"""Stream exported xctrace XML and isolate captured native command buffers.

This reads exports only. GPU intervals are joined by numeric CB ID against
PID-scoped application submissions. Durations use interval unions rather than
summing nested/overlapping rows. It never modifies or records a trace.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple
from xml.etree import ElementTree as ET


class Cell(NamedTuple):
    tag: str
    value: int | str | None
    pid: int | None = None


NUMERIC = {
    "start-time",
    "duration",
    "pid",
    "tid",
    "uint32",
    "uint64",
    "size-in-bytes",
    "metal-command-buffer-id",
    "metal-encoder-id",
    "metal-nesting-level",
    "connection-uuid64",
    "render-buffer-depth",
    "gpu-frame-number",
}


def iter_rows(path):
    """IDs are scoped to each export; register nested definitions recursively."""
    references = {}
    columns = None
    node = None

    def decode(element):
        if "ref" in element.attrib:
            reference = element.attrib["ref"]
            if reference not in references:
                raise ValueError(f"unresolved XML reference {reference} in {path}")
            return references[reference]
        children = [decode(child) for child in element]
        raw = (element.text or "").strip()
        value = (
            int(raw)
            if element.tag in NUMERIC and raw
            else element.attrib.get("fmt", raw or None)
        )
        pid = None
        if element.tag == "process":
            pid = next((child.value for child in children if child.tag == "pid"), None)
            if pid is None and isinstance(value, str):
                found = re.search(r"\(([0-9]+)\)$", value)
                pid = int(found.group(1)) if found else None
        cell = Cell(element.tag, value, pid)
        if "id" in element.attrib:
            references[element.attrib["id"]] = cell
        return cell

    for event, element in ET.iterparse(path, events=("start", "end")):
        if event == "start" and element.tag == "node":
            node = element
        elif event == "end" and element.tag == "schema":
            columns = [col.findtext("mnemonic") for col in element.findall("col")]
        elif event == "end" and element.tag == "row":
            if columns is None or len(element) != len(columns):
                raise ValueError("row/schema column count mismatch")
            row = {
                column: decode(cell)
                for column, cell in zip(columns, element, strict=True)
            }
            yield row
            element.clear()
            if node is not None:
                node.clear()


def merge_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def union_ns(intervals):
    return sum(end - start for start, end in merge_intervals(intervals))


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def distribution_ns(values):
    return {
        "count": len(values),
        "minimum_ms": min(values) / 1e6 if values else None,
        "median_ms": statistics.median(values) / 1e6 if values else None,
        "p95_ms": percentile(values, 0.95) / 1e6 if values else None,
        "maximum_ms": max(values) / 1e6 if values else None,
        "sum_ms": sum(values) / 1e6,
    }


def intersections_ns(intervals, start, end):
    return union_ns(
        (max(a, start), min(b, end)) for a, b in intervals if a < end and b > start
    )


def analyze(args):
    submissions = {}
    for row in iter_rows(args.submissions):
        if row["process"].pid != args.pid:
            continue
        cb = row["cmdbuffer-id"].value
        if cb in submissions:
            raise ValueError("duplicate native command-buffer ID")
        submissions[cb] = {
            "cb_id": cb,
            "cb_hex": hex(cb),
            "creation_ns": row["start"].value,
            "submission_duration_ns": row["duration"].value,
            "encoder_time_ns": row["encoder-time"].value,
            "encoders": row["num-encoders"].value,
            "event_type": row["event-type"].value,
        }
    completions = {}
    global_completed_rows = 0
    for row in iter_rows(args.completions):
        global_completed_rows += 1
        cb = row["cmdbuffer-id"].value
        if cb in submissions:
            completions[cb] = row["timestamp"].value
    gpu = defaultdict(list)
    active_other = []
    all_rows = 0
    excluded_native = defaultdict(list)
    states = Counter()
    channels = Counter()
    mismatched_pid = 0
    for row in iter_rows(args.gpu):
        all_rows += 1
        cb = row["cmdbuffer-id"].value
        pid = row["process"].pid
        start, duration = row["start"].value, row["duration"].value
        state = row["state"].value
        if state == "Active" and duration > 0 and pid != args.pid:
            active_other.append((start, start + duration))
        if cb not in submissions:
            if pid == args.pid:
                excluded_native[cb].append((start, start + duration))
            continue
        if pid != args.pid:
            mismatched_pid += 1
            continue
        states[state] += 1
        channels[row["channel-name"].value] += 1
        gpu[cb].append(
            {
                "start_ns": start,
                "end_ns": start + duration,
                "state": state,
                "channel": row["channel-name"].value,
                "depth": row["event-depth"].value,
                "label": row["event-label"].value,
                "reported_cpu_to_gpu_latency_ns": row["start-latency"].value,
                "encoder_id": row["encoder-id"].value,
                "gpu_submission_id": row["gpu-submission-id"].value,
                "connection_uuid": row["connection-UUID"].value,
                "bytes": row["bytes"].value,
            }
        )
    backgrounds = merge_intervals(active_other)
    matched_rows = []
    all_active = []
    for cb, submit in sorted(
        submissions.items(), key=lambda pair: pair[1]["creation_ns"]
    ):
        records = gpu.get(cb, [])
        active = [row for row in records if row["state"] == "Active"]
        intervals = [(row["start_ns"], row["end_ns"]) for row in active]
        all_active.extend(intervals)
        first = min(active, key=lambda row: row["start_ns"]) if active else None
        begin = first["start_ns"] if first else None
        end = max(row["end_ns"] for row in active) if active else None
        submit_end = submit["creation_ns"] + submit["submission_duration_ns"]
        summary = {
            **submit,
            "gpu_interval_rows": len(records),
            "active_gpu_interval_rows": len(active),
            "first_gpu_active_ns": begin,
            "last_gpu_active_ns": end,
            "union_gpu_active_ns": union_ns(intervals),
            "completed_ns": completions.get(cb),
            "creation_to_first_gpu_active_ns": begin - submit["creation_ns"]
            if begin is not None
            else None,
            "submission_span_end_to_first_gpu_active_ns": begin - submit_end
            if begin is not None
            else None,
            "reported_cpu_to_gpu_latency_at_first_active_ns": first[
                "reported_cpu_to_gpu_latency_ns"
            ]
            if first
            else None,
            "first_gpu_cpu_latency_origin_ns": begin
            - first["reported_cpu_to_gpu_latency_ns"]
            if first
            else None,
            "reported_latency_origin_minus_submission_end_ns": begin
            - first["reported_cpu_to_gpu_latency_ns"]
            - submit_end
            if first
            else None,
            "first_gpu_label": first["label"] if first else None,
            "states": dict(Counter(row["state"] for row in records)),
            "channels": dict(Counter(row["channel"] for row in records)),
        }
        matched_rows.append(summary)
    groups = []
    for row in matched_rows:
        previous_end = (
            (
                groups[-1][-1]["completed_ns"]
                or groups[-1][-1]["last_gpu_active_ns"]
                or groups[-1][-1]["creation_ns"]
            )
            if groups
            else None
        )
        if (
            not groups
            or row["creation_ns"] - previous_end > args.cluster_gap_seconds * 1e9
        ):
            groups.append([])
        groups[-1].append(row)
    clusters = []
    for index, group in enumerate(groups):
        ids = {row["cb_id"] for row in group}
        intervals = [
            (row["start_ns"], row["end_ns"])
            for cb in ids
            for row in gpu.get(cb, [])
            if row["state"] == "Active"
        ]
        merged = merge_intervals(intervals)
        start = min(row["creation_ns"] for row in group)
        end = max(
            [
                row["completed_ns"] or row["last_gpu_active_ns"] or row["creation_ns"]
                for row in group
            ]
        )
        gaps = [
            (left[1], right[0])
            for left, right in zip(merged, merged[1:])
            if right[0] > left[1]
        ]
        first_gpu = merged[0][0] if merged else None
        pre_gpu = (first_gpu - start) if first_gpu is not None else 0
        clusters.append(
            {
                "index": index,
                "cb_count": len(group),
                "first_creation_seconds": start / 1e9,
                "last_completion_or_gpu_end_seconds": end / 1e9,
                "captured_window_seconds": (end - start) / 1e9,
                "matched_gpu_active_union_seconds": union_ns(intervals) / 1e9,
                "native_inactive_captured_window_seconds": (
                    (end - start) - union_ns(intervals)
                )
                / 1e9,
                "before_first_gpu_active_seconds": pre_gpu / 1e9,
                "other_process_gpu_active_before_first_native_ms": intersections_ns(
                    backgrounds, start, first_gpu
                )
                / 1e6
                if first_gpu is not None
                else 0,
                "gpu_internal_gaps": distribution_ns([b - a for a, b in gaps]),
                "gpu_gap_other_process_overlap_ms": sum(
                    intersections_ns(backgrounds, a, b) for a, b in gaps
                )
                / 1e6,
                "submission_end_to_first_gpu_active": distribution_ns(
                    [
                        row["submission_span_end_to_first_gpu_active_ns"]
                        for row in group
                        if row["first_gpu_active_ns"] is not None
                    ]
                ),
                "creation_to_first_gpu_active": distribution_ns(
                    [
                        row["creation_to_first_gpu_active_ns"]
                        for row in group
                        if row["first_gpu_active_ns"] is not None
                    ]
                ),
                "reported_cpu_to_gpu_latency_first_active": distribution_ns(
                    [
                        row["reported_cpu_to_gpu_latency_at_first_active_ns"]
                        for row in group
                        if row["first_gpu_active_ns"] is not None
                    ]
                ),
                "longest_internal_gaps": [
                    {
                        "start_seconds": a / 1e9,
                        "end_seconds": b / 1e9,
                        "duration_ms": (b - a) / 1e6,
                        "other_process_gpu_overlap_ms": intersections_ns(
                            backgrounds, a, b
                        )
                        / 1e6,
                    }
                    for a, b in sorted(
                        gaps, key=lambda pair: pair[1] - pair[0], reverse=True
                    )[:20]
                ],
                "cb_ids": [row["cb_hex"] for row in group],
            }
        )
    queue_events = [
        (row["creation_ns"] + row["submission_duration_ns"], 1) for row in matched_rows
    ]
    queue_events += [
        (row["completed_ns"], -1)
        for row in matched_rows
        if row["completed_ns"] is not None
    ]
    outstanding = maximum_outstanding = 0
    for timestamp, change in sorted(
        queue_events, key=lambda event: (event[0], event[1])
    ):
        outstanding += change
        maximum_outstanding = max(maximum_outstanding, outstanding)
    origin_offsets = [
        row["reported_latency_origin_minus_submission_end_ns"]
        for row in matched_rows
        if row["first_gpu_active_ns"] is not None
    ]
    result = {
        "schema": "splash-flash-matched-metal-system-trace-cpu-analysis-v1",
        "native_pid": args.pid,
        "gpu_executed_by_analysis": False,
        "application_submission_count": len(submissions),
        "global_completion_rows": global_completed_rows,
        "matched_completion_count": len(completions),
        "global_gpu_interval_rows": all_rows,
        "matched_native_gpu_interval_rows": sum(len(rows) for rows in gpu.values()),
        "matched_native_cb_count_with_gpu_rows": len(gpu),
        "matching_cb_but_wrong_process_rows": mismatched_pid,
        "matched_states": dict(states),
        "matched_channels": dict(channels),
        "matched_gpu_active_union_seconds": union_ns(all_active) / 1e9,
        "reported_gpu_latency_origin_matches_submission_span_end_count": sum(
            value == 0 for value in origin_offsets
        ),
        "reported_gpu_latency_origin_offsets": distribution_ns(origin_offsets),
        "maximum_captured_native_outstanding_command_buffers": maximum_outstanding,
        "excluded_native_gpu_cb_ids": [
            {
                "cb_hex": hex(cb),
                "intervals": len(rows),
                "begin_seconds": min(a for a, b in rows) / 1e9,
                "end_seconds": max(b for a, b in rows) / 1e9,
            }
            for cb, rows in excluded_native.items()
        ],
        "clusters": clusters,
        "command_buffers": matched_rows,
        "scope_notes": [
            "Only native application-submission CB IDs are joined to completions/GPU intervals.",
            "Raw xctrace time integers are nanoseconds relative to trace start.",
            "Application submission start column is labeled Creation. End-to-GPU latency is conservatively labeled submission-span-end until CPU commit semantics are independently established.",
            "GPU active duration uses unions, avoiding double counting nested/channel-overlapping rows.",
            "Other-process overlap is reported separately and cannot establish causation or physical queue contention.",
            "This is an instrumented capture; no unprofiled performance conclusion.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "application_submission_count",
                    "matched_completion_count",
                    "global_gpu_interval_rows",
                    "matched_native_gpu_interval_rows",
                    "matched_native_cb_count_with_gpu_rows",
                    "matched_states",
                    "matched_channels",
                    "matched_gpu_active_union_seconds",
                )
            }
        )
    )
    print(
        json.dumps(
            {
                "clusters": [
                    {
                        key: value
                        for key, value in cluster.items()
                        if key not in ("cb_ids", "longest_internal_gaps")
                    }
                    for cluster in clusters
                ]
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submissions", type=Path, required=True)
    parser.add_argument("--completions", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cluster-gap-seconds", type=float, default=1)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()

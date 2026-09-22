#!/usr/bin/env python3
"""Summarize private trained-head attribution JSONL without running Metal.

Rows contain case, phase, family_attribution and command (ProfilingJson.hpp).
Use explicit context_per_lane, true_rows_per_lane, physical_rows and lanes when
available, either on the row or in a case object. Missing geometry stays null.
Stage/dispatch counter durations describe an instrumented schedule, not an
unprofiled performance baseline. The command-minus-dispatch sum is signed and
is not an estimate of CPU, scheduling, or missing-kernel cost.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Iterable

PHASES = {"proposal", "committed_fold", "joint_proposal"}
COUNTER_MODES = {"stage", "dispatch"}
MODES = COUNTER_MODES | {"off", "command"}
STATUSES = {
    "pending",
    "complete",
    "unsupported",
    "sample_limit_exceeded",
    "allocation_failed",
    "resolve_failed",
    "invalid_timestamps",
    "command_failed",
}
TIMING_NOTES = (
    "Stage mode inserts an encoder boundary for each dispatch; dispatch mode "
    "inserts timestamp sampling barriers. Both can change GPU scheduling and "
    "cache behavior. No unprofiled overhead or throughput claim is inferred. "
    "Only complete counter profiles with valid finite dispatch timestamps "
    "contribute kernel/family times. Untimed or unsupported dispatches are "
    "counted, not assigned zero time. Command GPU durations remain separate. "
    "A zero recorded command duration does not prove a valid GPU interval. "
    "Command wall time starts at host encoding and excludes host preparation. "
    "Command minus the sum of emitted timed dispatches is a signed diagnostic; "
    "gaps, overlap, untimed/truncated metadata and instrumentation can affect "
    "it. It is not CPU or scheduling time. Family times use the fixture's "
    "classification; no pipeline-name classification is guessed."
    " Family aggregates may cover more metadata than a truncated emitted "
    "dispatch prefix; these two sums retain their separate coverage."
)


def finite_number(value, *, nonnegative=True):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and (not nonnegative or value >= 0)
    except OverflowError:
        return False


def count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def distribution_ms(values):
    return {
        "count": len(values),
        "sum_ms": math.fsum(values) * 1000 if values else None,
        "mean_ms": math.fsum(values) * 1000 / len(values) if values else None,
        "median_ms": median(values) * 1000 if values else None,
        "minimum_ms": min(values) * 1000 if values else None,
        "maximum_ms": max(values) * 1000 if values else None,
    }


def close(left, right):
    return math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-9)


def geometry(row):
    case = row.get("case")
    nested = case if isinstance(case, dict) else {}
    label = next((nested[key] for key in ("name", "id", "case") if key in nested), case)
    if not isinstance(label, str) or not label:
        raise ValueError(
            "case must be a nonempty string or an object with name/id/case"
        )

    def select(*names):
        return next(
            (
                source[name]
                for source in (row, nested)
                for name in names
                if name in source
            ),
            None,
        )

    result = {
        "case": label,
        "phase": row.get("phase"),
        "context_per_lane": select("context_per_lane", "context_rows", "context"),
        "true_rows_per_lane": select(
            "true_rows_per_lane", "real_rows_per_lane", "rows"
        ),
        "physical_rows": select("physical_rows"),
        "lanes": select("lanes"),
    }
    if not isinstance(result["phase"], str) or result["phase"] not in PHASES:
        raise ValueError(f"unknown head phase {result['phase']!r}")
    for name, value in result.items():
        if name not in {"case", "phase"} and value is not None and not count(value):
            raise ValueError(f"{name} must be a nonnegative integer or null")
    return result


@dataclass
class Totals:
    commands: int = 0
    statuses: Counter = field(default_factory=Counter)
    modes: Counter = field(default_factory=Counter)
    reasons: Counter = field(default_factory=Counter)
    command_gpu: list = field(default_factory=list)
    command_wall: list = field(default_factory=list)
    dispatch_sum: list = field(default_factory=list)
    remainder: list = field(default_factory=list)
    dispatches: int = 0
    emitted: int = 0
    timed: int = 0
    truncated_commands: int = 0
    dropped_profiles: int = 0
    classification_complete: int = 0
    complete_commands: int = 0
    altered_encoders: int = 0
    sampling_barriers: int = 0
    pipelines: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0, []]))
    families: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0, []]))
    cases: set = field(default_factory=set)

    def add(self, row):
        self.commands += 1
        self.cases.add(row["geometry"]["case"])
        self.statuses[row["status"]] += 1
        self.modes[row["mode"]] += 1
        if row["reason"]:
            self.reasons[row["reason"]] += 1
        self.dispatches += row["dispatch_count"]
        self.emitted += len(row["dispatches"])
        self.timed += row["timed_dispatches"]
        self.truncated_commands += row["truncated"]
        self.dropped_profiles += row["dropped"]
        self.classification_complete += row["classification_complete"]
        self.complete_commands += row["attribution_complete"]
        self.altered_encoders += row["altered_encoders"]
        self.sampling_barriers += row["sampling_barriers"]
        for key, destination in (
            ("command_gpu", self.command_gpu),
            ("command_wall", self.command_wall),
            ("dispatch_sum", self.dispatch_sum),
            ("remainder", self.remainder),
        ):
            if row[key] is not None:
                destination.append(row[key])
        for name, seconds in row["dispatches"]:
            entry = self.pipelines[name]
            entry[0] += 1
            if seconds is not None:
                entry[1] += 1
                entry[2].append(seconds)
        for name, (dispatches, timed, seconds) in row["families"].items():
            entry = self.families[name]
            entry[0] += dispatches
            entry[1] += timed
            if seconds is not None:
                entry[2].append(seconds)

    def report(self):
        kernel_seconds = math.fsum(self.dispatch_sum)
        family_seconds = math.fsum(
            seconds for _, _, values in self.families.values() for seconds in values
        )

        def ranked(entries, name_key, total):
            result = []
            for name, (dispatches, timed, values) in entries.items():
                seconds = math.fsum(values) if values else None
                result.append(
                    {
                        name_key: name,
                        "dispatches": dispatches,
                        "timed_dispatches": timed,
                        "untimed_dispatches": dispatches - timed,
                        "gpu_ms": seconds * 1000 if seconds is not None else None,
                        "share_of_timed_gpu": seconds / total
                        if seconds is not None and total > 0
                        else None,
                    }
                )
            return sorted(
                result,
                key=lambda item: (
                    item["gpu_ms"] is None,
                    -(item["gpu_ms"] or 0),
                    item[name_key],
                ),
            )

        return {
            "commands": self.commands,
            "cases": sorted(self.cases),
            "command_status_counts": dict(sorted(self.statuses.items())),
            "mode_counts": dict(sorted(self.modes.items())),
            "status_reason_counts": dict(sorted(self.reasons.items())),
            "command_gpu": distribution_ms(self.command_gpu),
            "command_wall": distribution_ms(self.command_wall),
            "sum_emitted_timed_dispatch_gpu": distribution_ms(self.dispatch_sum),
            "command_minus_emitted_timed_dispatch_gpu": distribution_ms(self.remainder),
            "dispatches_reported": self.dispatches,
            "dispatches_emitted": self.emitted,
            "timed_dispatches": self.timed,
            "untimed_emitted_dispatches": self.emitted - self.timed,
            "dispatches_not_emitted": self.dispatches - self.emitted,
            "timed_dispatch_coverage": self.timed / self.dispatches
            if self.dispatches
            else None,
            "truncated_commands": self.truncated_commands,
            "dropped_profiles_before": self.dropped_profiles,
            "family_classification_complete_commands": self.classification_complete,
            "attribution_complete": self.commands > 0
            and self.complete_commands == self.commands,
            "perturbation": {
                "counter_mode_commands": sum(
                    self.modes[mode] for mode in COUNTER_MODES
                ),
                "encoder_boundaries_altered_commands": self.altered_encoders,
                "sampling_barrier_commands": self.sampling_barriers,
            },
            "pipelines": ranked(self.pipelines, "pipeline", kernel_seconds),
            "families": ranked(self.families, "family", family_seconds),
        }


def normalize(row, errors, location):
    identity = geometry(row)
    command, attribution = row.get("command"), row.get("family_attribution")
    if not isinstance(command, dict) or not isinstance(attribution, dict):
        raise ValueError("command and family_attribution must be objects")
    mode, status = command.get("mode"), command.get("status")
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError(f"unknown command profiling mode {mode!r}")
    if not isinstance(status, str) or status not in STATUSES:
        raise ValueError(f"unknown command profile status {status!r}")
    dispatches = command.get("dispatches")
    families = attribution.get("families")
    if not isinstance(dispatches, list) or not isinstance(families, dict):
        raise ValueError(
            "command dispatches must be an array and attribution families an object"
        )
    error_start = len(errors)

    def error(message):
        errors.append(f"{location}: {message}")

    total = command.get(
        "dispatch_count", command.get("dispatches_total", len(dispatches))
    )
    if not count(total) or total < len(dispatches):
        raise ValueError(
            "dispatch count must be an integer at least as large as emitted metadata"
        )
    truncated = (
        bool(command.get("dispatches_truncated"))
        or bool(command.get("dispatch_metadata_truncated"))
        or len(dispatches) < total
    )
    for key, expected in (
        ("dispatches_emitted", len(dispatches)),
        ("dispatches_total", total),
    ):
        if key in command and command[key] != expected:
            error(f"{key} disagrees with dispatch metadata")
    metadata_count = command.get("dispatches_metadata_count", total)
    if not count(metadata_count) or not len(dispatches) <= metadata_count <= total:
        raise ValueError(
            "dispatches_metadata_count must lie between emitted and total dispatch counts"
        )
    dropped = command.get("dropped_profiles_before", 0)
    if not count(dropped):
        raise ValueError("dropped_profiles_before must be a nonnegative integer")
    eligible = status == "complete" and mode in COUNTER_MODES
    normalized_dispatches = []
    for dispatch in dispatches:
        if not isinstance(dispatch, dict) or not isinstance(
            dispatch.get("pipeline"), str
        ):
            raise ValueError("every dispatch must have a pipeline string")
        seconds = dispatch.get("gpu_seconds")
        valid = dispatch.get("timestamps_valid") is True and finite_number(seconds)
        if dispatch.get("timestamps_valid") is True and not finite_number(seconds):
            error(f"invalid GPU duration for pipeline {dispatch['pipeline']}")
        normalized_dispatches.append(
            (dispatch["pipeline"], seconds if eligible and valid else None)
        )
    timed = sum(seconds is not None for _, seconds in normalized_dispatches)
    dispatch_sum = (
        math.fsum(
            seconds for _, seconds in normalized_dispatches if seconds is not None
        )
        if timed
        else None
    )
    gpu, wall = command.get("gpu_seconds"), command.get("wall_seconds")
    if not finite_number(gpu):
        error("invalid command GPU duration")
        gpu = None
    if wall is not None and not finite_number(wall):
        error("invalid command wall duration")
        wall = None
    normalized_families = {}
    for name, family in families.items():
        if not isinstance(family, dict):
            raise ValueError(f"family {name} must be an object")
        dispatch_count, timed_count = (
            family.get("dispatches"),
            family.get("timed_dispatches"),
        )
        if (
            not count(dispatch_count)
            or not count(timed_count)
            or timed_count > dispatch_count
        ):
            raise ValueError(f"invalid dispatch counts for family {name}")
        seconds = family.get("gpu_seconds")
        if timed_count and not finite_number(seconds):
            error(f"invalid GPU duration for family {name}")
        normalized_families[name] = (
            dispatch_count,
            timed_count if eligible and finite_number(seconds) else 0,
            seconds if eligible and timed_count and finite_number(seconds) else None,
        )
    family_dispatches = sum(value[0] for value in normalized_families.values())
    family_timed = sum(value[1] for value in normalized_families.values())
    family_sum = math.fsum(
        value[2] for value in normalized_families.values() if value[2] is not None
    )
    classification_complete = attribution.get("classification_complete") is True
    if family_dispatches > metadata_count:
        error("family dispatch counts exceed available command metadata")
    if classification_complete and family_dispatches != total:
        error("complete family classification does not cover all reported dispatches")
    if "timed_dispatches" in attribution:
        raw_family_timed = sum(
            family["timed_dispatches"] for family in families.values()
        )
        if (
            not count(attribution["timed_dispatches"])
            or attribution["timed_dispatches"] != raw_family_timed
        ):
            error("attribution timed_dispatches disagrees with family counts")
    if not truncated:
        if family_dispatches != total:
            error("family dispatch counts do not cover emitted command dispatches")
        if eligible:
            if family_timed != timed:
                error("family timed counts disagree with command dispatches")
            if not close(family_sum, dispatch_sum or 0):
                error("family GPU sum disagrees with emitted timed dispatch sum")
    # Serialized families may summarize more dispatches than a truncated JSON
    # command array. Never equate that larger family sum with emitted pipelines.
    for key, derived in (
        ("full_command_gpu_seconds", gpu),
        ("sum_timed_dispatch_gpu_seconds", family_sum if eligible else None),
        (
            "command_minus_timed_dispatch_seconds",
            gpu - family_sum if gpu is not None and eligible else None,
        ),
    ):
        if key in attribution and derived is not None:
            if not finite_number(
                attribution[key],
                nonnegative=key != "command_minus_timed_dispatch_seconds",
            ) or not close(attribution[key], derived):
                error(f"attribution {key} disagrees with command timing")
    return {
        "geometry": identity,
        "mode": mode,
        "status": status,
        "reason": str(command.get("reason", "")),
        "command_gpu": gpu,
        "command_wall": wall,
        "dispatch_sum": dispatch_sum,
        "remainder": gpu - dispatch_sum
        if gpu is not None and dispatch_sum is not None
        else None,
        "dispatch_count": total,
        "dispatches": normalized_dispatches,
        "timed_dispatches": timed,
        "families": normalized_families,
        "truncated": truncated,
        "dropped": dropped,
        "classification_complete": classification_complete,
        "altered_encoders": command.get("encoder_boundaries_altered") is True,
        "sampling_barriers": command.get("sampling_barriers") is True,
        "attribution_complete": eligible
        and not truncated
        and not dropped
        and timed == total
        and classification_complete
        and len(errors) == error_start,
    }


@dataclass(frozen=True)
class LocatedRow:
    line_number: int
    value: dict


def summarize_records(records: Iterable[dict | LocatedRow], *, source="synthetic"):
    errors, overall = [], Totals()
    by_case, by_geometry = defaultdict(Totals), defaultdict(Totals)
    rows = 0
    for rows, record in enumerate(records, 1):
        line_number = record.line_number if isinstance(record, LocatedRow) else rows
        record = record.value if isinstance(record, LocatedRow) else record
        location = f"{source}:{line_number}"
        try:
            if not isinstance(record, dict):
                raise ValueError("trace row must be an object")
            row = normalize(record, errors, location)
        except ValueError as error:
            errors.append(f"{location}: {error}")
            continue
        identity = {**row["geometry"], "mode": row["mode"]}
        overall.add(row)
        by_case[tuple(identity.items())].add(row)
        by_geometry[
            tuple((key, value) for key, value in identity.items() if key != "case")
        ].add(row)

    def groups(values):
        return [
            {**dict(key), **value.report()}
            for key, value in sorted(values.items(), key=lambda item: repr(item[0]))
        ]

    if not rows:
        errors.append(f"{source}: no trace rows")
    aggregate = overall.report()
    aggregate["attribution_complete"] = aggregate["attribution_complete"] and not errors
    return {
        "schema": "splash-flash-head-profile-summary-v1",
        "source": source,
        "valid": not errors,
        "validation_errors": errors,
        "trace_rows": rows,
        "overall": aggregate,
        "by_case": groups(by_case),
        "by_context_rows": groups(by_geometry),
        "timing_notes": TIMING_NOTES,
    }


def analyze(path):
    def records():
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        yield LocatedRow(line_number, json.loads(line))
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"{path}:{line_number}: invalid JSON: {error.msg}"
                        ) from error

    return summarize_records(records(), source=str(path.resolve()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, help="write the complete JSON summary")
    parser.add_argument(
        "--top",
        type=int,
        default=8,
        help="pipeline/family entries printed per context/rows group",
    )
    args = parser.parse_args(argv)
    if args.top < 1:
        parser.error("--top must be at least 1")
    try:
        reports = [analyze(path) for path in args.traces]
    except (OSError, ValueError) as error:
        parser.exit(1, f"ERROR: {error}\n")
    result = {"valid": all(report["valid"] for report in reports), "reports": reports}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for report in reports:
        print(
            f"{Path(report['source']).name}: {report['overall']['commands']} commands; "
            f"valid={report['valid']}; attribution_complete={report['overall']['attribution_complete']}"
        )
        for group in report["by_context_rows"]:
            print(
                f"  {group['phase']} mode={group['mode']} context={group['context_per_lane']} "
                f"rows/lane={group['true_rows_per_lane']} physical_rows={group['physical_rows']} "
                f"lanes={group['lanes']}: {group['timed_dispatches']}/{group['dispatches_reported']} timed dispatches; "
                f"statuses={group['command_status_counts']}"
            )

            def total_ms(key):
                value = group[key]["sum_ms"]
                return f"{value:.3f} ms" if value is not None else "unavailable"

            print(
                f"    command GPU sum={total_ms('command_gpu')}; "
                f"emitted counter sum={total_ms('sum_emitted_timed_dispatch_gpu')}; "
                f"signed command-minus-counter sum={total_ms('command_minus_emitted_timed_dispatch_gpu')}"
            )
            for key, name in (("pipelines", "pipeline"), ("families", "family")):
                entries = [
                    f"{entry[name]} {entry['gpu_ms']:.3f} ms"
                    if entry["gpu_ms"] is not None
                    else f"{entry[name]} untimed"
                    for entry in group[key][: args.top]
                ]
                print(f"    {key}: " + "; ".join(entries))
        for error in report["validation_errors"]:
            print("  ERROR:", error)
    print(TIMING_NOTES)
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Read frozen native observations; no GPU or external dependencies."""

import collections
import json
import random
import statistics
import sys
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def aggregate(path):
    value = read(path)
    meta = read(path.with_suffix(".meta.json"))
    rows = value["measurements"]
    cycles = [c for m in rows for c in m["cycles"]]
    output = sum(len(m["output_tokens"]) for m in rows)
    wall = sum(m["wall_s"] for m in rows)
    emitted = sum(c["emitted"] for c in cycles)
    cycle_wall = sum(c["wall_s"] for c in cycles)
    return {
        "output_tokens": output,
        "request_wall_s": wall,
        "request_tokens_per_second": output / wall,
        "cycle_wall_s": cycle_wall,
        "cycle_gpu_s": sum(c["gpu_s"] for c in cycles),
        "cycle_emitted_tokens": emitted,
        "cycles": len(cycles),
        "cycle_tokens_per_second": emitted / cycle_wall,
        "tokens_per_cycle": emitted / len(cycles),
        "accepted_draft_tokens": sum(c["accepted"] for c in cycles),
        "mean_cycle_wall_ms": 1000 * cycle_wall / len(cycles),
        "verify_rows_histogram": dict(
            collections.Counter(c["verify_rows"] for c in cycles)
        )
        if cycles and "verify_rows" in cycles[0]
        else None,
        "mean_ttft_s": statistics.mean(m["ttft_s"] for m in rows),
        "preparation_s": value["preparation_s"],
        "warmup_s": value["warmup_s"],
        "conversion_s": value.get("conversion_s"),
        "process_wall_s": meta["process_wall_s"],
        "peak_rss_bytes": meta["peak_rss_bytes"],
        "additional_payload_bytes": value.get("additional_payload_bytes"),
        "power_before": meta["power_before"],
        "power_after": meta["power_after"],
    }


def tokens(path):
    return [(m["input_tokens"], m["output_tokens"]) for m in read(path)["measurements"]]


def main(root):
    result = {
        "protocol": read(root / "protocol.json"),
        "families": {},
        "long_context_pilots": [],
        "scope": "Same native binary within each family. Fixed known prompts; no broad quality or serving claim.",
        "telemetry_note": "macmon raw observations retained. No controlled energy-efficiency or total-system power claim.",
    }
    for name in ["dense-prefix5", "dense-adaptive", "dictionary"]:
        p = root / (name + "-groups.json")
        if not p.exists():
            continue
        groups = read(p)
        stable = [g for g in groups if g["stable"]]
        jobs = {
            f"g{g['group']}-{arm}": aggregate(root / f"{name}-g{g['group']}-{arm}.json")
            for g in groups
            for arm in ["A1", "B1", "B2", "A2"]
        }
        item = {
            "all_groups": groups,
            "stable_groups": len(stable),
            "excluded_groups": len(groups) - len(stable),
            "jobs": jobs,
            "all_complete_token_arrays_equal": all(
                g["complete_token_arrays_equal"] for g in groups
            ),
        }
        if stable:
            control = [
                statistics.mean([g["latency_s_per_token"][a] for a in ["A1", "A2"]])
                for g in stable
            ]
            candidate = [
                statistics.mean([g["latency_s_per_token"][a] for a in ["B1", "B2"]])
                for g in stable
            ]
            noise = max(
                max(g["endpoint_spread_control"], g["endpoint_spread_candidate"])
                for g in stable
            )
            floor = max(0.02, 2 * noise)
            gain = statistics.mean(control) / statistics.mean(candidate) - 1
            reduction = 1 - statistics.mean(candidate) / statistics.mean(control)
            rng = random.Random(658)
            samples = []
            for _ in range(5000):
                selected = [rng.randrange(len(stable)) for _ in stable]
                samples.append(
                    100
                    * (
                        statistics.mean(control[i] for i in selected)
                        / statistics.mean(candidate[i] for i in selected)
                        - 1
                    )
                )
            samples.sort()
            item.update(
                {
                    "request_tokens_per_second_control": 1 / statistics.mean(control),
                    "request_tokens_per_second_candidate": 1
                    / statistics.mean(candidate),
                    "speedup_percent": 100 * gain,
                    "latency_reduction_percent": 100 * reduction,
                    "practical_floor_percent": 100 * floor,
                    "descriptive_group_bootstrap95_speedup_percent": [
                        samples[124],
                        samples[4874],
                    ],
                    "qualified_positive_speedup": len(stable) >= 3
                    and reduction > floor
                    and samples[124] > 0
                    and item["all_complete_token_arrays_equal"],
                }
            )
            item["qualified_regression"] = (
                len(stable) >= 3 and -reduction > floor and samples[4874] < 0
            )
            item["verdict"] = (
                "insufficient stable groups"
                if len(stable) < 3
                else "complete outputs differ"
                if not item["all_complete_token_arrays_equal"]
                else "qualified positive speedup"
                if item["qualified_positive_speedup"]
                else "qualified regression"
                if item["qualified_regression"]
                else "timing change below practical floor"
            )
            for arm, positions in [
                ("control", ["A1", "A2"]),
                ("candidate", ["B1", "B2"]),
            ]:
                observations = [
                    jobs[f"g{g['group']}-{p}"] for g in stable for p in positions
                ]
                item[arm + "_means"] = {
                    key: statistics.mean(j[key] for j in observations)
                    for key in [
                        "cycle_tokens_per_second",
                        "tokens_per_cycle",
                        "mean_cycle_wall_ms",
                        "mean_ttft_s",
                        "preparation_s",
                        "warmup_s",
                        "process_wall_s",
                        "peak_rss_bytes",
                    ]
                }
                hist = collections.Counter()
                for job in observations:
                    hist.update(job["verify_rows_histogram"] or {})
                item[arm + "_verify_rows_histogram"] = dict(hist)
        result["families"][name] = item
    if (root / "dense-prefix5-g1-A1.meta.json").exists() and (
        root / "dense-adaptive-g1-A1.meta.json"
    ).exists():
        result["dense_fixed_five_vs_full_complete_token_arrays_equal"] = tokens(
            root / "dense-prefix5-g1-A1.json"
        ) == tokens(root / "dense-adaptive-g1-A1.json")
    for context in ["runtime-8192", "runtime-16384"]:
        for arm in ["dense-five", "dense-adaptive", "dictionary-candidate"]:
            a = root / (
                context
                + (
                    "-dictionary-control.json"
                    if arm == "dictionary-candidate"
                    else "-dense-all.json"
                )
            )
            b = root / (context + "-" + arm + ".json")
            if a.exists() and b.exists():
                try:
                    x, y = aggregate(a), aggregate(b)
                except (ValueError, FileNotFoundError):
                    continue
                result["long_context_pilots"].append(
                    {
                        "context": context,
                        "candidate": arm,
                        "complete_token_arrays_equal": tokens(a) == tokens(b),
                        "control": x,
                        "experiment": y,
                        "speedup_percent_unreplicated": 100
                        * (
                            y["request_tokens_per_second"]
                            / x["request_tokens_per_second"]
                            - 1
                        ),
                        "qualified_for_speed_claim": False,
                    }
                )
    for name in ["dictionary-fidelity-control", "dictionary-fidelity-candidate"]:
        if (root / (name + ".meta.json")).exists():
            result[name] = aggregate(root / (name + ".json"))
    if "dictionary-fidelity-candidate" in result:
        result["dictionary_initial_fidelity"] = tokens(
            root / "dictionary-fidelity-control.json"
        ) == tokens(root / "dictionary-fidelity-candidate.json")
    output = root / "analysis.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {k: v for k, v in data.items() if k not in ["jobs", "all_groups"]}
                for name, data in result["families"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main(
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parents[1] / "qualification"
    )

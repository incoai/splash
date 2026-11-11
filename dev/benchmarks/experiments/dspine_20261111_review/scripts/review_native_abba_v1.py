"""Supplement a frozen native ABBA analysis with provenance and cross-group gates."""

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import re
import sys
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observed(path):
    digest = sha(path)
    assert str(path) not in INPUTS or INPUTS[str(path)] == digest
    INPUTS[str(path)] = digest
    return digest


def read(path):
    observed(path)
    return json.loads(path.read_text())


def token_hash(tokens):
    value = 1469598103934665603
    for token in tokens:
        for shift in range(0, 32, 8):
            value = ((value ^ ((token >> shift) & 255)) * 1099511628211) & (
                (1 << 64) - 1
            )
    return str(value)


def load_protocol(path):
    protocol = read(path)
    assert protocol["groups"] == 3
    assert protocol["order_each_group"] == [
        "control",
        "candidate",
        "candidate",
        "control",
    ]
    prefix = protocol["prefix"]
    barrier = read(ROOT / "notes" / (prefix + ".artifact-barrier.json"))
    assert barrier["sha256"][str(path)] == observed(path)
    jobs_path = ROOT / "notes" / (prefix + ".jobs.json")
    jobs = read(jobs_path)
    keys = [
        f"g{group}_{arm}" for group in range(1, 4) for arm in ["a1", "b1", "b2", "a2"]
    ]
    assert list(jobs) == keys
    assert protocol["samples_per_round"] > 0
    assert protocol["temperature"] == protocol["ane_ffn_share"] == 0
    assert protocol["context_tokens"] == 32768 and protocol["verify_rows"] == "all"
    cohort_path = Path(protocol["cohort"])
    assert observed(cohort_path) == protocol["cohort_sha256"]
    cases = {}
    for line in cohort_path.read_text().splitlines():
        name, cap, *tokens = line.split()
        assert name not in cases
        cases[name] = {"cap": int(cap), "tokens": [int(v) for v in tokens]}
    assert set(cases) == set(protocol["case_names"])
    assert len(cases) == protocol["cases"] == len(protocol["registered_cases"])
    for case in protocol["registered_cases"]:
        original = cases[case["name"]]
        assert case["output_cap"] == original["cap"]
        assert case["prompt_tokens"] == len(original["tokens"])
        assert (
            case["prompt_tokens_sha256"]
            == hashlib.sha256(
                json.dumps(original["tokens"], separators=(",", ":")).encode()
            ).hexdigest()
        )
    builds = {}
    for arm, nomination in protocol["artifact_nomination"].items():
        manifest_path = Path(nomination["build_manifest"])
        assert observed(manifest_path) == nomination["build_manifest_sha256"]
        manifest = read(manifest_path)
        assert manifest == nomination["build_manifest_data"]
        assert (
            manifest["binary_sha256"]
            == nomination["bundle_sha256"][nomination["driver"]]
        )
        for artifact, digest in {
            **nomination["bundle_sha256"],
            **nomination["model_metadata_sha256"],
        }.items():
            assert observed(Path(artifact)) == digest
        headers = [
            (Path(p), digest)
            for p, digest in manifest["inputs"].items()
            if Path(p).name == "BuildIdentity.hpp"
        ]
        assert len(headers) == 1
        header, digest = headers[0]
        assert observed(header) == digest
        builds[arm] = re.search(
            r'#define SPLASH_BUILD_ID "([^"]+)"', header.read_text()
        )[1]
    for key, job in jobs.items():
        ident = prefix + "-" + key.replace("_", "-")
        assert job["id"] == ident
        arm_name = "control" if key.endswith(("a1", "a2")) else "candidate"
        arm = protocol["arms"][arm_name]
        nominee = protocol["artifact_nomination"][arm["variant"]]
        expected = [
            protocol["execution_python"],
            str(ROOT / "scripts/run_frozen_abba_round_v1.py"),
            str(path),
            ident,
            nominee["driver"],
            nominee["library"],
            arm["model_root"],
            "--draft-block-size",
            str(arm["rows"]),
            "--cohort",
            str(cohort_path),
            "--samples",
            str(protocol["samples_per_round"]),
            "--ane-ffn-share",
            "0",
            "--max-context",
            "32768",
            "--temperature",
            "0",
            "--study-width",
            str(protocol["clone_width"]),
        ]
        expected += ["--progress", str(ROOT / "results" / (ident + ".progress.jsonl"))]
        assert job["command"] == expected
        assert set(job["artifacts"]) == set(job["expected_sha256"])
        for artifact, digest in job["expected_sha256"].items():
            assert observed(Path(artifact)) == digest
    analyzer_path = ROOT / "scripts" / protocol["analyzer_file"]
    assert observed(analyzer_path) == protocol["analyzer_sha256"]
    assert observed(ROOT / "scripts/abba.py") == protocol["abba_sha256"]
    return protocol, jobs, jobs_path, cases, builds


def load_round(protocol, jobs_path, job, cases, builds):
    ident = job["id"]
    receipt = read(ROOT / "results" / (ident + ".receipt.json"))
    assert (
        receipt["id"] == ident
        and receipt["state"] == "complete"
        and receipt["exit_code"] == 0
    )
    assert receipt["protocol_sha256"] == observed(jobs_path)
    assert receipt["job"] == job
    assert (
        receipt["artifact_sha256"]
        == receipt["artifact_sha256_after"]
        == job["expected_sha256"]
    )
    operating = read(ROOT / "results" / (ident + ".operating.json"))
    assert operating["id"] == ident and operating["exit_code"] == 0
    assert operating["command"] == job["command"][4:]
    for gate in [
        "operating_gate_passed",
        "no_serious_or_critical_state",
        "no_swap",
        "ac_throughout_measurement",
    ]:
        assert operating[gate] is True
    events = [v for v in operating["events"] if v["phase"] == "measurement"]
    assert events
    assert all(
        v["ac_power"] and v["thermal_state"] < 2 and v["memory"]["swap_usage"] == 0
        for v in events
    )
    doc = read(ROOT / "results" / (ident + ".stdout"))
    arm_name = "control" if ident.endswith(("a1", "a2")) else "candidate"
    arm = protocol["arms"][arm_name]
    assert doc["build_id"] == builds[arm_name]
    assert doc["geometry"]["verify_rows"] == arm["rows"]
    assert doc["identity"]["model_root"] == arm["model_root"]
    assert (
        doc["identity"]["loaded_model_layout_sha256"]
        == arm["loaded_model_layout_sha256"]
    )
    assert doc["study_temperature"] == doc["ane_ffn_share"] == 0
    assert doc["max_context_tokens"] == protocol["context_tokens"]
    assert (
        doc["performance_pass"]
        and not doc["performance_failures"]
        and not doc["skipped_context_lengths"]
    )
    count, width = protocol["samples_per_round"], protocol["clone_width"]
    values = doc["measurements"] if width == 1 else doc["decode_throughput"]["samples"]
    assert len(values) == count * len(cases)
    found = {}
    for row in values:
        name, sample = row["scenario"], row["sample"]
        assert name in cases and type(sample) is int and 0 <= sample < count
        assert (name, sample) not in found
        case = cases[name]
        assert row["prompt_tokens"] == len(case["tokens"])
        outputs = [row["output_tokens"]] if width == 1 else row["lane_outputs"]
        assert len(outputs) == width
        assert all(isinstance(v, list) and 1 < len(v) <= case["cap"] for v in outputs)
        assert all(type(token) is int and token >= 0 for v in outputs for token in v)
        if width > 1:
            assert row["width"] == width and row["output_cap"] == case["cap"]
            assert row["input_token_hash"] == token_hash(case["tokens"])
            assert row["output_token_hash"] == token_hash(outputs[0])
            assert (
                row["output_tokens"]
                == row["decode_output_tokens"]
                == sum(map(len, outputs))
            )
        else:
            assert row["cache_status"] == "miss" and row["matched_tokens"] == 0
        assert all(v == outputs[0] for v in outputs), "clone lanes differ"
        found[name, sample] = outputs
    assert set(found) == {(name, sample) for name in cases for sample in range(count)}
    progress = ROOT / "results" / (ident + ".progress.jsonl")
    observed(progress)
    journal = [json.loads(line) for line in progress.read_text().splitlines()]
    starts = [v for v in journal if v.get("event") == "started"]
    identities = [v for v in journal if v.get("event") == "identity"]
    finishes = [v for v in journal if v.get("event") == "finished"]
    assert len(starts) == len(identities) == len(finishes) == 1
    assert (
        starts[0]["build_id"] == builds[arm_name]
        and starts[0]["samples_requested"] == count
    )
    assert identities[0]["identity"] == doc["identity"]
    assert (
        finishes[0]["result"] == "pass"
        and finishes[0]["performance_failure_count"] == 0
    )
    if width == 1:
        logged = {}
        for row in journal:
            if row.get("event") != "complete" or row.get("kind") != "request":
                continue
            key = row["scenario"], row["sample"]
            assert key in found and key not in logged
            assert row["input_token_hash"] == token_hash(cases[key[0]]["tokens"])
            assert row["output_token_hash"] == token_hash(row["output_tokens"])
            logged[key] = [row["output_tokens"]]
        assert logged == found
    else:
        complete = [
            v
            for v in journal
            if v.get("event") == "complete" and v.get("kind") == "decode_throughput"
        ]
        assert len(complete) == len(values)
        for event, row in zip(complete, values, strict=True):
            assert event["scenario"] == "B" + str(width)
            for key in [
                "sample",
                "width",
                "prompt_tokens",
                "decode_batches",
                "drafted_tokens",
                "accepted_draft_tokens",
            ]:
                assert event[key] == row[key]
    assert all(
        found[name, sample] == found[name, 0]
        for name in cases
        for sample in range(count)
    )
    return receipt, {name: found[name, 0] for name in cases}


def analyze(path, protocol_only=False, output=None):
    observed(Path(__file__))
    protocol, jobs, jobs_path, cases, builds = load_protocol(path)
    if protocol_only:
        print(
            json.dumps(
                {
                    "registration": str(path),
                    "jobs": len(jobs),
                    "build_ids": builds,
                    "protocol_sha256": observed(path),
                    "status": "registration_validated",
                },
                indent=2,
            )
        )
        return
    arrays, failures, receipts = {}, {}, []
    for key, job in jobs.items():
        try:
            receipt, arrays[key] = load_round(protocol, jobs_path, job, cases, builds)
            receipts.append(receipt)
        except (OSError, ValueError, AssertionError, KeyError) as error:
            failures[key] = str(error) or type(error).__name__
    chronological = all(
        dt.datetime.fromisoformat(a["end_utc"])
        <= dt.datetime.fromisoformat(b["start_utc"])
        for a, b in pairwise(receipts)
    )
    cross_group = {}
    if len(arrays) == 12:
        reference = next(iter(arrays.values()))
        for name in cases:
            matching = [
                key for key, result in arrays.items() if result[name] == reference[name]
            ]
            cross_group[name] = {
                "all_12_round_arrays_match": len(matching) == 12,
                "matching_rounds": matching,
                "array_sha256_by_round": {
                    key: hashlib.sha256(
                        json.dumps(result[name], separators=(",", ":")).encode()
                    ).hexdigest()
                    for key, result in arrays.items()
                },
            }
    analyzer_path = ROOT / "scripts" / protocol["analyzer_file"]
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "frozen_native_abba_analysis", analyzer_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frozen = module.evaluate(ROOT, protocol)
    directions = [group.get("direction") for group in frozen["groups"]]
    same_direction = len(set(directions)) == 1 and None not in directions
    all_arrays = len(cross_group) == len(cases) and all(
        v["all_12_round_arrays_match"] for v in cross_group.values()
    )
    report = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "complete" if not failures and len(arrays) == 12 else "incomplete",
        "protocol_sha256": observed(path),
        "jobs_sha256": observed(jobs_path),
        "build_ids": builds,
        "provenance_failures": failures,
        "all_registered_jobs_retained": True,
        "rounds_validated": len(arrays),
        "registered_rounds": 12,
        "nonoverlapping_registered_order": chronological,
        "cross_group_output_gate": all_arrays,
        "cross_group_case_arrays": cross_group,
        "all_group_directions_consistent": same_direction,
        "direction": directions[0] if same_direction else None,
        "primary_reportable": not failures
        and chronological
        and all_arrays
        and same_direction
        and frozen["qualified"],
        "frozen_analysis": frozen,
        "scope": "The unchanged frozen analyzer computes each registered ABBA group. This supplement checks full job/registration identity, raw AC/thermal/swap observations, nominated runtime build IDs, cohort/cap/token counts, and exact arrays across every arm and all three groups. Every registered group and failure is retained. Clone journal events contain counters without token arrays; full clone arrays come from complete stdout. Directional primary conclusions require every original group gate plus this cross-group output/provenance gate.",
    }
    for name, digest in INPUTS.items():
        assert sha(Path(name)) == digest, "Source changed during review: " + name
    report["source_sha256"] = dict(sorted(INPUTS.items()))
    output = output or ROOT / "results" / (protocol["prefix"] + ".review.json")
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(output),
                **{
                    key: report[key]
                    for key in [
                        "status",
                        "rounds_validated",
                        "provenance_failures",
                        "cross_group_output_gate",
                        "all_group_directions_consistent",
                        "primary_reportable",
                    ]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    analyze(args.protocol.resolve(), args.protocol_only, args.output)

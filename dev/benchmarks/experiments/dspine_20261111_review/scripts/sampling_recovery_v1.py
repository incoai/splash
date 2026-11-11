"""Recover the missing sampling seeds while preserving the timed-out study."""

import argparse
import ast
import datetime as dt
import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path("/Users/linson/Documents/ai/splash-dspine-13h-20261110")
PROTOCOL = ROOT / "notes/sampling-recovery-v1.protocol.json"
TAIL_ID = "sampling-recovery-tail-v1"
INTERRUPTED = "sampling4-DSpine8_control-chunk1-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def token_hash(tokens):
    value = 1469598103934665603
    for token in tokens:
        for shift in range(0, 32, 8):
            value = ((value ^ ((token >> shift) & 255)) * 1099511628211) & (
                (1 << 64) - 1
            )
    return str(value)


def cohort(path):
    result = {}
    for line in path.read_text().splitlines():
        name, cap, *tokens = line.split()
        result[name] = {"cap": int(cap), "tokens": [int(v) for v in tokens]}
    return result


def journal(path, names, expected_identity, build_id, samples):
    events = [json.loads(line) for line in path.read_text().splitlines()]
    starts = [v for v in events if v.get("event") == "started"]
    identities = [v for v in events if v.get("event") == "identity"]
    assert len(starts) == len(identities) == 1
    assert (
        starts[0]["build_id"] == build_id and starts[0]["samples_requested"] == samples
    )
    assert identities[0]["identity"] == expected_identity
    found = {}
    for row in events:
        if row.get("event") != "complete" or row.get("kind") != "request":
            continue
        name, sample, outputs = row["scenario"], row["sample"], row["output_tokens"]
        key = name, sample
        assert name in names and 0 <= sample < samples and key not in found
        assert isinstance(outputs, list) and all(
            type(v) is int and v >= 0 for v in outputs
        )
        assert len(outputs) <= names[name]["cap"]
        assert row["input_token_hash"] == token_hash(names[name]["tokens"])
        assert row["output_token_hash"] == token_hash(outputs)
        assert row["cache_status"] == "miss" and row["matched_tokens"] == 0
        found[key] = outputs
    return found


def executed(receipt, command):
    actual = receipt["job"]["command"]
    index = actual.index(command[0])
    assert actual[index:] == command, "executed command differs from frozen command"


def old_command(arm, ident, nomination):
    entry = nomination["artifact_nomination"][arm["variant"]]
    index = arm["jobs"].index(ident)
    return [
        entry["driver"],
        entry["library"],
        arm["model_root"],
        "--draft-block-size",
        str(arm["verify_rows"]),
        "--cohort",
        nomination["cohort"],
        "--samples",
        "500",
        "--temperature",
        "1",
        "--seed-base",
        str(arm["seed_start"] + index * 500),
        "--ane-ffn-share",
        "0",
        "--max-context",
        "32768",
        "--progress",
        str(ORIGINAL / "results" / (ident + ".progress.jsonl")),
    ]


def verify_artifacts(receipt, entry, nomination, barrier):
    assert receipt["artifact_sha256"] == receipt["artifact_sha256_after"]
    for key in ["driver", "library", "host_step"]:
        path = entry[key]
        assert receipt["artifact_sha256"][path] == barrier["sha256"][path]
    assert (
        receipt["artifact_sha256"][nomination["cohort"]]
        == barrier["sha256"][nomination["cohort"]]
    )


def retained_data(old_root):
    nomination = read(old_root / "notes/sampling4-nominations-v1.json")
    original_protocol = read(old_root / "notes/sampling4-protocol-v1.json")
    barrier = read(old_root / "notes/sampling4-v1.artifact-barrier.json")
    names = cohort(old_root / "cohorts/sampling4-v1.txt")
    datasets, provenance, files = {}, [], set()
    for relative in [
        "notes/sampling4-nominations-v1.json",
        "notes/sampling4-protocol-v1.json",
        "notes/sampling4-v1.artifact-barrier.json",
        "cohorts/sampling4-v1.txt",
        "results/sampling4-comparison-v1.json",
        "scripts/analyze_sampling4_v1.py",
    ]:
        files.add(relative)
    control_identity, control_build = None, None
    for arm in [nomination["control"], *nomination["candidates"]]:
        values = defaultdict(dict)
        entry = nomination["artifact_nomination"][arm["variant"]]
        for ident in arm["jobs"]:
            receipt_relative = "results/" + ident + ".receipt.json"
            progress_relative = "results/" + ident + ".progress.jsonl"
            receipt = read(old_root / receipt_relative)
            files.update([receipt_relative, progress_relative])
            executed(receipt, old_command(arm, ident, nomination))
            verify_artifacts(receipt, entry, nomination, barrier)
            if ident == INTERRUPTED:
                assert receipt["state"] == "timeout" and receipt["exit_code"] == -2
                found = journal(
                    old_root / progress_relative,
                    names,
                    control_identity,
                    control_build,
                    500,
                )
                assert set(found) == {
                    (name, sample) for name in names for sample in range(401)
                }
                seed_base = arm["seed_start"] + 500
                kind = "retained_partial_journal"
            else:
                assert receipt["state"] == "complete" and receipt["exit_code"] == 0
                relative = "results/" + ident + ".stdout"
                files.add(relative)
                document = read(old_root / relative)
                assert (
                    document["study_temperature"] == 1
                    and document["ane_ffn_share"] == 0
                )
                assert document["max_context_tokens"] == 32768
                assert document["geometry"]["verify_rows"] == arm["verify_rows"]
                assert document["identity"]["model_root"] == arm["model_root"]
                assert (
                    document["identity"]["loaded_model_layout_sha256"]
                    == arm["loaded_model_layout_sha256"]
                )
                seed_base = document["study_seed_base"]
                assert seed_base == arm["seed_start"] + arm["jobs"].index(ident) * 500
                found = {
                    (m["scenario"], m["sample"]): m["output_tokens"]
                    for m in document["measurements"]
                }
                assert len(document["measurements"]) == len(found) == 2000
                assert set(found) == {
                    (name, sample) for name in names for sample in range(500)
                }
                logged = journal(
                    old_root / progress_relative,
                    names,
                    document["identity"],
                    document["build_id"],
                    500,
                )
                assert logged == found, (
                    "retained complete journal arrays differ from original stdout"
                )
                if arm is nomination["control"]:
                    control_identity, control_build = (
                        document["identity"],
                        document["build_id"],
                    )
                kind = "retained_complete_stdout_and_matching_journal"
            for (name, sample), outputs in found.items():
                seed = seed_base + sample
                assert seed not in values[name]
                values[name][seed] = outputs
            encoded = json.dumps(
                sorted(
                    (name, sample, outputs) for (name, sample), outputs in found.items()
                ),
                separators=(",", ":"),
            ).encode()
            provenance.append(
                {
                    "job": ident,
                    "kind": kind,
                    "requests": len(found),
                    "canonical_arrays_sha256": hashlib.sha256(encoded).hexdigest(),
                }
            )
        datasets[arm["name"]] = values
    return (
        nomination,
        original_protocol,
        names,
        datasets,
        provenance,
        files,
        control_identity,
        control_build,
    )


def prepare():
    nomination, original_protocol, names, data, provenance, files, identity, build = (
        retained_data(ORIGINAL)
    )
    assert (
        read(ORIGINAL / "results/sampling4-comparison-v1.json")["status"]
        == "incomplete"
    )
    assert all(
        len(values) == (901 if arm == nomination["control"]["name"] else 1000)
        for arm, contexts in data.items()
        for values in contexts.values()
    )
    control = nomination["control"]
    entry = nomination["artifact_nomination"][control["variant"]]
    progress = ROOT / "results" / (TAIL_ID + ".progress.jsonl")
    command = [
        entry["driver"],
        entry["library"],
        control["model_root"],
        "--draft-block-size",
        "8",
        "--cohort",
        nomination["cohort"],
        "--samples",
        "99",
        "--temperature",
        "1",
        "--seed-base",
        "4100901",
        "--ane-ffn-share",
        "0",
        "--max-context",
        "32768",
        "--progress",
        str(progress),
    ]
    artifacts = [entry[k] for k in ["driver", "library", "host_step"]] + [
        nomination["cohort"]
    ]
    expected = {path: sha(Path(path)) for path in artifacts}
    barrier = read(ORIGINAL / "notes/sampling4-v1.artifact-barrier.json")["sha256"]
    assert all(expected[path] == barrier[path] for path in artifacts)
    record = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "study_kind": "Supplemental completion of seed ranges interrupted at the original fixed time cutoff",
        "source_study_root": str(ORIGINAL),
        "original_screen_status": "incomplete",
        "retained_files_sha256": {
            relative: sha(ORIGINAL / relative) for relative in sorted(files)
        },
        "recovery_analyzer_sha256": sha(Path(__file__)),
        "retained_validation": provenance,
        "control_identity": identity,
        "control_build_id": build,
        "tail": {
            "id": TAIL_ID,
            "command": command,
            "artifacts": artifacts,
            "expected_sha256": expected,
            "timeout_seconds": 1800,
            "stdout": "results/" + TAIL_ID + ".stdout",
            "receipt": "results/" + TAIL_ID + ".receipt.json",
            "progress": "results/" + TAIL_ID + ".progress.jsonl",
            "seed_start": 4100901,
            "samples": 99,
            "requests": 396,
        },
        "analysis": {
            "family_size": 280,
            "samples_per_arm_per_context": 1000,
            "method": "Execute the original frozen analyzer feature/test functions and unchanged analysis body via AST, with a provenance-checked recovered load function.",
            "complete_seed_ranges": {
                "DSpine8_control": [4100000, 4100999],
                "DSpine16_candidate": [4200000, 4200999],
                "DFlash8_candidate": [4300000, 4300999],
            },
        },
        "scope": "The original timeout receipt, incomplete result and 13-hour report stay immutable. This separately named supplemental screen restores every prespecified seed without repeating an observed request or changing sources, settings, tests or family size. It adds no throughput claim.",
    }
    with PROTOCOL.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "protocol": str(PROTOCOL),
                "protocol_sha256": sha(PROTOCOL),
                "tail_job": record["tail"],
            },
            indent=2,
        )
    )


def analyze(old_root, output):
    frozen = read(PROTOCOL)
    assert sha(Path(__file__)) == frozen["recovery_analyzer_sha256"]
    assert all(
        sha(old_root / path) == digest
        for path, digest in frozen["retained_files_sha256"].items()
    )
    nomination, original_protocol, names, datasets, sources, _, identity, build = (
        retained_data(old_root)
    )
    tail = frozen["tail"]
    receipt = read(ROOT / tail["receipt"])
    assert receipt["state"] == "complete" and receipt["exit_code"] == 0
    executed(receipt, tail["command"])
    assert receipt["artifact_sha256"] == receipt["artifact_sha256_after"]
    assert all(
        receipt["artifact_sha256"][path] == digest
        for path, digest in tail["expected_sha256"].items()
    )
    document = read(ROOT / tail["stdout"])
    assert document["identity"] == identity and document["build_id"] == build
    assert (
        document["geometry"]["verify_rows"] == 8 and document["study_temperature"] == 1
    )
    assert document["ane_ffn_share"] == 0 and document["max_context_tokens"] == 32768
    assert document["study_seed_base"] == tail["seed_start"]
    found = {
        (m["scenario"], m["sample"]): m["output_tokens"]
        for m in document["measurements"]
    }
    assert len(document["measurements"]) == len(found) == tail["requests"]
    assert set(found) == {
        (name, sample) for name in names for sample in range(tail["samples"])
    }
    assert (
        journal(ROOT / tail["progress"], names, identity, build, tail["samples"])
        == found
    )
    control = datasets[nomination["control"]["name"]]
    for (name, sample), outputs in found.items():
        seed = tail["seed_start"] + sample
        assert seed not in control[name]
        control[name][seed] = outputs
    sources.append(
        {
            "job": TAIL_ID,
            "kind": "new_missing_seed_tail",
            "requests": len(found),
            "receipt_sha256": sha(ROOT / tail["receipt"]),
            "stdout_sha256": sha(ROOT / tail["stdout"]),
            "journal_sha256": sha(ROOT / tail["progress"]),
        }
    )
    for arm in [nomination["control"], *nomination["candidates"]]:
        expected = set(range(arm["seed_start"], arm["seed_start"] + 1000))
        assert set(datasets[arm["name"]]) == set(names)
        assert all(set(values) == expected for values in datasets[arm["name"]].values())

    import numpy as np
    from scipy.stats import chi2_contingency

    original_path = old_root / "scripts/analyze_sampling4_v1.py"
    tree = ast.parse(original_path.read_text(), str(original_path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"feature", "test"}
    ]
    start = next(
        i
        for i, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "errors"
            for target in node.targets
        )
    )
    with tempfile.TemporaryDirectory(prefix="dspine-recovered-stats-") as temporary:
        work = Path(temporary)
        (work / "results").mkdir()
        namespace = {
            "Counter": Counter,
            "np": np,
            "chi2_contingency": chi2_contingency,
            "protocol": original_protocol,
            "nomination": nomination,
            "r": work,
            "json": json,
            "load": lambda arm: (
                datasets[arm["name"]],
                [
                    s
                    for s in sources
                    if s["job"] in arm["jobs"]
                    or (
                        s["job"] == TAIL_ID
                        and arm["name"] == nomination["control"]["name"]
                    )
                ],
            ),
        }
        exec(
            compile(
                ast.Module(body=functions + tree.body[start:], type_ignores=[]),
                str(original_path),
                "exec",
            ),
            namespace,
        )
        report = read(work / "results/sampling4-comparison-v1.json")
    assert report["status"] == "complete" and report["tests_observed"] == 280
    report["study_kind"] = "supplemental_completion"
    report["original_study_status"] = "incomplete"
    report["recovery_protocol_sha256"] = sha(PROTOCOL)
    report["recovery_provenance"] = sources
    report["requests_by_source"] = {
        "retained_complete_chunks": 10000,
        "retained_interrupted_journal": 1604,
        "new_missing_tail": 396,
    }
    report["interpretation"] = (
        "A supplemental complete finite regression screen using all original seed ranges. "
        "The original 13-hour study remains incomplete. The recovered screen preserves the original 280-test "
        "family and statistical code. Non-rejection does not prove distribution equality or rare-event safety."
    )
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "tests_observed": report["tests_observed"],
                "informative_tests": sum(
                    row["informative"] for row in report["results"]
                ),
                "rejections": report["rejections"],
                "output": str(output),
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "analyze"])
    parser.add_argument("--source-root", type=Path, default=ORIGINAL)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/sampling-recovery-v1.comparison.json",
    )
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        analyze(args.source_root, args.output)


if __name__ == "__main__":
    main()

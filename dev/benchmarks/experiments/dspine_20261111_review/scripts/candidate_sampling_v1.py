"""Register and analyze a new candidate against the recovered frozen control."""

import argparse
import ast
import datetime as dt
import hashlib
import importlib.util
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = Path("/Users/linson/Documents/ai/splash-dspine-13h-20261110")
PREFIX = "candidate-sampling-v1"
PROTOCOL = ROOT / "notes" / (PREFIX + ".protocol.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def recovery_module():
    path = ROOT / "scripts/sampling_recovery_v1.py"
    registered = read(ROOT / "notes/sampling-recovery-v1.protocol.json")
    assert sha(path) == registered["recovery_analyzer_sha256"]
    spec = importlib.util.spec_from_file_location("frozen_sampling_recovery", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_arrays(values):
    rows = [
        (name, seed, tokens)
        for name, seeds in sorted(values.items())
        for seed, tokens in sorted(seeds.items())
    ]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def baseline(old_root):
    helper = recovery_module()
    recovery_path = ROOT / "notes/sampling-recovery-v1.protocol.json"
    recovery = read(recovery_path)
    for relative, expected in recovery["retained_files_sha256"].items():
        assert sha(old_root / relative) == expected, (
            "original recovery input changed: " + relative
        )
    nomination, original_protocol, names, data, sources, _, identity, build = (
        helper.retained_data(old_root)
    )
    tail = recovery["tail"]
    receipt = read(ROOT / tail["receipt"])
    assert receipt["state"] == "complete" and receipt["exit_code"] == 0
    assert receipt["protocol_sha256"] == sha(recovery_path)
    helper.executed(receipt, tail["command"])
    assert receipt["artifact_sha256"] == receipt["artifact_sha256_after"]
    assert all(
        receipt["artifact_sha256"][path] == digest
        for path, digest in tail["expected_sha256"].items()
    )
    document = read(ROOT / tail["stdout"])
    assert document["identity"] == identity and document["build_id"] == build
    assert document["geometry"]["verify_rows"] == 8
    assert document["study_temperature"] == 1 and document["ane_ffn_share"] == 0
    assert (
        document["max_context_tokens"] == 32768
        and document["study_seed_base"] == 4100901
    )
    found = {
        (row["scenario"], row["sample"]): row["output_tokens"]
        for row in document["measurements"]
    }
    assert len(document["measurements"]) == len(found) == 396
    assert set(found) == {(name, sample) for name in names for sample in range(99)}
    assert helper.journal(ROOT / tail["progress"], names, identity, build, 99) == found
    control = data[nomination["control"]["name"]]
    for (name, sample), outputs in found.items():
        seed = 4100901 + sample
        assert seed not in control[name]
        control[name][seed] = outputs
    assert set(control) == set(names)
    assert all(
        set(values) == set(range(4100000, 4101000)) for values in control.values()
    )
    sources.append(
        {
            "job": helper.TAIL_ID,
            "kind": "new_missing_seed_tail",
            "requests": 396,
            "receipt_sha256": sha(ROOT / tail["receipt"]),
            "stdout_sha256": sha(ROOT / tail["stdout"]),
            "journal_sha256": sha(ROOT / tail["progress"]),
        }
    )
    result_path = ROOT / "results/sampling-recovery-v1.comparison.json"
    completion = read(result_path)
    assert completion["study_kind"] == "supplemental_completion"
    assert completion["original_study_status"] == "incomplete"
    assert completion["status"] == "complete" and completion["tests_observed"] == 280
    assert completion["family_size_reserved"] == 280
    assert completion["recovery_protocol_sha256"] == sha(recovery_path)
    assert completion["recovery_provenance"] == sources
    assert completion["requests_by_source"] == {
        "retained_complete_chunks": 10000,
        "retained_interrupted_journal": 1604,
        "new_missing_tail": 396,
    }
    assert (
        sha(old_root / "scripts/analyze_sampling4_v1.py")
        == nomination["analyzer_sha256"]
    )
    review_files = [
        "scripts/sampling_recovery_v1.py",
        "notes/sampling-recovery-v1.protocol.json",
        "results/sampling-recovery-v1.comparison.json",
        tail["receipt"],
        tail["stdout"],
        tail["progress"],
    ]
    pin = {
        "original_files_sha256": recovery["retained_files_sha256"],
        "review_files_sha256": {
            relative: sha(ROOT / relative) for relative in review_files
        },
        "canonical_control_arrays_sha256": canonical_arrays(control),
        "recovery_report_sha256": sha(result_path),
        "control_build_id": build,
        "control_identity": identity,
        "source_provenance": sources,
    }
    return helper, original_protocol, nomination, names, control, pin


def prepare(args):
    _, original, old_nomination, _, _, pin = baseline(OLD)
    original_driver = OLD / "scripts/study_driver_v2.mm"
    driver_source_sha = sha(original_driver)
    bundle, model_root = args.bundle.resolve(), args.model_root.resolve()
    paths = {
        key: bundle / filename
        for key, filename in [
            ("driver", "study-driver"),
            ("library", "splash.metallib"),
            ("host_step", "host-step.metallib"),
        ]
    }
    bundle_sha = {str(path): sha(path) for path in paths.values()}
    manifest = read(args.build_manifest)
    assert manifest["binary_sha256"] == bundle_sha[str(paths["driver"])]
    assert all(
        sha(Path(path)) == digest for path, digest in manifest["inputs"].items()
    ), "candidate build inputs differ from the build manifest"
    source_entries = [
        digest
        for path, digest in manifest["inputs"].items()
        if Path(path).name == "study_driver_v2.mm"
    ]
    assert source_entries == [driver_source_sha], (
        "candidate must use the identical sampling driver source"
    )
    assert args.layout_sha256 == old_nomination["control"]["loaded_model_layout_sha256"]
    cohort = ROOT / "cohorts/candidate-sampling4-v1.txt"
    assert not cohort.exists()
    for index in range(2):
        ident = PREFIX + "-chunk" + str(index)
        assert not list((ROOT / "results").glob(ident + ".*")), (
            "candidate observations already exist"
        )
    nomination_path = ROOT / "notes" / (PREFIX + ".nomination.json")
    barrier_path = ROOT / "notes" / (PREFIX + ".artifact-barrier.json")
    jobs_path = ROOT / "notes" / (PREFIX + ".jobs.json")
    assert not any(
        path.exists() for path in [nomination_path, PROTOCOL, barrier_path, jobs_path]
    )
    shutil.copy2(OLD / "cohorts/sampling4-v1.txt", cohort)
    assert sha(cohort) == original["cohort_sha256"]
    candidate = {
        "name": "DSpine16_final_review",
        "rows": 16,
        "seed_start": 4400000,
        "model_root": str(model_root),
        "loaded_model_layout_sha256": args.layout_sha256,
        "build_id": args.build_id,
        "source_label": args.source_label,
        "source_patch": str(args.source_patch.resolve()),
        "source_patch_sha256": sha(args.source_patch),
        "build_manifest": str(args.build_manifest.resolve()),
        "build_manifest_sha256": sha(args.build_manifest),
        "sampling_driver_source_sha256": driver_source_sha,
        "bundle": {key: str(path) for key, path in paths.items()},
        "bundle_sha256": bundle_sha,
    }
    control = {
        "name": "DSpine8_recovered_control",
        "rows": 8,
        "seed_start": 4100000,
        "build_id": pin["control_build_id"],
        "model_root": pin["control_identity"]["model_root"],
        "loaded_model_layout_sha256": pin["control_identity"][
            "loaded_model_layout_sha256"
        ],
        "kind": "Reused original complete chunk, retained partial journal and registered recovery tail",
        "canonical_arrays_sha256": pin["canonical_control_arrays_sha256"],
        "recovery_report_sha256": pin["recovery_report_sha256"],
    }
    nomination = {
        "frozen_utc": utc(),
        "prefix": PREFIX,
        "control": control,
        "candidates": [candidate],
        "baseline_reuse": "This original control also appears in the historical 280-test supplemental screen. Candidate observations use a new independent seed range and frozen source.",
        "scope": "The two candidate chunks belong to this nominated bundle only. Changed binaries require a new registration and new candidate observations.",
    }
    save(nomination_path, nomination)
    analysis = {
        key: original[key]
        for key in [
            "contexts",
            "output_cap",
            "samples_per_arm_per_context",
            "chunks_per_arm",
            "samples_per_chunk",
            "settings",
            "positions",
            "adjacent_pair_starts",
            "additional_features",
            "statistics",
        ]
    }
    analysis.update(
        {
            "frozen_utc": nomination["frozen_utc"],
            "cohort_sha256": sha(cohort),
            "family_size_reserved": 140,
            "arithmetic": "4 contexts * (24 positions + 10 adjacent pairs + 1 length) * 1 new candidate = 140 tests.",
            "statistics": original["statistics"].replace(
                "all 280 reserved tests", "all 140 reserved tests"
            ),
            "output_feature_encoding": {
                "valid_token_ids": "Every integer token ID >= 0, including token 0, remains an observed token.",
                "absorbing_stop": -1,
                "early_stop_rule": "Every position after sequence termination is STOP; do not condition on surviving to a position.",
                "empty_output_rule": "A zero-length output has length 0 and STOP at every tested position.",
            },
            "nomination_rule": "Freeze this single source and artifact bundle before its first new candidate output. Preserve both reserved chunks, all contexts and failed outcomes. Do not substitute a source, shrink the sample or repeat to pass.",
            "scope": "A distinct finite regression screen for the new nominated source against the reused recovered control.",
        }
    )
    common = [
        Path(__file__).resolve(),
        cohort,
        nomination_path,
        args.source_patch.resolve(),
        args.build_manifest.resolve(),
        ROOT / "scripts/sampling_recovery_v1.py",
        OLD / "scripts/analyze_sampling4_v1.py",
        original_driver,
        model_root / "model.json",
        model_root / "config.json",
        model_root / "draft/config.json",
        *paths.values(),
    ]
    artifact_files = sorted({str(path) for path in common})
    expected = {path: sha(Path(path)) for path in artifact_files}
    protocol = {
        "frozen_utc": utc(),
        "prefix": PREFIX,
        "analysis_protocol": analysis,
        "nomination_file": str(nomination_path),
        "nomination_sha256": sha(nomination_path),
        "barrier_file": str(barrier_path),
        "source_study_root": str(OLD),
        "baseline_pin": pin,
        "analyzer_sha256": sha(Path(__file__)),
        "original_statistical_source_sha256": sha(
            OLD / "scripts/analyze_sampling4_v1.py"
        ),
        "cohort": str(cohort),
        "cohort_sha256": sha(cohort),
        "job_keys": ["chunk0", "chunk1"],
        "candidate": candidate,
        "control": control,
        "statistics_execution": "Reuse the original frozen feature/test functions and analysis body via AST with one candidate, a 140-test family and provenance-checked load functions.",
        "failure_rule": "Every context needs exactly 1000 candidate seeds. A failed or partial chunk produces an incomplete screen. No reduced-sample substitution.",
        "scope": "This new candidate screen has distinct source identities and independent candidate seeds. Historical studies and recovery records stay immutable. No throughput or distribution-equivalence claim.",
    }
    for index in range(2):
        ident = PREFIX + "-chunk" + str(index)
        command = [
            str(paths["driver"]),
            str(paths["library"]),
            str(model_root),
            "--draft-block-size",
            "16",
            "--cohort",
            str(cohort),
            "--samples",
            "500",
            "--temperature",
            "1",
            "--seed-base",
            str(4400000 + index * 500),
            "--ane-ffn-share",
            "0",
            "--max-context",
            "32768",
            "--progress",
            str(ROOT / "results" / (ident + ".progress.jsonl")),
        ]
        protocol["chunk" + str(index)] = {
            "id": ident,
            "command": command,
            "artifacts": artifact_files,
            "expected_sha256": expected,
            "timeout_seconds": 3600,
            "stdout": "results/" + ident + ".stdout",
        }
    save(PROTOCOL, protocol)
    save(
        barrier_path,
        {
            "created_utc": utc(),
            "protocol_sha256": sha(PROTOCOL),
            "nomination_sha256": sha(nomination_path),
            "sha256": expected,
        },
    )
    save(
        jobs_path,
        {
            "protocol": str(PROTOCOL),
            "protocol_sha256": sha(PROTOCOL),
            "jobs": {key: protocol[key] for key in protocol["job_keys"]},
        },
    )
    print(
        json.dumps(
            {
                "protocol": str(PROTOCOL),
                "protocol_sha256": sha(PROTOCOL),
                "nomination": str(nomination_path),
                "barrier": str(barrier_path),
                "jobs": str(jobs_path),
                "job_keys": protocol["job_keys"],
                "family_size_reserved": 140,
            },
            indent=2,
        )
    )


def candidate_data(protocol, helper, names, nomination):
    candidate = nomination["candidates"][0]
    cases = {name: {} for name in names}
    sources, chunks, failures = [], [], []
    for key in protocol["job_keys"]:
        job = protocol[key]
        ident = job["id"]
        chunk_status = {
            "job": ident,
            "expected_requests": 2000,
            "validated_requests": 0,
        }
        try:
            receipt_path = ROOT / "results" / (ident + ".receipt.json")
            receipt = read(receipt_path)
            chunk_status.update(state=receipt["state"], exit_code=receipt["exit_code"])
            assert receipt["state"] == "complete" and receipt["exit_code"] == 0, (
                "candidate chunk failed"
            )
            assert receipt["protocol_sha256"] == sha(PROTOCOL), (
                "executed registration differs"
            )
            assert receipt["job"] == job, "executed job differs"
            assert (
                receipt["artifact_sha256"]
                == receipt["artifact_sha256_after"]
                == job["expected_sha256"]
            )
            stdout = ROOT / job["stdout"]
            doc = read(stdout)
            assert doc["build_id"] == candidate["build_id"]
            assert doc["identity"]["model_root"] == candidate["model_root"]
            assert (
                doc["identity"]["loaded_model_layout_sha256"]
                == candidate["loaded_model_layout_sha256"]
            )
            assert doc["geometry"]["verify_rows"] == 16
            assert (
                doc["study_temperature"] == 1
                and doc["ane_ffn_share"] == 0
                and doc["max_context_tokens"] == 32768
            )
            index = protocol["job_keys"].index(key)
            seed_base = 4400000 + index * 500
            assert doc["study_seed_base"] == seed_base
            found = {}
            for row in doc["measurements"]:
                name, sample, outputs = (
                    row["scenario"],
                    row["sample"],
                    row["output_tokens"],
                )
                assert (
                    name in names and 0 <= sample < 500 and (name, sample) not in found
                )
                assert type(sample) is int and isinstance(outputs, list)
                assert all(type(token) is int and token >= 0 for token in outputs)
                assert (
                    len(outputs) <= 24
                    and row["cache_status"] == "miss"
                    and row["matched_tokens"] == 0
                )
                for cycle in row.get("draft_cycles", []):
                    assert (
                        0
                        <= cycle["accepted_draft_tokens"]
                        <= cycle["drafted_tokens"]
                        <= 15
                    )
                    assert 0 <= cycle["output_tokens"] <= 17
                found[name, sample] = outputs
            assert len(doc["measurements"]) == len(found) == 2000
            assert set(found) == {
                (name, sample) for name in names for sample in range(500)
            }
            progress = ROOT / "results" / (ident + ".progress.jsonl")
            logged = helper.journal(
                progress, names, doc["identity"], doc["build_id"], 500
            )
            assert logged == found, "candidate journal differs from complete stdout"
            for (name, sample), outputs in found.items():
                seed = seed_base + sample
                assert seed not in cases[name]
                cases[name][seed] = outputs
            source = {
                "job": ident,
                "stdout_sha256": sha(stdout),
                "receipt_sha256": sha(receipt_path),
                "progress_sha256": sha(progress),
                "build_id": doc["build_id"],
                "geometry": doc["geometry"],
                "requests": 2000,
            }
            sources.append(source)
            chunk_status["validated_requests"] = 2000
        except (OSError, ValueError, AssertionError, KeyError) as error:
            failures.append(ident + ": " + (str(error) or type(error).__name__))
            chunk_status["error"] = failures[-1]
        chunks.append(chunk_status)
    complete = not failures and all(
        set(values) == set(range(4400000, 4401000)) for values in cases.values()
    )
    return cases, sources, chunks, failures, complete


def analyze(args):
    protocol = read(PROTOCOL)
    assert sha(Path(__file__)) == protocol["analyzer_sha256"]
    nomination = read(Path(protocol["nomination_file"]))
    assert sha(Path(protocol["nomination_file"])) == protocol["nomination_sha256"]
    barrier = read(Path(protocol["barrier_file"]))
    assert barrier["protocol_sha256"] == sha(PROTOCOL)
    assert barrier["nomination_sha256"] == protocol["nomination_sha256"]
    assert all(
        protocol[key]["expected_sha256"] == barrier["sha256"]
        for key in protocol["job_keys"]
    )
    assert sha(Path(protocol["cohort"])) == protocol["cohort_sha256"]
    for relative, digest in protocol["baseline_pin"]["original_files_sha256"].items():
        assert sha(args.source_root / relative) == digest, (
            "baseline source changed: " + relative
        )
    for relative, digest in protocol["baseline_pin"]["review_files_sha256"].items():
        assert sha(ROOT / relative) == digest, "baseline recovery changed: " + relative
    helper, _, _, names, control, current_pin = baseline(args.source_root)
    assert current_pin == protocol["baseline_pin"], (
        "baseline provenance differs from registration"
    )
    candidate, candidate_sources, chunks, failures, complete = candidate_data(
        protocol, helper, names, nomination
    )
    control_name = nomination["control"]["name"]
    candidate_name = nomination["candidates"][0]["name"]

    import numpy as np
    from scipy.stats import chi2_contingency

    def load(arm):
        if arm["name"] == control_name:
            return control, protocol["baseline_pin"]["source_provenance"]
        assert arm["name"] == candidate_name
        if not complete:
            raise ValueError(
                "; ".join(failures) or "incomplete candidate seed coverage"
            )
        return candidate, candidate_sources

    source_path = args.source_root / "scripts/analyze_sampling4_v1.py"
    assert sha(source_path) == protocol["original_statistical_source_sha256"]
    tree = ast.parse(source_path.read_text(), str(source_path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"feature", "test"}
    ]
    start = next(
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "errors"
            for target in node.targets
        )
    )
    with tempfile.TemporaryDirectory(
        prefix="candidate-sampling-analysis-"
    ) as directory:
        work = Path(directory)
        (work / "results").mkdir()
        namespace = {
            "Counter": Counter,
            "np": np,
            "chi2_contingency": chi2_contingency,
            "protocol": protocol["analysis_protocol"],
            "nomination": nomination,
            "r": work,
            "json": json,
            "load": load,
        }
        exec(  # noqa: S102 - executes only the SHA-pinned original statistical body.
            compile(
                ast.Module(body=functions + tree.body[start:], type_ignores=[]),
                str(source_path),
                "exec",
            ),
            namespace,
        )
        report = read(work / "results/sampling4-comparison-v1.json")
    assert report["family_size_reserved"] == 140
    assert report["tests_observed"] == (140 if complete else 0)
    report.update(
        study_kind="new_candidate_against_reused_recovered_control",
        registration_protocol_sha256=sha(PROTOCOL),
        nomination_sha256=protocol["nomination_sha256"],
        candidate_chunk_status=chunks,
        candidate_complete_requests=sum(
            chunk["validated_requests"] for chunk in chunks
        ),
        baseline_requests=4000,
        baseline_reuse=protocol["baseline_pin"],
        informative_tests=sum(result["informative"] for result in report["results"]),
        candidate_canonical_arrays_sha256=canonical_arrays(candidate)
        if complete
        else None,
    )
    report["interpretation"] = (
        "A distinct 140-test finite regression screen for this nominated source. "
        "The recovered baseline observations were also used in the earlier 280-test screen. "
        "A complete candidate covers exactly 1000 new independent seeds per context. "
        "Failed chunks remain incomplete without smaller-sample substitution. "
        "Non-rejection does not establish equivalent distributions or rare-event safety."
    )
    output = args.output or ROOT / "results" / (PREFIX + ".comparison.json")
    save(output, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in [
                    "status",
                    "family_size_reserved",
                    "tests_observed",
                    "informative_tests",
                    "rejections",
                    "candidate_complete_requests",
                    "candidate_chunk_status",
                ]
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--bundle", required=True, type=Path)
    prepare_parser.add_argument("--model-root", required=True, type=Path)
    prepare_parser.add_argument("--build-id", required=True)
    prepare_parser.add_argument("--layout-sha256", required=True)
    prepare_parser.add_argument("--source-label", required=True)
    prepare_parser.add_argument("--source-patch", required=True, type=Path)
    prepare_parser.add_argument("--build-manifest", required=True, type=Path)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--source-root", type=Path, default=OLD)
    analyze_parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()

"""Compare a candidate build with a baseline build on one installed model.

Run as ``python -m dev.benchmarks.backend_regression --baseline CHECKOUT
--package MODEL_ROOT``. Each checkout's build/ holds splash.metallib and
engine-tests/backend-benchmark. The native benchmark's decode and partial
scenarios run in ABBA order (baseline, candidate, candidate, baseline) on this
machine, which must be otherwise idle:

- outputs: every width's output_token_hash and accepted/drafted counts and
  every partial request's output tokens are identical in all four rounds. With
  --expect-output-change (or EXPECT_OUTPUT_CHANGE=1) each build must still
  repeat itself, and the candidate's acceptance rate per width may be at most
  0.02 below the baseline's.
- speed (abba.compare): decode GPU milliseconds per step for B1-B4, the GPU
  time of the 14,096-token cold prefill (partial_4k_cold) and the TTFT of its
  partial hit (partial_4k_hit).
- prepared bytes: when the builds' preparation identities differ, the
  baseline prepares into its own cache, <output dir>/baseline-weights, and
  the candidate keeps the cache its other release steps use, so neither
  re-prepares between steps; the candidate's cache must hold the bytes of
  every entry the baseline prepared from the model (prepared.compare). Equal
  identities share every entry, which then holds by construction.

The candidate's own benchmark invariants must hold too, and the baseline must
be another build: one with the candidate's build_id compares nothing. The
result is <output dir>/backend-regression.json with each round's benchmark
output beside it; the exit status is nonzero on any failure.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dev.benchmarks import abba, prepared
from dev.tests import smoke_real as smoke

ROOT = Path(__file__).resolve().parents[2]
ROUNDS = ("baseline", "candidate", "candidate", "baseline")
SCENARIOS = ("decode", "partial")
WIDTHS = (1, 2, 3, 4)
ACCEPTANCE_TOLERANCE = 0.02
BENCHMARK = Path("build/engine-tests/backend-benchmark")
METALLIB = Path("build/splash.metallib")
PARTIAL = ("partial_4k_cold", "partial_4k_seed", "partial_4k_hit")


class RegressionError(RuntimeError):
    pass


def supports_scenario_list(benchmark: Path) -> bool:
    """Whether a backend-benchmark takes a comma-separated --scenario; its
    usage, printed without arguments, says so. Older builds take one."""
    usage = subprocess.run(
        [str(benchmark)], capture_output=True, text=True, timeout=60
    ).stderr
    return "NAME[,NAME...]" in usage


def invocations(combined: bool) -> list[str]:
    """The --scenario values of one round: both scenarios in one model load
    when both builds take a list, else one load each, so both builds always
    run the same work."""
    return [",".join(SCENARIOS)] if combined else list(SCENARIOS)


def parse_document(stdout: str, returncode: int) -> dict:
    """The benchmark's JSON. It exits 1 when its own performance invariants
    fail and still prints a complete document; anything else is an error."""
    try:
        document = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RegressionError(f"benchmark exited {returncode} without JSON") from error
    if not isinstance(document, dict) or "performance_pass" not in document:
        raise RegressionError(f"benchmark exited {returncode} with incomplete JSON")
    if returncode not in (0, 1) or (returncode == 1) == document["performance_pass"]:
        raise RegressionError(
            f"benchmark exit status {returncode} disagrees with its document"
        )
    return document


def run_round(tree: Path, package: Path, round_index: int, version: str, args, env):
    documents = []
    for scenario in invocations(args.combined):
        stem = args.output_dir / (
            f"round-{round_index + 1}-{version}-{scenario.replace(',', '-')}"
        )
        command = [
            str(tree / BENCHMARK),
            str(tree / METALLIB),
            str(package),
            "--samples",
            str(args.samples),
            "--scenario",
            scenario,
            "--progress",
            str(stem.with_suffix(".progress.jsonl")),
        ]
        print(f"round {round_index + 1} {version}: {scenario}", file=sys.stderr)
        with stem.with_suffix(".log").open("w") as log:
            finished = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=log, text=True, env=env
            )
        stem.with_suffix(".json").write_text(finished.stdout)
        try:
            documents.append(parse_document(finished.stdout, finished.returncode))
        except RegressionError as error:
            log = stem.with_suffix(".log")
            tail = "".join(log.read_text(errors="replace").splitlines(True)[-3:])
            raise RegressionError(f"{stem.name}: {error} ({log}):\n{tail}") from error
    return documents


def round_record(version: str, documents: list[dict]) -> dict:
    """What one round measured: decode samples per width and partial
    requests per scenario, with the identity each load reported."""
    decode = {width: [] for width in WIDTHS}
    partial = {scenario: [] for scenario in PARTIAL}
    for document in documents:
        for sample in document.get("decode_throughput", {}).get("samples", []):
            decode[sample["width"]].append(
                {
                    "sample": sample["sample"],
                    "output_token_hash": sample["output_token_hash"],
                    "accepted_draft_tokens": sample["accepted_draft_tokens"],
                    "drafted_tokens": sample["drafted_tokens"],
                    "decode_batches": sample["decode_batches"],
                    "decode_gpu_ms": sample["decode_gpu_ms"],
                    "gpu_ms_per_step": sample["decode_gpu_ms"]
                    / sample["decode_batches"],
                }
            )
        for measurement in document.get("measurements", []):
            if measurement["scenario"] in partial:
                partial[measurement["scenario"]].append(
                    {
                        "sample": measurement["sample"],
                        "output_tokens": measurement["output_tokens"],
                        "prefill_gpu_ms": measurement["prefill_gpu_ms"],
                        "ttft_ms": measurement["ttft_ms"],
                    }
                )
    if not all(decode.values()) or not all(partial.values()):
        raise RegressionError(f"a {version} round lacks decode or partial samples")
    return {
        "version": version,
        "identities": [
            {**document["identity"], "build_id": document["build_id"]}
            for document in documents
        ],
        "performance_failures": [
            failure
            for document in documents
            for failure in document.get("performance_failures", [])
        ],
        "decode": decode,
        "partial": partial,
    }


def metrics(rounds: list[dict]) -> dict:
    """Each speed metric's samples per round, in ABBA order."""
    result = {}
    for width in WIDTHS:
        result[f"decode_B{width}_gpu_ms_per_step"] = [
            [sample["gpu_ms_per_step"] for sample in record["decode"][width]]
            for record in rounds
        ]
    result["partial_4k_cold_prefill_gpu_ms"] = [
        [request["prefill_gpu_ms"] for request in record["partial"]["partial_4k_cold"]]
        for record in rounds
    ]
    result["partial_4k_hit_ttft_ms"] = [
        [request["ttft_ms"] for request in record["partial"]["partial_4k_hit"]]
        for record in rounds
    ]
    return result


def outputs(record: dict) -> dict:
    """What must repeat: per width and sample the output hash and draft
    counts, per partial request its output tokens."""
    result = {}
    for width, samples in record["decode"].items():
        for sample in samples:
            result[f"B{width} sample {sample['sample']}"] = (
                sample["output_token_hash"],
                sample["accepted_draft_tokens"],
                sample["drafted_tokens"],
            )
    for scenario, requests in record["partial"].items():
        for request in requests:
            result[f"{scenario} sample {request['sample']}"] = tuple(
                request["output_tokens"]
            )
    return result


def acceptance(records: list[dict], width: int) -> float:
    accepted = sum(
        sample["accepted_draft_tokens"]
        for record in records
        for sample in record["decode"][width]
    )
    drafted = sum(
        sample["drafted_tokens"]
        for record in records
        for sample in record["decode"][width]
    )
    if not drafted:
        raise RegressionError(f"B{width} drafted no tokens")
    return accepted / drafted


def differences(first: dict, second: dict) -> list[str]:
    return sorted(
        key for key in first.keys() | second.keys() if first.get(key) != second.get(key)
    )


def summarize(rounds: list[dict], expect_output_change: bool) -> dict:
    """The comparison of four rounds in ABBA order: failures of outputs,
    identity and invariants, and the speed verdict per metric."""
    if [record["version"] for record in rounds] != list(ROUNDS):
        raise RegressionError("rounds are not in ABBA order")
    failures = []
    baseline, candidate = [rounds[0], rounds[3]], [rounds[1], rounds[2]]
    builds = {}
    for name, records in (("baseline", baseline), ("candidate", candidate)):
        identities = [
            {
                key: identity.get(key)
                for key in ("build_id", "loaded_model_layout_sha256")
            }
            for record in records
            for identity in record["identities"]
        ]
        if any(identity != identities[0] for identity in identities):
            failures.append(
                f"the {name} build or its loaded model changed between rounds"
            )
        builds[name] = identities[0]["build_id"]
        if changed := differences(outputs(records[0]), outputs(records[1])):
            failures.append(f"the {name} build did not repeat its outputs: {changed}")
    if builds["baseline"] == builds["candidate"]:
        failures.append(
            f"the baseline has the candidate's build_id {builds['baseline']}"
        )
    places = {
        (identity.get("model_root"), identity.get("device"))
        for record in rounds
        for identity in record["identities"]
    }
    if len(places) != 1:
        failures.append(f"rounds ran different models or devices: {sorted(places)}")
    rates = {
        f"B{width}": {
            "baseline": acceptance(baseline, width),
            "candidate": acceptance(candidate, width),
        }
        for width in WIDTHS
    }
    if expect_output_change:
        for width, rate in rates.items():
            if rate["candidate"] < rate["baseline"] - ACCEPTANCE_TOLERANCE:
                failures.append(
                    f"{width} acceptance fell from {rate['baseline']:.4f} "
                    f"to {rate['candidate']:.4f}"
                )
    elif changed := differences(outputs(rounds[0]), outputs(rounds[1])):
        failures.append(f"candidate outputs or acceptance differ: {changed}")
    if invariants := sorted(
        {f for record in candidate for f in record["performance_failures"]}
    ):
        failures.append(f"candidate benchmark invariants failed: {invariants}")
    speed = {
        name: abba.compare_samples(samples) for name, samples in metrics(rounds).items()
    }
    for name, result in speed.items():
        if not result["pass"]:
            failures.append(f"{name}: {abba.describe(result)}")
    return {
        "expect_output_change": expect_output_change,
        "acceptance": rates,
        "baseline_performance_failures": sorted(
            {f for record in baseline for f in record["performance_failures"]}
        ),
        "speed": speed,
        "failures": failures,
        "pass": not failures,
    }


def package_slug(package: Path) -> str:
    """The name of a package's results under build/release: its path below
    the models directory, or its own name elsewhere."""
    path = Path(os.path.abspath(package))
    models = Path(os.path.abspath(smoke.model_artifacts.MODELS))
    try:
        return "--".join(path.relative_to(models).parts)
    except ValueError:
        return path.name


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--baseline", required=True, type=Path, help="baseline checkout"
    )
    parser.add_argument(
        "--candidate", type=Path, default=ROOT, help="candidate checkout (this one)"
    )
    parser.add_argument(
        "--package", required=True, type=Path, help="installed model root"
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument(
        "--output-dir", type=Path, help="results directory (build/release/<package>)"
    )
    parser.add_argument(
        "--expect-output-change",
        action="store_true",
        default=os.environ.get("EXPECT_OUTPUT_CHANGE", "") not in ("", "0"),
        help="allow changed outputs with acceptance within 0.02 (EXPECT_OUTPUT_CHANGE=1)",
    )
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error("--samples must be positive")
    for tree in (args.baseline, args.candidate):
        for path in (tree / BENCHMARK, tree / METALLIB):
            if not path.is_file():
                parser.error(f"missing retained benchmark or library: {path}")
    if prepared.preparation_identity(args.candidate / "build") is None:
        parser.error(f"the candidate build has no {prepared.IDENTITY_HEADER}")
    args.kind = smoke.model_artifacts.installation_kind(args.package)
    if args.kind is None:
        parser.error(f"missing installed model: {args.package}")
    if args.output_dir is None:
        args.output_dir = ROOT / "build/release" / package_slug(args.package)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    smoke.hold_package(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trees = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    shared = prepared.preparation_identity(
        trees["baseline"] / "build"
    ) == prepared.preparation_identity(trees["candidate"] / "build")
    environments = {"baseline": dict(os.environ), "candidate": dict(os.environ)}
    if not shared:
        environments["baseline"].update(prepared.baseline_environment(args.output_dir))
    args.combined = all(
        supports_scenario_list(tree / BENCHMARK) for tree in trees.values()
    )
    document = {
        "schema_version": 1,
        "timing": "native GPU time and TTFT; ABBA rule of dev/benchmarks/abba.py",
        "package": str(args.package),
        "trees": {name: str(tree) for name, tree in trees.items()},
        "samples": args.samples,
        "scenario_invocations": invocations(args.combined),
        "pass": False,
    }
    try:
        rounds = [
            round_record(
                version,
                run_round(
                    trees[version],
                    args.package,
                    index,
                    version,
                    args,
                    environments[version],
                ),
            )
            for index, version in enumerate(ROUNDS)
        ]
        document["rounds"] = rounds
        document["comparison"] = summarize(rounds, args.expect_output_change)
        document["prepared"] = (
            {"shared_identity": True, "pass": True}
            if shared
            else {
                "shared_identity": False,
                **prepared.compare(
                    prepared.cache_root(environments["baseline"]),
                    prepared.cache_root(environments["candidate"]),
                    package=args.package,
                    required=args.kind == smoke.model_artifacts.ASSEMBLY,
                ),
            }
        )
        document["pass"] = (
            document["comparison"]["pass"] and document["prepared"]["pass"]
        )
    except Exception as error:
        document["error"] = str(error)
        raise
    finally:
        output = args.output_dir / "backend-regression.json"
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(document, indent=2) + "\n")
        temporary.replace(output)
    report(document)
    return 0 if document["pass"] else 1


def report(document: dict) -> None:
    comparison = document["comparison"]
    for name, result in comparison["speed"].items():
        rounds = " ".join(f"{value:.2f}" for value in result["rounds"])
        print(f"{name}: {abba.describe(result)} (ABBA {rounds})")
    for width, rate in comparison["acceptance"].items():
        print(
            f"{width} acceptance: baseline {rate['baseline']:.4f}, "
            f"candidate {rate['candidate']:.4f}"
        )
    prepared_bytes = document["prepared"]
    print(
        "prepared bytes: "
        + ("shared identity" if prepared_bytes.get("shared_identity") else "compared")
        + (" PASS" if prepared_bytes["pass"] else f" FAIL {prepared_bytes['failures']}")
    )
    for failure in comparison["failures"]:
        print(f"FAIL: {failure}")
    print(f"backend regression: {'PASS' if document['pass'] else 'FAIL'}")


if __name__ == "__main__":
    raise SystemExit(main())

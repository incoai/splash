"""Compare a candidate build with a baseline build on one installed model.

Run as ``python -m dev.benchmarks.backend_regression --baseline CHECKOUT
--model-root MODEL_ROOT``. Each checkout's build/ holds splash.metallib and
engine-tests/backend-benchmark, the candidate's engine-tests/weight-digests
too. The native benchmark's decode, partial, short and transcripts scenarios
(short and transcripts when both builds take them) run in ABBA order
(baseline, candidate, candidate, baseline) on this machine, which must be
otherwise idle:

- outputs: every width's output_token_hash and accepted/drafted counts and
  every partial, short and transcript request's output tokens are identical in
  all four rounds. The decode requests ask for counting, and the partial and
  short ones for a token after random tokens, which a change in the target's
  arithmetic seldom moves; the transcripts, greedy answers of 512 tokens to
  real requests, move with it. With --expect-output-change (or
  EXPECT_OUTPUT_CHANGE=1) each build must still repeat itself, and the
  candidate's acceptance rate per width may be at most 0.02 below the
  baseline's.
- the prefill FFN's Neural Engine split: with --ane-ffn-share every round of
  a build that takes the option runs that share and must report it, and a
  build that does not must report none. Without it startup calibrates the
  split from timings, so the first share and least chunk rows a round reports
  are those every later round runs when its build takes the options.
- speed (abba.compare): decode GPU milliseconds per step for B1-B4, the GPU
  time of the 14,096-token cold prefill (partial_4k_cold), the TTFT of its
  partial hit (partial_4k_hit) and the GPU time of each short cold prefill
  (short_<rows>_rows_prefill_gpu_ms, by the rows of its first chunk).
- memory plan: the candidate's plan holds no less context without the Neural
  Engine split than the baseline's holds without it, what a baseline before
  the split serves; the split takes its memory from the KV cache. The context
  each serves, the candidate's what its plan holds with the split it
  calibrated, and the change of the plan's elastic state/KV budget are
  reported.
- weight bytes: after the rounds both builds load the model's weight images
  once more and must hold the same images with the same bytes
  (weights.compare_builds). A baseline of an earlier release prepares into a
  cache of its own, <output dir>/baseline-weights.

The candidate's own benchmark invariants must hold too, and the baseline must
be another build: one with the candidate's build_id compares nothing. The
result is <output dir>/backend-regression.json with each round's benchmark
output beside it; the exit status is nonzero on any failure.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from dev.benchmarks import abba, weights
from dev.tests import smoke_real as smoke

ROOT = Path(__file__).resolve().parents[2]
ROUNDS = ("baseline", "candidate", "candidate", "baseline")
SCENARIOS = ("decode", "partial", "short", "transcripts")
# The scenarios a build may lack, run only when both builds take them.
OPTIONAL_SCENARIOS = ("short", "transcripts")
WIDTHS = (1, 2, 3, 4)
ACCEPTANCE_TOLERANCE = 0.02
BENCHMARK = Path("build/engine-tests/backend-benchmark")
METALLIB = Path("build/splash.metallib")
PARTIAL = ("partial_4k_cold", "partial_4k_seed", "partial_4k_hit")
# The rows of the first chunk of the short scenario's cold prefills, whose
# prompts hold a token more.
SHORT = (480, 512, 544, 640, 672, 1024, 1536, 2016)
# The transcripts scenario's requests, one per case of the benchmark's.
TRANSCRIPTS = tuple(f"transcript_{case}" for case in range(8))
# The memory plan a load served, as the benchmark reports it.
MEMORY_PLAN = (
    "max_context_tokens",
    "no_ane_context_tokens",
    "dynamic_budget_bytes",
    "ane_ffn_bytes",
)


class RegressionError(RuntimeError):
    pass


def usage(benchmark: Path) -> str:
    """A backend-benchmark's usage, which it prints without arguments. It
    names the options a build takes: older builds take one --scenario and no
    short or transcripts scenario, and builds before the Neural Engine split no
    --ane-ffn-share."""
    return subprocess.run(
        [str(benchmark)], capture_output=True, text=True, timeout=60
    ).stderr


def ane_ffn_share(value: str) -> float:
    """An --ane-ffn-share argument: a share in [0, 1)."""
    share = float(value)
    if not 0 <= share < 1:
        raise argparse.ArgumentTypeError(f"{value} is not a share in [0, 1)")
    return share


def pin_ane_ffn(args, version: str, document: dict) -> None:
    """Holds every round to one Neural Engine split, the given share or else
    the first share and least chunk rows a round reports, and fails a round
    that reported another. With a share given, a build that does not take
    --ane-ffn-share runs without it and must report none."""
    share = document.get("ane_ffn_share")
    if args.ane_ffn_share_given:
        if not args.takes_ane_ffn_share[version]:
            if share is not None:
                raise RegressionError(
                    f"takes no --ane-ffn-share but reported the share {share}"
                )
        elif share != args.ane_ffn_share:
            raise RegressionError(
                f"ran the Neural Engine split at share {share}, "
                f"not the given {args.ane_ffn_share}"
            )
        return
    if share is None:
        return
    rows = document.get("ane_ffn_minimum_rows")
    if args.ane_ffn_share is None:
        args.ane_ffn_share, args.ane_ffn_minimum_rows = share, rows
    elif (share, rows) != (args.ane_ffn_share, args.ane_ffn_minimum_rows):
        raise RegressionError(
            f"ran the Neural Engine split at share {share} from {rows} rows, "
            f"not the first round's {args.ane_ffn_share} from "
            f"{args.ane_ffn_minimum_rows}"
        )


def invocations(combined: bool, scenarios) -> list[str]:
    """The --scenario values of one round: its scenarios in one model load
    when both builds take a list, else one load each, so both builds always
    run the same work."""
    return [",".join(scenarios)] if combined else list(scenarios)


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


def run_round(tree: Path, model_root: Path, round_index: int, version: str, args, env):
    documents = []
    for scenario in invocations(args.combined, args.scenarios):
        stem = args.output_dir / (
            f"round-{round_index + 1}-{version}-{scenario.replace(',', '-')}"
        )
        command = [
            str(tree / BENCHMARK),
            str(tree / METALLIB),
            str(model_root),
            "--samples",
            str(args.samples),
            "--scenario",
            scenario,
            "--progress",
            str(stem.with_suffix(".progress.jsonl")),
        ]
        # A first round that ran no split pins none: later rounds take the
        # outcome the build remembered, and the check holds them to it.
        if (
            args.ane_ffn_share_given or args.ane_ffn_share
        ) and args.takes_ane_ffn_share[version]:
            command += ["--ane-ffn-share", repr(args.ane_ffn_share)]
            if args.ane_ffn_minimum_rows:
                command += ["--ane-ffn-minimum-rows", str(args.ane_ffn_minimum_rows)]
        print(f"round {round_index + 1} {version}: {scenario}", file=sys.stderr)
        with stem.with_suffix(".log").open("w") as log:
            finished = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=log, text=True, env=env
            )
        stem.with_suffix(".json").write_text(finished.stdout)
        try:
            document = parse_document(finished.stdout, finished.returncode)
            pin_ane_ffn(args, version, document)
            documents.append(document)
        except RegressionError as error:
            log = stem.with_suffix(".log")
            tail = "".join(log.read_text(errors="replace").splitlines(True)[-3:])
            raise RegressionError(f"{stem.name}: {error} ({log}):\n{tail}") from error
    return documents


def round_record(version: str, documents: list[dict], scenarios) -> dict:
    """What one round of these scenarios measured: decode samples per width,
    partial and transcript requests per scenario and short requests per first
    chunk, with the identity and memory plan each load reported."""
    decode = {width: [] for width in WIDTHS}
    partial = {scenario: [] for scenario in PARTIAL}
    short = {rows: [] for rows in SHORT} if "short" in scenarios else {}
    transcripts = (
        {name: [] for name in TRANSCRIPTS} if "transcripts" in scenarios else {}
    )
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
            request = {
                "sample": measurement["sample"],
                "output_tokens": measurement["output_tokens"],
                "prefill_gpu_ms": measurement["prefill_gpu_ms"],
                "ttft_ms": measurement["ttft_ms"],
            }
            if measurement["scenario"] in partial:
                partial[measurement["scenario"]].append(request)
            elif measurement["scenario"] in transcripts:
                transcripts[measurement["scenario"]].append(request)
            elif (
                measurement["scenario"] == "short"
                and measurement["prompt_tokens"] - 1 in short
            ):
                short[measurement["prompt_tokens"] - 1].append(request)
    measured = {
        "decode": decode,
        "partial": partial,
        "short": short,
        "transcripts": transcripts,
    }
    if lacking := [name for name, found in measured.items() if not all(found.values())]:
        raise RegressionError(
            f"a {version} round lacks {' and '.join(lacking)} samples"
        )
    return {
        "version": version,
        "identities": [
            {**document["identity"], "build_id": document["build_id"]}
            for document in documents
        ],
        "memory_plans": [
            {key: document.get(key) for key in MEMORY_PLAN} for document in documents
        ],
        "performance_failures": [
            failure
            for document in documents
            for failure in document.get("performance_failures", [])
        ],
        **measured,
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
    for rows in rounds[0]["short"]:
        result[f"short_{rows}_rows_prefill_gpu_ms"] = [
            [request["prefill_gpu_ms"] for request in record["short"][rows]]
            for record in rounds
        ]
    return result


def outputs(record: dict) -> dict:
    """What must repeat: per width and sample the output hash and draft
    counts, per partial, short and transcript request its output tokens."""
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
    for rows, requests in record["short"].items():
        for request in requests:
            result[f"short {rows} rows sample {request['sample']}"] = tuple(
                request["output_tokens"]
            )
    for name, requests in record["transcripts"].items():
        for request in requests:
            result[f"{name} sample {request['sample']}"] = tuple(
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
    identity, memory plan and invariants, and the speed verdict per
    metric."""
    if [record["version"] for record in rounds] != list(ROUNDS):
        raise RegressionError("rounds are not in ABBA order")
    failures = []
    baseline, candidate = [rounds[0], rounds[3]], [rounds[1], rounds[2]]
    builds, plans = {}, {}
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
        loads = [plan for record in records for plan in record["memory_plans"]]
        if any(plan != loads[0] for plan in loads):
            failures.append(f"the {name} build's memory plan changed between rounds")
        plans[name] = loads[0]
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
    # What an older build's benchmark does not report is None, and not
    # compared; one before the Neural Engine split serves its plan's context.
    context = {
        name: plan["max_context_tokens"]
        if plan.get("no_ane_context_tokens") is None
        else plan["no_ane_context_tokens"]
        for name, plan in plans.items()
    }
    if None not in context.values() and context["candidate"] < context["baseline"]:
        failures.append(
            f"the candidate holds {context['candidate']} tokens of context "
            f"without the Neural Engine split, less than the baseline's "
            f"{context['baseline']}"
        )
    budget = {name: plan["dynamic_budget_bytes"] for name, plan in plans.items()}
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
        "memory_plan": {
            **plans,
            "dynamic_budget_change_bytes": None
            if None in budget.values()
            else budget["candidate"] - budget["baseline"],
        },
        "baseline_performance_failures": sorted(
            {f for record in baseline for f in record["performance_failures"]}
        ),
        "speed": speed,
        "failures": failures,
        "pass": not failures,
    }


def results_slug(model_root: Path) -> str:
    """The name of a model root's results under build/release: its path
    below the models directory, or its own name elsewhere."""
    path = Path(os.path.abspath(model_root))
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
        "--model-root", required=True, type=Path, help="installed model root"
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument(
        "--output-dir", type=Path, help="results directory (build/release/<model root>)"
    )
    parser.add_argument(
        "--expect-output-change",
        action="store_true",
        default=os.environ.get("EXPECT_OUTPUT_CHANGE", "") not in ("", "0"),
        help="allow changed outputs with acceptance within 0.02 (EXPECT_OUTPUT_CHANGE=1)",
    )
    parser.add_argument(
        "--ane-ffn-share",
        type=ane_ffn_share,
        metavar="SHARE",
        help="the prefill FFN's Neural Engine share in [0, 1) every round runs "
        "(0: the GPU alone; default: the first share a round reports)",
    )
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error("--samples must be positive")
    for tree in (args.baseline, args.candidate):
        for path in (tree / BENCHMARK, tree / METALLIB):
            if not path.is_file():
                parser.error(f"missing retained benchmark or library: {path}")
    if not weights.loads_in_memory(args.candidate / "build"):
        parser.error(f"the candidate build has no {weights.WEIGHT_DIGESTS}")
    if (
        smoke.model_artifacts.installation_kind(args.model_root)
        != smoke.model_artifacts.ASSEMBLY
    ):
        parser.error(f"missing installed model: {args.model_root}")
    if args.output_dir is None:
        args.output_dir = ROOT / "build/release" / results_slug(args.model_root)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    smoke.hold_model_root(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trees = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    environments = {"baseline": dict(os.environ), "candidate": dict(os.environ)}
    if not weights.loads_in_memory(trees["baseline"] / "build"):
        environments["baseline"].update(weights.baseline_environment(args.output_dir))
    usages = {name: usage(tree / BENCHMARK) for name, tree in trees.items()}
    args.combined = all("NAME[,NAME...]" in text for text in usages.values())
    # Both builds run an optional scenario only when both take it.
    args.scenarios = [
        name
        for name in SCENARIOS
        if name not in OPTIONAL_SCENARIOS
        or all(re.search(rf"\b{name}\b", text) for text in usages.values())
    ]
    args.takes_ane_ffn_share = {
        name: "--ane-ffn-share" in text for name, text in usages.items()
    }
    args.ane_ffn_share_given = args.ane_ffn_share is not None
    args.ane_ffn_minimum_rows = None
    document = {
        "schema_version": 1,
        "timing": "native GPU time and TTFT; ABBA rule of dev/benchmarks/abba.py",
        "model_root": str(args.model_root),
        "trees": {name: str(tree) for name, tree in trees.items()},
        "samples": args.samples,
        "scenario_invocations": invocations(args.combined, args.scenarios),
        "pass": False,
    }
    try:
        rounds = [
            round_record(
                version,
                run_round(
                    trees[version],
                    args.model_root,
                    index,
                    version,
                    args,
                    environments[version],
                ),
                args.scenarios,
            )
            for index, version in enumerate(ROUNDS)
        ]
        document["rounds"] = rounds
        document["ane_ffn_share"] = args.ane_ffn_share
        document["ane_ffn_minimum_rows"] = args.ane_ffn_minimum_rows
        document["comparison"] = summarize(rounds, args.expect_output_change)
        document["weights"] = weights.compare_builds(
            trees["baseline"] / "build",
            trees["candidate"] / "build",
            args.model_root,
            environments["baseline"],
        )
        document["pass"] = (
            document["comparison"]["pass"] and document["weights"]["pass"]
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
    for name, what in (("short", "short prompts"), ("transcripts", "transcripts")):
        if not any(name in scenario for scenario in document["scenario_invocations"]):
            print(f"{what}: not compared, a build's backend-benchmark lacks them")
    for width, rate in comparison["acceptance"].items():
        print(
            f"{width} acceptance: baseline {rate['baseline']:.4f}, "
            f"candidate {rate['candidate']:.4f}"
        )
    plans = comparison["memory_plan"]
    print(
        f"context: baseline {plans['baseline']['max_context_tokens']}, "
        f"candidate {plans['candidate']['max_context_tokens']} tokens "
        f"({plans['candidate']['no_ane_context_tokens']} without the Neural "
        f"Engine split); "
        f"elastic budget change: {plans['dynamic_budget_change_bytes']} bytes"
    )
    images = document["weights"]
    print(
        f"weight bytes: {len(images['images'])} images "
        + ("PASS" if images["pass"] else f"FAIL {images['failures']}")
    )
    for failure in comparison["failures"]:
        print(f"FAIL: {failure}")
    print(f"backend regression: {'PASS' if document['pass'] else 'FAIL'}")


if __name__ == "__main__":
    raise SystemExit(main())

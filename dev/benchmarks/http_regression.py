"""Compare retained servers with matched HTTP requests in ABBA order.

Run as ``python -m dev.benchmarks.http_regression --baseline-binary PATH``.
Reuses the real HTTP test lifecycle; never contacts an existing server.
With ``--burst N`` it measures bursts of N concurrent requests instead, and
with ``--follow-up`` each conversation's next turn after its burst.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path

from dev.benchmarks import abba
from dev.benchmarks import prepared as prepared_weights
from dev.tests import smoke_real as smoke


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prompts(tokenizer, contexts: list[int], samples: int, nonce: str) -> dict:
    result = {}
    filler = tokenizer.encode(
        " The archive contains routine project notes and implementation details",
        add_special_tokens=False,
    )
    for sample in range(samples):
        for context in contexts:
            prefix = digest([nonce, sample, context]) + "\n"
            suffix = "\nWrite the integers from 1 to 200, one per line."
            rows = context
            for _ in range(8):
                content = (
                    prefix
                    + tokenizer.decode((filler * (rows // len(filler) + 1))[:rows])
                    + suffix
                )
                count = len(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": content}],
                        tokenize=True,
                        return_dict=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
                if count == context:
                    break
                rows -= count - context
            else:
                raise ValueError(f"cannot construct an exact {context}-token prompt")
            result[sample, context] = content
    return result


def measure(server, model, content, output_tokens, scenario, context, timeout):
    _, before = smoke.request(server.port, "GET", "/status")
    started = time.monotonic()
    code, response = smoke.request(
        server.port,
        "POST",
        "/v1/chat/completions",
        smoke.chat_body(model, content, max_completion_tokens=output_tokens),
        timeout=timeout,
    )
    elapsed = (time.monotonic() - started) * 1000
    smoke.require(code == 200, f"{scenario}: HTTP {code}: {response!r}")
    _, after = smoke.request(server.port, "GET", "/status")
    smoke.validate_status(after)
    metrics, usage = response["metrics"], response["usage"]
    smoke.require(usage["prompt_tokens"] == context, "prompt token count changed")
    reused = metrics["cache"]["matched_tokens"]
    prefill = metrics["prefill"]["tokens"]
    if scenario == "exact":
        smoke.require(0 < reused < context and prefill < context, "exact hit missing")
    else:
        smoke.require(reused == 0 and prefill == context, "cold prompt reused cache")
    if scenario == "decode":
        smoke.require(
            usage["completion_tokens"] == output_tokens,
            "decode sample ended before the requested output length",
        )
    for phase in (
        "queued",
        "waiting_resources",
        "prefilling",
        "decoding",
        "waiting_mask",
    ):
        smoke.require(after["scheduler"][phase] == 0, f"request left {phase} work")
    smoke.require(
        after["state"]["active_lanes"] == 0 and after["kv"]["pages_active"] == 0,
        "active resources leaked",
    )
    keys = [
        "prefill_wall_ms",
        "decode_wall_ms",
        "prefill_input_tokens",
        "decode_output_tokens",
        "drafted_tokens",
        "accepted_draft_tokens",
    ]
    # A build older than the engine's decode cycle timing does not report it.
    if "decode_cycle_ms" in after["metrics"]:
        keys.append("decode_cycle_ms")
    delta = {key: after["metrics"][key] - before["metrics"][key] for key in keys}
    return {
        "scenario": scenario,
        "context": context,
        "prompt_sha256": digest(content),
        "response_sha256": digest(response["choices"][0]["message"]),
        "usage": usage,
        "metrics": metrics,
        "native_delta": delta,
        "http_wall_ms": elapsed,
    }


# A burst's requests are short, so their replay points come due together.
BURST_OUTPUT_TOKENS = 16
FOLLOW_UP = "Continue with the next ten integers."


def measure_burst(server, model, contents, follow_up, timeout):
    """N chat requests of equal length at once, then, with follow_up, each
    conversation's next turn one after another: the publication failures of
    the burst's replay points, its decode rate, and whether each next turn
    resumed at its conversation's replay point."""
    _, before = smoke.request(server.port, "GET", "/status")
    responses = [None] * len(contents)

    def send(index):
        responses[index] = smoke.request(
            server.port,
            "POST",
            "/v1/chat/completions",
            smoke.chat_body(
                model, contents[index], max_completion_tokens=BURST_OUTPUT_TOKENS
            ),
            timeout=timeout,
        )

    threads = [threading.Thread(target=send, args=(i,)) for i in range(len(contents))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for code, response in responses:
        smoke.require(code == 200, f"burst: HTTP {code}: {response!r}")
    _, after = smoke.request(server.port, "GET", "/status")
    smoke.validate_status(after)
    decode_ms = after["metrics"]["decode_wall_ms"] - before["metrics"]["decode_wall_ms"]
    decoded = (
        after["metrics"]["decode_output_tokens"]
        - before["metrics"]["decode_output_tokens"]
    )
    failures = "replay_state_publication_failures"
    row = {
        "scenario": "burst",
        "replay_state_publication_failures": after["cache"][failures]
        - before["cache"][failures],
        "decode_ms_per_token": decode_ms / max(1, decoded),
        "maximum_context_tokens": after["maximum_context_tokens"],
        "follow_ups": [],
    }
    if not follow_up:
        return row
    page_tokens = after["memory_plan"]["budget"]["kv_page_tokens"]
    for content, (_, response) in zip(contents, responses):
        answer = response["choices"][0]["message"]["content"]
        body = smoke.chat_body(model, FOLLOW_UP, max_completion_tokens=1)
        body["messages"] = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": FOLLOW_UP},
        ]
        code, turn = smoke.request(
            server.port, "POST", "/v1/chat/completions", body, timeout=timeout
        )
        smoke.require(code == 200, f"follow-up: HTTP {code}: {turn!r}")
        # The replay point lies in the first turn's last page or the page
        # before its generation prompt; any other match is a checkpoint at
        # least a page below it.
        matched = turn["metrics"]["cache"]["matched_tokens"]
        resumed = matched >= response["usage"]["prompt_tokens"] - 2 * page_tokens
        row["follow_ups"].append(
            {
                "ttft_ms": turn["metrics"]["request_latency"]["ttft_ms"],
                "matched_tokens": matched,
                "resumed": resumed,
            }
        )
    return row


ROUNDS = ("baseline", "candidate", "candidate", "baseline")


def summarize(records: list[dict]) -> list[dict]:
    """Per context and scenario, the ABBA verdict (abba.compare) of the
    median latency of each of the four rounds; the matched transcripts of
    both versions must be identical."""
    groups = defaultdict(lambda: defaultdict(list))
    outputs = {}
    # Decode is judged by the engine's cycle, so host work between commands
    # counts, unless a version does not report it: then both versions are
    # judged by the GPU command's wall, never one metric against the other.
    decode_metric = (
        "decode_cycle_ms"
        if all(
            "decode_cycle_ms" in row["native_delta"]
            for row in records
            if row["scenario"] == "decode"
        )
        else "decode_wall_ms"
    )
    for row in records:
        key = row["sample"], row["context"], row["scenario"]
        output = (
            row["prompt_sha256"],
            row["response_sha256"],
            row["usage"]["completion_tokens"],
        )
        paired = outputs.setdefault(key, {})
        if row["version"] in paired:
            raise ValueError(f"duplicate performance sample: {key}")
        if paired and next(iter(paired.values())) != output:
            raise ValueError(f"baseline/candidate transcript differs: {key}")
        paired[row["version"]] = output
        if row["scenario"] == "decode":
            latency = row["native_delta"][decode_metric] / max(
                1, row["native_delta"]["decode_output_tokens"]
            )
        else:
            latency = row["metrics"]["request_latency"]["ttft_ms"]
        groups[row["context"], row["scenario"]][row["round"]].append(latency)
    results = []
    if any(set(pair) != {"baseline", "candidate"} for pair in outputs.values()):
        raise ValueError("unpaired request samples")
    for (context, scenario), rounds in sorted(groups.items()):
        if sorted(rounds) != list(range(len(ROUNDS))):
            raise ValueError(f"a round has no samples: {context} {scenario}")
        comparison = abba.compare_samples(rounds[index] for index in range(len(ROUNDS)))
        results.append(
            {
                "context": context,
                "scenario": scenario,
                "metric": (
                    f"{decode_metric}_per_token" if scenario == "decode" else "ttft_ms"
                ),
                "samples_per_round": [
                    len(rounds[index]) for index in range(len(ROUNDS))
                ],
                "baseline_median": statistics.median(rounds[0] + rounds[3]),
                "candidate_median": statistics.median(rounds[1] + rounds[2]),
                **comparison,
            }
        )
    return results


def summarize_bursts(records: list[dict]) -> list[dict]:
    """Per context, each version's replay-point publication failures per
    burst, the next turns that resumed at their replay point and the
    advertised context, and the ABBA verdicts of the bursts' decode time per
    token and of the next turns' time to first token, when they were sent.
    The candidate is kept when it publishes every replay point where the
    baseline lost some, a conclusive verdict finds its decode at most 2%
    slower, and next turns were sent and start at least 15% sooner with no
    round overlapping the baseline's."""
    groups = defaultdict(list)
    for row in records:
        groups[row["context"]].append(row)
    results = []
    for context, rows in sorted(groups.items()):
        versions = {}
        for version in ("baseline", "candidate"):
            bursts = [row for row in rows if row["version"] == version]
            follow_ups = [turn for row in bursts for turn in row["follow_ups"]]
            versions[version] = {
                "bursts": len(bursts),
                "failures_per_burst": statistics.mean(
                    row["replay_state_publication_failures"] for row in bursts
                ),
                "resumed_follow_ups": sum(turn["resumed"] for turn in follow_ups),
                "follow_ups": len(follow_ups),
                "maximum_context_tokens": min(
                    row["maximum_context_tokens"] for row in bursts
                ),
            }

        def per_round(sample):
            return [
                [
                    value
                    for row in rows
                    if row["round"] == index
                    for value in sample(row)
                ]
                for index in range(len(ROUNDS))
            ]

        decode = abba.compare_samples(
            per_round(lambda row: [row["decode_ms_per_token"]])
        )
        follow_ups = per_round(
            lambda row: [turn["ttft_ms"] for turn in row["follow_ups"]]
        )
        ttft = abba.compare_samples(follow_ups) if all(follow_ups) else None
        baseline, candidate = versions["baseline"], versions["candidate"]
        keep = (
            baseline["failures_per_burst"] > 0
            and candidate["failures_per_burst"] == 0
            and decode["verdict"] != abba.INCONCLUSIVE
            and decode["regression"] <= 0.02
            and ttft is not None
            and ttft["regression"] <= -0.15
            and max(ttft["rounds"][1:3]) < min(ttft["rounds"][0], ttft["rounds"][3])
        )
        results.append(
            {
                "context": context,
                "scenario": "burst",
                **versions,
                "follow_up_ttft_ms": ttft,
                "decode_ms_per_token": decode,
                "keep": keep,
            }
        )
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    smoke.add_server_arguments(parser)
    parser.add_argument("--baseline-binary", required=True, type=Path)
    parser.add_argument("--contexts", default="2048,10000")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument(
        "--burst",
        type=int,
        default=0,
        help="measure bursts of this many concurrent requests per context instead",
    )
    parser.add_argument(
        "--follow-up",
        action="store_true",
        help="after each burst, send each conversation's next turn",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("build/release/http-regression.json")
    )
    args = smoke.resolve_server_arguments(parser.parse_args(argv))
    args.contexts = [int(value) for value in args.contexts.split(",")]
    if args.samples < 2 or not args.contexts or min(args.contexts) < 256:
        parser.error("at least two samples and contexts >= 256 are required")
    if args.burst < 0 or args.burst == 1 or (args.follow_up and not args.burst):
        parser.error("--burst needs two or more requests, and --follow-up a burst")
    if args.request_timeout <= 0:
        parser.error("the request timeout must be positive")
    if len(set(args.contexts)) != len(args.contexts):
        parser.error("contexts must be unique")
    for binary in (args.baseline_binary, args.binary):
        for path in (binary, binary.parent / "splash.metallib"):
            if not path.is_file():
                parser.error(f"missing retained executable/library: {path}")
    args.kind = smoke.model_artifacts.installation_kind(args.package)
    if args.kind is None:
        parser.error(f"missing installed model: {args.package}")
    return args


def check_identity(status: dict, version: str, rounds: list[dict], shared: bool):
    """Every round serves the same model and KV format. A build's rounds
    load the same executable and prepared files; builds of one preparation
    identity load the same prepared files too, while builds of different
    identities prepare under different keys, so their bytes are compared
    after the rounds instead (prepared.compare)."""
    identity = status["identity"]
    for previous in rounds:
        expected = previous["identity"]
        smoke.require(
            smoke.kv_identity(identity) == smoke.kv_identity(expected),
            "KV identity changed",
        )
        same_layout = (
            identity["cache"]["loaded_model_layout_sha256"]
            == expected["cache"]["loaded_model_layout_sha256"]
        )
        if previous["version"] == version:
            smoke.require(
                identity["cache"]["build_id"] == expected["cache"]["build_id"],
                "executable source changed between rounds",
            )
            smoke.require(same_layout, f"the {version} loaded another model layout")
        elif shared:
            smoke.require(same_layout, "loaded target/draft changed")


def main(argv=None):
    args = parse_args(argv)
    smoke.hold_package(args)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.package / "tokenizer", local_files_only=True
    )
    nonce = uuid.uuid4().hex
    # A burst's requests each have a prefix of their own.
    prepared = prompts(
        tokenizer, [128, *args.contexts], args.samples * max(1, args.burst), nonce
    )
    binaries = {"baseline": args.baseline_binary, "candidate": args.binary}
    shared = prepared_weights.preparation_identity(
        args.baseline_binary.resolve().parent
    ) == prepared_weights.preparation_identity(args.binary.resolve().parent)
    environments = {"baseline": None, "candidate": None}
    if not shared:
        # Builds of different preparation identities must not share a cache;
        # the candidate keeps its own, which its other steps use.
        environments["baseline"] = prepared_weights.baseline_environment(
            args.output.parent
        )
    document = {
        "schema_version": 1,
        "timing": "HTTP/native wall; not GPU time",
        "package": str(args.package.resolve()),
        "rounds": [],
        "samples": [],
        "correctness_pass": False,
        "performance_pass": False,
    }
    try:
        for round_id, version in enumerate(ROUNDS):
            run_args = argparse.Namespace(**vars(args))
            run_args.binary = binaries[version]
            server = smoke.RealServer(run_args, environments[version])
            try:
                status = server.wait_ready(args.startup_timeout)
                smoke.validate_status(status, args.kv_format)
                check_identity(status, version, document["rounds"], shared)
                document["rounds"].append(
                    {
                        "version": version,
                        "binary": str(run_args.binary.resolve()),
                        "identity": status["identity"],
                    }
                )
                for sample in range(int(round_id >= 2), args.samples, 2):
                    if args.burst:
                        for context in args.contexts:
                            contents = [
                                prepared[sample * args.burst + index, context]
                                for index in range(args.burst)
                            ]
                            row = measure_burst(
                                server,
                                args.model,
                                contents,
                                args.follow_up,
                                args.request_timeout,
                            )
                            row.update(
                                version=version,
                                round=round_id,
                                sample=sample,
                                context=context,
                            )
                            document["samples"].append(row)
                            print(
                                f"{version} sample={sample} burst context={context}",
                                file=sys.stderr,
                                flush=True,
                            )
                        continue
                    cases = [(128, "decode", 64)] + [
                        (context, scenario, 1)
                        for context in args.contexts
                        for scenario in ("cold", "exact")
                    ]
                    cold_outputs = {}
                    for context, scenario, output_tokens in cases:
                        row = measure(
                            server,
                            args.model,
                            prepared[sample, context],
                            output_tokens,
                            scenario,
                            context,
                            args.request_timeout,
                        )
                        if scenario == "cold":
                            cold_outputs[context] = row["response_sha256"]
                        elif scenario == "exact":
                            smoke.require(
                                row["response_sha256"] == cold_outputs[context],
                                "exact hit changed the cold transcript",
                            )
                        row.update(version=version, round=round_id, sample=sample)
                        document["samples"].append(row)
                        print(
                            f"{version} sample={sample} {scenario} context={context}",
                            file=sys.stderr,
                            flush=True,
                        )
            except Exception:
                document["server_tail"] = server.tail()
                raise
            finally:
                server.close()
        document["comparison"] = (summarize_bursts if args.burst else summarize)(
            document["samples"]
        )
        document["prepared"] = (
            {"shared_identity": True, "pass": True}
            if shared
            else {
                "shared_identity": False,
                **prepared_weights.compare(
                    prepared_weights.cache_root(environments["baseline"]),
                    prepared_weights.cache_root(os.environ),
                    # The model root RealServer gives both builds.
                    package=args.package.resolve(),
                    required=args.kind == smoke.model_artifacts.ASSEMBLY,
                ),
            }
        )
        smoke.require(
            document["prepared"]["pass"],
            f"prepared bytes differ: {document['prepared'].get('failures')}",
        )
        document["correctness_pass"] = True
        document["performance_pass"] = all(
            row["keep"] if args.burst else row["pass"] for row in document["comparison"]
        )
    except Exception as error:
        document["error"] = str(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(document, indent=2) + "\n")
        temporary.replace(args.output)
    print(json.dumps(document["comparison"], indent=2))
    return 0 if document["performance_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

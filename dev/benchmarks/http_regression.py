"""Compare retained servers with matched HTTP requests in ABBA order.

Run as ``python -m dev.benchmarks.http_regression --baseline-binary PATH``.
Reuses the real HTTP test lifecycle; never contacts an existing server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

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
        after["state"]["active_cells"] == 0 and after["kv"]["pages_active"] == 0,
        "active resources leaked",
    )
    delta = {
        key: after["metrics"][key] - before["metrics"][key]
        for key in (
            "prefill_wall_ms",
            "decode_wall_ms",
            "prefill_input_tokens",
            "decode_output_tokens",
            "drafted_tokens",
            "accepted_draft_tokens",
        )
    }
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


def summarize(records: list[dict], maximum_regression: float) -> list[dict]:
    groups = defaultdict(lambda: defaultdict(list))
    outputs = {}
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
            latency = row["native_delta"]["decode_wall_ms"] / max(
                1, row["native_delta"]["decode_output_tokens"]
            )
        else:
            latency = row["metrics"]["request_latency"]["ttft_ms"]
        groups[row["context"], row["scenario"]][row["version"]].append(latency)
    results = []
    if any(set(pair) != {"baseline", "candidate"} for pair in outputs.values()):
        raise ValueError("unpaired request samples")
    for (context, scenario), versions in sorted(groups.items()):
        if len(versions["baseline"]) != len(versions["candidate"]):
            raise ValueError("unpaired performance samples")
        baseline = statistics.median(versions["baseline"])
        candidate = statistics.median(versions["candidate"])
        ratio = candidate / baseline
        results.append(
            {
                "context": context,
                "scenario": scenario,
                "metric": "decode_ms_per_token" if scenario == "decode" else "ttft_ms",
                "samples_per_binary": len(versions["baseline"]),
                "baseline_median": baseline,
                "candidate_median": candidate,
                "ratio": ratio,
                "pass": ratio <= 1 + maximum_regression,
            }
        )
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    smoke.add_server_arguments(parser)
    parser.add_argument("--baseline-binary", required=True, type=Path)
    parser.add_argument("--contexts", default="2048,10000")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--maximum-regression", type=float, default=0.02)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument(
        "--output", type=Path, default=Path("build/release/http-regression.json")
    )
    args = smoke.resolve_server_arguments(parser.parse_args(argv))
    args.contexts = [int(value) for value in args.contexts.split(",")]
    if args.samples < 2 or not args.contexts or min(args.contexts) < 256:
        parser.error("at least two samples and contexts >= 256 are required")
    if args.maximum_regression < 0 or args.request_timeout <= 0:
        parser.error("regression must be nonnegative and timeout positive")
    if len(set(args.contexts)) != len(args.contexts):
        parser.error("contexts must be unique")
    for binary in (args.baseline_binary, args.binary):
        for path in (binary, binary.parent / "splash.metallib"):
            if not path.is_file():
                parser.error(f"missing retained executable/library: {path}")
    for path in (args.package / "manifest.json", args.package / "tokenizer"):
        if not path.exists():
            parser.error(f"missing installed package: {path}")
    return args


def main(argv=None):
    args = parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.package / "tokenizer", local_files_only=True
    )
    nonce = uuid.uuid4().hex
    prepared = prompts(tokenizer, [128, *args.contexts], args.samples, nonce)
    binaries = {"baseline": args.baseline_binary, "candidate": args.binary}
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
        for round_id, version in enumerate(
            ("baseline", "candidate", "candidate", "baseline")
        ):
            run_args = argparse.Namespace(**vars(args))
            run_args.binary = binaries[version]
            server = smoke.RealServer(run_args)
            try:
                status = server.wait_ready(args.startup_timeout)
                smoke.validate_status(status, args.kv_format)
                if document["rounds"]:
                    expected = document["rounds"][0]["identity"]
                    smoke.require(
                        smoke.kv_identity(status["identity"])
                        == smoke.kv_identity(expected)
                        and status["identity"]["cache"]["loaded_model_layout_sha256"]
                        == expected["cache"]["loaded_model_layout_sha256"],
                        "loaded target/draft or KV identity changed",
                    )
                for previous in document["rounds"]:
                    if previous["version"] == version:
                        smoke.require(
                            status["identity"]["cache"]["build_id"]
                            == previous["identity"]["cache"]["build_id"],
                            "executable source changed between rounds",
                        )
                document["rounds"].append(
                    {
                        "version": version,
                        "binary": str(run_args.binary.resolve()),
                        "identity": status["identity"],
                    }
                )
                for sample in range(int(round_id >= 2), args.samples, 2):
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
        document["comparison"] = summarize(document["samples"], args.maximum_regression)
        document["correctness_pass"] = True
        document["performance_pass"] = all(
            row["pass"] for row in document["comparison"]
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

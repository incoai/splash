"""Paired HTTP ABBA measurements on two resident servers.

Only one server runs a measured request at a time. Alternating resident
servers shortens the time between comparison arms without restarting or
warming a model between them. This needs enough memory for both servers;
memory pressure and their shared GPU still limit what the result generalizes
to. Use the standard sequential release gate as the independent rollout check.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import http.client
import json
import math
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

from dev.benchmarks import abba
from dev.benchmarks.trace_report import percentile
from dev.tests import smoke_real as smoke

WORKLOADS = {
    "code": "Implement a thread-safe bounded LRU cache in Python with expiration, "
    "type annotations, documentation and comprehensive unit tests. Explain every design choice.",
    "math": "Derive the central limit theorem for independent identically distributed "
    "random variables. Explain assumptions and every intermediate step in detail.",
    "prose": "Explain how operating systems implement virtual memory, page tables, "
    "copy-on-write and page reclamation. Give extensive examples and implementation details.",
}
METRICS = (
    "decode_cycle_ms",
    "decode_wall_ms",
    "decode_output_tokens",
    "drafted_tokens",
    "accepted_draft_tokens",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def fingerprint(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def environment():
    return {
        key: subprocess.check_output(command, text=True).strip()
        for key, command in (
            ("power", ["pmset", "-g", "batt"]),
            ("thermal", ["pmset", "-g", "therm"]),
        )
    }


@contextlib.contextmanager
def child_environment(values):
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def prepared_prompt(tokenizer, content, context, identity):
    filler = tokenizer.encode(
        " The service uses a durable queue with retries and transactional state.",
        add_special_tokens=False,
    )
    prefix = digest(identity) + "\nProject notes: "
    suffix = "\n" + content + "\nGive a complete, extensive answer."
    rows = context
    for _ in range(12):
        prompt = (
            prefix
            + tokenizer.decode(
                (filler * (max(rows, 0) // len(filler) + 1))[: max(rows, 0)]
            )
            + suffix
        )
        count = len(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        if count == context:
            return prompt
        rows -= count - context
        if rows < 0:
            raise ValueError("context is too short for this workload")
    raise ValueError(f"could not prepare exactly {context} prompt tokens")


def completion(server, args, content, tokens=None, **extra):
    extra.setdefault("cache_salt", "serving-regression-" + uuid.uuid4().hex)
    code, response = smoke.request(
        server.port,
        "POST",
        "/v1/chat/completions",
        smoke.chat_body(
            args.model, content, max_completion_tokens=tokens or args.tokens, **extra
        ),
        timeout=args.request_timeout,
    )
    smoke.require(code == 200, f"completion failed: HTTP {code}: {response}")
    choice = response["choices"][0]
    return {
        "usage": response["usage"],
        "finish": choice["finish_reason"],
        "message_sha256": digest(choice["message"]),
        "message": choice["message"],
    }


def status(server, args):
    code, result = smoke.request(server.port, "GET", "/status")
    smoke.require(code == 200, "status request failed")
    smoke.validate_status(result, args.kv_format)
    return result


def warm_servers(servers, args, tokenizer):
    elapsed = [0.0] * len(servers)
    rounds = 0
    while (
        rounds < getattr(args, "warmup_rounds", 2) or min(elapsed) < args.warmup_seconds
    ):
        prompt = WORKLOADS["code"]
        if args.warmup_context:
            prompt = prepared_prompt(
                tokenizer, prompt, args.warmup_context, ["warmup", rounds]
            )
        for index, server in enumerate(servers):
            started = time.monotonic()
            completion(server, args, prompt, tokens=args.warmup_tokens)
            elapsed[index] += time.monotonic() - started
        rounds += 1
    return {"rounds_per_arm": rounds, "request_wall_seconds_by_arm": elapsed}


def measure(server, args, prompts, context):
    before = status(server, args)
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        responses = list(
            pool.map(lambda prompt: completion(server, args, prompt), prompts)
        )
    elapsed = time.monotonic() - started
    after = status(server, args)
    smoke.require(
        after["state"]["active_lanes"] == 0 and after["kv"]["pages_active"] == 0,
        "request resources remain active after completion",
    )
    for response in responses:
        smoke.require(
            response["usage"]["prompt_tokens"] == context, "prompt rows changed"
        )
        smoke.require(
            response["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
            == 0,
            "cold benchmark unexpectedly reused a prompt prefix",
        )
        smoke.require(
            response["usage"]["completion_tokens"] == args.tokens,
            "workload ended before its requested output budget",
        )
    delta = {key: after["metrics"][key] - before["metrics"][key] for key in METRICS}
    smoke.require(
        delta["decode_output_tokens"] > 0 and delta["decode_cycle_ms"] > 0,
        "missing decode measurements",
    )
    return {
        "seconds_per_token": delta["decode_cycle_ms"]
        / 1000
        / delta["decode_output_tokens"],
        "wall_seconds": elapsed,
        "delta": delta,
        "responses": responses,
        "thermal_before": before.get("thermal_state"),
        "thermal_after": after.get("thermal_state"),
        "memory_before": before.get("memory_actual"),
        "memory_after": after.get("memory_actual"),
        "memory_pressure_before": before.get("memory_pressure"),
        "memory_pressure_after": after.get("memory_pressure"),
        "decode_batches_by_width": {
            key: value - before["scheduler"]["decode_batches_by_width"][key]
            for key, value in after["scheduler"]["decode_batches_by_width"].items()
        },
    }


def summarize(rounds):
    result = abba.compare(*(row["seconds_per_token"] for row in rounds))
    result["output_equal"] = (
        len({tuple(r["message_sha256"] for r in row["responses"]) for row in rounds})
        == 1
    )
    result["throughput_gain"] = result["baseline"] / result["candidate"] - 1
    for name in (
        "short_ttft_ms",
        "long_ttft_ms",
        "active_gap_p95_ms",
        "active_gap_max_ms",
    ):
        if all(row.get(name, 0) > 0 for row in rounds):
            result[name] = abba.compare(*(row[name] for row in rounds))
    return result


def fallback_probes(server, args):
    sampled = completion(
        server,
        args,
        "Explain binary search with annotated Python code.",
        tokens=128,
        temperature=0.7,
        seed=1234,
    )
    value = "A bounded cache keeps recent entries and expires stale values. " * 8
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string", "const": value}},
        "required": ["text"],
        "additionalProperties": False,
    }
    constrained = completion(
        server,
        args,
        "Return the required JSON object.",
        tokens=256,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "probe", "strict": True, "schema": schema},
        },
    )
    smoke.require(
        json.loads(constrained["message"]["content"]) == {"text": value},
        "constrained fallback returned invalid JSON",
    )
    return {"sampled": sampled, "constrained": constrained}


def streamed(
    server, args, prompt, tokens, first=None, *, cache_salt=None, expect_cached=False
):
    """Client arrival/chunk timestamps, including HTTP/tokenization latency."""
    connection = http.client.HTTPConnection(
        "127.0.0.1", server.port, timeout=args.request_timeout
    )
    started = time.monotonic()
    chunks, text, usage = [], [], None
    try:
        body = smoke.chat_body(
            args.model,
            prompt,
            max_completion_tokens=tokens,
            stream=True,
            stream_options={"include_usage": True},
            cache_salt=cache_salt or "mixed-regression-" + uuid.uuid4().hex,
        )
        connection.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(body),
            smoke.request_headers(True),
        )
        response = connection.getresponse()
        smoke.require(response.status == 200, f"stream failed: HTTP {response.status}")
        for line in iter(response.readline, b""):
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            document = json.loads(payload)
            if document.get("usage"):
                usage = document["usage"]
            for choice in document.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    chunks.append(time.monotonic())
                    text.append(content)
                    if first:
                        first.set()
        smoke.require(
            chunks and usage is not None, "stream omitted content or final usage"
        )
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        smoke.require(
            cached > 0 if expect_cached else cached == 0,
            "mixed workload cache reuse does not match the requested scenario",
        )
        return {
            "started": started,
            "ended": time.monotonic(),
            "chunks": chunks,
            "ttft_ms": (chunks[0] - started) * 1000,
            "usage": usage,
            "message_sha256": digest("".join(text)),
        }
    finally:
        if first:
            first.set()
        connection.close()


def prepare_mixed_cache(server, args, long_prompt):
    if args.cache_mode == "cold":
        return None
    salt = "mixed-primer-" + uuid.uuid4().hex
    primer = (
        long_prompt
        if args.cache_mode == "cached"
        else long_prompt[: len(long_prompt) // 2]
    )
    response = completion(server, args, primer, tokens=1, cache_salt=salt)
    smoke.require(
        response["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0) == 0,
        "a fresh cache primer unexpectedly reused earlier work",
    )
    return salt


def native_sample(server, args, before_short=False):
    value = status(server, args)
    smoke.require(not value.get("transport", {}).get("status_stale", False),
                  "stale native status cannot prove work overlap")
    return {
        "at": time.monotonic(),
        "before_short": before_short,
        "prefill_rows": value["scheduler"]["prefill_rows"],
        "memory_pressure": value["memory_pressure"],
        "requests": value["active_requests"],
    }


def native_overlap(
    samples, active_id, context, cached, long_response, active_response, initial_rows
):
    """Prove native work overlap; concurrent HTTP lifetimes alone are insufficient."""
    long_id = None
    for sample in samples:
        if sample["at"] >= active_response["ended"]:
            continue
        active = any(
            row["id"] == active_id and row["phase"] == "decode"
            for row in sample["requests"]
        )
        if not active:
            continue
        peers = [
            row
            for row in sample["requests"]
            if row["id"] != active_id
            and row["prompt_tokens"] == context
            and row["max_new_tokens"] == 32
        ]
        if sample["before_short"]:
            if peers:
                long_id = peers[0]["id"]
            # No other request can have added prefill rows in this interval.
            if sample["prefill_rows"] > initial_rows:
                return "long-prefill"
        for row in peers:
            if (context != 512 or row["id"] == long_id) and row[
                "prompt_processed"
            ] > cached:
                return "long-prefill"
        if (
            cached >= long_response["usage"]["prompt_tokens"]
            and long_response["ended"] <= sample["at"]
        ):
            return "cached-continuation-without-prefill"
    return None


def measure_mixed(server, args, active_prompt, long_prompt, short_prompt, context):
    long_salt = prepare_mixed_cache(server, args, long_prompt)
    before = status(server, args)
    started = time.monotonic()
    first = threading.Event()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    try:
        active = pool.submit(streamed, server, args, active_prompt, args.tokens, first)
        smoke.require(
            first.wait(min(60, args.request_timeout)), "active stream did not start"
        )
        if active.done():
            active.result()
            raise smoke.SmokeFailure(
                "active stream ended before the contended arrivals"
            )
        initial = native_sample(server, args, before_short=True)
        active_rows = [
            row for row in initial["requests"] if row["max_new_tokens"] == args.tokens
        ]
        smoke.require(
            len(active_rows) == 1 and active_rows[0]["phase"] == "decode",
            "the initial native request is no longer decoding",
        )
        active_id = active_rows[0]["id"]
        native_samples = []
        time.sleep(args.arrival_delay)
        long = pool.submit(
            streamed,
            server,
            args,
            long_prompt,
            32,
            cache_salt=long_salt,
            expect_cached=args.cache_mode != "cold",
        )
        time.sleep(args.arrival_delay)
        native_samples.append(native_sample(server, args, before_short=True))
        short = pool.submit(streamed, server, args, short_prompt, 32)
        while not active.done():
            native_samples.append(native_sample(server, args))
            time.sleep(args.status_interval)
        responses = [
            future.result(timeout=args.request_timeout)
            for future in (active, long, short)
        ]
    finally:
        # On failure the outer server cleanup closes outstanding requests;
        # do not block that cleanup by waiting here for another HTTP timeout.
        pool.shutdown(wait=False, cancel_futures=True)
    elapsed = time.monotonic() - started
    after = status(server, args)
    smoke.require(
        after["state"]["active_lanes"] == 0 and after["kv"]["pages_active"] == 0,
        "mixed requests left resources active",
    )
    active, long, short = responses
    cached = long["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
    overlap = native_overlap(
        native_samples,
        active_id,
        context,
        cached,
        long,
        active,
        initial["prefill_rows"],
    )
    smoke.require(
        overlap is not None,
        "requests overlapped but native long prefill/continuation did not overlap decoding; "
        "inspect memory/admission limits or use --residency sequential",
    )
    gaps = [
        (right - left) * 1000
        for left, right in zip(active["chunks"], active["chunks"][1:])
        if left < long["chunks"][0] and right > long["started"]
    ]
    smoke.require(
        gaps and active["ended"] > long["started"], "no decode/prefill overlap occurred"
    )
    total = sum(row["usage"]["completion_tokens"] for row in responses)
    return {
        "seconds_per_token": elapsed / total,
        "wall_seconds": elapsed,
        "native_overlap": overlap,
        "native_samples": native_samples,
        "decode_batches_by_width": {
            key: value - before["scheduler"]["decode_batches_by_width"][key]
            for key, value in after["scheduler"]["decode_batches_by_width"].items()
        },
        "short_ttft_ms": short["ttft_ms"],
        "long_ttft_ms": long["ttft_ms"],
        "active_gap_p95_ms": percentile(gaps, 0.95),
        "active_gap_max_ms": max(gaps),
        "overlapping_chunk_gaps": len(gaps),
        "responses": responses,
        "delta": {
            key: after["metrics"][key] - before["metrics"][key] for key in METRICS
        },
        "thermal_before": before.get("thermal_state"),
        "thermal_after": after.get("thermal_state"),
        "memory_before": before.get("memory"),
        "memory_after": after.get("memory"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    smoke.add_server_arguments(parser)
    parser.add_argument("--baseline-binary", type=Path, required=True)
    parser.add_argument("--contexts", default="2048,8192,31744")
    parser.add_argument("--workloads", default="code,math,prose")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmup-context", type=int, default=0)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument(
        "--residency", choices=("paired", "sequential"), default="paired"
    )
    parser.add_argument("--status-interval", type=float, default=0.1)
    parser.add_argument("--warmup-tokens", type=int, default=128)
    parser.add_argument("--warmup-seconds", type=float, default=0)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1, choices=range(1, 5))
    parser.add_argument("--request-timeout", type=float, default=1200)
    parser.add_argument("--baseline-env", action="append", default=[])
    parser.add_argument("--candidate-env", action="append", default=[])
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--probes",
        action="store_true",
        help="exercise sampled and multi-cycle constrained fallbacks",
    )
    parser.add_argument("--scenario", choices=("decode", "mixed"), default="decode")
    parser.add_argument("--arrival-delay", type=float, default=0.05)
    parser.add_argument(
        "--cache-mode", choices=("cold", "cached", "prefix-resume"), default="cold"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = smoke.resolve_server_arguments(parser.parse_args())
    args.contexts = [int(value) for value in args.contexts.split(",")]
    args.workloads = args.workloads.split(",")
    args.max_context = args.max_context or 32768
    args.max_memory = args.max_memory or "20G"
    args.max_cache_disk = args.max_cache_disk or "0"
    if args.samples < 1 or args.tokens < 8 or min(args.contexts) < 256:
        parser.error(
            "need positive samples, at least eight output tokens and contexts >=256"
        )
    if max(args.contexts) + args.tokens > args.max_context:
        parser.error("prompt plus output exceeds maximum context")
    if set(args.workloads) - WORKLOADS.keys():
        parser.error("unknown workload")
    if args.scenario == "mixed" and (
        args.batch != 1 or args.tokens < 128 or args.arrival_delay < 0
    ):
        parser.error(
            "mixed workloads need batch=1, tokens>=128 and a nonnegative arrival delay"
        )
    if (
        not math.isfinite(args.warmup_seconds)
        or args.warmup_seconds < 0
        or args.warmup_context < 0
        or args.warmup_tokens < 1
        or args.warmup_context + args.warmup_tokens > args.max_context
    ):
        parser.error("invalid warmup duration/context/output budget")
    if args.scenario != "mixed" and args.cache_mode != "cold":
        parser.error("cached and prefix-resume modes require --scenario mixed")
    if (
        args.warmup_rounds < 1
        or not math.isfinite(args.status_interval)
        or args.status_interval <= 0
    ):
        parser.error("need positive warmup rounds and status interval")
    if args.residency == "sequential" and args.warmup_seconds:
        parser.error(
            "sequential residency uses fixed --warmup-rounds with --warmup-seconds 0"
        )
    args.environments = []
    for values in (args.baseline_env, args.candidate_env):
        env = {
            "HF_HUB_OFFLINE": "1",
            "MTL_SHADER_VALIDATION": None,
            "SPLASH_SERVING_TRACE": None,
        }
        for value in values:
            key, separator, content = value.partition("=")
            if not key or not separator:
                parser.error("environment options have the form KEY=VALUE")
            env[key] = content
        args.environments.append(env)
    return args


def case_specs(args, tokenizer):
    for context in args.contexts:
        for workload in args.workloads:
            for sample in range(args.samples):
                yield {
                    "context": context,
                    "workload": workload,
                    "sample": sample,
                    "prompts": [
                        prepared_prompt(
                            tokenizer,
                            WORKLOADS[workload],
                            context,
                            [workload, sample, lane],
                        )
                        for lane in range(args.batch)
                    ],
                    "rounds": [],
                }


def measure_case(server, args, tokenizer, case):
    if args.scenario != "mixed":
        return measure(server, args, case["prompts"], case["context"])
    active_prompt = prepared_prompt(
        tokenizer, WORKLOADS["code"], 2048, ["active", case["sample"]]
    )
    short_prompt = prepared_prompt(
        tokenizer, WORKLOADS["prose"], 512, ["short", case["sample"]]
    )
    return measure_mixed(
        server, args, active_prompt, case["prompts"][0], short_prompt, case["context"]
    )


def run_sequential(args, tokenizer, report, binaries):
    """Ordinary ABBA rounds with one owned resident server at a time."""
    report["method"] = (
        "one resident server at a time; full workload rounds in ABBA order; cold salts"
    )
    report["cases"] = list(case_specs(args, tokenizer))
    report["server_runs"] = []
    report["final_status"] = []
    if args.probes:
        report["probes"] = []
    path = args.output / "report.json"
    try:
        for arm, index in enumerate((0, 1, 1, 0), 1):
            settings = argparse.Namespace(**{**vars(args), "binary": binaries[index]})
            env = dict(args.environments[index])
            if args.trace:
                env["SPLASH_SERVING_TRACE"] = str(
                    (args.output / f"round-{arm}-trace.json").resolve()
                )
            record = {"arm": arm, "server": index}
            report["server_runs"].append(record)
            with contextlib.ExitStack() as cleanup:
                with child_environment(env):
                    server = smoke.RealServer(settings)
                cleanup.callback(server.close)
                try:
                    print(
                        f"Starting round {arm} ({'baseline' if index == 0 else 'candidate'})",
                        flush=True,
                    )
                    record["initial_status"] = server.wait_ready(args.startup_timeout)
                    report["initial_status"].append(record["initial_status"])
                    record["warmup"] = warm_servers([server], args, tokenizer)
                    for case in report["cases"]:
                        row = measure_case(server, args, tokenizer, case)
                        row.update(arm=arm, server=index)
                        case["rounds"].append(row)
                        if len(case["rounds"]) == 4:
                            case["comparison"] = summarize(case["rounds"])
                            print(
                                f"{case['workload']} c{case['context']} B{args.batch} sample{case['sample']}: "
                                f"{abba.describe(case['comparison'])}",
                                flush=True,
                            )
                        path.write_text(json.dumps(report, indent=2) + "\n")
                    if args.probes:
                        report["probes"].append(fallback_probes(server, args))
                    record["final_status"] = status(server, args)
                    report["final_status"].append(record["final_status"])
                finally:
                    server.stop()
                    server.log.seek(0)
                    (args.output / f"round-{arm}.log").write_text(server.log.read())
        report["final_environment"] = environment()
        report["complete"] = True
    except BaseException as error:
        report["error"] = str(error)
        raise
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    smoke.hold_model_root(args)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_root / "tokenizer", local_files_only=True
    )
    binaries = [args.baseline_binary, args.binary]
    report = {
        "method": "two resident servers, serial cold requests with unique cache salts, paired ABBA",
        "scenario": args.scenario,
        "residency": args.residency,
        "tool_sha256": fingerprint(Path(__file__)),
        "primary_metric": "native decode cycle seconds/token"
        if args.scenario == "decode"
        else "total workload wall seconds/output token",
        "environment": environment(),
        "model": args.model,
        "model_manifest_sha256": fingerprint(args.model_root / "model.json"),
        "binaries": [
            {
                "sha256": fingerprint(p),
                "metallib_sha256": fingerprint(p.parent / "splash.metallib"),
            }
            for p in binaries
        ],
        "settings": {
            "max_context": args.max_context,
            "max_memory": args.max_memory,
            "kv_format": args.kv_format,
            "tokens": args.tokens,
            "batch": args.batch,
            "samples": args.samples,
            "cache_mode": args.cache_mode,
            "warmup_rounds": args.warmup_rounds,
            "status_interval_seconds": args.status_interval,
            "warmup_context": args.warmup_context,
            "warmup_tokens": args.warmup_tokens,
            "warmup_seconds_per_arm": args.warmup_seconds,
            "arrival_delay_seconds": args.arrival_delay,
            "mixed_active_context": 2048,
            "mixed_short_context": 512,
            "mixed_peer_output_tokens": 32,
        },
        "environments": args.environments,
        "initial_status": [],
        "cases": [],
        "complete": False,
    }
    if args.residency == "sequential":
        run_sequential(args, tokenizer, report, binaries)
        return
    servers = []
    cleanup = contextlib.ExitStack()
    try:
        for index, binary in enumerate(binaries):
            settings = argparse.Namespace(**{**vars(args), "binary": binary})
            env = dict(args.environments[index])
            if args.trace:
                env["SPLASH_SERVING_TRACE"] = str(
                    (args.output / f"server-{index}-trace.json").resolve()
                )
            print(
                f"Starting {'baseline' if index == 0 else 'candidate'} server",
                flush=True,
            )
            with child_environment(env):
                server = smoke.RealServer(settings)
            servers.append(server)
            cleanup.callback(server.close)
            report["initial_status"].append(server.wait_ready(args.startup_timeout))
            print(f"Server {index} ready", flush=True)
        # Alternate complete matched warmup rounds; the faster arm never
        # receives extra requests merely because it reaches the time first.
        report["warmup"] = warm_servers(servers, args, tokenizer)
        for context in args.contexts:
            for workload in args.workloads:
                for sample in range(args.samples):
                    prompts = [
                        prepared_prompt(
                            tokenizer,
                            WORKLOADS[workload],
                            context,
                            [workload, sample, lane],
                        )
                        for lane in range(args.batch)
                    ]
                    rounds = []
                    if args.scenario == "mixed":
                        active_prompt = prepared_prompt(
                            tokenizer, WORKLOADS["code"], 2048, ["active", sample]
                        )
                        short_prompt = prepared_prompt(
                            tokenizer, WORKLOADS["prose"], 512, ["short", sample]
                        )
                    for arm, server_index in enumerate([0, 1, 1, 0], 1):
                        if args.scenario == "mixed":
                            row = measure_mixed(
                                servers[server_index],
                                args,
                                active_prompt,
                                prompts[0],
                                short_prompt,
                                context,
                            )
                        else:
                            row = measure(servers[server_index], args, prompts, context)
                        row.update(arm=arm, server=server_index)
                        rounds.append(row)
                    result = summarize(rounds)
                    report["cases"].append(
                        {
                            "context": context,
                            "workload": workload,
                            "sample": sample,
                            "prompts": prompts,
                            "rounds": rounds,
                            "comparison": result,
                        }
                    )
                    (args.output / "report.json").write_text(
                        json.dumps(report, indent=2) + "\n"
                    )
                    print(
                        f"{workload} c{context} B{args.batch} sample{sample}: {abba.describe(result)}, outputs equal={result['output_equal']}",
                        flush=True,
                    )
        if args.probes:
            report["probes"] = [fallback_probes(server, args) for server in servers]
        report["final_status"] = [status(server, args) for server in servers]
        report["final_environment"] = environment()
        report["complete"] = True
    except BaseException as error:
        report["error"] = str(error)
        raise
    finally:
        for index, server in enumerate(servers):
            try:
                server.stop()
                server.log.seek(0)
                (args.output / f"server-{index}.log").write_text(server.log.read())
            except Exception as error:
                report.setdefault("cleanup_errors", []).append(str(error))
        try:
            cleanup.close()
        finally:
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()

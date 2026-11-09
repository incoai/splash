"""Exploratory HTTP ABBA comparison of two builds or installed drafters.

Run on an otherwise idle Mac with --baseline-binary, --baseline-model-root,
--candidate-model-root and --output DIR. The candidate binary defaults to
build/splash. Each binary needs its own splash.metallib beside it. The four
rounds restart the server, warm up, then run four greedy coding prompts and
the same prompts concurrently. Results retain output hashes, speculative
counts, build/binary identities and the repository's ABBA noise calculation.
This small sample is not the release performance gate.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from dev.benchmarks import abba
from dev.tests import smoke_real as smoke

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = (
    "Write a Python function that returns the nth Fibonacci number using iteration. Return only the function.",
    "Implement a stable merge sort in Python. Include a short explanation of its complexity.",
    "Write a Python LRU cache using a doubly linked list and a dictionary. Explain invariants briefly.",
    "Write a TypeScript function that retries an asynchronous operation with exponential backoff.",
)
NAMES = ("fibonacci", "merge_sort", "lru_cache", "typescript_retry")


def target_sources(model_root: Path) -> dict:
    """Target source identity, independent of draft-specific image plans.

    /status's target_model_sha256 is a prepared-image manifest identity;
    capture-layer differences can change it for the same source weights.
    Installed model.json records the verified source content digests.
    """
    manifest = json.loads((model_root / "model.json").read_text())
    weights = {
        name: {"bytes": value["bytes"], "digest": value["digest"]}
        for name, value in manifest["files"].items()
        if name.startswith("target/")
    }
    smoke.require(bool(weights), "model root has no target source digests")
    return {
        "config_sha256": hashlib.sha256(
            (model_root / "config.json").read_bytes()
        ).hexdigest(),
        "weights": weights,
    }


def completion(port: int, prompt: str, salt: str, max_tokens: int) -> dict:
    started = time.perf_counter()
    code, document = smoke.request(
        port,
        "POST",
        "/v1/chat/completions",
        smoke.chat_body(
            "local/comparison",
            prompt,
            max_completion_tokens=max_tokens,
            seed=42,
            cache_salt=salt,
        ),
        timeout=180,
    )
    smoke.require(code == 200, f"completion returned {code}: {document}")
    text = document["choices"][0]["message"]["content"]
    smoke.require(bool(text), "completion produced no text")
    metrics = document["metrics"]
    return {
        "usage": document["usage"],
        "latency": metrics["request_latency"],
        "speculation": metrics["speculation"],
        "cache": metrics["cache"],
        "finish_reason": document["choices"][0]["finish_reason"],
        "client_wall_seconds": time.perf_counter() - started,
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def round_run(args, name: str, binary: Path, model_root: Path) -> dict:
    port = smoke.available_port()
    salt = f"draft-comparison-{uuid.uuid4().hex}"
    command = [
        sys.executable,
        "-m",
        "server.server",
        str(model_root.resolve()),
        "--binary",
        str(binary.resolve()),
        "--model",
        "local/comparison",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-context",
        "32768",
        "--kv-format",
        "int8",
        "--state-format",
        "fp32",
        "--disable-ane",
        "--verify-rows",
        "all",
        "--persistent-cache",
        "--cache-dir",
        str((args.output / f"cache-{name}").resolve()),
        "--max-cache-disk",
        "4G",
    ]
    with (args.output / f"{name}.server.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            deadline = time.monotonic() + 300
            while True:
                smoke.require(process.poll() is None, f"{name} exited during startup")
                try:
                    code, status = smoke.request(port, "GET", "/status", timeout=1)
                    if code == 200 and status.get("ready"):
                        break
                except (OSError, smoke.SmokeFailure):
                    pass
                smoke.require(time.monotonic() < deadline, f"{name} startup timed out")
                time.sleep(0.25)
            smoke.validate_status(status, "int8")
            deadline = time.monotonic() + 300
            while status["thermal_state"] != "nominal":
                smoke.require(
                    time.monotonic() < deadline, f"{name} did not cool to nominal"
                )
                print(f"{name}: cooling ({status['thermal_state']})", flush=True)
                time.sleep(5)
                code, status = smoke.request(port, "GET", "/status")
                smoke.require(code == 200, "status failed while cooling")
            print(f"{name}: ready", flush=True)
            completion(port, PROMPTS[0], f"{salt}-warmup", 128)
            solo = {
                key: completion(port, prompt, f"{salt}-solo-{key}", args.max_tokens)
                for key, prompt in zip(NAMES, PROMPTS, strict=True)
            }
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [
                    pool.submit(
                        completion, port, prompt, f"{salt}-batch-{key}", args.max_tokens
                    )
                    for key, prompt in zip(NAMES, PROMPTS, strict=True)
                ]
                batch = dict(zip(NAMES, (f.result() for f in futures), strict=True))
            batch_wall = time.perf_counter() - started
            code, final_status = smoke.request(port, "GET", "/status")
            smoke.require(code == 200, "final status failed")
            smoke.validate_status(final_status, "int8")
            result = {
                "name": name,
                "build_id": status["identity"]["cache"]["build_id"],
                "target_model_sha256": status["identity"]["kv"]["target_model_sha256"],
                "target_sources": target_sources(model_root),
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "metallib_sha256": hashlib.sha256(
                    (binary.parent / "splash.metallib").read_bytes()
                ).hexdigest(),
                "thermal_state_start": status["thermal_state"],
                "thermal_state_end": final_status["thermal_state"],
                "solo": solo,
                "batch": batch,
                "batch_wall_seconds": batch_wall,
            }
            (args.output / f"{name}.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            return result
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    code = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                    raise smoke.SmokeFailure(f"{name} did not stop cleanly") from None
                smoke.require(code == 0, f"{name} stopped with status {code}")


def summarize(rounds: list[dict]) -> dict:
    comparisons = {}
    for key in NAMES:
        comparisons[key] = abba.compare(
            *(
                1000 / r["solo"][key]["latency"]["stream_tokens_per_second"]
                for r in rounds
            )
        )
    # All measured outputs exclude the first token selected during prefill.
    solo_tps = [
        sum(v["usage"]["completion_tokens"] - 1 for v in r["solo"].values())
        / sum(
            (v["usage"]["completion_tokens"] - 1)
            / v["latency"]["stream_tokens_per_second"]
            for v in r["solo"].values()
        )
        for r in rounds
    ]
    comparisons["combined_solo"] = abba.compare(*(1000 / t for t in solo_tps))
    comparisons["four_concurrent"] = abba.compare(
        *(
            1000
            * r["batch_wall_seconds"]
            / sum(v["usage"]["completion_tokens"] for v in r["batch"].values())
            for r in rounds
        )
    )
    return {
        "milliseconds_per_token": comparisons,
        "solo_output_hashes_match": all(
            r["solo"][key]["output_sha256"] == rounds[0]["solo"][key]["output_sha256"]
            for r in rounds[1:]
            for key in NAMES
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-binary", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path, default=ROOT / "build/splash")
    parser.add_argument("--baseline-model-root", type=Path, required=True)
    parser.add_argument("--candidate-model-root", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    smoke.require(args.max_tokens > 1, "--max-tokens must exceed one")
    args.output.mkdir(parents=True, exist_ok=False)
    rounds = []
    for name in ("a1", "b1", "b2", "a2"):
        baseline = name.startswith("a")
        rounds.append(
            round_run(
                args,
                name,
                args.baseline_binary if baseline else args.candidate_binary,
                args.baseline_model_root if baseline else args.candidate_model_root,
            )
        )
        print(f"{name}: finished", flush=True)
    smoke.require(
        all(r["target_sources"] == rounds[0]["target_sources"] for r in rounds),
        "the rounds loaded different target models",
    )
    result = {
        "schema_version": 1,
        "prompts": dict(zip(NAMES, PROMPTS, strict=True)),
        "method": {
            "order": ["baseline", "candidate", "candidate", "baseline"],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "seed": 42,
            "reasoning_effort": "none",
            "warmup_tokens": 128,
            "max_context": 32768,
            "kv_format": "int8",
            "state_format": "fp32",
            "ane": False,
            "verify_rows": "all",
            "persistent_cache_disk_bytes": 4 * 1024**3,
            "unique_cache_salts": True,
            "quietness": "No continuous machine quietness trace; thermal state sampled at round boundaries.",
        },
        "rounds": rounds,
        "comparison": summarize(rounds),
    }
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()

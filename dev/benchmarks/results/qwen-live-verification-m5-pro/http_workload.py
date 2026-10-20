"""Exact workload snapshot for the October 20, 2026 M5 Pro measurements.

Run from the repository root with:
    python -m dev.benchmarks.results.qwen-live-verification-m5-pro.http_workload ...

Requires --model-root and --output; targets the Qwen3.6 MoE model named below.
This snapshot preserves the experiment; use the standard regression harnesses
for release qualification. It starts and stops its own servers.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

from dev.benchmarks import abba
from dev.tests import smoke_real as smoke

p = argparse.ArgumentParser()
p.add_argument("--threshold", type=float, default=0.1)
p.add_argument(
    "--baseline-binary", type=Path, default=Path("build/deepinfer-candidate/splash")
)
p.add_argument(
    "--candidate-binary", type=Path, default=Path("build/deepinfer-candidate/splash")
)
p.add_argument("--baseline-threshold", type=float, default=0)
p.add_argument("--model-root", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--tokens", type=int, default=512)
p.add_argument("--samples", type=int, default=2)
p.add_argument("--cases", default="code,math,prose,long-code,batch4")
p.add_argument("--warmup-seconds", type=float, default=0)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
model = "unsloth/Qwen3.6-35B-A3B-GGUF"
settings = argparse.Namespace(
    model_root=a.model_root,
    binary=a.candidate_binary,
    model=model,
    max_context=32768,
    max_memory="34G",
    max_cache_disk="0",
    max_image_pixels=None,
    kv_format="int8",
)
smoke.hold_model_root(settings)
base = [
    (
        "code",
        "Implement a thread-safe bounded LRU cache in Python with expiration, type annotations, documentation and comprehensive unit tests.",
    ),
    (
        "math",
        "Derive the central limit theorem for independent identically distributed random variables. Explain assumptions and every intermediate step in detail.",
    ),
    (
        "prose",
        "Explain how operating systems implement virtual memory, page tables, copy-on-write and page reclamation. Give extensive examples.",
    ),
    (
        "long-code",
        "Project notes: "
        + (
            "The service uses a durable queue with retries, timeouts and transactional state transitions. "
            * 600
        )
        + "\nImplement a complete Python job queue with persistence, retries, graceful shutdown and tests.",
    ),
]
selected = a.cases.split(",")
if set(selected) - {"code", "math", "prose", "long-code", "batch4"}:
    p.error("unknown case")
code_prompt = base[0][1]
base = [x for x in base if x[0] in selected]
keys = [
    "decode_cycle_ms",
    "decode_wall_ms",
    "decode_output_tokens",
    "drafted_tokens",
    "accepted_draft_tokens",
]


def req(server, content, **extra):
    body = smoke.chat_body(model, content, max_completion_tokens=a.tokens, **extra)
    code, response = smoke.request(
        server.port, "POST", "/v1/chat/completions", body, timeout=600
    )
    smoke.require(code == 200, f"HTTP {code}: {response}")
    smoke.require(response.get("choices"), f"missing choices: {response}")
    return {
        "usage": response["usage"],
        "message": response["choices"][0]["message"],
        "finish": response["choices"][0]["finish_reason"],
        "hash": hashlib.sha256(
            json.dumps(response["choices"][0]["message"], sort_keys=True).encode()
        ).hexdigest(),
    }


def measure(server, label, contents):
    _, before = smoke.request(server.port, "GET", "/status")
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(contents)) as pool:
        responses = list(pool.map(lambda content: req(server, content), contents))
    _, after = smoke.request(server.port, "GET", "/status")
    smoke.validate_status(after, "int8")
    smoke.require(
        after["state"]["active_lanes"] == 0 and after["kv"]["pages_active"] == 0,
        "active resources leaked",
    )
    delta = {k: after["metrics"][k] - before["metrics"][k] for k in keys}
    smoke.require(
        delta["decode_cycle_ms"] > 0 and delta["decode_output_tokens"] > 0,
        "missing decode metrics",
    )
    return {
        "id": label,
        "seconds_per_token": delta["decode_cycle_ms"]
        / 1000
        / delta["decode_output_tokens"],
        "wall_seconds": time.monotonic() - start,
        "delta": delta,
        "responses": responses,
    }


rounds = []
for index, on in enumerate([False, True, True, False], 1):
    settings.binary = a.candidate_binary if on else a.baseline_binary
    os.environ["SPLASH_LIVE_VERIFY_THRESHOLD"] = str(
        a.threshold if on else a.baseline_threshold
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.pop("MTL_SHADER_VALIDATION", None)
    server = smoke.RealServer(settings)
    try:
        status = server.wait_ready(1800)
        print(
            "Round",
            index,
            "candidate" if on else "baseline",
            "thermal",
            status.get("thermal_state"),
            flush=True,
        )
        req(server, "Write a Python function adding two integers and explain it.")
        warm_until = time.monotonic() + a.warmup_seconds
        while time.monotonic() < warm_until:
            req(
                server,
                "Write a complete annotated Python expression parser with precedence, syntax error recovery, documentation and comprehensive tests. Explain every design choice in detail.",
            )
        rows = []
        for sample in range(a.samples):
            for label, content in base:
                row = measure(
                    server,
                    label,
                    [content + f"\nUse descriptive names. Version {sample}."],
                )
                rows.append(row)
                print(
                    " ",
                    label,
                    round(1 / row["seconds_per_token"], 2),
                    "tok/s",
                    flush=True,
                )
            if "batch4" in selected:
                rows.append(
                    measure(
                        server,
                        "batch4",
                        [
                            code_prompt + f"\nImplement variant {sample}, request {i}."
                            for i in range(4)
                        ],
                    )
                )
        # Sampled and structured paths must remain usable with the feature enabled.
        sampled = req(
            server,
            "Explain binary search and provide annotated Python code.",
            temperature=0.7,
            seed=1234,
        )
        schema = {
            "type": "object",
            "properties": {"result": {"type": "string", "const": "ok"}},
            "required": ["result"],
            "additionalProperties": False,
        }
        structured = req(
            server,
            "Return the required JSON.",
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "probe", "strict": True, "schema": schema},
            },
        )
        smoke.require(
            json.loads(structured["message"]["content"]) == {"result": "ok"},
            "structured fallback failed",
        )
        with settings.binary.open("rb") as binary_file:
            binary_sha256 = hashlib.file_digest(binary_file, "sha256").hexdigest()
        rounds.append(
            {
                "enabled": on,
                "binary_sha256": binary_sha256,
                "identity": status.get("identity"),
                "rows": rows,
                "sampled": sampled,
                "structured": structured,
                "environment": {
                    "power": subprocess.check_output(
                        ["pmset", "-g", "batt"], text=True
                    ),
                    "thermal": status.get("thermal_state"),
                },
            }
        )
        (a.output / "rounds.json").write_text(json.dumps(rounds, indent=2) + "\n")
    finally:
        server.stop()
        server.log.seek(0)
        (a.output / f"server-{index}.log").write_text(server.log.read())
        server.close()
report = []
for label in [x[0] for x in base] + (["batch4"] if "batch4" in selected else []):
    values = [
        [r["seconds_per_token"] for r in record["rows"] if r["id"] == label]
        for record in rounds
    ]
    result = abba.compare_samples(values)
    result.update(
        id=label,
        tok_s=[1 / statistics.median(v) for v in values],
        output_equal_by_sample=[
            len(
                {
                    tuple(
                        x["hash"]
                        for x in [r for r in record["rows"] if r["id"] == label][
                            sample
                        ]["responses"]
                    )
                    for record in rounds
                }
            )
            == 1
            for sample in range(a.samples)
        ],
    )
    report.append(result)
    print(label, abba.describe(result), flush=True)
(a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

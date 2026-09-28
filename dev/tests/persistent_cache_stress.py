#!/usr/bin/env python3
"""Real-model persistence/churn/concurrency gate; writes only a temporary database.

Compare every deterministic completion against persistence-disabled references.
The JSON artifact keeps per-request timings, status and physical file sizes.
"""

import argparse
import ast
import concurrent.futures
import copy
import http.client
import json
import os
import signal
import sqlite3
import subprocess
import tempfile
import time
from contextlib import closing
from pathlib import Path

from persistent_cache_real import settle
from smoke_real import (
    RealServer,
    chat_body,
    request,
    request_headers,
    require,
    validate_status,
)


def comparable_message(message):
    """Normalize Python presentation in persistence-disabled batch controls.

    Preserve the raw output in the report. Never execute generated code, and
    do not normalize names, statements, values, reasoning or tool calls.
    Both fenced and bare functions occur in persistence-disabled pair batches.
    """
    content = message.get("content", "")
    if not isinstance(content, str):
        return message
    if content.startswith("```python\n") and content.endswith("```"):
        content = content[len("```python\n") : -3]
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return message
    if len(tree.body) != 1 or not isinstance(
        tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        return message
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.returns = None
        elif isinstance(node, ast.arg):
            node.annotation = None
    return {**message, "content": ast.dump(tree)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--binary", type=Path, default=Path("build/splash"))
    parser.add_argument("--max-memory", default="24G")
    parser.add_argument("--kv-format", choices=("int8", "bf16"), default="int8")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--quota", default="512M")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.max_context = None
    args.persistent_cache = None
    args.max_cache_disk = None
    records = "\n".join(
        f"Record {i}: apple banana cherry date elderberry." for i in range(260)
    )
    bodies = []
    for i in range(8):
        # Four divergent sessions share a large system prompt; four have unique
        # prefixes to force real quota pressure instead of only deduplicating.
        system = (
            "You are a precise assistant.\n" if i < 4 else f"Workspace {i}.\n"
        ) + records
        body = chat_body(args.model, "", max_completion_tokens=32)
        body["messages"] = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Write a Python function adding {i + 2} to an integer. Code only.",
            },
        ]
        bodies.append(body)
    report = {
        "model": args.model,
        "kv_format": args.kv_format,
        "rounds": args.rounds,
        "stages": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def complete(server, body):
        start = time.monotonic()
        status, response = request(
            server.port, "POST", "/v1/chat/completions", body, timeout=240
        )
        require(status == 200, f"HTTP {status}: {response}")
        return {"elapsed_ms": 1000 * (time.monotonic() - start), "response": response}

    def batch(server, requests, parallel):
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
            return list(pool.map(lambda body: complete(server, body), requests))

    def files():
        return {
            file.name: file.stat().st_size
            for file in args.cache_file.parent.glob("prefix.sqlite*")
        }

    def kill(server):
        children = subprocess.check_output(
            ["pgrep", "-P", str(server.process.pid)], text=True
        ).split()
        require(bool(children), "native child missing")
        for pid in children:
            os.kill(int(pid), signal.SIGKILL)
        server.process.kill()
        server.process.wait(10)

    with tempfile.TemporaryDirectory(prefix="splash-stress-") as directory:
        args.cache_file = Path(directory) / "prefix.sqlite"
        baseline = None
        concurrent_baseline = None
        for stage, persistent, temporary, parallel in (
            ("reference", None, None, 1),
            ("reference-concurrent", None, None, 4),
            ("fill", args.quota, None, 1),
            ("restart-concurrent", args.quota, None, 4),
            ("both-churn", args.quota, "256M", 4),
            ("shrink", "256M", None, 2),
            ("grow", "4G", None, 4),
        ):
            args.persistent_cache, args.max_cache_disk = persistent, temporary
            server = RealServer(
                args, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            )
            entry = {"stage": stage, "batches": []}
            report["stages"].append(entry)
            try:
                entry["before"] = server.wait_ready(180)
                if stage in ("restart-concurrent", "both-churn"):
                    require(
                        entry["before"]["persistent_cache"]["restored"] > 0,
                        f"{stage}: no committed prefixes survived restart",
                    )
                rounds = args.rounds if stage == "both-churn" else 1
                for turn in range(rounds):
                    results = batch(server, bodies, parallel)
                    entry["batches"].append(results)
                    save()
                    if baseline is None:
                        baseline = [
                            x["response"]["choices"][0]["message"] for x in results
                        ]
                    if stage == "reference-concurrent":
                        concurrent_baseline = [
                            x["response"]["choices"][0]["message"] for x in results
                        ]
                    for i, result in enumerate(results):
                        if result["response"]["choices"][0]["message"] != baseline[i]:
                            report.setdefault("output_differences", []).append(
                                {
                                    "stage": stage,
                                    "round": turn,
                                    "request": i,
                                    "expected": baseline[i],
                                    "actual": result["response"]["choices"][0][
                                        "message"
                                    ],
                                    "matched_tokens": result["response"]["metrics"][
                                        "cache"
                                    ]["matched_tokens"],
                                }
                            )
                        allowed = [baseline[i]]
                        if parallel > 1 and concurrent_baseline is not None:
                            allowed.append(concurrent_baseline[i])
                        actual_message = result["response"]["choices"][0]["message"]
                        if parallel > 1:
                            actual_message = comparable_message(actual_message)
                            allowed = [
                                comparable_message(message) for message in allowed
                            ]
                        require(
                            actual_message in allowed,
                            f"{stage}/{turn}/{i}: output differs from both persistence-disabled controls",
                        )
                    status = settle(server)
                    validate_status(status, args.kv_format)
                    require(
                        status["persistent_cache"]["failures"] == 0,
                        f"{stage}: persistence failed: {status}",
                    )
                    require(
                        status["persistent_cache"]["used_bytes"]
                        <= status["persistent_cache"]["capacity_bytes"],
                        "durable quota exceeded",
                    )
                    entry["after"] = status
                    entry["files"] = files()
                    save()
                    print(
                        json.dumps(
                            {
                                "stage": stage,
                                "round": turn,
                                "persistent": status["persistent_cache"],
                                "matched": [
                                    x["response"]["metrics"]["cache"]["matched_tokens"]
                                    for x in results
                                ],
                                "files": entry["files"],
                            }
                        ),
                        flush=True,
                    )
                if stage == "grow":
                    # Once the quota holds the complete working set, immutable
                    # hot prefixes must cause no further payload writes.
                    before_written = entry["after"]["disk"]["written_bytes"]
                    report["steady_warm_requests"] = batch(server, bodies, parallel)
                    report["steady_warm_status"] = settle(server)
                    report["warm_payload_bytes_written"] = (
                        report["steady_warm_status"]["disk"]["written_bytes"]
                        - before_written
                    )
                    require(
                        report["warm_payload_bytes_written"] == 0,
                        "hot prefixes rewrote persisted payloads with sufficient quota",
                    )
                    # A real multi-turn continuation, including the assistant's
                    # actual answer, must also survive another process lifetime.
                    continuation = copy.deepcopy(bodies[0])
                    continuation["messages"] += [
                        baseline[0],
                        {"role": "user", "content": "Now add a docstring. Code only."},
                    ]
                    continuation["tools"] = [
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "description": "Read a project file.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"path": {"type": "string"}},
                                    "required": ["path"],
                                },
                            },
                        }
                    ]
                    continuation["messages"] += [
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_read",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"main.py"}',
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call_read",
                            "content": baseline[0]["content"],
                        },
                        {
                            "role": "user",
                            "content": "Return the revised function with a docstring. No tools.",
                        },
                    ]
                    expected = complete(server, continuation)
                    settle(server)
                    report["continuation_before"] = expected
                    # Cancel a streaming request after the first response byte;
                    # cache writes and request cleanup must not poison peers.
                    conn = http.client.HTTPConnection(
                        "127.0.0.1", server.port, timeout=120
                    )
                    cancelled = copy.deepcopy(bodies[1])
                    cancelled.update(stream=True, max_completion_tokens=256)
                    conn.request(
                        "POST",
                        "/v1/chat/completions",
                        json.dumps(cancelled),
                        request_headers(True),
                    )
                    response = conn.getresponse()
                    require(response.status == 200, "cancellation request failed")
                    response.read(1)
                    response.close()
                    conn.close()
                    report["after_cancel"] = complete(server, bodies[2])
                    report["cancel_status"] = settle(server)
                    require(
                        report["cancel_status"]["requests"]["cancelled"] > 0,
                        "stream fixture did not cancel a live request",
                    )
                if stage in ("fill", "both-churn", "grow"):
                    kill(server)
            except Exception:
                entry["log"] = server.tail()
                save()
                print(entry["log"], flush=True)
                raise
            finally:
                server.close()
            if persistent:
                with closing(sqlite3.connect(args.cache_file)) as db:
                    require(
                        db.execute("PRAGMA integrity_check").fetchone()[0] == "ok",
                        "SQLite integrity failed",
                    )
                    require(
                        not db.execute("PRAGMA foreign_key_check").fetchall(),
                        "broken publication references",
                    )
        args.persistent_cache = "4G"
        args.max_cache_disk = None
        server = RealServer(args, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        try:
            server.wait_ready(180)
            actual = complete(server, continuation)
            report["continuation_after"] = actual
            require(
                actual["response"]["choices"][0]["message"]
                == expected["response"]["choices"][0]["message"],
                "multi-turn restart changed completion",
            )
            require(
                actual["response"]["metrics"]["cache"]["matched_tokens"] > 2048,
                "multi-turn restart missed",
            )
            report["final"] = settle(server)
            save()
        finally:
            server.close()
        # Damage the newest complete state while the database is closed. A
        # request must fall back to a healthy boundary, repair, and stay usable.
        with closing(sqlite3.connect(args.cache_file)) as db:
            db.execute(
                "UPDATE slots SET checksum=checksum+1 WHERE id=(SELECT id FROM prefixes ORDER BY used DESC LIMIT 1)"
            )
            db.commit()
        server = RealServer(args, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        try:
            server.wait_ready(180)
            repaired = complete(server, continuation)
            require(
                repaired["response"]["choices"][0]["message"]
                == expected["response"]["choices"][0]["message"],
                "corrupted state changed completion instead of falling back",
            )
            report["corruption_recovery"] = {
                "result": repaired,
                "status": settle(server),
            }
            require(
                report["corruption_recovery"]["status"]["state"]["invalidations"] > 0,
                "corruption fixture did not exercise a failed state restore",
            )
            save()
            # Interrupt the actual GPU/server path with publication outstanding,
            # in addition to the storage-level randomized crash tests.
            interrupted = chat_body(
                args.model,
                "Independent workspace.\n"
                + records * 3
                + "\nWrite a long Python program.",
                max_completion_tokens=256,
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(complete, server, interrupted)
                deadline = time.monotonic() + 90
                while True:
                    _, status = request(server.port, "GET", "/status")
                    if status["persistent_cache"]["writing"]:
                        report["interrupted_publication"] = status["persistent_cache"]
                        kill(server)
                        break
                    require(
                        not future.done() and time.monotonic() < deadline,
                        "could not interrupt an outstanding real publication",
                    )
                    time.sleep(0.01)
                try:
                    future.result()
                except (OSError, http.client.HTTPException, RuntimeError):
                    pass
            save()
        finally:
            server.close()
        server = RealServer(args, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        try:
            server.wait_ready(180)
            recovered = complete(server, continuation)
            require(
                recovered["response"]["choices"][0]["message"]
                == expected["response"]["choices"][0]["message"],
                "interrupted publication changed the earlier committed continuation",
            )
            require(
                recovered["response"]["metrics"]["cache"]["matched_tokens"] > 2048,
                "interrupted publication lost the earlier committed continuation",
            )
            report["interrupted_recovery"] = {
                "result": recovered,
                "status": settle(server),
            }
            save()
        finally:
            server.close()
        with closing(sqlite3.connect(args.cache_file)) as db:
            db.execute("""UPDATE slots SET checksum=checksum+1 WHERE id=(
                SELECT slot FROM refs WHERE prefix=(SELECT id FROM prefixes ORDER BY used DESC LIMIT 1)
                AND slot != prefix ORDER BY slot LIMIT 1)""")
            db.commit()
        server = RealServer(args, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        try:
            server.wait_ready(180)
            recovered = complete(server, continuation)
            require(
                recovered["response"]["choices"][0]["message"]
                == expected["response"]["choices"][0]["message"],
                "corrupted KV changed completion instead of falling back",
            )
            report["kv_corruption_recovery"] = {
                "result": recovered,
                "status": settle(server),
            }
            require(
                report["kv_corruption_recovery"]["status"]["disk"][
                    "kv_restore_failures"
                ]
                > 0,
                "corruption fixture did not exercise a failed KV restore",
            )
            save()
        finally:
            server.close()
    print("Persistent cache stress gate passed", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Real-model persistence/churn/concurrency gate; writes only a temporary database.

Compare serial completions exactly; concurrent batches check restore and response properties.
The JSON artifact keeps per-request timings, status and physical file sizes.
"""

import argparse
import concurrent.futures
import copy
import http.client
import json
import os
import signal
import sqlite3
import struct
import subprocess
import tempfile
import time
from contextlib import closing, contextmanager
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


def corrupt_component(cache_file, kind):
    with closing(sqlite3.connect(cache_file)) as db:
        blob = db.execute(
            "SELECT metadata FROM prefixes ORDER BY used DESC LIMIT 1"
        ).fetchone()[0]
        words = struct.unpack(f"<{len(blob) // 8}Q", blob)
        require(words[0] == 3, "unexpected persistent metadata version")
        cursor = 3
        group_slots = {0: [], 1: []}
        for _ in range(words[2]):
            group, begin, end, metadata_count, record_count = words[cursor : cursor + 5]
            cursor += 5 + metadata_count
            if group in group_slots:
                group_slots[group].extend(words[cursor : cursor + 2 * record_count : 2])
            cursor += 2 * record_count
        group = 0 if kind == "recurrent" else 1
        require(group_slots[group], "state component fixture is empty")
        slot = words[cursor + 1] if kind == "target" else group_slots[group][0]
        db.execute("UPDATE slots SET checksum=checksum+1 WHERE id=?", (slot,))
        db.commit()


def crc32c(data):
    value = 0xFFFFFFFF
    for byte in data:
        value ^= byte
        for _ in range(8):
            value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
    return value ^ 0xFFFFFFFF


def corrupt_group_range(cache_file):
    # A descriptor spanning two fixed pages must be rejected before import.
    with closing(sqlite3.connect(cache_file)) as db:
        prefix, blob = db.execute(
            "SELECT id, metadata FROM prefixes ORDER BY used DESC LIMIT 1"
        ).fetchone()
        words = list(struct.unpack(f"<{len(blob) // 8}Q", blob))
        require(words[0] == 3, "unexpected persistent metadata version")
        cursor = 3
        for _ in range(words[2]):
            group, begin, end, metadata_count, record_count = words[cursor : cursor + 5]
            if group == 1 and begin >= 32:
                words[cursor + 1] = begin - 32
                changed = struct.pack(f"<{len(words)}Q", *words)
                # Keep the checksum valid so this exercises descriptor validation.
                db.execute(
                    "UPDATE prefixes SET metadata=?, checksum=? WHERE id=?",
                    (changed, crc32c(changed), prefix),
                )
                db.commit()
                return
            cursor += 5 + metadata_count + 2 * record_count
        raise AssertionError("no complete draft page in range-corruption fixture")


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
                    for i, result in enumerate(results):
                        response = result["response"]
                        choice = response["choices"][0]
                        if parallel == 1:
                            require(
                                choice["message"] == baseline[i],
                                f"{stage}/{turn}/{i}: serial output differs from reference",
                            )
                        else:
                            require(
                                choice["finish_reason"] in ("stop", "length")
                                and isinstance(choice["message"].get("content"), str)
                                and bool(choice["message"]["content"]),
                                f"{stage}/{turn}/{i}: malformed concurrent response",
                            )
                        require(
                            0
                            <= response["metrics"]["cache"]["matched_tokens"]
                            <= response["usage"]["prompt_tokens"],
                            f"{stage}/{turn}/{i}: invalid restored prefix length",
                        )
                    if stage == "restart-concurrent":
                        require(
                            any(
                                r["response"]["metrics"]["cache"]["matched_tokens"] > 0
                                for r in results
                            ),
                            "concurrent restart restored no cached tokens",
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
                    require(
                        status["disk"]["used_bytes"]
                        <= status["disk"]["capacity_bytes"],
                        "temporary quota exceeded",
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
                    before_written = sum(
                        entry["after"][tier]["written_bytes"]
                        for tier in ("disk", "persistent_cache")
                    )
                    report["steady_warm_requests"] = batch(server, bodies, parallel)
                    report["steady_warm_status"] = settle(server)
                    report["warm_payload_bytes_written"] = (
                        sum(
                            report["steady_warm_status"][tier]["written_bytes"]
                            for tier in ("disk", "persistent_cache")
                        )
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

        @contextmanager
        def running_server():
            server = RealServer(
                args, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            )
            try:
                yield server, server.wait_ready(180)
            finally:
                server.close()

        def resume(server, label, must_hit=False):
            actual = complete(server, continuation)
            require(
                actual["response"]["choices"][0]["message"]
                == expected["response"]["choices"][0]["message"],
                f"{label}: restart changed the serial continuation",
            )
            if must_hit:
                require(
                    actual["response"]["metrics"]["cache"]["matched_tokens"] > 2048,
                    f"{label}: committed continuation was lost",
                )
            report[label] = {"result": actual, "status": settle(server)}
            save()
            return report[label]

        with running_server() as (server, _):
            resume(server, "continuation_after", must_hit=True)
            # Kill the actual serving process with publication outstanding.
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
        with running_server() as (server, _):
            resume(server, "interrupted_recovery", must_hit=True)

        # Damage one layer at a time, with a healthy server restart between
        # cases. Every case must exercise its failure counter and preserve the
        # unmodified serial completion, whether recovery hits or recomputes.
        for kind, label, counter in (
            ("recurrent", "corruption_recovery", ("state", "invalidations")),
            ("draft", "draft_corruption_recovery", ("state", "invalidations")),
            ("target", "kv_corruption_recovery", ("disk", "kv_restore_failures")),
            ("range", "range_corruption_recovery", ("persistent_cache", "failures")),
        ):
            if kind == "range":
                corrupt_group_range(args.cache_file)
            else:
                corrupt_component(args.cache_file, kind)
            with running_server() as (server, startup):
                result = resume(server, label)
                result["startup"] = startup
                observed = startup if kind == "range" else result["status"]
                require(
                    observed[counter[0]][counter[1]] > 0,
                    f"{kind}: corrupt data was not rejected at the expected layer",
                )
                save()
    print("Persistent cache stress gate passed", flush=True)


if __name__ == "__main__":
    main()

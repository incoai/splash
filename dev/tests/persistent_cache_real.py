#!/usr/bin/env python3
"""Manual real-model gate: runtime save, abrupt exit and equivalent resumed decode.

Uses only temporary cache files and loopback ports; never touches an existing
server. Run with the same Python environment as smoke_real.py.
"""

import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from smoke_real import RealServer, chat_body, request, require


def settle(server):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        _, status = request(server.port, "GET", "/status")
        if not status["persistent_cache"]["writing"]:
            return status
        time.sleep(0.05)
    raise RuntimeError("persistent publication did not finish")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--binary", type=Path, default=Path("build/splash"))
    parser.add_argument("--max-memory", default="auto")
    parser.add_argument("--kv-format", choices=("int8", "bf16"), default="int8")
    parser.add_argument(
        "--records",
        type=int,
        default=260,
        help="number of reference records; increase to exercise longer prefixes",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.max_context = None
    prompt = (
        "Reference records:\n"
        + "\n".join(
            f"Record {i}: apple banana cherry date elderberry."
            for i in range(args.records)
        )
        + "\nWrite a Python function returning the Fibonacci numbers below 100. Code only."
    )
    body = chat_body(args.model, prompt, max_completion_tokens=96)
    results = []
    with tempfile.TemporaryDirectory(prefix="splash-real-persistence-") as directory:
        args.cache_file = Path(directory) / "prefix.sqlite"
        for stage, persistent, temporary in (
            ("cold", "2G", None),
            ("restart", "2G", None),
            ("both", "2G", "1G"),
            ("disabled", None, None),
            ("temporary-only", None, "1G"),
            ("reenabled", "2G", None),
        ):
            args.persistent_cache, args.max_cache_disk = persistent, temporary
            server = RealServer(
                args, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            )
            try:
                before = server.wait_ready(180)
                http_status, response = request(
                    server.port, "POST", "/v1/chat/completions", body, timeout=180
                )
                require(http_status == 200, f"request failed: {response}")
                after = settle(server)
                metrics = response["metrics"]
                result = {
                    "stage": stage,
                    "response": response,
                    "before": before,
                    "after": after,
                }
                results.append(result)
                print(
                    json.dumps(
                        {
                            "stage": stage,
                            "metrics": metrics,
                            "persistent": after["persistent_cache"],
                        }
                    ),
                    flush=True,
                )
                if stage == "cold":
                    require(
                        after["persistent_cache"]["saved"] > 0,
                        "RAM prefix was not saved at runtime",
                    )
                elif persistent:
                    require(
                        before["persistent_cache"]["restored"] > 0,
                        "restart did not load an index",
                    )
                    require(
                        metrics["cache"]["matched_tokens"] > 2048,
                        "test did not restore past the draft window",
                    )
                else:
                    require(
                        metrics["cache"]["matched_tokens"] == 0,
                        "disabled persistence loaded a prefix",
                    )
                require(
                    response["choices"][0]["message"]
                    == results[0]["response"]["choices"][0]["message"],
                    "restored greedy generation differs from the cold run",
                )
                if stage == "cold":
                    # These are this test's own processes. Kill native first,
                    # then the frontend, so neither can export cache on exit.
                    native = subprocess.check_output(
                        ["pgrep", "-P", str(server.process.pid)], text=True
                    ).split()
                    require(bool(native), "native child not found")
                    for child in native:
                        os.kill(int(child), signal.SIGKILL)
                    server.process.kill()
                    server.process.wait(10)
            except Exception:
                print(server.tail(), flush=True)
                raise
            finally:
                server.close()
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    print("Persistent real-model gate passed", flush=True)


if __name__ == "__main__":
    main()

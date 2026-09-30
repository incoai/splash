#!/usr/bin/env python3
"""Real-model gate for optional persistence failures and temporary-tier fallback."""

import argparse
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

from smoke_real import RealServer, chat_body, request, require, validate_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--binary", type=Path, default=Path("build/splash"))
    parser.add_argument("--max-memory", default="24G")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.max_context = None
    args.kv_format = "int8"
    args.persistent_cache = "512M"
    args.max_cache_disk = None
    environment = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    results = []
    with tempfile.TemporaryDirectory(prefix="splash-startup-fallback-") as directory:
        root = Path(directory)
        args.cache_file = root / "seed.sqlite"
        seed = RealServer(args, environment)
        try:
            validate_status(seed.wait_ready(180))
        finally:
            seed.close()
        for failure in ("corrupt-index", "locked-index", "mismatched-payload"):
            for temporary in (None, "256M"):
                args.cache_file = root / f"{failure}-{temporary}.sqlite"
                args.max_cache_disk = temporary
                for suffix in ("", ".data"):
                    shutil.copyfile(
                        str(root / "seed.sqlite") + suffix,
                        str(args.cache_file) + suffix,
                    )
                lock = None
                if failure == "corrupt-index":
                    args.cache_file.write_bytes(b"not a SQLite database")
                elif failure == "locked-index":
                    lock = sqlite3.connect(args.cache_file)
                    lock.execute("BEGIN EXCLUSIVE")
                else:
                    with Path(str(args.cache_file) + ".data").open("r+b") as data:
                        data.seek(8)
                        original = data.read(1)[0]
                        data.seek(8)
                        data.write(bytes((original ^ 1,)))
                server = None
                try:
                    server = RealServer(args, environment)
                    before = server.wait_ready(180)
                    validate_status(before)
                    require(
                        before["persistent_cache"]["capacity_bytes"] == 0,
                        "broken persistence remained enabled",
                    )
                    require(
                        before["disk"]["capacity_bytes"]
                        == (256 * 1024**2 if temporary else 0),
                        "fallback changed the requested temporary quota",
                    )
                    code, response = request(
                        server.port,
                        "POST",
                        "/v1/chat/completions",
                        chat_body(args.model, "Reply OK.", max_completion_tokens=8),
                        timeout=180,
                    )
                    require(code == 200, f"fallback request failed: {response}")
                    log = server.tail()
                    require(
                        str(args.cache_file) in log,
                        "startup diagnostic omitted the cache path",
                    )
                    results.append(
                        {
                            "failure": failure,
                            "temporary": temporary,
                            "before": before,
                            "response": response,
                            "log": log,
                        }
                    )
                    args.output.write_text(json.dumps(results, indent=2) + "\n")
                    print(f"passed {failure}, temporary={temporary}", flush=True)
                finally:
                    if server:
                        server.close()
                    if lock:
                        lock.close()
    print("Persistent startup fallback gate passed", flush=True)


if __name__ == "__main__":
    main()

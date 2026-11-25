#!/usr/bin/env python3
"""Run a paired LoopCD benchmark from a Splash source checkout (Python stdlib only)."""

import argparse
from contextlib import contextmanager
from decimal import Decimal
import json
import os
from pathlib import Path
import random
import re
import secrets
import signal
import socket
import statistics
import subprocess
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
MODEL = "unsloth/Inkling-2-Mini-GGUF:UD-IQ1_S"
GSM8K = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl"
)
SPEED_PROMPTS = [
    f"Write a detailed 800-word beginner's guide to {topic}."
    for topic in (
        "growing tomatoes", "making sourdough", "repairing a bicycle tire",
        "learning to draw", "planning a hiking trip", "organizing family photos",
    )
]
FINAL_NUMBER = re.compile(r"(?m)^\s*(?:####\s*|Answer:\s*)(\$?[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?:[.!])?[ \t]*$")


def final_number(text):
    matches = FINAL_NUMBER.findall(text)
    return Decimal(matches[-1].replace(",", "").lstrip("$")) if matches else None


def questions(count):
    with urlopen(GSM8K, timeout=60) as response:
        rows = [json.loads(line) for line in response]
    return [(i, rows[i]) for i in random.Random(42).sample(range(len(rows)), count)]


def request(port, key, path, data=None, timeout=240):
    req = Request(
        f"http://127.0.0.1:{port}{path}",
        None if data is None else json.dumps(data).encode(),
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)


@contextmanager
def server(args, enabled, log_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    key = secrets.token_hex(16)
    command = [
        "./splash", "serve", "--model", args.model, "--language-only", "--max-context", "2K",
        "--max-concurrent-requests", "1", "--verify-rows", "all", "--reasoning-effort", "none",
        "--host", "127.0.0.1", "--port", str(port), "--api-key", key, "--no-webui",
    ]
    if enabled:
        command.append("--loop-cd")
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            deadline = time.monotonic() + 3600
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    request(port, key, "/ready", timeout=3)
                    model = request(port, key, "/v1/models", timeout=3)["data"][0]["id"]
                except (URLError, TimeoutError):
                    time.sleep(1)
                else:
                    break
            else:
                raise RuntimeError(f"Splash did not become ready (exit={process.poll()}); see {log_path}")
            yield port, key, model
        finally:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def complete(connection, prompt, max_tokens):
    port, key, model = connection
    started = time.monotonic()
    response = request(port, key, "/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": max_tokens,
    })
    choice = response["choices"][0]
    return {
        "text": choice["message"].get("content") or "", "finish_reason": choice["finish_reason"],
        "timings": response["timings"], "wall_s": round(time.monotonic() - started, 4),
    }


def summarize(records):
    summary = {}
    answers = {}
    for mode in ("off", "on"):
        speed = [r["timings"] for r in records if r["mode"] == mode and r["kind"] == "speed"]
        usable = [t["predicted_per_second"] for t in speed
                  if t["predicted_n"] >= 32 and t["predicted_per_second"] > 0]
        drafted = sum(t.get("draft_n", 0) for t in speed)
        accepted = sum(t.get("draft_n_accepted", 0) for t in speed)
        math = [r for r in records if r["mode"] == mode and r["kind"] == "gsm8k"]
        answers[mode] = {r["id"]: r["correct"] for r in math}
        summary[mode] = {
            "decode_tok_s_median": statistics.median(usable) if usable else None,
            "decode_samples": len(usable), "speed_requests": len(speed),
            "first_output_ms_median": statistics.median(t["prompt_ms"] for t in speed),
            "output_tokens_median": statistics.median(t["predicted_n"] for t in speed),
            "draft_acceptance": accepted / drafted if drafted else None,
            "gsm8k_correct": sum(answers[mode].values()), "gsm8k_questions": len(math),
            "gsm8k_unparsed": sum(r["answer"] is None for r in math),
        }
    pairs = [(answers["off"][i], answers["on"][i]) for i in answers["off"].keys() & answers["on"].keys()]
    summary["paired_gsm8k"] = {"wrong_to_right": pairs.count((False, True)),
                               "right_to_wrong": pairs.count((True, False))}
    off, on = (summary[mode]["decode_tok_s_median"] for mode in ("off", "on"))
    summary["decode_change_pct"] = 100 * (on / off - 1) if on is not None and off else None
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--questions", type=int, default=100, help="GSM8K sample size (0 for speed only)")
    args = parser.parse_args()
    if not 0 <= args.questions <= 1319:
        parser.error("--questions must be between 0 and 1319")
    if not os.access(ROOT / "splash", os.X_OK):
        parser.error(f"run chmod +x {ROOT / 'splash'} first")
    out = Path(tempfile.mkdtemp(prefix="loop-cd-", dir=Path.cwd()))
    print(f"Results and server logs: {out}", flush=True)
    sample = questions(args.questions) if args.questions else []
    records = []
    (out / "setup.json").write_text(json.dumps({"model": args.model, "dataset": GSM8K,
                                                "question_ids": [i for i, _ in sample]}, indent=2))
    with (out / "requests.jsonl").open("w", buffering=1) as raw:
        for run, mode in enumerate(("on", "off", "off", "on"), 1):
            log = out / f"{run}-{mode}.log"
            print(f"Run {run}/4: LoopCD {mode} (see {log.name} while Splash starts)", flush=True)
            with server(args, mode == "on", log) as connection:
                complete(connection, "Describe a rainbow in two sentences.", 32)
                for i, prompt in enumerate(SPEED_PROMPTS):
                    record = {"mode": mode, "run": run, "kind": "speed", "id": i,
                              **complete(connection, prompt, 256)}
                    records.append(record)
                    raw.write(json.dumps(record) + "\n")
                # Measure speed first in every run; match accuracy prompts between modes.
                if run >= 3:
                    for n, (i, row) in enumerate(sample, 1):
                        prompt = (f"{row['question']}\nShow your work, then put the final numeric "
                                  "answer on a separate line beginning with ####, with no units.")
                        record = complete(connection, prompt, 768)
                        gold, answer = final_number(row["answer"]), final_number(record["text"])
                        if gold is None:
                            raise ValueError(f"Invalid GSM8K reference answer at row {i}")
                        record.update(mode=mode, run=run, kind="gsm8k", id=i, correct=answer == gold,
                                      answer=None if answer is None else str(answer), expected=str(gold))
                        records.append(record)
                        raw.write(json.dumps(record) + "\n")
                        if n % 10 == 0 or n == len(sample):
                            print(f"  GSM8K {mode}: {n}/{len(sample)}", flush=True)
    result = json.dumps(summarize(records), indent=2)
    (out / "summary.json").write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()

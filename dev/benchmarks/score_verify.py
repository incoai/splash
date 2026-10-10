#!/usr/bin/env python3
"""Verification of POST /v1/score against a running Splash server.

Each check prints its numbers and appends a JSON record to --out.

  score_verify.py agree      --jsonl FILE     argmax vs. the first generated token
  score_verify.py determinism                 one request 20x
  score_verify.py concurrency                 1 / 4 / 24 concurrent requests
  score_verify.py cache                       cold vs. warm prefix cache
  score_verify.py sanity                      probabilities, mass, 400 on a multi-token label
  score_verify.py perf       --jsonl FILE     latency and throughput vs. generation
  score_verify.py memory     --count N        RSS and KV pages over N requests
  score_verify.py image      --image FILE     one scored request with an image
  score_verify.py regress    --save F | --check-file F   Chat outputs on a fixed prompt set

JSONL items: {"system": str, "user": str, "prefix": str, "labels": [str, ...]}
(labels default to [" true", " false"], prefix to '{"same":').
Common options: --url, --model, --api-key (default $SPLASH_API_KEY).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import statistics
import subprocess
import time
import urllib.error
import urllib.request
import uuid

DEFAULT_PREFIX = '{"same":'
DEFAULT_LABELS = [" true", " false"]


class Client:
    def __init__(self, url, model, api_key):
        self.url = url.rstrip("/")
        self.model = model
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = "Bearer " + api_key

    def post(self, path, body, timeout=3600):
        request = urllib.request.Request(
            self.url + path, json.dumps(body).encode(), self.headers
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def get(self, path):
        request = urllib.request.Request(self.url + path, headers=self.headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)


def messages_of(item):
    messages = []
    if item.get("system"):
        messages.append({"role": "system", "content": item["system"]})
    messages.append({"role": "user", "content": item["user"]})
    return messages


def score_body(client, item, **extra):
    return {
        "model": client.model,
        "messages": item.get("messages") or messages_of(item),
        "assistant_prefix": item.get("prefix", DEFAULT_PREFIX),
        "labels": item.get("labels", DEFAULT_LABELS),
        "reasoning_effort": "none",
        **extra,
    }


def load_items(path):
    with open(path) as file:
        return [json.loads(line) for line in file if line.strip()]


SYNTHETIC_PREFIX = os.environ.get("SCORE_PREFIX", '{\n"same":')


def synthetic_items(count):
    """Product-matching prompts for runs without the client's JSONL."""
    pairs = [
        ("Bosch GSR 12V-15 2x2Ah", "Bosch GSR 12V-15 Professional, 2 aku 2.0 Ah"),
        ("Makita DHP482Z", "Makita DHP482RFJ 18V 3Ah"),
        ("Philips HD9252/90 Airfryer", "Philips Essential Airfryer HD9252/90"),
        ("Samsung Galaxy S24 128GB černý", "Samsung Galaxy S24 256GB černý"),
        ("Kärcher K 5 Power Control", "Kärcher K5 Power Control Home"),
        ("Apple AirPods Pro 2 USB-C", "Apple AirPods 3"),
        ("LEGO 42115 Lamborghini Sián", "LEGO Technic 42115 Lamborghini Sián FKP 37"),
        ("Dyson V15 Detect Absolute", "Dyson V15 Detect Absolute filtr"),
    ]
    system = (
        "You compare two product listings from different shops. Decide whether "
        "they are the same product (same model and variant). Answer only with "
        'JSON {"same": true|false, "difference": "..."}.'
    )
    items = []
    for index in range(count):
        a, b = pairs[index % len(pairs)]
        items.append(
            {
                "system": system,
                "user": f"Listing {index}.\nA: {a}\nB: {b}",
                "prefix": SYNTHETIC_PREFIX,
                "labels": DEFAULT_LABELS,
            }
        )
    return items


def record(args, name, data):
    print(json.dumps({name: data}, ensure_ascii=False), flush=True)
    if args.out:
        with open(args.out, "a") as file:
            file.write(
                json.dumps(
                    {"check": name, "time": time.time(), **data}, ensure_ascii=False
                )
                + "\n"
            )


def items_of(args):
    return load_items(args.jsonl) if args.jsonl else synthetic_items(args.count)


# ---------------------------------------------------------------- checks


def check_agree(client, args):
    """Argmax of /v1/score against (a) the token /v1/chat/completions writes
    at T=0 right after the prefix and (b) /v1/completions over the same
    rendered prompt plus prefix, one token at T=0."""
    items = items_of(args)
    agree_chat = agree_completion = compared_chat = 0
    prefix_mismatch = []
    natural_prefixes = {}
    errors = []
    for index, item in enumerate(items):
        status, score = client.post("/v1/score", score_body(client, item))
        if status != 200:
            errors.append({"index": index, "score_status": status, "error": score})
            continue
        prefix = item.get("prefix", DEFAULT_PREFIX)
        labels = item.get("labels", DEFAULT_LABELS)
        status, chat = client.post(
            "/v1/chat/completions",
            {
                "model": client.model,
                "messages": item.get("messages") or messages_of(item),
                "temperature": 0,
                "max_tokens": 64,
                "reasoning_effort": "none",
            },
        )
        content = (
            (chat["choices"][0]["message"].get("content") or "")
            if status == 200
            else ""
        )
        # The prefix the model itself writes: its output up to the end of the
        # prefix's key (for '{"same":' the text through '"same":'), which may
        # differ from the given prefix in whitespace such as '{\n"same":'.
        key = prefix.lstrip("{[ \n")
        position = content.find(key) if key else -1
        if position < 0:
            prefix_mismatch.append({"index": index, "generated": content[:60]})
        else:
            natural = content[: position + len(key)]
            after = content[position + len(key) :]
            generated = next(
                (label for label in labels if after.startswith(label)), None
            )
            if natural != prefix:
                natural_prefixes[natural] = natural_prefixes.get(natural, 0) + 1
                status, natural_score = client.post(
                    "/v1/score", score_body(client, {**item, "prefix": natural})
                )
                if status != 200:
                    errors.append(
                        {
                            "index": index,
                            "kind": "chat",
                            "natural_prefix": natural,
                            "score_status": status,
                            "error": natural_score,
                        }
                    )
                    continue
            else:
                natural_score = score
            compared_chat += 1
            if generated == natural_score["argmax"]:
                agree_chat += 1
            else:
                errors.append(
                    {
                        "index": index,
                        "kind": "chat",
                        "prefix": natural,
                        "argmax": natural_score["argmax"],
                        "generated": after[:20],
                        "labels": natural_score["labels"],
                        "item": item,
                    }
                )
        status, template = client.post(
            "/apply-template",
            {
                "model": client.model,
                "messages": item.get("messages") or messages_of(item),
                "reasoning_effort": "none",
            },
        )
        status, completion = client.post(
            "/v1/completions",
            {
                "model": client.model,
                "prompt": template["prompt"] + prefix,
                "max_tokens": 1,
                "temperature": 0,
            },
        )
        token = completion["choices"][0]["text"] if status == 200 else None
        if token == score["argmax"]:
            agree_completion += 1
        else:
            errors.append(
                {
                    "index": index,
                    "kind": "completion",
                    "argmax": score["argmax"],
                    "generated": token,
                    "labels": score["labels"],
                    "item": item,
                }
            )
        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(items)}", flush=True)
    record(
        args,
        "agree",
        {
            "items": len(items),
            "completion_agree": agree_completion,
            "chat_compared": compared_chat,
            "chat_agree": agree_chat,
            "chat_key_missing": len(prefix_mismatch),
            "natural_prefixes": natural_prefixes,
            "prefix_mismatch_examples": prefix_mismatch[:5],
            "disagreements": errors,
        },
    )


def check_determinism(client, args):
    item = items_of(args)[0]
    runs = []
    for _ in range(20):
        status, score = client.post("/v1/score", score_body(client, item))
        assert status == 200, score
        runs.append(
            [entry["prob"] for entry in score["labels"]]
            + [entry["logprob"] for entry in score["labels"]]
        )
    spread = max(max(column) - min(column) for column in zip(*runs))
    record(
        args, "determinism", {"runs": 20, "max_spread": spread, "pass": spread <= 1e-6}
    )


def check_concurrency(client, args):
    items = items_of(args)[:24]
    baseline = []
    for item in items:
        status, score = client.post("/v1/score", score_body(client, item))
        baseline.append(score)
    results = {}
    for width in (1, 4, 24):
        with concurrent.futures.ThreadPoolExecutor(width) as pool:
            scores = list(
                pool.map(
                    lambda item: client.post("/v1/score", score_body(client, item))[1],
                    items,
                )
            )
        argmax_same = all(a["argmax"] == b["argmax"] for a, b in zip(scores, baseline))
        spread = max(
            abs(x["prob"] - y["prob"])
            for a, b in zip(scores, baseline)
            for x, y in zip(a["labels"], b["labels"])
        )
        results[str(width)] = {
            "argmax_same": argmax_same,
            "max_prob_diff": spread,
            "pass": argmax_same and spread <= 1e-3,
        }
    record(args, "concurrency", results)


def check_cache(client, args):
    out = []
    for item in items_of(args)[:8]:
        nonce = f"[{uuid.uuid4()}] "
        cold_item = {**item, "system": nonce + item.get("system", "")}
        status, cold = client.post("/v1/score", score_body(client, cold_item))
        status, warm = client.post("/v1/score", score_body(client, cold_item))
        diff = max(
            abs(a["logprob"] - b["logprob"])
            for a, b in zip(cold["labels"], warm["labels"])
        )
        out.append(
            {
                "cold_cached": cold["usage"]["cached_tokens"],
                "warm_cached": warm["usage"]["cached_tokens"],
                "prompt_tokens": warm["usage"]["prompt_tokens"],
                "max_logprob_diff": diff,
            }
        )
    worst = max(entry["max_logprob_diff"] for entry in out)
    record(
        args, "cache", {"pairs": out, "max_logprob_diff": worst, "pass": worst <= 1e-4}
    )


def check_sanity(client, args):
    problems = []
    for item in items_of(args)[:20]:
        status, score = client.post("/v1/score", score_body(client, item))
        total = sum(entry["prob"] for entry in score["labels"])
        if abs(total - 1) > 1e-9 or not 0 <= score["label_mass"] <= 1:
            problems.append({"sum": total, "mass": score["label_mass"]})
    status, error = client.post(
        "/v1/score",
        score_body(client, items_of(args)[0], labels=[" true false", " no"]),
    )
    record(
        args,
        "sanity",
        {
            "problems": problems,
            "multi_token_status": status,
            "multi_token_error": error.get("error"),
            "pass": not problems and status == 400,
        },
    )


def latencies(values):
    values = sorted(values)
    return {
        "p50_ms": statistics.median(values) * 1000,
        "p95_ms": values[max(0, math.ceil(len(values) * 0.95) - 1)] * 1000,
    }


def check_perf(client, args):
    items = items_of(args)
    status_before = client.get("/status")
    report = {}
    for mode in ("score", "generate"):
        for width in (1, 4, 12, 24):

            def one(item):
                started = time.monotonic()
                if mode == "score":
                    client.post("/v1/score", score_body(client, item))
                else:
                    client.post(
                        "/v1/chat/completions",
                        {
                            "model": client.model,
                            "messages": item.get("messages") or messages_of(item),
                            "temperature": 0,
                            "max_tokens": args.max_tokens,
                            "reasoning_effort": "none",
                        },
                    )
                return time.monotonic() - started

            started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(width) as pool:
                times = list(pool.map(one, items))
            wall = time.monotonic() - started
            report[f"{mode}_{width}"] = {
                **latencies(times),
                "requests_per_min": len(items) / wall * 60,
            }
            print(f"  {mode} x{width}: {report[f'{mode}_{width}']}", flush=True)
    status_after = client.get("/status")

    def timing(status):
        return status.get("model_timing", {})

    report["gpu_ms"] = {
        phase: timing(status_after)[phase]["total_gpu_ms"]
        - timing(status_before)[phase]["total_gpu_ms"]
        for phase in ("prefill", "decode")
    }
    report["score_counters"] = status_after.get("score")
    record(args, "perf", report)


def check_memory(client, args):
    items = items_of(args)
    status = client.get("/status")
    pid = status["instance"]["pid"]

    def rss_of(pid):
        output = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True
        ).stdout.strip()
        return int(output) * 1024 if output else None

    children = subprocess.run(
        ["pgrep", "-P", str(pid)], capture_output=True, text=True
    ).stdout.split()
    native_pid = int(children[0]) if children else None
    samples = []
    for index in range(args.count):
        item = dict(items[index % len(items)])
        item["user"] = f"#{index} " + item["user"]
        client.post("/v1/score", score_body(client, item))
        if index % max(1, args.count // 20) == 0 or index == args.count - 1:
            snapshot = client.get("/status")
            samples.append(
                {
                    "requests": index + 1,
                    "server_rss": rss_of(pid),
                    "engine_rss": rss_of(native_pid) if native_pid else None,
                    "kv_pages_active": snapshot.get("kv", {}).get("pages_active"),
                    "kv_pages_cache": snapshot.get("kv", {}).get("pages_cache"),
                    "memory_current": snapshot.get("memory_actual", {}).get(
                        "current_bytes"
                    ),
                }
            )
            print(f"  {samples[-1]}", flush=True)
    record(args, "memory", {"samples": samples})


def check_image(client, args):
    import base64

    with open(args.image, "rb") as file:
        data = base64.b64encode(file.read()).decode()
    kind = "png" if args.image.endswith(".png") else "jpeg"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{kind};base64,{data}"},
                },
                {"type": "text", "text": args.question},
            ],
        }
    ]
    status, score = client.post(
        "/v1/score",
        {
            "model": client.model,
            "messages": messages,
            "labels": args.labels,
            "reasoning_effort": "none",
        },
    )
    status_chat, chat = client.post(
        "/v1/chat/completions",
        {
            "model": client.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 8,
            "reasoning_effort": "none",
        },
    )
    generated = (
        chat["choices"][0]["message"].get("content") if status_chat == 200 else None
    )
    record(
        args,
        "image",
        {
            "status": status,
            "score": score,
            "generated": generated,
            "agree": generated is not None
            and generated.strip().startswith(score.get("argmax", "?").strip()),
        },
    )


def check_regress(client, args):
    """Chat outputs at T=0 on a fixed prompt set, saved or compared bit by bit."""
    items = synthetic_items(12)
    prompts = [messages_of(item) for item in items] + [
        [{"role": "user", "content": "Napiš krátkou báseň o podzimu v Praze."}],
        [
            {
                "role": "user",
                "content": "Explain how a hash map works in three sentences.",
            }
        ],
        [{"role": "user", "content": "List five prime numbers above 100."}],
    ]
    submitted_before = client.get("/status")["requests"]["submitted"]
    outputs = []
    for messages in prompts:
        status, chat = client.post(
            "/v1/chat/completions",
            {
                "model": client.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 120,
                "reasoning_effort": "none",
            },
        )
        outputs.append(
            chat["choices"][0]["message"].get("content")
            if status == 200
            else f"HTTP {status}"
        )
    if args.save:
        with open(args.save, "w") as file:
            json.dump(outputs, file, ensure_ascii=False, indent=1)
        record(args, "regress_save", {"prompts": len(outputs), "file": args.save})
    else:
        with open(args.check_file) as file:
            before = json.load(file)
        same = [a == b for a, b in zip(before, outputs)]
        # Requests other clients sent during the run: batching with them can
        # change greedy output (Splash repeats it only when a request runs alone).
        foreign = (
            client.get("/status")["requests"]["submitted"]
            - submitted_before
            - len(prompts)
        )
        record(
            args,
            "regress_check",
            {
                "prompts": len(outputs),
                "identical": sum(same),
                "differing": [i for i, s in enumerate(same) if not s],
                "foreign_requests_during_run": foreign,
                "pass": all(same),
            },
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "check",
        choices=[
            "agree",
            "determinism",
            "concurrency",
            "cache",
            "sanity",
            "perf",
            "memory",
            "image",
            "regress",
        ],
    )
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=os.environ.get("SPLASH_API_KEY"))
    parser.add_argument("--jsonl")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--image")
    parser.add_argument(
        "--question", default="Is this a cordless drill? Answer yes or no."
    )
    parser.add_argument("--labels", nargs="+", default=["yes", "no"])
    parser.add_argument("--save")
    parser.add_argument("--check-file")
    parser.add_argument("--out")
    args = parser.parse_args()
    client = Client(args.url, args.model, args.api_key)
    {
        "agree": check_agree,
        "determinism": check_determinism,
        "concurrency": check_concurrency,
        "cache": check_cache,
        "sanity": check_sanity,
        "perf": check_perf,
        "memory": check_memory,
        "image": check_image,
        "regress": check_regress,
    }[args.check](client, args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""End-to-end throughput probe against a running splash HTTP server (non-streaming; uses the server's own request metrics).
Per repeat: (1) single-stream decode, short prompt, long answer; (2) prefill, ~2k-token prompt, short answer; (3) four concurrent
short prompts. One JSON line per measurement; raw server metrics are kept in each line."""

import argparse
import json
import threading
import time
import urllib.request


def complete(url, model, prompt, max_tokens):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    wall = time.perf_counter() - t0
    m = d.get("metrics", {})
    lat = m.get("request_latency", {})
    usage = d.get("usage", {})
    return dict(
        wall_s=wall,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        ttft_s=(lat.get("start_to_first_token_ms") or 0) / 1e3 or None,
        decode_s=(lat.get("first_token_to_done_ms") or 0) / 1e3 or None,
        server_wall_s=(lat.get("wall_ms") or 0) / 1e3 or None,
        metrics=m,
        text=(
            d["choices"][0]["message"].get("content")
            or d["choices"][0]["message"].get("reasoning_content")
            or ""
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8011")
    ap.add_argument("--model", default="incoai/Qwen3.8-27B-Splash")
    ap.add_argument("--tag", default="")
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()

    def emit(kind, **kw):
        print(json.dumps(dict(tag=a.tag, kind=kind, **kw)), flush=True)

    complete(a.url, a.model, "Say hello in one word.", 8)  # warmup
    essay = "Write a long, detailed essay about the history of the Roman Empire, covering its founding, expansion, culture, and fall. Do not stop early."
    for i in range(a.repeats):
        r = complete(a.url, a.model, essay, a.decode_tokens)
        n = (r["completion_tokens"] or 0) - 1
        emit(
            "decode_single",
            repeat=i,
            completion_tokens=r["completion_tokens"],
            decode_tok_s=(n / r["decode_s"]) if (r["decode_s"] and n > 0) else None,
            ttft_s=r["ttft_s"],
            wall_s=r["wall_s"],
            metrics=r["metrics"],
        )
    filler = "The archive contains routine project notes and implementation details about the storage layer, the scheduler and the deployment scripts. "
    long_prompt = filler * 95 + "\nSummarize the above in one sentence."
    for i in range(a.repeats):
        r = complete(a.url, a.model, f"(variant {i}) " + long_prompt, 16)
        emit(
            "prefill_long",
            repeat=i,
            prompt_tokens=r["prompt_tokens"],
            ttft_s=r["ttft_s"],
            prefill_tok_s=(r["prompt_tokens"] / r["ttft_s"])
            if (r["prompt_tokens"] and r["ttft_s"])
            else None,
            metrics=r["metrics"],
        )
    for i in range(a.repeats):
        results = [None] * 4

        def worker(j):
            results[j] = complete(
                a.url,
                a.model,
                f"Write a detailed story about a lighthouse keeper who discovers something unexpected (story {i}-{j}). Make it long.",
                a.decode_tokens,
            )

        threads = [threading.Thread(target=worker, args=(j,)) for j in range(4)]
        t0 = time.perf_counter()
        [t.start() for t in threads]
        [t.join() for t in threads]
        wall = time.perf_counter() - t0
        total = sum((r["completion_tokens"] or 0) for r in results)
        emit(
            "decode_concurrent4",
            repeat=i,
            completion_tokens=total,
            wall_s=wall,
            aggregate_tok_s=total / wall,
            per_stream_tok_s=[
                (((r["completion_tokens"] or 1) - 1) / r["decode_s"])
                if r["decode_s"]
                else None
                for r in results
            ],
        )
    emit(
        "sample",
        text=complete(
            a.url, a.model, "What is the capital of France? Answer in one sentence.", 40
        )["text"][:300],
    )


if __name__ == "__main__":
    main()

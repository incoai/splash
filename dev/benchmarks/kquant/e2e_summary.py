#!/usr/bin/env python3
import collections
import json
import statistics as st
import sys

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
by = collections.defaultdict(list)
for r in rows:
    tag = r["tag"]
    pkg = tag.split("#")[0]
    if r["kind"] == "decode_single" and r.get("decode_tok_s"):
        by[(pkg, "decode 1 stream, tok/s")].append(r["decode_tok_s"])
        by[(pkg, "decode 1 stream TTFT, ms")].append(r["ttft_s"] * 1e3)
    if r["kind"] == "prefill_long" and r.get("prefill_tok_s"):
        by[(pkg, "prefill ~2k prompt, tok/s")].append(r["prefill_tok_s"])
        by[(pkg, "prefill ~2k prompt TTFT, ms")].append(r["ttft_s"] * 1e3)
        by[(pkg, "prompt tokens")].append(r["prompt_tokens"])
    if r["kind"] == "decode_concurrent4":
        by[(pkg, "decode 4 concurrent aggregate, tok/s")].append(r["aggregate_tok_s"])
    if r["kind"] == "decode_single":
        by[(pkg, "single-stream completion tokens")].append(r["completion_tokens"])
metrics = [
    "decode 1 stream, tok/s",
    "decode 1 stream TTFT, ms",
    "decode 4 concurrent aggregate, tok/s",
    "prefill ~2k prompt, tok/s",
    "prefill ~2k prompt TTFT, ms",
    "prompt tokens",
    "single-stream completion tokens",
]
print(
    "| metric | splash package (mean, min..max, n) | K-quant package (mean, min..max, n) | K-quant / splash |"
)
print("|---|---|---|---|")
for m in metrics:
    a = by.get(("splash", m), [])
    b = by.get(("kquant", m), [])

    def fmt(v):
        return (
            f"{st.mean(v):.1f} ({min(v):.1f}..{max(v):.1f}, n={len(v)})" if v else "-"
        )

    ratio = (st.mean(b) / st.mean(a)) if a and b else None
    print(
        f"| {m} | {fmt(a)} | {fmt(b)} | {ratio:.3f} |"
        if ratio
        else f"| {m} | {fmt(a)} | {fmt(b)} | - |"
    )
print("\nper run:")
for r in rows:
    if r["kind"] == "decode_single":
        print(
            f"  {r['tag']:<10} decode_single  {r['decode_tok_s']:.1f} tok/s  ttft {r['ttft_s'] * 1e3:.0f} ms  tokens {r['completion_tokens']}"
        )
    if r["kind"] == "prefill_long":
        print(
            f"  {r['tag']:<10} prefill_long   {r['prefill_tok_s']:.0f} tok/s  ttft {r['ttft_s'] * 1e3:.0f} ms  prompt {r['prompt_tokens']}"
        )
    if r["kind"] == "decode_concurrent4":
        print(
            f"  {r['tag']:<10} concurrent4    {r['aggregate_tok_s']:.1f} tok/s aggregate  wall {r['wall_s']:.1f} s"
        )
    if r["kind"] == "sample":
        print(f"  {r['tag']:<10} sample: {r['text'][:120]!r}")

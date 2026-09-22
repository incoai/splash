#!/usr/bin/env python3
"""Tables from bench.jsonl (per-engine, per-domain TTFT / decode rate), tasks.jsonl (accuracy) and gemm_results.jsonl."""
import json, statistics as st, sys, collections
def load(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except FileNotFoundError: return []
base = sys.argv[1] if len(sys.argv) > 1 else "."
bench, tasks, gemm = load(f"{base}/bench.jsonl"), load(f"{base}/tasks.jsonl"), load(f"{base}/gemm_results.jsonl")
if bench:
    print("## Single-stream generation (client-side streaming timing, greedy, thinking off, max 512 tokens)\n")
    print("| engine | domain | prompts | prompt tok (mean) | output tok (mean) | TTFT s (median) | decode tok/s (median) | decode tok/s (min..max) |")
    print("|---|---|---|---|---|---|---|---|")
    by = collections.defaultdict(list)
    for r in bench: by[(r["engine"], r["domain"])].append(r)
    for (eng, dom), rs in sorted(by.items()):
        rates = [r["decode_tok_s"] for r in rs if r.get("decode_tok_s")]
        ttft = [r["ttft_s"] for r in rs if r.get("ttft_s")]
        print(f"| {eng} | {dom} | {len(rs)} | {st.mean(r['prompt_tokens'] or 0 for r in rs):.0f} | {st.mean(r['completion_tokens'] or 0 for r in rs):.0f} | {st.median(ttft):.3f} | {st.median(rates):.1f} | {min(rates):.1f}..{max(rates):.1f} |")
    print("\n| engine | all prompts | decode tok/s (median) | mean | TTFT s (median) | total output tokens | total wall s |")
    print("|---|---|---|---|---|---|---|")
    for eng in sorted({r["engine"] for r in bench}):
        rs = [r for r in bench if r["engine"] == eng]; rates = [r["decode_tok_s"] for r in rs if r.get("decode_tok_s")]
        print(f"| {eng} | {len(rs)} | {st.median(rates):.1f} | {st.mean(rates):.1f} | {st.median([r['ttft_s'] for r in rs if r.get('ttft_s')]):.3f} | {sum(r['completion_tokens'] or 0 for r in rs)} | {sum(r['wall_s'] for r in rs):.0f} |")
    # same-prompt output agreement between engines (greedy)
    engines = sorted({r["engine"] for r in bench})
    if len(engines) > 1:
        print("\n### Greedy output agreement on identical prompts (exact text match / same first 64 chars)\n")
        texts = {(r["engine"], r["domain"], r["prompt"]): r["text"] for r in bench}
        for i in range(len(engines)):
            for j in range(i + 1, len(engines)):
                keys = [k for k in texts if k[0] == engines[i] and (engines[j], k[1], k[2]) in texts]
                same = sum(texts[k] == texts[(engines[j], k[1], k[2])] for k in keys)
                pre = sum(texts[k][:64] == texts[(engines[j], k[1], k[2])][:64] for k in keys)
                print(f"- {engines[i]} vs {engines[j]}: {same}/{len(keys)} identical, {pre}/{len(keys)} share the first 64 characters")
if tasks:
    print("\n## Task battery (greedy, thinking off)\n")
    print("| engine | GSM8K (n) | HumanEval pass@1 (n) | mean output tokens gsm8k / humaneval |")
    print("|---|---|---|---|")
    for eng in sorted({r["engine"] for r in tasks}):
        rs = [r for r in tasks if r["engine"] == eng]
        g = [r for r in rs if r["task"] == "gsm8k"]; h = [r for r in rs if r["task"] == "humaneval"]
        gt = st.mean(r["tokens"] or 0 for r in g) if g else 0; ht = st.mean(r["tokens"] or 0 for r in h) if h else 0
        print(f"| {eng} | {sum(r['correct'] for r in g)}/{len(g)} ({sum(r['correct'] for r in g)/max(len(g),1)*100:.1f}%) | {sum(r['correct'] for r in h)}/{len(h)} ({sum(r['correct'] for r in h)/max(len(h),1)*100:.1f}%) | {gt:.0f} / {ht:.0f} |")
    engines = sorted({r["engine"] for r in tasks})
    if len(engines) > 1:
        res = {(r["engine"], r["task"], str(r["id"])): r["correct"] for r in tasks}
        print("\nPer-item agreement of correct/incorrect between engines:")
        for i in range(len(engines)):
            for j in range(i + 1, len(engines)):
                keys = [k for k in res if k[0] == engines[i] and (engines[j], k[1], k[2]) in res]
                agree = sum(res[k] == res[(engines[j], k[1], k[2])] for k in keys)
                print(f"- {engines[i]} vs {engines[j]}: {agree}/{len(keys)} items agree")
if gemm:
    print("\n## Same-tensor GEMM comparison (1024 rows of a real UD-Q4_K_M tensor, identical bf16 activations)\n")
    print("| format | K | M | splash: mean rel err vs fp64 | splash: within 1 bf16 ulp | ggml-metal: mean rel err | ggml-cpu: mean rel err | splash vs bf16(ggml-metal): identical / ≤1 ulp / max ulp |")
    print("|---|---|---|---|---|---|---|---|")
    for r in gemm:
        b = r["splash_vs_bf16(ggml_metal)"]
        print(f"| {r['fmt']} | {r['K']} | {r['M']} | {r['splash']['mean_abs_err/mean_abs_ref']:.2e} | {r['splash']['within_1_bf16_ulp']*100:.1f}% | {r['ggml-metal']['mean_abs_err/mean_abs_ref']:.1e} | {r['ggml-cpu']['mean_abs_err/mean_abs_ref']:.1e} | {b['identical']*100:.1f}% / {b['within_1_ulp']*100:.1f}% / {b['max_ulp']} |")

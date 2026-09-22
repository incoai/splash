#!/usr/bin/env python3
"""End-to-end parity between splash (GGUF-direct) and llama.cpp on identical token ids.

  reference  : query llama-server for the top-K logprobs at every prefix position of the text chunks
               and along its own greedy generations for chat prompts; writes jobs.tsv for splash's
               score-prefix tool and reference.json.
  compare    : read score-prefix logits and compute per-source metrics.
"""
import argparse, json, math, sys, time, urllib.request

def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def topk_of(entry):
    """(ids, logprobs) of a llama-server completion_probabilities entry (handles both response shapes)."""
    cands = entry.get("top_logprobs") or entry.get("probs") or []
    ids, lps = [], []
    for c in cands:
        ids.append(int(c["id"]))
        lps.append(float(c["logprob"]) if "logprob" in c else math.log(max(float(c["prob"]), 1e-30)))
    return ids, lps

def reference(args):
    chunks = json.load(open(args.chunks))
    prompts = json.load(open(args.prompts)) if args.prompts else {}
    jobs, ref, skipped = [], [], []
    base = args.llama.rstrip("/")
    def checkpoint():
        with open(args.jobs, "w") as f:
            for prefix, cand in jobs:
                f.write(",".join(map(str, prefix)) + "\t" + ",".join(map(str, cand)) + "\n")
        json.dump(ref, open(args.reference, "w"))
        json.dump(skipped, open(args.reference + ".skipped.json", "w"))
    t0 = time.time()
    # (1) teacher forcing on natural text: every prefix position from `start`
    for ci, chunk in enumerate(chunks):
        ids = chunk["ids"]
        for p in range(args.start, len(ids)):
            prefix = ids[:p]
            d = post(base + "/completion", {"prompt": prefix, "n_predict": 1, "n_probs": args.topk, "cache_prompt": True,
                                            "temperature": 0.0, "samplers": [], "top_k": 0, "min_keep": 1})
            if not d.get("completion_probabilities"):
                skipped.append({"chunk": ci, "pos": p, "response": {k: v for k, v in d.items() if k not in ("prompt", "generation_settings")}})
                print(f"  skipped chunk {ci} pos {p}: {json.dumps(skipped[-1]['response'])[:300]}", file=sys.stderr, flush=True)
                continue
            cp = d["completion_probabilities"][0]
            cand, lps = topk_of(cp)
            greedy = int(cp["id"])  # the sampled (greedy) token at this position
            truth = ids[p]
            for extra in (greedy, truth):
                if extra not in cand:
                    cand.append(extra); lps.append(None)
            jobs.append((prefix, cand))
            ref.append({"source": chunk["source"], "chunk": ci, "pos": p, "cand": cand, "llama_logprob": lps, "llama_greedy": greedy, "truth": truth, "kind": "text"})
        print(f"chunk {ci} ({chunk['source']}) done, {len(jobs)} jobs, {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
        checkpoint()
    # (2) llama.cpp's own greedy generations on chat prompts (rendered by llama-server's template, thinking off)
    for domain, texts in prompts.items():
        for pi, text in enumerate(texts[: args.gen_prompts]):
            d = post(base + "/apply-template", {"messages": [{"role": "user", "content": text}], "chat_template_kwargs": {"enable_thinking": False}})
            rendered = d["prompt"]
            tok = post(base + "/tokenize", {"content": rendered, "add_special": False})["tokens"]
            g = post(base + "/completion", {"prompt": tok, "n_predict": args.gen_tokens, "n_probs": args.topk, "cache_prompt": True,
                                            "temperature": 0.0, "samplers": [], "top_k": 0, "min_keep": 1})
            if not g.get("completion_probabilities"):
                print(f"  generation {domain}/{pi} returned no probabilities: {json.dumps({k: v for k, v in g.items() if k not in ('prompt',)})[:300]}", file=sys.stderr, flush=True)
                continue
            gen = [int(cp["id"]) for cp in g["completion_probabilities"]]
            for t, cp in enumerate(g["completion_probabilities"]):
                cand, lps = topk_of(cp)
                if gen[t] not in cand:
                    cand.append(gen[t]); lps.append(None)
                jobs.append((tok + gen[:t], cand))
                ref.append({"source": f"gen-{domain}", "chunk": pi, "pos": len(tok) + t, "cand": cand, "llama_logprob": lps, "llama_greedy": gen[t], "truth": gen[t], "kind": "generation"})
            print(f"generation {domain}/{pi}: {len(gen)} tokens, stop={g.get('stop_type')}", file=sys.stderr, flush=True)
            checkpoint()
    checkpoint()
    print(f"wrote {len(jobs)} jobs ({len(skipped)} positions skipped) -> {args.jobs}, {args.reference} in {time.time()-t0:.0f}s", file=sys.stderr)

def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e); return [v / s for v in e]

def compare(args):
    ref = json.load(open(args.reference))
    lines = [l.strip() for l in open(args.logits) if l.strip()]
    assert len(lines) == len(ref), f"{len(lines)} logit lines for {len(ref)} jobs"
    per = {}
    rows = []
    for r, line in zip(ref, lines):
        splash = [float(v) for v in line.split(",")]
        cand = r["cand"]; lp = r["llama_logprob"]
        # llama.cpp distribution restricted to the top-K set (entries without a logprob are outside its top-K)
        known = [i for i, v in enumerate(lp) if v is not None]
        mass = sum(math.exp(lp[i]) for i in known)          # llama.cpp probability mass covered by the top-K set
        pl = softmax([lp[i] for i in known]); ps = softmax([splash[i] for i in known])
        kl_ls = sum(a * (math.log(a) - math.log(max(b, 1e-30))) for a, b in zip(pl, ps) if a > 0)
        kl_sl = sum(b * (math.log(b) - math.log(max(a, 1e-30))) for a, b in zip(pl, ps) if b > 0)
        splash_arg = cand[max(range(len(cand)), key=lambda i: splash[i])]
        top1 = splash_arg == r["llama_greedy"]
        # logit agreement: llama.cpp logprobs and splash logits differ by a per-position constant; compare centered values
        cl = [lp[i] for i in known]; cs = [splash[i] for i in known]
        ml, ms = sum(cl) / len(cl), sum(cs) / len(cs)
        mad = sum(abs((a - ml) - (b - ms)) for a, b in zip(cl, cs)) / len(cl)
        maxd = max(abs((a - ml) - (b - ms)) for a, b in zip(cl, cs))
        # margin between the two best llama.cpp tokens when they disagree
        srt = sorted(known, key=lambda i: -lp[i]); margin = (lp[srt[0]] - lp[srt[1]]) if len(srt) > 1 else float("inf")
        key = r["source"]
        d = per.setdefault(key, {"n": 0, "top1": 0, "kl_ls": 0.0, "kl_sl": 0.0, "mad": 0.0, "maxd": 0.0, "mass": 0.0, "truth_top1_llama": 0, "truth_top1_splash": 0, "disagree_margins": []})
        d["n"] += 1; d["top1"] += top1; d["kl_ls"] += kl_ls; d["kl_sl"] += kl_sl; d["mad"] += mad; d["maxd"] = max(d["maxd"], maxd); d["mass"] += mass
        d["truth_top1_llama"] += r["llama_greedy"] == r["truth"]; d["truth_top1_splash"] += splash_arg == r["truth"]
        if not top1: d["disagree_margins"].append(round(margin, 4))
        rows.append({"source": key, "pos": r["pos"], "top1": top1, "kl": kl_ls, "mad": mad})
    print("| source | positions | top-1 agreement | KL(llama‖splash) mean | KL(splash‖llama) mean | mean abs centered logit diff | max diff | top-K mass | next-token acc llama / splash |")
    print("|---|---|---|---|---|---|---|---|---|")
    tot = {"n": 0, "top1": 0, "kl_ls": 0.0, "kl_sl": 0.0, "mad": 0.0, "maxd": 0.0, "mass": 0.0}
    for key, d in per.items():
        n = d["n"]
        print(f"| {key} | {n} | {d['top1']/n*100:.2f}% | {d['kl_ls']/n:.2e} | {d['kl_sl']/n:.2e} | {d['mad']/n:.4f} | {d['maxd']:.3f} | {d['mass']/n:.4f} | {d['truth_top1_llama']/n*100:.1f}% / {d['truth_top1_splash']/n*100:.1f}% |")
        for k in tot: tot[k] = max(tot[k], d[k]) if k == "maxd" else tot[k] + d[k]
    n = tot["n"]
    print(f"| **all** | {n} | {tot['top1']/n*100:.2f}% | {tot['kl_ls']/n:.2e} | {tot['kl_sl']/n:.2e} | {tot['mad']/n:.4f} | {tot['maxd']:.3f} | {tot['mass']/n:.4f} | |")
    dis = [(k, m) for k, d in per.items() for m in d["disagree_margins"]]
    if dis:
        print(f"\ntop-1 disagreements: {len(dis)}; llama.cpp margin between its best two tokens at those positions (nats): " + ", ".join(f"{m:.3f}" for _, m in sorted(dis, key=lambda x: x[1])[:20]) + (" ..." if len(dis) > 20 else ""))
    json.dump({"per_source": {k: {kk: vv for kk, vv in d.items() if kk != 'disagree_margins'} for k, d in per.items()}, "rows": rows}, open(args.out, "w"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("reference"); a.add_argument("--llama", default="http://127.0.0.1:8080"); a.add_argument("--chunks", required=True); a.add_argument("--prompts"); a.add_argument("--start", type=int, default=32); a.add_argument("--topk", type=int, default=64); a.add_argument("--gen-prompts", type=int, default=2); a.add_argument("--gen-tokens", type=int, default=160); a.add_argument("--jobs", default="jobs.tsv"); a.add_argument("--reference", default="reference.json"); a.set_defaults(fn=reference)
    b = sub.add_parser("compare"); b.add_argument("--reference", default="reference.json"); b.add_argument("--logits", required=True); b.add_argument("--out", default="parity_metrics.json"); b.set_defaults(fn=compare)
    args = ap.parse_args(); args.fn(args)

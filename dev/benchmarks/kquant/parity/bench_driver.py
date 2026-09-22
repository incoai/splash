#!/usr/bin/env python3
"""Streaming benchmark + task battery against an OpenAI-compatible chat endpoint (splash or llama-server).

  bench : per prompt, one streaming greedy chat completion; client-side TTFT and decode rate.
  tasks : GSM8K (numeric answer) and HumanEval (executes the generated function against the tests).
Both disable model thinking: splash via reasoning_effort=none, llama-server via chat_template_kwargs.enable_thinking=false.
"""
import argparse, concurrent.futures, json, math, os, re, subprocess, sys, tempfile, time, urllib.request

TOKENIZER = None
def count_tokens(text):
    """Token count from the model tokenizer when the server sends no usage (splash streams several tokens per event)."""
    if TOKENIZER is None: return None
    return len(TOKENIZER.encode(text, add_special_tokens=False).ids)

def stream_chat(url, model, prompt, max_tokens, engine, timeout=1800):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}}
    if engine == "splash": body["reasoning_effort"] = "none"
    else: body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; last = t0; text = []; usage = None; chunks = 0; timings = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"): continue
            payload = line[5:].strip()
            if payload == "[DONE]": break
            d = json.loads(payload)
            if d.get("usage"): usage = d["usage"]
            if d.get("timings"): timings = d["timings"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    now = time.perf_counter()
                    if first is None: first = now
                    last = now; text.append(piece); chunks += 1
    end = time.perf_counter()
    out = "".join(text)
    completion = (usage or {}).get("completion_tokens") or count_tokens(out)
    return {"text": out, "prompt_tokens": (usage or {}).get("prompt_tokens"), "completion_tokens": completion, "chunks": chunks,
            "ttft_s": (first - t0) if first else None, "decode_s": (last - first) if first else None, "wall_s": end - t0, "timings": timings}

def bench(args):
    prompts = json.load(open(args.prompts))
    out = open(args.out, "a")
    for domain, texts in prompts.items():
        if args.domains and domain not in args.domains: continue
        for i, text in enumerate(texts):
            r = stream_chat(args.url, args.model, text, args.max_tokens, args.engine)
            n = r["completion_tokens"] or count_tokens(r["text"]) or r["chunks"]
            rate = (n - 1) / r["decode_s"] if r["decode_s"] and n and n > 1 else None
            rec = {"engine": args.label, "domain": domain, "prompt": i, "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"], "chunks": r["chunks"],
                   "ttft_s": r["ttft_s"], "decode_tok_s": rate, "wall_s": r["wall_s"], "timings": r["timings"], "text_sha": hash(r["text"]) & 0xffffffff, "text": r["text"][:4000]}
            out.write(json.dumps(rec) + "\n"); out.flush()
            print(f"{args.label} {domain}/{i}: prompt {r['prompt_tokens']} tok, {r['completion_tokens']} tok, TTFT {r['ttft_s']:.3f}s, {rate and round(rate,1)} tok/s", flush=True)

def gsm_answer(text):
    m = re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", text)
    if not m: m = re.findall(r"(-?[\d,]*\.?\d+)", text)
    if not m: return None
    return m[-1].replace(",", "")

def num_eq(a, b):
    try: return abs(float(a) - float(b)) < 1e-6
    except Exception: return False

def run_humaneval(sample, completion):
    code = completion
    m = re.search(r"```(?:python)?\n(.*?)```", completion, re.S)
    if m: code = m.group(1)
    if "def " + sample["entry_point"] not in code: code = sample["prompt"] + code   # model returned only a body
    program = code + "\n\n" + sample["test"] + f"\n\ncheck({sample['entry_point']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f: f.write(program); path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, timeout=20)
        return p.returncode == 0
    except subprocess.TimeoutExpired: return False
    finally: os.unlink(path)

def tasks(args):
    data = json.load(open(args.tasks))
    out = open(args.out, "a")
    def one(kind, item):
        if kind == "gsm8k":
            prompt = f"{item['question']}\nSolve step by step, then give the final numeric answer on the last line as 'Answer: <number>'."
            r = stream_chat(args.url, args.model, prompt, args.max_tokens, args.engine)
            pred = gsm_answer(r["text"]); ok = pred is not None and num_eq(pred, item["answer"])
            return {"engine": args.label, "task": "gsm8k", "id": item["id"], "correct": ok, "pred": pred, "gold": item["answer"], "tokens": r["completion_tokens"], "wall_s": r["wall_s"], "text": r["text"][:3000]}
        prompt = f"Complete the following Python function. Return the complete function in one ```python code block.\n\n{item['prompt']}"
        r = stream_chat(args.url, args.model, prompt, args.max_tokens, args.engine)
        ok = run_humaneval(item, r["text"])
        return {"engine": args.label, "task": "humaneval", "id": item["id"], "correct": ok, "tokens": r["completion_tokens"], "wall_s": r["wall_s"], "text": r["text"][:3000]}
    for kind in ("gsm8k", "humaneval"):
        if args.only and kind != args.only: continue
        items = data[kind][: args.limit] if args.limit else data[kind]
        t0 = time.time(); correct = 0
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
            for rec in pool.map(lambda it: one(kind, it), items):
                out.write(json.dumps(rec) + "\n"); out.flush(); correct += bool(rec["correct"])
        print(f"{args.label} {kind}: {correct}/{len(items)} correct in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("bench", bench), ("tasks", tasks)):
        p = sub.add_parser(name); p.add_argument("--url", required=True); p.add_argument("--model", required=True); p.add_argument("--engine", choices=["splash", "llama"], required=True)
        p.add_argument("--label", required=True); p.add_argument("--out", required=True); p.add_argument("--max-tokens", type=int, default=768); p.add_argument("--tokenizer"); p.set_defaults(fn=fn)
        if name == "bench": p.add_argument("--prompts", required=True); p.add_argument("--domains", nargs="*")
        else: p.add_argument("--tasks", required=True); p.add_argument("--limit", type=int); p.add_argument("--concurrency", type=int, default=1); p.add_argument("--only")
    args = ap.parse_args()
    if args.tokenizer:
        from tokenizers import Tokenizer
        TOKENIZER = Tokenizer.from_file(args.tokenizer)
    args.fn(args)

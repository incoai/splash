#!/bin/bash
# ggml-metal Q4_K precision across batch sizes on two Q4_K tensors (probe of the 5e-3 error seen in the sweep)
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"
P=/Users/liang2kl/dev/q4k-m5/parity; G=/Users/liang2kl/dev/q4k-m5/models/Qwen3.8-27B-UD-Q4_K_M.gguf; PY=/Users/liang2kl/dev/splash/.venv/bin/python
cd $P; : > q4k_probe.jsonl
for t in blk.1.attn_gate.weight blk.3.ffn_gate.weight blk.4.ffn_down.weight; do
  for M in 1 2 4 8 16 32 64 128 512; do
    d=gemm/probe-$t-m$M; mkdir -p "$d"
    ./ggml_gemm_parity $G $t 1024 $M 3 "$d" > "$d/ggml.log" 2>&1 || { echo "ggml failed $t $M"; continue; }
    fmt=$(cut -d' ' -f1 "$d/meta.txt")
    if [ $M -le 32 ] || [ $M -ge 128 ]; then /Users/liang2kl/dev/q4k-m5/harness_prod /Users/liang2kl/dev/splash/build/splash.metallib file "$d" > "$d/splash.log" 2>&1 || echo "splash failed $t $M"; fi
    if [ -f "$d/Y_splash.f32" ]; then $PY gemm_compare.py "$d" | $PY -c "import json,sys; r=json.load(sys.stdin); print(json.dumps({'tensor': '$t', 'fmt': r['fmt'], 'M': r['M'], 'ggml_metal': r['ggml-metal'], 'splash': r['splash'], 'ggml_cpu_mean': r['ggml-cpu']['mean_abs_err/mean_abs_ref']}))" | tee -a q4k_probe.jsonl | cut -c1-260
    else grep "ggml-metal" "$d/ggml.log" | sed "s/^/$t M=$M: /"; fi
  done
done
echo "== Q4K PROBE DONE"

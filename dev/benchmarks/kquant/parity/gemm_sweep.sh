#!/bin/bash
# same-tensor GEMM parity for every format on real UD-Q4_K_M tensors, decode (M=8) and prefill (M=512) shapes
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"
P=/Users/liang2kl/dev/q4k-m5/parity; G=/Users/liang2kl/dev/q4k-m5/models/Qwen3.8-27B-UD-Q4_K_M.gguf
PY=/Users/liang2kl/dev/splash/.venv/bin/python; export SPLASH_GGML_ORACLE=/Users/liang2kl/dev/llama.cpp/build-metal/bin/libggml-base.dylib
cd $P; : > gemm_results.jsonl
while read -r fmt tensor; do
  for M in 8 512; do
    d=gemm/$fmt-m$M; mkdir -p $d
    ./ggml_gemm_parity $G $tensor 1024 $M 1 $d > $d/ggml.log 2>&1 || { echo "ggml failed $fmt $M"; tail -2 $d/ggml.log; continue; }
    /Users/liang2kl/dev/q4k-m5/harness_prod /Users/liang2kl/dev/splash/build/splash.metallib file $d > $d/splash.log 2>&1 || { echo "splash failed $fmt $M"; tail -2 $d/splash.log; continue; }
    $PY gemm_compare.py $d | tee -a gemm_results.jsonl | cut -c1-200
    grep "timing" $d/ggml.log
  done
done <<'LIST'
q4k blk.1.attn_gate.weight
iq4xs blk.0.ffn_gate.weight
iq4nl blk.1.ffn_down.weight
q5k blk.0.attn_qkv.weight
q6k blk.1.ssm_out.weight
q3k blk.0.ffn_up.weight
q80 blk.11.attn_v.weight
iq3s blk.11.ffn_gate.weight
LIST
echo "== GEMM SWEEP DONE"

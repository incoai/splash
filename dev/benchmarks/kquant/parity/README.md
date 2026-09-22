# Parity and performance campaign: splash GGUF target vs llama.cpp vs splash uniform Q4

Scripts used on the M5 Pro (paths inside point at that machine's `~/dev/q4k-m5/parity` rig; adjust).

Kernel level (same tensor, same bf16 activations, fp64 references):
- `ggml_gemm_parity.cpp`: llama.cpp's ggml Metal and CPU `mul_mat` on N rows of a real GGUF tensor; writes
  `W.native`, `X.bf16`, the ggml outputs and an fp64 reference from ggml's own dequantization. Build against a
  llama.cpp build: `clang++ -std=c++17 -O2 -I<llama.cpp>/ggml/include ggml_gemm_parity.cpp -L<build>/bin -lggml
  -lggml-base -lggml-metal -lggml-cpu -Wl,-rpath,<build>/bin -o ggml_gemm_parity`.
- `harness_prod.mm file <dir>` (in the parent directory) runs splash's production kernel on the same inputs.
- `gemm_compare.py <dir>`: error statistics of both engines against fp64 and against each other (bf16 ulps).
- `gemm_sweep.sh`, `q4k_probe.sh`: the eight-format sweep and the Q4_K batch-size probe.

End to end (identical token ids in both engines):
- `parity_driver.py reference`: asks llama-server for the top-K logprobs at every prefix position of the text chunks
  and along its own greedy generations; writes `jobs.tsv` for splash's `score-prefix` tool
  (`make benchmark-score-prefix`, `dev/benchmarks/score_prefix.mm`) and `reference.json`.
- `parity_driver.py compare`: top-1 agreement, restricted-support KL both ways, centered logit differences.

Performance and task battery (OpenAI-compatible chat endpoints, greedy, thinking off, client-side streaming timing):
- `bench_driver.py bench|tasks`, `phases.sh`, `chain.sh` (one engine on the GPU at a time, 60 s gap before every
  timed run), `summarize.py` (tables).

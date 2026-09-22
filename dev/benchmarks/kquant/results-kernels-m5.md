# K-quant GEMM kernels vs splash native Q4 on M5 Pro (16-core GPU, 48 GB), 2026-09-21

Kernels: `kq.metal` (staged fp16 dequant -> `matmul2d`, per-simdgroup private staging for decode, shared 128-row tiles for prefill,
split-K for narrow N, bf16 activations). Harness: `harness_kq.mm quick` (random blocks in native GGUF layouts, llama.cpp-faithful
reference; real-tensor check via `real` mode = bit-exact repack vs gguf-py for all 8 formats). Splash reference: its own Apple10
dispatch policy (`decode_linear_q4_*`, `prefill_linear_q4_n128_sg4`). Ratio = splash time / best K-quant kernel time, same shape.
Q5_K/Q6_K/Q8_0 stream 5.5/6.56/8.5 bits per weight vs splash's 4.5, so their time ratio is below 1 by construction; per byte they
are at or above splash. IQ4_XS (4.25) and Q3_K/IQ3_S (3.44) stream fewer bytes.

| phase | rows | shape | q4k | iq4xs | iq4nl | q5k | q6k | q3k | q80 | iq3s |
|---|---|---|---|---|---|---|---|---|---|---|
| decode | 8 | gate/up 17408x5120 | 0.93 | 0.88 | 0.81 | 0.84 |  | 0.93 |  | 0.90 |
| decode | 16 | gate/up 17408x5120 | 1.13 | 1.11 | 1.00 | 1.02 |  | 1.15 |  | 1.13 |
| decode | 24 | gate/up 17408x5120 | 1.05 | 1.12 | 1.04 | 0.91 |  | 1.12 |  | 1.10 |
| decode | 32 | gate/up 17408x5120 | 1.16 | 1.22 | 1.13 | 0.96 |  | 1.25 |  | 1.20 |
| decode | 8 | down 5120x17408 | 1.10 | 1.00 | 0.94 | 0.93 | 0.83 | 1.12 |  | 1.02 |
| decode | 16 | down 5120x17408 | 1.30 | 1.26 | 1.18 | 1.11 | 0.93 | 1.37 |  | 1.33 |
| decode | 24 | down 5120x17408 | 1.60 | 1.65 | 1.53 | 1.46 | 1.28 | 1.69 |  | 1.63 |
| decode | 32 | down 5120x17408 | 1.56 | 1.63 | 1.52 | 1.40 | 1.26 | 1.68 |  | 1.64 |
| decode | 8 | attn_qkv 10240x5120 | 0.96 | 0.87 | 0.83 | 0.86 |  |  |  |  |
| decode | 16 | attn_qkv 10240x5120 | 1.06 | 0.97 | 0.94 | 0.93 |  |  |  |  |
| decode | 24 | attn_qkv 10240x5120 | 1.06 | 1.00 | 0.97 | 0.94 |  |  |  |  |
| decode | 32 | attn_qkv 10240x5120 | 1.38 | 1.30 | 1.26 | 1.18 |  |  |  |  |
| decode | 8 | attn_kv 1024x5120 | 2.41 |  |  | 2.16 | 2.28 |  | 2.16 |  |
| decode | 16 | attn_kv 1024x5120 | 3.12 |  |  | 2.78 | 3.12 |  | 2.78 |  |
| decode | 24 | attn_kv 1024x5120 | 3.86 |  |  | 3.52 | 3.68 |  | 3.52 |  |
| decode | 32 | attn_kv 1024x5120 | 3.62 |  |  | 3.30 | 3.45 |  | 3.30 |  |
| decode | 8 | out 5120x6144 |  | 0.94 |  | 0.98 | 0.89 |  |  |  |
| decode | 16 | out 5120x6144 |  | 1.13 |  | 1.19 | 0.94 |  |  |  |
| decode | 24 | out 5120x6144 |  | 1.44 |  | 1.41 | 1.27 |  |  |  |
| decode | 32 | out 5120x6144 |  | 1.55 |  | 1.53 | 1.38 |  |  |  |
| decode | 8 | ab 256x5120 |  |  |  |  |  |  | 3.64 |  |
| decode | 16-32 | ab 256x5120 |  |  |  |  |  |  | 4.0-5.6 |  |
| prefill | 17 | gate/up 17408x5120 | 1.28 | 1.26 | 1.19 | 1.01 |  | 1.31 |  | 1.27 |
| prefill | 128 | gate/up 17408x5120 | 1.07 | 1.08 | 1.08 | 1.07 |  | 1.06 |  | 1.06 |
| prefill | 512 | gate/up 17408x5120 | 1.02 | 1.03 | 1.05 | 1.00 |  | 0.98 |  | 0.99 |
| prefill | 2048 | gate/up 17408x5120 | 1.03 | 1.05 | 1.05 | 0.97 |  | 0.97 |  | 0.95 |
| prefill | 17 | down 5120x17408 | 1.51 | 1.65 | 1.42 | 1.38 | 1.46 | 1.45 |  | 1.46 |
| prefill | 128 | down 5120x17408 | 1.38 | 1.42 | 1.40 | 1.40 | 1.38 | 1.34 |  | 1.39 |
| prefill | 512 | down 5120x17408 | 1.12 | 1.13 | 1.07 | 1.06 | 1.09 | 1.11 |  | 1.09 |
| prefill | 2048 | down 5120x17408 | 1.00 | 1.00 | 0.99 | 1.00 | 0.96 | 0.99 |  | 0.99 |
| prefill | 17 | attn_qkv 10240x5120 | 1.35 | 1.28 | 1.26 | 1.20 |  |  |  |  |
| prefill | 128 | attn_qkv 10240x5120 | 1.18 | 1.18 | 1.20 | 1.16 |  |  |  |  |
| prefill | 512 | attn_qkv 10240x5120 | 1.06 | 1.07 | 1.08 | 1.06 |  |  |  |  |
| prefill | 2048 | attn_qkv 10240x5120 | 1.05 | 1.08 | 1.08 | 1.06 |  |  |  |  |
| prefill | 17-2048 | attn_kv 1024x5120 | 1.09-1.44 |  |  | 1.07-1.30 | 1.09-1.36 |  | 1.09-1.34 |  |
| prefill | 17-2048 | out 5120x6144 |  | 1.02-1.55 |  | 1.01-1.37 | 1.00-1.37 |  |  |  |
| prefill | 17-2048 | ab 256x5120 |  |  |  |  |  |  | 1.40-1.56 |  |

Numerics: every format's repack + decode is bit-exact vs gguf-py `dequantize` on real UD-Q4_K_M tensors; kernel outputs vs fp64
reference have meanrel 2-4e-3 (bf16 output rounding + fp16 staging), same class as llama.cpp's Metal path.

Design notes (what mattered on M5): direct uint4b hardware path only works for K=64 slices (K32 = 0.39x, masked K64 doubles MACs) so
K-quants are staged; per-simdgroup private staging (no threadgroup barriers) beat cooperative staging; K-slice 32 beats 64 for decode;
non-persistent grids beat persistent loops; split-K (2-8) for N <= 12288; M=24 runs as a zero-padded 32-row tile; simd_shuffle LUT for
IQ4 codebooks is 2x slower than constant-memory tables; threadgroup 256-entry byte-pair LUT best for IQ4_XS; bf16 activations cost ~2%.
Package conventions (from layer-0.bin vs GGUF): norms stored as 1+w like GGUF; conv1d/decay(-exp A_log)/dt_bias/v/z rows in HF grouped
head order (GGUF is tiled); gdn_in rows = [q 2048|k 2048|v 6144|z 6144|beta 48|alpha 48|zeros 160]; out_proj K in grouped order.

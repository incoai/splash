# Batch-1 (single-lane) decode: attribution and improvements, M5 Pro, 2026-09-21

Instrument: splash's `decode-profile` (every dispatch of a B1 DFlash cycle replayed as its own command, plus the fused cycle time).

| | splash package | K-quant package, first integration | K-quant, after fusion work |
|---|---|---|---|
| B1 cycle, fused GPU time | 87.34 ms | 100.02 ms | 98.81 ms |
| sum of separate dispatches | 88.17 ms (625 dispatches) | 103.49 ms (1185) | 101.30 ms (673) |
| GEMM dispatches for the target | 4 per layer (fused input, out, gate+up, down) + 2 lm_head | ~13 per GDN layer | 4 per layer + 2 lm_head |

Where the 11.5 ms gap to splash comes from (per cycle, B1):
- bytes: 16.20 vs 15.45 GB streamed (+4.8%, 4.79 vs 4.5 bits/element) = ~4 ms at splash's 208 GB/s; of which the Q6_K lm_head alone is
  +3 ms (2 x 4.2 ms at 247 GB/s vs 2 x 2.7 ms; the cycle runs the head twice in both engines).
- dequantization ALU at M=8: the staged path spends ~2 (Q4_K) to ~3.5 (IQ4_XS) ALU ops per weight on the regular ALUs, which the
  matrix-unit-fed splash kernels do not; serialized (no-overlap) bandwidth on 17408x5120: Q4_K 205-212 GB/s (0.94-0.98 of splash),
  Q5_K 226 GB/s per byte (0.87 in time), IQ4_XS 171-181 GB/s (0.81-0.85); IQ4_XS is 33% of the file -> ~3 ms.
- fused input projection kqf (qkv|z|ab, q|k|v with per-tile format switch): 17.5 ms vs ~15 ms at splash's rate (~2 ms).
- everything else (norms, GDN, attention, draft, sampling) is identical: ~13 ms in both.

What was tried:
- one dispatch per projection (kqf multi-format segments, kqgu gate+up, kqs split-K with in-kernel last-arriver reduce):
  1185 -> 673 dispatches, 100.0 -> 99.2 ms. The first kqgu version used 17 KB of threadgroup memory and was 50% slower (occupancy);
  8 KB shared stage fixed it. Heaviest-first segment order + split tuning: 99.2 -> 98.8 ms.
- runtime format switch costs <= 6% (Q4_K) vs compile-time kernels; IQ4 codebook as an exact degree-4 polynomial (fp32) validates but is
  no faster than the threadgroup byte-pair LUT (ALU-bound either way); persistent grids of 64-256 threadgroups are slower; single vs
  double staging buffer and prefetch depth 1-3 change <4%; mmapped no-copy weights stream as fast as Metal-allocated (218-244 GB/s warm).
- DFlash acceptance is the same for both packages on a 3-prompt mix (26.1% vs 26.7% of drafted tokens), but it is text dependent:
  on the essay prompt used in the ABAB runs the K-quant model accepted fewer draft tokens per cycle (3.3 vs 4.0), which is why the
  first e2e single-stream ratio (0.73) was below the cycle-time ratio (0.87).

## End-to-end after the fusion work (ABAB, 60 s gaps, same build for both packages, e2e_results_v2.jsonl)

| metric | splash package | K-quant package | ratio |
|---|---|---|---|
| decode, 1 stream, 256 new tokens | 42.2 tok/s (41.6..42.7) | 39.4 tok/s (38.9..40.1) | 0.93 |
| decode, 4 concurrent, aggregate | 79.9 tok/s | 75.4 tok/s | 0.94 |
| prefill, 2062-token prompt | 370 tok/s | 361 tok/s | 0.97 |

The machine ran ~8% slower than in the first ABAB for both packages (thermal state); ratios are the comparable quantity. The
single-stream ratio moved from 0.73 to 0.93 mostly because DFlash acceptance on the essay prompt differed between runs; the
prompt-independent measure is the B1 cycle time, 98.8 vs 87.3 ms (0.88).

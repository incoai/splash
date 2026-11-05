# Step 1: Ripple draft-kernel geometry prototypes — findings

Run: 2026-11-05, MacBook Pro M5 Max (Mac17,6), macOS 27.0.1, splash `main`
031a25a6 + this prototype branch. Receipts in `receipts.txt`; raw log in
`sweep.log`. Harness: `dev/benchmarks/ripple/` in the
`.worktrees/splash-ripple-step1` worktree (branch `ripple-step1-prototypes`).

## What was prototyped

The production DFlash2 draft attention (`draft_attention_bf16_split/reduce`)
and dynamic convolution (`draft_conv`), re-instantiated with their geometry as
compile-time parameters (`ripple_draft_variant.metal`), driven by a sweep that
checks every case against the CPU reference of
`dev/tests/engine/draft_attention_metal_test.mm` (generalized to a group's row
count) and times each dispatch as its own command. Cases: the shipped geometry
(32 query heads, 4 per KV group, 2048 ring, 4 splits) on both the production
metallib and the variant library (harness cross-check), and Ripple's draft
geometry (80 query heads, 8 KV heads, 128-dim, 4096 ring) as

- h5: two 40-row threadgroups per (lane, KV head), 5 query heads each,
  splits 4 and 8, ring 2048 and 4096;
- h10: one 80-row threadgroup per (lane, KV head);
- the 4-tap dynamic convolution beside the shipped 2-tap one (hidden 5120).

## Findings

1. **The one-80-row-group design is refused by the device.** Pipeline creation
   fails with `Threadgroup memory size (41920) exceeds the maximum threadgroup
   memory allowed (32768)`. The two-40-row-group design (21 KB workspace) is
   the viable shape, as planned.
2. **MPP's `matmul2d` requires M or N to be a multiple of 16.** The ring tiles
   (N=128) are legal at any M; the current-rows tile (N=8) is not. A padded
   16-wide current tile compiled but miscomputed rows of the shipped geometry
   (rows 4-7, 12-15, 17-31 of every head block), so the prototype computes the
   eight current scores per row as plain fp32 dot products (~0.4% of a ring
   tile's MACs) and keeps MPP for the ring QK, ring PV and current PV. The
   production engine change should do the same.
3. **The 40-row tile is at least as efficient per query head as the shipped
   32-row tile.** At 4 lanes, length 2047: baseline 22.3 µs/lane (0.70 µs per
   query head) vs Ripple-h5 50.7-51.2 µs/lane (0.63-0.64 µs per query head) —
   the larger tile amortizes the K/V tile loads over more rows. Split count
   4 and 8 tie within noise (s8's floor is marginally better: 47.8 vs 49.9);
   no deal-out cliff. The 4096 ring at full window costs 2× the tiles of the
   2048 ring and 1.83× the time (94.8 vs 51.9 µs) — real work, not overhead.
4. **The 4-tap dynamic convolution costs +14%** over the 2-tap one at hidden
   5120 (2.52 vs 2.21 µs/lane/stage floor), memory-bound as expected.
5. **Harness validation:** the variant library's shipped-geometry instantiation
   matches the production kernel's output digest bit for bit at every ring
   length and lane count (at length 0 the digest differs, as the manual dot's
   fp32 accumulation order differs from the MPP matmul's; it stays within the
   reference tolerance). Every correctness check passes at cache lengths
   0/512/2047/4096 and lanes 1 and 4.

## Verdict

Step 1 de-risks the draft-kernel extension: the Ripple geometry compiles,
checks against the reference, and runs at per-head efficiency equal to or
better than the shipped kernels, with the split-count knob free. The next step
(step 2 of the plan) is the production integration: per-geometry
instantiations in the production kernel sources, the draft-variant table on
the host side, and the ring/context-window sizing for 4096.

## Step 2 design note: the FFN clamp (for the target integration)

Ripple's SwiGLU clamps its gate and up activations at 20 before the silu
product. The FFN runs fused: prefill as three passes (gate, up-with-gate,
down) whose epilogues the linear kernels own, and decode as one staged-pair
kernel whose epilogue computes `bf16(up) * silu(bf16(gate))` directly from
both accumulators. The clamp therefore lands as:

- prefill: a small elementwise kernel clamping the gate scratch between the
  gate pass and the up pass - no GEMM kernel changes;
- decode: the staged-pair epilogue gains a clamped variant, the clamp value
  read from a uniform, the shipped kernels' codegen untouched - the variant
  selected by the projection's plan as the gate's is, and measured against
  the unclamped one before it serves.

The value residual's mix lands beside it: the v projection's input staging
reads the mixed row (sigma(lambda) * normed x + (1 - sigma(lambda)) * scaled
embeddings), which the mixer's capture of the block's input rows already
holds.

## Step 2 design note: the value residual (for the target's forward)

Ripple's block mixes its v input as
`v_in = sigmoid(lambda) * attn_norm(x) + (1 - sigmoid(lambda)) * scaled_emb`,
the scaled token embeddings the same rows every block mixes back in. The
mixer's q, k and v run as one fused projection of one input, which cannot
give v its own; the residual therefore splits algebraically:

- the fused projection's v rows are scaled by sigmoid(lambda) at load, a
  per-layer constant folded where the Qwen3.5 gate rows are already folded,
  so the fused projection computes the first term for its v columns at no
  runtime cost;
- the second term is a per-layer projection of the scaled embedding rows
  through the v rows alone, which the prepared weights hold contiguously as
  the fused projection's last section (a view, no extra memory), weighted by
  1 - sigmoid(lambda) from the block's lambda buffer; at decode that is a
  1024x5120 GEMV per block, about half a percent of a cycle's weight reads;
- the term adds to the fused projection's v columns where the attention's
  q/k/v prepare stages them - the prepare kernels are per-geometry
  instantiations already, and prefill's chunked rows take the same add.

The scaled embeddings are the gather's output (the gather writes them
scaled), which the block's capture already holds for the draft's taps.

## Step 2 design note: the clamp (supersedes the earlier note's decode part)

The decode epilogue's clamped variant reads the clamp from a uniform of the
family's ffn_clamp_threshold; the prefill clamps the gate scratch between
the passes as written before.

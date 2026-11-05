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

## Step 2 spec: the block graph, from the llama.cpp port (PR #31692, KLD 0)

Per block (the port's build_arch_graph, verbatim semantics):

    residual = x                                  # saved pre-norm
    a = attn_norm(x)
    q, k = Wq(a), Wk(a); per-head norms; RoPE iff sliding (NoPE on full)
    v_in = sigmoid(λ)·a + (1 - sigmoid(λ))·scaled_emb    # scaled_emb = the
        # gather's rows, the same tensor every block mixes back in
    v = Wv(v_in); attn = softmax(q·k/sqrt(d))·v; gated: attn *= sigmoid(Wg(a))
    attn_out = Wo(attn)
    attn_post = post_attention_norm(attn_out)     # a BRANCH norm, pre-residual
    ffn_in = attn_post + residual
    ffn_normed = ffn_norm(ffn_in)                 # the residual's norm, pre-FFN
    ffn_out = Wdown(clamp20(silu(Wgate(ffn_normed)) * Wup(ffn_normed)))
    ffn_post = post_ffw_norm(ffn_out)             # a BRANCH norm, pre-residual
    x = ffn_post + ffn_in

This is a post-norm architecture: each branch's output is normed before its
residual add, which the QwenTarget block helpers (inputNorm → mixer →
residual → postAttentionNorm → FFN → residual) do not express - RippleTarget
composes its own blocks over the shared step machinery, as FlintTarget does
for its composition. The loader and planner are unaffected (they read the
four norms by name); the earlier note's "post-FFN norm stage" meant this
branch norm.

The residual split of the earlier note stands, with the input named: the
fused projection's v rows scale by sigmoid(λ) at load (term one through the
fused projection, whose input is a); term two is (1 - sigmoid(λ)) · Wv ·
scaled_emb, a per-block projection through the v rows alone (the prepared
weights' last fused section), adding where the prepare stages v. The scale
of the embeddings is the gather's (landed), and kq_scale is 1/sqrt(128).

## Step 2 spec: the value residual's final form (supersedes the row-scale split)

The mixed v input is materialized and the v projection reads it, which keeps
the attention kernels' packed row order untouched and adds no weight bytes:

    packed row = [ q | gate | k | v ]     (22528 columns, as today)
    fused projection: q + gate + k  ->  packed[0 .. 21504)   (input: a)
    v mix kernel:     v_in = sigmoid(λ)·a + (1 - sigmoid(λ))·scaled_emb
    v projection:     attn_v (1024 rows) -> packed[21504 .. 22528)
                      (input: v_in; out_offset = 21504, which
                       GgufDecodeParams.out_offset carries and the staged
                       kernels already apply)

Per block, three dispatches (fused, mix, v) instead of one; the weight bytes
are the same (the fused projection drops the v rows it used to carry, the v
projection picks them up). The attention's q/k/v prepare reads the packed v
columns at the tail as it does for every family, unchanged. The λ is the
block's F32 section; the mix kernel computes its sigmoid (a scalar per
block). scaled_emb is the gather's output rows, which the block's capture
already holds for the draft's taps.

Consequences for the loader and planner: the fused read is q+gate+k
(21504 rows) and attn_v plans as its own projection section, so the mixer's
reading is Ripple's own (readQwenAttention serves the Qwen fused three; the
generic reader's mixer hook takes a family's reader, as its FFN hook does).
The layout's packedFullWidth stays 22528 - the packed row the kernels see -
and the fused read's width is the loader's business.

## The server integration gap: the template's tags are text, not tokens

Ripple's vocabulary (100352, OLMo/cl100k-style BPE) keeps none of the
standard tags as dedicated tokens: THINK_START/THINK_END, TOOL_CALL_OPEN
and the call's close are text its tokenizer splits into byte-level BPE
tokens, as the release's tokenizer.json splits the template's own text.
The server's thinking and tool-call machinery reads single token IDs
(chat_format's think_end_id and call_open_id), which startup validates
(constraints.validate_tokenizer) — a Ripple tokenizer fails that check.

The engine is complete without them: the family's descriptor, layout,
planner, loader and target stand, and the draft's subset proposal serves
exactly. The server's integration needs the thinking and call detection to
match the tags' text sequences across the token stream, which the
output.py's pending machinery is shaped for but the chat format's
single-token fields do not state. That is the remaining server work; until
it lands, families.py stays without Ripple, so install does not open to a
serve that would fail at startup.

## The block composition's validation

The post-norm block's pieces are individually tested: the mix and the
placed value projection (ripple-value-mix), the attention kernels
(attention-variants), the norms and the clamped epilogue (the goldens and
the quantized kernel tests). The composition's ordering — which norm feeds
which add, and where the residual joins — is code review against the
pinned graph spec.

The full numerical gate is the parity harness: a tiny Ripple model loaded
end-to-end, its prefill run through the engine and through a CPU reference
of the block graph, the logits compared. The test's block-level attempt
found the float path's epilogue coverage too narrow for an isolated block
test (the gated epilogues are quantized-path only); the harness runs the
real quantized path through the full loading machinery, which the goldens
and the planning tests already exercise.

## The pipeline's end-to-end validation

The tiny model (2 layers at the family's attention geometry, hidden 256,
with a 1-layer shipped-variant draft) loads through the full pipeline:
the descriptor constructs, the planner plans the GGUF's images, the
preparation writes them, the loader reads them into the target's and the
draft's weights, and the target constructs (ripple-load). The fixture's
builder writes the GGUF and the checkpoint at test dimensions with the real
tensor names and formats.

The block composition's compute is validated piece by piece (the mix and
the placed projection, the attention kernels, the norms, the clamped
epilogue) and code-reviewed against the pinned graph. The full numerical
gate is the parity harness: the tiny model's prefill run through the
engine's Runtime and through a CPU reference of the block graph, the logits
compared. The Runtime's setup (the states, the KV pages, the scheduler) is
the oracle test's infrastructure, whose Ripple case needs the tiny model's
fixture and the family's oracle tokens.

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

## The parity gate and what it caught

The parity harness (ripple-parity) prefills the tiny model through the
engine's real machinery — the loaded weights, the prefill arena, a paged
K/V page, the target's dispatches and the head — and compares all four
positions' logits with an fp64 reference that reads the fixture's GGUF
itself and runs the pinned block graph. The gate is the pipeline's
accumulated bf16 rounding, measured at about half a percent of a row's
largest logit; the bound is 0.008 of the scale plus a floor. The worst
delta measured: 0.71 on logits of magnitude 50-90.

It caught two defects the per-kernel tests had missed:

- The window's off-by-one. The port's SWA masks at p1 - p0 >= n_swa
  (llama-hparams.h is_masked_swa, STANDARD), so a row attends the last
  n_swa keys including its own — 1023 keys before it for Ripple's sliding
  window of 1024. The kernels bake the keys before a row, and were
  compiled with 1024. The prefill and verify split and reduce kernels and
  the compiled entry now hold 1023. The draft's kernels take the window as
  the total (draft.metal's row_start is query_position - (Window - 1)), so
  the draft's 4096 is right as it is.
- The geometry's q/k/v prepare and output-gate kernels were never
  instantiated: the variants test exercised only the smaller geometry's
  prepare, and the compiled entry had folded the window into its suffix,
  which the prepare's name does not take. The suffix is the bare geometry;
  splitSuffix builds the windowed split and reduce names.

withPrefillArena hands its caller the whole prefill buffer set (the ANE
FFN's timing and preparation take it apart as before), which the harness
uses to reach the arena's RoPE tables, hidden ring and attention scratch
without the Runtime.

## The model runtime serves the real model

The real model runs through the model::Runtime the engine serves through
(ripple-serve): the request's tokens and page table prefill in the runtime's
chunks, and its decode cycles run the draft's proposal and the target's
verification end to end. Measured on this machine (M5 Max, 128 GB):

- a 32-token greedy continuation: 44.8 tok/s of decode, the draft proposed
  42 tokens of which 25 were accepted (59.5%);
- a 128-token reasoning prompt: 71.4 tok/s, 109 of 132 accepted (82.6%);
- a chat-templated prompt at temperature 0.7: 78.6 tok/s, 111 of 116
  accepted (95.7%);
- a short reasoning question: 28.7 tok/s, 13 of 56 accepted (23.2%).

Against the single-token weight-read ceiling of about 19 tok/s (40.4 GB at
800 GB/s), the draft's accepted rows put the served rate at 1.5-4x the
ceiling, and the acceptance varies with the task as a draft's does.

The model's own format serves: at 'What is 2+2?' it opens its reasoning
block, reasons briefly, closes it and answers 4 - the tags it writes are
the text ones the server's detection has to read (chat_format's gap, which
still gates families.py).

## The server reads the tags as text

families.py carries the family, whose chat format holds the think and call
tags as text (the token-id fields None). The server's machinery reads them
where the format says the tags are text:

- startup checks each tag by encoding it and decoding it back unchanged;
- the reasoning's tokens count through the decoded stream, from the opener's
  text to the close's first character (or a call's opening where one ends
  the reasoning);
- a thinking budget has no engine cutoff token: the reasoning runs to the
  model's own close;
- a tool grammar's thinking block is a suffix rule, whose lexeme runs to
  the close's first spelling and ends with it. The shipped shape, a TEXT
  terminal and a close literal, fails with multi-token tags: the lexer
  commits the close's earlier tokens to the TEXT, and the close's last
  token satisfies nothing. And a TEXT exclusion of the tags' bytes cannot
  stand beside the suffix rule: its forbidden path prunes the shared
  lexer's path through the close, which the suffix's completion needs. The
  TEXT rule of a text-tags format excludes nothing; the lexer prefers the
  structural literals over TEXT regardless;
- the calls' arguments are one JSON value (the syntax carries
  json_arguments): a strict tool's grammar generates its schema's JSON, a
  loose one's any JSON, and the projector streams the value as it is
  written, framed by the name's close and the call's close tag.

The tool responses render through the model's own template, which the
installer carries from the GGUF; nothing on the prompt side needs the
format's tags.

## The served tool loop, and the mask's word boundary

The installed model serves through the server over HTTP: plain
completions, a reasoning turn, and a full tool loop — the call emitted
with its arguments as one JSON value, the tool result rendered back
through the template, and the model's answer reading it ("It's currently
14:00 (2:00 PM) CET in Paris") — at about 45 tok/s of decode per cycle.

The tool loop's second turn crashed the engine: a page of the Slim runway
refused layer 0's K/V. The mask's countBelow read a layer past the first
word's bits as count minus the word's suffix, which only holds when no
windowed layer sits at or above the layer; the first family whose window
runs past bit 64 read every later layer's region wrong, and the trim
moved its pages by the wrong regions. The count walks the whole words
below the layer's, then its word's bits below its bit.

Two smaller catches on the way: the spilled history's one-layer staging
layout drops its windowed mask (a one-region page cannot hold a windowed
layer's region), and the engine's memory plan carries the family's own
prefill scratch, which the audit counted against the plan it did not
know.

## The runtime's own numbers (SPLASH_SERVE_BENCH, M5 Max)

The served path's costs, measured black-box through the model runtime (the
draft proposes, the target verifies; the AD-Q4_K_M target and the DFlash2
draft, 42.8 GB of weights together):

- prefill 32 tokens: 0.127 s (253 tok/s); 128: 0.40 s (317); 512: 1.20 s
  (428); 2048: 4.88 s (420). Long prefill is compute-bound at about 55
  effective TFLOPS (65B × 2 FLOPs per token); the short chunk's 0.127 s
  sits about 1.7× over the memory floor of one pass over the weights
  (42.8 GB at the machine's 546 GB/s is 78 ms).
- decode: mean cycle 0.111 s at a 96-token budget (57.5 tok/s, 4.2 tokens
  a cycle, 78% accepted), 0.097 s at a 192-token one (73.2 tok/s, 4.9 a
  cycle, 87.8% accepted over 27 cycles). One cycle reads the whole 42.8 GB
  once: 386-441 GB/s effective, 70-80% of the machine's bandwidth — the
  headroom is real but bounded, and the acceptance is the bigger lever
  (each accepted row is bandwidth the target did not spend).

Earlier "12x off the ceiling" figures from the server's TTFT were the
HTTP path's own overheads, not the runtime's.

## The same harness on the shipped family (Qwen3.8-27B, UD-Q3_K_XL GGUF)

The bench is family-agnostic now (the descriptor comes from the model
root, inspectModelRoot), so the shipped DFlash2 path runs the same
protocol on this machine:

- prefill 2048 tokens: 1.85 s (1106 tok/s) — 27B × 2 FLOPs per token is
  about 60 effective TFLOPS, the same kernel efficiency as Ripple's;
- decode: 24 cycles, mean 0.0400 s, tight (best 0.0398, worst 0.0415),
  200 tok/s, 8 tokens a cycle, every proposal accepted (greedy, a
  deterministic prompt). 14.7 GB a cycle is 368 GB/s effective.

So the per-byte kernel efficiency is the same implementation's — both
families run 368-466 GB/s effective per cycle. The decode rate differs
where the models differ: Ripple's weights are 2.9x the bytes per pass,
its draft is 8.1 GB (5 layers of the target's FFN size) against Qwen's
2.2 GB, and its block proposes 16 candidates against Qwen's 8 — whose
greedy acceptance is 100% where Ripple's subset-proposed 16 reach 78-88%.

The decode levers, in the order the numbers put them: the trained
selector's 32-candidate kernels (the subset proposal caps acceptance),
the acceptance itself on real workloads, and the last 15-30% of
bandwidth.

## Why the decode levers are not a constants bump

The trained proposal wants what the ABI does not carry: Ripple's draft was
trained for 16-row blocks and a 32-candidate selector; the kernels run
8-row blocks (SPLASH_DRAFT_QUERY_ROWS / SPLASH_TARGET_VERIFY_ROWS) and
keep 16 candidates (SPLASH_DRAFT_CANDIDATES) — the shipped families'
numbers (Qwen3.8's draft: block 8, top-16). Both sides are real: the
config's block and candidate count name what the draft was trained for,
and the ABI names what the kernels, buffers, state layout and wire carry.

A global bump is a no-go: the descriptor refuses a draft whose
block_size or selector_top_k is below the kernels' (the shipped
Qwen3.8-27B draft, block 8 and top-16, would be refused by kernels at 16
and 32). The fix is the attention variants' own pattern: the draft
variant carries its proposal rows and candidate count, the affected
kernels instantiate per variant, and the buffers, state layout and
descriptor check read the variant's numbers.

The consumers, enumerated: the metal kernels that bake the constants
(draft.metal's forward and proposal, sampling.metal's three selector
kernels, the decode qkv and verify staging, paged attention's verify
tiles, embedding and gdn staging, sparse attention, ngram embedding, the
store) instantiate per variant; the decode arena's tensors (FullPacked,
logits, verify rows, proposal probabilities) size from the geometry's
variant; the state layout's rings and the KV block's rows follow; the
descriptor's check becomes "the config's counts are ones the build
compiles"; the wire's proposal arrays follow the descriptor.

The measured prize: the draft proposes ~7 of its 16 rows a cycle (a
confidence early-stop the truncated candidate list makes fire early) and
88% of them are accepted; the trained 16-row block at the trained 32
candidates would propose and accept more per cycle at the same 42.8 GB
pass. That, with the last 15% of bandwidth, is the decode headroom the
refactor buys.

# Optional BF16 target KV

`--kv-format bf16` selects unquantized BF16 target KV at startup. INT8 remains
the default. Target weights, draft KV, GDN state, sampling, and scheduling are
unchanged. BF16 uses about twice the target KV memory; it is a quality/memory
choice, not a general speed improvement.

## Implementation boundaries

- One `kv::Layout` supplies allocation, admission, cache identity, and attention
  planning. Format is part of execution-policy keys, so policies cannot cross
  formats, and of the cache identity `/status` reports. One process serves one
  format and keeps KV only in its own extents and unlinked slot files, so
  cached blocks cannot cross formats either.
- `PageStorage` owns both formats. BF16 extents have no scale regions. Every
  region of an extent starts 64 KiB aligned, and extents target about 128 MiB.
  Logical prefix blocks remain 32 tokens.
- The shared Metal page loop specializes on the stored element type. BF16 stores
  preserve source bits and attention omits quantization scales at compile time.
  INT8 entry points, argument order, arithmetic, and dispatch policies remain.
  Both formats share the FP32 split reduction. Format-generic code is named for
  paged KV; only the INT8 entry points, quantization and scales say q8.
- Status reports the selected format and its actual byte geometry.

## Validation on 2026-09-21

Devices: M3 Max (Apple9, 40 GPU cores, 128 GB) and M5 Pro (Apple10, 16 GPU cores,
48 GB). Both ran on AC. The M3's 65 W supply could not sustain long full-load
runs; it was paused to recharge. Results are not a claim about every device or
power condition.

- Full runtime oracle: both 27B/35B models, both KV formats, both devices,
  eight runs passed. Covers state restore, prefill partitioning, constrained
  sampling, preemption, and batched execution.
- Shader-validated independent attention references: both formats and both
  attention geometries, including 131,072 and 260,096 history tokens. The latter
  prefill adds 2,048 rows and reaches 262,144 tokens. Decode includes B1–B4;
  long numerical cases use B4. Store checks include page boundaries, untouched
  bytes, and exact BF16 edge-pattern preservation.
- Storage admission/release and format-separated plan/cache identities pass.
- Final default-INT8 attention comparison against main (`e8fffde`): bit-identical
  outputs across histories 0, 2,048, 131,072, 260,096 and both model geometries.
  At long histories, elapsed-time changes ranged from -0.19% to +0.09% on M3 and
  -0.41% to +0.57% on M5 (17 alternating samples). Short microsecond cases are
  noisier and are not used to assert universal non-regression.
- Real-model decode profiling at 512 prompt tokens, B1–B4, seven cycles per
  width, order main/INT8/BF16/BF16/INT8/main: default INT8 cycle-time changes
  ranged from -0.08% to +0.81% on M3 and -0.71% to +0.07% on M5.
- Native long-context runs completed through 256K for 35B in both formats on
  both devices. With PR #92 integrated, the 27B 256K matrix completed
  as well: both formats on both devices, 262,016 input tokens plus 128 output
  tokens, followed by exact-prefix replay. Each replay reused 261,984 tokens
  and matched its cold run's 128-token output exactly. Active KV pages and
  state cells returned to zero after each request. The M5 INT8 output also
  matched the retained main run on the same input token-for-token.

The benchmark tools accept `--kv-format int8|bf16|zbf16`. `attention-sweep` accepts
`--compare-metallib BASELINE` and checks exact output equality (the baseline
must be built from a tree with `residency_kick`, which every backend loads);
`paged-attention-plan METALLIB --long` runs the long independent references.

## Combined serving validation with PR #92

The integrated build passed all four real HTTP and runtime-oracle combinations
(27B/35B, INT8/BF16) on M5 Pro 20. HTTP coverage includes tools, structured
output, images, prefix reuse, mixed scoring/chat, and timeout recovery. One
initial 35B load was refused by the host-memory preflight after switching
models; an unchanged retry passed. No memory guard or numerical threshold was
relaxed. M5 Pro 16 also passed all four runtime oracles and shader-validated
128K/256K independent attention references in both formats.

The combined Python suite ran 766 tests: 764 passed and two opt-in external CLI
routing tests were skipped. Production/CPU/sanitizers and Python 3.12–3.14 CI
passed, as did the full Metal gate with shader validation on M5 Pro 20.

The real HTTP smoke and ABBA tools accept `--kv-format int8|bf16|zbf16` and check
the running format and its quantization/scale identity. M5 Pro 20 HTTP ABBA
against main passed transcript/usage equality and the unchanged 2% regression
limit for both models (three samples per build, 2K/8K cold and cached requests,
plus 64-token decode). Native 256K runs establish completion and cache/resource
behavior on a synthetic repeated prompt; they are not a task-quality score or
a universal performance guarantee.

## State-oracle scope and numerical investigation

The runtime commit test now compares a budget-truncated greedy cycle with a
constrained cycle that retains the same proposal prefix using a larger budget.
It requires exact draft-cache bytes, proposal identity, and retained counts; a
cycle's GDN state is its cell from before the cycle and the retained rows,
which the next cycle folds in. Independent GDN kernel references cover the
recurrence and that fold.

This replaces the former whole-network prefill/decode state-cosine comparison;
it changes coverage rather than lowering that comparison's tolerance. The old
27B BF16 comparison still fails on M5: convolution cosine 0.984408 against 0.99.
The first differing GDN recurrence produces three one-step BF16 differences out
of 49,152 values; later layers amplify the perturbation. All 96 captured
layer/path checks satisfy the existing FP64-reference bounds. No production
arithmetic was changed to make the new test pass.

A separate diagnostic compared the exact failing eight-token sample with
unmodified MLX-LM 0.31.3 / MLX 0.32.0, after checking 677 target-weight sections.
Both Splash paths and MLX's aligned batch/single-token paths chose the same top
token at all eight positions. Mean KL against those references was
0.0000594–0.0002685; mean Splash cross-phase KL was 0.0001955. Repeated captures
were byte-identical and retained the original state-cosine failure. This is a
focused reproducer check, not a general task-quality or long-generation bound.
Diagnostic capture code and model tensors are not part of production sources.

## Lossless ZBF16 (2026-10-07)

`--kv-format zbf16` stores BF16 target KV bit for bit in 0.8125 of its bytes
(`runtime/metal/abi/KvZip.h`): each value's sign and mantissa byte as it is,
its exponent as a 4-bit code against a per-(layer, tensor, KV head, dimension)
16-binade window calibrated per model family, and the values outside their
window as escapes in a 252-entry table per 32-token slab of one head, the
coding SplitZip (Guo and Joshi, 2026) uses for KV transfer. Pages keep the
BF16 extent geometry: the data regions hold sign/mantissa bytes and codes,
the scale regions the escape tables, so the pool, prefix cache, SSD tier,
persistent cache and memory plan handle the pages unchanged.

Boundaries:

- `kv::Format::ZipBFloat16` is a third format of the same `kv::Layout`;
  `PageStorage` owns the codec buffer (a header with the escape-overflow count,
  then the bases of `runtime/model/kvzip/<family>.inc`), which the store and
  attention kernels bind after their page tables.
- A chunk of more than 32 rows runs the BF16 split and reduce: before them,
  `addPrefill` expands the history the chunk attends, its own rows (which the
  store just encoded) included, into a one-layer BF16 scratch of the prefill
  arena (`PrefillTensor::ZipScratch`), from the view it attends, so pages
  staged back from the SSD tier expand as the pool's do; one bandwidth pass
  per layer and chunk (27B: 0.9 ms at 32K, 3.8 ms at 128K). The scratch's
  table names its pages by address, so the arena is allocated
  `BufferReach::Addressed` when it holds one. A chunk of up to 32 rows (four
  query tiles) skips the expansion: its split decodes each page into a
  threadgroup tile as verify's does, and the BF16 reduce follows
  (`PrefillAttentionPlan::zipExpand`).
- Only dense layers of 256 dimensions take ZBF16 (`kv::Layout::valid`): no
  block index, no window and the dense loop's value order, the 27B's and the
  35B's geometries.
- Verify decodes each page into a 16 KiB threadgroup tile, token-major, and
  runs the BF16 loop's MPP products and softmax over it, bit for bit.
- A slab with more escapes than its table stores the rest with code 0 (never
  an infinity or NaN); the store counts it, `/status` reports
  `identity.kv.overflow_slabs` and the server logs an error. Real KV needs 0.8
  escapes per slab on average (at most 26 over 150K captured tokens of both
  models); prompts of one repeated token up to ~200.
- The persistent cache namespace includes the bases' digest and the page
  layout's version (`SPLASH_KVZIP_LAYOUT_VERSION`).

Validation on M5 Pro (16 cores, 48 GB), both models, GGUF UD-Q4_K_M:

- `kv-zip` (shader-validated): the stores write the host reference codec's
  bytes; every row decodes to its bits; prefill and verify attention equal the
  BF16 kernels' output bit for bit, including rewritten rejected rows and a
  slab past its escape table.
- Greedy HTTP outputs on four prompts (short, story, 30K code, 40K prose):
  identical to `--kv-format bf16` on both models; prefill times within noise
  of BF16's.
- `attention-sweep --kv-sample` on captured KV, per layer, verify. The first
  version decoded slower than BF16 (27B 3.06 ms at 128K against 2.71, 35B
  1.79 against 1.54). Two changes (2026-10-28) make the 27B faster than BF16
  at every context and bring the 35B within 4-7%:
  - The 27B's scores have their own scratch, so a page's values decode in
    its softmax's phase: four barriers a page instead of six.
  - Decoding costs a third less: the page layout orders each word's
    sign/mantissa bytes and codes so that a word decodes as two BF16 pairs
    in 32-bit arithmetic (`splash_kvzip_pair`), the codec holds the bases in
    the same order, and each slab's bases load with its bytes instead of
    sitting in registers across the page loop (held, the compiler hoisted
    their pairs and the 35B lost 30% to the registers).
  Timing probes on the 27B at 128K before the second change: the tile's
  products and barriers alone 1.44 ms, the slab loads 0.83, the decode
  arithmetic 0.48. ABBA medians on 38K captured tokens of code and prose
  per model, M5 Pro 16 (ms; the PR's earlier head in parentheses):

  | Model | History | Lanes | BF16 | ZBF16 |
  | --- | ---: | ---: | ---: | ---: |
  | 27B | 2,048 | 1 | 0.086 | 0.079 (0.092) |
  | 27B | 32,768 | 1 | 0.779 | 0.740 (0.812) |
  | 27B | 32,768 | 4 | 2.848 | 2.789 (3.003) |
  | 27B | 131,072 | 1 | 2.730 | 2.683 (2.873) |
  | 27B | 131,072 | 4 | 10.637 | 10.488 (11.286) |
  | 35B | 2,048 | 1 | 0.037 | 0.050 (0.061) |
  | 35B | 32,768 | 1 | 0.464 | 0.492 (0.526) |
  | 35B | 131,072 | 1 | 1.536 | 1.648 (1.768) |
  | 35B | 131,072 | 4 | 5.823 | 6.201 (6.625) |

  Outputs are unchanged: every digest equals BF16's and the earlier
  layout's, and greedy outputs of the 27B (a short prompt, and 34K tokens
  of code) equal BF16's token for token.

  Rebased onto main of 2026-10-28 (its tile parts, windows, sinks and
  staged history), the same ABBA (ms):

  | Model | History | Lanes | BF16 | ZBF16 |
  | --- | ---: | ---: | ---: | ---: |
  | 27B | 2,048 | 1 | 0.086 | 0.080 |
  | 27B | 2,048 | 4 | 0.305 | 0.271 |
  | 27B | 32,768 | 1 | 0.766 | 0.732 |
  | 27B | 32,768 | 4 | 2.834 | 2.767 |
  | 27B | 131,072 | 1 | 2.688 | 2.646 |
  | 27B | 131,072 | 4 | 10.405 | 10.308 |
  | 35B | 2,048 | 1 | 0.038 | 0.062 |
  | 35B | 32,768 | 1 | 0.462 | 0.452 |
  | 35B | 32,768 | 4 | 1.592 | 1.685 |
  | 35B | 131,072 | 1 | 1.530 | 1.631 |
  | 35B | 131,072 | 4 | 5.704 | 6.110 |

  INT8 and BF16 against main's kernels: every digest equal, every time
  within noise (24 cases). Greedy outputs of the 27B equal BF16's again.
- The 35B's verify (eight query heads per KV head, 64 rows a tile) stays
  4-7% behind BF16 at 32K-128K. Its kernel, probed at 128K: products,
  softmax and barriers over the tile 0.94 ms, slab loads and tile writes
  0.46, decode arithmetic 0.16. Three kernels tailored to the shape lost or
  tied (ABBA, captured KV): scores apart from the tile (-5 to -8% at 2K,
  +6% at 32K: its 8 KiB leaves two threadgroups a core instead of three);
  a pipelined loop over two 8 KiB half tiles that decodes each half beside
  a product (+26-30%: half-width MPP products cost more than the barriers
  they hide); and loading a page's values with its keys (within 3%: the
  registers cost the occupancy the early loads save). End to end decode
  does not show the gap: `backend-benchmark --scenario decode,context
  --max-context 65536` on the 35B (ABBA, two samples a run, 2026-10-30)
  decodes 64 tokens a lane in 232, 257, 334 and 356 ms of GPU time at one
  to four lanes (BF16 233, 256, 333, 355), with the same output tokens, and
  cold prefills of 2K-50K tokens are within 0.4%. Short resumes over long
  histories were not: an answer's first token after 10K and 50K cached
  tokens, 16 new rows, took 39.0 and 53.4 ms (BF16 37.6 and 46.9). Each
  layer expanded the whole history for 16 rows; below, short chunks decode
  their pages instead.
- The 35B's verify now picks its split per step: scores apart from the tile
  while every lane's history is at most 16K tokens, shared past it. Both
  splits compute the same bits. ABBA against the shared split alone
  (captured KV, split ms): four lanes 0.131 against 0.144 at 2K, 0.235
  against 0.256 at 4K and 0.428 against 0.452 at 8K, a tie at 16K, and
  1% slower at 32K. With one lane the two are within 3% up to 16K. Against
  BF16 (ms, fused):

  | History | Lanes | BF16 | ZBF16 |
  | ---: | ---: | ---: | ---: |
  | 2,048 | 1 | 0.038 | 0.059 |
  | 2,048 | 4 | 0.190 | 0.176 |
  | 8,192 | 1 | 0.126 | 0.138 |
  | 8,192 | 4 | 0.506 | 0.500 |
  | 16,384 | 4 | 0.858 | 0.910 |
  | 32,768 | 4 | 1.617 | 1.686 |
  | 131,072 | 4 | 5.708 | 6.198 |

  The other probes of the long-context gap lost or tied, which places it
  in the per-page decode phases at three threadgroups a core:
  - Threadgroup memory sits at the edge of the third threadgroup. The
    split uses 21,264 bytes; 2 KiB more cost 11% at 32K-128K and 8 KiB
    more cost 15-20%.
  - A fourth threadgroup does not pay. A 16 KiB layout (probabilities and
    statistics in the tile), alone or with a lower-register decode, ties.
  - Neither does a lower register peak: a decode in two 16-dimension
    passes ties.
  - Issuing loads earlier costs registers. Values loaded beside the
    softmax are 10% slower; values loaded with the keys tie.
  - Tile stores rotated across banks are 0-3% slower.
  - The verify store encoding eight rows a barrier round ties. Its 7 us
    at 2K (BF16 3 us) is latency, not barriers.
- Prefill chunks of up to 32 rows (four query tiles) decode their pages in
  the split, as verify does, instead of expanding the history into the BF16
  scratch: a resume's few rows paid a bandwidth pass over the whole history
  in every layer. Both paths compute the same bits; `kv-zip` checks each
  side of the threshold. Per layer, captured KV, ABBA (fused ms; expanded in
  parentheses):

  | Model | History | Rows | BF16 | ZBF16 |
  | --- | ---: | ---: | ---: | ---: |
  | 27B | 8,192 | 16 | 0.340 | 0.350 (0.559) |
  | 27B | 32,768 | 16 | 1.137 | 1.216 (2.002) |
  | 27B | 65,536 | 16 | 2.172 | 2.414 (3.910) |
  | 27B | 65,536 | 32 | 3.995 | 4.703 (5.749) |
  | 35B | 8,192 | 16 | 0.223 | 0.258 (0.331) |
  | 35B | 32,768 | 16 | 0.794 | 0.874 (1.239) |
  | 35B | 65,536 | 16 | 1.543 | 1.684 (2.423) |
  | 35B | 65,536 | 32 | 2.919 | 3.256 (3.752) |

  Eight tiles lose to the expansion at short histories (35B, 64 rows at 2K:
  0.264 against 0.230). The 35B's split shares its tile with the scores at
  every history: own scores are 1-3% slower here, where a short chunk's
  grid leaves the cores room. End to end (the same `backend-benchmark`, two
  ABBA rounds), an answer after 50K cached tokens reaches its first token
  in 49.0 ms (BF16 47.2; 53.4 before) and after 10K in 38.4 (BF16 38.5);
  decode, cold prefills and output tokens are unchanged. Longer chunks
  still expand: a 176-row continuation after 50K is 2% slower than BF16.
- INT8 and BF16 unchanged: every attention kernel's bits and times match the
  upstream metallib (36 cases), and the ABBA regression against upstream
  (`test-performance-real` with `BASELINE`, idle, ANE split off on both)
  passes every metric within 1.2%.

Calibration: serve the model with `--kv-format bf16 --max-cache-disk 40G
--persistent-cache --cache-dir DIR`, send prompts as long as the contexts it
will serve, stop it cleanly and run `dev/tools/kvzip_calibrate.py` on the
directory's `kv.slots`; the earlier calibration (NPLG, 3-bit codes in groups
promoted to 16-binade or raw tiers, 0.734 of BF16's bytes) decoded 1.8x
slower than BF16 and is in this branch's history.

## Lossless INT8 (2026-10-28): not worth a format

INT8 codes computed exactly as the store writes them (per-token absmax / 127)
from the same captured BF16 KV hold 6.85-7.0 bits each (order-0 entropy), and
6.4-6.75 bits per (layer, KV head, dimension); deltas along tokens hold no
less. The best fixed-width codings a kernel could decode keep 0.93-0.98 of
INT8's bytes: groups of 16 at 5-8 bits with a 2-bit tag, or a calibrated
width per dimension with escapes (27B K 0.953, V 0.983; 35B K 0.934, V
0.956). ZBF16's verify spends about 6% over BF16 on decoding through
threadgroup memory, which INT8's MPP products, reading device memory
directly, do not pay; at most 7% fewer bytes cannot buy that back.

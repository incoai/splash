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
It requires exact GDN and draft-cache bytes, proposal identity, and retained
counts. Independent GDN kernel references cover the recurrence and commits.

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
- Prefill runs the BF16 split and reduce: the store expands a chunk's committed
  history into a one-layer BF16 scratch of the prefill arena
  (`PrefillTensor::ZipScratch`) and writes the chunk's rows there as well, one
  bandwidth pass per layer and chunk (27B: 0.9 ms at 32K, 3.8 ms at 128K).
- Verify decodes each page into a 16 KiB threadgroup tile, token-major, and
  runs the BF16 loop's MPP products and softmax over it, bit for bit.
- A slab with more escapes than its table stores the rest with code 0 (never
  an infinity or NaN); the store counts it, `/status` reports
  `identity.kv.overflow_slabs` and the server logs an error. Real KV needs 0.8
  escapes per slab on average (at most 26 over 150K captured tokens of both
  models); prompts of one repeated token up to ~200.
- The persistent cache namespace includes the bases' digest.

Validation on M5 Pro (16 cores, 48 GB), both models, GGUF UD-Q4_K_M:

- `kv-zip` (shader-validated): the stores write the host reference codec's
  bytes; every row decodes to its bits; prefill and verify attention equal the
  BF16 kernels' output bit for bit, including rewritten rejected rows and a
  slab past its escape table.
- Greedy HTTP outputs on four prompts (short, story, 30K code, 40K prose):
  identical to `--kv-format bf16` on both models; prefill times within noise
  of BF16's.
- `attention-sweep --kv-sample` on captured KV, per layer at 128K history,
  one lane: verify 27B 3.06 ms (BF16 2.71), 35B 1.79 ms (BF16 1.54); at
  32K 0.85 (0.76) and 0.53 (0.44). The BF16 verify kernel runs within ~25%
  of its compute floor under 8-row speculative verify, so a decoded page's
  wait for its own bytes costs more than the 19% fewer bytes save: lossless
  pages give capacity (23% more context), not a decode speedup, on this GPU.
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

# Experimental exact AF4 coefficient dictionaries

This draft implements the exact native metadata experiment in
[issue #658](https://github.com/incoai/splash/issues/658). An explicit
development call returns a projection with owned U16 dictionary indices and
U32 entries containing the original raw BF16 scale/bias pair. It preserves
every bit, the AF4 codes, FP32 affine expression, FP16 staging, matrix geometry
and split reduction order. Serving loaders do not attach dictionaries.

The supported path is one unrotated AF4G64 plane tensor, and staged decode at
8/16/32 physical rows, including residual, FP32 plain output, and regular or
fused gate/up. Unsupported formats, views and capacity overflow fall back to
the original projection. Register decode, prefill and fused multi-segment
consumers retain original metadata. There is no on-disk format change.

**This prototype adds memory.** It retains original metadata for fallback
consumers. Replacing that metadata across every consumer would be a separate
format/loader/export change. The dictionary has projection lifetime ownership;
there is no global cache or cross-model retention. Unique entries are per
projection, so their sum can exceed the whole model's global unique-pair count.

## Run it

Use a supported Mac with a Metal toolchain and a Qwen3.8-27B AF4 model root
with its DFlash2 draft already installed:

```sh
make -j4 build/engine-tests/affine-dictionary-experiment build/engine-tests/gguf-projection
MTL_SHADER_VALIDATION=1 build/engine-tests/gguf-projection build/splash.metallib

# Exhaustive actual-model GPU coefficient audit; separate from timed runs.
build/engine-tests/affine-dictionary-experiment build/splash.metallib MODEL_ROOT \
  --dictionary 1 --audit 1 --prompt-index 0 --output-tokens 16 > audit.json

build/engine-tests/affine-dictionary-experiment build/splash.metallib MODEL_ROOT \
  --dictionary 0 > control.json
build/engine-tests/affine-dictionary-experiment build/splash.metallib MODEL_ROOT \
  --dictionary 1 > dictionary.json
```

The benchmark attaches dictionaries to the target's MLP gate/up/down
projections only. It warms up once, then runs six frozen `TranscriptPrompts`
as cold greedy requests, with a 256-token limit, INT8 KV, FP32 recurrent state
and GPU execution. It calls `Runtime` directly, excluding HTTP, the serving
scheduler, ANE and cache reuse. `--prompt-index 0..5`, `--tokens-file`
(whitespace separated U32 IDs), and `--output-tokens 1..4096` support individual
and long-context probes. Token IDs and context extent are validated.

JSON includes complete input/output token arrays, request wall time and TTFT,
cycle wall/GPU time, accepted/emitted tokens, conversion time, preparation,
warmup, dictionary modules/units/entries and additional payload bytes.
`physical_rows` is the fixed eight-row arena width, not an observed selector
count. One-row verification cycles are included; a terminal anchor without a
GPU cycle is excluded. Measure complete process wall time and peak RSS
separately. `--audit 1` reconstructs every actual coefficient on the GPU and
compares its bits; keep that audit outside matched timing runs.

## Qualification

Construction uses exact U32 keys in deterministic first-occurrence order and
audits every original pair on the CPU. More than 65,536 distinct pairs falls
back to the original projection. The operator tests additionally reconstruct
all coefficients on the GPU, then require bit-identical output and independent
FP64 error bounds for the supported row counts, applicable splits and both
spread settings. They include fused/two-pass gate/up, residual and FP32 output,
metadata buffer offsets, output padding, leading views, register/prefill
fallback and capacity/format fallback. The full shader-validated projection
suite passes, as does the native CPU suite.

The default staged reader is the original packed load; only the new kernels
instantiate the dictionary reader. On the tested compiler, all 2,104 original
normalized AIR functions are identical; 16 experimental functions are added.
This demonstrates original kernel isolation, not a performance benefit.

On the pinned actual model, the separate GPU audit covers 192 MLP projections
and 267,386,880 coefficient units. Added payload is 536,946,968 bytes
(approximately 0.500 GiB), including 543,302 summed per-projection unique pairs.
That figure is buffer payload, not allocator peak RSS. The accompanying dated
results retain startup/RSS, repeated native comparisons, full token arrays,
long-context pilots and practical noise floors. MLX results are historical
motivation only and must not be presented as native gains.

The native format already loads a packed U32 pair. Indexing therefore adds a
dependent dictionary access in exchange for fewer streaming metadata bytes.
The affected ideal AF4 stream falls from 36 to about 34 bytes per 64 weights,
which bounds byte-only kernel benefit around 5.88% under ideal bandwidth-only
conditions. It does not bound other cache effects or predict model speed.
Before adoption, measure wider workloads/devices, separate kernel/cache effects
from conversion thermal history, cover all consumers, and implement/measure
offline packaging and complete metadata replacement if RAM saving is a goal.

Pinned native observations and verdict: [2026-11-03 results](results/dictionary-native-20261103/README.md).

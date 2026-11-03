# Experimental dense verification prefixes

This draft implements the dense, B1 part of [issue #658](https://github.com/incoai/splash/issues/658).
It reuses Splash's existing verification selector and acceptance model. The
new instrumented control can cap the selected prefix at 1–8 rows, or supply
eight independently measured cycle costs to the existing adaptive selector.
Serving does not link this control API. Default dense and MoE policy stays
unchanged.

The selected prefix removes current-step GDN convolution, recurrence and gate
work for the suffix. Every thread still reaches the barriers and every pending
committed row is folded into state. Convolution carry is preserved, unused
hidden rows are zeroed, and unused tape rows remain untouched. Physical storage
and strides remain eight rows. Dense projections, attention, draft generation
and other producers still execute their original width. This is a partial work
reduction experiment, with FP32 recurrent state only.

## Run it

Use a supported Mac, its Metal toolchain, and a Qwen3.8 dense model root with
its DFlash2 draft already installed. No model conversion is performed.

```sh
make -j4 build/engine-tests/dense-verify-experiment build/engine-tests/gdn-decode
MTL_SHADER_VALIDATION=1 build/engine-tests/gdn-decode build/splash.metallib

build/engine-tests/dense-verify-experiment build/splash.metallib MODEL_ROOT \
  --rows all > control.json
build/engine-tests/dense-verify-experiment build/splash.metallib MODEL_ROOT \
  --rows 5 --prefix-gdn 0 > fixed-five-full-gdn.json
build/engine-tests/dense-verify-experiment build/splash.metallib MODEL_ROOT \
  --rows 5 --prefix-gdn 1 > fixed-five-prefix-gdn.json
```

Each command warms up once, then runs six frozen `TranscriptPrompts` as cold
greedy requests, with a 256-token limit, INT8 KV, FP32 recurrent state and GPU
execution. The tool calls `Runtime` directly; it excludes HTTP, the serving
scheduler, ANE and cache reuse. Optional `--prompt-index 0..5`, `--tokens-file`
(whitespace separated U32 token IDs), and `--output-tokens 1..4096` permit
individual and long-context probes. IDs and total context (at most 32,768,
including output/runway) are checked before weight loading.

For adaptive selection, first measure fixed rows 1–8 on an input separate
from the timing cohort. Take the median **full cycle wall time** of cycles
whose actual `verify_rows` equals that fixed limit, freeze all eight costs,
then pass the positive values in row order:

```sh
build/engine-tests/dense-verify-experiment build/splash.metallib MODEL_ROOT \
  --rows 1 --tokens-file calibration.tokens --output-tokens 96
# Repeat with rows 2..8; do not fit costs to expected task answers.
build/engine-tests/dense-verify-experiment build/splash.metallib MODEL_ROOT \
  --rows adaptive --costs C1,C2,C3,C4,C5,C6,C7,C8 > adaptive.json
```

Costs must describe the chosen prefix-kernel setting and workload. The
acceptance model starts from the existing prior for each measured request,
learns within that request and explores the full prefix periodically. Costs
remain externally supplied. There is no automatic context-dependent cost
model, persistent cross-request learning, or new acceptance algorithm here.
Do not reuse short-context costs as evidence of a suitable long-context policy.

JSON retains complete input/output token arrays, request wall time and TTFT,
actual selected rows, accepted/emitted tokens, and full cycle wall/GPU time.
One-row cycles are included even though their drafted-token count is zero;
a terminal anchor emitted without a GPU cycle is excluded from cycle records.
Preparation and warmup are separate from measured requests. Process wall time
must be measured separately when assessing first-use latency.

## Qualification

The new GDN tests exercise every compiled shape, 1/4 lanes, both norm storage
types, grouped/tiled heads and plain/Table16 output. They compare live hidden
rows, every live tape field and the entire resulting committed state bitwise
against the unchanged eight-row kernel, across pending counts 0–8 and live
prefixes 1–8. Poisoned suffix tape must remain untouched; dead hidden rows
must be zero; fused tables must match independently prepared tables. Short
prefix-count buffers and BF16 state are rejected before encoding.

The original helpers live in `common/gdn_decode.h`; a local preprocessor path
is selected only by the new `gdn_prefix.metal` translation unit. On the tested
compiler, all 2,104 original normalized AIR functions are unchanged; the
experiment adds 16 FP32-state entry points. This checks isolation, not speed.

The native CPU suite and shader-validated GDN suite pass. The accompanying
dated results contain the native comparison, frozen calibration, artifact
hashes, complete token observations and exclusions. Use its practical floor
and workload scope when interpreting timing differences. Neither unit tests
nor repeated known prompts establish broad task-quality noninferiority.

Before adopting a serving policy, measure context-dependent costs and wider
workloads, integrate row/shape planning for the remaining expensive operators,
and qualify BF16 state, multi-request scheduling, preemption, cache restore,
constraints and other devices. The current B1 development control does not
claim those serving contracts.

Pinned native observations and verdict: [2026-11-03 results](results/dense-native-20261103/README.md).

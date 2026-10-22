# Qwen3.8-27B real-checkpoint CUDA smoke

These runs qualify the optional training implementation at
`d066629ff370d94e645bbdd6d32f73a4c8194838` on actual CUDA hardware.
They do not establish a model improvement. No experimental weights or serving
defaults are proposed for deployment.

## Inputs and execution

- Teacher: `Qwen/Qwen3.8-27B` at
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (BF16).
- Initial draft: `incoai/Qwen3.8-27B-DFlash2` at
  `015e795645c74b1a0eeef3b570031fb62e769bc5` (BF16).
- One NVIDIA A100-SXM4-80GB; Python 3.12.3, torch 2.13.0+cu130,
  CUDA 13.0, Transformers 5.15.1 and safetensors 0.8.0.
- Eight authored training prompts, four validation prompts and four distinct
  native-test prompts. These tiny smoke splits are not a representative corpus.
- Twenty optimizer steps per recipe, batch size one, four accumulated blocks,
  sequence limit 2048, seed 42, learning rate 1e-5 and gradient clipping at 1.
  Validation evaluates eight blocks from four continuations. All four selected
  step 10, the earliest tied checkpoint.
- The separate memory probe has 1,895 prompt tokens and up to 128 generated
  tokens (2,023 combined). It performs one complete eight-row draft block's
  backward pass and optimizer update. It is not a full sequence-sized batch.

All CUDA runs use the combined training stack, rather than each isolated parent
PR head. The recipes and small raw results beside this document preserve that
scope. Training and validation capture tensor payloads were exactly equal
across the four runs. BF16 export checks found changes only in the requested
training scopes; the recall run preserved all selector tensors exactly.

## Results

| Recipe | Steps | Training peak in memory probe | Validation matching proposals, before → after |
| --- | ---: | ---: | ---: |
| Selector continuation | 20 | 10.6 GiB | 3.875 → 3.875 |
| Selector with AF4G64 context/selector QAT | 20 | 12.0 GiB | 3.875 → 3.875 |
| Selector with prefix loss 0.1 and position decay 0.9 | 20 | 10.6 GiB | 3.875 → 3.875 |
| Recall with QAT, prefix loss and position decay | 20 | 16.2 GiB | 3.875 → 3.875 |
| Full draft, plain objective | One preflight update | 40.7 GiB | Not evaluated |

Teacher capture peaked at 51.8 GiB separately from draft optimization.
The measurements are PyTorch peak allocated bytes, not total device memory
requirements. They qualify these contexts and recipes on this A100; they do
not guarantee the same footprint at longer contexts or larger batches.

The recall recipe combines several controls to exercise their interaction. It
cannot isolate a recall improvement from QAT or objective changes. All 80
training steps, validation and finite BF16 exports completed successfully.

The research-only pair-context and nonlinear selectors were also tested on a
real cached teacher continuation with hidden size 5120, rank 256, vocabulary
248320 and top-16 support. Both reproduced baseline scores and greedy paths
exactly at zero initialization, produced finite effective optimizer updates,
preserved the frozen base and reloaded their research artifacts exactly.
Their capability metadata remains `native_compatible: false`. This is a
component test, not end-to-end research-architecture training or native support.

## Native qualification and interpretation

Native probes use the diagnostics implementation at
`b4006b12cdd7e0eff0b759b2f5f1aa9d53e93610` on an Apple M5 Pro, a fixed
`mlx-community/Qwen3.8-27B-4bit` target at
`10c35caafbb80f7dc6a7a432cdd11af10a6d4818`, INT8 KV, FP32 recurrent
state, greedy decoding, thinking disabled, all verification rows and frozen
budget learning. The BF16 training teacher and low-bit native target are
distinct artifacts. Quantizing draft projections during training does not
make their target features and output heads identical.

The JSON evidence records native acceptance separately from frozen BF16 trace
replay. Native mean retained tokens includes the anchor; validation matching
proposals excludes it. Do not compare these two numbers directly.

All four trained exports passed native loading and capture/replay. The complete
native matrix has 540 lane-cycles: 108 per model, with 32 each for code, math and
prose and 12 before termination for JSON. Non-draft input digests match across
all five assemblies. Target output common prefixes agree with the released
draft in every comparison.

| Native prompt | Released | Selector | QAT | Prefix | Recall combination |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code | 7.15625 | 7.03125 | 7.03125 | 7.03125 | 7.03125 |
| Math | 4.84375 | 4.84375 | 4.84375 | 4.84375 | 4.84375 |
| Prose | 2.5 | 2.53125 | 2.53125 | 2.53125 | 2.5 |
| JSON | 7.16667 | 7.16667 | 7.16667 | 7.16667 | 7.16667 |

There is no consistent acceptance gain in these short probes, and no serving
throughput benchmark was run. Cycles within one prompt are correlated and
different acceptance lengths can change later block boundaries. These results
must not be interpreted as a statistical comparison or used to tune on the
held-out test prompts.

The next effectiveness study needs more independent prompts, a frozen final
test, matched target features/quantization, equal training-token budgets and
paired native timing. A wider support, larger selector, wider convolutions or
deeper backbone is not justified by this smoke. Issues #542–#548 retain their
research and deployment gates.

## Reproduction

Check out the training source commit above, install its optional training
dependencies and fetch the two pinned model revisions into separate directories.
With `TARGET`, `DRAFT` and `OUTPUT` pointing to those directories and a new
output location, run each of `baseline`, `qat`, `prefix` and `recall`:

```sh
python -m draft_training train --target "$TARGET" --init-draft "$DRAFT" \
  --data train.jsonl --validation-data validation.jsonl \
  --config baseline.yaml --output "$OUTPUT" --device cuda
```

For the memory probe, substitute `context-probe.jsonl` for training data and add
`--preflight`. Use `full.yaml` only with `--preflight` to reproduce the single
full-draft update recorded here. Outputs must not overlap the source checkpoints.

The original run/selection manifests are retained as produced: their
`native_validation: not run` describes the CUDA trainer's own execution.
Subsequent native evidence is recorded separately, identified by the verified
export hashes. It does not retroactively alter those manifests.

`summary.json` links the measured scopes and exact source revisions. `cuda/`
contains each run's original metrics, preflight, selection and run manifests.
`native/` contains replay reports and compressed captures. Only local machine
paths in capture metadata were replaced with placeholders. `capture_sha256`
records the original trace; `published_capture_sha256` records the decompressed
sanitized trace. Model fingerprints and numerical trace values are preserved.

At the native diagnostics commit above, a published capture can be replayed:

```sh
gzip -dc native/qat-prose.jsonl.gz > /tmp/qat-prose.jsonl
python -m dev.benchmarks.draft_replay /tmp/qat-prose.jsonl \
  --output /tmp/qat-prose.report.json
```

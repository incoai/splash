# Experimental DFlash2 fine-tuning

The optional source-checkout command fine-tunes an existing Qwen3.8-27B
DFlash2 draft. It exports the BF16 safetensors format Splash already accepts
through `--draft-model`. It does not add support for other target families or
reproduce Inco's undisclosed original training recipe. Related: [issue #435](https://github.com/incoai/splash/issues/435).

The tensor implementation is checked against Z Lab's public DFlash code at
`07ebd93db9f472af339b644bb70221ad8428328a`. The training objective is
experimental: unary cross-entropy plus selector cross-entropy over the strict
unary top-16 candidates. Missing selector targets are skipped and coverage is
reported. Candidate selection and tie-breaking follow Splash's serving contract.

## Current-main integration

Rechecked against main `1482f6bbc2abf12375332725302019594150b6cb` on
October 21, 2026. Main's adaptive MoE verify-row policy (#487) controls native
execution; it does not add training commands or an acceptance-confidence head
to this trainer. The trainer retains its existing, deliberately restricted
Qwen3.8-27B adapter and BF16 export schema. Main now derives the draft context
projection width from its target-layer layout; the supported five target
layers still imply the same `fc.weight` shape.

The source launcher, CI and Python checks are integrated with the current
server/options layout. No native runtime or shader change is part of this PR.
Small-model/metadata checks do not replace production CUDA training,
real-checkpoint native loading or native acceptance/throughput qualification.
For deployment comparisons, record the current all/auto verify-row setting and
the actual target/draft revisions. Historical pre-merge test and timing
results below or in the PR description keep their original source scope.

## Setup

Use Python 3.12–3.14 and a separate training environment. CUDA is the default
for real training. CPU execution supports small reference diagnostics; large
BF16 checkpoints still need memory for all their weights. Hardware capacity
must be measured with preflight; this implementation has not yet completed a
production Qwen3.8-27B CUDA run or a trained-checkpoint native speed benchmark.

```bash
python3.12 -m venv /path/to/training-env
. /path/to/training-env/bin/activate
python -m pip install -r training/requirements.txt
python -m draft_training --help
```

Run from the checkout root. The `./splash train-draft` source wrapper prefers
`SPLASH_TRAINING_PYTHON`, then the activated `VIRTUAL_ENV` interpreter. Normal
serving retains its existing interpreter selection. Packaged Splash releases
do not advertise or include this optional training command.

## Checkpoints and data

`--target` requires a local, unquantized BF16 Transformers checkpoint with
`config.json`, tokenizer files, a chat template, and `model.safetensors` or
indexed safetensors shards. The first adapter uses the `qwen3_5` configuration
and `Qwen3_5ForConditionalGeneration` architecture for the supported
Qwen3.8-27B family. GGUF, MLX, FP8, and NVFP4 training targets are rejected.
`--init-draft` requires the matching released BF16 DFlash2 checkpoint.

Both inputs stay local; commands never download checkpoints, provision a
machine, upload weights, or start a paid job. Input files are fingerprinted
for capture reuse and resume. Hashing a full target can take time.

Provide separate training and validation JSONL files:

```json
{"messages":[{"role":"user","content":"Write a Python CSV reader."}]}
```

Only text `system`, `user`, and `assistant` messages are supported initially.
Each user turn yields a prompt prefix. The default `target-generated` mode
replaces its continuation with the frozen teacher's greedy generation and
never includes later source turns in that continuation. `supplied-answers`
uses the assistant turn immediately following each user turn; it is a
separate experiment and may reduce agreement with the teacher.

Keep every source conversation and derived prompt in one split. Duplicate
normalized prompt prefixes or an explicit shared `source_id` across splits are
rejected. Split before generating responses, and reserve a third test file
for final evaluation. Training cannot accept a `--test-data` argument.

## Inspect and probe

```bash
python -m draft_training train \
  --target /models/target-transformers-bf16 \
  --init-draft /models/released-dflash2 \
  --data train.jsonl --validation-data validation.jsonl \
  --config training/qwen3_8_27b.yaml --output runs/my-draft \
  --dry-run
```

Dry-run validates headers, family settings, recipe, and split isolation
without loading tensor data or creating output. It does not establish tensor
finiteness or native loading. Replace `--dry-run` with `--preflight` to generate
one continuation, capture selected decoder outputs, and execute a gradient
and optimizer probe. It reports separate CUDA capture and training peaks and
writes no run. The first probe must contain a full assistant block.

A query block contains one known anchor and seven masked proposals. Its
captured target context ends strictly before the anchor and respects each
query's 2048-token window. Query attention is noncausal inside that single
block; future target features and other blocks are never visible.

## Capture, train, and resume

Run the previous command without either probe flag to train. `capture` accepts
the same arguments when feature preparation should be completed separately;
continue that matching run with `train ... --resume` afterward.

The example YAML starts with selector-only training. Set `trainable: all`
for an independently evaluated full-draft experiment. AdamW uses FP32 trainable
master parameters and moments with BF16 autocast for forward matmuls. The
learning rate is constant; selector loss weight ramps over the configured
optimizer steps. The effective streamed batch is `batch_size` times
`gradient_accumulation`, processed one isolated block at a time.

The target is frozen throughout. After capture, its decoder and vision weights
are released; training retains only the frozen embedding and output head.
Gradients through the output head still reach the draft. Capture uses bounded
continuation shards; BF16 selected features alone occupy about 51.2 GB per
million tokens for this family.

A run contains:

```text
run.json                    # inputs, recipe, implementation hashes, versions
features/train/             # safetensors shards and capture manifest
features/validation/
baseline-validation.json    # released draft on validation traces
metrics.jsonl               # optimizer steps and validation metrics
checkpoints/step-*/          # model, optimizer/RNG state, cursor, progress
export/                     # selected BF16 serving checkpoint
selection.json              # frozen selection and export digests
```

Use the identical arguments and recipe with `--resume` after interruption.
The command rejects changes to inputs, implementation, library versions, device
context, or settings before allocating the teacher model.
Resume restores the latest committed checkpoint, optimizer, RNG, and absolute
block cursor in a deterministically shuffled epoch. Nonempty run directories
require explicit resume. Exports use new directories and are published only
after finite-value and format checks. If interruption occurs after export
publication but before `selection.json` is written, resume verifies every
exported tensor and its provenance against the selected checkpoint before
finishing the manifest. It never replaces a published export. An independent
`export` subcommand can export a local trainer checkpoint.

## Evaluation and native serving

Checkpoint selection uses validation trace replay only. It maximizes mean
consecutive matching proposals; the earlier checkpoint wins ties. Validation
uses the BF16 weights that will be exported and then restores the FP32 masters
before training continues. The export,
recipe, input identities, implementation, library versions, device context,
and split fingerprints are frozen before final tests. Evaluation checks these
before opening the test data.

```bash
python -m draft_training evaluate \
  --target /models/target-transformers-bf16 \
  --draft runs/my-draft/export \
  --baseline-draft /models/released-dflash2 \
  --test-data test.jsonl \
  --selection-manifest runs/my-draft/selection.json \
  --output runs/my-draft/final-test
```

This compares original and trained drafts on the same untouched test traces.
It reports unary CE, top-16 coverage, teacher-forced selector accuracy, and a
histogram of consecutive matches along self-conditioned greedy paths. Anchor
and bonus tokens are excluded. The configured block limit bounds evaluation
and is distributed across scorable continuations using the frozen seed.
Answers shorter than one full block are skipped, and unused per-continuation
quotas are redistributed until the budget or available blocks are exhausted.
Coverage counts are included in the report. Supplied-answer replay measures
matches to those answers, not acceptance by the verifier.

Replay is not a native throughput measurement or sampled verification test.
Do not tune on final results; use a new untouched test split if those results
influence another training experiment. Final reports cannot be overwritten.

Transfer the export to a supported Mac and use a compatible deployment target:

```bash
./splash serve --model OWNER/COMPATIBLE-TARGET \
  --draft-model /path/to/export
```

The CUDA BF16 teacher and Splash's quantized deployment target are distinct
artifacts. Record the deployment revision and tokenizer compatibility in the
native benchmark results. Check actual weight preparation/loading and compare
acceptance, decode throughput, and latency against the released draft using
identical target, sampling, cache, and memory settings with alternating runs.
Exports remain experimental until these real-model gates pass. A compatible
checkpoint or reduced loss alone does not establish a speedup.

## Evidence required before leaving draft

- Complete a CUDA preflight and short training run with the real BF16 target
  and released initial draft. Record GPU, peak memory, step time, exact input
  revisions, recipe, and split fingerprints.
- Prepare and load both an unchanged round-trip export and the trained export
  in native Splash on a supported Mac. Verify the deployment tokenizer and
  token IDs against the training teacher.
- Compare the released and trained drafts with alternating native runs on the
  same deployment target. Report acceptance, decode throughput, and latency,
  with sampling, cache, memory settings, hardware, and revisions held fixed.

The CPU fixture tests establish implementation behavior at small dimensions.
They do not satisfy these hardware gates. Fresh initialization, additional
target families, and a Mac training backend are outside this PR's scope.

## Development tests

```bash
python -m unittest discover -s dev/tests/training -t . -v
# Optional numerical comparison with a checkout of the pinned reference:
DFLASH_REFERENCE=/path/to/dflash/dflash/model.py \
  python -m unittest dev.tests.training.test_model -v
make check
```

Tensor tests skip when optional dependencies are absent. Metadata and launcher
checks still run in the ordinary suite. The separate Linux CI job installs
CPU training dependencies and tests the pinned reference without model downloads.

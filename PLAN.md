# DFlash2 draft training command

Status: initial implementation under test. Small-model CPU training,
reference parity, capture, resume, and export tests pass. Production CUDA
training, real-checkpoint native loading, and speed/acceptance benchmarks are
still required before declaring the plan complete.

Created: 2026-10-20. Current-main integration reviewed: 2026-10-21.
The October 20 validation record below is historical; the PR description
records the current-main rerun separately.
Related: [Splash issue #435](https://github.com/incoai/splash/issues/435).

## Objective

Build a command that fine-tunes a DFlash2 draft and exports a checkpoint
Splash can load through its existing `--draft-model` option. Start with the
existing Qwen3.8-27B draft and a user's text conversation dataset. Add
Qwen3.6-35B-A3B after the first family passes the complete flow.

Success means reproducible training, a compatible export, and measured
speculative decoding behavior. A checkpoint that loads is an intermediate
milestone; it is not evidence of improved acceptance or throughput.

## Implementation and validation record

The source-only wrapper and independent module implement metadata inspection,
explicit device preflight, target capture, streamed training, exact cursor/RNG
resume, checkpoint selection, BF16 export, and separate final greedy replay.
The production entry point accepts only the existing Qwen3.8-27B family; tiny
fixtures exercise the same components without downloading production weights.

Verified locally on 2026-10-20:

- 35 optional training tests pass on Python 3.12.11 with PyTorch 2.13.0 and
  Transformers 5.15.1, including the actual tiny Qwen3.5 target adapter and
  pinned Z Lab reference forward/selector parity.
- Tests cover future-token isolation through the loss, FP32 rotary frequencies
  at long positions, FP32 master updates, BF16 export, split isolation,
  validation budget distribution, and interrupted resume across shuffled epochs.
- The repository Python suite passed 1,298 tests with 21 expected skips under
  the ordinary environment without PyTorch. Ruff and architecture checks pass.
- The aggregate native check stopped at an unchanged Neural Engine timing
  assertion: the GPU layer measured 151 ms against a required <100 ms. An
  isolated probe using baseline test code reproduced the assertion at 116 ms.
  Remaining Metal tests are being checked separately. Do not record the
  aggregate `make check` as passing or relax its timing threshold for this PR.
- Final evaluation now checks implementation, library versions, and device
  context against the frozen selection before reading test examples.

Remaining gates: a production CUDA run with the actual BF16 target and released
initial draft, unchanged-checkpoint native loading, trained-checkpoint native
loading, deployment tokenizer/revision verification, and native acceptance and
throughput comparison. These require real checkpoints and hardware, and remain
unchecked below. The initial PR must remain a draft until the required evidence
is available. Setup and measured scope are described in
[the training guide](docs/dflash2-training.md).

## Scope and assumptions

- First version: text conversations, one supported target family, and an
  existing draft as initial weights. Fresh initialization follows later.
- Use a separate optional PyTorch training environment, initially on Linux
  with CUDA. Keep training dependencies out of normal Splash installation.
- The Linux entry point must work without compiling Splash or importing its
  Apple-specific runtime. Validate the export and benchmark on a supported Mac.
- Train where the target, data, and accelerator reside. No automatic cloud
  provisioning, paid jobs, remote execution, or model uploads.
- Accept local target and draft directories first. Add Hub downloads with
  pinned revisions later. The initial training target must be a Transformers
  BF16 safetensors checkpoint; GGUF, MLX-quantized, FP8, and NVFP4 target
  loading are outside the initial CUDA adapter's contract.
- Use target-generated assistant responses as the baseline supervision.
  Supplied-answer training is an explicit separate experiment, since dataset
  answers may differ from the frozen target's predictions.
- Record the training teacher and deployed target separately, including their
  formats, precision, revisions, and tokenizer identities. Verify that they
  describe the intended model family and token mapping; do not assume every
  deployment quantization or fine-tune behaves like the training teacher.
- Hardware, dataset, and training budget are not specified. Measure memory and
  step time before choosing production settings or promising completion times.
- Vision training and arbitrary target architectures are later work. A new
  draft does not by itself add support for a new target to Splash.

## Evidence and compatibility contract

Splash exposes the inference architecture and checkpoint contract. It does
not establish the original training loss, data mixture, optimizer schedule,
initialization procedure, or compute budget. Our recipe must be explicitly
experimental and evaluated on its own results.

| Repository source | What it establishes |
| --- | --- |
| `runtime/model/DraftCheckpoint.cpp` | Required tensor names, BF16 dtypes, shapes, and preparation into runtime images |
| `runtime/model/DFlashDraft.cpp` | Context projection, normalization, attention, convolution, MLP, and selection flow |
| `runtime/metal/kernels/decode/draft.metal` | Dynamic convolution and draft attention semantics |
| `runtime/ops/DraftSelector.cpp` and selector Metal kernels | Top-16 candidates, transition scores, and proposal path behavior |
| `runtime/model/ModelDescriptor.mm`, `validateDraftConfig` | Native configuration acceptance rules |
| `runtime/model/Qwen3_8.hpp` and `Qwen3_6Moe.hpp` | Family dimensions and target capture layers |
| `runtime/metal/abi/ExecutionGeometry.h` and `DraftAttention.h` | Compiled block, selector, and attention geometry |
| `install/upstream.py`, `_draft_files` | Local draft discovery and configuration checks |
| `dev/tests/fixtures/model-configs/*/draft/config.json` | Reference configurations |

| Property | Qwen3.8-27B | Qwen3.6-35B-A3B |
| --- | --- | --- |
| Draft layers | 5 | 6 |
| Draft hidden width | 5120 | 2048 |
| Draft MLP intermediate width | 17408 | 6144 |
| Target capture layer IDs | `[5, 19, 33, 47, 61]` | `[1, 6, 11, 16, 22, 27, 32, 37]` |
| Concatenated target feature width | 25600 | 16384 |
| Mask token ID | 248070 | 248077 |

Both layouts require vocabulary size 248320, block size 8 (one anchor and
seven draft proposals), sliding window 2048, convolution kernel size 2,
convolution group size 16, selector rank 256, and top-16 candidates. Draft
attention uses 32 query heads, 8 KV heads, and head dimension 128. Match each
reference config's remaining settings, including noncausal attention and RoPE.
These are current runtime requirements, not first-version tuning knobs.

## Training target input contract

`--target` names a local Hugging Face Transformers directory with `config.json`,
the target tokenizer and chat template, and BF16 safetensors weights, either
one file or indexed shards. The first adapter supports the Qwen3.8-27B family
through the `qwen3_5` configuration and `Qwen3_5ForConditionalGeneration`
architecture represented by the repository's reference target config. Run
text-only forwards and verify the actual pinned Transformers class, language
decoder capture points, embedding access, and output head in Phase 0. Do not
infer compatibility from a repository name or substitute the draft's `qwen3`
architecture for the target model's class.

Reject unsupported serialization, quantization metadata, architecture,
dimensions, tensor dtypes, or missing tokenizer/template files during metadata
preflight, before feature capture or training allocation. `--preflight` then
loads a tiny text sample and checks capture shapes, layer ordering, and frozen
embedding/output-head access on the selected CUDA device. Record the exact
Transformers and PyTorch versions in the resolved run configuration.

The deployment target may be a compatible GGUF or MLX variant served by
Splash. It is a distinct artifact from the BF16 teacher. Provenance must map
the teacher to the deployment checkpoint and verify vocabulary/token IDs and
special tokens; native evaluation uses the actual deployment variant.

## Command interface

The initial implementation exposes these commands; see [setup and limitations](docs/dflash2-training.md).

```bash
# Run on a CUDA training host with an optional training environment.
python -m draft_training train \
  --target /models/target-transformers-bf16 \
  --init-draft /models/existing-dflash2 \
  --data ./conversations.jsonl \
  --validation-data ./validation.jsonl \
  --supervision target-generated \
  --config ./training/qwen3_8_27b.yaml \
  --output ./runs/my-dflash2

# Convenience wrapper using the same arguments on a configured training host.
./splash train-draft \
  --target /models/target-transformers-bf16 \
  --init-draft /models/existing-dflash2 \
  --data ./conversations.jsonl \
  --validation-data ./validation.jsonl \
  --supervision target-generated \
  --config ./training/qwen3_8_27b.yaml \
  --output ./runs/my-dflash2

# Validate export metadata and tensor compatibility without training.
python -m draft_training validate \
  --target /models/target-transformers-bf16 \
  --draft ./runs/my-dflash2/export

# Run final test evaluation only after checkpoint and settings are frozen.
python -m draft_training evaluate \
  --target /models/target-transformers-bf16 \
  --draft ./runs/my-dflash2/export \
  --baseline-draft /models/existing-dflash2 \
  --test-data ./test.jsonl \
  --selection-manifest ./runs/my-dflash2/selection.json \
  --output ./runs/my-dflash2/final-test

# After transferring the export to a supported Mac, validate native serving.
./splash serve \
  --model OWNER/COMPATIBLE-TARGET \
  --draft-model ./runs/my-dflash2/export
```

The training command orchestrates preflight, capture, optimization, export,
and validation-set evaluation. Final test evaluation is a separate operation;
test data is not an argument to the training command. Also expose module
subcommands `capture`, `export`, and `evaluate` to reuse expensive stages.
Provide `--dry-run` for metadata-only
planning, `--preflight` for a measured short device probe, and `--resume` for
continuing an interrupted run.

YAML holds seed, sequence length, batch size, gradient accumulation, maximum
steps, optimizer settings, precision, checkpoint intervals, loss weights, and
teacher generation settings. `target-generated` is the baseline supervision
mode; `supplied-answers` must be explicitly selected and separately reported.
CLI paths take precedence; save the resolved configuration. Reject nonempty
output directories unless explicitly resuming a matching run. Keep training
checkpoints and serving exports distinct.

## Implementation layout

```text
draft_training/
  __main__.py          # Linux-compatible command dispatch
  config.py            # Validated run settings and family contracts
  model.py             # Differentiable reference model
  capture.py           # Frozen target feature extraction
  data.py              # Conversation parsing, masks, and blocks
  losses.py            # Unary and selector objectives
  trainer.py           # Optimization, checkpoints, and resume
  export.py            # Serving checkpoint conversion and validation
  evaluate.py          # Validation metrics and separate final test runs
training/
  requirements.txt     # Pinned optional environment
  qwen3_8_27b.yaml     # First family recipe
dev/tests/training/    # Focused CPU and optional accelerator tests
```

Add lazy dispatch in `install/launcher.py` after the independent module works.
The launcher currently assumes every command has a port: scope port handling
and server validation to applicable commands. Help, installation, and serving
must continue to work without PyTorch installed.

## Phase 0: Freeze reference semantics and the export contract

- [ ] Record the Splash commit and immutable upstream implementation revisions.
- [ ] Enforce the BF16 Transformers target input contract. Verify the first
  target adapter's text-only forward, selected hidden states, shared embeddings,
  output head, and pinned library versions with a short device probe.
- [ ] Record separate teacher/deployment identities and check tokenizer and
  special-token compatibility before generating supervision.
- [ ] Compare [Z Lab DFlash](https://github.com/z-lab/dflash) and
  [TorchSpec's DFlash2 implementation](https://github.com/lightseekorg/TorchSpec/blob/main/torchspec/models/draft/dflash2.py)
  with native operations. Select reuse or a small reference implementation
  based on that comparison; preserve licenses and notices for reused code.
- [ ] Verify exporter mappings, including context projection to `fc`, context
  norm to `hidden_norm`, and final norm to `norm`.
- [ ] Resolve target layer IDs versus Transformers hidden-state tuple indices;
  verify the embedding offset rather than assuming the indices coincide.
- [ ] Specify feature concatenation order, anchor/label alignment, attention
  visibility, convolution resets at block boundaries, and shared frozen target
  embeddings and output head.
- [ ] Define the visibility matrix for a block anchored at position `a`:
  captured target features may come only from positions strictly before `a`,
  subject to each query row's sliding window. The query block contains the
  known anchor token followed by seven mask tokens and permits noncausal
  attention within that block. Other blocks' query rows and future target
  features must be invisible. Ground-truth continuation tokens are labels,
  never query inputs or visible target features for that block.
- [ ] Implement strict configuration and safetensors inspection for the first
  family, with actionable errors for unsupported shapes or missing tensors.

Gate: inspect a released draft, verify target capture on a small sample, and
reject incompatible target formats, configurations, tokenizer mappings, and
tensor fixtures before any optimization run.

## Phase 1: Reproduce the forward pass and unchanged export

- [ ] Implement or adapt a differentiable model and load the released weights.
- [ ] Freeze the target, including reused embeddings and output projection.
- [ ] Match normalization, context projection, Q/K normalization, RoPE, sliding
  context, noncausal query-block visibility, convolutions, residuals, and selector.
- [ ] Compare intermediate tensors and proposal tokens with the pinned public
  BF16 reference on short prefixes, window boundaries, and multiple blocks.
- [ ] Add future-token leakage tests: keep a block's prefix and anchor fixed,
  then independently change future tokens, future cached target features, and
  later packed blocks. Its unary logits and selector scores must remain
  unchanged. Test at short prefixes and sliding-window boundaries; include a
  deliberately unmasked implementation to prove the checks detect leakage.
- [ ] Justify numerical tolerances from dtype and operations. Separate reference
  arithmetic differences from native quantization effects; do not demand
  bitwise equality between BF16 and quantized implementations or hide a
  mismatch by loosening tolerances.
- [ ] Export the original weights unchanged, compare tensor values and config,
  and load that export in Splash before implementing full-scale training.

Gate: reference parity and future-token isolation are established, and the
unchanged export passes native loading. This isolates architecture, leakage,
and conversion errors from training errors.

## Phase 2: Prepare data and capture target features

- [ ] Define validated JSONL records containing `messages` with role/content
  pairs. Use the target tokenizer and chat template.
- [ ] Preserve assistant loss masks; exclude padding and prompt-only positions.
  Prevent blocks and convolutions from crossing conversation boundaries.
- [ ] Establish distinct training, validation, and final test splits before
  generating responses or sampling blocks. Group by source conversation or
  prompt identity so derived examples stay together. Check normalized prompt
  and conversation overlap and record split manifests and data fingerprints.
  A separate manifest check may inspect test identities for overlap; the
  training and model-selection paths must not load test examples or metrics.
- [ ] Default to target-generated assistant responses with the frozen BF16
  teacher and recorded generation settings. Start with greedy generation for
  the first recipe. Build prompt prefixes ending at the assistant-generation
  boundary and generate replacement continuations; preserve source grouping
  and never use original future turns as input to those continuations.
- [ ] Treat `supplied-answers` as a separate, explicitly selected experiment.
  Measure its agreement with teacher predictions and native acceptance against
  the target-generated baseline; do not equate dataset likelihood with target
  imitation. Record supervision mode and generation provenance in every shard.
- [ ] Capture only selected hidden states with the target frozen. Save token
  IDs, masks, boundaries, positions, and feature ordering.
- [ ] Apply the per-anchor visibility matrix even when a shard contains an
  entire conversation's captured states. Re-run future-feature perturbation
  checks through the data loader and loss path, not just the reference model.
- [ ] Write bounded, restartable shards and a manifest binding them to target
  revision/precision, tokenizer, template, capture layers, and configuration.
- [ ] Validate shard integrity and compatibility before reuse. Avoid storing
  full-vocabulary logits per token; compute vocabulary losses in bounded chunks.

Estimate storage before capture. Selected BF16 features alone require
`tokens * concatenated_feature_width * 2` bytes: about 51.2 GB per million
tokens for the 27B layout, before other data and checkpoints. Measure capture
memory, disk throughput, and time on a small sample before selecting shard
sizes and a production dataset size.

Gate: tiny-data capture/reload preserves positions, boundaries, assistant masks,
split membership, and per-anchor isolation. Reject overlapping splits, stale
or incompatible features, and mismatched supervision provenance.

## Phase 3: Fine-tune with an explicit experimental objective

- [ ] Start with cross-entropy against the teacher-generated assistant tokens
  on masked, anchored blocks. Teacher-forced labels and selector predecessors
  must come from that same generated sequence. Preserve the full vocabulary
  and serving architecture; supplied-answer objectives are separate ablations.
- [ ] Add selector cross-entropy over strict unary top-16 candidates, conditioned
  on the correct predecessor during training. Skip selector loss when the
  target is absent and report candidate coverage. Never insert the correct
  token into inference or evaluation candidates.
- [ ] Document valid-token denominators and handle batches without valid
  selector targets. Report unary and selector losses separately.
- [ ] Run a selector-only diagnostic, then full-draft training if validation
  results support it. Use validation only for hyperparameters, early stopping,
  checkpoint selection, and native acceptance experiments. Record trainable
  parameters and whether selector gradients reach the backbone and unary scores.
- [ ] Implement loss-weight warmup, accumulation, clipping, finite-loss checks,
  and optimizer-step accounting.
- [ ] Save model, optimizer, scheduler, RNG state, step, data cursor, and run
  identity for resume. Distinguish exact resume from a new warm-start run.
- [ ] Select the final checkpoint using a documented validation criterion.
  Freeze its weight/config digests, recipe, teacher/deployment identities,
  generation and benchmark settings, and validation summary in `selection.json`
  before granting the final evaluator access to test examples.
- [ ] Measure peak device memory and step time before a full run. Set a maximum
  step count and select hardware/budget from the measured probe.
- [ ] Consider teacher-distribution distillation only as a follow-up experiment;
  specify direction, temperature, and approximations if introduced.

The loss is our starting hypothesis. [Speculators' DFlash2 documentation](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dflash2.md)
also identifies its objective as experimental because the original public
inference implementation does not disclose its training objective.

Gate: a short run updates intended parameters with finite gradients, learns a
diagnostic sample, resumes correctly, and reports validation coverage, path
behavior, and acceptance against the released draft. A selection manifest is
written before final testing. Reduced training loss alone does not establish
utility.

## Phase 4: Export and verify native compatibility

- [ ] Export `config.json`, BF16 safetensors, and an index if sharding is needed.
  Match `DraftCheckpoint.cpp` exactly; omit optimizer state from serving files.
- [ ] Check required names, shapes, dtypes, finite values, shard references,
  capture layers, and family configuration before declaring export ready.
- [ ] Write into a temporary directory and publish only after validation; do
  not overwrite another run or a released baseline checkpoint.
- [ ] Include revisions, target/draft identities, data fingerprints, seed,
  resolved recipe, supervision provenance, split manifests, training step,
  validation results, and the frozen selection manifest in run provenance.
- [ ] Exercise native configuration validation, weight preparation, and actual
  serving on a supported Mac. Python inspection does not replace native checks.

Gate: the trained draft loads through `--draft-model` and serves valid output
without changing the existing runtime geometry or checkpoint layout.

## Phase 5: Evaluate utility and integrate the command

- [ ] Verify that the export and evaluation settings match `selection.json`
  before opening the final test set. Evaluate the released baseline and frozen
  trained draft on the same untouched test prompts. Do not tune, choose another
  checkpoint, or early-stop from test results. If test findings drive changes,
  report the prior run as exploratory and obtain a new untouched test split
  before making a fresh final performance claim.
- [ ] Compare released and trained drafts with identical target, precision,
  tokenizer, prompts, sampling settings, memory budget, and cache conditions.
- [ ] Report unary top-16 recall, selector accuracy with the correct predecessor,
  and actual self-conditioned greedy paths separately.
- [ ] Report mean and distribution of accepted draft proposals per verification
  step. Keep anchor and bonus tokens separate from accepted proposals.
- [ ] Measure throughput and latency with repeated alternating runs on final
  test coding tasks and general prompts, recording domain regressions too.
  Keep validation benchmark results clearly separate from final test results.
- [ ] Compare against the established verifier baseline, accounting for known
  target-only versus batched numerical differences. Check greedy behavior and
  sampled verification separately; shared-seed text equality is not a test of
  sampling correctness.
- [ ] Report exports as experimental, regressed, or validated for the measured
  target/hardware scope. Preserve the original default draft.
- [ ] Add lazy launcher dispatch, dependency messages, setup documentation, and
  Linux-to-Mac export transfer instructions.

Gate: reproduce the entire flow and its measurements. Claim improvement only
when untouched final test results support it. The command may work correctly
while a particular trained draft fails to outperform the released baseline.

## Testing and validation

Use focused CPU tests for contract rejection, block/mask alignment, selector
loss edge cases, split overlap, selection-manifest enforcement, export
round-trips, and checkpoint resume. Include model-level and data-loader/loss
future-token leakage tests with a failing unmasked control. Verify the trainer
cannot load final test data through its normal arguments or selection path.
Use tiny accelerator models for gradient and reference parity tests, including
gradient flow through the frozen output head to the draft. Keep large downloads
and production training outside default CI; real-checkpoint checks are optional
and explicitly invoked.

When editing the launcher, verify help and existing serve/client commands
without training dependencies. Run repository checks appropriate to touched
code, including `make check` on a supported Mac for integration changes.
Record commands, environment, and outcomes in the implementation PR.

## Risks and unresolved inputs

| Risk or unknown | Resolution before proceeding |
| --- | --- |
| Original objective and data are unknown | Version an experimental recipe and compare with the released draft |
| Hidden-state indexing or visibility mismatch | Establish reference traces before training |
| Future target features leak into masked blocks | Enforce per-anchor visibility and pass perturbation tests through the model and loss path |
| Upstream config/export differences | Pin revisions and round-trip unchanged weights |
| GPU memory and feature storage | Measure a short run; bound shards, sequence lengths, and vocabulary projections |
| Teacher forcing hides selector errors | Evaluate self-conditioned paths and actual native acceptance |
| Supplied answers disagree with the teacher | Use target-generated baseline supervision; measure supplied-answer training separately |
| Validation choices contaminate final claims | Freeze checkpoint/settings before final testing; keep a separate untouched test split |
| Training teacher differs from deployment quantization | Record variants and evaluate the deployed target |
| Unsupported training checkpoint format or target capture | Reject unsupported formats early and verify the pinned target adapter in Phase 0 |
| New target family requested | Plan runtime support and draft training as separate deliverables |

Before the first substantial training run, select CUDA hardware, maximum
runtime, dataset, target revision, and initial draft revision. These remain
open inputs, not reasons to block the local implementation and tiny tests.
Also select the deployment checkpoint, teacher generation settings, split
manifests, and validation selection criterion before the substantial run.

## Later milestones

1. Add Qwen3.6-35B-A3B with its own reference parity and native serving gates.
2. Add pinned Hub downloads and reusable family recipes.
3. Train from fresh initialization after validating the objective and pipeline.
   Verify identity convolution and neutral selector initialization against the
   reference; measure quality and compute against warm-starting.
4. Evaluate a Mac-native training backend separately after the CUDA flow works.
   Measure feasibility before promising large-model training on a given Mac.
5. Add other families alongside their capture adapters, checkpoint contracts,
   and any necessary Splash runtime support.

## Completion checklist

- [ ] Independent Linux module and lazy Splash wrapper work.
- [ ] One family completes capture, fine-tuning, resume, export, evaluation,
  and native serving.
- [ ] Optional dependencies, reproduction steps, and provenance are documented.
- [ ] The export passes tensor/config checks and native loading.
- [ ] The BF16 target input contract and teacher/deployment mapping are enforced.
- [ ] Future-token leakage checks pass for the model, data loader, and loss path.
- [ ] Target-generated supervision and disjoint train/validation/test splits
  are recorded, and final testing follows a frozen selection manifest.
- [ ] Original-versus-trained acceptance and speed results include regressions,
  device details, and precision limitations.
- [ ] Existing Splash installation and serving checks remain passing.

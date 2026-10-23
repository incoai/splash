# Train a DFlash2 draft on your own data

This workflow continues released Qwen3.8-27B or Qwen3.6-35B-A3B DFlash2 drafts using your
text dataset. It includes data preparation, validation, teacher capture,
optimization, interruption recovery, BF16 export and held-out comparison.
It supports selector-only or full-draft training against a frozen dense or MoE
teacher. Other model families and
training a new architecture from scratch are outside this first workflow.

Training produces a candidate; whether it improves acceptance depends on your
data, recipe and deployment. The selection and final reports compare it with
the released draft. Keep the released checkpoint when your candidate does not
meet your validation and native deployment criteria.

## 1. Set up the training host and local checkpoints

Use a source checkout containing `draft_training`, Python 3.12–3.14, a CUDA
host supporting BF16 and sufficient CPU RAM, GPU memory and disk. Training
does not require building the Apple Metal runtime. Native deployment testing
later requires a supported Mac. #499 itself supports CUDA and CPU diagnostics;
experimental MPS training is introduced separately in #537.

For Qwen3.8-27B, one A100-SXM4-80GB completed the published combined-stack smoke. Teacher
capture peaked at 51.8 GiB allocated, selector optimization at 10.6 GiB and
one full-draft preflight update at 40.7 GiB, measured separately near 2K context.
Those are measured allocated bytes, rather than total hardware requirements.
Use preflight on your host and recipe. Feature caches alone use about 51.2 GB
per million captured tokens for that family, plus model weights and optimizer
checkpoints. Qwen3.6-35B-A3B caches use about 32.8 GB per million tokens. Its
teacher still holds all experts in memory even though only a subset is active
per token. Do not infer its full memory requirement from the 3B active count.
The published 27B GPU measurements do not qualify the full-size MoE workflow.

```bash
python3.12 -m venv /path/to/training-env
. /path/to/training-env/bin/activate
python -m pip install -r training/requirements.txt
python -m draft_training --help
```

Obtain these checkpoints locally before training. For example, the installed
Hugging Face CLI can download the revisions used in the published smoke:

```bash
hf download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir /models/qwen38-teacher
hf download incoai/Qwen3.8-27B-DFlash2 \
  --revision 015e795645c74b1a0eeef3b570031fb62e769bc5 \
  --local-dir /models/qwen38-draft
```

The teacher must be the unquantized BF16 Transformers checkpoint, including
its tokenizer and chat template. A GGUF or 4-bit serving package cannot be
used as the training teacher. Commands below use only local checkpoints;
they do not rent a GPU, download models or upload data automatically.

For **Qwen3.6-35B-A3B MoE**, use the matching local teacher and released draft:

```bash
hf download Qwen/Qwen3.6-35B-A3B \
  --revision 995ad96eacd98c81ed38be0c5b274b04031597b0 \
  --local-dir /models/qwen36-teacher
hf download incoai/Qwen3.6-35B-A3B-DFlash2 \
  --revision 51ef7b6923ad6c14cb1bb41c37a9041446496ab6 \
  --local-dir /models/qwen36-draft
```

Use these two paths in place of the `qwen38` paths in every command below,
and copy `training/qwen3_6_35b_a3b.yaml` or
`training/qwen3_6_35b_a3b_full.yaml` as your recipe. Family detection is automatic
from checkpoint metadata; mixing the dense and MoE checkpoints is rejected.
The full-size MoE path has metadata/tokenizer validation and real small-MoE
pipeline tests, while real-checkpoint CUDA and native qualification are still
required. There is no MoE acceptance or serving speed claim.

## 2. Prepare your JSONL dataset

Use one conversation per line in `corpus.jsonl`:

```json
{"source_id":"repository-or-conversation-a","messages":[{"role":"user","content":"Write a Python CSV reader."},{"role":"assistant","content":"A complete answer goes here."}]}
{"source_id":"repository-or-conversation-b","prompt":"Explain a SQL left join.","response":"A complete answer goes here."}
{"source_id":"repository-or-conversation-c","prompt":"Write a JSON schema for an invoice."}
```

`prepare-data` accepts `messages`, `prompt` with `response` or `completion`,
and prompt-only records. It writes canonical text conversations. Prompt-only
records work with the default `target-generated` supervision: the frozen
teacher generates the continuation, and source answers are not used as labels.
For `supplied-answers`, provide an assistant answer immediately after each
user turn you want to train. All messages must contain nonempty text; tools,
images and other structured content are not accepted by this adapter.

```bash
python -m draft_training prepare-data \
  --data corpus.jsonl --output data/my-domain \
  --validation-fraction 0.1 --test-fraction 0.1 --seed 42
```

This creates `train.jsonl`, `validation.jsonl`, `test.jsonl` and
`preparation.json`. At least three independent groups are required. Real
training needs substantially more representative data than this three-line
format example. Fraction targets apply to groups; actual conversation counts
are printed and recorded. Preparation is deterministic for the same input
and seed and refuses an existing output directory.

Records sharing a `source_id` or normalized prompt prefix stay together,
including transitive connections. Set `source_id` to your repository,
conversation or other leakage group before splitting. The tool cannot infer
unlabelled relationships or semantic duplicates. You can also provide your
own already isolated `messages` JSONL splits directly to training.

## 3. Choose a recipe and check the data before allocating weights

```bash
cp training/qwen3_8_27b.yaml my-domain.yaml
# For full-draft continuation instead:
# cp training/qwen3_8_27b_full.yaml my-domain.yaml
# For the MoE teacher, use qwen3_6_35b_a3b.yaml or qwen3_6_35b_a3b_full.yaml.
```

Edit `my-domain.yaml` before starting. Choose the training budget, learning
rate, sequence limit, generated continuation length and validation block
budget. `batch_size * gradient_accumulation` is the number of isolated blocks
per optimizer step; each full block has seven supervised proposals. The
included 100-step recipes are starting diagnostics, not established budgets
for an arbitrary dataset. Increase the budget using training and validation
evidence. `max_validation_blocks` caps the validation subset used throughout
selection and final replay, so choose it to cover your evaluation distribution.
`max_steps` counts optimizer steps rather than epochs. For a known total of
N scorable blocks and B blocks per step, an epoch takes about N/B steps;
teacher-generated block counts become known only after capture.

Match `enable_thinking` and `reasoning_effort` to the intended deployment.
The pinned Qwen3.6 template ignores effort; its thinking flag still applies.
Set `enable_thinking: false` for non-thinking serving. With thinking enabled,
short generated continuations may mostly contain reasoning tokens.

```bash
python -m draft_training check-data \
  --target /models/qwen38-teacher \
  --data data/my-domain/train.jsonl \
  --validation-data data/my-domain/validation.jsonl \
  --config my-domain.yaml
```

This loads the local tokenizer, checks split isolation and renders every
training/validation example with your recipe. It reports prompt/sequence
lengths and catches overlong examples before loading teacher or draft weights.
For generated supervision, sequence lengths include the requested
`max_new_tokens` budget; early teacher termination can still produce a short,
unscorable continuation. For supplied answers, actual lengths and full-block
counts are checked. Fix rejected examples or adjust the recipe and rerun.
The trainer also runs these checks automatically for a new run.

Use `--supervision supplied-answers` consistently on `check-data`, preflight,
capture and train if you want to use provided answers. That measures agreement
with those answers rather than automatically matching the teacher policy.

## 4. Preflight, train and resume

Run from the checkout root in your training environment:

```bash
python -m draft_training train \
  --target /models/qwen38-teacher --init-draft /models/qwen38-draft \
  --data data/my-domain/train.jsonl \
  --validation-data data/my-domain/validation.jsonl \
  --config my-domain.yaml --output runs/my-domain --device cuda --preflight
```

Preflight finds a scorable continuation and performs a backward/optimizer update
without publishing a run. It reports capture and training memory separately.
It checks up to eight early continuations for a full assistant query block.
This does not automatically exercise your longest example; compare the
token-check maxima and probe a representative long example separately when
qualifying memory. `--dry-run` instead
checks metadata and split isolation without importing tensor dependencies;
use `check-data` for tokenized length checks.

Remove `--preflight` to capture both splits, train, select on validation and
publish `runs/my-domain/export` plus `selection.json`. Progress logs aggregate
every block in each optimizer step. The frozen teacher is released after
capture; its embedding and output head remain frozen during draft training.

After interruption, rerun the identical training command with `--resume`.
It restores optimizer/RNG state and the block cursor from the last committed
checkpoint; uncheckpointed work is repeated. Keep the same dataset files,
recipe, source checkout, library versions and device context. To prepare
features separately, replace `train` with `capture`, then run `train --resume`
with the same arguments. Existing runs require their original implementation
to resume; source changes intentionally invalidate the run identity.

## 5. Compare, freeze and evaluate

Inspect `baseline-validation.json`, `metrics.jsonl` and `selection.json`.
The selection manifest records the released and selected scores, their delta,
and whether the trained candidate improved, tied or regressed on validation.
Loss reduction alone does not establish better speculative acceptance.
Do not use the final test to choose a recipe or checkpoint.

Once you have frozen a candidate:

```bash
python -m draft_training evaluate \
  --target /models/qwen38-teacher \
  --draft runs/my-domain/export --baseline-draft /models/qwen38-draft \
  --test-data data/my-domain/test.jsonl \
  --selection-manifest runs/my-domain/selection.json \
  --output runs/my-domain/final-test --device cuda
```

This reports both drafts on the same held-out teacher traces and saves
`test-report.json`. It enforces split isolation, frozen artifacts and execution
context and refuses to overwrite a report. If final results influence more
tuning, reserve a new untouched test set for the next candidate.

Copy the BF16 export to a supported Mac and load it with a compatible native
target through `--draft-model /path/to/export`. Compare acceptance, throughput
and latency with the released draft under the same target revision, thinking,
sampling, cache and memory settings. The BF16 teacher and quantized serving
target remain different artifacts. Adopt the candidate only after it meets
your native deployment criteria.

See the [training reference](dflash2-training.md) for contracts, provenance,
export details, tests and the scope of the published CUDA/native smoke.

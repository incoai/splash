# Qwen3.6 MoE draft-training CUDA smoke

This experiment tests the own-data training workflow in #499 using the actual
BF16 Qwen3.6-35B-A3B teacher and released DFlash2 draft. It is an engineering
smoke with 16 authored prompts, not a model-effectiveness study. The MoE teacher,
including its routers and experts, stays frozen; optimization adapts its dense
DFlash2 draft.

## Source and environment

- Tested source: `1aae5d5dcb45805aad63915de54e845dd6be7c37`, the isolated #499
  head. No downstream training PRs or pod-only source changes were used.
- One NVIDIA A100-SXM4-80GB, driver 595.91.07, on RunPod.
- Python 3.12.3, torch 2.13.0+cu130, CUDA 13.0, Transformers 5.15.1,
  safetensors 0.8.0 and PyYAML 6.0.3. Eight torch CPU threads; SDPA attention.
- Teacher: `Qwen/Qwen3.6-35B-A3B` at
  `995ad96eacd98c81ed38be0c5b274b04031597b0`.
- Initial draft: `incoai/Qwen3.6-35B-A3B-DFlash2` at
  `51ef7b6923ad6c14cb1bb41c37a9041446496ab6`.

The run/selection manifests record checkpoint-file hashes, all 13 trainer-module
hashes, native family contracts, library versions and execution settings.
Later documentation commits do not change the tested trainer implementation.

## Dataset and recipe

`prepare-data` split [corpus.jsonl](corpus.jsonl) into eight training, four
validation and four test conversations using seed 42 and 0.25/0.25 fractions.
The prepared files and preparation manifest are included. Source groups and
normalized prompt identities are isolated across splits. The prompts cover
code, math, prose and structured output; they are not a representative corpus.

Both recipes were fixed before final testing. Each uses 20 optimizer steps,
four accumulated blocks per step, learning rate `1e-5`, a 2048-token sequence
limit and at most 128 generated tokens. Supervision is greedy teacher-generated
text. Thinking is explicitly disabled; the pinned template ignores the recorded
`reasoning_effort: medium` setting. Validation and final replay each use eight
blocks spread over all four continuations, only 56 proposal positions per split.
The final test is not used for checkpoint or recipe selection.

The separate [context probe](context-probe.jsonl) has 1,773 rendered prompt
tokens and a maximum 128-token continuation budget. It exercises a longer
context than the short training prompts. Preflight checks one complete draft
block's backward/optimizer update, rather than a full long-context training run.

## Results

| Recipe | Optimizer updates | Selected checkpoint | Validation mean matching proposals | Test mean matching proposals |
| --- | ---: | --- | --- | --- |
| Selector only | 20 | step 10 | 3.5 → 3.5 | 4.5 → 4.5 |
| Full draft | 20 | step 10 | 3.5 → 3.75 | 4.5 → 4.5 |

These scores measure consecutive matching proposals, from zero to seven, on
frozen BF16 teacher traces. They are not native speculative acceptance,
throughput or sampled-decoding results. The full-draft validation gain did not
produce a held-out matching-proposal gain. Neither candidate demonstrated an
effectiveness improvement on the tiny test set.

Both selector and full-draft long-context preflights passed:

| Preflight | Teacher capture peak allocated | Draft backward/optimizer peak allocated |
| --- | ---: | ---: |
| Selector only | 65.9 GiB | 5.1 GiB |
| Full draft | 65.9 GiB | 11.8 GiB |

These are separate CUDA allocator peaks, not total device-memory requirements
or sizing guarantees for other datasets and recipes. All MoE experts remain
resident during teacher capture. The decoder is released before optimization.

The full run was interrupted with SIGINT after its step-10 checkpoint directory
committed, exiting with code 130. That checkpoint recorded ten updates and cursor
40. The identical CLI command with `--resume` completed step 20 with 20 updates
and cursor 80. Its metric sequence contains steps 1–20 exactly once. This tests
recovery on the real checkpoint; equality against a separate uninterrupted CUDA
run was not measured. Deterministic optimizer/RNG recovery is covered separately
by the component tests.

Both automatic exports contain finite BF16 tensors. The selector export changed
only its three candidate-selector tensors; its other 93 tensor payloads stayed
identical to the released draft. Full-draft optimization changed 70 of the 96
exported BF16 tensor payloads. A standalone export of the selected full checkpoint
matched the automatic export tensor for tensor. Resuming the already completed
full run also passed: it verified the existing export and selection without
adding optimizer steps or duplicated metric entries.

## Reproduction

Use the exact source, model revisions and pinned training requirements above.
The [own-data quickstart](../../dflash2-training-quickstart.md) describes setup
and downloading the local BF16 checkpoints. Run from the source checkout:

```bash
python -m draft_training prepare-data \
  --data /path/to/this-report/corpus.jsonl --output data/moe-smoke \
  --validation-fraction 0.25 --test-fraction 0.25 --seed 42

python -m draft_training check-data \
  --target /models/qwen36-teacher \
  --data data/moe-smoke/train.jsonl \
  --validation-data data/moe-smoke/validation.jsonl \
  --config /path/to/this-report/selector.yaml

python -m draft_training train \
  --target /models/qwen36-teacher --init-draft /models/qwen36-draft \
  --data data/moe-smoke/train.jsonl \
  --validation-data data/moe-smoke/validation.jsonl \
  --config /path/to/this-report/selector.yaml \
  --output runs/moe-selector --device cuda --preflight
```

For the measured long-context preflight, replace `--data` with the report's
`context-probe.jsonl`. Remove `--preflight` to train on the prepared training
split. Use `full.yaml` and a separate output directory for full-draft training.
After interruption, rerun the identical command with `--resume`; keep inputs,
recipe, implementation, versions and device context unchanged.

```bash
python -m draft_training evaluate \
  --target /models/qwen36-teacher \
  --draft runs/moe-selector/export --baseline-draft /models/qwen36-draft \
  --test-data data/moe-smoke/test.jsonl \
  --selection-manifest runs/moe-selector/selection.json \
  --output runs/moe-selector/final-test --device cuda
```

Repeat evaluation for the independently selected full-draft candidate. The
commands preserve the released baseline and refuse to overwrite final reports.

## Remaining qualification

Trained MoE exports still need native Splash loading and matched-target
acceptance, latency and throughput measurements. Larger representative datasets,
longer budgets and independent seeds are required to assess effectiveness.
This smoke does not qualify arbitrary MoE architectures, fresh draft
initialization, quantized teachers or every downstream training objective.
No trained checkpoint becomes a serving default.

## Recorded artifacts

- [Summary and BF16 tensor hashes](summary.json), [environment](environment.json)
  and [source/transport verification](artifact-verification.json).
- [Selector metrics](runs/selector/metrics.jsonl), [selection](runs/selector/selection.json)
  and [held-out report](runs/selector/final-test/test-report.json).
- [Full-draft metrics](runs/full/metrics.jsonl), [selection](runs/full/selection.json)
  and [held-out report](runs/full/final-test/test-report.json).
- [Selector preflight](selector-preflight.json), [full preflight](full-preflight.json)
  and [preparation manifest](prepared/preparation.json).
- [Stage history](driver-status.json), [interruption](logs/full-interrupted-train.log),
  [resume](logs/full-resume.log) and [completed-run resume](logs/full-completed-resume.log).

Checkpoint weights, optimizer states and feature tensor payloads are excluded
from the repository. Their identities and capture manifests are recorded here.

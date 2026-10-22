# DFlash model acceptance and training analysis

The strongest candidate for Splash is to adapt the existing DFlash2 selector
to the deployed target before enlarging the drafter. In the short greedy
probes below, **75–96% of first rejections were selection misses**: the target
token was already among the 16 candidates. Changing the selection objective
alone was inconsistent across prompts. A better trained conditional selector
is therefore a stronger experiment than immediately doubling block size,
adding transformer layers, or retaining more candidates.

These are measurements of acceptance on two cached checkpoints, not a trained
model improvement or a throughput claim. Training and architecture proposals
below are hypotheses to test against this baseline.

The strongest candidate is filed as
[#542](https://github.com/incoai/splash/issues/542).

## Scope and implementation

The measurement checkout starts at `6e8849b6`; its native model sources match
upstream main `1482f6bb` before adding the development accessor. Upstream main
does not contain Inco's checkpoint training implementation, optimizer
configuration, training corpus, or checkpoint ablations. Consequently, the
source cannot establish whether a proposed training technique was already
used to produce these weights. Public DFlash2 training is available in
[SpecForge](https://sgl-project.github.io/SpecForge/basic_usage/training.html);
checkpoint training changes belong in such a training implementation, with
export and evaluation here.

Existing draft PR [#499](https://github.com/incoai/splash/pull/499), reviewed at
`bf3e6681`, provides optional selector-only/full-draft continuation with unary
and strict-candidate selector cross entropy, a BF16 teacher, and BF16 export
evaluation. Draft PR [#537](https://github.com/incoai/splash/pull/537), reviewed
at `2fe5fe39`, adds experimental MPS support. These provide the training
baseline; the proposed quantized-teacher, affine-4bit-aware and prefix-survival
experiments are additional work. Neither draft PR establishes a real-checkpoint
native speedup.

| Property | Evidence in this checkout | Consequence |
| --- | --- | --- |
| One anchor and seven proposals | `ExecutionGeometry.h`, `DFlashDraft::addDecode` | At most eight retained tokens per full cycle. |
| Local dynamic convolutions before and after attention and MLP | `DFlashDraft.cpp`, `DraftAttention.h` | Adding the DFlash2 convolution is already implemented. |
| Top 16 unary candidates, rank 256 adjacent-token selector | `DraftSelector.cpp`, `sampling.metal` | The selector can repair choices within this support; it cannot invent an absent candidate. |
| Greedy or ancestral walk through locally normalized scores | `draft_select_dflash` | The current greedy choice does not optimize whole-prefix survival. |
| Shared target vocabulary projection | End of `DFlashDraft::addDecode` | Target head quantization also changes the draft's unary logits. |
| Every draft projection converted to affine 4-bit, groups of 64 | `DraftCheckpoint.cpp` | Context fusion, dynamic kernels, and selector hidden projection all encounter this rounding. Norms, convolution base kernels, and selector codebooks stay BF16. |
| One fused target context representation, with per-layer KV projections | `addContextKv`, `contextProjection` | Per-layer KV projections already differ. A proposal merely to untie them duplicates existing behavior. |
| A 2048-position draft ring | `DraftAttention.h`, `readDraftLayout` | Direct draft attention has a local window; target hidden features still contain the target's longer-context information. |
| Temperature passed to the selector | `SelectorBatchParams`, `DraftSelector::add` | Target penalties and target top-k/top-p/min-p filtering are not mirrored by the current draft selection. This is compatible with exact rejection sampling, but may affect overlap. |
| Online MoE verification budget | `VerifyBudget.cpp`, `draft_verify_rows` | Adaptive verification already exists. A higher accepted/proposed ratio can result from checking fewer proposals, without a better model. |

The dense fixture uses five draft layers and target layers `[5,19,33,47,61]`.
The MoE fixture uses six draft layers and `[1,6,11,16,22,27,32,37]`.
The loader supports alternative layer counts, intermediate sizes, and selected
target layers within its limits; attention geometry, selector rank/support,
block size, and window must match the compiled kernels.

## What success means

Measure useful retained tokens per cycle, conditional acceptance at each
proposal position, and wall time per output token. Keep output quality as the
target's responsibility. Raising an acceptance threshold or relaxing target
verification is not a model improvement.

For a full block without terminal clipping, let `s_i` be the probability that
proposal `i` is accepted conditional on every earlier proposal being accepted.
Then:

```text
tau = 1 + s_1 + s_1*s_2 + ... + s_1*s_2*...*s_7
time per token = cycle wall time / tau
```

This explains why isolated late-position accuracy and ordinary token accuracy
can be misleading: a late prediction only pays when its prefix survives.
For example, increasing uniform conditional acceptance from 0.80 to 0.85
raises this eight-row model's theoretical tau from 4.16 to 4.85. Those are
illustrative calculations, not measured model gains.

For stochastic proposals with conditional distributions `q` and `p`, exact
rejection sampling accepts with probability `sum_v min(p(v),q(v))`, or
`1 - TV(p,q)`. Here `p` must include the request's actual penalties and
sampling filters. Since `q` is supported on 16 candidates, its acceptance
cannot exceed the target mass of that set. Renormalizing target probabilities
within the candidates hides the lost tail mass. Greedy top-16 recall and
stochastic target mass within the top 16 answer different questions.

## Measured acceptance

Measured October 22, 2026 on an Apple M5 Pro with 48 GiB unified memory,
macOS 26.6, INT8 target KV and FP32 recurrent state. The tool uses the GPU
path, with no ANE FFN. Every cycle verifies all eight rows, avoiding the
MoE budget's selection of which suffixes get observed. Prompts use the cached
dense tokenizer's chat template with thinking disabled. The JSON prompt is
ordinary text generation, without a schema constraint.

Each B1 row below is **one authored prompt and 64 consecutive cycles**.
Cycles from one answer are correlated; these are diagnostic probes, not
independent benchmark samples or statistical evidence of general gains.

| Model | Prompt | Mean retained tokens | Selection misses | Candidate recall misses | Selection share of first rejections |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B MLX 4-bit | Code | 4.344 | 43 | 7 | 86.0% |
| Qwen3.8-27B MLX 4-bit | Math | 4.781 | 47 | 2 | 95.9% |
| Qwen3.8-27B MLX 4-bit | Prose | 3.125 | 53 | 6 | 89.8% |
| Qwen3.8-27B MLX 4-bit | JSON | 5.375 | 31 | 6 | 83.8% |
| Qwen3.6-35B-A3B MLX 4-bit | Code | 3.906 | 45 | 10 | 81.8% |
| Qwen3.6-35B-A3B MLX 4-bit | Prose | 3.406 | 45 | 15 | 75.0% |

Full acceptance cycles are absent from the two miss counts. Candidate recall
misses still matter, especially on the MoE prose probe; the data does not
establish that the backbone needs no improvement.

The dense code probe accepted position 1 on 53/64 reached prefixes and position
7 on 14/17. Dense prose accepted position 3 on 18/36 reached prefixes. This
does not support a universal explanation based on last-position decay alone.
The late-position samples are particularly small.

### Replaying selection objectives

For cycles with all seven reference tokens available, replay can compare paths
against the recorded greedy continuation. Both alternatives use the existing
edge table. Expected-length dynamic programming maximizes the sum of draft
prefix probabilities; Viterbi maximizes joint conditional probability.

| B1 probe | Complete cycles | Current path tau | Expected-length path tau | Viterbi path tau | Candidate membership oracle tau |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense code | 64 | 4.344 | 4.406 | 4.531 | 6.781 |
| Dense math | 64 | 4.781 | 4.781 | 4.781 | 7.328 |
| Dense prose | 62 | 3.129 | 3.177 | 3.081 | 5.677 |
| Dense JSON | 64 | 5.375 | 5.375 | 5.266 | 7.234 |
| MoE code | 63 | 3.921 | 3.921 | 3.968 | 6.032 |
| MoE prose | 62 | 3.468 | 3.452 | 3.403 | 5.548 |

The oracle chooses the recorded target token whenever it occurs in each
position's candidate set. It is an upper bound on this fixed-continuation
comparison, not a deployable policy. Repaired prefixes require later true
tokens: target rows after the first rejection are conditioned on the rejected
draft and cannot supply those labels. The tool instead reconstructs the true
continuation from later committed outputs and excludes incomplete tails.

Changing paths shifts subsequent cycle starts and can change numerical results
through different verify batches. These replay numbers are local estimates;
they do not predict an exact alternative rollout or speedup. In particular,
Viterbi helps dense code while hurting prose and JSON. Issue
[#259](https://github.com/incoai/splash/issues/259) and its earlier pilot
[#261](https://github.com/incoai/splash/pull/261) were closed as not planned
after finding no acceptance gain. These replay probes do not justify reopening
that decision.

A separate dense code probe at presence penalty 1.5 retained 4.672 tokens per
cycle, with 41 selection and 5 recall misses. Its generated continuation
differs from the unpenalized run. Comparing those means does not isolate the
effect of penalty mismatch; a controlled comparison must score both policies
on the same contexts.

### Checking scalar edge calibration

An exploratory sweep of `unary + scale * edge` checked scales
`0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2` over the same complete-reference
cycles. All values below are greedy local prefix yields, including the anchor.

| B1 probe | Scale 0 | Scale 0.5 | Scale 1 | Scale 1.5 | Scale 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense code | 4.125 | 4.391 | 4.344 | 4.141 | 4.047 |
| Dense math | 4.531 | 4.844 | 4.781 | 4.969 | 4.922 |
| Dense prose | 2.806 | 3.129 | 3.129 | 3.177 | 3.226 |
| Dense JSON | 5.078 | 5.297 | 5.375 | 5.156 | 5.031 |
| MoE code | 3.571 | 3.937 | 3.921 | 3.857 | 3.619 |
| MoE prose | 3.323 | 3.403 | 3.468 | 3.355 | 3.194 |

No tested nondefault scale improves all six sequences. This checks the
low-cost calibration candidate but does not select a deployment parameter:
there is no held-out calibration split. Scaling recorded edge scores also
does not reproduce rounding after scaling and exporting a BF16 codebook.
The full sweep is in the measurement directory's `edge-scale-replay.json`.

## Training changes in priority order

### Adapt the existing selector to the deployed target

First freeze the backbone and adapt its selector using target-generated,
correct-prefix examples from the actual quantized deployment. Use Inco's
training implementation or an existing DFlash2 trainer, exporting the same
checkpoint shapes. Compare selector-only adaptation with an equal-budget
continuation of the existing training recipe. The measured selection misses
make this the cheapest targeted experiment; they do not prove that training
mismatch caused those misses.

Use three explicit stages:

1. Collect target continuations and hidden features from code, reasoning,
   prose, JSON, tool boundaries, and multilingual prompts. Sample multiple
   prompts per domain and split by conversation/repository/template, keeping
   held-out prompts outside all tuning decisions. Include thinking modes and
   deployment context lengths. The current captures contain token IDs and
   scores, **not the hidden features or full target probabilities required
   for training**.
2. Optimize candidate conditional cross entropy on the verifier-correct
   predecessor, with hard negatives from actual selection mistakes. Track
   natural top-16 coverage separately: do not silently inject the true token
   into evaluation candidates. Mix ordinary examples with error examples so
   frequent failures do not erase good domains.
3. Adapt with the exact affine 4-bit/group-64 rounding used by the loader for
   selector/context projections, and with features and vocabulary-head
   behavior matched to the selected target quantization. Keep a separate
   BF16-draft control in the trainer to measure how much rounding costs.

For frozen-selector adaptation, [#543](https://github.com/incoai/splash/issues/543)
also proposes exporting native draft final hidden rows, natural candidates
and correct-prefix labels. Training on that frozen cache can avoid loading
the BF16 teacher and draft backbone on the training device. It needs a new
feature-export schema beyond the score-only captures here, and does not
replace joint backbone training or native evaluation of the trained export.

Try scalar unary/edge scale calibration before adding capacity; the current
score is `unary + edge` with no configurable learned scale at inference. A
checkpoint-compatible edge scale can be folded into one trained codebook.
Optimize held-out prefix yield, rather than only selector training loss.

Quantization-specialized drafter adaptation has precedent: a Qwen3.5-4B study
uses a high-precision stage followed by adaptation to its quantized target.
Its GPU results do not establish a gain for Splash, its kernels, or DFlash2.
[Quantize the Target, Quantize the Drafter](https://arxiv.org/abs/2607.04244)

### Train against the acceptance objective and deployment policies

Use correct-prefix conditional acceptance as an auxiliary objective. A
practical teacher-forced surrogate is to maximize a sum of cumulative
overlaps, using distribution overlap for sampled requests and smooth
correct-token probabilities for greedy requests. This surrogate is not an
exact optimization of the complete rollout. Retain ordinary token/selector
cross entropy for stability and compare several position weight schedules;
the original DFlash training already includes position-dependent loss decay.
[DFlash training and ablations](https://arxiv.org/html/2602.06036v2)

Collect rejection examples from the actual draft–verify loop, then recompute
teacher labels under correct prefixes. Do not treat the target's suffix
behind an incorrect draft prefix as the correct continuation. Draft-OPD
specifically revisits verification-exposed error positions, while AdaFlash
studies on-policy adaptation for diffusion drafters. Both motivate an
experiment; neither proves an improvement over these DFlash2 checkpoints.
[Draft-OPD](https://arxiv.org/abs/2605.29343),
[AdaFlash](https://arxiv.org/abs/2607.19223)

Train and evaluate policy mixtures deliberately: greedy, target-default
sampling, and representative repetition/presence/frequency penalties. A
selector that learns unpenalized distributions need not approximate a
penalized request equally well. Mirroring penalties at runtime is another
candidate, but it must account for the selected within-block prefix and
candidate retrieval, as well as the proposal probabilities used for exact
sampling. Applying a penalty only after top-16 retrieval cannot recover
excluded candidates. Keep this as an ablation until identical-context
distribution-overlap measurements show a benefit.

### Improve candidate recall only where it limits acceptance

If recall misses dominate a domain, selector tuning alone cannot fix them.
Jointly adapt the late backbone, context fusion and convolution projections,
using the actual target features and quantized shared LM head. Sweep draft
loss decay, random anchor placement, and context-length mixture with equal
compute budgets. Focus sampling on surviving prefixes while retaining domain
coverage. Evaluate improvements at positions that are actually reached.

Keep quantized-target and BF16-target quality separate. Training the draft
to match a low-bit target improves agreement with that target; it does not
restore quality that target quantization lost.

## Architecture experiments

| Candidate | Why it could help | Changes needed | Priority |
| --- | --- | --- | --- |
| Pair-context selector gate `g_t = W_current*h_t + W_previous*h_(t-1)` | The existing gate uses the current draft hidden row and predecessor token identity; explicit preceding hidden context may improve selection without another backbone pass. | New trained gate projection, export/loader support and a shifted gate preparation. Keep rank 256 and the current edge walk initially. | First architecture ablation after selector-only tuning. |
| Small candidate-conditioned nonlinear corrector | Bilinear rank-256 compatibility may underfit context-specific token transitions. | Train a compact corrector and compare against equal-budget bilinear adaptation. Requires new weights and kernels. | Medium, if the existing selector remains the limiting factor. |
| Wider or dilated local convolution | Could address recall decay over longer suffixes. | Train the altered convolutions; update the two-tap ABI/kernels and tests. | Conditional on reached-prefix recall evidence. |
| Dynamic per-layer target-feature fusion | Could avoid sharing one fused context row across all draft layers. | Multiple trained fusions and revised context-window/cache accounting. Static layer reweighting alone can already be represented by the existing `fc`; per-layer KV projections already differ. | Higher cost; test externally before a native port. |
| Selective global draft memory | Could help long-context retrieval beyond direct access to the 2048-row draft ring. | Trained global/summary attention plus new ring, capture, restore and memory-plan contracts. | Only after long-context diagnosis. |
| More layers, larger rank, more candidates, or a 16-row block | More capacity/support can help, but increases bandwidth, selection work, or verify work. | Depth/intermediate changes may fit today's loader; rank/support/window/block changes need native implementations. | Last, after a measured bottleneck justifies each axis. |

Candidate-conditioned contextual correction and layer-wise feature fusion
have published precedents, but their reported comparisons are not matched
ablations against Splash's deployed DFlash2. They should guide experiment
design rather than supply an expected speedup.
[Contextual Correction](https://arxiv.org/html/2610.12570v1),
[DFlare](https://arxiv.org/html/2606.02091v1)

Increasing selector rank is not free: the two stored BF16 codebooks at
vocabulary 248320 and rank 256 occupy approximately 242.5 MiB. Doubling rank
doubles that storage and the edge dot-product width. Increasing candidates
from 16 to 32 quadruples the interior pair table. Prefer measuring better
conditioning at the existing rank and support first.

Longer blocks were discussed in
[#187](https://github.com/incoai/splash/issues/187), which maintainers closed
as not planned because the released drafts fill eight rows. A new 16-position
trained checkpoint and current-kernel qualification would be prerequisites,
rather than padding the existing draft. Their last rows only pay
when early prefixes survive. The existing convolutions and selector are the
two central DFlash2 additions, so proposing either as a new DFlash2 feature
would repeat the released design.
[DFlash2 design](https://inco.ai/blog/dflash2/)

## Recommendation coverage and tracking

Every recommended approach has an implementation, an experiment issue, or a
recorded reason to defer it. Experimental architecture issues require trained
weights and held-out results before a native implementation is proposed.

| Approach | Current result or prerequisite | Tracking |
| --- | --- | --- |
| Selector-only adaptation | Baseline already implemented in the optional training PR; real-checkpoint native qualification remains. | [#499](https://github.com/incoai/splash/pull/499), [#542](https://github.com/incoai/splash/issues/542) |
| Apple GPU training | Existing experimental MPS follow-up, pending real-checkpoint qualification. | [#537](https://github.com/incoai/splash/pull/537) |
| Scalar unary/edge calibration | Implemented offline sweep; no tested nondefault scale improves every baseline sequence. | This diagnostic PR and [#542](https://github.com/incoai/splash/issues/542) |
| Quantized teacher/head matching and affine-4bit-aware draft training | Distinct from the BF16 baseline; needs a controlled trainer adapter and native export evaluation. | [#543](https://github.com/incoai/splash/issues/543) |
| Prefix-survival objective, position loss schedule and verification-error replay | Needs a correct-prefix dataset and equal-compute loss/replay ablation. | [#544](https://github.com/incoai/splash/issues/544) |
| Request-policy matching | Identical-context distribution overlap must justify penalty/filter changes and their cost. | [#545](https://github.com/incoai/splash/issues/545) |
| Pair-context gate and nonlinear selector | Needs trained comparisons against adapted existing-shape selectors first. | [#546](https://github.com/incoai/splash/issues/546) |
| Joint backbone/context adaptation and wider/dilated convolution | Conditional on reached-prefix candidate-recall evidence. | [#547](https://github.com/incoai/splash/issues/547) |
| Per-layer target-feature fusion | Existing KV projections already differ; new fusion needs trained representations and cache accounting. | [#547](https://github.com/incoai/splash/issues/547) |
| Deeper/wider draft, larger rank or more candidates | Training/cost ablation first; rank/support changes require native shape and sampling support. | [#547](https://github.com/incoai/splash/issues/547) |
| Long-context mixture, cold/restored prefixes and selective global memory | Diagnose matched-context recall/reconstruction before selecting a new memory architecture. | [#548](https://github.com/incoai/splash/issues/548) |
| Expected-length or Viterbi search | Implemented offline; mixed results. Maintainers declined the earlier native pilot. | Closed [#259](https://github.com/incoai/splash/issues/259)/[#261](https://github.com/incoai/splash/pull/261) |
| Sixteen-row blocks | Released checkpoints fill eight rows; maintainers closed the previous proposal. | Closed [#187](https://github.com/incoai/splash/issues/187) |
| Existing local convolution, candidate selector, per-layer KV and adaptive verification | Already implemented; adding them again is not a new model change. | Current main |

## Qualification plan

Keep a factorial comparison: released checkpoint; extra training with the
existing recipe; selector-only adaptation; quantization-aware adaptation;
joint backbone adaptation; then one architecture change at a time. Hold
training tokens and compute constant where possible, and report additional
training cost where they differ.

Start with both measured checkpoint families, then Flash-Next, and cover
code/math/prose/tools/JSON, thinking on/off, multiple independent prompts per domain, and
contexts 2K, 8K, 32K and 128K. Evaluate cold and restored prefixes, INT8/BF16
KV, relevant target quantizations, widths 1–4, and both fixed and adaptive
verification. For sampling, record actual `q` along the selected path and
target mass inside the candidate set, and preserve rejection correction.

A reasonable proposed promotion gate is at least 3% more retained tokens per
cycle on held-out workloads, no domain regression exceeding 1%, and an
end-to-end wall-time improvement larger than the paired run's uncertainty.
These thresholds are proposed experiment criteria, not an established project
policy. Require native numerical, sampling, state/cache restore and terminal
tests, followed by sustained serving measurements on Apple9 and Apple10.

Do not promote from these replay probes: no candidate weights have been
trained, and no alternative selector has been measured end to end.

## Reproduction and added tooling

`decode-profile --draft-dump` runs ordinary greedy cycles at a fixed batch
width with all verify rows. Its instrumented-only accessor reads completed
Shared scratch, uses the actual packed dispatch width to locate edge tables,
and omits unwritten predecessor rows at position zero. The production binary
does not link that accessor. The normal profiling mode remains available.

```sh
make build/engine-tests/decode-profile
build/engine-tests/decode-profile build/splash.metallib MODEL_ROOT \
  --draft-dump build/dflash-code.jsonl --prompt-file code.u32 \
  --cycles 64 --width 1
.venv/bin/python -m dev.benchmarks.draft_replay build/dflash-code.jsonl \
  --output build/dflash-code.report.json
```

Prompt files contain chat-formatted little-endian uint32 IDs, without further
templating by the native tool. Captures end with a record count; interrupted
files are rejected. Replay verifies the greedy walk and confidence against
the dump, rounds score sums as FP32, keeps request continuations separate,
censors unavailable suffixes and terminal labels, and validates complete batch
records. It also accepts gzip captures.

Probe inputs, summaries, source hashes, checkpoint revisions and capture
checksums are in [the measurement record](results/dflash-acceptance-m5-pro/README.md).
Compressed raw captures are included with the measurement record; local
uncompressed copies are under `build/dflash-acceptance-m5-pro/`.
The work overlaps the diagnostic scope of
[#258](https://github.com/incoai/splash/issues/258); stage attribution and
stochastic-distribution capture from that issue remain separate work.

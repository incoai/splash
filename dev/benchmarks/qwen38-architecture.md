# Qwen3.8-27B architecture experiments on a 48 GiB M5 Pro

2026-10-25. Local experimental work; no architecture change is enabled in the production checkout or the existing optimization PRs.

## Result

Architecture changes can remove measurable work, but this round found **no quality-preserving architecture improvement ready to ship**. Eleven draft ablations retain the original verifier. All preserve the sampled answers in the screen, but reduce speculation efficiency; the best-looking draft ablation loses **10.91%** throughput in repeated held-out ABBA tests. A four-FFN rank-64 target surrogate reduces fixed-input cycle latency **3.02%**, but changes predictions and worsens held-out next-token likelihood. Removing the same FFNs saves **3.16%** cycle latency and also changes the model.

The positive result is a measured compute opportunity, with a quality problem still to solve. Natural-generation screens suggest larger token/sec gains for some approximate targets, but those runs generate different text and cannot establish equivalent model performance.

[Portable summaries and experimental sources](results/qwen38-architecture/README.md) accompany this report. The earlier [runtime investigation](https://github.com/linson007/splash/blob/45598d5e/dev/benchmarks/qwen38-m5-pro.md) covers the production kernel improvements already proposed in PRs [#583](https://github.com/incoai/splash/pull/583), [#584](https://github.com/incoai/splash/pull/584), and [#585](https://github.com/incoai/splash/pull/585). The subsequent [deployment audit](qwen38-rejection-audit.md) tests a frozen-feature selector pilot, restored context, policy overlap and representative M16 serving qualification; it also finds no newly qualified architecture/checkpoint.

## What the installed architecture actually computes

The pinned target is `mlx-community/Qwen3.8-27B-4bit` revision `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`; the BF16 source draft is `incoai/Qwen3.8-27B-DFlash2` revision `015e795645c74b1a0eeef3b570031fb62e769bc5`. The runtime quantizes the draft projections to affine 4-bit groups of 64. The reference already includes the locally measured M16 decode and short-prefill improvements, so architecture variants do not get credited for those kernel changes.

The target has hidden width 5120, intermediate width 17408, vocabulary 248320, and 64 layers: 48 GDN mixers and 16 full-attention mixers, each followed by a dense SwiGLU FFN. The five-layer draft consumes captured target outputs at layers 5, 19, 33, 47, 61, projects their concatenation to width 5120, and predicts a block of eight rows: one known anchor and seven proposals. Each draft layer has attention and FFN residual branches with trained dynamic convolutions. Its selector uses rank-256 predecessor/successor codebooks and top-16 candidates. The draft retains a 2048-token context ring.

The architecture contracts are spread across `ModelDescriptor.mm`, `DraftCheckpoint.cpp`, `DFlashDraft.cpp`, `QwenTarget.cpp`, the execution geometry headers, and the Metal selector/attention/GDN kernels. Depth reduction must resize loaded layer images, persistent rings, and hidden-buffer parity together. FFN width reduction must slice corresponding gate/up rows and down columns, preserving the original source-row stride and 64-input quantization groups. Dropping a branch must still write its residual into the expected hidden buffer. Experiments implement those contracts rather than merely zeroing weights inside unchanged dense kernels.

One target FFN holds `3 * 5120 * 17408 * 36/64 = 150,405,120` bytes of affine weights, or 143.44 MiB. All 64 account for about 9.63 GB, roughly 64% of target weight bytes. Removing four eliminates approximately 602 MB of logical weight reads per target pass. Four F32 rank-64 pairs add only 10.49 MB of replacement matrices. **The prototype still loads the original FFN weights**, so this is an execution/traffic experiment, not a demonstrated reduction in resident memory. The memory planner also does not account separately for the surrogate's extra allocation.

For a speculative cycle of cost C producing an average tau emitted tokens, token rate is tau/C. A modification wins only if `tau_new/tau_old > C_new/C_old`. A 3% cheaper cycle can tolerate less than a 3% decrease in useful tokens per cycle. This explains why shrinking an already small draft is fragile: its acceptance loss can exceed its saving against a full 27B verification pass.

## Measurement and correctness controls

Apple M5 Pro, raw Apple GPU family 10, 20 GPU cores, 48 GiB physical memory, macOS 26.6; AC power, power mode 0, ANE disabled, INT8 KV and FP32 recurrent state. Fresh server processes, configured memory 34G, max context 32768, disk cache disabled. Each measured request asserts zero matched cached tokens and uses native complete-cycle milliseconds divided by actual emitted decode tokens. Warmups are discarded. GPU experiments are serial. No OS memory, power, or cache setting is changed.

The default experimental binary's full same-prefix logits are bitwise equal to the original reference, including after adding capture instrumentation. Independent shorter-depth and narrower-width controls also leave target logits bitwise equal at both quality prefixes. An additional CPU check validates the low-rank FFN orientation, BF16 rounding, and residual addition against 32 captured native output rows of layer 4, with bitwise agreement. Experimental serving identities hash source, architecture flags, and loaded surrogate tensor content so cached state identities cannot mix variants.

The first draft screen has substantial drift: repeated controls slow 11–23% depending on prompt. Its timing rankings are exploratory. Two subsequent ABBA blocks use four new held-out prompt domains and 192 emitted tokens per request. Controls and candidate answers are identical within each block, while their cycle counts consistently differ. The draft conclusion is driven by this acceptance loss and paired measurements, not a correction for the initial drift. No confidence interval based on treating individual GPU cycles as independent trials is claimed.

There are 31 fresh serving processes and 124 measured requests in this round, plus separate teacher-forced correctness, calibration, sensitivity, and fixed-input timing runs. OS reports no thermal warning and normal pressure at the recorded boundaries; pre-existing swap is present. Those signals do not rule out clock or thermal drift. Raw before/after power, pressure, swap, response, and native metrics are preserved.

## Draft architecture ablations: original target retained

The screen uses four authored domains: code, mathematics, Chinese prose, and JSON. Each runs 128 output tokens. Control verification cycles are 29 / 33 / 75 / 25 respectively. Every candidate below has the same four generated messages as that control. Width pruning is a deliberately simple **prefix-channel ablation**, not activation-aware selection or a reproduction of a published pruning algorithm.

| Variant | Loaded draft weights, decimal MB | Cycles: code / math / Chinese / JSON |
| --- | ---: | --- |
| `draft4-no-last` | 1079.1 | 40 / 45 / 119 / 40 |
| `draft4-no-middle` | 1079.1 | 37 / 37 / 88 / 35 |
| `draft4-no-first` | 1079.1 | 127 / 127 / 127 / 127 |
| `draft3-spaced` | 891.5 | 43 / 41 / 94 / 39 |
| `draft-mlp-no-middle` | 1266.6 | 33 / 36 / 83 / 28 |
| `draft-mlp-no-last` | 1266.6 | 30 / 35 / 98 / 31 |
| `draft-mlp-no-two` | 1266.6 | 37 / 40 / 86 / 31 |
| `draft-ffn16384` | 1222.4 | 30 / 34 / 77 / 28 |
| `draft-ffn12288` | 1045.5 | 33 / 33 / 80 / 29 |
| `draft-ffn8704` | 890.6 | 36 / 39 / 88 / 33 |
| `draft-no-convs` | 1266.6 | 40 / 44 / 84 / 35 |

Removing the first draft layer causes 127 cycles for 128 emitted tokens in every domain: speculation has almost collapsed to one useful token per cycle. The trained convolutions are also functional components, not removable overhead. Fewer complete layers and narrower FFNs physically reduce draft images; FFN/convolution skip flags bypass execution while keeping their original weights resident.

The best-looking screen candidate, skipping draft FFN source layer 2, receives two repeated ABBA blocks on new prompts. Geometric-mean throughput falls 10.91%; useful emitted tokens/cycle fall 10.64%; measured complete-cycle cost is approximately unchanged (+0.31%). Domain throughput changes are code +0.05%, math −16.83%, Chinese −5.86%, JSON −19.59%. There is no qualified win. These results reject these untrained ablations, not all retrained smaller drafts.

## Target FFN selection and quality sensitivity

A local influence proxy averages `1-cos(residual, residual+FFN)` over the last 32 prefill rows of three calibration prefixes. It is inspired by [ShortGPT's block-influence approach](https://arxiv.org/abs/2403.03853), but scores individual FFN branches in this hybrid target; it is not a reproduction of its whole-layer experiments. Layers 0 and 63 are protected from low-influence selection. Selected layers are (zero-based indices):

- least1: 4; least2: 4, 5; least4: 3, 4, 5, 9; least8: 1, 3, 4, 5, 8, 9, 11, 12.
- fixed dispersed comparison: 7, 23, 39, 55; high-influence comparison: 63.

All mixers, recurrent updates, attention caches, and draft capture positions remain active. FFN bypass is applied in both prefill and verification, so each variant defines a consistent approximate target instead of using different target functions in those phases.

Sensitivity is evaluated on a source-code corpus held out from calibration, at 1024- and 4096-token prefixes, 129 next-token distributions each. The table compares full-vocabulary FP32 logits to the unchanged quantized target. KL is `KL(original || variant)`; positive NLL delta means worse likelihood of the true continuation, in nats/token. Top1 agreement measures agreement with the original, **not task accuracy**. Two short source-text windows cannot certify broad quality.

| Variant | Top1 agreement at 1024 | At 4096 | Mean KL: 1024 / 4096 | True-token NLL delta: 1024 / 4096 |
| --- | ---: | ---: | ---: | ---: |
| `target-ffn-least1` | 127/129 | 122/129 | 0.0078 / 0.0171 | -0.00004 / +0.0335 |
| `target-ffn-least2` | 125/129 | 120/129 | 0.0116 / 0.0476 | +0.0012 / +0.0379 |
| `target-ffn-least4` | 123/129 | 117/129 | 0.0283 / 0.1536 | +0.0105 / +0.1259 |
| `target-ffn-least8` | 117/129 | 111/129 | 0.1245 / 0.3920 | +0.1697 / +0.2724 |
| `target-ffn-random4` | 126/129 | 116/129 | 0.0485 / 0.2048 | +0.0711 / +0.0817 |
| `target-ffn-high1` | 115/129 | 103/129 | 0.3351 / 0.5085 | +0.3786 / +0.7481 |
| `surrogate1` | 129/129 | 123/129 | 0.0064 / 0.0169 | +0.0028 / +0.0452 |
| `surrogate4` | 123/129 | 115/129 | 0.0420 / 0.1653 | +0.0325 / +0.1507 |

Even the low-influence four-FFN deletion changes 6/129 short-context and 12/129 long-context top choices. Cosine similarity to a large residual stream therefore does not establish that a small branch is unimportant to the final vocabulary distribution. The dispersed and high-influence controls reinforce that selection matters, but no deletion is quality-preserving.

## Calibrated low-rank replacement

A second architecture replaces selected FFNs with `residual + B(A(RMSNorm(residual)))`, rank 64. The replacement has two F32 matrices, a BF16 intermediate and projected output, and a BF16 residual add. It uses existing tested float-projection kernels and a small experimental residual kernel. It has no new recurrent state and preserves every mixer and capture site.

For each selected FFN, 1536 captured prefill rows from three calibration prefixes fit its final residual delta. Output PCA plus ridge regression produces A and B. Leave-one-calibration-prefix-out folds select among ranks 16/32/64 and relative ridge strengths 0.001/0.01/0.1/1. All four choose rank 64. Leave-one-prefix-out calibration relative squared errors are about 0.75–0.85 against deleting the branch. The Chinese calibration prefix repeats a 59-token authored paragraph, so 1536 rows should not be mistaken for 1536 independent diverse examples. Fitting uses NumPy/Accelerate on CPU; this is local regression calibration, not end-to-end distillation or target finetuning.

A one-FFN surrogate slightly improves top1 agreement and KL versus deletion on these windows, but worsens true-token NLL. Four surrogates do not improve the quality trade-off: short/long KL 0.0420/0.1653 and NLL deltas +0.0325/+0.1507 are worse than simply bypassing the same four FFNs. The tiny calibration set and restricted linear form do not recover the nonlinear target sufficiently.

## Is there an actual compute saving?

Yes, for approximate targets. A dedicated native probe times only the complete synchronous `Runtime::decode` call. It forces the identical 2048-token held-out input prefix and 128-token continuation, retaining all eight rows per cycle. CPU logit scoring and file output occur outside the timer. Each candidate uses fresh-process ABBA; the first two of 16 cycles in each process are discarded. All control repeats and candidate repeats are independently bitwise stable.

| Approximate architecture | Baseline process means, ms/cycle | Candidate process means, ms/cycle | Cycle latency reduction |
| --- | ---: | ---: | ---: |
| `skip4` | 69.817 / 69.888 | 67.578 / 67.707 | 3.16% |
| `surrogate4` | 69.577 / 69.516 | 67.468 / 67.428 | 3.02% |

Control/candidate process spreads stay below 0.20% in this specific experiment. The roughly 3% cost reduction is repeatable at this input. It is not a claim of 3% equivalent-quality end-to-end throughput, larger-context performance, or multi-lane performance.

The separate natural-generation target screen shows +7.9–8.7% geometric-mean token rate for selected FFN deletions relative to bracket controls. A later held-out prompt screen shows +12.75% for four surrogates and −0.26% for one surrogate. Text and acceptance trajectories differ; bracket drift reaches 5.6% in that later screen. These are descriptive prototype outcomes, not qualified product gains. The fixed-input experiment isolates only about 3% physical saving; the larger natural-generation difference includes changed model behavior.

## Priorities after these experiments

Literature review after the initial recommendation: the original [DFlash depth ablation, Section 5.5.2, Table 6](https://arxiv.org/html/2602.06036v2#S5.SS5.SSS2) and Inco's [DFlash2 depth/convolution study](https://inco.ai/blog/dflash2/#suffix-decay-is-a-local-problem) already investigate this trade-off. The earlier recommendation to make a smaller draft the strongest next direction was premature. Our local layer-removal experiments are untrained surgery on the released checkpoint, not comparisons between independently trained depths. I did not find a published depth sweep for Qwen3.8-27B on this M5 Pro.

1. **Retain the trained five-layer DFlash2 as the architecture baseline.** Prioritize measured reductions in execution cost while preserving its predictions and acceptance. Revisit a smaller trained draft only with a specific deployment hypothesis and a budget that justifies reproducing existing depth research. Score the joint objective tau/C, not draft loss or draft latency alone; a local pruning failure does not settle the performance of an independently trained smaller model.
2. **Test structured nonlinear FFN compression with recovery training.** Use activation-aware neuron groups or a compact SwiGLU student, hardware-aligned widths, broader multilingual/code calibration, and joint validation of downstream logits and draft acceptance. The fixed-input result bounds the current four-branch compute opportunity at about 3%; a small quality regression cannot be waved away by similar hidden-state angles. Original weights must be excluded by the exporter/loader before claiming resident-memory savings.
3. **Investigate a cheap trained draft exit/selector improvement.** DFlash2 already predicts the block in parallel and uses a small pairwise selector; its authors distinguish candidate-selection error from suffix/backbone error. A learned selector or exit gate may improve tau without another full vocabulary projection. Measure added cost explicitly. [DFlash2's design and ablations](https://inco.ai/blog/dflash2/) support studying both error sources; their reported gains do not transfer automatically to this Mac/checkpoint.
4. **Treat whole-target early exit, residual-width surgery, attention-to-GDN conversion, and dense-to-MoE as new model projects.** [LayerSkip](https://arxiv.org/abs/2404.16710) trains with layer dropout and early-exit objectives; stock Qwen does not gain that ability from a flag. [SliceGPT](https://arxiv.org/abs/2401.15024) uses transformations and smaller dense matrices, not zero masks in unchanged kernels. Hybrid recurrent state, RoPE/head geometry, target feature captures, quantization groups, and draft training must remain compatible. Replacing an attention block with GDN requires learned recurrent parameters and new behavior. These directions are untested here.

Block length also couples the trained draft, selector, buffers, and GDN verification tape. More rows help only if the expected retained prefix grows faster than cycle cost; shorter blocks can waste the same weight bandwidth. Changing a compile-time row constant is not enough evidence. Similarly, arbitrary sparse neurons need a kernel that avoids reads; with eight speculative rows, their union can erase a hoped-for sparsity gain. These are implementation and training hypotheses for later bounded experiments, not results of this round.

No architecture variant is promoted to the existing production PRs. The private branch and source archives keep the failed and approximate prototypes reviewable and reproducible.

# Qwen3.8-27B DFlash2 deployment audit on a 48 GiB M5 Pro

2026-10-25–26. Research evidence only. The released five-layer draft, installed target weights, and production defaults are unchanged.

## Decisions

The fresh audit confirms a **candidate-selection opportunity**, but the small local adaptation does not realize it. Across 192 cases and 6144 natural greedy cycles, 84.89% of first rejections occur with the correct target token already in the top-16 support. A frozen-backbone selector pilot improves its validation proxy slightly, then **loses 1.17% tokens/cycle on held-out task families**. It fails promotion. Matched-prefix penalty adjustments and restored-context checks do not identify a compelling replacement policy or memory architecture in these samples.

The representative M16 serving comparison preserves outputs but establishes **no general throughput improvement**. Timing drift is large relative to the observed differences. PR [#584](https://github.com/incoai/splash/pull/584) remains draft; earlier narrow counting results must not be advertised as a broad speedup.

The preceding [architecture experiments](qwen38-architecture.md) cover 19 structural variants. This report follows them with diagnosis and a narrowly trained selector experiment, rather than another untrained depth reduction. The original [DFlash depth ablation](https://arxiv.org/html/2602.06036v2#S5.SS5.SSS2) and [DFlash2 depth/convolution study](https://inco.ai/blog/dflash2/#suffix-decay-is-a-local-problem) already investigate the depth/cost trade-off. Neither settles this particular quantized checkpoint on this Mac.

[Portable evidence, verification and reconstruction](results/qwen38-rejection-audit/README.md) accompany the report. Existing work is tracked in [#542](https://github.com/incoai/splash/issues/542), [#543](https://github.com/incoai/splash/issues/543), [#548](https://github.com/incoai/splash/issues/548), [#580](https://github.com/incoai/splash/issues/580), and [#581](https://github.com/incoai/splash/issues/581). No duplicate research issue is needed.

## Reference and measurement contracts

Target: `mlx-community/Qwen3.8-27B-4bit` revision `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`. Draft: `incoai/Qwen3.8-27B-DFlash2` revision `015e795645c74b1a0eeef3b570031fb62e769bc5`. Target geometry is 64 layers, hidden 5120, FFN 17408, vocabulary 248320; 48 mixers are GDN and 16 use attention. The draft uses five layers, seven proposals plus an anchor, top-16 support, rank-256 pairwise selector, and a 2048-token context ring. Splash prepares its BF16 draft projection sources into affine 4-bit/group-64 weights.

The measured production library combines the existing M16 and short-prefill changes at local integration `950ffef9`, upstream base `3135ab45`. Diagnostic accessors reuse PRs [#550](https://github.com/incoai/splash/pull/550) and [#556](https://github.com/incoai/splash/pull/556). Production preprocessing is bitwise unchanged by these accessors. A same-build single-case/batch control matches all eight recorded cycles exactly. The original historical capture has a different build and is not treated as a bitwise golden file.

Hardware: Apple M5 Pro, raw Apple GPU family 10, 20 GPU cores, 18 CPU cores, 48 GiB, macOS 26.6. Experiments run serially on AC power with ANE disabled, INT8 KV and FP32 recurrent state. No OS power, memory or cache setting is changed. HTTP runs request a 34G memory limit, context 32768 and zero disk-cache capacity. Native feature capture and full-vocabulary logit capture are diagnostics; their elapsed times are **not serving throughput**. Configured memory and the status planner are not measurements of peak resident memory.

## Fresh first-rejection audit

There are 32 authored task families, eight each for code, mathematics, Chinese prose and JSON. Every family runs with thinking disabled/enabled and short/approximately 2K/approximately 8K contexts: 192 cases. Actual prompt lengths are 63–81, 1987–2048 and 8124–8192 tokens. Longer prompts use deterministic fictional supporting records with shared templates across families; this is not a retrieval or task-accuracy benchmark.

Task-family splits are fixed before capture: 16 development families, eight validation, eight final test. Final-test metrics remain uninspected until the candidate checkpoint is frozen. Each case starts with fresh request/state and runs 32 natural greedy cycles, all eight verification rows, B1 and frozen budget learning. All 192 complete 32 cycles without terminal censoring. Rejection diagnosis uses only the accepted prefix and first mismatch: later target rows conditioned on incorrect proposals are not gold labels.

| Domain | Cases | Mean emitted tokens/cycle | Selection first misses | Support-recall first misses |
| --- | ---: | ---: | ---: | ---: |
| Code | 48 | 3.7461 | 1115 | 201 |
| Mathematics | 48 | 4.9297 | 973 | 87 |
| Chinese | 48 | 3.5273 | 1029 | 252 |
| JSON | 48 | 4.6413 | 916 | 178 |
| All | 192 | 4.2111 | 4033 | 718 |

Selection accounts for `4033/(4033+718) = 84.887%` of first misses. This diagnoses where the greedy proposal fails; it does not establish that quantization caused the failure, that a different selector can recover every miss, or that an offline oracle yields a new rollout speedup. Conditional proposal-position acceptance is 81.4%, 78.2%, 79.9%, 81.8%, 80.9%, 81.2%, 83.0%, with respective reached denominators 6144, 5004, 3911, 3123, 2554, 2066, 1678. Those surviving-prefix denominators matter when interpreting suffix behavior.

## Frozen-feature selector pilot

The local pilot keeps the existing draft backbone, both codebooks and target frozen. Only the selector hidden projection receives a merged rank-8 update: 43008 trainable parameters. Deployment geometry and selector compute stay unchanged. Native normalized hidden rows and correct reached-prefix predecessors supply 11570 supported training examples from 96 development cases. Unsupported reached labels are counted (372/11942), then excluded from candidate cross-entropy; no gold candidates are injected.

Four equal-budget fits compare BF16-forward and loader-matched AF4-forward training at learning rates 0.0003/0.001: 200 steps, batch 256, shared seed and case-balanced sampling, candidate CE plus 0.25 baseline KL, position decay and clipping. This reuses the existing PR [#551](https://github.com/incoai/splash/pull/551) quantizer. All 1310720 projection weights match an independent native-loader AF4 reference exactly, on both CPU and MPS. Native hidden-to-edge reconstruction agrees on 776/776 conditional argmaxes; edge RMSE is 4.23e-5, maximum absolute error 0.001223. Anchoring score changes to captured native edges avoids crediting CPU/Metal accumulation differences as learning.

Validation chooses BF16-forward, learning rate 0.001, step 100. The original fixed-continuation proxy rises from 4.28581 to 4.31225 (+0.617%). Its checkpoint SHA256 is `21b9cc1b35b2ea73cafe4a8dec1f75eccbf7dd2bf804babe7b35f5039e1c3c0d`. This proxy is evaluated at old cycle starts, not on a new trajectory.

**A post-decision audit found a proxy-tail bug.** Eight of 1483 proxy cycles could use unobserved suffix tokens from target rows after rejection. No conflicting labels were found at emitted positions; training's reached-prefix labels are unaffected. Rescoring only the released and already-locked candidate on 1475 complete emitted-reference cycles gives 4.28445 → 4.30818 (+0.554%). There is no refitting or reselection after final-test inspection. The original measured script and the corrective audit are both archived; future proxies must exclude these tails.

The exporter changes only that BF16 projection region in a private copy. Every other draft byte is checked equal; target/tokenizer/vision use the original files. A zero-update export reproduces the complete source weight file and an eight-cycle native rollout bitwise. Installed file digests are rechecked when packaging.

| Native rollout qualification | Cases / task families | Geometric mean tokens/cycle change | Family-cluster bootstrap 95% interval |
| --- | ---: | ---: | --- |
| Short validation | 16 / 8 | +1.016% | −1.278% to +3.370% |
| Frozen final test, all six context/mode cells | 48 / 8 | **−1.173%** | **−2.031% to −0.262%** |

Final-test changes by domain are code −2.464%, mathematics −1.151%, Chinese −0.377%, JSON −0.688%. Short/direct and short/thinking lose 2.471%/2.976%; 8K cells gain only 0.549%/0.160%. Initial hidden rows, candidates and unary scores remain bitwise equal in all 48 cases. The model is rejected, with no serving speedup or task-quality claim. This is a small frozen-feature pilot, not a conclusive test of full-draft recovery training or all quantization-aware adaptation.

## One divergent continuation

Common emitted prefixes match in 47/48 final-test cases. The remaining `math-7-counting-paths-8k-direct` case first differs at generated token index 138 (zero-based). Excluding it descriptively still gives −1.175% tokens/cycle across 47 cases. It is not removed from the primary result.

Fresh captures of all raw target logit rows reproduce both original rollouts exactly. At the first difference, the correct token history, verification start position 8307 and logit row 5 match, but earlier acceptance schedules differ. The released run ranks token 332 above 92217 by 0.01518; the candidate reverses them by 0.00398. Centered full-vocabulary logit difference RMSE is 0.02092. A separate identical forced continuation/schedule produces **bitwise-equal target logits** across the two model assemblies: 129 rows × 248320 vocabulary, SHA256 `42af5cba1528fdabb9244de88286605eaac70b7057e9c2979653fee5032f5dbc`.

This is consistent with finite-precision sensitivity to different acceptance/state-update schedules, rather than changed target weight bytes. It does not prove a kernel-level cause, establish task equivalence, or waive output qualification. No modified checkpoint is promoted.

## Policy and restored-context probes

A matched-prefix reference compares target/proposal distribution overlap on 64 correct-prefix positions from eight development families, unit temperature, without top-k/top-p filtering. These greedy contexts deliberately select first misses or the last reached position and are not a representative sampling workload. FP64 policy math is a reference, not native sampling-probability parity or a sampled rollout.

Without penalties, mean overlap is 0.48936. At presence penalty +1.5, existing/history/history-plus-prefix proposal policies give 0.48449/0.48326/0.48363; at −0.5 they give 0.49240/0.49332/0.49312. There is no consistent gain from simply mirroring penalties into proposal scores. Candidate support holds about 86.98% of target probability mass in the unpenalized reference. No runtime policy changes result.

The context experiment pairs materialized-boundary prefill with snapshot/restore into another state lane, matching prefill chunk partition. Eight development families run direct mode at 2K and 8K: 16 pairs, 32 cases, 510 matched context/anchor observations. Initial draft hidden relative L2 drift is 1.11–1.99%; proposals differ at 17 matched observations and retained counts at two. **Every pair has exactly the same mean tokens/cycle after 32 cycles, and every complete emitted token stream matches.** This sample supplies no acceptance-loss case for adding global draft memory or widening its window. It does not rule out effects on other contexts or workloads.

## Representative M16 serving qualification

Original target and draft are used in both arms: baseline versus M16-only, four fresh server processes in A/B/B/A order. Eight held-out task families × two modes × short/2K contexts produce 32 requests/round, 128 measured requests total, 128 output tokens/request; warmups are discarded. Counters include the complete native decode cycle divided by actual emitted tokens. Every request asserts zero cache reuse and no leaked active scheduler, state-lane or KV resources. All messages, usage and speculation records match across arms.

| Stratum | Observed token/sec change | Same-arm decode spread | TTFT change |
| --- | ---: | ---: | ---: |
| Short/direct | +3.251% | 11.063% | +7.481% |
| Short/thinking | −0.374% | 18.826% | +5.431% |
| 2K/direct | −0.467% | 3.545% | +5.237% |
| 2K/thinking | −0.643% | 4.537% | +4.259% |

These are descriptive ABBA summaries, not qualified gains or proof of a TTFT regression. The predeclared promotion rule requires decode gain above max(2%, twice same-arm spread), spread at most 5%, and an interpretable passing TTFT check in every stratum. No stratum passes the gain rule; TTFT same-arm spreads are 13–18%. Short/direct baseline means drift from 16.90 to 18.88 ms/emitted token; this is much larger than a small kernel win. AC stays connected, pressure is normal, existing swap remains 1108 MB and macOS reports no thermal warning. Those boundary signals do not measure clocks or eliminate thermal drift.

## Prioritized follow-up

1. Keep the released checkpoint and the M16 PR's draft status. Continue #581 with shorter interleaved/bracketed comparisons and explicit stationarity checks before more tuning. Preserve actual cycle accounting from #583; do not normalize away drifting controls.
2. Add a repeatable identical-prefix/schedule qualification case for any future checkpoint experiment under #542/#543. Repair offline proxy reference construction before reuse. A new adaptation study needs fresh held-out task families: this final set is now exposed.
3. Treat selection as a measured research opportunity, not a solved optimization. Broader recovery training or a small conditional selector change needs validation gains that survive native rollouts, unchanged outputs, full cycle cost, sampling, TTFT and memory checks. This pilot does not justify another trained draft or production precision change.
4. Keep memory/window architecture and penalty-policy changes behind their existing evidence gates (#548/#545). Current probes do not justify their added execution/state cost. The target FFN compression opportunity remains about 3% fixed-input cycle cost in the preceding architecture study, with unresolved quality loss.

No experiment here qualifies a new architecture or checkpoint for production. Negative results and the proxy correction are retained so subsequent work can start from measured constraints.

# Qwen serving experiments and research archive

The serving tools compare existing verification policies and compatible
builds. They introduce no production kernel, scheduler, model or default
change, and do not claim a new inference speedup.

## Archive and source boundaries

The full DeepInfer-inspired investigation is preserved at immutable Splash
commit [`3e8911fc89a6`](https://github.com/linson007/splash/tree/3e8911fc89a6ed009e7b0561bfa7c23e58ea594f/dev/benchmarks),
also named by tag `archive/deepinfer-qwen-m5-pro-20261021` in `linson007/splash`.
It includes the original plan, source audit, prototype links, all raw reports,
trace archives, excluded captures, failures, commands and validation records.
Confidence/prefill analysis tools and their tests also remain in that snapshot.
Tracking issue [#508](https://github.com/incoai/splash/issues/508) links the
research archive separately from the benchmark tooling PR.

The experiments used one M5 Pro, 20 GPU cores and 48 GiB. Most optimization
measurements compare pinned older Splash revisions. Main subsequently merged
adaptive verification in [#487](https://github.com/incoai/splash/pull/487) and
changed scheduling and kernel costs. Historical gains or regressions do not
describe the effect of merging these tools into current main.

## Measured decisions

| Experiment | Recorded outcome and decision |
| --- | --- |
| Selective MoE verification | The initial older-baseline experiment measured 6–7% native gains and 6.52% on one long-code HTTP workload. The expanded 72-comparison matrix had 30 gains above individual bounds, six inconclusive comparisons and three differing-hash cases. Main's independently merged #487 supersedes that runtime prototype. |
| Pending-tape GDN pruning | Shader/model correctness checks passed; serving/native gains did not meet promotion thresholds. The runtime prototype stays archived. |
| Cost-aware verification | Eighteen held-out comparisons of an early #487 revision had eight gains, eight inconclusive outcomes and two stable regressions. Later merged code differs; these are not current-main regression claims. |
| Host allocation reduction | Removed one allocation per dispatch, but awake serving repeats showed no qualified throughput gain. The runtime prototype stays archived. |
| Context-aware prefill | Prediction error improved. MoE 2k long-request TTFT improved 12.7–14.5%, while maximum active-stream gaps worsened 41.4–45.0%. The latency tradeoff failed promotion. |

For complete sources, gates and counterexamples, start with the archived
[decision ledger](https://github.com/linson007/splash/blob/3e8911fc89a6ed009e7b0561bfa7c23e58ea594f/dev/benchmarks/deepinfer-qwen-completion.md).
The [current-main validation bundle](https://github.com/linson007/splash/tree/3e8911fc89a6ed009e7b0561bfa7c23e58ea594f/dev/benchmarks/results/qwen-current-main-m5-pro)
retains all sixteen executed model/KV/state cells: fifteen passes and one
original penalty-oracle assertion failure. The separate test correction in
[#526](https://github.com/incoai/splash/pull/526) passes the failing invocation
and four representative final-source cells; it is not a model speedup or a
rerun of all sixteen corrected cells.

## New measurements

Use [the serving guide](serving-measurements.md) and
[maintainer recipe](qwen-maintainer-runs.md). Current policy comparisons use
supported `--verify-rows all|auto`, actual native overlap and complete output
budgets. Other-machine jobs remain unexecuted until reports are returned;
historical results do not qualify a shared performance default.

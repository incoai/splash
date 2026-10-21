# Fixed versus adaptive verification policy on M5 Pro

**Historical data:** these measurements apply to the source/build identities in this bundle. Current main has since merged a different runtime stack; see [the source reconciliation](../../current-main-audit.md). They are not current-main qualification.


This evaluates step 4 of [the roadmap](../../deepinfer-qwen-plan.md) against
the existing [PR #487](https://github.com/incoai/splash/pull/487), at immutable
revision `67bb94069179470125d393da5e5725e3f1501f9d`. **The held-out promotion
gate is not met:** both storage paths have a stable regression against their
selected fixed control. Keep the roadmap's experimental/default-off policy;
these data do not justify broad adaptive promotion on this device.

The independent full-width MLX reference crossed sleep and is excluded. Its
awake replacement is complete in `adaptive-full-reference-mlx-awake.json`; all
three trials remain inconclusive (8.62–19.91% noise). The apparent 10.53–21.22%
cost reductions therefore do not qualify a benefit. Its UTC/awake discrepancy
is below one second. This extra reference does not change the held-out
negative decision.

## Source and controls

The upstream branch changes physical verification work and speaks protocol 11.
All policy comparisons therefore use its own frozen executable and server in
both arms, rather than mixing its dependency stack with main/protocol 10.
The [qualification patch](../../patches/pr-487-fixed-budget.patch) adds
`SPLASH_BENCH_FIXED_VERIFY_ROWS=1..8` for unconstrained all-greedy MoE batches.
It caps the existing selector's maximum rows and requests its complete capped
prefix. With that variable absent, the upstream adaptive choice is unchanged.
The patch reverse-checks cleanly, and the upstream `verify-budget` CPU test
passes. Online fitting overhead remains present in fixed controls.

The inspected upstream model uses conditional acceptance
`logit(p) = slope*logit(max-candidate-confidence) + intercept[position]`.
It refits with a Gaussian prior every 64 cycles over up to 4,096 observed
labels. The prior slope is 0.7 with position-specific intercepts. Its cost fit
is linear in total verified rows, separated by lane count and constrained
status, with decay 0.97, a five-cycle prior and bounded outlier updates.
Every 32nd cycle explores full width. Model state is runtime-local.

The confidence feature differs from this roadmap's cumulative selected-path
heuristic; `confidence-gguf.json` and `confidence-mlx.json` calibrate that heuristic, not
the upstream adaptive model. The reference has 4,676 GGUF and 4,452 MLX labels
with no censoring; candidates have 22 and 52 censored records respectively.
Candidate suffix censoring is excluded rather than labelled rejection. The upstream cost input is command wall time; the
benchmark's primary metric includes complete native decode-cycle gaps too.
The inspected cost table has no explicit context class. That is a possible
reason to investigate transfer across contexts, not a demonstrated cause of
the regressions below.

## Training and held-out protocol

Training sweeps fixed rows 1–7 against eight rows on two 2k code prompts,
256 output tokens per request, four ABBA arms and matched warmup histories.
Selection minimizes mean candidate/reference cost among complete training
reports with every trial's noise <=5%, with full width as an available control
of relative cost 1. Training and held-out prompts remain separate. GGUF rows
3/5/7 are excluded for excessive spread; row 5 also crossed sleep. All those
reports remain attached. The selected controls are **8 rows for GGUF IQ3_XXS**
and **6 rows for MLX 4-bit**. These are selected measurable controls, not a
claim that inconclusive settings or every workload have been optimized.

Held-out workloads are 2k math/prose and 8k code, three trials each, with 512
output tokens. One resident server runs each complete ABBA round; it closes
before the next. Every round has the same four 2k/512-token warmup requests
and workload history. Both sides retain online fitting. MoE MLX uses a 26 GiB
cap, GGUF 20 GiB. Every request must be cold and produce the full output budget.
No tracing or shader validation is enabled for timing. All model, binary,
metallib and tool hashes, actual counters and environment snapshots are retained.

## Held-out results

| Target / context | Trials | Gains above gate | Inconclusive | Stable regressions |
| --- | ---: | ---: | ---: | ---: |
| GGUF / 2k math and prose | 6 | 5 | 1 | 0 |
| GGUF / 8k code | 3 | 0 | 2 | 1 |
| MLX / 2k math and prose | 6 | 3 | 2 | 1 |
| MLX / 8k code | 3 | 0 | 3 | 0 |

The GGUF 8k code sample 2 loses **8.03% throughput** at **1.48% noise**
(3.0% cost-regression bound after rounding). The MLX 2k prose sample 0 loses
**3.95% throughput** at **1.24% noise** (2.48% bound). Those failures are outside
the regression tolerance; positive trials cannot erase them in an aggregate.
All 18 held-out comparisons contain differing greedy output hashes.
The upstream policy documents timing/history-dependent outputs; these finite
performance/probe checks do not establish quality or sampling-distribution
preservation. This evaluation's primary workloads are greedy only.

The original additional MLX full-eight-row reference has three inconclusive
trials and crossed sleep. `qualification_validity.usable=false` excludes it
regardless of any apparent gain. The completed awake repeat is attached separately and remains inconclusive.
Its strict-JSON probe hashes match; the four seeded sampled probes have three
distinct hashes. Probe execution is not a claim of sampled output equality or
distribution preservation. The fixed-depth qualification override applies
only to unconstrained all-greedy batches; sampled probes retain upstream
adaptive behavior in both arms.
The four held-out report intervals above are unaffected by the power audit.

## Evidence and interpretation

`policy-summary.json` identifies selected controls, every training artifact,
held-out counts, pending reference replacements and the control-patch hash.
The 14 training reports and four held-out reports retain every arm and sample;
generated answer text is removed and prompt text is represented by SHA-256.
Raw `complete` means a harness executed; qualification validity and ABBA
verdicts must also be checked. Early sequential reports have a generic legacy
`method` string; their `residency` field and four `server_runs` identify the
actual one-resident-server method.

Promotion requires gain > `max(2%,2*noise)` and no applicable correctness or
latency regression; noise >5% is inconclusive. A regression `pass` does not
itself mean a useful gain. Oscillation was not separately instrumented in this
upstream evaluation; the throughput gate already fails, so no stability or
rollout claim depends on an unmeasured policy histogram. Other Apple hardware
remains unexecuted in the maintainer package.

Reproduce using the immutable upstream checkout, control patch and its server
as described in [the maintainer note](../../qwen-maintainer-runs.md). Preserve
the recorded training/held-out split and full ABBA server histories. New runs
use process-scoped sleep prevention and reject a UTC/awake-clock discrepancy.

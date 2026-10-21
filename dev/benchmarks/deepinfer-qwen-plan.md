# DeepInfer lessons and Qwen optimization plan

**Current integration: [PR #523](https://github.com/incoai/splash/pull/523).**
Upstream advanced substantially during this investigation. The contribution now
contains current-main benchmark tools and source-pinned experimental evidence;
it changes no production runtime or default. See [the completion/review ledger](deepinfer-qwen-completion.md), [current-main
source audit](current-main-audit.md) and [qualification recipe](qwen-maintainer-runs.md).
Runtime instructions in the original design below describe the frozen legacy
sources, rather than settings supported by today's main.


Splash should first measure its current execution gaps and evaluate the existing
greedy MoE live-row prototype against the latest runtime. A general rewrite for
variable verification shapes should follow only if measurements justify it.
DeepInfer supplies useful scheduling principles, but its accelerator topology,
model architecture and trained confidence head differ from Splash's.

## Execution status

The roadmap is tracked in [#508](https://github.com/incoai/splash/issues/508).
Completing a conditional optimization means implementing and testing its
experiment, measuring its gate, and recording the resulting decision. A failed
speed or latency gate leaves that experiment out of the product. Historical
runtime PRs #507/#509/#510/#511/#522 are preserved separately; #523 is the
reviewable contribution based on current main.

| Step | Current status | Remaining gate |
| --- | --- | --- |
| 1. Runtime measurement | Complete: per-width operator tables, full-engine/mask/ANE traces, full trace archives and five display windows | Current-main functional/model validation is a separate pending scope |
| 2. Greedy MoE prefixes | Complete: 72 ABBA trials across both formats/contexts/widths; three exact-prompt disabled-path replays match | Upstream #487 supersedes the old runtime prototype; no current threshold flag |
| 3. Pending-tape GDN pruning | Complete: 128 shader combinations, real-model oracles, native/HTTP controls; promotion gate not met | Keep historical prototype off; no current runtime change |
| 4. Cost-aware policy | Complete: 14 training reports, 18 held-out trials and awake full-width reference against #487 revision `67bb9406`; held-out promotion fails | Later merged #487 has different code; no current-main regression claim |
| 5. Host overhead | Complete: one allocation saved per dispatch, ownership/GPU checks and awake serving repeats; no qualified gain | Retain runtime baseline; no larger reuse pipeline justified |
| 6. Mixed prefill/decode | Complete: 30 production comparisons, causal prediction analysis, cold/cache/continuation and actual-path traces; scheduler ASAN/UBSAN/TSAN pass | Latency gate fails: larger ragged commands improve long TTFT but worsen maximum stream gaps; retain baseline |

All six historical local experiments now have measured decisions. The
current-main native build, CPU/sanitizer suites and full Python check also pass
(1,348 tests, 27 skips; lint/format pass). The fresh four-model KV/state oracle
matrix and four functional all/auto HTTP checks are still executing. A fresh
MoE GGUF BF16-KV/FP32-state cell failed the upstream cold-score penalty
assertion; the original failure is retained while a same-forward test oracle
is verified separately. These results are not inferred from historical runs. The [review ledger](deepinfer-qwen-completion.md) links
the scopes and evidence.

Only this M5 Pro is available for local runs. Other GPU families will receive
[a repeatable maintainer-run package](qwen-maintainer-runs.md); those tests remain unexecuted until their
results exist. Completing the local experiments does not qualify a shared
default or establish missing hardware coverage.

Reviewed on October 20, 2026, against these immutable source identities:

| Source | Revision |
| --- | --- |
| Splash main | `be3b083c5f413673d44e44881042e875e674e333` |
| DeepInfer | `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` |
| Existing Splash branch `pr-283-live-verify` | `1f82545359898626a84e1e1058253d6bc47f4557` |
| Existing stash on that branch | `440f39ba31e4d10ab260859370f1334f36c063f7` |

The findings below describe the frozen revisions above. The subsequent
[local implementation and evaluation](deepinfer-qwen-results.md) ports the MoE
prototype to `deepinfer-qwen-eval`, adds native timeline and threshold-sweep
tools, and verifies both installed Qwen families. Threshold 0.1 improves the
qualified MoE measurements; broader HTTP qualification remains inconclusive,
so the feature stays disabled by default. The original branch and stash remain
unchanged. GDN pruning, adaptive policy and scheduler changes remain conditional
on the gates below.

## Findings confirmed against code

### Verification has fixed storage and execution shapes on main

[ExecutionLimits](../../runtime/model/Model.hpp) declares eight draft query
rows, seven proposals, eight target verification rows and four maximum lanes.
[ExecutionGeometry.h](../../runtime/metal/abi/ExecutionGeometry.h) repeats the
host and Metal ABI constants. `Runtime::decodeAsync` explicitly retains this
physical graph even when the remaining output budget is shorter.

This assumption spans `QwenTarget::addVerify`, `Linear::decodePlan`,
`GDN::addDecode`, `PagedAttention::verifyParams`, `MoE::decodePlan`, sampling,
and the decode arena. Linear validation accepts decode row counts in multiples
of eight; the GDN shader maps eight SIMD groups onto verification rows.

Reducing `maximumRetained` alone changes output retention, not target compute.
Supporting arbitrary compact row counts would therefore be an operator and ABI
change, rather than a scheduler setting.

Sources: [Runtime.cpp](../../runtime/model/Runtime.cpp),
[QwenTarget.cpp](../../runtime/model/QwenTarget.cpp),
[Linear.cpp](../../runtime/ops/Linear.cpp),
[GDN.cpp](../../runtime/ops/GDN.cpp),
[PagedAttention.cpp](../../runtime/ops/PagedAttention.cpp),
[Sampling.cpp](../../runtime/ops/Sampling.cpp),
[RuntimeArenas.hpp](../../runtime/model/RuntimeArenas.hpp).

### The existing branch already implements selective MoE work

Commit `80ea673` on `pr-283-live-verify` adds:

- `VerifyLiveRows`, a GPU-produced count and prefix-confidence vector per lane.
- `DraftSelector::addLiveRows` and the `draft_verify_live_rows` shader.
- MoE route grouping and combination that exclude inactive suffix rows,
  including their shared-expert routes.
- Acceptance restricted to the selected prefix, plus live-row telemetry,
  benchmark switches and focused Metal tests.

The implementation preserves eight-row storage and dense projections. It can
still remove real work: excluding expert routes reduces expert computation and
may reduce the number of expert tiles. Savings depend on tile occupancy and
expert overlap.

Its policy is deliberately limited to unconstrained sparse MoE batches with
greedy lanes. Sampled lanes remain full width, constrained batches and dense
targets bypass it, and threshold zero disables it. Its confidence is derived
from the selector's candidate and edge scores; it is explicitly a heuristic,
not a trained acceptance predictor.

Inspect the frozen branch with `git show 1f82545:PATH`, particularly
`runtime/metal/abi/LiveRows.h`, `runtime/ops/DraftSelector.cpp`,
`runtime/metal/kernels/decode/sampling.metal`,
`runtime/metal/kernels/shared/moe.metal`, and
`dev/tests/engine/live_verify_rows_metal_test.mm`.

### The stashed GDN pruning needs a redesign for current main

The stash adds live-prefix GDN execution and tests against the former
`GDN::addCommit` implementation. Current main instead keeps per-step GDN tapes;
the next step folds the previous step's retained rows into its incoming state.
`QwenStateStorage::swapDecodeParity` records those pending rows.

The stashed implementation and tests provide design material, but cannot be
transplanted as whole files. A new implementation must preserve pending-tape
consumption, convolution carry, tape ownership, and snapshot materialization.
It must also support main's FP32 and BF16 state storage.

Sources: [GdnTapeLane](../../runtime/ops/GDN.hpp),
[GDN decode shader](../../runtime/metal/kernels/decode/gdn.metal),
[QwenState.cpp](../../runtime/model/QwenState.cpp).

### Asynchronous submission exists and some host work already overlaps

`MetalBackend::submitCommandAsync` returns a completion ticket.
`Engine::tick` processes control events while a batch is pending, but waits
for that batch before applying its results and selecting another batch.
`Scheduler::next` also refuses a plan while `active_` exists.

Unconstrained decoding already combines draft, target verification, acceptance
and draft-context commit into one submitted graph. Constrained decoding has a
separate ticket that launches target forward while the host obtains grammar
masks. Describing Splash as lacking asynchronous execution or mask overlap
would be incorrect.

The remaining opportunity is to reduce preparation between batches, or prepare
safe portions of future work during execution. Device-side state transitions
would be needed for a more aggressive pipeline that starts the next decode
before CPU acceptance processing.

Sources: [Engine.cpp](../../runtime/engine/Engine.cpp),
[Scheduler.cpp](../../runtime/engine/Scheduler.cpp),
[`Runtime::decodeAsync` and `ConstrainedDecodeTicket`](../../runtime/model/Runtime.cpp).

### Profiling and prefill adaptation already exist

`decode-profile` measures fused GPU and wall time at B1 through B4 and can
attribute dispatch time. `/status` exposes both `decode_wall_ms` and
`decode_cycle_ms`; the latter includes time between commands. Existing
optimization notes already describe this distinction.

Dispatch attribution submits each dispatch separately and waits, so it changes
execution. Use ordinary fused execution and HTTP measurements for throughput;
use attribution to identify expensive operators. Wall time minus GPU time is
an aggregate overhead estimate, not a measurement of pure CPU scheduling cost.

Prefill already uses an exponential estimate of milliseconds per token, halves
contended row budgets toward a 500 ms target, and stops at 64 rows. It also
accounts for priority, starvation prevention, and decode time owed. The
remaining cost-model gap is that `observePrefill(rows, milliseconds)` has no
explicit context-length or execution-path features.

Sources: [decode_profile.mm](decode_profile.mm),
[BackendInstrumentation.hpp](../../runtime/metal/BackendInstrumentation.hpp),
[prior optimization measurements](remaining-decode-optimizations.md),
[Scheduler.cpp](../../runtime/engine/Scheduler.cpp).

### MoE fusion and reproducibility tests are established foundations

`MoE::add` already groups routes and launches expert passes through indirect
grids containing live tiles. Compatible staged GGUF paths combine gate and up
projections. Broadly proposing grouped or fused MoE would duplicate existing
work. Further changes need a measured routing, padding, intermediate-memory or
expert-kernel bottleneck.

The model oracle already checks same-graph lane equivalence, permuted lanes,
multiple batch widths, restore, preemption, sampled reproducibility and mask
overlap. It explicitly treats B1 versus B4 output equality as diagnostic because
different numerical graphs can change acceptance and subsequent tokens.
Request-local random streams do not establish exact batch invariance.

Sources: [MoE.cpp](../../runtime/ops/MoE.cpp),
[model_runtime_oracle_test.mm](../../dev/tests/engine/model_runtime_oracle_test.mm),
[target_sampling_metal_test.mm](../../dev/tests/engine/target_sampling_metal_test.mm).

## DeepInfer mechanisms that transfer

DeepInfer's [MTPVarlen scheduler][deepinfer-policy] selects a total query budget
using expected output and measured forward/sampler costs. It separates older
confidence used to select that budget from more recent confidence used for
per-request allocation. The source explicitly prevents later draft confidence
from influencing whether an earlier draft is verified. Its
[device allocator][deepinfer-allocation] distributes rows using cumulative
probabilities. These are stronger semantics than a single probability cutoff.

Its [worker pipeline][deepinfer-worker] launches current forward before
processing previous results. GPU correction carries previous sampled tokens,
acceptance counts and metadata into the new step. Copying the CPU loop alone
would not reproduce this behavior in Splash.

The [DeepInfer draft][deepinfer-draft] loads a trained confidence projection.
Splash's current `DFlashDraftWeights` and loader have a selector projection and
codebooks but no equivalent acceptance-confidence head. Any new head would
require compatible training and checkpoint support.

DeepInfer's [release scope][deepinfer-readme] is offline DeepSeek inference on
Linux with NVIDIA SM100 or Ascend 950 hardware. Its production cache service
is not fully included. Its [deployment Q&A][deepinfer-qa] describes approximate
SWA replay tied to its architecture and gives a low-concurrency example where
dynamic verification loses throughput. Neither its replay method nor its
reported speedups should be assumed to transfer to Qwen on a Mac.

## Implementation sequence

### Step 1 Measure the current runtime

Extend the existing benchmark tools before changing scheduling or kernel policy.
Record ordinary fused cycle time, GPU time, committed output tokens, acceptance
by draft position, and CPU preparation/completion intervals. Keep token counters
distinct: proposed, eligible for verification, actually verified, accepted, and
committed. The old branch's live-row count describes eligibility, not all rows
executed by its unchanged target operators.

Add optional timestamp tracing around engine result application, planning,
resource preparation, runtime graph construction, Metal submission and
completion. Use GPU timing or an aligned device trace to confirm idle gaps;
host completion notifications can arrive after GPU work ends. Trace constrained
mask waits and ANE event waits separately.

Use `decode_cycle_ms` for serving comparisons and retain `decode_wall_ms` for
command timing. Preserve the existing interpretation of both metrics.

Deliverables: benchmark JSON/CSV, a CPU/GPU timeline, and a bottleneck table for
each model and batch width. This determines whether host work, verification
suffixes, attention, or expert execution deserves the next optimization.

### Step 2 Reuse the greedy MoE prototype on current main

Create an isolated branch or worktree from reviewed main. Port the narrowly
scoped logic from `80ea673` and its tests; retain the original branch and stash.
Reconcile it with current MoE indirect grids, quantization support, sampling
contracts and arena accounting. Keep it experimental and disabled by default.

Preserve the fixed eight-row ABI, the eight-row draft and the sampled/constrained
fallback. Start with the MoE family. Excluding expert routes can save work
without introducing arbitrary matrix row counts across every operator.

Sweep threshold zero and a small candidate set, for example 0.01, 0.05, 0.1 and
0.2, as experimental values. Measure expert tiles, expert time, cycle time and
committed tokens per second. Reducing eligible rows alone is insufficient.

Gate: the disabled path matches baseline behavior, invalid confidence falls
back safely, live prefixes preserve token and state accounting, and selected
settings improve complete serving runs beyond measured noise. If shorter
prefixes reduce useful tokens more than execution time, stop or revise policy.

### Step 3 Adapt GDN pruning to pending tapes when profiling justifies it

Use the stash as a reference for selective row arithmetic and test intent.
Implement against main's `GdnTapeLane`, not the removed explicit commit path.
Keep maximum tape capacity and address geometry stable initially.

The current step must consume all previously retained pending rows even when
its own eligible prefix is shorter. Its output tape must contain every row a
future retention or snapshot can consume. Inactive outputs and any prepared
tables read by full-width consumers need defined contents. Preserve barriers
when inactive SIMD groups skip arithmetic.

Test transitions such as previous/current lengths 8/1, 1/8 and mixed per-lane
lengths, every retained prefix, both state formats, lane permutations, snapshot,
disk restore and preemption. Port applicable tests from the stash, then add
pending-tape-specific cases against an independent recurrence reference.

Gate: live outputs and materialized committed state match the reference under
the existing numerical contract, and a complete decode cycle improves. A GDN
kernel win alone does not justify enabling it.

### Step 4 Select a policy using measured costs

Begin with benchmark-selected fixed settings per model/device/workload and a
full-width fallback. Then evaluate a cost-aware choice using only information
available to the scheduling decision. Compare against both fixed eight-row
verification and the best fixed experimental setting.

Optimize committed tokens per total cycle time, including draft generation,
target work, selection overhead and host coordination. Estimate prefix survival
from held-out observations; selector probability is only a candidate feature.
Evaluate calibration separately for code, prose, temperature and quantization.

For a future curve cache, key by device family and core count, model/draft
identity, quantization, KV/state formats, kernel build and relevant execution
settings. Include context buckets and bounded online updates where measurements
show they matter. Start with benchmark artifacts before adding persistent
runtime calibration or startup work.

Keep adaptive truncation greedy-only initially. Sample-dependent truncation can
bias generation. Extending it to sampling requires a reviewed rule governing
which information may influence each prefix decision, correct proposal and
residual distributions, and statistical validation. DeepInfer's separate budget
and allocation inputs are a useful reference, not a correctness proof for
Splash's selector. A fixed seed is not a distribution-preservation test.

Gate: held-out serving runs beat the best fixed policy without oscillation or
material latency regressions. Otherwise retain the measured fixed setting.

### Step 5 Reduce measured host overhead

This work depends on Step 1, independently of the verification prototype.
First evaluate immutable graph-description reuse, reusable host containers and
cached execution plans or binding templates. Runtime-dependent pointers,
positions, page-table revisions and state parity must remain current.

Only then evaluate preparation overlap while the existing batch runs. Separate
preparation from mutations of the shared decode arena and request state.
Invalidate prepared work after cancellation, resource changes, mask responses,
or a different acceptance result. Queue admission and request priority must not
be committed from stale assumptions.

Launching a next decode before CPU result application is a separate, larger
project. It requires device-resident token/length propagation, explicit arena
ownership and potentially additional buffers. Measure the memory cost before
adopting it on constrained Macs.

Gate: a shorter measured inter-command gap produces a serving improvement;
cancel/deadline, mask overlap, cache restore, page lifetime and memory-pressure
tests remain correct. Preserve state transitions on the engine thread.

### Step 6 Improve mixed prefill and decode latency

Extend the current timing estimator only after collecting prediction errors
by new-row count, history length, target family and ANE use. Model actual
ragged batches rather than only the leading request. Try a small bucketed
estimator before a more complicated fitted model.

Retain priority, overtaking bounds, decode debt, the row ceiling and conservative
fallbacks for insufficient samples. Measure p50/p95 time to first token and
streaming gaps while a long prompt arrives beside decoding requests. Include
cold, cached and resumed prompts; do not fit restore or image costs as ordinary
text prefill without distinguishing them.

Gate: lower contended latency without starvation or a throughput regression
outside the existing benchmark tolerance. Choose any revised latency target
from these measurements rather than importing a cluster budget.

## Conditional followups

Dense target compaction and true 2/4/8-row matrix plans follow only if fixed
storage with selective work leaves a measured bottleneck. Keep draft geometry,
target logical rows, physical strides and padded matrix tiles distinct. A
two-row logical plan padded to eight arithmetic rows may save little.

Any such change spans linear plans and kernels, GDN, attention and KV staging,
head selection, sampling, captured hidden states, draft-context commit and arena
extents. Prove end-to-end behavior first in a B1 four-row target experiment,
then qualify B2/B3/B4 and other lengths. Avoid changing the shared draft constant
to simulate a shorter target. Dense and MoE results require separate decisions.

Further MoE fusion needs operator attribution and memory-traffic evidence.
Exact batch invariance needs an explicit product requirement and a separate
numerical/performance design. Neither is a prerequisite for testing live rows.

Lower-bit KV is a separate quality and capacity experiment. Qwen GDN state
cannot be replaced by DeepInfer's bounded SWA suffix replay. Distributed expert
replication, RDMA and prefill/decode disaggregation are outside this Mac plan.

## Validation and rollout

Start with both Qwen families, one representative safetensors target and one
GGUF per family. Cover B1/B2/B3/B4, approximately 2K/8K/32K contexts, and code,
prose and strict tool output. Fixed-context teacher-forced comparisons help
attribute numerical changes before autoregressive paths diverge. HTTP runs
must also report the actual committed output counts and acceptance behavior.

Use a current local Apple GPU first, then qualify on both Apple9 and Apple10
families before making shared defaults. Expand to long-context limits, a small
memory configuration and additional quantization formats after a candidate
passes the initial gates. Hold model bytes, prompts, memory limits, KV/state
formats, ANE setting and power conditions constant within each comparison.

Reuse [ABBA](abba.py), [backend_regression.py](backend_regression.py) and
[http_regression.py](http_regression.py). Existing regression tolerance is
`max(2%, 2 * measured noise)`; spread above 5% is inconclusive and requires a
rerun. Proposed promotion criterion: a repeatable primary-workload gain greater
than that same noise bound, plus passing regression checks on the agreed
workload set. Report affected-device regressions explicitly rather than hiding
them in an aggregate average.

Extend existing linear, GDN, MoE, attention, sampling and model-oracle tests
according to the changed paths. Include all-accepted and early-rejected prefixes,
EOS, token limits, thinking cutoffs, penalties, invalid scores and mixed sampled
lanes. For mask-related changes, exercise partial tool output and thinking
budgets over the API. Preserve documented numerical tolerances; cross-width
bitwise identity is not an existing acceptance criterion.

Run relevant focused tests during development, then the repository's CPU/Metal
suites and real-model oracle for each selected model. Include shader validation
for correctness runs and ordinary execution for speed. Run native build and
sanitizer checks when control-plane ownership or scheduler code changes.

Keep experiments independently switchable with a baseline fallback. Ship a
default only after correctness, useful output throughput, contended latency,
memory use and startup cost are established. The immediate implementation
deliverable is Step 1; the first verification candidate reuses Step 2's existing
MoE work. Steps 3 through 6 follow the measured bottlenecks.

[deepinfer-policy]: https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/csrc/engine/mtp_varlen.hpp
[deepinfer-allocation]: https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/layers/tilelang_ops/schedule_mtp_varlen.py
[deepinfer-worker]: https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/worker/worker.py
[deepinfer-draft]: https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/deepinfer/model_executor/models/mtp/dflash.py
[deepinfer-readme]: https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/README.md
[deepinfer-qa]: https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/doc/inference_deployment_qa.md

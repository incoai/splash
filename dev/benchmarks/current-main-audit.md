# Current-main reconciliation of the DeepInfer/Qwen plan

Rechecked October 21, 2026, at upstream
`1482f6bbc2abf12375332725302019594150b6cb`. The original experiments used
main `be3b083c5f413673d44e44881042e875e674e333`; their source/model/build
identities remain attached. They do not qualify the newer runtime.

During the experiments upstream merged #487 and the following runtime/release
stack. The reviewable current-main contribution therefore consists of benchmark
tools, qualification recipes and versioned evidence. It changes no production
kernel, scheduler, model API or default. The earlier runtime branches are
historical experiments, rather than patches to transplant into this release.

| Area | Reconfirmed current code | Consequence |
| --- | --- | --- |
| MoE depth | `Runtime.cpp` owns an optional `VerifyBudget`; `DraftSelector::addVerifyRows` selects rows from learned acceptance and cost. `QwenTargetVerifyBuffers::verifyRows` feeds the target. | The original fixed-threshold prototype overlaps merged #487. Its environment flag is not a current-main configuration. |
| Product control | `NativeArguments.cpp`, `RuntimeResources` and `serve_options.py` expose `--verify-rows all|auto`; `/status.verify_rows` reports effective behavior. | The new harness sets the option per arm and checks native status. Dense targets may report `all` under `auto`. No private patch is needed for the all/auto comparison. |
| Policy revision | Merged #487 head is `507f6f3fe1e28d6090bc1a8623b955ec5dc7bb0d`. It skips invalid confidence labels, unlike the evaluated earlier `67bb9406` head. Later merged kernels also change costs. | The old held-out regressions are evidence for their recorded revision; they must not be presented as current-main regressions. |
| Prefill cost | `Scheduler::observePrefill` separates fixed host-step cost from per-row cost. `attentionPairs` and `maximumAttentionPairs_` bound every ragged lane's query/key work. | The old single-scalar description is historical. The bucket prototype cannot simply replace the new estimator. |
| Streaming/passes | `prefillGroupRows`, `prefillPasses`, `passesFor` and `streamingRows` account for streamed experts/history and multi-pass commands. Decode can run between passes. | A future estimator must preserve those semantics and train on complete representative observations. The old 2,048-row-only prototype does not cover them. |
| Diagnostics | Status exposes `model_timing`, `constraint_masks`, streamed-resource diagnostics and host-step observations. | Reuse the merged diagnostics. The historical bounded full-engine trace implementation is not silently reapplied over the new backend. |
| GDN | The shader adds a sigmoid gate and a third packed layout (16512 columns) for Flash-Next. Ordinary pending tapes remain. | The old two-layout live-GDN experiment is not current coverage of the new layout/gate. Its old serving gate failed; no GDN flag is added here. |

The existing eight-row storage constants still do not mean every operator uses
the same amount of arithmetic. Current selective MoE work, logical verification
rows, physical storage and other operators' execution must be distinguished.
No general dense compaction or device-side next-step pipeline is implemented
by these benchmark tools.

Current code references: [Runtime.cpp](../../runtime/model/Runtime.cpp),
[VerifyBudget.hpp](../../runtime/model/VerifyBudget.hpp),
[Scheduler.cpp](../../runtime/engine/Scheduler.cpp),
[NativeArguments.cpp](../../runtime/engine/NativeArguments.cpp),
[Status.cpp](../../runtime/engine/Status.cpp),
[GDN shader](../../runtime/metal/kernels/decode/gdn.metal).

Use [the current qualification recipe](qwen-maintainer-runs.md) for new runs.
Only the available M5 Pro is locally exercised; the prepared jobs for other
Apple hardware remain unexecuted. Historical numerical evidence, pending
replacement runs and current-main functional validation are recorded separately.

# Foreground lane admission: replay feasibility

Source base: downstream production commit
`9753a803a8d7acc4caf6e164292ca279ea69d830`.
The initial characterization commit adds fake-model engine tests only. It changes no scheduler,
model runtime, service configuration, memory limit or cache policy. No native
fairness experiment is enabled or deployed.

## What the tests establish

`testForegroundLaneCapacityDoesNotRotateStreamingPeers` submits four independent
foreground streams, waits until all are streaming, then submits four more.
Twenty engine ticks do not admit the waiting streams or suspend a resident.
Cancelling one resident permits a waiter to enter; all remaining requests finish
with their requested output budgets. This distinguishes unavailable native
lanes from the gateway's previously fixed missed-wakeup delay.

`testStreamingSuspensionReplayCostGrowsWithGeneratedHistory` exercises the real
Engine with the existing deterministic Executor. A lower-priority victim is
suspended through the normal resource-admission path after progressively longer
output. It restores a 32-token prefill checkpoint, replays committed history,
and preserves the output anchor/count contract across resume. Total prefill
rows must equal both original prompts plus exactly those replay rows. The
replay must grow with output history; the victim still emits exactly 192 tokens
and reports matching usage.

A command already in flight can finish between the threshold event and the
suspension. Accounting therefore uses the actual supplied replay history and restored
boundary, not the earlier threshold. Treating the threshold as the exact pause
boundary understates replay work.

`testCancelledStreamingContinuationDoesNotResumeOrEmitAgain` cancels the victim
both while suspended and just after lane reacquisition. It checks that no token
is emitted after cancellation, no continuation is resurrected, the unrelated
request finishes, and model/cache active ownership is released.

These tests use the fake model's constant token 42. They establish scheduling,
row accounting, anchor/count continuity and cancellation behavior, not numerical
Flash-Next equivalence, real latency, throughput or coding accuracy.

## Why simple equal-priority preemption is not ready

`Engine::preemptBelow` currently permits only strictly lower-priority victims.
Changing that comparison would admit equal-priority waiters but make streaming
requests pay the replay demonstrated above. No throughput benefit follows from
that change alone.

There are also two separate checkpoint limitations:

1. `QwenStateStorage::snapshot` requires committed, page-aligned lengths and a
   current complete draft context window. Decode updates draft rings while the
   context window can remain behind. Prefill cache snapshots do not represent
   arbitrary decode safe points.
2. Decode can retain pending GDN rows outside its state cells. The next decode
   folds those rows from per-state-lane slots in the shared decode arena.
   `QwenStateStorage::requireCommitted` rejects snapshots with pending rows.
   Moving the GDN cells alone would not preserve the continuation when another
   request reuses that arena slot. Existing `RuntimeInstrumentation::pendingGdnRows`
   illustrates the separate tape fields, but is not a production resume API.

## Next implementation boundary

Prototype a complete RAM-only continuation capsule before a scheduling policy.
At a fully drained command boundary it must retain or copy both GDN parity cells,
pending tape rows and metadata, private draft rings/window, KV ownership, anchor,
RNG/sampling/constraint continuation and request-local planning state. Restore
must rebuild any lane-indexed views without executing an extra sampling step.
Keep parked KV referenced; ending its cache request as ordinary suspension does
would require a different ownership contract.

All retained buffers and extra incoming-lane allocation must remain charged to
the existing governor. Refusal leaves the resident untouched and scheduling
unchanged. A parked capsule must not silently spill to SSD, exceed the memory
limit, or survive cancellation. Do not create a new activation flag until there
is a complete tested path to guard.

Exact resume parity tests must include nonconstant generated tokens, pending
GDN rows, both parities, partial speculative acceptance, RNG state, constrained
masks, cancellation at every transition, lane-index rebinding, allocation
failure and cleanup. Fake-model tests cannot substitute for those Metal tests.

Only after that boundary is qualified should an opt-in one-borrow policy be
considered. Protect streaming progress, bound each borrow and resume debt, and
prevent same-tier cycling/starvation. A separately reviewed matched workload
comparison must measure time to 32 visible tokens, inter-token gaps, completion
latency, aggregate throughput, correctness and SSD writes alongside p95 TTFT.
Do not qualify a one-token latency improvement that merely parks the stream.

## Validation

The three new characterization cases and the complete fake-model Engine binary
pass: **458 passed, 0 failed**. Built with the repository target and its
`-Wall -Wextra -Werror` flags. See `foreground-lane-replay-tests.json` for hashes,
commands and read-only production verification. No Metal numerical tests or
real-model benchmarks were run, and nothing was deployed.

The subsequent RAM-only Runtime prototype and its targeted qualification are
documented in [ram-continuation-prototype.md](ram-continuation-prototype.md).
Engine rotation and production activation remain absent.

# RFC 193: a scoped host scheduling experiment

Related issue: [#193](https://github.com/incoai/splash/issues/193).
Rechecked against main `1482f6bbc2abf12375332725302019594150b6cb`.

## Current runtime and remaining proposal

Main already reduces per-submit host work and batches buffer bindings.
[`MetalBackend`](../../runtime/metal/MetalBackend.mm) has asynchronous command
completion, host steps and a host-step worker that requests
`QOS_CLASS_USER_INTERACTIVE`. Its worker QoS request is distinct from an
engine-thread work interval around graph construction/submission. No
WorkInterval integration or `SPLASH_ENGINE_WORK_INTERVAL` option is implemented.

[Maintainer feedback](https://github.com/incoai/splash/issues/193#issuecomment-5975595014)
keeps the work-interval proposal open while asking for measurement after the
existing host improvements. Main has subsequently changed submission, host
steps, streaming and decode scheduling further. Earlier third-party timings
and the old RFC's host-gap assumptions are not Splash measurements of this
revision. A scheduling hint does not establish a P-core placement guarantee.

The unused flag/configuration header and tests that only asserted its strings
are removed from this documentation proposal. They did not call a scheduling
API, wrap an engine scope or demonstrate a performance improvement.

## Experiment design

First measure graph preparation, submission, completion and the interval
between commands with matching current builds. Keep GPU command spans and
host scopes separate; subtracting GPU time from wall time does not isolate
pure scheduler cost. Existing
[`decode-profile`](../../dev/benchmarks/decode_profile.mm) supports native
operator/cycle diagnostics, and the [engine](../../runtime/engine/Engine.cpp)
and [scheduler](../../runtime/engine/Scheduler.cpp) define the serving loop.

If a material host scheduling delay remains, an opt-in implementation can
compare a bounded hint around engine preparation/submission against the same
code without the hint. API choice, supported OS behavior, ownership and
cleanup remain design work. Include success, exception, cancellation and
shutdown paths; do not retain a hint over blocking waits. Keep the native
protocol, model geometry, memory plan and kernel code unchanged for the
experiment.

Use matched warmups and alternating controls, report B1 through B4 and full
serving latency/throughput, and record power/thermal conditions. Include
competing UI/audio work and both hint setup/teardown cost and host/GPU gaps.
A host-phase improvement alone is insufficient to change a shared default.
Output hashes should be reported; identical arithmetic kernels do not prove
identical mixed-serving batches or output trajectories.

## Review and validation scope

This is a documentation-only RFC. There is no implementation, active flag,
new timing result or speedup claim in this PR. The current production path
and defaults are unchanged. The proposal remains experimental until an
implementation and measurements justify it; it is independent of main's
adaptive MoE verify-row policy in #487.

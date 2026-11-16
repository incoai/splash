# RAM-only decode continuation prototype

The existing resource-suspension path releases lane state and replays committed
history. This prototype adds a separate native Runtime capability to retain a
decoder in RAM and resume without replay. Engine admission does not call it;
there is no protocol, CLI or production activation path. Nothing is deployed.

## Retained state and accounting

`QwenStateStorage::ParkedLane` owns both recurrent-state parity cells, including
family auxiliary state, private draft rings, context window and logical metadata.
It moves ownership rather than copying those large buffers. Pending GDN rows
live separately in the shared decode arena: Runtime packs only retained rows,
field by field, into a governor-admitted Metal buffer. The same buffer holds
sampling-penalty words. Its actual allocated bytes remain in the state's ledger.

Request-local anchor, sampling RNG, thinking/constraint continuation, token
history and draft planning stay in Runtime's request entry. Restore rebinds the
state to a free lane, restores tape/penalty words and invalidates GPU page-table
bindings so the next command rewrites them even at an unchanged page revision.
The caller must retain the request's KV-page references throughout parking.

Both operations reject an outstanding backend command. Parking also rejects
unfinished prefill, invalid results, masked verification in progress, an absent
anchor and terminal/budget-exhausted anchors. Allocation refusal or a throwing
scratch-save callback leaves the source assigned. A full-lane restore keeps the
continuation intact for retry. Ordinary replay resume rejects parked requests.

Cancellation destroys the capsule and releases its buffers directly; it does
not allocate pool-vector capacity during cleanup. Storage-level handles can
outlive storage while keeping their owned allocations alive, and release them
without touching a destroyed storage object. No checkpoint-file write is part
of park or restore.

## Targeted qualification

- All 458 fake-model Engine tests pass, including eight-user lane saturation,
  suspension/replay accounting and cancellation characterization.
- Storage unit cases pass with FP32 and BF16, nonconstant bytes in both cells,
  auxiliary state, draft/window buffers, pending-row metadata and scratch.
  Coverage includes different-lane rebinding, owner/storage/occupied-destination
  rejection, consumed handles, memory refusal, callback exceptions, alignment
  overflow, zero scratch, cancellation and a handle outliving storage.
- Numerical unit cases use generated synthetic weights, never installed model
  weights. Dense Qwen3.8 and Flash-Next architectures each pass in FP32/BF16.
  The Flash-Next fixture uses executable 2560-wide hyper-connections and eight
  experts, PLE and hot KV readers. Fixed verification rows isolate continuation
  parity from adaptive verification decisions. Sampled, nonconstant tokens,
  anchors, recurrent/draft state and speculative counts match uninterrupted
  decoding after two rotations covering both state parities. Another request
  overwrites the vacated lane between park and restore. Busy-command and
  full-lane refusals, terminal-anchor refusal and cancellation are checked.
- The repository architecture checker passes on Python 3.14.

These are correctness unit tests, not a performance comparison. Synthetic
weights and the exercised draft acceptance do not qualify all real Flash-Next
sampling, thinking or constrained-generation outcomes. Tests with active hot
readers retain full resident KV pages; they do not qualify an actual spilled-KV
read/reclaim failure across parking.

## Activation gates

A scheduling policy is intentionally absent. Native Engine source currently
uses 30102 production lines of its 30105-line architecture budget. Adding a
second suspension/admission mechanism without simplifying existing ownership
would both exceed that gate and risk bypassing resource, cache and cancellation
contracts. Do not raise the budget or add an exception for this experiment.

Before integration, agree on a shared lifecycle for replay suspension and RAM
parking, including KV ownership, deadlines, claim accounting and page movement.
Then test bounded eight-user rotation and starvation, partial/full speculative
acceptance, constrained/thinking continuation, resource denial and actual
spilled-KV failure paths. Keep that policy disabled until matched coding tests
show lower p95 TTFT and time to 32 tokens without unacceptable inter-token gaps,
completion/correctness regression, extra SSD writes or throughput loss.

## Sanitizers and reproducibility

The uniformly instrumented native library, construction-only control, storage
unit cases and all four numerical cases pass ASan/UBSan with container checking
enabled. An initial partial-instrumentation build reported a libc++ vector
container overflow before parking; a construction-only control reproduced it
with the mixed objects, while the ordinary control and fully instrumented
control passed. The dedicated `make test-ram-continuation-sanitizers` target
builds a separate complete instrumented library to prevent that test-build error.
It does not disable container checking. Leak detection follows the repository's
existing macOS sanitizer settings.

The default Metal compiler was unavailable on both test hosts. No shaders changed:
executables were built locally and unit tests ran on the M5 against the unchanged
matching installed metallib, read-only. Full `make check` was not run; the
architecture checker was run directly with Python 3.14 after the make target's
environment bootstrap failed to fetch dependencies. These limits are separate
from the passing focused tests.

Production's ten health/readiness endpoints passed afterward, with the native
instance/identity unchanged, zero transport restarts, Samsung slots open on the
verified external volume, Flash-Next at 96 GiB RAM/512 GiB cache, eight queue
workers and admission/M4 disabled. Fusion and deployment were unchanged.
See `ram-continuation-tests-20261116.json` for sanitized evidence.

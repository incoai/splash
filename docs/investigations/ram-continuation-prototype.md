# RAM-only decode continuation prototype

The existing resource-suspension path releases lane state and replays committed
history. This prototype adds a separate native Runtime capability to retain a
decoder in RAM and resume without replay. Engine admission does not call it;
there is no protocol, CLI or production activation path. Nothing is deployed.

## Current baseline

The newest reconciliation targets upstream main `61f6d51` (1.7.0 source,
including 1.6.2), not a production upgrade. Earlier receipts below remain
historical evidence on their named 1.6.1 bases. The tree decoder uses a
`pendingPath` mask and at most 16 retained rows; parking copies precisely those
physical tape rows while retaining the running FP32/BF16 cells exactly, without
the cached-state codec. Grammar masks follow tree depth. Request-local lookup,
grammar, RNG and thinking continuation remain with the request.

Upstream decode-time checkpoints are enabled only without auxiliary state.
Flash-Next's PLE/QSA state excludes it: the concrete Runtime capability check
asserts this in both state formats. A controlled CPU characterization finds
8/52/140 replay rows without decode captures and 8/20/12 with captures at reply
thresholds 1/41/129. Those are fake-model work counts for the two capabilities,
not Flash-Next timings or a measured improvement of production. Captures may
also be refused under pressure. Equal-priority foreground streams still do not
rotate; the eight-request characterization remains unchanged.

The shared lifecycle, KV ownership and deterministic fairness acceptance matrix
are in [ram-continuation-scheduler-qualification.md](ram-continuation-scheduler-qualification.md).
It is a review specification, with no Engine scheduling implementation or
activation. Upstream's architecture budget is 30594/30594; this patch adds no
Engine production lines and does not raise that budget.

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

A scheduling policy is intentionally absent. The deployed-stack comparison
uses 30102 Engine production lines of its 30105-line architecture budget;
the refreshed upstream-main review base uses all 30105 lines. Adding a
second suspension/admission mechanism without simplifying existing ownership
would both exceed that gate and risk bypassing resource, cache and cancellation
contracts. Do not raise the budget or add an exception for this experiment.

Before integration, agree on a shared lifecycle for replay suspension and RAM
parking, including KV ownership, deadlines, claim accounting and page movement.
Then test bounded eight-user rotation and starvation, real-model speculative
variation, combined constrained/thinking continuation, resource denial and actual
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

## Focused upstream review branch

The patch was transplanted onto refreshed upstream main `d8af105` without the
earlier cache/counter patches. Its full fake-model Engine binary passes 458/458;
storage cases and all four synthetic numerical cases also pass on the M5. The
architecture checker passes at 30105/30105 Engine production lines.
`Runtime.cpp` is byte-identical to the uniformly ASan/UBSan-tested deployed-stack
variant. ASan/UBSan were not repeated on this separate upstream base; the
base-specific receipts keep those validations distinct. See
`ram-continuation-upstream-tests-20261116.json`.

The final numerical driver also includes a greedy control accepting all seven
draft tokens, exercising the full retained block across a park/rebind/next-step
sequence. These controls pass for both architectures and state formats in the
normal upstream build and with the uniformly instrumented deployed-stack native
library. Intermediate speculative-acceptance patterns and real-model variation
remain separate activation gates.

## Second correctness cycle

The expanded driver passes 24 partial-acceptance scenarios: one through six
accepted drafts (two through seven pending GDN rows), for dense/Flash-Next in
FP32/BF16. Each forces acceptance through actual grammar masks, parks and rebinds
the request after a different request overwrites its old lane, checks exact
state bytes and verifies the next masked output. It also verifies that an
initial-mask waiter and a verify-mask waiter with an idle GPU cannot park.

Another 16 scenarios use thinking budgets 0, 1, 5 and 12. They pause immediately
after prefill with no pending GDN rows, preserve the anchor/budget continuation,
and emit the two-token thinking-end sequence exactly once and in order. Along
with the sampled and full-acceptance controls, the driver now exercises pending
GDN counts zero through eight.

Both normal upstream binaries and the uniformly ASan/UBSan-instrumented library
pass all new scenarios. The latter uses matching deployed-stack substrate
headers, with byte-identical Runtime and parked-storage implementations; this
is not a claim that the full upstream library was rebuilt under sanitizers.
The architecture checker remains green at 30105/30105 Engine lines.

No runtime patch was necessary: only tests and evidence changed. These runs used
synthetic weights in observed production-idle windows. They are not real coding
benchmarks; combined constrained/thinking behavior, actual spilled-KV failure
paths and scheduler integration remain activation gates. See
`ram-continuation-cycle2-20261116.json` for hashes and counts.

## Third correctness cycle

Added 32 combined grammar/thinking cases: greedy and sampled decoding, thinking
budgets 0, 1, 5 and 12, dense/Flash-Next and FP32/BF16. Each parks twice: inside
a grammar-delayed thinking-end sequence, then with the first end token selected
but not committed. Another request overwrites the vacated lane before restore.
Exact state, subsequent tokens and speculative counts match the uninterrupted
control; the two-token end sequence appears exactly once and in order.

All 32 pass normally on the upstream candidate and under the same uniformly
instrumented ASan/UBSan library described above. The previous 40 partial/thinking
cases and existing controls also pass again. A temporary, test-only Runtime clone
that deliberately drops request 902's thinking continuation fails at the expected
next-output parity assertion (exit 1), while the positive driver exits 0. This
checks test sensitivity; it is not a discovered production bug or runtime fix.
Container checks remain enabled; architecture remains 30105/30105.

These tests close the synthetic combined grammar/thinking gap from cycle two.
They do not close the spilled-KV boundary: the active hot-reader fixture retains
Full resident pages. Existing oracle helpers serialize pages before moving them
to Index form and retain dense-prefix/window pages; a future continuation fixture
must preserve those ownership rules and exercise read failures explicitly rather
than merely attach a hot cache. Physical SSD failure behavior remains separate.
Engine scheduling, fairness and real-model verification variation also remain
unqualified, so a RAM-continuation TTFT benchmark would be premature.

Production verification after the runs passed all ten health/readiness checks.
Native identity and instance remained unchanged, with zero transport restarts;
Samsung cache files stayed open on the verified external volume. Flash-Next,
96 GiB RAM, 512 GiB cache and the disabled flags were unchanged. No deployment
or real-model performance benchmark occurred. Source/binary hashes and sanitized
acceptance are recorded in `ram-continuation-cycle3-20261116.json`.

## Fourth correctness cycle: spilled reads

The new opt-in synthetic attention fixture uses deterministic bounded nonzero
weights for attention, embedding/output and the indexer, with unit norms. GDN,
FFN and PLE projection weights remain zero; this is not a trained-model test.
The existing default fixtures and production code are unchanged. Zero target
weights were rejected for this test because they selected only tied resident
keys and made no copy reads; generic quantization-fixture weights were also
rejected because their extreme scales produced non-finite long-prefill rows.

For each FP32/BF16 state format, the driver runs a successful cold-copy read,
an injected failed cold-copy read, and a fresh successful request after ending
the failure. It prefills 2208 tokens, decodes once to leave speculative GDN rows,
serializes four committed history pages with SlotFormat, then moves those pages
to Index form. Dense-prefix and the trailing full page stay resident. Old Full
page K/V is poisoned only after the move, following the existing spill oracle.

Parking releases the request's hot-cache floor; reclaimable cache extents are
released. Another request overwrites the vacated decoder lane before restore
into a different lane. Exact retained state matches, the restored request
reacquires its hot floor, and its new page-table revision identifies Index pages
with retained TestKvCopy ownership. Successful results match a resident control's
non-tied logits bit for bit, tokens and speculative counts. Each successful case
makes 16 copy reads; each failed case makes 32 attempted reads. Failures identify
only that request's source blocks and return no output tokens. Failed requests
are ended, as Engine requires; no unsafe in-place retry is attempted.

The six scenarios pass normally and with the uniformly instrumented ASan/UBSan
library described above, including the non-tied-logit guard. All prior driver
cases pass again. Architecture remains 30105/30105. The read-failure injection
uses in-memory copies through the actual spilled-KV reader; physical Samsung IO,
corruption/crash recovery and latency are not qualified by these tests.

The last commit's CI sanitizers and Python 3.12/3.13/3.14 passed. Its primary job
failed before compilation while downloading Apple's Metal toolchain catalog
(exit 70). This account cannot rerun the upstream job; this test commit triggers
fresh CI. Keep the PR draft pending clean CI and lifecycle/design review. Engine
scheduling, fairness, real-model verification variation and matched performance
qualification remain open gates; no TTFT or throughput improvement is claimed.

See `ram-continuation-cycle4-20261117.json` for hashes, test counts and production
acceptance. All ten production health/readiness checks passed again, with the
same native identity/instance, zero transport restarts and Samsung slot files
open on the verified external volume. Flash-Next, 96 GiB RAM, 512 GiB cache and
the disabled flags stayed unchanged. No runtime code or production settings
changed in this cycle.

## 1.7 tree-contract qualification

The current base passes 480/480 CPU Engine tests. Normal M5 and fresh full-library
ASan/UBSan runs pass dense/Flash-Next in FP32/BF16, including 15 accepted proposals
and all 16 retained rows, 24 partial-acceptance cases (one nonconsecutive path per
configuration), 16 thinking and 32 combined grammar/thinking cases, and six
spilled-read scenarios. Masks follow parent depth; exact grammar park points use
one-row steps before restoring ordinary tree choice, and thinking tests run to a
bounded token goal rather than assume three cycles finish the budget.

Four additional byte-ownership cases put canaries in a sparse pending tape path
(rows 0/3/15), overwrite the old lane/slot and verify exact restore elsewhere.
These requests end without numerically decoding the invalid canary bytes.
Storage tests use codec-aligned dimensions and preserve the same sparse metadata
path, with refusal, throwing-save rollback, cancellation and late-handle cleanup.

A uniformly instrumented temporary clone that incorrectly copies a consecutive
prefix of popcount rows instead of the selected physical rows exits 1 at the
expected state/tape parity check. The correct driver exits 0. No fault code is
committed; this is a test-sensitivity control, not a serving defect. The entire
native library and instrumentation were rebuilt on the current upstream base
with matching headers, container checks enabled and macOS leak detection off.

Matching 1.7 shaders were built with the locally downloaded Apple Metal compiler
component. The normal correctness run lacked the optional sibling host-step
keepawake shader and reported its latency warning; the sanitizer run included
that matching shader. No timing result is inferred from either. Physical SSD
faults, trained-model variation and Engine scheduling remain unqualified.

All ten production health/readiness checks pass after the runs. Native identity
and instance remain unchanged with zero transport restarts; Samsung cache files
remain open on the verified external volume. Flash-Next, 96 GiB RAM/512 GiB cache,
eight workers and the disabled flags are unchanged. No deployment, queue pause,
production restart or real-model benchmark occurred. Hashes and sanitized
acceptance are in `ram-continuation-upstream170-20261117.json`.

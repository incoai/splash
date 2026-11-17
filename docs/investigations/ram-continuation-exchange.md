# Allocation-free return to a full decoder lane

Local experimental follow-on to the dormant RAM continuation prototype, based
on `5efd09b` and upstream `61f6d51`. No Engine policy, Model-interface activation,
CLI serving option or production deployment is included. This is a correctness
qualification, not a TTFT or throughput benchmark.

## Why the free-lane primitive was insufficient

An ordinary parked request still needs an unassigned lane to return. If its
borrower is streaming and all four physical lanes are occupied, returning it
would require parking the borrower too. A fresh allocation could fail precisely
when the old stream must return. That is unsuitable for bounded foreground
scheduling.

The exchange preparation admits one tracked scratch bank containing two
maximum-size packets before it detaches the first owner. Each packet fits all
16 pending GDN rows and lane-indexed sampling penalties. Thereafter, the
outgoing packet is copied to the unused half, the incoming packet is restored,
and running-state ownership swaps without allocating host or Metal storage.
Both recurrent cells, draft rings/window, auxiliary state and pending metadata
move with their owner. Request-local RNG, lookup and grammar state stay attached
to their original request. Page-table bindings are invalidated on each switch.

A bank stays bound to its original physical lane. It retains that lane's
pre-reserved hot-reader floor; another lane becoming free does not migrate it.
Cross-lane exchange, nested parking, an undrained command, an unsupported decode
stage, terminal continuation or missing floor refuses without detaching owners.
Cancellation releases the bank directly. If the resident ends first, the bank
keeps the floor until restoration or its own cancellation; if the parked owner
ends first, the resident keeps the floor. A bank canceled before any borrower
starts gives its otherwise orphaned floor back.

The previous scratch helper created a new host buffer view while copying tape.
It now uses the arena's existing per-lane view, with an explicit parity offset.
The positive tests run return exchange with caller-thread `operator new`
configured to fail on its very next allocation, as well as critical governor
pressure. They check unchanged Metal bytes and denial counters.

## Qualification scope

The normal and uniformly instrumented native-library ASan/UBSan drivers cover
dense and Flash-Next targets in FP32 and BF16. Each configuration executes
34 exchanges with all physical lanes occupied, sampled token and penalty-state
parity, critical-pressure returns, initial allocation refusal and cancellation
in both orders. Cross-lane refusal and restoration with a lower-numbered free
lane protect physical-floor ownership. Existing compact continuation, partial
path, thinking/grammar and successful/failed spilled-reader cases remain in the
driver. Sanitizers retain container checks; macOS leak detection is disabled.

The development oracle adds `--continuation-exchange` to
`--continuation-only --continuation-prompt TOKEN_FILE`. The installed trained
Flash-Next weights pass greedy/sampled cases at fixed 8/16-row budgets on a
7,379-token public-repository coding context. A baseline and a borrower control
retain separate physical lanes, a filler occupies the fourth, and two streams
exchange the remaining lane. Both streams' subsequent tokens, target-logit
digests and speculative counts match their controls; both tape parities and
exact state/window/anchor restoration are checked. Presence/frequency penalties
are active. Return exchange changes neither Metal allocation nor the governor's
denial counter.

Each case matches 35/36/32/38 candidate tokens respectively (141 total). Two
return observations per case have means 0.726/0.477/0.349/0.384 ms. A bank uses
44,793,856 bytes (about 42.72 MiB), in addition to the parked owner's retained
running state and KV. Initial preparation and subsequent reverse exchange are
mixed in the separately reported detach mean; that mean does not isolate bank
allocation cost. These are operation micro timings, not p95s or end-to-end gains.

The isolated oracle enforces 96 GiB RAM, BF16 state, int8 KV, exact cached-state
layout, 4 GiB PLE and 256K served context. Private resident test KV does not write
the Samsung persistent cache. The guarded idle-window controller restored the
unchanged signed production build, all ten health/readiness checks, Samsung
slots and volume identity, queue database/history, eight workers and disabled
settings. Configuration hashes, model identity, 96 GiB RAM and 512 GiB cache
were preserved. No new serving build was left deployed.

CPU Engine tests pass 480/480. The full local app suite passes 250 tests with
one existing known issue; its isolated CI-readiness test also passes locally.
The latest #778 CI still has one desktop-app readiness timeout, with native
sanitisers and Python/build checks green. No app product/test change was made
to hide a failure that has not reproduced.

## Gates before a user-visible benefit

This concrete Runtime API is not an Engine scheduling mechanism. The next
change must define a Model-level continuation contract and share terminal,
claim, cache-reference, deadline and pressure-recovery handling with replay
suspension. Parked requests must keep active KV ownership; today's `vacateLane`
ends it and cannot be reused unchanged. A resident paired with a parked owner
must not enter ordinary resource suspension without first resolving the pair.

A single global bank would help only the fifth of eight simultaneous requests.
Qualification must cover up to one bank per physical slot, both owners making
sustained progress, FIFO return debt under repeated arrivals, cancellation,
expiry, startup refusal and actual spilled-KV failures. Do not manufacture an
improved first-token result by emitting one token then parking indefinitely.
Strict request priorities must still apply. The eight-user EngineHarness tests
must precede a matched real coding comparison of p95 TTFT, time to 32 tokens,
decode gaps, completion latency, throughput, answer correctness and SSD writes.

Architecture remains at the unchanged 30594/30594 Engine line budget. Scheduler
integration requires a genuine shared-lifecycle simplification; no raised budget,
new exception or cosmetic line removal is justified. Issue #779 remains the
upstream lifecycle discussion, with no agreement or merge readiness implied.

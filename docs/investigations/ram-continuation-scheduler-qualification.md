# Scheduler qualification after upstream 1.7

This is a review specification, not an implemented scheduling policy. It applies
to upstream main `61f6d51` plus the dormant continuation prototype. Production is
unchanged. Keep admission, fusion and M4 disabled, Flash-Next at 96 GiB RAM and
512 GiB persistent cache. No performance benefit is qualified by this document.

## What changed upstream

The 1.7 decode contract holds up to 16 verify rows as a tree. Pending GDN state
is a row-bitmask (`pendingPath`), not a consecutive row count. Parking must retain
and scatter precisely those rows, preserving both running-state parities; it
must not route the exact continuation through the new cached-state codec.
Grammar masks follow parent depth, rather than a verify row's physical index.
Request-local lookup history, grammar chains and sampling continuation remain
owned by the live request while parked.

Upstream can capture checkpoints during decoding, but Runtime enables this only
for a target without auxiliary state. Flash-Next has PLE/QSA auxiliary state and
is excluded. The synthetic capability test must assert that distinction. In a
controlled Engine test, replies suspended at thresholds 1/41/129 replay 8/52/140
rows without decode captures, versus 8/20/12 with them. These are fake-model work
counts, not timings or a Flash-Next gain. Captures can also be refused under
pressure, so a future policy must not assume a checkpoint always exists.

Equal-priority foreground waiters still do not rotate streaming peers; the
existing eight-request characterization verifies this. Cheaper replay alone
therefore does not provide same-priority lane access.

## Shared lifecycle boundary

Keep Model as the Engine boundary. Do not call concrete Runtime parking APIs
from Engine. Agree on an opaque exact-continuation capability or a unified
suspend/resume contract in issue #779 before implementing the larger lifecycle.

| State | Model state | KV ownership | Hot-reader floor |
|---|---|---|---|
| Resident | Running lane, tape and host continuation | Active cache request | Held when required |
| Replay-suspended | Host continuation; device lane released | `vacateLane` ends the active cache request; resume reacquires history | Released |
| RAM-parked, proposed | Governor-charged exact capsule and host continuation | Must retain active KV ownership, with page moves forwarded | Released; restore admission must reacquire it |
| Ended/cancelled | No continuation | No active KV references | Released |

A RAM park cannot use today's `vacateLane` unchanged: that path calls
`cache_.endRequest`. Define one owner of request state, cache pins, admission
claims and deadlines, with exact-once disposal. Restoration refusal must leave
the capsule and KV ownership intact. No hidden checkpoint writes or uncharged
scratch are acceptable. The Engine architecture budget is 30594/30594 on this
base; integration requires simplification, not a raised limit or exception.

## Deterministic integration acceptance

Use the existing EngineHarness and controlled clock, then the real Runtime
fixtures. Do not simulate a policy separately and call that Engine coverage.

- Four resident foreground streams plus four waiting peers: each borrower gets
  sustained service, each displaced stream returns within a bounded borrow,
  and repeated arrivals cannot bypass accumulated return debt. Preserve strict
  priority ordering; test cancellation and expiry before selection, while
  parked, during restore admission and after restore.
- Distinguish first token from sustained progress: assert service through at
  least 32 visible tokens, bounded inter-token gaps and eventual completion.
  A first token followed by indefinite parking is a failure.
- Borrow only after drained commands and complete pass boundaries. Initial
  grammar waits, outstanding verify masks, terminal anchors and unfinished
  prefill remain ineligible. Cancellation must never emit a late token.
- Exercise allocation refusal, a full lane set, hot-floor refusal, another
  request taking the prospective lane, page compaction/moves, indexed KV reads
  and failed copies. Refusal leaves ownership unchanged and current scheduling
  usable. Failed reads end the request; no in-place numerical retry.
- Reconcile all governor, cache-reference, claim and deadline counts at every
  terminal transition, including shutdown with parked requests. Bound the
  retained-memory population and resume debt explicitly.
- Cover both state formats, both parities, sparse pending paths and all 16
  retained rows, lookup and grammar chains, constrained/thinking continuation
  and variable real-model tree/verification choices. Synthetic equality alone
  is insufficient for activation.

## Performance acceptance after integration

Only an integrated path can test RAM parking's TTFT effect. Compare the exact
serving build and candidate with identical Flash-Next weights, precision,
context, RAM/cache limits and workload. Also compare against upstream's
checkpoint-capable baseline where the target supports it. Separate cold/warm
prefix reuse and normalize output volume; retain responses and run generated
coding tests against the actual repository.

Report per-request p50/p95 TTFT, time to 32 visible tokens, inter-token gap p95
and maximum, completion latency, aggregate output tokens/s, completed-request
rate, correctness, memory peaks, replay rows, park/restore refusals, return-debt
age and SSD payload/host writes. Use matched repeated order (such as ABBA) and
state the acceptance bounds before running. Do not enable or deploy on the
strength of a one-token latency improvement or synthetic token parity.

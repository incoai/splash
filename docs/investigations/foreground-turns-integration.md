# Experimental foreground decode turns

Follow-on to #778/#779, separate from the dormant primitive PR. This implements an opt-in policy through the Model exact-suspension interface. `--foreground-turn-tokens` defaults to zero. No production deployment or real-workload speed claim is attached to this upstream variant.

Only unconstrained text-only Foreground streams with a warm remainder of at most 512 tokens can borrow a decoding lane after its current owner's quantum. At most four pairs/eight owners retain their live KV, running recurrent cells, pending tape, draft rings/window, penalty words and host sampling state. An admitted two-packet scratch bank lets each pair swap without host/Metal allocation. Bank admission uses optional governor rights and falls back on refusal. Physical slot reservations prevent a concurrent cancellation from exposing another parked stream's return slot to an unrelated admission.

The scheduler preserves exact/replay ownership through terminal cleanup. Cancellations unlink pairs; orphan return waits for the next drained tick. Pressure reclaim resolves a pair before replay suspension. Existing strict-priority preemption semantics are preserved for this upstream base, including admission after abandoning a superseded command. The separate 1.6 production backport preserves that older base's different preemption boundary.

## Current validation

- Full CPU engine suite: 486 passed, zero failed.
- CLI parsing and nonzero parked/quantum status cases were added, with default zero and explicit disable coverage.
- Python shared serve-option contracts pass; repository Ruff formatting applied.
- Architecture stays within the unchanged upstream budget.
- The production-compatible eight-row backport separately passes 461 CPU tests normally and under ASan/UBSan, plus normal and fully instrumented M5 native tests for dense/Flash-Next FP32/BF16, four banks/eight owners, allocation-free exchange, reserved return slots, cancellation and spilled-reader recovery.

These results do not substitute one decoder version for another. Current upstream uses tree-path pending metadata and up to 16 rows; the deployed backport keeps contiguous pending rows and its original eight-row shaders. Model-weight micro controls and previous synthetic receipts do not establish end-to-end TTFT or throughput.

## Activation gates

Use a matched real coding workload, repeated off/on arms, unchanged model/RAM/cache and disabled admission/fusion/M4 flags. Verify actual parked owners, TTFT p50/p95, aggregate throughput, completion latency, trusted tests and the model's own generated tests, cache reuse, physical host writes and SSD temperature. Restore and verify production/queue after the experiment. Enable only after the improvement survives correctness and throughput review; preserve deployed SSD attribution and cache-access fixes.

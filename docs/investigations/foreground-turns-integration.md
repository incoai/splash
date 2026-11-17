# Experimental foreground decode turns

Follow-on to #778/#779, separate from the dormant primitive PR. This implements an opt-in policy through the Model exact-suspension interface. `--foreground-turn-tokens` defaults to zero. No production deployment or real-workload speed claim is attached to this upstream variant.

Only unconstrained text-only Foreground streams with a warm remainder of at most 512 tokens can borrow a decoding lane after its current owner's quantum. At most four pairs/eight owners retain their live KV, running recurrent cells, pending tape, draft rings/window, penalty words and host sampling state. An admitted two-packet scratch bank lets each pair swap without host/Metal allocation. Bank admission uses optional governor rights and falls back on refusal. When a complete pair leaves a physical slot free, an optional hot-floor reservation splits another pair into two independent decoders; if refused, the allocation-free exchange continues. Physical slot reservations prevent a concurrent cancellation from exposing another parked stream's return slot to an unrelated admission.

The scheduler preserves exact/replay ownership through terminal cleanup. Cancellations unlink pairs; orphan return waits for the next drained tick. Pressure reclaim resolves a pair before replay suspension. Existing strict-priority preemption semantics are preserved for this upstream base, including admission after abandoning a superseded command. The separate 1.6 production backport preserves that older base's different preemption boundary.

## Current validation

- Full CPU engine suite: 487 passed, zero failed.
- CLI parsing and nonzero parked/quantum status cases were added, with default zero and explicit disable coverage.
- Python shared serve-option contracts pass; repository Ruff formatting applied.
- Architecture stays within the unchanged upstream budget.
- The production-compatible eight-row backport separately passes 462 CPU tests normally and under ASan/UBSan, plus normal and fully instrumented M5 native tests for dense/Flash-Next FP32/BF16, four banks/eight owners, allocation-free exchange, reserved return slots, cancellation and spilled-reader recovery.

These results do not substitute one decoder version for another. Current upstream uses tree-path pending metadata and up to 16 rows; the deployed backport keeps contiguous pending rows and its original eight-row shaders. Model-weight micro controls and previous synthetic receipts do not establish end-to-end TTFT or throughput.

## Activation gates

Use a matched real coding workload, repeated off/on arms, unchanged model/RAM/cache and disabled admission/fusion/M4 flags. Verify actual parked owners, TTFT p50/p95, aggregate throughput, completion latency, trusted tests and the model's own generated tests, cache reuse, physical host writes and SSD temperature. Restore and verify production/queue after the experiment. Enable only after the improvement survives correctness and throughput review; preserve deployed SSD attribution and cache-access fixes.

## Qualification history

The first same-binary off/on coding comparison reduced warm p95 TTFT from 20.823 s to 1.243 s while aggregate throughput was 379.14 versus 380.33 tokens/s. The stricter exact-deployed-build control then showed 380.24 versus 371.98 tokens/s, a 2.17% throughput decrease. That initial candidate was **not activated**. Receipts preserve both outcomes rather than replacing the unfavorable result.

The decoder-tail investigation found parked owners still tied to their original pair while another physical decoder slot was free. The revised implementation can fund a return into that free slot, preserves the original resident and both owners' exact state, releases both physical floors independently, and falls back to an in-place exchange if the optional claim is refused. A scheduler regression covers funded/refused returns; a deliberate no-expansion fault fails the test. A separate exact-state native test covers sampling/speculative state and whole-buffer parity. Production-derived and upstream decoder variants are qualified separately.

Saved coding receipts: [same-binary comparison](foreground-turns-coding-20261117.json), [exact deployed-build control](foreground-turns-exact-production-20261117.json). Neither receipt claims an accuracy improvement or attributable SSD-write savings.

## Revised coding and activation result

The eight-row production backport's second exact-build comparison used eight concurrent public-repo coding tasks and three warm rounds/build in ABBA order. Warm TTFT p50/p95 fell from 5.235/19.787 s to 0.463/1.291 s; client time to the 32nd visible token p95 fell from 20.316 to 1.737 s. Coding completion p95 fell from 31.873 to 26.054 s, while median completion rose from 21.027 to 21.849 s. Trusted task checks passed 17/24 on each build; generated and combined checks passed 10/24 old versus 9/24 new. No accuracy increase is claimed.

Raw coding throughput was 400.75 versus 382.95 tokens/s, with different answer lengths and one old length-limited answer (23/24 old complete versus 24/24 new). A separate ABBA capacity control retained the same coding messages but forced exactly 1,024 output tokens/request using `ignore_eos`, three warm rounds/build. It measured 397.68 versus 395.57 tokens/s (-0.53%), within observed round variation. Those truncated outputs are capacity evidence, never coding-quality passes. This does not establish a throughput increase or statistical non-inferiority.

Both builds had 24/24 warm cache hits and reused 177,600 prefix tokens; engine payload bytes were identical. Host-write differences do not establish SSD savings. Maximum coding SSD temperature was 47.85 C old / 49.85 C new. Longest visible SSE chunk gap grew from 0.174 to 0.860 s as streams shared the decoder. Full numbers, failed grades, methodology and restoration are retained in [the capacity receipt](foreground-turns-capacity-20261117.json).

The signed production-compatible backport is active at 32 tokens, preserving the prior production shaders, cache/profiling/access fixes, Flash-Next, 96 GiB RAM, 512 GiB Samsung cache and disabled admission/fusion/M4. Live acceptance verified eight RMM-style interactive-header requests without body priority, actual four-bank rotation, preserved queue DB/history, ten health/readiness endpoints and Samsung takeback/read reuse after restart without another permission prompt. [Sanitized live receipt](foreground-turns-live-20261117.json).

The upstream tree-path/16-row variant in this PR remains opt-in with default zero. Production-backport workload and deployment results do not qualify the upstream decoder's real-model throughput; local upstream CPU/native checks and CI are distinct review gates.

## CI environment follow-up

The final runtime/app job and Python 3.12 job passed. Python 3.13/3.14 failed before tests while Apple's Metal toolchain asset catalog was unavailable. CI now checks for an existing usable Metal compiler, otherwise tries installation up to three times with bounded waits and verifies the compiler after download; persistent failure remains a failed job. Four isolated command-fixture tests cover installed, transient, false-success and persistent-failure behavior. No test job or native check is skipped. The local PDF failures came from a borrowed dependency path outside the worker sandbox; a self-contained environment fixes the controls and all 30 previously failing test methods pass without weakening the sandbox or changing PDF code.

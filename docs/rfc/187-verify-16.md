# RFC 187 — Deeper verify rows (8 → 16)

Upstream issue: https://github.com/incoai/splash/issues/187

Source: `mlxfast-bonsai2-27b` `Qwen35MTPTopTwo.swift` (verify `L=16`
folded `repeats*L` rows, vocab-head `16x32x16` kernel),
`AttentionV1 verifyFoldRepeats` (bit-identical `BK/SK` reduction).

## Problem

Splash pins `TARGET_VERIFY_ROWS=8`, `DRAFT_QUERY_ROWS=8`,
`DRAFT_PROPOSAL_TOKENS=7` (`runtime/metal/abi/ExecutionGeometry.h:5`,
`model/Model.hpp:269`); decode shapes are lanes×8 only. The 27B dense
slabs (`5120×17408`) and 35B MoE tiles stay under-occupied. `PLAN.md:26`
scoped deeper verify out; bonsai evidence reopens it. Related: #168, #169.

## Proposal

Tunable verify axis, default 8:

1. `runtime/ops/Rfc187VerifyRows.hpp` declares the flag, env var
   (`SPLASH_VERIFY_ROWS`), and candidate set `{8, 16}`.
2. Follow-up threads it through `ExecutionGeometry.h`,
   `PagedAttention::verifyPlan`, `Linear::decodePlan`, arena packing, and
   `SPLASH_SPECULATIVE_SCRATCH_TOKENS`; draft depth fills the wider window.
3. Tuner (`decodeCandidates()`) picks per device; `MAX_BATCH_WIDTH`
   coordination follows.

## Contracts

- Bit-identical verify (folded view, same reduction order).
- Memory plan bounds extra scratch at startup; ABBA acceptance gate.

## Validation

- `benchmark-decode-profile` B1–B4 + `benchmark-gguf-moe` rows 8/16 +
  serving ABBA (MLX + GGUF, both models).

## Rollout

1. This RFC (flag + doc + test).
2. Plan/ABI threading (default 8).
3. Per-device promotion with evidence.

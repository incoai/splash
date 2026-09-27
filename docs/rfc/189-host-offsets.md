# RFC 189 — Host-side position offsets

Upstream issue: https://github.com/incoai/splash/issues/189

Source: bonsai `CBv2HostPositionOffsets` (`MLXFAST_HOST_POSITION_OFFSETS`,
host `+L` + shared `MLXArray`, kills 16× 1-elem device adds per verify).

## Problem

Per-layer 1-elem device adds for position/offset bookkeeping waste
dispatch slots on every attention/GDN layer. RoPE math itself is fine
(`ops/RoPE.*`, `kernels/shared/rope.metal`); only the offset delivery
needs to move.

## Proposal

1. `runtime/ops/Rfc189HostOffsets.hpp` declares the flag + env var
   (`SPLASH_HOST_POSITION_OFFSETS`).
2. Follow-up computes offsets on host in `PagedAttention` /
   `DraftAttention` dispatch and passes via shared/uniform buffer.

## Contracts

- Bit-identical positions; dispatch-only change; ABBA neutral required.

## Code change (landed)

`runtime/ops/Rfc189HostOffsets.hpp` now carries the host arithmetic
(`offsetFor`, `offsetsFit`, `LaneOffsets`) as `constexpr` with
`static_assert` cases — no dispatch change yet, and deliberately no
`.metal` change: positions are consumed by the untouched RoPE kernels,
so there is no kernel to add. `clang++ -fsyntax-only` clean.

## Validation

- `benchmark-decode-profile` per-pipeline ms; revert is a flag flip.

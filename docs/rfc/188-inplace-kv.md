# RFC 188 — In-place KV append + squeezed verify update

Upstream issue: https://github.com/incoai/splash/issues/188

Source: bonsai `FullSequenceKV.swift` (`inPlaceAppendDestination`,
`CBv2SqueezedKVUpdate`), `LayerCacheV2.swift`
(`updateAndAttendQueryBlocks`, `attendAfterInPlaceAppend`),
`KVCache.swift` (`DFlash2BlockKVCache`).

## Problem

Verify pays a full-context `updateConcat`-style copy plus a reshape copy
on head-transposed V, and one extra launch per layer per round (~64/cycle
on 27B). Splash paged KV (`ops/PagedKv.hpp Page32`, `ops/KvCopy.*`,
`kernels/shared/kv_copy.metal`) has the same shape.

## Proposal

1. `runtime/ops/Rfc188InPlaceKv.hpp` declares the flag + env var
   (`SPLASH_INPLACE_KV_APPEND`).
2. Follow-up adds an in-place append destination: fused Q/K prework
   writes K/V directly; squeezed (B=1) slice-assign for verify slices;
   pre-size `maxSize+block` once.
3. Keep `LayoutGuard` + prefix-tree + SSD-tier semantics.

## Code change (landed)

- Device: `splash_chunk_base_for_lane()` in
  `kernels/common/paged_store_row.h` — squeezed (`B=1`) chunk-base
  addressing for the verify store. Unused until dispatch passes
  `squeezed=true`; `attention_q8_store.metal` recompiles clean.
- Host: `chunkBaseForLane()` mirror in `Rfc188InPlaceKv.hpp` as
  `constexpr` with `static_assert` cases; test asserts both sides spell
  the squeezed base identically. `clang++ -fsyntax-only` clean.

## Contracts

- Bit-identical KV bytes; no alloc while encoding; ABBA gate.

## Validation

- `benchmark-decode-profile` fused-vs-parts gap + B1–B4; memory-plan
  workspace tests; ABBA.

# RFC 190 — Deferred GDN recurrent replay

Upstream issue: https://github.com/incoai/splash/issues/190

Source: bonsai `CBv2DeferredRecurrentReplay` (`keep/inputs/build/resolve`,
`RecurrentStateV2.swift`), `Qwen35A3BTargetVerify GDNReplayFused`.

## Problem

27B has 48 GDN layers vs 16 full-attention; GDN scans dominate 32K
prefill and add per-verify traffic. On partial accept Splash rebuilds
the strict-prefix SSM eagerly; bonsai fuses it into the next verify scan
(or lazy `replay(keep)`).

## Proposal

1. `runtime/ops/Rfc190DeferredGdn.hpp` declares the flag + env var
   (`SPLASH_DEFERRED_GDN_REPLAY`).
2. Follow-up adds deferred-replay state to `QwenGdnCell` / GDN plan
   (`ops/GDN.*`, `model/QwenState.hpp`): keep unbuilt prefix inputs,
   resolve in next verify; `finalSSM:nil + fullAcceptance` fast path.
3. Scratch sized at startup via memory plan.

## Code change (landed)

`runtime/ops/Rfc190DeferredGdn.hpp` now carries the transition contract
(`ReplayState`, `DeferredReplay{state, pendingTokens}`, `onVerify()`)
as `constexpr` with `static_assert` cases. No new `.metal`: the resolve
target is the existing `verify_gdn_fused*` family in
`kernels/decode/gdn.metal`, which already fuses the scan — the follow-up
only teaches `QwenGdnCell` / GDN plan to defer into it.
`clang++ -fsyntax-only` clean.

## Contracts

- Bit-identical recurrent state after resolve; `StateCache` / SSD-tier
  semantics preserved; ABBA gate.

## Validation

- 32K prefill + 32K replay TTFT + B1–B4 profile; `test-engine-metal` GDN
  bounds; ABBA.

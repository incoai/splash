# RFC 192 — Fused prompt-causal attention (8 → 3 launches)

Upstream issue: https://github.com/incoai/splash/issues/192

Source: bonsai `AttentionV1 CBv2PromptCausalAttention` (unscaled QK GEMM
+ fused `scale+causal-select` + single-row softmax, 8→3 launches,
`warmVerifyBlock` pipeline pre-build).

## Problem

Splash prompt/prefill attention (`ops/PagedAttention.*`,
`decode|prefill/attention_q8*.metal`) pays separate scale / mask /
softmax dispatches on 512-row prompt blocks and L=16 verify windows.

## Proposal

1. `runtime/ops/Rfc192FusedAttention.hpp` declares the flag + env var
   (`SPLASH_FUSED_PROMPT_ATTN`).
2. Follow-up adds a fused prompt-causal candidate to
   `PagedAttention::prefillCandidates()` / `verifyPlan` with identical
   accumulation order, plus verify-block pipeline retention.

## Code change (landed)

`runtime/ops/Rfc192FusedAttention.hpp` now carries `fusedScaleAvailable()`
(the `(4,6)`/`(2,8)` combos whose fused scale-in-softmax split kernels
already ship) plus the 6-entry prefill warm-up roster (split+reduce pairs
+ BF16 splits), `constexpr` with `static_assert` cases. Deliberately no
new `.metal`: the bonsai scale-epilogue fusion already ships here as
`ScaleInSoftmax`, and split+reduce fusion is deferred with reason (split
grids over heads×tiles×splits vs reduce over heads×M×tiles — one grid
cannot cover both unpadded). The follow-up pre-builds/retains the roster.
`clang++ -fsyntax-only` clean.

## Contracts

- Bit-identical outputs (one rounding boundary); tuner-gated, default off.

## Validation

- 32K prefill tok/s + cached TTFT replay + B1–B4 profile; attention unit
  tests + ABBA.

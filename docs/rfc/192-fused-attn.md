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

## Contracts

- Bit-identical outputs (one rounding boundary); tuner-gated, default off.

## Validation

- 32K prefill tok/s + cached TTFT replay + B1–B4 profile; attention unit
  tests + ABBA.

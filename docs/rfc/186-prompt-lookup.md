# RFC 186 — Prompt-lookup drafter alongside DFlash2

Upstream issue: https://github.com/incoai/splash/issues/186

Source: `mlxfast-bonsai2-27b` `879410a..831fae7`,
`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/MTP/CBv2PromptLookupDraft.swift`
(+366, unique `>=16` prompt-span lookup, `MLXFAST_DFLASH_LOOKUP/SKIP`).

## Problem

Splash drafts with DFlash2 only (`runtime/model/DFlashDraft.*`,
`runtime/ops/DraftAttention.*`, `runtime/model/DraftContextPlan.cpp`).
Repeated spans (system prompts, file prefixes, agent retries) pay a full
drafter forward even when the prompt itself contains the next tokens.

## Proposal

Additive, default-off second drafter:

1. New `runtime/ops/Rfc186PromptLookup.hpp` declares the feature flag,
   env var (`SPLASH_PROMPT_LOOKUP`), and minimum span (`>=16`).
2. Follow-up wires a `PromptLookup` plan hook into `DraftContextPlan` /
   `BatchPlan`: on lookup hit, skip the `DFlashDraft` forward and verify
   the lookup span as the draft.
3. Tag lookup vs draft origin in `Status.cpp` acceptance accounting
   (`drafted/accepted_draft_tokens`).

## Contracts

- Target decides every token; lookup output is token-identical after verify.
- ABBA gate: output text + `accepted_draft_tokens` match baseline exactly
  (short/32K, code + technical + repeat, widths 1–4).
- No allocation while encoding; workspace via existing arenas.

## Validation

- `benchmark-decode-profile` B1–B4 before/after + serving ABBA per `PLAN.md §9`.
- Report hit rate and skipped drafter forwards separately.

## Code change (landed)

`runtime/ops/Rfc186PromptLookup.hpp` now carries the host-side matcher
(no dispatch change, no kernel — lookup reuses the existing target
verify kernels, which is the point):

- `PromptSpan{start, length}`, `meetsMinSpan()` (`>=16` gate),
  `hashTokens()` (FNV-1a candidate selection), `matchAt()` (exact span
  equality), all `constexpr` with `static_assert` cases.
- `clang++ -std=c++20 -fsyntax-only` clean; python test mirrors the
  match semantics.

## Rollout

1. This RFC (flag + doc + test, no behavior change).
2. Plan hook + accounting (still default off).
3. Tuner/config promotion per device with ABBA evidence.

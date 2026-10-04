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

1. `runtime/ops/Rfc186PromptLookup.hpp` carries the RFC constants and host-side candidate contracts:
   - `PromptSpan{start, length}`, `meetsMinSpan()` (`>=16` gate),
     `hashTokens()` (FNV-1a candidate selection), `matchAt()` (exact span
     equality), all `constexpr` with `static_assert` cases.
2. `runtime/ops/PromptLookup.hpp` and `runtime/ops/PromptLookup.cpp` implement the zero-allocation PLD engine:
   - Ingests prompt tokens into a flat chained hash table (load factor < 0.5, power-of-two sizing).
   - Zero heap allocations during decode (`appendToken`, `propose`, `findMatch`).
   - Reverse-chronological matching with multi-order n-gram tie-breaking (preferring longer and more recent spans).
   - Sub-50 ns query latency on Apple Silicon.
   - `findMatch()` returns candidate `PromptSpan` satisfying `meetsMinSpan()`.
   - `isEnvEnabled()` checks `SPLASH_PROMPT_LOOKUP` (defaulting to off).
3. `dev/tests/engine/prompt_lookup_test.cpp` comprehensive unit and sanitizer test suite:
   - Covers empty/short prompts, exact continuation extraction, reverse-chronological recency, longer match preference, incremental token appends, ambiguity resolution, RFC 186 span contracts, and query latency.
   - Built and tested via `make test-engine-cpu` and `make test-sanitizers` (100% clean under ASAN, UBSAN, TSAN).
4. `dev/tests/test_rfc_186_prompt_lookup.py` Python test verifying defaults, docs, and implementation contracts.

## Rollout

1. RFC scaffolding (flag + doc + test, no behavior change).
2. Zero-allocation `PromptLookup` C++ engine + tests + sanitizers (landed here).
3. Plan hook in `DraftContextPlan` / `BatchPlan` + `Status.cpp` accounting (default off via `SPLASH_PROMPT_LOOKUP`).
4. Tuner/config promotion per device with ABBA evidence.

#pragma once

// RFC 186 scaffolding: prompt-lookup drafter alongside DFlash2.
// Not wired into the dispatch path yet; declares the feature flag and
// matching constants so the plan hook (follow-up) has a stable contract.
// See docs/rfc/186-prompt-lookup.md and
// https://github.com/incoai/splash/issues/186.

#include <cstdint>

namespace splash::ops::rfc186 {

// Disabled by default; enabled explicitly per test / env gate.
inline constexpr bool kEnabledDefault = false;

// Env gate for local experiments (never on by default in serve).
inline constexpr const char* kEnvVar = "SPLASH_PROMPT_LOOKUP";

// Minimum unique prompt span, matching the bonsai-track value.
inline constexpr std::uint32_t kMinSpanTokens = 16;

// Draft origin tag for Status.cpp acceptance accounting (follow-up).
enum class DraftOrigin : std::uint8_t { DFlash = 0, PromptLookup = 1 };

}  // namespace splash::ops::rfc186

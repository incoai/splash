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

// A candidate span: [start, start + length) in the committed prefix.
// The target verifies the span, so this machinery only *proposes*.
struct PromptSpan final {
  std::uint32_t start = 0;
  std::uint32_t length = 0;
};

// Minimum-span gate: shorter spans never pay back the verify slot.
[[nodiscard]] constexpr bool meetsMinSpan(std::uint32_t length) noexcept {
  return length >= kMinSpanTokens;
}

// FNV-1a over a token span (host candidate selection only).
[[nodiscard]] constexpr std::uint64_t hashTokens(const std::uint32_t* tokens,
                                                std::uint32_t n) noexcept {
  std::uint64_t h = 1469598103934665603ULL;
  for (std::uint32_t i = 0; i < n; ++i) {
    h ^= tokens[i];
    h *= 1099511628211ULL;
  }
  return h;
}

// True when needle[0, n) equals haystack[off, off + n).
[[nodiscard]] constexpr bool matchAt(const std::uint32_t* haystack,
                                     std::uint32_t haystackLen,
                                     std::uint32_t off,
                                     const std::uint32_t* needle,
                                     std::uint32_t n) noexcept {
  if (n == 0 || off + n > haystackLen) return false;
  for (std::uint32_t i = 0; i < n; ++i)
    if (haystack[off + i] != needle[i]) return false;
  return true;
}

namespace detail {
inline constexpr std::uint32_t kToksA[] = {1, 2, 3, 4};
inline constexpr std::uint32_t kToksB[] = {0, 1, 2, 3, 4, 5};
}  // namespace detail

static_assert(meetsMinSpan(16) && !meetsMinSpan(15));
static_assert(hashTokens(detail::kToksA, 0) == 1469598103934665603ULL);
static_assert(matchAt(detail::kToksB, 6, 1, detail::kToksA, 4));
static_assert(!matchAt(detail::kToksB, 6, 2, detail::kToksA, 4));
static_assert(!matchAt(detail::kToksB, 6, 3, detail::kToksA, 4));

}  // namespace splash::ops::rfc186

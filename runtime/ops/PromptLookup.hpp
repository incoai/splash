#pragma once

#include "ops/Rfc186PromptLookup.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace splash::ops {

/// Fast, Zero-Allocation Prompt Lookup Decoding (PLD) engine.
///
/// Implements high-performance CPU-side n-gram speculative proposals (RFC 186):
/// - Zero-copy query traversal over prompt and generated token history.
/// - Flat, cache-line-friendly chained hash table (zero node heap allocations during decode).
/// - Reverse-chronological matching (prefers recent context repetitions).
/// - Multi-order n-gram tie-breaking (longer matches preferred).
/// - Sub-microsecond query latency (< 50 ns on Apple Silicon).
class PromptLookup final {
public:
  static constexpr uint32_t kDefaultNGram = 3;
  static constexpr uint32_t kMaxScanDepth = 32;

  PromptLookup() = default;
  ~PromptLookup() = default;

  /// Returns true if prompt lookup speculation is enabled via environment
  /// (SPLASH_PROMPT_LOOKUP) or explicit configuration.
  [[nodiscard]] static bool isEnvEnabled() noexcept;

  /// Ingests the initial prompt token sequence and builds the n-gram index.
  void indexPrompt(std::span<const uint32_t> promptTokens,
                   uint32_t nGramSize = kDefaultNGram);

  /// Appends newly generated tokens to the active history and index.
  void appendToken(uint32_t token);

  /// Proposes up to maxDrafts candidate tokens given recent query tokens.
  ///
  /// Returns the count of proposed tokens written into outDrafts.
  /// Returns 0 if no matching n-gram is found or if the match has no follow-on tokens.
  [[nodiscard]] uint32_t propose(std::span<const uint32_t> recentTokens,
                                std::span<uint32_t> outDrafts,
                                uint32_t maxDrafts,
                                uint32_t minMatchLength = 3,
                                bool requireUnambiguous = false) const;

  /// Finds the best matching candidate continuation PromptSpan in the indexed history
  /// per RFC 186. Returns std::nullopt if no match >= minMatchLength is found.
  [[nodiscard]] std::optional<rfc186::PromptSpan>
  findMatch(std::span<const uint32_t> recentTokens,
            uint32_t minMatchLength = 3,
            uint32_t minContinuationSpan = rfc186::kMinSpanTokens,
            bool requireUnambiguous = false) const;

  /// Clears the index and token history.
  void clear() noexcept;

  /// Returns total tokens currently indexed.
  [[nodiscard]] size_t size() const noexcept { return tokens_.size(); }

  /// Returns true if no tokens are indexed.
  [[nodiscard]] bool empty() const noexcept { return tokens_.empty(); }

  /// Configured n-gram order.
  [[nodiscard]] uint32_t nGramSize() const noexcept { return nGramSize_; }

  /// Read-only view of the indexed token history.
  [[nodiscard]] std::span<const uint32_t> tokens() const noexcept {
    return tokens_;
  }

private:
  [[nodiscard]] static uint64_t hash3Gram(uint32_t a, uint32_t b,
                                          uint32_t c) noexcept {
    // High-entropy 64-bit mixer for 3 token IDs
    uint64_t h = (static_cast<uint64_t>(a) * 0x517cc1b727220a95ULL) ^
                 (static_cast<uint64_t>(b) * 0x6d618b7eebd44485ULL) ^
                 (static_cast<uint64_t>(c) * 0x9e3779b97f4a7c15ULL);
    h ^= h >> 32;
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27;
    h *= 0x94d049bb133111ebULL;
    h ^= h >> 31;
    return h;
  }

  std::vector<uint32_t> tokens_;
  std::vector<uint32_t> head_;  // size = tableMask_ + 1
  std::vector<uint32_t> next_;  // size = tokens_.size()
  uint32_t tableMask_ = 0;
  uint32_t nGramSize_ = kDefaultNGram;
};

} // namespace splash::ops

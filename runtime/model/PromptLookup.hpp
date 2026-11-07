#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace splash::model {

// An optional proposal source, never an acceptance rule. All returned tokens
// still pass through the target's ordinary causal verify and commit.
struct PromptLookup final {
  static constexpr uint32_t kMatchTokens = 16;
  static constexpr uint32_t kProposalTokens = 7;
  static constexpr uint32_t kSearchTokens = 65536;
  using Proposal = std::array<uint32_t, kProposalTokens>;

  // history excludes the known pending anchor. Search only the most recent
  // kSearchTokens of that history, without allocation or persistent indexes.
  // Every fully observed match must have the same seven-token continuation.
  // Incomplete occurrences cannot supply proposals and are not candidates.
  [[nodiscard]] static std::optional<Proposal> find(
      std::span<const uint32_t> history, uint32_t anchor) noexcept {
    if (history.size() < kMatchTokens + kProposalTokens)
      return std::nullopt;
    const size_t begin = history.size() - std::min<size_t>(history.size(), kSearchTokens);
    const size_t last = history.size() - kMatchTokens - kProposalTokens;
    const size_t suffix = history.size() - (kMatchTokens - 1);
    std::optional<Proposal> found;
    for (size_t at = begin; at <= last; ++at) {
      if (history[at + kMatchTokens - 1] != anchor ||
          !std::equal(history.begin() + static_cast<std::ptrdiff_t>(at),
                      history.begin() + static_cast<std::ptrdiff_t>(at + kMatchTokens - 1),
                      history.begin() + static_cast<std::ptrdiff_t>(suffix)))
        continue;
      Proposal proposal;
      std::copy_n(history.begin() + static_cast<std::ptrdiff_t>(at + kMatchTokens),
                  kProposalTokens, proposal.begin());
      if (found && *found != proposal)
        return std::nullopt;
      found = proposal;
    }
    return found;
  }
};

} // namespace splash::model

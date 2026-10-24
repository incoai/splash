#pragma once

#include "metal/abi/Sampling.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <stdexcept>

namespace splash::ops {

// A target's stop tokens, which end a sequence: up to SPLASH_STOP_TOKENS ids,
// every place past them SPLASH_NO_TOKEN, which no vocabulary holds and which
// is no stop token, as the sampling kernels compare them
// (metal/abi/Sampling.h). A family lists its own (the Qwen families'
// {248044, 248046}); a list longer than the kernels compare, or holding
// SPLASH_NO_TOKEN, throws, and in a family's constant does not compile.
struct StopTokens final {
  std::array<uint32_t, SPLASH_STOP_TOKENS> ids{};

  constexpr StopTokens() noexcept { ids.fill(SPLASH_NO_TOKEN); }
  constexpr StopTokens(std::initializer_list<uint32_t> tokens) : StopTokens() {
    if (tokens.size() > ids.size() || std::ranges::find(tokens, SPLASH_NO_TOKEN) != tokens.end())
      throw std::invalid_argument("more stop tokens than the sampling kernels compare, or SPLASH_NO_TOKEN");
    std::ranges::copy(tokens, ids.begin());
  }

  // The stop tokens without their padding.
  [[nodiscard]] constexpr std::span<const uint32_t> tokens() const noexcept {
    return std::span(ids).first(static_cast<size_t>(std::ranges::find(ids, SPLASH_NO_TOKEN) - ids.begin()));
  }
  // Whether `token` is a stop token, as the kernels' stop_token answers.
  [[nodiscard]] constexpr bool contains(uint32_t token) const noexcept {
    return token != SPLASH_NO_TOKEN && std::ranges::find(ids, token) != ids.end();
  }
  bool operator==(const StopTokens &) const = default;
};

} // namespace splash::ops

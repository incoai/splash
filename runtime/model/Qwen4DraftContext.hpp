#pragma once

#include "model/Qwen4DraftVocabulary.hpp"

#include <array>
#include <cassert>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace splash::model {

// Host context and fallback history survive suspension with the request;
// device row ownership does not. Scanning never depends on the head chosen
// for the current cycle, while copying rows remains lazy.
struct Qwen4DraftContext final {
  std::vector<uint32_t> tokens;
  std::vector<uint64_t> bits;
  size_t scanned = 0;
  float misses = 0.0F;
  float observations = 0.0F;

  [[nodiscard]] bool contains(uint32_t token) const noexcept {
    return inQwen4DraftVocabulary(token) ||
           (token / 64 < bits.size() && ((bits[token / 64] >> (token % 64)) & 1));
  }
  void scan(std::span<const uint32_t> history, uint32_t vocabulary,
            uint32_t capacity = kQwen4DraftContextTokens) {
    if (bits.empty()) bits.assign((vocabulary + 63) / 64, 0);
    for (; scanned < history.size(); ++scanned) {
      if (tokens.size() >= capacity) {
        scanned = history.size();
        break;
      }
      const uint32_t token = history[scanned];
      if (token >= vocabulary || contains(token)) continue;
      bits[token / 64] |= uint64_t{1} << (token % 64);
      tokens.push_back(token);
    }
  }
  void record(std::span<const uint32_t> selected) noexcept {
    for (const uint32_t token : selected) {
      misses = 0.98F * misses + (contains(token) ? 0.0F : 1.0F);
      observations = 0.98F * observations + 1.0F;
    }
  }
  [[nodiscard]] bool useVocabulary() const noexcept { return misses <= 0.03F * observations; }
};

// A lane's rows are valid only during one activation. Runtime releases them
// on completion, cancellation and suspension, even if the next request has
// the same numeric ID. bind() returns the first row requiring a GPU copy.
template <size_t Lanes> class Qwen4DraftContextCache final {
public:
  void release(uint32_t lane) noexcept {
    owners_[lane].reset();
    synced_[lane] = 0;
  }
  [[nodiscard]] uint32_t bind(uint32_t lane, uint64_t request, uint32_t count) {
    if (owners_[lane] != request) {
      owners_[lane] = request;
      synced_[lane] = 0;
    }
    const uint32_t synced = synced_[lane];
    if (synced > count) throw std::logic_error("draft context shrank within a lane activation");
    assert(synced <= count);
    synced_[lane] = count;
    return synced;
  }

private:
  std::array<std::optional<uint64_t>, Lanes> owners_{};
  std::array<uint32_t, Lanes> synced_{};
};

} // namespace splash::model

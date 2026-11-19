#pragma once

#include <cstdint>
#include <span>

namespace splash {

// A tokenizer-compiled byte marker automaton. Class zero is the default;
// sorted exceptions pack token << 8 | class. Transitions pack four states
// per word. Each active state also names the tokens that finish its close.
// The bounded program binds as one Metal constant buffer per lane.
class TokenAutomaton final {
public:
  static constexpr uint32_t maximumWords = 1024;
  explicit TokenAutomaton(std::span<const uint32_t> words) : words_(words) {}
  bool valid(uint32_t vocabulary = 0xffffffffu) const noexcept {
    if (words_.size() < 8 || words_.size() > maximumWords) return false;
    const auto states = words_[0], classes = words_[1], open = words_[2];
    if (!states || states > 64 || !classes || classes > 256 || open >= states ||
        words_[3] != states || words_[4] != states + 1 || words_[5] >= states ||
        words_[6] > words_.size()) return false;
    const uint32_t matrix = (states * classes + 3) / 4;
    const uint32_t suffixes = 8 + matrix + states;
    const uint32_t entries = words_[7];
    if (uint64_t{entries} + words_[6] != words_.size() || entries < suffixes) return false;
    for (uint32_t i = 0; i < states * classes; ++i)
      if (((words_[8 + i / 4] >> (8 * (i % 4))) & 255) > states + 1) return false;
    for (uint32_t s = 0; s < states; ++s) {
      const uint32_t offset = words_[8 + matrix + s];
      if (s < open) { if (offset) return false; continue; }
      if (offset < suffixes || offset >= entries || !words_[offset] ||
          words_[offset] > 64 || offset + 1 + words_[offset] > entries) return false;
      for (const auto token : words_.subspan(offset + 1, words_[offset]))
        if (token >= vocabulary) return false;
    }
    uint32_t previous = 0;
    for (uint32_t i = 0; i < words_[6]; ++i) {
      const auto entry = words_[entries + i];
      if ((entry >> 8) >= vocabulary || !(entry & 255) || (entry & 255) >= classes ||
          (i && (entry >> 8) <= previous)) return false;
      previous = entry >> 8;
    }
    for (uint32_t state = open; state < states; ++state) {
      uint32_t next = state;
      for (const auto token : suffix(state)) next = advance(next, token);
      if (next != words_[3]) return false;
    }
    return true;
  }
  uint32_t initial() const noexcept { return words_[5]; }
  bool opened(uint32_t state) const noexcept { return state >= words_[2] && state < words_[0]; }
  bool terminal(uint32_t state) const noexcept { return state >= words_[0]; }
  uint32_t advance(uint32_t state, uint32_t token) const noexcept {
    if (terminal(state)) return state;
    const uint32_t entries = words_[7], count = words_[6];
    uint32_t low = 0, high = count;
    while (low < high) {
      const uint32_t middle = (low + high) / 2;
      if ((words_[entries + middle] >> 8) < token) low = middle + 1;
      else high = middle;
    }
    const uint32_t kind = low < count && (words_[entries + low] >> 8) == token ? words_[entries + low] & 255 : 0;
    const uint32_t index = state * words_[1] + kind;
    return (words_[8 + index / 4] >> (8 * (index % 4))) & 255;
  }
  std::span<const uint32_t> suffix(uint32_t state) const noexcept {
    const uint32_t matrix = (words_[0] * words_[1] + 3) / 4;
    const uint32_t offset = words_[8 + matrix + state];
    return words_.subspan(offset + 1, words_[offset]);
  }
private:
  std::span<const uint32_t> words_;
};
} // namespace splash

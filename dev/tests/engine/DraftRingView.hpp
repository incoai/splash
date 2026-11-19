#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace splash::test {

// Logical [head][position][dimension] rows retained by one layer's ring.
// Keys store [head][slot][dimension]; values store [head][dimension][slot].
class DraftRingView final {
public:
  DraftRingView(std::span<const uint16_t> keys, std::span<const uint16_t> values,
                uint32_t heads, uint32_t dimensions, uint32_t window,
                uint64_t begin, uint64_t end)
      : keys_(keys), values_(values), dimensions_(dimensions), window_(window) {
    if (!heads || !dimensions || !window || end < begin)
      throw std::invalid_argument("invalid draft ring geometry");
    const uint64_t capacity = uint64_t{heads} * window * dimensions;
    if (keys.size() < capacity || values.size() < capacity)
      throw std::invalid_argument("draft ring buffer is smaller than its layer window");
    begin_ = std::max(begin, end - std::min<uint64_t>(end, window));
    rows_ = end - begin_;
    elements_ = uint64_t{heads} * rows_ * dimensions;
  }

  [[nodiscard]] uint64_t size() const { return elements_; }
  [[nodiscard]] uint64_t position(uint64_t index) const {
    check(index);
    return begin_ + index / dimensions_ % rows_;
  }
  [[nodiscard]] uint16_t key(uint64_t index) const {
    const uint64_t slot = position(index) % window_;
    const uint64_t head = index / (rows_ * dimensions_);
    return keys_[(head * window_ + slot) * dimensions_ + index % dimensions_];
  }
  [[nodiscard]] uint16_t value(uint64_t index) const {
    const uint64_t slot = position(index) % window_;
    const uint64_t head = index / (rows_ * dimensions_);
    return values_[(head * dimensions_ + index % dimensions_) * window_ + slot];
  }

private:
  void check(uint64_t index) const {
    if (index >= elements_)
      throw std::out_of_range("draft ring logical index is outside retained rows");
  }
  std::span<const uint16_t> keys_, values_;
  uint32_t dimensions_, window_;
  uint64_t begin_ = 0, rows_ = 0, elements_ = 0;
};

} // namespace splash::test

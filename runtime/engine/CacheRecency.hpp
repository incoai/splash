#pragma once

#include <cstdint>
#include <limits>

namespace splash::engine {

// Shared monotonic access order for KV blocks and composite states.
class CacheRecency final {
public:
  [[nodiscard]] uint64_t next() noexcept {
    if (value_ != std::numeric_limits<uint64_t>::max())
      ++value_;
    return value_;
  }
  // Continues after a value an earlier process handed out, for what a
  // persistent tier takes back.
  void continueAfter(uint64_t value) noexcept {
    if (value > value_)
      value_ = value;
  }

private:
  uint64_t value_ = 0;
};

struct CacheEvictionCandidate final {
  uint64_t id = 0;
  uint64_t lastUsed = 0;
};

} // namespace splash::engine

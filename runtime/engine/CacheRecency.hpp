#pragma once

#include <cstdint>
#include <limits>
#include <optional>

namespace splash::engine {

// Maintenance holds storage alive without making its content more recent.
enum class CacheAccess : uint8_t { Request, Maintenance };

// Shared monotonic access order for KV blocks and group states.
class CacheRecency final {
public:
  [[nodiscard]] uint64_t next() noexcept {
    if (value_ != std::numeric_limits<uint64_t>::max())
      ++value_;
    return value_;
  }

private:
  uint64_t value_ = 0;
};

struct CacheEvictionCandidate final {
  uint64_t id = 0;
  uint64_t lastUsed = 0;
  std::optional<uint32_t> group{};
};

} // namespace splash::engine

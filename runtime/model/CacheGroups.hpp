#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace splash {

class StatePayload;

// IDs belong to a model's cache namespace, not to engine architecture names.
using CacheGroupId = uint32_t;
enum class CacheGroupKind : uint8_t { Checkpoint, SlidingWindow };

struct CacheGroupSpec final {
  CacheGroupId id = 0;
  CacheGroupKind kind = CacheGroupKind::Checkpoint;
  uint32_t windowTokens = 0;
  bool operator==(const CacheGroupSpec &) const = default;
};

// A checkpoint has begin == end. A window block covers [begin, end).
// Several logical slices may share the same immutable physical payload.
struct CachedStateBlock final {
  CacheGroupId group = 0;
  uint32_t begin = 0;
  uint32_t end = 0;
  std::shared_ptr<const StatePayload> payload;
};

struct RestoreState final {
  uint32_t boundary = 0;
  std::vector<CachedStateBlock> blocks;

  // The contiguous suffix of fixed-size window pages immediately before B.
  [[nodiscard]] uint32_t windowBegin(CacheGroupId group, uint32_t window) const;
  [[nodiscard]] bool resident() const noexcept;
  [[nodiscard]] uint64_t bytes() const;
};

// Reject malformed declarations before any cache record can be published.
void validateCacheGroups(std::span<const CacheGroupSpec> groups);

} // namespace splash

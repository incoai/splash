#include "model/CacheGroups.hpp"
#include "model/Model.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace splash {

void validateCacheGroups(std::span<const CacheGroupSpec> groups) {
  std::unordered_set<CacheGroupId> ids;
  for (const auto &group : groups) {
    if (!ids.insert(group.id).second ||
        (group.kind == CacheGroupKind::Checkpoint && group.windowTokens) ||
        (group.kind == CacheGroupKind::SlidingWindow && !group.windowTokens) ||
        (group.kind != CacheGroupKind::Checkpoint &&
         group.kind != CacheGroupKind::SlidingWindow))
      throw std::invalid_argument("invalid cache group declaration");
  }
}

bool RestoreState::resident() const noexcept {
  return std::all_of(blocks.begin(), blocks.end(), [](const auto &block) {
    return block.payload &&
           block.payload->residentBytes() == block.payload->bytes();
  });
}

uint64_t RestoreState::bytes() const {
  uint64_t bytes = 0;
  std::unordered_set<const void *> seen;
  for (const auto &block : blocks)
    for (const auto &resource : block.payload->resources())
      if (seen.insert(resource.identity).second)
        bytes += resource.bytes;
  return bytes;
}

} // namespace splash

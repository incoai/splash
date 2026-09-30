#include "model/CacheGroups.hpp"
#include "model/Model.hpp"
#include "ops/PagedKv.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace splash {

void validateCacheGroups(std::span<const CacheGroupSpec> groups) {
  std::unordered_set<CacheGroupId> ids;
  for (const auto &group : groups) {
    if (!ids.insert(group.id).second ||
        (group.kind == CacheGroupKind::Checkpoint && group.windowTokens) ||
        (group.kind == CacheGroupKind::SlidingWindow &&
         (!group.windowTokens || group.windowTokens % kv::kPageTokens)) ||
        (group.kind != CacheGroupKind::Checkpoint &&
         group.kind != CacheGroupKind::SlidingWindow))
      throw std::invalid_argument("invalid cache group declaration");
  }
}

uint32_t RestoreState::windowBegin(CacheGroupId group, uint32_t window) const {
  const uint32_t first = boundary - std::min(boundary, window);
  uint32_t begin = boundary;
  while (begin > first) {
    const auto found =
        std::find_if(blocks.begin(), blocks.end(), [&](const auto &block) {
          return block.group == group && block.end == begin &&
                 block.end - block.begin == kv::kPageTokens;
        });
    if (found == blocks.end())
      break;
    begin -= kv::kPageTokens;
  }
  return begin;
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

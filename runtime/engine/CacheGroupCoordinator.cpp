#include "engine/CacheGroupCoordinator.hpp"
#include "model/WindowPlan.hpp"
#include "ops/PagedKv.hpp"

#include <algorithm>
#include <stdexcept>

namespace splash::engine {

CacheGroupCoordinator::CacheGroupCoordinator(std::vector<CacheGroupSpec> groups)
    : groups_(std::move(groups)) {
  validateCacheGroups(groups_);
  // Checkpoint groups reject incomplete boundaries before window lookup.
  std::stable_partition(groups_.begin(), groups_.end(), [](const auto &group) {
    return group.kind == CacheGroupKind::Checkpoint;
  });
}

bool CacheGroupCoordinator::collect(uint32_t boundary, const Find &find,
                                    const Visit &visit, uint32_t prompt) const {
  for (const auto &group : groups_) {
    if (group.kind == CacheGroupKind::Checkpoint) {
      const auto block = find(group.id, boundary);
      if (!block || block->group != group.id || block->begin != boundary ||
          block->end != boundary) {
        return false;
      }
      if (visit)
        visit(*block);
      continue;
    }
    const uint32_t begin = boundary - std::min(boundary, group.windowTokens);
    uint32_t available = boundary;
    while (available > begin) {
      const auto block = find(group.id, available);
      if (!block)
        break;
      if (block->group != group.id || block->end != available ||
          block->end - block->begin != kv::kPageTokens)
        return false;
      available -= kv::kPageTokens;
    }
    if (!planWindow(boundary, prompt ? prompt : boundary, group.windowTokens,
                    available)
             .usable)
      return false;
    if (visit)
      for (uint32_t end = boundary; end > available; end -= kv::kPageTokens)
        visit(*find(group.id, end));
  }
  return true;
}

std::optional<RestoreState>
CacheGroupCoordinator::match(uint32_t maximumBoundary, uint32_t alignment,
                             const Find &find, uint32_t prompt) const {
  if (!alignment)
    throw std::invalid_argument("cache boundary alignment is zero");
  for (uint32_t boundary = maximumBoundary / alignment * alignment; boundary;
       boundary -= alignment) {
    if (!collect(boundary, find, {}, prompt))
      continue;
    RestoreState result{boundary, {}};
    collect(
        boundary, find,
        [&](const auto &block) { result.blocks.push_back(block); }, prompt);
    return result;
  }
  return std::nullopt;
}

} // namespace splash::engine

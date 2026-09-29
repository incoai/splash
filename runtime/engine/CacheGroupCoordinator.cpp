#include "engine/CacheGroupCoordinator.hpp"

#include <algorithm>
#include <stdexcept>

namespace splash::engine {

CacheGroupCoordinator::CacheGroupCoordinator(std::vector<CacheGroupSpec> groups)
    : groups_(std::move(groups)) {
  validateCacheGroups(groups_);
  // Exact-state groups reject incomplete boundaries cheaply; visiting their
  // reusable dependencies afterward also preserves dependency-aware recency.
  std::stable_partition(groups_.begin(), groups_.end(), [](const auto &group) {
    return group.kind == CacheGroupKind::Checkpoint;
  });
}

bool CacheGroupCoordinator::collect(uint32_t boundary, const Find &find,
                                    const Visit &visit) const {
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
    uint32_t end = boundary;
    while (end > begin) {
      const auto block = find(group.id, end);
      if (!block || block->group != group.id || block->end != end ||
          block->begin >= end) {
        return false;
      }
      if (visit)
        visit(*block);
      end = block->begin;
    }
  }
  return true;
}

bool CacheGroupCoordinator::complete(uint32_t boundary,
                                     const Find &find) const {
  return collect(boundary, find, {});
}

void CacheGroupCoordinator::visitComplete(uint32_t boundary, const Find &find,
                                          const Visit &visit) const {
  if (complete(boundary, find))
    collect(boundary, find, visit);
}

std::optional<RestoreState>
CacheGroupCoordinator::match(uint32_t maximumBoundary, uint32_t alignment,
                             const Find &find) const {
  if (!alignment)
    throw std::invalid_argument("cache boundary alignment is zero");
  // Ranges may overlap and end at different token boundaries. A missing
  // fragment at one candidate does not prove that an earlier candidate using
  // another range is incomplete; check each aligned boundary independently.
  for (uint32_t boundary = maximumBoundary / alignment * alignment; boundary;
       boundary -= alignment) {
    if (!complete(boundary, find))
      continue;
    RestoreState result{boundary, {}};
    collect(boundary, find,
            [&](const auto &block) { result.blocks.push_back(block); });
    return result;
  }
  return std::nullopt;
}

} // namespace splash::engine

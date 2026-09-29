#pragma once

#include "model/CacheGroups.hpp"

#include <functional>
#include <optional>

namespace splash::engine {

// The index supplies only blocks on the already matched target prefix chain.
// Payloads stay opaque: checkpoint and window coverage are sufficient to find
// a common restore boundary for any number of model-defined groups.
class CacheGroupCoordinator final {
public:
  using Find =
      std::function<std::optional<CachedStateBlock>(CacheGroupId, uint32_t)>;

  using Visit = std::function<void(const CachedStateBlock &)>;

  explicit CacheGroupCoordinator(std::vector<CacheGroupSpec> groups);
  [[nodiscard]] std::span<const CacheGroupSpec> groups() const noexcept {
    return groups_;
  }
  [[nodiscard]] std::optional<RestoreState>
  match(uint32_t maximumBoundary, uint32_t alignment, const Find &find) const;
  [[nodiscard]] bool complete(uint32_t boundary, const Find &find) const;
  // Visit only complete restore points, with checkpoints before dependencies.
  void visitComplete(uint32_t boundary, const Find &find,
                     const Visit &visit) const;

private:
  bool collect(uint32_t boundary, const Find &find, const Visit &visit) const;
  std::vector<CacheGroupSpec> groups_;
};

} // namespace splash::engine

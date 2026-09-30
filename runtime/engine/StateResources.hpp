#pragma once

#include "model/Model.hpp"

#include <span>
#include <unordered_map>

namespace splash::engine {

// Counts backing once even when several restore points reference it. The
// resource descriptors are retained in each entry, so releasing is allocation
// free and does not depend on a model object's lifetime.
class StateResources final {
public:
  uint64_t retain(std::span<const StateResource> resources) {
    uint64_t added = 0;
    size_t retained = 0;
    try {
      for (const auto &resource : resources) {
        auto &[count, bytes] = references_[resource.identity];
        if (!count++) {
          bytes = resource.bytes;
          added += bytes;
        }
        ++retained;
      }
    } catch (...) {
      release(resources.first(retained));
      throw;
    }
    return added;
  }
  uint64_t release(std::span<const StateResource> resources) noexcept {
    uint64_t removed = 0;
    for (const auto &resource : resources) {
      auto found = references_.find(resource.identity);
      if (!--found->second.count) {
        removed += found->second.bytes;
        references_.erase(found);
      }
    }
    return removed;
  }

private:
  struct Reference {
    uint64_t count = 0, bytes = 0;
  };
  std::unordered_map<const void *, Reference> references_;
};
} // namespace splash::engine

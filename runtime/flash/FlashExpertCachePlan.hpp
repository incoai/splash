#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace splash::flash {

// CPU metadata only. The cache constructor must independently validate actual
// projection geometry, coefficient policy and governor-accounted allocations.
struct FlashExpertCachePlan final {
  std::array<std::vector<uint32_t>, 48> selectedExperts;
  std::string sourceIdentity;
  std::string planSha256;
  std::string phaseFilter;
  uint32_t requestedLimit = 0;
  uint64_t observedAssignments = 0;
  uint64_t selectedAssignments = 0;
  double observedHitRate = 0.0;
  uint64_t estimatedCoefficientBytes = 0;
};

[[nodiscard]] FlashExpertCachePlan
loadFlashExpertCachePlan(const std::filesystem::path &path,
                         std::string_view expectedSourceIdentity);

} // namespace splash::flash

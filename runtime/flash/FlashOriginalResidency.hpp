#pragma once

#include <cstdint>
#include <limits>
#include <string_view>

namespace splash::flash {

enum class OriginalResidencyCategory : uint32_t {
  Text = 1, MTP = 2, PLE = 4, Vision = 8, Unknown = 16,
};

inline uint32_t flashOriginalResidencyCategory(std::string_view name) noexcept {
  if (name.starts_with("vision_tower.") || name.find("vision") != std::string_view::npos)
    return static_cast<uint32_t>(OriginalResidencyCategory::Vision);
  if (name.find(".ple.") != std::string_view::npos || name.find("ngram") != std::string_view::npos)
    return static_cast<uint32_t>(OriginalResidencyCategory::PLE);
  if (name.starts_with("mtp.")) return static_cast<uint32_t>(OriginalResidencyCategory::MTP);
  if (name.starts_with("language_model.")) return static_cast<uint32_t>(OriginalResidencyCategory::Text);
  return static_cast<uint32_t>(OriginalResidencyCategory::Unknown);
}

inline bool flashOriginalResidencyBaseEligible(uint32_t mask) noexcept {
  constexpr uint32_t allowed = static_cast<uint32_t>(OriginalResidencyCategory::Text) |
      static_cast<uint32_t>(OriginalResidencyCategory::MTP);
  return (mask & allowed) != 0 && (mask & ~allowed) == 0;
}

inline bool flashOriginalResidencyHostAllowed(bool hostValid, bool growthAllowed,
    bool pressureNormal, uint64_t hostHeadroomBytes, uint64_t originalMappedBytes) noexcept {
  constexpr uint64_t margin = 2ULL << 30;
  return hostValid && growthAllowed && pressureNormal && originalMappedBytes &&
      originalMappedBytes <= std::numeric_limits<uint64_t>::max() - margin &&
      hostHeadroomBytes >= originalMappedBytes + margin;
}

} // namespace splash::flash

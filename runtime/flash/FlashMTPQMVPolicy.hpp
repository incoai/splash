#pragma once

#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace splash::flash {

inline constexpr const char *kFlashMTPQMVF32ProposalSemantics =
    "mtp-proposal-fc-hidden-only-original-q4g64-contiguous-qmv-f32xsum-physical4to128-logical1to8-v1";

// Strict proposal-only switch; constructor dependency validation keeps the
// same F32 activation-sum profile as the already-qualified target runtime.
[[nodiscard]] inline bool flashMTPQMVF32Switch(const char *raw, bool globalF32) {
  if (!raw || std::string_view(raw) == "0") return false;
  if (std::string_view(raw) != "1")
    throw std::invalid_argument("SPLASH_FLASH_MTP_QMV_F32 must be 0 or 1");
  if (!globalF32)
    throw std::invalid_argument("SPLASH_FLASH_MTP_QMV_F32 requires SPLASH_FLASH_QMV_F32=1");
  return true;
}

[[nodiscard]] constexpr bool flashMTPQMVF32Geometry(
    uint32_t experts, uint32_t outputs, uint32_t inputs, uint32_t bits,
    uint32_t group, uint64_t weightStride, uint64_t parameterStride) noexcept {
  return experts == 1 && outputs == 2560 && inputs == 2560 && bits == 4 &&
      group == 64 && weightStride == 1280 && parameterStride == 80;
}

// A real pair has four stream vectors. Joint callers pass their minimum real
// fold length; concatenating lanes never changes the priming eligibility.
[[nodiscard]] constexpr bool flashMTPQMVF32Window(
    uint32_t physicalRows, uint32_t logicalRows) noexcept {
  return logicalRows >= 1 && logicalRows <= 8 && physicalRows >= 4 &&
      physicalRows <= 128 && physicalRows % 4 == 0 && physicalRows >= logicalRows * 4;
}

} // namespace splash::flash

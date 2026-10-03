#pragma once

// Layout constants of the weight files: preparation writes them, the weight
// store reads them. Preparation code takes them from this header, so the
// preparation identity does not follow the reader's API.
// Editing this file re-prepares every affine and vision model.

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string_view>

namespace splash::model {

inline constexpr uint32_t kQ4GroupElements = 64;
inline constexpr uint64_t kBFloat16Bytes = 2;

inline constexpr uint64_t kWeightFileAlignment = 16 * 1024;
inline constexpr uint32_t kQ4StorageN = 256;

// The packed vision tower.
inline constexpr std::string_view kVisionMagic = "MDFV0001";

// offset rounded up to the next section boundary.
[[nodiscard]] inline constexpr uint64_t alignWeightOffset(uint64_t offset) {
  return (offset + kWeightFileAlignment - 1) & ~(kWeightFileAlignment - 1);
}

// The 16-byte header a weight file starts with: its eight-byte magic, then
// its layer and type, little-endian.
[[nodiscard]] inline std::array<uint8_t, 16> weightFileHeader(std::string_view magic, uint32_t layer,
                                                              uint32_t type) {
  if (magic.size() != 8) throw std::invalid_argument("a weight file magic is eight bytes");
  std::array<uint8_t, 16> header{};
  std::memcpy(header.data(), magic.data(), 8);
  std::memcpy(header.data() + 8, &layer, 4);
  std::memcpy(header.data() + 12, &type, 4);
  return header;
}

} // namespace splash::model

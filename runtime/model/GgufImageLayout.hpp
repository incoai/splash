#pragma once

// The layout of a GGUF target image: the planner writes it, the weight store
// reads it. After the 16-byte file header (model/WeightLayout.hpp)
// come 16 KiB-aligned sections. A quantized tensor is a descriptor section,
// then its plane0, optional plane1 and meta sections in its format's layout
// (metal/abi/QuantFormat.h). A tensor copied as stored (the token
// embedding's native rows; the F32 MoE router, shared-expert gate and
// alpha/beta) is a descriptor and its rows; a norm, the convolution and the
// GDN head vectors are their rows alone, F32 or narrowed to the bf16 values
// they equal.

#include "metal/abi/QuantFormat.h"

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>

namespace splash::model {

inline constexpr std::string_view kGgufImageMagic = "MDGG0001";

// Columns of a superblock, the unit a quantized tensor's width is a multiple of.
inline constexpr uint64_t kGgufBlockColumns = 256;

// A tensor's descriptor section: its seven 32-bit fields from byte 0, then
// its three 64-bit fields from byte 32, little-endian.
struct GgufTensorDescriptor {
  static constexpr uint64_t kBytes = 56;

  uint32_t type = 0;       // ggml type
  uint32_t outputSize = 0; // rows
  uint32_t inputSize = 0;  // columns
  uint32_t p0 = 0, p1 = 0; // plane0 and plane1 bytes per 32 columns of a row
  uint32_t metaBytes = 0;  // meta bytes per metaGroups 32-column groups of a row
  uint32_t metaGroups = 0;
  uint64_t plane0Bytes = 0, plane1Bytes = 0, metaTotalBytes = 0;

  [[nodiscard]] std::array<uint8_t, kBytes> encode() const noexcept {
    const std::array<uint32_t, 7> words{type, outputSize, inputSize, p0, p1, metaBytes, metaGroups};
    const std::array<uint64_t, 3> totals{plane0Bytes, plane1Bytes, metaTotalBytes};
    std::array<uint8_t, kBytes> bytes{};
    std::memcpy(bytes.data(), words.data(), sizeof words);
    std::memcpy(bytes.data() + 32, totals.data(), sizeof totals);
    return bytes;
  }
  [[nodiscard]] static GgufTensorDescriptor decode(std::span<const uint8_t, kBytes> bytes) noexcept {
    std::array<uint32_t, 7> words;
    std::array<uint64_t, 3> totals;
    std::memcpy(words.data(), bytes.data(), sizeof words);
    std::memcpy(totals.data(), bytes.data() + 32, sizeof totals);
    return {words[0], words[1], words[2], words[3], words[4], words[5], words[6], totals[0], totals[1], totals[2]};
  }
};
static_assert(std::endian::native == std::endian::little, "GGUF image fields are little-endian");

struct GgufPlaneBytes {
  uint64_t plane0, plane1, meta;
};

// The planes of a [rows, columns] tensor in format.
[[nodiscard]] inline constexpr GgufPlaneBytes ggufPlaneBytes(const QuantFormat &format, uint64_t rows,
                                                             uint64_t columns) {
  const uint64_t groups = columns / 32;
  return {rows * groups * format.plane0_bytes, rows * groups * format.plane1_bytes,
          rows * (groups / format.meta_groups) * format.meta_bytes};
}

// Bytes of `columns` values of a row in format's GGUF blocks.
[[nodiscard]] inline constexpr uint64_t ggufRowBytes(const QuantFormat &format, uint64_t columns) {
  return columns / format.block_elements * format.block_bytes;
}

} // namespace splash::model

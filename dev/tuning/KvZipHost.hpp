#pragma once

#include "metal/abi/KvExtent.h"
#include "metal/abi/KvZip.h"
#include "tuning/HostKvExtents.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

// Host reference codec of ZBF16 pages (metal/abi/KvZip.h): what the store
// kernels write and the attention kernels read, for kernel tests and the
// attention fixture.
namespace splash::ops::tuning::kvzip {

constexpr uint32_t kRows = SPLASH_KVZIP_ROWS;
constexpr uint32_t kDims = SPLASH_KVZIP_DIMENSIONS;

// A head's windows: ZBF16 codes against base4; base3 is the calibration
// format's other window.
struct HeadBases final {
  std::array<uint8_t, kDims> base3{};
  std::array<uint8_t, kDims> base4{};
};

// One ZBF16 page of a host pool, in one layer.
struct ZipPage final {
  const HostKvExtents &extents;
  uint32_t kvHeads, page, layer;

  uint8_t *data(uint32_t tensor, uint32_t head) const {
    return extents.slab<uint8_t>(layer, tensor ? SPLASH_KV_VALUES : SPLASH_KV_KEYS, page) +
           head * SPLASH_KVZIP_DATA_BYTES_PER_HEAD;
  }
  uint8_t *aux(uint32_t tensor, uint32_t head) const {
    return extents.slab<uint8_t>(layer, tensor ? SPLASH_KV_VALUE_SCALES : SPLASH_KV_KEY_SCALES, page) +
           head * SPLASH_KVZIP_AUX_BYTES_PER_HEAD;
  }
  uint32_t count(uint32_t tensor, uint32_t head) const {
    uint32_t value;
    std::memcpy(&value, aux(tensor, head), 4);
    return value;
  }
  uint32_t *escapes(uint32_t tensor, uint32_t head) const {
    return reinterpret_cast<uint32_t *>(aux(tensor, head) + SPLASH_KVZIP_ESCAPE_HEADER_BYTES);
  }
};

// One row's codes and escapes.
struct RowCode final {
  std::array<uint8_t, kDims> code{};
  std::vector<uint32_t> escapes; // position | exponent << 16
};

inline RowCode encodeRow(const uint16_t *row, uint32_t rowIndex, const HeadBases &bases) {
  RowCode code;
  for (uint32_t dimension = 0; dimension < kDims; ++dimension) {
    const uint32_t exponent = (row[dimension] >> 7) & 0xFF;
    const int delta = int(exponent) - bases.base4[dimension];
    if (delta >= 0 && delta < 16) {
      code.code[dimension] = uint8_t(delta);
    } else {
      code.escapes.push_back((rowIndex * kDims + dimension) | (exponent << 16));
    }
  }
  return code;
}

// Stores rows [begin, end) of a slab as the store kernels do; `rows` holds
// the page's 32 token-major rows. Rows before `begin` keep their escapes.
// Returns whether an escape was dropped.
inline bool storeRows(const ZipPage &page, uint32_t tensor, uint32_t head, const uint16_t *rows,
                      uint32_t begin, uint32_t end, const HeadBases &bases) {
  uint8_t *data = page.data(tensor, head);
  uint32_t *escapes = page.escapes(tensor, head);
  uint32_t count = 0;
  if (begin) {
    count = std::min(page.count(tensor, head), SPLASH_KVZIP_ESCAPES);
    for (uint32_t index = 0; index < count; ++index)
      if ((escapes[index] & 0xFFFF) / kDims >= begin) {
        count = index;
        break;
      }
  }
  bool dropped = false;
  for (uint32_t row = begin; row < end; ++row) {
    const RowCode code = encodeRow(rows + row * kDims, row, bases);
    for (uint32_t dimension = 0; dimension < kDims; ++dimension) {
      const uint32_t bits = rows[row * kDims + dimension];
      data[row * kDims + dimension] = uint8_t(((bits >> 8) & 0x80) | (bits & 0x7F));
    }
    for (uint32_t pair = 0; pair < kDims / 2; ++pair)
      data[SPLASH_KVZIP_SM_BYTES + row * SPLASH_KVZIP_CODE_BYTES_PER_ROW + pair] =
          uint8_t(code.code[2 * pair] | (code.code[2 * pair + 1] << 4));
    for (const uint32_t escape : code.escapes) {
      if (count < SPLASH_KVZIP_ESCAPES)
        escapes[count++] = escape;
      else
        dropped = true;
    }
  }
  std::memcpy(page.aux(tensor, head), &count, 4);
  return dropped;
}

// Decodes rows [0, rows) of a slab into BF16 bits, token-major; a dropped
// escape's element decodes as the kernels decode it.
inline std::vector<uint16_t> decodeSlab(const ZipPage &page, uint32_t tensor, uint32_t head,
                                        uint32_t rows, const HeadBases &bases) {
  const uint8_t *data = page.data(tensor, head);
  std::vector<uint16_t> result(rows * kDims);
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t dimension = 0; dimension < kDims; ++dimension) {
      const uint8_t pair = data[SPLASH_KVZIP_SM_BYTES + row * SPLASH_KVZIP_CODE_BYTES_PER_ROW + dimension / 2];
      const uint32_t code = dimension & 1 ? pair >> 4 : pair & 0xF;
      const uint32_t exponent = (bases.base4[dimension] + code) & 0xFF;
      const uint32_t sm = data[row * kDims + dimension];
      result[row * kDims + dimension] = uint16_t(((sm & 0x80) << 8) | (exponent << 7) | (sm & 0x7F));
    }
  const uint32_t count = std::min(page.count(tensor, head), SPLASH_KVZIP_ESCAPES);
  const uint32_t *escapes = page.escapes(tensor, head);
  for (uint32_t index = 0; index < count; ++index) {
    const uint32_t position = escapes[index] & 0xFFFF;
    if (position / kDims < rows)
      result[position] = uint16_t((result[position] & 0x807F) | (((escapes[index] >> 16) & 0xFF) << 7));
  }
  return result;
}

// The best 8- and 16-binade windows of one (tensor, KV head) from each
// dimension's exponent histogram, as dev/tools/kvzip_calibrate.py fits them.
inline HeadBases windows(const std::vector<std::array<uint32_t, 256>> &histograms) {
  HeadBases bases;
  for (uint32_t dimension = 0; dimension < kDims; ++dimension) {
    std::array<uint64_t, 257> cumulative{};
    for (uint32_t exponent = 0; exponent < 256; ++exponent)
      cumulative[exponent + 1] = cumulative[exponent] + histograms[dimension][exponent];
    for (uint32_t width : {8U, 16U}) {
      uint64_t best = 0;
      uint32_t base = 0;
      for (uint32_t start = 0; start + width <= 256; ++start)
        if (cumulative[start + width] - cumulative[start] > best)
          best = cumulative[start + width] - cumulative[start], base = start;
      (width == 8 ? bases.base3 : bases.base4)[dimension] = uint8_t(base);
    }
  }
  return bases;
}

} // namespace splash::ops::tuning::kvzip

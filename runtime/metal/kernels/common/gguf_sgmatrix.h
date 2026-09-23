#pragma once
#include "metal/kernels/common/q4_sgmatrix.h"

// The eight-row X^T table the Apple9 GGUF register kernel reads
// (LinearInput::Table16). A 64-element span occupies 512 bfloat in the
// q4sg::xt_offset order; its fragments follow the chunk order of the GGUF
// image (metal/abi/QuantFormat.h): fragment 4q + f holds pair f of every
// chunk of the span's 32-element group q, so fragments 4q + 2h and
// 4q + 2h + 1 cover its 16-element group h. Row sums, one fp32 per row and
// 16 or 32 inputs, are stored row-minor so a lane reads its two rows at once.
namespace q16sg {

// Physical k inside a span -> (fragment j, row k').
inline uint2 klogical(uint k) {
  const uint q = k >> 5, h = (k >> 4) & 1, a = (k >> 2) & 3, b = (k >> 1) & 1, e = k & 1;
  return uint2(4 * q + 2 * h + b, 2 * a + e);
}

// Per 8-row tile of width K: K * 8 bfloat, then sums16 [K / 16][8 rows] and
// sums32 [K / 32][8 rows].
inline ulong sums32_offset(uint width) { return ulong(width) / 16 * 8; }
inline ulong sums_per_tile(uint width) { return sums32_offset(width) + ulong(width) / 32 * 8; }

// One simdgroup writes one span of one row; the lane holds elements
// 2 lane, 2 lane + 1.
inline void write_input(device bfloat *table, device float *sums, uint width, uint span,
                        uint row, uint lane, bfloat a, bfloat b) {
  const uint2 l = klogical(2 * lane);
  table[span * q4sg::kXtPerGroup + q4sg::xt_offset(l.x, l.y, row)] = a;
  table[span * q4sg::kXtPerGroup + q4sg::xt_offset(l.x, l.y + 1, row)] = b;
  float s = float(a) + float(b);
  s += simd_shuffle_xor(s, 1u);
  s += simd_shuffle_xor(s, 2u);
  s += simd_shuffle_xor(s, 4u);
  if ((lane & 7) == 0) sums[(span * 4 + lane / 8) * 8 + row] = s;
  const float s32 = s + simd_shuffle_xor(s, 8u);
  if ((lane & 15) == 0) sums[sums32_offset(width) + (span * 2 + lane / 16) * 8 + row] = s32;
}

struct Table16 {
  static ulong sums_per_tile(uint width) { return q16sg::sums_per_tile(width); }
  static void write(device bfloat *table, device float *sums, uint width, uint span, uint row,
                    uint lane, bfloat a, bfloat b) {
    write_input(table, sums, width, span, row, lane, a, b);
  }
};

} // namespace q16sg

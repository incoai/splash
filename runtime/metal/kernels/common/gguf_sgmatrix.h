#pragma once
#include "metal/abi/Gguf.h"
#include <metal_stdlib>
using namespace metal;

// The eight-row X^T table the Apple9 GGUF register kernel reads
// (LinearInput::Table16, laid out as metal/abi/Gguf.h states). A 64-element
// span occupies 512 bfloat in the xt_offset order; its fragments follow the chunk order of the GGUF
// image (metal/abi/QuantFormat.h): fragment 4q + f holds pair f of every
// chunk of the span's 32-element group q, so fragments 4q + 2h and
// 4q + 2h + 1 cover its 16-element group h. Per row, the sum of every 32
// inputs (formats with a min) and the chain seed of every 16 inputs (formats
// with a zero point) are stored row-minor, so a lane reads its two rows at
// once.
namespace gguf_sg {

// Formats with a zero point (Q6_K, Q3_K) enter the MMA as 160 + code - zero,
// exact in bf16 (160 = 128 + 32, Q6_K's zero point), so each 16-input chain
// starts from -160 times the sum of its inputs: the seed the table stores.
constant constexpr uint kZeroPointOffset = 160;

// The rows of a table: one lane's verify rows.
constant constexpr uint kRows = 8;
static_assert(GGUF_TABLE16_SPAN_VALUES == GGUF_TABLE16_SPAN_INPUTS * kRows, "a span holds every row's inputs");

// A span's values laid out as quads, so the lane (fm, fn) reads fragments
// 4 jq .. 4 jq + 3 for (k' = fm, m = fn, fn + 1) as one vec<T, 8>:
//   offset(j, k', m) = (((j >> 2) 8 + k') 4 + (m >> 1)) 8 + (j & 3) 2 + (m & 1)
inline uint xt_offset(uint j, uint kp, uint m) {
  return (((j >> 2) * 8 + kp) * 4 + (m >> 1)) * 8 + (j & 3) * 2 + (m & 1);
}

// Physical k inside a span -> (fragment j, row k').
inline uint2 klogical(uint k) {
  const uint q = k >> 5, h = (k >> 4) & 1, a = (k >> 2) & 3, b = (k >> 1) & 1, e = k & 1;
  return uint2(4 * q + 2 * h + b, 2 * a + e);
}

// One simdgroup writes one span of one row; the lane holds elements
// 2 lane, 2 lane + 1.
inline void write_input(device bfloat *table, device float *sums, uint width, uint span,
                        uint row, uint lane, bfloat a, bfloat b) {
  const uint2 l = klogical(2 * lane);
  table[span * GGUF_TABLE16_SPAN_VALUES + xt_offset(l.x, l.y, row)] = a;
  table[span * GGUF_TABLE16_SPAN_VALUES + xt_offset(l.x, l.y + 1, row)] = b;
  float s = float(a) + float(b);
  s += simd_shuffle_xor(s, 1u);
  s += simd_shuffle_xor(s, 2u);
  s += simd_shuffle_xor(s, 4u);
  if ((lane & 7) == 0) sums[(span * (GGUF_TABLE16_SPAN_SEEDS / kRows) + lane / 8) * kRows + row] = -float(kZeroPointOffset) * s;
  const float s32 = s + simd_shuffle_xor(s, 8u);
  if ((lane & 15) == 0) sums[table16_sums32_offset(width) + (span * (GGUF_TABLE16_SPAN_SUMS / kRows) + lane / 16) * kRows + row] = s32;
}

struct Table16 {
  static ulong sums_per_tile(uint width) { return table16_sums_per_tile(width); }
  static void write(device bfloat *table, device float *sums, uint width, uint span, uint row,
                    uint lane, bfloat a, bfloat b) {
    write_input(table, sums, width, span, row, lane, a, b);
  }
};

} // namespace gguf_sg

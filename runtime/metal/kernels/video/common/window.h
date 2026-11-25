#pragma once

// Register layout of the 16-row attention tiles and the final store shared by the window-attention kernels.
//
// MPP cooperative tensors of one simdgroup (QK destination 16 x 32, P left input 16 x 32, PV destination 16 x D),
// measured on M5 / macOS 27 and verified at start-up (shared/layout_check.metal):
//   slot i -> row r0 + 8 ((i >> 2) & 1), column 16 (i >> 3) + cb + (i & 3),
//   r0 = ((lane >> 1) & 3) + 4 (lane >> 4), cb = 4 (lane & 1) + 8 ((lane >> 3) & 1).
// The 4 lanes sharing a row are {L, L^1, L^8, L^9}: row reductions are simd_shuffle_xor 1 and 8.
#include "metal/abi/video/AttentionABI.h"
#include "metal/kernels/video/common/tensor.h"

namespace h3 {
constant constexpr int kHeadDim = 128, kRowsPerTile = 16, kKeysPerTile = 32, kScoreSlots = 16, kOutSlots = 64;

inline ushort lane_r0(ushort lane) { return ((lane >> 1) & 3) + 4 * (lane >> 4); }
inline ushort lane_cb(ushort lane) { return 4 * (lane & 1) + 8 * ((lane >> 3) & 1); }

// col inside the first n of the row's (up to 3) half-open key ranges
inline bool in3(int col, int n, int2 r0, int2 r1, int2 r2) {
  return (n > 0 && col >= r0.x && col < r0.y) || (n > 1 && col >= r1.x && col < r1.y) ||
         (n > 2 && col >= r2.x && col < r2.y);
}

// Final store of one 16-row tile: direct (normalized bf16 into O) or a key-chunk partial (normalized fp16 O plus
// (m, l); l == 0 -> m = -inf) that dit_window_combine merges.
template <class OT>
inline void window_finish(thread OT &oT, float mA, float mB, float lA, float lB, int h, int hl, int q0, int slot,
                          int local0, ushort r0, ushort cb, constant WindowParams &p, device bfloat *O,
                          device half *Op, device float2 *ML) {
  lA += simd_shuffle_xor(lA, ushort(1));
  lA += simd_shuffle_xor(lA, ushort(8));
  lB += simd_shuffle_xor(lB, ushort(1));
  lB += simd_shuffle_xor(lB, ushort(8));
  const float iA = lA > 0.0f ? 1.0f / lA : 0.0f, iB = lB > 0.0f ? 1.0f / lB : 0.0f;
  if (slot < 0) {
    device bfloat *Oh = O + h * kHeadDim;
#pragma unroll
    for (ushort i = 0; i < kOutSlots; i += 4) {
      const bool b = (i >> 2) & 1;
      const int row = q0 + r0 + (b ? 8 : 0);
      const int col = 16 * (i >> 3) + cb;
      if (row >= p.S) continue;
      float4 v = float4(oT[i], oT[i + 1], oT[i + 2], oT[i + 3]) * (b ? iB : iA);
      *(device vec<bfloat, 4> *)(Oh + ulong(row) * p.ldo + col) = vec<bfloat, 4>(v);
    }
    return;
  }
  const ulong base = (ulong(hl) * p.nslot + slot) * 64;
#pragma unroll
  for (ushort i = 0; i < kOutSlots; i += 4) {
    const bool b = (i >> 2) & 1;
    const int lr = local0 + r0 + (b ? 8 : 0);
    const int col = 16 * (i >> 3) + cb;
    float4 v = float4(oT[i], oT[i + 1], oT[i + 2], oT[i + 3]) * (b ? iB : iA);
    *(device half4 *)(Op + (base + lr) * kHeadDim + col) = half4(v);
  }
  if (cb == 0) {
    ML[base + local0 + r0] = float2(lA > 0.0f ? mA : -INFINITY, lA);
    ML[base + local0 + r0 + 8] = float2(lB > 0.0f ? mB : -INFINITY, lB);
  }
}
}  // namespace h3

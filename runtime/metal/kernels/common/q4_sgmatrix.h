#pragma once
#include "metal/abi/Linear.h"
#include <metal_stdlib>
using namespace metal;

// Eight-row decode activation layout shared by the normalizer, pre-pass and
// register matrix kernel. One 64-element group occupies 512 bfloat values.
// A batch concatenates these eight-row tables, one per request lane.
namespace q4sg {
// simdgroup_matrix lane -> element mapping (verified by the driver's probe):
// a lane's thread_elements() are M[fm][fn] and M[fm][fn + 1].
struct Lane {
  ushort fm;
  ushort fn;
};
inline Lane lane_map(uint lane) {
  const uint qid = lane >> 2;
  Lane l;
  l.fm = ushort((qid & 4) | ((lane >> 1) & 3));
  l.fn = ushort(((qid & 2) << 1) | ((lane & 1) << 1));
  return l;
}

// Inverse: physical k -> (fragment j, row k').
inline uint2 klogical(uint k) {
  const uint c = k >> 4, r = k & 15;
  return uint2((r >> 3) * 4 + (r & 3), 2 * c + ((r >> 2) & 1));
}

template <typename T>
__attribute__((always_inline)) inline thread vec<T, 2> &te(thread simdgroup_matrix<T, 8, 8> &m) {
  return reinterpret_cast<thread vec<T, 2> &>(m.thread_elements());
}

// One 8x8x8 MMA on plain-register operands: c += a x b. The simdgroup_matrix
// objects live only inside this call (the MLX steel pattern), so the
// persistent accumulator is an ordinary float2 the compiler keeps in
// registers; a persistent simdgroup_matrix read through thread_elements()
// each group was kept in thread memory instead.
template <typename T>
__attribute__((always_inline)) inline void
mma_acc(thread float2 &c, vec<T, 2> a, vec<T, 2> b) {
  simdgroup_matrix<T, 8, 8> A, B;
  simdgroup_matrix<float, 8, 8> C, D;
  te(A) = a;
  te(B) = b;
  te(C) = c;
  simdgroup_multiply_accumulate(D, A, B, C);
  c = te(D);
}

// X^T table: 512 T per group, laid out as quads so the lane (fm, fn) reads
// fragments 4 jq .. 4 jq + 3 for (k' = fm, m = fn, fn + 1) as one vec<T, 8>:
//   offset(g, j, k', m) = g 512 + (((j >> 2) 8 + k') 4 + (m >> 1)) 8
//                         + (j & 3) 2 + (m & 1)
constexpr constant uint kRows = 8;
constexpr constant uint kXtPerGroup = 512;
inline uint xt_offset(uint j, uint kp, uint m) {
  return (((j >> 2) * 8 + kp) * 4 + (m >> 1)) * 8 + (j & 3) * 2 + (m & 1);
}


inline void write_input(device bfloat *table, device float *sums,
                        uint group, uint row, uint lane, bfloat a, bfloat b) {
  const uint2 logical = klogical(2 * lane);
  table[group * kXtPerGroup + xt_offset(logical.x, logical.y, row)] = a;
  table[group * kXtPerGroup + xt_offset(logical.x + 1, logical.y, row)] = b;
  const float sum = simd_sum(float(a) + float(b));
  if (lane == 0) sums[group * kRows + row] = sum;
}

// The affine table as a producer target (LinearInput::Table64): per 8-row
// tile of width K, K * 8 bfloat and K / 8 sums. One simdgroup writes one
// 64-element group of one row; the lane holds elements 2 lane, 2 lane + 1.
struct Table64 {
  static ulong sums_per_tile(uint width) { return width / 8; }
  static void write(device bfloat *table, device float *sums, uint, uint group, uint row,
                    uint lane, bfloat a, bfloat b) {
    write_input(table, sums, group, row, lane, a, b);
  }
};
} // namespace q4sg

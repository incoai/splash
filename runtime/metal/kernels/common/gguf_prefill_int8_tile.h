#pragma once
// The int8-activation tile for MLX affine 4- and 5-bit weights in groups of 64 (af4g64, af5g64), which the dense
// prefill GEMMs (kernels/prefill/gguf_prefill_int8.metal) and the MoE experts' prefill passes
// (kernels/shared/moe_int8.metal) run: see gguf_prefill_int8.metal for its arithmetic and fragment schedule.
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/gguf_tile.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;

// What a tile reads and writes: K inputs a row, `rows` rows of codes from the codes' start (the scales' row stride
// storage_rows), destination rows of out_stride columns from out_offset on, and planes of plane_input_size inputs a
// row.
struct PrefillInt8Shape {
  uint input_size, rows, storage_rows, out_stride, out_offset, plane_input_size;
};

constant constexpr int kInt8FragM = 16, kInt8FragN = 32, kInt8FragK = 16, kInt8RowFrags = 2, kInt8Elements = 16;

// This lane's (column, row) in matmul2d's int8 16x16 fragment layout, the operands' and the destination's, which
// gguf-projection checks on each family. Taking the destination's from get_multidimensional_index instead loads each
// element's scales alone: up to 25% more time (16-core M5 Pro, measured).
inline short2 prefill_int8_coord(ushort lane) {
  const short qid = lane >> 2;
  return short2(((qid & 2) | (lane & 1)) * 4, (qid & 4) | ((lane >> 1) & 3));
}

// Four 5-bit codes: the step's nibbles and the fifth bits of chunk byte `b` at slots {0, 4, 1, 5} (or, shifted by
// two, {2, 6, 3, 7}) moved to bit 4 of each byte.
inline uint prefill_int8_fifth(uint b) {
  return ((b & 1u) | ((b & 0x10u) << 4) | ((b & 2u) << 15) | ((b & 0x20u) << 19)) << 4;
}

// A tile of GGUF_PREFILL_INT8_SIMDGROUPS simdgroups, each 32 rows (two 16-row fragments) by 32 columns, at (column
// tile, row tile) `tile`.
template <int BITS, GgufEpilogue Ep>
inline void prefill_int8_tile(device const char *codes, device const float2 *scales, device const uchar *w0,
                              device const uchar *w1, device const uchar *meta, device bfloat *output,
                              device const bfloat *aux, PrefillInt8Shape p, uint2 tile, ushort lane,
                              ushort simd_group) {
  using namespace mpp::tensor_ops;
  constexpr int BM = kInt8RowFrags * kInt8FragM, BN = GGUF_TILE_COLUMNS;
  static_assert(BN == kInt8FragN * GGUF_PREFILL_INT8_SIMDGROUPS, "the simdgroups cover the column tile");
  static_assert(BM == GGUF_PREFILL_INT8_ROWS, "a simdgroup holds the tile's rows");
  constexpr uint T = QUANT_TILE_ROWS;
  const int K = int(p.input_size), M = int(p.rows), groups = K / GGUF_PREFILL_INT8_GROUP;
  const uint plane_groups32 = p.plane_input_size / 32, plane_units = p.plane_input_size / GGUF_PREFILL_INT8_GROUP;
  const int row_base = int(tile.y) * BM, col_base = int(tile.x) * BN + simd_group * kInt8FragN;
  const short2 coord = prefill_int8_coord(lane);
  const int cx = coord.x >> 2;

  constexpr auto desc = matmul2d_descriptor(kInt8FragM, kInt8FragN, kInt8FragK, false, true, false,
                                            matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto desc_set = matmul2d_descriptor(kInt8FragM, kInt8FragN, kInt8FragK, false, true, false,
                                                matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  matmul2d<desc_set, execution_simdgroup> op_set;
  auto ct_a = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_b = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto acc0 = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), int32_t>();
  auto acc1 = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), int32_t>();

  float cf[kInt8RowFrags][kInt8Elements];
#pragma unroll
  for (int i = 0; i < kInt8RowFrags; ++i)
#pragma unroll
    for (int e = 0; e < kInt8Elements; ++e) cf[i][e] = 0.0f;

  // The lane's weight column n_lane + 8 (q & 1) + 16 (q >> 1) for right-operand run q, all in one plane tile (a
  // 64-column tile never straddles one): plane0 records of 16 bytes, plane1 of 4, a 32-group T records further on.
  // The lane reads bytes 8 (x & 1) .. +7 of 32-group 2g + (x >> 1).
  const uint n_lane = uint(col_base + coord.y);
  const ulong record0 = ulong(n_lane / T) * plane_groups32 * T + n_lane % T;
  device const uchar *wl0 = w0 + record0 * 16 + 8 * (cx & 1);
  device const uchar *wl1 = w1 + record0 * 4 + 2 * (cx & 1);
  const int h32 = cx >> 1;
  // Meta (s, z as bf16 in one uint) of the lane's destination columns n_run + 16h + j: 4 consecutive records.
  const uint n_run = uint(col_base + coord.x);
  device const uint *ml = (device const uint *)(meta) + ulong(n_run / T) * plane_units * T + n_run % T;
  const int m_base = row_base + coord.y;

  for (int g = 0; g < groups; ++g) {
    const ulong g32 = ulong(2 * g + h32) * T;
    uint2 wv[4];
    uint hb[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int off = (q & 1) * 8 + (q >> 1) * 16;
      wv[q] = *((device const uint2 *)(wl0 + (g32 + off) * 16));
      if (BITS == 5) hb[q] = *((device const ushort *)(wl1 + (g32 + off) * 4));
    }
#pragma unroll
    for (int t = 0; t < 4; ++t) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint bits = (wv[q][t >> 1] >> (4 * (t & 1))) & 0x0f0f0f0fu;
        if (BITS == 5) bits |= prefill_int8_fifth((hb[q] >> (8 * (t >> 1) + 2 * (t & 1))) & 0xFFu);
        const char4 quad = as_type<char4>(bits);
        const int base = (q >> 1) * 8 + (q & 1) * 4;
        ct_b[base + 0] = quad.x; ct_b[base + 1] = quad.y; ct_b[base + 2] = quad.z; ct_b[base + 3] = quad.w;
      }
#pragma unroll
      for (int hf = 0; hf < kInt8RowFrags; ++hf) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const int m = min(m_base + hf * kInt8FragM + r * 8, M - 1);
          const ulong at = ulong(m) * K + g * GGUF_PREFILL_INT8_GROUP + cx * 16 + t * 4;
          const char4 quad = as_type<char4>(*((device const uint *)(codes + at)));
          ct_a[r * 4 + 0] = quad.x; ct_a[r * 4 + 1] = quad.y; ct_a[r * 4 + 2] = quad.z; ct_a[r * 4 + 3] = quad.w;
        }
        if (t == 0) { if (hf == 0) op_set.run(ct_a, ct_b, acc0); else op_set.run(ct_a, ct_b, acc1); }
        else { if (hf == 0) op.run(ct_a, ct_b, acc0); else op.run(ct_a, ct_b, acc1); }
      }
    }
    // The group's correction: s, z of the lane's two runs of four columns, and the scales of its four rows.
    const uint4 mv0 = *((device const uint4 *)(ml + ulong(g) * T));
    const uint4 mv1 = *((device const uint4 *)(ml + ulong(g) * T + 16));
    float2 sa[kInt8RowFrags][2];
#pragma unroll
    for (int i = 0; i < kInt8RowFrags; ++i)
#pragma unroll
      for (int r = 0; r < 2; ++r)
        sa[i][r] = scales[ulong(g) * p.storage_rows + min(m_base + i * kInt8FragM + r * 8, M - 1)];
#pragma unroll
    for (int e = 0; e < kInt8Elements; ++e) {
      const int r = (e & 7) >> 2;
      const uint mt = (e >> 3) ? mv1[e & 3] : mv0[e & 3];
      const float s = float(as_type<bfloat>(ushort(mt & 0xFFFF))), z = float(as_type<bfloat>(ushort(mt >> 16)));
      cf[0][e] = fma(sa[0][r].x * s, float(acc0[e]), fma(z, sa[0][r].y, cf[0][e]));
      cf[1][e] = fma(sa[1][r].x * s, float(acc1[e]), fma(z, sa[1][r].y, cf[1][e]));
    }
  }
#pragma unroll
  for (int i = 0; i < kInt8RowFrags; ++i)
#pragma unroll
    for (int e = 0; e < kInt8Elements; ++e) {
      const int m = m_base + i * kInt8FragM + ((e & 7) >> 2) * 8;
      if (m >= M) continue;
      const ulong o = ulong(m) * p.out_stride + p.out_offset + n_run + (e & 3) + 16 * (e >> 3);
      output[o] = gguf_epilogue<Ep>(cf[i][e], aux, o);
    }
}

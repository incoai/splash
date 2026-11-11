// Int8-activation prefill GEMMs for MLX affine 4- and 5-bit weights in groups of 64 (af4g64, af5g64)
// (ops::LinearTile::GgufPrefillInt8). Each 64-input group g of a chunk row is quantized to int8 codes qa with its own
// scale sa_g, and with w = s * q + z per weight group:
//
//   y[m][n] = sum_g ( (sa_g[m] s_g[n]) sum_{k in g} qa[m][k] q[n][k]  +  z_g[n] (sa_g[m] sum_{k in g} qa[m][k]) )
//
// The weights' codes feed int8 16x32x16 matmul2d fragments straight from the planes; each group's int32 sums fold
// into fp32 once. The weights are exact; only the activations round, per row and group.
//
// The fragment schedule and the activations' slot order are adapted from oMLX's oQ A8 kernel (Apache-2.0 notice in
// THIRD_PARTY_NOTICES): in fragment step t (0-3) of a group, lane column x (coord.x / 4) supplies inputs
// 16x + 4t .. 16x + 4t + 3, 32-group x >> 1, chunk 2 (x & 1) + (t >> 1) of the planes (each weight column's 16 codes
// of the lane are one uint2 of its 4-bit plane), nibbles (word >> 4 (t & 1)) & 0x0f0f0f0f: chunk slots {0, 4, 1, 5}
// or {2, 6, 3, 7} (metal/abi/QuantFormat.h). The quantizer writes each 32-group in that order.
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/gguf_tile.h"
#include "metal/kernels/common/int8_activations.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;

// The schedule position, within its 32-group, of the input at slot s of the planes: chunk c = s >> 3 is lane half
// c >> 1 and step half c & 1; chunk slot u holds step bit (u >> 1) & 1 and lane input ((u & 1) << 1) | (u >> 2).
inline uint prefill_int8_position(uint slot) {
  const uint c = slot >> 3, u = slot & 7;
  return 16 * (c >> 1) + 4 * (2 * (c & 1) + ((u >> 1) & 1)) + (((u & 1) << 1) | (u >> 2));
}

// grid (groups / GGUF_PREFILL_INT8_QUANTIZE_SIMDGROUPS, rows): simdgroup s of threadgroup (b, row) quantizes group
// b * GGUF_PREFILL_INT8_QUANTIZE_SIMDGROUPS + s of the row by the int8-activation rule
// (kernels/common/int8_activations.h), lane l its inputs l and 32 + l, and stores the group's scale and the scale
// times its codes' sum.
kernel void gguf_prefill_int8_quantize(device const bfloat *input [[buffer(0)]], device char *codes [[buffer(1)]],
                                       device float2 *scales [[buffer(2)]],
                                       constant GgufPrefillInt8QuantizeParams &p [[buffer(3)]],
                                       uint2 tid [[threadgroup_position_in_grid]],
                                       ushort lane [[thread_index_in_simdgroup]],
                                       ushort simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint g = tid.x * GGUF_PREFILL_INT8_QUANTIZE_SIMDGROUPS + simd_group, row = tid.y;
  if (g >= p.input_size / GGUF_PREFILL_INT8_GROUP || row >= p.rows) return;
  const ulong at = ulong(row) * p.input_size + g * GGUF_PREFILL_INT8_GROUP;
  const bfloat b0 = input[at + lane], b1 = input[at + 32 + lane];
  const float x0 = float(b0), x1 = float(b1);
  const Int8Block block = int8_activation_block(simd_max(max(fabs(x0), fabs(x1))));
  const int q0 = int8_activation_code(x0, block.inverse), q1 = int8_activation_code(x1, block.inverse);
  const uint position = prefill_int8_position(quant_slot(lane));
  codes[at + position] = char(q0);
  codes[at + 32 + position] = char(q1);
  const int sum = simd_sum(q0 + q1);
  const bool finite = simd_all(!int8_activation_not_finite(b0) && !int8_activation_not_finite(b1));
  if (lane != 0) return;
  ((device uint2 *)scales)[ulong(g) * p.storage_rows + row] =
      uint2(int8_activation_bits(block.scale, finite), int8_activation_bits(block.scale * float(sum), finite));
}

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

// The (column tile, row tile) of tile index `at` of grid.x column tiles by grid.y row tiles:
// GGUF_PREFILL_INT8_WALK row tiles take each column tile in turn before the next column tile, the last block of row
// tiles its remainder.
inline uint2 prefill_int8_walk(uint at, uint2 grid) {
  constexpr uint S = GGUF_PREFILL_INT8_WALK;
  const uint block = at / (S * grid.x), height = min(S, grid.y - block * S), i = at - block * S * grid.x;
  return uint2(i / height, block * S + i % height);
}

// A tile of GGUF_PREFILL_INT8_SIMDGROUPS simdgroups, each 32 rows (two 16-row fragments) by 32 columns, at (column
// tile, row tile) `tile`.
template <int BITS, GgufEpilogue Ep>
inline void prefill_int8_tile(device const char *codes, device const float2 *scales, device const uchar *w0,
                              device const uchar *w1, device const uchar *meta, device bfloat *output,
                              device const bfloat *aux, constant GgufPrefillInt8Params &p, uint2 tile, ushort lane,
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

// grid (GGUF_PREFILL_INT8_WALK, its tiles / GGUF_PREFILL_INT8_WALK rounded up), GGUF_PREFILL_INT8_THREADS threads:
// threadgroup (x, y) takes tile index y GGUF_PREFILL_INT8_WALK + x of the segment's column tiles by the chunk's row
// tiles of GGUF_PREFILL_INT8_ROWS (prefill_int8_walk), and those past the tiles none. A run of 16 threadgroups along
// x is so one column tile's row tiles of a block, whatever the column count; a grid of (column tiles, row tiles)
// kept them together only where the column tiles were a multiple of 16 (M6: 260 column tiles took 29% longer than
// 256 at 2048 rows). aux (buffer 6) is the residual or the gate, the output for the plain epilogue, which reads none.
#define GGUF_PREFILL_INT8_KERNEL(B, f, ep, Ep)                                                                         \
  kernel void gguf_prefill_int8_##f##_##ep(                                                                            \
      device const char *codes [[buffer(0)]], device const float2 *scales [[buffer(1)]],                               \
      device const uchar *w0 [[buffer(2)]], device const uchar *w1 [[buffer(3)]],                                      \
      device const uchar *meta [[buffer(4)]], device bfloat *output [[buffer(5)]],                                     \
      device const bfloat *aux [[buffer(6)]], constant GgufPrefillInt8Params &p [[buffer(7)]],                         \
      uint2 tid [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],                           \
      ushort simd_group [[simdgroup_index_in_threadgroup]]) {                                                          \
    const uint rows = (p.rows + GGUF_PREFILL_INT8_ROWS - 1) / GGUF_PREFILL_INT8_ROWS,                                  \
               at = tid.y * GGUF_PREFILL_INT8_WALK + tid.x;                                                            \
    if (at >= p.column_tiles * rows) return;                                                                           \
    prefill_int8_tile<B, Ep>(codes, scales, w0, w1, meta, output, aux, p,                                              \
                             prefill_int8_walk(at, uint2(p.column_tiles, rows)), lane, simd_group);                    \
  }
#define GGUF_PREFILL_INT8_FORMAT(B, f)                                                                                 \
  GGUF_PREFILL_INT8_KERNEL(B, f, a, EpNone) GGUF_PREFILL_INT8_KERNEL(B, f, r, EpResidual)                              \
  GGUF_PREFILL_INT8_KERNEL(B, f, g, EpUpWithGate)
GGUF_PREFILL_INT8_FORMAT(4, af4g64)
GGUF_PREFILL_INT8_FORMAT(5, af5g64)
#undef GGUF_PREFILL_INT8_FORMAT
#undef GGUF_PREFILL_INT8_KERNEL

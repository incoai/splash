#pragma once
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/gguf_staged.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;
using namespace mpp::tensor_ops;

// The matrix tile (kernels/decode/linear_gguf_matrix.metal): the column tile's geometry, GGUF_COLUMN_SPLITS
// simdgroups along K over the same GGUF_COLUMN_COLUMNS columns, whose simdgroups run matmul2d(16, 32, 16) on
// fragments of 16 rows instead of a lane's FMAs: each lane decodes the chunks its right-operand elements hold from the
// planes into registers (dequant_chunk) and MPP reads the rows from device memory, with no stage and no barrier in the
// K loop. A decode step of 3 to 64 rows costs its fragments' 16 rows each, so 9 rows cost what 16 do and 3 what 16 do.
// Includers set `#pragma clang fp reassociate(off)` first.

constant constexpr ushort kMatrixFragmentRows = 16, kMatrixStep = 16;

// This lane's place in matmul2d(16, 32, 16)'s simdgroup layouts (Metal 4 MPP for macOS 26.4, every family; half and
// int8 alike; gguf-projection checks it): x = 8 bit 3 + 4 bit 0 of the lane, y = 4 bit 4 + bits 1-2. Right operand
// element 4j + i is column y + 8j, input x + i of the 16-input step; destination element 8h + 4r + i is row y + 8r,
// column x + 16h + i. So in a 32-input group, chunk x / 4 of each of the lane's four columns holds its elements of
// both steps: pairs 0, 1 the first step's, pairs 2, 3 the second's.
inline ushort2 matrix_coord(ushort lane) {
  return ushort2(((lane >> 3) & 1) * 8 + (lane & 1) * 4, ((lane >> 4) & 1) * 4 + ((lane >> 1) & 3));
}

// A simdgroup's sums over the groups g_begin, g_begin + stride, ... below g_end of K of the 32 columns from `column0`
// for Fragments fragments of 16 rows of `x` (rows [0, 16 Fragments) of a matrix of K inputs a row, which MPP reads from
// device memory), fragment by fragment: fold(f, elements) for each, where elements(write) hands write(row, column,
// sum) the lane's sums of fragment f (row within the fragment), so the caller can meet its simdgroups' barriers around
// them. Every row's sums are its own. Per group a lane loads and decodes the chunk of each of its four columns its
// elements hold (dequant_chunk). Measured against this on a 12-core M6 (Inkling-2-Mini's k, o and k|v at 3 to 16
// rows): the rows loaded into the left operand from registers, live rows alone and the next group's chunks loaded
// ahead took 10-39% more time; the four lanes of a column set sharing its coefficients by simd_shuffle (each decoding
// one column's) 3% more over six formats, Q4_K 9-14%; simdgroups on contiguous K rather than interleaved 2.5% more;
// four plane addresses a lane rather than one 8% more.
template <class F, ushort Fragments, class Fold>
inline void matrix_sums(device const bfloat *x, uint input_size, device uchar *w0, device uchar *w1, device uchar *meta,
                        uint column0, uint g_begin, uint g_end, uint stride, threadgroup half2 *tl, ushort lane,
                        Fold fold) {
  constexpr auto descriptor = matmul2d_descriptor(kMatrixFragmentRows, GGUF_COLUMN_COLUMNS, kMatrixStep, false, true,
                                                  false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroup> op;
  // A step's 16 inputs of a fragment's 16 rows, a tensor of static extents at its offset (no bounds to check).
  device bfloat *rows = const_cast<device bfloat *>(x);
  const array<int, 2> strides{1, int(input_size)};
  const auto step = [&](uint input, ushort fi) __attribute__((always_inline)) {
    return tensor(rows + input + ulong(fi) * kMatrixFragmentRows * input_size,
                  extents<int, kMatrixStep, kMatrixFragmentRows>(), strides);
  };
  auto first = op.template get_right_input_cooperative_tensor<bfloat, half, float>();
  auto second = op.template get_right_input_cooperative_tensor<bfloat, half, float>();
  auto slice = step(0, 0);
  // A cooperative tensor is no array element: the fragments' sums are four tensors, those past Fragments unused.
  auto s0 = op.template get_destination_cooperative_tensor<decltype(slice), decltype(first), float>();
  auto s1 = op.template get_destination_cooperative_tensor<decltype(slice), decltype(first), float>();
  auto s2 = op.template get_destination_cooperative_tensor<decltype(slice), decltype(first), float>();
  auto s3 = op.template get_destination_cooperative_tensor<decltype(slice), decltype(first), float>();
  const auto each = [&](auto fn) __attribute__((always_inline)) {
    fn(ushort(0), s0);
    if constexpr (Fragments > 1) fn(ushort(1), s1);
    if constexpr (Fragments > 2) fn(ushort(2), s2);
    if constexpr (Fragments > 3) fn(ushort(3), s3);
  };
  each([&](ushort, thread auto &sums) __attribute__((always_inline)) {
#pragma unroll
    for (ushort e = 0; e < sums.get_capacity(); ++e) sums[e] = 0.0f;
  });
  const ushort2 coord = matrix_coord(lane);
  const ushort chunk = coord.x / 4;
  const uint groups = input_size / 32, units = groups / F::MetaGroups;
  // The lane's columns y + 8 j of the 32 from column0 (a multiple of 32) lie in one plane tile of QUANT_TILE_ROWS
  // ([N / 64][G][64][P]): one address a plane, column j's 8 j plane rows on.
  const uint plane_tile = column0 / QUANT_TILE_ROWS, plane_row = column0 % QUANT_TILE_ROWS + coord.y;
  device uchar *c0 = w0 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P0;
  device uchar *c1 = w1 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P1;
  device uchar *cm = meta + (ulong(plane_tile) * units * QUANT_TILE_ROWS + plane_row) * F::MetaBytes;
  for (uint g = g_begin; g < g_end; g += stride) {
    device uchar *g0 = c0 + ulong(g) * QUANT_TILE_ROWS * F::P0, *g1 = c1 + ulong(g) * QUANT_TILE_ROWS * F::P1,
                 *gm = cm + ulong(g / F::MetaGroups) * QUANT_TILE_ROWS * F::MetaBytes;
#pragma unroll
    for (ushort j = 0; j < 4; ++j) {
      const typename F::Chunk q = F::loadChunk(g0 + 8 * j * F::P0, g1 + 8 * j * F::P1, chunk);
      const typename F::Meta m = F::loadMeta(gm + 8 * j * F::MetaBytes);
      QuantCoef k;
      if constexpr (F::ScaleInChunk) k = F::coef(m, q); else k = F::coef(m, ushort(g % F::MetaGroups));
      half4 lo, hi;
      dequant_chunk<F>(q, k, tl, lo, hi);
#pragma unroll
      for (ushort i = 0; i < 4; ++i) {
        first[4 * j + i] = lo[i];
        second[4 * j + i] = hi[i];
      }
    }
    each([&](ushort fi, thread auto &sums) __attribute__((always_inline)) {
      auto a = step(g * 32, fi);
      op.run(a, first, sums);
      auto b = step(g * 32 + kMatrixStep, fi);
      op.run(b, second, sums);
    });
  }
  each([&](ushort fi, thread auto &sums) __attribute__((always_inline)) {
    fold(fi, [&](auto write) __attribute__((always_inline)) {
#pragma unroll
      for (ushort e = 0; e < 16; ++e)
        write(uint(coord.y + 8 * ((e >> 2) & 1)), uint(coord.x + 16 * (e >> 3) + (e & 3)), float(sums[e]));
    });
  });
}

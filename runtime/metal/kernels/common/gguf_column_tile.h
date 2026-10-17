#pragma once
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/gguf_staged.h"

#include <metal_stdlib>
using namespace metal;

// The column tile of kernels/shared/moe_gguf.metal: a GEMV of few rows in registers. Lane l of a simdgroup owns
// output column `column`, decodes its column's groups of 32 into registers (dequant32_chunks) and multiplies each
// weight into the fp32 sums of its rows: no stage, no barrier, Rows FMAs per weight. The staged tile runs a
// dequantize, barrier and matmul chain per 32-input step on padded 8-row MPP tiles, which bounds it at few rows, not
// DRAM, so the column tile runs the MoE expert tiles of one or two live rows (moe_column_tile). Includers set
// `#pragma clang fp reassociate(off)` first.

// Each row's sum over groups [g_begin, g_end) of K, of its first `rows` rows (at most Rows): per group one fp32 chain
// of its 32 products in element order, then added to the row's sum. Every row sums alone, so a row's result does not
// depend on the other rows, on its place among them or on Rows. The chain is written as multiplies and adds, whose
// order `reassociate(off)` keeps: Metal's fma() is emitted with every fast-math flag, reassociation among them, which
// the pragma does not clear. Each product of a half weight and a bf16 input is exact in fp32, so fused or not, every
// step rounds once (and as fast as the fma() chain). Each thread loads the next group's payload while the group's FMAs
// run, and its meta unit when the next group enters a new one, and reads the activations 16 bytes at a time (their
// rows of K, a multiple of 64, start 16-byte aligned in page-aligned buffers).
template <class F, ushort Rows>
inline void column_dot(device const bfloat *x, uint rows, uint input_size, device uchar *w0, device uchar *w1,
                       device uchar *meta, uint column, uint g_begin, uint g_end, threadgroup half2 *tl,
                       thread float (&sums)[Rows]) {
  const uint groups = input_size / 32, units = groups / F::MetaGroups;
  const uint plane_tile = column / QUANT_TILE_ROWS, plane_row = column % QUANT_TILE_ROWS;
  device uchar *tw0 = w0 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P0;
  device uchar *tw1 = w1 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P1;
  device uchar *tmeta = meta + (ulong(plane_tile) * units * QUANT_TILE_ROWS + plane_row) * F::MetaBytes;
#pragma unroll
  for (ushort r = 0; r < Rows; ++r) sums[r] = 0.0f;
  typename F::Payload w =
      F::load(tw0 + ulong(g_begin) * QUANT_TILE_ROWS * F::P0, tw1 + ulong(g_begin) * QUANT_TILE_ROWS * F::P1);
  typename F::Meta m = F::loadMeta(tmeta + ulong(g_begin / F::MetaGroups) * QUANT_TILE_ROWS * F::MetaBytes);
  for (uint g = g_begin; g < g_end; ++g) {
    half4 v[8];   // v[i]: elements 4i..4i+3 of the group
    dequant32_chunks<F>(w, m, ushort(g % F::MetaGroups), tl,
                        [&](ushort c, half4 lo, half4 hi) __attribute__((always_inline)) {
                          v[c] = lo;
                          v[4 + c] = hi;
                        });
    if (g + 1 < g_end) {
      w = F::load(tw0 + ulong(g + 1) * QUANT_TILE_ROWS * F::P0, tw1 + ulong(g + 1) * QUANT_TILE_ROWS * F::P1);
      if ((g + 1) / F::MetaGroups != g / F::MetaGroups)
        m = F::loadMeta(tmeta + ulong((g + 1) / F::MetaGroups) * QUANT_TILE_ROWS * F::MetaBytes);
    }
#pragma unroll
    for (ushort r = 0; r < Rows; ++r) {
      if (r >= rows) break;
      device const uint4 *xr = reinterpret_cast<device const uint4 *>(x + ulong(r) * input_size + g * 32);
      float sum = 0.0f;
#pragma unroll
      for (ushort i = 0; i < 8; ++i) {
        const uint4 q = xr[i / 2];
        const float4 xv = float4(as_type<bfloat4>(i % 2 ? q.zw : q.xy)), wv = float4(v[i]);
        sum = sum + wv.x * xv.x;
        sum = sum + wv.y * xv.y;
        sum = sum + wv.z * xv.z;
        sum = sum + wv.w * xv.w;
      }
      sums[r] += sum;
    }
  }
}

// column_dot in the run-time format `format` (uniform per threadgroup), after its pair table, which every thread of
// the threadgroup fills.
template <ushort Rows>
inline void column_dot_any(uint format, device const bfloat *x, uint rows, uint input_size, device uchar *w0,
                           device uchar *w1, device uchar *meta, uint column, uint g_begin, uint g_end,
                           threadgroup half2 *tl, uint thread_index, uint threads, thread float (&sums)[Rows]) {
  quant_format_switch(format, [&](auto f) {
    typedef decltype(f) F;
    quant_pair_table<F>(tl, thread_index, threads);
    column_dot<F, Rows>(x, rows, input_size, w0, w1, meta, column, g_begin, g_end, tl, sums);
  });
}

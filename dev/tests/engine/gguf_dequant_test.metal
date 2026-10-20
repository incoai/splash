// Test-only entry points expose the production dequantizer before matmul/output rounding.
#pragma clang fp reassociate(off)
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/gguf_live_rows.h"
#include "metal/kernels/common/gguf_staged.h"

#define DEQUANT_TEST(F, name) \
kernel void gguf_test_dequant_##name(device uchar *w0 [[buffer(0)]], \
    device uchar *w1 [[buffer(1)]], device uchar *meta [[buffer(2)]], \
    device half *output [[buffer(3)]], constant GgufDecodeParams &p [[buffer(4)]], \
    uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) { \
  threadgroup half stage[32 * 32]; \
  threadgroup half2 lut[256]; \
  quant_pair_table<F>(lut, lane, 32); \
  const uint groups = p.input_size / 32, row = tid / groups, g = tid % groups; \
  const ulong payload = quant_tile_index(row, g, groups); \
  const ulong header = quant_tile_index(row, g / F::MetaGroups, groups / F::MetaGroups); \
  dequant32<F>(F::load(w0 + payload * F::P0, w1 + payload * F::P1), \
               F::loadMeta(meta + header * F::MetaBytes), g % F::MetaGroups, lut, stage + lane * 32); \
  threadgroup_barrier(mem_flags::mem_threadgroup); \
  for (uint i = 0; i < 32; ++i) output[ulong(tid) * 32 + i] = stage[lane * 32 + i]; \
}

QUANT_FORMATS(DEQUANT_TEST)
#undef DEQUANT_TEST

// The register form (kernels/common/gguf_live_rows.h: dequant32_values, and the chunks of a grid format read from a
// threadgroup copy of its grid), written in the stage's order.
template <class F> inline void live_rows_test_grid(threadgroup uint *grid, uint lane) {
  if constexpr (LiveRowsGrid<F>::Words) LiveRowsGrid<F>::fill(grid, lane, 32);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
#define LIVE_ROWS_TEST(F, name) \
kernel void gguf_test_live_rows_##name(device uchar *w0 [[buffer(0)]], \
    device uchar *w1 [[buffer(1)]], device uchar *meta [[buffer(2)]], \
    device half *output [[buffer(3)]], device half *tg_output [[buffer(4)]], constant GgufDecodeParams &p [[buffer(5)]], \
    uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) { \
  threadgroup half2 lut[256]; \
  threadgroup uint grid[LiveRowsGrid<F>::Words ? LiveRowsGrid<F>::Words : 1]; \
  quant_pair_table<F>(lut, lane, 32); \
  live_rows_test_grid<F>(grid, lane); \
  const uint groups = p.input_size / 32, row = tid / groups, g = tid % groups; \
  const ulong payload = quant_tile_index(row, g, groups); \
  const ulong header = quant_tile_index(row, g / F::MetaGroups, groups / F::MetaGroups); \
  const F::Payload w = F::load(w0 + payload * F::P0, w1 + payload * F::P1); \
  const F::Meta m = F::loadMeta(meta + header * F::MetaBytes); \
  half4 v[8]; \
  dequant32_values<F>(w, m, g % F::MetaGroups, lut, v); \
  const QuantCoef k = live_rows_coef<F>(w, m, g % F::MetaGroups); \
  for (ushort c = 0; c < 4; ++c) { \
    half4 lo, hi; \
    live_rows_chunk<F, LiveRowsGrid<F>::Words != 0>(F::chunk(w, c), k, lut, grid, lo, hi); \
    for (ushort i = 0; i < 4; ++i) { \
      output[ulong(tid) * 32 + 4 * c + i] = v[c][i]; \
      output[ulong(tid) * 32 + 16 + 4 * c + i] = v[4 + c][i]; \
      tg_output[ulong(tid) * 32 + 4 * c + i] = lo[i]; \
      tg_output[ulong(tid) * 32 + 16 + 4 * c + i] = hi[i]; \
    } \
  } \
}

QUANT_FORMATS(LIVE_ROWS_TEST)
#undef LIVE_ROWS_TEST

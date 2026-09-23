// Test-only entry points expose the production dequantizer before matmul/output rounding.
#include "metal/kernels/shared/gguf_linear.metal"

#define DEQUANT_TEST(F, name) \
kernel void gguf_test_dequant_##name(device uchar *w0 [[buffer(0)]], \
    device uchar *w1 [[buffer(1)]], device uchar *meta [[buffer(2)]], \
    device half *output [[buffer(3)]], constant GgufParams &p [[buffer(4)]], \
    uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) { \
  threadgroup half stage[32 * 32]; \
  threadgroup half2 lut[256]; \
  gguf_init_lut(lut, lane, 32); \
  const uint groups = p.input_size / 32, row = tid / groups, g = tid % groups; \
  const ulong payload = quant_tile_index(row, g, groups); \
  const ulong header = quant_tile_index(row, g / F::MetaGroups, groups / F::MetaGroups); \
  dequant32<F>(F::load(w0 + payload * F::P0, w1 + payload * F::P1), \
               F::loadMeta(meta + header * F::MetaBytes), g % F::MetaGroups, lut, stage + lane * 32); \
  threadgroup_barrier(mem_flags::mem_threadgroup); \
  for (uint i = 0; i < 32; ++i) output[ulong(tid) * 32 + i] = stage[lane * 32 + i]; \
}

DEQUANT_TEST(FmtQ4K, q4k)
DEQUANT_TEST(FmtIQ4XS, iq4xs)
DEQUANT_TEST(FmtIQ4NL, iq4nl)
DEQUANT_TEST(FmtQ5K, q5k)
DEQUANT_TEST(FmtQ6K, q6k)
DEQUANT_TEST(FmtQ3K, q3k)
DEQUANT_TEST(FmtQ80, q80)
DEQUANT_TEST(FmtIQ3S, iq3s)

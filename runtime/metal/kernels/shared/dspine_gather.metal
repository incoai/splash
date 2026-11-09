#pragma clang fp reassociate(off)
#include "metal/abi/DSpine.h"
#include "metal/kernels/common/gguf_staged.h"

// Gather selected vocabulary rows from the same planes the target head uses.
#define DSPINE_GATHER(F, name) \
kernel void dspine_gather_##name(device const uint *ids [[buffer(0)]], \
    device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]], device uchar *meta [[buffer(3)]], \
    device bfloat *output [[buffer(4)]], constant DSpineGatherParams &p [[buffer(5)]], \
    uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) { \
  threadgroup half stage[32 * 32]; \
  threadgroup half2 lut[256]; \
  quant_pair_table<F>(lut, lane, 32); \
  const uint groups = p.width / 32, at = tid / groups, g = tid % groups; \
  const uint row = at < p.rows ? ids[at] : 0; \
  const ulong payload = quant_tile_index(row, g, groups); \
  const ulong header = quant_tile_index(row, g / F::MetaGroups, groups / F::MetaGroups); \
  dequant32<F>(F::load(w0 + payload * F::P0, w1 + payload * F::P1), \
               F::loadMeta(meta + header * F::MetaBytes), g % F::MetaGroups, lut, stage + lane * 32); \
  threadgroup_barrier(mem_flags::mem_threadgroup); \
  if (at < p.rows) for (uint i = 0; i < 32; ++i) output[ulong(tid) * 32 + i] = bfloat(stage[lane * 32 + i]); \
}
QUANT_FORMATS(DSPINE_GATHER)
#undef DSPINE_GATHER

kernel void dspine_gather_f32(device const uint *ids [[buffer(0)]], device const float *table [[buffer(1)]],
    device bfloat *output [[buffer(2)]], constant DSpineGatherParams &p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i < p.rows * p.width) output[i] = bfloat(table[ulong(ids[i / p.width]) * p.width + i % p.width]);
}

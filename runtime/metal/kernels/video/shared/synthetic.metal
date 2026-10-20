// Synthetic weights (h3 bench --synthetic): metal/abi/SyntheticABI.h.
#include <metal_stdlib>
#include "metal/abi/video/SyntheticABI.h"

using namespace metal;

kernel void synthetic_fill(device uchar *dst [[buffer(0)]], constant SyntheticFill &p [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
  if (i >= p.count) return;
  const ulong seed = (ulong(p.seed_hi) << 32) | p.seed_lo;
  const uint32_t h = h3_syn_hash(seed, i);
  if (p.type == H3_SYN_I8) {
    ((device char *)dst)[i] = char(int(h % 255u) - 127);
    return;
  }
  const float r = float(int32_t(h) >> 8) * 0x1p-23f;
  const float v = fma(p.scale, r, p.offset);
  if (p.type == H3_SYN_F32) ((device float *)dst)[i] = v;
  else if (p.type == H3_SYN_F16) ((device half *)dst)[i] = half(v);
  else ((device bfloat *)dst)[i] = bfloat(v);
}

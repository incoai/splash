// TAEH3 preview decoder input and memory-block state copy.
#include <metal_stdlib>
#include "metal/abi/video/PreviewABI.h"

using namespace metal;

// fp32 latents [n][H][W][24] (normalized diffusion latents) -> tanh(x / 3) * 3 -> fp16 padded frames, 32 channels.
kernel void preview_input(device const float *lat [[buffer(0)]], device half *X [[buffer(1)]],
                          constant PreviewConvParams &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
  const int x = gid.x, y = gid.y, f = gid.z;
  if (x >= p.W || y >= p.H) return;
  device const float *s = lat + ((long(f) * p.H + y) * p.W + x) * 24;
  device half *d = X + (long(f) * p.FSout + (y + 1) * p.Wp + x + 1) * 32;
  for (int c = 0; c < 24; c += 4) {
    float4 v = *(device const float4 *)(s + c);
    *(device half4 *)(d + c) = half4(tanh(v / 3.0f) * 3.0f);
  }
  for (int c = 24; c < 32; c += 4) *(device half4 *)(d + c) = half4(0.0h);
}

// MemBlock state: copy n uint4 from src to dst (last frame of the chunk -> memory slot).
kernel void preview_copy(device const uint4 *src [[buffer(0)]], device uint4 *dst [[buffer(1)]],
                         constant uint &n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) dst[gid] = src[gid];
}

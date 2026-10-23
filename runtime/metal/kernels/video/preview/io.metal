// TAEH3 preview decoder input, memory-block state copy and yuv420p output.
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

// uint8 RGB frames -> planar yuv420p (BT.601 limited range, 2x2 box chroma), piped straight into the encoder.
// One thread per 2x2 pixel block.
kernel void preview_rgb_to_yuv420(device const uchar *rgb [[buffer(0)]], device uchar *yuv [[buffer(1)]],
                                  constant PreviewYuvParams &d [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
  const int bx = gid.x, by = gid.y, f = gid.z, H = d.H, W = d.W;
  if (2 * bx >= W || 2 * by >= H) return;
  device const uchar *src = rgb + long(f) * H * W * 3;
  device uchar *Y = yuv + long(f) * H * W * 3 / 2;
  device uchar *U = Y + long(H) * W;
  device uchar *V = U + long(H / 2) * (W / 2);
  float3 acc = 0.0f;
  for (int j = 0; j < 2; ++j)
    for (int i = 0; i < 2; ++i) {
      const long p = long(2 * by + j) * W + 2 * bx + i;
      float3 c = float3(src[p * 3], src[p * 3 + 1], src[p * 3 + 2]);
      acc += c;
      Y[p] = uchar(clamp(16.0f + dot(c, float3(0.256788f, 0.504129f, 0.097906f)) + 0.5f, 0.0f, 255.0f));
    }
  acc *= 0.25f;
  const long q = long(by) * (W / 2) + bx;
  U[q] = uchar(clamp(128.0f + dot(acc, float3(-0.148223f, -0.290993f, 0.439216f)) + 0.5f, 0.0f, 255.0f));
  V[q] = uchar(clamp(128.0f + dot(acc, float3(0.439216f, -0.367788f, -0.071427f)) + 0.5f, 0.0f, 255.0f));
}

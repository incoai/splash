#pragma once

// Per-token int8 quantizer, Hadamard(128) transforms and activation functions shared by the W8A8 producers.
//
// Hadamard convention (tools/convert_dit.py, tools/convert_vae.py): for every group g of 128 input channels
//   x'_g = (1/sqrt(128)) * H128 * (s_g (.) x_g),  H128[i][j] = (-1)^popcount(i & j),  s = +-1 (splitmix64 signs).
// Weights are stored rotated the same way, so x'.w' = x.w. Kernels compute the unnormalized u = H128 (s (.) x) and
// fold 1/sqrt(128) into the per-token scale (absmax quantization is invariant to it).
#include "metal/kernels/video/common/tensor.h"

namespace h3 {
constant constexpr float kQMax = 127.0f;
constant constexpr float kRs128 = 0.08838834764831845f;  // 1/sqrt(128)

// s = amax / 127, q = rint(clamp(x * (127 / amax), -127, 127))
inline float q8_inv(float amax) { return amax > 0.0f ? kQMax / amax : 0.0f; }
inline char q8_round(float x, float inv) { return char(rint(clamp(x * inv, -kQMax, kQMax))); }
inline uint q4pack(float4 v, float inv) { return as_type<uint>(char4(rint(clamp(v * inv, -kQMax, kQMax)))); }

inline float silu_mul(float g, float v) { return g * v / (1.0f + precise::exp(-g)); }
inline float sigm(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float4 silu4(float4 x) { return x / (1.0f + exp(-x)); }

// Unnormalized WHT of one 128-group held by a 16-lane half simdgroup: lane j (= lane & 15) holds elements
// 8j..8j+3 in a and 8j+4..8j+7 in b. Strides 1, 2, 4 in registers; strides 8..64 pair lanes j ^ 1, 2, 4, 8.
inline void wht128(thread float4 &a, thread float4 &b, ushort lane) {
  float4 x = a, y = b;
  x = float4(x.x + x.y, x.x - x.y, x.z + x.w, x.z - x.w);
  y = float4(y.x + y.y, y.x - y.y, y.z + y.w, y.z - y.w);
  x = float4(x.x + x.z, x.y + x.w, x.x - x.z, x.y - x.w);
  y = float4(y.x + y.z, y.y + y.w, y.x - y.z, y.y - y.w);
  a = x + y;
  b = x - y;
  for (ushort m = 1; m < 16; m <<= 1) {
    float4 pa = simd_shuffle_xor(a, m), pb = simd_shuffle_xor(b, m);
    bool hi = (lane & m) != 0;
    a = hi ? pa - a : a + pa;
    b = hi ? pb - b : b + pb;
  }
}

// Unnormalized WHT over 128 values held by a full simdgroup: lane holds 4 consecutive (4 lane .. 4 lane + 3).
inline float4 wht128_lane4(float4 x, ushort lane) {
  x = float4(x.x + x.y, x.x - x.y, x.z + x.w, x.z - x.w);
  x = float4(x.x + x.z, x.y + x.w, x.x - x.z, x.y - x.w);
  for (ushort m = 1; m < 32; m <<= 1) {
    float4 o = simd_shuffle_xor(x, m);
    x = (lane & m) ? o - x : x + o;
  }
  return x;
}
}  // namespace h3

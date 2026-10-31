#include "metal/abi/LatentRotation.h"

#include <metal_stdlib>

using namespace metal;

// Latent rotation (metal/abi/LatentRotation.h, DeepSeek-V4.1's), as
// dev/tools/deepseek41_reference.py computes it after model.py: rms_norm,
// rotate (the last rope_pairs pairs, interleaved) and fp8_round or fp4_round,
// in fp32 with the reference's separate roundings: no contraction into fma,
// and exact divisions where a scale is no power of two. One simdgroup per
// vector; each lane holds width / 32 consecutive values.

constant constexpr uint kMaxValues = LATENT_ROTATION_MAX_WIDTH / 32;
constant constexpr float kFp8Max = 448.0f;
constant constexpr float kFp4Max = 6.0f;

// 2^ceil(log2(x)) of x > 0, as the reference's ceil_pow2 takes it from frexp.
inline float latent_ceil_pow2(float x) {
  int exponent;
  const float mantissa = frexp(x, exponent);
  return ldexp(1.0f, mantissa == 0.5f ? exponent - 1 : exponent);
}

// x on the E4M3 grid (3 mantissa bits, normals from 2^-6, subnormals in steps
// of 2^-9), rounded to nearest even.
inline float latent_round_e4m3(float x) {
  int exponent;
  frexp(max(fabs(x), 0.015625f), exponent);
  const float step = ldexp(1.0f, exponent - 4);
  return copysign(rint(fabs(x) / step) * step, x);
}

// x on the E2M1 grid (0, 0.5, 1, 1.5, 2, 3, 4, 6), rounded to nearest even.
inline float latent_round_e2m1(float x) {
  const float magnitude = fabs(x);
  const float step = magnitude < 2.0f ? 0.5f : (magnitude < 4.0f ? 1.0f : 2.0f);
  return copysign(rint(magnitude / step) * step, x);
}

template <typename In>
inline void latent_rotate(device const In *input, device const float *norm, device const float *frequencies,
                          device const uint *positions, device bfloat *output, constant LatentRotationParams &p,
                          uint2 group, uint lane) {
#pragma clang fp contract(off)
  const uint row = group.x, vector = group.y;
  if (row >= p.rows || vector >= p.vectors) return;
  const uint n = p.width / 32;
  const uint first = lane * n;
  const ulong base = (ulong(row) * p.vectors + vector) * p.width + first;
  float v[kMaxValues];
  for (uint i = 0; i < n; ++i) v[i] = float(input[base + i]);
  if (p.normalize) {
    float squares = 0.0f;
    for (uint i = 0; i < n; ++i) squares += v[i] * v[i];
    const float inverse = precise::divide(1.0f, precise::sqrt(simd_sum(squares) / float(p.width) + p.norm_epsilon));
    for (uint i = 0; i < n; ++i) v[i] = v[i] * inverse * norm[first + i];
  }
  // The rotated pairs, interleaved: pair j holds values tail + 2j and tail + 2j + 1.
  const uint tail = p.width - 2 * p.rope_pairs;
  const float position = float(positions[row]);
  for (uint i = 0; i < n; i += 2) {
    if (first + i < tail) continue;
    const float angle = position * frequencies[(first + i - tail) / 2];
    const float cosine = precise::cos(angle), sine = p.inverse ? -precise::sin(angle) : precise::sin(angle);
    const float even = v[i], odd = v[i + 1];
    v[i] = even * cosine - odd * sine;
    v[i + 1] = even * sine + odd * cosine;
  }
  if (p.rounding != LATENT_ROUND_NONE) {
    const uint block = p.rounding == LATENT_ROUND_FP4_E4M3_SCALE ? 16 : 32;
    // A lane holds n / block whole blocks, or a block spans block / n lanes.
    const uint span = n >= block ? block : n;
    for (uint at = 0; at < n; at += span) {
      float amax = 0.0f;
      for (uint i = at; i < at + span; ++i) amax = max(amax, fabs(v[i]));
      for (uint lanes = 1; lanes * n < block; lanes <<= 1) amax = max(amax, simd_shuffle_xor(amax, ushort(lanes)));
      if (p.rounding == LATENT_ROUND_FP8) {
        const float scale = latent_ceil_pow2(precise::divide(max(amax, 1e-4f), kFp8Max));
        for (uint i = at; i < at + span; ++i)
          v[i] = latent_round_e4m3(clamp(v[i] / scale, -kFp8Max, kFp8Max)) * scale;
      } else {
        const bool e4m3 = p.rounding == LATENT_ROUND_FP4_E4M3_SCALE;
        const float floor = e4m3 ? kFp4Max * 0x1p-9f : kFp4Max * 0x1p-126f;
        const float ratio = precise::divide(max(amax, floor), kFp4Max);
        const float scale = e4m3 ? latent_round_e4m3(ratio) : latent_ceil_pow2(ratio);
        for (uint i = at; i < at + span; ++i)
          v[i] = latent_round_e2m1(clamp(precise::divide(v[i], scale), -kFp4Max, kFp4Max)) * scale;
      }
    }
  }
  for (uint i = 0; i < n; ++i) output[base + i] = bfloat(v[i]);
}

#define LATENT_ROTATION_ENTRY(Name, In)                                                                         \
  kernel void Name(device const In *input [[buffer(0)]], device const float *norm [[buffer(1)]],               \
                   device const float *frequencies [[buffer(2)]], device const uint *positions [[buffer(3)]], \
                   device bfloat *output [[buffer(4)]], constant LatentRotationParams &p [[buffer(5)]],       \
                   uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {    \
    latent_rotate<In>(input, norm, frequencies, positions, output, p, group, lane);                             \
  }
LATENT_ROTATION_ENTRY(latent_rotation_bf16, bfloat)
LATENT_ROTATION_ENTRY(latent_rotation_f32, float)
#undef LATENT_ROTATION_ENTRY

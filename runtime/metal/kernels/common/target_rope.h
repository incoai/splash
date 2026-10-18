#pragma once

#include <metal_stdlib>

using namespace metal;

// Pair `pair` of the target's RoPE row at the (t, h, w) position `position`:
// its cosine and sine, Qwen3.5's interleaved M-RoPE taking frequency `pair` on
// axis pair % 3, both scaled by `scale`, YaRN's attention factor (1 without
// YaRN). Every kernel that rotates by the target's RoPE computes its rows
// with it, so they all rotate alike (ops::RoPE).
inline float2 target_rope_pair(uint3 position, device const float *inverse_frequencies, uint pair,
                               float scale) {
  const float angle = float(position[pair % 3]) * inverse_frequencies[pair];
  return float2(cos(angle), sin(angle)) * scale;
}

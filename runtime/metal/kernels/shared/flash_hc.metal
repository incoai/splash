// RMS reduction traversal adapted from MLX v0.32.2 rms_norm.metal.
// Copyright (c) 2024 Apple Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "metal/abi/FlashHC.h"
#include <metal_stdlib>

using namespace metal;

// Explicit storage-type boundaries match the canonical BF16 array operations.
// In particular, neither the stream mean nor the residual update is a single
// FP32 expression followed by a final BF16 cast.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

inline bfloat flash_hc_sigmoid(bfloat source) {
  // HC calls standalone mx.sigmoid, whose installed GPU primitive uses precise
  // exp. Compiled nn.silu/SwiGLU has a distinct fast-exp contract. Both retain
  // BF16 exp/add/reciprocal/subtraction stages; CPU promotion is different.
  const bfloat exponential =
      bfloat(metal::precise::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f) + exponential;
  const bfloat tail = bfloat(1.0f) / denominator;
  return source < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}

inline bfloat flash_hc_injection_weight(bfloat raw, uint streams) {
  const bfloat divided = bfloat(float(raw) / float(streams));
  const bfloat gate = flash_hc_sigmoid(divided);
  return bfloat(2.0f * float(gate));
}

inline bfloat flash_hc_mix_element(device const bfloat *normalized,
                                 device const bfloat *raw_up,
                                 FlashHCParams params, uint row,
                                 uint column) {
  bfloat total = bfloat(0.0f);
  for (uint stream = 0; stream < params.streams; ++stream) {
    const ulong index =
        (ulong(row) * params.streams + stream) * params.width + column;
    const bfloat gate = flash_hc_sigmoid(raw_up[index]);
    const bfloat product = bfloat(float(gate) * float(normalized[index]));
    total = bfloat(float(product) + float(total));
  }
  return bfloat(float(total) / float(params.streams));
}

kernel void flash_hc_expand(
    device const bfloat *input [[buffer(0)]],
    device bfloat *expanded [[buffer(1)]],
    constant FlashHCParams &params [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint2 group_size [[threads_per_threadgroup]]) {
  const uint group_width = group_size.x;
  const uint column = group.x * group_width + thread_index;
  if (group.y >= params.rows || column >= params.width)
    return;
  const bfloat value = input[ulong(group.y) * params.width + column];
  for (uint stream = 0; stream < params.streams; ++stream) {
    expanded[(ulong(group.y) * params.streams + stream) * params.width +
             column] = value;
  }
}

template <class Weight>
inline void flash_hc_grouped_norm(
    device const bfloat *input, device const Weight *weight,
    device bfloat *output, FlashHCParams params, uint2 group,
    uint thread_index, uint group_width, uint lane, uint simd_group,
    threadgroup float *partials) {
  const ulong base =
      (ulong(group.x) * params.streams + group.y) * params.width;
  // Match MLX's four-contiguous-read RMS traversal and two SIMD reductions.
  float square_sum = 0.0f;
  for (ulong origin = ulong(thread_index) * 4; origin < params.width;
       origin += ulong(group_width) * 4) {
    for (uint element = 0; element < 4; ++element) {
      const ulong column = origin + element;
      const float value = column < params.width ? float(input[base + column])
                                                : 0.0f;
      square_sum += value * value;
    }
  }
  square_sum = simd_sum(square_sum);
  if (simd_group == 0)
    partials[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0)
    partials[simd_group] = square_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    const float total = simd_sum(partials[lane]);
    if (lane == 0) {
      partials[0] = metal::precise::rsqrt(total / float(params.width) +
                                        params.epsilon);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inverse = partials[0];
  const ulong weight_base = ulong(group.y) * params.width;
  for (ulong origin = ulong(thread_index) * 4; origin < params.width;
       origin += ulong(group_width) * 4) {
    for (uint element = 0; element < 4; ++element) {
      const ulong column = origin + element;
      if (column < params.width) {
        const float raw_weight = float(weight[weight_base + column]);
        const float scale = params.norm_convention == FlashHCOnePlusWeight
                                ? 1.0f + raw_weight
                                : raw_weight;
        const float normalized = float(input[base + column]) * inverse;
        output[base + column] = bfloat(normalized * scale);
      }
    }
  }
}

#define FLASH_HC_NORM_ENTRY(NAME, WEIGHT)                                      \
  kernel void NAME(                                                          \
      device const bfloat *input [[buffer(0)]],                              \
      device const WEIGHT *weight [[buffer(1)]],                             \
      device bfloat *output [[buffer(2)]],                                   \
      constant FlashHCParams &params [[buffer(3)]],                          \
      uint2 group [[threadgroup_position_in_grid]],                          \
      uint thread_index [[thread_index_in_threadgroup]],                     \
      uint2 group_size [[threads_per_threadgroup]],                          \
      uint lane [[thread_index_in_simdgroup]],                               \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                   \
    if (group.x >= params.rows || group.y >= params.streams)                  \
      return;                                                               \
    threadgroup float partials[32];                                         \
    flash_hc_grouped_norm(input, weight, output, params, group, thread_index, \
                          group_size.x, lane, simd_group, partials);         \
  }

FLASH_HC_NORM_ENTRY(flash_hc_norm_bf16_weight, bfloat)
FLASH_HC_NORM_ENTRY(flash_hc_norm_f32_weight, float)

#undef FLASH_HC_NORM_ENTRY

kernel void flash_hc_mix(
    device const bfloat *normalized [[buffer(0)]],
    device const bfloat *raw_up [[buffer(1)]],
    device bfloat *mixed [[buffer(2)]],
    constant FlashHCParams &params [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint2 group_size [[threads_per_threadgroup]]) {
  const uint group_width = group_size.x;
  const uint column = group.x * group_width + thread_index;
  if (group.y >= params.rows || column >= params.width)
    return;
  mixed[ulong(group.y) * params.width + column] =
      flash_hc_mix_element(normalized, raw_up, params, group.y, column);
}

// The four injection weights are produced once per row alongside the mix,
// before its branch executes. No atomics or duplicate writes are needed.
kernel void flash_hc_mix_with_injection(
    device const bfloat *normalized [[buffer(0)]],
    device const bfloat *raw_up [[buffer(1)]],
    device const bfloat *raw_injection [[buffer(2)]],
    device bfloat *mixed [[buffer(3)]],
    device bfloat *injection_weights [[buffer(4)]],
    constant FlashHCParams &params [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint2 group_size [[threads_per_threadgroup]]) {
  const uint group_width = group_size.x;
  if (group.y >= params.rows)
    return;
  if (group.x == 0) {
    for (uint stream = thread_index; stream < params.streams;
         stream += group_width) {
      const ulong index = ulong(group.y) * params.streams + stream;
      injection_weights[index] =
          flash_hc_injection_weight(raw_injection[index], params.streams);
    }
  }
  const uint column = group.x * group_width + thread_index;
  if (column < params.width) {
    mixed[ulong(group.y) * params.width + column] =
        flash_hc_mix_element(normalized, raw_up, params, group.y, column);
  }
}

kernel void flash_hc_inject(
    device const bfloat *hyper_input [[buffer(0)]],
    device const bfloat *branch [[buffer(1)]],
    device const bfloat *injection_weights [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant FlashHCParams &params [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint2 group_size [[threads_per_threadgroup]]) {
  const uint group_width = group_size.x;
  const uint column = group.x * group_width + thread_index;
  if (group.y >= params.rows || column >= params.width)
    return;
  const bfloat branch_value = branch[ulong(group.y) * params.width + column];
  for (uint stream = 0; stream < params.streams; ++stream) {
    const bfloat gate =
        injection_weights[ulong(group.y) * params.streams + stream];
    const bfloat product = bfloat(float(branch_value) * float(gate));
    const ulong index =
        (ulong(group.y) * params.streams + stream) * params.width + column;
    output[index] = bfloat(float(hyper_input[index]) + float(product));
  }
}

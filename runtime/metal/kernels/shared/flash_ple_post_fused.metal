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

#include "metal/abi/FlashPLEPostFused.h"
#include <metal_stdlib>

using namespace metal;

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

inline bool flash_ple_post_fused_geometry(FlashPLEPostParams p) {
  return p.lanes && p.rows && p.width && p.streams && p.streams <= 8 &&
         !(p.flags & ~1u) && p.state_rows == 9 && p.taps == 4 && p.dilation == 3;
}

inline float flash_ple_post_fused_weight(device const uchar *weight,
                                       ulong index, bool f32) {
  return f32 ? reinterpret_cast<device const float *>(weight)[index] :
               float(reinterpret_cast<device const bfloat *>(weight)[index]);
}

inline bfloat flash_ple_post_fused_normalize(
    bfloat input, device const uchar *weight, ulong index, bool f32,
    float inverse, uint convention) {
  const float raw_weight = flash_ple_post_fused_weight(weight, index, f32);
  const float scale = convention == FlashHCOnePlusWeight ?
      1.0f + raw_weight : raw_weight;
  const float normalized = float(input) * inverse;
  return bfloat(normalized * scale);
}

// The caller may launch extra gate threads. Only the source RMS partition
// contributes; inactive SIMD groups cannot change its two-level reduction.
inline float flash_ple_post_fused_inverse(
    device const bfloat *input, ulong base, uint width, uint norm_threads,
    float epsilon, uint thread_index, uint lane, uint simd_group,
    threadgroup float *partials) {
  float square_sum = 0.0f;
  if (thread_index < norm_threads) {
    for (ulong origin = ulong(thread_index) * 4; origin < width;
         origin += ulong(norm_threads) * 4) {
      for (uint element = 0; element < 4; ++element) {
        const ulong column = origin + element;
        const float value = column < width ? float(input[base + column]) : 0.0f;
        square_sum += value * value;
      }
    }
  }
  square_sum = simd_sum(square_sum);
  if (simd_group == 0) partials[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0 && simd_group < norm_threads / 32)
    partials[simd_group] = square_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    const float total = simd_sum(partials[lane]);
    if (lane == 0)
      partials[0] = metal::precise::rsqrt(total / float(width) + epsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return partials[0];
}

// Independent RMS sums share synchronization, never arithmetic. Interleaving
// the two sums preserves each input's exact local order and SIMD hierarchy.
inline float2 flash_ple_post_fused_inverse_pair(
    device const bfloat *keys, device const bfloat *queries, ulong base,
    uint width, uint norm_threads, float epsilon, uint thread_index,
    uint lane, uint simd_group, threadgroup float *key_partials,
    threadgroup float *query_partials) {
  float key_square_sum = 0.0f;
  float query_square_sum = 0.0f;
  if (thread_index < norm_threads) {
    for (ulong origin = ulong(thread_index) * 4; origin < width;
         origin += ulong(norm_threads) * 4) {
      for (uint element = 0; element < 4; ++element) {
        const ulong column = origin + element;
        const float key = column < width ? float(keys[base + column]) : 0.0f;
        const float query = column < width ? float(queries[base + column]) : 0.0f;
        key_square_sum += key * key;
        query_square_sum += query * query;
      }
    }
  }
  key_square_sum = simd_sum(key_square_sum);
  query_square_sum = simd_sum(query_square_sum);
  if (simd_group == 0) {
    key_partials[lane] = 0.0f;
    query_partials[lane] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0 && simd_group < norm_threads / 32) {
    key_partials[simd_group] = key_square_sum;
    query_partials[simd_group] = query_square_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    const float key_total = simd_sum(key_partials[lane]);
    const float query_total = simd_sum(query_partials[lane]);
    if (lane == 0) {
      key_partials[0] = metal::precise::rsqrt(key_total / float(width) + epsilon);
      query_partials[0] = metal::precise::rsqrt(query_total / float(width) + epsilon);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return float2(key_partials[0], query_partials[0]);
}

inline bfloat flash_ple_post_fused_gate_sigmoid(bfloat input) {
  const bfloat exponential =
      bfloat(metal::precise::exp(metal::abs(float(input))));
  const bfloat denominator = bfloat(1.0f) + exponential;
  const bfloat tail = bfloat(1.0f) / denominator;
  return input < bfloat(0.0f) ? tail : bfloat(1.0f) - tail;
}

inline bfloat flash_ple_post_fused_silu_sigmoid(bfloat input) {
  const bfloat exponential = bfloat(metal::exp(metal::abs(float(input))));
  auto tail = 1 / (1 + exponential);
  return input < 0 ? tail : 1 - tail;
}

// One group owns exactly one token and hyper stream. All cross-thread writes
// to gated remain BF16 and are complete before this same group normalizes
// them. Temporal convolution runs in a later dispatch because it reads other
// rows; its normalized BF16 array also serves speculative-prefix restoration.
kernel void flash_ple_post_norm_gate_fused(
    device const bfloat *keys [[buffer(0)]],
    device const bfloat *queries [[buffer(1)]],
    device const bfloat *values [[buffer(2)]],
    device const uchar *norm_key [[buffer(3)]],
    device const uchar *norm_query [[buffer(4)]],
    device const uchar *norm_convolution [[buffer(5)]],
    device const uint *mask [[buffer(6)]],
    device bfloat *normalized_keys [[buffer(7)]],
    device bfloat *normalized_queries [[buffer(8)]],
    device bfloat *gated [[buffer(9)]],
    device bfloat *normalized_convolution [[buffer(10)]],
    device atomic_uint *diagnostics [[buffer(11)]],
    constant FlashPLEPostFusedParams &params [[buffer(12)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint2 group_size [[threads_per_threadgroup]]) {
  const FlashPLEPostParams p = params.geometry;
  if (!flash_ple_post_fused_geometry(p) ||
      !metal::isfinite(params.epsilon) || params.epsilon <= 0 ||
      params.norm_convention > FlashHCDirectGamma ||
      (params.norm_weight_flags & ~7u) ||
      !params.norm_threads || params.norm_threads > 1024 ||
      params.norm_threads % 32 || !params.gate_threads ||
      params.gate_threads > 1024 || params.gate_threads % 32 ||
      group_size.x != max(params.norm_threads, params.gate_threads) ||
      params.reserved0 || params.reserved1 || params.reserved2) {
    if (thread_index == 0)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  if (ulong(group.x) >= ulong(p.lanes) * p.rows || group.y >= p.streams) return;
  const ulong base = (ulong(group.x) * p.streams + group.y) * p.width;
  const ulong weight_base = ulong(group.y) * p.width;
  threadgroup float rms_partials[32];
  threadgroup float query_rms_partials[32];
  threadgroup bfloat gate_partials[32];
  threadgroup bfloat gate;
  const float2 inverses = flash_ple_post_fused_inverse_pair(
      keys, queries, base, p.width, params.norm_threads, params.epsilon,
      thread_index, lane, simd_group, rms_partials, query_rms_partials);
  const float inverse_key = inverses.x;
  const float inverse_query = inverses.y;
  const bool key_f32 = params.norm_weight_flags & FlashPLEPostFusedKeyF32;
  const bool query_f32 = params.norm_weight_flags & FlashPLEPostFusedQueryF32;
  bfloat reduced = bfloat(0.0f);
  if (p.width <= 64) {
    if (thread_index == 0) {
      for (uint channel = 0; channel < p.width; ++channel) {
        const bfloat key = flash_ple_post_fused_normalize(
            keys[base + channel], norm_key, weight_base + channel, key_f32,
            inverse_key, params.norm_convention);
        const bfloat query = flash_ple_post_fused_normalize(
            queries[base + channel], norm_query, weight_base + channel, query_f32,
            inverse_query, params.norm_convention);
        normalized_keys[base + channel] = key;
        normalized_queries[base + channel] = query;
        const bfloat product = bfloat(float(key) * float(query));
        reduced = bfloat(float(product) + float(reduced));
      }
    }
  } else {
    bfloat total = bfloat(0.0f);
    if (thread_index < params.gate_threads) {
      for (ulong block = 0; block < p.width;
           block += ulong(params.gate_threads) * 4) {
        for (uint read = 0; read < 4; ++read) {
          const ulong channel = block + ulong(thread_index) * 4 + read;
          if (channel >= p.width) break;
          const bfloat key = flash_ple_post_fused_normalize(
              keys[base + channel], norm_key, weight_base + channel, key_f32,
              inverse_key, params.norm_convention);
          const bfloat query = flash_ple_post_fused_normalize(
              queries[base + channel], norm_query, weight_base + channel, query_f32,
              inverse_query, params.norm_convention);
          normalized_keys[base + channel] = key;
          normalized_queries[base + channel] = query;
          const bfloat product = bfloat(float(key) * float(query));
          total = bfloat(float(product) + float(total));
        }
      }
    }
    const bfloat simd_total = bfloat(simd_sum(float(total)));
    if (lane == 0 && simd_group < params.gate_threads / 32)
      gate_partials[simd_group] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint simd_count = params.gate_threads / 32;
    const bfloat partial = thread_index < simd_count ?
        gate_partials[thread_index] : bfloat(0.0f);
    reduced = bfloat(simd_sum(float(partial)));
  }
  if (thread_index == 0) {
    const bfloat divisor = bfloat(metal::sqrt(float(p.width)));
    const bfloat divided = bfloat(float(reduced) / float(divisor));
    const bfloat magnitude = bfloat(metal::max(metal::abs(float(divided)),
                                               float(bfloat(1e-6f))));
    const bfloat root = bfloat(metal::sqrt(float(magnitude)));
    const float sign = float(divided) < 0 ? -1.0f :
                       float(divided) > 0 ? 1.0f : 0.0f;
    gate = flash_ple_post_fused_gate_sigmoid(bfloat(sign * float(root)));
    if (!metal::isfinite(float(reduced)) ||
        !metal::isfinite(float(divided)) || !metal::isfinite(float(root)) ||
        !metal::isfinite(float(gate)))
      atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool enabled = !(p.flags & 1u) || mask[group.x] != 0;
  for (uint channel = thread_index; channel < p.width; channel += group_size.x) {
    const bfloat value = bfloat(float(gate) *
        float(values[ulong(group.x) * p.width + channel]));
    if (!metal::isfinite(float(value)))
      atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
    gated[base + channel] = enabled ? value : bfloat(0.0f);
  }
  threadgroup_barrier(mem_flags::mem_device);
  const float inverse_convolution = flash_ple_post_fused_inverse(
      gated, base, p.width, params.norm_threads, params.epsilon,
      thread_index, lane, simd_group, rms_partials);
  const bool convolution_f32 =
      params.norm_weight_flags & FlashPLEPostFusedConvolutionF32;
  if (thread_index < params.norm_threads) {
    for (ulong origin = ulong(thread_index) * 4; origin < p.width;
         origin += ulong(params.norm_threads) * 4) {
      for (uint element = 0; element < 4; ++element) {
        const ulong channel = origin + element;
        if (channel < p.width)
          normalized_convolution[base + channel] = flash_ple_post_fused_normalize(
              gated[base + channel], norm_convolution, weight_base + channel,
              convolution_f32, inverse_convolution, params.norm_convention);
      }
    }
  }
}

// No thread mutates convolution history here. An exact hyper/output alias is
// safe because the only hyper read is this thread's own element, and query
// normalization has completed in the preceding command-graph dispatch.
kernel void flash_ple_post_convolution_inject_fused(
    device const bfloat *normalized [[buffer(0)]],
    device const bfloat *gated [[buffer(1)]],
    device const bfloat *state [[buffer(2)]],
    device const bfloat *weights [[buffer(3)]],
    device const bfloat *hyper [[buffer(4)]],
    device bfloat *ple_output [[buffer(5)]],
    device bfloat *injected_output [[buffer(6)]],
    device atomic_uint *diagnostics [[buffer(7)]],
    constant FlashPLEPostParams &p [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  if (!flash_ple_post_fused_geometry(p)) {
    if (thread_index == 0)
      atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint hyper_width = p.width * p.streams;
  const uint channel = group.x * 256 + thread_index;
  if (channel >= hyper_width || ulong(group.y) >= ulong(p.lanes) * p.rows) return;
  const uint lane = group.y / p.rows;
  const uint row = group.y % p.rows;
  float sum = 0.0f;
  for (uint tap = 0; tap < 4; ++tap) {
    const uint timeline = row + tap * 3;
    const bfloat value = timeline < 9 ?
        state[(ulong(lane) * 9 + timeline) * hyper_width + channel] :
        normalized[(ulong(lane) * p.rows + timeline - 9) * hyper_width + channel];
    sum += float(value) * float(weights[ulong(channel) * 4 + tap]);
  }
  const bfloat convolution = bfloat(sum);
  const bfloat sigmoid = flash_ple_post_fused_silu_sigmoid(convolution);
  const bfloat activated = bfloat(float(convolution) * float(sigmoid));
  const ulong index = ulong(group.y) * hyper_width + channel;
  const bfloat ple = bfloat(float(gated[index]) + float(activated));
  if (!metal::isfinite(float(ple)))
    atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
  ple_output[index] = ple;
  injected_output[index] = bfloat(float(hyper[index]) + float(ple));
}

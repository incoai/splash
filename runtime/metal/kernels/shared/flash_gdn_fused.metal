#include "metal/abi/FlashGDN.h"
#include "metal/abi/FlashGDNFused.h"
#include "metal/abi/FlashGDNSeparate.h"
#include <metal_stdlib>

using namespace metal;

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

constant uint GFKeyWidth = 2048;
constant uint GFValueWidth = 6144;
constant uint GFConvWidth = 10240;

inline void gf_flag(device atomic_uint &diagnostics, uint flag) {
  atomic_fetch_or_explicit(&diagnostics, flag, memory_order_relaxed);
}

inline bool gf_valid(FlashGDNParams p, device atomic_uint &diagnostics) {
  const bool valid = p.rows && p.rows <= 2048 && p.lanes && p.lanes <= 32 &&
      p.key_heads == 16 && p.value_heads == 48 && p.key_dimension == 128 &&
      p.value_dimension == 128 && p.convolution_taps == 4 &&
      isfinite(p.norm_epsilon) && p.norm_epsilon > 0.0f &&
      p.convolution_lane_stride_bytes >= ulong(3) * GFConvWidth * 2 &&
      p.convolution_lane_stride_bytes % 2 == 0 &&
      p.recurrent_lane_stride_bytes >= ulong(48) * 128 * 128 * 4 &&
      p.recurrent_lane_stride_bytes % 4 == 0;
  if (!valid) gf_flag(diagnostics, FlashGDNInvalidParameters);
  return valid;
}

inline void gf_check(float value, device atomic_uint &diagnostics) {
  if (!isfinite(value)) gf_flag(diagnostics, FlashGDNNonFinite);
}

inline bfloat gf_sigmoid_compiled(bfloat source) {
  const bfloat exponent = bfloat(metal::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f + float(exponent));
  const bfloat tail = bfloat(1.0f / float(denominator));
  return float(source) < 0.0f ? tail : bfloat(1.0f - float(tail));
}

inline bfloat gf_sigmoid_unary(bfloat source) {
  const bfloat exponent =
      bfloat(metal::precise::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f + float(exponent));
  const bfloat tail = bfloat(1.0f / float(denominator));
  return float(source) < 0.0f ? tail : bfloat(1.0f - float(tail));
}

inline bfloat gf_softplus(bfloat source) {
  const bfloat exponent = bfloat(metal::exp(-metal::abs(float(source))));
  const float xp1 = 1.0f + float(exponent);
  const bfloat logarithm = xp1 == 1.0f ? exponent :
      bfloat(float(exponent) * (metal::log(xp1) / (xp1 - 1.0f)));
  return bfloat(max(float(source), 0.0f) + float(logarithm));
}

inline float gf_sigmoid_z(float source) {
  const float tail =
      1.0f / (1.0f + metal::precise::exp(metal::abs(source)));
  return source < 0.0f ? tail : 1.0f - tail;
}

template <bool Separate = false>
inline bfloat gf_conv(device const bfloat *qkv,
                       device const bfloat *weights,
                       device const bfloat *history, FlashGDNParams p,
                       uint batch, uint token, uint channel) {
  device const bfloat *lane_history =
      history + (Separate ? 0 : ulong(batch) * p.convolution_lane_stride_bytes / 2);
  device const bfloat *lane_input = qkv + ulong(batch) * p.rows * GFConvWidth;
  float total = 0.0f;
  for (uint tap = 0; tap < 4; ++tap) {
    const uint position = token + tap;
    const bfloat source = position < 3
        ? lane_history[position * GFConvWidth + channel]
        : lane_input[ulong(position - 3) * GFConvWidth + channel];
    total += float(source) * float(weights[channel * 4 + tap]);
  }
  const bfloat convolution = bfloat(total);
  return bfloat(float(convolution) * float(gf_sigmoid_compiled(convolution)));
}

template <bool Separate = false>
inline void gf_qk(device const bfloat *qkv,
                   device const bfloat *weights,
                   device const bfloat *history, FlashGDNParams p,
                   uint batch, uint token, uint key_head, uint lane,
                   thread bfloat *query, thread bfloat *key) {
  float q[4], k[4];
  bfloat qp = bfloat(0.0f), kp = bfloat(0.0f);
  for (uint i = 0; i < 4; ++i) {
    q[i] = float(gf_conv<Separate>(qkv, weights, history, p, batch, token,
                         key_head * 128 + 4 * lane + i));
    k[i] = float(gf_conv<Separate>(qkv, weights, history, p, batch, token,
                         GFKeyWidth + key_head * 128 + 4 * lane + i));
    qp = bfloat(float(qp) + float(bfloat(q[i] * q[i])));
    kp = bfloat(float(kp) + float(bfloat(k[i] * k[i])));
  }
  const bfloat qs = bfloat(simd_sum(float(qp)));
  const bfloat ks = bfloat(simd_sum(float(kp)));
  const bfloat epsilon = bfloat(1e-6f);
  const bfloat qd = bfloat(float(qs) + float(epsilon));
  const bfloat kd = bfloat(float(ks) + float(epsilon));
  const float qi = float(bfloat(metal::precise::rsqrt(float(qd))));
  const float ki = float(bfloat(metal::precise::rsqrt(float(kd))));
  const bfloat scale = bfloat(0.08838834764831845f);
  for (uint i = 0; i < 4; ++i) {
    query[i] = bfloat(float(bfloat(q[i] * qi)) * float(scale));
    key[i] = bfloat(k[i] * ki);
  }
}

inline void gf_gates(device const bfloat *a, device const bfloat *b,
                      device const bfloat *a_log,
                      device const bfloat *time_bias, ulong index, uint head,
                      thread float &decay, thread bfloat &beta) {
  const bfloat sum = bfloat(float(a[index]) + float(time_bias[head]));
  decay = metal::exp(-metal::exp(float(a_log[head])) * float(gf_softplus(sum)));
  beta = gf_sigmoid_unary(b[index]);
}

// One SIMD prepares q/k; three SIMD groups prepare the three value heads
// sharing that key head. No shared memory or inter-SIMD synchronization.
kernel void flash_gdn_fused_prepare(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *a [[buffer(1)]],
    device const bfloat *b [[buffer(2)]],
    device const bfloat *weights [[buffer(3)]],
    device const bfloat *a_log [[buffer(4)]],
    device const bfloat *time_bias [[buffer(5)]],
    device const bfloat *history [[buffer(6)]],
    device bfloat *mixed [[buffer(7)]],
    device float *decay [[buffer(8)]],
    device bfloat *beta [[buffer(9)]],
    device atomic_uint &diagnostics [[buffer(10)]],
    constant FlashGDNParams &p [[buffer(11)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (!gf_valid(p, diagnostics) || group.x >= 16 || group.y >= p.rows ||
      group.z >= p.lanes) return;
  const ulong row = ulong(group.z) * p.rows + group.y;
  if (simd_group == 0) {
    bfloat query[4], key[4];
    gf_qk(qkv, weights, history, p, group.z, group.y, group.x, lane,
           query, key);
    for (uint i = 0; i < 4; ++i) {
      mixed[row * GFConvWidth + group.x * 128 + 4 * lane + i] = query[i];
      mixed[row * GFConvWidth + GFKeyWidth + group.x * 128 + 4 * lane + i] = key[i];
      gf_check(float(query[i]), diagnostics);
      gf_check(float(key[i]), diagnostics);
    }
  } else if (simd_group < 4) {
    const uint head = group.x * 3 + simd_group - 1;
    for (uint i = 0; i < 4; ++i) {
      const uint channel = 2 * GFKeyWidth + head * 128 + 4 * lane + i;
      const bfloat value = gf_conv(qkv, weights, history, p, group.z, group.y,
                                    channel);
      mixed[row * GFConvWidth + channel] = value;
      gf_check(float(value), diagnostics);
    }
    if (lane == 0) {
      float d; bfloat be;
      gf_gates(a, b, a_log, time_bias, row * 48 + head, head, d, be);
      decay[row * 48 + head] = d;
      beta[row * 48 + head] = be;
      gf_check(d, diagnostics);
      gf_check(float(be), diagnostics);
    }
  }
}

// A full value head owns its F32 state. The eight SIMD groups each advance
// sixteen independent value dimensions, preserving the qualified SIMD32 dot
// traversal. Shared q/k history is read-only until the later carry dispatch.
template <bool Capture, uint Simdgroups, bool Separate = false>
inline void gf_persistent(
    device const bfloat *qkv, device const bfloat *z,
    device const bfloat *a, device const bfloat *b,
    device const bfloat *weights, device const bfloat *a_log,
    device const bfloat *time_bias, device const bfloat *norm,
    device const bfloat *history, device float *recurrent,
    device bfloat *mixed, device float *decay, device bfloat *beta,
    device bfloat *recurrent_rows, device bfloat *output,
    device atomic_uint &diagnostics, FlashGDNParams p,
    device float *captured, ulong capture_row_stride,
    ulong capture_lane_stride, uint capture_rows, uint2 group, uint thread_index,
    uint lane, uint simd_group,
    threadgroup bfloat *query, threadgroup bfloat *key,
    threadgroup bfloat *value, threadgroup bfloat *rows,
    threadgroup float *gates) {
  if (!gf_valid(p, diagnostics) || group.x >= 48 || group.y >= p.lanes) return;
  const uint head = group.x, batch = group.y, key_head = head / 3;
  const bool shared_writer = head % 3 == 0;
  constexpr uint ValueRows = 128 / Simdgroups;
  float state[ValueRows][4];
  const ulong state_lane = Separate ? 0 : ulong(batch) * p.recurrent_lane_stride_bytes / 4;
  for (uint r = 0; r < ValueRows; ++r) {
    const uint dimension = r * Simdgroups + simd_group;
    const ulong base = state_lane + (head * 128 + dimension) * 128 + 4 * lane;
    for (uint i = 0; i < 4; ++i) state[r][i] = recurrent[base + i];
  }
  for (uint token = 0; token < p.rows; ++token) {
    const ulong row = ulong(batch) * p.rows + token;
    if (simd_group == 0) {
      bfloat q[4], k[4];
      gf_qk<Separate>(qkv, weights, history, p, batch, token, key_head, lane, q, k);
      for (uint i = 0; i < 4; ++i) {
        query[4 * lane + i] = q[i]; key[4 * lane + i] = k[i];
        if (shared_writer) {
          mixed[row * GFConvWidth + key_head * 128 + 4 * lane + i] = q[i];
          mixed[row * GFConvWidth + GFKeyWidth + key_head * 128 + 4 * lane + i] = k[i];
        }
        gf_check(float(q[i]), diagnostics); gf_check(float(k[i]), diagnostics);
      }
    }
    if (thread_index < 128) {
      const uint channel = 2 * GFKeyWidth + head * 128 + thread_index;
      value[thread_index] = gf_conv<Separate>(qkv, weights, history, p, batch, token,
                                     channel);
      mixed[row * GFConvWidth + channel] = value[thread_index];
      gf_check(float(value[thread_index]), diagnostics);
    }
    if (thread_index == 0) {
      float d; bfloat be;
      gf_gates(a, b, a_log, time_bias, row * 48 + head, head, d, be);
      gates[0] = d; gates[1] = float(be);
      decay[row * 48 + head] = d; beta[row * 48 + head] = be;
      gf_check(d, diagnostics); gf_check(float(be), diagnostics);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q[4], k[4];
    for (uint i = 0; i < 4; ++i) {
      q[i] = float(query[4 * lane + i]); k[i] = float(key[4 * lane + i]);
    }
    for (uint r = 0; r < ValueRows; ++r) {
      const uint dimension = r * Simdgroups + simd_group;
      float memory = 0.0f;
      for (uint i = 0; i < 4; ++i) {
        state[r][i] = state[r][i] * gates[0]; memory += state[r][i] * k[i];
      }
      memory = simd_sum(memory);
      const float delta = (float(value[dimension]) - memory) * gates[1];
      float result = 0.0f;
      for (uint i = 0; i < 4; ++i) {
        state[r][i] = state[r][i] + k[i] * delta; result += state[r][i] * q[i];
      }
      result = simd_sum(result);
      if (lane == 0) {
        rows[dimension] = bfloat(result);
        recurrent_rows[row * GFValueWidth + head * 128 + dimension] = rows[dimension];
        gf_check(result, diagnostics);
      }
      if (Capture && token < capture_rows) {
        const ulong capture_base =
            (ulong(batch) * capture_lane_stride + ulong(token) * capture_row_stride) / 4 +
            (head * 128 + dimension) * 128 + 4 * lane;
        for (uint i = 0; i < 4; ++i) captured[capture_base + i] = state[r][i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
      float sum = 0.0f;
      for (uint i = 0; i < 4; ++i) {
        const float item = float(rows[4 * lane + i]); sum += item * item;
      }
      sum = simd_sum(sum);
      if (lane == 0) gates[2] = metal::precise::rsqrt(sum / 128 + p.norm_epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < 128) {
      const bfloat normalized = bfloat(float(rows[thread_index]) * gates[2]);
      const bfloat weighted = bfloat(float(normalized) * float(norm[thread_index]));
      const ulong index = row * GFValueWidth + head * 128 + thread_index;
      const bfloat result = bfloat(float(weighted) * gf_sigmoid_z(float(z[index])));
      output[index] = result; gf_check(float(result), diagnostics);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint r = 0; r < ValueRows; ++r) {
    const uint dimension = r * Simdgroups + simd_group;
    const ulong base = state_lane + (head * 128 + dimension) * 128 + 4 * lane;
    for (uint i = 0; i < 4; ++i) {
      recurrent[base + i] = state[r][i]; gf_check(state[r][i], diagnostics);
    }
  }
}

#define GF_PERSISTENT_COMMON_BINDINGS                                       \
    device const bfloat *qkv [[buffer(0)]],                                 \
    device const bfloat *z [[buffer(1)]],                                   \
    device const bfloat *a [[buffer(2)]],                                   \
    device const bfloat *b [[buffer(3)]],                                   \
    device const bfloat *weights [[buffer(4)]],                             \
    device const bfloat *a_log [[buffer(5)]],                               \
    device const bfloat *time_bias [[buffer(6)]],                           \
    device const bfloat *norm [[buffer(7)]],                                \
    device const bfloat *history [[buffer(8)]],                             \
    device float *recurrent [[buffer(9)]],                                 \
    device bfloat *mixed [[buffer(10)]],                                    \
    device float *decay [[buffer(11)]],                                    \
    device bfloat *beta [[buffer(12)]],                                    \
    device bfloat *recurrent_rows [[buffer(13)]],                           \
    device bfloat *output [[buffer(14)]],                                  \
    device atomic_uint &diagnostics [[buffer(15)]]

#define GF_PERSISTENT_GRID_BINDINGS                                        \
    uint2 group [[threadgroup_position_in_grid]],                          \
    uint thread_index [[thread_index_in_threadgroup]],                     \
    uint lane [[thread_index_in_simdgroup]],                               \
    uint simd_group [[simdgroup_index_in_threadgroup]]

#define GF_PERSISTENT_ENTRY(NAME, SG)                                       \
  kernel void NAME(GF_PERSISTENT_COMMON_BINDINGS,                           \
      constant FlashGDNParams &p [[buffer(16)]],                            \
      GF_PERSISTENT_GRID_BINDINGS) {                                       \
    threadgroup bfloat query[128], key[128], value[128], rows[128];          \
    threadgroup float gates[3];                                            \
    gf_persistent<false, SG>(qkv, z, a, b, weights, a_log, time_bias, norm,  \
        history, recurrent, mixed, decay, beta, recurrent_rows, output,     \
        diagnostics, p, recurrent, 0, 0, 0, group, thread_index, lane,        \
        simd_group, query, key, value, rows, gates);                         \
  }

#define GF_CAPTURE_ENTRY(NAME, SG)                                         \
  kernel void NAME(GF_PERSISTENT_COMMON_BINDINGS,                           \
      device float *captured [[buffer(16)]],                                \
      constant FlashGDNCaptureParams &p [[buffer(17)]],                      \
      GF_PERSISTENT_GRID_BINDINGS) {                                       \
    if (p.gdn.rows > 16 || p.capture_rows > p.gdn.rows || p.reserved != 0 ||  \
        p.capture_row_stride_bytes < ulong(48) * 128 * 128 * 4 ||           \
        p.capture_row_stride_bytes % 4 ||                                  \
        p.capture_lane_stride_bytes < p.capture_rows *                     \
            p.capture_row_stride_bytes || p.capture_lane_stride_bytes % 4) { \
      gf_flag(diagnostics, FlashGDNInvalidParameters); return;              \
    }                                                                     \
    threadgroup bfloat query[128], key[128], value[128], rows[128];          \
    threadgroup float gates[3];                                            \
    gf_persistent<true, SG>(qkv, z, a, b, weights, a_log, time_bias, norm,   \
        history, recurrent, mixed, decay, beta, recurrent_rows, output,     \
        diagnostics, p.gdn, captured, p.capture_row_stride_bytes,           \
        p.capture_lane_stride_bytes, p.capture_rows, group, thread_index,   \
        lane, simd_group, query, key, value, rows, gates);                   \
  }

GF_PERSISTENT_ENTRY(flash_gdn_fused_persistent, 8)
GF_PERSISTENT_ENTRY(flash_gdn_fused_persistent_sg16, 16)
GF_PERSISTENT_ENTRY(flash_gdn_fused_persistent_sg32, 32)
GF_CAPTURE_ENTRY(flash_gdn_fused_persistent_capture, 8)
GF_CAPTURE_ENTRY(flash_gdn_fused_persistent_capture_sg16, 16)
GF_CAPTURE_ENTRY(flash_gdn_fused_persistent_capture_sg32, 32)

#undef GF_PERSISTENT_ENTRY
#undef GF_CAPTURE_ENTRY

#undef GF_PERSISTENT_COMMON_BINDINGS
#undef GF_PERSISTENT_GRID_BINDINGS

kernel void flash_gdn_fused_separate_sg16(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *z [[buffer(1)]],
    device const bfloat *a [[buffer(2)]],
    device const bfloat *b [[buffer(3)]],
    device const bfloat *weights [[buffer(4)]],
    device const bfloat *a_log [[buffer(5)]],
    device const bfloat *time_bias [[buffer(6)]],
    device const bfloat *norm [[buffer(7)]],
    device const bfloat *history0 [[buffer(8)]],
    device const bfloat *history1 [[buffer(9)]],
    device const bfloat *history2 [[buffer(10)]],
    device const bfloat *history3 [[buffer(11)]],
    device float *recurrent0 [[buffer(12)]],
    device float *recurrent1 [[buffer(13)]],
    device float *recurrent2 [[buffer(14)]],
    device float *recurrent3 [[buffer(15)]],
    device bfloat *mixed [[buffer(16)]],
    device float *decay [[buffer(17)]],
    device bfloat *beta [[buffer(18)]],
    device bfloat *recurrent_rows [[buffer(19)]],
    device bfloat *output [[buffer(20)]],
    device atomic_uint &diagnostics [[buffer(21)]],
    constant FlashGDNSeparateParams &params [[buffer(22)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (!params.lanes || params.lanes > 4 || !isfinite(params.norm_epsilon) ||
      params.norm_epsilon <= 0.0f) {
    gf_flag(diagnostics, FlashGDNInvalidParameters); return;
  }
  if (group.x >= 48 || group.z >= params.lanes) return;
  device const bfloat *history = group.z == 0 ? history0 : group.z == 1 ? history1
                                          : group.z == 2 ? history2 : history3;
  device float *recurrent = group.z == 0 ? recurrent0 : group.z == 1 ? recurrent1
                                       : group.z == 2 ? recurrent2 : recurrent3;
  const FlashGDNParams p{1, params.lanes, 16, 48, 128, 128, 4,
                         params.norm_epsilon, ulong(3) * GFConvWidth * 2,
                         ulong(48) * 128 * 128 * 4};
  threadgroup bfloat query[128], key[128], value[128], rows[128];
  threadgroup float gates[3];
  gf_persistent<false, 16, true>(qkv, z, a, b, weights, a_log, time_bias, norm,
      history, recurrent, mixed, decay, beta, recurrent_rows, output,
      diagnostics, p, recurrent, 0, 0, 0, uint2(group.x, group.z), thread_index,
      lane, simd_group, query, key, value, rows, gates);
}

kernel void flash_gdn_fused_separate_carry(
    device const bfloat *qkv [[buffer(0)]],
    device bfloat *history0 [[buffer(1)]],
    device bfloat *history1 [[buffer(2)]],
    device bfloat *history2 [[buffer(3)]],
    device bfloat *history3 [[buffer(4)]],
    device atomic_uint &diagnostics [[buffer(5)]],
    constant FlashGDNSeparateParams &p [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  if (!p.lanes || p.lanes > 4 || !isfinite(p.norm_epsilon) || p.norm_epsilon <= 0.0f) {
    gf_flag(diagnostics, FlashGDNInvalidParameters); return;
  }
  const uint channel = group.x * 256 + thread_index;
  if (channel >= GFConvWidth || group.z >= p.lanes) return;
  device bfloat *history = group.z == 0 ? history0 : group.z == 1 ? history1
                                    : group.z == 2 ? history2 : history3;
  history[channel] = history[GFConvWidth + channel];
  history[GFConvWidth + channel] = history[2 * GFConvWidth + channel];
  history[2 * GFConvWidth + channel] = qkv[ulong(group.z) * GFConvWidth + channel];
}

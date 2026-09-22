#include "metal/abi/FlashGDNLazyRollback.h"
#include <metal_stdlib>

using namespace metal;

// Private lazy rollback candidate. Helpers and the persistent SG16 math are
// copied from the qualified fused kernel; all names are private to this file.
// The initial load adds one snapshot store. Capture is instantiated false, so
// there are no per-token recurrent snapshot stores.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

constant uint LazyGFKeyWidth = 2048;
constant uint LazyGFValueWidth = 6144;
constant uint LazyGFConvWidth = 10240;

inline void lazy_gf_flag(device atomic_uint &diagnostics, uint flag) {
  atomic_fetch_or_explicit(&diagnostics, flag, memory_order_relaxed);
}

inline bool lazy_gf_valid(FlashGDNParams p, device atomic_uint &diagnostics) {
  const bool valid = p.rows && p.rows <= 16 && p.lanes && p.lanes <= 4 &&
      p.key_heads == 16 && p.value_heads == 48 && p.key_dimension == 128 &&
      p.value_dimension == 128 && p.convolution_taps == 4 &&
      isfinite(p.norm_epsilon) && p.norm_epsilon > 0.0f &&
      p.convolution_lane_stride_bytes >= ulong(3) * LazyGFConvWidth * 2 &&
      p.convolution_lane_stride_bytes % 2 == 0 &&
      (p.lanes <= 1 || p.convolution_lane_stride_bytes <=
          (ulong(-1) - ulong(3) * LazyGFConvWidth * 2) / (p.lanes - 1)) &&
      p.recurrent_lane_stride_bytes >= ulong(48) * 128 * 128 * 4 &&
      p.recurrent_lane_stride_bytes % 4 == 0 &&
      (p.lanes <= 1 || p.recurrent_lane_stride_bytes <=
          (ulong(-1) - ulong(48) * 128 * 128 * 4) / (p.lanes - 1));
  if (!valid) lazy_gf_flag(diagnostics, FlashGDNInvalidParameters);
  return valid;
}

inline void lazy_gf_check(float value, device atomic_uint &diagnostics) {
  if (!isfinite(value)) lazy_gf_flag(diagnostics, FlashGDNNonFinite);
}

inline bfloat lazy_gf_sigmoid_compiled(bfloat source) {
  const bfloat exponent = bfloat(metal::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f + float(exponent));
  const bfloat tail = bfloat(1.0f / float(denominator));
  return float(source) < 0.0f ? tail : bfloat(1.0f - float(tail));
}

inline bfloat lazy_gf_sigmoid_unary(bfloat source) {
  const bfloat exponent =
      bfloat(metal::precise::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f + float(exponent));
  const bfloat tail = bfloat(1.0f / float(denominator));
  return float(source) < 0.0f ? tail : bfloat(1.0f - float(tail));
}

inline bfloat lazy_gf_softplus(bfloat source) {
  const bfloat exponent = bfloat(metal::exp(-metal::abs(float(source))));
  const float xp1 = 1.0f + float(exponent);
  const bfloat logarithm = xp1 == 1.0f ? exponent :
      bfloat(float(exponent) * (metal::log(xp1) / (xp1 - 1.0f)));
  return bfloat(max(float(source), 0.0f) + float(logarithm));
}

inline float lazy_gf_sigmoid_z(float source) {
  const float tail =
      1.0f / (1.0f + metal::precise::exp(metal::abs(source)));
  return source < 0.0f ? tail : 1.0f - tail;
}

template <bool Separate = false>
inline bfloat lazy_gf_conv(device const bfloat *qkv,
                       device const bfloat *weights,
                       device const bfloat *history, FlashGDNParams p,
                       uint batch, uint token, uint channel) {
  device const bfloat *lane_history =
      history + (Separate ? 0 : ulong(batch) * p.convolution_lane_stride_bytes / 2);
  device const bfloat *lane_input = qkv + ulong(batch) * p.rows * LazyGFConvWidth;
  float total = 0.0f;
  for (uint tap = 0; tap < 4; ++tap) {
    const uint position = token + tap;
    const bfloat source = position < 3
        ? lane_history[position * LazyGFConvWidth + channel]
        : lane_input[ulong(position - 3) * LazyGFConvWidth + channel];
    total += float(source) * float(weights[channel * 4 + tap]);
  }
  const bfloat convolution = bfloat(total);
  return bfloat(float(convolution) * float(lazy_gf_sigmoid_compiled(convolution)));
}

template <bool Separate = false>
inline void lazy_gf_qk(device const bfloat *qkv,
                   device const bfloat *weights,
                   device const bfloat *history, FlashGDNParams p,
                   uint batch, uint token, uint key_head, uint lane,
                   thread bfloat *query, thread bfloat *key) {
  float q[4], k[4];
  bfloat qp = bfloat(0.0f), kp = bfloat(0.0f);
  for (uint i = 0; i < 4; ++i) {
    q[i] = float(lazy_gf_conv<Separate>(qkv, weights, history, p, batch, token,
                         key_head * 128 + 4 * lane + i));
    k[i] = float(lazy_gf_conv<Separate>(qkv, weights, history, p, batch, token,
                         LazyGFKeyWidth + key_head * 128 + 4 * lane + i));
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

inline void lazy_gf_gates(device const bfloat *a, device const bfloat *b,
                      device const bfloat *a_log,
                      device const bfloat *time_bias, ulong index, uint head,
                      thread float &decay, thread bfloat &beta) {
  const bfloat sum = bfloat(float(a[index]) + float(time_bias[head]));
  decay = metal::exp(-metal::exp(float(a_log[head])) * float(lazy_gf_softplus(sum)));
  beta = lazy_gf_sigmoid_unary(b[index]);
}

template <bool Capture, uint Simdgroups, bool Separate = false>
inline void lazy_gf_persistent(
    device const bfloat *qkv, device const bfloat *z,
    device const bfloat *a, device const bfloat *b,
    device const bfloat *weights, device const bfloat *a_log,
    device const bfloat *time_bias, device const bfloat *norm,
    device const bfloat *history, device float *recurrent,
    device bfloat *mixed, device float *decay, device bfloat *beta,
    device bfloat *recurrent_rows, device bfloat *output,
    device atomic_uint &diagnostics, FlashGDNParams p,
    device float *captured, ulong capture_row_stride,
    ulong capture_lane_stride, uint capture_rows, device float *initial_snapshot,
    ulong snapshot_lane_stride, uint2 group, uint thread_index,
    uint lane, uint simd_group,
    threadgroup bfloat *query, threadgroup bfloat *key,
    threadgroup bfloat *value, threadgroup bfloat *rows,
    threadgroup float *gates) {
  if (!lazy_gf_valid(p, diagnostics) || group.x >= 48 || group.y >= p.lanes) return;
  const uint head = group.x, batch = group.y, key_head = head / 3;
  const bool shared_writer = head % 3 == 0;
  constexpr uint ValueRows = 128 / Simdgroups;
  float state[ValueRows][4];
  const ulong state_lane = Separate ? 0 : ulong(batch) * p.recurrent_lane_stride_bytes / 4;
  for (uint r = 0; r < ValueRows; ++r) {
    const uint dimension = r * Simdgroups + simd_group;
    const ulong base = state_lane + (head * 128 + dimension) * 128 + 4 * lane;
    const ulong snapshot_base = ulong(batch) * snapshot_lane_stride / 4 +
        (head * 128 + dimension) * 128 + 4 * lane;
    for (uint i = 0; i < 4; ++i) {
      state[r][i] = recurrent[base + i];
      initial_snapshot[snapshot_base + i] = state[r][i];
    }
  }
  for (uint token = 0; token < p.rows; ++token) {
    const ulong row = ulong(batch) * p.rows + token;
    if (simd_group == 0) {
      bfloat q[4], k[4];
      lazy_gf_qk<Separate>(qkv, weights, history, p, batch, token, key_head, lane, q, k);
      for (uint i = 0; i < 4; ++i) {
        query[4 * lane + i] = q[i]; key[4 * lane + i] = k[i];
        if (shared_writer) {
          mixed[row * LazyGFConvWidth + key_head * 128 + 4 * lane + i] = q[i];
          mixed[row * LazyGFConvWidth + LazyGFKeyWidth + key_head * 128 + 4 * lane + i] = k[i];
        }
        lazy_gf_check(float(q[i]), diagnostics); lazy_gf_check(float(k[i]), diagnostics);
      }
    }
    if (thread_index < 128) {
      const uint channel = 2 * LazyGFKeyWidth + head * 128 + thread_index;
      value[thread_index] = lazy_gf_conv<Separate>(qkv, weights, history, p, batch, token,
                                     channel);
      mixed[row * LazyGFConvWidth + channel] = value[thread_index];
      lazy_gf_check(float(value[thread_index]), diagnostics);
    }
    if (thread_index == 0) {
      float d; bfloat be;
      lazy_gf_gates(a, b, a_log, time_bias, row * 48 + head, head, d, be);
      gates[0] = d; gates[1] = float(be);
      decay[row * 48 + head] = d; beta[row * 48 + head] = be;
      lazy_gf_check(d, diagnostics); lazy_gf_check(float(be), diagnostics);
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
        recurrent_rows[row * LazyGFValueWidth + head * 128 + dimension] = rows[dimension];
        lazy_gf_check(result, diagnostics);
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
      const ulong index = row * LazyGFValueWidth + head * 128 + thread_index;
      const bfloat result = bfloat(float(weighted) * lazy_gf_sigmoid_z(float(z[index])));
      output[index] = result; lazy_gf_check(float(result), diagnostics);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint r = 0; r < ValueRows; ++r) {
    const uint dimension = r * Simdgroups + simd_group;
    const ulong base = state_lane + (head * 128 + dimension) * 128 + 4 * lane;
    for (uint i = 0; i < 4; ++i) {
      recurrent[base + i] = state[r][i]; lazy_gf_check(state[r][i], diagnostics);
    }
  }
}

constant ulong LazyRecurrentLaneBytes = ulong(48) * 128 * 128 * 4;
constant ulong LazyConvolutionLaneBytes = ulong(3) * 10240 * 2;

inline bool lazy_snapshot_stride_valid(ulong stride, uint lanes,
                                       device atomic_uint &diagnostics) {
  const bool valid = stride >= LazyRecurrentLaneBytes && stride % 4 == 0 &&
      (lanes <= 1 || stride <= (ulong(-1) - LazyRecurrentLaneBytes) / (lanes - 1));
  if (!valid) lazy_gf_flag(diagnostics, FlashGDNInvalidParameters);
  return valid;
}

inline bool lazy_grid_valid(uint3 group_count, uint3 threads, uint simd_width,
                            uint lanes, uint groups, uint required_threads,
                            device atomic_uint &diagnostics) {
  const bool valid = group_count.x == groups && group_count.y == lanes &&
      group_count.z == 1 && threads.x == required_threads && threads.y == 1 &&
      threads.z == 1 && simd_width == 32;
  if (!valid) lazy_gf_flag(diagnostics, FlashGDNInvalidParameters);
  return valid;
}

inline bool lazy_replay_valid(constant FlashGDNLazyReplayParams &params,
                              device atomic_uint &diagnostics) {
  if (!lazy_gf_valid(params.gdn, diagnostics) ||
      !lazy_snapshot_stride_valid(params.snapshot_lane_stride_bytes,
                                   params.gdn.lanes, diagnostics)) return false;
  for (uint lane = 0; lane < params.gdn.lanes; ++lane) {
    if (params.retained[lane] > params.gdn.rows) {
      lazy_gf_flag(diagnostics, FlashGDNInvalidParameters);
      return false;
    }
  }
  return true;
}

kernel void flash_gdn_lazy_verify_sg16(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *z [[buffer(1)]],
    device const bfloat *a [[buffer(2)]],
    device const bfloat *b [[buffer(3)]],
    device const bfloat *weights [[buffer(4)]],
    device const bfloat *a_log [[buffer(5)]],
    device const bfloat *time_bias [[buffer(6)]],
    device const bfloat *norm [[buffer(7)]],
    device const bfloat *history [[buffer(8)]],
    device float *recurrent [[buffer(9)]],
    device bfloat *mixed [[buffer(10)]],
    device float *decay [[buffer(11)]],
    device bfloat *beta [[buffer(12)]],
    device bfloat *recurrent_rows [[buffer(13)]],
    device bfloat *output [[buffer(14)]],
    device atomic_uint &diagnostics [[buffer(15)]],
    device float *initial_snapshot [[buffer(16)]],
    constant FlashGDNLazyVerifyParams &params [[buffer(17)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 group_count [[threadgroups_per_grid]],
    uint3 threads [[threads_per_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (!lazy_grid_valid(group_count, threads, simd_width, params.gdn.lanes,
                        48, 512, diagnostics) ||
      !lazy_gf_valid(params.gdn, diagnostics) ||
      !lazy_snapshot_stride_valid(params.snapshot_lane_stride_bytes,
                                   params.gdn.lanes, diagnostics)) return;
  threadgroup bfloat query[128], key[128], value[128], rows[128];
  threadgroup float gates[3];
  lazy_gf_persistent<false, 16>(qkv, z, a, b, weights, a_log, time_bias, norm,
      history, recurrent, mixed, decay, beta, recurrent_rows, output,
      diagnostics, params.gdn, initial_snapshot, 0, 0, 0, initial_snapshot,
      params.snapshot_lane_stride_bytes, uint2(group.x, group.y), thread_index,
      lane, simd_group, query, key, value, rows, gates);
}

// Reconstruct only the state recurrence, using the exact prepared BF16 operands
// and F32 gates saved by verification. The SIMD32 reduction and operation order
// match lazy_gf_persistent<false, 16>; no q/output/norm work is required.
kernel void flash_gdn_lazy_replay_sg16(
    device const bfloat *mixed [[buffer(0)]],
    device const float *decay [[buffer(1)]],
    device const bfloat *beta [[buffer(2)]],
    device const float *initial_snapshot [[buffer(3)]],
    device float *recurrent [[buffer(4)]],
    device atomic_uint &diagnostics [[buffer(5)]],
    constant FlashGDNLazyReplayParams &params [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 group_count [[threadgroups_per_grid]],
    uint3 threads [[threads_per_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const FlashGDNParams p = params.gdn;
  if (!lazy_grid_valid(group_count, threads, simd_width, p.lanes,
                        48, 512, diagnostics) ||
      !lazy_replay_valid(params, diagnostics)) return;
  if (group.x >= 48 || group.y >= p.lanes || group.z != 0) return;
  const uint kept = params.retained[group.y];
  if (kept == p.rows) return;
  const uint head = group.x, batch = group.y, key_head = head / 3;
  constexpr uint Simdgroups = 16;
  constexpr uint ValueRows = 128 / Simdgroups;
  float state[ValueRows][4];
  const ulong snapshot_lane = ulong(batch) * params.snapshot_lane_stride_bytes / 4;
  for (uint r = 0; r < ValueRows; ++r) {
    const uint dimension = r * Simdgroups + simd_group;
    const ulong base = snapshot_lane + (head * 128 + dimension) * 128 + 4 * lane;
    for (uint i = 0; i < 4; ++i) state[r][i] = initial_snapshot[base + i];
  }
  for (uint token = 0; token < kept; ++token) {
    const ulong row = ulong(batch) * p.rows + token;
    float k[4];
    for (uint i = 0; i < 4; ++i)
      k[i] = float(mixed[row * LazyGFConvWidth + LazyGFKeyWidth +
                         key_head * 128 + 4 * lane + i]);
    const float d = decay[row * 48 + head];
    const float be = float(beta[row * 48 + head]);
    for (uint r = 0; r < ValueRows; ++r) {
      const uint dimension = r * Simdgroups + simd_group;
      float memory = 0.0f;
      for (uint i = 0; i < 4; ++i) {
        state[r][i] = state[r][i] * d;
        memory += state[r][i] * k[i];
      }
      memory = simd_sum(memory);
      const float value = float(mixed[row * LazyGFConvWidth +
                                       2 * LazyGFKeyWidth + head * 128 + dimension]);
      const float delta = (value - memory) * be;
      for (uint i = 0; i < 4; ++i)
        state[r][i] = state[r][i] + k[i] * delta;
    }
  }
  const ulong state_lane = ulong(batch) * p.recurrent_lane_stride_bytes / 4;
  for (uint r = 0; r < ValueRows; ++r) {
    const uint dimension = r * Simdgroups + simd_group;
    const ulong base = state_lane + (head * 128 + dimension) * 128 + 4 * lane;
    for (uint i = 0; i < 4; ++i) {
      recurrent[base + i] = state[r][i];
      lazy_gf_check(state[r][i], diagnostics);
    }
  }
}

kernel void flash_gdn_lazy_restore_convolution(
    device const bfloat *initial_history [[buffer(0)]],
    device const bfloat *saved_qkv [[buffer(1)]],
    device bfloat *history [[buffer(2)]],
    device atomic_uint &diagnostics [[buffer(3)]],
    constant FlashGDNLazyReplayParams &params [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 group_count [[threadgroups_per_grid]],
    uint3 threads [[threads_per_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  const FlashGDNParams p = params.gdn;
  if (!lazy_grid_valid(group_count, threads, simd_width, p.lanes,
                        40, 256, diagnostics) ||
      !lazy_replay_valid(params, diagnostics)) return;
  const uint channel = group.x * 256 + thread_index;
  if (channel >= LazyGFConvWidth || group.y >= p.lanes || group.z != 0) return;
  const uint kept = params.retained[group.y];
  if (kept == p.rows) return;
  const ulong history_lane = ulong(group.y) * p.convolution_lane_stride_bytes / 2;
  const ulong initial_lane = ulong(group.y) * LazyConvolutionLaneBytes / 2;
  const ulong input_lane = ulong(group.y) * p.rows * LazyGFConvWidth;
  for (uint row = 0; row < 3; ++row) {
    const bfloat source = kept >= 3
        ? saved_qkv[input_lane + ulong(kept - 3 + row) * LazyGFConvWidth + channel]
        : row < 3 - kept
            ? initial_history[initial_lane + ulong(row + kept) * LazyGFConvWidth + channel]
            : saved_qkv[input_lane + ulong(row - (3 - kept)) * LazyGFConvWidth + channel];
    history[history_lane + ulong(row) * LazyGFConvWidth + channel] = source;
  }
}

kernel void flash_gdn_lazy_copy(
    device const uchar *input [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant FlashGDNLazyCopyParams &params [[buffer(2)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 group_count [[threadgroups_per_grid]],
    uint3 threads [[threads_per_threadgroup]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  const ulong required_groups = params.bytes / 256 + (params.bytes % 256 != 0);
  if (threads.x != 256 || threads.y != 1 || threads.z != 1 ||
      group_count.x != required_groups || group_count.y != 1 || group_count.z != 1)
    return;
  const ulong index = ulong(group.x) * 256 + thread_index;
  if (index < params.bytes) output[index] = input[index];
}

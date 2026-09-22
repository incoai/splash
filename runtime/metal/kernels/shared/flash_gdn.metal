#include "metal/abi/FlashGDN.h"
#include <metal_stdlib>

using namespace metal;

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

constant uint FlashGDNKeyHeads = 16;
constant uint FlashGDNValueHeads = 48;
constant uint FlashGDNHeadDimension = 128;
constant uint FlashGDNKeyWidth = 2048;
constant uint FlashGDNValueWidth = 6144;
constant uint FlashGDNConvolutionWidth = 10240;

inline void flash_gdn_flag(device atomic_uint &diagnostics, uint flag) {
  atomic_fetch_or_explicit(&diagnostics, flag, memory_order_relaxed);
}

inline bool flash_gdn_parameters(FlashGDNParams p,
                                 device atomic_uint &diagnostics) {
  const bool valid =
      p.rows >= 1 && p.rows <= 2048 && p.lanes >= 1 && p.lanes <= 32 &&
      p.key_heads == FlashGDNKeyHeads &&
      p.value_heads == FlashGDNValueHeads &&
      p.key_dimension == FlashGDNHeadDimension &&
      p.value_dimension == FlashGDNHeadDimension && p.convolution_taps == 4 &&
      isfinite(p.norm_epsilon) && p.norm_epsilon > 0.0f &&
      p.convolution_lane_stride_bytes >=
          ulong(3) * FlashGDNConvolutionWidth * sizeof(bfloat) &&
      p.convolution_lane_stride_bytes % sizeof(bfloat) == 0 &&
      p.recurrent_lane_stride_bytes >=
          ulong(FlashGDNValueHeads) * FlashGDNHeadDimension *
              FlashGDNHeadDimension * sizeof(float) &&
      p.recurrent_lane_stride_bytes % sizeof(float) == 0;
  if (!valid)
    flash_gdn_flag(diagnostics, FlashGDNInvalidParameters);
  return valid;
}

inline float flash_gdn_sigmoid(float x) {
  // The final z gate is an unfused F32 mx.sigmoid unary operator.
  const float tail = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
  return x < 0.0f ? tail : 1.0f - tail;
}

inline bfloat flash_gdn_sigmoid_bf16_compiled(bfloat source) {
  // MLX's bfloat exp overload and integer-literal arithmetic each retain
  // BF16. A single F32 sigmoid followed by a cast has different semantics.
  const bfloat exponent = bfloat(metal::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f + float(exponent));
  const bfloat tail = bfloat(1.0f / float(denominator));
  return float(source) < 0.0f ? tail : bfloat(1.0f - float(tail));
}

inline bfloat flash_gdn_sigmoid_bf16_unary(bfloat source) {
  // _compute_g_beta's independent one-op beta branch stays an unfused MLX
  // Sigmoid primitive, whose GPU exponential is precise. Convolution SiLU
  // has connected sigmoid/multiply nodes and uses the compiled fast variant.
  const bfloat exponent =
      bfloat(metal::precise::exp(metal::abs(float(source))));
  const bfloat denominator = bfloat(1.0f + float(exponent));
  const bfloat tail = bfloat(1.0f / float(denominator));
  return float(source) < 0.0f ? tail : bfloat(1.0f - float(tail));
}

inline bfloat flash_gdn_softplus_bf16(bfloat source) {
  // MLX LogAddExp(x,0): bfloat exp, its bfloat log1p helper, then the
  // bfloat addition to max(x,0). The helper keeps accuracy for tiny tails.
  const bfloat exponent = bfloat(metal::exp(-metal::abs(float(source))));
  const float xp1 = 1.0f + float(exponent);
  const bfloat logarithm =
      xp1 == 1.0f
          ? exponent
          : bfloat(float(exponent) * (metal::log(xp1) / (xp1 - 1.0f)));
  return bfloat(max(float(source), 0.0f) + float(logarithm));
}

inline void flash_gdn_check(float value, device atomic_uint &diagnostics) {
  if (!isfinite(value))
    flash_gdn_flag(diagnostics, FlashGDNNonFinite);
}

kernel void flash_gdn_convolution(
    device const bfloat *qkv [[buffer(0)]],
    device const bfloat *weights [[buffer(1)]],
    device const bfloat *history [[buffer(2)]],
    device bfloat *mixed [[buffer(3)]],
    device atomic_uint &diagnostics [[buffer(4)]],
    constant FlashGDNParams &p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  if (!flash_gdn_parameters(p, diagnostics))
    return;
  const uint channel = group.x * 256 + thread_index;
  const uint token = group.y;
  const uint lane = group.z;
  if (channel >= FlashGDNConvolutionWidth || token >= p.rows || lane >= p.lanes)
    return;
  device const bfloat *lane_history =
      history + ulong(lane) * p.convolution_lane_stride_bytes / sizeof(bfloat);
  device const bfloat *lane_input =
      qkv + ulong(lane) * p.rows * FlashGDNConvolutionWidth;
  float total = 0.0f;
  for (uint tap = 0; tap < 4; ++tap) {
    const uint position = token + tap;
    const bfloat source =
        position < 3
            ? lane_history[position * FlashGDNConvolutionWidth + channel]
            : lane_input[ulong(position - 3) * FlashGDNConvolutionWidth + channel];
    total += float(source) * float(weights[channel * 4 + tap]);
  }
  const bfloat convolution = bfloat(total);
  // nn.silu on a BF16 array has a BF16 sigmoid before the BF16 product.
  const bfloat gate = flash_gdn_sigmoid_bf16_compiled(convolution);
  const bfloat result = bfloat(float(convolution) * float(gate));
  mixed[(ulong(lane) * p.rows + token) * FlashGDNConvolutionWidth + channel] =
      result;
  flash_gdn_check(float(result), diagnostics);
}

kernel void flash_gdn_normalize_qk(
    device bfloat *mixed [[buffer(0)]],
    device atomic_uint &diagnostics [[buffer(1)]],
    constant FlashGDNParams &p [[buffer(2)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (!flash_gdn_parameters(p, diagnostics))
    return;
  if (group.x >= FlashGDNKeyHeads || group.y >= p.rows || group.z >= p.lanes)
    return;
  const ulong base = (ulong(group.z) * p.rows + group.y) *
                         FlashGDNConvolutionWidth +
                     group.x * FlashGDNHeadDimension;
  float q[4], k[4];
  bfloat q_partial = bfloat(0.0f), k_partial = bfloat(0.0f);
  // MLX's width128 BF16 sum gives each lane four contiguous values, rounds
  // each local addition BF16, then widens partials for the SIMD reduction.
  for (uint i = 0; i < 4; ++i) {
    q[i] = float(mixed[base + 4 * lane + i]);
    k[i] = float(mixed[base + FlashGDNKeyWidth + 4 * lane + i]);
    q_partial = bfloat(float(q_partial) + float(bfloat(q[i] * q[i])));
    k_partial = bfloat(float(k_partial) + float(bfloat(k[i] * k[i])));
  }
  const bfloat q_total = bfloat(simd_sum(float(q_partial)));
  const bfloat k_total = bfloat(simd_sum(float(k_partial)));
  const bfloat epsilon = bfloat(1e-6f);
  const bfloat q_denominator = bfloat(float(q_total) + float(epsilon));
  const bfloat k_denominator = bfloat(float(k_total) + float(epsilon));
  const float q_inverse =
      float(bfloat(metal::precise::rsqrt(float(q_denominator))));
  const float k_inverse =
      float(bfloat(metal::precise::rsqrt(float(k_denominator))));
  const bfloat query_scale = bfloat(0.08838834764831845f);
  for (uint i = 0; i < 4; ++i) {
    const bfloat normalized_q = bfloat(q[i] * q_inverse);
    const bfloat normalized_k = bfloat(k[i] * k_inverse);
    const bfloat query = bfloat(float(normalized_q) * float(query_scale));
    mixed[base + 4 * lane + i] = query;
    mixed[base + FlashGDNKeyWidth + 4 * lane + i] = normalized_k;
    flash_gdn_check(float(query), diagnostics);
    flash_gdn_check(float(normalized_k), diagnostics);
  }
}

kernel void flash_gdn_gates(
    device const bfloat *a [[buffer(0)]],
    device const bfloat *b [[buffer(1)]],
    device const bfloat *a_log [[buffer(2)]],
    device const bfloat *time_bias [[buffer(3)]],
    device float *decay [[buffer(4)]],
    device bfloat *beta [[buffer(5)]],
    device atomic_uint &diagnostics [[buffer(6)]],
    constant FlashGDNParams &p [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint head [[thread_index_in_threadgroup]]) {
  if (!flash_gdn_parameters(p, diagnostics))
    return;
  if (head >= FlashGDNValueHeads || group.y >= p.rows || group.z >= p.lanes)
    return;
  const ulong index = (ulong(group.z) * p.rows + group.y) * FlashGDNValueHeads +
                      head;
  const bfloat sum = bfloat(float(a[index]) + float(time_bias[head]));
  const bfloat softplus = flash_gdn_softplus_bf16(sum);
  const float value =
      metal::exp(-metal::exp(float(a_log[head])) * float(softplus));
  const bfloat beta_value = flash_gdn_sigmoid_bf16_unary(b[index]);
  decay[index] = value;
  beta[index] = beta_value;
  flash_gdn_check(value, diagnostics);
  flash_gdn_check(float(beta_value), diagnostics);
}

kernel void flash_gdn_recurrence(
    device const bfloat *mixed [[buffer(0)]],
    device const float *decay [[buffer(1)]],
    device const bfloat *beta [[buffer(2)]],
    device float *recurrent [[buffer(3)]],
    device bfloat *rows [[buffer(4)]],
    device atomic_uint &diagnostics [[buffer(5)]],
    constant FlashGDNParams &p [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  if (!flash_gdn_parameters(p, diagnostics))
    return;
  const uint head = group.x;
  const uint value_dimension = group.y * 4 + simd_group;
  const uint batch = group.z;
  if (head >= FlashGDNValueHeads || value_dimension >= FlashGDNHeadDimension ||
      batch >= p.lanes)
    return;
  const uint key_head = head / (FlashGDNValueHeads / FlashGDNKeyHeads);
  const ulong state_base = ulong(batch) * p.recurrent_lane_stride_bytes /
                              sizeof(float) +
                          (head * FlashGDNHeadDimension + value_dimension) *
                              FlashGDNHeadDimension;
  float state[4];
  for (uint i = 0; i < 4; ++i)
    state[i] = recurrent[state_base + 4 * lane + i];
  for (uint token = 0; token < p.rows; ++token) {
    const ulong mixed_base =
        (ulong(batch) * p.rows + token) * FlashGDNConvolutionWidth;
    const ulong key_base = mixed_base + key_head * FlashGDNHeadDimension;
    const ulong gate_index =
        (ulong(batch) * p.rows + token) * FlashGDNValueHeads + head;
    float q[4], k[4];
    float memory = 0.0f;
    for (uint i = 0; i < 4; ++i) {
      q[i] = float(mixed[key_base + 4 * lane + i]);
      k[i] = float(mixed[key_base + FlashGDNKeyWidth + 4 * lane + i]);
      state[i] = state[i] * decay[gate_index];
      memory += state[i] * k[i];
    }
    memory = simd_sum(memory);
    const float value =
        float(mixed[mixed_base + 2 * FlashGDNKeyWidth +
                    head * FlashGDNHeadDimension + value_dimension]);
    const float delta = (value - memory) * float(beta[gate_index]);
    float output = 0.0f;
    for (uint i = 0; i < 4; ++i) {
      state[i] = state[i] + k[i] * delta;
      output += state[i] * q[i];
    }
    output = simd_sum(output);
    if (lane == 0) {
      rows[(ulong(batch) * p.rows + token) * FlashGDNValueWidth +
           head * FlashGDNHeadDimension + value_dimension] = bfloat(output);
      flash_gdn_check(output, diagnostics);
    }
  }
  for (uint i = 0; i < 4; ++i) {
    recurrent[state_base + 4 * lane + i] = state[i];
    flash_gdn_check(state[i], diagnostics);
  }
}

kernel void flash_gdn_output(
    device const bfloat *recurrent_rows [[buffer(0)]],
    device const bfloat *z [[buffer(1)]],
    device const bfloat *norm [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    device atomic_uint &diagnostics [[buffer(4)]],
    constant FlashGDNParams &p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  if (!flash_gdn_parameters(p, diagnostics))
    return;
  if (group.x >= FlashGDNValueHeads || group.y >= p.rows || group.z >= p.lanes)
    return;
  const ulong base = (ulong(group.z) * p.rows + group.y) * FlashGDNValueWidth +
                     group.x * FlashGDNHeadDimension;
  float values[4];
  float square_sum = 0.0f;
  for (uint i = 0; i < 4; ++i) {
    values[i] = float(recurrent_rows[base + 4 * lane + i]);
    square_sum += values[i] * values[i];
  }
  const float total = simd_sum(square_sum);
  const float inverse = metal::precise::rsqrt(total / FlashGDNHeadDimension +
                                            p.norm_epsilon);
  for (uint i = 0; i < 4; ++i) {
    const uint dimension = 4 * lane + i;
    // The MLX fast RMS kernel rounds x*inverse before the BF16 gamma product.
    const bfloat normalized = bfloat(values[i] * inverse);
    const bfloat weighted = bfloat(float(normalized) * float(norm[dimension]));
    const float gate = flash_gdn_sigmoid(float(z[base + dimension]));
    const bfloat result = bfloat(float(weighted) * gate);
    output[base + dimension] = result;
    flash_gdn_check(float(result), diagnostics);
  }
}

kernel void flash_gdn_convolution_carry(
    device const bfloat *qkv [[buffer(0)]],
    device bfloat *history [[buffer(1)]],
    device atomic_uint &diagnostics [[buffer(2)]],
    constant FlashGDNParams &p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  if (!flash_gdn_parameters(p, diagnostics))
    return;
  const uint channel = group.x * 256 + thread_index;
  const uint lane = group.y;
  if (channel >= FlashGDNConvolutionWidth || lane >= p.lanes)
    return;
  device bfloat *lane_history =
      history + ulong(lane) * p.convolution_lane_stride_bytes / sizeof(bfloat);
  device const bfloat *lane_input =
      qkv + ulong(lane) * p.rows * FlashGDNConvolutionWidth;
  // Ascending destination order preserves the old source rows for rows<3.
  for (uint row = 0; row < 3; ++row) {
    const uint source = p.rows + row;
    lane_history[row * FlashGDNConvolutionWidth + channel] =
        source < 3
            ? lane_history[source * FlashGDNConvolutionWidth + channel]
            : lane_input[ulong(source - 3) * FlashGDNConvolutionWidth + channel];
  }
}

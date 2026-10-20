#include "video/TextEncoderABI.h"

using namespace metal;

// The video model's text encoder attention (video/TextEncoder.mm): Qwen3-VL's
// Q/K RMS norms, rotary embedding and causal GQA attention over one prompt,
// in one pass. Both kernels run one simdgroup per (row, head), each lane
// holding four of the head's 128 dimensions; rotate_half pairs dimension i
// with i + 64, which lane l holds with lane l ^ 16.

constant constexpr uint kHead = SPLASH_TEXT_ENCODER_HEAD_DIMENSION;

// The fused projection's rows [rows][queries | keys | values] become
// head-major queries [query head][row][128] and keys and values
// [kv head][row][128]. Queries and keys are RMS normalized and rounded to
// bf16 as the norm writes them, then rotated by the row's position.
kernel void text_encoder_qk_norm_rope(
    device const bfloat *packed [[buffer(0)]],
    device const bfloat *query_norm [[buffer(1)]],
    device const bfloat *key_norm [[buffer(2)]],
    device const float *inverse_frequencies [[buffer(3)]],
    device bfloat *queries [[buffer(4)]],
    device bfloat *keys [[buffer(5)]],
    device bfloat *values [[buffer(6)]],
    constant TextEncoderHeadParams &params [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = group.x;
  const uint head = group.y;
  const uint heads = params.query_heads + 2 * params.kv_heads;
  const float4 x = float4(reinterpret_cast<device const bfloat4 *>(
      packed + (ulong(row) * heads + head) * kHead)[lane]);
  if (head >= params.query_heads + params.kv_heads) {
    const uint value_head = head - params.query_heads - params.kv_heads;
    reinterpret_cast<device bfloat4 *>(
        values + (ulong(value_head) * params.rows + row) * kHead)[lane] = bfloat4(x);
    return;
  }
  const bool query = head < params.query_heads;
  const float4 weight =
      float4(reinterpret_cast<device const bfloat4 *>(query ? query_norm : key_norm)[lane]);
  const float inverse_rms = rsqrt(simd_sum(dot(x, x)) / float(kHead) + params.value);
  const float4 normalized = float4(bfloat4(x * inverse_rms * weight));
  const float4 partner = simd_shuffle_xor(normalized, 16);
  const float4 angle = float(row) * reinterpret_cast<device const float4 *>(inverse_frequencies)[lane & 15];
  const float4 cosine = cos(angle);
  const float4 sine = sin(angle);
  const float4 rotated =
      lane < 16 ? normalized * cosine - partner * sine : normalized * cosine + partner * sine;
  device bfloat *destination =
      query ? queries + (ulong(head) * params.rows + row) * kHead
            : keys + (ulong(head - params.query_heads) * params.rows + row) * kHead;
  reinterpret_cast<device bfloat4 *>(destination)[lane] = bfloat4(rotated);
}

// out[row][head * 128 + d]: softmax(q k^T * scale) v over keys 0..row of the
// head's KV group, an online softmax in fp32.
kernel void text_encoder_attention(
    device const bfloat *queries [[buffer(0)]],
    device const bfloat *keys [[buffer(1)]],
    device const bfloat *values [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant TextEncoderHeadParams &params [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = group.x;
  const uint head = group.y;
  const uint kv_head = head / (params.query_heads / params.kv_heads);
  const float4 query = float4(reinterpret_cast<device const bfloat4 *>(
                           queries + (ulong(head) * params.rows + row) * kHead)[lane]) *
                       params.value;
  device const bfloat4 *key = reinterpret_cast<device const bfloat4 *>(
      keys + ulong(kv_head) * params.rows * kHead) + lane;
  device const bfloat4 *value = reinterpret_cast<device const bfloat4 *>(
      values + ulong(kv_head) * params.rows * kHead) + lane;
  float maximum = -INFINITY;
  float total = 0.0f;
  float4 accumulated = 0.0f;
  for (uint j = 0; j <= row; ++j) {
    const float score = simd_sum(dot(query, float4(key[j * (kHead / 4)])));
    const float next = max(maximum, score);
    const float scale = exp(maximum - next);
    const float weight = exp(score - next);
    total = total * scale + weight;
    accumulated = accumulated * scale + weight * float4(value[j * (kHead / 4)]);
    maximum = next;
  }
  reinterpret_cast<device bfloat4 *>(
      output + ulong(row) * params.query_heads * kHead + head * kHead)[lane] =
      bfloat4(accumulated / total);
}

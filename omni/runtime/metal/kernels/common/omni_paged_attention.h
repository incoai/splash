#pragma once

// The paged decoder attention's shared pieces (kernels/omni/attention*.metal):
// Qwen3's 16 query heads over 8 KV heads of 128 dimensions, bf16 KV in
// 32-token pages (metal/abi/Omni.h), and the simdgroup page loop of the
// decode kernels.

#include "metal/abi/Omni.h"
#include <metal_stdlib>

using namespace metal;

constant constexpr uint kQueryHeads = OMNI_DECODER_QUERY_HEADS;
constant constexpr uint kKvHeads = OMNI_DECODER_KV_HEADS;
constant constexpr uint kGroup = kQueryHeads / kKvHeads;
constant constexpr uint kHeadDim = OMNI_DECODER_HEAD_DIM;
constant constexpr uint kPageTokens = OMNI_PAGE_TOKENS;
constant constexpr uint kQkvWidth = OMNI_DECODER_QKV_WIDTH;
constant constexpr uint kAttentionWidth = OMNI_DECODER_ATTENTION_WIDTH;
constant constexpr float kScale = 0.08838834764831845f; // 1 / sqrt(128)

// The page of `token` of sequence `s` in `layer`.
inline uint omni_page_of(constant OmniSequenceParams &s, device const uint *page_table, uint layer, uint token) {
  return page_table[s.page_table_offset + layer * s.page_table_stride + token / kPageTokens];
}

inline device bfloat *omni_page_keys(device uchar *pages, uint page, uint kv_head) {
  return reinterpret_cast<device bfloat *>(pages + ulong(page) * OMNI_PAGE_BYTES) +
         ulong(kv_head) * kPageTokens * kHeadDim;
}
inline device bfloat *omni_page_values(device uchar *pages, uint page, uint kv_head) {
  return reinterpret_cast<device bfloat *>(pages + ulong(page) * OMNI_PAGE_BYTES + OMNI_PAGE_KEY_BYTES) +
         ulong(kv_head) * kPageTokens * kHeadDim;
}

// One page of one KV head against M query vectors held in threadgroup
// memory, by one simdgroup: lane l reads 16 bytes (8 dims) of token
// 2 i + l / 16 in load i (16 loads cover the page's 32 x 128 keys, each
// one contiguous 512-byte span), so a token's score is its 16 lanes'
// partial dots reduced by four xor shuffles; the probabilities then weight
// the values read the same way, lane l accumulating dims 8 (l % 16) .. +7
// of its tokens, the two token parities added at the end. The running
// softmax state (max, sum) and the lane's 8-dim accumulators carry across
// pages.
template <uint M>
struct OmniPageState {
  float running_max[M], running_sum[M], acc[M][8];
  inline void clear() {
    for (uint m = 0; m < M; ++m) {
      running_max[m] = -INFINITY;
      running_sum[m] = 0.0f;
      for (uint j = 0; j < 8; ++j) acc[m][j] = 0.0f;
    }
  }
};

inline void omni_unpack_bf16x8(uint4 packed, thread float (&out)[8]) {
  out[0] = as_type<float>(packed.x << 16);
  out[1] = as_type<float>(packed.x & 0xffff0000u);
  out[2] = as_type<float>(packed.y << 16);
  out[3] = as_type<float>(packed.y & 0xffff0000u);
  out[4] = as_type<float>(packed.z << 16);
  out[5] = as_type<float>(packed.z & 0xffff0000u);
  out[6] = as_type<float>(packed.w << 16);
  out[7] = as_type<float>(packed.w & 0xffff0000u);
}

// `visible_of(m)` is the number of tokens query vector m may see.
template <uint M, class Visible>
inline void omni_attention_page(thread OmniPageState<M> &state, device const bfloat *keys,
                                device const bfloat *values, threadgroup const float (*query)[OMNI_DECODER_HEAD_DIM],
                                uint first_token, uint lane, Visible visible_of) {
  constexpr uint D = kHeadDim;
  const uint parity = lane / 16, dim0 = (lane % 16) * 8;
  // Scores: load i holds token 2 i + parity.
  float score[M][16];
  for (uint i = 0; i < 16; ++i) {
    const uint4 packed = *reinterpret_cast<device const uint4 *>(keys + (2 * i + parity) * D + dim0);
    float k[8];
    omni_unpack_bf16x8(packed, k);
#pragma unroll
    for (uint m = 0; m < M; ++m) {
      float dot = 0.0f;
#pragma unroll
      for (uint j = 0; j < 8; ++j) dot += query[m][dim0 + j] * k[j];
      dot += simd_shuffle_xor(dot, 8);
      dot += simd_shuffle_xor(dot, 4);
      dot += simd_shuffle_xor(dot, 2);
      dot += simd_shuffle_xor(dot, 1);
      score[m][i] = dot;
    }
  }
#pragma unroll
  for (uint m = 0; m < M; ++m) {
    const uint visible = visible_of(m);
    float lane_max = -INFINITY;
    for (uint i = 0; i < 16; ++i) {
      const uint token = first_token + 2 * i + parity;
      score[m][i] = token < visible ? score[m][i] * kScale : -INFINITY;
      lane_max = max(lane_max, score[m][i]);
    }
    const float page_max = simd_max(lane_max);
    const float next_max = max(state.running_max[m], page_max);
    const float scale = state.running_max[m] == -INFINITY ? 0.0f : fast::exp(state.running_max[m] - next_max);
    float lane_sum = 0.0f;
    for (uint i = 0; i < 16; ++i) {
      const float probability = score[m][i] == -INFINITY ? 0.0f : fast::exp(score[m][i] - next_max);
      score[m][i] = probability;
      lane_sum += probability;
    }
    // Every token's probability sits in its 16 lanes: count one of them.
    state.running_sum[m] = state.running_sum[m] * scale + simd_sum(lane % 16 == 0 ? lane_sum : 0.0f);
    state.running_max[m] = next_max;
#pragma unroll
    for (uint j = 0; j < 8; ++j) state.acc[m][j] *= scale;
  }
  // Values, read as the keys were: load i holds token 2 i + parity.
  for (uint i = 0; i < 16; ++i) {
    const uint4 packed = *reinterpret_cast<device const uint4 *>(values + (2 * i + parity) * D + dim0);
    float v[8];
    omni_unpack_bf16x8(packed, v);
#pragma unroll
    for (uint m = 0; m < M; ++m) {
      const float weight = score[m][i];
#pragma unroll
      for (uint j = 0; j < 8; ++j) state.acc[m][j] += weight * v[j];
    }
  }
}

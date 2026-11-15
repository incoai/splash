#pragma once

#include "metal/abi/PagedAttention.h"
#include "metal/kernels/common/kv_extent.h"
#include "metal/kernels/common/kv_paging.h"
#include <metal_stdlib>

using namespace metal;

// Writes one row's key and value of one KV head into the row's page slot,
// through the slot's page entry (splash_store_page): a threadgroup of HeadDim
// threads, each one dimension of both. The slot is the row's own, at the
// sequence's committed tokens plus the row: a decode step
// writes all its rows, a lane's at the slots of their rows of its block; its
// commit moves the accepted path's rows to the slots of their depths
// (splash_path_commit_phase, paged_attention_store.metal), and the rows past
// them stay unreachable until the next command starting at the same logical
// position overwrites them. INT8 pages take a scale per row and head, the
// largest magnitude of the head's values over 127, and codes of the values
// over it rounded to nearest within +-127 (all zero for a zero row); BF16
// pages take the values' bits. Slots are addressed inside the head's slab of
// the page: the page index functions at page zero and head zero.

// The scratch of a row's scale reductions, by CacheElement: a maximum per
// simdgroup of the HeadDim threads for the keys and then the values, for
// INT8; BF16 has none.
#define SPLASH_STORE_MAXIMA_int8_t(HeadDim) threadgroup float maxima[2 * ((HeadDim) / 32)];
#define SPLASH_STORE_MAXIMA_bfloat(HeadDim) threadgroup float *maxima = nullptr;

// The largest magnitude of each of the threadgroup's two values, keys' first
// in `maxima`, values' after them: every thread's, through a maximum per
// simdgroup and then over the simdgroups'. A maximum is exact, so its order
// is any.
template <uint HeadDim>
inline float2 splash_store_maxima(float key, float value, threadgroup float *maxima, uint simd_lane,
                                  uint simd_group) {
  constexpr uint Groups = HeadDim / 32;
  const float key_maximum = simd_max(abs(key)), value_maximum = simd_max(abs(value));
  if (simd_lane == 0) {
    maxima[simd_group] = key_maximum;
    maxima[Groups + simd_group] = value_maximum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float2 maximum = 0.0f;
#pragma unroll
  for (uint group = 0; group < Groups; ++group)
    maximum = max(maximum, float2(maxima[group], maxima[Groups + group]));
  return maximum;
}

// The INT8 code of `value` at a row's largest magnitude `maximum`.
inline char splash_store_code(float value, float maximum) {
  return maximum == 0.0f ? char(0) : char(clamp(int(rint(value * 127.0f / maximum)), -127, 127));
}

// The page entry of a sequence's row, which a store reads before its row's
// values are ready: the slot's place follows the parameters alone.
inline SplashKvPage splash_store_page(device const SplashKvPage *page_table, constant SplashAttentionSequence &sequence,
                                      uint row) {
  return page_table[(sequence.committed_tokens + row) / SplashKvPageTokens];
}

template <uint KVHeads, uint HeadDim, typename CacheElement, bool ValuesTokenMajor>
__attribute__((always_inline)) inline void splash_store_kv_slot(
    SplashKvPage page, constant SplashAttentionSequence &sequence, SplashKvLayer kv, threadgroup float *maxima,
    uint head, uint row, bfloat key, bfloat value, uint dimension, uint simd_lane, uint simd_group) {
  const SplashKvPageTensors<CacheElement> slab =
      SplashKvAddressing<KVHeads, HeadDim, CacheElement>(kv, head).page(page);
  const uint page_token = (sequence.committed_tokens + row) % SplashKvPageTokens;
  const ulong key_index = splash_kv_key_index<KVHeads, HeadDim>(0, 0, page_token, dimension);
  const ulong value_index = splash_kv_value_index<KVHeads, HeadDim, ValuesTokenMajor>(0, 0, page_token, dimension);
  if constexpr (is_same<CacheElement, bfloat>::value) {
    slab.keys[key_index] = key;
    slab.values[value_index] = value;
  } else {
    const float2 maximum = splash_store_maxima<HeadDim>(float(key), float(value), maxima, simd_lane, simd_group);
    slab.keys[key_index] = splash_store_code(float(key), maximum.x);
    slab.values[value_index] = splash_store_code(float(value), maximum.y);
    if (dimension == 0) {
      const ulong scale_index = splash_q8_scale_index<KVHeads>(0, 0, page_token);
      slab.key_scales[scale_index] = maximum.x == 0.0f ? 0.0f : maximum.x / 127.0f;
      slab.value_scales[scale_index] = maximum.y == 0.0f ? 0.0f : maximum.y / 127.0f;
    }
  }
}

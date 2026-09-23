#pragma once

#include "metal/abi/PagedAttention.h"
#include "metal/kernels/common/q8_paging.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

constant uint SplashChunkMaximumRows = SPLASH_PREFILL_TOKEN_BUDGET;
constant uint SplashChunkMaximumPhysicalTokens =
    SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS;

inline bool
splash_chunk_contract_valid(constant SplashChunkedPrefillParams &params) {
  uint required_pages = (params.committed_tokens + params.chunk_tokens +
                         SplashQ8PageTokens - 1) /
                        SplashQ8PageTokens;
  return params.committed_tokens <= SplashChunkMaximumPhysicalTokens &&
         params.chunk_tokens > 0 &&
         params.chunk_tokens <= SplashChunkMaximumRows &&
         params.committed_tokens + params.chunk_tokens <=
             SplashChunkMaximumPhysicalTokens &&
         params.chunk_stride >= params.chunk_tokens &&
         params.chunk_stride <= SplashChunkMaximumRows &&
         params.chunk_stride % SPLASH_TARGET_KV_BLOCK_TOKENS == 0 &&
         params.page_table_entries >= required_pages &&
         params.physical_page_count > 0 && params.reserved0 == 0 &&
         params.reserved1 == 0 && params.reserved2 == 0;
}

inline ulong splash_current_key_index(uint stride, uint head, uint token,
                                        uint dimension) {
  return (ulong(head) * stride + token) * SplashQ8HeadDimension + dimension;
}

inline ulong splash_current_value_index(uint stride, uint head, uint token,
                                          uint dimension) {
  return (ulong(head) * SplashQ8HeadDimension + dimension) * stride + token;
}

// One lane per dimension stores a current row in its final page slot.
// INT8 derives a per-row scale; BF16 copies the original bits.
template <uint KVHeads, typename CacheElement>
__attribute__((always_inline)) inline void splash_store_kv_row(
    device const bfloat *chunk_keys, device const bfloat *chunk_values,
    device CacheElement *cache_keys, device float *key_scales_buffer, device CacheElement *cache_values,
    device float *value_scales_buffer, device const uint *page_table,
    constant SplashChunkedPrefillParams &params, threadgroup float *maxima,
    bool value_tensor, uint head, uint chunk_token, uint dimension,
    uint simd_lane, uint simd_group) {
  uint logical_token = params.committed_tokens + chunk_token;
  uint physical_page = page_table[logical_token / SplashQ8PageTokens];
  if (physical_page >= params.physical_page_count)
    return;
  uint page_token = logical_token % SplashQ8PageTokens;
  ulong source_index =
      value_tensor ? splash_current_value_index(params.chunk_stride, head,
                                                  chunk_token, dimension)
                   : splash_current_key_index(params.chunk_stride, head,
                                                chunk_token, dimension);
  bfloat source = value_tensor ? chunk_values[source_index] : chunk_keys[source_index];

  if constexpr (is_same<CacheElement, bfloat>::value) {
    if (value_tensor)
      cache_values[splash_q8_value_index<KVHeads>(physical_page, head, page_token,
                                               dimension)] = source;
    else
      cache_keys[splash_q8_key_index<KVHeads>(physical_page, head, page_token,
                                           dimension)] = source;
    return;
  } else {

    float value = float(source);
    float local_maximum = simd_max(abs(value));
    if (simd_lane == 0)
      maxima[simd_group] = local_maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
      for (uint offset = 4; offset != 0; offset >>= 1) {
        if (simd_lane < offset)
          maxima[simd_lane] = max(maxima[simd_lane], maxima[simd_lane + offset]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum = maxima[0];
    float scale = maximum == 0.0f ? 0.0f : maximum / 127.0f;
    int quantized = maximum == 0.0f
                        ? 0
                        : clamp(int(rint(value * 127.0f / maximum)), -127, 127);

    if (value_tensor) {
      cache_values[splash_q8_value_index<KVHeads>(physical_page, head, page_token,
                                                dimension)] = char(quantized);
      if (dimension == 0)
        value_scales_buffer[splash_q8_scale_index<KVHeads>(physical_page, head,
                                                         page_token)] = scale;
    } else {
      cache_keys[splash_q8_key_index<KVHeads>(physical_page, head, page_token,
                                            dimension)] = char(quantized);
      if (dimension == 0)
        key_scales_buffer[splash_q8_scale_index<KVHeads>(physical_page, head,
                                                       page_token)] = scale;
    }
  }
}

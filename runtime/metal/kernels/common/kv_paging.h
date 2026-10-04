#pragma once

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/KvExtent.h"
#include <metal_stdlib>

using namespace metal;

// KV page geometry of both formats, shared by the attention and store
// kernels. A page holds SplashKvPageTokens tokens of every KV head: keys
// token-major, values dimension-major. INT8 adds one fp32 scale per (KV head,
// token) for each tensor.
constant uint SplashKvPageTokens = SPLASH_TARGET_KV_BLOCK_TOKENS;
constant uint SplashKvHeadDimension = SPLASH_KV_HEAD_DIMENSION;

// An element's index in a layer's region of one tensor: its page's slab,
// then its place in the slab (abi/KvExtent.h).
template <uint KVHeads>
inline ulong splash_kv_key_index(uint page, uint head, uint token,
                                   uint dimension) {
  constexpr ulong ElementsPerPage =
      ulong(KVHeads) * SplashKvPageTokens * SplashKvHeadDimension;
  return ulong(page) * ElementsPerPage +
         splash_kv_key_element(head, token, dimension);
}

template <uint KVHeads>
inline ulong splash_kv_value_index(uint page, uint head, uint token,
                                     uint dimension) {
  constexpr ulong ElementsPerPage =
      ulong(KVHeads) * SplashKvPageTokens * SplashKvHeadDimension;
  return ulong(page) * ElementsPerPage +
         splash_kv_value_element(head, token, dimension);
}

template <uint KVHeads>
inline ulong splash_q8_scale_index(uint page, uint head, uint token) {
  constexpr ulong ScalesPerPage = ulong(KVHeads) * SplashKvPageTokens;
  return ulong(page) * ScalesPerPage + splash_kv_scale_element(head, token);
}

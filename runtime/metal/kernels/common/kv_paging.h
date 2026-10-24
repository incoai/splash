#pragma once

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/KvExtent.h"
#include <metal_stdlib>

using namespace metal;

// KV page geometry of both formats, shared by the attention and store
// kernels. A page holds SplashKvPageTokens tokens of every KV head of HeadDim
// dimensions, a value each kernel is compiled for: keys token-major, values
// dimension-major, or token-major like the keys where a layout's are
// (ValuesTokenMajor, kv::Layout::valuesTokenMajor): a block of a head's
// tokens then holds its values contiguously. INT8 adds one fp32 scale per
// (KV head, token) for each tensor.
constant uint SplashKvPageTokens = SPLASH_TARGET_KV_BLOCK_TOKENS;

// An element's index in a layer's region of one tensor: its page's slab,
// then its place in the slab (abi/KvExtent.h).
template <uint KVHeads, uint HeadDim>
inline ulong splash_kv_key_index(uint page, uint head, uint token,
                                   uint dimension) {
  constexpr ulong ElementsPerPage = ulong(KVHeads) * SplashKvPageTokens * HeadDim;
  return ulong(page) * ElementsPerPage +
         splash_kv_key_element(head, token, dimension, HeadDim);
}

template <uint KVHeads, uint HeadDim, bool ValuesTokenMajor = false>
inline ulong splash_kv_value_index(uint page, uint head, uint token,
                                     uint dimension) {
  constexpr ulong ElementsPerPage = ulong(KVHeads) * SplashKvPageTokens * HeadDim;
  return ulong(page) * ElementsPerPage +
         (ValuesTokenMajor ? splash_kv_key_element(head, token, dimension, HeadDim)
                           : splash_kv_value_element(head, token, dimension, HeadDim));
}

template <uint KVHeads>
inline ulong splash_q8_scale_index(uint page, uint head, uint token) {
  constexpr ulong ScalesPerPage = ulong(KVHeads) * SplashKvPageTokens;
  return ulong(page) * ScalesPerPage + splash_kv_scale_element(head, token);
}

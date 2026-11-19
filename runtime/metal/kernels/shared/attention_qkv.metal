#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/attention_qkv_prepare.h"

// The q/k norms and RoPE of a layer's packed projection rows, which stage
// its queries and store its keys and values in their pages, over the rows of
// a dispatch's sequences (SplashAttentionParams): a threadgroup per (head,
// row, sequence), each sequence's rows from the command's layout (`layout`,
// a SplashLaneRows a sequence).

// W: the q/k norm weights' stored type (float: a GGUF's F32 norms, _f32);
// CacheElement and ValuesTokenMajor: the layer's pages' (int8_t for q8_, and
// token-major values for the selected blocks' readers); HeadDim, RotaryPairs
// and Gated: full_qkv_storage_phase's. A sequence's page table is its binding
// among the four (SPLASH_LANE_BINDING).
#define ATTENTION_QKV(Name, QHeads, KHeads, HeadDim, RotaryPairs, Gated, W, CacheElement, ValuesTokenMajor)     \
  kernel void Name(                                                                                          \
      device const bfloat *qkv [[buffer(0)]],                                                                \
      device const W *q_norm [[buffer(1)]],                                                                  \
      device const W *k_norm [[buffer(2)]],                                                                  \
      device const float *rope_cos [[buffer(3)]],                                                            \
      device const float *rope_sin [[buffer(4)]],                                                            \
      device bfloat *queries [[buffer(5)]],                                                                  \
      device const SplashKvPage *page_table0 [[buffer(6)]],                                                  \
      device const SplashKvPage *page_table1 [[buffer(7)]],                                                  \
      device const SplashKvPage *page_table2 [[buffer(8)]],                                                  \
      device const SplashKvPage *page_table3 [[buffer(9)]],                                                  \
      device const SplashLaneRows *layout [[buffer(10)]],                                                    \
      constant SplashAttentionParams &params [[buffer(11)]],                                                 \
      uint3 group [[threadgroup_position_in_grid]],                                                          \
      uint thread_index [[thread_index_in_threadgroup]],                                                     \
      uint lane [[thread_index_in_simdgroup]],                                                               \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                                                  \
    threadgroup float reductions[HeadDim / 32];                                                              \
    threadgroup bfloat normalized[HeadDim];                                                                  \
    SPLASH_STORE_MAXIMA_##CacheElement(HeadDim)                                                              \
    full_qkv_storage_phase<QHeads, KHeads, HeadDim, RotaryPairs, Gated, W, CacheElement, ValuesTokenMajor>(  \
        qkv, q_norm, k_norm, rope_cos, rope_sin, queries,                                                    \
        SPLASH_LANE_BINDING(group.z, page_table0, page_table1, page_table2, page_table3), layout, params,     \
        reductions, normalized, maxima, group, thread_index, lane, simd_group);                              \
  }

// A layer without RoPE (NoPE) binds no tables.
#define ATTENTION_QKV_NOPE(Name, QHeads, KHeads, HeadDim, Gated, W, CacheElement, ValuesTokenMajor)            \
  kernel void Name(                                                                                          \
      device const bfloat *qkv [[buffer(0)]],                                                                \
      device const W *q_norm [[buffer(1)]],                                                                  \
      device const W *k_norm [[buffer(2)]],                                                                  \
      device bfloat *queries [[buffer(3)]],                                                                  \
      device const SplashKvPage *page_table0 [[buffer(4)]],                                                  \
      device const SplashKvPage *page_table1 [[buffer(5)]],                                                  \
      device const SplashKvPage *page_table2 [[buffer(6)]],                                                  \
      device const SplashKvPage *page_table3 [[buffer(7)]],                                                  \
      device const SplashLaneRows *layout [[buffer(8)]],                                                     \
      constant SplashAttentionParams &params [[buffer(9)]],                                                  \
      uint3 group [[threadgroup_position_in_grid]],                                                          \
      uint thread_index [[thread_index_in_threadgroup]],                                                     \
      uint lane [[thread_index_in_simdgroup]],                                                               \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                                                  \
    threadgroup float reductions[HeadDim / 32];                                                              \
    threadgroup bfloat normalized[HeadDim];                                                                  \
    SPLASH_STORE_MAXIMA_##CacheElement(HeadDim)                                                              \
    full_qkv_storage_phase<QHeads, KHeads, HeadDim, 0, Gated, W, CacheElement, ValuesTokenMajor>(            \
        qkv, q_norm, k_norm, nullptr, nullptr, queries,                                                      \
        SPLASH_LANE_BINDING(group.z, page_table0, page_table1, page_table2, page_table3), layout, params,     \
        reductions, normalized, maxima, group, thread_index, lane, simd_group);                              \
  }
// Each geometry's instances for its pages' formats and the norm types its
// families store: bf16 (an MLX target's) and F32, or F32 alone for
// Qwen3.8-Flash-Next's (_kv2_g12, whose pages hold token-major values), which
// loads from GGUFs only, and Flint's (_kv4_g8_d128), whose MLX images widen its
// norms to the F32 its GGUFs hold (model/SafetensorsImage.hpp float32Norms).
#define ATTENTION_QKV_FORMATS(Suffix, QHeads, KHeads, HeadDim, RotaryPairs, Gated, W, ValuesTokenMajor)         \
  ATTENTION_QKV(attention_q8_qkv##Suffix, QHeads, KHeads, HeadDim, RotaryPairs, Gated, W, int8_t,               \
                ValuesTokenMajor)                                                                            \
  ATTENTION_QKV(attention_bf16_qkv##Suffix, QHeads, KHeads, HeadDim, RotaryPairs, Gated, W, bfloat,             \
                ValuesTokenMajor)
ATTENTION_QKV_FORMATS(, 24, 4, 256, 32, true, bfloat, false)
ATTENTION_QKV_FORMATS(_kv2_g8, 16, 2, 256, 32, true, bfloat, false)
ATTENTION_QKV_FORMATS(_f32, 24, 4, 256, 32, true, float, false)
ATTENTION_QKV_FORMATS(_kv2_g8_f32, 16, 2, 256, 32, true, float, false)
ATTENTION_QKV_FORMATS(_kv2_g12_f32, 24, 2, 256, 32, true, float, true)
ATTENTION_QKV_FORMATS(_kv2_g16, 32, 2, 256, 32, true, bfloat, false)
ATTENTION_QKV_FORMATS(_kv2_g16_f32, 32, 2, 256, 32, true, float, false)
// Four KV heads of 128 dimensions without an output gate, their whole heads
// rotated (64 pairs) or unrotated.
ATTENTION_QKV_FORMATS(_kv4_g8_d128_f32, 32, 4, 128, 64, false, float, false)
ATTENTION_QKV_NOPE(attention_q8_qkv_kv4_g8_d128_nope_f32, 32, 4, 128, false, float, int8_t, false)
ATTENTION_QKV_NOPE(attention_bf16_qkv_kv4_g8_d128_nope_f32, 32, 4, 128, false, float, bfloat, false)
#undef ATTENTION_QKV_FORMATS
#undef ATTENTION_QKV_NOPE
#undef ATTENTION_QKV

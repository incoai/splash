#include "metal/abi/KernelABI.h"
#include "metal/abi/ShortConvolution.h"
#include "metal/kernels/common/attention_qkv_prepare.h"
#include "metal/kernels/common/verify_tree.h"

// The q/k norms and RoPE of a layer's packed projection rows, which stage
// its queries and store its keys and values in their pages, over the rows of
// a dispatch's sequences (SplashAttentionParams): a threadgroup per (head,
// row, sequence), each sequence's rows from the command's layout (`layout`,
// a SplashLaneRows a sequence).

// W: the q/k norm weights' stored type (float: a GGUF's F32 norms, _f32);
// CacheElement and ValuesTokenMajor: the layer's pages' (int8_t for q8_, and
// token-major values for the selected blocks' readers); HeadDim, RotaryPairs
// and Gated: full_qkv_storage_phase's. A sequence's page table is its binding
// among the eight (SPLASH_LANE_BINDING).
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
      device const SplashKvPage *page_table3 [[buffer(9)]], device const SplashKvPage *page_table4 [[buffer(10)]], device const SplashKvPage *page_table5 [[buffer(11)]], device const SplashKvPage *page_table6 [[buffer(12)]], device const SplashKvPage *page_table7 [[buffer(13)]],                                                  \
      device const SplashLaneRows *layout [[buffer(14)]],                                                    \
      constant SplashAttentionParams &params [[buffer(15)]],                                                 \
      uint3 group [[threadgroup_position_in_grid]],                                                          \
      uint thread_index [[thread_index_in_threadgroup]],                                                     \
      uint lane [[thread_index_in_simdgroup]],                                                               \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                                                  \
    threadgroup float reductions[HeadDim / 32];                                                              \
    threadgroup bfloat normalized[HeadDim];                                                                  \
    SPLASH_STORE_MAXIMA_##CacheElement(HeadDim)                                                              \
    full_qkv_storage_phase<QHeads, KHeads, HeadDim, RotaryPairs, Gated, W, CacheElement, ValuesTokenMajor>(  \
        qkv, q_norm, k_norm, rope_cos, rope_sin, queries,                                                    \
        SPLASH_LANE_BINDING(group.z, page_table0, page_table1, page_table2, page_table3, page_table4, page_table5, page_table6, page_table7), layout, params,     \
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
      device const SplashKvPage *page_table3 [[buffer(7)]], device const SplashKvPage *page_table4 [[buffer(8)]], device const SplashKvPage *page_table5 [[buffer(9)]], device const SplashKvPage *page_table6 [[buffer(10)]], device const SplashKvPage *page_table7 [[buffer(11)]],                                                  \
      device const SplashLaneRows *layout [[buffer(12)]],                                                     \
      constant SplashAttentionParams &params [[buffer(13)]],                                                  \
      uint3 group [[threadgroup_position_in_grid]],                                                          \
      uint thread_index [[thread_index_in_threadgroup]],                                                     \
      uint lane [[thread_index_in_simdgroup]],                                                               \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                                                  \
    threadgroup float reductions[HeadDim / 32];                                                              \
    threadgroup bfloat normalized[HeadDim];                                                                  \
    SPLASH_STORE_MAXIMA_##CacheElement(HeadDim)                                                              \
    full_qkv_storage_phase<QHeads, KHeads, HeadDim, 0, Gated, W, CacheElement, ValuesTokenMajor>(            \
        qkv, q_norm, k_norm, nullptr, nullptr, queries,                                                      \
        SPLASH_LANE_BINDING(group.z, page_table0, page_table1, page_table2, page_table3, page_table4, page_table5, page_table6, page_table7), layout, params,     \
        reductions, normalized, maxima, group, thread_index, lane, simd_group);                              \
  }
// Each geometry's instances for its pages' formats and the norm types its
// families load: bf16 (an MLX target's) and F32, or F32 alone for
// Qwen3.8-Flash-Next's (_kv2_g12, whose pages hold token-major values) and
// Flint's (_kv4_g8_d128), whose MLX images widen their norms to the F32 their
// GGUFs hold (model/SafetensorsImage.hpp float32Norms).
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

// A relative geometry's prepare (relative_qkv_storage_phase) of a layer's
// query side rows and its keys and values after their convolutions, with
// its F32 query and relative-state scales, staging its queries and writing
// its relative states; a position-scaled layer's (_scaled) multiplies both by
// tau. A decode step binds its tree rows, whose depths place its rows; a
// prompt binds another buffer of the dispatch in their place, of which it
// reads nothing.
#define ATTENTION_RELATIVE_QKV(Name, QHeads, KHeads, HeadDim, PositionScaled, CacheElement)                  \
  kernel void Name(                                                                                          \
      device const float *query_side [[buffer(0)]],                                                          \
      device const float *keys [[buffer(1)]],                                                                \
      device const float *values [[buffer(2)]],                                                              \
      device const float *query_scale [[buffer(3)]],                                                         \
      device const float *relative_scale [[buffer(4)]],                                                      \
      device bfloat *queries [[buffer(5)]],                                                                  \
      device float *relative [[buffer(6)]],                                                                  \
      device const SplashKvPage *page_table0 [[buffer(7)]],                                                  \
      device const SplashKvPage *page_table1 [[buffer(8)]],                                                  \
      device const SplashKvPage *page_table2 [[buffer(9)]],                                                  \
      device const SplashKvPage *page_table3 [[buffer(10)]],                                                 \
      device const SplashKvPage *page_table4 [[buffer(11)]],                                                 \
      device const SplashKvPage *page_table5 [[buffer(12)]],                                                 \
      device const SplashKvPage *page_table6 [[buffer(13)]],                                                 \
      device const SplashKvPage *page_table7 [[buffer(14)]],                                                 \
      device const SplashTreeRow *tree [[buffer(15)]],                                                       \
      device const SplashLaneRows *layout [[buffer(16)]],                                                    \
      constant SplashAttentionParams &params [[buffer(17)]],                                                 \
      uint3 group [[threadgroup_position_in_grid]],                                                          \
      uint thread_index [[thread_index_in_threadgroup]],                                                     \
      uint lane [[thread_index_in_simdgroup]],                                                               \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                                                  \
    threadgroup float2 partials[HeadDim / 32];                                                               \
    SPLASH_STORE_MAXIMA_##CacheElement(HeadDim)                                                              \
    const auto key_value = [&](ulong packed, uint channel, uint) {                                           \
      return float2(keys[packed * (KHeads * HeadDim) + channel], values[packed * (KHeads * HeadDim) + channel]); \
    };                                                                                                       \
    relative_qkv_storage_phase<QHeads, KHeads, HeadDim, PositionScaled, CacheElement>(                       \
        query_side, key_value, query_scale, relative_scale, queries, relative,                               \
        SPLASH_LANE_BINDING(group.z, page_table0, page_table1, page_table2, page_table3, page_table4,        \
                            page_table5, page_table6, page_table7),                                          \
        tree, layout, params, partials, maxima, group, thread_index, lane, simd_group);                      \
  }
#define ATTENTION_RELATIVE_QKV_FORMATS(Suffix, QHeads, KHeads, HeadDim)                                      \
  ATTENTION_RELATIVE_QKV(attention_q8_qkv##Suffix, QHeads, KHeads, HeadDim, false, int8_t)                   \
  ATTENTION_RELATIVE_QKV(attention_bf16_qkv##Suffix, QHeads, KHeads, HeadDim, false, bfloat)                 \
  ATTENTION_RELATIVE_QKV(attention_q8_qkv##Suffix##_scaled, QHeads, KHeads, HeadDim, true, int8_t)           \
  ATTENTION_RELATIVE_QKV(attention_bf16_qkv##Suffix##_scaled, QHeads, KHeads, HeadDim, true, bfloat)
// Eight KV heads of 128 dimensions under 24 query heads (Inkling 2's).
ATTENTION_RELATIVE_QKV_FORMATS(_kv8_g3_d128, 24, 8, 128)
#undef ATTENTION_RELATIVE_QKV_FORMATS
#undef ATTENTION_RELATIVE_QKV

// Channel c of a verify row's key or value convolution, the arithmetic of
// short_convolution (metal/abi/ShortConvolution.h, a static or dynamic
// convolution over F32 inputs with a BF16 history): the row's own input from
// its side, and the three before it along its path, which come from the
// block's rows or, before the block, from the lane's history, as a history
// keeps them. The row's input goes to the tape at its packed row for the
// commit. `offset` is the convolution's first channel in a history or tape
// row: 0 for the keys, value_offset for the values.
inline float relative_convolved(device const float *side, device const float *taps, device const bfloat *history,
                                device bfloat *tape, uint c, uint offset, ulong packed, uint first,
                                SplashTreeRow node, constant SplashRelativeConvolution &p) {
  constexpr uint Taps = SHORT_CONVOLUTION_TAPS;
  constexpr int History = int(SHORT_CONVOLUTION_HISTORY);
  const auto past = [&](int t) -> float {
    return t >= 0 ? float(bfloat(side[ulong(first + uint(t)) * p.side_stride + c]))
                  : float(history[ulong(History + t) * p.history_stride + offset + c]);
  };
  const float x = side[packed * p.side_stride + c];
  device const float *dynamic =
      side + packed * p.side_stride + p.lambda_offset + (p.head_channels ? c / p.head_channels : 0) * Taps;
  float sum = 0.0f;
  for (uint j = 0; j < Taps; ++j) {
    const uint lag = Taps - 1 - j;
    const float weight = taps[ulong(c) * Taps + j] + (p.head_channels ? dynamic[lag] : 0.0f);
    const int back = int(node.depth) - int(lag);
    const float value = lag == 0 ? x : past(back >= 0 ? int(splash_path_row(node.ancestors, uint(back))) : back);
    sum += weight * value;
  }
  tape[packed * p.tape_stride + offset + c] = bfloat(x);
  return (x + sum) * p.scale;
}

// A verify step's relative prepare that convolves the KV heads' keys and
// values itself (SplashRelativeConvolutionParams) from their sides: one
// dispatch where the two convolutions and the prepare would take three
// (0.17 ms of an 8-row verify cycle); a lane binds a page table and a history.
#define ATTENTION_RELATIVE_QKV_CONV(Name, QHeads, KHeads, HeadDim, PositionScaled, CacheElement)             \
  kernel void Name(                                                                                          \
      device const float *query_side [[buffer(0)]],                                                          \
      device const float *key_side [[buffer(1)]],                                                            \
      device const float *value_side [[buffer(2)]],                                                          \
      device const float *query_scale [[buffer(3)]],                                                         \
      device const float *relative_scale [[buffer(4)]],                                                      \
      device bfloat *queries [[buffer(5)]],                                                                  \
      device float *relative [[buffer(6)]],                                                                  \
      device const SplashKvPage *page_table0 [[buffer(7)]],                                                  \
      device const SplashKvPage *page_table1 [[buffer(8)]],                                                  \
      device const SplashKvPage *page_table2 [[buffer(9)]],                                                  \
      device const SplashKvPage *page_table3 [[buffer(10)]],                                                 \
      device const SplashKvPage *page_table4 [[buffer(11)]],                                                 \
      device const SplashKvPage *page_table5 [[buffer(12)]],                                                 \
      device const SplashKvPage *page_table6 [[buffer(13)]],                                                 \
      device const SplashKvPage *page_table7 [[buffer(14)]],                                                 \
      device const SplashTreeRow *tree [[buffer(15)]],                                                       \
      device const SplashLaneRows *layout [[buffer(16)]],                                                    \
      device const float *key_taps [[buffer(17)]],                                                           \
      device const float *value_taps [[buffer(18)]],                                                         \
      device const bfloat *history0 [[buffer(19)]],                                                          \
      device const bfloat *history1 [[buffer(20)]],                                                          \
      device const bfloat *history2 [[buffer(21)]],                                                          \
      device const bfloat *history3 [[buffer(22)]],                                                          \
      device const bfloat *history4 [[buffer(23)]],                                                          \
      device const bfloat *history5 [[buffer(24)]],                                                          \
      device const bfloat *history6 [[buffer(25)]],                                                          \
      device const bfloat *history7 [[buffer(26)]],                                                          \
      device bfloat *tape [[buffer(27)]],                                                                    \
      constant SplashRelativeConvolutionParams &params [[buffer(28)]],                                       \
      uint3 group [[threadgroup_position_in_grid]],                                                          \
      uint thread_index [[thread_index_in_threadgroup]],                                                     \
      uint lane [[thread_index_in_simdgroup]],                                                               \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                                                  \
    threadgroup float2 partials[HeadDim / 32];                                                               \
    SPLASH_STORE_MAXIMA_##CacheElement(HeadDim)                                                              \
    if (group.z >= SPLASH_MAXIMUM_BATCH_WIDTH) return;                                                       \
    device const bfloat *history = SPLASH_LANE_BINDING(group.z, history0, history1, history2, history3,      \
                                                       history4, history5, history6, history7);              \
    constant SplashRelativeConvolution &p = params.convolution;                                              \
    const auto key_value = [&](ulong packed, uint channel, uint first) {                                     \
      const SplashTreeRow node = tree[packed];                                                               \
      return float2(relative_convolved(key_side, key_taps, history, tape, channel, 0, packed, first, node, p), \
                    relative_convolved(value_side, value_taps, history, tape, channel, p.value_offset, packed, \
                                       first, node, p));                                                     \
    };                                                                                                       \
    relative_qkv_storage_phase<QHeads, KHeads, HeadDim, PositionScaled, CacheElement>(                       \
        query_side, key_value, query_scale, relative_scale, queries, relative,                               \
        SPLASH_LANE_BINDING(group.z, page_table0, page_table1, page_table2, page_table3, page_table4,        \
                            page_table5, page_table6, page_table7),                                          \
        tree, layout, params.attention, partials, maxima, group, thread_index, lane, simd_group);            \
  }
#define ATTENTION_RELATIVE_QKV_CONV_FORMATS(Suffix, QHeads, KHeads, HeadDim)                                 \
  ATTENTION_RELATIVE_QKV_CONV(attention_q8_qkv_conv##Suffix, QHeads, KHeads, HeadDim, false, int8_t)         \
  ATTENTION_RELATIVE_QKV_CONV(attention_bf16_qkv_conv##Suffix, QHeads, KHeads, HeadDim, false, bfloat)       \
  ATTENTION_RELATIVE_QKV_CONV(attention_q8_qkv_conv##Suffix##_scaled, QHeads, KHeads, HeadDim, true, int8_t) \
  ATTENTION_RELATIVE_QKV_CONV(attention_bf16_qkv_conv##Suffix##_scaled, QHeads, KHeads, HeadDim, true, bfloat)
ATTENTION_RELATIVE_QKV_CONV_FORMATS(_kv8_g3_d128, 24, 8, 128)
#undef ATTENTION_RELATIVE_QKV_CONV_FORMATS
#undef ATTENTION_RELATIVE_QKV_CONV

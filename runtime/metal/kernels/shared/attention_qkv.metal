#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/attention_gate.h"
#include "metal/kernels/common/attention_qkv_prepare.h"
#include "metal/kernels/common/gguf_sgmatrix.h"

// The q/k norms and RoPE that stage a layer's packed projection rows for its
// store and attention, and the gate that makes the attention rows the
// out-projection's input rows, over the rows of a dispatch's sequences
// (SplashAttentionParams): a threadgroup per (head, row, sequence) and a
// thread per element of a (row, sequence) respectively, each sequence's rows
// from the command's layout (`layout`, a SplashLaneRows a sequence).

// W: the q/k norm weights' stored type (float: a GGUF's F32 norms, _f32);
// HeadDim, RotaryPairs and Gated: full_qkv_storage_phase's.
#define ATTENTION_QKV(Name, QHeads, KHeads, HeadDim, RotaryPairs, Gated, W)                                  \
  kernel void Name(                                                                                          \
      device const bfloat *qkv [[buffer(0)]],                                                                \
      device const W *q_norm [[buffer(1)]],                                                                  \
      device const W *k_norm [[buffer(2)]],                                                                  \
      device const float *rope_cos [[buffer(3)]],                                                            \
      device const float *rope_sin [[buffer(4)]],                                                            \
      device bfloat *queries [[buffer(5)]],                                                                  \
      device bfloat *chunk_keys [[buffer(6)]],                                                               \
      device bfloat *chunk_values [[buffer(7)]],                                                             \
      device const SplashLaneRows *layout [[buffer(8)]],                                                     \
      constant SplashAttentionParams &params [[buffer(9)]],                                                  \
      uint3 group [[threadgroup_position_in_grid]],                                                          \
      uint thread_index [[thread_index_in_threadgroup]],                                                     \
      uint lane [[thread_index_in_simdgroup]],                                                               \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                                                  \
    threadgroup float reductions[HeadDim / 32];                                                              \
    threadgroup bfloat normalized[HeadDim];                                                                  \
    full_qkv_storage_phase<QHeads, KHeads, HeadDim, RotaryPairs, Gated>(                                     \
        qkv, q_norm, k_norm, rope_cos, rope_sin, queries, chunk_keys, chunk_values, layout, params,          \
        reductions, normalized, group, thread_index, lane, simd_group);                                      \
  }

// A layer without RoPE (NoPE) binds no tables.
#define ATTENTION_QKV_NOPE(Name, QHeads, KHeads, HeadDim, Gated, W)                                          \
  kernel void Name(                                                                                          \
      device const bfloat *qkv [[buffer(0)]],                                                                \
      device const W *q_norm [[buffer(1)]],                                                                  \
      device const W *k_norm [[buffer(2)]],                                                                  \
      device bfloat *queries [[buffer(3)]],                                                                  \
      device bfloat *chunk_keys [[buffer(4)]],                                                               \
      device bfloat *chunk_values [[buffer(5)]],                                                             \
      device const SplashLaneRows *layout [[buffer(6)]],                                                     \
      constant SplashAttentionParams &params [[buffer(7)]],                                                  \
      uint3 group [[threadgroup_position_in_grid]],                                                          \
      uint thread_index [[thread_index_in_threadgroup]],                                                     \
      uint lane [[thread_index_in_simdgroup]],                                                               \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                                                  \
    threadgroup float reductions[HeadDim / 32];                                                              \
    threadgroup bfloat normalized[HeadDim];                                                                  \
    full_qkv_storage_phase<QHeads, KHeads, HeadDim, 0, Gated>(                                               \
        qkv, q_norm, k_norm, nullptr, nullptr, queries, chunk_keys, chunk_values, layout, params, reductions, \
        normalized, group, thread_index, lane, simd_group);                                                  \
  }
// Each geometry's instances for the norm types its families store: bf16 (an
// MLX target's) and F32, or F32 alone for Qwen3.8-Flash-Next's (_kv2_g12) and
// Flint's (_kv4_g8_d128), which load from GGUFs only.
ATTENTION_QKV(attention_qkv, 24, 4, 256, 32, true, bfloat)
ATTENTION_QKV(attention_qkv_kv2_g8, 16, 2, 256, 32, true, bfloat)
ATTENTION_QKV(attention_qkv_f32, 24, 4, 256, 32, true, float)
ATTENTION_QKV(attention_qkv_kv2_g8_f32, 16, 2, 256, 32, true, float)
ATTENTION_QKV(attention_qkv_kv2_g12_f32, 24, 2, 256, 32, true, float)
ATTENTION_QKV(attention_qkv_kv2_g16, 32, 2, 256, 32, true, bfloat)
ATTENTION_QKV(attention_qkv_kv2_g16_f32, 32, 2, 256, 32, true, float)
// Four KV heads of 128 dimensions without an output gate, their whole heads
// rotated (64 pairs) or unrotated.
ATTENTION_QKV(attention_qkv_kv4_g8_d128_f32, 32, 4, 128, 64, false, float)
ATTENTION_QKV_NOPE(attention_qkv_kv4_g8_d128_nope_f32, 32, 4, 128, false, float)
#undef ATTENTION_QKV_NOPE
#undef ATTENTION_QKV

// The gate and gather: group (element group, row, sequence) of
// SPLASH_ATTENTION_GATE_THREADS threads, each one element of the row's
// QHeads x HeadDim, writes the row of the dispatch's output rows
// (splash_gate_row); a row past its sequence's rows writes nothing.
constant constexpr uint SplashAttentionGateThreads = SPLASH_ATTENTION_GATE_THREADS;

template <uint QHeads, uint KHeads, uint HeadDim, bool Gated>
inline void full_attention_output_phase(device const bfloat *packed_qkv, device const bfloat *attention,
                                        device bfloat *hidden, device const SplashLaneRows *layout,
                                        constant SplashAttentionParams &params, uint3 group, uint thread_index) {
  constexpr uint Width = QHeads * HeadDim;
  static_assert(Width % SplashAttentionGateThreads == 0, "a row's elements fill whole groups");
  const SplashGateRow row = splash_gate_row(layout, params, group.y, group.z);
  if (!row.live)
    return;
  const uint element = group.x * SplashAttentionGateThreads + thread_index;
  hidden[row.output * Width + element] = full_attention_output_value<QHeads, KHeads, HeadDim, Gated>(
      packed_qkv, attention, params.sequence[group.z], group.y, row.packed, element);
}

#define ATTENTION_GATE(Name, QHeads, KHeads, HeadDim)                                                        \
  kernel void Name(device const bfloat *packed_qkv [[buffer(0)]],                                            \
                   device const bfloat *attention [[buffer(1)]],                                             \
                   device bfloat *hidden [[buffer(2)]],                                                      \
                   device const SplashLaneRows *layout [[buffer(3)]],                                        \
                   constant SplashAttentionParams &params [[buffer(4)]],                                     \
                   uint3 group [[threadgroup_position_in_grid]],                                             \
                   uint thread_index [[thread_index_in_threadgroup]]) {                                      \
    full_attention_output_phase<QHeads, KHeads, HeadDim, true>(packed_qkv, attention, hidden, layout, params, \
                                                               group, thread_index);                         \
  }
ATTENTION_GATE(attention_gate, 24, 4, 256)
ATTENTION_GATE(attention_gate_kv2_g8, 16, 2, 256)
ATTENTION_GATE(attention_gate_kv2_g12, 24, 2, 256)
ATTENTION_GATE(attention_gate_kv2_g16, 32, 2, 256)
#undef ATTENTION_GATE

// A layer without an output gate gathers its attention rows into the
// out-projection's input rows as they are.
#define ATTENTION_GATHER(Name, QHeads, KHeads, HeadDim)                                                      \
  kernel void Name(device const bfloat *attention [[buffer(0)]],                                             \
                   device bfloat *hidden [[buffer(1)]],                                                      \
                   device const SplashLaneRows *layout [[buffer(2)]],                                        \
                   constant SplashAttentionParams &params [[buffer(3)]],                                     \
                   uint3 group [[threadgroup_position_in_grid]],                                             \
                   uint thread_index [[thread_index_in_threadgroup]]) {                                      \
    full_attention_output_phase<QHeads, KHeads, HeadDim, false>(nullptr, attention, hidden, layout, params,  \
                                                                group, thread_index);                        \
  }
ATTENTION_GATHER(attention_gather_kv4_g8_d128, 32, 4, 128)
#undef ATTENTION_GATHER

// The gate's rows written as the out-projection's table too: a thread per
// two elements of a row, a group's 512 elements of one row.
template <uint QHeads, uint KHeads, uint HeadDim, bool Gated, class Layout>
inline void full_attention_output_table_phase(device const bfloat *packed_qkv, device const bfloat *attention,
                                              device bfloat *hidden, device bfloat *table, device float *sums,
                                              device const SplashLaneRows *layout,
                                              constant SplashAttentionParams &params, uint3 group,
                                              uint thread_index, uint lane) {
  constexpr uint Width = QHeads * HeadDim;
  static_assert(Width % (2 * SplashAttentionGateThreads) == 0, "a row's element pairs fill whole groups");
  const SplashGateRow row = splash_gate_row(layout, params, group.y, group.z);
  if (!row.live)
    return;
  constant SplashAttentionSequence &sequence = params.sequence[group.z];
  const uint element = 2 * (group.x * SplashAttentionGateThreads + thread_index);
  const bfloat a = full_attention_output_value<QHeads, KHeads, HeadDim, Gated>(packed_qkv, attention, sequence,
                                                                               group.y, row.packed, element);
  const bfloat b = full_attention_output_value<QHeads, KHeads, HeadDim, Gated>(packed_qkv, attention, sequence,
                                                                               group.y, row.packed, element + 1);
  hidden[row.output * Width + element] = a;
  hidden[row.output * Width + element + 1] = b;
  Layout::write(table + row.output / 8 * Width * 8, sums + row.output / 8 * Layout::sums_per_tile(Width), Width,
                element / 64, row.output % 8, lane, a, b);
}

#define ATTENTION_GATE_TABLE(Name, QHeads, KHeads, HeadDim, Gated, Layout)                                   \
  kernel void Name(device const bfloat *packed [[buffer(0)]],                                                \
                   device const bfloat *attention [[buffer(1)]],                                             \
                   device bfloat *hidden [[buffer(2)]],                                                      \
                   device bfloat *table [[buffer(3)]], device float *sums [[buffer(4)]],                     \
                   device const SplashLaneRows *layout [[buffer(5)]],                                        \
                   constant SplashAttentionParams &params [[buffer(6)]],                                     \
                   uint3 group [[threadgroup_position_in_grid]],                                             \
                   uint thread_index [[thread_index_in_threadgroup]],                                        \
                   uint lane [[thread_index_in_simdgroup]]) {                                                \
    full_attention_output_table_phase<QHeads, KHeads, HeadDim, Gated, Layout>(                               \
        packed, attention, hidden, table, sums, layout, params, group, thread_index, lane);                  \
  }
ATTENTION_GATE_TABLE(attention_gate_table16, 24, 4, 256, true, gguf_sg::Table16)
ATTENTION_GATE_TABLE(attention_gate_table16_kv2_g8, 16, 2, 256, true, gguf_sg::Table16)
ATTENTION_GATE_TABLE(attention_gate_table16_kv2_g12, 24, 2, 256, true, gguf_sg::Table16)
ATTENTION_GATE_TABLE(attention_gate_table16_kv2_g16, 32, 2, 256, true, gguf_sg::Table16)
#undef ATTENTION_GATE_TABLE

#define ATTENTION_GATHER_TABLE(Name, QHeads, KHeads, HeadDim, Layout)                                        \
  kernel void Name(device const bfloat *attention [[buffer(0)]],                                             \
                   device bfloat *hidden [[buffer(1)]],                                                      \
                   device bfloat *table [[buffer(2)]], device float *sums [[buffer(3)]],                     \
                   device const SplashLaneRows *layout [[buffer(4)]],                                        \
                   constant SplashAttentionParams &params [[buffer(5)]],                                     \
                   uint3 group [[threadgroup_position_in_grid]],                                             \
                   uint thread_index [[thread_index_in_threadgroup]],                                        \
                   uint lane [[thread_index_in_simdgroup]]) {                                                \
    full_attention_output_table_phase<QHeads, KHeads, HeadDim, false, Layout>(                               \
        nullptr, attention, hidden, table, sums, layout, params, group, thread_index, lane);                 \
  }
ATTENTION_GATHER_TABLE(attention_gather_table16_kv4_g8_d128, 32, 4, 128, gguf_sg::Table16)
#undef ATTENTION_GATHER_TABLE

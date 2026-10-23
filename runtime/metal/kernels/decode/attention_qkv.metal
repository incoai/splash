#include "metal/kernels/common/gguf_sgmatrix.h"
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/attention_gate.h"
#include "metal/kernels/common/attention_qkv_prepare.h"

template <uint QHeads, uint KHeads, uint HeadDim, uint RotaryPairs, bool Gated, class W>
inline void full_qkv_decode_phase(
    device const bfloat *qkv, device const W *q_norm,
    device const W *k_norm, device const float *rope_cos,
    device const float *rope_sin, device bfloat *queries,
    device bfloat *keys, device bfloat *values, threadgroup float *reductions,
    threadgroup bfloat *normalized, uint2 group, uint thread_index, uint lane,
    uint simd_group) {
  constexpr uint QStride = Gated ? 2 * HeadDim : HeadDim;
  constexpr uint PackedStride = QHeads * QStride + 2 * KHeads * HeadDim;
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint Stride = SPLASH_VERIFY_CHUNK_STRIDE;
  uint batch = group.y;
  constexpr ulong kv_lane_stride = ulong(KHeads) * Stride * HeadDim;
  FullPrefillParams lane_params{Rows, Stride};
  full_qkv_storage_phase<QHeads, KHeads, HeadDim, RotaryPairs, Gated>(
      qkv + ulong(batch) * Rows * PackedStride, q_norm, k_norm,
      rope_cos + ulong(batch) * Rows * RotaryPairs,
      rope_sin + ulong(batch) * Rows * RotaryPairs,
      queries + ulong(batch) * QHeads * Stride * HeadDim,
      keys + ulong(batch) * kv_lane_stride,
      values + ulong(batch) * kv_lane_stride, lane_params, reductions,
      normalized, group.x, thread_index, lane, simd_group);
}

// W: the q/k norm weights' stored type (float: a GGUF's F32 norms, _f32);
// HeadDim, RotaryPairs and Gated: full_qkv_storage_phase's.
#define VERIFY_ATTENTION_QKV(Name, QHeads, KHeads, HeadDim, RotaryPairs,      \
                             Gated, W)                                        \
  kernel void Name(                                                           \
      device const bfloat *qkv [[buffer(0)]],                                 \
      device const W *q_norm [[buffer(1)]],                                   \
      device const W *k_norm [[buffer(2)]],                                   \
      device const float *rope_cos [[buffer(3)]],                             \
      device const float *rope_sin [[buffer(4)]],                             \
      device bfloat *queries [[buffer(5)]], device bfloat *keys [[buffer(6)]], \
      device bfloat *values [[buffer(7)]],                                    \
      uint2 group [[threadgroup_position_in_grid]],                           \
      uint thread_index [[thread_index_in_threadgroup]],                      \
      uint lane [[thread_index_in_simdgroup]],                                \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                   \
    threadgroup float reductions[HeadDim / 32];                               \
    threadgroup bfloat normalized[HeadDim];                                   \
    full_qkv_decode_phase<QHeads, KHeads, HeadDim, RotaryPairs, Gated>(       \
        qkv, q_norm, k_norm, rope_cos, rope_sin, queries, keys, values,       \
        reductions, normalized, group, thread_index, lane, simd_group);       \
  }

// A layer without RoPE (NoPE) binds no tables.
#define VERIFY_ATTENTION_QKV_NOPE(Name, QHeads, KHeads, HeadDim, Gated, W)    \
  kernel void Name(                                                           \
      device const bfloat *qkv [[buffer(0)]],                                 \
      device const W *q_norm [[buffer(1)]],                                   \
      device const W *k_norm [[buffer(2)]],                                   \
      device bfloat *queries [[buffer(3)]], device bfloat *keys [[buffer(4)]], \
      device bfloat *values [[buffer(5)]],                                    \
      uint2 group [[threadgroup_position_in_grid]],                           \
      uint thread_index [[thread_index_in_threadgroup]],                      \
      uint lane [[thread_index_in_simdgroup]],                                \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                   \
    threadgroup float reductions[HeadDim / 32];                               \
    threadgroup bfloat normalized[HeadDim];                                   \
    full_qkv_decode_phase<QHeads, KHeads, HeadDim, 0, Gated>(                 \
        qkv, q_norm, k_norm, nullptr, nullptr, queries, keys, values,         \
        reductions, normalized, group, thread_index, lane, simd_group);       \
  }
VERIFY_ATTENTION_QKV(verify_attention_qkv, 24, 4, 256, 32, true, bfloat)
VERIFY_ATTENTION_QKV(verify_attention_qkv_kv2_g8, 16, 2, 256, 32, true, bfloat)
VERIFY_ATTENTION_QKV(verify_attention_qkv_f32, 24, 4, 256, 32, true, float)
VERIFY_ATTENTION_QKV(verify_attention_qkv_kv2_g8_f32, 16, 2, 256, 32, true, float)
VERIFY_ATTENTION_QKV(verify_attention_qkv_kv2_g12, 24, 2, 256, 32, true, bfloat)
VERIFY_ATTENTION_QKV(verify_attention_qkv_kv2_g12_f32, 24, 2, 256, 32, true, float)
VERIFY_ATTENTION_QKV(verify_attention_qkv_kv2_g16, 32, 2, 256, 32, true, bfloat)
VERIFY_ATTENTION_QKV(verify_attention_qkv_kv2_g16_f32, 32, 2, 256, 32, true, float)
// Four KV heads of 128 dimensions without an output gate, their whole heads
// rotated (64 pairs) or unrotated.
VERIFY_ATTENTION_QKV(verify_attention_qkv_kv4_g8_d128, 32, 4, 128, 64, false, bfloat)
VERIFY_ATTENTION_QKV(verify_attention_qkv_kv4_g8_d128_f32, 32, 4, 128, 64, false, float)
VERIFY_ATTENTION_QKV_NOPE(verify_attention_qkv_kv4_g8_d128_nope, 32, 4, 128, false, bfloat)
VERIFY_ATTENTION_QKV_NOPE(verify_attention_qkv_kv4_g8_d128_nope_f32, 32, 4, 128, false, float)
#undef VERIFY_ATTENTION_QKV_NOPE
#undef VERIFY_ATTENTION_QKV

// Element `element` of every verify lane's rows, lane by lane.
template <uint QHeads, uint KHeads, uint HeadDim, bool Gated>
inline bfloat verify_attention_output_value(device const bfloat *packed_qkv,
                                            device const bfloat *attention,
                                            uint element) {
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint per_lane = Rows * QHeads * HeadDim;
  uint batch = element / per_lane;
  return full_attention_output_value<QHeads, KHeads, HeadDim, Gated>(
      packed_qkv, attention, batch, Rows, SPLASH_VERIFY_CHUNK_STRIDE,
      element % per_lane);
}

template <uint QHeads, uint KHeads, uint HeadDim>
inline void full_attention_gate_decode_phase(
    device const bfloat *packed_qkv, device const bfloat *attention,
    device bfloat *hidden, constant FullDecodeBatchParams &params, uint index,
    uint grid_size) {
  const uint count = params.lanes * SPLASH_TARGET_VERIFY_ROWS * QHeads * HeadDim;
  for (uint element = index; element < count; element += grid_size)
    hidden[element] =
        verify_attention_output_value<QHeads, KHeads, HeadDim, true>(packed_qkv, attention, element);
}

#define VERIFY_ATTENTION_GATE(Name, QHeads, KHeads, HeadDim)                   \
  kernel void Name(device const bfloat *packed_qkv [[buffer(0)]],              \
                   device const bfloat *attention [[buffer(1)]],               \
                   device bfloat *hidden [[buffer(2)]],                        \
                   constant FullDecodeBatchParams &params [[buffer(3)]],       \
                   uint index [[thread_position_in_grid]],                     \
                   uint grid_size [[threads_per_grid]]) {                      \
    full_attention_gate_decode_phase<QHeads, KHeads, HeadDim>(                 \
        packed_qkv, attention, hidden, params, index, grid_size);              \
  }
VERIFY_ATTENTION_GATE(verify_attention_gate, 24, 4, 256)
VERIFY_ATTENTION_GATE(verify_attention_gate_kv2_g8, 16, 2, 256)
VERIFY_ATTENTION_GATE(verify_attention_gate_kv2_g12, 24, 2, 256)
VERIFY_ATTENTION_GATE(verify_attention_gate_kv2_g16, 32, 2, 256)
#undef VERIFY_ATTENTION_GATE

// A layer without an output gate gathers its attention rows into the
// out-projection's input rows as they are.
#define VERIFY_ATTENTION_GATHER(Name, QHeads, KHeads, HeadDim)                 \
  kernel void Name(device const bfloat *attention [[buffer(0)]],               \
                   device bfloat *hidden [[buffer(1)]],                        \
                   constant FullDecodeBatchParams &params [[buffer(2)]],       \
                   uint index [[thread_position_in_grid]],                     \
                   uint grid_size [[threads_per_grid]]) {                      \
    const uint count = params.lanes * SPLASH_TARGET_VERIFY_ROWS * QHeads *     \
                       HeadDim;                                                \
    for (uint element = index; element < count; element += grid_size)          \
      hidden[element] =                                                        \
          verify_attention_output_value<QHeads, KHeads, HeadDim, false>(       \
              nullptr, attention, element);                                    \
  }
VERIFY_ATTENTION_GATHER(verify_attention_gather_kv4_g8_d128, 32, 4, 128)
#undef VERIFY_ATTENTION_GATHER

#define ATTENTION_GATE_TABLE(Name, QHeads, KHeads, HeadDim, Layout) \
  kernel void Name( \
      device const bfloat *packed [[buffer(0)]], \
      device const bfloat *attention [[buffer(1)]], \
      device bfloat *hidden [[buffer(2)]], \
      device bfloat *table [[buffer(3)]], device float *sums [[buffer(4)]], \
      uint index [[thread_position_in_grid]], \
      uint lane [[thread_index_in_simdgroup]]) { \
    constexpr uint width = QHeads * HeadDim; \
    const uint element = 2 * index; \
    const bfloat a = verify_attention_output_value<QHeads, KHeads, HeadDim, true>(packed, attention, element); \
    const bfloat b = verify_attention_output_value<QHeads, KHeads, HeadDim, true>(packed, attention, element + 1); \
    hidden[element] = a; hidden[element + 1] = b; \
    const uint row = element / width; \
    Layout::write(table + ulong(row / 8) * width * 8, sums + ulong(row / 8) * Layout::sums_per_tile(width), \
                  width, (element % width) / 64, row % 8, lane, a, b); \
  }
ATTENTION_GATE_TABLE(verify_attention_gate_table16, 24, 4, 256, gguf_sg::Table16)
ATTENTION_GATE_TABLE(verify_attention_gate_table16_kv2_g8, 16, 2, 256, gguf_sg::Table16)
ATTENTION_GATE_TABLE(verify_attention_gate_table16_kv2_g12, 24, 2, 256, gguf_sg::Table16)
ATTENTION_GATE_TABLE(verify_attention_gate_table16_kv2_g16, 32, 2, 256, gguf_sg::Table16)
#undef ATTENTION_GATE_TABLE

// The gather's rows written as the out-projection's table too.
#define ATTENTION_GATHER_TABLE(Name, QHeads, KHeads, HeadDim, Layout) \
  kernel void Name( \
      device const bfloat *attention [[buffer(0)]], \
      device bfloat *hidden [[buffer(1)]], \
      device bfloat *table [[buffer(2)]], device float *sums [[buffer(3)]], \
      uint index [[thread_position_in_grid]], \
      uint lane [[thread_index_in_simdgroup]]) { \
    constexpr uint width = QHeads * HeadDim; \
    const uint element = 2 * index; \
    const bfloat a = verify_attention_output_value<QHeads, KHeads, HeadDim, false>(nullptr, attention, element); \
    const bfloat b = verify_attention_output_value<QHeads, KHeads, HeadDim, false>(nullptr, attention, element + 1); \
    hidden[element] = a; hidden[element + 1] = b; \
    const uint row = element / width; \
    Layout::write(table + ulong(row / 8) * width * 8, sums + ulong(row / 8) * Layout::sums_per_tile(width), \
                  width, (element % width) / 64, row % 8, lane, a, b); \
  }
ATTENTION_GATHER_TABLE(verify_attention_gather_table16_kv4_g8_d128, 32, 4, 128, gguf_sg::Table16)
#undef ATTENTION_GATHER_TABLE

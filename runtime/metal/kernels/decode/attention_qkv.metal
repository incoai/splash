#include "metal/kernels/common/gguf_sgmatrix.h"
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/attention_gate.h"
#include "metal/kernels/common/attention_qkv_prepare.h"

template <uint QHeads, uint KHeads, class W>
inline void full_qkv_decode_phase(
    device const bfloat *qkv, device const W *q_norm,
    device const W *k_norm, device const float *rope_cos,
    device const float *rope_sin, device bfloat *queries,
    device bfloat *keys, device bfloat *values, threadgroup float *reductions,
    threadgroup bfloat *normalized, uint2 group, uint thread_index, uint lane,
    uint simd_group) {
  constexpr uint HeadDim = 256, RotaryPairs = SPLASH_TARGET_ROPE_PAIRS, QStride = 2 * HeadDim;
  constexpr uint PackedStride = QHeads * QStride + 2 * KHeads * HeadDim;
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint Stride = SPLASH_VERIFY_CHUNK_STRIDE;
  uint batch = group.y;
  constexpr ulong kv_lane_stride = ulong(KHeads) * Stride * HeadDim;
  FullPrefillParams lane_params{Rows, Stride};
  full_qkv_storage_phase<QHeads, KHeads>(
      qkv + ulong(batch) * Rows * PackedStride, q_norm, k_norm,
      rope_cos + ulong(batch) * Rows * RotaryPairs,
      rope_sin + ulong(batch) * Rows * RotaryPairs,
      queries + ulong(batch) * QHeads * Stride * HeadDim,
      keys + ulong(batch) * kv_lane_stride,
      values + ulong(batch) * kv_lane_stride, lane_params, reductions,
      normalized, group.x, thread_index, lane, simd_group);
}

// W: the q/k norm weights' stored type (float: a GGUF's F32 norms, _f32).
#define VERIFY_ATTENTION_QKV(Name, QHeads, KHeads, W)                         \
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
    threadgroup float reductions[8];                                          \
    threadgroup bfloat normalized[256];                                       \
    full_qkv_decode_phase<QHeads, KHeads>(                                    \
        qkv, q_norm, k_norm, rope_cos, rope_sin, queries, keys, values,       \
        reductions, normalized, group, thread_index, lane, simd_group);       \
  }
VERIFY_ATTENTION_QKV(verify_attention_qkv, 24, 4, bfloat)
VERIFY_ATTENTION_QKV(verify_attention_qkv_kv2_g8, 16, 2, bfloat)
VERIFY_ATTENTION_QKV(verify_attention_qkv_f32, 24, 4, float)
VERIFY_ATTENTION_QKV(verify_attention_qkv_kv2_g8_f32, 16, 2, float)
#undef VERIFY_ATTENTION_QKV

// Element `element` of every verify lane's rows, lane by lane.
template <uint QHeads, uint KHeads>
inline bfloat verify_attention_gate_value(device const bfloat *packed_qkv,
                                          device const bfloat *attention,
                                          uint element) {
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint per_lane = Rows * QHeads * 256;
  uint batch = element / per_lane;
  return full_attention_gate_value<QHeads, KHeads>(
      packed_qkv, attention, batch, Rows, SPLASH_VERIFY_CHUNK_STRIDE,
      element % per_lane);
}

template <uint QHeads, uint KHeads>
inline void full_attention_gate_decode_phase(
    device const bfloat *packed_qkv, device const bfloat *attention,
    device bfloat *hidden, constant FullDecodeBatchParams &params, uint index,
    uint grid_size) {
  const uint count = params.lanes * SPLASH_TARGET_VERIFY_ROWS * QHeads * 256;
  for (uint element = index; element < count; element += grid_size)
    hidden[element] =
        verify_attention_gate_value<QHeads, KHeads>(packed_qkv, attention, element);
}

#define VERIFY_ATTENTION_GATE(Name, QHeads, KHeads)                            \
  kernel void Name(device const bfloat *packed_qkv [[buffer(0)]],              \
                   device const bfloat *attention [[buffer(1)]],               \
                   device bfloat *hidden [[buffer(2)]],                        \
                   constant FullDecodeBatchParams &params [[buffer(3)]],       \
                   uint index [[thread_position_in_grid]],                     \
                   uint grid_size [[threads_per_grid]]) {                      \
    full_attention_gate_decode_phase<QHeads, KHeads>(                          \
        packed_qkv, attention, hidden, params, index, grid_size);              \
  }
VERIFY_ATTENTION_GATE(verify_attention_gate, 24, 4)
VERIFY_ATTENTION_GATE(verify_attention_gate_kv2_g8, 16, 2)
#undef VERIFY_ATTENTION_GATE

#define ATTENTION_GATE_TABLE(Name, QHeads, KHeads, Layout) \
  kernel void Name( \
      device const bfloat *packed [[buffer(0)]], \
      device const bfloat *attention [[buffer(1)]], \
      device bfloat *hidden [[buffer(2)]], \
      device bfloat *table [[buffer(3)]], device float *sums [[buffer(4)]], \
      uint index [[thread_position_in_grid]], \
      uint lane [[thread_index_in_simdgroup]]) { \
    constexpr uint width = QHeads * 256; \
    const uint element = 2 * index; \
    const bfloat a = verify_attention_gate_value<QHeads, KHeads>(packed, attention, element); \
    const bfloat b = verify_attention_gate_value<QHeads, KHeads>(packed, attention, element + 1); \
    hidden[element] = a; hidden[element + 1] = b; \
    const uint row = element / width; \
    Layout::write(table + ulong(row / 8) * width * 8, sums + ulong(row / 8) * Layout::sums_per_tile(width), \
                  width, (element % width) / 64, row % 8, lane, a, b); \
  }
ATTENTION_GATE_TABLE(verify_attention_gate_table16, 24, 4, gguf_sg::Table16)
ATTENTION_GATE_TABLE(verify_attention_gate_table16_kv2_g8, 16, 2, gguf_sg::Table16)
#undef ATTENTION_GATE_TABLE

#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/attention_qkv_prepare.h"

// W: the q/k norm weights' stored type (float: a GGUF's F32 norms, _f32).
#define PREFILL_ATTENTION_QKV(Name, QHeads, KHeads, W)                        \
  kernel void Name(                                                           \
      device const bfloat *qkv [[buffer(0)]],                                 \
      device const W *q_norm [[buffer(1)]],                                   \
      device const W *k_norm [[buffer(2)]],                                   \
      device const float *rope_cos [[buffer(3)]],                             \
      device const float *rope_sin [[buffer(4)]],                             \
      device bfloat *queries [[buffer(5)]],                                   \
      device bfloat *key_cache [[buffer(6)]],                                 \
      device bfloat *value_cache [[buffer(7)]],                               \
      constant FullPrefillParams &params [[buffer(8)]],                       \
      uint task [[threadgroup_position_in_grid]],                             \
      uint thread_index [[thread_index_in_threadgroup]],                      \
      uint lane [[thread_index_in_simdgroup]],                                \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                   \
    threadgroup float reductions[8];                                          \
    threadgroup bfloat normalized[256];                                       \
    full_qkv_storage_phase<QHeads, KHeads>(                                   \
        qkv, q_norm, k_norm, rope_cos, rope_sin, queries, key_cache,          \
        value_cache, params, reductions, normalized, task, thread_index,      \
        lane, simd_group);                                                    \
  }
PREFILL_ATTENTION_QKV(prefill_attention_qkv, 24, 4, bfloat)
PREFILL_ATTENTION_QKV(prefill_attention_qkv_kv2_g8, 16, 2, bfloat)
PREFILL_ATTENTION_QKV(prefill_attention_qkv_f32, 24, 4, float)
PREFILL_ATTENTION_QKV(prefill_attention_qkv_kv2_g8_f32, 16, 2, float)
#undef PREFILL_ATTENTION_QKV

template <uint QHeads, uint KHeads>
inline void full_attention_gate_prefill_phase(
    device const bfloat *packed_qkv, device const bfloat *attention,
    device bfloat *hidden, constant FullPrefillParams &params, uint index,
    uint grid_size) {
  constexpr uint HeadDim = 256, QStride = 2 * HeadDim;
  constexpr uint PackedStride = QHeads * QStride + 2 * KHeads * HeadDim;
  constexpr uint HeadsPerKV = QHeads / KHeads;
  uint count = params.tokens * QHeads * HeadDim;
  for (uint element = index; element < count; element += grid_size) {
    uint row = element / (QHeads * HeadDim);
    uint remainder = element % (QHeads * HeadDim);
    uint query_head = remainder / HeadDim;
    uint dim = remainder % HeadDim;
    float gate = float(packed_qkv[ulong(row) * PackedStride +
                                  query_head * QStride + HeadDim + dim]);
    float sigmoid = 1.0f / (1.0f + fast::exp2(-1.44269504089f * gate));
    uint kv_head = query_head / HeadsPerKV;
    uint local_head = query_head % HeadsPerKV;
    hidden[element] = bfloat(
        float(attention[((ulong(kv_head) * params.row_stride + row) *
                             HeadsPerKV +
                         local_head) *
                            HeadDim +
                        dim]) *
        sigmoid);
  }
}

kernel void
prefill_attention_gate(device const bfloat *packed_qkv [[buffer(0)]],
                            device const bfloat *attention [[buffer(1)]],
                            device bfloat *hidden [[buffer(2)]],
                            constant FullPrefillParams &params [[buffer(3)]],
                            uint index [[thread_position_in_grid]],
                            uint grid_size [[threads_per_grid]]) {
  full_attention_gate_prefill_phase<24, 4>(
      packed_qkv, attention, hidden, params, index, grid_size);
}

kernel void prefill_attention_gate_kv2_g8(
    device const bfloat *packed_qkv [[buffer(0)]],
    device const bfloat *attention [[buffer(1)]],
    device bfloat *hidden [[buffer(2)]],
    constant FullPrefillParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  full_attention_gate_prefill_phase<16, 2>(
      packed_qkv, attention, hidden, params, index, grid_size);
}

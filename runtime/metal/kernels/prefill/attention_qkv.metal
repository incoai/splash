#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/attention_gate.h"
#include "metal/kernels/common/attention_qkv_prepare.h"

// W: the q/k norm weights' stored type (float: a GGUF's F32 norms, _f32);
// HeadDim, RotaryPairs and Gated: full_qkv_storage_phase's.
#define PREFILL_ATTENTION_QKV(Name, QHeads, KHeads, HeadDim, RotaryPairs,     \
                              Gated, W)                                       \
  kernel void Name(                                                           \
      device const bfloat *qkv [[buffer(0)]],                                 \
      device const W *q_norm [[buffer(1)]],                                   \
      device const W *k_norm [[buffer(2)]],                                   \
      device const float *rope_cos [[buffer(3)]],                             \
      device const float *rope_sin [[buffer(4)]],                             \
      device bfloat *queries [[buffer(5)]],                                   \
      device bfloat *chunk_keys [[buffer(6)]],                                \
      device bfloat *chunk_values [[buffer(7)]],                              \
      constant FullPrefillParams &params [[buffer(8)]],                       \
      uint task [[threadgroup_position_in_grid]],                             \
      uint thread_index [[thread_index_in_threadgroup]],                      \
      uint lane [[thread_index_in_simdgroup]],                                \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                   \
    threadgroup float reductions[HeadDim / 32];                               \
    threadgroup bfloat normalized[HeadDim];                                   \
    full_qkv_storage_phase<QHeads, KHeads, HeadDim, RotaryPairs, Gated>(      \
        qkv, q_norm, k_norm, rope_cos, rope_sin, queries, chunk_keys,         \
        chunk_values, params, reductions, normalized, task, thread_index,     \
        lane, simd_group);                                                    \
  }

// A layer without RoPE (NoPE) binds no tables.
#define PREFILL_ATTENTION_QKV_NOPE(Name, QHeads, KHeads, HeadDim, Gated, W)   \
  kernel void Name(                                                           \
      device const bfloat *qkv [[buffer(0)]],                                 \
      device const W *q_norm [[buffer(1)]],                                   \
      device const W *k_norm [[buffer(2)]],                                   \
      device bfloat *queries [[buffer(3)]],                                   \
      device bfloat *chunk_keys [[buffer(4)]],                                \
      device bfloat *chunk_values [[buffer(5)]],                              \
      constant FullPrefillParams &params [[buffer(6)]],                       \
      uint task [[threadgroup_position_in_grid]],                             \
      uint thread_index [[thread_index_in_threadgroup]],                      \
      uint lane [[thread_index_in_simdgroup]],                                \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                   \
    threadgroup float reductions[HeadDim / 32];                               \
    threadgroup bfloat normalized[HeadDim];                                   \
    full_qkv_storage_phase<QHeads, KHeads, HeadDim, 0, Gated>(                \
        qkv, q_norm, k_norm, nullptr, nullptr, queries, chunk_keys,           \
        chunk_values, params, reductions, normalized, task, thread_index,     \
        lane, simd_group);                                                    \
  }
PREFILL_ATTENTION_QKV(prefill_attention_qkv, 24, 4, 256, 32, true, bfloat)
PREFILL_ATTENTION_QKV(prefill_attention_qkv_kv2_g8, 16, 2, 256, 32, true, bfloat)
PREFILL_ATTENTION_QKV(prefill_attention_qkv_f32, 24, 4, 256, 32, true, float)
PREFILL_ATTENTION_QKV(prefill_attention_qkv_kv2_g8_f32, 16, 2, 256, 32, true, float)
PREFILL_ATTENTION_QKV(prefill_attention_qkv_kv2_g12, 24, 2, 256, 32, true, bfloat)
PREFILL_ATTENTION_QKV(prefill_attention_qkv_kv2_g12_f32, 24, 2, 256, 32, true, float)
PREFILL_ATTENTION_QKV(prefill_attention_qkv_kv2_g16, 32, 2, 256, 32, true, bfloat)
PREFILL_ATTENTION_QKV(prefill_attention_qkv_kv2_g16_f32, 32, 2, 256, 32, true, float)
// Four KV heads of 128 dimensions without an output gate, their whole heads
// rotated (64 pairs) or unrotated.
PREFILL_ATTENTION_QKV(prefill_attention_qkv_kv4_g8_d128, 32, 4, 128, 64, false, bfloat)
PREFILL_ATTENTION_QKV(prefill_attention_qkv_kv4_g8_d128_f32, 32, 4, 128, 64, false, float)
PREFILL_ATTENTION_QKV_NOPE(prefill_attention_qkv_kv4_g8_d128_nope, 32, 4, 128, false, bfloat)
PREFILL_ATTENTION_QKV_NOPE(prefill_attention_qkv_kv4_g8_d128_nope_f32, 32, 4, 128, false, float)
#undef PREFILL_ATTENTION_QKV_NOPE
#undef PREFILL_ATTENTION_QKV

template <uint QHeads, uint KHeads, uint HeadDim>
inline void full_attention_gate_prefill_phase(
    device const bfloat *packed_qkv, device const bfloat *attention,
    device bfloat *hidden, constant FullPrefillParams &params, uint index,
    uint grid_size) {
  uint count = params.tokens * QHeads * HeadDim;
  for (uint element = index; element < count; element += grid_size)
    hidden[element] = full_attention_output_value<QHeads, KHeads, HeadDim, true>(
        packed_qkv, attention, 0, 0, params.stride, element);
}

#define PREFILL_ATTENTION_GATE(Name, QHeads, KHeads, HeadDim)                  \
  kernel void Name(device const bfloat *packed_qkv [[buffer(0)]],              \
                   device const bfloat *attention [[buffer(1)]],               \
                   device bfloat *hidden [[buffer(2)]],                        \
                   constant FullPrefillParams &params [[buffer(3)]],           \
                   uint index [[thread_position_in_grid]],                     \
                   uint grid_size [[threads_per_grid]]) {                      \
    full_attention_gate_prefill_phase<QHeads, KHeads, HeadDim>(                \
        packed_qkv, attention, hidden, params, index, grid_size);              \
  }
PREFILL_ATTENTION_GATE(prefill_attention_gate, 24, 4, 256)
PREFILL_ATTENTION_GATE(prefill_attention_gate_kv2_g8, 16, 2, 256)
PREFILL_ATTENTION_GATE(prefill_attention_gate_kv2_g12, 24, 2, 256)
PREFILL_ATTENTION_GATE(prefill_attention_gate_kv2_g16, 32, 2, 256)
#undef PREFILL_ATTENTION_GATE

// A layer without an output gate gathers its attention rows into the
// out-projection's input rows as they are.
#define PREFILL_ATTENTION_GATHER(Name, QHeads, KHeads, HeadDim)                \
  kernel void Name(device const bfloat *attention [[buffer(0)]],               \
                   device bfloat *hidden [[buffer(1)]],                        \
                   constant FullPrefillParams &params [[buffer(2)]],           \
                   uint index [[thread_position_in_grid]],                     \
                   uint grid_size [[threads_per_grid]]) {                      \
    const uint count = params.tokens * QHeads * HeadDim;                       \
    for (uint element = index; element < count; element += grid_size)          \
      hidden[element] =                                                        \
          full_attention_output_value<QHeads, KHeads, HeadDim, false>(         \
              nullptr, attention, 0, 0, params.stride, element);               \
  }
PREFILL_ATTENTION_GATHER(prefill_attention_gather_kv4_g8_d128, 32, 4, 128)
#undef PREFILL_ATTENTION_GATHER

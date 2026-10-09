#pragma once

#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/rms_inverse.h"

// q_norm and k_norm are read in their stored type W: bfloat for an MLX
// target's, float for a GGUF's F32 norms.
template <uint QHeads, uint KHeads, class W>
inline void full_qkv_storage_phase(
    device const bfloat *qkv, device const W *q_norm,
    device const W *k_norm, device const float *rope_cos,
    device const float *rope_sin, device bfloat *queries,
    device bfloat *chunk_keys, device bfloat *chunk_values,
    FullPrefillParams params, threadgroup float *reductions,
    threadgroup bfloat *normalized, uint task, uint thread_index, uint lane,
    uint simd_group) {
  static_assert(QHeads % KHeads == 0);
  constexpr uint HeadDim = 256, RotaryPairs = SPLASH_TARGET_ROPE_PAIRS, QStride = 2 * HeadDim;
  constexpr uint PackedStride = QHeads * QStride + 2 * KHeads * HeadDim;
  constexpr uint QWidth = QHeads * QStride, KWidth = KHeads * HeadDim;
  uint query_tasks = params.tokens * QHeads;
  bool query = task < query_tasks;
  uint local_task = query ? task : task - query_tasks;
  uint heads = query ? QHeads : KHeads;
  uint row = local_task / heads;
  uint head_index = local_task % heads;
  uint position = row;
  device const bfloat *source =
      query ? qkv + ulong(row) * PackedStride + head_index * QStride
            : qkv + ulong(row) * PackedStride + QWidth + head_index * HeadDim;
  device const W *weight = query ? q_norm : k_norm;
  uint kv_head = head_index / (QHeads / KHeads);
  uint local_head = head_index % (QHeads / KHeads);
  device bfloat *destination =
      queries + ((ulong(kv_head) * params.stride + row) *
                     (QHeads / KHeads) +
                 local_head) *
                    HeadDim;
  if (!query) {
    ulong key_offset =
        (ulong(head_index) * params.stride + position) * HeadDim;
    destination = chunk_keys + key_offset;
  }

  float element = float(source[thread_index]);
  const float inverse = rms_inverse_of_sums(element * element, HeadDim,
                                            reductions, thread_index, lane,
                                            simd_group);
  normalized[thread_index] =
      bfloat(element * inverse * float(weight[thread_index]));
  if (!query) {
    ulong value_offset =
        (ulong(head_index) * HeadDim + thread_index) * params.stride +
        position;
    chunk_values[value_offset] = source[KWidth + thread_index];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < RotaryPairs) {
    float first = float(normalized[thread_index]);
    float second = float(normalized[thread_index + RotaryPairs]);
    float cosine = rope_cos[ulong(row) * RotaryPairs + thread_index];
    float sine = rope_sin[ulong(row) * RotaryPairs + thread_index];
    destination[thread_index] = bfloat(first * cosine - second * sine);
    destination[thread_index + RotaryPairs] = bfloat(second * cosine + first * sine);
  } else if (thread_index >= 2 * RotaryPairs) {
    destination[thread_index] = normalized[thread_index];
  }
}

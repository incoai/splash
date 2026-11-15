#pragma once

#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/attention_sequence.h"
#include "metal/kernels/common/rms_inverse.h"

// A threadgroup of HeadDim threads normalizes one head of one row of the
// packed projection: each query head's queries (and, Gated, its output gate
// after them), then the K and V heads; group (head, row, sequence) takes row
// `row` of the dispatch's sequence (SplashAttentionParams), read from its
// packed row (the sequence's layout entry, attention_sequence.h) and staged
// at the sequence's staging rows, and a row past the sequence's rows returns.
// q_norm and k_norm are read in their stored type W: bfloat for an MLX
// target's, float for a GGUF's F32 norms. The pairs of dimensions
// (i, i + RotaryPairs) then rotate by the packed row's RoPE table row of
// RotaryPairs pairs, the half split of the first 2 x RotaryPairs dimensions:
// Qwen3.5's first 64, or with HeadDim / 2 pairs the whole head (NeoX's).
// With no pairs (NoPE) the head is stored as normalized, and no table is read.
template <uint QHeads, uint KHeads, uint HeadDim, uint RotaryPairs, bool Gated, class W>
inline void full_qkv_storage_phase(
    device const bfloat *qkv, device const W *q_norm,
    device const W *k_norm, device const float *rope_cos,
    device const float *rope_sin, device bfloat *queries,
    device bfloat *chunk_keys, device bfloat *chunk_values,
    device const SplashLaneRows *layout, constant SplashAttentionParams &params,
    threadgroup float *reductions, threadgroup bfloat *normalized, uint3 group, uint thread_index, uint lane,
    uint simd_group) {
  static_assert(QHeads % KHeads == 0);
  static_assert(2 * RotaryPairs <= HeadDim, "the rotated pairs are dimensions of the head");
  constexpr uint QStride = Gated ? 2 * HeadDim : HeadDim;
  constexpr uint PackedStride = QHeads * QStride + 2 * KHeads * HeadDim;
  constexpr uint QWidth = QHeads * QStride, KWidth = KHeads * HeadDim;
  const uint sequence_row = group.y;
  const SplashAttentionRows rows = splash_attention_rows(layout, params, group.z);
  if (sequence_row >= rows.rows)
    return;
  constant SplashAttentionSequence &sequence = params.sequence[group.z];
  const bool query = group.x < QHeads;
  const uint head_index = query ? group.x : group.x - QHeads;
  const ulong packed = ulong(rows.first) + sequence_row;
  device const bfloat *source =
      query ? qkv + packed * PackedStride + head_index * QStride
            : qkv + packed * PackedStride + QWidth + head_index * HeadDim;
  device const W *weight = query ? q_norm : k_norm;
  uint kv_head = head_index / (QHeads / KHeads);
  uint local_head = head_index % (QHeads / KHeads);
  device bfloat *destination =
      queries + ((ulong(sequence.stage_row) + ulong(kv_head) * sequence.chunk_stride + sequence_row) *
                     (QHeads / KHeads) +
                 local_head) *
                    HeadDim;
  if (!query) {
    ulong key_offset =
        (ulong(sequence.stage_row) + ulong(head_index) * sequence.chunk_stride + sequence_row) * HeadDim;
    destination = chunk_keys + key_offset;
  }

  float element = float(source[thread_index]);
  const float inverse = rms_inverse_of_sums<HeadDim / 32>(element * element, HeadDim,
                                                          reductions, thread_index, lane,
                                                          simd_group);
  normalized[thread_index] =
      bfloat(element * inverse * float(weight[thread_index]));
  if (!query) {
    ulong value_offset = ulong(sequence.stage_row) * HeadDim +
                         (ulong(head_index) * HeadDim + thread_index) * sequence.chunk_stride + sequence_row;
    chunk_values[value_offset] = source[KWidth + thread_index];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < RotaryPairs) {
    float first = float(normalized[thread_index]);
    float second = float(normalized[thread_index + RotaryPairs]);
    float cosine = rope_cos[packed * RotaryPairs + thread_index];
    float sine = rope_sin[packed * RotaryPairs + thread_index];
    destination[thread_index] = bfloat(first * cosine - second * sine);
    destination[thread_index + RotaryPairs] = bfloat(second * cosine + first * sine);
  } else if (thread_index >= 2 * RotaryPairs) {
    destination[thread_index] = normalized[thread_index];
  }
}

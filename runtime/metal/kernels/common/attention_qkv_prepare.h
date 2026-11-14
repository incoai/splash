#pragma once

#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/attention_sequence.h"
#include "metal/kernels/common/rms_inverse.h"

// A threadgroup of HeadDim threads normalizes one head of one row of the
// packed projection: each query head's queries (and, Gated, its output gate
// after them), then the K and V heads, over the dispatch's rows
// (SplashAttentionParams), each read from its sequence's packed row and
// staged at its sequence's staging rows. q_norm and k_norm are read in their
// stored type W: bfloat for an MLX target's, float for a GGUF's F32 norms.
// The pairs of dimensions (i, i + RotaryPairs) then rotate by the packed
// row's RoPE table row of RotaryPairs pairs, the half split of the first
// 2 x RotaryPairs dimensions: Qwen3.5's first 64, or with HeadDim / 2 pairs
// the whole head (NeoX's). With no pairs (NoPE) the head is stored as
// normalized, and no table is read.
template <uint QHeads, uint KHeads, uint HeadDim, uint RotaryPairs, bool Gated, class W>
inline void full_qkv_storage_phase(
    device const bfloat *qkv, device const W *q_norm,
    device const W *k_norm, device const float *rope_cos,
    device const float *rope_sin, device bfloat *queries,
    device bfloat *chunk_keys, device bfloat *chunk_values,
    constant SplashAttentionParams &params, threadgroup float *reductions,
    threadgroup bfloat *normalized, uint task, uint thread_index, uint lane,
    uint simd_group) {
  static_assert(QHeads % KHeads == 0);
  static_assert(2 * RotaryPairs <= HeadDim, "the rotated pairs are dimensions of the head");
  constexpr uint QStride = Gated ? 2 * HeadDim : HeadDim;
  constexpr uint PackedStride = QHeads * QStride + 2 * KHeads * HeadDim;
  constexpr uint QWidth = QHeads * QStride, KWidth = KHeads * HeadDim;
  uint query_tasks = params.rows * QHeads;
  bool query = task < query_tasks;
  uint local_task = query ? task : task - query_tasks;
  uint heads = query ? QHeads : KHeads;
  uint row = local_task / heads;
  uint head_index = local_task % heads;
  constant SplashAttentionSequence &sequence = params.sequence[splash_attention_sequence_of_row(params, row)];
  // The row's place in its sequence's staging and its packed row.
  uint sequence_row = row - sequence.first_row;
  ulong packed = ulong(sequence.row_begin) + sequence_row;
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

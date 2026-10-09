#pragma once

#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/rms_inverse.h"

// One KV head of one context row: a row of context_kv holds the row's keys,
// then its values (KWidth each). The keys are normalized and rotated into
// the ring slot of the row's position; the values are copied there.
inline void draft_context_kv_phase(
    device const bfloat *context_kv, device const bfloat *k_norm,
    device const float *rope_cos, device const float *rope_sin,
    device bfloat *keys, device bfloat *values, uint start_position,
    uint active_tokens, uint task,
    uint thread_index, uint lane, uint simd_group,
    threadgroup float *reductions, threadgroup bfloat *normalized) {
  constexpr uint KVHeads = SPLASH_DRAFT_KV_HEADS, HeadDim = SPLASH_DRAFT_HEAD_DIMENSION,
                 Window = SPLASH_DRAFT_SLIDING_WINDOW;
  constexpr uint KWidth = KVHeads * HeadDim, RowWidth = 2 * KWidth;
  uint row = task / KVHeads;
  if (row >= active_tokens)
    return;
  uint head_index = task % KVHeads;
  uint position = start_position + row;
  uint slot = position % Window;
  device const bfloat *source =
      context_kv + ulong(row) * RowWidth + head_index * HeadDim;
  device bfloat *key = keys + (ulong(head_index) * Window + slot) * HeadDim;
  device bfloat *value = values + ulong(head_index) * HeadDim * Window + slot;

  float element = thread_index < HeadDim ? float(source[thread_index]) : 0.0f;
  const float inverse = rms_inverse_of_sums(element * element, HeadDim,
                                            reductions, thread_index, lane,
                                            simd_group);
  if (thread_index < HeadDim) {
    normalized[thread_index] =
        bfloat(element * inverse * float(k_norm[thread_index]));
    value[ulong(thread_index) * Window] = source[KWidth + thread_index];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < HeadDim / 2) {
    float first = float(normalized[thread_index]);
    float second = float(normalized[thread_index + HeadDim / 2]);
    float cosine = rope_cos[ulong(row) * SPLASH_DRAFT_ROPE_PAIRS + thread_index];
    float sine = rope_sin[ulong(row) * SPLASH_DRAFT_ROPE_PAIRS + thread_index];
    key[thread_index] = bfloat(first * cosine - second * sine);
    key[thread_index + HeadDim / 2] = bfloat(second * cosine + first * sine);
  }
}

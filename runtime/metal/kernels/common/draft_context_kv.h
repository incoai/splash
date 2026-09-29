#pragma once

#include "metal/abi/KernelABI.h"

inline void draft_context_kv_phase(
    device const bfloat *context_qkv, device const bfloat *k_norm,
    device const float *rope_cos, device const float *rope_sin,
    device bfloat *keys, device bfloat *values, uint cache_stride,
    uint start_position, bool paged, constant DraftKvAddresses &pages,
    uint active_tokens, uint task, uint thread_index, uint lane,
    uint simd_group, threadgroup float *reductions,
    threadgroup bfloat *normalized) {
  constexpr uint KVHeads = 8, HeadDim = 128, Window = SPLASH_DRAFT_SLIDING_WINDOW;
  constexpr uint QWidth = 4096, KWidth = 1024, PackedWidth = 6144;
  uint row = task / KVHeads;
  if (row >= active_tokens)
    return;
  uint head_index = task % KVHeads;
  uint position = start_position + row;
  uint slot = position % Window;
  device const bfloat *source =
      context_qkv + ulong(row) * PackedWidth + QWidth + head_index * HeadDim;
  uint stride = cache_stride;
  if (paged) {
    uint page = slot / SPLASH_DRAFT_PAGE_TOKENS;
    keys = reinterpret_cast<device bfloat *>(pages.keys[page]);
    values = keys + KVHeads * SPLASH_DRAFT_PAGE_TOKENS * HeadDim;
    slot %= SPLASH_DRAFT_PAGE_TOKENS;
    stride = SPLASH_DRAFT_PAGE_TOKENS;
  }
  device bfloat *key = keys + (ulong(head_index) * stride + slot) * HeadDim;
  device bfloat *value = values + ulong(head_index) * HeadDim * stride + slot;

  float element = thread_index < HeadDim ? float(source[thread_index]) : 0.0f;
  float square_sum = simd_sum(element * element);
  if (lane == 0)
    reductions[simd_group] = square_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float total = 0.0f;
    for (uint i = 0; i < 8; ++i)
      total += reductions[i];
    reductions[0] = rsqrt(total / HeadDim + 1e-6f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < HeadDim) {
    normalized[thread_index] =
        bfloat(element * reductions[0] * float(k_norm[thread_index]));
    value[ulong(thread_index) * stride] = source[KWidth + thread_index];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < HeadDim / 2) {
    float first = float(normalized[thread_index]);
    float second = float(normalized[thread_index + HeadDim / 2]);
    float cosine = rope_cos[ulong(row) * (HeadDim / 2) + thread_index];
    float sine = rope_sin[ulong(row) * (HeadDim / 2) + thread_index];
    key[thread_index] = bfloat(first * cosine - second * sine);
    key[thread_index + HeadDim / 2] = bfloat(second * cosine + first * sine);
  }
}

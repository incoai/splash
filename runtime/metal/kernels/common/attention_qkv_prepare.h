#pragma once

#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/attention_sequence.h"
#include "metal/kernels/common/attention_store.h"
#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/rms_inverse.h"

// A threadgroup of HeadDim threads normalizes one head of one row of the
// packed projection: each query head's queries (and, Gated, its output gate
// after them), then the K heads, whose threadgroups also take the V head of
// their index; group (head, row, sequence) takes row `row` of the dispatch's
// sequence (SplashAttentionParams), read from its packed row (the sequence's
// layout entry, attention_sequence.h), and a row past the sequence's rows
// returns. q_norm and k_norm are read in their stored type W: bfloat for an
// MLX target's, float for a GGUF's F32 norms. The pairs of dimensions
// (i, i + RotaryPairs) then rotate by the packed row's RoPE table row of
// RotaryPairs pairs, the half split of the first 2 x RotaryPairs dimensions:
// Qwen3.5's first 64, or with HeadDim / 2 pairs the whole head (NeoX's).
// With no pairs (NoPE) the head is kept as normalized, and no table is read.
// A query head is staged at the sequence's staging rows for the attention; a
// K head's key and its V head's value go to the row's page slot through the
// sequence's page table (`page_table`, its binding, SPLASH_LANE_BINDING), as
// the layer's pages store them (splash_store_kv_slot), before any tile of
// the attention reads them.
template <uint QHeads, uint KHeads, uint HeadDim, uint RotaryPairs, bool Gated, class W, typename CacheElement,
          bool ValuesTokenMajor, bool NormalizeValue = false, bool SharedKv = false>
inline void full_qkv_storage_phase(
    device const bfloat *qkv, device const W *q_norm,
    device const W *k_norm, device const float *rope_cos,
    device const float *rope_sin, device bfloat *queries,
    device const SplashKvPage *page_table, device const SplashLaneRows *layout,
    constant SplashAttentionParams &params, threadgroup float *reductions, threadgroup bfloat *normalized,
    threadgroup float *maxima, uint3 group, uint thread_index, uint lane, uint simd_group) {
  static_assert(QHeads % KHeads == 0);
  static_assert(2 * RotaryPairs <= HeadDim, "the rotated pairs are dimensions of the head");
  constexpr uint QStride = Gated ? 2 * HeadDim : HeadDim;
  constexpr uint PackedStride = QHeads * QStride + (SharedKv ? 1 : 2) * KHeads * HeadDim;
  constexpr uint QWidth = QHeads * QStride, KWidth = KHeads * HeadDim;
  const uint sequence_row = group.y;
  constant SplashAttentionSequence &sequence = params.sequence[group.z];
  const bool query = group.x < QHeads;
  // A K head stores its row where its sequence's pages, staging and bound hold
  // it, as the host checks them; its page entry, whose place the parameters
  // give, loads beside the layout's entry and the row's values.
  const bool stores =
      !query && splash_attention_rows_valid(params, sequence) && sequence_row < sequence.row_bound;
  const SplashKvPage page = stores ? splash_store_page(page_table, sequence, sequence_row) : SPLASH_KV_PAGE_ABSENT;
  const SplashAttentionRows rows = splash_attention_rows(layout, params, group.z);
  if (sequence_row >= rows.rows || (!query && !stores))
    return;
  const uint head_index = query ? group.x : group.x - QHeads;
  const ulong packed = ulong(rows.first) + sequence_row;
  device const bfloat *source =
      query ? qkv + packed * PackedStride + head_index * QStride
            : qkv + packed * PackedStride + QWidth + head_index * HeadDim;
  device const W *weight = query ? q_norm : k_norm;

  float element = float(source[thread_index]);
  bfloat value = query ? bfloat(0.0f) : source[(SharedKv ? 0 : KWidth) + thread_index];
  const float inverse = rms_inverse_of_sums<HeadDim / 32>(element * element, HeadDim,
                                                          reductions, thread_index, lane,
                                                          simd_group);
  normalized[thread_index] =
      bfloat(element * inverse * float(weight[thread_index]));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // The thread's dimension of the head: a rotated pair's first or second
  // half, or the rest as normalized.
  bfloat result = normalized[thread_index];
  if constexpr (RotaryPairs != 0) {
    if (thread_index < 2 * RotaryPairs) {
      const uint pair = thread_index % RotaryPairs;
      const float first = float(normalized[pair]), second = float(normalized[pair + RotaryPairs]);
      const float cosine = rope_cos[packed * RotaryPairs + pair];
      const float sine = rope_sin[packed * RotaryPairs + pair];
      result = thread_index < RotaryPairs ? bfloat(first * cosine - second * sine)
                                          : bfloat(second * cosine + first * sine);
    }
  }
  if (query) {
    const uint kv_head = head_index / (QHeads / KHeads), local_head = head_index % (QHeads / KHeads);
    queries[((ulong(sequence.stage_row) + ulong(kv_head) * sequence.chunk_stride + sequence_row) *
                 (QHeads / KHeads) +
             local_head) *
                HeadDim +
            thread_index] = result;
    return;
  }
  if constexpr (NormalizeValue) {
    // Values use the raw projection even when it supplies the keys, with
    // a unit RMS weight and no rotary transform.
    const float raw = float(value);
    value = bfloat(raw * rms_inverse_of_sums<HeadDim / 32>(raw * raw, HeadDim, reductions, thread_index, lane,
                                                           simd_group));
  }
  splash_store_kv_slot<KHeads, HeadDim, CacheElement, ValuesTokenMajor>(
      page, sequence, params.kv, maxima, head_index, sequence_row, result, value, thread_index, lane, simd_group);
}

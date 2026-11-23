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
          bool ValuesTokenMajor>
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
  constexpr uint PackedStride = QHeads * QStride + 2 * KHeads * HeadDim;
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
  const bfloat value = query ? bfloat(0.0f) : source[KWidth + thread_index];
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
  splash_store_kv_slot<KHeads, HeadDim, CacheElement, ValuesTokenMajor>(
      page, sequence, params.kv, maxima, head_index, sequence_row, result, value, thread_index, lane, simd_group);
}

// A relative geometry's rows (metal/abi/PagedAttention.h): a threadgroup of
// HeadDim threads per (head, row, sequence), as full_qkv_storage_phase's. A
// query head's group reads its queries and relative state from the row's
// query side row (F32, SPLASH_RELATIVE_QUERY_SIDE floats), scales each by the
// layer's F32 scales (query_scale [QHeads][HeadDim], relative_scale [QHeads]
// [SPLASH_RELATIVE_WIDTH]) and clips them, and on a PositionScaled layer
// multiplies both by tau of the row's position (the sequence's committed
// tokens plus its row, or for a tree row plus its depth): its queries staged
// as full_qkv_storage_phase stages them, its relative state at its packed row
// of `relative` ([row][QHeads][SPLASH_RELATIVE_WIDTH], F32). A K head's group
// clips its key from `keys` and normalizes its value from `values` (each
// [row][KHeads x HeadDim], F32) by its RMS without a gamma, and stores both
// in the row's page slot. Each norm adds its squares per simdgroup and then
// the simdgroups' in order.
inline float relative_tau(uint position) {
  return 1.0f + SPLASH_RELATIVE_LOG_SCALE_ALPHA *
                    precise::log(max(1.0f, float(position + 1) / SPLASH_RELATIVE_LOG_SCALE_FLOOR));
}

inline float relative_clip_factor(float squares, float threshold) {
  return min(1.0f, threshold / (precise::sqrt(squares) + SPLASH_RELATIVE_CLIP_EPSILON));
}

// Both sums of the threadgroup's HeadDim threads, each thread's own pair
// added per simdgroup and then over the simdgroups in their order.
template <uint HeadDim>
inline float2 relative_head_sums(float2 own, threadgroup float2 *partials, uint lane, uint simd_group) {
#pragma clang fp reassociate(off)
  const float2 sum = float2(simd_sum(own.x), simd_sum(own.y));
  if (lane == 0)
    partials[simd_group] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float2 total = 0.0f;
  for (uint group = 0; group < HeadDim / 32; ++group)
    total += partials[group];
  return total;
}

template <uint QHeads, uint KHeads, uint HeadDim, bool PositionScaled, typename CacheElement, class KeyValue>
inline void relative_qkv_storage_phase(
    device const float *query_side, KeyValue key_value,
    device const float *query_scale, device const float *relative_scale, device bfloat *queries,
    device float *relative, device const SplashKvPage *page_table, device const SplashTreeRow *tree,
    device const SplashLaneRows *layout, constant SplashAttentionParams &params, threadgroup float2 *partials,
    threadgroup float *maxima, uint3 group, uint thread_index, uint lane, uint simd_group) {
  static_assert(QHeads % KHeads == 0 && HeadDim % 32 == 0 && SPLASH_RELATIVE_WIDTH <= 32,
                "a relative state's dimensions are the first simdgroup's lanes");
  constexpr uint Group = QHeads / KHeads, Width = SPLASH_RELATIVE_WIDTH;
  constexpr uint SideStride = SPLASH_RELATIVE_QUERY_SIDE(QHeads, HeadDim);
  const uint sequence_row = group.y;
  constant SplashAttentionSequence &sequence = params.sequence[group.z];
  const bool query = group.x < QHeads;
  const bool stores =
      !query && splash_attention_rows_valid(params, sequence) && sequence_row < sequence.row_bound;
  const SplashKvPage page = stores ? splash_store_page(page_table, sequence, sequence_row) : SPLASH_KV_PAGE_ABSENT;
  const SplashAttentionRows rows = splash_attention_rows(layout, params, group.z);
  if (sequence_row >= rows.rows || (!query && !stores))
    return;
  const uint head = query ? group.x : group.x - QHeads;
  const ulong packed = ulong(rows.first) + sequence_row;
  if (query) {
    device const float *side = query_side + packed * SideStride;
    const float element = query_scale[head * HeadDim + thread_index] * side[head * HeadDim + thread_index];
    const bool state = thread_index < Width;
    const float coordinate =
        state ? relative_scale[head * Width + thread_index] *
                    side[SPLASH_RELATIVE_STATES(QHeads, HeadDim) + head * Width + thread_index]
              : 0.0f;
    const float2 squares =
        relative_head_sums<HeadDim>(float2(element * element, coordinate * coordinate), partials, lane, simd_group);
    float tau = 1.0f;
    if constexpr (PositionScaled)
      tau = relative_tau(sequence.committed_tokens + (params.trees ? uint(tree[packed].depth) : sequence_row));
    const float scaled = element * relative_clip_factor(squares.x, SPLASH_RELATIVE_QK_CLIP) * tau;
    const uint kv_head = head / Group, local_head = head % Group;
    queries[((ulong(sequence.stage_row) + ulong(kv_head) * sequence.chunk_stride + sequence_row) * Group +
             local_head) *
                HeadDim +
            thread_index] = bfloat(scaled);
    if (state)
      relative[(packed * QHeads + head) * Width + thread_index] =
          coordinate * relative_clip_factor(squares.y, SPLASH_RELATIVE_STATE_CLIP) * tau;
    return;
  }
  const float2 convolved = key_value(packed, head * HeadDim + thread_index, rows.first);
  const float key = convolved.x, value = convolved.y;
  const float2 squares = relative_head_sums<HeadDim>(float2(key * key, value * value), partials, lane, simd_group);
  const float clipped = key * relative_clip_factor(squares.x, SPLASH_RELATIVE_QK_CLIP);
  const float normalized = value * rsqrt(squares.y / HeadDim + kRmsEpsilon);
  splash_store_kv_slot<KHeads, HeadDim, CacheElement, false>(page, sequence, params.kv, maxima, head, sequence_row,
                                                             bfloat(clipped), bfloat(normalized), thread_index, lane,
                                                             simd_group);
}

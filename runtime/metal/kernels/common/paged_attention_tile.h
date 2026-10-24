#pragma once

#include "metal/abi/PagedAttention.h"
#include "metal/kernels/common/kv_extent.h"
#include "metal/kernels/common/kv_paging.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

// Paged attention over INT8 or BF16 KV. INT8 has one fp32 scale per
// (token, KV head); BF16 reads the stored values directly. The preceding
// store writes current rows to their final slots; committed_tokens controls
// visibility, and the next command overwrites rejected verify rows.
//
// A threadgroup owns one KV head, query tile (or a part of one,
// splash_attention_tile_part) and history split. It processes the GQA group
// as M = rows x query heads per KV head, at most 64, of HeadDim dimensions
// each, the head dimension a kernel is compiled for. Each split writes fp32
// partials and statistics for a fixed-order reduce.
//
// A kernel compiled with a Window attends, for a row at position p, only the
// keys p - Window to p (Window + 1 with its own; a sliding-window layer's),
// and a reduce given a layer's learned sinks adds one per query head
// (splash_attention_combine_splits). Window 0 and no sinks is plain causal
// attention over the whole history.
//
// Prefill and verify share this device-operand page loop.
constant uint SplashVerifyMaximumSplits =
    SPLASH_VERIFY_ATTENTION_MAXIMUM_SPLITS;
constant uint SplashPrefillTileRows = SPLASH_PREFILL_ATTENTION_TILE_ROWS;
constant uint SplashPrefillMaximumSplits =
    SPLASH_PREFILL_ATTENTION_MAXIMUM_SPLITS;

inline bool splash_prefill_attention_contract_valid(
    constant SplashPrefillAttentionParams &params) {
  return params.rows > 0 && params.rows <= SPLASH_PREFILL_TOKEN_BUDGET &&
         params.chunk_stride >= params.rows &&
         params.chunk_stride <= SPLASH_PREFILL_TOKEN_BUDGET &&
         params.chunk_stride % SPLASH_TARGET_KV_BLOCK_TOKENS == 0 &&
         params.page_table_entries >=
             (params.committed_tokens + params.rows +
              SplashKvPageTokens - 1) /
                 SplashKvPageTokens &&
         params.kv.extent_pages > 0 && params.split_count > 0 &&
         params.split_count <= SplashPrefillMaximumSplits &&
         ulong(params.committed_tokens) + params.rows <=
             ulong(SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS);
}

inline bool splash_verify_attention_contract_valid(
    constant SplashVerifyAttentionParams &params) {
  return params.page_table_entries >=
             (params.committed_tokens + SPLASH_TARGET_VERIFY_ROWS +
              SplashKvPageTokens - 1) /
                 SplashKvPageTokens &&
         params.kv.extent_pages > 0 &&
         ulong(params.committed_tokens) + SPLASH_TARGET_VERIFY_ROWS <=
             ulong(SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS) &&
         params.split_count > 0 &&
         params.split_count <= SplashVerifyMaximumSplits &&
         params.slot_splits >= params.split_count &&
         params.slot_splits <= SplashVerifyMaximumSplits;
}

// The query tile a threadgroup owns, of one KV head: the offset of its fused
// rows in the queries and the output, the position of its first row, its
// active rows, its history splits and the partial slot of its first split. A
// verify tile is a lane's eight rows, a prefill tile SplashPrefillTileRows
// rows of the chunk; a tile whose parameters fail the contract, or past the
// chunk's rows, is inactive. Split and reduce entries, dense and sparse, take
// their tiles from here, so they agree on every offset.
struct SplashAttentionTile {
  ulong offset;
  ulong first_slot;
  uint first_position;
  uint active_rows;
  uint splits;
  bool active;
};

template <uint KVHeads, uint QueryHeadsPerKVHead, uint HeadDim>
inline SplashAttentionTile splash_verify_attention_tile(
    constant SplashVerifyAttentionParams &lane, uint kv_head, uint batch) {
  if (!splash_verify_attention_contract_valid(lane) || kv_head >= KVHeads)
    return {};
  const ulong head = ulong(batch) * KVHeads + kv_head;
  return {head * SPLASH_VERIFY_CHUNK_STRIDE * QueryHeadsPerKVHead * HeadDim,
          head * lane.slot_splits, lane.committed_tokens, SPLASH_TARGET_VERIFY_ROWS,
          lane.split_count, true};
}

template <uint KVHeads, uint QueryHeadsPerKVHead, uint HeadDim>
inline SplashAttentionTile splash_prefill_attention_tile(
    constant SplashPrefillAttentionParams &params, uint kv_head, uint tile) {
  const uint start = tile * SplashPrefillTileRows;
  if (!splash_prefill_attention_contract_valid(params) || kv_head >= KVHeads ||
      start >= params.rows)
    return {};
  return {(ulong(kv_head) * params.chunk_stride + start) * QueryHeadsPerKVHead *
              HeadDim,
          (ulong(tile) * KVHeads + kv_head) * params.split_count,
          params.committed_tokens + start, min(SplashPrefillTileRows, params.rows - start),
          params.split_count, true};
}

// Part `part` of a tile run as Parts parts of RowsPerTile / Parts rows each
// (SPLASH_ATTENTION_TILE_PARTS: at most 64 fused rows each, the 256-thread page
// loop's), each a threadgroup of the split over the same pages. A part is a
// tile of its own rows: their queries and output, positions and active rows,
// and partial slots of its own, slot_stride (the slot splits of each of the
// tile's slots) apart. A part past the tile's active rows is inactive. One
// part is the tile.
template <uint QueryHeadsPerKVHead, uint RowsPerTile, uint Parts, uint HeadDim>
inline SplashAttentionTile splash_attention_tile_part(SplashAttentionTile tile, uint part,
                                                      uint slot_stride) {
  static_assert(Parts > 0 && RowsPerTile % Parts == 0, "parts of whole rows");
  constexpr uint Rows = RowsPerTile / Parts;
  const uint first = part * Rows;
  if (!tile.active || first >= tile.active_rows)
    return {};
  return {tile.offset + ulong(first) * QueryHeadsPerKVHead * HeadDim,
          tile.first_slot * Parts + ulong(part) * slot_stride, tile.first_position + first,
          min(Rows, tile.active_rows - first), tile.splits, true};
}

// Split and reduce derive the same balanced partition of each query tile's
// visible pages (splash_attention_first_page on). Causal and window masking
// remain per query row inside each split.
inline uint splash_attention_pages(uint visible_tokens) {
  return (visible_tokens + SplashKvPageTokens - 1) / SplashKvPageTokens;
}

inline uint splash_attention_pages_per_split(uint pages, uint splits) {
  return (pages + splits - 1) / splits;
}

// The first key a row at `position` attends: Window keys before it, or with
// no window (0) the history's first.
template <uint Window> inline uint splash_attention_window_start(uint position) {
  return Window && position > Window ? position - Window : 0;
}

// The first page a tile whose first row is at first_position reads: its
// first row's window start's, the tile's others starting their windows on it
// or after it and masking the keys before them. Its pages end with its last
// row's (splash_attention_pages), and with no window begin at page 0.
template <uint Window> inline uint splash_attention_first_page(uint first_position) {
  return splash_attention_window_start<Window>(first_position) / SplashKvPageTokens;
}

// The scale of a head's scores, 1/sqrt(HeadDim), as the literal of each head
// dimension the kernels are compiled for.
template <uint HeadDim> constexpr float splash_attention_scale() {
  static_assert(HeadDim == 128 || HeadDim == 256, "a score scale for each compiled head dimension");
  return HeadDim == 256 ? 0.0625f : 0.08838834764831845f;
}

// One Page32 block of one tile: key-scaled scores become value-scaled bf16
// probabilities and the row statistics advance. Four lanes own one fused row,
// eight consecutive tokens each, so the row maximum and sum are two xor
// shuffles instead of a 32-lane reduction per row and the eight bf16
// probabilities leave as one 16-byte store. Threads past 4 x FusedRows
// (kv4_g6: simdgroups 6 and 7) only take part in the caller's barriers.
// The float4 loads need 16-byte aligned slabs (alignas in the entries) and
// the page's 32 key and value scales as float4 vectors, in device memory
// (splash_q8_scale_index is a multiple of 32 floats and every region of an
// extent starts 64 KiB-aligned).
// Rows whose running maximum grew atomically set the shared boolean rescale
// flag. Existing threadgroup barriers separate reset, concurrent set, and
// read; relaxed atomics make the same-value writes safe without changing
// arithmetic.
template <uint QueryHeadsPerKVHead, uint RowsPerTile, uint HeadDim, uint Window, bool Quantized>
inline void splash_attention_page_softmax(
    threadgroup const float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    device const float4 *key_scales, device const float4 *value_scales,
    uint token_start,
    uint visible_tokens, uint committed_tokens, uint active_rows,
    uint thread_index) {
  constexpr uint N = SplashKvPageTokens;
  constexpr uint FusedRows = RowsPerTile * QueryHeadsPerKVHead;
  constexpr uint TokensPerLane = 8;
  constexpr uint LanesPerRow = N / TokensPerLane;
  constexpr float Scale = splash_attention_scale<HeadDim>();
  static_assert(N == 32, "four lanes of eight tokens span one page");
  static_assert(LanesPerRow * FusedRows <= 256,
                "one softmax lane per thread of the 256-thread tile");
  if (thread_index >= LanesPerRow * FusedRows)
    return;
  const uint fused_row = thread_index / LanesPerRow;
  const uint column = thread_index % LanesPerRow * TokensPerLane;
  const uint query_row = fused_row / QueryHeadsPerKVHead;
  const uint causal_end =
      committed_tokens + min(query_row, active_rows - 1) + 1;
  // The row sees the tokens of [start, its causal end): its window's, or
  // with no window its whole history's. Tokens and limit count from start,
  // so one unsigned comparison tests both ends: a token before start wraps
  // past every limit.
  const uint start = splash_attention_window_start<Window>(causal_end - 1);
  const uint limit = min(visible_tokens, causal_end) - start;
  const uint token = token_start + column - start;
  threadgroup const float4 *scores4 =
      reinterpret_cast<threadgroup const float4 *>(scores + fused_row * N +
                                                   column);
  const uint vector = column / 4;
  float score[TokensPerLane];
  {
    float4 low = scores4[0], high = scores4[1];
    if constexpr (Quantized) {
      low *= key_scales[vector];
      high *= key_scales[vector + 1];
    }
    low *= Scale;
    high *= Scale;
    score[0] = low.x, score[1] = low.y, score[2] = low.z, score[3] = low.w;
    score[4] = high.x, score[5] = high.y, score[6] = high.z, score[7] = high.w;
  }
  float local_max = -INFINITY;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    score[j] = token + j < limit ? score[j] : -INFINITY;
    local_max = max(local_max, score[j]);
  }
  local_max = max(local_max, simd_shuffle_xor(local_max, 1));
  local_max = max(local_max, simd_shuffle_xor(local_max, 2));
  const float previous_max = row_max[fused_row];
  const float next_max = max(previous_max, local_max);
  float probability[TokensPerLane];
  float local_sum = 0.0f;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    probability[j] = token + j < limit ? fast::exp(score[j] - next_max) : 0.0f;
    local_sum += probability[j];
  }
  local_sum += simd_shuffle_xor(local_sum, 1);
  local_sum += simd_shuffle_xor(local_sum, 2);
  if (column == 0) {
    const float scale = next_max == -INFINITY || next_max == previous_max
                            ? 1.0f
                            : fast::exp(previous_max - next_max);
    previous_scale[fused_row] = scale;
    row_sum[fused_row] = row_sum[fused_row] * scale + local_sum;
    row_max[fused_row] = next_max;
    if (scale != 1.0f)
      atomic_store_explicit(rescale, 1u, memory_order_relaxed);
  }
  // Masked tokens stay exactly zero whatever their stored value scale holds.
  float4 low_scales(1.0f), high_scales(1.0f);
  if constexpr (Quantized) {
    low_scales = value_scales[vector];
    high_scales = value_scales[vector + 1];
  }
  const float4 low(token + 0 < limit ? probability[0] * low_scales.x : 0.0f,
                   token + 1 < limit ? probability[1] * low_scales.y : 0.0f,
                   token + 2 < limit ? probability[2] * low_scales.z : 0.0f,
                   token + 3 < limit ? probability[3] * low_scales.w : 0.0f);
  const float4 high(token + 4 < limit ? probability[4] * high_scales.x : 0.0f,
                    token + 5 < limit ? probability[5] * high_scales.y : 0.0f,
                    token + 6 < limit ? probability[6] * high_scales.z : 0.0f,
                    token + 7 < limit ? probability[7] * high_scales.w : 0.0f);
  threadgroup bfloat4 *probabilities4 = reinterpret_cast<threadgroup bfloat4 *>(
      probabilities + fused_row * N + column);
  probabilities4[0] = bfloat4(low);
  probabilities4[1] = bfloat4(high);
}

// One tile over its split's pages, whose K/V (INT8 with their scales, or
// BF16) it consumes in place as device tensor operands: the prefill tile of
// SPLASH_PREFILL_ATTENTION_TILE_ROWS rows and the verify tile of
// SPLASH_TARGET_VERIFY_ROWS (RowsPerTile). Queries are KV-head-major
// [kv head][row][query head in group][dimension], so the tile's fused rows
// form one contiguous M x D tensor. Each page is reached through its table
// entry (kv_extent.h).
// Three barriers per page order the score store, the softmax and the
// probability reads of PV.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint RowsPerTile, uint HeadDim,
          uint Window, typename CacheElement>
inline void splash_paged_attention_tile(
    device bfloat *tile_queries, device const SplashKvPage *page_table,
    SplashKvLayer kv, uint kv_head, uint committed_tokens, uint active_rows, uint splits,
    uint split, device float *partials, device float *statistics, ulong slot,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    uint thread_index) {
  constexpr ushort M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr bool Quantized = is_same<CacheElement, int8_t>::value;
  constexpr ushort N = SplashKvPageTokens;
  constexpr ushort D = HeadDim;
  uint visible_tokens = committed_tokens + active_rows;
  uint pages = splash_attention_pages(visible_tokens);
  uint first_page = 0;
  if constexpr (Window != 0)
    first_page = splash_attention_first_page<Window>(committed_tokens);
  uint per_split = splash_attention_pages_per_split(pages - first_page, splits);
  uint page_begin = first_page + split * per_split;
  if (page_begin >= pages)
    return;
  uint page_end = min(pages, page_begin + per_split);
  if (thread_index < M) {
    row_max[thread_index] = -INFINITY;
    row_sum[thread_index] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto qt = tensor(tile_queries, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto st = tensor(scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto pt = tensor(probabilities, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto p0 = pt.slice<N, M>(0, 0);
  auto key_type = tensor(static_cast<device CacheElement *>(nullptr),
                         dextents<int, 2>{D, N}, array<int, 2>{1, D});
  auto value_type = tensor(static_cast<device CacheElement *>(nullptr),
                           dextents<int, 2>{N, D}, array<int, 2>{1, N});
  const SplashKvAddressing<KVHeads, HeadDim, CacheElement> addressing(kv, kv_head);
  auto q0 = qt.slice<D, M>(0, 0);
  auto k0 = key_type.template slice<D, N>(0, 0);
  auto v0 = value_type.template slice<N, D>(0, 0);
  // QK writes a complete page score tile, the softmax applies an INT8 page's
  // key scales and PV accumulates the running output.
  constexpr auto qk_descriptor =
      matmul2d_descriptor(M, N, D, false, true, false,
                          matmul2d_descriptor::mode::multiply);
  constexpr auto pv_descriptor =
      matmul2d_descriptor(M, D, N, false, true, true,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
  matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
  auto running = pv.template get_destination_cooperative_tensor<
      decltype(p0), decltype(v0), float>();
  // Uniform capacity partitions the logical tile across eight 32-lane groups.
  // Equality proves no padding; otherwise only valid entries may be accessed.
  const bool running_full =
      uint(running.get_capacity()) * (8u * 32u) == uint(M) * D;
#pragma unroll
  for (ushort index = 0; index < running.get_capacity(); ++index) {
    if (running_full || running.is_valid_element(index))
      running[index] = 0.0f;
  }

  for (uint page = page_begin; page < page_end; ++page) {
    const SplashKvPageTensors<CacheElement> tensors =
        addressing.page(page_table[page]);
    uint token_start = page * N;
    auto kt = tensor(tensors.keys, dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto vt = tensor(tensors.values, dextents<int, 2>{N, D}, array<int, 2>{1, N});
    device const float *key_scales = tensors.key_scales;
    device const float *value_scales = tensors.value_scales;

    auto page_scores = qk.template get_destination_cooperative_tensor<
        decltype(q0), decltype(k0), float>();
    // One full-dimension product initializes the score CT through MPP.
    auto ks = kt.template slice<D, N>(0, 0);
    qk.run(q0, ks, page_scores);
    page_scores.store(st.slice<N, M>(0, 0));
    if (thread_index == 0)
      atomic_store_explicit(rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    splash_attention_page_softmax<QueryHeadsPerKVHead, RowsPerTile, HeadDim, Window, Quantized>(
        scores, probabilities, row_max, row_sum, previous_scale, rescale,
        reinterpret_cast<device const float4 *>(key_scales),
        reinterpret_cast<device const float4 *>(value_scales), token_start,
        visible_tokens, committed_tokens, active_rows, thread_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(rescale, memory_order_relaxed)) {
#pragma unroll
      for (ushort index = 0; index < running.get_capacity(); ++index) {
        if (!running_full && !running.is_valid_element(index))
          continue;
        auto coordinates = running.get_multidimensional_index(index);
        running[index] *= previous_scale[coordinates[1]];
      }
    }
    auto vs = vt.template slice<N, D>(0, 0);
    pv.run(p0, vs, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  auto target = tensor(partials + slot * M * D, dextents<int, 2>{D, M},
                       array<int, 2>{1, D});
  running.store(target.slice<D, M>(0, 0));
  if (thread_index < M) {
    statistics[(slot * M + thread_index) * 2] = row_max[thread_index];
    statistics[(slot * M + thread_index) * 2 + 1] = row_sum[thread_index];
  }
}

// The sink of a fused row's query head among its group's, or with none a
// logit below every score.
template <uint QueryHeadsPerKVHead> inline float splash_row_sink(uint) { return -INFINITY; }

template <uint QueryHeadsPerKVHead>
inline float splash_row_sink(uint fused_row, device const float *group_sinks) {
  return group_sinks[fused_row % QueryHeadsPerKVHead];
}

// Partials are [slot][fused row][dimension]; statistics are
// [slot][fused row]{max, sum}. The callers lay slots out as
// [tile][KV head][part][split] (prefill) and [lane][KV head][part][split]
// (verify), a sparse tile being one part.
// Combines the first `written` splits of one fused row in split order, a
// threadgroup of HeadDim threads, one per output dimension. The statistics
// are shared by all output dimensions: their weights are computed once per
// row, the splits' maxima by the threads at their split's index (written
// never exceeds HeadDim), then each lane streams one dimension of the
// partials. A row past the tile's active rows is written as zeros; a
// threadgroup owns one fused row, so it returns uniformly.
// A layer's learned sinks come as group_sinks, a logit per query head of the
// tile's KV head (its GQA group), and none without: the row's query head's
// sink is one more key, with no value, of the softmax (the scores' scale
// already applied), so it enters the maximum and, after the splits in their
// order, the denominator.
template <uint QueryHeadsPerKVHead, uint RowsPerTile, uint HeadDim, class... GroupSinks>
inline void splash_attention_combine_splits(
    device const float *partials, device const float *statistics,
    device bfloat *tile_output, uint active_rows, uint written,
    ulong head_slot, uint fused_row, uint thread_index,
    threadgroup float *weights, threadgroup float *group_values,
    GroupSinks... group_sinks) {
  constexpr bool Sinks = sizeof...(GroupSinks) != 0;
  constexpr uint M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr uint D = HeadDim;
  static_assert(D >= SPLASH_VERIFY_ATTENTION_MAXIMUM_SPLITS &&
                    D >= SPLASH_PREFILL_ATTENTION_MAXIMUM_SPLITS,
                "a thread per split");
  if (fused_row / QueryHeadsPerKVHead >= active_rows) {
    tile_output[fused_row * D + thread_index] = bfloat(0.0f);
    return;
  }
  const uint lane = thread_index % 32, sg = thread_index / 32;
  const ulong stat = ((head_slot + thread_index) * M + fused_row) * 2;
  const float maximum = thread_index < written ? statistics[stat] : -INFINITY;
  const float group_maximum = simd_max(maximum);
  if (lane == 0) group_values[sg] = group_maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float split_maximum = simd_max(lane < D / 32 ? group_values[lane] : -INFINITY);
  const float sink = splash_row_sink<QueryHeadsPerKVHead>(fused_row, group_sinks...);
  const float row_maximum = Sinks ? max(split_maximum, sink) : split_maximum;
  if (thread_index < written)
    weights[thread_index] = fast::exp(maximum - row_maximum);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float numerator = 0.0f, denominator = 0.0f;
  // Keep both accumulations in the original split order. A parallel sum of
  // the denominator changes speculative acceptance on real model prompts.
  for (uint split = 0; split < written; ++split) {
    const float weight = weights[split];
    const ulong stat = ((head_slot + split) * M + fused_row) * 2;
    numerator += weight *
        partials[((head_slot + split) * M + fused_row) * D + thread_index];
    denominator += weight * statistics[stat + 1];
  }
  if constexpr (Sinks)
    denominator += fast::exp(sink - row_maximum);
  tile_output[fused_row * D + thread_index] =
      bfloat(denominator > 0.0f ? numerator / denominator : 0.0f);
}

// The sinks of KV head kv_head's query heads (its GQA group) among a
// layer's, one per query head.
template <uint QueryHeadsPerKVHead>
inline device const float *splash_group_sinks(uint kv_head, device const float *sinks) {
  return sinks + kv_head * QueryHeadsPerKVHead;
}

// Fused row `fused_row` of a tile run as Parts parts
// (splash_attention_tile_part) combines the splits of its part's page loop:
// those that own at least one page wrote their partials, and the partition
// is recomputed here, of the same Window. A part past the tile's active rows
// ran no split; its rows are written as zeros.
template <uint QueryHeadsPerKVHead, uint RowsPerTile, uint Parts, uint HeadDim,
          uint Window, class... GroupSinks>
inline void splash_attention_reduce_row(
    device const float *partials, device const float *statistics,
    device bfloat *output, SplashAttentionTile whole, uint slot_stride,
    uint fused_row, uint thread_index, threadgroup float *weights,
    threadgroup float *group_values, GroupSinks... group_sinks) {
  constexpr uint PartRows = RowsPerTile / Parts;
  constexpr uint PartM = PartRows * QueryHeadsPerKVHead;
  const SplashAttentionTile tile =
      splash_attention_tile_part<QueryHeadsPerKVHead, RowsPerTile, Parts, HeadDim>(
          whole, fused_row / PartM, slot_stride);
  if (!tile.active) {
    output[whole.offset + ulong(fused_row) * HeadDim +
           thread_index] = bfloat(0.0f);
    return;
  }
  uint pages = splash_attention_pages(tile.first_position + tile.active_rows);
  if constexpr (Window != 0)
    pages -= splash_attention_first_page<Window>(tile.first_position);
  const uint per_split = splash_attention_pages_per_split(pages, tile.splits);
  const uint written = (pages + per_split - 1) / per_split;
  splash_attention_combine_splits<QueryHeadsPerKVHead, PartRows, HeadDim>(
      partials, statistics, output + tile.offset, tile.active_rows, written,
      tile.first_slot, fused_row % PartM, thread_index, weights, group_values,
      group_sinks...);
}

#pragma once

#include "metal/abi/PagedAttention.h"
#include "metal/abi/VerifyTree.h"
#include "metal/kernels/common/attention_sequence.h"
#include "metal/kernels/common/int8_activations.h"
#include "metal/kernels/common/kv_extent.h"
#include "metal/kernels/common/kv_paging.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

// Paged attention over INT8 or BF16 KV. INT8 has one fp32 scale per
// (token, KV head); BF16 reads the stored values directly. The preceding
// store writes each row's K/V to its row's slot: a prompt row's is final, a
// decode tree row's is its own row of the block, whose accepted path a commit
// moves to the slots of its depths; committed_tokens controls visibility,
// and the next command overwrites a decode step's rejected rows.
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
// Every attention dispatch's tiles share this device-operand page loop.
constant uint SplashAttentionMaximumSplits = SPLASH_ATTENTION_MAXIMUM_SPLITS;
constant uint SplashAttentionTileRows = SPLASH_ATTENTION_TILE_ROWS;

// A sequence's tiles run while its rows and splits are within bounds
// (attention_sequence.h).
inline bool splash_attention_contract_valid(constant SplashAttentionParams &params,
                                            constant SplashAttentionSequence &s) {
  return splash_attention_rows_valid(params, s) && s.split_count > 0 && s.split_count <= SplashAttentionMaximumSplits;
}

// The query tile a threadgroup owns, of one KV head: the offset of its fused
// rows in the queries and the output, the position of its first row, its
// active rows, its history splits and the partial slot of its first split.
// A sequence's tile t holds its rows (`rows`, its layout entry's,
// attention_sequence.h) from t SplashAttentionTileRows on; a tile whose
// sequence fails the contract, or past its rows (a spare slot of its bound),
// is inactive. Split and reduce entries, dense and sparse, take their tiles
// from here, so they agree on every offset.
struct SplashAttentionTile {
  ulong offset;
  ulong first_slot;
  uint first_position;
  uint active_rows;
  uint splits;
  bool active;
};

template <uint KVHeads, uint QueryHeadsPerKVHead, uint HeadDim>
inline SplashAttentionTile splash_attention_tile(constant SplashAttentionParams &params,
                                                 constant SplashAttentionSequence &sequence, SplashAttentionRows rows,
                                                 uint kv_head, uint tile) {
  const uint start = tile * SplashAttentionTileRows;
  if (!splash_attention_contract_valid(params, sequence) || kv_head >= KVHeads || start >= rows.rows)
    return {};
  return {(ulong(sequence.stage_row) + ulong(kv_head) * sequence.chunk_stride + start) * QueryHeadsPerKVHead *
              HeadDim,
          ulong(sequence.first_slot) + (ulong(tile) * KVHeads + kv_head) * sequence.split_count,
          sequence.committed_tokens + start, min(SplashAttentionTileRows, rows.rows - start),
          sequence.split_count, true};
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

// A tile of a decode lane's block as a tree (SplashAttentionParams::trees):
// the place in its tree of the tile's first row, and the slot of the block's
// anchor, from which the block's rows' K/V sit at the slots of their rows.
// A prompt's tile has none (null rows) and attends causally.
struct SplashAttentionTree {
  device const SplashTreeRow *rows;
  uint anchor;
};

// The tree of a sequence's tile whose first row is at `first_position`, of
// the command's tree rows (a row each of its packed rows).
inline SplashAttentionTree splash_attention_tree(constant SplashAttentionParams &params,
                                                 constant SplashAttentionSequence &sequence, SplashAttentionRows rows,
                                                 device const SplashTreeRow *tree, uint first_position) {
  if (!params.trees)
    return {nullptr, 0};
  return {tree + rows.first + (first_position - sequence.committed_tokens), sequence.committed_tokens};
}

// The position a tile's window starts from: its first row's, or a tree's
// anchor's, since a tree row's depth may be less than its row.
inline uint splash_attention_window_position(bool tree, uint anchor, uint first_position) {
  return tree ? anchor : first_position;
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

// The first page a tile whose window position (splash_attention_window_position)
// is first_position reads: that position's window start's, the tile's rows
// starting their windows on it or after it and masking the keys before them.
// Its pages end with its last row's (splash_attention_pages), and with no
// window begin at page 0.
template <uint Window> inline uint splash_attention_first_page(uint first_position) {
  return splash_attention_window_start<Window>(first_position) / SplashKvPageTokens;
}

// The scale of a head's scores, 1/sqrt(HeadDim), as the literal of each head
// dimension the kernels are compiled for.
template <uint HeadDim> constexpr float splash_attention_scale() {
  static_assert(HeadDim == 128 || HeadDim == 256, "a score scale for each compiled head dimension");
  return HeadDim == 256 ? 0.0625f : 0.08838834764831845f;
}

// The lanes of one fused row in the page loop: the softmax's, eight of a
// page's tokens each, and the query quantization's, so that each lane holds
// its own row's query scale.
constant uint SplashAttentionLanesPerRow = 4;

// One Page32 block of one tile: key-scaled scores (fp32, or the int32
// products of INT8 queries) times score_scale, the row's, become value-scaled
// bf16 probabilities and the row statistics advance. Four lanes own one
// fused row, eight consecutive tokens each, so the row maximum and sum are
// two xor shuffles instead of a 32-lane reduction per row and the eight bf16
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
// A tree's row (SplashAttentionTree) attends the history of its position's
// window, its position the anchor's plus its `depth`, and of the block's
// rows those its `ancestors` hold, itself among them; only a page that
// holds rows of the block tests them.
template <uint QueryHeadsPerKVHead, uint RowsPerTile, uint Window, bool Quantized, typename Score>
inline void splash_attention_page_softmax(
    threadgroup const Score *scores, float score_scale, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    device const float4 *key_scales, device const float4 *value_scales,
    uint token_start,
    uint visible_tokens, uint committed_tokens, uint active_rows,
    bool tree, uint anchor, uint ancestors, uint depth,
    uint thread_index) {
  constexpr uint N = SplashKvPageTokens;
  constexpr uint FusedRows = RowsPerTile * QueryHeadsPerKVHead;
  constexpr uint TokensPerLane = 8;
  constexpr uint LanesPerRow = SplashAttentionLanesPerRow;
  static_assert(N == LanesPerRow * TokensPerLane, "four lanes of eight tokens span one page");
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
  const uint start = splash_attention_window_start<Window>(tree ? anchor + depth : causal_end - 1);
  const uint limit = min(visible_tokens, causal_end) - start;
  const uint token = token_start + column - start;
  // Bit j: the lane's token j is a row of the tree's block off the row's
  // path, which it does not see.
  uint off_path = 0;
  if (tree && token_start + N > anchor) {
#pragma unroll
    for (uint j = 0; j < TokensPerLane; ++j) {
      const uint slot = token_start + column + j;
      off_path |= uint(slot >= anchor && !((ancestors >> min(slot - anchor, 31u)) & 1u)) << j;
    }
  }
  bool seen[TokensPerLane];
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j)
    seen[j] = token + j < limit && !((off_path >> j) & 1u);
  threadgroup const vec<Score, 4> *scores4 =
      reinterpret_cast<threadgroup const vec<Score, 4> *>(scores + fused_row * N + column);
  const uint vector = column / 4;
  float score[TokensPerLane];
  {
    float4 low = float4(scores4[0]), high = float4(scores4[1]);
    if constexpr (Quantized) {
      low *= key_scales[vector];
      high *= key_scales[vector + 1];
    }
    low *= score_scale;
    high *= score_scale;
    score[0] = low.x, score[1] = low.y, score[2] = low.z, score[3] = low.w;
    score[4] = high.x, score[5] = high.y, score[6] = high.z, score[7] = high.w;
  }
  float local_max = -INFINITY;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    score[j] = seen[j] ? score[j] : -INFINITY;
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
    probability[j] = seen[j] ? fast::exp(score[j] - next_max) : 0.0f;
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
  const float4 low(seen[0] ? probability[0] * low_scales.x : 0.0f,
                   seen[1] ? probability[1] * low_scales.y : 0.0f,
                   seen[2] ? probability[2] * low_scales.z : 0.0f,
                   seen[3] ? probability[3] * low_scales.w : 0.0f);
  const float4 high(seen[4] ? probability[4] * high_scales.x : 0.0f,
                    seen[5] ? probability[5] * high_scales.y : 0.0f,
                    seen[6] ? probability[6] * high_scales.z : 0.0f,
                    seen[7] ? probability[7] * high_scales.w : 0.0f);
  threadgroup bfloat4 *probabilities4 = reinterpret_cast<threadgroup bfloat4 *>(
      probabilities + fused_row * N + column);
  probabilities4[0] = bfloat4(low);
  probabilities4[1] = bfloat4(high);
}

// The tile's queries as INT8 codes with one scale per fused row, by the
// int8-activation rule (kernels/common/int8_activations.h), for the INT8 x
// INT8 score product. A row's softmax lanes quantize it, a quarter of its
// dimensions each, so each returns its own row's scale: NaN for a row holding
// an infinity or a NaN, whose scores are then NaN, as with bf16 queries.
// Lanes past those of M rows return 0.
template <uint M, uint HeadDim>
inline float splash_attention_query_codes(device const bfloat *queries, device int8_t *codes,
                                          uint thread_index) {
  constexpr uint LanesPerRow = SplashAttentionLanesPerRow;
  static_assert(LanesPerRow == 4, "two xor shuffles reduce a row's four lanes");
  constexpr uint Quarter = HeadDim / LanesPerRow;
  if (thread_index >= LanesPerRow * M)
    return 0.0f;
  const uint first = thread_index / LanesPerRow * HeadDim + thread_index % LanesPerRow * Quarter;
  device const bfloat4 *source = reinterpret_cast<device const bfloat4 *>(queries + first);
  float maximum = 0.0f;
  uint finite = 1;
#pragma unroll
  for (uint i = 0; i < Quarter / 4; ++i) {
    const bfloat4 query = source[i];
    finite &= !int8_activation_not_finite(query);
    const float4 value = abs(float4(query));
    maximum = max(maximum, max(max(value.x, value.y), max(value.z, value.w)));
  }
  maximum = max(maximum, simd_shuffle_xor(maximum, 1));
  maximum = max(maximum, simd_shuffle_xor(maximum, 2));
  finite &= simd_shuffle_xor(finite, 1);
  finite &= simd_shuffle_xor(finite, 2);
  const Int8Block block = int8_activation_block(maximum);
  device char4 *target = reinterpret_cast<device char4 *>(codes + first);
#pragma unroll
  for (uint i = 0; i < Quarter / 4; ++i)
    target[i] = char4(int8_activation_code(float4(source[i]), block.inverse));
  return as_type<float>(int8_activation_bits(block.scale, finite != 0));
}

// One tile over its split's pages, whose K/V (INT8 with their scales, or
// BF16) it consumes in place as device tensor operands: a tile (or a part of
// one) of RowsPerTile rows. Queries are KV-head-major
// [kv head][row][query head in group][dimension], so the tile's fused rows
// form one contiguous M x D tensor. Each page is reached through its table
// entry (kv_extent.h).
// Three barriers per page order the score store, the softmax and the
// probability reads of PV.
// Int8Queries (INT8 K/V only) scores the tile's queries as INT8 codes
// against the INT8 keys, exact in int32 (|sum| < 2^22 at 256 dimensions);
// the codes live in the threadgroup's own partials slot, which it writes
// only after its last page.
// A tree's tile (SplashAttentionTree) masks each row to its path's rows of
// the block; its K/V, the store's, sit at the slots of their rows before
// any tile runs, so a row of a later tile reads its ancestors' from them.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint RowsPerTile, uint HeadDim,
          uint Window, typename CacheElement, bool Int8Queries>
inline void splash_paged_attention_tile(
    device bfloat *tile_queries, device const SplashKvPage *page_table,
    SplashKvLayer kv, uint kv_head, uint committed_tokens, uint active_rows, uint splits,
    uint split, device float *partials, device float *statistics, ulong slot,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    SplashAttentionTree tree, uint thread_index) {
  constexpr ushort M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr bool Quantized = is_same<CacheElement, int8_t>::value;
  constexpr ushort N = SplashKvPageTokens;
  constexpr ushort D = HeadDim;
  const bool in_tree = tree.rows != nullptr;
  uint visible_tokens = committed_tokens + active_rows;
  uint pages = splash_attention_pages(visible_tokens);
  uint first_page = 0;
  if constexpr (Window != 0)
    first_page = splash_attention_first_page<Window>(
        splash_attention_window_position(in_tree, tree.anchor, committed_tokens));
  uint per_split = splash_attention_pages_per_split(pages - first_page, splits);
  uint page_begin = first_page + split * per_split;
  if (page_begin >= pages)
    return;
  uint page_end = min(pages, page_begin + per_split);
  static_assert(Quantized || !Int8Queries, "INT8 queries take INT8 keys");
  using QueryElement = conditional_t<Int8Queries, int8_t, bfloat>;
  using Score = conditional_t<Int8Queries, int, float>;
  device QueryElement *queries;
  float score_scale = splash_attention_scale<HeadDim>();
  if constexpr (Int8Queries) {
    queries = reinterpret_cast<device int8_t *>(partials + slot * M * D);
    score_scale *= splash_attention_query_codes<M, HeadDim>(tile_queries, queries, thread_index);
  } else {
    queries = tile_queries;
  }
  if (thread_index < M) {
    row_max[thread_index] = -INFINITY;
    row_sum[thread_index] = 0.0f;
  }
  // A tree row's ancestors and depth, for its softmax lanes.
  uint ancestors = 0, depth = 0;
  if (in_tree && thread_index < SplashAttentionLanesPerRow * M) {
    const SplashTreeRow place =
        tree.rows[min(thread_index / SplashAttentionLanesPerRow / QueryHeadsPerKVHead, active_rows - 1)];
    ancestors = place.ancestors;
    depth = place.depth;
  }
  // The query codes are device writes the whole threadgroup reads.
  threadgroup_barrier(Int8Queries ? mem_flags::mem_threadgroup | mem_flags::mem_device
                                  : mem_flags::mem_threadgroup);

  threadgroup Score *page_score_tile = reinterpret_cast<threadgroup Score *>(scores);
  auto qt = tensor(queries, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto st = tensor(page_score_tile, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto pt = tensor(probabilities, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto p0 = pt.slice<N, M>(0, 0);
  auto key_type = tensor(static_cast<device CacheElement *>(nullptr),
                         dextents<int, 2>{D, N}, array<int, 2>{1, D});
  auto value_type = tensor(static_cast<device CacheElement *>(nullptr),
                           dextents<int, 2>{N, D}, array<int, 2>{1, N});
  const SplashKvAddressing<KVHeads, HeadDim, CacheElement> addressing(kv, kv_head);
  auto q0 = qt.template slice<D, M>(0, 0);
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
        decltype(q0), decltype(k0), Score>();
    // One full-dimension product initializes the score CT through MPP.
    auto ks = kt.template slice<D, N>(0, 0);
    qk.run(q0, ks, page_scores);
    page_scores.store(st.template slice<N, M>(0, 0));
    if (thread_index == 0)
      atomic_store_explicit(rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    splash_attention_page_softmax<QueryHeadsPerKVHead, RowsPerTile, Window, Quantized>(
        page_score_tile, score_scale, probabilities, row_max, row_sum, previous_scale, rescale,
        reinterpret_cast<device const float4 *>(key_scales),
        reinterpret_cast<device const float4 *>(value_scales), token_start,
        visible_tokens, committed_tokens, active_rows, in_tree, tree.anchor, ancestors, depth, thread_index);
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
// [slot][fused row]{max, sum}. The callers lay a sequence's slots out from its
// first_slot as [tile][KV head][part][split], a sparse tile being one part.
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
  static_assert(D >= SPLASH_ATTENTION_MAXIMUM_SPLITS, "a thread per split");
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
// is recomputed here, of the same Window and, for a tree's tile, from the
// same anchor. A part past the tile's active rows ran no split; its rows are
// written as zeros.
template <uint QueryHeadsPerKVHead, uint RowsPerTile, uint Parts, uint HeadDim,
          uint Window, class... GroupSinks>
inline void splash_attention_reduce_row(
    device const float *partials, device const float *statistics,
    device bfloat *output, SplashAttentionTile whole, uint slot_stride,
    bool tree, uint anchor, uint fused_row, uint thread_index, threadgroup float *weights,
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
    pages -= splash_attention_first_page<Window>(splash_attention_window_position(tree, anchor, tile.first_position));
  const uint per_split = splash_attention_pages_per_split(pages, tile.splits);
  const uint written = (pages + per_split - 1) / per_split;
  splash_attention_combine_splits<QueryHeadsPerKVHead, PartRows, HeadDim>(
      partials, statistics, output + tile.offset, tile.active_rows, written,
      tile.first_slot, fused_row % PartM, thread_index, weights, group_values,
      group_sinks...);
}

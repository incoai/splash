#pragma once

#include "metal/abi/Gguf.h"
#include "metal/abi/RelativeAttention.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/attention_sequence.h"
#include "metal/kernels/common/attention_store.h"
#include "metal/kernels/common/gguf_sgmatrix.h"
#include "metal/kernels/common/paged_attention_tile.h"
#include "metal/kernels/common/rms_inverse.h"

// Relative attention (ops::RelativeAttention, metal/abi/RelativeAttention.h):
// the paged attention of kernels/common/paged_attention_tile.h, over the same
// pages, tiles, splits and reduce, whose scores also take a bias of each
// key's backward distance mixed by the row's relative states, and whose
// softmax takes a sink of the row's own for each query head. Its page
// softmax, page loop, combine and output row are this file's, each written
// as the shared one is (the same order and names) with these terms added, so
// that the two read side by side; every other helper is the shared one.

// A projection row of relative attention, a row each of the command's packed
// rows (ops::RelativeAttentionRows): every query head's queries, then their
// output gates, their relative states and their sinks, each a block. The
// projection writing them has whole tiles of GGUF_TILE_COLUMNS outputs
// (ops::Linear), so its rows are Stride apart, the blocks' Width rounded up
// to a tile; the columns past Width are padding, which nothing reads.
template <uint QHeads, uint HeadDim> struct RelativeAttentionRow {
  static constexpr constant uint States = SPLASH_RELATIVE_ATTENTION_STATES;
  static constexpr constant uint GateOffset = QHeads * HeadDim;
  static constexpr constant uint StateOffset = 2 * QHeads * HeadDim;
  static constexpr constant uint SinkOffset = StateOffset + QHeads * States;
  static constexpr constant uint Width = SinkOffset + QHeads;
  static constexpr constant uint Stride = (Width + GGUF_TILE_COLUMNS - 1) / GGUF_TILE_COLUMNS * GGUF_TILE_COLUMNS;
};

// ---- Prepare --------------------------------------------------------------

// The sum of each thread's `value` over a threadgroup of Simdgroups
// simdgroups, in simdgroup order, returned to every thread; `reductions`
// holds a partial per simdgroup, which the threadgroup does not reuse.
template <ushort Simdgroups>
inline float relative_attention_sum(float value, threadgroup float *reductions, uint lane, uint simd_group) {
#pragma clang fp reassociate(off)
  value = simd_sum(value);
  if (lane == 0)
    reductions[simd_group] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint i = 0; i < Simdgroups; ++i)
    total += reductions[i];
  return total;
}

// The factor that clips a head whose squares sum to `squares` to an L2 norm
// of at most `clip`, in fp32 as the source's: min(1, clip / (|x| + epsilon)).
inline float relative_attention_clip(float squares, float clip) {
  return min(clip / (sqrt(squares) + SPLASH_RELATIVE_ATTENTION_CLIP_EPSILON), 1.0f);
}

// A position-scaled layer's factor of a row at `position`, in fp32 as the
// source's.
inline float relative_attention_log_scale(uint position) {
  return 1.0f + SPLASH_RELATIVE_ATTENTION_LOG_SCALE_ALPHA *
                    precise::log(max(float(position + 1) / SPLASH_RELATIVE_ATTENTION_LOG_SCALE_FLOOR, 1.0f));
}

// A threadgroup of HeadDim threads per (head, row, sequence) of a dispatch's
// sequences (SplashAttentionParams), as full_qkv_storage_phase
// (attention_qkv_prepare.h): query heads, then K heads, whose threadgroups
// also take the V head of their index; a row past its sequence's rows
// returns. A row's projection row is `projection` (RelativeAttentionRow),
// its convolved K and V rows `keys` and `values` (KHeads x HeadDim each), at
// its packed row.
// A query head's queries are scaled by query_scale ([query head][dimension]),
// clipped and rounded to bf16, as the source's projection returns them, and,
// PositionScaled, scaled again by the row's log scale and rounded; staged as
// the paged attention's ([KV head][row][query head in group][dimension]).
// Its relative states, the first simdgroup's lanes, are scaled by
// relative_scale ([query head][state]), clipped and scaled alike, and staged
// as F32 in the same order (States a fused row); its sink is staged as F32
// (one a fused row). A row's position is its sequence's committed tokens plus
// its row, or a tree row's (SplashAttentionParams::trees) its anchor's plus
// its depth.
// A K head's key is clipped, its V head's value normalized (times the inverse
// RMS of the head, unweighted), each rounded to bf16, and both stored in the
// row's page slot (splash_store_kv_slot) through its sequence's page table,
// before any tile reads them.
template <uint QHeads, uint KHeads, uint HeadDim, bool PositionScaled, typename CacheElement>
inline void relative_attention_prepare_phase(
    device const bfloat *projection, device const bfloat *keys, device const bfloat *values,
    device const float *query_scale, device const float *relative_scale, device bfloat *queries,
    device float *relative_states, device float *sinks, device const SplashKvPage *page_table,
    device const SplashTreeRow *tree, device const SplashLaneRows *layout, constant SplashAttentionParams &params,
    threadgroup float *reductions, threadgroup float *maxima, uint3 group, uint thread_index, uint lane,
    uint simd_group) {
  static_assert(QHeads % KHeads == 0);
  using Row = RelativeAttentionRow<QHeads, HeadDim>;
  constexpr uint Group = QHeads / KHeads, States = Row::States;
  constexpr ushort Simdgroups = HeadDim / 32;
  static_assert(States <= 32, "one simdgroup clips a head's relative states");
  const uint sequence_row = group.y;
  constant SplashAttentionSequence &sequence = params.sequence[group.z];
  const bool query = group.x < QHeads;
  const bool stores =
      !query && splash_attention_rows_valid(params, sequence) && sequence_row < sequence.row_bound;
  const SplashKvPage page = stores ? splash_store_page(page_table, sequence, sequence_row) : SPLASH_KV_PAGE_ABSENT;
  const SplashAttentionRows rows = splash_attention_rows(layout, params, group.z);
  if (sequence_row >= rows.rows || (!query && !stores))
    return;
  const ulong packed = ulong(rows.first) + sequence_row;
  if (!query) {
    const uint head = group.x - QHeads;
    const ulong element = packed * (KHeads * HeadDim) + head * HeadDim + thread_index;
    const float key = float(keys[element]), value = float(values[element]);
    const float clip = relative_attention_clip(relative_attention_sum<Simdgroups>(key * key, reductions, lane,
                                                                                 simd_group),
                                               SPLASH_RELATIVE_ATTENTION_QK_CLIP);
    const float inverse = rms_inverse_of_sums<Simdgroups>(value * value, HeadDim, reductions + Simdgroups,
                                                          thread_index, lane, simd_group);
    splash_store_kv_slot<KHeads, HeadDim, CacheElement, false>(page, sequence, params.kv, maxima, head,
                                                               sequence_row, bfloat(key * clip),
                                                               bfloat(value * inverse), thread_index, lane,
                                                               simd_group);
    return;
  }
  const uint head = group.x;
  float log_scale = 1.0f;
  if constexpr (PositionScaled)
    log_scale = relative_attention_log_scale(sequence.committed_tokens +
                                             (params.trees ? uint(tree[packed].depth) : sequence_row));
  device const bfloat *row = projection + packed * Row::Stride;
  const float element = float(row[head * HeadDim + thread_index]) * query_scale[head * HeadDim + thread_index];
  const float clip = relative_attention_clip(
      relative_attention_sum<Simdgroups>(element * element, reductions, lane, simd_group),
      SPLASH_RELATIVE_ATTENTION_QK_CLIP);
  bfloat result = bfloat(element * clip);
  if constexpr (PositionScaled)
    result = bfloat(float(result) * log_scale);
  const uint kv_head = head / Group, local_head = head % Group;
  const ulong fused =
      (ulong(sequence.stage_row) + ulong(kv_head) * sequence.chunk_stride + sequence_row) * Group + local_head;
  queries[fused * HeadDim + thread_index] = result;
  if (simd_group != 0)
    return;
  const float state =
      lane < States ? float(row[Row::StateOffset + head * States + lane]) * relative_scale[head * States + lane]
                    : 0.0f;
  bfloat clipped =
      bfloat(state * relative_attention_clip(simd_sum(state * state), SPLASH_RELATIVE_ATTENTION_STATE_CLIP));
  if constexpr (PositionScaled)
    clipped = bfloat(float(clipped) * log_scale);
  if (lane < States)
    relative_states[fused * States + lane] = float(clipped);
  if (lane == 0)
    sinks[fused] = float(row[Row::SinkOffset + head]);
}

// ---- Page loop ------------------------------------------------------------

// A softmax lane's bias: its fused row's relative states, in registers for
// the tile's whole page loop, the layer's table, and the row's position. Each
// of the lane's eight keys at a backward distance d below the extent adds the
// states' dot with table row d: 16 products a key in the lane's registers,
// against a table that stays in cache (64 KiB). A table of each row's 1024
// biases a head, written before the tiles, would cost 96 KiB a row (200 MB a
// 2048-row chunk) and a dispatch more an application. A key past the row
// wraps past the extent: it is masked anyway.
// A tree row's keys of its block sit at the slots of their rows, not of
// their positions: an ancestor at block row j is at depth j's place on the
// row's path, the count of its path's rows before j (metal/abi/VerifyTree.h:
// a path's rows ascend), so its distance is the row's depth less that count.
// `block` is a tree's anchor slot, and past every slot for any other tile.
struct RelativeAttentionLane {
  device const float4 *table;
  float4 state[SPLASH_RELATIVE_ATTENTION_STATES / 4];
  uint position;
  uint block;
  uint ancestors;
  uint depth;
};

inline void relative_attention_add_bias(thread float *score, uint first_key, RelativeAttentionLane lane) {
  static_assert(SPLASH_RELATIVE_ATTENTION_STATES == 16, "four float4 states a row");
#pragma unroll
  for (uint j = 0; j < 8; ++j) {
    const uint key = first_key + j;
    const uint distance =
        key >= lane.block ? lane.depth - popcount(lane.ancestors & ((1u << min(key - lane.block, 31u)) - 1u))
                          : lane.position - key;
    if (distance < SPLASH_RELATIVE_ATTENTION_EXTENT) {
      device const float4 *bias = lane.table + ulong(distance) * 4;
      score[j] += dot(lane.state[0], bias[0]) + dot(lane.state[1], bias[1]) + dot(lane.state[2], bias[2]) +
                  dot(lane.state[3], bias[3]);
    }
  }
}

// splash_attention_page_softmax with the bias: Biased, a page that holds a
// key within the extent of some row of the tile, adds the lane's to its
// scaled scores (after an INT8 page's key scales and the query codes' scale);
// a page below every row's extent takes none.
template <uint QueryHeadsPerKVHead, uint RowsPerTile, uint Window, bool Quantized, bool Masked, bool Biased,
          typename Score>
inline void relative_attention_page_softmax(
    threadgroup const Score *scores, float score_scale, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    device const float4 *key_scales, device const float4 *value_scales,
    uint token_start,
    uint visible_tokens, uint committed_tokens, uint active_rows,
    bool tree, uint anchor, uint ancestors, uint depth,
    RelativeAttentionLane relative, uint thread_index) {
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
  bool seen[TokensPerLane];
  if constexpr (Masked) {
    const uint query_row = fused_row / QueryHeadsPerKVHead;
    const uint causal_end =
        committed_tokens + min(query_row, active_rows - 1) + 1;
    const uint start = splash_attention_window_start<Window>(tree ? anchor + depth : causal_end - 1);
    const uint limit = min(visible_tokens, causal_end) - start;
    const uint token = token_start + column - start;
    uint off_path = 0;
    if (tree && token_start + N > anchor) {
#pragma unroll
      for (uint j = 0; j < TokensPerLane; ++j) {
        const uint slot = token_start + column + j;
        off_path |= uint(slot >= anchor && !((ancestors >> min(slot - anchor, 31u)) & 1u)) << j;
      }
    }
#pragma unroll
    for (uint j = 0; j < TokensPerLane; ++j)
      seen[j] = token + j < limit && !((off_path >> j) & 1u);
  } else {
#pragma unroll
    for (uint j = 0; j < TokensPerLane; ++j)
      seen[j] = true;
  }
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
  if constexpr (Biased)
    relative_attention_add_bias(score, token_start + column, relative);
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

// The page-softmax call of one page: Masked unless every row of the tile
// sees it wholly, Biased from the tile's first biased page on.
template <uint QueryHeadsPerKVHead, uint RowsPerTile, uint Window, bool Quantized, bool Biased, typename Score>
inline void relative_attention_page(
    bool masked, threadgroup const Score *scores, float score_scale, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum, threadgroup float *previous_scale,
    threadgroup atomic_uint *rescale, device const float *key_scales, device const float *value_scales,
    uint token_start, uint visible_tokens, uint committed_tokens, uint active_rows, bool tree, uint anchor,
    uint ancestors, uint depth, RelativeAttentionLane relative, uint thread_index) {
  if (masked)
    relative_attention_page_softmax<QueryHeadsPerKVHead, RowsPerTile, Window, Quantized, true, Biased>(
        scores, score_scale, probabilities, row_max, row_sum, previous_scale, rescale,
        reinterpret_cast<device const float4 *>(key_scales), reinterpret_cast<device const float4 *>(value_scales),
        token_start, visible_tokens, committed_tokens, active_rows, tree, anchor, ancestors, depth, relative,
        thread_index);
  else
    relative_attention_page_softmax<QueryHeadsPerKVHead, RowsPerTile, Window, Quantized, false, Biased>(
        scores, score_scale, probabilities, row_max, row_sum, previous_scale, rescale,
        reinterpret_cast<device const float4 *>(key_scales), reinterpret_cast<device const float4 *>(value_scales),
        token_start, visible_tokens, committed_tokens, active_rows, tree, anchor, ancestors, depth, relative,
        thread_index);
}

// The pages a tile's history split `split` of `splits` reads, as the split,
// the reduce and the single pass derive them: the balanced partition of the
// pages from its window's first (splash_attention_first_page, from its first
// row's position or a tree's anchor's) to its last row's, the split's run
// [begin, end), empty past the pages, and the runs that hold pages
// (`written`), whose partials a combine reads.
struct RelativeAttentionPages {
  uint begin;
  uint end;
  uint written;
};

template <uint Window>
inline RelativeAttentionPages relative_attention_pages(uint first_position, uint active_rows, bool tree, uint anchor,
                                                       uint splits, uint split) {
  const uint pages = splash_attention_pages(first_position + active_rows);
  uint first_page = 0;
  if constexpr (Window != 0)
    first_page = splash_attention_first_page<Window>(splash_attention_window_position(tree, anchor, first_position));
  const uint per_split = splash_attention_pages_per_split(pages - first_page, splits);
  const uint begin = first_page + split * per_split;
  return {begin, min(pages, begin + per_split), (pages - first_page + per_split - 1) / per_split};
}

// splash_paged_attention_tile with the bias, over a split's nonempty run of
// pages (relative_attention_pages): the tile's fused rows' relative states
// (`states`, staged as their queries) and the layer's bias table (`table`).
// Its softmax lanes load their rows' states and positions once; the pages
// from the first that holds a key within the extent of the tile's first row's
// position (a tree's anchor's, as its window's) take the bias, the pages
// below it, past every row's extent, none. Int8Queries writes the tile's
// query codes to `query_codes` (M x HeadDim bytes the threadgroup owns).
// Then `store` stores the running output of the tile's fused rows (an
// M x HeadDim cooperative tensor, its argument, every element of which
// running_full says is valid) and their statistics, which row_max and row_sum
// hold.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint RowsPerTile, uint HeadDim,
          uint Window, typename CacheElement, bool Int8Queries, class Store>
inline void relative_attention_page_loop(
    device bfloat *tile_queries, device int8_t *query_codes, device const float *states,
    device const float *table, device const SplashKvPage *page_table,
    SplashKvLayer kv, uint kv_head, uint committed_tokens, uint active_rows, RelativeAttentionPages run,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    SplashAttentionTree tree, uint thread_index, Store store) {
  constexpr ushort M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr bool Quantized = is_same<CacheElement, int8_t>::value;
  constexpr ushort N = SplashKvPageTokens;
  constexpr ushort D = HeadDim;
  constexpr uint States = SPLASH_RELATIVE_ATTENTION_STATES;
  const bool in_tree = tree.rows != nullptr;
  const uint visible_tokens = committed_tokens + active_rows;
  static_assert(Quantized || !Int8Queries, "INT8 queries take INT8 keys");
  using QueryElement = conditional_t<Int8Queries, int8_t, bfloat>;
  using Score = conditional_t<Int8Queries, int, float>;
  device QueryElement *queries;
  float score_scale = splash_attention_scale<HeadDim>();
  if constexpr (Int8Queries) {
    queries = query_codes;
    score_scale *= splash_attention_query_codes<M, HeadDim>(tile_queries, queries, thread_index);
  } else {
    queries = tile_queries;
  }
  if (thread_index < M) {
    row_max[thread_index] = -INFINITY;
    row_sum[thread_index] = 0.0f;
  }
  uint ancestors = 0, depth = 0;
  if (in_tree && thread_index < SplashAttentionLanesPerRow * M) {
    const SplashTreeRow place =
        tree.rows[min(thread_index / SplashAttentionLanesPerRow / QueryHeadsPerKVHead, active_rows - 1)];
    ancestors = place.ancestors;
    depth = place.depth;
  }
  // The lane's fused row's states and position, a row past the tile's active
  // rows its last's, as its depth.
  RelativeAttentionLane lane{};
  if (thread_index < SplashAttentionLanesPerRow * M) {
    const uint fused_row = thread_index / SplashAttentionLanesPerRow;
    const uint row = min(fused_row / QueryHeadsPerKVHead, active_rows - 1);
    device const float4 *row_states = reinterpret_cast<device const float4 *>(
        states + (row * QueryHeadsPerKVHead + fused_row % QueryHeadsPerKVHead) * States);
#pragma unroll
    for (uint i = 0; i < States / 4; ++i)
      lane.state[i] = row_states[i];
    lane.table = reinterpret_cast<device const float4 *>(table);
    lane.position = in_tree ? tree.anchor + depth : committed_tokens + row;
    lane.block = in_tree ? tree.anchor : 0xffffffffu;
    lane.ancestors = ancestors;
    lane.depth = depth;
  }
  // The first page that holds a key within the extent of the tile's first
  // row: keys from its position less Extent - 1 on.
  const uint biased_page = splash_attention_first_page<SPLASH_RELATIVE_ATTENTION_EXTENT - 1>(
      splash_attention_window_position(in_tree, tree.anchor, committed_tokens));
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
  const bool running_full =
      uint(running.get_capacity()) * (8u * 32u) == uint(M) * D;
#pragma unroll
  for (ushort index = 0; index < running.get_capacity(); ++index) {
    if (running_full || running.is_valid_element(index))
      running[index] = 0.0f;
  }

  const SplashAttentionWholePages whole =
      splash_attention_whole_pages<Window>(committed_tokens, active_rows, in_tree, tree.anchor);
  for (uint page = run.begin; page < run.end; ++page) {
    const SplashKvPageTensors<CacheElement> tensors =
        addressing.page(page_table[page]);
    uint token_start = page * N;
    auto kt = tensor(tensors.keys, dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto vt = tensor(tensors.values, dextents<int, 2>{N, D}, array<int, 2>{1, N});

    auto page_scores = qk.template get_destination_cooperative_tensor<
        decltype(q0), decltype(k0), Score>();
    auto ks = kt.template slice<D, N>(0, 0);
    qk.run(q0, ks, page_scores);
    page_scores.store(st.template slice<N, M>(0, 0));
    if (thread_index == 0)
      atomic_store_explicit(rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool masked = !(page >= whole.begin && page < whole.end);
    if (page >= biased_page)
      relative_attention_page<QueryHeadsPerKVHead, RowsPerTile, Window, Quantized, true>(
          masked, page_score_tile, score_scale, probabilities, row_max, row_sum, previous_scale, rescale,
          tensors.key_scales, tensors.value_scales, token_start, visible_tokens, committed_tokens, active_rows,
          in_tree, tree.anchor, ancestors, depth, lane, thread_index);
    else
      relative_attention_page<QueryHeadsPerKVHead, RowsPerTile, Window, Quantized, false>(
          masked, page_score_tile, score_scale, probabilities, row_max, row_sum, previous_scale, rescale,
          tensors.key_scales, tensors.value_scales, token_start, visible_tokens, committed_tokens, active_rows,
          in_tree, tree.anchor, ancestors, depth, lane, thread_index);
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
  store(running, running_full);
}

// ---- Reduce ---------------------------------------------------------------

// splash_attention_combine_splits with the row's own sink, `sink` (the
// scores' scale is not applied to it): one more key with no value, in the
// maximum and, after the splits in their order, the denominator.
template <uint QueryHeadsPerKVHead, uint RowsPerTile, uint HeadDim>
inline bfloat relative_attention_combine_splits(
    device const float *partials, device const float *statistics, uint active_rows, uint written,
    ulong head_slot, uint fused_row, uint thread_index,
    threadgroup float *weights, threadgroup float *group_values, float sink) {
  constexpr uint M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr uint D = HeadDim;
  static_assert(D >= SPLASH_ATTENTION_MAXIMUM_SPLITS, "a thread per split");
  if (fused_row / QueryHeadsPerKVHead >= active_rows)
    return bfloat(0.0f);
  const uint lane = thread_index % 32, sg = thread_index / 32;
  const ulong stat = ((head_slot + thread_index) * M + fused_row) * 2;
  const float maximum = thread_index < written ? statistics[stat] : -INFINITY;
  const float group_maximum = simd_max(maximum);
  if (lane == 0) group_values[sg] = group_maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float split_maximum = simd_max(lane < D / 32 ? group_values[lane] : -INFINITY);
  const float row_maximum = max(split_maximum, sink);
  if (thread_index < written)
    weights[thread_index] = fast::exp(maximum - row_maximum);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float numerator = 0.0f, denominator = 0.0f;
  // Both accumulations in the original split order, as the shared combine's.
  for (uint split = 0; split < written; ++split) {
    const float weight = weights[split];
    const ulong stat = ((head_slot + split) * M + fused_row) * 2;
    numerator += weight *
        partials[((head_slot + split) * M + fused_row) * D + thread_index];
    denominator += weight * statistics[stat + 1];
  }
  denominator += fast::exp(sink - row_maximum);
  return bfloat(denominator > 0.0f ? numerator / denominator : 0.0f);
}

// The attention's output of one fused row, as splash_attention_output_row
// (attention_gate.h) writes a gated one: `value` times the sigmoid of its
// query head's gate in its projection row (RelativeAttentionRow's gate
// block), at row `output` of `hidden` and, with Table, of the
// out-projection's Table16 input and its sums.
template <uint QHeads, uint HeadDim, bool Table>
inline void relative_attention_output_row(bfloat value, device const bfloat *projection, device bfloat *hidden,
                                          device bfloat *table, device float *sums, ulong packed, ulong output,
                                          uint query_head, uint dimension, bool writes, threadgroup bfloat *pairs) {
  using Row = RelativeAttentionRow<QHeads, HeadDim>;
  constexpr uint Width = QHeads * HeadDim;
  static_assert(HeadDim % 64 == 0, "a head's elements fill whole simdgroups of pairs");
  bfloat element = value;
  if (writes) {
    const float gate = float(projection[packed * Row::Stride + Row::GateOffset + query_head * HeadDim + dimension]);
    element = bfloat(float(value) * splash_sigmoid(gate));
    hidden[output * Width + query_head * HeadDim + dimension] = element;
  }
  if constexpr (Table) {
    pairs[dimension] = element;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (writes && dimension < HeadDim / 2) {
      const uint first = query_head * HeadDim + 2 * dimension;
      gguf_sg::Table16::write(table + output / 8 * Width * 8,
                              sums + output / 8 * gguf_sg::Table16::sums_per_tile(Width), Width, first / 64,
                              output % 8, dimension % 32, pairs[2 * dimension], pairs[2 * dimension + 1]);
    }
  }
}

// ---- Single pass ----------------------------------------------------------

// A decode step's attention of a layer in one dispatch (ops::RelativeAttention
// ::addDecode, relative_attention_decode_phase): each threadgroup of a lane's
// tile prepares what its split reads itself, then runs the split's page loop,
// and the split that finishes last combines the tile's splits as the reduce
// does. These compute every value as the prepare and the reduce do, in the same
// order, a simdgroup a head where they spread a head over a threadgroup: lane l
// holds elements l + 32 i of a head, whose squares it adds a block of 32 at a
// time (simd_sum, the prepare's per-simdgroup partial), block after block.

// The prepare of a query head's row (relative_attention_prepare_phase's query
// threadgroup) by one simdgroup: its queries, scaled, clipped and rounded,
// and, PositionScaled, scaled by the log scale of the row's `position` and
// rounded again, to `queries` (HeadDim bf16), and its relative states to
// `states` (F32, the first States lanes'), from the row's projection row
// `row` (RelativeAttentionRow).
template <uint QHeads, uint HeadDim, bool PositionScaled>
inline void relative_attention_stage_head(device const bfloat *row, device const float *query_scale,
                                          device const float *relative_scale, uint head, uint position,
                                          device bfloat *queries, device float *states, uint lane) {
#pragma clang fp reassociate(off)
  using Row = RelativeAttentionRow<QHeads, HeadDim>;
  constexpr uint Blocks = HeadDim / 32, States = Row::States;
  float element[Blocks];
  float squares = 0.0f;
#pragma unroll
  for (uint i = 0; i < Blocks; ++i) {
    const uint dimension = head * HeadDim + 32 * i + lane;
    element[i] = float(row[dimension]) * query_scale[dimension];
    squares += simd_sum(element[i] * element[i]);
  }
  const float clip = relative_attention_clip(squares, SPLASH_RELATIVE_ATTENTION_QK_CLIP);
  float log_scale = 1.0f;
  if constexpr (PositionScaled)
    log_scale = relative_attention_log_scale(position);
#pragma unroll
  for (uint i = 0; i < Blocks; ++i) {
    bfloat result = bfloat(element[i] * clip);
    if constexpr (PositionScaled)
      result = bfloat(float(result) * log_scale);
    queries[32 * i + lane] = result;
  }
  const float state =
      lane < States ? float(row[Row::StateOffset + head * States + lane]) * relative_scale[head * States + lane]
                    : 0.0f;
  bfloat clipped =
      bfloat(state * relative_attention_clip(simd_sum(state * state), SPLASH_RELATIVE_ATTENTION_STATE_CLIP));
  if constexpr (PositionScaled)
    clipped = bfloat(float(clipped) * log_scale);
  if (lane < States)
    states[lane] = float(clipped);
}

// The prepare of a K head's row and its V head's (relative_attention_prepare_
// phase's K threadgroup) by one simdgroup: the key clipped and the value
// normalized, each rounded to bf16, from the row's convolved K and V heads at
// `element` of `keys` and `values`, stored in slot `token` of the page entry
// `page` as splash_store_kv_slot stores them (INT8 codes at the row's largest
// magnitude of each, and their scales).
template <uint KHeads, uint HeadDim, typename CacheElement>
inline void relative_attention_store_head(device const bfloat *keys, device const bfloat *values, ulong element,
                                          SplashKvPage page, SplashKvLayer kv, uint head, uint token, uint lane) {
#pragma clang fp reassociate(off)
  constexpr uint Blocks = HeadDim / 32;
  float key[Blocks], value[Blocks];
  float key_squares = 0.0f, value_squares = 0.0f;
#pragma unroll
  for (uint i = 0; i < Blocks; ++i) {
    key[i] = float(keys[element + 32 * i + lane]);
    value[i] = float(values[element + 32 * i + lane]);
    key_squares += simd_sum(key[i] * key[i]);
    value_squares += simd_sum(value[i] * value[i]);
  }
  const float clip = relative_attention_clip(key_squares, SPLASH_RELATIVE_ATTENTION_QK_CLIP);
  const float inverse = rsqrt(value_squares / HeadDim + kRmsEpsilon);
  bfloat clipped[Blocks], normalized[Blocks];
#pragma unroll
  for (uint i = 0; i < Blocks; ++i) {
    clipped[i] = bfloat(key[i] * clip);
    normalized[i] = bfloat(value[i] * inverse);
  }
  const SplashKvPageTensors<CacheElement> slab = SplashKvAddressing<KHeads, HeadDim, CacheElement>(kv, head).page(page);
  if constexpr (is_same<CacheElement, bfloat>::value) {
#pragma unroll
    for (uint i = 0; i < Blocks; ++i) {
      slab.keys[splash_kv_key_index<KHeads, HeadDim>(0, 0, token, 32 * i + lane)] = clipped[i];
      slab.values[splash_kv_value_index<KHeads, HeadDim>(0, 0, token, 32 * i + lane)] = normalized[i];
    }
  } else {
    float2 maximum = 0.0f;
#pragma unroll
    for (uint i = 0; i < Blocks; ++i)
      maximum = max(maximum, abs(float2(float(clipped[i]), float(normalized[i]))));
    maximum = float2(simd_max(maximum.x), simd_max(maximum.y));
#pragma unroll
    for (uint i = 0; i < Blocks; ++i) {
      slab.keys[splash_kv_key_index<KHeads, HeadDim>(0, 0, token, 32 * i + lane)] =
          splash_store_code(float(clipped[i]), maximum.x);
      slab.values[splash_kv_value_index<KHeads, HeadDim>(0, 0, token, 32 * i + lane)] =
          splash_store_code(float(normalized[i]), maximum.y);
    }
    if (lane == 0) {
      const ulong scale_index = splash_q8_scale_index<KHeads>(0, 0, token);
      slab.key_scales[scale_index] = maximum.x == 0.0f ? 0.0f : maximum.x / 127.0f;
      slab.value_scales[scale_index] = maximum.y == 0.0f ? 0.0f : maximum.y / 127.0f;
    }
  }
}

// relative_attention_combine_splits of a tile's fused rows by the split that
// arrives last, in two phases over the published splits' partials and
// statistics (split_reduce.h), read coherently. First a thread a fused row:
// its maximum over its splits' and its sink, each split's weight into
// `weights` ([fused row][split]) and its denominator, the splits' in their
// order and then the sink's, into `denominators`. Then, past a barrier, a
// simdgroup a fused row, a lane each four of its dimensions: their
// numerators, the splits' in their order, whose loads a lane issues
// CombineSplits splits at a time before it adds them, and their quotients,
// as the reduce's. A simdgroup that combined two dimensions of a row at a
// time, its lanes holding the splits' weights, had waited on each split's
// load of each of its six rows of an 8-row tile in turn: a verify step's
// layer took 10% longer (8 rows) and 12% (16), a 1-row step's 6% less
// (20-core M5 Pro).
template <uint QueryHeadsPerKVHead, uint RowsPerTile, uint HeadDim>
inline void relative_attention_combine_weights(device coherent(device) const float *statistics, uint written,
                                               ulong head_slot, uint fused_row, float sink, threadgroup float *weights,
                                               threadgroup float *denominators) {
  constexpr uint M = RowsPerTile * QueryHeadsPerKVHead;
  float maximum = sink;
  for (uint split = 0; split < written; ++split)
    maximum = max(maximum, statistics[((head_slot + split) * M + fused_row) * 2]);
  float denominator = 0.0f;
  for (uint split = 0; split < written; ++split) {
    const ulong stat = ((head_slot + split) * M + fused_row) * 2;
    const float weight = fast::exp(statistics[stat] - maximum);
    weights[fused_row * written + split] = weight;
    denominator += weight * statistics[stat + 1];
  }
  denominators[fused_row] = denominator + fast::exp(sink - maximum);
}

template <uint QueryHeadsPerKVHead, uint RowsPerTile, uint HeadDim>
inline float4 relative_attention_combine_row(device coherent(device) const float *partials, uint written,
                                             ulong head_slot, uint fused_row, uint dimension,
                                             threadgroup const float *weights, float denominator) {
  constexpr uint M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr uint D = HeadDim;
  constexpr uint CombineSplits = 4;
  float4 numerator = 0.0f;
  for (uint first = 0; first < written; first += CombineSplits) {
    float4 partial[CombineSplits];
#pragma unroll
    for (uint i = 0; i < CombineSplits; ++i)
      partial[i] = first + i < written ? *reinterpret_cast<device coherent(device) const float4 *>(
                                             partials + ((head_slot + first + i) * M + fused_row) * D + dimension)
                                       : float4(0.0f);
#pragma unroll
    for (uint i = 0; i < CombineSplits && first + i < written; ++i) {
      const float weight = weights[fused_row * written + first + i];
      numerator.x += weight * partial[i].x;
      numerator.y += weight * partial[i].y;
      numerator.z += weight * partial[i].z;
      numerator.w += weight * partial[i].w;
    }
  }
  return denominator > 0.0f ? float4(numerator.x / denominator, numerator.y / denominator,
                                     numerator.z / denominator, numerator.w / denominator)
                            : float4(0.0f);
}

// relative_attention_output_row of a fused row's four dimensions from
// `dimension` on, a lane of the row's simdgroup each: their values (the
// combine's quotients) rounded, times the sigmoids of their gates, at the
// row's output place `place` of `hidden` and, with Table, of the
// out-projection's Table16 input, which the simdgroup writes a span of 64 at
// a time, lane l the span's elements 2l and 2l + 1, from the lanes that hold
// them.
template <uint QHeads, uint HeadDim, bool Table>
inline void relative_attention_output_quad(float4 value, device const bfloat *projection, device bfloat *hidden,
                                           device bfloat *table, device float *sums, SplashAttentionOutput place,
                                           uint query_head, uint dimension, uint lane) {
  using Row = RelativeAttentionRow<QHeads, HeadDim>;
  constexpr uint Width = QHeads * HeadDim;
  static_assert(HeadDim == 128, "a lane's four dimensions of a simdgroup's row");
  if (!place.writes)
    return;
  const bfloat4 gates = *reinterpret_cast<device const bfloat4 *>(projection + place.packed * Row::Stride +
                                                                   Row::GateOffset + query_head * HeadDim + dimension);
  const bfloat4 element(bfloat(float(bfloat(value.x)) * splash_sigmoid(float(gates.x))),
                        bfloat(float(bfloat(value.y)) * splash_sigmoid(float(gates.y))),
                        bfloat(float(bfloat(value.z)) * splash_sigmoid(float(gates.z))),
                        bfloat(float(bfloat(value.w)) * splash_sigmoid(float(gates.w))));
  *reinterpret_cast<device bfloat4 *>(hidden + place.output * Width + query_head * HeadDim + dimension) = element;
  if constexpr (Table)
#pragma unroll
    for (uint span = 0; span < 2; ++span) {
      const bfloat4 held = as_type<bfloat4>(simd_shuffle(as_type<uint2>(element), ushort(span * 16 + lane / 2)));
      gguf_sg::Table16::write(table + place.output / 8 * Width * 8,
                              sums + place.output / 8 * gguf_sg::Table16::sums_per_tile(Width), Width,
                              (query_head * HeadDim) / 64 + span, place.output % 8, lane,
                              lane % 2 ? held.z : held.x, lane % 2 ? held.w : held.y);
    }
}

#pragma once

#include "metal/kernels/common/paged_attention_tile.h"

// Sparse paged attention: a tile of RowsPerTile query rows of one KV head
// attends only to the blocks of SPLASH_ATTENTION_BLOCK_TOKENS tokens its block
// list names, each entry a block and the tile's rows that attend it
// (metal/abi/PagedAttention.h), or, without a list, to every block of its
// history. Its KV layout keeps values token-major like keys
// (kv::Layout::valuesTokenMajor), so each block's keys and values are a
// contiguous run of its page. A step gathers SplashSparseStep::Blocks blocks'
// keys into threadgroup memory, a tile of Tokens tokens, and runs the same
// MPP QK as the page loop on it; the softmax masks each row's tokens in
// blocks it does not attend and past its causal end; the step's values then
// take the keys' place for PV. Splits share the list out in equal runs of
// whole partition units.

// The list entry of a step's slot past the list's end.
constant uint SplashSparseNoEntry = 0xffffffffu;
// Splits take runs of whole units of this many blocks, a multiple of every
// format's step, so that the split and reduce kernels of either format share
// one partition.
constant uint SplashSparsePartitionBlocks = 8;
// A split takes at least this many entries of a long list: it writes
// partials of every fused row of the tile, which the reduce reads back, and
// fewer blocks per split would move more partials than KV. A short list
// (one below the QSA budget, or a lane's without one) spreads over at least
// SplashSparseMinimumSplits splits of a partition unit each instead, as the
// page loop's history does, so that a single lane's few threadgroups do not
// leave the GPU idle.
constant uint SplashSparseSplitBlocks = SPLASH_SPARSE_SPLIT_BLOCKS;
constant uint SplashSparseMinimumSplits = SPLASH_SPARSE_MINIMUM_SPLITS;
constant uint SplashSparseBlockTokens = SPLASH_ATTENTION_BLOCK_TOKENS;

template <typename CacheElement> struct SplashSparseStep {
  // A step's staged keys or values take 8 KiB in either format, which leaves
  // the 96-row tile's scores and probabilities room in 32 KiB: 32 INT8 tokens
  // or 16 BF16 ones.
  static constexpr constant uint Blocks =
      is_same<CacheElement, int8_t>::value ? 8 : 4;
  static constexpr constant uint Tokens = Blocks * SplashSparseBlockTokens;
};

static_assert(SplashSparsePartitionBlocks % SplashSparseStep<int8_t>::Blocks == 0 &&
                  SplashSparsePartitionBlocks % SplashSparseStep<bfloat>::Blocks == 0,
              "a partition unit is whole steps of either format");
static_assert(SplashKvPageTokens % SplashSparseBlockTokens == 0,
              "a page holds whole blocks");

// The entries one split takes of a list of `count`, of at most `splits`
// splits: an equal share in whole partition units, of at least
// SplashSparseSplitBlocks entries, over at least SplashSparseMinimumSplits
// splits where the list has as many units. It depends on the list alone, so
// a lane's partition is its own whatever lanes run with it. Every split past
// the written ones takes none.
inline uint splash_sparse_split_entries(uint count, uint splits) {
  const uint units = (count + SplashSparsePartitionBlocks - 1) / SplashSparsePartitionBlocks;
  const uint used = clamp(max((count + SplashSparseSplitBlocks - 1) / SplashSparseSplitBlocks,
                              min(SplashSparseMinimumSplits, units)),
                          1u, splits);
  const uint share = (count + used - 1) / used;
  return (share + SplashSparsePartitionBlocks - 1) / SplashSparsePartitionBlocks *
         SplashSparsePartitionBlocks;
}

inline uint splash_sparse_written_splits(uint count, uint splits) {
  const uint per_split = splash_sparse_split_entries(count, splits);
  return per_split ? (count + per_split - 1) / per_split : 0;
}

// The blocks a tile's rows see without a list: every block up to the last
// row's token.
inline uint splash_sparse_history_blocks(uint first_position, uint active_rows) {
  return (first_position + active_rows + SplashSparseBlockTokens - 1) /
         SplashSparseBlockTokens;
}

// The scratch of a sparse tile: the step's list entries and their pages, its
// staged keys or values [token][dimension] and, for INT8, their scales, and
// the page loop's scores, probabilities and row statistics.
template <typename CacheElement, uint M> struct SplashSparseScratch {
  static constexpr constant uint Blocks = SplashSparseStep<CacheElement>::Blocks;
  static constexpr constant uint Tokens = SplashSparseStep<CacheElement>::Tokens;
  uint entries[Blocks];
  SplashKvPage pages[Blocks];
  alignas(16) CacheElement stage[Tokens * SplashKvHeadDimension];
  alignas(16) float key_scales[Tokens];
  alignas(16) float value_scales[Tokens];
  alignas(16) float scores[M * Tokens];
  alignas(16) bfloat probabilities[M * Tokens];
  float row_max[M];
  float row_sum[M];
  float previous_scale[M];
  atomic_uint rescale;
};

// Copies one tensor's rows of the step's tokens (keys or values, both
// token-major) into the stage, 16 bytes a thread at a time; tokens of a slot
// past the list's end are zero, so that their products are too.
template <uint KVHeads, typename CacheElement, uint Tokens>
inline void splash_sparse_gather(
    threadgroup CacheElement *stage, threadgroup const uint *entries,
    threadgroup const SplashKvPage *pages,
    const SplashKvAddressing<KVHeads, CacheElement> addressing, bool values,
    uint thread_index) {
  constexpr uint D = SplashKvHeadDimension;
  constexpr uint RowChunks = D * sizeof(CacheElement) / 16;
  for (uint chunk = thread_index; chunk < Tokens * RowChunks; chunk += 256) {
    const uint token = chunk / RowChunks;
    const uint slot = token / SplashSparseBlockTokens;
    uint4 data = 0;
    if (entries[slot] != SplashSparseNoEntry) {
      const uint block = entries[slot] >> SPLASH_ATTENTION_BLOCK_ROW_BITS;
      const uint page_token =
          block % (SplashKvPageTokens / SplashSparseBlockTokens) * SplashSparseBlockTokens +
          token % SplashSparseBlockTokens;
      const SplashKvPageTensors<CacheElement> tensors = addressing.page(pages[slot]);
      device const CacheElement *row =
          (values ? tensors.values : tensors.keys) + ulong(page_token) * D;
      data = reinterpret_cast<device const uint4 *>(row)[chunk % RowChunks];
    }
    reinterpret_cast<threadgroup uint4 *>(stage + token * D)[chunk % RowChunks] = data;
  }
}

// One softmax lane of a step: eight consecutive tokens (two blocks) of one
// fused row, whose key-scaled scores become value-scaled bf16 probabilities
// as in splash_attention_page_softmax. A token counts when the row's
// bit of its block's entry is set and it is not past the row's causal end;
// rows past the tile's active ones mirror the last active row.
template <uint QueryHeadsPerKVHead, uint Tokens, bool Quantized>
inline void splash_sparse_softmax_lane(
    threadgroup const float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    threadgroup const float *key_scales, threadgroup const float *value_scales,
    threadgroup const uint *entries, uint first_position, uint active_rows,
    uint lane_index) {
  constexpr uint TokensPerLane = 8;
  constexpr uint LanesPerRow = Tokens / TokensPerLane;
  static_assert(TokensPerLane == 2 * SplashSparseBlockTokens, "a lane's tokens are two blocks");
  static_assert(LanesPerRow == 2 || LanesPerRow == 4, "a row's lanes shuffle as a pair or a quad");
  const uint fused_row = lane_index / LanesPerRow;
  const uint column = lane_index % LanesPerRow * TokensPerLane;
  const uint row = min(fused_row / QueryHeadsPerKVHead, active_rows - 1);
  const uint causal_end = first_position + row + 1;
  uint allowed = 0;
#pragma unroll
  for (uint slot = 0; slot < 2; ++slot) {
    const uint entry = entries[column / SplashSparseBlockTokens + slot];
    if (entry == SplashSparseNoEntry || !((entry >> row) & 1u))
      continue;
    const uint first = (entry >> SPLASH_ATTENTION_BLOCK_ROW_BITS) * SplashSparseBlockTokens;
#pragma unroll
    for (uint t = 0; t < SplashSparseBlockTokens; ++t)
      if (first + t < causal_end)
        allowed |= 1u << (slot * SplashSparseBlockTokens + t);
  }
  float score[TokensPerLane];
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    float value = scores[fused_row * Tokens + column + j];
    if constexpr (Quantized)
      value *= key_scales[column + j];
    score[j] = (allowed >> j) & 1u ? value * 0.0625f : -INFINITY;
  }
  float local_max = -INFINITY;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j)
    local_max = max(local_max, score[j]);
  local_max = max(local_max, simd_shuffle_xor(local_max, 1));
  if (LanesPerRow == 4)
    local_max = max(local_max, simd_shuffle_xor(local_max, 2));
  const float previous_max = row_max[fused_row];
  const float next_max = max(previous_max, local_max);
  float probability[TokensPerLane];
  float local_sum = 0.0f;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    probability[j] = (allowed >> j) & 1u ? fast::exp(score[j] - next_max) : 0.0f;
    local_sum += probability[j];
  }
  local_sum += simd_shuffle_xor(local_sum, 1);
  if (LanesPerRow == 4)
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
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    float weighted = probability[j];
    if constexpr (Quantized)
      weighted = (allowed >> j) & 1u ? weighted * value_scales[column + j] : 0.0f;
    probabilities[fused_row * Tokens + column + j] = bfloat(weighted);
  }
}

// One tile over its split's run of its block list (`list`, `count` entries),
// or with a null list over its history's blocks, writing the split's partials
// and statistics as the page loop does. The tile's rows are at first_position
// on; queries are its fused rows [row][query head][dimension] of one KV head.
// An entry past the tile's history counts as none, so no list makes the tile
// read past its page table. Five barriers a step order the step's entries
// and pages before the gather; the staged keys and scales before QK; the
// scores before the softmax, and QK's reads of the keys before the values
// replace them; the probabilities and values before PV; and PV's reads before
// the next step's entries and stage.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint RowsPerTile,
          typename CacheElement>
inline void splash_sparse_attention_tile(
    device bfloat *tile_queries, device const SplashKvPage *page_table,
    SplashKvLayer kv, uint kv_head, uint first_position, uint active_rows,
    device const uint *list, uint count, uint splits, uint split,
    device float *partials, device float *statistics, ulong slot,
    threadgroup SplashSparseScratch<CacheElement, RowsPerTile * QueryHeadsPerKVHead> &scratch,
    uint thread_index) {
  constexpr ushort M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr ushort N = SplashSparseStep<CacheElement>::Tokens;
  constexpr uint Blocks = SplashSparseStep<CacheElement>::Blocks;
  constexpr ushort D = SplashKvHeadDimension;
  constexpr bool Quantized = is_same<CacheElement, int8_t>::value;
  constexpr uint AllRows = (1u << RowsPerTile) - 1u;
  static_assert(RowsPerTile <= SPLASH_ATTENTION_BLOCK_ROW_BITS, "an entry has a bit per row");
  const uint per_split = splash_sparse_split_entries(count, splits);
  const uint begin = split * per_split;
  if (begin >= count)
    return;
  const uint end = min(count, begin + per_split);
  const uint history = splash_sparse_history_blocks(first_position, active_rows);
  if (thread_index < M) {
    scratch.row_max[thread_index] = -INFINITY;
    scratch.row_sum[thread_index] = 0.0f;
  }

  auto qt = tensor(tile_queries, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto st = tensor(scratch.scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto pt = tensor(scratch.probabilities, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  // The stage holds the step's keys, then its values, both [token][dimension].
  tensor<threadgroup CacheElement, dextents<int, 2>, tensor_inline> kv_stage(
      scratch.stage, dextents<int, 2>{D, N}, array<int, 2>{1, D});
  auto q0 = qt.slice<D, M>(0, 0);
  auto p0 = pt.template slice<N, M>(0, 0);
  auto stage0 = kv_stage.template slice<D, N>(0, 0);
  const SplashKvAddressing<KVHeads, CacheElement> addressing(kv, kv_head);
  constexpr auto qk_descriptor =
      matmul2d_descriptor(M, N, D, false, true, false,
                          matmul2d_descriptor::mode::multiply);
  constexpr auto pv_descriptor =
      matmul2d_descriptor(M, D, N, false, false, true,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
  matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
  auto running = pv.template get_destination_cooperative_tensor<
      decltype(p0), decltype(stage0), float>();
  const bool running_full =
      uint(running.get_capacity()) * (8u * 32u) == uint(M) * D;
#pragma unroll
  for (ushort index = 0; index < running.get_capacity(); ++index) {
    if (running_full || running.is_valid_element(index))
      running[index] = 0.0f;
  }

  for (uint step = begin; step < end; step += Blocks) {
    if (thread_index < Blocks) {
      const uint index = step + thread_index;
      uint entry = SplashSparseNoEntry;
      if (index < end)
        entry = list ? list[index] : index << SPLASH_ATTENTION_BLOCK_ROW_BITS | AllRows;
      if ((entry >> SPLASH_ATTENTION_BLOCK_ROW_BITS) >= history)
        entry = SplashSparseNoEntry;
      scratch.entries[thread_index] = entry;
      scratch.pages[thread_index] =
          entry == SplashSparseNoEntry
              ? SplashKvPage(0)
              : page_table[(entry >> SPLASH_ATTENTION_BLOCK_ROW_BITS) /
                           (SplashKvPageTokens / SplashSparseBlockTokens)];
    }
    if (thread_index == 0)
      atomic_store_explicit(&scratch.rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    splash_sparse_gather<KVHeads, CacheElement, N>(scratch.stage, scratch.entries, scratch.pages,
                                                    addressing, false, thread_index);
    if constexpr (Quantized) {
      if (thread_index < N) {
        const uint slot_index = thread_index / SplashSparseBlockTokens;
        const uint entry = scratch.entries[slot_index];
        float key_scale = 0.0f, value_scale = 0.0f;
        if (entry != SplashSparseNoEntry) {
          const SplashKvPageTensors<CacheElement> tensors = addressing.page(scratch.pages[slot_index]);
          const uint page_token =
              (entry >> SPLASH_ATTENTION_BLOCK_ROW_BITS) %
                  (SplashKvPageTokens / SplashSparseBlockTokens) * SplashSparseBlockTokens +
              thread_index % SplashSparseBlockTokens;
          key_scale = tensors.key_scales[page_token];
          value_scale = tensors.value_scales[page_token];
        }
        scratch.key_scales[thread_index] = key_scale;
        scratch.value_scales[thread_index] = value_scale;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto step_scores = qk.template get_destination_cooperative_tensor<
        decltype(q0), decltype(stage0), float>();
    qk.run(q0, stage0, step_scores);
    step_scores.store(st.template slice<N, M>(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The values replace the keys in the stage while the softmax reads only
    // the scores and scales.
    splash_sparse_gather<KVHeads, CacheElement, N>(scratch.stage, scratch.entries, scratch.pages,
                                                    addressing, true, thread_index);
    for (uint lane = thread_index; lane < M * N / 8; lane += 256)
      splash_sparse_softmax_lane<QueryHeadsPerKVHead, N, Quantized>(
          scratch.scores, scratch.probabilities, scratch.row_max, scratch.row_sum,
          scratch.previous_scale, &scratch.rescale, scratch.key_scales,
          scratch.value_scales, scratch.entries, first_position, active_rows, lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(&scratch.rescale, memory_order_relaxed)) {
#pragma unroll
      for (ushort index = 0; index < running.get_capacity(); ++index) {
        if (!running_full && !running.is_valid_element(index))
          continue;
        auto coordinates = running.get_multidimensional_index(index);
        running[index] *= scratch.previous_scale[coordinates[1]];
      }
    }
    pv.run(p0, stage0, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  auto target = tensor(partials + slot * M * D, dextents<int, 2>{D, M},
                       array<int, 2>{1, D});
  running.store(target.slice<D, M>(0, 0));
  if (thread_index < M) {
    statistics[(slot * M + thread_index) * 2] = scratch.row_max[thread_index];
    statistics[(slot * M + thread_index) * 2 + 1] = scratch.row_sum[thread_index];
  }
}

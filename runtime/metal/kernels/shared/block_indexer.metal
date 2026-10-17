#include "metal/abi/BlockIndexer.h"
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/kv_extent.h"
#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/rms_inverse.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

// A block indexer (Qwen3.8-Flash-Next's QSA, for one). Past top_blocks
// complete blocks of BLOCK_INDEXER_BLOCK tokens, a query row attends to the
// top_blocks complete blocks its indexer scores highest and to the tokens
// after its last complete block, which are none when its own token completes
// its block. A block's score is sum over the indexer's heads of
// relu(q_h . k_b) / sqrt(dimension), accumulated in fp32: q_h the row's
// normalized, rotated query heads, k_b the block's pooled key, the
// normalized mean of its tokens' raw keys rotated to its first token's
// position, both stored bf16 as the model computes them. The kernels write
// each row's bitmap of the blocks it attends to (ops/BlockIndexer.cpp), which
// the sparse attention lists its tiles' blocks from
// (ops::PagedAttention::addBlockLists).

constant constexpr uint kDimension = BLOCK_INDEXER_DIMENSION;
constant constexpr uint kHeads = BLOCK_INDEXER_HEADS;
constant constexpr uint kBlock = BLOCK_INDEXER_BLOCK;
constant constexpr uint kPending = BLOCK_INDEXER_PENDING;
constant constexpr uint kPairs = SPLASH_TARGET_ROPE_PAIRS; // rotated pairs (i, i + kPairs)
constant constexpr uint kPageTokens = SPLASH_TARGET_KV_BLOCK_TOKENS;
constant constexpr uint kPageBlocks = kPageTokens / kBlock;

// Lane `lane` of a dispatch (BlockIndexerParams), none past its lanes.
inline bool indexer_lane(uint lane, constant BlockIndexerParams &p) {
  return lane < min(p.lanes, SPLASH_MAXIMUM_BATCH_WIDTH);
}

// The pooled keys of a page's blocks, through its index entry.
inline device bfloat *indexer_index_keys(SplashKvPage entry, constant BlockIndexerParams &p) {
  const uint index = splash_kv_page_index(entry);
  return reinterpret_cast<device bfloat *>(splash_kv_extent(entry, index) + p.index_offset) +
         ulong(index) * kPageBlocks * kDimension;
}

// The inverse RMS of the threadgroup's kDimension values, one per thread.
inline float indexer_rms_scale(float value, threadgroup float *scratch, uint lane, uint sg) {
  const float sum = simd_sum(value * value);
  if (lane == 0) scratch[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint g = 0; g < kDimension / 32; ++g) total += scratch[g];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return rsqrt(total / kDimension + kRmsEpsilon);
}

// Rotates the first 2 kPairs dimensions as the attention does (pairs i,
// i + kPairs by the table's frequency i) and writes the head as bf16.
template <class Table>
inline void indexer_rotate_store(float value, threadgroup float *rotated, Table cosine,
                                 Table sine, device bfloat *out, uint tid) {
  rotated[tid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float result = value;
  if (tid < kPairs) result = value * cosine[tid] - rotated[tid + kPairs] * sine[tid];
  else if (tid < 2 * kPairs) result = value * cosine[tid - kPairs] + rotated[tid - kPairs] * sine[tid - kPairs];
  out[tid] = bfloat(result);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Threadgroup (row, head, lane) of 128 threads: query heads are normalized,
// rotated by the row's RoPE table and written as bf16 [rows][heads][dim];
// the key head (head == heads) is written raw, bf16, to the row's keys.
kernel void block_indexer_project(device const float *query [[buffer(0)]],
                                  device const float *key [[buffer(1)]],
                                  device const float *query_norm [[buffer(2)]],
                                  device const float *rope_cos [[buffer(3)]],
                                  device const float *rope_sin [[buffer(4)]],
                                  device bfloat *queries [[buffer(5)]],
                                  device bfloat *keys [[buffer(6)]],
                                  constant BlockIndexerParams &p [[buffer(7)]],
                                  uint3 group [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kDimension / 32];
  threadgroup float rotated[kDimension];
  const uint head = group.y, batch = group.z;
  if (group.x >= p.rows || !indexer_lane(batch, p)) return;
  const ulong row = ulong(batch) * p.rows + group.x;
  if (head == kHeads) {
    keys[row * kDimension + tid] = bfloat(key[row * kDimension + tid]);
    return;
  }
  const float value = query[(row * kHeads + head) * kDimension + tid];
  const float scale = indexer_rms_scale(value, scratch, lane, sg);
  indexer_rotate_store(value * scale * query_norm[tid], rotated, rope_cos + row * kPairs, rope_sin + row * kPairs,
                       queries + (row * kHeads + head) * kDimension, tid);
}

// Threadgroup (block, lane) of 128 threads: the mean of the block's four
// tokens' raw keys, normalized and rotated to its first token's position.
// A token from the lane's first row on is a row's of the step; one before it
// is in the lane's pending keys, whose open block this is.
kernel void block_indexer_pool(device const BlockIndexerBlock *blocks [[buffer(0)]],
                               device const bfloat *keys [[buffer(1)]],
                               device const bfloat *pending0 [[buffer(2)]],
                               device const bfloat *pending1 [[buffer(3)]],
                               device const bfloat *pending2 [[buffer(4)]],
                               device const bfloat *pending3 [[buffer(5)]],
                               device const float *key_norm [[buffer(6)]],
                               device const float *inverse_frequencies [[buffer(7)]],
                               device const SplashKvPage *index_table0 [[buffer(8)]],
                               device const SplashKvPage *index_table1 [[buffer(9)]],
                               device const SplashKvPage *index_table2 [[buffer(10)]],
                               device const SplashKvPage *index_table3 [[buffer(11)]],
                               constant BlockIndexerParams &p [[buffer(12)]],
                               uint2 group [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kDimension / 32];
  threadgroup float rotated[kDimension];
  threadgroup float cosine[kPairs], sine[kPairs];
  const uint index = group.x, batch = group.y;
  if (!indexer_lane(batch, p) || index >= p.counts[batch]) return;
  device const SplashKvPage *index_table =
      SPLASH_LANE_BINDING(batch, index_table0, index_table1, index_table2, index_table3);
  device const bfloat *pending = SPLASH_LANE_BINDING(batch, pending0, pending1, pending2, pending3);
  const BlockIndexerBlock block = blocks[ulong(batch) * p.lane_records + index];
  // The block's table row as rope_build_tables computes one: frequency i
  // on axis i % 3 (interleaved M-RoPE).
  if (tid < kPairs) {
    const float angle = float(block.rope[tid % 3]) * inverse_frequencies[tid];
    cosine[tid] = cos(angle);
    sine[tid] = sin(angle);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint first = p.first_position[batch];
  device const bfloat *lane_keys = keys + ulong(batch) * p.rows * kDimension;
  float sum = 0.0f;
  for (uint t = 0; t < kBlock; ++t) {
    const uint position = block.position + t;
    sum += float(position >= first ? lane_keys[ulong(position - first) * kDimension + tid]
                                   : pending[t * kDimension + tid]);
  }
  const float mean = sum / kBlock;
  const float scale = indexer_rms_scale(mean, scratch, lane, sg);
  indexer_rotate_store(mean * scale * key_norm[tid], rotated, cosine, sine,
                       indexer_index_keys(index_table[block.position / kPageTokens], p) +
                           (block.position % kPageTokens / kBlock) * kDimension,
                       tid);
}

// One layer's pending keys of a lane after the first `retained` of its rows
// from `first` (BlockIndexerPendingParams): the tokens of the rows' last block
// before their end, from the rows' keys or, before `first`, the pending keys
// before them, and zero past the end. A thread per dimension.
inline void indexer_store_pending(device const bfloat *keys, device const bfloat *pending_in,
                                  device bfloat *pending_out, uint first, uint retained, uint tid) {
  const uint end = first + retained, open = end / kBlock * kBlock;
  for (uint slot = 0; slot < kPending; ++slot) {
    const uint position = open + slot;
    bfloat value = bfloat(0.0f);
    if (position < end)
      value = position >= first ? keys[ulong(position - first) * kDimension + tid]
                                : pending_in[slot * kDimension + tid];
    pending_out[slot * kDimension + tid] = value;
  }
}

// One threadgroup of kDimension threads: the pending keys of one layer of
// one lane after its first retained[0] rows, a count the host knows.
kernel void block_indexer_pending(device const bfloat *keys [[buffer(0)]],
                                  device const bfloat *pending_in [[buffer(1)]],
                                  device bfloat *pending_out [[buffer(2)]],
                                  constant BlockIndexerPendingParams &p [[buffer(3)]],
                                  uint tid [[thread_index_in_threadgroup]]) {
  indexer_store_pending(keys, pending_in, pending_out, p.first_position[0], min(p.retained[0], p.rows), tid);
}

// Threadgroup (layer, lane) of kDimension threads: each layer's pending keys
// of each lane of a verify after the rows it retained, a count the GPU
// wrote. A lane's pending keys of every layer follow each other.
kernel void block_indexer_commit(device const bfloat *keys [[buffer(0)]],
                                 device const uint *retained [[buffer(1)]],
                                 device const bfloat *pending_in0 [[buffer(2)]],
                                 device const bfloat *pending_in1 [[buffer(3)]],
                                 device const bfloat *pending_in2 [[buffer(4)]],
                                 device const bfloat *pending_in3 [[buffer(5)]],
                                 device bfloat *pending_out0 [[buffer(6)]],
                                 device bfloat *pending_out1 [[buffer(7)]],
                                 device bfloat *pending_out2 [[buffer(8)]],
                                 device bfloat *pending_out3 [[buffer(9)]],
                                 constant BlockIndexerPendingParams &p [[buffer(10)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
  const uint layer = group.x, lane = group.y;
  if (lane >= min(p.lanes, SPLASH_MAXIMUM_BATCH_WIDTH)) return;
  const ulong pending = ulong(layer) * kPending * kDimension;
  indexer_store_pending(keys + (ulong(layer) * p.layer_rows + ulong(lane) * p.rows) * kDimension,
                        SPLASH_LANE_BINDING(lane, pending_in0, pending_in1, pending_in2, pending_in3) + pending,
                        SPLASH_LANE_BINDING(lane, pending_out0, pending_out1, pending_out2, pending_out3) + pending,
                        p.first_position[lane], min(retained[lane], p.rows), tid);
}

// Threadgroup per tile of kScoreRows rows of a lane by kScoreBlocks blocks:
// the rows' query heads and the blocks' pooled keys, gathered from the
// index into threadgroup memory, make every head's dot product of the tile
// in one MPP product, and each row's score of a block sums its heads' relu
// over sqrt(dimension). The scratch holds the dispatch's rows, lane after
// lane.
constant constexpr uint kScoreRows = BLOCK_INDEXER_SCORE_ROWS;
constant constexpr uint kScoreBlocks = BLOCK_INDEXER_SCORE_BLOCKS;
constant constexpr uint kScoreThreads = BLOCK_INDEXER_SCORE_THREADS;
kernel void block_indexer_score(device const bfloat *queries [[buffer(0)]],
                                device const SplashKvPage *index_table0 [[buffer(1)]],
                                device const SplashKvPage *index_table1 [[buffer(2)]],
                                device const SplashKvPage *index_table2 [[buffer(3)]],
                                device const SplashKvPage *index_table3 [[buffer(4)]],
                                device float *scores [[buffer(5)]],
                                constant BlockIndexerParams &p [[buffer(6)]],
                                uint3 group [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]]) {
  constexpr ushort M = kScoreRows * kHeads, N = kScoreBlocks, D = kDimension;
  // 16-byte chunks of a head's (or a key's) row.
  constexpr uint Chunks = D * sizeof(bfloat) / sizeof(uint4);
  alignas(16) threadgroup bfloat q[M * D];
  alignas(16) threadgroup bfloat k[N * D];
  threadgroup float products[M * N];
  const uint row0 = group.y * kScoreRows, block0 = group.x * N, batch = group.z;
  if (!indexer_lane(batch, p)) return;
  const uint last = min(p.rows, row0 + kScoreRows);
  // Rows attend sparsely past top_blocks complete blocks; complete blocks
  // grow with the row.
  const uint complete = (p.first_position[batch] + p.row_offset + last) / kBlock;
  if (block0 >= complete || complete <= p.top_blocks) return;
  device const SplashKvPage *index_table =
      SPLASH_LANE_BINDING(batch, index_table0, index_table1, index_table2, index_table3);
  device const uint4 *tile_queries = reinterpret_cast<device const uint4 *>(
      queries + (ulong(batch) * p.rows + p.row_offset + row0) * kHeads * D);
  for (uint chunk = tid; chunk < M * Chunks; chunk += kScoreThreads)
    reinterpret_cast<threadgroup uint4 *>(q)[chunk] =
        row0 + chunk / (kHeads * Chunks) < p.rows ? tile_queries[chunk] : uint4(0);
  for (uint chunk = tid; chunk < N * Chunks; chunk += kScoreThreads) {
    const uint block = block0 + chunk / Chunks;
    uint4 data = 0;
    if (block < complete)
      data = reinterpret_cast<device const uint4 *>(indexer_index_keys(index_table[block / kPageBlocks], p) +
                                                    (block % kPageBlocks) * D)[chunk % Chunks];
    reinterpret_cast<threadgroup uint4 *>(k)[chunk] = data;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto qt = tensor(q, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto kt = tensor(k, dextents<int, 2>{D, N}, array<int, 2>{1, D});
  auto pt = tensor(products, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto q0 = qt.template slice<D, M>(0, 0);
  auto k0 = kt.template slice<D, N>(0, 0);
  constexpr auto descriptor =
      matmul2d_descriptor(M, N, D, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<kScoreThreads / 32>> product;
  auto dots = product.template get_destination_cooperative_tensor<decltype(q0), decltype(k0), float>();
  product.run(q0, k0, dots);
  dots.store(pt.template slice<N, M>(0, 0));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint index = tid; index < kScoreRows * N; index += kScoreThreads) {
    const uint r = index / N, b = index % N;
    if (row0 + r >= p.rows || block0 + b >= complete) continue;
    float score = 0.0f;
    for (uint h = 0; h < kHeads; ++h) score += max(products[(r * kHeads + h) * N + b], 0.0f);
    scores[(ulong(batch) * p.rows + row0 + r) * p.score_stride + block0 + b] = score / sqrt(float(D));
  }
}

// Threadgroup (row, lane): the row's bitmap of attended blocks. A row with at
// most top_blocks complete blocks attends causally to all of them; past
// that, to the top_blocks highest-scoring complete blocks (ties broken
// towards the earlier block). In both cases it also attends to the block
// after its complete ones (the tail), which holds its tokens past them, if
// any. The top blocks come from a radix select of the top_blocks-th highest
// score over its bits below the sign (a score is never negative),
// kSelectBits a pass from the highest: every thread sums a run of the pass's
// histogram from its top, and the runs' prefix sums find the bin of the
// threshold. The first pass reads the row; its threshold bin's scores, a few
// hundred of a long row, then go to threadgroup memory, where the later
// passes run, unless more than kCandidates fall in it, when they read the
// row again. Once a pass's bin holds just the scores still wanted, all of
// them are taken and the select ends. Each simdgroup then writes a run of the
// bitmap's words from ballots of 32 consecutive scores: those above the
// threshold are taken, and of those equal to it the earliest still wanted. A row
// writes the words the tiles that hold it list from (attention_block_lists:
// up to their last row's tail, at most BLOCK_INDEXER_TILE_SLACK blocks past
// its own).
constant constexpr uint kSelectBits = 11;
constant constexpr uint kSelectThreads = BLOCK_INDEXER_SELECT_THREADS;
constant constexpr uint kCandidates = 4096;

// A pass's threshold over its histogram of `bins` bins: the bin of the
// `wanted`-th highest value and how many of it are wanted, in threshold[0]
// and [1], and in threshold[2] whether that is all of it.
inline void indexer_threshold(threadgroup atomic_uint *histogram, uint bins, uint wanted,
                              threadgroup uint *simd_counts, threadgroup uint *threshold, uint tid, uint lane,
                              uint sg) {
  constexpr uint kRun = (1u << kSelectBits) / kSelectThreads;
  static_assert((1u << kSelectBits) % kSelectThreads == 0, "the threads share the bins out evenly");
  // Thread t sums the t-th run of bins below the top.
  uint run[kRun];
  uint local = 0;
  for (uint j = 0; j < kRun; ++j) {
    const uint below = tid * kRun + j;
    run[j] = below < bins ? atomic_load_explicit(&histogram[bins - 1 - below], memory_order_relaxed) : 0u;
    local += run[j];
  }
  const uint inclusive = simd_prefix_inclusive_sum(local);
  if (lane == 31) simd_counts[sg] = inclusive;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint above = inclusive - local;
  for (uint g = 0; g < sg; ++g) above += simd_counts[g];
  if (above < wanted && above + local >= wanted)
    for (uint j = 0; j < kRun; ++j) {
      if (above + run[j] >= wanted) {
        threshold[0] = bins - 1 - (tid * kRun + j);
        threshold[1] = wanted - above;
        threshold[2] = run[j] == wanted - above;
        break;
      }
      above += run[j];
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void block_indexer_select(device const float *scores [[buffer(0)]],
                                 device uint *mask [[buffer(1)]],
                                 constant BlockIndexerParams &p [[buffer(2)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint kBins = 1u << kSelectBits;
  threadgroup atomic_uint histogram[kBins];
  threadgroup uint candidates[kCandidates];
  threadgroup atomic_uint candidate_count;
  threadgroup uint simd_counts[kSelectThreads / 32];
  threadgroup uint threshold[3];
  const uint batch = group.y;
  if (group.x >= p.rows || !indexer_lane(batch, p)) return;
  const uint position = p.first_position[batch] + p.row_offset + group.x;
  const uint complete = (position + 1) / kBlock;
  const uint words = min(p.mask_words, (complete + BLOCK_INDEXER_TILE_SLACK) / 32 + 1);
  device uint *bits = mask + (ulong(batch) * p.rows + p.row_offset + group.x) * p.mask_words;
  if (complete <= p.top_blocks) {
    // Blocks 0..complete, the tail included; later bits stay clear.
    for (uint w = tid; w < words; w += kSelectThreads) {
      const uint first = w * 32;
      const uint end = complete + 1;
      bits[w] = first >= end ? 0u : (end - first >= 32 ? ~0u : (1u << (end - first)) - 1u);
    }
    return;
  }
  device const float *s = scores + (ulong(batch) * p.rows + group.x) * p.score_stride;
  uint prefix = 0, prefix_mask = 0, wanted = p.top_blocks;
  bool gathered = false, all = false;
  for (int high = 30; high >= 0 && !all; high -= int(kSelectBits)) {
    const uint width = min(kSelectBits, uint(high) + 1), shift = uint(high) + 1 - width;
    const uint bins = 1u << width;
    for (uint i = tid; i < kBins; i += kSelectThreads) atomic_store_explicit(&histogram[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gathered) {
      const uint count = atomic_load_explicit(&candidate_count, memory_order_relaxed);
      for (uint i = tid; i < count; i += kSelectThreads) {
        const uint v = candidates[i];
        if ((v & prefix_mask) == prefix)
          atomic_fetch_add_explicit(&histogram[(v >> shift) & (bins - 1)], 1u, memory_order_relaxed);
      }
    } else {
      for (uint b = tid; b < complete; b += kSelectThreads) {
        const uint v = as_type<uint>(s[b]);
        if ((v & prefix_mask) == prefix)
          atomic_fetch_add_explicit(&histogram[(v >> shift) & (bins - 1)], 1u, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    indexer_threshold(histogram, bins, wanted, simd_counts, threshold, tid, lane, sg);
    prefix |= threshold[0] << shift;
    prefix_mask |= (bins - 1) << shift;
    wanted = threshold[1];
    all = threshold[2];
    // The first pass's bin, if it holds fewer than kCandidates scores, for
    // the passes after it.
    if (high == 30 && !all && atomic_load_explicit(&histogram[threshold[0]], memory_order_relaxed) <= kCandidates) {
      if (tid == 0) atomic_store_explicit(&candidate_count, 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint b = tid; b < complete; b += kSelectThreads) {
        const uint v = as_type<uint>(s[b]);
        if ((v & prefix_mask) == prefix)
          candidates[atomic_fetch_add_explicit(&candidate_count, 1u, memory_order_relaxed)] = v;
      }
      gathered = true;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // Each simdgroup writes a run of whole words, a ballot of 32 consecutive
  // scores each. Unless every score equal to the threshold is taken, a pass
  // first counts each run's, so that the earliest are.
  constexpr uint kSimdgroups = kSelectThreads / 32;
  const uint per = (words + kSimdgroups - 1) / kSimdgroups;
  const uint w0 = min(sg * per, words), w1 = min(w0 + per, words);
  uint before = 0;
  if (!all) {
    uint equal = 0;
    for (uint w = w0; w < w1; ++w) {
      const uint b = w * 32 + lane;
      equal += b < complete && (as_type<uint>(s[b]) & prefix_mask) == prefix ? 1u : 0u;
    }
    equal = simd_sum(equal);
    if (lane == 0) simd_counts[sg] = equal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = 0; g < sg; ++g) before += simd_counts[g];
  }
  for (uint w = w0; w < w1; ++w) {
    const uint b = w * 32 + lane;
    const uint v = b < complete ? as_type<uint>(s[b]) & prefix_mask : 0u;
    const bool equal = b < complete && v == prefix;
    const uint equals = uint(static_cast<simd_vote::vote_t>(simd_ballot(equal)));
    const bool take =
        (b < complete && v > prefix) || (equal && (all || before + popcount(equals & ((1u << lane) - 1u)) < wanted));
    uint word = uint(static_cast<simd_vote::vote_t>(simd_ballot(take)));
    if (w == complete / 32) word |= 1u << (complete % 32);
    if (lane == 0) bits[w] = word;
    before += popcount(equals);
  }
}

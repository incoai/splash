#include "metal/abi/BlockIndexer.h"
#include "metal/abi/KernelABI.h"
#include "metal/abi/VerifyTree.h"
#include "metal/kernels/common/kv_extent.h"
#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/rms_inverse.h"
#include "metal/kernels/common/target_rope.h"

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
// (ops::PagedAttention::addBlockLists). A verify block's row (the _tree
// kernels, metal/abi/BlockIndexer.h) is at its anchor's position plus its
// depth, and the blocks its path completes are its path's own: it scores
// each from the block key of its path's row that completes it, in the same
// product as a block of the index, so that it scores what the path's rows run
// one at a time would.

constant constexpr uint kDimension = BLOCK_INDEXER_DIMENSION;
constant constexpr uint kHeads = BLOCK_INDEXER_HEADS;
constant constexpr uint kBlock = BLOCK_INDEXER_BLOCK;
constant constexpr uint kPending = BLOCK_INDEXER_PENDING;
constant constexpr uint kPairs = SPLASH_TARGET_ROPE_PAIRS; // rotated pairs (i, i + kPairs)
constant constexpr uint kPageTokens = SPLASH_TARGET_KV_BLOCK_TOKENS;
constant constexpr uint kPageBlocks = kPageTokens / kBlock;

// Lane `lane` of a dispatch of `lanes`, none past a batch.
inline bool indexer_lane(uint lane, uint lanes) {
  return lane < min(lanes, SPLASH_MAXIMUM_BATCH_WIDTH);
}

// The pooled key of block `block` of a lane's index, its index table's entry
// of the block's page, for the layer whose index is `offset` bytes into each
// entry's plane.
inline device bfloat *indexer_block_key(device const SplashKvPage *index_table, uint offset, uint block) {
  const SplashKvPage entry = index_table[block / kPageBlocks];
  const uint index = splash_kv_page_index(entry);
  return reinterpret_cast<device bfloat *>(splash_kv_extent(entry, index) + offset) +
         (ulong(index) * kPageBlocks + block % kPageBlocks) * kDimension;
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
  if (!indexer_lane(batch, p.lanes) || group.x >= p.rows[batch]) return;
  const ulong row = ulong(p.first_row[batch]) + group.x;
  if (head == kHeads) {
    keys[row * kDimension + tid] = bfloat(key[row * kDimension + tid]);
    return;
  }
  const float value = query[(row * kHeads + head) * kDimension + tid];
  const float scale = indexer_rms_scale(value, scratch, lane, sg);
  indexer_rotate_store(value * scale * query_norm[tid], rotated, rope_cos + row * kPairs, rope_sin + row * kPairs,
                       queries + (row * kHeads + head) * kDimension, tid);
}

// A block's pooled key from the sum of its tokens' raw keys, a thread per
// dimension: their mean, normalized and rotated to the block's first token's
// position (its record) by the target's RoPE scaled by rope_scale, into
// `out` as bf16.
inline void indexer_pool_store(float sum, BlockIndexerBlock block, device const float *key_norm,
                               device const float *inverse_frequencies, float rope_scale, device bfloat *out,
                               threadgroup float *scratch, threadgroup float *rotated, threadgroup float *cosine,
                               threadgroup float *sine, uint tid, uint lane, uint sg) {
  // The block's row of the target's RoPE table, scaled as the queries' are.
  if (tid < kPairs) {
    const float2 rope =
        target_rope_pair(uint3(block.rope[0], block.rope[1], block.rope[2]), inverse_frequencies, tid, rope_scale);
    cosine[tid] = rope.x;
    sine[tid] = rope.y;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float mean = sum / kBlock;
  const float scale = indexer_rms_scale(mean, scratch, lane, sg);
  indexer_rotate_store(mean * scale * key_norm[tid], rotated, cosine, sine, out, tid);
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
  if (!indexer_lane(batch, p.lanes) || index >= p.counts[batch]) return;
  device const SplashKvPage *index_table =
      SPLASH_LANE_BINDING(batch, index_table0, index_table1, index_table2, index_table3);
  device const bfloat *pending = SPLASH_LANE_BINDING(batch, pending0, pending1, pending2, pending3);
  const BlockIndexerBlock block = blocks[ulong(batch) * p.lane_records + index];
  const uint first = p.first_position[batch];
  device const bfloat *lane_keys = keys + ulong(p.first_row[batch]) * kDimension;
  float sum = 0.0f;
  for (uint t = 0; t < kBlock; ++t) {
    const uint position = block.position + t;
    sum += float(position >= first ? lane_keys[ulong(position - first) * kDimension + tid]
                                   : pending[t * kDimension + tid]);
  }
  indexer_pool_store(sum, block, key_norm, inverse_frequencies, p.rope_scale,
                     indexer_block_key(index_table, p.index_offset, block.position / kBlock), scratch, rotated,
                     cosine, sine, tid, lane, sg);
}

// Threadgroup (row, lane) of 128 threads over a verify step's lanes' rows on
// the device: a row whose position, its lane's anchor plus its depth, ends a
// block pools the block from its path, its own raw key and its parents' back
// to the anchor and, for the block's tokens before the anchor, the lane's
// pending keys, in the order of the tokens as block_indexer_pool sums them;
// into its block key, rotated by the lane's record of the block (from the
// anchor's block on). Every other row pools nothing.
kernel void block_indexer_pool_tree(device const BlockIndexerBlock *blocks [[buffer(0)]],
                                    device const bfloat *keys [[buffer(1)]],
                                    device const SplashTreeRow *tree [[buffer(2)]],
                                    device const SplashLaneRows *lane_rows [[buffer(3)]],
                                    device const bfloat *pending0 [[buffer(4)]],
                                    device const bfloat *pending1 [[buffer(5)]],
                                    device const bfloat *pending2 [[buffer(6)]],
                                    device const bfloat *pending3 [[buffer(7)]],
                                    device const float *key_norm [[buffer(8)]],
                                    device const float *inverse_frequencies [[buffer(9)]],
                                    device bfloat *block_keys [[buffer(10)]],
                                    constant BlockIndexerParams &p [[buffer(11)]],
                                    uint2 group [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kDimension / 32];
  threadgroup float rotated[kDimension];
  threadgroup float cosine[kPairs], sine[kPairs];
  const uint batch = group.y;
  if (!indexer_lane(batch, p.lanes)) return;
  const SplashLaneRows rows = lane_rows[batch];
  if (group.x >= rows.rows) return;
  const uint anchor = p.first_position[batch], last = anchor + tree[rows.first + group.x].depth;
  const uint index = last / kBlock - anchor / kBlock;
  if (last % kBlock != kBlock - 1 || index >= p.counts[batch]) return;
  const BlockIndexerBlock block = blocks[ulong(batch) * p.lane_records + index];
  // The rows of the block's tokens from the anchor on: the row at its last,
  // each parent one token before its child.
  uint path[kBlock] = {};
  uint at = group.x;
  for (uint t = kBlock; t-- > 0 && block.position + t >= anchor;) {
    path[t] = at;
    if (block.position + t > anchor) at = tree[rows.first + at].parent;
  }
  device const bfloat *pending = SPLASH_LANE_BINDING(batch, pending0, pending1, pending2, pending3);
  float sum = 0.0f;
  for (uint t = 0; t < kBlock; ++t)
    sum += float(block.position + t >= anchor ? keys[(ulong(rows.first) + path[t]) * kDimension + tid]
                                              : pending[t * kDimension + tid]);
  indexer_pool_store(sum, block, key_norm, inverse_frequencies, p.rope_scale,
                     block_keys + (ulong(rows.first) + group.x) * kDimension, scratch, rotated, cosine, sine, tid,
                     lane, sg);
}

// One layer's pending keys of a lane after its tokens before `end` (a thread
// per dimension): those of the block they leave open, from
// end / kBlock * kBlock on, and zero from `end` on. Slot s of the block holds
// the step's key row rows[s] from position `first` on, or before `first`
// the lane's pending keys before the step.
inline void indexer_store_pending(device const bfloat *keys, thread const uint (&rows)[kPending],
                                  device const bfloat *pending_in, device bfloat *pending_out, uint first,
                                  uint end, uint tid) {
  const uint open = end / kBlock * kBlock;
  for (uint slot = 0; slot < kPending; ++slot) {
    const uint position = open + slot;
    bfloat value = bfloat(0.0f);
    if (position < end)
      value = position >= first ? keys[ulong(rows[slot]) * kDimension + tid] : pending_in[slot * kDimension + tid];
    pending_out[slot * kDimension + tid] = value;
  }
}

// One threadgroup of kDimension threads: a sequence's pending keys of one
// layer after its first `retained` rows from first_position, a count the
// host knows, its rows' keys in their order.
kernel void block_indexer_pending(device const bfloat *keys [[buffer(0)]],
                                  device const bfloat *pending_in [[buffer(1)]],
                                  device bfloat *pending_out [[buffer(2)]],
                                  constant BlockIndexerPendingParams &p [[buffer(3)]],
                                  uint tid [[thread_index_in_threadgroup]]) {
  const uint end = p.first_position + p.retained, open = end / kBlock * kBlock;
  uint rows[kPending] = {};
  for (uint slot = 0; slot < kPending; ++slot)
    if (open + slot >= p.first_position) rows[slot] = open + slot - p.first_position;
  indexer_store_pending(keys, rows, pending_in, pending_out, p.first_position, end, tid);
}

// Threadgroup (layer, lane) of kDimension threads, once a verify accepted a
// path of each lane's rows (a row mask, metal/abi/VerifyTree.h, of the lane's
// rows on the device): each block a row of the path completes goes into the
// layer's index, its pooled key that row's block key (block_indexer_pool_tree),
// and the lane's pending keys of the layer become those of the block the path
// leaves open, its rows' raw keys there by depth. A lane's pending keys of
// every layer follow each other.
kernel void block_indexer_commit(device const bfloat *keys [[buffer(0)]],
                                 device const bfloat *block_keys [[buffer(1)]],
                                 device const uint *paths [[buffer(2)]],
                                 device const SplashLaneRows *lane_rows [[buffer(3)]],
                                 device const bfloat *pending_in0 [[buffer(4)]],
                                 device const bfloat *pending_in1 [[buffer(5)]],
                                 device const bfloat *pending_in2 [[buffer(6)]],
                                 device const bfloat *pending_in3 [[buffer(7)]],
                                 device bfloat *pending_out0 [[buffer(8)]],
                                 device bfloat *pending_out1 [[buffer(9)]],
                                 device bfloat *pending_out2 [[buffer(10)]],
                                 device bfloat *pending_out3 [[buffer(11)]],
                                 device const SplashKvPage *index_table0 [[buffer(12)]],
                                 device const SplashKvPage *index_table1 [[buffer(13)]],
                                 device const SplashKvPage *index_table2 [[buffer(14)]],
                                 device const SplashKvPage *index_table3 [[buffer(15)]],
                                 constant BlockIndexerCommitParams &p [[buffer(16)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
  const uint layer = group.x, batch = group.y;
  if (!indexer_lane(batch, p.lanes)) return;
  const uint path = paths[batch], anchor = p.first_position[batch], end = anchor + popcount(path);
  const ulong first = ulong(layer) * p.layer_rows + lane_rows[batch].first;
  device const SplashKvPage *index_table =
      SPLASH_LANE_BINDING(batch, index_table0, index_table1, index_table2, index_table3);
  // The path's rows by depth, the anchor's first.
  uint depth = 0;
  for (uint rest = path; rest; rest &= rest - 1, ++depth)
    if ((anchor + depth) % kBlock == kBlock - 1)
      indexer_block_key(index_table, layer * p.index_layer_bytes, (anchor + depth) / kBlock)[tid] =
          block_keys[(first + ctz(rest)) * kDimension + tid];
  // The rows of the path's tokens in the block it leaves open, its deepest
  // the last.
  const uint open = end / kBlock * kBlock;
  uint rows[kPending] = {};
  uint up = path;
  for (uint position = end; position-- > max(open, anchor);) {
    rows[position - open] = 31 - clz(up);
    up &= ~(1u << rows[position - open]);
  }
  const ulong pending = ulong(layer) * kPending * kDimension;
  indexer_store_pending(keys + first * kDimension, rows,
                        SPLASH_LANE_BINDING(batch, pending_in0, pending_in1, pending_in2, pending_in3) + pending,
                        SPLASH_LANE_BINDING(batch, pending_out0, pending_out1, pending_out2, pending_out3) + pending,
                        anchor, end, tid);
}

// The score kernels' tiles: kScoreRows rows of a lane by kScoreBlocks blocks,
// a threadgroup of kScoreThreads threads each. The tile's query heads and its
// blocks' pooled keys are gathered into threadgroup memory, every head's dot
// product of the tile is one MPP product, and a row's score of a block sums
// its heads' relu over sqrt(dimension).
constant constexpr uint kScoreRows = BLOCK_INDEXER_SCORE_ROWS;
constant constexpr uint kScoreBlocks = BLOCK_INDEXER_SCORE_BLOCKS;
constant constexpr uint kScoreThreads = BLOCK_INDEXER_SCORE_THREADS;
// 16-byte chunks of a head's (or a key's) row.
constant constexpr uint kScoreChunks = kDimension * sizeof(bfloat) / sizeof(uint4);

// The scratch of a score tile.
struct IndexerScoreTile {
  alignas(16) bfloat q[kScoreRows * kHeads * kDimension];
  alignas(16) bfloat k[kScoreBlocks * kDimension];
  float products[kScoreRows * kHeads * kScoreBlocks];
};

// The tile's query heads: `rows` rows from `queries` (the first's heads),
// zero past them.
inline void indexer_load_queries(threadgroup IndexerScoreTile &tile, device const bfloat *queries, uint rows,
                                 uint tid) {
  device const uint4 *source = reinterpret_cast<device const uint4 *>(queries);
  for (uint chunk = tid; chunk < kScoreRows * kHeads * kScoreChunks; chunk += kScoreThreads)
    reinterpret_cast<threadgroup uint4 *>(tile.q)[chunk] =
        chunk / (kHeads * kScoreChunks) < rows ? source[chunk] : uint4(0);
}

// Every head's dot product of the tile's rows and keys, once both are loaded:
// products[(row * kHeads + head) * kScoreBlocks + key].
inline void indexer_tile_products(threadgroup IndexerScoreTile &tile) {
  constexpr ushort M = kScoreRows * kHeads, N = kScoreBlocks, D = kDimension;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto qt = tensor(tile.q, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto kt = tensor(tile.k, dextents<int, 2>{D, N}, array<int, 2>{1, D});
  auto pt = tensor(tile.products, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto q0 = qt.template slice<D, M>(0, 0);
  auto k0 = kt.template slice<D, N>(0, 0);
  constexpr auto descriptor =
      matmul2d_descriptor(M, N, D, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<kScoreThreads / 32>> product;
  auto dots = product.template get_destination_cooperative_tensor<decltype(q0), decltype(k0), float>();
  product.run(q0, k0, dots);
  dots.store(pt.template slice<N, M>(0, 0));
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The tile's row `row`'s score of its key `key`.
inline float indexer_tile_score(threadgroup const IndexerScoreTile &tile, uint row, uint key) {
  float score = 0.0f;
  for (uint h = 0; h < kHeads; ++h) score += max(tile.products[(row * kHeads + h) * kScoreBlocks + key], 0.0f);
  return score / sqrt(float(kDimension));
}

// Key `key` of the tile, blocks of the index from block0: zero past `blocks`.
inline void indexer_load_index(threadgroup IndexerScoreTile &tile, device const SplashKvPage *index_table,
                               uint offset, uint block0, uint blocks, uint tid) {
  for (uint chunk = tid; chunk < kScoreBlocks * kScoreChunks; chunk += kScoreThreads) {
    const uint block = block0 + chunk / kScoreChunks;
    uint4 data = 0;
    if (block < blocks)
      data = reinterpret_cast<device const uint4 *>(indexer_block_key(index_table, offset, block))[chunk % kScoreChunks];
    reinterpret_cast<threadgroup uint4 *>(tile.k)[chunk] = data;
  }
}

// Threadgroup per tile of kScoreRows rows of a lane by kScoreBlocks blocks:
// each row's scores of the tile's complete blocks. The scratch holds the
// dispatch's rows from its first lane's first row, as the queries do; a tile
// past its lane's rows has none.
kernel void block_indexer_score(device const bfloat *queries [[buffer(0)]],
                                device const SplashKvPage *index_table0 [[buffer(1)]],
                                device const SplashKvPage *index_table1 [[buffer(2)]],
                                device const SplashKvPage *index_table2 [[buffer(3)]],
                                device const SplashKvPage *index_table3 [[buffer(4)]],
                                device float *scores [[buffer(5)]],
                                constant BlockIndexerParams &p [[buffer(6)]],
                                uint3 group [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]]) {
  threadgroup IndexerScoreTile tile;
  const uint row0 = group.y * kScoreRows, block0 = group.x * kScoreBlocks, batch = group.z;
  if (!indexer_lane(batch, p.lanes) || row0 >= p.rows[batch]) return;
  const uint rows = p.rows[batch], last = min(rows, row0 + kScoreRows);
  // Rows attend sparsely past top_blocks complete blocks; complete blocks
  // grow with the row.
  const uint complete = (p.first_position[batch] + p.row_offset + last) / kBlock;
  if (block0 >= complete || complete <= p.top_blocks) return;
  indexer_load_queries(tile, queries + (ulong(p.first_row[batch]) + p.row_offset + row0) * kHeads * kDimension,
                       last - row0, tid);
  indexer_load_index(tile, SPLASH_LANE_BINDING(batch, index_table0, index_table1, index_table2, index_table3),
                     p.index_offset, block0, complete, tid);
  indexer_tile_products(tile);
  for (uint index = tid; index < kScoreRows * kScoreBlocks; index += kScoreThreads) {
    const uint r = index / kScoreBlocks, b = index % kScoreBlocks;
    if (row0 + r >= rows || block0 + b >= complete) continue;
    scores[(ulong(p.first_row[batch] - p.first_row[0]) + row0 + r) * p.score_stride + block0 + b] =
        indexer_tile_score(tile, r, b);
  }
}

// Threadgroup per tile of kScoreRows rows of a verify lane's rows on the
// device by kScoreBlocks blocks; a row's scores sit at its row of the step.
// The grid's last tile of a row of tiles takes the lane's rows' block keys
// as its keys (zero for a row that completes no block) and scores, for each
// of its rows, every block the row's path completes from the key of the
// path's row at the block's last token; the tiles before it score the blocks
// complete before the lane's anchor from the index. A tile none of whose
// rows can attend sparsely (its last row's position bounds theirs) scores
// nothing.
kernel void block_indexer_score_tree(device const bfloat *queries [[buffer(0)]],
                                     device const bfloat *block_keys [[buffer(1)]],
                                     device const SplashTreeRow *tree [[buffer(2)]],
                                     device const SplashLaneRows *lane_rows [[buffer(3)]],
                                     device const SplashKvPage *index_table0 [[buffer(4)]],
                                     device const SplashKvPage *index_table1 [[buffer(5)]],
                                     device const SplashKvPage *index_table2 [[buffer(6)]],
                                     device const SplashKvPage *index_table3 [[buffer(7)]],
                                     device float *scores [[buffer(8)]],
                                     constant BlockIndexerParams &p [[buffer(9)]],
                                     uint3 group [[threadgroup_position_in_grid]],
                                     uint3 groups [[threadgroups_per_grid]],
                                     uint tid [[thread_index_in_threadgroup]]) {
  threadgroup IndexerScoreTile tile;
  const uint row0 = group.y * kScoreRows, batch = group.z;
  if (!indexer_lane(batch, p.lanes)) return;
  const SplashLaneRows lane = lane_rows[batch];
  if (row0 >= lane.rows) return;
  const uint rows = min(kScoreRows, lane.rows - row0), anchor = p.first_position[batch];
  const uint history = anchor / kBlock, block0 = group.x * kScoreBlocks;
  const bool own = group.x + 1 == groups.x;
  if ((anchor + row0 + rows) / kBlock <= p.top_blocks || (!own && block0 >= history)) return;
  const uint first = lane.first + row0;
  indexer_load_queries(tile, queries + ulong(first) * kHeads * kDimension, rows, tid);
  if (own) {
    for (uint chunk = tid; chunk < kScoreBlocks * kScoreChunks; chunk += kScoreThreads) {
      const uint row = chunk / kScoreChunks;
      uint4 data = 0;
      if (row < lane.rows && (anchor + tree[lane.first + row].depth) % kBlock == kBlock - 1)
        data = reinterpret_cast<device const uint4 *>(block_keys +
                                                      (ulong(lane.first) + row) * kDimension)[chunk % kScoreChunks];
      reinterpret_cast<threadgroup uint4 *>(tile.k)[chunk] = data;
    }
  } else {
    indexer_load_index(tile, SPLASH_LANE_BINDING(batch, index_table0, index_table1, index_table2, index_table3),
                       p.index_offset, block0, history, tid);
  }
  indexer_tile_products(tile);
  if (own) {
    // A row's path, its rows by depth from the anchor's.
    if (tid < rows) {
      uint depth = 0;
      for (uint rest = tree[first + tid].ancestors; rest; rest &= rest - 1, ++depth)
        if ((anchor + depth) % kBlock == kBlock - 1)
          scores[ulong(first + tid) * p.score_stride + (anchor + depth) / kBlock] =
              indexer_tile_score(tile, tid, ctz(rest));
    }
    return;
  }
  for (uint index = tid; index < kScoreRows * kScoreBlocks; index += kScoreThreads) {
    const uint r = index / kScoreBlocks, b = index % kScoreBlocks;
    if (r >= rows || block0 + b >= history) continue;
    scores[ulong(first + r) * p.score_stride + block0 + b] = indexer_tile_score(tile, r, b);
  }
}

// A row's bitmap of attended blocks, a threadgroup per row. A row with at
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

// The scratch of a select.
struct IndexerSelectScratch {
  atomic_uint histogram[1u << kSelectBits];
  uint candidates[kCandidates];
  atomic_uint candidate_count;
  uint simd_counts[kSelectThreads / 32];
  uint threshold[3];
};

// A pass's threshold over its histogram of `bins` bins: the bin of the
// `wanted`-th highest value and how many of it are wanted, in threshold[0]
// and [1], and in threshold[2] whether that is all of it.
inline void indexer_threshold(threadgroup IndexerSelectScratch &x, uint bins, uint wanted, uint tid, uint lane,
                              uint sg) {
  constexpr uint kRun = (1u << kSelectBits) / kSelectThreads;
  static_assert((1u << kSelectBits) % kSelectThreads == 0, "the threads share the bins out evenly");
  // Thread t sums the t-th run of bins below the top.
  uint run[kRun];
  uint local = 0;
  for (uint j = 0; j < kRun; ++j) {
    const uint below = tid * kRun + j;
    run[j] = below < bins ? atomic_load_explicit(&x.histogram[bins - 1 - below], memory_order_relaxed) : 0u;
    local += run[j];
  }
  const uint inclusive = simd_prefix_inclusive_sum(local);
  if (lane == 31) x.simd_counts[sg] = inclusive;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint above = inclusive - local;
  for (uint g = 0; g < sg; ++g) above += x.simd_counts[g];
  if (above < wanted && above + local >= wanted)
    for (uint j = 0; j < kRun; ++j) {
      if (above + run[j] >= wanted) {
        x.threshold[0] = bins - 1 - (tid * kRun + j);
        x.threshold[1] = wanted - above;
        x.threshold[2] = run[j] == wanted - above;
        break;
      }
      above += run[j];
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Words [0, words) of the bitmap of a row with `complete` complete blocks,
// whose scores `s` holds.
inline void indexer_select_row(device const float *s, uint complete, uint words, device uint *bits, uint top_blocks,
                               threadgroup IndexerSelectScratch &x, uint tid, uint lane, uint sg) {
  constexpr uint kBins = 1u << kSelectBits;
  if (complete <= top_blocks) {
    // Blocks 0..complete, the tail included; later bits stay clear.
    for (uint w = tid; w < words; w += kSelectThreads) {
      const uint first = w * 32;
      const uint end = complete + 1;
      bits[w] = first >= end ? 0u : (end - first >= 32 ? ~0u : (1u << (end - first)) - 1u);
    }
    return;
  }
  uint prefix = 0, prefix_mask = 0, wanted = top_blocks;
  bool gathered = false, all = false;
  for (int high = 30; high >= 0 && !all; high -= int(kSelectBits)) {
    const uint width = min(kSelectBits, uint(high) + 1), shift = uint(high) + 1 - width;
    const uint bins = 1u << width;
    for (uint i = tid; i < kBins; i += kSelectThreads) atomic_store_explicit(&x.histogram[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gathered) {
      const uint count = atomic_load_explicit(&x.candidate_count, memory_order_relaxed);
      for (uint i = tid; i < count; i += kSelectThreads) {
        const uint v = x.candidates[i];
        if ((v & prefix_mask) == prefix)
          atomic_fetch_add_explicit(&x.histogram[(v >> shift) & (bins - 1)], 1u, memory_order_relaxed);
      }
    } else {
      for (uint b = tid; b < complete; b += kSelectThreads) {
        const uint v = as_type<uint>(s[b]);
        if ((v & prefix_mask) == prefix)
          atomic_fetch_add_explicit(&x.histogram[(v >> shift) & (bins - 1)], 1u, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    indexer_threshold(x, bins, wanted, tid, lane, sg);
    prefix |= x.threshold[0] << shift;
    prefix_mask |= (bins - 1) << shift;
    wanted = x.threshold[1];
    all = x.threshold[2];
    // The first pass's bin, if it holds fewer than kCandidates scores, for
    // the passes after it.
    if (high == 30 && !all &&
        atomic_load_explicit(&x.histogram[x.threshold[0]], memory_order_relaxed) <= kCandidates) {
      if (tid == 0) atomic_store_explicit(&x.candidate_count, 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint b = tid; b < complete; b += kSelectThreads) {
        const uint v = as_type<uint>(s[b]);
        if ((v & prefix_mask) == prefix)
          x.candidates[atomic_fetch_add_explicit(&x.candidate_count, 1u, memory_order_relaxed)] = v;
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
    if (lane == 0) x.simd_counts[sg] = equal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = 0; g < sg; ++g) before += x.simd_counts[g];
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

kernel void block_indexer_select(device const float *scores [[buffer(0)]],
                                 device uint *mask [[buffer(1)]],
                                 constant BlockIndexerParams &p [[buffer(2)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup IndexerSelectScratch scratch;
  const uint batch = group.y;
  if (!indexer_lane(batch, p.lanes) || group.x >= p.rows[batch]) return;
  const uint position = p.first_position[batch] + p.row_offset + group.x;
  const uint complete = (position + 1) / kBlock;
  indexer_select_row(scores + (ulong(p.first_row[batch] - p.first_row[0]) + group.x) * p.score_stride, complete,
                     min(p.mask_words, (complete + BLOCK_INDEXER_TILE_SLACK) / 32 + 1),
                     mask + (ulong(p.first_row[batch]) + p.row_offset + group.x) * p.mask_words, p.top_blocks,
                     scratch, tid, lane, sg);
}

// The same for a verify lane's rows on the device, a row at its anchor's
// position plus its depth. Its K/V sit at the slots of the lane's rows from
// the anchor's (metal/abi/VerifyTree.h), so its bitmap names the blocks of
// slots it reads: below the anchor's block the blocks it attends, and from
// that block on the blocks of the slots of its path's rows in blocks it
// attends, which it writes as its row mask (a bit per row of its lane), and,
// as a chain's do, the words the tiles that hold it list from, by its slot.
// The anchor's block holds the slots of the lane's first rows, at positions
// in it whatever their depths, so its bit is the block's, as below it.
kernel void block_indexer_select_tree(device const float *scores [[buffer(0)]],
                                      device const SplashTreeRow *tree [[buffer(1)]],
                                      device const SplashLaneRows *lane_rows [[buffer(2)]],
                                      device uint *mask [[buffer(3)]],
                                      device uint *row_masks [[buffer(4)]],
                                      constant BlockIndexerParams &p [[buffer(5)]],
                                      uint2 group [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]],
                                      uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup IndexerSelectScratch scratch;
  const uint batch = group.y;
  if (!indexer_lane(batch, p.lanes)) return;
  const SplashLaneRows rows = lane_rows[batch];
  if (group.x >= rows.rows) return;
  const uint row = rows.first + group.x, anchor = p.first_position[batch], slot = anchor + group.x;
  const SplashTreeRow place = tree[row];
  const uint words = min(p.mask_words, ((slot + 1) / kBlock + BLOCK_INDEXER_TILE_SLACK) / 32 + 1);
  device uint *bits = mask + ulong(row) * p.mask_words;
  indexer_select_row(scores + ulong(row) * p.score_stride, (anchor + place.depth + 1) / kBlock, words, bits,
                     p.top_blocks, scratch, tid, lane, sg);
  threadgroup_barrier(mem_flags::mem_device);
  if (tid) return;
  uint attended = 0, depth = 0;
  for (uint rest = place.ancestors; rest; rest &= rest - 1, ++depth) {
    const uint block = (anchor + depth) / kBlock;
    if ((bits[block / 32] >> (block % 32)) & 1u) attended |= 1u << ctz(rest);
  }
  row_masks[row] = attended;
  const uint first = anchor / kBlock;
  for (uint w = first / 32; w < min(words, (slot + 1) / kBlock / 32 + 1); ++w)
    bits[w] &= w == first / 32 ? (1u << (first % 32)) - 1u : 0u;
  for (uint rest = attended; rest; rest &= rest - 1) {
    const uint block = (anchor + ctz(rest)) / kBlock;
    bits[block / 32] |= 1u << (block % 32);
  }
}

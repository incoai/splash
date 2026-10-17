#include "metal/abi/KernelABI.h"
#include "metal/abi/Qwen4.h"
#include "metal/kernels/common/kv_extent.h"
#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/rms_inverse.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

// Qwen3.8-Flash-Next's QSA. Past top_blocks complete blocks of QWEN4_QSA_BLOCK tokens, a query row attends
// to the top_blocks complete blocks its indexer scores highest and to the
// tokens after its last complete block, which are none when its own token
// completes its block. A block's score is sum over the indexer's heads of
// relu(q_h . k_b) / sqrt(dimension), accumulated in fp32: q_h the row's
// normalized, rotated query heads, k_b the block's pooled key, the
// normalized mean of its tokens' raw keys rotated to its first token's
// position, both stored bf16 as the model computes them. The kernels write
// each row's bitmap of the blocks it attends to (ops/Qwen4.cpp), which the
// sparse attention lists its tiles' blocks from
// (ops::PagedAttention::addBlockLists).

constant constexpr uint kDimension = QWEN4_QSA_DIMENSION;
constant constexpr uint kHeads = QWEN4_QSA_HEADS;
constant constexpr uint kBlock = QWEN4_QSA_BLOCK;
constant constexpr uint kPairs = SPLASH_TARGET_ROPE_PAIRS; // rotated pairs (i, i + kPairs)
constant constexpr uint kPageTokens = SPLASH_TARGET_KV_BLOCK_TOKENS;
constant constexpr uint kPageBlocks = kPageTokens / kBlock;

// Lane `lane` of a dispatch (Qwen4QsaParams), none past its lanes.
inline bool qsa_lane(uint lane, constant Qwen4QsaParams &p) { return lane < min(p.lanes, SPLASH_MAXIMUM_BATCH_WIDTH); }

// A page's raw keys and pooled keys in its extent's index regions.
inline device bfloat *qsa_raw_keys(SplashKvPage entry, constant Qwen4QsaParams &p) {
  const uint index = splash_kv_page_index(entry);
  return reinterpret_cast<device bfloat *>(splash_kv_extent(entry, index) + p.keys_offset) +
         ulong(index) * kPageTokens * kDimension;
}
inline device bfloat *qsa_pooled_keys(SplashKvPage entry, constant Qwen4QsaParams &p) {
  const uint index = splash_kv_page_index(entry);
  return reinterpret_cast<device bfloat *>(splash_kv_extent(entry, index) + p.pooled_offset) +
         ulong(index) * kPageBlocks * kDimension;
}

// The inverse RMS of the threadgroup's kDimension values, one per thread.
inline float qsa_rms_scale(float value, threadgroup float *scratch, uint lane, uint sg) {
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
inline void qsa_rotate_store(float value, threadgroup float *rotated, Table cosine,
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
// the key head (head == heads) is stored raw, bf16, at the row's slot of
// the index store.
kernel void qwen4_qsa_project(device const float *query [[buffer(0)]],
                              device const float *key [[buffer(1)]],
                              device const float *query_norm [[buffer(2)]],
                              device const float *rope_cos [[buffer(3)]],
                              device const float *rope_sin [[buffer(4)]],
                              device const SplashKvPage *page_table0 [[buffer(5)]],
                              device const SplashKvPage *page_table1 [[buffer(6)]],
                              device const SplashKvPage *page_table2 [[buffer(7)]],
                              device const SplashKvPage *page_table3 [[buffer(8)]],
                              device bfloat *queries [[buffer(9)]],
                              constant Qwen4QsaParams &p [[buffer(10)]],
                              uint3 group [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kDimension / 32];
  threadgroup float rotated[kDimension];
  const uint head = group.y, batch = group.z;
  if (group.x >= p.rows || !qsa_lane(batch, p)) return;
  const uint row = batch * p.rows + group.x;
  if (head == kHeads) {
    device const SplashKvPage *page_table =
        SPLASH_LANE_BINDING(batch, page_table0, page_table1, page_table2, page_table3);
    const uint position = p.first_position[batch] + group.x;
    qsa_raw_keys(page_table[position / kPageTokens], p)[(position % kPageTokens) * kDimension + tid] =
        bfloat(key[ulong(row) * kDimension + tid]);
    return;
  }
  const float value = query[(ulong(row) * kHeads + head) * kDimension + tid];
  const float scale = qsa_rms_scale(value, scratch, lane, sg);
  qsa_rotate_store(value * scale * query_norm[tid], rotated, rope_cos + ulong(row) * kPairs,
                   rope_sin + ulong(row) * kPairs, queries + (ulong(row) * kHeads + head) * kDimension, tid);
}

// Threadgroup (block, lane) of 128 threads: the mean of the block's four
// tokens' raw keys, normalized and rotated to its first token's position.
kernel void qwen4_qsa_pool(device const Qwen4QsaBlock *blocks [[buffer(0)]],
                           device const float *key_norm [[buffer(1)]],
                           device const float *inverse_frequencies [[buffer(2)]],
                           device const SplashKvPage *page_table0 [[buffer(3)]],
                           device const SplashKvPage *page_table1 [[buffer(4)]],
                           device const SplashKvPage *page_table2 [[buffer(5)]],
                           device const SplashKvPage *page_table3 [[buffer(6)]],
                           constant Qwen4QsaParams &p [[buffer(7)]],
                           uint2 group [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kDimension / 32];
  threadgroup float rotated[kDimension];
  threadgroup float cosine[kPairs], sine[kPairs];
  const uint index = group.x, batch = group.y;
  if (!qsa_lane(batch, p) || index >= p.counts[batch]) return;
  device const SplashKvPage *page_table =
      SPLASH_LANE_BINDING(batch, page_table0, page_table1, page_table2, page_table3);
  const Qwen4QsaBlock block = blocks[ulong(batch) * p.lane_records + index];
  // The block's table row as rope_build_tables computes one: frequency i
  // on axis i % 3 (interleaved M-RoPE).
  if (tid < kPairs) {
    const float angle = float(block.rope[tid % 3]) * inverse_frequencies[tid];
    cosine[tid] = cos(angle);
    sine[tid] = sin(angle);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const SplashKvPage page = page_table[block.position / kPageTokens];
  const uint slot = block.position % kPageTokens;
  device const bfloat *raw_keys = qsa_raw_keys(page, p);
  float sum = 0.0f;
  for (uint t = 0; t < kBlock; ++t) sum += float(raw_keys[(slot + t) * kDimension + tid]);
  const float mean = sum / kBlock;
  const float scale = qsa_rms_scale(mean, scratch, lane, sg);
  qsa_rotate_store(mean * scale * key_norm[tid], rotated, cosine, sine,
                   qsa_pooled_keys(page, p) + (slot / kBlock) * kDimension, tid);
}

// Threadgroup per tile of kScoreRows rows of a lane by kScoreBlocks blocks:
// the rows' query heads and the blocks' pooled keys, gathered from their
// pages into threadgroup memory, make every head's dot product of the tile
// in one MPP product, and each row's score of a block sums its heads' relu
// over sqrt(dimension). The scratch holds the dispatch's rows, lane after
// lane.
constant constexpr uint kScoreRows = QWEN4_QSA_SCORE_ROWS;
constant constexpr uint kScoreBlocks = QWEN4_QSA_SCORE_BLOCKS;
constant constexpr uint kScoreThreads = QWEN4_QSA_SCORE_THREADS;
kernel void qwen4_qsa_score(device const bfloat *queries [[buffer(0)]],
                            device const SplashKvPage *page_table0 [[buffer(1)]],
                            device const SplashKvPage *page_table1 [[buffer(2)]],
                            device const SplashKvPage *page_table2 [[buffer(3)]],
                            device const SplashKvPage *page_table3 [[buffer(4)]],
                            device float *scores [[buffer(5)]],
                            constant Qwen4QsaParams &p [[buffer(6)]],
                            uint3 group [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
  constexpr ushort M = kScoreRows * kHeads, N = kScoreBlocks, D = kDimension;
  // 16-byte chunks of a head's (or a key's) row.
  constexpr uint Chunks = D * sizeof(bfloat) / sizeof(uint4);
  alignas(16) threadgroup bfloat q[M * D];
  alignas(16) threadgroup bfloat k[N * D];
  threadgroup float products[M * N];
  const uint row0 = group.y * kScoreRows, block0 = group.x * N, batch = group.z;
  if (!qsa_lane(batch, p)) return;
  const uint last = min(p.rows, row0 + kScoreRows);
  // Rows attend sparsely past top_blocks complete blocks; complete blocks
  // grow with the row.
  const uint complete = (p.first_position[batch] + p.row_offset + last) / kBlock;
  if (block0 >= complete || complete <= p.top_blocks) return;
  device const SplashKvPage *page_table =
      SPLASH_LANE_BINDING(batch, page_table0, page_table1, page_table2, page_table3);
  device const uint4 *tile_queries = reinterpret_cast<device const uint4 *>(
      queries + (ulong(batch) * p.rows + p.row_offset + row0) * kHeads * D);
  for (uint chunk = tid; chunk < M * Chunks; chunk += kScoreThreads)
    reinterpret_cast<threadgroup uint4 *>(q)[chunk] =
        row0 + chunk / (kHeads * Chunks) < p.rows ? tile_queries[chunk] : uint4(0);
  for (uint chunk = tid; chunk < N * Chunks; chunk += kScoreThreads) {
    const uint block = block0 + chunk / Chunks;
    uint4 data = 0;
    if (block < complete)
      data = reinterpret_cast<device const uint4 *>(qsa_pooled_keys(page_table[block / kPageBlocks], p) +
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
// threshold. Once that bin holds just the scores still wanted, all of them
// are taken and the select ends.
constant constexpr uint kSelectBits = 11;
kernel void qwen4_qsa_select(device const float *scores [[buffer(0)]],
                             device uint *mask [[buffer(1)]],
                             constant Qwen4QsaParams &p [[buffer(2)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint kThreads = QWEN4_QSA_SELECT_THREADS;
  constexpr uint kBins = 1u << kSelectBits;
  // The bins a thread sums.
  constexpr uint kRun = kBins / kThreads;
  static_assert(kBins % kThreads == 0, "the threads share the bins out evenly");
  threadgroup atomic_uint histogram[kBins];
  threadgroup uint simd_counts[kThreads / 32];
  // The threshold's bin, the scores in it still wanted, and whether they
  // are all of its scores.
  threadgroup uint threshold[3];
  const uint batch = group.y;
  if (group.x >= p.rows || !qsa_lane(batch, p)) return;
  const uint position = p.first_position[batch] + p.row_offset + group.x;
  const uint complete = (position + 1) / kBlock;
  device uint *bits = mask + (ulong(batch) * p.rows + p.row_offset + group.x) * p.mask_words;
  if (complete <= p.top_blocks) {
    // Blocks 0..complete, the tail included; later bits stay clear.
    for (uint w = tid; w < p.mask_words; w += kThreads) {
      const uint first = w * 32;
      const uint end = complete + 1;
      bits[w] = first >= end ? 0u : (end - first >= 32 ? ~0u : (1u << (end - first)) - 1u);
    }
    return;
  }
  device const float *s = scores + (ulong(batch) * p.rows + group.x) * p.score_stride;
  uint prefix = 0, prefix_mask = 0, wanted = p.top_blocks;
  for (int high = 30; high >= 0; high -= int(kSelectBits)) {
    const uint width = min(kSelectBits, uint(high) + 1), shift = uint(high) + 1 - width;
    const uint bins = 1u << width;
    for (uint i = tid; i < kBins; i += kThreads) atomic_store_explicit(&histogram[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint b = tid; b < complete; b += kThreads) {
      const uint v = as_type<uint>(s[b]);
      if ((v & prefix_mask) == prefix)
        atomic_fetch_add_explicit(&histogram[(v >> shift) & (bins - 1)], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
    prefix |= threshold[0] << shift;
    prefix_mask |= (bins - 1) << shift;
    wanted = threshold[1];
    const bool all = threshold[2];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (all) break;
  }
  // A score whose resolved bits exceed the threshold's is taken, and of
  // those equal to it the `wanted` earliest: each thread owns a contiguous
  // run of words.
  const uint words = (complete + 31) / 32;
  const uint per = (words + kThreads - 1) / kThreads;
  const uint w0 = min(tid * per, words), w1 = min(w0 + per, words);
  uint equal = 0;
  for (uint b = w0 * 32; b < min(w1 * 32, complete); ++b)
    equal += (as_type<uint>(s[b]) & prefix_mask) == prefix ? 1u : 0u;
  const uint inclusive = simd_prefix_inclusive_sum(equal);
  if (lane == 31) simd_counts[sg] = inclusive;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint before = inclusive - equal;
  for (uint g = 0; g < sg; ++g) before += simd_counts[g];
  for (uint w = w0; w < w1; ++w) {
    uint word = 0;
    for (uint i = 0; i < 32; ++i) {
      const uint b = w * 32 + i;
      if (b >= complete) break;
      const uint v = as_type<uint>(s[b]) & prefix_mask;
      bool take = v > prefix;
      if (v == prefix) {
        take = before < wanted;
        ++before;
      }
      if (take) word |= 1u << i;
    }
    // The tail block, from the thread that writes its word: a threadgroup
    // barrier does not order another thread's device stores.
    if (w == complete / 32) word |= 1u << (complete % 32);
    bits[w] = word;
  }
  // Past the complete blocks: the tail block when it starts a word, then
  // nothing.
  for (uint w = words + tid; w < p.mask_words; w += kThreads)
    bits[w] = w == complete / 32 ? 1u << (complete % 32) : 0u;
}

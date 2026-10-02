#include "metal/abi/KernelABI.h"
#include "metal/abi/Qwen4.h"
#include "metal/kernels/common/kv_extent.h"

#include <metal_stdlib>

using namespace metal;

// Qwen3.8-Flash-Next's QSA (llama.cpp qwen4exp.cpp build_qsa_sel). Past
// top_blocks complete blocks of four tokens, a query row attends to the
// top_blocks blocks its indexer scores highest and to the block it is in.
// A block's score is sum over the indexer's heads of relu(q_h . k_b) /
// sqrt(dimension), in fp32: q_h the row's normalized, rotated query heads,
// k_b the block's pooled key, the normalized mean of its tokens' raw keys
// rotated to its first token's position. The kernels write each row's
// bitmap of the blocks it attends to (ops/Qwen4.cpp), which the GQA-12
// attention kernels read.

constant constexpr uint kDimension = QWEN4_QSA_DIMENSION;
constant constexpr uint kHeads = QWEN4_QSA_HEADS;
constant constexpr uint kBlock = QWEN4_QSA_BLOCK;
constant constexpr uint kPairs = 32;   // rotated pairs (i, i + 32) of the first 64 dimensions
constant constexpr uint kPageTokens = SPLASH_TARGET_KV_BLOCK_TOKENS;
constant constexpr uint kPageBlocks = kPageTokens / kBlock;

// A page's raw keys and pooled keys in its extent's index regions.
inline device bfloat *qsa_raw_keys(SplashKvPage entry, constant Qwen4QsaParams &p) {
  const uint index = splash_kv_page_index(entry);
  return reinterpret_cast<device bfloat *>(splash_kv_extent(entry, index) + p.keys_offset) +
         ulong(index) * kPageTokens * kDimension;
}
inline device float *qsa_pooled_keys(SplashKvPage entry, constant Qwen4QsaParams &p) {
  const uint index = splash_kv_page_index(entry);
  return reinterpret_cast<device float *>(splash_kv_extent(entry, index) + p.pooled_offset) +
         ulong(index) * kPageBlocks * kDimension;
}

inline float qsa_rms_scale(float value, threadgroup float *scratch, uint lane, uint sg) {
  const float sum = simd_sum(value * value);
  if (lane == 0) scratch[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint g = 0; g < kDimension / 32; ++g) total += scratch[g];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return rsqrt(total / kDimension + QWEN4_RMS_EPSILON);
}

// Rotates the first 2 kPairs dimensions as the attention does (pairs i,
// i + 32 by the table's frequency i) and writes the head.
template <class Table>
inline void qsa_rotate_store(float value, threadgroup float *rotated, Table cosine,
                             Table sine, device float *out, uint tid) {
  rotated[tid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float result = value;
  if (tid < kPairs) result = value * cosine[tid] - rotated[tid + kPairs] * sine[tid];
  else if (tid < 2 * kPairs) result = value * cosine[tid - kPairs] + rotated[tid - kPairs] * sine[tid - kPairs];
  out[tid] = result;
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Threadgroup (row, head) of 128 threads: query heads are normalized,
// rotated by the row's RoPE table and written as fp32 [rows][heads][dim];
// the key head (head == heads) is stored raw, bf16, at the row's slot of
// the index store.
kernel void qwen4_qsa_project(device const float *query [[buffer(0)]],
                              device const float *key [[buffer(1)]],
                              device const float *query_norm [[buffer(2)]],
                              device const float *rope_cos [[buffer(3)]],
                              device const float *rope_sin [[buffer(4)]],
                              device const SplashKvPage *page_table [[buffer(5)]],
                              device float *queries [[buffer(6)]],
                              constant Qwen4QsaParams &p [[buffer(7)]],
                              uint2 group [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kDimension / 32];
  threadgroup float rotated[kDimension];
  const uint row = group.x, head = group.y;
  if (row >= p.rows) return;
  if (head == kHeads) {
    const uint position = p.first_position + row;
    qsa_raw_keys(page_table[position / kPageTokens], p)[(position % kPageTokens) * kDimension + tid] =
        bfloat(key[ulong(row) * kDimension + tid]);
    return;
  }
  const float value = query[(ulong(row) * kHeads + head) * kDimension + tid];
  const float scale = qsa_rms_scale(value, scratch, lane, sg);
  qsa_rotate_store(value * scale * query_norm[tid], rotated, rope_cos + ulong(row) * kPairs,
                   rope_sin + ulong(row) * kPairs, queries + (ulong(row) * kHeads + head) * kDimension, tid);
}

// Threadgroup per block of 128 threads: the mean of its four tokens' raw
// keys, normalized and rotated to its first token's position.
kernel void qwen4_qsa_pool(device const Qwen4QsaBlock *blocks [[buffer(0)]],
                           device const float *key_norm [[buffer(1)]],
                           device const float *inverse_frequencies [[buffer(2)]],
                           device const SplashKvPage *page_table [[buffer(3)]],
                           constant Qwen4QsaParams &p [[buffer(4)]],
                           uint index [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kDimension / 32];
  threadgroup float rotated[kDimension];
  threadgroup float cosine[kPairs], sine[kPairs];
  if (index >= p.blocks) return;
  const Qwen4QsaBlock block = blocks[index];
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

// The scores of a tile of 8 rows by 256 blocks: thread b streams block b's
// pooled key once and scores it for the tile's rows that attend sparsely.
// Queries and positions are those of rows row_offset + r; the scratch holds
// the dispatch's rows from its first.
constant constexpr uint kScoreRows = 8;
kernel void qwen4_qsa_score(device const float *queries [[buffer(0)]],
                            device const SplashKvPage *page_table [[buffer(1)]],
                            device float *scores [[buffer(2)]],
                            constant Qwen4QsaParams &p [[buffer(3)]],
                            uint2 group [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float q[kScoreRows * kHeads * kDimension];
  const uint row0 = group.y * kScoreRows;
  for (uint i = tid; i < kScoreRows * kHeads * kDimension; i += 256) {
    const uint r = row0 + i / (kHeads * kDimension);
    q[i] = r < p.rows ? queries[(ulong(p.row_offset + r) * kHeads) * kDimension + i % (kHeads * kDimension)] : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint block = group.x * 256 + tid;
  const uint last = min(p.rows, row0 + kScoreRows);
  // Rows attend sparsely past top_blocks complete blocks; complete blocks
  // grow with the row.
  const uint complete = (p.first_position + p.row_offset + last) / kBlock;
  if (block >= complete || complete <= p.top_blocks) return;
  device const float *k =
      qsa_pooled_keys(page_table[block / kPageBlocks], p) + (block % kPageBlocks) * kDimension;
  float acc[kScoreRows][kHeads];
  for (uint r = 0; r < kScoreRows; ++r)
    for (uint h = 0; h < kHeads; ++h) acc[r][h] = 0.0f;
  for (uint d = 0; d < kDimension; d += 4) {
    const float4 kv = *reinterpret_cast<device const float4 *>(k + d);
    for (uint r = 0; r < kScoreRows; ++r)
      for (uint h = 0; h < kHeads; ++h) {
        threadgroup const float *qv = q + (r * kHeads + h) * kDimension + d;
        acc[r][h] = fma(qv[0], kv.x, fma(qv[1], kv.y, fma(qv[2], kv.z, fma(qv[3], kv.w, acc[r][h]))));
      }
  }
  for (uint r = 0; r < kScoreRows && row0 + r < p.rows; ++r) {
    float score = 0.0f;
    for (uint h = 0; h < kHeads; ++h) score += max(acc[r][h], 0.0f);
    scores[ulong(row0 + r) * p.score_stride + block] = score / sqrt(float(kDimension));
  }
}

// Threadgroup per row: the row's bitmap of attended blocks. A row with at
// most top_blocks complete blocks attends causally to all of them; past
// that, to the top_blocks highest-scoring complete blocks (ties broken
// towards the earlier block) and, in both cases, to the block it is in.
kernel void qwen4_qsa_select(device const float *scores [[buffer(0)]],
                             device uint *mask [[buffer(1)]],
                             constant Qwen4QsaParams &p [[buffer(2)]],
                             uint group [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint kThreads = QWEN4_QSA_SELECT_THREADS;
  threadgroup atomic_uint histogram[256];
  threadgroup uint shared[4];
  threadgroup uint simd_counts[kThreads / 32];
  const uint row = p.row_offset + group;
  if (group >= p.rows) return;
  const uint position = p.first_position + row;
  const uint complete = (position + 1) / kBlock;
  device uint *bits = mask + ulong(row) * p.mask_words;
  if (complete <= p.top_blocks) {
    // Blocks 0..complete, the tail included; later bits stay clear.
    for (uint w = tid; w < p.mask_words; w += kThreads) {
      const uint first = w * 32;
      const uint end = complete + 1;
      bits[w] = first >= end ? 0u : (end - first >= 32 ? ~0u : (1u << (end - first)) - 1u);
    }
    return;
  }
  device const float *s = scores + ulong(group) * p.score_stride;
  // Radix-select the top_blocks-th largest score; scores are >= 0, so their
  // bits order as their values.
  uint prefix = 0, prefix_mask = 0, wanted = p.top_blocks;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (uint i = tid; i < 256; i += kThreads) atomic_store_explicit(&histogram[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint b = tid; b < complete; b += kThreads) {
      const uint v = as_type<uint>(s[b]);
      if ((v & prefix_mask) == prefix)
        atomic_fetch_add_explicit(&histogram[(v >> shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      uint above = 0, bin = 255;
      for (;; --bin) {
        const uint count = atomic_load_explicit(&histogram[bin], memory_order_relaxed);
        if (above + count >= wanted || bin == 0) {
          shared[0] = bin;
          shared[1] = wanted - above;
          break;
        }
        above += count;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix |= shared[0] << shift;
    prefix_mask |= 255u << shift;
    wanted = shared[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // `prefix` is the threshold; `wanted` of the scores equal to it are taken,
  // the earliest first: each thread owns a contiguous run of words.
  const uint words = (complete + 31) / 32;
  const uint per = (words + kThreads - 1) / kThreads;
  const uint w0 = min(tid * per, words), w1 = min(w0 + per, words);
  uint equal = 0;
  for (uint b = w0 * 32; b < min(w1 * 32, complete); ++b) equal += as_type<uint>(s[b]) == prefix ? 1u : 0u;
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
      const uint v = as_type<uint>(s[b]);
      bool take = v > prefix;
      if (v == prefix) {
        take = before < wanted;
        ++before;
      }
      if (take) word |= 1u << i;
    }
    bits[w] = word;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // The tail block, and nothing past it.
  for (uint w = words + tid; w < p.mask_words; w += kThreads) bits[w] = 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) bits[complete / 32] |= 1u << (complete % 32);
}

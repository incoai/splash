#include "metal/abi/LatentIndex.h"
#include "metal/kernels/common/kv_extent.h"

#include <metal_stdlib>

using namespace metal;

// DeepSeek-V4.1's compressor and indexer (metal/abi/LatentIndex.h), as
// DeepSeek's model.py computes them (Compressor before its norm, Indexer's
// scores; in dev/tools/deepseek41_reference.py Forward.latents and
// Forward.select).

constant constexpr uint kThreads = LATENT_INDEX_THREADS;
constant constexpr uint kDim = LATENT_INDEX_DIM;
constant constexpr uint kMaxHeads = LATENT_INDEX_MAX_HEADS;

// A compressed group's latent, fp32: each value the sum of its `ratio` rows'
// values weighted by the softmax of their gate scores of that value, from
// bf16 projections or fp32 ones (BF16 weights project in fp32). A thread per
// group and value.
template <typename In>
inline void latent_compress_value(device const In *kv, device const In *gate, device float *out,
                                  constant LatentCompressParams &p, uint2 group, uint tid) {
  const uint g = group.x, d = group.y * kThreads + tid;
  if (g >= p.groups || d >= p.width) return;
  const ulong first = ulong(g) * p.ratio * p.width + d;
  float scores[LATENT_COMPRESS_MAX_RATIO];
  float peak = -numeric_limits<float>::infinity();
  for (uint i = 0; i < p.ratio; ++i) {
    scores[i] = float(gate[first + ulong(i) * p.width]);
    peak = max(peak, scores[i]);
  }
  float sum = 0.0f;
  for (uint i = 0; i < p.ratio; ++i) sum += scores[i] = exp(scores[i] - peak);
  float value = 0.0f;
  for (uint i = 0; i < p.ratio; ++i) value += float(kv[first + ulong(i) * p.width]) * (scores[i] / sum);
  out[ulong(g) * p.width + d] = value;
}
#define LATENT_COMPRESS_ENTRY(Name, In)                                                                          \
  kernel void Name(device const In *kv [[buffer(0)]], device const In *gate [[buffer(1)]],                      \
                   device float *out [[buffer(2)]], constant LatentCompressParams &p [[buffer(3)]],             \
                   uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {     \
    latent_compress_value<In>(kv, gate, out, p, group, tid);                                                      \
  }
LATENT_COMPRESS_ENTRY(latent_compress_bf16, bfloat)
LATENT_COMPRESS_ENTRY(latent_compress_f32, float)
#undef LATENT_COMPRESS_ENTRY

// The indexer's score of each group a row reaches: the sum over its heads of
// the head's weight, scaled, times the ReLU of the head's query with the
// group's key; -inf for the groups it does not reach. A threadgroup scores
// LATENT_INDEX_THREADS groups of a row, a thread each, the row's queries and
// weights in threadgroup memory.
// Where a pass's index keys are: group g's the g-th of a pool, or the first
// values of the value at its place in a sequence's pages, the place of
// position (g + 1) * ratio - 1, through the sequence's page table.
struct PooledIndexKeys {
  using Source = device const bfloat *;
  static device const bfloat *at(Source keys, constant LatentIndexParams &, uint g) { return keys + ulong(g) * kDim; }
};
struct PagedIndexKeys {
  using Source = device const SplashKvPage *;
  static device const bfloat *at(Source pages, constant LatentIndexParams &p, uint g) {
    const uint position = (g + 1) * p.ratio - 1;
    const SplashKvPage entry = pages[position / SplashKvPageTokens];
    const uint page = splash_kv_page_index(entry);
    // BF16 pages: a region's keys of every page, then their values.
    const ulong values = ulong(p.kv.extent_pages) * SplashKvPageTokens * LATENT_STORE_WIDTH * sizeof(bfloat);
    return reinterpret_cast<device const bfloat *>(splash_kv_extent(entry, page) + p.kv.offset + values) +
           splash_kv_value_index<1, LATENT_STORE_WIDTH, true>(page, 0, position % SplashKvPageTokens, 0);
  }
};

template <class Keys>
inline void latent_index_row_scores(device const bfloat *q, device const float *weights,
                                    typename Keys::Source keys, device const uint *positions, device float *scores,
                                    constant LatentIndexParams &p, uint2 group, uint tid,
                                    threadgroup float *row_q, threadgroup float *row_weights) {
  const uint row = group.x;
  if (row >= p.rows) return;
  for (uint i = tid; i < p.heads * kDim; i += kThreads) row_q[i] = float(q[ulong(row) * p.heads * kDim + i]);
  for (uint h = tid; h < p.heads; h += kThreads) row_weights[h] = weights[ulong(row) * p.heads + h] * p.weight_scale;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint g = group.y * kThreads + tid;
  if (g >= p.stride) return;
  const uint reachable = min(p.groups, (positions[row] + 1) / p.ratio);
  float score = -numeric_limits<float>::infinity();
  if (g < reachable) {
    device const bfloat *key = Keys::at(keys, p, g);
    float dots[kMaxHeads];
    for (uint h = 0; h < kMaxHeads; ++h) dots[h] = 0.0f;
    for (uint d = 0; d < kDim; d += 4) {
      const float4 k = float4(*reinterpret_cast<device const bfloat4 *>(key + d));
      for (uint h = 0; h < kMaxHeads; ++h)
        if (h < p.heads) dots[h] += dot(*reinterpret_cast<threadgroup const float4 *>(row_q + h * kDim + d), k);
    }
    score = 0.0f;
    for (uint h = 0; h < kMaxHeads; ++h)
      if (h < p.heads) score += max(dots[h], 0.0f) * row_weights[h];
  }
  scores[ulong(row) * p.stride + g] = score;
}
kernel void latent_index_scores(device const bfloat *q [[buffer(0)]], device const float *weights [[buffer(1)]],
                                device const bfloat *keys [[buffer(2)]], device const uint *positions [[buffer(3)]],
                                device float *scores [[buffer(4)]], constant LatentIndexParams &p [[buffer(5)]],
                                uint2 group [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float row_q[kMaxHeads * kDim];
  threadgroup float row_weights[kMaxHeads];
  latent_index_row_scores<PooledIndexKeys>(q, weights, keys, positions, scores, p, group, tid, row_q, row_weights);
}
kernel void latent_index_scores_paged(device const bfloat *q [[buffer(0)]], device const float *weights [[buffer(1)]],
                                      device const SplashKvPage *pages [[buffer(2)]],
                                      device const uint *positions [[buffer(3)]], device float *scores [[buffer(4)]],
                                      constant LatentIndexParams &p [[buffer(5)]],
                                      uint2 group [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float row_q[kMaxHeads * kDim];
  threadgroup float row_weights[kMaxHeads];
  latent_index_row_scores<PagedIndexKeys>(q, weights, pages, positions, scores, p, group, tid, row_q, row_weights);
}

// A KV source's compressed groups into a sequence's pages (LatentStoreParams):
// a threadgroup per group, each thread four values of its latent and one of
// its index key.
kernel void latent_index_store(device const bfloat *latents [[buffer(0)]], device const bfloat *index_keys [[buffer(1)]],
                               device const SplashKvPage *pages [[buffer(2)]],
                               constant LatentStoreParams &p [[buffer(3)]],
                               uint i [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  if (i >= p.groups) return;
  const uint position = (p.first + i + 1) * p.ratio - 1;
  const SplashKvPage entry = pages[position / SplashKvPageTokens];
  const uint page = splash_kv_page_index(entry), token = position % SplashKvPageTokens;
  device uchar *region = splash_kv_extent(entry, page) + p.kv.offset;
  const ulong values = ulong(p.kv.extent_pages) * SplashKvPageTokens * LATENT_STORE_WIDTH * sizeof(bfloat);
  device bfloat *key = reinterpret_cast<device bfloat *>(region) +
                       splash_kv_key_index<1, LATENT_STORE_WIDTH>(page, 0, token, 0);
  device bfloat *value = reinterpret_cast<device bfloat *>(region + values) +
                         splash_kv_value_index<1, LATENT_STORE_WIDTH, true>(page, 0, token, 0);
  const uint d = tid * 4;
  if (d < LATENT_STORE_WIDTH)
    *reinterpret_cast<device bfloat4 *>(key + d) =
        *reinterpret_cast<device const bfloat4 *>(latents + ulong(i) * LATENT_STORE_WIDTH + d);
  if (tid < kDim) value[tid] = index_keys[ulong(i) * kDim + tid];
}

// A score's key in the order of the scores: its bits, the sign flipped for a
// positive score and all of them for a negative one.
inline uint latent_score_key(float score) {
  const uint bits = as_type<uint>(score);
  return bits ^ ((bits >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}

constant constexpr uint kSelectThreads = LATENT_INDEX_SELECT_THREADS;
constant constexpr uint kSelectSimdgroups = kSelectThreads / 32;

// The exclusive prefix sum of each thread's `value` over the threadgroup, and
// the total in `total`.
inline uint latent_select_scan(uint value, threadgroup uint *partials, uint lane, uint sg, thread uint &total) {
  const uint before = simd_prefix_exclusive_sum(value);
  if (lane == 31) partials[sg] = before + value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const uint sums = lane < kSelectSimdgroups ? partials[lane] : 0u;
    const uint prefix = simd_prefix_exclusive_sum(sums);
    if (lane < kSelectSimdgroups) partials[kSelectSimdgroups + lane] = prefix;
    if (lane == kSelectSimdgroups - 1) partials[2 * kSelectSimdgroups] = prefix + sums;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  total = partials[2 * kSelectSimdgroups];
  const uint result = partials[kSelectSimdgroups + sg] + before;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}

// A row's top k scores (metal/abi/LatentIndex.h LatentSelectParams): the
// k-th highest key by an 8-bit radix select over the keys of the scores
// above -inf, then the scores above it and as many equal to it as k leaves,
// the lowest first, written in ascending order.
kernel void latent_index_select(device const float *scores [[buffer(0)]],
                                device uint *selected [[buffer(1)]],
                                constant LatentSelectParams &p [[buffer(2)]],
                                uint row [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup atomic_uint histogram[256];
  threadgroup uint partials[2 * kSelectSimdgroups + 1];
  threadgroup uint state[3];
  if (row >= p.rows) return;
  device const float *line = scores + ulong(row) * p.stride;
  device uint *out = selected + ulong(row) * p.out_stride;
  const float none = -numeric_limits<float>::infinity();
  // The scores above -inf.
  uint live = 0;
  for (uint i = tid; i < p.count; i += kSelectThreads) live += line[i] > none;
  uint total;
  latent_select_scan(live, partials, lane, sg, total);
  const bool all = total <= p.k;
  uint prefix = 0, mask = 0, remaining = p.k;
  for (int shift = 24; !all && shift >= 0; shift -= 8) {
    for (uint b = tid; b < 256; b += kSelectThreads) atomic_store_explicit(&histogram[b], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < p.count; i += kSelectThreads) {
      const float s = line[i];
      if (!(s > none)) continue;
      const uint key = latent_score_key(s);
      if ((key & mask) == prefix)
        atomic_fetch_add_explicit(&histogram[(key >> shift) & 0xFFu], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      // The bin the k-th highest key falls in, from the highest bin down.
      uint above = 0, bin = 255;
      for (;; --bin) {
        const uint n = atomic_load_explicit(&histogram[bin], memory_order_relaxed);
        if (above + n >= remaining || bin == 0) break;
        above += n;
      }
      state[0] = prefix | (bin << shift);
      state[1] = mask | (0xFFu << shift);
      state[2] = remaining - above;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    prefix = state[0];
    mask = state[1];
    remaining = state[2];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // In index order, chunk by chunk: the keys above the threshold and the first
  // `remaining` equal to it.
  uint ties = 0, written = 0;
  for (uint first = 0; first < p.count; first += kSelectThreads) {
    const uint i = first + tid;
    const float s = i < p.count ? line[i] : none;
    const bool alive = s > none;
    const uint key = alive ? latent_score_key(s) : 0u;
    const bool equal = !all && alive && key == prefix;
    uint chunkTies;
    const uint tieRank = ties + latent_select_scan(equal ? 1u : 0u, partials, lane, sg, chunkTies);
    const bool take = alive && (all || key > prefix || (equal && tieRank < remaining));
    uint chunkTaken;
    const uint at = written + latent_select_scan(take ? 1u : 0u, partials, lane, sg, chunkTaken);
    if (take && at < p.k) out[at] = i;
    ties += chunkTies;
    written += chunkTaken;
  }
  for (uint at = min(written, p.k) + tid; at < p.k; at += kSelectThreads) out[at] = LATENT_INDEX_NONE;
}

// Each row's block scores: the highest score of each block of `block` groups,
// -inf past the groups the row reaches, and +inf for the block of the last
// group it reaches. A thread per row and block.
kernel void latent_index_blocks(device const float *scores [[buffer(0)]],
                                device const uint *positions [[buffer(1)]],
                                device float *blocks [[buffer(2)]],
                                constant LatentBlockParams &p [[buffer(3)]],
                                uint2 group [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]]) {
  const uint row = group.x, b = group.y * kThreads + tid;
  if (row >= p.rows || b >= p.blocks) return;
  const uint reachable = min(p.groups, (positions[row] + 1) / p.ratio);
  float best = -numeric_limits<float>::infinity();
  for (uint g = b * p.block; g < min(reachable, (b + 1) * p.block); ++g)
    best = max(best, scores[ulong(row) * p.stride + g]);
  if (reachable && b == (reachable - 1) / p.block) best = numeric_limits<float>::infinity();
  blocks[ulong(row) * p.blocks + b] = best;
}

// A later layer's scores in place, -inf outside the row's selected blocks
// (an ascending list, LATENT_INDEX_NONE after its last). A thread per row and
// group, which finds its block in the list by bisection.
kernel void latent_index_mask(device float *scores [[buffer(0)]],
                              device const uint *kept [[buffer(1)]],
                              constant LatentBlockParams &p [[buffer(2)]],
                              uint2 group [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
  const uint row = group.x, g = group.y * kThreads + tid;
  if (row >= p.rows || g >= p.groups) return;
  device const uint *list = kept + ulong(row) * p.list_stride;
  const uint b = g / p.block;
  uint low = 0, high = p.list_stride;
  while (low < high) {
    const uint middle = (low + high) / 2;
    if (list[middle] < b) low = middle + 1;
    else high = middle;
  }
  if (low == p.list_stride || list[low] != b) scores[ulong(row) * p.stride + g] = -numeric_limits<float>::infinity();
}

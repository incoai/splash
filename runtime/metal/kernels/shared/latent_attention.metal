#include "metal/abi/LatentAttention.h"
#include "metal/kernels/common/kv_extent.h"
#include "metal/kernels/common/sgmatrix.h"

#include <metal_stdlib>

using namespace metal;

// Latent attention (metal/abi/LatentAttention.h, DeepSeek-V4.1's), as
// DeepSeek's model.py computes it (sparse_attn; in
// dev/tools/deepseek41_reference.py Forward.attention): each head's scores
// over the row's keys, scaled; their softmax beside the head's sink logit,
// which takes its share of the denominator and no value; and the sum of the
// keys, each its own value, by those weights. Three passes: the scores of
// every head and key, their softmax in place, and the weighted sums, over
// parts of the keys added by a fourth when a pass splits them.

constant constexpr uint kDim = LATENT_ATTENTION_DIM;
constant constexpr uint kThreads = LATENT_ATTENTION_THREADS;
constant constexpr uint kKeyBlock = LATENT_ATTENTION_KEY_BLOCK;
constant constexpr uint kHeadBlock = LATENT_ATTENTION_HEAD_BLOCK;
constant constexpr uint kKeysPerSimdgroup = kKeyBlock / (kThreads / 32);

// The value of LATENT_INDEX_NONE (metal/abi/LatentIndex.h), which ends a
// row's selection.
constant constexpr uint kNoGroup = 0xFFFFFFFFu;

// Where a pass's compressed keys are: a pool of latents, a key its index-th,
// or a sequence's pages, a key the latent at the place of the position its
// index names, through the sequence's page table (abi/KvExtent.h).
struct PooledKeys {
  using Source = device const bfloat *;
  static device const bfloat *at(Source pool, constant LatentAttentionParams &, uint index) {
    return pool + ulong(index) * kDim;
  }
};
struct PagedKeys {
  using Source = device const SplashKvPage *;
  static device const bfloat *at(Source pages, constant LatentAttentionParams &p, uint position) {
    const SplashKvPage entry = pages[position / SplashKvPageTokens];
    const uint page = splash_kv_page_index(entry);
    return reinterpret_cast<device const bfloat *>(splash_kv_extent(entry, page) + p.kv.offset) +
           splash_kv_key_index<1, kDim>(page, 0, position % SplashKvPageTokens, 0);
  }
};

// Key `index` of a row's list: a window latent, of the ring before the step
// below slot p.ring and of the step's past it, or a compressed one.
template <class Keys>
inline device const bfloat *latent_key(device const bfloat *window, device const bfloat *ring,
                                       typename Keys::Source compressed, constant LatentAttentionParams &p,
                                       uint index) {
  if (!(index & LATENT_ATTENTION_WINDOW_KEY)) return Keys::at(compressed, p, index);
  const uint slot = index & ~LATENT_ATTENTION_WINDOW_KEY;
  return slot < p.ring ? ring + ulong(slot) * kDim : window + ulong(slot - p.ring) * kDim;
}

// The scaled scores of every head of a row with LATENT_ATTENTION_KEY_BLOCK
// of its keys: each simdgroup takes some of the keys, each lane the heads
// lane and lane + 32.
template <class Keys>
inline void latent_scores(device const bfloat *q, device const bfloat *window, typename Keys::Source compressed,
                          device const uint *indices, device const uint *counts, device float *scores,
                          constant LatentAttentionParams &p, device const bfloat *ring, uint2 group, uint lane,
                          uint sg) {
  const uint row = group.x;
  if (row >= p.rows) return;
  const uint count = counts[row];
  device const bfloat *row_q = q + ulong(row) * p.heads * kDim;
  for (uint i = 0; i < kKeysPerSimdgroup; ++i) {
    const uint k = group.y * kKeyBlock + sg * kKeysPerSimdgroup + i;
    if (k >= count) return;
    device const bfloat *key = latent_key<Keys>(window, ring, compressed, p, indices[ulong(row) * p.key_stride + k]);
    for (uint h = lane; h < p.heads; h += 32) {
      device const bfloat *head = row_q + ulong(h) * kDim;
      float sum = 0.0f;
      for (uint d = 0; d < kDim; d += 4)
        sum += dot(float4(*reinterpret_cast<device const bfloat4 *>(head + d)),
                   float4(*reinterpret_cast<device const bfloat4 *>(key + d)));
      scores[(ulong(row) * p.heads + h) * p.key_stride + k] = sum * p.scale;
    }
  }
}
kernel void latent_attention_scores(device const bfloat *q [[buffer(0)]], device const bfloat *window [[buffer(1)]],
                                    device const bfloat *compressed [[buffer(2)]],
                                    device const uint *indices [[buffer(3)]], device const uint *counts [[buffer(4)]],
                                    device float *scores [[buffer(5)]],
                                    constant LatentAttentionParams &p [[buffer(7)]],
                                    device const bfloat *ring [[buffer(6)]],
                                    uint2 group [[threadgroup_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]]) {
  latent_scores<PooledKeys>(q, window, compressed, indices, counts, scores, p, ring, group, lane, sg);
}
kernel void latent_attention_scores_paged(device const bfloat *q [[buffer(0)]],
                                          device const bfloat *window [[buffer(1)]],
                                          device const SplashKvPage *pages [[buffer(2)]],
                                          device const uint *indices [[buffer(3)]],
                                          device const uint *counts [[buffer(4)]],
                                          device float *scores [[buffer(5)]],
                                          constant LatentAttentionParams &p [[buffer(7)]],
                                          device const bfloat *ring [[buffer(6)]],
                                          uint2 group [[threadgroup_position_in_grid]],
                                          uint lane [[thread_index_in_simdgroup]],
                                          uint sg [[simdgroup_index_in_threadgroup]]) {
  latent_scores<PagedKeys>(q, window, pages, indices, counts, scores, p, ring, group, lane, sg);
}

// Each row's key list (LatentAttentionKeysParams): its window's slots in
// position order, then its selected groups, and their count; none past a
// limited lane's rows. A threadgroup per row; its threads count the
// selection's groups, which end at its first LATENT_INDEX_NONE.
kernel void latent_attention_keys(device const uint *positions [[buffer(0)]],
                                  device const uint *selected [[buffer(1)]],
                                  device uint *indices [[buffer(2)]],
                                  device uint *counts [[buffer(3)]],
                                  device const uint *limits [[buffer(4)]],
                                  constant LatentAttentionKeysParams &p [[buffer(5)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint partials[kThreads / 32];
  if (row >= p.rows) return;
  if (p.limited && row >= limits[p.lane]) {
    if (tid == 0) counts[row] = 0;
    return;
  }
  const uint position = positions[row], window = min(p.window, position + 1);
  device uint *out = indices + ulong(row) * p.key_stride;
  for (uint i = tid; i < window; i += kThreads) {
    const uint key = position + 1 - window + i;
    const uint slot = !p.split ? key % p.ring : key < p.committed ? key % p.ring : p.ring + key - p.committed;
    out[i] = slot | LATENT_ATTENTION_WINDOW_KEY;
  }
  device const uint *groups = selected + ulong(row) * p.selected;
  uint count = 0;
  for (uint i = tid; i < p.selected; i += kThreads) count += groups[i] != kNoGroup;
  count = simd_sum(count);
  if (lane == 0) partials[sg] = count;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint total = 0;
  for (uint g = 0; g < kThreads / 32; ++g) total += partials[g];
  for (uint i = tid; i < total; i += kThreads) out[window + i] = p.ratio ? (groups[i] + 1) * p.ratio - 1 : groups[i];
  if (tid == 0) counts[row] = window + total;
}

// A head's scores of a row in place by their softmax beside its sink: exp(s -
// peak) over their sum plus exp(sink - peak), peak the largest of them and
// the sink. A simdgroup per row and head.
kernel void latent_attention_softmax(device float *scores [[buffer(0)]],
                                     device const uint *counts [[buffer(1)]],
                                     device const float *sinks [[buffer(2)]],
                                     constant LatentAttentionParams &p [[buffer(3)]],
                                     uint2 group [[threadgroup_position_in_grid]],
                                     uint lane [[thread_index_in_simdgroup]]) {
  const uint row = group.x, h = group.y;
  if (row >= p.rows || h >= p.heads) return;
  const uint count = counts[row];
  device float *s = scores + (ulong(row) * p.heads + h) * p.key_stride;
  const float sink = sinks[h];
  float peak = sink;
  for (uint k = lane; k < count; k += 32) peak = max(peak, s[k]);
  peak = simd_max(peak);
  float sum = 0.0f;
  for (uint k = lane; k < count; k += 32) sum += exp(s[k] - peak);
  sum = simd_sum(sum) + exp(sink - peak);
  const float inverse = 1.0f / sum;
  for (uint k = lane; k < count; k += 32) s[k] = exp(s[k] - peak) * inverse;
}

// Each head's output of a row: its keys summed by its weights, fp32; with
// p.splits parts, part `split` of it, over that part of the keys, at
// out [rows][splits][heads][LATENT_ATTENTION_DIM]. A threadgroup per row,
// LATENT_ATTENTION_HEAD_BLOCK heads and part; a thread takes four values of
// each of the heads, so each key's value is read once for them.
template <class Keys>
inline void latent_values(device const float *weights, device const bfloat *window, typename Keys::Source compressed,
                          device const uint *indices, device const uint *counts, device float *out,
                          constant LatentAttentionParams &p, device const bfloat *ring, uint3 group, uint tid) {
  const uint row = group.x, first = group.y * kHeadBlock, split = group.z, d = tid * 4;
  if (row >= p.rows || first >= p.heads || split >= p.splits || d >= kDim) return;
  const uint heads = min(kHeadBlock, p.heads - first);
  const uint count = counts[row], begin = split * count / p.splits, end = (split + 1) * count / p.splits;
  device const uint *row_indices = indices + ulong(row) * p.key_stride;
  device const float *row_weights = weights + (ulong(row) * p.heads + first) * p.key_stride;
  float4 sums[kHeadBlock];
  for (uint h = 0; h < kHeadBlock; ++h) sums[h] = 0.0f;
  for (uint k = begin; k < end; ++k) {
    const float4 value =
        float4(*reinterpret_cast<device const bfloat4 *>(latent_key<Keys>(window, ring, compressed, p, row_indices[k]) + d));
    for (uint h = 0; h < kHeadBlock; ++h)
      if (h < heads) sums[h] = fma(float4(row_weights[ulong(h) * p.key_stride + k]), value, sums[h]);
  }
  device float *row_out = out + ((ulong(row) * p.splits + split) * p.heads + first) * kDim + d;
  for (uint h = 0; h < heads; ++h) *reinterpret_cast<device float4 *>(row_out + ulong(h) * kDim) = sums[h];
}
kernel void latent_attention_values(device const float *weights [[buffer(0)]],
                                    device const bfloat *window [[buffer(1)]],
                                    device const bfloat *compressed [[buffer(2)]],
                                    device const uint *indices [[buffer(3)]], device const uint *counts [[buffer(4)]],
                                    device float *out [[buffer(5)]], constant LatentAttentionParams &p [[buffer(7)]],
                                    device const bfloat *ring [[buffer(6)]],
                                    uint3 group [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
  latent_values<PooledKeys>(weights, window, compressed, indices, counts, out, p, ring, group, tid);
}
kernel void latent_attention_values_paged(device const float *weights [[buffer(0)]],
                                          device const bfloat *window [[buffer(1)]],
                                          device const SplashKvPage *pages [[buffer(2)]],
                                          device const uint *indices [[buffer(3)]],
                                          device const uint *counts [[buffer(4)]], device float *out [[buffer(5)]],
                                          constant LatentAttentionParams &p [[buffer(7)]],
                                          device const bfloat *ring [[buffer(6)]],
                                          uint3 group [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]]) {
  latent_values<PagedKeys>(weights, window, pages, indices, counts, out, p, ring, group, tid);
}

// The matrix passes of a step of many rows (LATENT_ATTENTION_MMA_*), whose
// rows each multiply their 64 heads' queries by their keys and their weights
// by the keys' values in 8x8 simdgroup matrices (common/sgmatrix.h): a
// row's 64 heads are a whole matrix of their own, whose keys no other row
// shares in its order.
constant constexpr uint kMmaKeys = LATENT_ATTENTION_MMA_KEYS;
constant constexpr uint kMmaValues = LATENT_ATTENTION_MMA_VALUES;
constant constexpr uint kMmaHeads = LATENT_ATTENTION_MAX_HEADS;
// The latent values a scores step stages of the queries and the keys.
constant constexpr uint kMmaChunk = 64;
// The keys a values step stages.
constant constexpr uint kMmaValueKeys = 32;
// A pass's threads each stage half a head's and half a key's chunk, or a
// quarter of a key's values; its simdgroups each take 16 heads.
static_assert(LATENT_ATTENTION_MMA_THREADS == 2 * kMmaHeads && LATENT_ATTENTION_MMA_THREADS == 2 * kMmaKeys &&
                  LATENT_ATTENTION_MMA_THREADS == 4 * kMmaValueKeys &&
                  LATENT_ATTENTION_MMA_THREADS / 32 * 16 == kMmaHeads,
              "the matrix passes' staging takes LATENT_ATTENTION_MMA_THREADS threads");

// The scaled scores of LATENT_ATTENTION_MMA_KEYS keys of a row for every
// head, bf16 products summed in fp32: the queries [heads][64] and the keys
// transposed [64][keys] staged a chunk of 64 latent values at a time,
// simdgroup sg the heads 16 sg to 16 sg + 15 against every key of the block.
// Heads past p.heads stage zeros and keys past the row's count zeros, whose
// scores are not written.
template <class Keys>
inline void latent_scores_mma(device const bfloat *q, device const bfloat *window, typename Keys::Source compressed,
                              device const uint *indices, device const uint *counts, device float *scores,
                              constant LatentAttentionParams &p, device const bfloat *ring, uint2 group, uint tid,
                              uint lane, uint sg, threadgroup bfloat *queries, threadgroup bfloat *keys) {
  const uint row = group.x, first = group.y * kMmaKeys;
  if (row >= p.rows) return;
  const uint count = counts[row];
  if (first >= count) return;
  device const bfloat *row_q = q + ulong(row) * p.heads * kDim;
  // The thread's share of the staging: head tid / 2's 32 values of a chunk,
  // and key tid / 2's.
  const uint stage = tid / 2, part = (tid % 2) * 32;
  device const bfloat *head = stage < p.heads ? row_q + ulong(stage) * kDim : nullptr;
  device const bfloat *key =
      first + stage < count
          ? latent_key<Keys>(window, ring, compressed, p, indices[ulong(row) * p.key_stride + first + stage])
          : nullptr;
  const sgmatrix::Lane m = sgmatrix::lane_map(lane);
  float2 acc[2][kMmaKeys / 8];
  for (uint a = 0; a < 2; ++a)
    for (uint b = 0; b < kMmaKeys / 8; ++b) acc[a][b] = 0.0f;
  for (uint chunk = 0; chunk < kDim; chunk += kMmaChunk) {
    for (uint i = 0; i < 32; i += 4) {
      *reinterpret_cast<threadgroup bfloat4 *>(queries + stage * kMmaChunk + part + i) =
          head ? *reinterpret_cast<device const bfloat4 *>(head + chunk + part + i) : bfloat4(bfloat(0.0f));
      const bfloat4 v = key ? *reinterpret_cast<device const bfloat4 *>(key + chunk + part + i) : bfloat4(bfloat(0.0f));
      for (uint j = 0; j < 4; ++j) keys[(part + i + j) * kMmaKeys + stage] = v[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = 0; d < kMmaChunk; d += 8) {
      bfloat2 a[2];
      for (uint i = 0; i < 2; ++i)
        a[i] = *reinterpret_cast<threadgroup const bfloat2 *>(queries + ((sg * 2 + i) * 8 + m.fm) * kMmaChunk + d + m.fn);
      for (uint b = 0; b < kMmaKeys / 8; ++b) {
        const bfloat2 k = *reinterpret_cast<threadgroup const bfloat2 *>(keys + (d + m.fm) * kMmaKeys + b * 8 + m.fn);
        sgmatrix::mma_acc<bfloat>(acc[0][b], a[0], k);
        sgmatrix::mma_acc<bfloat>(acc[1][b], a[1], k);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 2; ++i) {
    const uint h = (sg * 2 + i) * 8 + m.fm;
    if (h >= p.heads) continue;
    device float *row_scores = scores + (ulong(row) * p.heads + h) * p.key_stride;
    for (uint b = 0; b < kMmaKeys / 8; ++b) {
      const uint k = first + b * 8 + m.fn;
      if (k < count) row_scores[k] = acc[i][b].x * p.scale;
      if (k + 1 < count) row_scores[k + 1] = acc[i][b].y * p.scale;
    }
  }
}
kernel void latent_attention_scores_mma(device const bfloat *q [[buffer(0)]],
                                        device const bfloat *window [[buffer(1)]],
                                        device const bfloat *compressed [[buffer(2)]],
                                        device const uint *indices [[buffer(3)]],
                                        device const uint *counts [[buffer(4)]], device float *scores [[buffer(5)]],
                                        constant LatentAttentionParams &p [[buffer(7)]],
                                        device const bfloat *ring [[buffer(6)]],
                                        uint2 group [[threadgroup_position_in_grid]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup bfloat queries[kMmaHeads * kMmaChunk];
  threadgroup bfloat keys[kMmaChunk * kMmaKeys];
  latent_scores_mma<PooledKeys>(q, window, compressed, indices, counts, scores, p, ring, group, tid, lane, sg, queries,
                                keys);
}
kernel void latent_attention_scores_mma_paged(device const bfloat *q [[buffer(0)]],
                                              device const bfloat *window [[buffer(1)]],
                                              device const SplashKvPage *pages [[buffer(2)]],
                                              device const uint *indices [[buffer(3)]],
                                              device const uint *counts [[buffer(4)]],
                                              device float *scores [[buffer(5)]],
                                              constant LatentAttentionParams &p [[buffer(7)]],
                                              device const bfloat *ring [[buffer(6)]],
                                              uint2 group [[threadgroup_position_in_grid]],
                                              uint tid [[thread_index_in_threadgroup]],
                                              uint lane [[thread_index_in_simdgroup]],
                                              uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup bfloat queries[kMmaHeads * kMmaChunk];
  threadgroup bfloat keys[kMmaChunk * kMmaKeys];
  latent_scores_mma<PagedKeys>(q, window, pages, indices, counts, scores, p, ring, group, tid, lane, sg, queries,
                               keys);
}

// Each head's output of a row over LATENT_ATTENTION_MMA_VALUES of its latent
// values, in fp32: its weights [heads][keys] (the softmax of its scores) by
// the keys' values, 32 keys at a time staged as fp32 [keys][values]; simdgroup
// sg the heads 16 sg to 16 sg + 15 against every value of the block. Weights
// past the row's count read as zeros, and their keys stage zeros.
template <class Keys>
inline void latent_values_mma(device const float *weights, device const bfloat *window,
                              typename Keys::Source compressed, device const uint *indices, device const uint *counts,
                              device float *out, constant LatentAttentionParams &p, device const bfloat *ring,
                              uint2 group, uint tid, uint lane, uint sg, threadgroup float *values) {
  const uint row = group.x, first = group.y * kMmaValues;
  if (row >= p.rows) return;
  const uint count = counts[row];
  device const uint *row_indices = indices + ulong(row) * p.key_stride;
  device const float *row_weights = weights + ulong(row) * p.heads * p.key_stride;
  // The thread's share of the staging: key tid / 4's 16 values of the block.
  const uint stage = tid / 4, part = (tid % 4) * 16;
  const sgmatrix::Lane m = sgmatrix::lane_map(lane);
  float2 acc[2][kMmaValues / 8];
  for (uint a = 0; a < 2; ++a)
    for (uint b = 0; b < kMmaValues / 8; ++b) acc[a][b] = 0.0f;
  for (uint block = 0; block < count; block += kMmaValueKeys) {
    const uint k = block + stage;
    device const bfloat *key = k < count ? latent_key<Keys>(window, ring, compressed, p, row_indices[k]) : nullptr;
    for (uint i = 0; i < 16; i += 4)
      *reinterpret_cast<threadgroup float4 *>(values + stage * kMmaValues + part + i) =
          key ? float4(*reinterpret_cast<device const bfloat4 *>(key + first + part + i)) : float4(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 0; step < kMmaValueKeys; step += 8) {
      float2 w[2];
      const uint kw = block + step + m.fn;
      for (uint i = 0; i < 2; ++i) {
        const uint h = (sg * 2 + i) * 8 + m.fm;
        device const float *head = row_weights + ulong(h) * p.key_stride;
        w[i] = float2(h < p.heads && kw < count ? head[kw] : 0.0f, h < p.heads && kw + 1 < count ? head[kw + 1] : 0.0f);
      }
      for (uint b = 0; b < kMmaValues / 8; ++b) {
        const float2 v = *reinterpret_cast<threadgroup const float2 *>(values + (step + m.fm) * kMmaValues + b * 8 + m.fn);
        sgmatrix::mma_acc<float>(acc[0][b], w[0], v);
        sgmatrix::mma_acc<float>(acc[1][b], w[1], v);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 2; ++i) {
    const uint h = (sg * 2 + i) * 8 + m.fm;
    if (h >= p.heads) continue;
    device float *row_out = out + (ulong(row) * p.heads + h) * kDim + first;
    for (uint b = 0; b < kMmaValues / 8; ++b) *reinterpret_cast<device float2 *>(row_out + b * 8 + m.fn) = acc[i][b];
  }
}
kernel void latent_attention_values_mma(device const float *weights [[buffer(0)]],
                                        device const bfloat *window [[buffer(1)]],
                                        device const bfloat *compressed [[buffer(2)]],
                                        device const uint *indices [[buffer(3)]],
                                        device const uint *counts [[buffer(4)]], device float *out [[buffer(5)]],
                                        constant LatentAttentionParams &p [[buffer(7)]],
                                        device const bfloat *ring [[buffer(6)]],
                                        uint2 group [[threadgroup_position_in_grid]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float values[kMmaValueKeys * kMmaValues];
  latent_values_mma<PooledKeys>(weights, window, compressed, indices, counts, out, p, ring, group, tid, lane, sg,
                                values);
}
kernel void latent_attention_values_mma_paged(device const float *weights [[buffer(0)]],
                                              device const bfloat *window [[buffer(1)]],
                                              device const SplashKvPage *pages [[buffer(2)]],
                                              device const uint *indices [[buffer(3)]],
                                              device const uint *counts [[buffer(4)]],
                                              device float *out [[buffer(5)]],
                                              constant LatentAttentionParams &p [[buffer(7)]],
                                              device const bfloat *ring [[buffer(6)]],
                                              uint2 group [[threadgroup_position_in_grid]],
                                              uint tid [[thread_index_in_threadgroup]],
                                              uint lane [[thread_index_in_simdgroup]],
                                              uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float values[kMmaValueKeys * kMmaValues];
  latent_values_mma<PagedKeys>(weights, window, pages, indices, counts, out, p, ring, group, tid, lane, sg, values);
}

// Each head's output of a row from the values pass's parts, parts [rows]
// [splits][heads][LATENT_ATTENTION_DIM], added in part order. A thread per
// four values.
kernel void latent_attention_sum(device const float *parts [[buffer(0)]],
                                 device float *out [[buffer(1)]],
                                 constant LatentAttentionParams &p [[buffer(2)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
  const uint row = group.x, i = (group.y * kThreads + tid) * 4, width = p.heads * kDim;
  if (row >= p.rows || i >= width) return;
  device const float *part = parts + ulong(row) * p.splits * width + i;
  float4 sum = 0.0f;
  for (uint s = 0; s < p.splits; ++s) sum += *reinterpret_cast<device const float4 *>(part + ulong(s) * width);
  *reinterpret_cast<device float4 *>(out + ulong(row) * width + i) = sum;
}

// A sequence's ring after a step (LatentRingParams): a threadgroup per slot,
// a thread four values of its latent.
kernel void latent_ring_update(device const bfloat *ring_in [[buffer(0)]], device const bfloat *step [[buffer(1)]],
                               device bfloat *ring_out [[buffer(2)]], device const uint *retained [[buffer(3)]],
                               constant LatentRingParams &p [[buffer(4)]], uint slot [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
  const uint d = tid * 4;
  if (slot >= p.ring || d >= kDim) return;
  const uint kept = p.retained ? min(retained[p.lane], p.rows) : p.rows, end = p.committed + kept;
  device const bfloat *from = ring_in + ulong(slot) * kDim;
  if (end) {
    const uint last = end - 1, back = (last + p.ring - slot) % p.ring;
    if (last >= back && last - back >= p.committed) from = step + ulong(last - back - p.committed) * kDim;
  }
  *reinterpret_cast<device bfloat4 *>(ring_out + ulong(slot) * kDim + d) =
      *reinterpret_cast<device const bfloat4 *>(from + d);
}

// A paired KV source's open group after a verify (LatentOpenParams): a thread
// per value of its kv and gate projections.
kernel void latent_open_commit(device const float *kv [[buffer(0)]], device const float *gate [[buffer(1)]],
                               device float *open [[buffer(2)]], device const uint *retained [[buffer(3)]],
                               constant LatentOpenParams &p [[buffer(4)]], uint d [[thread_position_in_grid]]) {
  const uint kept = retained[p.lane];
  if (d >= p.width || !((p.committed + kept) & 1)) return;
  const ulong row = ulong(kept) * p.width;
  open[d] = kv[row + d];
  open[p.width + d] = gate[row + d];
}

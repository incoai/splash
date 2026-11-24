// Decoder attention of the Omni models (ops::OmniAttention): Qwen3's 16
// query heads over 8 KV heads of 128 dimensions, bf16 KV in 32-token pages
// (metal/abi/Omni.h). The prepare kernel normalizes and rotates each row's
// queries and keys and stores the row's keys and values into its page slot;
// prefill tiles of 8 rows run MPP matmuls over the pages of their sequence,
// split over the history and reduced in split order; decode rows run one
// simdgroup kernel per (lane, KV head, split).
//
// Queries are KV-head-major: [kv head][row][query head in group][128], the
// rows of a dispatch padded to whole tiles of 8, so a tile's 16 fused rows
// (8 rows x 2 heads) are one contiguous M x D tensor.

#include "metal/abi/Omni.h"
#include "metal/kernels/common/split_reduce.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr uint kQueryHeads = OMNI_DECODER_QUERY_HEADS;
constant constexpr uint kKvHeads = OMNI_DECODER_KV_HEADS;
constant constexpr uint kGroup = kQueryHeads / kKvHeads;
constant constexpr uint kHeadDim = OMNI_DECODER_HEAD_DIM;
constant constexpr uint kPageTokens = OMNI_PAGE_TOKENS;
constant constexpr uint kQkvWidth = OMNI_DECODER_QKV_WIDTH;
constant constexpr uint kAttentionWidth = OMNI_DECODER_ATTENTION_WIDTH;
constant constexpr float kScale = 0.08838834764831845f; // 1 / sqrt(128)

inline uint omni_padded_rows(uint rows) { return (rows + OMNI_PREFILL_TILE_ROWS - 1) / OMNI_PREFILL_TILE_ROWS * OMNI_PREFILL_TILE_ROWS; }

// The page of `token` of sequence `s` in `layer`.
inline uint omni_page_of(constant OmniSequenceParams &s, device const uint *page_table, uint layer, uint token) {
  return page_table[s.page_table_offset + layer * s.page_table_stride + token / kPageTokens];
}

inline device bfloat *omni_page_keys(device uchar *pages, uint page, uint kv_head) {
  return reinterpret_cast<device bfloat *>(pages + ulong(page) * OMNI_PAGE_BYTES) +
         ulong(kv_head) * kPageTokens * kHeadDim;
}
inline device bfloat *omni_page_values(device uchar *pages, uint page, uint kv_head) {
  return reinterpret_cast<device bfloat *>(pages + ulong(page) * OMNI_PAGE_BYTES + OMNI_PAGE_KEY_BYTES) +
         ulong(kv_head) * kPageTokens * kHeadDim;
}

// One task per (row, head): 16 query heads then 8 KV heads; 128 threads, one
// per dimension. q/k RMSNorm in fp32 over the head, then rotate-half RoPE at
// the row's position (angles in fp32, as the reference computes them).
kernel void omni_qkv_prepare(device const bfloat *qkv [[buffer(0)]], device const bfloat *q_norm [[buffer(1)]],
                             device const bfloat *k_norm [[buffer(2)]],
                             device const float *inverse_frequencies [[buffer(3)]],
                             device const uint *row_sequence [[buffer(4)]],
                             constant OmniSequenceParams *sequences [[buffer(5)]],
                             device const uint *page_table [[buffer(6)]], device uchar *pages [[buffer(7)]],
                             device bfloat *queries [[buffer(8)]], constant OmniQkvParams &p [[buffer(9)]],
                             uint task [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float reductions[4];
  threadgroup float normalized[kHeadDim];
  const uint row = task / (kQueryHeads + kKvHeads);
  const uint head = task % (kQueryHeads + kKvHeads);
  if (row >= p.rows) return;
  const uint sequence = row_sequence[row];
  if (sequence >= p.sequences) return;
  constant OmniSequenceParams &s = sequences[sequence];
  const uint local = row - s.row_offset;
  const uint position = s.position_offset + local;
  const uint token = s.committed_tokens + local;
  const bool query = head < kQueryHeads;
  device const bfloat *source = qkv + ulong(row) * kQkvWidth + head * kHeadDim;
  device const bfloat *weight = query ? q_norm : k_norm;
  float element = float(source[tid]);
  float sum = simd_sum(element * element);
  if (lane == 0) reductions[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inverse = rsqrt((reductions[0] + reductions[1] + reductions[2] + reductions[3]) / float(kHeadDim) +
                              p.rms_epsilon);
  normalized[tid] = float(bfloat(element * inverse * float(weight[tid])));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device bfloat *destination;
  if (query) {
    const uint kv_head = head / kGroup, member = head % kGroup;
    destination = queries + ((ulong(kv_head) * omni_padded_rows(p.rows) + row) * kGroup + member) * kHeadDim;
  } else {
    const uint kv_head = head - kQueryHeads;
    const uint page = omni_page_of(s, page_table, p.layer, token);
    destination = omni_page_keys(pages, page, kv_head) + (token % kPageTokens) * kHeadDim;
    // The row's values go to the same slot of the value plane as stored.
    device bfloat *values = omni_page_values(pages, page, kv_head) + (token % kPageTokens) * kHeadDim;
    values[tid] = source[kKvHeads * kHeadDim + tid];
  }
  constexpr uint rotary = kHeadDim / 2;
  if (tid < rotary) {
    const float angle = float(position) * inverse_frequencies[tid];
    const float cosine = precise::cos(angle), sine = precise::sin(angle);
    const float first = normalized[tid], second = normalized[tid + rotary];
    destination[tid] = bfloat(first * cosine - second * sine);
    destination[tid + rotary] = bfloat(second * cosine + first * sine);
  }
}

// One 32-token page of a tile: scaled scores become bf16 probabilities and
// the fused rows' statistics advance (online softmax). Four lanes own one
// fused row, eight consecutive tokens each. A token is visible to a query row
// when it is before the row's own position plus one.
template <uint FusedRows, uint RowsPerTile>
inline void omni_page_softmax(threadgroup const float *scores, threadgroup bfloat *probabilities,
                              threadgroup float *row_max, threadgroup float *row_sum,
                              threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
                              uint token_start, uint committed_tokens, uint active_rows, uint tid) {
  constexpr uint N = kPageTokens, TokensPerLane = 8, LanesPerRow = N / TokensPerLane;
  if (tid >= LanesPerRow * FusedRows) return;
  const uint fused_row = tid / LanesPerRow;
  const uint column = tid % LanesPerRow * TokensPerLane;
  const uint query_row = fused_row / (FusedRows / RowsPerTile);
  const uint limit = committed_tokens + min(query_row, active_rows - 1) + 1;
  const uint token = token_start + column;
  float score[TokensPerLane];
  float local_max = -INFINITY;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    score[j] = token + j < limit ? scores[fused_row * N + column + j] * kScale : -INFINITY;
    local_max = max(local_max, score[j]);
  }
  local_max = max(local_max, simd_shuffle_xor(local_max, 1));
  local_max = max(local_max, simd_shuffle_xor(local_max, 2));
  const float previous_max = row_max[fused_row];
  const float next_max = max(previous_max, local_max);
  float local_sum = 0.0f;
  float probability[TokensPerLane];
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    probability[j] = token + j < limit ? fast::exp(score[j] - next_max) : 0.0f;
    local_sum += probability[j];
  }
  local_sum += simd_shuffle_xor(local_sum, 1);
  local_sum += simd_shuffle_xor(local_sum, 2);
  if (column == 0) {
    const float scale = next_max == -INFINITY || next_max == previous_max ? 1.0f : fast::exp(previous_max - next_max);
    previous_scale[fused_row] = scale;
    row_sum[fused_row] = row_sum[fused_row] * scale + local_sum;
    row_max[fused_row] = next_max;
    if (scale != 1.0f) atomic_store_explicit(rescale, 1u, memory_order_relaxed);
  }
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) probabilities[fused_row * N + column + j] = bfloat(probability[j]);
}

// A prefill tile: group.x the KV head, group.y the tile, group.z the split.
// Partials are [tile][kv head][split][16][128] fp32; statistics
// [slot][16]{max, sum}.
kernel void omni_attention_prefill_split(device bfloat *queries [[buffer(0)]], device const uint *page_table [[buffer(1)]],
                                         device uchar *pages [[buffer(2)]],
                                         constant OmniSequenceParams *sequences [[buffer(3)]],
                                         constant OmniPrefillTile *tiles [[buffer(4)]],
                                         device float *partials [[buffer(5)]], device float *statistics [[buffer(6)]],
                                         constant OmniAttentionParams &p [[buffer(7)]],
                                         uint3 group [[threadgroup_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]]) {
  constexpr ushort RowsPerTile = OMNI_PREFILL_TILE_ROWS;
  constexpr ushort M = RowsPerTile * kGroup;
  constexpr ushort N = kPageTokens;
  constexpr ushort D = kHeadDim;
  alignas(16) threadgroup float scores[M * N];
  alignas(16) threadgroup bfloat probabilities[M * N];
  threadgroup float row_max[M], row_sum[M], previous_scale[M];
  threadgroup atomic_uint rescale;
  const uint kv_head = group.x, tile_index = group.y, split = group.z;
  if (kv_head >= kKvHeads || tile_index >= p.tiles) return;
  constant OmniPrefillTile &tile = tiles[tile_index];
  if (tile.sequence >= p.sequences) return;
  constant OmniSequenceParams &s = sequences[tile.sequence];
  if (split >= s.splits) return;
  const uint committed = s.committed_tokens + (tile.first_row - s.row_offset);
  const uint active_rows = tile.rows;
  const uint visible = committed + active_rows;
  const uint page_count = (visible + N - 1) / N;
  const uint per_split = (page_count + s.splits - 1) / s.splits;
  const uint page_begin = split * per_split;
  if (page_begin >= page_count) return;
  const uint page_end = min(page_count, page_begin + per_split);
  if (tid < M) {
    row_max[tid] = -INFINITY;
    row_sum[tid] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device bfloat *tile_queries = queries + ((ulong(kv_head) * p.padded_rows + tile.first_row) * kGroup) * D;
  auto qt = tensor(tile_queries, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto st = tensor(scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto pt = tensor(probabilities, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto p0 = pt.slice<N, M>(0, 0);
  auto q0 = qt.slice<D, M>(0, 0);
  auto key_type = tensor(static_cast<device bfloat *>(nullptr), dextents<int, 2>{D, N}, array<int, 2>{1, D});
  auto value_type = tensor(static_cast<device bfloat *>(nullptr), dextents<int, 2>{D, N}, array<int, 2>{1, D});
  auto k0 = key_type.slice<D, N>(0, 0);
  auto v0 = value_type.slice<D, N>(0, 0);
  constexpr auto qk_descriptor = matmul2d_descriptor(M, N, D, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto pv_descriptor =
      matmul2d_descriptor(M, D, N, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
  matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
  auto running = pv.template get_destination_cooperative_tensor<decltype(p0), decltype(v0), float>();
#pragma unroll
  for (ushort index = 0; index < running.get_capacity(); ++index)
    if (running.is_valid_element(index)) running[index] = 0.0f;

  for (uint page = page_begin; page < page_end; ++page) {
    const uint page_index = omni_page_of(s, page_table, p.layer, page * N);
    auto kt = tensor(omni_page_keys(pages, page_index, kv_head), dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto vt = tensor(omni_page_values(pages, page_index, kv_head), dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto page_scores = qk.template get_destination_cooperative_tensor<decltype(q0), decltype(k0), float>();
    auto ks = kt.slice<D, N>(0, 0);
    qk.run(q0, ks, page_scores);
    page_scores.store(st.slice<N, M>(0, 0));
    if (tid == 0) atomic_store_explicit(&rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    omni_page_softmax<M, RowsPerTile>(scores, probabilities, row_max, row_sum, previous_scale, &rescale, page * N,
                                      committed, active_rows, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(&rescale, memory_order_relaxed)) {
#pragma unroll
      for (ushort index = 0; index < running.get_capacity(); ++index) {
        if (!running.is_valid_element(index)) continue;
        auto coordinates = running.get_multidimensional_index(index);
        running[index] *= previous_scale[coordinates[1]];
      }
    }
    auto vs = vt.slice<D, N>(0, 0);
    pv.run(p0, vs, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const ulong slot = (ulong(tile_index) * kKvHeads + kv_head) * OMNI_PREFILL_MAXIMUM_SPLITS + split;
  auto target = tensor(partials + slot * M * D, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  running.store(target.slice<D, M>(0, 0));
  if (tid < M) {
    statistics[(slot * M + tid) * 2] = row_max[tid];
    statistics[(slot * M + tid) * 2 + 1] = row_sum[tid];
  }
}

// Combines one fused row's splits in split order into the attention output
// row [row][16 heads x 128]. group.x the KV head, group.y the tile (or decode
// lane), group.z the fused row; 128 threads, one per dimension.
template <uint M>
inline void omni_attention_reduce_row(device const float *partials, device const float *statistics,
                                      device bfloat *output, uint committed, uint active_rows, uint splits,
                                      uint maximum_splits, uint kv_head, uint head_slot, uint fused_row,
                                      uint first_row, uint tid, threadgroup float *weights) {
  constexpr uint D = kHeadDim;
  const uint query_row = fused_row / kGroup, member = fused_row % kGroup;
  if (query_row >= active_rows) return;
  const uint visible = committed + active_rows;
  const uint page_count = (visible + kPageTokens - 1) / kPageTokens;
  const uint per_split = (page_count + splits - 1) / splits;
  const uint written = (page_count + per_split - 1) / per_split;
  float maximum = -INFINITY;
  for (uint split = 0; split < written; ++split)
    maximum = max(maximum, statistics[((head_slot + split) * M + fused_row) * 2]);
  if (tid < written) weights[tid] = fast::exp(statistics[((head_slot + tid) * M + fused_row) * 2] - maximum);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float numerator = 0.0f, denominator = 0.0f;
  for (uint split = 0; split < written; ++split) {
    const float weight = weights[split];
    numerator += weight * partials[((head_slot + split) * M + fused_row) * D + tid];
    denominator += weight * statistics[((head_slot + split) * M + fused_row) * 2 + 1];
  }
  (void)maximum_splits;
  output[ulong(first_row + query_row) * kAttentionWidth + (kv_head * kGroup + member) * D + tid] =
      bfloat(denominator > 0.0f ? numerator / denominator : 0.0f);
}

kernel void omni_attention_prefill_reduce(device const float *partials [[buffer(0)]],
                                          device const float *statistics [[buffer(1)]],
                                          device bfloat *output [[buffer(2)]],
                                          constant OmniSequenceParams *sequences [[buffer(3)]],
                                          constant OmniPrefillTile *tiles [[buffer(4)]],
                                          constant OmniAttentionParams &p [[buffer(5)]],
                                          uint3 group [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint M = OMNI_PREFILL_TILE_ROWS * kGroup;
  threadgroup float weights[OMNI_PREFILL_MAXIMUM_SPLITS];
  const uint kv_head = group.x, tile_index = group.y, fused_row = group.z;
  if (kv_head >= kKvHeads || tile_index >= p.tiles || fused_row >= M) return;
  constant OmniPrefillTile &tile = tiles[tile_index];
  if (tile.sequence >= p.sequences) return;
  constant OmniSequenceParams &s = sequences[tile.sequence];
  const uint committed = s.committed_tokens + (tile.first_row - s.row_offset);
  omni_attention_reduce_row<M>(partials, statistics, output, committed, tile.rows, s.splits,
                               OMNI_PREFILL_MAXIMUM_SPLITS, kv_head,
                               (tile_index * kKvHeads + kv_head) * OMNI_PREFILL_MAXIMUM_SPLITS, fused_row,
                               tile.first_row, tid, weights);
}

// One page of one KV head against M query vectors held in threadgroup
// memory, by one simdgroup: lane l reads 16 bytes (8 dims) of token
// 2 i + l / 16 in load i (16 loads cover the page's 32 x 128 keys, each
// one contiguous 512-byte span), so a token's score is its 16 lanes'
// partial dots reduced by four xor shuffles; the probabilities then weight
// the values read the same way, lane l accumulating dims 8 (l % 16) .. +7
// of its tokens, the two token parities added at the end. The running
// softmax state (max, sum) and the lane's 8-dim accumulators carry across
// pages.
template <uint M>
struct OmniPageState {
  float running_max[M], running_sum[M], acc[M][8];
  inline void clear() {
    for (uint m = 0; m < M; ++m) {
      running_max[m] = -INFINITY;
      running_sum[m] = 0.0f;
      for (uint j = 0; j < 8; ++j) acc[m][j] = 0.0f;
    }
  }
};

inline void omni_unpack_bf16x8(uint4 packed, thread float (&out)[8]) {
  out[0] = as_type<float>(packed.x << 16);
  out[1] = as_type<float>(packed.x & 0xffff0000u);
  out[2] = as_type<float>(packed.y << 16);
  out[3] = as_type<float>(packed.y & 0xffff0000u);
  out[4] = as_type<float>(packed.z << 16);
  out[5] = as_type<float>(packed.z & 0xffff0000u);
  out[6] = as_type<float>(packed.w << 16);
  out[7] = as_type<float>(packed.w & 0xffff0000u);
}

// `visible_of(m)` is the number of tokens query vector m may see.
template <uint M, class Visible>
inline void omni_attention_page(thread OmniPageState<M> &state, device const bfloat *keys,
                                device const bfloat *values, threadgroup const float (*query)[OMNI_DECODER_HEAD_DIM],
                                uint first_token, uint lane, Visible visible_of) {
  constexpr uint D = kHeadDim;
  const uint parity = lane / 16, dim0 = (lane % 16) * 8;
  // Scores: load i holds token 2 i + parity.
  float score[M][16];
  for (uint i = 0; i < 16; ++i) {
    const uint4 packed = *reinterpret_cast<device const uint4 *>(keys + (2 * i + parity) * D + dim0);
    float k[8];
    omni_unpack_bf16x8(packed, k);
#pragma unroll
    for (uint m = 0; m < M; ++m) {
      float dot = 0.0f;
#pragma unroll
      for (uint j = 0; j < 8; ++j) dot += query[m][dim0 + j] * k[j];
      dot += simd_shuffle_xor(dot, 8);
      dot += simd_shuffle_xor(dot, 4);
      dot += simd_shuffle_xor(dot, 2);
      dot += simd_shuffle_xor(dot, 1);
      score[m][i] = dot;
    }
  }
#pragma unroll
  for (uint m = 0; m < M; ++m) {
    const uint visible = visible_of(m);
    float lane_max = -INFINITY;
    for (uint i = 0; i < 16; ++i) {
      const uint token = first_token + 2 * i + parity;
      score[m][i] = token < visible ? score[m][i] * kScale : -INFINITY;
      lane_max = max(lane_max, score[m][i]);
    }
    const float page_max = simd_max(lane_max);
    const float next_max = max(state.running_max[m], page_max);
    const float scale = state.running_max[m] == -INFINITY ? 0.0f : fast::exp(state.running_max[m] - next_max);
    float lane_sum = 0.0f;
    for (uint i = 0; i < 16; ++i) {
      const float probability = score[m][i] == -INFINITY ? 0.0f : fast::exp(score[m][i] - next_max);
      score[m][i] = probability;
      lane_sum += probability;
    }
    // Every token's probability sits in its 16 lanes: count one of them.
    state.running_sum[m] = state.running_sum[m] * scale + simd_sum(lane % 16 == 0 ? lane_sum : 0.0f);
    state.running_max[m] = next_max;
#pragma unroll
    for (uint j = 0; j < 8; ++j) state.acc[m][j] *= scale;
  }
  // Values, read as the keys were: load i holds token 2 i + parity.
  for (uint i = 0; i < 16; ++i) {
    const uint4 packed = *reinterpret_cast<device const uint4 *>(values + (2 * i + parity) * D + dim0);
    float v[8];
    omni_unpack_bf16x8(packed, v);
#pragma unroll
    for (uint m = 0; m < M; ++m) {
      const float weight = score[m][i];
#pragma unroll
      for (uint j = 0; j < 8; ++j) state.acc[m][j] += weight * v[j];
    }
  }
}

// Decode rows over long histories: one threadgroup per (KV head, lane,
// split), its eight simdgroups taking the split's pages round-robin. The
// simdgroups' running states merge in threadgroup memory into one partial
// per split, [lane][kv head][split][2][128].
kernel void omni_attention_decode_split(device const bfloat *queries [[buffer(0)]],
                                        device const uint *page_table [[buffer(1)]], device uchar *pages [[buffer(2)]],
                                        constant OmniSequenceParams *sequences [[buffer(3)]],
                                        device float *partials [[buffer(4)]], device float *statistics [[buffer(5)]],
                                        constant OmniAttentionParams &p [[buffer(6)]],
                                        uint3 group [[threadgroup_position_in_grid]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint D = kHeadDim, N = kPageTokens, M = kGroup;
  threadgroup float query[M][D];
  threadgroup float group_output[8][M][D];
  threadgroup float group_max[8][M], group_sum[8][M];
  const uint kv_head = group.x, sequence = group.y, split = group.z;
  if (kv_head >= kKvHeads || sequence >= p.sequences) return;
  constant OmniSequenceParams &s = sequences[sequence];
  if (split >= s.splits) return;
  const uint visible = s.committed_tokens + 1;
  const uint page_count = (visible + N - 1) / N;
  const uint per_split = (page_count + s.splits - 1) / s.splits;
  const uint page_begin = split * per_split;
  if (page_begin >= page_count) return;
  const uint page_end = min(page_count, page_begin + per_split);
  for (uint i = tid; i < M * D; i += 256)
    query[i / D][i % D] = float(queries[((ulong(kv_head) * p.padded_rows + s.row_offset) * M) * D + i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  OmniPageState<M> state;
  state.clear();
  for (uint page = page_begin + sg; page < page_end; page += 8) {
    const uint page_index = omni_page_of(s, page_table, p.layer, page * N);
    omni_attention_page<M>(state, omni_page_keys(pages, page_index, kv_head), omni_page_values(pages, page_index, kv_head),
                           query, page * N, lane, [&](uint) { return visible; });
  }
  // The two token parities of each dim, then the simdgroup states merged
  // in simdgroup order by simdgroup 0.
#pragma unroll
  for (uint m = 0; m < M; ++m) {
#pragma unroll
    for (uint j = 0; j < 8; ++j) {
      const float both = state.acc[m][j] + simd_shuffle_xor(state.acc[m][j], 16);
      if (lane < 16) group_output[sg][m][lane * 8 + j] = both;
    }
    if (lane == 0) {
      group_max[sg][m] = state.running_max[m];
      group_sum[sg][m] = state.running_sum[m];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  const ulong slot = (ulong(sequence) * kKvHeads + kv_head) * OMNI_DECODE_MAXIMUM_SPLITS + split;
#pragma unroll
  for (uint m = 0; m < M; ++m) {
    float maximum = -INFINITY;
    for (uint g = 0; g < 8; ++g) maximum = max(maximum, group_max[g][m]);
    float total = 0.0f, merged[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint g = 0; g < 8; ++g) {
      const float weight = group_max[g][m] == -INFINITY ? 0.0f : fast::exp(group_max[g][m] - maximum);
      total += weight * group_sum[g][m];
#pragma unroll
      for (uint j = 0; j < 4; ++j) merged[j] += weight * group_output[g][m][lane + 32 * j];
    }
#pragma unroll
    for (uint j = 0; j < 4; ++j) partials[(slot * M + m) * D + lane + 32 * j] = merged[j];
    if (lane == 0) {
      statistics[(slot * M + m) * 2] = maximum;
      statistics[(slot * M + m) * 2 + 1] = total;
    }
  }
}

kernel void omni_attention_decode_reduce(device const float *partials [[buffer(0)]],
                                         device const float *statistics [[buffer(1)]],
                                         device bfloat *output [[buffer(2)]],
                                         constant OmniSequenceParams *sequences [[buffer(3)]],
                                         constant OmniAttentionParams &p [[buffer(4)]],
                                         uint3 group [[threadgroup_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint M = kGroup;
  threadgroup float weights[OMNI_DECODE_MAXIMUM_SPLITS];
  const uint kv_head = group.x, sequence = group.y, fused_row = group.z;
  if (kv_head >= kKvHeads || sequence >= p.sequences || fused_row >= M) return;
  constant OmniSequenceParams &s = sequences[sequence];
  omni_attention_reduce_row<M>(partials, statistics, output, s.committed_tokens, 1, s.splits,
                               OMNI_DECODE_MAXIMUM_SPLITS, kv_head,
                               (sequence * kKvHeads + kv_head) * OMNI_DECODE_MAXIMUM_SPLITS, fused_row,
                               s.row_offset, tid, weights);
}

// Dense attention of the encoders: p.heads heads of 64 (16 for the 1.7B
// tower, 14 for the 0.6B's), every head its own keys, Q, K and V rows of
// qkv_stride elements (a head's 64 at head * 64), output rows of heads * 64. A threadgroup owns one head and 16 query rows; it walks the
// key blocks of 32 that the rows' ranges cover and masks per element: a key
// is visible to a row when it is inside the row's [key_begin, key_end) and,
// for the causal kind, key index > query index - window and <= query index.
kernel void omni_attention_dense(device bfloat *q [[buffer(0)]], device bfloat *k [[buffer(1)]],
                                 device bfloat *v [[buffer(2)]], device bfloat *output [[buffer(3)]],
                                 constant OmniDenseRow *rows [[buffer(4)]], constant OmniDenseParams &p [[buffer(5)]],
                                 uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr ushort M = OMNI_DENSE_TILE_ROWS, N = OMNI_DENSE_KEY_BLOCK, D = OMNI_ENCODER_HEAD_DIM;
  const uint Heads = p.heads;
  alignas(16) threadgroup float scores[M * N];
  alignas(16) threadgroup bfloat probabilities[M * N];
  threadgroup float row_max[M], row_sum[M], previous_scale[M];
  threadgroup atomic_uint rescale;
  const uint head = group.y;
  const uint first_row = group.x * M;
  if (head >= Heads || first_row >= p.rows) return;
  const uint active = min(uint(M), p.rows - first_row);
  uint key_lo = 0xffffffffu, key_hi = 0;
  for (uint r = 0; r < active; ++r) {
    key_lo = min(key_lo, rows[first_row + r].key_begin);
    key_hi = max(key_hi, rows[first_row + r].key_end);
  }
  if (tid < M) {
    row_max[tid] = -INFINITY;
    row_sum[tid] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int stride = int(p.qkv_stride);
  auto qt = tensor(q + ulong(first_row) * p.qkv_stride + head * D, dextents<int, 2>{D, M}, array<int, 2>{1, stride});
  auto st = tensor(scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto pt = tensor(probabilities, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto p0 = pt.slice<N, M>(0, 0);
  auto q0 = qt.slice<D, M>(0, 0);
  auto key_type = tensor(static_cast<device bfloat *>(nullptr), dextents<int, 2>{D, N}, array<int, 2>{1, stride});
  auto k0 = key_type.slice<D, N>(0, 0);
  constexpr auto qk_descriptor = matmul2d_descriptor(M, N, D, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto pv_descriptor =
      matmul2d_descriptor(M, D, N, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
  matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
  auto running = pv.template get_destination_cooperative_tensor<decltype(p0), decltype(k0), float>();
#pragma unroll
  for (ushort index = 0; index < running.get_capacity(); ++index)
    if (running.is_valid_element(index)) running[index] = 0.0f;
  const uint block_begin = key_lo / N, block_end = (key_hi + N - 1) / N;
  for (uint block = block_begin; block < block_end; ++block) {
    const uint key_start = block * N;
    auto kt = tensor(k + ulong(key_start) * p.qkv_stride + head * D, dextents<int, 2>{D, N}, array<int, 2>{1, stride});
    auto vt = tensor(v + ulong(key_start) * p.qkv_stride + head * D, dextents<int, 2>{D, N}, array<int, 2>{1, stride});
    auto block_scores = qk.template get_destination_cooperative_tensor<decltype(q0), decltype(k0), float>();
    auto ks = kt.slice<D, N>(0, 0);
    qk.run(q0, ks, block_scores);
    block_scores.store(st.slice<N, M>(0, 0));
    if (tid == 0) atomic_store_explicit(&rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // One thread per row: the row's 32 keys of this block, masked.
    if (tid < M) {
      const uint row = tid;
      const bool live = row < active;
      constant OmniDenseRow &descriptor = rows[first_row + min(row, active - 1)];
      float value[N];
      float local_max = -INFINITY;
      for (uint j = 0; j < N; ++j) {
        const uint key = key_start + j;
        bool visible = live && key >= descriptor.key_begin && key < descriptor.key_end;
        if (p.mask == 1u) {
          const uint query_index = descriptor.query_index;
          visible = visible && key <= query_index && (p.window == 0 || key + p.window > query_index);
        }
        value[j] = visible ? scores[row * N + j] * p.scale : -INFINITY;
        local_max = max(local_max, value[j]);
      }
      const float previous_max = row_max[row];
      const float next_max = max(previous_max, local_max);
      float local_sum = 0.0f;
      for (uint j = 0; j < N; ++j) {
        const float probability = value[j] == -INFINITY ? 0.0f : fast::exp(value[j] - next_max);
        probabilities[row * N + j] = bfloat(probability);
        local_sum += probability;
      }
      const float scale = next_max == -INFINITY || next_max == previous_max ? 1.0f : fast::exp(previous_max - next_max);
      previous_scale[row] = scale;
      row_sum[row] = row_sum[row] * scale + local_sum;
      row_max[row] = next_max;
      if (scale != 1.0f) atomic_store_explicit(&rescale, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(&rescale, memory_order_relaxed)) {
#pragma unroll
      for (ushort index = 0; index < running.get_capacity(); ++index) {
        if (!running.is_valid_element(index)) continue;
        auto coordinates = running.get_multidimensional_index(index);
        running[index] *= previous_scale[coordinates[1]];
      }
    }
    auto vs = vt.slice<D, N>(0, 0);
    pv.run(p0, vs, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // Normalize and store the active rows.
#pragma unroll
  for (ushort index = 0; index < running.get_capacity(); ++index) {
    if (!running.is_valid_element(index)) continue;
    auto coordinates = running.get_multidimensional_index(index);
    const uint row = coordinates[1], d = coordinates[0];
    if (row >= active) continue;
    const float denominator = row_sum[row];
    output[ulong(first_row + row) * (Heads * D) + head * D + d] =
        bfloat(denominator > 0.0f ? running[index] / denominator : 0.0f);
  }
  (void)lane;
  (void)sg;
}

// Prepares row r of a fused decode dispatch in one simdgroup: `part` 0 and 1
// normalize and rotate its two query heads of the threadgroup's KV head into
// `query`, 2 its key head into the row's page slot, 3 copies its values
// there: lane l holds dims l, l+32, l+64, l+96.
inline void omni_fused_prepare(device const bfloat *qkv, device const bfloat *q_norm, device const bfloat *k_norm,
                               device const float *inverse_frequencies, constant OmniSequenceParams &s,
                               device const uint *page_table, device uchar *pages, device float2 *rotations,
                               constant OmniQkvParams &p, threadgroup float (*query)[OMNI_DECODER_HEAD_DIM],
                               uint kv_head, uint r, uint part, uint lane) {
  constexpr uint D = kHeadDim, N = kPageTokens;
  const uint row = s.row_offset + r;
  const uint position = s.position_offset + r;
  const uint token = s.committed_tokens + r;
  const uint page_index = omni_page_of(s, page_table, p.layer, token);
  device bfloat *stored_keys = omni_page_keys(pages, page_index, kv_head) + (token % N) * D;
  device bfloat *stored_values = omni_page_values(pages, page_index, kv_head) + (token % N) * D;
  device const bfloat *source = qkv + ulong(row) * kQkvWidth;
  if (part < 3) {
    const uint head = part < 2 ? kv_head * kGroup + part : kQueryHeads + kv_head;
    device const bfloat *weight = part < 2 ? q_norm : k_norm;
    float x[4];
#pragma unroll
    for (uint j = 0; j < 4; ++j) x[j] = float(source[head * D + lane + 32 * j]);
    // A later layer loads the row's rotations alongside the row.
    device float2 *row_rotations = rotations + ulong(row) * (D / 2);
    float2 rotation[2];
    if (p.rotations == OMNI_ROTATIONS_LOAD) {
      rotation[0] = row_rotations[lane];
      rotation[1] = row_rotations[lane + 32];
    }
    const float sum = simd_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
    const float inverse = rsqrt(sum / float(D) + p.rms_epsilon);
    float n[4];
#pragma unroll
    for (uint j = 0; j < 4; ++j) n[j] = float(bfloat(x[j] * inverse * float(weight[lane + 32 * j])));
    if (p.rotations != OMNI_ROTATIONS_LOAD) {
      const float a0 = float(position) * inverse_frequencies[lane];
      rotation[0] = float2(precise::cos(a0), precise::sin(a0));
      const float a1 = float(position) * inverse_frequencies[lane + 32];
      rotation[1] = float2(precise::cos(a1), precise::sin(a1));
      if (p.rotations == OMNI_ROTATIONS_STORE && kv_head == 0 && part == 0) {
        row_rotations[lane] = rotation[0];
        row_rotations[lane + 32] = rotation[1];
      }
    }
    // Pairs (lane, lane + 64) and (lane + 32, lane + 96): x[0],x[2] and
    // x[1],x[3], rotated as first * cos - second * sin and
    // second * cos + first * sin. The fused multiply-adds are the
    // contraction the compiler chose for that expression with the cosines
    // and sines computed in place; spelled out, every mode gives those bits.
    float rotated[4];
#pragma unroll
    for (uint j = 0; j < 2; ++j) {
      const float c = rotation[j].x, s = rotation[j].y;
      rotated[j] = fma(n[j], c, -(n[j + 2] * s));
      rotated[j + 2] = fma(n[j], s, n[j + 2] * c);
    }
    if (part < 2) {
#pragma unroll
      for (uint j = 0; j < 4; ++j) query[r * kGroup + part][lane + 32 * j] = float(bfloat(rotated[j]));
    } else {
#pragma unroll
      for (uint j = 0; j < 4; ++j) stored_keys[lane + 32 * j] = bfloat(rotated[j]);
    }
  } else {
#pragma unroll
    for (uint j = 0; j < 4; ++j) stored_values[lane + 32 * j] = source[(kQueryHeads + kKvHeads + kv_head) * D + lane + 32 * j];
  }
}

// A decode sequence's whole attention in one dispatch, R (1 or 2) rows per
// sequence (a lone row past OMNI_FUSED_MAXIMUM_PAGES pages takes
// omni_attention_decode_long instead):
// group.x the KV head, group.y the sequence. The threadgroup normalizes and
// rotates its sequence's R rows' two query heads and key head, stores the
// keys and values into the rows' page slots, then runs the page loop of
// omni_attention_decode_split over every page for the R * 2 query vectors
// (row r sees the tokens through its own) and writes the normalized output
// rows.
template <uint R>
inline void omni_attention_decode_fused_body(device const bfloat *qkv, device const bfloat *q_norm,
                                             device const bfloat *k_norm, device const float *inverse_frequencies,
                                             constant OmniSequenceParams *sequences, device const uint *page_table,
                                             device uchar *pages, device bfloat *output, device float2 *rotations,
                                             constant OmniQkvParams &p,
                                             threadgroup float (*query)[OMNI_DECODER_HEAD_DIM],
                                             threadgroup float (*group_output)[R * OMNI_DECODER_QUERY_HEADS / OMNI_DECODER_KV_HEADS][OMNI_DECODER_HEAD_DIM],
                                             threadgroup float (*group_max)[R * OMNI_DECODER_QUERY_HEADS / OMNI_DECODER_KV_HEADS],
                                             threadgroup float (*group_sum)[R * OMNI_DECODER_QUERY_HEADS / OMNI_DECODER_KV_HEADS],
                                             uint2 group, uint lane, uint sg) {
  constexpr uint D = kHeadDim, N = kPageTokens, M = R * kGroup;
  const uint kv_head = group.x, sequence = group.y;
  if (kv_head >= kKvHeads || sequence >= p.sequences) return;
  constant OmniSequenceParams &s = sequences[sequence];
  // Simdgroups 4r..4r+3 prepare row r.
  if (sg / 4 < R)
    omni_fused_prepare(qkv, q_norm, k_norm, inverse_frequencies, s, page_table, pages, rotations, p, query, kv_head, sg / 4,
                       sg % 4, lane);
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

  // Row r sees tokens 0 .. committed + r.
  const uint visible_last = s.committed_tokens + R;
  const uint page_count = (visible_last + N - 1) / N;
  OmniPageState<M> state;
  state.clear();
  for (uint page = sg; page < page_count; page += 8) {
    const uint index = omni_page_of(s, page_table, p.layer, page * N);
    omni_attention_page<M>(state, omni_page_keys(pages, index, kv_head), omni_page_values(pages, index, kv_head), query,
                           page * N, lane, [&](uint m) { return s.committed_tokens + m / kGroup + 1; });
  }
#pragma unroll
  for (uint m = 0; m < M; ++m) {
#pragma unroll
    for (uint j = 0; j < 8; ++j) {
      const float both = state.acc[m][j] + simd_shuffle_xor(state.acc[m][j], 16);
      if (lane < 16) group_output[sg][m][lane * 8 + j] = both;
    }
    if (lane == 0) {
      group_max[sg][m] = state.running_max[m];
      group_sum[sg][m] = state.running_sum[m];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
#pragma unroll
  for (uint m = 0; m < M; ++m) {
    float maximum = -INFINITY;
    for (uint g = 0; g < 8; ++g) maximum = max(maximum, group_max[g][m]);
    float total = 0.0f, merged[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint g = 0; g < 8; ++g) {
      const float weight = group_max[g][m] == -INFINITY ? 0.0f : fast::exp(group_max[g][m] - maximum);
      total += weight * group_sum[g][m];
#pragma unroll
      for (uint j = 0; j < 4; ++j) merged[j] += weight * group_output[g][m][lane + 32 * j];
    }
    const uint row = s.row_offset + m / kGroup;
    device bfloat *out = output + ulong(row) * kAttentionWidth + (kv_head * kGroup + m % kGroup) * D;
#pragma unroll
    for (uint j = 0; j < 4; ++j) out[lane + 32 * j] = bfloat(total > 0.0f ? merged[j] / total : 0.0f);
  }
}

#define OMNI_FUSED_DECODE_KERNEL(Name, R)                                                                         \
  kernel void Name(device const bfloat *qkv [[buffer(0)]], device const bfloat *q_norm [[buffer(1)]],             \
                   device const bfloat *k_norm [[buffer(2)]], device const float *inverse_frequencies [[buffer(3)]], \
                   constant OmniSequenceParams *sequences [[buffer(4)]], device const uint *page_table [[buffer(5)]], \
                   device uchar *pages [[buffer(6)]], device bfloat *output [[buffer(7)]],                         \
                   device float2 *rotations [[buffer(8)]], constant OmniQkvParams &p [[buffer(9)]],                \
                   uint2 group [[threadgroup_position_in_grid]],                                                   \
                   uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {          \
    threadgroup float query[R * kGroup][kHeadDim];                                                                \
    threadgroup float group_output[8][R * kGroup][kHeadDim];                                                      \
    threadgroup float group_max[8][R * kGroup], group_sum[8][R * kGroup];                                         \
    omni_attention_decode_fused_body<R>(qkv, q_norm, k_norm, inverse_frequencies, sequences, page_table, pages,   \
                                        output, rotations, p, query, group_output, group_max, group_sum, group,   \
                                        lane, sg);                                                                \
  }
OMNI_FUSED_DECODE_KERNEL(omni_attention_decode_fused, 1)
OMNI_FUSED_DECODE_KERNEL(omni_attention_decode_fused_r2, 2)
#undef OMNI_FUSED_DECODE_KERNEL

// The fused decode attention of one row per sequence by query head: group.x
// the query head, group.y the sequence. omni_attention_decode_fused (R = 1)
// with each of a KV head's two query vectors in a threadgroup of its own,
// which doubles a step's threadgroups and halves each simdgroup's work per
// page where few lanes leave the GPU idle. Simdgroup 0 prepares the query
// head, 2 the key head and 3 the values; both threadgroups of a KV head
// store its key and values (the same bits into the same slots). A vector's
// page loop and merge are the fused kernel's for that vector, so the
// outputs are its bits.
kernel void omni_attention_decode_fused_head(device const bfloat *qkv [[buffer(0)]],
                                             device const bfloat *q_norm [[buffer(1)]],
                                             device const bfloat *k_norm [[buffer(2)]],
                                             device const float *inverse_frequencies [[buffer(3)]],
                                             constant OmniSequenceParams *sequences [[buffer(4)]],
                                             device const uint *page_table [[buffer(5)]], device uchar *pages [[buffer(6)]],
                                             device bfloat *output [[buffer(7)]], device float2 *rotations [[buffer(8)]],
                                             constant OmniQkvParams &p [[buffer(9)]],
                                             uint2 group [[threadgroup_position_in_grid]],
                                             uint lane [[thread_index_in_simdgroup]],
                                             uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint D = kHeadDim, N = kPageTokens;
  threadgroup float query[kGroup][D];
  threadgroup float group_output[8][D];
  threadgroup float group_max[8], group_sum[8];
  const uint kv_head = group.x / kGroup, part = group.x % kGroup, sequence = group.y;
  if (kv_head >= kKvHeads || sequence >= p.sequences) return;
  constant OmniSequenceParams &s = sequences[sequence];
  if (sg == 0 || sg == 2 || sg == 3)
    omni_fused_prepare(qkv, q_norm, k_norm, inverse_frequencies, s, page_table, pages, rotations, p, query, kv_head, 0,
                       sg == 0 ? part : sg, lane);
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

  const uint visible = s.committed_tokens + 1;
  const uint page_count = (visible + N - 1) / N;
  OmniPageState<1> state;
  state.clear();
  for (uint page = sg; page < page_count; page += 8) {
    const uint index = omni_page_of(s, page_table, p.layer, page * N);
    omni_attention_page<1>(state, omni_page_keys(pages, index, kv_head), omni_page_values(pages, index, kv_head),
                           query + part, page * N, lane, [&](uint) { return visible; });
  }
#pragma unroll
  for (uint j = 0; j < 8; ++j) {
    const float both = state.acc[0][j] + simd_shuffle_xor(state.acc[0][j], 16);
    if (lane < 16) group_output[sg][lane * 8 + j] = both;
  }
  if (lane == 0) {
    group_max[sg] = state.running_max[0];
    group_sum[sg] = state.running_sum[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  float maximum = -INFINITY;
  for (uint g = 0; g < 8; ++g) maximum = max(maximum, group_max[g]);
  float total = 0.0f, merged[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint g = 0; g < 8; ++g) {
    const float weight = group_max[g] == -INFINITY ? 0.0f : fast::exp(group_max[g] - maximum);
    total += weight * group_sum[g];
#pragma unroll
    for (uint j = 0; j < 4; ++j) merged[j] += weight * group_output[g][lane + 32 * j];
  }
  device bfloat *out = output + ulong(s.row_offset) * kAttentionWidth + group.x * D;
#pragma unroll
  for (uint j = 0; j < 4; ++j) out[lane + 32 * j] = bfloat(total > 0.0f ? merged[j] / total : 0.0f);
}

// A lone decode row's attention over a long history in one dispatch, by
// query head: group.x the query head, group.z the split (s.splits of them,
// the pages partitioned as omni_attention_decode_split partitions them).
// Each threadgroup prepares its query vector as the fused kernel does (the
// split holding the row's own token also stores its key and values), runs
// the split kernel's page loop and merge for that vector and stores the
// split's partial; the last split to finish reduces the partials in split
// order as omni_attention_decode_reduce does (kernels/common/split_reduce.h)
// and writes the output. The outputs are the bits of omni_qkv_prepare, the
// split kernel and the reduce, in one dispatch where those take three, with
// one query vector per threadgroup where one row leaves the GPU idle.
kernel void omni_attention_decode_long(device const bfloat *qkv [[buffer(0)]], device const bfloat *q_norm [[buffer(1)]],
                                       device const bfloat *k_norm [[buffer(2)]],
                                       device const float *inverse_frequencies [[buffer(3)]],
                                       constant OmniSequenceParams *sequences [[buffer(4)]],
                                       device const uint *page_table [[buffer(5)]], device uchar *pages [[buffer(6)]],
                                       device bfloat *output [[buffer(7)]], device float2 *rotations [[buffer(8)]],
                                       device coherent(device) float *partials [[buffer(9)]],
                                       device coherent(device) float *statistics [[buffer(10)]],
                                       device atomic_uint *counters [[buffer(11)]],
                                       constant OmniQkvParams &p [[buffer(12)]],
                                       uint3 group [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]],
                                       uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint D = kHeadDim, N = kPageTokens, M = kGroup;
  threadgroup float query[kGroup][D];
  threadgroup float group_output[8][D];
  threadgroup float group_max[8], group_sum[8];
  threadgroup uint arrival;
  const uint kv_head = group.x / kGroup, part = group.x % kGroup, sequence = group.y, split = group.z;
  if (kv_head >= kKvHeads || sequence >= p.sequences) return;
  constant OmniSequenceParams &s = sequences[sequence];
  const uint visible = s.committed_tokens + 1;
  const uint page_count = (visible + N - 1) / N;
  const uint per_split = (page_count + s.splits - 1) / s.splits;
  const uint written = (page_count + per_split - 1) / per_split;
  if (split >= written) return;
  const uint page_begin = split * per_split, page_end = min(page_count, page_begin + per_split);
  // Simdgroup 0 prepares the query head; 2 and 3 the key head and values
  // in the split that holds the row's token.
  if (sg == 0 || (page_end == page_count && (sg == 2 || sg == 3)))
    omni_fused_prepare(qkv, q_norm, k_norm, inverse_frequencies, s, page_table, pages, rotations, p, query, kv_head, 0,
                       sg == 0 ? part : sg, lane);
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

  OmniPageState<1> state;
  state.clear();
  for (uint page = page_begin + sg; page < page_end; page += 8) {
    const uint index = omni_page_of(s, page_table, p.layer, page * N);
    omni_attention_page<1>(state, omni_page_keys(pages, index, kv_head), omni_page_values(pages, index, kv_head),
                           query + part, page * N, lane, [&](uint) { return visible; });
  }
#pragma unroll
  for (uint j = 0; j < 8; ++j) {
    const float both = state.acc[0][j] + simd_shuffle_xor(state.acc[0][j], 16);
    if (lane < 16) group_output[sg][lane * 8 + j] = both;
  }
  if (lane == 0) {
    group_max[sg] = state.running_max[0];
    group_sum[sg] = state.running_sum[0];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // The split's partial in the split kernel's slot, [lane][kv head][split][2][128].
  const ulong slot = (ulong(sequence) * kKvHeads + kv_head) * OMNI_DECODE_MAXIMUM_SPLITS;
  if (sg == 0) {
    float maximum = -INFINITY;
    for (uint g = 0; g < 8; ++g) maximum = max(maximum, group_max[g]);
    float total = 0.0f, merged[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint g = 0; g < 8; ++g) {
      const float weight = group_max[g] == -INFINITY ? 0.0f : fast::exp(group_max[g] - maximum);
      total += weight * group_sum[g];
#pragma unroll
      for (uint j = 0; j < 4; ++j) merged[j] += weight * group_output[g][lane + 32 * j];
    }
#pragma unroll
    for (uint j = 0; j < 4; ++j) partials[((slot + split) * M + part) * D + lane + 32 * j] = merged[j];
    if (lane == 0) {
      statistics[((slot + split) * M + part) * 2] = maximum;
      statistics[((slot + split) * M + part) * 2 + 1] = total;
    }
  }
  device atomic_uint *counter = counters + sequence * kQueryHeads + group.x;
  if (!split_arrive_last(counter, written, tid, &arrival)) return;
  if (sg == 0) {
    float maximum = -INFINITY;
    for (uint other = 0; other < written; ++other) maximum = max(maximum, statistics[((slot + other) * M + part) * 2]);
    float numerator[4] = {0.0f, 0.0f, 0.0f, 0.0f}, denominator = 0.0f;
    for (uint other = 0; other < written; ++other) {
      const float weight = fast::exp(statistics[((slot + other) * M + part) * 2] - maximum);
#pragma unroll
      for (uint j = 0; j < 4; ++j) numerator[j] += weight * partials[((slot + other) * M + part) * D + lane + 32 * j];
      denominator += weight * statistics[((slot + other) * M + part) * 2 + 1];
    }
    device bfloat *out = output + ulong(s.row_offset) * kAttentionWidth + group.x * D;
#pragma unroll
    for (uint j = 0; j < 4; ++j) out[lane + 32 * j] = bfloat(denominator > 0.0f ? numerator[j] / denominator : 0.0f);
  }
  split_release(counter, tid);
}

// The fused decode attention of sequences whose history and R rows fit one
// page (at most 32 tokens; the code predictor's): the fused kernel without
// its page loop and merge, in 4 R simdgroups. Simdgroup 4 r + part prepares
// part `part` of row r as omni_fused_prepare does; the query simdgroups
// (parts 0 and 1) then run their own query vector m = 2 r + part, over only
// the loads that hold a token it sees, with omni_attention_page's
// arithmetic for that vector (the same scores, softmax and value sums in the
// same order; the loads it skips hold masked tokens, whose probabilities are
// zeros), and write its output: one page's state is the whole result. The
// scores of loads older than the step's rows are taken before the
// threadgroup waits for the rows' keys and values.
template <uint R>
inline void omni_attention_decode_page_body(device const bfloat *qkv, device const bfloat *q_norm,
                                            device const bfloat *k_norm, device const float *inverse_frequencies,
                                            constant OmniSequenceParams *sequences, device const uint *page_table,
                                            device uchar *pages, device bfloat *output, device float2 *rotations,
                                            constant OmniQkvParams &p,
                                            threadgroup float (*query)[OMNI_DECODER_HEAD_DIM], uint2 group, uint lane,
                                            uint sg) {
  constexpr uint D = kHeadDim;
  const uint kv_head = group.x, sequence = group.y;
  if (kv_head >= kKvHeads || sequence >= p.sequences) return;
  constant OmniSequenceParams &s = sequences[sequence];
  const uint r = sg / 4, part = sg % 4, m = r * kGroup + part;
  omni_fused_prepare(qkv, q_norm, k_norm, inverse_frequencies, s, page_table, pages, rotations, p, query, kv_head, r,
                     part, lane);
  // Vector m (row r) sees tokens 0 .. committed + r: loads
  // 0 .. ceil(visible / 2) - 1, load i holding tokens 2 i and 2 i + 1;
  // the loads before `settled` hold only tokens stored by earlier steps.
  const uint visible = s.committed_tokens + r + 1;
  const uint loads = (visible + 1) / 2, settled = s.committed_tokens / 2;
  const uint index = omni_page_of(s, page_table, p.layer, 0);
  device const bfloat *keys = omni_page_keys(pages, index, kv_head);
  device const bfloat *values = omni_page_values(pages, index, kv_head);
  const uint parity = lane / 16, dim0 = (lane % 16) * 8;
  float score[16];
  const auto score_load = [&](uint i) {
    const uint4 packed = *reinterpret_cast<device const uint4 *>(keys + (2 * i + parity) * D + dim0);
    float k[8];
    omni_unpack_bf16x8(packed, k);
    float dot = 0.0f;
#pragma unroll
    for (uint j = 0; j < 8; ++j) dot += query[m][dim0 + j] * k[j];
    dot += simd_shuffle_xor(dot, 8);
    dot += simd_shuffle_xor(dot, 4);
    dot += simd_shuffle_xor(dot, 2);
    dot += simd_shuffle_xor(dot, 1);
    score[i] = dot;
  };
  if (part < 2) {
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint i = 0; i < 16; ++i) {
      if (i >= settled) break;
      score_load(i);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  if (part >= 2) return;
#pragma unroll
  for (uint i = 0; i < 16; ++i) {
    if (i >= loads) break;
    if (i >= settled) score_load(i);
  }
  float lane_max = -INFINITY;
#pragma unroll
  for (uint i = 0; i < 16; ++i) {
    if (i >= loads) break;
    score[i] = 2 * i + parity < visible ? score[i] * kScale : -INFINITY;
    lane_max = max(lane_max, score[i]);
  }
  const float page_max = simd_max(lane_max);
  float lane_sum = 0.0f;
#pragma unroll
  for (uint i = 0; i < 16; ++i) {
    if (i >= loads) break;
    const float probability = score[i] == -INFINITY ? 0.0f : fast::exp(score[i] - page_max);
    score[i] = probability;
    lane_sum += probability;
  }
  // Every token's probability sits in its 16 lanes: count one of them.
  const float sum = simd_sum(lane % 16 == 0 ? lane_sum : 0.0f);
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (uint i = 0; i < 16; ++i) {
    if (i >= loads) break;
    const uint4 packed = *reinterpret_cast<device const uint4 *>(values + (2 * i + parity) * D + dim0);
    float v[8];
    omni_unpack_bf16x8(packed, v);
    const float weight = score[i];
#pragma unroll
    for (uint j = 0; j < 8; ++j) acc[j] += weight * v[j];
  }
  // The two token parities of each dim; lane l < 16 holds dims 8 l .. 8 l + 7.
  device bfloat *out = output + ulong(s.row_offset + r) * kAttentionWidth + (kv_head * kGroup + part) * D;
#pragma unroll
  for (uint j = 0; j < 8; ++j) {
    const float both = acc[j] + simd_shuffle_xor(acc[j], 16);
    if (lane < 16) out[dim0 + j] = bfloat(sum > 0.0f ? both / sum : 0.0f);
  }
}

#define OMNI_PAGE_DECODE_KERNEL(Name, R)                                                                          \
  kernel void Name(device const bfloat *qkv [[buffer(0)]], device const bfloat *q_norm [[buffer(1)]],             \
                   device const bfloat *k_norm [[buffer(2)]], device const float *inverse_frequencies [[buffer(3)]], \
                   constant OmniSequenceParams *sequences [[buffer(4)]], device const uint *page_table [[buffer(5)]], \
                   device uchar *pages [[buffer(6)]], device bfloat *output [[buffer(7)]],                         \
                   device float2 *rotations [[buffer(8)]], constant OmniQkvParams &p [[buffer(9)]],                \
                   uint2 group [[threadgroup_position_in_grid]],                                                   \
                   uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {          \
    threadgroup float query[R * kGroup][kHeadDim];                                                                \
    omni_attention_decode_page_body<R>(qkv, q_norm, k_norm, inverse_frequencies, sequences, page_table, pages,    \
                                       output, rotations, p, query, group, lane, sg);                             \
  }
OMNI_PAGE_DECODE_KERNEL(omni_attention_decode_page, 1)
OMNI_PAGE_DECODE_KERNEL(omni_attention_decode_page_r2, 2)
#undef OMNI_PAGE_DECODE_KERNEL

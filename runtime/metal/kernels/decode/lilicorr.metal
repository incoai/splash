#include "metal/abi/KernelABI.h"
#include "metal/abi/LiLiCorr.h"
#include "metal/abi/VerifyTree.h"

// The LiLiCorr selector of Bolt's draft (metal/abi/LiLiCorr.h,
// ops/LiLiCorr.cpp): everything around its projections, which ops::Linear
// runs. A lane's nodes are rows lane * 56 + position * 8 + rank of every
// [rows][width] buffer; its residual rows are fp32, every projection's input
// and output bf16, and each projection's bias is added where its output is
// read.

constant constexpr uint kWidth = SPLASH_LILICORR_HIDDEN;
constant constexpr uint kNodes = SPLASH_LILICORR_NODES;
constant constexpr uint kRanks = SPLASH_LILICORR_CANDIDATES;
constant constexpr uint kHeadDimension = SPLASH_LILICORR_HEAD_DIMENSION;
constant constexpr float kRmsEpsilon = 1e-6f;

// The sum of a value of each thread of a group of `threads` threads.
inline float group_sum(float value, threadgroup float *scratch, uint threads, uint lane, uint simd_group) {
  value = simd_sum(value);
  if (lane == 0) scratch[simd_group] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint i = 0; i < threads / 32; ++i) total += scratch[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}

// One group per proposal row: the logsumexp of the lane's draft logits row
// that drafts it, row position + 1 of its query rows.
kernel void lilicorr_lse(device const float *logits [[buffer(0)]], device float *lse [[buffer(1)]],
                         constant uint &vocabulary [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                         uint thread_index [[thread_index_in_threadgroup]],
                         uint threads [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                         uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Positions = SPLASH_DRAFT_PROPOSAL_TOKENS;
  device const float *values =
      logits + (ulong(row / Positions) * SPLASH_DRAFT_QUERY_ROWS + row % Positions + 1) * vocabulary;
  threadgroup float scratch[32];
  float maximum = -INFINITY;
  for (uint i = thread_index; i < vocabulary; i += threads) maximum = max(maximum, values[i]);
  maximum = simd_max(maximum);
  if (lane == 0) scratch[simd_group] = maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maximum = scratch[0];
  for (uint i = 1; i < threads / 32; ++i) maximum = max(maximum, scratch[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum = 0.0f;
  for (uint i = thread_index; i < vocabulary; i += threads) sum += exp(values[i] - maximum);
  sum = group_sum(sum, scratch, threads, lane, simd_group);
  if (thread_index == 0) lse[row] = maximum + log(sum);
}

// x = token + pass + features (each with its bias) + slot and rank
// embeddings; a lane's pass row is its draft's final hidden row of the
// node's position, row position + 1 of its query rows.
kernel void lilicorr_embed_sum(device const bfloat *token [[buffer(0)]], device const bfloat *token_bias [[buffer(1)]],
                               device const bfloat *pass [[buffer(2)]], device const bfloat *pass_bias [[buffer(3)]],
                               device const bfloat *feature [[buffer(4)]],
                               device const bfloat *feature_bias [[buffer(5)]],
                               device const bfloat *slot [[buffer(6)]], device const bfloat *rank [[buffer(7)]],
                               device float *x [[buffer(8)]], uint2 at [[thread_position_in_grid]]) {
  const uint d = at.x, node = at.y;
  const uint lane = node / kNodes, position = node % kNodes / kRanks, candidate = node % kRanks;
  const ulong i = ulong(node) * kWidth + d;
  x[i] = float(token[i]) + float(token_bias[d]) +
         float(pass[(ulong(lane) * SPLASH_DRAFT_QUERY_ROWS + position + 1) * kWidth + d]) + float(pass_bias[d]) +
         float(feature[i]) + float(feature_bias[d]) + float(slot[position * kWidth + d]) +
         float(rank[candidate * kWidth + d]);
}

// One group of 256 threads per row: the bf16 RMS norm of an fp32 row.
kernel void lilicorr_rms(device const float *x [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
                         device bfloat *output [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                         uint thread_index [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                         uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Each = kWidth / 256;
  threadgroup float scratch[8];
  float values[Each];
  float squares = 0.0f;
  for (uint i = 0; i < Each; ++i) {
    values[i] = x[ulong(row) * kWidth + thread_index + i * 256];
    squares += values[i] * values[i];
  }
  const float inverse = rsqrt(group_sum(squares, scratch, 256, lane, simd_group) / kWidth + kRmsEpsilon);
  for (uint i = 0; i < Each; ++i) {
    const uint d = thread_index + i * 256;
    output[ulong(row) * kWidth + d] = bfloat(values[i] * inverse * float(weight[d]));
  }
}

// One simdgroup per lane, head and query node: softmax(q k / sqrt(128) +
// attention_bias[head][query][key]) v over the lane's 56 nodes, from the
// in-projection's [q | k | v] rows and their bias.
kernel void lilicorr_attention(device const bfloat *qkv [[buffer(0)]], device const bfloat *bias [[buffer(1)]],
                               device const bfloat *attention_bias [[buffer(2)]], device bfloat *output [[buffer(3)]],
                               uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint Each = kHeadDimension / 32;
  const uint query = group.x, head = group.y, base = group.z * kNodes;
  const float scale = rsqrt(float(kHeadDimension));
  float q[Each], accumulated[Each];
  for (uint i = 0; i < Each; ++i) {
    const uint d = head * kHeadDimension + lane + i * 32;
    q[i] = (float(qkv[ulong(base + query) * 3 * kWidth + d]) + float(bias[d])) * scale;
    accumulated[i] = 0.0f;
  }
  device const bfloat *biases = attention_bias + (head * kNodes + query) * kNodes;
  float maximum = -INFINITY, sum = 0.0f;
  for (uint key = 0; key < kNodes; ++key) {
    device const bfloat *row = qkv + ulong(base + key) * 3 * kWidth;
    float dot = 0.0f;
    for (uint i = 0; i < Each; ++i) {
      const uint d = head * kHeadDimension + lane + i * 32;
      dot += q[i] * (float(row[kWidth + d]) + float(bias[kWidth + d]));
    }
    const float score = simd_sum(dot) + float(biases[key]);
    const float next = max(maximum, score);
    const float rescale = exp(maximum - next), weight = exp(score - next);
    sum = sum * rescale + weight;
    for (uint i = 0; i < Each; ++i) {
      const uint d = head * kHeadDimension + lane + i * 32;
      accumulated[i] = accumulated[i] * rescale + weight * (float(row[2 * kWidth + d]) + float(bias[2 * kWidth + d]));
    }
    maximum = next;
  }
  for (uint i = 0; i < Each; ++i)
    output[ulong(base + query) * kWidth + head * kHeadDimension + lane + i * 32] = bfloat(accumulated[i] / sum);
}

// x += y + bias.
kernel void lilicorr_residual(device float *x [[buffer(0)]], device const bfloat *y [[buffer(1)]],
                              device const bfloat *bias [[buffer(2)]], uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * kWidth + at.x;
  x[i] += float(y[i]) + float(bias[at.x]);
}

// y = silu(y + bias) over rows of `width` values.
kernel void lilicorr_bias_silu(device bfloat *y [[buffer(0)]], device const bfloat *bias [[buffer(1)]],
                               constant uint &width [[buffer(2)]], uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * width + at.x;
  const float v = float(y[i]) + float(bias[at.x]);
  y[i] = bfloat(v / (1.0f + exp(-v)));
}

// One group of 256 threads per lane: its anchor state, the RMS norm of its
// context projection plus bias, in fp32 for the factor input and in bf16 for
// the anchor head.
kernel void lilicorr_anchor(device const bfloat *projected [[buffer(0)]], device const bfloat *bias [[buffer(1)]],
                            device const bfloat *weight [[buffer(2)]], device float *state [[buffer(3)]],
                            device bfloat *state_rows [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                            uint thread_index [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                            uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Each = kWidth / 256;
  threadgroup float scratch[8];
  float values[Each];
  float squares = 0.0f;
  for (uint i = 0; i < Each; ++i) {
    const uint d = thread_index + i * 256;
    values[i] = float(projected[ulong(row) * kWidth + d]) + float(bias[d]);
    squares += values[i] * values[i];
  }
  const float inverse = rsqrt(group_sum(squares, scratch, 256, lane, simd_group) / kWidth + kRmsEpsilon);
  for (uint i = 0; i < Each; ++i) {
    const uint d = thread_index + i * 256;
    const float v = values[i] * inverse * float(weight[d]);
    state[ulong(row) * kWidth + d] = v;
    state_rows[ulong(row) * kWidth + d] = bfloat(v);
  }
}

// One group of 256 threads per node: the factor projection's input [x, a,
// x * a] of its normalized row x and its lane's anchor state a.
kernel void lilicorr_factor_input(device const float *x [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
                                  device const float *state [[buffer(2)]], device bfloat *output [[buffer(3)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint thread_index [[thread_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Each = kWidth / 256;
  threadgroup float scratch[8];
  float values[Each];
  float squares = 0.0f;
  for (uint i = 0; i < Each; ++i) {
    values[i] = x[ulong(row) * kWidth + thread_index + i * 256];
    squares += values[i] * values[i];
  }
  const float inverse = rsqrt(group_sum(squares, scratch, 256, lane, simd_group) / kWidth + kRmsEpsilon);
  device bfloat *out = output + ulong(row) * 3 * kWidth;
  for (uint i = 0; i < Each; ++i) {
    const uint d = thread_index + i * 256;
    const float v = values[i] * inverse * float(weight[d]);
    const float a = state[ulong(row / kNodes) * kWidth + d];
    out[d] = bfloat(v);
    out[kWidth + d] = bfloat(a);
    out[2 * kWidth + d] = bfloat(v * a);
  }
}

// The length of a simdgroup's row of 1024 values plus bias, each lane
// holding 32 of them.
inline float simd_norm(thread const float (&v)[kWidth / 32]) {
  float squares = 0.0f;
  for (uint i = 0; i < kWidth / 32; ++i) squares += v[i] * v[i];
  return max(sqrt(simd_sum(squares)), SPLASH_LILICORR_VECTOR_EPSILON);
}

inline void load_row(thread float (&v)[kWidth / 32], device const bfloat *row, device const bfloat *bias, uint lane) {
  for (uint i = 0; i < kWidth / 32; ++i) v[i] = float(row[lane + i * 32]) + float(bias[lane + i * 32]);
}

// One group of eight simdgroups per lane and proposal position, simdgroup c
// for its candidate c: the position's 16 x 16 edge table that
// draft_select_dflash walks (draft_select_edges' layout, in the partial-values
// scratch after the shard partials), 8 * cos(out[pred], in[c]) from the
// previous position's candidates, or for position 0 from predecessor 0, the
// lane's anchor, 8 * cos(in[c], anchor_out). Other entries are 0; the
// candidates past the eighth have unary -inf (lilicorr_candidates).
kernel void lilicorr_scores(device const bfloat *out_head [[buffer(0)]], device const bfloat *out_bias [[buffer(1)]],
                            device const bfloat *in_head [[buffer(2)]], device const bfloat *in_bias [[buffer(3)]],
                            device const bfloat *anchor_out [[buffer(4)]],
                            device const bfloat *anchor_out_bias [[buffer(5)]], device float *partial_values [[buffer(6)]],
                            constant uint &lanes [[buffer(7)]], uint2 group [[threadgroup_position_in_grid]],
                            uint thread_index [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                            uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint Positions = SPLASH_DRAFT_PROPOSAL_TOKENS;
  constexpr uint Candidates = SPLASH_DRAFT_CANDIDATES;
  const uint position = group.x, batch = group.y;
  device float *table = partial_values + ulong(lanes) * Positions * SPLASH_DRAFT_SAMPLING_SHARDS * Candidates +
                        ulong(batch * Positions + position) * Candidates * Candidates;
  for (uint i = thread_index; i < Candidates * Candidates; i += 256) {
    const uint predecessor = i / Candidates, candidate = i % Candidates;
    if (predecessor >= kRanks || candidate >= kRanks || (position == 0 && predecessor > 0)) table[i] = 0.0f;
  }
  const uint candidate = simd_group;
  float in[kWidth / 32], other[kWidth / 32];
  load_row(in, in_head + ulong(batch * kNodes + position * kRanks + candidate) * kWidth, in_bias, lane);
  const float in_norm = simd_norm(in);
  const auto cosine = [&]() {
    float dot = 0.0f;
    for (uint i = 0; i < kWidth / 32; ++i) dot += in[i] * other[i];
    return SPLASH_LILICORR_LOGIT_SCALE * simd_sum(dot) / (in_norm * simd_norm(other));
  };
  if (position == 0) {
    load_row(other, anchor_out + ulong(batch) * kWidth, anchor_out_bias, lane);
    const float score = cosine();
    if (lane == 0) table[candidate] = score;
    return;
  }
  for (uint predecessor = 0; predecessor < kRanks; ++predecessor) {
    load_row(other, out_head + ulong(batch * kNodes + (position - 1) * kRanks + predecessor) * kWidth, out_bias, lane);
    const float score = cosine();
    if (lane == 0) table[predecessor * Candidates + candidate] = score;
  }
}

// Copies params.x bf16 values of row params.y, or with counts row
// counts[0] - 1, of `source` to row params.z of `destination`: a lane's
// anchor context row.
// params: hidden width, the source row (0xffffffff: lane params.w's deepest
// accepted row, the highest bit of its path mask, from its first row of the
// step), the destination row, the lane.
kernel void lilicorr_store_anchor(device const bfloat *source [[buffer(0)]], device const uint *paths [[buffer(1)]],
                                  device bfloat *destination [[buffer(2)]],
                                  device const SplashLaneRows *lanes [[buffer(3)]],
                                  constant uint4 &params [[buffer(4)]],
                                  uint d [[thread_position_in_grid]]) {
  const uint row =
      params.y == 0xffffffffu ? lanes[params.w].first + (31u - clz(paths[params.w])) : params.y;
  destination[ulong(params.z) * params.x + d] = source[ulong(row) * params.x + d];
}

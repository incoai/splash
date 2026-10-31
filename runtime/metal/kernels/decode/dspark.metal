#include "metal/abi/DSpark.h"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/ManifoldHyperConnection.h"

#include <metal_stdlib>

using namespace metal;

// DSpark's proposals (metal/abi/DSpark.h), as DeepSeek's inference/model.py
// draws them greedily: its Markov head's chain over each lane's block.

constant constexpr uint kThreads = DSPARK_STEP_THREADS;
constant constexpr uint kSimdgroups = kThreads / 32;
constant constexpr uint kProposals = SPLASH_DRAFT_PROPOSAL_TOKENS;
constant constexpr uint kCandidates = SPLASH_DRAFT_CANDIDATES;

// Step p.position of each lane's chain (DSparkStepParams), a threadgroup per
// lane: its token, the argmax of the row's logits plus the lane's bias, the
// lowest id among ties; its confidence, the sigmoid of the confidence head's
// dot with the row's pre-sum of the streams (by the pre weights of the
// row's last mix, rounded to bf16 as the sublayers read it) and the lane's
// Markov embedding. The draft distribution acceptance reads of the token
// (decode_accept_dflash) is the token alone: its candidates all name it, the
// first with probability one. The token is the next step's previous token.
kernel void dspark_markov_step(device const float *logits [[buffer(0)]],
                               device const float *bias [[buffer(1)]],
                               device const float *streams [[buffer(2)]],
                               device const float *mixes [[buffer(3)]],
                               device const bfloat *markov [[buffer(4)]],
                               device const float *weights [[buffer(5)]],
                               device uint *tokens [[buffer(6)]],
                               device float *confidence [[buffer(7)]],
                               device uint *candidates [[buffer(8)]],
                               device float *probabilities [[buffer(9)]],
                               device uint *previous [[buffer(10)]],
                               constant DSparkStepParams &p [[buffer(11)]],
                               uint batch [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float best_values[kSimdgroups];
  threadgroup uint best_ids[kSimdgroups];
  threadgroup float sums[kSimdgroups];
  if (batch >= p.lanes) return;
  const ulong row = ulong(batch) * p.lane_rows + p.position;
  device const float *row_logits = logits + row * p.vocabulary;
  device const float *lane_bias = bias + ulong(batch) * p.vocabulary;
  // A thread's tokens ascend, so its first largest is its lowest.
  float best = -INFINITY;
  uint best_id = 0xffffffffu;
  for (uint token = tid; token < p.vocabulary; token += kThreads) {
    const float value = row_logits[token] + lane_bias[token];
    if (value > best) {
      best = value;
      best_id = token;
    }
  }
  const float simd_best = simd_max(best);
  const uint simd_id = simd_min(best == simd_best ? best_id : 0xffffffffu);
  device const float *x = streams + row * MHC_STREAMS * p.hidden;
  device const float *pre = mixes + row * MHC_MIXES;
  float sum = 0.0f;
  for (uint c = tid; c < p.hidden; c += kThreads) {
    float value = 0.0f;
    for (uint s = 0; s < MHC_STREAMS; ++s) value = fma(pre[s], x[s * p.hidden + c], value);
    sum = fma(weights[c], float(bfloat(value)), sum);
  }
  device const bfloat *embedding = markov + ulong(batch) * p.rank;
  for (uint c = tid; c < p.rank; c += kThreads) sum = fma(weights[p.hidden + c], float(embedding[c]), sum);
  sum = simd_sum(sum);
  if (lane == 0) {
    best_values[sg] = simd_best;
    best_ids[sg] = simd_id;
    sums[sg] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid != 0) return;
  float top = best_values[0], score = 0.0f;
  uint token = best_ids[0];
  for (uint g = 0; g < kSimdgroups; ++g) {
    score += sums[g];
    if (best_values[g] > top || (best_values[g] == top && best_ids[g] < token)) {
      top = best_values[g];
      token = best_ids[g];
    }
  }
  // Rows of no finite value draft token 0.
  if (token >= p.vocabulary) token = 0;
  previous[batch] = token;
  const uint end = p.position + 1 == p.block ? kProposals : p.position + 1;
  for (uint position = p.position; position < end; ++position) {
    const ulong at = ulong(batch) * kProposals + position;
    tokens[at] = token;
    confidence[at] = position == p.position ? 1.0f / (1.0f + exp(-score)) : 0.0f;
    for (uint c = 0; c < kCandidates; ++c) {
      candidates[at * kCandidates + c] = token;
      probabilities[at * kCandidates + c] = c ? 0.0f : 1.0f;
    }
  }
}

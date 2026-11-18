#include "metal/abi/DSpark.h"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/ManifoldHyperConnection.h"
#include "metal/abi/Sampling.h"
#include "metal/abi/VerifyTree.h"

#include <metal_stdlib>

using namespace metal;

// DSpark's proposals (metal/abi/DSpark.h), drawn greedily: its Markov head's
// chain over each lane's block, each lane's tree.

constant constexpr uint kThreads = DSPARK_STEP_THREADS;
constant constexpr uint kSimdgroups = kThreads / 32;
constant constexpr uint kProposals = SPLASH_DRAFT_PROPOSAL_TOKENS;

// Step p.position of each lane's chain (DSparkStepParams), a threadgroup per
// lane: its token, the argmax of the row's logits plus the lane's bias, the
// lowest id among ties; its confidence, the sigmoid of the confidence head's
// dot with the row's pre-sum of the streams (by the pre weights of the
// row's last mix, rounded to bf16 as the sublayers read it) and the lane's
// Markov embedding. The token is the next step's previous token.
kernel void dspark_markov_step(device const float *logits [[buffer(0)]],
                               device const float *bias [[buffer(1)]],
                               device const float *streams [[buffer(2)]],
                               device const float *mixes [[buffer(3)]],
                               device const bfloat *markov [[buffer(4)]],
                               device const float *weights [[buffer(5)]],
                               device uint *tokens [[buffer(6)]],
                               device float *confidence [[buffer(7)]],
                               device uint *previous [[buffer(8)]],
                               constant DSparkStepParams &p [[buffer(9)]],
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
  const ulong at = ulong(batch) * kProposals + p.position;
  tokens[at] = token;
  confidence[at] = 1.0f / (1.0f + exp(-score));
}

// Each lane's tree (DSparkTreeParams), a thread per lane: its chain's rows
// in the selector's tree buffers.
kernel void dspark_tree(device const uint *proposals [[buffer(0)]],
                        device const float *confidence [[buffer(1)]],
                        device const uint *chain_tokens [[buffer(2)]],
                        device const float *chain_rates [[buffer(3)]],
                        device uint *tree_tokens [[buffer(4)]],
                        device SplashTreeRow *trees [[buffer(5)]],
                        device float2 *scores [[buffer(6)]],
                        device uint *sizes [[buffer(7)]],
                        device uchar *draws [[buffer(8)]],
                        device float *caps [[buffer(9)]],
                        constant DSparkTreeParams &p [[buffer(10)]],
                        uint lane [[thread_position_in_grid]]) {
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  if (lane >= p.lanes) return;
  const bool chained = p.chain[lane] != 0;
  const uint rows = 1 + min(min(chained ? p.chain[lane] : p.block, p.depth[lane]), Rows - 1);
  float estimate = 1.0f;
  for (uint row = 0; row < rows; ++row) {
    const ulong at = ulong(lane) * Rows + row;
    float q = 1.0f;
    uint token = p.anchor[lane];
    if (row && chained) {
      const ulong at_chain = ulong(lane) * SPLASH_TREE_CHAIN_TOKENS + row - 1;
      token = chain_tokens[at_chain];
      q = -1.0f;
      estimate *= chain_rates[at_chain * SPLASH_TREE_AGREEMENTS + SPLASH_TREE_AGREEMENTS - 1];
    } else if (row) {
      const ulong proposal = ulong(lane) * kProposals + row - 1;
      token = proposals[proposal];
      q = confidence[proposal];
      const float bounded = clamp(q, 1.0e-6f, 1.0f - 1.0e-6f);
      const float logit = log(bounded / (1.0f - bounded));
      estimate *= 1.0f / (1.0f + exp(-(p.slope[lane] * logit + p.intercept[lane][row - 1])));
    }
    tree_tokens[at] = token;
    trees[at] = {(2u << row) - 1u, ushort(row ? row - 1 : SPLASH_TREE_ROOT), ushort(row)};
    scores[at] = float2(q, estimate);
    caps[at] = 1.0f;
    for (uint child = 0; child < SPLASH_TREE_CHILDREN; ++child) draws[at * SPLASH_TREE_CHILDREN + child] = 0xff;
  }
  sizes[lane] = rows;
}

// DSpine's adjacent causal injection and serial final-write refinement.
// Equations follow Tencent/DSpine's Apache-2.0 reference implementation.
#pragma clang fp reassociate(off)
#include "metal/abi/DSpine.h"
#include "metal/abi/KernelABI.h"

inline float dspine_sum(float x, threadgroup float *scratch, uint tid, uint lane, uint sg) {
  x = simd_sum(x);
  if (lane == 0) scratch[sg] = x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float sum = 0;
    for (uint i = 0; i < 8; ++i) sum += scratch[i];
    scratch[0] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return scratch[0];
}

kernel void dspine_residual(device const bfloat *input [[buffer(0)]],
                            device const bfloat *residual [[buffer(1)]],
                            device bfloat *output [[buffer(2)]],
                            constant DSpineParams &p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i < p.rows * p.hidden) output[i] = bfloat(float(input[i]) + float(residual[i]));
}

kernel void dspine_read_input(device const bfloat *input [[buffer(0)]],
                              device bfloat *normalized [[buffer(1)]],
                              device bfloat *raw [[buffer(2)]], device float *rms [[buffer(3)]],
                              constant DSpineParams &p [[buffer(4)]],
                              uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[8];
  float sum = 0;
  for (uint h = tid; h < p.hidden; h += 256) {
    float x = float(input[ulong(row) * p.hidden + h]);
    sum += x * x;
  }
  float root = sqrt(max(dspine_sum(sum, scratch, tid, lane, sg) / p.hidden, 1.1754943508222875e-38f));
  if (tid == 0) rms[row] = root;
  for (uint h = tid; h < p.hidden; h += 256) {
    ulong i = ulong(row) * p.hidden + h;
    raw[i] = input[i];
    normalized[i] = bfloat(float(input[i]) / (root + 1e-6f));
  }
}

kernel void dspine_codes(device bfloat *codes [[buffer(0)]], device const bfloat *book [[buffer(1)]],
                         device const uint *tokens [[buffer(2)]], constant DSpineParams &p [[buffer(3)]],
                         uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[8];
  float sum = 0;
  for (uint c = tid; c < p.code; c += 256) {
    float x = float(codes[ulong(row) * p.code + c]);
    sum += x * x;
  }
  float norm = sqrt(dspine_sum(sum, scratch, tid, lane, sg));
  const bool anchor = row % SPLASH_DRAFT_QUERY_ROWS == 0;
  uint token = tokens[(row / SPLASH_DRAFT_QUERY_ROWS) * SPLASH_DRAFT_QUERY_ROWS];
  for (uint c = tid; c < p.code; c += 256) {
    ulong i = ulong(row) * p.code + c;
    codes[i] = bfloat(sqrt(float(p.code)) * (anchor ? float(book[ulong(token) * p.code + c])
                                                  : float(codes[i]) / (norm + 1e-6f)));
  }
}

kernel void dspine_message(device const bfloat *packed [[buffer(0)]], device const bfloat *bias [[buffer(1)]],
                           device bfloat *message [[buffer(2)]], constant DSpineParams &p [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
  if (i >= p.rows * p.rank) return;
  uint row = i / p.rank, r = i % p.rank;
  float value = 0;
  if (row % SPLASH_DRAFT_QUERY_ROWS != 0) {
    float q = float(packed[ulong(row) * 3 * p.rank + r]);
    float s = float(packed[ulong(row - 1) * 3 * p.rank + p.rank + r]);
    float v = float(packed[ulong(row - 1) * 3 * p.rank + 2 * p.rank + r]);
    value = v / (1.0f + exp(-(q + s + float(bias[r]))));
  }
  message[i] = bfloat(value);
}

kernel void dspine_write(device const bfloat *raw [[buffer(0)]], device const float *rms [[buffer(1)]],
                         device const bfloat *delta [[buffer(2)]], device bfloat *output [[buffer(3)]],
                         constant DSpineParams &p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.rows * p.hidden) return;
  uint row = i / p.hidden;
  output[i] = row % SPLASH_DRAFT_QUERY_ROWS == 0 ? raw[i] : bfloat(float(raw[i]) + rms[row] * float(delta[i]));
}

// One group per request. Only the selected predecessor is refined at each
// position, so there is no full-vocabulary derived table or 16x16xH tensor.
// The conditional q written here is the one consumed by rejection sampling.
kernel void dspine_refine(device const bfloat *raw [[buffer(0)]], device const float *rms [[buffer(1)]],
    device const bfloat *packed [[buffer(2)]], device const bfloat *book [[buffer(3)]],
    device const bfloat *pred_gate [[buffer(4)]], device const bfloat *previous [[buffer(5)]],
    device const bfloat *bias [[buffer(6)]], device const bfloat *write [[buffer(7)]],
    device const bfloat *norm [[buffer(8)]], device const bfloat *head [[buffer(9)]],
    device const uint *candidates [[buffer(10)]], device const float *uniforms [[buffer(11)]],
    device uint *tokens [[buffer(12)]], device float *probabilities [[buffer(13)]],
    device float *confidence [[buffer(14)]], constant DSpineRefineParams &params [[buffer(15)]],
    uint batch [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const DSpineParams p = params.shape;
  constexpr uint K = SPLASH_DRAFT_CANDIDATES, Positions = SPLASH_DRAFT_PROPOSAL_TOKENS;
  threadgroup float code[1024], message[512], state[5120], scores[K], scratch[8];
  threadgroup uint selected_token;
  if (tid == 0) selected_token = params.selection.anchor[batch];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint position = 0; position < Positions; ++position) {
    uint row = batch * SPLASH_DRAFT_QUERY_ROWS + position + 1;
    uint candidate_row = batch * Positions + position;
    for (uint c = tid; c < p.code; c += 256)
      code[c] = sqrt(float(p.code)) * float(book[ulong(selected_token) * p.code + c]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = tid; r < p.rank; r += 256) {
      float gate = 0, value = 0;
      for (uint c = 0; c < p.code; ++c) {
        gate += code[c] * float(pred_gate[ulong(r) * p.code + c]);
        value += code[c] * float(previous[ulong(r) * p.code + c]);
      }
      float q = float(packed[ulong(row) * 3 * p.rank + r]);
      message[r] = value / (1.0f + exp(-(q + gate + float(bias[r]))));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0;
    for (uint h = tid; h < p.hidden; h += 256) {
      float delta = 0;
      for (uint r = 0; r < p.rank; ++r) delta += message[r] * float(write[ulong(h) * p.rank + r]);
      float x = float(raw[ulong(row) * p.hidden + h]) + rms[row] * delta;
      state[h] = x;
      sum += x * x;
    }
    float inverse = rsqrt(dspine_sum(sum, scratch, tid, lane, sg) / p.hidden + 1e-6f);
    for (uint h = tid; h < p.hidden; h += 256) state[h] *= inverse * float(norm[h]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = sg; k < K; k += 8) {
      float score = 0;
      for (uint h = lane; h < p.hidden; h += 32)
        score += state[h] * float(head[(ulong(candidate_row) * K + k) * p.hidden + h]);
      score = simd_sum(score);
      if (lane == 0) scores[k] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float best = scores[0];
      uint choice = 0;
      for (uint k = 1; k < K; ++k)
        if (scores[k] > best) { best = scores[k]; choice = k; }
      float mass = 0;
      for (uint k = 0; k < K; ++k) mass += exp(scores[k] - best);
      confidence[candidate_row] = 1.0f / mass;
      if (params.selection.sampling_mask & (1u << batch)) {
        float total = 0;
        for (uint k = 0; k < K; ++k) {
          scores[k] = exp((scores[k] - best) / params.selection.temperature[batch]);
          total += scores[k];
        }
        float cumulative = 0;
        choice = K - 1;
        bool found = false;
        for (uint k = 0; k < K; ++k) {
          float q = scores[k] / total;
          probabilities[candidate_row * K + k] = q;
          cumulative += q;
          if (!found && cumulative > uniforms[batch * SPLASH_SAMPLING_UNIFORMS + SPLASH_UNIFORM_PROPOSALS + position]) {
            choice = k;
            found = true;
          }
        }
      }
      selected_token = candidates[candidate_row * K + choice];
      tokens[candidate_row] = selected_token;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

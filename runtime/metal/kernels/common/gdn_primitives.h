#pragma once

#include "metal/abi/KernelABI.h"

// Four-tap causal convolution of one channel at one token of the command,
// reading the three preceding tokens from the carried state, rounded to bf16
// and gated by SiLU.
inline bfloat gdn_conv_silu(device const bfloat *packed,
                            device const bfloat *conv_state_in,
                            device const bfloat *conv_weights,
                            uint packed_width, uint conv_dim, uint token,
                            uint channel) {
  float value = 0.0f;
  for (uint tap = 0; tap < 4; ++tap) {
    uint position = token + tap;
    bfloat input = position < 3
                       ? conv_state_in[position * conv_dim + channel]
                       : packed[(position - 3) * packed_width + channel];
    value += float(input) * float(conv_weights[channel * 4 + tap]);
  }
  value = float(bfloat(value));
  return bfloat(value / (1.0f + fast::exp2(-1.44269504089f * value)));
}

// Row `row` of the carried state after consumed_tokens: the last three inputs
// seen, still taken from the incoming state when fewer were consumed.
inline bfloat gdn_conv_carry(device const bfloat *packed,
                             device const bfloat *conv_state_in,
                             uint packed_width, uint conv_dim,
                             uint consumed_tokens, uint row, uint channel) {
  uint source = consumed_tokens + row;
  return source < 3 ? conv_state_in[source * conv_dim + channel]
                    : packed[(source - 3) * packed_width + channel];
}

// The gates of one (token, value head): beta = sigmoid(b) and
// decay = exp(a_scale * softplus(bf16(a + dt_bias))), the softplus rounded to
// bf16 as the reference does.
struct GdnGates {
  bfloat beta;
  float decay;
};

inline GdnGates gdn_gates(device const bfloat *packed_row,
                          device const bfloat *dt_bias,
                          device const float *a_scale, uint b_offset,
                          uint a_offset, uint head) {
  float b = float(packed_row[b_offset + head]);
  GdnGates gates;
  gates.beta = bfloat(1.0f / (1.0f + fast::exp2(-1.44269504089f * b)));
  bfloat x = bfloat(float(packed_row[a_offset + head]) + float(dt_bias[head]));
  float xf = float(x);
  bfloat softplus =
      bfloat(max(xf, 0.0f) +
             fast::log2(1.0f + fast::exp2(-1.44269504089f * abs(xf))) *
                 0.69314718056f);
  gates.decay = fast::exp(a_scale[head] * float(softplus));
  return gates;
}

inline void gdn_write_gates(device const bfloat *packed_row,
                            device const bfloat *dt_bias,
                            device const float *a_scale, uint b_offset,
                            uint a_offset, uint head, device bfloat &beta,
                            device float &decay) {
  const GdnGates gates =
      gdn_gates(packed_row, dt_bias, a_scale, b_offset, a_offset, head);
  beta = gates.beta;
  decay = gates.decay;
}

// The position of value head `head` among the GDN output's head blocks: the
// head itself, or with `tiled` llama.cpp's GGUF order, which puts value head
// j of every key head next to each other (GDNGatePrefillParams).
template <uint KeyHeads, uint ValueHeads>
inline uint gdn_output_head(uint head, bool tiled) {
  constexpr uint HeadsPerKey = ValueHeads / KeyHeads;
  return tiled ? (head % HeadsPerKey) * KeyHeads + head / HeadsPerKey : head;
}

// Gated RMSNorm of the recurrent rows, one task per (token, value head) and
// one lane per dimension, stored at the head's output position. Decode walks
// a lane's rows persistently; prefill dispatches one task per threadgroup.
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint Simdgroups = 8>
inline void
gdn_gate_phase(device const bfloat *recurrent, device const bfloat *packed,
               device const bfloat *norm_weight, device bfloat *hidden,
               uint tasks, uint groups, uint packed_width, bool tiled,
               threadgroup float *scratch, uint group, uint thread_index,
               uint lane, uint simd_group) {
  constexpr uint ZOffset = ConvDim;
  for (uint task = group; task < tasks; task += groups) {
    uint token = task / ValueHeads;
    uint head = task % ValueHeads;
    ulong base = ulong(task) * HeadDim;
    ulong hidden_base =
        (ulong(token) * ValueHeads +
         gdn_output_head<KeyHeads, ValueHeads>(head, tiled)) *
        HeadDim;
    float value =
        thread_index < HeadDim ? float(recurrent[base + thread_index]) : 0.0f;
    float square_sum = simd_sum(value * value);
    if (lane == 0)
      scratch[simd_group] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
      float total = 0.0f;
      for (uint i = 0; i < Simdgroups; ++i)
        total += scratch[i];
      scratch[0] = rsqrt(total / HeadDim + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index < HeadDim) {
      bfloat normalized =
          bfloat(value * scratch[0] * float(norm_weight[thread_index]));
      float gate = float(packed[token * packed_width + ZOffset +
                                head * HeadDim + thread_index]);
      float silu = gate / (1.0f + fast::exp2(-1.44269504089f * gate));
      hidden[hidden_base + thread_index] = bfloat(float(normalized) * silu);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

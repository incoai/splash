#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/gdn_primitives.h"

constant constexpr uint GdnScanBlock = 16;

// One threadgroup carries SPLASH_GDN_SCAN_STATE_ROWS rows of a value head's
// 128 x 128 fp32 recurrent state through the whole chunk, token by token:
// S = d S; m = S k; S += k (v - m) beta; o = S q. A lane owns sixteen key
// columns of one row, so eight lanes share a row and each key-dimension dot
// product costs three shuffle steps; a simdgroup covers four rows. Blocks of
// GdnScanBlock tokens of q and k (widened to fp32), v, decay and beta are
// staged in threadgroup memory while the following block is prefetched into
// registers, so the dependent per-token chain reads only threadgroup memory.
// The math is the decode recurrence; only the summation order of the dot
// products differs from its per-lane form.
template <uint KeyHeads, uint ValueHeads, uint HeadDim>
inline void gdn_scan_prefill_phase(
    device const bfloat *q, device const bfloat *k, device const bfloat *v,
    device const float *decay, device const bfloat *beta,
    device const float *state_in, device float *state_out,
    device bfloat *output, uint tokens, threadgroup float *keys,
    threadgroup float *queries, threadgroup float *values,
    threadgroup float *gates, uint group, uint thread_index, uint lane,
    uint simd_group) {
  constexpr uint Columns = 16;
  constexpr uint LanesPerRow = HeadDim / Columns;
  constexpr uint Rows = SPLASH_GDN_SCAN_STATE_ROWS;
  constexpr uint Threads = SPLASH_GDN_SCAN_THREADS;
  constexpr uint Block = GdnScanBlock;
  constexpr uint HeadsPerKey = ValueHeads / KeyHeads;
  constexpr uint GroupsPerHead = HeadDim / Rows;
  static_assert(Rows == (Threads / 32) * (32 / LanesPerRow),
                "the simdgroups of a threadgroup cover its state rows");
  static_assert(Block * HeadDim == Threads * 16,
                "every thread stages sixteen keys and sixteen queries");
  static_assert(Block * Rows == 32 * 8,
                "the first simdgroup stages the value block");

  const uint value_head = group / GroupsPerHead;
  const uint key_head = value_head / HeadsPerKey;
  const uint row_base = (group % GroupsPerHead) * Rows;
  const uint row = simd_group * (32 / LanesPerRow) + lane / LanesPerRow;
  const uint column = (lane % LanesPerRow) * Columns;
  const ulong state_base =
      (ulong(value_head) * HeadDim + row_base + row) * HeadDim + column;

  float4 state[Columns / 4];
  for (uint c = 0; c < Columns / 4; ++c) {
    state[c] =
        reinterpret_cast<device const float4 *>(state_in + state_base)[c];
  }

  bfloat4 key_fetch[2][2] = {}, query_fetch[2][2] = {}, value_fetch[2] = {};
  float decay_fetch = 0.0f, beta_fetch = 0.0f;
  // Registers for tokens at or past `tokens` keep their previous values and
  // are staged as they are: the compute loop stops at the chunk's token
  // count, so those slots are never read, and zeroing them would only add
  // instructions to the per-token chain.
  auto fetch = [&](uint start) {
    for (uint n = 0; n < 2; ++n) {
      const uint element = (n * Threads + thread_index) * 8;
      const uint token = start + element / HeadDim;
      if (token < tokens) {
        const ulong base =
            (ulong(token) * KeyHeads + key_head) * HeadDim + element % HeadDim;
        for (uint h = 0; h < 2; ++h) {
          key_fetch[n][h] = reinterpret_cast<device const bfloat4 *>(k + base)[h];
          query_fetch[n][h] =
              reinterpret_cast<device const bfloat4 *>(q + base)[h];
        }
      }
    }
    if (thread_index < 32) {
      const uint element = thread_index * 8;
      const uint token = start + element / Rows;
      if (token < tokens) {
        const ulong base = (ulong(token) * ValueHeads + value_head) * HeadDim +
                           row_base + element % Rows;
        for (uint h = 0; h < 2; ++h)
          value_fetch[h] = reinterpret_cast<device const bfloat4 *>(v + base)[h];
      }
    }
    if (thread_index < Block && start + thread_index < tokens) {
      const uint gate = (start + thread_index) * ValueHeads + value_head;
      decay_fetch = decay[gate];
      beta_fetch = float(beta[gate]);
    }
  };
  auto stage = [&]() {
    for (uint n = 0; n < 2; ++n) {
      const uint element = (n * Threads + thread_index) * 8;
      for (uint h = 0; h < 2; ++h) {
        reinterpret_cast<threadgroup float4 *>(keys + element)[h] =
            float4(key_fetch[n][h]);
        reinterpret_cast<threadgroup float4 *>(queries + element)[h] =
            float4(query_fetch[n][h]);
      }
    }
    if (thread_index < 32) {
      for (uint h = 0; h < 2; ++h) {
        reinterpret_cast<threadgroup float4 *>(values + thread_index * 8)[h] =
            float4(value_fetch[h]);
      }
    }
    if (thread_index < Block) {
      gates[thread_index] = decay_fetch;
      gates[Block + thread_index] = beta_fetch;
    }
  };

  fetch(0);
  stage();
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint start = 0; start < tokens; start += Block) {
    const bool more = start + Block < tokens;
    if (more)
      fetch(start + Block);
    const uint count = min(Block, tokens - start);
    for (uint t = 0; t < count; ++t) {
      const float d = gates[t], b = gates[Block + t];
      const threadgroup float4 *key = reinterpret_cast<const threadgroup float4 *>(
          keys + t * HeadDim + column);
      const threadgroup float4 *query =
          reinterpret_cast<const threadgroup float4 *>(queries + t * HeadDim +
                                                       column);
      float4 partial = 0.0f;
      for (uint c = 0; c < Columns / 4; ++c) {
        state[c] *= d;
        partial = fma(state[c], key[c], partial);
      }
      float memory = (partial.x + partial.y) + (partial.z + partial.w);
      for (uint s = 1; s < LanesPerRow; s <<= 1)
        memory += simd_shuffle_xor(memory, s);
      const float delta = (values[t * Rows + row] - memory) * b;
      partial = 0.0f;
      for (uint c = 0; c < Columns / 4; ++c) {
        state[c] = fma(key[c], delta, state[c]);
        partial = fma(state[c], query[c], partial);
      }
      float result = (partial.x + partial.y) + (partial.z + partial.w);
      for (uint s = 1; s < LanesPerRow; s <<= 1)
        result += simd_shuffle_xor(result, s);
      if (lane % LanesPerRow == 0) {
        output[(ulong(start + t) * ValueHeads + value_head) * HeadDim +
               row_base + row] = bfloat(result);
      }
    }
    if (more) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      stage();
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  for (uint c = 0; c < Columns / 4; ++c) {
    reinterpret_cast<device float4 *>(state_out + state_base)[c] = state[c];
  }
}

#define GDN_SCAN_PREFILL_ENTRY(Name, KeyHeads, ValueHeads, HeadDim)            \
  kernel void Name(                                                           \
      device const bfloat *q [[buffer(0)]],                                   \
      device const bfloat *k [[buffer(1)]],                                   \
      device const bfloat *v [[buffer(2)]],                                   \
      device const float *decay [[buffer(3)]],                                \
      device const bfloat *beta [[buffer(4)]],                                \
      device const float *state_in [[buffer(5)]],                             \
      device float *state_out [[buffer(6)]],                                  \
      device bfloat *output [[buffer(7)]],                                    \
      constant GDNPrefillParams &params [[buffer(8)]],                        \
      uint group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]],                      \
      uint lane [[thread_index_in_simdgroup]],                                \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                   \
    threadgroup float keys[GdnScanBlock * HeadDim];                           \
    threadgroup float queries[GdnScanBlock * HeadDim];                        \
    threadgroup float values[GdnScanBlock * SPLASH_GDN_SCAN_STATE_ROWS];    \
    threadgroup float gates[2 * GdnScanBlock];                                \
    gdn_scan_prefill_phase<KeyHeads, ValueHeads, HeadDim>(                    \
        q, k, v, decay, beta, state_in, state_out, output, params.tokens,     \
        keys, queries, values, gates, group, thread_index, lane, simd_group); \
  }

GDN_SCAN_PREFILL_ENTRY(prefill_gdn_scan, 16, 48, 128)
GDN_SCAN_PREFILL_ENTRY(prefill_gdn_scan_vh32, 16, 32, 128)
#undef GDN_SCAN_PREFILL_ENTRY

// Prepares one token of one key head: causal convolution, SiLU, q/k RMS
// normalization, value-head gates and (from the first token) convolution carry.
// Round q/k to bf16 before applying the 1/128 and 1/sqrt(128) scales.
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim>
inline void gdn_prepare_prefill_phase(
    device const bfloat *packed, device const bfloat *conv_weights,
    device const bfloat *conv_state_in, device bfloat *conv_state_out,
    device bfloat *q, device bfloat *k, device bfloat *v,
    device const float *a_scale, device const bfloat *dt_bias,
    device float *decay, device bfloat *beta,
    constant GDNPreparePrefillParams &params, threadgroup float *reductions,
    uint task, uint thread_index, uint lane, uint simd_group) {
  constexpr uint KeyWidth = KeyHeads * HeadDim;
  constexpr uint ValueWidth = ValueHeads * HeadDim;
  constexpr uint HeadsPerKey = ValueHeads / KeyHeads;
  constexpr uint BOffset = ConvDim + ValueWidth;
  constexpr uint AOffset = BOffset + ValueHeads;
  static_assert(HeadDim == 4 * 32, "one simdgroup per 32 head dimensions");
  const uint token = task / KeyHeads;
  const uint key_head = task % KeyHeads;
  const uint query_channel = key_head * HeadDim + thread_index;
  const uint key_channel = KeyWidth + query_channel;
  const uint value_channel =
      2 * KeyWidth + key_head * HeadsPerKey * HeadDim + thread_index;

  const float query =
      float(gdn_conv_silu(packed, conv_state_in, conv_weights,
                          params.packed_width, ConvDim, token, query_channel));
  const float key =
      float(gdn_conv_silu(packed, conv_state_in, conv_weights,
                          params.packed_width, ConvDim, token, key_channel));
  const float query_sum = simd_sum(query * query);
  const float key_sum = simd_sum(key * key);
  if (lane == 0) {
    reductions[simd_group] = query_sum;
    reductions[4 + simd_group] = key_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    reductions[0] =
        rsqrt((reductions[0] + reductions[1] + reductions[2] + reductions[3]) /
                  HeadDim +
              1e-6f);
    reductions[4] =
        rsqrt((reductions[4] + reductions[5] + reductions[6] + reductions[7]) /
                  HeadDim +
              1e-6f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const ulong key_row = ulong(token) * KeyWidth;
  q[key_row + query_channel] =
      bfloat(float(bfloat(query * reductions[0])) * 0.0078125f);
  k[key_row + query_channel] =
      bfloat(float(bfloat(key * reductions[4])) * 0.08838834765f);
  for (uint head = 0; head < HeadsPerKey; ++head) {
    const uint channel = value_channel + head * HeadDim;
    v[ulong(token) * ValueWidth + channel - 2 * KeyWidth] =
        gdn_conv_silu(packed, conv_state_in, conv_weights, params.packed_width,
                      ConvDim, token, channel);
  }
  if (thread_index < HeadsPerKey) {
    const uint head = key_head * HeadsPerKey + thread_index;
    const uint gate = token * ValueHeads + head;
    gdn_write_gates(packed + token * params.packed_width, dt_bias, a_scale,
                    BOffset, AOffset, head, beta[gate], decay[gate]);
  }
  if (token == 0) {
    for (uint row = 0; row < 3; ++row) {
      for (uint head = 0; head < HeadsPerKey + 2; ++head) {
        const uint channel = head == 0   ? query_channel
                             : head == 1 ? key_channel
                                         : value_channel + (head - 2) * HeadDim;
        conv_state_out[row * ConvDim + channel] =
            gdn_conv_carry(packed, conv_state_in, params.packed_width, ConvDim,
                           params.tokens, row, channel);
      }
    }
  }
}

#define GDN_PREPARE_PREFILL_ENTRY(Name, KeyHeads, ValueHeads, HeadDim,        \
                                  ConvDim)                                    \
  kernel void Name(                                                           \
      device const bfloat *packed [[buffer(0)]],                              \
      device const bfloat *conv_weights [[buffer(1)]],                        \
      device const bfloat *conv_state_in [[buffer(2)]],                       \
      device bfloat *conv_state_out [[buffer(3)]],                            \
      device bfloat *q [[buffer(4)]], device bfloat *k [[buffer(5)]],         \
      device bfloat *v [[buffer(6)]],                                         \
      device const float *a_scale [[buffer(7)]],                              \
      device const bfloat *dt_bias [[buffer(8)]],                             \
      device float *decay [[buffer(9)]], device bfloat *beta [[buffer(10)]],  \
      constant GDNPreparePrefillParams &params [[buffer(11)]],                \
      uint task [[threadgroup_position_in_grid]],                             \
      uint thread_index [[thread_index_in_threadgroup]],                      \
      uint lane [[thread_index_in_simdgroup]],                                \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                   \
    threadgroup float reductions[8];                                          \
    gdn_prepare_prefill_phase<KeyHeads, ValueHeads, HeadDim, ConvDim>(        \
        packed, conv_weights, conv_state_in, conv_state_out, q, k, v, a_scale,\
        dt_bias, decay, beta, params, reductions, task, thread_index, lane,   \
        simd_group);                                                          \
  }

GDN_PREPARE_PREFILL_ENTRY(prefill_gdn_prepare, 16, 48, 128, 10240)
GDN_PREPARE_PREFILL_ENTRY(prefill_gdn_prepare_vh32, 16, 32, 128, 8192)
#undef GDN_PREPARE_PREFILL_ENTRY

#define GDN_GATE_PREFILL_ENTRY(Name, KeyHeads, ValueHeads, HeadDim, ConvDim)   \
  kernel void Name(                                                           \
      device const bfloat *recurrent [[buffer(0)]],                           \
      device const bfloat *packed [[buffer(1)]],                              \
      device const bfloat *norm_weight [[buffer(2)]],                         \
      device bfloat *hidden [[buffer(3)]],                                    \
      constant GDNGatePrefillParams &params [[buffer(4)]],                    \
      uint task [[threadgroup_position_in_grid]],                             \
      uint thread_index [[thread_index_in_threadgroup]],                      \
      uint lane [[thread_index_in_simdgroup]],                                \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                   \
    threadgroup float reductions[4];                                          \
    gdn_gate_phase<KeyHeads, ValueHeads, HeadDim, ConvDim, 4>(                \
        recurrent, packed, norm_weight, hidden, params.tokens * ValueHeads,   \
        params.tokens * ValueHeads, params.packed_width,                      \
        params.tiled_heads != 0, reductions, task, thread_index, lane,        \
        simd_group);                                                          \
  }

GDN_GATE_PREFILL_ENTRY(prefill_gdn_gate, 16, 48, 128, 10240)
GDN_GATE_PREFILL_ENTRY(prefill_gdn_gate_vh32, 16, 32, 128, 8192)
#undef GDN_GATE_PREFILL_ENTRY

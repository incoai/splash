#include "metal/abi/KernelABI.h"
#include "metal/abi/NgramEmbedding.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/hyper_streams.h"
#include "metal/kernels/common/rms_inverse.h"

#include <metal_stdlib>

using namespace metal;

// The n-gram embedding (metal/abi/NgramEmbedding.h), Qwen3.8-Flash-Next's PLE,
// over a row's fp32 hyper-connected streams [rows][streams][hidden].

// The gate of a row: per stream the scaled dot product of the normalized
// key (the gathered n-gram rows' key projection) and the normalized stream,
// through a signed square root (of at least kNgramScoreFloor) and a sigmoid,
// gates the value projection. Writes the gated values (fp32, added to the
// streams) and their per-stream normalization (bf16), which the dilated
// convolution reads.
constant constexpr float kNgramScoreFloor = 1e-6f;
kernel void ngram_embedding_gate(device const bfloat *key [[buffer(0)]],
                                 device const bfloat *value [[buffer(1)]],
                                 device const float *streams [[buffer(2)]],
                                 device const float *key_norm [[buffer(3)]],
                                 device const float *query_norm [[buffer(4)]],
                                 device const float *convolution_norm [[buffer(5)]],
                                 device float *gated [[buffer(6)]],
                                 device bfloat *convolution_input [[buffer(7)]],
                                 constant NgramEmbeddingParams &p [[buffer(8)]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kSimdgroups * kStreams];
  if (row >= p.rows) return;
  const uint width = p.streams * p.hidden;
  device const bfloat *k = key + ulong(row) * width;
  device const float *x = streams + ulong(row) * width;
  device const bfloat *v = value + ulong(row) * p.hidden;
  float key_sums[kStreams] = {0.0f, 0.0f, 0.0f, 0.0f}, query_sums[kStreams] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint s = 0; s < p.streams; ++s)
    for (uint d = tid; d < p.hidden; d += kRowThreads) {
      const float kv = float(k[s * p.hidden + d]), qv = x[s * p.hidden + d];
      key_sums[s] += kv * kv;
      query_sums[s] += qv * qv;
    }
  reduce_streams(key_sums, p.streams, scratch, lane, sg);
  reduce_streams(query_sums, p.streams, scratch, lane, sg);
  float dots[kStreams] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint s = 0; s < p.streams; ++s) {
    const float ks = rsqrt(key_sums[s] / p.hidden + kRmsEpsilon);
    const float qs = rsqrt(query_sums[s] / p.hidden + kRmsEpsilon);
    for (uint d = tid; d < p.hidden; d += kRowThreads) {
      const uint c = s * p.hidden + d;
      dots[s] += (float(k[c]) * ks * key_norm[c]) * (x[c] * qs * query_norm[c]);
    }
  }
  reduce_streams(dots, p.streams, scratch, lane, sg);
  float gate[kStreams];
  for (uint s = 0; s < p.streams; ++s) {
    const float score = dots[s] / sqrt(float(p.hidden));
    gate[s] = splash_sigmoid(sign(score) * sqrt(max(abs(score), kNgramScoreFloor)));
  }
  // The gated values' norms: every stream holds the value times one gate.
  float value_sum = 0.0f;
  for (uint d = tid; d < p.hidden; d += kRowThreads) {
    const float vv = float(v[d]);
    value_sum += vv * vv;
  }
  float value_sums[kStreams] = {value_sum, 0.0f, 0.0f, 0.0f};
  reduce_streams(value_sums, 1, scratch, lane, sg);
  device float *g = gated + ulong(row) * width;
  device bfloat *out = convolution_input + ulong(row) * width;
  for (uint s = 0; s < p.streams; ++s) {
    const float cs = rsqrt(gate[s] * gate[s] * value_sums[0] / p.hidden + kRmsEpsilon);
    for (uint d = tid; d < p.hidden; d += kRowThreads) {
      const uint c = s * p.hidden + d;
      const float gv = float(v[d]) * gate[s];
      g[c] = gv;
      out[c] = bfloat(gv * cs * convolution_norm[c]);
    }
  }
}

// The depthwise convolution over one sequence's rows: tap k of channel
// c (taps[c][k]) reads the input (taps - 1 - k) * dilation rows back, before
// the first row from the history; silu of the sum and the gated values are
// added to the streams. With write_history, the history after the rows.
kernel void ngram_embedding_conv(device const bfloat *input [[buffer(0)]],
                                 device const bfloat *history_in [[buffer(1)]],
                                 device bfloat *history_out [[buffer(2)]],
                                 device const float *taps [[buffer(3)]],
                                 device const float *gated [[buffer(4)]],
                                 device float *streams [[buffer(5)]],
                                 constant NgramEmbeddingParams &p [[buffer(6)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
  const uint width = p.streams * p.hidden;
  const uint c = group.y * kRowThreads + tid, row = group.x;
  if (c >= width) return;
  const int history = int((p.taps - 1) * p.dilation);
  const auto at = [&](int t) -> float {
    return t >= 0 ? float(input[ulong(t) * width + c]) : float(history_in[ulong(history + t) * width + c]);
  };
  if (row < p.rows) {
    float sum = 0.0f;
    for (uint k = 0; k < p.taps; ++k)
      sum += taps[ulong(c) * p.taps + k] * at(int(row) - int((p.taps - 1 - k) * p.dilation));
    const ulong i = ulong(row) * width + c;
    streams[i] += gated[i] + splash_silu(sum);
  } else if (p.write_history && row - p.rows < uint(history)) {
    // Rows past the sequence write the history: row p.rows + j holds row
    // rows - history + j.
    const int j = int(row - p.rows);
    history_out[ulong(j) * width + c] = bfloat(at(int(p.rows) - history + j));
  }
}

// The history after a sequence's first `rows` rows: its history before,
// then the convolution inputs; threadgroup (j, y) writes row j's channels
// from y x kRowThreads on.
inline void ngram_history_rows(device const bfloat *input, device const bfloat *history_in,
                               device bfloat *history_out, constant NgramEmbeddingParams &p, uint rows,
                               uint2 group, uint tid) {
  const uint width = p.streams * p.hidden;
  const uint history = (p.taps - 1) * p.dilation;
  const uint c = group.y * kRowThreads + tid, j = group.x;
  if (c >= width || j >= history) return;
  const int source = int(rows) - int(history) + int(j);
  history_out[ulong(j) * width + c] = source >= 0 ? input[ulong(source) * width + c]
                                                  : history_in[ulong(int(history) + source) * width + c];
}

// The history after a prefill sequence's first p.rows rows (a state captured
// inside its command).
kernel void ngram_embedding_history(device const bfloat *input [[buffer(0)]],
                                    device const bfloat *history_in [[buffer(1)]],
                                    device bfloat *history_out [[buffer(2)]],
                                    constant NgramEmbeddingParams &p [[buffer(3)]],
                                    uint2 group [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
  ngram_history_rows(input, history_in, history_out, p, p.rows, group, tid);
}

// The history after a lane's retained rows of a verify step, a count the GPU
// wrote.
kernel void ngram_embedding_commit(device const bfloat *input [[buffer(0)]],
                                   device const bfloat *history_in [[buffer(1)]],
                                   device bfloat *history_out [[buffer(2)]],
                                   device const uint *retained [[buffer(3)]],
                                   constant NgramEmbeddingParams &p [[buffer(4)]],
                                   uint2 group [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
  ngram_history_rows(input, history_in, history_out, p, retained[0], group, tid);
}

// The table rows of each lane's verify rows: heads of the 2- to ngram-gram
// hashes of the row's token and those before it, cut at EOS or the
// sequence's start.
kernel void ngram_embedding_hash(device const uint *tokens [[buffer(0)]],
                                 device const uint *before [[buffer(1)]],
                                 device uint *indices [[buffer(2)]],
                                 constant NgramHashParams &p [[buffer(3)]],
                                 uint index [[thread_position_in_grid]]) {
  if (index >= p.rows) return;
  const uint lane = index / SPLASH_TARGET_VERIFY_ROWS, r = index % SPLASH_TARGET_VERIFY_ROWS;
  // The row's token, then the ngram - 1 tokens before it, EOS from a cut on.
  ulong context[NGRAM_EMBEDDING_MAXIMUM_NGRAM] = {tokens[index]};
  bool cut = false;
  for (uint back = 1; back < p.ngram; ++back) {
    const uint previous = back <= r ? tokens[index - back] : before[lane * (p.ngram - 1) + (back - r - 1)];
    cut = cut || previous == NGRAM_EMBEDDING_NONE || previous == p.eos;
    context[back] = cut ? p.eos : previous;
  }
  const uint heads = (p.ngram - 1) * p.heads_per_ngram;
  for (uint gram = 2; gram <= p.ngram; ++gram) {
    ulong mixed = context[0] * p.multipliers[0];
    for (uint j = 1; j < gram; ++j) mixed ^= context[j] * p.multipliers[j];
    for (uint g = 0; g < p.heads_per_ngram; ++g) {
      const uint head = (gram - 2) * p.heads_per_ngram + g;
      indices[ulong(index) * heads + head] = uint(mixed % p.vocabularies[head] + p.offsets[head]);
    }
  }
}

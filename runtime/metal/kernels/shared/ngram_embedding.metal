#include "metal/abi/KernelABI.h"
#include "metal/abi/NgramEmbedding.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/hyper_streams.h"
#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/rms_inverse.h"
#include "metal/kernels/common/verify_tree.h"

#include <metal_stdlib>

using namespace metal;

// The n-gram embedding (metal/abi/NgramEmbedding.h), Qwen3.8-Flash-Next's PLE
// (model/Qwen4Exp.hpp), over a row's fp32 hyper-connected streams
// [rows][streams][hidden].

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

// The depthwise convolution over each of a command's sequences' rows (a
// prompt chunk's sequences, or a verify step's lanes), threadgroup (row,
// channel group, sequence) of sequence s's rows its entry of the command's
// layout: tap k of channel c (taps[c][k]) reads the input (taps - 1 - k) *
// dilation rows back, before the sequence's first row from its history
// (history_in s); silu of the sum and the gated values are added to the
// streams. With write_history, the sequence's history after its rows
// (history_out s). A verify block's rows (p.tree, a row's place in `tree`)
// count back along their paths; a prompt's bind another buffer for the
// tree, which they do not read. A row past its sequence's rows and history,
// or a sequence whose rows run past the command's, writes nothing.
kernel void ngram_embedding_conv(device const bfloat *input [[buffer(0)]],
                                 device const bfloat *history_in0 [[buffer(1)]],
                                 device const bfloat *history_in1 [[buffer(2)]],
                                 device const bfloat *history_in2 [[buffer(3)]],
                                 device const bfloat *history_in3 [[buffer(4)]],
                                 device bfloat *history_out0 [[buffer(5)]],
                                 device bfloat *history_out1 [[buffer(6)]],
                                 device bfloat *history_out2 [[buffer(7)]],
                                 device bfloat *history_out3 [[buffer(8)]],
                                 device const float *taps [[buffer(9)]],
                                 device const float *gated [[buffer(10)]],
                                 device float *streams [[buffer(11)]],
                                 device const SplashLaneRows *layout [[buffer(12)]],
                                 device const SplashTreeRow *tree [[buffer(13)]],
                                 constant NgramEmbeddingParams &p [[buffer(14)]],
                                 uint3 group [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
  const uint width = p.streams * p.hidden;
  const uint c = group.y * kRowThreads + tid, row = group.x, s = group.z;
  if (c >= width || s >= min(p.sequences, SPLASH_MAXIMUM_BATCH_WIDTH)) return;
  const SplashLaneRows rows = layout[s];
  if (ulong(rows.first) + rows.rows > p.rows) return;
  device const bfloat *sequence = input + ulong(rows.first) * width;
  device const bfloat *history_in = SPLASH_LANE_BINDING(s, history_in0, history_in1, history_in2, history_in3);
  const int history = int((p.taps - 1) * p.dilation);
  const auto at = [&](int t) -> float {
    return t >= 0 ? float(sequence[ulong(t) * width + c]) : float(history_in[ulong(history + t) * width + c]);
  };
  if (row < rows.rows) {
    // The row's path and depth: a prompt's rows are a chain.
    const uint packed = rows.first + row;
    const uint path = p.tree ? tree[packed].ancestors : 0u;
    const int depth = p.tree ? int(tree[packed].depth) : int(row);
    float sum = 0.0f;
    for (uint k = 0; k < p.taps; ++k) {
      const int back = depth - int((p.taps - 1 - k) * p.dilation);
      sum += taps[ulong(c) * p.taps + k] * at(back >= 0 && p.tree ? int(splash_path_row(path, uint(back))) : back);
    }
    const ulong i = ulong(packed) * width + c;
    streams[i] += gated[i] + splash_silu(sum);
  } else if (p.write_history && row - rows.rows < uint(history)) {
    // Rows past the sequence's write its history: row rows + j holds row
    // rows - history + j.
    const int j = int(row - rows.rows);
    device bfloat *history_out = SPLASH_LANE_BINDING(s, history_out0, history_out1, history_out2, history_out3);
    history_out[ulong(j) * width + c] = bfloat(at(int(rows.rows) - history + j));
  }
}

// The history after a sequence's first `rows` rows, or the rows of a
// verify block's `path` (a row mask, its `rows` rows in order): its history
// before, then the convolution inputs; threadgroup (j, y) writes row j's
// channels from y x kRowThreads on.
inline void ngram_history_rows(device const bfloat *input, device const bfloat *history_in,
                               device bfloat *history_out, constant NgramEmbeddingParams &p, uint rows,
                               uint path, uint2 group, uint tid) {
  const uint width = p.streams * p.hidden;
  const uint history = (p.taps - 1) * p.dilation;
  const uint c = group.y * kRowThreads + tid, j = group.x;
  if (c >= width || j >= history) return;
  const int source = int(rows) - int(history) + int(j);
  if (source < 0) {
    history_out[ulong(j) * width + c] = history_in[ulong(int(history) + source) * width + c];
    return;
  }
  const uint row = path ? splash_path_row(path, uint(source)) : uint(source);
  history_out[ulong(j) * width + c] = input[ulong(row) * width + c];
}

// The history after a prefill sequence's first p.rows rows (a state captured
// inside its command).
kernel void ngram_embedding_history(device const bfloat *input [[buffer(0)]],
                                    device const bfloat *history_in [[buffer(1)]],
                                    device bfloat *history_out [[buffer(2)]],
                                    constant NgramEmbeddingParams &p [[buffer(3)]],
                                    uint2 group [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
  ngram_history_rows(input, history_in, history_out, p, p.rows, 0, group, tid);
}

// The history after each lane's accepted path of its verify block, a row
// mask the GPU wrote (acceptance's retained rows, paths[l]) of the lane's
// rows its entry of the step's layout holds: threadgroup (j, y, lane), from
// the lane's history before (history_in l) to its history after
// (history_out l). A lane whose rows run past the step's writes nothing.
kernel void ngram_embedding_commit(device const bfloat *input [[buffer(0)]],
                                   device const bfloat *history_in0 [[buffer(1)]],
                                   device const bfloat *history_in1 [[buffer(2)]],
                                   device const bfloat *history_in2 [[buffer(3)]],
                                   device const bfloat *history_in3 [[buffer(4)]],
                                   device bfloat *history_out0 [[buffer(5)]],
                                   device bfloat *history_out1 [[buffer(6)]],
                                   device bfloat *history_out2 [[buffer(7)]],
                                   device bfloat *history_out3 [[buffer(8)]],
                                   device const uint *paths [[buffer(9)]],
                                   device const SplashLaneRows *layout [[buffer(10)]],
                                   constant NgramEmbeddingParams &p [[buffer(11)]],
                                   uint3 group [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
  const uint lane = group.z;
  if (lane >= min(p.sequences, SPLASH_MAXIMUM_BATCH_WIDTH)) return;
  const SplashLaneRows rows = layout[lane];
  if (ulong(rows.first) + rows.rows > p.rows) return;
  const uint path = paths[lane];
  ngram_history_rows(input + ulong(rows.first) * p.streams * p.hidden,
                     SPLASH_LANE_BINDING(lane, history_in0, history_in1, history_in2, history_in3),
                     SPLASH_LANE_BINDING(lane, history_out0, history_out1, history_out2, history_out3), p,
                     popcount(path), path, group.xy, tid);
}

// The table rows of each lane's rows of a verify step (model/Qwen4Steps.cpp
// qwen4PleRows on the CPU): heads of the 2- to ngram-gram hashes of the row's
// token and those before it on its path, cut at EOS or the sequence's start.
// A thread per row of the step: its lane is the last whose first row is at
// most it, and a row past that lane's rows hashes nothing.
kernel void ngram_embedding_hash(device const uint *tokens [[buffer(0)]],
                                 device const SplashTreeRow *tree [[buffer(1)]],
                                 device const SplashLaneRows *lane_rows [[buffer(2)]],
                                 device const uint *before [[buffer(3)]],
                                 device uint *indices [[buffer(4)]],
                                 constant NgramHashParams &p [[buffer(5)]],
                                 uint index [[thread_position_in_grid]]) {
  uint lane = 0;
  for (uint l = 1; l < min(p.lanes, uint(SPLASH_MAXIMUM_BATCH_WIDTH)); ++l)
    lane = index >= lane_rows[l].first ? l : lane;
  const uint first = lane_rows[lane].first;
  if (index < first || index - first >= lane_rows[lane].rows) return;
  // The row's token, then the ngram - 1 tokens before it: its parents' up to
  // the anchor, then those before the lane's first row; EOS from a cut on.
  ulong context[NGRAM_EMBEDDING_MAXIMUM_NGRAM] = {tokens[index]};
  bool cut = false;
  uint row = index - first;
  const uint depth = tree[index].depth;
  for (uint back = 1; back < p.ngram; ++back) {
    uint previous;
    if (back <= depth) {
      row = tree[first + row].parent;
      previous = tokens[first + row];
    } else {
      previous = before[lane * (p.ngram - 1) + (back - depth - 1)];
    }
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

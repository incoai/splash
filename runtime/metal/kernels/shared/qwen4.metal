#include "metal/abi/KernelABI.h"
#include "metal/abi/QuantFormat.h"
#include "metal/abi/Qwen4.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/gguf_sgmatrix.h"
#include "metal/kernels/common/rms_inverse.h"

#include <metal_stdlib>

using namespace metal;

// Qwen3.8-Flash-Next's hyper-connections and PLE n-gram embedding
// (model/Qwen4Exp.hpp). The residual streams are fp32 [rows][streams][hidden];
// the projections between these kernels read and write bf16 rows.

constant constexpr uint kRowThreads = QWEN4_ROW_THREADS;
constant constexpr uint kSimdgroups = kRowThreads / 32;
constant constexpr uint kStreams = QWEN4_MAXIMUM_STREAMS;

// The sums of `values` (one per stream) over the threadgroup, in every thread.
inline void reduce_streams(thread float (&values)[kStreams], uint streams, threadgroup float *scratch,
                           uint lane, uint sg) {
  for (uint s = 0; s < streams; ++s) {
    const float sum = simd_sum(values[s]);
    if (lane == 0) scratch[sg * kStreams + s] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = 0; s < streams; ++s) {
    float total = 0.0f;
    for (uint g = 0; g < kSimdgroups; ++g) total += scratch[g * kStreams + s];
    values[s] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Writes the lane's two values of span `span` of row `row` into the row's
// 8-row tile of the Table16 table the register-tile projection reading the
// rows takes (gguf_sg::Table16, as decode_linear_gguf_prepare writes it).
inline void qwen4_table16_write(device bfloat *table, device float *sums, uint width, uint span, uint row,
                                uint lane, bfloat a, bfloat b) {
  gguf_sg::Table16::write(table + ulong(row / gguf_sg::kRows) * width * gguf_sg::kRows,
                          sums + ulong(row / gguf_sg::kRows) * gguf_sg::Table16::sums_per_tile(width), width,
                          span, row % gguf_sg::kRows, lane, a, b);
}

// A row's bf16 values into its table, read back from `values` after the
// threadgroup wrote them: a simdgroup per 64-value span, each of the
// threadgroup's `simdgroups` taking every simdgroups-th.
inline void qwen4_table16_row(device const bfloat *values, device bfloat *table, device float *sums, uint width,
                              uint row, uint lane, uint sg, uint simdgroups = kSimdgroups) {
  threadgroup_barrier(mem_flags::mem_device);
  for (uint g = sg; g < width / 64; g += simdgroups) {
    const uint k = g * 64 + 2 * lane;
    qwen4_table16_write(table, sums, width, g, row, lane, values[k], values[k + 1]);
  }
}

// The first mix step of a row: each stream's RMS norm times the mix's norm
// weights (the GGUF stores 1 + w) into bf16 rows of streams * hidden, and,
// when it injects, the injection weights 2 sigmoid(inject(normalized) /
// streams) of the block's output into each stream. Its kernels are compiled
// for Flash-Next's shape, kStreams streams of kNormHidden values. Each takes
// every load of a pass before the pass's arithmetic: at decode's few rows, a
// threadgroup per row that waits on each load in turn runs latency-bound.
// The two sum alike, so a row's output and weights are bit for bit the same
// whichever of them runs it.
constant constexpr uint kNormHidden = QWEN4_NORM_HIDDEN;
constant constexpr uint kNormValues = kNormHidden / kRowThreads; // per thread and stream
static_assert(kNormHidden % kRowThreads == 0, "a thread's values per stream are whole");

// Prefill's rows, and a mix that injects nothing: a threadgroup of
// kRowThreads threads per row; each thread holds its values of every stream.
template <bool Table>
inline void qwen4_hyper_norm_loaded_row(device const float *streams, device const float *norm,
                                        device const float *inject, device bfloat *normalized,
                                        device float *weights, device bfloat *table, device float *table_sums,
                                        constant Qwen4HyperParams &p, uint row, uint tid, uint lane, uint sg,
                                        threadgroup float *scratch) {
  if (row >= p.rows) return;
  const uint width = kStreams * kNormHidden;
  device const float *x = streams + ulong(row) * width;
  float xs[kStreams][kNormValues];
  for (uint s = 0; s < kStreams; ++s)
    for (uint i = 0; i < kNormValues; ++i) xs[s][i] = x[s * kNormHidden + tid + i * kRowThreads];
  float sums[kStreams] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint s = 0; s < kStreams; ++s)
    for (uint i = 0; i < kNormValues; ++i) sums[s] += xs[s][i] * xs[s][i];
  reduce_streams(sums, kStreams, scratch, lane, sg);
  float scale[kStreams];
  for (uint s = 0; s < kStreams; ++s) scale[s] = rsqrt(sums[s] / kNormHidden + kRmsEpsilon);
  float dots[kStreams] = {0.0f, 0.0f, 0.0f, 0.0f};
  device bfloat *out = normalized + ulong(row) * width;
  for (uint s = 0; s < kStreams; ++s) {
    float ns[kNormValues];
    for (uint i = 0; i < kNormValues; ++i) ns[i] = norm[s * kNormHidden + tid + i * kRowThreads];
    for (uint i = 0; i < kNormValues; ++i) {
      const uint c = s * kNormHidden + tid + i * kRowThreads;
      const float v = xs[s][i] * scale[s] * ns[i];
      out[c] = bfloat(v);
      if (p.inject)
        for (uint j = 0; j < kStreams; ++j) dots[j] += inject[ulong(j) * width + c] * v;
    }
  }
  if (Table) qwen4_table16_row(out, table, table_sums, width, row, lane, sg);
  if (!p.inject) return;
  reduce_streams(dots, kStreams, scratch, lane, sg);
  if (tid < kStreams) weights[row * kStreams + tid] = 2.0f * splash_sigmoid(dots[tid] / kStreams);
}

// Decode's rows of an injecting mix: a threadgroup of
// QWEN4_NORM_QUARTER_THREADS threads per row, whose quarter q stands for the
// loaded row's threadgroup in stream q's sum and in injection weight q's dot:
// each of its threads sums its values of stream q, then runs injection dot q
// over every stream's values in order, and writes stream q's normalized
// values.
template <bool Table>
inline void qwen4_hyper_norm_quarter_row(device const float *streams, device const float *norm,
                                         device const float *inject, device bfloat *normalized,
                                         device float *weights, device bfloat *table, device float *table_sums,
                                         constant Qwen4HyperParams &p, uint row, uint thread_id, uint lane,
                                         uint simdgroup, threadgroup float *scratch, threadgroup float *scales) {
  if (row >= p.rows) return;
  const uint q = thread_id / kRowThreads, tid = thread_id % kRowThreads;
  const uint width = kStreams * kNormHidden;
  device const float *x = streams + ulong(row) * width;
  float mine[kNormValues];
  for (uint i = 0; i < kNormValues; ++i) mine[i] = x[q * kNormHidden + tid + i * kRowThreads];
  float sum = 0.0f;
  for (uint i = 0; i < kNormValues; ++i) sum += mine[i] * mine[i];
  // The threadgroup's simdgroup `simdgroup` is quarter q's simdgroup
  // simdgroup % kSimdgroups.
  sum = simd_sum(sum);
  if (lane == 0) scratch[simdgroup] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float total = 0.0f;
    for (uint g = 0; g < kSimdgroups; ++g) total += scratch[q * kSimdgroups + g];
    scales[q] = rsqrt(total / kNormHidden + kRmsEpsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device bfloat *out = normalized + ulong(row) * width;
  float dot = 0.0f;
  if (!p.inject) {
    for (uint i = 0; i < kNormValues; ++i) {
      const uint c = q * kNormHidden + tid + i * kRowThreads;
      out[c] = bfloat(mine[i] * scales[q] * norm[c]);
    }
  } else {
    device const float *w = inject + ulong(q) * width;
    for (uint s = 0; s < kStreams; ++s) {
      const float scale = scales[s];
      float xv[kNormValues], ns[kNormValues], ws[kNormValues];
      for (uint i = 0; i < kNormValues; ++i) {
        const uint c = s * kNormHidden + tid + i * kRowThreads;
        xv[i] = x[c];
        ns[i] = norm[c];
        ws[i] = w[c];
      }
      for (uint i = 0; i < kNormValues; ++i) {
        const uint c = s * kNormHidden + tid + i * kRowThreads;
        const float v = xv[i] * scale * ns[i];
        if (s == q) out[c] = bfloat(v);
        dot += ws[i] * v;
      }
    }
  }
  if (Table) qwen4_table16_row(out, table, table_sums, width, row, lane, simdgroup, kStreams * kSimdgroups);
  if (!p.inject) return;
  dot = simd_sum(dot);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) scratch[simdgroup] = dot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid != 0) return;
  float total = 0.0f;
  for (uint g = 0; g < kSimdgroups; ++g) total += scratch[q * kSimdgroups + g];
  weights[row * kStreams + q] = 2.0f * splash_sigmoid(total / kStreams);
}

#define QWEN4_NORM_ARGUMENTS                                                                                     \
  device const float *streams [[buffer(0)]], device const float *norm [[buffer(1)]],                            \
      device const float *inject [[buffer(2)]], device bfloat *normalized [[buffer(3)]],                        \
      device float *weights [[buffer(4)]]

kernel void qwen4_hyper_norm_loaded(QWEN4_NORM_ARGUMENTS, constant Qwen4HyperParams &p [[buffer(5)]],
                                   uint row [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kSimdgroups * kStreams];
  qwen4_hyper_norm_loaded_row<false>(streams, norm, inject, normalized, weights, nullptr, nullptr, p, row, tid, lane,
                                     sg, scratch);
}

kernel void qwen4_hyper_norm_loaded_table16(QWEN4_NORM_ARGUMENTS, device bfloat *table [[buffer(5)]],
                                           device float *sums [[buffer(6)]],
                                           constant Qwen4HyperParams &p [[buffer(7)]],
                                           uint row [[threadgroup_position_in_grid]],
                                           uint tid [[thread_index_in_threadgroup]],
                                           uint lane [[thread_index_in_simdgroup]],
                                           uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kSimdgroups * kStreams];
  qwen4_hyper_norm_loaded_row<true>(streams, norm, inject, normalized, weights, table, sums, p, row, tid, lane, sg,
                                    scratch);
}

kernel void qwen4_hyper_norm_quarters(QWEN4_NORM_ARGUMENTS, constant Qwen4HyperParams &p [[buffer(5)]],
                                            uint row [[threadgroup_position_in_grid]],
                                            uint thread_id [[thread_index_in_threadgroup]],
                                            uint lane [[thread_index_in_simdgroup]],
                                            uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kStreams * kSimdgroups];
  threadgroup float scales[kStreams];
  qwen4_hyper_norm_quarter_row<false>(streams, norm, inject, normalized, weights, nullptr, nullptr, p, row,
                                      thread_id, lane, simdgroup, scratch, scales);
}

kernel void qwen4_hyper_norm_quarters_table16(QWEN4_NORM_ARGUMENTS, device bfloat *table [[buffer(5)]],
                                                    device float *sums [[buffer(6)]],
                                                    constant Qwen4HyperParams &p [[buffer(7)]],
                                                    uint row [[threadgroup_position_in_grid]],
                                                    uint thread_id [[thread_index_in_threadgroup]],
                                                    uint lane [[thread_index_in_simdgroup]],
                                                    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kStreams * kSimdgroups];
  threadgroup float scales[kStreams];
  qwen4_hyper_norm_quarter_row<true>(streams, norm, inject, normalized, weights, table, sums, p, row, thread_id,
                                     lane, simdgroup, scratch, scales);
}

#undef QWEN4_NORM_ARGUMENTS

// silu(low / streams) of the mix's low-rank projection, in place.
kernel void qwen4_hyper_low(device bfloat *low [[buffer(0)]],
                            constant Qwen4HyperParams &p [[buffer(1)]],
                            uint index [[thread_position_in_grid]]) {
  if (index >= p.rows * p.rank) return;
  low[index] = bfloat(splash_silu(float(low[index]) / p.streams));
}

// The same values into the up projection's table alone (a decode step's
// rows): a threadgroup per row, a simdgroup per 64-value span.
kernel void qwen4_hyper_low_table16(device const bfloat *low [[buffer(0)]],
                                    device bfloat *table [[buffer(1)]],
                                    device float *sums [[buffer(2)]],
                                    constant Qwen4HyperParams &p [[buffer(3)]],
                                    uint row [[threadgroup_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]]) {
  if (row >= p.rows) return;
  device const bfloat *x = low + ulong(row) * p.rank;
  for (uint g = sg; g < p.rank / 64; g += kSimdgroups) {
    const uint k = g * 64 + 2 * lane;
    qwen4_table16_write(table, sums, p.rank, g, row, lane, bfloat(splash_silu(float(x[k]) / p.streams)),
                        bfloat(splash_silu(float(x[k + 1]) / p.streams)));
  }
}

// The block's input: the mean over the streams of the normalized streams
// gated by sigmoid of the mix's up projection.
kernel void qwen4_hyper_mix(device const bfloat *normalized [[buffer(0)]],
                            device const bfloat *gate [[buffer(1)]],
                            device bfloat *mixed [[buffer(2)]],
                            constant Qwen4HyperParams &p [[buffer(3)]],
                            uint2 group [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
  const uint row = group.x, d = group.y * kRowThreads + tid;
  if (row >= p.rows || d >= p.hidden) return;
  const ulong base = ulong(row) * p.streams * p.hidden + d;
  float sum = 0.0f;
  for (uint s = 0; s < p.streams; ++s)
    sum += float(normalized[base + s * p.hidden]) * splash_sigmoid(float(gate[base + s * p.hidden]));
  mixed[ulong(row) * p.hidden + d] = bfloat(sum / p.streams);
}

// The mix with the next projection's table (a decode step's rows, hidden a
// multiple of the threadgroup): the threadgroup's 256 values are four
// 64-value spans, which its first four simdgroups write.
kernel void qwen4_hyper_mix_table16(device const bfloat *normalized [[buffer(0)]],
                                    device const bfloat *gate [[buffer(1)]],
                                    device bfloat *mixed [[buffer(2)]],
                                    device bfloat *table [[buffer(3)]],
                                    device float *sums [[buffer(4)]],
                                    constant Qwen4HyperParams &p [[buffer(5)]],
                                    uint2 group [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup bfloat values[kRowThreads];
  const uint row = group.x, d = group.y * kRowThreads + tid;
  if (row >= p.rows) return;
  const ulong base = ulong(row) * p.streams * p.hidden + d;
  float sum = 0.0f;
  for (uint s = 0; s < p.streams; ++s)
    sum += float(normalized[base + s * p.hidden]) * splash_sigmoid(float(gate[base + s * p.hidden]));
  const bfloat value = bfloat(sum / p.streams);
  mixed[ulong(row) * p.hidden + d] = value;
  values[tid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg >= kRowThreads / 64) return;
  const uint k = sg * 64 + 2 * lane;
  qwen4_table16_write(table, sums, p.hidden, group.y * (kRowThreads / 64) + sg, row, lane, values[k],
                      values[k + 1]);
}

// Every stream plus the block's output scaled by the stream's injection weight.
kernel void qwen4_hyper_combine(device float *streams [[buffer(0)]],
                                device const bfloat *branch [[buffer(1)]],
                                device const float *weights [[buffer(2)]],
                                constant Qwen4HyperParams &p [[buffer(3)]],
                                uint2 group [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]]) {
  const uint row = group.x, d = group.y * kRowThreads + tid;
  if (row >= p.rows || d >= p.hidden) return;
  const float b = float(branch[ulong(row) * p.hidden + d]);
  device float *x = streams + ulong(row) * p.streams * p.hidden + d;
  for (uint s = 0; s < p.streams; ++s) x[s * p.hidden] += b * weights[row * p.streams + s];
}

// The streams' start: every stream a copy of the token's embedding.
kernel void qwen4_hyper_init(device const bfloat *embedding [[buffer(0)]],
                             device float *streams [[buffer(1)]],
                             constant Qwen4HyperParams &p [[buffer(2)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
  const uint row = group.x, d = group.y * kRowThreads + tid;
  if (row >= p.rows || d >= p.hidden) return;
  const float e = float(embedding[ulong(row) * p.hidden + d]);
  device float *x = streams + ulong(row) * p.streams * p.hidden + d;
  for (uint s = 0; s < p.streams; ++s) x[s * p.hidden] = e;
}

// The PLE gate of a row: per stream the scaled dot product of the normalized
// key (the gathered n-gram rows' key projection) and the normalized stream,
// through a signed square root (of at least kPleScoreFloor) and a sigmoid,
// gates the value projection. Writes the gated values (fp32, added to the
// streams) and their per-stream normalization (bf16), which the dilated
// convolution reads.
constant constexpr float kPleScoreFloor = 1e-6f;
kernel void qwen4_ple_gate(device const bfloat *key [[buffer(0)]],
                           device const bfloat *value [[buffer(1)]],
                           device const float *streams [[buffer(2)]],
                           device const float *key_norm [[buffer(3)]],
                           device const float *query_norm [[buffer(4)]],
                           device const float *convolution_norm [[buffer(5)]],
                           device float *gated [[buffer(6)]],
                           device bfloat *convolution_input [[buffer(7)]],
                           constant Qwen4PleParams &p [[buffer(8)]],
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
    gate[s] = splash_sigmoid(sign(score) * sqrt(max(abs(score), kPleScoreFloor)));
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

// The PLE's depthwise convolution over one sequence's rows: tap k of channel
// c (taps[c][k]) reads the input (taps - 1 - k) * dilation rows back, before
// the first row from the history; silu of the sum and the gated values are
// added to the streams. With write_history, the history after the rows.
kernel void qwen4_ple_conv(device const bfloat *input [[buffer(0)]],
                           device const bfloat *history_in [[buffer(1)]],
                           device bfloat *history_out [[buffer(2)]],
                           device const float *taps [[buffer(3)]],
                           device const float *gated [[buffer(4)]],
                           device float *streams [[buffer(5)]],
                           constant Qwen4PleParams &p [[buffer(6)]],
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

// The PLE history after a sequence's first `rows` rows: its history before,
// then the convolution inputs; threadgroup (j, y) writes row j's channels
// from y x kRowThreads on.
inline void qwen4_ple_history_rows(device const bfloat *input, device const bfloat *history_in,
                                   device bfloat *history_out, constant Qwen4PleParams &p, uint rows,
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
kernel void qwen4_ple_history(device const bfloat *input [[buffer(0)]],
                              device const bfloat *history_in [[buffer(1)]],
                              device bfloat *history_out [[buffer(2)]],
                              constant Qwen4PleParams &p [[buffer(3)]],
                              uint2 group [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
  qwen4_ple_history_rows(input, history_in, history_out, p, p.rows, group, tid);
}

// The history after a lane's retained rows of a verify step, a count the GPU
// wrote.
kernel void qwen4_ple_commit(device const bfloat *input [[buffer(0)]],
                             device const bfloat *history_in [[buffer(1)]],
                             device bfloat *history_out [[buffer(2)]],
                             device const uint *retained [[buffer(3)]],
                             constant Qwen4PleParams &p [[buffer(4)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
  qwen4_ple_history_rows(input, history_in, history_out, p, retained[0], group, tid);
}

// The mean of each row's streams (bf16), at column p.column of row p.row + r
// of a destination of p.stride values: Qwen3.8-Flash-Next's capture of a
// layer's output, which its DFlash2 draft reads.
kernel void qwen4_stream_mean(device const float *streams [[buffer(0)]],
                              device bfloat *destination [[buffer(1)]],
                              constant Qwen4StreamMeanParams &p [[buffer(2)]],
                              uint2 group [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
  const uint row = group.x, d = group.y * kRowThreads + tid;
  if (row >= p.rows || d >= p.hidden) return;
  device const float *x = streams + ulong(row) * p.streams * p.hidden + d;
  float sum = 0.0f;
  for (uint s = 0; s < p.streams; ++s) sum += x[s * p.hidden];
  destination[ulong(p.row + row) * p.stride + p.column + d] = bfloat(sum / float(p.streams));
}

// ---- MTP head (model/Qwen4Exp.hpp Qwen4MtpWeights) ----

// A draft step's tokens: row r of each lane the anchor (r = 0), proposal
// r - 1, or, from p.limit on, a repeat of row limit - 1.
kernel void qwen4_mtp_tokens(device const uint *anchors [[buffer(0)]],
                             device const uint *proposals [[buffer(1)]],
                             device uint *tokens [[buffer(2)]],
                             constant Qwen4MtpParams &p [[buffer(3)]],
                             uint index [[thread_position_in_grid]]) {
  if (index >= p.rows) return;
  const uint lane = index / p.lane_rows, r = min(index % p.lane_rows, p.limit - 1);
  tokens[index] = r == 0 ? anchors[lane * p.lane_rows] : proposals[lane * p.proposals + r - 1];
}

// One lane's rows: per stream [enorm(embedding) ; hnorm_s(streams_s)] (bf16
// rows of 2 * hidden), the input of the head's eh projection, stream-major:
// stream s of row r of the dispatch at row s * lane_rows + p.row + r.
kernel void qwen4_mtp_input(device const bfloat *embedding [[buffer(0)]],
                            device const float *carried [[buffer(1)]],
                            device const float *history [[buffer(2)]],
                            device const float *embed_norm [[buffer(3)]],
                            device const float *hidden_norm [[buffer(4)]],
                            device bfloat *output [[buffer(5)]],
                            constant Qwen4MtpParams &p [[buffer(6)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kSimdgroups * kStreams];
  if (row >= p.rows) return;
  const uint source = min(row, p.limit - 1);
  const uint width = p.streams * p.hidden;
  device const bfloat *e = embedding + ulong(source) * p.hidden;
  device const float *h = source == 0 ? carried : history + ulong(source - 1) * width;
  float sums[kStreams] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint s = 0; s < p.streams; ++s)
    for (uint d = tid; d < p.hidden; d += kRowThreads) {
      const float v = h[s * p.hidden + d];
      sums[s] += v * v;
    }
  reduce_streams(sums, p.streams, scratch, lane, sg);
  float embed_sum[kStreams] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint d = tid; d < p.hidden; d += kRowThreads) {
    const float v = float(e[d]);
    embed_sum[0] += v * v;
  }
  reduce_streams(embed_sum, 1, scratch, lane, sg);
  const float embed_scale = rsqrt(embed_sum[0] / p.hidden + kRmsEpsilon);
  for (uint s = 0; s < p.streams; ++s) {
    const float scale = rsqrt(sums[s] / p.hidden + kRmsEpsilon);
    device bfloat *o = output + (ulong(s) * p.lane_rows + p.row + row) * 2 * p.hidden;
    for (uint d = tid; d < p.hidden; d += kRowThreads) {
      o[d] = bfloat(float(e[d]) * embed_scale * embed_norm[d]);
      o[p.hidden + d] = bfloat(h[s * p.hidden + d] * scale * hidden_norm[s * p.hidden + d]);
    }
  }
}

// fp32 streams [rows][streams][hidden] from the eh projection's
// stream-major bf16 rows [streams][rows][hidden].
kernel void qwen4_mtp_widen(device const bfloat *projected [[buffer(0)]],
                            device float *streams [[buffer(1)]],
                            constant Qwen4MtpParams &p [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
  const uint width = p.streams * p.hidden;
  if (index >= p.rows * width) return;
  const uint row = index / width, c = index % width, s = c / p.hidden, d = c % p.hidden;
  streams[index] = float(projected[(ulong(s) * p.lane_rows + row) * p.hidden + d]);
}

// One threadgroup per lane: the argmax of the lane's logits row p.row, over
// p.vocabulary logits: the tokens themselves or, Mapped, the logits of a
// draft vocabulary's tokens, ascending, of which `tokens` names each. Ties
// go toward the lower index, so toward the lower token either way, as its
// proposal p.row (and, p.last, every later one) and its first sparse
// candidate. A row without a finite logit drafts 0.
template <bool Mapped>
inline void qwen4_mtp_argmax_row(device const float *logits, device uint *proposals, device uint *candidates,
                                 device const uint *tokens, constant Qwen4MtpParams &p, uint lane_index, uint tid,
                                 uint threads, uint lane, uint sg, threadgroup float *best_values,
                                 threadgroup uint *best_ids) {
  device const float *x = logits + (ulong(lane_index) * p.lane_rows + p.row) * p.vocabulary;
  float best = -INFINITY;
  uint id = 0xffffffffu;
  for (uint v = tid; v < p.vocabulary; v += threads) {
    const float value = x[v];
    if (value > best) {
      best = value;
      id = v;
    }
  }
  for (uint offset = 16; offset > 0; offset >>= 1) {
    const float other = simd_shuffle_down(best, offset);
    const uint other_id = simd_shuffle_down(id, offset);
    if (other > best || (other == best && other_id < id)) {
      best = other;
      id = other_id;
    }
  }
  if (lane == 0) {
    best_values[sg] = best;
    best_ids[sg] = id;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid != 0) return;
  const uint groups = (threads + 31) / 32;
  for (uint g = 1; g < groups; ++g)
    if (best_values[g] > best || (best_values[g] == best && best_ids[g] < id)) {
      best = best_values[g];
      id = best_ids[g];
    }
  if (id >= p.vocabulary) id = 0;
  else if (Mapped) id = tokens[id];
  const uint end = p.last ? p.proposals : p.row + 1;
  for (uint slot = p.row; slot < end; ++slot) {
    proposals[lane_index * p.proposals + slot] = id;
    candidates[(lane_index * p.proposals + slot) * p.candidates] = id;
  }
}

kernel void qwen4_mtp_argmax(device const float *logits [[buffer(0)]],
                             device uint *proposals [[buffer(1)]],
                             device uint *candidates [[buffer(2)]],
                             constant Qwen4MtpParams &p [[buffer(3)]],
                             uint lane_index [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint threads [[threads_per_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float best_values[32];
  threadgroup uint best_ids[32];
  qwen4_mtp_argmax_row<false>(logits, proposals, candidates, nullptr, p, lane_index, tid, threads, lane, sg,
                              best_values, best_ids);
}

kernel void qwen4_mtp_argmax_mapped(device const float *logits [[buffer(0)]],
                                    device uint *proposals [[buffer(1)]],
                                    device uint *candidates [[buffer(2)]],
                                    device const uint *tokens [[buffer(3)]],
                                    constant Qwen4MtpParams &p [[buffer(4)]],
                                    uint lane_index [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint threads [[threads_per_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float best_values[32];
  threadgroup uint best_ids[32];
  qwen4_mtp_argmax_row<true>(logits, proposals, candidates, tokens, p, lane_index, tid, threads, lane, sg,
                             best_values, best_ids);
}

// Takes (value, t) as the best so far when it is larger, or as large with a
// lower token.
inline void qwen4_consider(thread float &best, thread uint &token, float value, uint t) {
  if (value > best || (value == best && t < token)) {
    best = value;
    token = t;
  }
}

// The byte offset of unit `unit` of `row` in planes [rows / T][units][T][bytes].
inline ulong qwen4_plane_offset(uint row, uint units, uint unit, uint bytes) {
  return ((ulong(row / QUANT_TILE_ROWS) * units + unit) * QUANT_TILE_ROWS + row % QUANT_TILE_ROWS) * bytes;
}

// One threadgroup per lane: the argmax of the lane's row p.row over the
// draft vocabulary's logits (`tokens` naming each) and the lane's context
// rows' logits (Qwen4MtpParams context_*), ties toward the lower token, as its
// proposal p.row (and, p.last, every later one) and its first candidate.
kernel void qwen4_mtp_argmax_context(device const float *logits [[buffer(0)]],
                                     device uint *proposals [[buffer(1)]],
                                     device uint *candidates [[buffer(2)]],
                                     device const uint *tokens [[buffer(3)]],
                                     device const float *context_logits [[buffer(4)]],
                                     device const uint *context_tokens [[buffer(5)]],
                                     device const Qwen4MtpContextLane *lanes [[buffer(6)]],
                                     constant Qwen4MtpParams &p [[buffer(7)]],
                                     uint lane_index [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint threads [[threads_per_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float best_values[32];
  threadgroup uint best_tokens[32];
  float best = -INFINITY;
  uint token = 0xffffffffu;
  device const float *x = logits + (ulong(lane_index) * p.lane_rows + p.row) * p.vocabulary;
  for (uint v = tid; v < p.vocabulary; v += threads) qwen4_consider(best, token, x[v], tokens[v]);
  const Qwen4MtpContextLane own = lanes[lane_index];
  const uint count = min(own.count, p.context_capacity);
  device const float *y = context_logits + ulong(lane_index) * p.context_slot + ulong(p.row) * p.context_capacity;
  device const uint *named = context_tokens + ulong(own.state_lane) * p.context_capacity;
  for (uint v = tid; v < count; v += threads) qwen4_consider(best, token, y[v], named[v]);
  for (uint offset = 16; offset > 0; offset >>= 1) {
    const float other = simd_shuffle_down(best, offset);
    const uint other_token = simd_shuffle_down(token, offset);
    qwen4_consider(best, token, other, other_token);
  }
  if (lane == 0) {
    best_values[sg] = best;
    best_tokens[sg] = token;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid != 0) return;
  const uint groups = (threads + 31) / 32;
  for (uint g = 1; g < groups; ++g) qwen4_consider(best, token, best_values[g], best_tokens[g]);
  // No finite logit drafts token 0, as without a draft vocabulary.
  if (token == 0xffffffffu) token = 0;
  const uint end = p.last ? p.proposals : p.row + 1;
  for (uint slot = p.row; slot < end; ++slot) {
    proposals[lane_index * p.proposals + slot] = token;
    candidates[(lane_index * p.proposals + slot) * p.candidates] = token;
  }
}

// One thread per (pair, group): copies the group's plane bytes of each
// pair's source row to its destination row, and its meta unit when the group
// starts one. Planes are [rows / T][units][T][unit bytes].
kernel void qwen4_copy_head_rows(device const uchar *source0 [[buffer(0)]],
                                 device const uchar *source1 [[buffer(1)]],
                                 device const uchar *source_meta [[buffer(2)]],
                                 device uchar *target0 [[buffer(3)]],
                                 device uchar *target1 [[buffer(4)]],
                                 device uchar *target_meta [[buffer(5)]],
                                 device const uint2 *pairs [[buffer(6)]],
                                 constant Qwen4HeadRowsParams &p [[buffer(7)]],
                                 uint2 index [[thread_position_in_grid]]) {
  const uint pair = index.x, group = index.y;
  if (pair >= p.count || group >= p.groups) return;
  const uint2 rows = pairs[pair];
  const ulong from0 = qwen4_plane_offset(rows.x, p.groups, group, p.plane0_bytes);
  const ulong to0 = qwen4_plane_offset(rows.y, p.groups, group, p.plane0_bytes);
  for (uint b = 0; b < p.plane0_bytes; ++b) target0[to0 + b] = source0[from0 + b];
  if (p.plane1_bytes) {
    const ulong from1 = qwen4_plane_offset(rows.x, p.groups, group, p.plane1_bytes);
    const ulong to1 = qwen4_plane_offset(rows.y, p.groups, group, p.plane1_bytes);
    for (uint b = 0; b < p.plane1_bytes; ++b) target1[to1 + b] = source1[from1 + b];
  }
  if (group % p.meta_groups == 0) {
    const uint units = p.groups / p.meta_groups, unit = group / p.meta_groups;
    const ulong from = qwen4_plane_offset(rows.x, units, unit, p.meta_bytes);
    const ulong to = qwen4_plane_offset(rows.y, units, unit, p.meta_bytes);
    for (uint b = 0; b < p.meta_bytes; ++b) target_meta[to + b] = source_meta[from + b];
  }
}

// A lane's (or sequence's) carried streams: one row of its streams.
kernel void qwen4_mtp_carry(device const float *streams [[buffer(0)]],
                            device float *carried [[buffer(1)]],
                            device const uint *retained [[buffer(2)]],
                            constant Qwen4MtpParams &p [[buffer(3)]],
                            uint index [[thread_position_in_grid]]) {
  const uint width = p.streams * p.hidden;
  if (index >= width) return;
  const uint row = (p.counted ? retained[0] : p.limit) - 1;
  carried[index] = streams[ulong(row) * width + index];
}

// The PLE rows of each lane's verify rows (model/Qwen4Steps.cpp qwen4PleRows
// on the CPU): heads of the 2- to ngram-gram hashes of the row's token and
// those before it, cut at EOS or the sequence's start.
kernel void qwen4_ple_hash(device const uint *tokens [[buffer(0)]],
                           device const uint *before [[buffer(1)]],
                           device uint *indices [[buffer(2)]],
                           constant Qwen4PleHashParams &p [[buffer(3)]],
                           uint index [[thread_position_in_grid]]) {
  if (index >= p.rows) return;
  const uint lane = index / SPLASH_TARGET_VERIFY_ROWS, r = index % SPLASH_TARGET_VERIFY_ROWS;
  // The row's token, then the ngram - 1 tokens before it, EOS from a cut on.
  ulong context[QWEN4_PLE_MAXIMUM_NGRAM] = {tokens[index]};
  bool cut = false;
  for (uint back = 1; back < p.ngram; ++back) {
    const uint previous = back <= r ? tokens[index - back] : before[lane * (p.ngram - 1) + (back - r - 1)];
    cut = cut || previous == QWEN4_PLE_NONE || previous == p.eos;
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

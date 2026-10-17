#include "metal/abi/KernelABI.h"
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

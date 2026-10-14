#include "metal/abi/KernelABI.h"
#include "metal/abi/Qwen4.h"
#include "metal/kernels/common/gguf_sgmatrix.h"

#include <metal_stdlib>

using namespace metal;

// Qwen3.8-Flash-Next's hyper-connections and PLE n-gram embedding
// (model/Qwen4Exp.hpp; llama.cpp qwen4exp.cpp build_hc_mix, build_hc_combine,
// build_ple). The residual streams are fp32 [rows][streams][hidden]; the
// projections between these kernels read and write bf16 rows.

constant constexpr uint kRowThreads = QWEN4_ROW_THREADS;
constant constexpr uint kSimdgroups = kRowThreads / 32;
constant constexpr uint kStreams = QWEN4_MAXIMUM_STREAMS;

inline float sigmoid_of(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float silu_of(float x) { return x * sigmoid_of(x); }

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

// The first mix step of a row: each stream's RMS norm times the mix's norm
// weights (the GGUF stores 1 + w) into bf16 rows of streams * hidden, and,
// when it injects, the injection weights 2 sigmoid(inject(normalized) /
// streams) of the block's output into each stream.
// A row's bf16 values as the register-tile projection that reads them takes
// them (gguf_sg::Table16, as decode_linear_gguf_prepare writes it), read back
// from `values` after the threadgroup wrote them: a simdgroup per 64-value
// span, each of the threadgroup's `simdgroups` taking every simdgroups-th.
inline void qwen4_table16_row(device const bfloat *values, device bfloat *table, device float *sums, uint width,
                              uint row, uint lane, uint sg, uint simdgroups = kSimdgroups) {
  threadgroup_barrier(mem_flags::mem_device);
  for (uint g = sg; g < width / 64; g += simdgroups) {
    const uint k = g * 64 + 2 * lane;
    gguf_sg::Table16::write(table + ulong(row / 8) * width * 8, sums + ulong(row / 8) * gguf_sg::Table16::sums_per_tile(width),
                            width, g, row % 8, lane, values[k], values[k + 1]);
  }
}

template <bool Table>
inline void qwen4_hyper_norm_row(device const float *streams, device const float *norm, device const float *inject,
                                 device bfloat *normalized, device float *weights, device bfloat *table,
                                 device float *table_sums, constant Qwen4HyperParams &p, uint row, uint tid, uint lane,
                                 uint sg, threadgroup float *scratch) {
  if (row >= p.rows) return;
  const uint width = p.streams * p.hidden;
  device const float *x = streams + ulong(row) * width;
  float sums[kStreams] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint s = 0; s < p.streams; ++s)
    for (uint d = tid; d < p.hidden; d += kRowThreads) {
      const float v = x[s * p.hidden + d];
      sums[s] += v * v;
    }
  reduce_streams(sums, p.streams, scratch, lane, sg);
  float scale[kStreams];
  for (uint s = 0; s < p.streams; ++s) scale[s] = rsqrt(sums[s] / p.hidden + QWEN4_RMS_EPSILON);
  float dots[kStreams] = {0.0f, 0.0f, 0.0f, 0.0f};
  device bfloat *out = normalized + ulong(row) * width;
  for (uint s = 0; s < p.streams; ++s)
    for (uint d = tid; d < p.hidden; d += kRowThreads) {
      const uint c = s * p.hidden + d;
      const float v = x[c] * scale[s] * norm[c];
      out[c] = bfloat(v);
      if (p.inject)
        for (uint j = 0; j < p.streams; ++j) dots[j] += inject[ulong(j) * width + c] * v;
    }
  if (Table) qwen4_table16_row(out, table, table_sums, width, row, lane, sg);
  if (!p.inject) return;
  reduce_streams(dots, p.streams, scratch, lane, sg);
  if (tid < p.streams) weights[row * p.streams + tid] = 2.0f * sigmoid_of(dots[tid] / p.streams);
}

kernel void qwen4_hyper_norm(device const float *streams [[buffer(0)]],
                             device const float *norm [[buffer(1)]],
                             device const float *inject [[buffer(2)]],
                             device bfloat *normalized [[buffer(3)]],
                             device float *weights [[buffer(4)]],
                             constant Qwen4HyperParams &p [[buffer(5)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kSimdgroups * kStreams];
  qwen4_hyper_norm_row<false>(streams, norm, inject, normalized, weights, nullptr, nullptr, p, row, tid, lane, sg,
                              scratch);
}

// The norm with the down projection's table (decode rows, whole lanes).
kernel void qwen4_hyper_norm_table16(device const float *streams [[buffer(0)]],
                                     device const float *norm [[buffer(1)]],
                                     device const float *inject [[buffer(2)]],
                                     device bfloat *normalized [[buffer(3)]],
                                     device float *weights [[buffer(4)]],
                                     device bfloat *table [[buffer(5)]],
                                     device float *sums [[buffer(6)]],
                                     constant Qwen4HyperParams &p [[buffer(7)]],
                                     uint row [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]],
                                     uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kSimdgroups * kStreams];
  qwen4_hyper_norm_row<true>(streams, norm, inject, normalized, weights, table, sums, p, row, tid, lane, sg, scratch);
}

// The same norm for Flash-Next's shape, four streams of QWEN4_NORM_HIDDEN
// values: the same sums in the same order, so bit for bit the same output and
// weights. qwen4_hyper_norm_row runs latency-bound at decode's few rows: its
// threadgroup per row waits on each load of its passes in turn. These take
// every load of a pass before its arithmetic.
constant constexpr uint kNormHidden = QWEN4_NORM_HIDDEN;
constant constexpr uint kNormValues = kNormHidden / kRowThreads; // per thread and stream
static_assert(kNormHidden % kRowThreads == 0, "a thread's values per stream are whole");

// Prefill's rows, and a mix that injects nothing: qwen4_hyper_norm_row's
// threadgroup per row of kRowThreads threads; each thread holds its values of
// every stream.
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
  for (uint s = 0; s < kStreams; ++s) scale[s] = rsqrt(sums[s] / p.hidden + QWEN4_RMS_EPSILON);
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
  if (tid < kStreams) weights[row * kStreams + tid] = 2.0f * sigmoid_of(dots[tid] / p.streams);
}

// Decode's rows of an injecting mix: a threadgroup of
// QWEN4_NORM_QUARTER_THREADS threads per row,
// whose quarter q stands for qwen4_hyper_norm_row's threadgroup in stream q's
// sum and in injection weight q's dot: each of its threads sums its values of
// stream q, then runs injection dot q over every stream's values in order,
// and writes stream q's normalized values.
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
  // Simdgroup simdgroup is the quarter's simdgroup simdgroup % kSimdgroups.
  sum = simd_sum(sum);
  if (lane == 0) scratch[simdgroup] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float total = 0.0f;
    for (uint g = 0; g < kSimdgroups; ++g) total += scratch[q * kSimdgroups + g];
    scales[q] = rsqrt(total / p.hidden + QWEN4_RMS_EPSILON);
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
  weights[row * kStreams + q] = 2.0f * sigmoid_of(total / p.streams);
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
  low[index] = bfloat(silu_of(float(low[index]) / p.streams));
}

// The same values into the up projection's table alone (decode rows, whole
// lanes): a threadgroup per row, a simdgroup per 64-value span.
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
    gguf_sg::Table16::write(table + ulong(row / 8) * p.rank * 8,
                            sums + ulong(row / 8) * gguf_sg::Table16::sums_per_tile(p.rank), p.rank, g, row % 8, lane,
                            bfloat(silu_of(float(x[k]) / p.streams)), bfloat(silu_of(float(x[k + 1]) / p.streams)));
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
    sum += float(normalized[base + s * p.hidden]) * sigmoid_of(float(gate[base + s * p.hidden]));
  mixed[ulong(row) * p.hidden + d] = bfloat(sum / p.streams);
}

// The mix with the next projection's table (decode rows, whole lanes, hidden
// a multiple of the threadgroup): the threadgroup's 256 values are four
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
    sum += float(normalized[base + s * p.hidden]) * sigmoid_of(float(gate[base + s * p.hidden]));
  const bfloat value = bfloat(sum / p.streams);
  mixed[ulong(row) * p.hidden + d] = value;
  values[tid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg >= kRowThreads / 64) return;
  const uint k = sg * 64 + 2 * lane;
  gguf_sg::Table16::write(table + ulong(row / 8) * p.hidden * 8,
                          sums + ulong(row / 8) * gguf_sg::Table16::sums_per_tile(p.hidden), p.hidden,
                          group.y * (kRowThreads / 64) + sg, row % 8, lane, values[k], values[k + 1]);
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
// through a signed square root and a sigmoid, gates the value projection.
// Writes the gated values (fp32, added to the streams) and their per-stream
// normalization (bf16), which the dilated convolution reads.
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
    const float ks = rsqrt(key_sums[s] / p.hidden + QWEN4_RMS_EPSILON);
    const float qs = rsqrt(query_sums[s] / p.hidden + QWEN4_RMS_EPSILON);
    for (uint d = tid; d < p.hidden; d += kRowThreads) {
      const uint c = s * p.hidden + d;
      dots[s] += (float(k[c]) * ks * key_norm[c]) * (x[c] * qs * query_norm[c]);
    }
  }
  reduce_streams(dots, p.streams, scratch, lane, sg);
  float gate[kStreams];
  for (uint s = 0; s < p.streams; ++s) {
    const float score = dots[s] / sqrt(float(p.hidden));
    gate[s] = sigmoid_of(sign(score) * sqrt(max(abs(score), 1e-6f)));
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
    const float cs = rsqrt(gate[s] * gate[s] * value_sums[0] / p.hidden + QWEN4_RMS_EPSILON);
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
    for (uint k = 0; k < p.taps; ++k) sum += taps[ulong(c) * p.taps + k] * at(int(row) - int((p.taps - 1 - k) * p.dilation));
    const ulong i = ulong(row) * width + c;
    streams[i] += gated[i] + silu_of(sum);
  } else if (p.write_history && row - p.rows < uint(history)) {
    // Rows past the sequence write the history: row p.rows + j holds row
    // rows - history + j.
    const int j = int(row - p.rows);
    history_out[ulong(j) * width + c] = bfloat(at(int(p.rows) - history + j));
  }
}

// The PLE history after each lane's retained rows of a verify step (p.rows
// 0) or after a prefill sequence's first p.rows rows (a state captured inside
// its command): its history before, then the convolution inputs.
kernel void qwen4_ple_commit(device const bfloat *input [[buffer(0)]],
                             device const bfloat *history_in [[buffer(1)]],
                             device bfloat *history_out [[buffer(2)]],
                             device const uint *retained [[buffer(3)]],
                             constant Qwen4PleParams &p [[buffer(4)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
  const uint width = p.streams * p.hidden;
  const uint history = (p.taps - 1) * p.dilation;
  const uint c = group.y * kRowThreads + tid, j = group.x;
  if (c >= width || j >= history) return;
  const int source = int(p.rows ? p.rows : retained[0]) - int(history) + int(j);
  history_out[ulong(j) * width + c] = source >= 0 ? input[ulong(source) * width + c]
                                                  : history_in[ulong(int(history) + source) * width + c];
}

// ---- MTP head ----

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
  const float embed_scale = rsqrt(embed_sum[0] / p.hidden + QWEN4_RMS_EPSILON);
  for (uint s = 0; s < p.streams; ++s) {
    const float scale = rsqrt(sums[s] / p.hidden + QWEN4_RMS_EPSILON);
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
  streams[index] = float(projected[(ulong(s) * p.rows + row) * p.hidden + d]);
}

// One threadgroup per lane: the argmax of the lane's logits row p.row.
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
  // Ties toward the lower id.
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
  const uint end = p.last ? p.proposals : p.row + 1;
  for (uint slot = p.row; slot < end; ++slot) {
    proposals[lane_index * p.proposals + slot] = id;
    candidates[(lane_index * p.proposals + slot) * p.candidates] = id;
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

// The PLE rows of each lane's verify rows (Runtime pleRows): heads of the
// 2- to ngram-gram hashes of the row's token and those before it, cut at
// EOS or the sequence's start.
kernel void qwen4_ple_hash(device const uint *tokens [[buffer(0)]],
                           device const uint *before [[buffer(1)]],
                           device uint *indices [[buffer(2)]],
                           constant Qwen4PleHashParams &p [[buffer(3)]],
                           uint index [[thread_position_in_grid]]) {
  if (index >= p.rows) return;
  const uint lane = index / p.lane_rows, r = index % p.lane_rows;
  ulong context[4] = {tokens[index], p.eos, p.eos, p.eos};
  bool cut = false;
  for (uint back = 1; back < p.ngram; ++back) {
    const uint previous = back <= r ? tokens[index - back] : before[lane * 2 + (back - r - 1)];
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

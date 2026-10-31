#include "metal/abi/ManifoldHyperConnection.h"
#include "metal/kernels/common/activation.h"

#include <metal_stdlib>

using namespace metal;

// Manifold-constrained hyper-connections (metal/abi/ManifoldHyperConnection.h,
// DeepSeek-V4.1's), as DeepSeek's model.py computes them: a mix's weights
// (hc_split_sinkhorn), the sum of the streams into a sublayer's input
// (hc_pre) and the streams after the sublayer (hc_post); in
// dev/tools/deepseek41_reference.py hc_mixes, hc_pre and hc_post. The streams
// are fp32 [rows][MHC_STREAMS][hidden]; a sublayer reads its input and writes
// its output as bf16 rows.

constant constexpr uint kStreams = MHC_STREAMS;
constant constexpr uint kMixes = MHC_MIXES;
constant constexpr uint kThreads = MHC_THREADS;
constant constexpr uint kSimdgroups = kThreads / 32;
constant constexpr uint kMixRows = MHC_MIX_ROWS;
// A row's partial sums: its dots with each of the mix's rows, then its sum
// of squares.
constant constexpr uint kRowPartials = MHC_MIX_ROW_PARTIALS;
constant constexpr uint kChunk = MHC_MIX_CHUNK;
constant constexpr uint kComb = 2 * kStreams;
// The mixes a simdgroup of mhc_mix_partial takes.
constant constexpr uint kSimdgroupMixes = kMixes / kSimdgroups;
static_assert(kSimdgroupMixes * kSimdgroups == kMixes, "the simdgroups take the mixes evenly");
static_assert(kMixRows <= kSimdgroups, "simdgroup r sums row r's squares");

// The partial sums of a sublayer's mix: the dots of MHC_MIX_ROWS rows'
// streams with the rows of the mix's fn [MHC_MIXES][streams * hidden], and
// their sums of squares, over one chunk of MHC_MIX_CHUNK columns, into
// partials [rows][chunks][MHC_MIX_ROW_PARTIALS]. A threadgroup per group of
// rows and chunk, so that a decode's few rows read fn on many cores; each
// simdgroup sums its share of the mixes, and simdgroup r row r's squares,
// each lane four columns at a time.
kernel void mhc_mix_partial(device const float *streams [[buffer(0)]],
                            device const float *fn [[buffer(1)]],
                            device float *chunks [[buffer(2)]],
                            constant ManifoldHyperParams &p [[buffer(3)]],
                            uint2 group [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint first = group.x * kMixRows;
  if (first >= p.rows) return;
  const uint rows = min(kMixRows, p.rows - first);
  const uint width = kStreams * p.hidden, count = (width + kChunk - 1) / kChunk;
  const uint begin = group.y * kChunk, end = min(width, begin + kChunk);
  device const float *mix_fn = fn + ulong(sg) * kSimdgroupMixes * width;
  float4 dots[kMixRows][kSimdgroupMixes];
  float4 squares = 0.0f;
  for (uint r = 0; r < kMixRows; ++r)
    for (uint k = 0; k < kSimdgroupMixes; ++k) dots[r][k] = 0.0f;
  // A constant count of steps, so that every load of a lane is in flight at
  // once.
  for (uint step = 0; step < kChunk / 128; ++step) {
    const uint c = begin + step * 128 + lane * 4;
    if (c >= end) break;
    float4 x[kMixRows];
    for (uint r = 0; r < kMixRows; ++r) {
      x[r] = r < rows ? *reinterpret_cast<device const float4 *>(streams + ulong(first + r) * width + c) : 0.0f;
      if (r == sg) squares = fma(x[r], x[r], squares);
    }
    for (uint k = 0; k < kSimdgroupMixes; ++k) {
      const float4 w = *reinterpret_cast<device const float4 *>(mix_fn + ulong(k) * width + c);
      for (uint r = 0; r < kMixRows; ++r) dots[r][k] = fma(x[r], w, dots[r][k]);
    }
  }
  device float *row_chunks = chunks + (ulong(first) * count + group.y) * kRowPartials;
  for (uint r = 0; r < kMixRows; ++r)
    for (uint k = 0; k < kSimdgroupMixes; ++k) {
      const float4 d = dots[r][k];
      const float sum = simd_sum((d.x + d.y) + (d.z + d.w));
      if (lane == 0 && r < rows) row_chunks[ulong(r) * count * kRowPartials + sg * kSimdgroupMixes + k] = sum;
    }
  if (sg >= rows) return;
  const float sum = simd_sum((squares.x + squares.y) + (squares.z + squares.w));
  if (lane == 0) row_chunks[ulong(sg) * count * kRowPartials + kMixes] = sum;
}

// The lanes of a mix's simdgroup: lane s < 2 * MHC_STREAMS takes the pre or
// post weight s, lane kSquaresLane the row's sum of squares, and lane
// kCombLane + i * MHC_STREAMS + j the comb matrix's row i, column j, so that
// a quad holds a row of it and lanes MHC_STREAMS apart its columns.
constant constexpr uint kSquaresLane = kComb;
constant constexpr uint kMaxChunks = (kStreams * MHC_MAX_HIDDEN + kChunk - 1) / kChunk;
constant constexpr uint kCombLane = 32 - kStreams * kStreams;
static_assert(kStreams == 4 && kCombLane == 16, "a quad holds a comb row, lanes 4 and 8 apart its column");

// A sublayer's mix: each row's MHC_MIXES weights from its chunks' partial
// sums, added in chunk order, its dots over the RMS of all its streams, each
// weight on its own lane. A simdgroup per row; its lanes read the row's
// chunks together first.
kernel void mhc_mix_finish(device const float *chunks [[buffer(0)]],
                           device const float *scale [[buffer(1)]],
                           device const float *base [[buffer(2)]],
                           device float *mixes [[buffer(3)]],
                           constant ManifoldHyperParams &p [[buffer(4)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float partials[kMaxChunks * kRowPartials];
  if (row >= p.rows) return;
  const uint width = kStreams * p.hidden, count = (width + kChunk - 1) / kChunk;
  device const float *row_chunks = chunks + ulong(row) * count * kRowPartials;
  for (uint i = lane; i < count * kRowPartials; i += 32) partials[i] = row_chunks[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // The partial this lane sums, if any: its weight's, or the squares'.
  const bool comb = lane >= kCombLane;
  const uint k = comb ? kComb + lane - kCombLane : lane == kSquaresLane ? kMixes : lane;
  float m = 0.0f;
  if (comb || lane <= kSquaresLane)
    for (uint c = 0; c < count; ++c) m += partials[c * kRowPartials + k];
  m *= rsqrt(simd_shuffle(m, kSquaresLane) / float(width) + p.norm_epsilon);
  device float *out = mixes + ulong(row) * kMixes;
  if (lane < kStreams) out[lane] = splash_sigmoid(m * scale[0] + base[lane]) + p.hc_epsilon;
  else if (lane < kComb) out[lane] = 2.0f * splash_sigmoid(m * scale[1] + base[lane]);
  // Each comb row's softmax plus eps, then its columns and rows in turn; the
  // first iteration's rows are the softmax's. Every lane computes, so that
  // the quads and shuffles see all of them; the comb lanes write.
  float v = m * scale[2] + base[comb ? k : 0];
  v = exp(v - quad_max(v));
  v = v / quad_sum(v) + p.hc_epsilon;
  for (uint iteration = 0; iteration < p.iterations; ++iteration) {
    if (iteration > 0) v /= quad_sum(v) + p.hc_epsilon;
    float column = v + simd_shuffle_xor(v, kStreams);
    column += simd_shuffle_xor(column, 2 * kStreams);
    v /= column + p.hc_epsilon;
  }
  if (comb) out[k] = v;
}

// The streams of each row, copies of its embedding (bf16 [rows][hidden]). A
// thread per row and value.
kernel void mhc_init(device const bfloat *embedding [[buffer(0)]],
                     device float *streams [[buffer(1)]],
                     constant ManifoldHyperParams &p [[buffer(2)]],
                     uint2 group [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]]) {
  const uint row = group.x, d = group.y * kThreads + tid;
  if (row >= p.rows || d >= p.hidden) return;
  const float e = float(embedding[ulong(row) * p.hidden + d]);
  device float *x = streams + ulong(row) * kStreams * p.hidden + d;
  for (uint s = 0; s < kStreams; ++s) x[s * p.hidden] = e;
}

// Each row's streams averaged (MhcMeanParams), as bf16 into columns from
// p.offset on of rows p.stride wide. A thread per row and value.
kernel void mhc_stream_mean(device const float *streams [[buffer(0)]],
                            device bfloat *out [[buffer(1)]],
                            constant MhcMeanParams &p [[buffer(2)]],
                            uint2 group [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
  const uint row = group.x, d = group.y * kThreads + tid;
  if (row >= p.rows || d >= p.hidden) return;
  device const float *x = streams + ulong(row) * kStreams * p.hidden + d;
  float sum = 0.0f;
  for (uint s = 0; s < kStreams; ++s) sum += x[s * p.hidden];
  out[ulong(row) * p.stride + p.offset + d] = bfloat(sum / float(kStreams));
}

// A sublayer's input: the streams summed by a mix's pre weights, then RMS
// normalized and scaled by the sublayer's norm, as bf16. A threadgroup per
// row; each thread keeps its share of the row.
kernel void mhc_pre_norm(device const float *streams [[buffer(0)]],
                         device const float *mixes [[buffer(1)]],
                         device const float *norm [[buffer(2)]],
                         device bfloat *output [[buffer(3)]],
                         constant ManifoldHyperParams &p [[buffer(4)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partials[kSimdgroups];
  if (row >= p.rows) return;
  device const float *x = streams + ulong(row) * kStreams * p.hidden;
  device const float *pre = mixes + ulong(row) * kMixes;
  float values[MHC_MAX_HIDDEN / kThreads];
  float squares = 0.0f;
  for (uint d = tid, i = 0; d < p.hidden; d += kThreads, ++i) {
    float v = 0.0f;
    for (uint s = 0; s < kStreams; ++s) v = fma(pre[s], x[s * p.hidden + d], v);
    values[i] = v;
    squares = fma(v, v, squares);
  }
  squares = simd_sum(squares);
  if (lane == 0) partials[sg] = squares;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint g = 0; g < kSimdgroups; ++g) total += partials[g];
  const float inverse = rsqrt(total / float(p.hidden) + p.norm_epsilon);
  device bfloat *out = output + ulong(row) * p.hidden;
  for (uint d = tid, i = 0; d < p.hidden; d += kThreads, ++i) out[d] = bfloat(values[i] * inverse * norm[d]);
}

// The streams after a sublayer: stream j is its output scaled by the mix's
// post weight j plus each stream i before it scaled by comb[i][j]. In place:
// a thread reads its value of every stream before it writes any.
kernel void mhc_post(device float *streams [[buffer(0)]],
                     device const bfloat *branch [[buffer(1)]],
                     device const float *mixes [[buffer(2)]],
                     constant ManifoldHyperParams &p [[buffer(3)]],
                     uint2 group [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]]) {
  const uint row = group.x, d = group.y * kThreads + tid;
  if (row >= p.rows || d >= p.hidden) return;
  device float *x = streams + ulong(row) * kStreams * p.hidden + d;
  device const float *m = mixes + ulong(row) * kMixes;
  const float b = float(branch[ulong(row) * p.hidden + d]);
  float residual[kStreams];
  for (uint s = 0; s < kStreams; ++s) residual[s] = x[s * p.hidden];
  for (uint j = 0; j < kStreams; ++j) {
    float v = m[kStreams + j] * b;
    for (uint i = 0; i < kStreams; ++i) v = fma(m[kComb + i * kStreams + j], residual[i], v);
    x[j * p.hidden] = v;
  }
}

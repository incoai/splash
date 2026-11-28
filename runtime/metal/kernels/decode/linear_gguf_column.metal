// Decode projections of at most GGUF_COLUMN_ROWS live rows (ops::LinearTile::GgufColumn): a GEMV in registers on the
// column tile the MoE expert tiles of few rows run (kernels/common/gguf_column_tile.h), each lane one output column, in
// the format its tensor holds at run time. The staged tiles run a dequantize, barrier and matmul chain per 32-input
// step on padded 8-row MPP tiles, and the Apple9 register tile reads a table its producer must write, which bound a
// step of one or two live rows above DRAM: the column tile reads the live rows alone. A pipeline decodes its
// dispatch's formats alone (fused_formats, kernels/common/gguf_segments.h), which holds registers for those: against
// one pipeline of every format, Inkling-2-Mini's k (1024 x 2304) took 4-8% less time at one row in af4g64, af8g64,
// Q4_K, Q6_K and Q8_0 on a 16-core M5 Pro, its four projections in eight formats 1.4% less in all.
//
// One threadgroup: GGUF_COLUMN_SPLITS simdgroups over the same 32 columns of one of the plan's K partitions
// (ops::columnSplits, grid.y), simdgroup s summing its share [b + s P / S, b + (s + 1) P / S) of the partition's P
// groups from group b for every live row (column_dot), then simdgroup 0 adding the parts in simdgroup order, so a
// column's value depends on its K partitions alone, not on the other columns or rows. Unsplit, simdgroup 0 stores it
// as the tiles do (kernels/common/gguf_tile.h): bf16, or fp32 (the plain epilogue's logits), plus the residual, or a
// one-pass gate/up's bf16(up) * silu(bf16(gate)); split, it stores its fp32 partials and the last partition to arrive
// adds them in partition order (kernels/common/split_reduce.h) and stores the sum so.
#pragma clang fp reassociate(off)
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/gguf_column_tile.h"
#include "metal/kernels/common/gguf_segments.h"
#include "metal/kernels/common/gguf_tile.h"
#include "metal/kernels/common/split_reduce.h"

namespace gguf_column {

// A tensor's planes, bound in slots (w0, w1, meta).
struct Planes {
  device uchar *w0, *w1, *meta;
};

// A threadgroup's memory: the codebook formats' pair table and each simdgroup's parts of the live rows' sums (a
// one-pass gate/up's of both tensors).
template <bool GateUp> struct Shared {
  half2 table[kQuantPairTableEntries];
  float parts[GGUF_COLUMN_SPLITS][GateUp ? 2 : 1][GGUF_COLUMN_ROWS][32];
};

template <GgufEpilogue Ep, bool GateUp, class Out>
inline void project(device const bfloat *input, Planes planes, Planes up, device const bfloat *aux,
                    device Out *output, device coherent(device) float *partials, device atomic_uint *counters,
                    const GgufColumnParams p, uint tile, uint split, uint sg, uint lane,
                    threadgroup Shared<GateUp> &shared) {
  constexpr ushort Rows = GGUF_COLUMN_ROWS, Tensors = GateUp ? 2 : 1;
  const uint tid = sg * 32 + lane, K = p.input_size, groups = K / 32;
  // The partition's groups, then the simdgroup's share of them.
  const uint begin = split * groups / p.splits, end = (split + 1) * groups / p.splits;
  const uint first = begin + sg * (end - begin) / GGUF_COLUMN_SPLITS,
             last = begin + (sg + 1) * (end - begin) / GGUF_COLUMN_SPLITS;
  const uint column = tile * GGUF_COLUMN_COLUMNS + lane;
  float sums[Tensors][Rows];
  fused_formats()(p.format, [&](auto f) {
    typedef decltype(f) F;
    quant_pair_table<F>(shared.table, tid, GGUF_COLUMN_THREADS);
    column_dot<F, Rows>(input, p.rows, K, planes.w0, planes.w1, planes.meta, column, first, last, shared.table,
                        sums[0]);
    if constexpr (GateUp)
      column_dot<F, Rows>(input, p.rows, K, up.w0, up.w1, up.meta, column, first, last, shared.table, sums[1]);
  });
#pragma unroll
  for (ushort t = 0; t < Tensors; ++t)
#pragma unroll
    for (ushort r = 0; r < Rows; ++r) shared.parts[sg][t][r][lane] = sums[t][r];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg) return;
#pragma unroll
  for (ushort t = 0; t < Tensors; ++t)
#pragma unroll
    for (ushort r = 0; r < Rows; ++r) {
      if (r >= p.rows) break;
      float total = 0.0f;
      for (uint s = 0; s < GGUF_COLUMN_SPLITS; ++s) total += shared.parts[s][t][r][lane];
      sums[t][r] = total;
    }
  const uint destination = p.out_offset + column;
  if (p.splits > 1) {
    // Partials [split][row][column], a gate/up's the gate's columns then the up's at out_stride on; the last
    // partition to arrive adds them in split order (kernels/common/split_reduce.h).
    const uint width = Tensors * p.out_stride;
    const auto at = [&](uint s, uint t, uint r) { return (ulong(s) * p.rows + r) * width + t * p.out_stride + destination; };
#pragma unroll
    for (ushort t = 0; t < Tensors; ++t)
#pragma unroll
      for (ushort r = 0; r < Rows; ++r) {
        if (r >= p.rows) break;
        partials[at(split, t, r)] = sums[t][r];
      }
    device atomic_uint *counter = counters + p.out_offset / GGUF_COLUMN_COLUMNS + tile;
    if (!split_arrive_last_simdgroup(counter, p.splits, lane)) return;
#pragma unroll
    for (ushort t = 0; t < Tensors; ++t)
#pragma unroll
      for (ushort r = 0; r < Rows; ++r) {
        if (r >= p.rows) break;
        sums[t][r] = split_sum(sums[t][r], split, p.splits, [&](uint s) { return partials[at(s, t, r)]; });
      }
    split_release(counter, lane);
  }
#pragma unroll
  for (ushort r = 0; r < Rows; ++r) {
    if (r >= p.rows) break;
    const ulong at = ulong(r) * p.out_stride + destination;
    if constexpr (GateUp)
      output[at] = Out(float(bfloat(sums[1][r])) * splash_silu(float(bfloat(sums[0][r]))));
    else
      output[at] = gguf_epilogue<Ep, Out>(sums[0][r], aux, at);
  }
}

} // namespace gguf_column

// Grid (the tensor's columns / GGUF_COLUMN_COLUMNS, K partitions), GGUF_COLUMN_THREADS threads. Bindings: input, the
// tensor's planes (w0, w1, meta), output; then the residual (_r), or a one-pass gate/up's up planes (_gate_up: the
// first are the gate's); then the split partials and counters (any buffer when unsplit) and GgufColumnParams.
#define GGUF_COLUMN_ARGS                                                                                       \
  device const bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]],   \
      device uchar *meta [[buffer(3)]]
#define GGUF_COLUMN_SPLIT_ARGS(i)                                                                              \
  device coherent(device) float *partials [[buffer(i)]], device atomic_uint *counters [[buffer(i + 1)]]
#define GGUF_COLUMN_THREAD_ARGS                                                                                \
  uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],                    \
      uint lane [[thread_index_in_simdgroup]]
#define GGUF_COLUMN_PLAIN(Name, Out)                                                                           \
  kernel void Name(GGUF_COLUMN_ARGS, device Out *output [[buffer(4)]], GGUF_COLUMN_SPLIT_ARGS(5),              \
                   constant GgufColumnParams &p [[buffer(7)]], GGUF_COLUMN_THREAD_ARGS) {                      \
    threadgroup gguf_column::Shared<false> shared;                                                             \
    gguf_column::project<EpNone, false, Out>(input, {w0, w1, meta}, {w0, w1, meta}, input, output, partials,   \
                                             counters, p, group.x, group.y, sg, lane, shared);                 \
  }
GGUF_COLUMN_PLAIN(gguf_column_a, bfloat)
GGUF_COLUMN_PLAIN(gguf_column_a_f32, float)
kernel void gguf_column_gate_up(GGUF_COLUMN_ARGS, device bfloat *output [[buffer(4)]],
                                device uchar *up_w0 [[buffer(5)]], device uchar *up_w1 [[buffer(6)]],
                                device uchar *up_meta [[buffer(7)]], GGUF_COLUMN_SPLIT_ARGS(8),
                                constant GgufColumnParams &p [[buffer(10)]], GGUF_COLUMN_THREAD_ARGS) {
  threadgroup gguf_column::Shared<true> shared;
  gguf_column::project<EpNone, true, bfloat>(input, {w0, w1, meta}, {up_w0, up_w1, up_meta}, input, output, partials,
                                             counters, p, group.x, group.y, sg, lane, shared);
}
kernel void gguf_column_r(GGUF_COLUMN_ARGS, device bfloat *output [[buffer(4)]],
                          device const bfloat *residual [[buffer(5)]], GGUF_COLUMN_SPLIT_ARGS(6),
                          constant GgufColumnParams &p [[buffer(8)]], GGUF_COLUMN_THREAD_ARGS) {
  threadgroup gguf_column::Shared<false> shared;
  gguf_column::project<EpResidual, false, bfloat>(input, {w0, w1, meta}, {w0, w1, meta}, residual, output, partials,
                                                  counters, p, group.x, group.y, sg, lane, shared);
}
// A fused projection's segments in one dispatch (kernels/common/gguf_segments.h), as gguf_decode_fused_m<rows> runs
// them on the staged tile: the threadgroup's columns pick their segment, whose planes and format they decode as the
// segment's own dispatch does, bit for bit. Grid (the segments' columns / GGUF_COLUMN_COLUMNS, K partitions).
// Bindings: input, the segments' planes, output, the split partials and counters; then GgufColumnFusedParams.
kernel void gguf_column_fused(device const bfloat *input [[buffer(0)]], GGUF_SEGMENTS,
                              device bfloat *output [[buffer(10)]], GGUF_COLUMN_SPLIT_ARGS(11),
                              constant GgufColumnFusedParams &p [[buffer(13)]], GGUF_COLUMN_THREAD_ARGS) {
  threadgroup gguf_column::Shared<false> shared;
  GGUF_SEGMENT_OF(group.x, p.cols, GGUF_COLUMN_COLUMNS);
  const GgufColumnParams segment{p.rows, p.input_size, p.out_stride, p.offset[s], p.fmt[s], p.splits};
  gguf_column::project<EpNone, false, bfloat>(input, {w0, w1, meta}, {w0, w1, meta}, input, output, partials,
                                              counters, segment, local, group.y, sg, lane, shared);
}
#undef GGUF_COLUMN_PLAIN
#undef GGUF_COLUMN_THREAD_ARGS
#undef GGUF_COLUMN_SPLIT_ARGS
#undef GGUF_COLUMN_ARGS
#undef GGUF_SEGMENT_OF
#undef GGUF_SEGMENT_PLANE
#undef GGUF_SEGMENT_INDEX
#undef GGUF_SEGMENTS
#undef GGUF_SEGMENT

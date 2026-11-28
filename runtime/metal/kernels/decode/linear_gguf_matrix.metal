// Decode projections of 3 to 64 rows (ops::LinearTile::GgufMatrix): the column tile's grid, a threadgroup of
// GGUF_COLUMN_SPLITS simdgroups along K over GGUF_COLUMN_COLUMNS columns of one K partition (grid.y), whose
// simdgroups run matmul2d on 16-row fragments with the weights decoded into registers (kernels/common/
// gguf_matrix_tile.h), in the format its tensor holds at run time (fused_formats, kernels/common/gguf_segments.h).
// The staged tile stages each 32-input step in threadgroup memory behind a barrier; the matrix tile streams the planes
// as the column tile does and keeps every simdgroup's chain to its share of K.
//
// After K, each fi's sums meet in threadgroup memory and fold in simdgroup order, so a value depends on its K
// partitions alone, not on the other columns or rows. Unsplit, the threadgroup stores it as the tiles do (kernels/
// common/gguf_tile.h): bf16, or fp32 (the plain epilogue's logits), plus the residual, or bf16(v) * silu(gate) (_g, the
// up pass of a gate/up pair); split, it stores its fp32 partials [split][row][destination column] and the last
// partition to arrive adds them in partition order (kernels/common/split_reduce.h).
#pragma clang fp reassociate(off)
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/gguf_matrix_tile.h"
#include "metal/kernels/common/gguf_segments.h"
#include "metal/kernels/common/gguf_tile.h"
#include "metal/kernels/common/split_reduce.h"

namespace gguf_matrix {

// A tensor's planes, bound in slots (w0, w1, meta).
struct Planes {
  device uchar *w0, *w1, *meta;
};

// A threadgroup's memory: the codebook formats' pair table, each simdgroup's sums of one fragment, and the split's
// arrival. (Folding them in two rounds through half of it, the upper simdgroups' sums added into the lower ones'
// first, took 6% more time over Inkling-2-Mini's k, o and k|v at 3 to 16 rows on a 12-core M6.)
struct Shared {
  half2 table[kQuantPairTableEntries];
  float parts[GGUF_COLUMN_SPLITS][kMatrixFragmentRows][GGUF_COLUMN_COLUMNS];
  uint arrival;
};

// The outputs a thread folds of a fi: elements t and t + 256 of its 16 x 32.
constant constexpr uint kOutputs = kMatrixFragmentRows * GGUF_COLUMN_COLUMNS / GGUF_COLUMN_THREADS;

template <GgufEpilogue Ep, ushort Fragments, class Out>
inline void project(device const bfloat *input, Planes planes, device const bfloat *aux, device Out *output,
                    device coherent(device) float *partials, device atomic_uint *counters, const GgufColumnParams p,
                    uint tile, uint split, uint rows_group, uint sg, uint lane, threadgroup Shared &shared) {
  const uint tid = sg * 32 + lane, K = p.input_size, groups = K / 32;
  // The row group's rows: [row0, row0 + live) of the plan's, Fragments fragments' worth each.
  constexpr uint kGroupRows = Fragments * kMatrixFragmentRows;
  const uint row0 = rows_group * kGroupRows, live = min(p.rows - row0, kGroupRows);
  // The partition's groups, simdgroup s taking every GGUF_COLUMN_SPLITS-th from its s-th: the threadgroup's
  // simdgroups read adjacent groups at once, one stream of the planes a threadgroup rather than one a simdgroup.
  const uint begin = split * groups / p.splits, end = (split + 1) * groups / p.splits;
  const uint column0 = tile * GGUF_COLUMN_COLUMNS, destination0 = p.out_offset + column0;
  const auto at = [&](uint row, uint column) { return ulong(row) * p.out_stride + destination0 + column; };
  const auto partial = [&](uint s, uint row, uint column) {
    return (ulong(s) * p.rows + row) * p.out_stride + destination0 + column;
  };
  float own[Fragments][kOutputs];
  fused_formats()(p.format, [&](auto f) {
    typedef decltype(f) F;
    quant_pair_table<F>(shared.table, tid, GGUF_COLUMN_THREADS);
    matrix_sums<F, Fragments>(
        input + ulong(row0) * K, K, planes.w0, planes.w1, planes.meta, column0, begin + sg, end, GGUF_COLUMN_SPLITS,
        shared.table, ushort(lane), [&](ushort fi, auto elements) __attribute__((always_inline)) {
          // The fragment's live rows, which alone meet: a plan's rows past them are padding.
          const uint first = uint(fi) * kMatrixFragmentRows,
                     rows = live > first ? min(live - first, uint(kMatrixFragmentRows)) : 0;
          threadgroup_barrier(mem_flags::mem_threadgroup);
          elements([&](uint row, uint column, float v) __attribute__((always_inline)) {
            if (row < rows) shared.parts[sg][row][column] = v;
          });
          threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
          for (uint o = 0; o < kOutputs; ++o) {
            const uint index = tid + o * GGUF_COLUMN_THREADS, row = index / GGUF_COLUMN_COLUMNS,
                       column = index % GGUF_COLUMN_COLUMNS, r = row0 + first + row;
            if (row >= rows) continue;
            float total = 0.0f;
            for (uint s = 0; s < GGUF_COLUMN_SPLITS; ++s) total += shared.parts[s][row][column];
            own[fi][o] = total;
            if (p.splits > 1) partials[partial(split, r, column)] = total;
            else output[at(r, column)] = gguf_epilogue<Ep, Out>(total, aux, at(r, column));
          }
        });
  });
  if (p.splits == 1) return;
  // One counter per 32 destination columns of each row group.
  device atomic_uint *counter =
      counters + rows_group * (p.out_stride / GGUF_COLUMN_COLUMNS) + destination0 / GGUF_COLUMN_COLUMNS;
  if (!split_arrive_last(counter, p.splits, tid, &shared.arrival)) return;
#pragma unroll
  for (ushort fi = 0; fi < Fragments; ++fi)
#pragma unroll
    for (uint o = 0; o < kOutputs; ++o) {
      const uint index = tid + o * GGUF_COLUMN_THREADS, row = index / GGUF_COLUMN_COLUMNS,
                 column = index % GGUF_COLUMN_COLUMNS, r = row0 + fi * kMatrixFragmentRows + row;
      if (r >= row0 + live) continue;
      const float total =
          split_sum(own[fi][o], split, p.splits, [&](uint s) { return partials[partial(s, r, column)]; });
      output[at(r, column)] = gguf_epilogue<Ep, Out>(total, aux, at(r, column));
    }
  split_release(counter, tid);
}

} // namespace gguf_matrix

// Grid (the tensor's columns / GGUF_COLUMN_COLUMNS, K partitions, row groups), GGUF_COLUMN_THREADS threads; a row
// group holds up to 16 F rows (F fragments: _m16, _m32, _m48, _m64), and the input buffer holds as many a row group
// (ops::matrixRowGroups: a decode step of 65 to 128 rows runs two). Bindings: input, the
// tensor's planes (w0, w1, meta), output; then the residual (_r) or the gate scratch (_g); then the split partials and
// counters (any buffer when unsplit) and GgufColumnParams.
#define GGUF_MATRIX_ARGS                                                                                       \
  device const bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]],   \
      device uchar *meta [[buffer(3)]]
#define GGUF_MATRIX_SPLIT_ARGS(i)                                                                              \
  device coherent(device) float *partials [[buffer(i)]], device atomic_uint *counters [[buffer(i + 1)]]
#define GGUF_MATRIX_THREAD_ARGS                                                                                \
  uint3 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],                    \
      uint lane [[thread_index_in_simdgroup]]
#define GGUF_MATRIX_PLAIN(Name, Out, F)                                                                        \
  kernel void Name(GGUF_MATRIX_ARGS, device Out *output [[buffer(4)]], GGUF_MATRIX_SPLIT_ARGS(5),              \
                   constant GgufColumnParams &p [[buffer(7)]], GGUF_MATRIX_THREAD_ARGS) {                      \
    threadgroup gguf_matrix::Shared shared;                                                                    \
    gguf_matrix::project<EpNone, F, Out>(input, {w0, w1, meta}, input, output, partials, counters, p, group.x, \
                                         group.y, group.z, sg, lane, shared);                                  \
  }
#define GGUF_MATRIX_AUX(Name, Ep, F)                                                                           \
  kernel void Name(GGUF_MATRIX_ARGS, device bfloat *output [[buffer(4)]], device const bfloat *aux [[buffer(5)]], \
                   GGUF_MATRIX_SPLIT_ARGS(6), constant GgufColumnParams &p [[buffer(8)]], GGUF_MATRIX_THREAD_ARGS) { \
    threadgroup gguf_matrix::Shared shared;                                                                    \
    gguf_matrix::project<Ep, F, bfloat>(input, {w0, w1, meta}, aux, output, partials, counters, p, group.x,    \
                                        group.y, group.z, sg, lane, shared);                                   \
  }
// A fused projection's segments in one dispatch (kernels/common/gguf_segments.h), as gguf_column_fused runs them:
// the threadgroup's columns pick their segment, whose planes and format they decode as the segment's own dispatch
// does. Grid (the segments' columns / GGUF_COLUMN_COLUMNS, K partitions). Bindings: input, the segments' planes,
// output, the split partials and counters; then GgufColumnFusedParams.
#define GGUF_MATRIX_FUSED(Name, F)                                                                             \
  kernel void Name(device const bfloat *input [[buffer(0)]], GGUF_SEGMENTS, device bfloat *output [[buffer(10)]], \
                   GGUF_MATRIX_SPLIT_ARGS(11), constant GgufColumnFusedParams &p [[buffer(13)]],               \
                   GGUF_MATRIX_THREAD_ARGS) {                                                                  \
    threadgroup gguf_matrix::Shared shared;                                                                    \
    GGUF_SEGMENT_OF(group.x, p.cols, GGUF_COLUMN_COLUMNS);                                                     \
    const GgufColumnParams segment{p.rows, p.input_size, p.out_stride, p.offset[s], p.fmt[s], p.splits};      \
    gguf_matrix::project<EpNone, F, bfloat>(input, {w0, w1, meta}, input, output, partials, counters, segment, \
                                            local, group.y, group.z, sg, lane, shared);                        \
  }
#define GGUF_MATRIX_ROWS(F, R)                                                                                 \
  GGUF_MATRIX_PLAIN(gguf_matrix_a_m##R, bfloat, F)                                                             \
  GGUF_MATRIX_PLAIN(gguf_matrix_a_f32_m##R, float, F)                                                          \
  GGUF_MATRIX_AUX(gguf_matrix_r_m##R, EpResidual, F)                                                           \
  GGUF_MATRIX_AUX(gguf_matrix_g_m##R, EpUpWithGate, F)                                                         \
  GGUF_MATRIX_FUSED(gguf_matrix_fused_m##R, F)
GGUF_MATRIX_ROWS(1, 16)
GGUF_MATRIX_ROWS(2, 32)
GGUF_MATRIX_ROWS(3, 48)
GGUF_MATRIX_ROWS(4, 64)
#undef GGUF_MATRIX_ROWS
#undef GGUF_MATRIX_FUSED
#undef GGUF_MATRIX_AUX
#undef GGUF_MATRIX_PLAIN
#undef GGUF_MATRIX_THREAD_ARGS
#undef GGUF_MATRIX_SPLIT_ARGS
#undef GGUF_MATRIX_ARGS
#undef GGUF_SEGMENT_OF
#undef GGUF_SEGMENT_PLANE
#undef GGUF_SEGMENT_INDEX
#undef GGUF_SEGMENTS
#undef GGUF_SEGMENT

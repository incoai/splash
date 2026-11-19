// Decode projections of at most GGUF_COLUMN_ROWS live rows (ops::LinearTile::GgufColumn): a GEMV in registers on the
// column tile the MoE expert tiles of few rows run (kernels/common/gguf_column_tile.h), each lane one output column, in
// the format its tensor holds at run time. The staged tiles run a dequantize, barrier and matmul chain per 32-input
// step on padded 8-row MPP tiles, and the Apple9 register tile reads a table its producer must write, which bound a
// step of one or two live rows above DRAM: the column tile reads the live rows alone. One pipeline of each kernel
// decodes every format: created cold, each took 0.31-0.64 s on the M5 Max, a one-format staged tile 0.06-0.09 s.
//
// One threadgroup: GGUF_COLUMN_SPLITS simdgroups over the same 32 columns, simdgroup s summing groups
// [s G / S, (s + 1) G / S) of K (G = K / 32) for every live row (column_dot), then simdgroup 0 adding the parts in
// simdgroup order, so a column's value depends neither on the grid nor on the other columns, and storing it as the
// tiles do (kernels/common/gguf_tile.h): bf16, or fp32 (the plain epilogue's logits), plus the residual, or a one-pass
// gate/up's bf16(up) * silu(bf16(gate)).
#pragma clang fp reassociate(off)
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/gguf_column_tile.h"
#include "metal/kernels/common/gguf_tile.h"

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
                    device Out *output, constant GgufColumnParams &p, uint tile, uint sg, uint lane,
                    threadgroup Shared<GateUp> &shared) {
  const uint tid = sg * 32 + lane, K = p.input_size, groups = K / 32;
  const uint first = sg * groups / GGUF_COLUMN_SPLITS, last = (sg + 1) * groups / GGUF_COLUMN_SPLITS;
  const uint column = tile * 32 + lane;
  float sums[GGUF_COLUMN_ROWS], up_sums[GGUF_COLUMN_ROWS];
  quant_format_switch(p.format, [&](auto f) {
    typedef decltype(f) F;
    quant_pair_table<F>(shared.table, tid, GGUF_COLUMN_THREADS);
    column_dot<F, GGUF_COLUMN_ROWS>(input, p.rows, K, planes.w0, planes.w1, planes.meta, column, first, last,
                                    shared.table, sums);
    if constexpr (GateUp)
      column_dot<F, GGUF_COLUMN_ROWS>(input, p.rows, K, up.w0, up.w1, up.meta, column, first, last, shared.table,
                                      up_sums);
  });
  for (uint r = 0; r < GGUF_COLUMN_ROWS; ++r) {
    shared.parts[sg][0][r][lane] = sums[r];
    if constexpr (GateUp) shared.parts[sg][1][r][lane] = up_sums[r];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg) return;
  for (uint r = 0; r < p.rows; ++r) {
    float total = 0.0f, up_total = 0.0f;
    for (uint s = 0; s < GGUF_COLUMN_SPLITS; ++s) {
      total += shared.parts[s][0][r][lane];
      if constexpr (GateUp) up_total += shared.parts[s][1][r][lane];
    }
    const ulong at = ulong(r) * p.out_stride + p.out_offset + column;
    if constexpr (GateUp)
      output[at] = Out(float(bfloat(up_total)) * splash_silu(float(bfloat(total))));
    else
      output[at] = gguf_epilogue<Ep, Out>(total, aux, at);
  }
}

} // namespace gguf_column

// Grid (the segment's columns / 32), GGUF_COLUMN_THREADS threads. Bindings: input, the tensor's planes (w0, w1,
// meta), output; then the residual (_r), or a one-pass gate/up's up planes (_gate_up: the first are the gate's); then
// GgufColumnParams.
#define GGUF_COLUMN_ARGS                                                                                       \
  device const bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]],   \
      device uchar *meta [[buffer(3)]]
#define GGUF_COLUMN_THREAD_ARGS                                                                                \
  uint tile [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],                      \
      uint lane [[thread_index_in_simdgroup]]
#define GGUF_COLUMN_PLAIN(Name, Out)                                                                           \
  kernel void Name(GGUF_COLUMN_ARGS, device Out *output [[buffer(4)]],                                         \
                   constant GgufColumnParams &p [[buffer(5)]], GGUF_COLUMN_THREAD_ARGS) {                      \
    threadgroup gguf_column::Shared<false> shared;                                                             \
    gguf_column::project<EpNone, false, Out>(input, {w0, w1, meta}, {w0, w1, meta}, input, output, p, tile, sg, \
                                             lane, shared);                                                    \
  }
GGUF_COLUMN_PLAIN(gguf_column_a, bfloat)
GGUF_COLUMN_PLAIN(gguf_column_a_f32, float)
kernel void gguf_column_gate_up(GGUF_COLUMN_ARGS, device bfloat *output [[buffer(4)]],
                                device uchar *up_w0 [[buffer(5)]], device uchar *up_w1 [[buffer(6)]],
                                device uchar *up_meta [[buffer(7)]], constant GgufColumnParams &p [[buffer(8)]],
                                GGUF_COLUMN_THREAD_ARGS) {
  threadgroup gguf_column::Shared<true> shared;
  gguf_column::project<EpNone, true, bfloat>(input, {w0, w1, meta}, {up_w0, up_w1, up_meta}, input, output, p, tile,
                                             sg, lane, shared);
}
kernel void gguf_column_r(GGUF_COLUMN_ARGS, device bfloat *output [[buffer(4)]],
                          device const bfloat *residual [[buffer(5)]], constant GgufColumnParams &p [[buffer(6)]],
                          GGUF_COLUMN_THREAD_ARGS) {
  threadgroup gguf_column::Shared<false> shared;
  gguf_column::project<EpResidual, false, bfloat>(input, {w0, w1, meta}, {w0, w1, meta}, residual, output, p, tile,
                                                  sg, lane, shared);
}
#undef GGUF_COLUMN_PLAIN
#undef GGUF_COLUMN_THREAD_ARGS
#undef GGUF_COLUMN_ARGS

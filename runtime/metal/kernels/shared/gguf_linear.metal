// GGUF quantized GEMMs (K-quants, i-quants, Q8_0) for Apple9 and Apple10.
// Decode weights with FP32 group coefficients, then round once to the half tile,
// matching llama.cpp Metal dequantize.h / mul_mm.metal (MIT notice in THIRD_PARTY_NOTICES). Keep activations BF16.
// Weight planes and meta in the MDGG0001 layout (metal/abi/QuantFormat.h), decoded by kernels/common/quant_formats.h.
// Activations bf16 [rows][K]; weights staged as fp16 in threadgroup memory; fp32 accumulation; bf16 output (fp32 from
// the _a_f32 instances, the logits). Rotated PQ2_0 decode also runs in int8 activations (gguf_decode_int8_*, below):
// int8 codes, int32 accumulation per 128-input block.
// Keep the source order of float operations, which Metal's default fast math lets the compiler reassociate. Set
// before the includes, so it also holds for the shared format and reduction code compiled here.
#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_staged_tile.h"
#include "metal/kernels/common/split_reduce.h"

// A tile's sums over every K partition, handed to store(row, column, sum). One partition stores its own; more publish
// fp32 partials [split][Rows][destination column] (kernels/common/split_reduce.h) and the last arriving partition adds
// them in split order. `column0` is the simdgroup's first destination column, `counter` its threadgroup's (one per 64
// destination columns: the segments of a projection never share one).
template <ushort Rows, class Acc, class Store>
inline void gguf_store_sums(thread Acc &acc, uint splits, uint split, device coherent(device) float *partials,
                            device atomic_uint *counter, uint stride, uint column0, uint thread_index,
                            threadgroup uint *arrival, Store store) {
  if (splits == 1) { gguf_elements(acc, store); return; }
  const auto at = [&](uint s, uint row, uint column) { return (ulong(s) * Rows + row) * stride + column0 + column; };
  gguf_elements(acc, [&](uint row, uint column, float v) { partials[at(split, row, column)] = v; });
  if (!split_arrive_last(counter, splits, thread_index, arrival)) return;
  gguf_elements(acc, [&](uint row, uint column, float v) {
    store(row, column, split_sum(v, split, splits, [&](uint s) { return partials[at(s, row, column)]; }));
  });
  split_release(counter, thread_index);
}
// A gate/up pair's sums over every K partition, stored as bf16(up) * silu(bf16(gate)). One partition stores its own;
// more publish fp32 partials [split][Rows][2 * stride], the gate's columns then the up's at stride on, and the last
// arriving partition adds each in split order. `counter` is the simdgroup's (one per 32 destination columns).
template <ushort Rows, class Acc>
inline void gguf_store_gate_up(thread Acc &gate, thread Acc &up, uint splits, uint split,
                               device coherent(device) float *partials, device atomic_uint *counter, uint stride,
                               uint column0, uint lane, threadgroup uint *arrival, device bfloat *output) {
  const auto value = [](float g, float u) { return bfloat(float(bfloat(u)) * splash_silu(float(bfloat(g)))); };
  const auto out = [&](uint row, uint column) { return ulong(row) * stride + column0 + column; };
  if (splits == 1) {
#pragma unroll
    for (ushort i = 0; i < gate.get_capacity(); ++i) {
      if (!gate.is_valid_element(i)) continue;
      const auto index = gate.get_multidimensional_index(i);
      output[out(index[1], index[0])] = value(gate[i], up[i]);
    }
    return;
  }
  const auto at = [&](uint s, uint row, uint column) {
    return (ulong(s) * Rows + row) * (2 * stride) + column0 + column;
  };
#pragma unroll
  for (ushort i = 0; i < gate.get_capacity(); ++i) {
    if (!gate.is_valid_element(i)) continue;
    const auto index = gate.get_multidimensional_index(i);
    partials[at(split, index[1], index[0])] = gate[i];
    partials[at(split, index[1], index[0]) + stride] = up[i];
  }
  if (!split_arrive_last(counter, splits, lane, arrival)) return;
#pragma unroll
  for (ushort i = 0; i < gate.get_capacity(); ++i) {
    if (!gate.is_valid_element(i)) continue;
    const auto index = gate.get_multidimensional_index(i);
    const uint row = index[1], column = index[0];
    const float g = split_sum(float(gate[i]), split, splits, [&](uint s) { return partials[at(s, row, column)]; });
    const float u = split_sum(float(up[i]), split, splits,
                              [&](uint s) { return partials[at(s, row, column) + stride]; });
    output[out(row, column)] = value(g, u);
  }
  split_release(counter, lane);
}

// ---------------- prefill tiles: a shared B stage (TileN x KS, all threads dequantize), each simdgroup owns RowsPerSG
// rows. `rows` counts the chunk's rows from the tile's first: simdgroups past them (the last tile of a chunk that is
// not a multiple of the tile) skip their matmuls and stores, so a chunk costs its rows rounded up to RowsPerSG rather
// than to the tile (a 33-row chunk: 1.7x faster than a 128-row tile). Full tiles run this loop too; a branch-free copy
// for them is no faster.
template <class F, ushort RowsPerSG, ushort Simdgroups, ushort TileN, ushort KS, GgufEpilogue Ep = EpNone>
inline void gguf_prefill_tile(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta, device bfloat *output,
                    uint input_size, uint output_origin, uint rows, threadgroup half *stage, threadgroup half2 *tl,
                    uint simd_lane, uint simd_group, uint out_stride, uint out_offset, device bfloat *aux = nullptr) {
  const bool owns_rows = simd_group * RowsPerSG < rows;   // uniform per simdgroup
  device bfloat *rows_input = input + ulong(simd_group) * RowsPerSG * input_size;
  auto acc = staged_accumulator<RowsPerSG, TileN, KS>(rows_input, input_size, stage);
  gguf_zero(acc);
  gguf_staged_steps<F, RowsPerSG, TileN, KS, Simdgroups * 32>(rows_input, w0, w1, meta, input_size, output_origin, stage, tl,
                                                              simd_group * 32 + simd_lane, 0, input_size / KS, 0, owns_rows,
                                                              acc);
  if (!owns_rows) return;
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto index = acc.get_multidimensional_index(i);
    const ulong o = (ulong(simd_group) * RowsPerSG + index[1]) * out_stride + out_offset + output_origin + index[0];
    output[o] = gguf_epilogue<Ep>(acc[i], aux, o);
  }
}

// The two stages of a prefill tile.
constant constexpr uint kPrefillStages = 2 * GGUF_TILE_COLUMNS * GGUF_PREFILL_STEP;

// ---------------- decode dispatches over (64-column tiles, K partitions): two simdgroups of 32 columns per
// threadgroup, every request lane in its tile, `splits` partitions of K (grid.y; kernels/common/split_reduce.h).
// MPP computes 16-row fragments, so a tile holds 8, 16 or 32 rows, or 64 for a prefill chunk of 33 to 64 rows on
// Apple10 and on Apple9 GPUs of 10 cores or fewer (these kernels and the fused ones): a 3-lane step
// runs the 32-row tile over the storage of four lanes, and a 1-lane step may run the 16-row tile over two
// (LinearPlan::storageRows); the padding lane's rows are computed and discarded. Rows are independent, so every active
// row is the bits of any other tile height (gguf-projection checks it); the 32-row tile at three lanes costs what it
// costs at four, 3-15% less than a 16-row plus an 8-row matmul per stage.
// Gate/up runs as a gate pass (a) into the gate scratch and an up pass (g) whose epilogue applies silu(gate) to the bf16
// up value, as the Apple9 register kernels do. The destination's type Out is bf16, or fp32 for the plain epilogue's
// logits (a_f32, ops::Projection::destination). Each column tile walks its partition from the step staged_first_step
// gives it (kernels/common/gguf_staged_tile.h): in lockstep, or spread when p.spread is set.
template <class F, ushort Rows, GgufEpilogue Ep, class Out>
inline void gguf_decode_tile(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta, device Out *output,
                             device coherent(device) float *partials, device atomic_uint *counters, device bfloat *aux,
                             constant GgufDecodeParams &p, uint2 group, uint simd_lane, uint simd_group,
                             threadgroup half *stage, threadgroup half2 *tl, threadgroup uint *arrival) {
  const uint per = p.input_size / GGUF_STAGED_STEP / p.splits, sb = group.y * per, se = sb + per,
             origin = group.x * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS, column0 = p.out_offset + origin;
  threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;
  auto acc = staged_accumulator<Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, p.input_size, my);
  gguf_zero(acc);
  staged_accumulate<F, Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, w0, w1, meta, p.input_size, origin, my, tl, simd_lane, sb, se,
                                          staged_first_step(group.x, sb, se, p.spread), acc);
  gguf_store_sums<Rows>(acc, p.splits, group.y, partials, counters + p.out_offset / GGUF_TILE_COLUMNS + group.x, p.out_stride, column0,
                        simd_group * 32 + simd_lane, arrival, [&](uint row, uint column, float v) {
    const ulong o = ulong(row) * p.out_stride + column0 + column;
    output[o] = gguf_epilogue<Ep, Out>(v, aux, o);
  });
}
// The staged decode kernels: grid (column tiles, K partitions), two simdgroups.
template <class F, ushort Rows, GgufEpilogue Ep, class Out>
kernel void gguf_decode(device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]],
                        device uchar *meta [[buffer(3)]], device Out *output [[buffer(4)]],
                        device coherent(device) float *partials [[buffer(5)]], device atomic_uint *counters [[buffer(6)]],
                        device bfloat *aux [[buffer(7)]], constant GgufDecodeParams &p [[buffer(8)]],
                        uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                        uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half2 tl[F::Kind == QuantCodebook ? kQuantPairTableEntries : 1];
  quant_pair_table<F>(tl, simd_group * 32 + simd_lane, GGUF_STAGED_THREADS);
  threadgroup half stage[kStagedStages];
  threadgroup uint arrival;
  gguf_decode_tile<F, Rows, Ep>(input, w0, w1, meta, output, partials, counters, aux, p, group, simd_lane, simd_group,
                                stage, tl, &arrival);
}
template <class Out>
using GgufDecodeKernel = void(device bfloat *, device uchar *, device uchar *, device uchar *, device Out *,
                              device coherent(device) float *, device atomic_uint *, device bfloat *,
                              constant GgufDecodeParams &, uint2, uint, uint);
#define GGUF_DECODE(F, f, R, ep, Ep, Out) \
  template [[host_name("gguf_decode_" #f "_m" #R "_" #ep)]] kernel GgufDecodeKernel<Out> gguf_decode<F, R, Ep, Out>;
#define GGUF_DECODE_ROWS(F, f, ep, Ep, Out) \
  GGUF_DECODE(F, f, 8, ep, Ep, Out) GGUF_DECODE(F, f, 16, ep, Ep, Out) GGUF_DECODE(F, f, 32, ep, Ep, Out)
// Prefill chunks of 33 to 64 rows run the 64-row tile (ops::Linear), whose bf16 epilogues prefill takes.
#define GGUF_DECODE_ROWS64(F, f, ep, Ep) GGUF_DECODE_ROWS(F, f, ep, Ep, bfloat) GGUF_DECODE(F, f, 64, ep, Ep, bfloat)
#define GGUF_DECODE_FORMAT(F, f)                                                                                  \
  GGUF_DECODE_ROWS64(F, f, a, EpNone) GGUF_DECODE_ROWS(F, f, a_f32, EpNone, float)                                \
  GGUF_DECODE_ROWS64(F, f, r, EpResidual) GGUF_DECODE_ROWS64(F, f, g, EpUpWithGate)
QUANT_FORMATS(GGUF_DECODE_FORMAT)
#undef GGUF_DECODE_FORMAT
#undef GGUF_DECODE_ROWS64
#undef GGUF_DECODE_ROWS
#undef GGUF_DECODE

// Gate/up in one dispatch, where gate and up share their format (ops::Linear, LinearConfig::oneGateUpPass): grid
// (32-column tiles, K partitions), threadgroups of one simdgroup that stages the gate's and the up's same 32 columns
// (gguf_staged_pair_steps), so the per-step latency of a tile's chain serves both tensors. Each walks from the step its
// 64-column tile walks from in the two passes, so at the same K split gate and up are the bits of the gate pass's and
// of the up pass's sums, and the output is bf16(up) * silu(bf16(gate)) as the up pass writes it (gguf_store_gate_up).
template <class F, ushort Rows>
kernel void gguf_decode_gate_up(device bfloat *input [[buffer(0)]], device uchar *gw0 [[buffer(1)]],
                                device uchar *gw1 [[buffer(2)]], device uchar *gmeta [[buffer(3)]],
                                device uchar *uw0 [[buffer(4)]], device uchar *uw1 [[buffer(5)]],
                                device uchar *umeta [[buffer(6)]], device bfloat *output [[buffer(7)]],
                                device coherent(device) float *partials [[buffer(8)]],
                                device atomic_uint *counters [[buffer(9)]], constant GgufDecodeParams &p [[buffer(10)]],
                                uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half2 tl[F::Kind == QuantCodebook ? kQuantPairTableEntries : 1];
  quant_pair_table<F>(tl, lane, GGUF_GATE_UP_THREADS);
  threadgroup half stage[2 * GGUF_STAGED_COLUMNS * GGUF_STAGED_STEP];
  threadgroup uint arrival;
  const uint per = p.input_size / GGUF_STAGED_STEP / p.splits, sb = group.y * per, se = sb + per,
             origin = group.x * GGUF_STAGED_COLUMNS, column0 = p.out_offset + origin;
  auto gate = staged_accumulator<Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, p.input_size, stage);
  auto up = staged_accumulator<Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, p.input_size, stage);
  gguf_zero(gate);
  gguf_zero(up);
  gguf_staged_pair_steps<F, Rows>(input, gw0, gw1, gmeta, uw0, uw1, umeta, p.input_size, origin, stage, tl, lane, sb,
                                  se, staged_first_step(group.x / 2, sb, se, p.spread), gate, up);
  gguf_store_gate_up<Rows>(gate, up, p.splits, group.y, partials,
                           counters + p.out_offset / GGUF_STAGED_COLUMNS + group.x, p.out_stride, column0, lane,
                           &arrival, output);
}
using GgufDecodeGateUpKernel = void(device bfloat *, device uchar *, device uchar *, device uchar *, device uchar *,
                                    device uchar *, device uchar *, device bfloat *, device coherent(device) float *,
                                    device atomic_uint *, constant GgufDecodeParams &, uint2, uint);
#define GGUF_DECODE_GATE_UP(F, f, R) \
  template [[host_name("gguf_decode_" #f "_m" #R "_gate_up")]] kernel GgufDecodeGateUpKernel gguf_decode_gate_up<F, R>;
#define GGUF_DECODE_GATE_UP_FORMAT(F, f) GGUF_DECODE_GATE_UP(F, f, 8) GGUF_DECODE_GATE_UP(F, f, 16) GGUF_DECODE_GATE_UP(F, f, 32)
QUANT_FORMATS(GGUF_DECODE_GATE_UP_FORMAT)
#undef GGUF_DECODE_GATE_UP_FORMAT
#undef GGUF_DECODE_GATE_UP

// Fused projections (qkv|z|ab, q|k|v): up to three column segments of any formats in one dispatch, so the small
// segments do not run as dispatches of their own. The threadgroup's tile picks its segment, and the segment's format
// picks the decode; every segment takes the same K splits, and its tiles walk as its projection alone would, from the
// step staged_first_step gives the tile's index within the segment. The fused and prefill kernels stay plain kernels:
// as template instantiations (weak_odr) the compiler infers fewer parameter attributes and inlines differently.
#define GGUF_SEGMENT(i, w0, w1, m) device uchar *w0 [[buffer(i)]], device uchar *w1 [[buffer(i + 1)]], device uchar *m [[buffer(i + 2)]]
#define GGUF_SEGMENTS GGUF_SEGMENT(1, w0a, w1a, ma), GGUF_SEGMENT(4, w0b, w1b, mb), GGUF_SEGMENT(7, w0c, w1c, mc)
// The segment s of column tile `tile` among the segments' tiles (cols, 0 past the last) and the tile's index `local`
// within it; GGUF_SEGMENT_PLANE the segment's `plane` of x##a, x##b and x##c; GGUF_SEGMENT_OF both, with the
// segment's planes w0, w1 and meta.
#define GGUF_SEGMENT_INDEX(tile, cols)                                                                             \
  const uint t0 = (cols)[0] / GGUF_TILE_COLUMNS, t1 = t0 + (cols)[1] / GGUF_TILE_COLUMNS;                          \
  const uint s = (tile) < t0 ? 0 : (tile) < t1 ? 1 : 2, local = (tile) - (s == 0 ? 0 : s == 1 ? t0 : t1)
#define GGUF_SEGMENT_PLANE(plane, x) device uchar *plane = s == 0 ? x##a : s == 1 ? x##b : x##c
#define GGUF_SEGMENT_OF(tile, cols)                                                                                \
  GGUF_SEGMENT_INDEX(tile, cols);                                                                                  \
  GGUF_SEGMENT_PLANE(w0, w0);                                                                                      \
  GGUF_SEGMENT_PLANE(w1, w1);                                                                                      \
  GGUF_SEGMENT_PLANE(meta, m)
// A fused pipeline decodes its dispatch's segment formats alone, function constants 0-2 (ops::Linear: each once, the
// last repeated past them): the GPU compiler builds a staged loop for each format a pipeline decodes, and a cold
// 16-row fused pipeline took 18 s to build with all 44 on the 16-core M5 Pro, 0.5 s with three.
// fused_formats runs body(F()) with the format type of `format`, one body per distinct format; an id the pipeline
// does not decode decodes as its first, so every id runs one body.
constant uint kFusedFormat0 [[function_constant(0)]];
constant uint kFusedFormat1 [[function_constant(1)]];
constant uint kFusedFormat2 [[function_constant(2)]];
struct fused_formats {
  template <class Body> void operator()(uint format, Body body) const {
    if (kFusedFormat2 != kFusedFormat1 && format == kFusedFormat2) quant_format_switch(kFusedFormat2, body);
    else if (kFusedFormat1 != kFusedFormat0 && format == kFusedFormat1) quant_format_switch(kFusedFormat1, body);
    else quant_format_switch(kFusedFormat0, body);
  }
};
#define GGUF_DECODE_FUSED(R)                                                                                       \
  kernel void gguf_decode_fused_m##R(device bfloat *input [[buffer(0)]], GGUF_SEGMENTS,                          \
                                     device bfloat *output [[buffer(10)]],                                        \
                                     device coherent(device) float *partials [[buffer(11)]],                      \
                                     device atomic_uint *counters [[buffer(12)]],                                 \
                                     constant GgufDecodeFusedParams &p [[buffer(13)]],                            \
                                     uint2 group [[threadgroup_position_in_grid]],                                \
                                     uint simd_lane [[thread_index_in_simdgroup]],                                \
                                     uint simd_group [[simdgroup_index_in_threadgroup]]) {                        \
    threadgroup half stage[kStagedStages]; threadgroup half2 tl[kQuantPairTableEntries]; threadgroup uint arrival; \
    GGUF_SEGMENT_OF(group.x, p.cols);                                                                             \
    const uint per = p.input_size / GGUF_STAGED_STEP / p.splits;                                                  \
    const uint sb = group.y * per, se = sb + per;                                                                 \
    const uint origin = local * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS, column0 = p.offset[s] + origin; \
    threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;                                            \
    auto acc = staged_accumulator<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, p.input_size, my);                                         \
    gguf_zero(acc);                                                                                               \
    staged_accumulate_any<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(p.fmt[s], input, w0, w1, meta, p.input_size, origin, my, tl, \
                                            simd_group * 32 + simd_lane, simd_lane, sb, se,                       \
                                            staged_first_step(local, sb, se, p.spread), acc, fused_formats());    \
    gguf_store_sums<R>(acc, p.splits, group.y, partials, counters + p.offset[s] / GGUF_TILE_COLUMNS + local,       \
                       p.out_stride, column0, simd_group * 32 + simd_lane, &arrival,                              \
                       [&](uint row, uint column, float v) { output[ulong(row) * p.out_stride + column0 + column] = bfloat(v); }); \
  }
// The 64-row instance runs fused prefill chunks of 33 to 64 rows (ops::Linear).
GGUF_DECODE_FUSED(8) GGUF_DECODE_FUSED(16) GGUF_DECODE_FUSED(32) GGUF_DECODE_FUSED(64)
#undef GGUF_DECODE_FUSED

// The prefill kernels: grid (GGUF_PREFILL_ROWS-row tiles of the chunk, column tiles), four simdgroups; the residual
// and gate kernels run one tensor and read aux at buffer 5.
#define GGUF_PREFILL_BUFFERS                                                                                       \
  device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]],             \
      device uchar *meta [[buffer(3)]], device bfloat *output [[buffer(4)]]
#define GGUF_PREFILL_THREAD                                                                                        \
  uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],                      \
      uint simd_group [[simdgroup_index_in_threadgroup]]
#define GGUF_PREFILL_TABLES(F)                                                                                     \
  threadgroup half2 tl[F::Kind == QuantCodebook ? kQuantPairTableEntries : 1];                                     \
  quant_pair_table<F>(tl, simd_group * 32 + simd_lane, GGUF_PREFILL_THREADS);                                      \
  threadgroup half stage[kPrefillStages]
// The plain kernel takes up to three segments of its format (GgufPrefillSegmentsParams), so the segments of a fused
// projection that share a format run as one dispatch, grid.y over their column tiles in segment order.
#define GGUF_PREFILL(F, f)                                                                                         \
  kernel void gguf_prefill_##f##_a(device bfloat *input [[buffer(0)]], GGUF_SEGMENTS,                            \
                                   device bfloat *output [[buffer(10)]],                                          \
                                   constant GgufPrefillSegmentsParams &p [[buffer(11)]], GGUF_PREFILL_THREAD) {    \
    GGUF_PREFILL_TABLES(F);                                                                                        \
    GGUF_SEGMENT_OF(group.y, p.cols);                                                                              \
    const uint first = group.x * GGUF_PREFILL_ROWS, rows = p.rows > first ? p.rows - first : 0;                    \
    gguf_prefill_tile<F, GGUF_PREFILL_SIMDGROUP_ROWS, GGUF_PREFILL_SIMDGROUPS, GGUF_TILE_COLUMNS, GGUF_PREFILL_STEP>(input + ulong(first) * p.input_size, w0, w1, meta,                        \
                                         output + ulong(first) * p.out_stride, p.input_size,                       \
                                         local * GGUF_TILE_COLUMNS, rows, stage, tl, simd_lane, simd_group,        \
                                         p.out_stride, p.offset[s]);                                               \
  }
#define GGUF_PREFILL_EPILOGUE(F, f, ep, Ep)                                                                        \
  kernel void gguf_prefill_##f##_##ep(GGUF_PREFILL_BUFFERS, device bfloat *aux [[buffer(5)]],                    \
                                      constant GgufPrefillParams &p [[buffer(6)]], GGUF_PREFILL_THREAD) {          \
    GGUF_PREFILL_TABLES(F);                                                                                        \
    const uint first = group.x * GGUF_PREFILL_ROWS, rows = p.rows > first ? p.rows - first : 0;                    \
    gguf_prefill_tile<F, GGUF_PREFILL_SIMDGROUP_ROWS, GGUF_PREFILL_SIMDGROUPS, GGUF_TILE_COLUMNS, GGUF_PREFILL_STEP, Ep>(input + ulong(first) * p.input_size, w0, w1, meta,                    \
                                             output + ulong(first) * p.out_stride, p.input_size,                   \
                                             group.y * GGUF_TILE_COLUMNS, rows, stage, tl, simd_lane, simd_group,  \
                                             p.out_stride, p.out_offset, aux + ulong(first) * p.out_stride);       \
  }
#define GGUF_PREFILL_FORMAT(F, f) \
  GGUF_PREFILL(F, f) GGUF_PREFILL_EPILOGUE(F, f, r, EpResidual) GGUF_PREFILL_EPILOGUE(F, f, g, EpUpWithGate)
QUANT_FORMATS(GGUF_PREFILL_FORMAT)
#undef GGUF_PREFILL_FORMAT

// The residual kernels over a view of the leading inputs of wider weight rows, for the Neural Engine FFN split's down
// projection, one per format. Each runs the tile of its residual kernel on the planes advanced past the groups and
// meta units the view leaves unread in the rows of the plane tiles before its column tile, which the tile then
// addresses as rows of input_size inputs.
#define GGUF_PREFILL_LEADING_INPUTS(F, f)                                                                          \
  kernel void gguf_prefill_##f##_r_leading_inputs(GGUF_PREFILL_BUFFERS, device bfloat *aux [[buffer(5)]],          \
                                                  constant GgufPrefillLeadingParams &leading [[buffer(6)]],        \
                                                  GGUF_PREFILL_THREAD) {                                           \
    GGUF_PREFILL_TABLES(F);                                                                                        \
    constant GgufPrefillParams &p = leading.prefill;                                                               \
    const uint first = group.x * GGUF_PREFILL_ROWS, rows = p.rows > first ? p.rows - first : 0;                    \
    const ulong before = ulong(group.y * GGUF_TILE_COLUMNS / QUANT_TILE_ROWS) * QUANT_TILE_ROWS;                   \
    const uint groups = p.input_size / 32, plane_groups = leading.plane_input_size / 32;                           \
    const ulong unread = before * (plane_groups - groups),                                                         \
                unread_units = before * (plane_groups / F::MetaGroups - groups / F::MetaGroups);                   \
    gguf_prefill_tile<F, GGUF_PREFILL_SIMDGROUP_ROWS, GGUF_PREFILL_SIMDGROUPS, GGUF_TILE_COLUMNS, GGUF_PREFILL_STEP, \
                      EpResidual>(input + ulong(first) * p.input_size, w0 + unread * F::P0, w1 + unread * F::P1,   \
                                  meta + unread_units * F::MetaBytes, output + ulong(first) * p.out_stride,        \
                                  p.input_size, group.y * GGUF_TILE_COLUMNS, rows, stage, tl, simd_lane,           \
                                  simd_group, p.out_stride, p.out_offset, aux + ulong(first) * p.out_stride);      \
  }
QUANT_FORMATS(GGUF_PREFILL_LEADING_INPUTS)
#undef GGUF_PREFILL_LEADING_INPUTS
#undef GGUF_PREFILL_EPILOGUE
#undef GGUF_PREFILL
#undef GGUF_PREFILL_TABLES
#undef GGUF_PREFILL_THREAD
#undef GGUF_PREFILL_BUFFERS

// ---------------- int8-activation decode of rotated projections (ops::Linear, LinearConfig::decodeInt8) in a format
// whose codes are int8 exactly (quant_decode_int8_format: PQ2_0). gguf_rotate_int8 writes the rotated input as int8
// rows with one scale s per row and rotation block (metal/abi/Gguf.h). The tiles are the staged decode tiles on int8
// stages: their columns (StagedColumn), walks (gguf_staged_tile_walk, gguf_staged_walk), K splits, split reductions
// (gguf_store_sums, gguf_store_gate_up), pass modes, epilogues, parameters and bindings; per step each lane stages its
// column's 32 two-bit codes q as int8 q - 1 in {-1, 0, 1, 2} in the order of the int8 rows
// (gguf_decode_int8_position), and one int8 matmul2d accumulates in int32 over a block's steps (|acc| <= 2 * 128 * 127
// for PQ2_0's 128 inputs, exact); the block's last step folds it into the fp32 sums as acc * d * s, so a tile walks
// from its block's first step. The tiles hold 16 or 32 rows (one lane on the 16-row tile), the 8-row int8 matmul
// taking longer than the 16-row one.

// Stages `step` of `column` as int8: its 32 codes q as q - 1 ((q + 0x7F) ^ 0x80 a byte: bytes 0x7F..0x82 before the
// xor, no carries) at the lane's 32 bytes of `stage`, and at a block's last step the block's d at d[lane], before the
// walk's load moves the column's meta to the next block.
template <class F>
inline void gguf_int8_stage(thread const StagedColumn<F> &column, uint step, threadgroup char *stage,
                            threadgroup float *d, uint lane) {
  static_assert(F::Kind == QuantLinear && F::Zero == 1 && F::P0 == 8 && F::P1 == 0,
                "the int8 stage takes two-bit codes of value d (q - 1)");
  threadgroup uint *dst = (threadgroup uint *)(stage + lane * GGUF_STAGED_STEP);
#pragma unroll
  for (ushort e = 0; e < 4; ++e) {
    dst[e] = (((column.packed.a.x >> (2 * e)) & 0x03030303u) + 0x7F7F7F7Fu) ^ 0x80808080u;
    dst[4 + e] = (((column.packed.a.y >> (2 * e)) & 0x03030303u) + 0x7F7F7F7Fu) ^ 0x80808080u;
  }
  if (step % F::MetaGroups == F::MetaGroups - 1) d[lane] = F::coef(column.hdr, 0).s.x;
}
// The step an int8 tile starts at: its staged_first_step, back to the first step of its block, so that each block's
// steps run in order and fold once.
template <class F> inline uint gguf_int8_first_step(uint tile, uint step_begin, uint step_end, uint spread) {
  return step_begin + (staged_first_step(tile, step_begin, step_end, spread) - step_begin) / F::MetaGroups * F::MetaGroups;
}
// A block's int32 sums into the fp32 sums, sums += acc * d[column] * s[row][block], acc reset. The two cooperative
// tensors are destinations of one matmul2d, element i the same (row, column) in both.
template <class Acc, class Sums>
inline void gguf_int8_fold(thread Acc &acc, thread Sums &sums, threadgroup const float *d, device const float *s,
                           uint blocks, uint block) {
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (acc.is_valid_element(i)) {
      const auto index = acc.get_multidimensional_index(i);
      sums[i] = fma(float(acc[i]), d[index[0]] * s[uint(index[1]) * blocks + block], float(sums[i]));
    }
    acc[i] = 0;
  }
}
// The int8 rows of a Rows-row tile (input [Rows][K] int8, then their scales) and a stage of 32 columns, as matmul2d
// operands.
template <ushort Rows> inline auto gguf_int8_rows(device char *input, uint input_size) {
  return tensor<device int8_t, dextents<int, 2>, tensor_inline>((device int8_t *)input, dextents<int, 2>{int(input_size), Rows},
                                                               array<int, 2>{1, int(input_size)});
}
inline auto gguf_int8_operand(threadgroup char *stage) {
  return tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline>(
             (threadgroup int8_t *)stage, dextents<int, 2>{GGUF_STAGED_STEP, GGUF_STAGED_COLUMNS}, array<int, 2>{1, GGUF_STAGED_STEP})
      .template slice<GGUF_STAGED_STEP, GGUF_STAGED_COLUMNS>(0, 0);
}
template <ushort Rows>
using GgufInt8Matmul = matmul2d<matmul2d_descriptor(Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP, false, true, false,
                                                    matmul2d_descriptor::mode::multiply_accumulate),
                                execution_simdgroups<1>>;
// The destination of a Rows-row int8 tile, of int (a block's sums) or float (the folded sums), not initialized, as
// staged_accumulator's.
template <ushort Rows, class T> inline auto gguf_int8_destination(device char *input, uint input_size, threadgroup char *stage) {
  GgufInt8Matmul<Rows> op;
  auto a0 = gguf_int8_rows<Rows>(input, input_size).template slice<GGUF_STAGED_STEP, Rows>(0, 0);
  auto b0 = gguf_int8_operand(stage);
  return op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), T>();
}

// The int8 decode tile loop of one simdgroup without the store, as staged_accumulate's: its GGUF_STAGED_COLUMNS
// columns from `origin` over steps [step_begin, step_end) of K from block-aligned step_first, folded into `sums`;
// `stages` holds two stages, d a block's scale of each column.
template <class F, ushort Rows, class Sums>
inline void gguf_int8_accumulate(device char *input, device uchar *w0, device uchar *w1, device uchar *meta,
                                 uint input_size, uint origin, threadgroup char *stages, threadgroup float *d,
                                 uint lane, uint step_begin, uint step_end, uint step_first, thread Sums &sums) {
  constexpr ushort Stage = GGUF_STAGED_COLUMNS * GGUF_STAGED_STEP;
  auto rows = gguf_int8_rows<Rows>(input, input_size);
  GgufInt8Matmul<Rows> op;
  auto b0 = gguf_int8_operand(stages), b1 = gguf_int8_operand(stages + Stage);
  auto acc = gguf_int8_destination<Rows, int>(input, input_size, stages);
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
  device const float *scales = (device const float *)(input + ulong(Rows) * input_size);
  StagedColumn<F> column;
  column.begin(w0, w1, meta, input_size, origin, lane, step_first, step_begin < step_end);
  const auto run_step = [&](uint step, uint next, bool more, uint buffer) __attribute__((always_inline)) {
    gguf_int8_stage(column, step, stages + buffer * Stage, d, lane);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (more) column.load(step, next);
    auto a = rows.template slice<GGUF_STAGED_STEP, Rows>(step * GGUF_STAGED_STEP, 0);
    if (buffer) op.run(a, b1, acc); else op.run(a, b0, acc);
    if (step % F::MetaGroups == F::MetaGroups - 1)
      gguf_int8_fold(acc, sums, d, scales, input_size / GGUF_ROTATION_BLOCK, step * GGUF_STAGED_STEP / GGUF_ROTATION_BLOCK);
  };
  gguf_staged_tile_walk(step_begin, step_end, step_first, run_step);
}
// The int8 decode kernels, as gguf_decode's: grid (column tiles, K partitions), two simdgroups. Their stores take the
// parameters they need by value: gguf_store_sums is not inlined into them, and under shader validation a load through
// the constant parameters' address that its closure held read as 0 in 1 run of 30 (a fused plan's p.out_stride).
template <class F, ushort Rows, GgufEpilogue Ep, class Out>
kernel void gguf_decode_int8(device char *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]],
                             device uchar *meta [[buffer(3)]], device Out *output [[buffer(4)]],
                             device coherent(device) float *partials [[buffer(5)]], device atomic_uint *counters [[buffer(6)]],
                             device bfloat *aux [[buffer(7)]], constant GgufDecodeParams &p [[buffer(8)]],
                             uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                             uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup char stages[GGUF_TILE_COLUMNS / GGUF_STAGED_COLUMNS][2 * GGUF_STAGED_COLUMNS * GGUF_STAGED_STEP];
  threadgroup float d[GGUF_TILE_COLUMNS / GGUF_STAGED_COLUMNS][GGUF_STAGED_COLUMNS];
  threadgroup uint arrival;
  const uint per = p.input_size / GGUF_STAGED_STEP / p.splits, sb = group.y * per, se = sb + per,
             origin = group.x * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS, column0 = p.out_offset + origin;
  auto sums = gguf_int8_destination<Rows, float>(input, p.input_size, stages[simd_group]);
  gguf_zero(sums);
  gguf_int8_accumulate<F, Rows>(input, w0, w1, meta, p.input_size, origin, stages[simd_group], d[simd_group], simd_lane,
                                sb, se, gguf_int8_first_step<F>(group.x, sb, se, p.spread), sums);
  const uint stride = p.out_stride;
  gguf_store_sums<Rows>(sums, p.splits, group.y, partials, counters + p.out_offset / GGUF_TILE_COLUMNS + group.x,
                        stride, column0, simd_group * 32 + simd_lane, &arrival, [=](uint row, uint column, float v) {
    const ulong o = ulong(row) * stride + column0 + column;
    output[o] = gguf_epilogue<Ep, Out>(v, aux, o);
  });
}
// A fused projection's segments in one dispatch, as gguf_decode_fused_m<rows> runs them, each of the kernel's format:
// the threadgroup's tile picks its segment, whose tiles walk as its projection alone would.
template <class F, ushort Rows>
kernel void gguf_decode_int8_fused(device char *input [[buffer(0)]], GGUF_SEGMENTS, device bfloat *output [[buffer(10)]],
                                   device coherent(device) float *partials [[buffer(11)]],
                                   device atomic_uint *counters [[buffer(12)]],
                                   constant GgufDecodeFusedParams &p [[buffer(13)]],
                                   uint2 group [[threadgroup_position_in_grid]],
                                   uint simd_lane [[thread_index_in_simdgroup]],
                                   uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup char stages[GGUF_TILE_COLUMNS / GGUF_STAGED_COLUMNS][2 * GGUF_STAGED_COLUMNS * GGUF_STAGED_STEP];
  threadgroup float d[GGUF_TILE_COLUMNS / GGUF_STAGED_COLUMNS][GGUF_STAGED_COLUMNS];
  threadgroup uint arrival;
  GGUF_SEGMENT_OF(group.x, p.cols);
  const uint per = p.input_size / GGUF_STAGED_STEP / p.splits, sb = group.y * per, se = sb + per,
             origin = local * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS, column0 = p.offset[s] + origin;
  auto sums = gguf_int8_destination<Rows, float>(input, p.input_size, stages[simd_group]);
  gguf_zero(sums);
  gguf_int8_accumulate<F, Rows>(input, w0, w1, meta, p.input_size, origin, stages[simd_group], d[simd_group], simd_lane,
                                sb, se, gguf_int8_first_step<F>(local, sb, se, p.spread), sums);
  const uint stride = p.out_stride;
  gguf_store_sums<Rows>(sums, p.splits, group.y, partials, counters + p.offset[s] / GGUF_TILE_COLUMNS + local,
                        stride, column0, simd_group * 32 + simd_lane, &arrival,
                        [=](uint row, uint column, float v) { output[ulong(row) * stride + column0 + column] = bfloat(v); });
}
// A gate/up pair in one pass, as gguf_decode_gate_up runs the bf16 pair: grid (32-column tiles, K partitions), one
// simdgroup staging both tensors' same 32 columns on stages of their own, two each; the output is
// bf16(up) * silu(bf16(gate)).
template <class F, ushort Rows>
kernel void gguf_decode_int8_gate_up(device char *input [[buffer(0)]], device uchar *gw0 [[buffer(1)]],
                                     device uchar *gw1 [[buffer(2)]], device uchar *gmeta [[buffer(3)]],
                                     device uchar *uw0 [[buffer(4)]], device uchar *uw1 [[buffer(5)]],
                                     device uchar *umeta [[buffer(6)]], device bfloat *output [[buffer(7)]],
                                     device coherent(device) float *partials [[buffer(8)]],
                                     device atomic_uint *counters [[buffer(9)]], constant GgufDecodeParams &p [[buffer(10)]],
                                     uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr ushort Stage = GGUF_STAGED_COLUMNS * GGUF_STAGED_STEP;
  threadgroup char stages[2][2 * Stage];   // [gate, up][stage]
  threadgroup float d[2][GGUF_STAGED_COLUMNS];
  threadgroup uint arrival;
  const uint per = p.input_size / GGUF_STAGED_STEP / p.splits, sb = group.y * per, se = sb + per,
             origin = group.x * GGUF_STAGED_COLUMNS, column0 = p.out_offset + origin;
  const uint first = gguf_int8_first_step<F>(group.x / 2, sb, se, p.spread);
  auto rows = gguf_int8_rows<Rows>(input, p.input_size);
  GgufInt8Matmul<Rows> op;
  auto g0 = gguf_int8_operand(stages[0]), g1 = gguf_int8_operand(stages[0] + Stage);
  auto u0 = gguf_int8_operand(stages[1]), u1 = gguf_int8_operand(stages[1] + Stage);
  auto gate = gguf_int8_destination<Rows, int>(input, p.input_size, stages[0]);
  auto up = gguf_int8_destination<Rows, int>(input, p.input_size, stages[0]);
  auto gateSums = gguf_int8_destination<Rows, float>(input, p.input_size, stages[0]);
  auto upSums = gguf_int8_destination<Rows, float>(input, p.input_size, stages[0]);
  gguf_zero(gateSums);
  gguf_zero(upSums);
#pragma unroll
  for (ushort i = 0; i < gate.get_capacity(); ++i) {
    gate[i] = 0;
    up[i] = 0;
  }
  device const float *scales = (device const float *)(input + ulong(Rows) * p.input_size);
  StagedColumn<F> g, u;
  g.begin(gw0, gw1, gmeta, p.input_size, origin, lane, first, sb < se);
  u.begin(uw0, uw1, umeta, p.input_size, origin, lane, first, sb < se);
  const auto run_step = [&](uint step, uint next, bool more, uint buffer) __attribute__((always_inline)) {
    gguf_int8_stage(g, step, stages[0] + buffer * Stage, d[0], lane);
    gguf_int8_stage(u, step, stages[1] + buffer * Stage, d[1], lane);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (more) {
      g.load(step, next);
      u.load(step, next);
    }
    auto a = rows.template slice<GGUF_STAGED_STEP, Rows>(step * GGUF_STAGED_STEP, 0);
    if (buffer) {
      op.run(a, g1, gate);
      op.run(a, u1, up);
    } else {
      op.run(a, g0, gate);
      op.run(a, u0, up);
    }
    if (step % F::MetaGroups == F::MetaGroups - 1) {
      const uint blocks = p.input_size / GGUF_ROTATION_BLOCK, block = step * GGUF_STAGED_STEP / GGUF_ROTATION_BLOCK;
      gguf_int8_fold(gate, gateSums, d[0], scales, blocks, block);
      gguf_int8_fold(up, upSums, d[1], scales, blocks, block);
    }
  };
  gguf_staged_walk(sb, se, first, run_step);
  gguf_store_gate_up<Rows>(gateSums, upSums, p.splits, group.y, partials,
                           counters + p.out_offset / GGUF_STAGED_COLUMNS + group.x, p.out_stride, column0, lane,
                           &arrival, output);
}
template <class Out>
using GgufDecodeInt8Kernel = void(device char *, device uchar *, device uchar *, device uchar *, device Out *,
                                  device coherent(device) float *, device atomic_uint *, device bfloat *,
                                  constant GgufDecodeParams &, uint2, uint, uint);
using GgufDecodeInt8GateUpKernel = void(device char *, device uchar *, device uchar *, device uchar *, device uchar *,
                                        device uchar *, device uchar *, device bfloat *, device coherent(device) float *,
                                        device atomic_uint *, constant GgufDecodeParams &, uint2, uint);
using GgufDecodeInt8FusedKernel = void(device char *, device uchar *, device uchar *, device uchar *, device uchar *,
                                       device uchar *, device uchar *, device uchar *, device uchar *, device uchar *,
                                       device bfloat *, device coherent(device) float *, device atomic_uint *,
                                       constant GgufDecodeFusedParams &, uint2, uint, uint);
#define GGUF_DECODE_INT8(F, f, R, ep, Ep, Out)                                                                     \
  template [[host_name("gguf_decode_int8_" #f "_m" #R "_" #ep)]] kernel GgufDecodeInt8Kernel<Out>                  \
      gguf_decode_int8<F, R, Ep, Out>;
#define GGUF_DECODE_INT8_ROWS(F, f, R)                                                                             \
  GGUF_DECODE_INT8(F, f, R, a, EpNone, bfloat) GGUF_DECODE_INT8(F, f, R, r, EpResidual, bfloat)                    \
  GGUF_DECODE_INT8(F, f, R, g, EpUpWithGate, bfloat) GGUF_DECODE_INT8(F, f, R, a_f32, EpNone, float)               \
  template [[host_name("gguf_decode_int8_" #f "_m" #R "_gate_up")]] kernel GgufDecodeInt8GateUpKernel              \
      gguf_decode_int8_gate_up<F, R>;                                                                              \
  template [[host_name("gguf_decode_int8_" #f "_fused_m" #R)]] kernel GgufDecodeInt8FusedKernel                    \
      gguf_decode_int8_fused<F, R>;
// The formats the int8 tiles take (quant_decode_int8_format).
#define GGUF_DECODE_INT8_FORMAT(F, f) GGUF_DECODE_INT8_ROWS(F, f, 16) GGUF_DECODE_INT8_ROWS(F, f, 32)
GGUF_DECODE_INT8_FORMAT(FmtPQ20, pq20)
#undef GGUF_DECODE_INT8_FORMAT
#undef GGUF_DECODE_INT8_ROWS
#undef GGUF_DECODE_INT8
#undef GGUF_SEGMENT_OF
#undef GGUF_SEGMENT_PLANE
#undef GGUF_SEGMENT_INDEX
#undef GGUF_SEGMENTS
#undef GGUF_SEGMENT

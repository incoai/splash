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
                                            staged_first_step(local, sb, se, p.spread), acc);                     \
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
#undef GGUF_SEGMENTS
#undef GGUF_SEGMENT

// ---------------- int8-activation decode of rotated PQ2_0 projections (ops::Linear, LinearConfig::decodeInt8).
// gguf_rotate_int8 writes the rotated input as int8 rows with one scale s per row and rotation block
// (metal/abi/Gguf.h). Per step each lane stages its column's 32 codes as int8 q - 1 in {-1, 0, 1}
// (((w >> 2e) & 0x03030303) + 0x7F7F7F7F, then ^ 0x80808080: bytes 0x7F..0x82, no carries) in the order of the int8
// rows (gguf_decode_int8_position), and one int8 matmul2d accumulates in int32 over a PQ2_0 block's four steps
// (|acc| <= 128 * 127, exact); the block's last step folds it into the fp32 sums as acc * d * s. K splits, spread walks
// (from a block's first step), the split reduction (gguf_store_sums, gguf_store_gate_up), pass modes and epilogues are
// the staged tile's; the tiles hold 16 or 32 rows (one lane on the 16-row tile), the 8-row int8 matmul taking longer
// than the 16-row one.

// One simdgroup's 32 columns of a PQ2_0 tensor: lane l stages column origin + l, holding the payload of the step it
// stages next and the d of that step's block.
struct Int8Column {
  device uchar *w0, *meta;
  uint2 packed;
  float d;
  void begin(device uchar *p0, device uchar *m, uint input_size, uint origin, uint lane, uint first) thread {
    const uint groups = input_size / 32, units = groups / FmtPQ20::MetaGroups;
    const uint plane_tile = origin / QUANT_TILE_ROWS, plane_row = origin % QUANT_TILE_ROWS + lane;
    w0 = p0 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * FmtPQ20::P0;
    meta = m + (ulong(plane_tile) * units * QUANT_TILE_ROWS + plane_row) * FmtPQ20::MetaBytes;
    load(first);
  }
  // `step`'s payload and, at a block's first step, the block's d.
  void load(uint step) thread {
    packed = *((device const uint2 *)(w0 + ulong(step) * QUANT_TILE_ROWS * FmtPQ20::P0));
    if (step % FmtPQ20::MetaGroups == 0)
      d = float(as_type<half>(
          *((device const ushort *)(meta + ulong(step / FmtPQ20::MetaGroups) * QUANT_TILE_ROWS * FmtPQ20::MetaBytes))));
  }
  // `step` into the lane's 32 bytes of `stage`; at a block's last step the block's d into scales[lane] for the fold.
  void stage(uint step, threadgroup char *stage, threadgroup float *scales, uint lane) thread {
    threadgroup uint *dst = (threadgroup uint *)(stage + lane * GGUF_STAGED_STEP);
#pragma unroll
    for (ushort e = 0; e < 4; ++e) {
      dst[e] = (((packed.x >> (2 * e)) & 0x03030303u) + 0x7F7F7F7Fu) ^ 0x80808080u;
      dst[4 + e] = (((packed.y >> (2 * e)) & 0x03030303u) + 0x7F7F7F7Fu) ^ 0x80808080u;
    }
    if (step % FmtPQ20::MetaGroups == FmtPQ20::MetaGroups - 1) scales[lane] = d;
  }
};
// A block's int32 sums into the fp32 sums, sums += acc * d[column] * s[row][block], acc reset. The two cooperative
// tensors are destinations of one matmul2d, element i the same (row, column) in both.
template <class Acc, class Sums>
inline void int8_fold(thread Acc &acc, thread Sums &sums, threadgroup const float *d, device const float *s,
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
// The walk over steps [sb, se) of K from block-aligned `first`, wrapping: per step stage(step, buffer), a barrier,
// load(next) where a step follows, run(step, buffer); the stages alternate per step walked.
template <class Stage, class Load, class Run>
inline void int8_walk(uint sb, uint se, uint first, Stage stage, Load load, Run run) {
  const uint per = se - sb;
  for (uint walked = 0; walked < per; ++walked) {
    const uint step = first + walked < se ? first + walked : first + walked - per;
    stage(step, walked & 1);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (walked + 1 < per) load(step + 1 < se ? step + 1 : sb);
    run(step, walked & 1);
  }
}
// The int8 operands of a Rows-row tile, declared in the kernel (a macro: the cooperative tensors take the types of op
// and a0): Cols, KS, the int8 rows `at`, the matmul's `descriptor` and `op`, and a0, its first input slice.
#define INT8_OPERANDS(Rows, input, K)                                                                             \
  constexpr ushort Cols = GGUF_STAGED_COLUMNS, KS = GGUF_STAGED_STEP;                                              \
  auto at = tensor<device int8_t, dextents<int, 2>, tensor_inline>((device int8_t *)(input),                       \
                                                                  dextents<int, 2>{int(K), Rows}, array<int, 2>{1, int(K)}); \
  constexpr auto descriptor =                                                                                      \
      matmul2d_descriptor(Rows, Cols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);     \
  matmul2d<descriptor, execution_simdgroups<1>> op;                                                                \
  auto a0 = at.template slice<KS, Rows>(0, 0)
// The int8 stage of 32 columns at `stage` as a matmul operand, of INT8_OPERANDS' KS and Cols (a macro: an inline
// function changes the kernels' AIR, unmeasured).
#define INT8_STAGE(stage)                                                                                         \
  tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline>((threadgroup int8_t *)(stage), dextents<int, 2>{KS, Cols}, \
                                                             array<int, 2>{1, KS})                                 \
      .template slice<KS, Cols>(0, 0)
template <ushort Rows, GgufEpilogue Ep, class Out>
kernel void gguf_decode_int8(device uchar *input [[buffer(0)]], device uchar *w0a [[buffer(1)]],
                             device uchar *ma [[buffer(2)]], device uchar *w0b [[buffer(3)]],
                             device uchar *mb [[buffer(4)]], device uchar *w0c [[buffer(5)]],
                             device uchar *mc [[buffer(6)]], device Out *output [[buffer(7)]],
                             device coherent(device) float *partials [[buffer(8)]],
                             device atomic_uint *counters [[buffer(9)]], device bfloat *aux [[buffer(10)]],
                             constant GgufDecodeInt8Params &p [[buffer(11)]],
                             uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                             uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup char stages[GGUF_TILE_COLUMNS / GGUF_STAGED_COLUMNS][2][GGUF_STAGED_COLUMNS * GGUF_STAGED_STEP];
  threadgroup float scales[GGUF_TILE_COLUMNS / GGUF_STAGED_COLUMNS][GGUF_STAGED_COLUMNS];
  threadgroup uint arrival;
  const uint K = p.input_size;
  GGUF_SEGMENT_INDEX(group.x, p.cols);
  GGUF_SEGMENT_PLANE(w0, w0);
  GGUF_SEGMENT_PLANE(meta, m);
  const uint origin = local * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS, column0 = p.offset[s] + origin;
  INT8_OPERANDS(Rows, input, K);
  auto b0 = INT8_STAGE(stages[simd_group][0]), b1 = INT8_STAGE(stages[simd_group][1]);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), int>();
  auto sums = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
  gguf_zero(sums);
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
  device const float *rowScales = (device const float *)(input + ulong(Rows) * K);
  const uint per = K / KS / p.splits, sb = group.y * per, se = sb + per;
  const uint first = sb + ((staged_first_step(local, sb, se, p.spread) - sb) & ~(FmtPQ20::MetaGroups - 1));
  Int8Column column;
  column.begin(w0, meta, K, origin, lane, first);
  threadgroup float *d = scales[simd_group];
  int8_walk(
      sb, se, first,
      [&](uint step, uint buffer) __attribute__((always_inline)) { column.stage(step, stages[simd_group][buffer], d, lane); },
      [&](uint next) __attribute__((always_inline)) { column.load(next); },
      [&](uint step, uint buffer) __attribute__((always_inline)) {
        const auto a = at.template slice<KS, Rows>(step * KS, 0);
        if (buffer) op.run(a, b1, acc); else op.run(a, b0, acc);
        if (step % FmtPQ20::MetaGroups == FmtPQ20::MetaGroups - 1)
          int8_fold(acc, sums, d, rowScales, K / GGUF_ROTATION_BLOCK, step * KS / GGUF_ROTATION_BLOCK);
      });
  gguf_store_sums<Rows>(sums, p.splits, group.y, partials, counters + p.offset[s] / GGUF_TILE_COLUMNS + local,
                        p.out_stride, column0, simd_group * 32 + lane, &arrival, [&](uint row, uint col, float v) {
    const ulong o = ulong(row) * p.out_stride + column0 + col;
    output[o] = gguf_epilogue<Ep, Out>(v, aux, o);
  });
}
// A gate/up pair of PQ2_0 tensors in one pass, as gguf_decode_gate_up runs the bf16 pair: grid (32-column tiles, K
// partitions), one simdgroup staging both tensors' same 32 columns; the output is bf16(up) * silu(bf16(gate)).
template <ushort Rows>
kernel void gguf_decode_int8_gate_up(device uchar *input [[buffer(0)]], device uchar *gw0 [[buffer(1)]],
                                     device uchar *gmeta [[buffer(2)]], device uchar *uw0 [[buffer(3)]],
                                     device uchar *umeta [[buffer(4)]], device bfloat *output [[buffer(5)]],
                                     device coherent(device) float *partials [[buffer(6)]],
                                     device atomic_uint *counters [[buffer(7)]],
                                     constant GgufDecodeInt8Params &p [[buffer(8)]],
                                     uint2 group [[threadgroup_position_in_grid]],
                                     uint lane [[thread_index_in_simdgroup]]) {
  threadgroup char stages[2][2][GGUF_STAGED_COLUMNS * GGUF_STAGED_STEP];   // [gate, up][buffer]
  threadgroup float scales[2][GGUF_STAGED_COLUMNS];
  threadgroup uint arrival;
  const uint K = p.input_size, origin = group.x * GGUF_STAGED_COLUMNS, column0 = p.offset[0] + origin;
  INT8_OPERANDS(Rows, input, K);
  auto g0 = INT8_STAGE(stages[0][0]), g1 = INT8_STAGE(stages[0][1]);
  auto u0 = INT8_STAGE(stages[1][0]), u1 = INT8_STAGE(stages[1][1]);
  auto gate = op.template get_destination_cooperative_tensor<decltype(a0), decltype(g0), int>();
  auto up = op.template get_destination_cooperative_tensor<decltype(a0), decltype(g0), int>();
  auto gateSums = op.template get_destination_cooperative_tensor<decltype(a0), decltype(g0), float>();
  auto upSums = op.template get_destination_cooperative_tensor<decltype(a0), decltype(g0), float>();
  gguf_zero(gateSums);
  gguf_zero(upSums);
#pragma unroll
  for (ushort i = 0; i < gate.get_capacity(); ++i) {
    gate[i] = 0;
    up[i] = 0;
  }
  device const float *rowScales = (device const float *)(input + ulong(Rows) * K);
  const uint per = K / KS / p.splits, sb = group.y * per, se = sb + per;
  const uint first = sb + ((staged_first_step(group.x / 2, sb, se, p.spread) - sb) & ~(FmtPQ20::MetaGroups - 1));
  Int8Column g, u;
  g.begin(gw0, gmeta, K, origin, lane, first);
  u.begin(uw0, umeta, K, origin, lane, first);
  int8_walk(
      sb, se, first,
      [&](uint step, uint buffer) __attribute__((always_inline)) {
        g.stage(step, stages[0][buffer], scales[0], lane);
        u.stage(step, stages[1][buffer], scales[1], lane);
      },
      [&](uint next) __attribute__((always_inline)) { g.load(next); u.load(next); },
      [&](uint step, uint buffer) __attribute__((always_inline)) {
        const auto a = at.template slice<KS, Rows>(step * KS, 0);
        if (buffer) {
          op.run(a, g1, gate);
          op.run(a, u1, up);
        } else {
          op.run(a, g0, gate);
          op.run(a, u0, up);
        }
        if (step % FmtPQ20::MetaGroups == FmtPQ20::MetaGroups - 1) {
          const uint blocks = K / GGUF_ROTATION_BLOCK, block = step * KS / GGUF_ROTATION_BLOCK;
          int8_fold(gate, gateSums, scales[0], rowScales, blocks, block);
          int8_fold(up, upSums, scales[1], rowScales, blocks, block);
        }
      });
  gguf_store_gate_up<Rows>(gateSums, upSums, p.splits, group.y, partials,
                           counters + p.offset[0] / GGUF_STAGED_COLUMNS + group.x, p.out_stride, column0, lane,
                           &arrival, output);
}
#undef INT8_STAGE
#undef INT8_OPERANDS
template <class Out>
using GgufDecodeInt8Kernel = void(device uchar *, device uchar *, device uchar *, device uchar *, device uchar *,
                                  device uchar *, device uchar *, device Out *, device coherent(device) float *,
                                  device atomic_uint *, device bfloat *, constant GgufDecodeInt8Params &, uint2, uint,
                                  uint);
using GgufDecodeInt8GateUpKernel = void(device uchar *, device uchar *, device uchar *, device uchar *, device uchar *,
                                        device bfloat *, device coherent(device) float *, device atomic_uint *,
                                        constant GgufDecodeInt8Params &, uint2, uint);
#define GGUF_DECODE_INT8(R, ep, Ep, Out)                                                                           \
  template [[host_name("gguf_decode_int8_pq20_m" #R "_" #ep)]] kernel GgufDecodeInt8Kernel<Out>                    \
      gguf_decode_int8<R, Ep, Out>;
#define GGUF_DECODE_INT8_ROWS(R)                                                                                   \
  GGUF_DECODE_INT8(R, a, EpNone, bfloat) GGUF_DECODE_INT8(R, r, EpResidual, bfloat)                                \
  GGUF_DECODE_INT8(R, g, EpUpWithGate, bfloat) GGUF_DECODE_INT8(R, a_f32, EpNone, float)                           \
  template [[host_name("gguf_decode_int8_pq20_m" #R "_gate_up")]] kernel GgufDecodeInt8GateUpKernel                \
      gguf_decode_int8_gate_up<R>;
GGUF_DECODE_INT8_ROWS(16) GGUF_DECODE_INT8_ROWS(32)
#undef GGUF_DECODE_INT8_ROWS
#undef GGUF_DECODE_INT8
#undef GGUF_SEGMENT_OF
#undef GGUF_SEGMENT_PLANE
#undef GGUF_SEGMENT_INDEX

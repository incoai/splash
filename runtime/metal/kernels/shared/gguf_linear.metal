// GGUF quantized GEMMs (K-quants, i-quants, Q8_0) for Apple9 and Apple10.
// Decode weights with FP32 group coefficients, then round once to the half tile,
// matching llama.cpp Metal dequantize.h / mul_mm.metal (MIT notice in THIRD_PARTY_NOTICES). Keep activations BF16.
// Weight planes and meta in the MDGG0001 layout (metal/abi/QuantFormat.h), decoded by kernels/common/quant_formats.h.
// Activations bf16 [rows][K]; weights staged as fp16 in threadgroup memory; fp32 accumulation; bf16 output (fp32 from
// the _a_f32 instances, the logits).
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

// ---------------- prefill tiles: a shared B stage (TileN x KS, all threads dequantize), each simdgroup owns RowsPerSG
// rows. `rows` counts the chunk's rows from the tile's first: simdgroups past them (the last tile of a chunk that is not
// a multiple of the tile) skip their matmuls and stores, so a chunk costs its rows rounded up to RowsPerSG rather than
// to the tile (a 33-row Q4_K 17408 x 5120 chunk: 1.7x faster on M5 and M3 than a 128-row tile). Full tiles run this
// loop too; a branch-free copy for them is no faster (M3 Max, M5 Max).
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
// MPP computes 16-row fragments, so a tile holds 8, 16 or 32 rows (these kernels and the fused ones): a 3-lane step
// runs the 32-row tile over the storage of four lanes (LinearPlan::storageRows) and the padding lane's rows are
// computed and discarded. Rows are independent, so every active row is the bits of any other tile height
// (gguf-projection checks it); the 32-row tile at three lanes costs what it costs at four, 3-15% less than a 16-row
// plus an 8-row matmul per stage (DRAM-cold, 16- and 20-core M5 Pro).
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
#define GGUF_DECODE_FORMAT(F, f)                                                                                  \
  GGUF_DECODE_ROWS(F, f, a, EpNone, bfloat) GGUF_DECODE_ROWS(F, f, a_f32, EpNone, float)                          \
  GGUF_DECODE_ROWS(F, f, r, EpResidual, bfloat) GGUF_DECODE_ROWS(F, f, g, EpUpWithGate, bfloat)
QUANT_FORMATS(GGUF_DECODE_FORMAT)
#undef GGUF_DECODE_FORMAT
#undef GGUF_DECODE_ROWS
#undef GGUF_DECODE

// Gate/up in one dispatch, where gate and up share their format (ops::Linear, LinearConfig::oneGateUpPass): grid
// (32-column tiles, K partitions), threadgroups of one simdgroup that stages the gate's and the up's same 32 columns
// (gguf_staged_pair_steps), so the per-step latency of a tile's chain serves both tensors. Each walks from the step its
// 64-column tile walks from in the two passes, so at the same K split gate and up are the bits of the gate pass's and
// of the up pass's sums, and the output is bf16(up) * silu(bf16(gate)) as the up pass writes it. Split partials
// [split][Rows][2 * out_stride]: the gate's columns, then the up's at out_stride on; one counter per 32 columns.
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
  const auto value = [](float g, float u) { return bfloat(float(bfloat(u)) * splash_silu(float(bfloat(g)))); };
  const auto out = [&](uint row, uint column) { return ulong(row) * p.out_stride + column0 + column; };
  if (p.splits == 1) {
#pragma unroll
    for (ushort i = 0; i < gate.get_capacity(); ++i) {
      if (!gate.is_valid_element(i)) continue;
      const auto index = gate.get_multidimensional_index(i);
      output[out(index[1], index[0])] = value(gate[i], up[i]);
    }
    return;
  }
  const auto at = [&](uint s, uint row, uint column) {
    return (ulong(s) * Rows + row) * (2 * p.out_stride) + column0 + column;
  };
#pragma unroll
  for (ushort i = 0; i < gate.get_capacity(); ++i) {
    if (!gate.is_valid_element(i)) continue;
    const auto index = gate.get_multidimensional_index(i);
    partials[at(group.y, index[1], index[0])] = gate[i];
    partials[at(group.y, index[1], index[0]) + p.out_stride] = up[i];
  }
  device atomic_uint *counter = counters + p.out_offset / GGUF_STAGED_COLUMNS + group.x;
  if (!split_arrive_last(counter, p.splits, lane, &arrival)) return;
#pragma unroll
  for (ushort i = 0; i < gate.get_capacity(); ++i) {
    if (!gate.is_valid_element(i)) continue;
    const auto index = gate.get_multidimensional_index(i);
    const uint row = index[1], column = index[0];
    const float g = split_sum(float(gate[i]), group.y, p.splits, [&](uint s) { return partials[at(s, row, column)]; });
    const float u = split_sum(float(up[i]), group.y, p.splits,
                              [&](uint s) { return partials[at(s, row, column) + p.out_stride]; });
    output[out(row, column)] = value(g, u);
  }
  split_release(counter, lane);
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
// The segment s of column tile `tile` among the segments' tiles (cols, 0 past the last), the tile's index `local`
// within it, and the segment's planes w0, w1 and meta.
#define GGUF_SEGMENT_OF(tile, cols)                                                                                \
  const uint t0 = (cols)[0] / GGUF_TILE_COLUMNS, t1 = t0 + (cols)[1] / GGUF_TILE_COLUMNS;                          \
  const uint s = (tile) < t0 ? 0 : (tile) < t1 ? 1 : 2, local = (tile) - (s == 0 ? 0 : s == 1 ? t0 : t1);          \
  device uchar *w0 = s == 0 ? w0a : s == 1 ? w0b : w0c;                                                           \
  device uchar *w1 = s == 0 ? w1a : s == 1 ? w1b : w1c;                                                           \
  device uchar *meta = s == 0 ? ma : s == 1 ? mb : mc
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
GGUF_DECODE_FUSED(8) GGUF_DECODE_FUSED(16) GGUF_DECODE_FUSED(32)
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
#undef GGUF_SEGMENT_OF
#undef GGUF_SEGMENTS
#undef GGUF_SEGMENT

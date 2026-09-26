// GGUF quantized GEMMs (K-quants, i-quants, Q8_0) for Apple9 and Apple10.
// Decode weights with FP32 group coefficients, then round once to the half tile,
// matching llama.cpp Metal dequantize.h / mul_mm.metal (MIT notice in THIRD_PARTY_NOTICES). Keep activations BF16.
// Weight planes and meta in the MDGG0001 layout (metal/abi/QuantFormat.h), decoded by kernels/common/quant_formats.h.
// Activations bf16 [rows][K]; weights staged as fp16 in threadgroup memory; fp32 accumulation; bf16 output.
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
// a multiple of the tile) still stage but skip their matmuls and stores, so a chunk costs its rows rounded up to
// RowsPerSG rather than to the tile (a 33-row Q4_K 17408 x 5120 chunk: 1.7x faster on M5 and M3 than a 128-row tile).
template <class F, ushort RowsPerSG, ushort Simdgroups, ushort TileN, ushort KS, GgufEpilogue Ep = EpNone>
inline void gguf_prefill_tile(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta, device bfloat *output,
                    uint output_size, uint input_size, uint output_origin, uint rows, threadgroup half *stage,
                    threadgroup half2 *tl, uint simd_lane, uint simd_group, uint out_stride = 0, uint out_offset = 0,
                    device bfloat *aux = nullptr) {
  // 0 = output_size (Gguf.h). The host always passes the stride, but dropping the fallback changes these kernels'
  // code, a change to measure on its own.
  if (out_stride == 0) out_stride = output_size;
  const bool owns_rows = simd_group * RowsPerSG < rows;   // uniform per simdgroup
  // Prefetch: the steps whose weights are loaded ahead of the one being staged.
  constexpr ushort Prefetch = 1, Threads = Simdgroups * 32, GPS = KS / 32, Items = TileN * GPS,
                   IPT = (Items + Threads - 1) / Threads;
  auto a = tensor(input + ulong(simd_group) * RowsPerSG * input_size, dextents<int, 2>{int(input_size), RowsPerSG}, array<int, 2>{1, int(input_size)});
  constexpr auto descriptor = matmul2d_descriptor(RowsPerSG, TileN, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  const uint groups = input_size / 32, steps = groups / GPS, units = groups / F::MetaGroups;
  const uint plane_tile = output_origin / QUANT_TILE_ROWS, plane_row = output_origin % QUANT_TILE_ROWS;
  device uchar *tw0 = w0 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P0;
  device uchar *tw1 = w1 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P1;
  device uchar *tmeta = meta + (ulong(plane_tile) * units * QUANT_TILE_ROWS + plane_row) * F::MetaBytes;
  auto a0 = a.template slice<KS, RowsPerSG>(0, 0);
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(stage, dextents<int, 2>{KS, TileN}, array<int, 2>{1, KS});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(stage + KS * TileN, dextents<int, 2>{KS, TileN}, array<int, 2>{1, KS});
  auto b0 = bt0.slice<KS, TileN>(0, 0), b1 = bt1.slice<KS, TileN>(0, 0);
  auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
  const uint thread_index = simd_group * 32 + simd_lane;
  typename F::Payload packed[Prefetch][IPT]; typename F::Meta hdr[IPT]; uint hdr_unit[IPT];
#pragma unroll
  for (ushort it = 0; it < IPT; ++it) {
    const uint item = thread_index + it * Threads; const bool live = item < Items;
    const uint col = live ? item % TileN : 0, gi = live ? item / TileN : 0;
#pragma unroll
    for (ushort pf = 0; pf < Prefetch; ++pf) {
      const ulong g = ulong(pf) * GPS + gi;
      if (live && pf < steps) packed[pf][it] = F::load(tw0 + (g * QUANT_TILE_ROWS + col) * F::P0, tw1 + (g * QUANT_TILE_ROWS + col) * F::P1);
    }
    hdr[it] = F::loadMeta(tmeta + col * F::MetaBytes); hdr_unit[it] = 0;
  }
  // The step loop with or without this simdgroup's matmuls, one copy each: a full tile runs the loop unchanged (a
  // branch around the matmul inside the loop cost 1-3% at full tiles on the M3 Max).
  const auto run_steps = [&](auto with_matmuls) {
    for (uint step = 0; step < steps; ++step) {
      threadgroup half *buf = stage + (step & 1) * (KS * TileN);
#pragma unroll
      for (ushort it = 0; it < IPT; ++it) {
        const uint item = thread_index + it * Threads; if (item >= Items) break;
        const uint col = item % TileN, gi = item / TileN, g = step * GPS + gi, unit = g / F::MetaGroups; const ushort j = g % F::MetaGroups;
        if (unit != hdr_unit[it]) { hdr[it] = F::loadMeta(tmeta + (ulong(unit) * QUANT_TILE_ROWS + col) * F::MetaBytes); hdr_unit[it] = unit; }
        dequant32<F>(packed[0][it], hdr[it], j, tl, buf + col * KS + gi * 32);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (step + Prefetch < steps) {
#pragma unroll
        for (ushort it = 0; it < IPT; ++it) {
          const uint item = thread_index + it * Threads; if (item >= Items) break;
          const uint col = item % TileN, gi = item / TileN; const ulong g = ulong(step + Prefetch) * GPS + gi;
          packed[Prefetch - 1][it] = F::load(tw0 + (g * QUANT_TILE_ROWS + col) * F::P0, tw1 + (g * QUANT_TILE_ROWS + col) * F::P1);
        }
      }
      if constexpr (decltype(with_matmuls)::value) {
        auto a_slice = a.template slice<KS, RowsPerSG>(step * KS, 0);
        if (step & 1) operation.run(a_slice, b1, acc); else operation.run(a_slice, b0, acc);
      }
    }
  };
  if (!owns_rows) { run_steps(false_type{}); return; }
  run_steps(true_type{});
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
// (gguf-projection full); on a 16-core M5 Pro the 32-row tile at three lanes costs what it costs at four, 3-15% less
// than a 16-row plus an 8-row matmul per stage (0.207 vs 0.218 ms, Q4_K 12288 x 5120, DRAM-cold; 20-core: 0.173 vs
// 0.203).
// Gate/up runs as a gate pass (a) into the gate scratch and an up pass (g) whose epilogue applies silu(gate) to the bf16
// up value, as the Apple9 register kernels do. The destination's type Out is bf16, or fp32 for the plain epilogue's
// logits (a_f32, ops::Projection::destination).
template <class F, ushort Rows, GgufEpilogue Ep, class Out>
inline void gguf_decode_tile(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta, device Out *output,
                             device coherent(device) float *partials, device atomic_uint *counters, device bfloat *aux,
                             constant GgufDecodeParams &p, uint2 group, uint simd_lane, uint simd_group,
                             threadgroup half *stage, threadgroup half2 *tl, threadgroup uint *arrival) {
  const uint per = p.input_size / GGUF_STAGED_STEP / p.splits,
             origin = group.x * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS, column0 = p.out_offset + origin;
  threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;
  auto acc = staged_accumulator<Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, p.input_size, my);
  gguf_zero(acc);
  staged_accumulate<F, Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, w0, w1, meta, p.input_size, origin, my, tl, simd_lane, group.y * per,
                                          (group.y + 1) * per, acc);
  gguf_store_sums<Rows>(acc, p.splits, group.y, partials, counters + p.out_offset / GGUF_TILE_COLUMNS + group.x, p.out_stride, column0,
                        simd_group * 32 + simd_lane, arrival, [&](uint row, uint column, float v) {
    // gguf_epilogue inline: calling it here reorders the lambda's captures.
    const ulong o = ulong(row) * p.out_stride + column0 + column;
    if constexpr (Ep == EpResidual) v += float(aux[o]);
    if constexpr (Ep == EpUpWithGate) v = float(bfloat(v)) * gguf_silu(float(aux[o]));
    output[o] = Out(v);
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

// Fused projections (qkv|z|ab, q|k|v): up to three column segments of any formats in one dispatch, so the small
// segments do not run as dispatches of their own. The threadgroup's tile picks its segment, and the segment's format
// picks the decode; every segment takes the same K splits. The fused and prefill kernels stay plain kernels: as
// template instantiations (weak_odr) the compiler infers fewer parameter attributes and inlines differently.
#define GGUF_SEGMENT(i, w0, w1, m) device uchar *w0 [[buffer(i)]], device uchar *w1 [[buffer(i + 1)]], device uchar *m [[buffer(i + 2)]]
#define GGUF_DECODE_FUSED(R)                                                                                       \
  kernel void gguf_decode_fused_m##R(device bfloat *input [[buffer(0)]], GGUF_SEGMENT(1, w0a, w1a, ma),          \
                                     GGUF_SEGMENT(4, w0b, w1b, mb), GGUF_SEGMENT(7, w0c, w1c, mc),                \
                                     device bfloat *output [[buffer(10)]],                                        \
                                     device coherent(device) float *partials [[buffer(11)]],                      \
                                     device atomic_uint *counters [[buffer(12)]],                                 \
                                     constant GgufDecodeFusedParams &p [[buffer(13)]],                            \
                                     uint2 group [[threadgroup_position_in_grid]],                                \
                                     uint simd_lane [[thread_index_in_simdgroup]],                                \
                                     uint simd_group [[simdgroup_index_in_threadgroup]]) {                        \
    threadgroup half stage[kStagedStages]; threadgroup half2 tl[kQuantPairTableEntries]; threadgroup uint arrival; \
    const uint t0 = p.cols[0] / GGUF_TILE_COLUMNS, t1 = t0 + p.cols[1] / GGUF_TILE_COLUMNS;                        \
    const uint s = group.x < t0 ? 0 : group.x < t1 ? 1 : 2;                                                       \
    device uchar *w0 = s == 0 ? w0a : s == 1 ? w0b : w0c;                                                         \
    device uchar *w1 = s == 0 ? w1a : s == 1 ? w1b : w1c;                                                         \
    device uchar *meta = s == 0 ? ma : s == 1 ? mb : mc;                                                          \
    const uint local = group.x - (s == 0 ? 0 : s == 1 ? t0 : t1), per = p.input_size / GGUF_STAGED_STEP / p.splits; \
    const uint origin = local * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS, column0 = p.offset[s] + origin; \
    threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;                                            \
    auto acc = staged_accumulator<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, p.input_size, my);                                         \
    gguf_zero(acc);                                                                                               \
    staged_accumulate_any<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(p.fmt[s], input, w0, w1, meta, p.input_size, origin, my, tl, \
                                            simd_group * 32 + simd_lane, simd_lane, group.y * per, (group.y + 1) * per, acc); \
    gguf_store_sums<R>(acc, p.splits, group.y, partials, counters + p.offset[s] / GGUF_TILE_COLUMNS + local,       \
                       p.out_stride, column0, simd_group * 32 + simd_lane, &arrival,                              \
                       [&](uint row, uint column, float v) { output[ulong(row) * p.out_stride + column0 + column] = bfloat(v); }); \
  }
GGUF_DECODE_FUSED(8) GGUF_DECODE_FUSED(16) GGUF_DECODE_FUSED(32)
#undef GGUF_DECODE_FUSED
#undef GGUF_SEGMENT

// The prefill kernels: grid (GGUF_PREFILL_ROWS-row tiles of the chunk, column tiles of the segment), four
// simdgroups; the residual and gate kernels read aux at buffer 5, the plain one binds none.
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
#define GGUF_PREFILL(F, f)                                                                                         \
  kernel void gguf_prefill_##f##_a(GGUF_PREFILL_BUFFERS, constant GgufPrefillParams &p [[buffer(5)]],            \
                                   GGUF_PREFILL_THREAD) {                                                          \
    GGUF_PREFILL_TABLES(F);                                                                                        \
    const uint first = group.x * GGUF_PREFILL_ROWS, rows = p.rows > first ? p.rows - first : 0;                    \
    gguf_prefill_tile<F, GGUF_PREFILL_SIMDGROUP_ROWS, GGUF_PREFILL_SIMDGROUPS, GGUF_TILE_COLUMNS, GGUF_PREFILL_STEP>(input + ulong(first) * p.input_size, w0, w1, meta,                        \
                                         output + ulong(first) * (p.out_stride ? p.out_stride : p.output_size),   \
                                         p.output_size, p.input_size, group.y * GGUF_TILE_COLUMNS, rows, stage, tl, \
                                         simd_lane, simd_group, p.out_stride, p.out_offset);                      \
  }
#define GGUF_PREFILL_EPILOGUE(F, f, ep, Ep)                                                                        \
  kernel void gguf_prefill_##f##_##ep(GGUF_PREFILL_BUFFERS, device bfloat *aux [[buffer(5)]],                    \
                                      constant GgufPrefillParams &p [[buffer(6)]], GGUF_PREFILL_THREAD) {          \
    GGUF_PREFILL_TABLES(F);                                                                                        \
    const uint rs = p.out_stride ? p.out_stride : p.output_size;                                                   \
    const uint first = group.x * GGUF_PREFILL_ROWS, rows = p.rows > first ? p.rows - first : 0;                    \
    gguf_prefill_tile<F, GGUF_PREFILL_SIMDGROUP_ROWS, GGUF_PREFILL_SIMDGROUPS, GGUF_TILE_COLUMNS, GGUF_PREFILL_STEP, Ep>(input + ulong(first) * p.input_size, w0, w1, meta,                    \
                                             output + ulong(first) * rs, p.output_size, p.input_size,              \
                                             group.y * GGUF_TILE_COLUMNS, rows, stage, tl, simd_lane, simd_group,  \
                                             p.out_stride, p.out_offset, aux + ulong(first) * rs);                 \
  }
#define GGUF_PREFILL_FORMAT(F, f) \
  GGUF_PREFILL(F, f) GGUF_PREFILL_EPILOGUE(F, f, r, EpResidual) GGUF_PREFILL_EPILOGUE(F, f, g, EpUpWithGate)
QUANT_FORMATS(GGUF_PREFILL_FORMAT)
#undef GGUF_PREFILL_FORMAT
#undef GGUF_PREFILL_EPILOGUE
#undef GGUF_PREFILL
#undef GGUF_PREFILL_TABLES
#undef GGUF_PREFILL_THREAD
#undef GGUF_PREFILL_BUFFERS

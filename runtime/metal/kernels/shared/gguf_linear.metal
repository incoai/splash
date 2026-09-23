// GGUF quantized GEMMs (K-quants, i-quants, Q8_0) for Apple9 and Apple10.
// Decode weights with FP32 group coefficients, then round once to the half tile,
// matching llama.cpp Metal dequantize.h / mul_mm.metal. Keep activations BF16.
// Weight planes and meta in the MDGG0001 layout (metal/abi/QuantFormat.h), decoded by kernels/common/quant_formats.h.
// Activations fp16 or bf16 [rows][K]; weights staged as fp16 in threadgroup memory; fp32 accumulation; bf16 output.
// Keep the source order of float operations, which Metal's default fast math lets the compiler reassociate. Set
// before the includes, so it also holds for the shared format and reduction code compiled here.
#pragma clang fp reassociate(off)
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/moe_expert_slab.h"
#include "metal/kernels/common/quant_formats.h"
#include "metal/kernels/common/split_reduce.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;
using namespace mpp::tensor_ops;
enum Epilogue : ushort { EpNone = GGUF_EPILOGUE_NONE, EpResidual = GGUF_EPILOGUE_RESIDUAL, EpUpWithGate = GGUF_EPILOGUE_UP_WITH_GATE };
inline float silu_gate(float g) { return g / (1.0f + fast::exp2(-1.44269504089f * g)); }
constant constexpr ushort kStorageN = 256;

// ---- staged dequantization (kernels/common/quant_formats.h): one thread writes one column's group of 32 as half,
// each value rounded once; chunk c's pairs 0, 1 go to dst + 4c and pairs 2, 3 to dst + 16 + 4c.
template <class F>
inline half2 staged_linear(uint pair, float s, float m) {
  const float2 code = float2(as_type<half2>(pair | 0x64006400u) - half2(half(1024 + F::Zero)));   // exact
  if constexpr (F::Zero) return half2(code * s);
  else return half2(fma(code, float2(s), float2(m)));
}
template <class F>
inline void dequant32(typename F::Payload w, typename F::Meta meta, ushort j, threadgroup half2 *tl, threadgroup half *dst) {
  QuantCoef k;
  if constexpr (F::Kind == QuantGrid) k = F::coef(meta, F::chunk(w, 0)); else k = F::coef(meta, j);
#pragma unroll
  for (ushort c = 0; c < 4; ++c) {
    const typename F::Chunk q = F::chunk(w, c);
    half4 lo, hi;
    if constexpr (F::Kind == QuantLinear) {
      const uint4 p = F::codes(q);
      lo = half4(staged_linear<F>(p.x, k.s.x, k.m), staged_linear<F>(p.y, k.s.x, k.m));
      hi = half4(staged_linear<F>(p.z, k.s.y, k.m), staged_linear<F>(p.w, k.s.y, k.m));
    } else if constexpr (F::Kind == QuantCodebook) {   // value * s in Scale: one rounding to half either way
      typedef typename F::Scale S;
      const uchar4 b = as_type<uchar4>(F::indices(q));
      lo = half4(half2(vec<S, 2>(tl[b.x]) * S(k.s.x)), half2(vec<S, 2>(tl[b.y]) * S(k.s.x)));
      hi = half4(half2(vec<S, 2>(tl[b.z]) * S(k.s.y)), half2(vec<S, 2>(tl[b.w]) * S(k.s.y)));
    } else if constexpr (F::Kind == QuantInt8) {
      typedef typename F::Scale S;
      const uint2 v = F::values(q);
      lo = half4(vec<S, 4>(as_type<char4>(v.x)) * S(k.s.x));
      hi = half4(vec<S, 4>(as_type<char4>(v.y)) * S(k.s.y));
    } else {
      const uint2 g = F::grid(q); const uint s = F::signs(q);
      lo = half4(float4(as_type<uchar4>(g.x)) * k.s.x);
      hi = half4(float4(as_type<uchar4>(g.y)) * k.s.y);
      lo = select(lo, -lo, bool4(s & 1, s & 2, s & 4, s & 8));
      hi = select(hi, -hi, bool4(s & 16, s & 32, s & 64, s & 128));
    }
    *((threadgroup half4 *)(dst + 4 * c)) = lo; *((threadgroup half4 *)(dst + 16 + 4 * c)) = hi;
  }
}

// ---------------- decode tiles: each simdgroup stages its own Cols x KS sub-tile privately and runs matmul2d alone.
// MPP computes 16-row fragments, so a tile holds 8, 16 or 32 rows: a 3-lane step runs the 32-row tile over the storage
// of four lanes (LinearPlan::storageRows) and the padding lane's rows are computed and discarded. Rows are independent,
// so every active row is the bits of any other tile height (gguf-projection full); on a 16-core M5 Pro the 32-row tile
// at three lanes costs what it costs at four, 3-15% less than a 16-row plus an 8-row matmul per stage (0.207 vs 0.218 ms,
// Q4_K 12288 x 5120, DRAM-cold; 20-core: 0.173 vs 0.203).

// The destination of a Rows-row tile, zeroed by the caller: returning an initialized cooperative tensor loses its
// initial values on Apple9 in runtime-format kernels (also with shader validation).
template <typename TA, ushort Rows, ushort Cols, ushort KS>
inline auto gguf_make_acc(device TA *input, uint input_size, threadgroup half *stage) {
  auto a = tensor(input, dextents<int, 2>{int(input_size), Rows}, array<int, 2>{1, int(input_size)});
  constexpr auto descriptor = matmul2d_descriptor(Rows, Cols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  auto a0 = a.template slice<KS, Rows>(0, 0);
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(stage, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  auto b0 = bt0.slice<KS, Cols>(0, 0);
  return operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
}
template <class Acc> inline void gguf_zero(thread Acc &acc) {
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
}
// fn(row, column, value) for every element of a tile's destination.
template <class Acc, class Fn> inline void gguf_elements(thread Acc &acc, Fn fn) {
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto index = acc.get_multidimensional_index(i);
    fn(uint(index[1]), uint(index[0]), float(acc[i]));
  }
}

// The decode tile loop without the store: dequantize one KS-input step of the Cols columns into the stage, then run
// the tile's matmul2d on it, over steps [step_begin, step_end) of K.
template <class F, typename TA, ushort Rows, ushort Cols, ushort KS, ushort Buffers, ushort Prefetch, class Acc>
inline void sg_accum(device TA *input, device uchar *w0, device uchar *w1, device uchar *meta, uint input_size, uint output_origin,
                     threadgroup half *stage, threadgroup half2 *tl, uint simd_lane, uint step_begin, uint step_end, thread Acc &acc) {
  constexpr ushort GPS = KS / 32, Items = Cols * GPS, IPT = (Items + 31) / 32;
  auto a = tensor(input, dextents<int, 2>{int(input_size), Rows}, array<int, 2>{1, int(input_size)});
  constexpr auto descriptor = matmul2d_descriptor(Rows, Cols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  const uint groups = input_size / 32, units = groups / F::MetaGroups;
  const uint tile = output_origin / kStorageN, tile_offset = output_origin % kStorageN;
  device uchar *tw0 = w0 + (ulong(tile) * groups * kStorageN + tile_offset) * F::P0;
  device uchar *tw1 = w1 + (ulong(tile) * groups * kStorageN + tile_offset) * F::P1;
  device uchar *tmeta = meta + (ulong(tile) * units * kStorageN + tile_offset) * F::MetaBytes;
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(stage, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(stage + (Buffers > 1 ? KS * Cols : 0), dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  auto b0 = bt0.slice<KS, Cols>(0, 0), b1 = bt1.slice<KS, Cols>(0, 0);
  typename F::Payload packed[Prefetch][IPT]; typename F::Meta hdr[IPT]; uint hdr_unit[IPT];
  const uint unit0 = (step_begin * GPS) / F::MetaGroups;
#pragma unroll
  for (ushort it = 0; it < IPT; ++it) {
    const uint item = simd_lane + it * 32; const bool live = item < Items;
    const uint col = live ? item % Cols : 0, gi = live ? item / Cols : 0;
#pragma unroll
    for (ushort pf = 0; pf < Prefetch; ++pf) {
      const ulong g = ulong(step_begin + pf) * GPS + gi;
      if (live && step_begin + pf < step_end) packed[pf][it] = F::load(tw0 + (g * kStorageN + col) * F::P0, tw1 + (g * kStorageN + col) * F::P1);
    }
    hdr[it] = F::loadMeta(tmeta + (ulong(unit0) * kStorageN + col) * F::MetaBytes); hdr_unit[it] = unit0;
  }
  for (uint step = step_begin; step < step_end; ++step) {
    threadgroup half *buf = stage + (Buffers > 1 ? (step & 1) * (KS * Cols) : 0);
    if constexpr (Buffers == 1) simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (ushort it = 0; it < IPT; ++it) {
      const uint item = simd_lane + it * 32; if (item >= Items) break;
      const uint col = item % Cols, gi = item / Cols, g = step * GPS + gi, unit = g / F::MetaGroups; const ushort j = g % F::MetaGroups;
      if (unit != hdr_unit[it]) { hdr[it] = F::loadMeta(tmeta + (ulong(unit) * kStorageN + col) * F::MetaBytes); hdr_unit[it] = unit; }
      dequant32<F>(packed[0][it], hdr[it], j, tl, buf + col * KS + gi * 32);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (ushort pf = 0; pf + 1 < Prefetch; ++pf)
#pragma unroll
      for (ushort it = 0; it < IPT; ++it) packed[pf][it] = packed[pf + 1][it];
    if (step + Prefetch < step_end) {
#pragma unroll
      for (ushort it = 0; it < IPT; ++it) {
        const uint item = simd_lane + it * 32; if (item >= Items) break;
        const uint col = item % Cols, gi = item / Cols; const ulong g = ulong(step + Prefetch) * GPS + gi;
        packed[Prefetch - 1][it] = F::load(tw0 + (g * kStorageN + col) * F::P0, tw1 + (g * kStorageN + col) * F::P1);
      }
    }
    auto a_slice = a.template slice<KS, Rows>(step * KS, 0);
    if (Buffers > 1 && (step & 1)) operation.run(a_slice, b1, acc); else operation.run(a_slice, b0, acc);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);   // the stage may be reused by a following accumulate
}

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

// ---------------- pf: prefill with a shared B stage (TileN x KS, all threads dequantize), each simdgroup owns RowsPerSG
// rows. `rows` counts the chunk's rows from the tile's first: simdgroups past them (the last tile of a chunk that is not
// a multiple of the tile) still stage but skip their matmuls and stores, so a chunk costs its rows rounded up to
// RowsPerSG rather than to the tile (a 33-row Q4_K 17408 x 5120 chunk: 1.7x faster on M5 and M3 than a 128-row tile).
template <class F, typename TA, ushort RowsPerSG, ushort Simdgroups, ushort TileN, ushort KS, ushort Prefetch, ushort Ep = EpNone>
inline void pf_tile(device TA *input, device uchar *w0, device uchar *w1, device uchar *meta, device bfloat *output,
                    uint output_size, uint input_size, uint output_origin, uint rows, threadgroup half *stage,
                    threadgroup half2 *tl, uint simd_lane, uint simd_group, uint out_stride = 0, uint out_offset = 0,
                    device bfloat *aux = nullptr) {
  if (out_stride == 0) out_stride = output_size;
  const bool owns_rows = simd_group * RowsPerSG < rows;   // uniform per simdgroup
  constexpr ushort Threads = Simdgroups * 32, GPS = KS / 32, Items = TileN * GPS, IPT = (Items + Threads - 1) / Threads;
  auto a = tensor(input + ulong(simd_group) * RowsPerSG * input_size, dextents<int, 2>{int(input_size), RowsPerSG}, array<int, 2>{1, int(input_size)});
  constexpr auto descriptor = matmul2d_descriptor(RowsPerSG, TileN, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  const uint groups = input_size / 32, steps = groups / GPS, units = groups / F::MetaGroups;
  const uint tile = output_origin / kStorageN, tile_offset = output_origin % kStorageN;
  device uchar *tw0 = w0 + (ulong(tile) * groups * kStorageN + tile_offset) * F::P0;
  device uchar *tw1 = w1 + (ulong(tile) * groups * kStorageN + tile_offset) * F::P1;
  device uchar *tmeta = meta + (ulong(tile) * units * kStorageN + tile_offset) * F::MetaBytes;
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
      if (live && pf < steps) packed[pf][it] = F::load(tw0 + (g * kStorageN + col) * F::P0, tw1 + (g * kStorageN + col) * F::P1);
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
        if (unit != hdr_unit[it]) { hdr[it] = F::loadMeta(tmeta + (ulong(unit) * kStorageN + col) * F::MetaBytes); hdr_unit[it] = unit; }
        dequant32<F>(packed[0][it], hdr[it], j, tl, buf + col * KS + gi * 32);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
      for (ushort pf = 0; pf + 1 < Prefetch; ++pf)
#pragma unroll
        for (ushort it = 0; it < IPT; ++it) packed[pf][it] = packed[pf + 1][it];
      if (step + Prefetch < steps) {
#pragma unroll
        for (ushort it = 0; it < IPT; ++it) {
          const uint item = thread_index + it * Threads; if (item >= Items) break;
          const uint col = item % TileN, gi = item / TileN; const ulong g = ulong(step + Prefetch) * GPS + gi;
          packed[Prefetch - 1][it] = F::load(tw0 + (g * kStorageN + col) * F::P0, tw1 + (g * kStorageN + col) * F::P1);
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
    float v = acc[i];
    if constexpr (Ep == EpResidual) v += float(aux[o]);
    if constexpr (Ep == EpUpWithGate) v = float(bfloat(v)) * silu_gate(float(aux[o]));
    output[o] = bfloat(v);
  }
}

#define TGLUT_INIT(F)                                                                                     \
  threadgroup half2 tl[F::Kind == QuantCodebook ? 256 : 1];                                                \
  if constexpr (F::Kind == QuantCodebook) quant_iq4_pair_table(tl, simd_group * 32 + simd_lane, Threads);
#define ABUF(TA, P) device TA *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]], \
                 device uchar *meta [[buffer(3)]], device bfloat *output [[buffer(4)]], constant P &p [[buffer(5)]]
#define ABUFE(P) device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]], \
                 device uchar *meta [[buffer(3)]], device bfloat *output [[buffer(4)]], device bfloat *aux [[buffer(5)]], constant P &p [[buffer(6)]]
#define IDS uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]
#define PF_ROWS(R, S) const uint first = group.x * (R * S), rows = p.rows > first ? p.rows - first : 0
#define PFE_K(F, f, EP, ep, R, S, N, KS, P)                                                               \
  kernel void pf##ep##_##f##_r##R##_sg##S##_n##N##_k##KS##_p##P(ABUFE(GgufPrefillParams), uint2 group [[threadgroup_position_in_grid]], IDS) { \
    constexpr ushort Threads = S * 32; TGLUT_INIT(F)                                                      \
    threadgroup half stage[2 * KS * N]; const uint rs = p.out_stride ? p.out_stride : p.output_size;      \
    PF_ROWS(R, S);                                                                                        \
    pf_tile<F, bfloat, R, S, N, KS, P, EP>(input + ulong(first) * p.input_size, w0, w1, meta, output + ulong(first) * rs, \
                                   p.output_size, p.input_size, group.y * N, rows, stage, tl, simd_lane, simd_group, p.out_stride, p.out_offset, aux + ulong(first) * rs); }
#define PF_K(F, f, TA, ta, R, S, N, KS, P)                                                                \
  kernel void pf##ta##_##f##_r##R##_sg##S##_n##N##_k##KS##_p##P(ABUF(TA, GgufPrefillParams), uint2 group [[threadgroup_position_in_grid]], IDS) { \
    constexpr ushort Threads = S * 32; TGLUT_INIT(F)                                                      \
    threadgroup half stage[2 * KS * N];                                                                   \
    PF_ROWS(R, S);                                                                                        \
    pf_tile<F, TA, R, S, N, KS, P>(input + ulong(first) * p.input_size, w0, w1, meta, output + ulong(first) * (p.out_stride ? p.out_stride : p.output_size), \
                                   p.output_size, p.input_size, group.y * N, rows, stage, tl, simd_lane, simd_group, p.out_stride, p.out_offset); }
// runtime dequantizer selection (uniform per threadgroup)
template <typename TA, ushort Rows, ushort Cols, ushort KS, ushort Buffers, ushort Prefetch, class Acc>
inline void gguf_accum_any(uint fmt, device TA *input, device uchar *w0, device uchar *w1, device uchar *meta, uint input_size, uint origin,
                           threadgroup half *stage, threadgroup half2 *tl, uint simd_lane, uint sb, uint se, thread Acc &acc) {
  quant_format_switch(fmt, [&](auto format) {
    sg_accum<decltype(format), TA, Rows, Cols, KS, Buffers, Prefetch>(input, w0, w1, meta, input_size, origin, stage, tl,
                                                                     simd_lane, sb, se, acc);
  });
}

// ---------------- decode dispatches over (64-column tiles, K partitions): two simdgroups of 32 columns per
// threadgroup, every request lane in its tile, `splits` partitions of K (grid.y; kernels/common/split_reduce.h).
// Gate/up runs as a gate pass (a) into the gate scratch and an up pass (g) whose epilogue applies silu(gate) to the bf16
// up value, as the Apple9 register kernels do.
template <class F, ushort Rows, ushort Ep>
inline void gguf_decode_tile(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta, device bfloat *output,
                             device coherent(device) float *partials, device atomic_uint *counters, device bfloat *aux,
                             constant GgufDecodeParams &p, uint2 group, uint simd_lane, uint simd_group,
                             threadgroup half *stage, threadgroup half2 *tl, threadgroup uint *arrival) {
  const uint per = p.input_size / 32 / p.splits, origin = group.x * 64 + simd_group * 32, column0 = p.out_offset + origin;
  threadgroup half *my = stage + simd_group * (2 * 32 * 32);
  auto acc = gguf_make_acc<bfloat, Rows, 32, 32>(input, p.input_size, my);
  gguf_zero(acc);
  sg_accum<F, bfloat, Rows, 32, 32, 2, 1>(input, w0, w1, meta, p.input_size, origin, my, tl, simd_lane, group.y * per,
                                          (group.y + 1) * per, acc);
  gguf_store_sums<Rows>(acc, p.splits, group.y, partials, counters + p.out_offset / 64 + group.x, p.out_stride, column0,
                        simd_group * 32 + simd_lane, arrival, [&](uint row, uint column, float v) {
    const ulong o = ulong(row) * p.out_stride + column0 + column;
    if constexpr (Ep == EpResidual) v += float(aux[o]);
    if constexpr (Ep == EpUpWithGate) v = float(bfloat(v)) * silu_gate(float(aux[o]));
    output[o] = bfloat(v);
  });
}
#define SEGBUF(i, w0, w1, m) device uchar *w0 [[buffer(i)]], device uchar *w1 [[buffer(i + 1)]], device uchar *m [[buffer(i + 2)]]
#define GGUF_DECODE_K(F, f, R, EP, ep)                                                                               \
  kernel void gguf_decode_##f##_m##R##_##ep(device bfloat *input [[buffer(0)]], SEGBUF(1, w0, w1, meta), device bfloat *output [[buffer(4)]], \
                                            device coherent(device) float *partials [[buffer(5)]], device atomic_uint *counters [[buffer(6)]], \
                                            device bfloat *aux [[buffer(7)]], constant GgufDecodeParams &p [[buffer(8)]], \
                                            uint2 group [[threadgroup_position_in_grid]], IDS) {                      \
    constexpr ushort Threads = 64; TGLUT_INIT(F) threadgroup half stage[2 * 2 * 32 * 32]; threadgroup uint arrival;  \
    gguf_decode_tile<F, R, EP>(input, w0, w1, meta, output, partials, counters, aux, p, group, simd_lane, simd_group, stage, tl, &arrival); }
#define GGUF_DECODE_ROWS(F, f, EP, ep) GGUF_DECODE_K(F, f, 8, EP, ep) GGUF_DECODE_K(F, f, 16, EP, ep) GGUF_DECODE_K(F, f, 32, EP, ep)
#define GGUF_DECODE_SET(F, f) GGUF_DECODE_ROWS(F, f, EpNone, a) GGUF_DECODE_ROWS(F, f, EpResidual, r) GGUF_DECODE_ROWS(F, f, EpUpWithGate, g)
GGUF_DECODE_SET(FmtQ4K, q4k) GGUF_DECODE_SET(FmtIQ4XS, iq4xs) GGUF_DECODE_SET(FmtIQ4NL, iq4nl) GGUF_DECODE_SET(FmtQ5K, q5k)
GGUF_DECODE_SET(FmtQ6K, q6k) GGUF_DECODE_SET(FmtQ3K, q3k) GGUF_DECODE_SET(FmtQ80, q80) GGUF_DECODE_SET(FmtIQ3S, iq3s)

// Fused projections (qkv|z|ab, q|k|v): up to three column segments of any formats in one dispatch, so the small
// segments do not run as dispatches of their own. The threadgroup's tile picks its segment, and the segment's format
// picks the decode; every segment takes the same K splits.
#define GGUF_DECODE_FUSED_K(R)                                                                                     \
  kernel void gguf_decode_fused_m##R(device bfloat *input [[buffer(0)]], SEGBUF(1, w0a, w1a, ma), SEGBUF(4, w0b, w1b, mb), \
                                     SEGBUF(7, w0c, w1c, mc), device bfloat *output [[buffer(10)]],          \
                                     device coherent(device) float *partials [[buffer(11)]], device atomic_uint *counters [[buffer(12)]], \
                                     constant GgufDecodeFusedParams &p [[buffer(13)]], uint2 group [[threadgroup_position_in_grid]], IDS) { \
    threadgroup half stage[2 * 2 * 32 * 32]; threadgroup half2 tl[256]; threadgroup uint arrival;           \
    quant_iq4_pair_table(tl, simd_group * 32 + simd_lane, 64);                                               \
    const uint t0 = p.cols[0] / 64, t1 = t0 + p.cols[1] / 64, s = group.x < t0 ? 0 : group.x < t1 ? 1 : 2;  \
    device uchar *w0 = s == 0 ? w0a : s == 1 ? w0b : w0c;                                                   \
    device uchar *w1 = s == 0 ? w1a : s == 1 ? w1b : w1c;                                                   \
    device uchar *meta = s == 0 ? ma : s == 1 ? mb : mc;                                                    \
    const uint local = group.x - (s == 0 ? 0 : s == 1 ? t0 : t1), per = p.input_size / 32 / p.splits;       \
    const uint origin = local * 64 + simd_group * 32, column0 = p.offset[s] + origin;                       \
    threadgroup half *my = stage + simd_group * (2 * 32 * 32);                                              \
    auto acc = gguf_make_acc<bfloat, R, 32, 32>(input, p.input_size, my);                                   \
    gguf_zero(acc);                                                                                         \
    gguf_accum_any<bfloat, R, 32, 32, 2, 1>(p.fmt[s], input, w0, w1, meta, p.input_size, origin, my, tl, simd_lane, \
                                            group.y * per, (group.y + 1) * per, acc);                      \
    gguf_store_sums<R>(acc, p.splits, group.y, partials, counters + p.offset[s] / 64 + local, p.out_stride, column0, \
                       simd_group * 32 + simd_lane, &arrival, [&](uint row, uint column, float v) {        \
      output[ulong(row) * p.out_stride + column0 + column] = bfloat(v);                                     \
    });                                                                                                     \
  }
GGUF_DECODE_FUSED_K(8) GGUF_DECODE_FUSED_K(16) GGUF_DECODE_FUSED_K(32)


#define PROD_SET(F, f) \
  PF_K(F, f, bfloat, a, 32, 4, 64, 64, 1) PFE_K(F, f, EpResidual, r, 32, 4, 64, 64, 1) PFE_K(F, f, EpUpWithGate, g, 32, 4, 64, 64, 1)
PROD_SET(FmtQ4K, q4k)
PROD_SET(FmtIQ4XS, iq4xs)
PROD_SET(FmtIQ4NL, iq4nl)
PROD_SET(FmtQ5K, q5k)
PROD_SET(FmtQ6K, q6k)
PROD_SET(FmtQ3K, q3k)
PROD_SET(FmtQ80, q80)
PROD_SET(FmtIQ3S, iq3s)

// ---------------- MoE experts (ops/MoE.cpp; kernels/shared/moe.metal groups the rows): threadgroup (x, y) computes
// 64 columns of grouped tile y with the weights of the tile's expert (moe_gguf_segment), in the format the tile picks
// at run time: on a 16-core M5 Pro one run-time-format dispatch over two segments is within -11..+8% of a dispatch
// per format (time-sg at 23040x2048 Q4_K and 92160x512 Q5_K, one to four lanes). aux is the gate of the up pass.
// Two simdgroups each stream their own 32 columns through the decode tile, grid (N / 64, tiles), on 8-row tiles
// (decode steps, short prefill chunks) or 32-row tiles (longer chunks, ops::moeGgufPrefillTile), where a tile runs the
// 16- or 32-row matmul that holds its live rows: an expert's last tile is mostly partial. On the 35B's real prefill
// routes (wikitext, chat, code; the three passes of a layer on a 16-core M5 Pro) 32-row tiles take 2.42-2.78 ms at
// 512 rows and 6.70-6.83 ms at 2048 rows against 3.44-3.86 and 8.20-8.23 for 64-row tiles sharing one 64-column stage
// over four 16-row simdgroups, and 1.29-1.71 ms against 1.59-2.60 for 8-row tiles at 128-256 rows.
template <ushort Rows, ushort Ep>
inline void moe_gguf_expert_tile(device bfloat *input, device const MoeTileDescriptor *tiles, device const uint *tile_count,
                                 device uchar *w0, device uchar *w1, device uchar *meta, device uchar *sw0, device uchar *sw1,
                                 device uchar *smeta, device bfloat *output, device bfloat *aux,
                                 constant MoeGgufExpertParams &p, uint2 group, uint simd_lane, uint simd_group,
                                 threadgroup half *stage, threadgroup half2 *tl) {
  if (group.y >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[group.y];
  const MoeGgufSegment s = moe_gguf_segment(tile.expert, p, w0, w1, meta, sw0, sw1, smeta);
  device bfloat *x = input + ulong(group.y) * Rows * p.input_size;
  const ulong out = ulong(group.y) * Rows * p.output_size;
  const uint origin = group.x * 64 + simd_group * 32;
  threadgroup half *my = stage + simd_group * (2 * 32 * 32);
  quant_iq4_pair_table(tl, simd_group * 32 + simd_lane, 64);
  const auto run = [&](auto rows) {
    constexpr ushort R = decltype(rows)::value;
    auto acc = gguf_make_acc<bfloat, R, 32, 32>(x, p.input_size, my);
    gguf_zero(acc);
    gguf_accum_any<bfloat, R, 32, 32, 2, 1>(s.format, x, s.w0, s.w1, s.meta, p.input_size, origin, my, tl, simd_lane, 0,
                                            p.input_size / 32, acc);
    gguf_elements(acc, [&](uint row, uint column, float v) {
      const ulong o = out + ulong(row) * p.output_size + origin + column;
      if constexpr (Ep == EpUpWithGate) v = float(bfloat(v)) * silu_gate(float(aux[o]));
      output[o] = bfloat(v);
    });
  };
  if constexpr (Rows == 8) run(integral_constant<ushort, 8>{});
  else if (tile.rows <= 16) run(integral_constant<ushort, 16>{});
  else run(integral_constant<ushort, 32>{});
}
#define MOE_GGUF_K(Name, Rows, Ep)                                                                                      \
  kernel void Name(device bfloat *input [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],            \
                   device const uint *tile_count [[buffer(2)]], SEGBUF(3, w0, w1, meta), SEGBUF(6, sw0, sw1, smeta),     \
                   device bfloat *output [[buffer(9)]], device bfloat *aux [[buffer(10)]],                               \
                   constant MoeGgufExpertParams &p [[buffer(11)]], uint2 group [[threadgroup_position_in_grid]], IDS) {  \
    threadgroup half stage[2 * 2 * 32 * 32]; threadgroup half2 tl[256];                                                \
    moe_gguf_expert_tile<Rows, Ep>(input, tiles, tile_count, w0, w1, meta, sw0, sw1, smeta, output, aux, p, group,     \
                                   simd_lane, simd_group, stage, tl);                                                  \
  }
MOE_GGUF_K(moe_expert_gguf_m8, 8, EpNone)
MOE_GGUF_K(moe_expert_gguf_m8_up, 8, EpUpWithGate)
MOE_GGUF_K(moe_expert_gguf_m32, 32, EpNone)
MOE_GGUF_K(moe_expert_gguf_m32_up, 32, EpUpWithGate)
#undef MOE_GGUF_K

// ---------------- token embedding gather from native block_q4_K rows (row = K/256 blocks of 144 B)
kernel void gguf_embed_q4k(device const uint *tokens [[buffer(0)]], device const uchar *table [[buffer(1)]], device bfloat *output [[buffer(2)]],
                      constant GgufEmbedParams &p [[buffer(3)]], uint index [[thread_position_in_grid]]) {
  const uint elements = p.rows * p.hidden; if (index >= elements) return;
  const uint row = index / p.hidden, dim = index % p.hidden;
  uint token = tokens[row]; token = token < p.vocabulary ? token : 0;
  device const uchar *blk = table + (ulong(token) * (p.hidden / 256) + dim / 256) * 144;
  const uint j = (dim % 256) / 32, l = dim % 32;
  const half d = as_type<half>(ushort(blk[0] | (blk[1] << 8))), dmin = as_type<half>(ushort(blk[2] | (blk[3] << 8)));
  device const uchar *sc = blk + 4; uchar s, m;
  if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; } else { s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
  const uchar q = (blk[16 + (j / 2) * 32 + l] >> ((j % 2) * 4)) & 15;
  output[index] = bfloat(float(d) * float(s) * float(q) - float(dmin) * float(m));
}
// native block_q6_K rows (210 B per 256 weights): ql[128] | qh[64] | int8 scales[16] | half d
kernel void gguf_embed_q6k(device const uint *tokens [[buffer(0)]], device const uchar *table [[buffer(1)]], device bfloat *output [[buffer(2)]],
                      constant GgufEmbedParams &p [[buffer(3)]], uint index [[thread_position_in_grid]]) {
  const uint elements = p.rows * p.hidden; if (index >= elements) return;
  const uint row = index / p.hidden, dim = index % p.hidden;
  uint token = tokens[row]; token = token < p.vocabulary ? token : 0;
  device const uchar *blk = table + (ulong(token) * (p.hidden / 256) + dim / 256) * 210;
  const uint l = dim % 256, n = l / 128, r = l % 128, quarter = r / 32, pos = r % 32;
  const uchar lo = (blk[n * 64 + (quarter & 1) * 32 + pos] >> ((quarter >> 1) * 4)) & 15;
  const uchar hi = (blk[128 + n * 32 + pos] >> (2 * quarter)) & 3;
  const char sc = as_type<char>(blk[192 + n * 8 + 2 * quarter + pos / 16]);
  const half d = as_type<half>(ushort(blk[208] | (blk[209] << 8)));
  output[index] = bfloat(float(d) * float(sc) * float(int(lo | (hi << 4)) - 32));
}
// native block_q8_0 rows (34 B per 32 weights): half d | int8 qs[32]
kernel void gguf_embed_q80(device const uint *tokens [[buffer(0)]], device const uchar *table [[buffer(1)]], device bfloat *output [[buffer(2)]],
                      constant GgufEmbedParams &p [[buffer(3)]], uint index [[thread_position_in_grid]]) {
  const uint elements = p.rows * p.hidden; if (index >= elements) return;
  const uint row = index / p.hidden, dim = index % p.hidden;
  uint token = tokens[row]; token = token < p.vocabulary ? token : 0;
  device const uchar *blk = table + (ulong(token) * (p.hidden / 32) + dim / 32) * 34;
  const half d = as_type<half>(ushort(blk[0] | (blk[1] << 8)));
  output[index] = bfloat(float(d) * float(as_type<char>(blk[2 + dim % 32])));
}

// ---- load-time repack: native GGUF rows -> MDGG0001 planes (metal/abi/QuantFormat.h) ----
// One thread per (destination row n, 32-wide K group g). Rows >= permute_from_row are read from
// llama.cpp's tiled value-head order so the image holds splash's grouped order.
static inline uint gguf_repack_source_row(uint n, constant GgufRepackParams &p) {
  if (n < p.permute_from_row) return n;
  const uint head = (n - p.permute_from_row) / p.permute_head_rows, e = (n - p.permute_from_row) % p.permute_head_rows;
  const uint source = (head % p.permute_groups) * p.permute_group_heads + head / p.permute_groups;
  return p.permute_from_row + source * p.permute_head_rows + e;
}
// Word w of the little-endian string of 32 slot values of `bits` bits each (1, 2, 4 or 8).
static inline uint gguf_bit_word(thread const uchar *slots, uint bits, uint w) {
  const uint per = 32 / bits;
  uint word = 0;
  for (uint i = 0; i < per; ++i) word |= uint(slots[w * per + i]) << (bits * i);
  return word;
}
static inline void gguf_store_bits(thread const uchar *slots, uint bits, device uchar *dst) {
  for (uint w = 0; w < bits; ++w) ((device uint *)dst)[w] = gguf_bit_word(slots, bits, w);
}
// 4-bit linear codes: word c holds slots 8c..8c+7, pair p at bits 4p (e0) and 16 + 4p (e1).
static inline void gguf_store_pairs(thread const uchar *slots, device uchar *dst) {
  for (uint c = 0; c < 4; ++c) {
    uint word = 0;
    for (uint i = 0; i < 8; ++i) word |= uint(slots[8 * c + i]) << ((i & 1) * 16 + 4 * (i >> 1));
    ((device uint *)dst)[c] = word;
  }
}
kernel void gguf_repack(device const uchar *src [[buffer(0)]], device uchar *dst [[buffer(1)]],
                      constant GgufRepackParams &p [[buffer(2)]], uint t [[thread_position_in_grid]]) {
  const uint G = p.input_size / 32;
  if (t >= p.rows * G) return;
  const uint n = t / G, g = t % G, r = gguf_repack_source_row(n, p);
  constant QuantFormat &f = kQuantFormats[p.fmt];
  const uint b = g / f.meta_groups, j = g % f.meta_groups;   // native block b holds meta unit b
  device const uchar *blk = src + p.src_offset + ulong(r) * p.src_row_bytes + ulong(b) * f.block_bytes;
  device uchar *out0 = dst + p.dst_plane0 + quant_tile_index(n, g, G) * f.plane0_bytes;
  device uchar *out1 = dst + p.dst_plane1 + quant_tile_index(n, g, G) * f.plane1_bytes;
  device uchar *meta = dst + p.dst_meta + quant_tile_index(n, b, G / f.meta_groups) * f.meta_bytes;
  uchar lo[32], hi[32];   // per slot: the (low) code and its high bits
  switch (p.fmt) {
    case GGUF_FMT_Q4K: {
      for (uint e = 0; e < 32; ++e) lo[quant_slot(e)] = (blk[16 + (j / 2) * 32 + e] >> (4 * (j % 2))) & 15;
      gguf_store_pairs(lo, out0);
      if (j == 0) for (uint i = 0; i < 16; ++i) meta[i] = blk[i];
      break;
    }
    case GGUF_FMT_Q5K: {
      for (uint e = 0; e < 32; ++e) {
        lo[quant_slot(e)] = (blk[48 + (j / 2) * 32 + e] >> (4 * (j % 2))) & 15;
        hi[quant_slot(e)] = (blk[16 + e] >> j) & 1;
      }
      gguf_store_pairs(lo, out0);
      gguf_store_bits(hi, 1, out1);
      if (j == 0) for (uint i = 0; i < 16; ++i) meta[i] = blk[i];
      break;
    }
    case GGUF_FMT_IQ4XS: {
      for (uint l = 0; l < 16; ++l) { const uchar q = blk[8 + 16 * j + l]; lo[quant_slot(l)] = q & 15; lo[quant_slot(16 + l)] = q >> 4; }
      gguf_store_bits(lo, 4, out0);
      if (j == 0) for (uint i = 0; i < 8; ++i) meta[i] = blk[i];
      break;
    }
    case GGUF_FMT_IQ4NL: {
      for (uint l = 0; l < 16; ++l) { const uchar q = blk[2 + l]; lo[quant_slot(l)] = q & 15; lo[quant_slot(16 + l)] = q >> 4; }
      gguf_store_bits(lo, 4, out0);
      meta[0] = blk[0]; meta[1] = blk[1];
      break;
    }
    case GGUF_FMT_Q6K: {
      const uint hb = j / 4, quarter = j % 4;
      for (uint e = 0; e < 32; ++e) {
        lo[quant_slot(e)] = (blk[64 * hb + 32 * (quarter & 1) + e] >> (4 * (quarter >> 1))) & 15;
        hi[quant_slot(e)] = (blk[128 + 32 * hb + e] >> (2 * quarter)) & 3;
      }
      gguf_store_pairs(lo, out0);
      gguf_store_bits(hi, 2, out1);
      if (j == 0) { for (uint i = 0; i < 16; ++i) meta[i] = blk[192 + i]; meta[16] = blk[208]; meta[17] = blk[209]; meta[18] = 0; meta[19] = 0; }
      break;
    }
    case GGUF_FMT_Q3K: {
      const uint hb = j / 4, jj = j % 4;
      for (uint e = 0; e < 32; ++e) {
        lo[quant_slot(e)] = (blk[32 + 32 * hb + e] >> (2 * jj)) & 3;
        hi[quant_slot(e)] = (blk[e] >> j) & 1;
      }
      gguf_store_bits(lo, 2, out0);
      gguf_store_bits(hi, 1, out1);
      if (j == 0) { meta[0] = blk[108]; meta[1] = blk[109]; meta[2] = 0; meta[3] = 0; for (uint i = 0; i < 12; ++i) meta[4 + i] = blk[96 + i]; }
      break;
    }
    case GGUF_FMT_Q80: {
      for (uint e = 0; e < 32; ++e) lo[quant_slot(e)] = blk[2 + e];
      gguf_store_bits(lo, 8, out0);
      meta[0] = blk[0]; meta[1] = blk[1];
      break;
    }
    default: {  // IQ3_S: grid entry t covers elements 4t..4t+3 (qs[t], ninth bit t of qh); sign bit e negates element e
      device const uchar *qs = blk + 2 + 8 * j, *signs = blk + 74 + 4 * j;
      const uint qh = blk[66 + j], scale = (blk[106 + j / 2] >> (4 * (j % 2))) & 15;
      for (uint e = 0; e < 32; ++e) hi[quant_slot(e)] = (signs[e / 8] >> (e % 8)) & 1;
      const uint sign = gguf_bit_word(hi, 1, 0);
      for (uint c = 0; c < 4; ++c)
        ((device uint *)out0)[c] = uint(qs[c]) | uint(qs[4 + c]) << 8 | ((sign >> (8 * c)) & 0xFF) << 16 |
                                   ((qh >> c) & 1) << 24 | ((qh >> (4 + c)) & 1) << 25 | scale << 26;
      if (j == 0) { meta[0] = blk[0]; meta[1] = blk[1]; }
      break;
    }
  }
}
// byte copy for native embedding rows: one thread per 16 bytes
kernel void gguf_copy(device const uchar *src [[buffer(0)]], device uchar *dst [[buffer(1)]],
                    constant GgufCopyParams &p [[buffer(2)]], uint t [[thread_position_in_grid]]) {
  const uint begin = t * 16;
  if (begin >= p.bytes) return;
  if (begin + 16 <= p.bytes) *(device uint4 *)(dst + p.dst_offset + begin) = *(device const uint4 *)(src + p.src_offset + begin);
  else for (uint i = begin; i < p.bytes; ++i) dst[p.dst_offset + i] = src[p.src_offset + i];
}

// dependency kernel for serialized profiling: touching the output forces the next dispatch to wait
kernel void gguf_touch(device bfloat *y [[buffer(0)]], uint tid [[thread_position_in_grid]]) { if (tid == 0) y[0] = bfloat(float(y[0]) + 0.0f); }

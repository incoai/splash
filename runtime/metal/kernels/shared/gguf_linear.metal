// GGUF quantized GEMMs (K-quants, i-quants, Q8_0) for Apple9 and Apple10.
// Decode weights with FP32 group coefficients, then round once to the half tile,
// matching llama.cpp Metal dequantize.h / mul_mm.metal. Keep activations BF16.
// Weight planes and meta in the MDGG0001 layout (metal/abi/QuantFormat.h), decoded by kernels/common/quant_formats.h.
// Activations fp16 or bf16 [rows][K]; weights staged as fp16 in threadgroup memory; fp32 accumulation; bf16 output.
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/quant_formats.h"

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
// IQ4 value pairs for the codebook formats' threadgroup table: entry b = (value[b & 15], value[b >> 4]).
inline void gguf_init_lut(threadgroup half2 *tl, uint thread_index, uint threads) {
  for (uint i = thread_index; i < 256; i += threads) tl[i] = half2(half(kIQ4NLValues[i & 15]), half(kIQ4NLValues[i >> 4]));
  threadgroup_barrier(mem_flags::mem_threadgroup);
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

// ---------------- sg: each simdgroup stages its own Cols x KS sub-tile privately and runs matmul2d alone
template <class F, typename TA, typename TO, ushort Rows, ushort Cols, ushort KS, ushort Buffers, ushort Prefetch, ushort Ep = EpNone>
inline void sg_tile(device TA *input, device uchar *w0, device uchar *w1, device uchar *meta, device TO *output,
                    uint output_size, uint input_size, uint output_origin, threadgroup half *stage, threadgroup half2 *tl,
                    uint simd_lane, uint step_begin, uint step_end, uint out_stride = 0, uint out_offset = 0, device bfloat *aux = nullptr) {
  if (out_stride == 0) out_stride = output_size;
  constexpr ushort GPS = KS / 32, Items = Cols * GPS, IPT = (Items + 31) / 32;
  auto a = tensor(input, dextents<int, 2>{int(input_size), Rows}, array<int, 2>{1, int(input_size)});
  constexpr auto descriptor = matmul2d_descriptor(Rows, Cols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  const uint groups = input_size / 32, units = groups / F::MetaGroups;
  const uint tile = output_origin / kStorageN, tile_offset = output_origin % kStorageN;
  device uchar *tw0 = w0 + (ulong(tile) * groups * kStorageN + tile_offset) * F::P0;
  device uchar *tw1 = w1 + (ulong(tile) * groups * kStorageN + tile_offset) * F::P1;
  device uchar *tmeta = meta + (ulong(tile) * units * kStorageN + tile_offset) * F::MetaBytes;
  auto a0 = a.template slice<KS, Rows>(0, 0);
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(stage, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(stage + (Buffers > 1 ? KS * Cols : 0), dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  auto b0 = bt0.slice<KS, Cols>(0, 0), b1 = bt1.slice<KS, Cols>(0, 0);
  auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
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
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto index = acc.get_multidimensional_index(i);
    const ulong o = ulong(index[1]) * out_stride + out_offset + output_origin + index[0];
    float v = acc[i];
    if constexpr (Ep == EpResidual) v += float(aux[o]);
    if constexpr (Ep == EpUpWithGate) v = float(bfloat(v)) * silu_gate(float(aux[o]));
    output[o] = TO(v);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
}


// ---------------- pf: prefill with a shared B stage (TileN x KS, all threads dequantize), each simdgroup owns RowsPerSG rows
template <class F, typename TA, ushort RowsPerSG, ushort Simdgroups, ushort TileN, ushort KS, ushort Prefetch, ushort Ep = EpNone>
inline void pf_tile(device TA *input, device uchar *w0, device uchar *w1, device uchar *meta, device bfloat *output,
                    uint output_size, uint input_size, uint output_origin, threadgroup half *stage, threadgroup half2 *tl,
                    uint simd_lane, uint simd_group, uint out_stride = 0, uint out_offset = 0, device bfloat *aux = nullptr) {
  if (out_stride == 0) out_stride = output_size;
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
    auto a_slice = a.template slice<KS, RowsPerSG>(step * KS, 0);
    if (step & 1) operation.run(a_slice, b1, acc); else operation.run(a_slice, b0, acc);
  }
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
  if constexpr (F::Kind == QuantCodebook) gguf_init_lut(tl, simd_group * 32 + simd_lane, Threads);
#define ABUF(TA) device TA *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]], \
                 device uchar *meta [[buffer(3)]], device bfloat *output [[buffer(4)]], constant GgufParams &p [[buffer(5)]]
#define ABUFE device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]], \
                 device uchar *meta [[buffer(3)]], device bfloat *output [[buffer(4)]], device bfloat *aux [[buffer(5)]], constant GgufParams &p [[buffer(6)]]
#define IDS uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]
// decode: one simdgroup per C columns, S simdgroups per threadgroup, grid = tiles (persistent_groups >= tiles)
#define SG_K(F, f, TA, ta, R, C, S, KS, B, P)                                                             \
  kernel void sg##ta##_##f##_m##R##_c##C##_sg##S##_k##KS##_b##B##_p##P(ABUF(TA), uint group [[threadgroup_position_in_grid]], IDS) { \
    constexpr ushort Threads = S * 32; TGLUT_INIT(F)                                                      \
    threadgroup half stage[S * B * KS * C]; const uint tiles = p.output_size / (S * C), steps = p.input_size / KS; \
    for (uint tile = group; tile < tiles; tile += p.persistent_groups)                                    \
      sg_tile<F, TA, bfloat, R, C, KS, B, P>(input, w0, w1, meta, output, p.output_size, p.input_size, tile * (S * C) + simd_group * C, \
                                             stage + simd_group * (B * KS * C), tl, simd_lane, 0, steps, p.out_stride, p.out_offset); }
// split-K decode: group.y = split, p.persistent_groups = splits, fp32 partials [split][R][N]
#define SGK_K(F, f, TA, ta, R, C, S, KS, B, P)                                                            \
  kernel void sgk##ta##_##f##_m##R##_c##C##_sg##S##_k##KS##_b##B##_p##P(device TA *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], \
             device uchar *w1 [[buffer(2)]], device uchar *meta [[buffer(3)]], device float *partials [[buffer(4)]], constant GgufParams &p [[buffer(5)]], \
             uint2 group [[threadgroup_position_in_grid]], IDS) {                                          \
    constexpr ushort Threads = S * 32; TGLUT_INIT(F)                                                      \
    threadgroup half stage[S * B * KS * C]; const uint steps = p.input_size / KS, per = steps / p.persistent_groups; \
    sg_tile<F, TA, float, R, C, KS, B, P>(input, w0, w1, meta, partials + ulong(group.y) * R * p.output_size, p.output_size, p.input_size, \
        group.x * (S * C) + simd_group * C, stage + simd_group * (B * KS * C), tl, simd_lane, group.y * per, (group.y + 1) * per); }
kernel void gguf_splitk_reduce(device float *partials [[buffer(0)]], device bfloat *output [[buffer(1)]], device bfloat *aux [[buffer(2)]],
                             constant GgufReduceParams &rp [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
  const uint count = rp.rows * rp.cols; if (tid >= count) return; float s = 0.0f;
  for (uint k = 0; k < rp.splits; ++k) s += partials[ulong(k) * count + tid];
  const uint row = tid / rp.cols, col = tid % rp.cols; const ulong o = ulong(row) * (rp.out_stride ? rp.out_stride : rp.cols) + rp.out_offset + col;
  if (rp.epilogue == EpResidual) s += float(aux[o]);
  if (rp.epilogue == EpUpWithGate) s = float(bfloat(s)) * silu_gate(float(aux[o]));
  output[o] = bfloat(s);
}
// epilogue entry points (bf16 activations): residual add or silu(gate)*acc, aux in buffer(6)
#define SGE_K(F, f, EP, ep, R, C, S, KS, B, P)                                                            \
  kernel void sg##ep##_##f##_m##R##_c##C##_sg##S##_k##KS##_b##B##_p##P(ABUFE, uint group [[threadgroup_position_in_grid]], IDS) { \
    constexpr ushort Threads = S * 32; TGLUT_INIT(F)                                                      \
    threadgroup half stage[S * B * KS * C]; const uint tiles = p.output_size / (S * C), steps = p.input_size / KS; \
    for (uint tile = group; tile < tiles; tile += p.persistent_groups)                                    \
      sg_tile<F, bfloat, bfloat, R, C, KS, B, P, EP>(input, w0, w1, meta, output, p.output_size, p.input_size, tile * (S * C) + simd_group * C, \
                                             stage + simd_group * (B * KS * C), tl, simd_lane, 0, steps, p.out_stride, p.out_offset, aux); }
#define PFE_K(F, f, EP, ep, R, S, N, KS, P)                                                               \
  kernel void pf##ep##_##f##_r##R##_sg##S##_n##N##_k##KS##_p##P(ABUFE, uint2 group [[threadgroup_position_in_grid]], IDS) { \
    constexpr ushort Threads = S * 32; TGLUT_INIT(F)                                                      \
    threadgroup half stage[2 * KS * N]; const uint rs = p.out_stride ? p.out_stride : p.output_size;      \
    pf_tile<F, bfloat, R, S, N, KS, P, EP>(input + ulong(group.x) * (R * S) * p.input_size, w0, w1, meta, output + ulong(group.x) * (R * S) * rs, \
                                   p.output_size, p.input_size, group.y * N, stage, tl, simd_lane, simd_group, p.out_stride, p.out_offset, aux + ulong(group.x) * (R * S) * rs); }
#define PF_K(F, f, TA, ta, R, S, N, KS, P)                                                                \
  kernel void pf##ta##_##f##_r##R##_sg##S##_n##N##_k##KS##_p##P(ABUF(TA), uint2 group [[threadgroup_position_in_grid]], IDS) { \
    constexpr ushort Threads = S * 32; TGLUT_INIT(F)                                                      \
    threadgroup half stage[2 * KS * N];                                                                   \
    pf_tile<F, TA, R, S, N, KS, P>(input + ulong(group.x) * (R * S) * p.input_size, w0, w1, meta, output + ulong(group.x) * (R * S) * (p.out_stride ? p.out_stride : p.output_size), \
                                   p.output_size, p.input_size, group.y * N, stage, tl, simd_lane, simd_group, p.out_stride, p.out_offset); }
// ---------------- sg_accum: the decode tile loop without the store; acc is created by the caller (gguf_make_acc) so
// several formats can accumulate into the same cooperative tensor type (fused segments, gate+up).
// Initialize in the caller after construction: returning an initialized cooperative tensor
// loses its initial values on Apple9 in runtime-format kernels (also with shader validation).
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
// runtime dequantizer selection (uniform per threadgroup)
template <typename TA, ushort Rows, ushort Cols, ushort KS, ushort Buffers, ushort Prefetch, class Acc>
inline void gguf_accum_any(uint fmt, device TA *input, device uchar *w0, device uchar *w1, device uchar *meta, uint input_size, uint origin,
                         threadgroup half *stage, threadgroup half2 *tl, uint simd_lane, uint sb, uint se, thread Acc &acc) {
  switch (fmt) {
  case GGUF_FMT_Q4K: sg_accum<FmtQ4K, TA, Rows, Cols, KS, Buffers, Prefetch>(input, w0, w1, meta, input_size, origin, stage, tl, simd_lane, sb, se, acc); break;
  case GGUF_FMT_IQ4XS: sg_accum<FmtIQ4XS, TA, Rows, Cols, KS, Buffers, Prefetch>(input, w0, w1, meta, input_size, origin, stage, tl, simd_lane, sb, se, acc); break;
  case GGUF_FMT_IQ4NL: sg_accum<FmtIQ4NL, TA, Rows, Cols, KS, Buffers, Prefetch>(input, w0, w1, meta, input_size, origin, stage, tl, simd_lane, sb, se, acc); break;
  case GGUF_FMT_Q5K: sg_accum<FmtQ5K, TA, Rows, Cols, KS, Buffers, Prefetch>(input, w0, w1, meta, input_size, origin, stage, tl, simd_lane, sb, se, acc); break;
  case GGUF_FMT_Q6K: sg_accum<FmtQ6K, TA, Rows, Cols, KS, Buffers, Prefetch>(input, w0, w1, meta, input_size, origin, stage, tl, simd_lane, sb, se, acc); break;
  case GGUF_FMT_Q3K: sg_accum<FmtQ3K, TA, Rows, Cols, KS, Buffers, Prefetch>(input, w0, w1, meta, input_size, origin, stage, tl, simd_lane, sb, se, acc); break;
  case GGUF_FMT_Q80: sg_accum<FmtQ80, TA, Rows, Cols, KS, Buffers, Prefetch>(input, w0, w1, meta, input_size, origin, stage, tl, simd_lane, sb, se, acc); break;
  default: sg_accum<FmtIQ3S, TA, Rows, Cols, KS, Buffers, Prefetch>(input, w0, w1, meta, input_size, origin, stage, tl, simd_lane, sb, se, acc); break;
  }
}

// ---------------- fused segments: one dispatch over up to three column segments of different formats (decode rows)
#define SEGBUF(i, w0, w1, m) device uchar *w0 [[buffer(i)]], device uchar *w1 [[buffer(i + 1)]], device uchar *m [[buffer(i + 2)]]
#define GGUF_FUSED_K(R)                                                                                            \
  kernel void gguf_fused_m##R(device bfloat *input [[buffer(0)]], SEGBUF(1, w0a, w1a, ma), SEGBUF(4, w0b, w1b, mb), SEGBUF(7, w0c, w1c, mc), \
                       device bfloat *output [[buffer(10)]], constant GgufFusedParams &p [[buffer(11)]],       \
                       uint group [[threadgroup_position_in_grid]], IDS) {                                   \
    threadgroup half stage[2 * 2 * 32 * 32]; threadgroup half2 tl[256];                                     \
    gguf_init_lut(tl, simd_group * 32 + simd_lane, 64);                                                      \
    const uint t0 = p.cols[0] / 64, t1 = t0 + p.cols[1] / 64;                                              \
    device uchar *w0 = w0a; device uchar *w1 = w1a; device uchar *meta = ma; uint fmt = p.fmt[0], off = p.offset[0], local = group; \
    if (group >= t1) { w0 = w0c; w1 = w1c; meta = mc; fmt = p.fmt[2]; off = p.offset[2]; local = group - t1; }                    \
    else if (group >= t0) { w0 = w0b; w1 = w1b; meta = mb; fmt = p.fmt[1]; off = p.offset[1]; local = group - t0; }                \
    const uint origin = local * 64 + simd_group * 32, steps = p.input_size / 32;                            \
    threadgroup half *my = stage + simd_group * (2 * 32 * 32);                                              \
    auto acc = gguf_make_acc<bfloat, R, 32, 32>(input, p.input_size, my);                                     \
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;                                    \
    gguf_accum_any<bfloat, R, 32, 32, 2, 1>(fmt, input, w0, w1, meta, p.input_size, origin, my, tl, simd_lane, 0, steps, acc); \
    for (ushort i = 0; i < acc.get_capacity(); ++i) {                                                       \
      if (!acc.is_valid_element(i)) continue;                                                              \
      auto index = acc.get_multidimensional_index(i);                                                      \
      output[ulong(index[1]) * p.out_stride + off + origin + index[0]] = bfloat(acc[i]);                   \
    }                                                                                                       \
  }
GGUF_FUSED_K(8) GGUF_FUSED_K(16) GGUF_FUSED_K(24) GGUF_FUSED_K(32)

// ---------------- gate + up in one dispatch: output = silu(gate) * up (both rounded to bf16 first, as splash does)
#define GGUF_GATEUP_K(R)                                                                                           \
  kernel void gguf_gateup_m##R(device bfloat *input [[buffer(0)]], SEGBUF(1, gw0, gw1, gm), SEGBUF(4, uw0, uw1, um),  \
                        device bfloat *output [[buffer(7)]], constant GgufGateUpParams &p [[buffer(8)]],        \
                        uint group [[threadgroup_position_in_grid]], IDS) {                                  \
    threadgroup half stage[2 * 2 * 32 * 32]; threadgroup half2 tl[256];                                     \
    gguf_init_lut(tl, simd_group * 32 + simd_lane, 64);                                                      \
    const uint origin = group * 64 + simd_group * 32, steps = p.input_size / 32;                           \
    threadgroup half *my = stage + simd_group * (2 * 32 * 32);   /* one 8 KB stage for both passes: occupancy */ \
    auto gate = gguf_make_acc<bfloat, R, 32, 32>(input, p.input_size, my);                                    \
    for (ushort i = 0; i < gate.get_capacity(); ++i) gate[i] = 0.0f;                                   \
    gguf_accum_any<bfloat, R, 32, 32, 2, 1>(p.gate_fmt, input, gw0, gw1, gm, p.input_size, origin, my, tl, simd_lane, 0, steps, gate); \
    auto up = gguf_make_acc<bfloat, R, 32, 32>(input, p.input_size, my);                                      \
    for (ushort i = 0; i < up.get_capacity(); ++i) up[i] = 0.0f;                                     \
    gguf_accum_any<bfloat, R, 32, 32, 2, 1>(p.up_fmt, input, uw0, uw1, um, p.input_size, origin, my, tl, simd_lane, 0, steps, up); \
    for (ushort i = 0; i < gate.get_capacity(); ++i) {                                                      \
      if (!gate.is_valid_element(i)) continue;                                                             \
      auto index = gate.get_multidimensional_index(i);                                                     \
      const float g = float(bfloat(gate[i])), u = float(bfloat(up[i]));                                    \
      output[ulong(index[1]) * p.out_stride + origin + index[0]] = bfloat(silu_gate(g) * u);               \
    }                                                                                                       \
  }
GGUF_GATEUP_K(8) GGUF_GATEUP_K(16) GGUF_GATEUP_K(24) GGUF_GATEUP_K(32)

// ---------------- split-K with last-arriver reduction (per format), epilogue applied by the reducing threadgroup
template <class F, ushort Rows>
inline void gguf_splitk_tile(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta, device float *partials,
                           device atomic_uint *counters, device bfloat *output, device bfloat *aux, constant GgufSplitParams &p,
                           uint2 group, uint simd_lane, uint simd_group, threadgroup half *stage, threadgroup half2 *tl, threadgroup uint *arrival) {
  const uint steps = p.input_size / 32, per = steps / p.splits, thread_index = simd_group * 32 + simd_lane;
  const uint origin = group.x * 64 + simd_group * 32;
  threadgroup half *my = stage + simd_group * (2 * 32 * 32);
  auto acc = gguf_make_acc<bfloat, Rows, 32, 32>(input, p.input_size, my);
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
  sg_accum<F, bfloat, Rows, 32, 32, 2, 1>(input, w0, w1, meta, p.input_size, origin, my, tl, simd_lane, group.y * per, (group.y + 1) * per, acc);
  const ulong N = p.output_size;
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto index = acc.get_multidimensional_index(i);
    partials[(ulong(group.y) * Rows + index[1]) * N + origin + index[0]] = acc[i];
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (thread_index == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope::thread_scope_device);
    *arrival = atomic_fetch_add_explicit(counters + group.x, 1u, memory_order_relaxed);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope::thread_scope_device);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  if (*arrival != p.splits - 1) return;
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto index = acc.get_multidimensional_index(i);
    float total = 0.0f;
    for (uint s = 0; s < p.splits; ++s)
      total += s == group.y ? acc[i] : partials[(ulong(s) * Rows + index[1]) * N + origin + index[0]];
    const ulong o = ulong(index[1]) * p.out_stride + p.out_offset + origin + index[0];
    if (p.epilogue == GGUF_EPILOGUE_RESIDUAL) total += float(aux[o]);
    if (p.epilogue == GGUF_EPILOGUE_UP_WITH_GATE) total = float(bfloat(total)) * silu_gate(float(aux[o]));
    output[o] = bfloat(total);
  }
  if (thread_index == 0) atomic_store_explicit(counters + group.x, 0u, memory_order_relaxed);
}
#define GGUF_SPLITK_K(F, f, R)                                                                                      \
  kernel void gguf_splitk_##f##_m##R(device bfloat *input [[buffer(0)]], SEGBUF(1, w0, w1, meta), device float *partials [[buffer(4)]], \
                             device atomic_uint *counters [[buffer(5)]], device bfloat *output [[buffer(6)]], device bfloat *aux [[buffer(7)]], \
                             constant GgufSplitParams &p [[buffer(8)]], uint2 group [[threadgroup_position_in_grid]], IDS) { \
    threadgroup half stage[2 * 2 * 32 * 32]; threadgroup half2 tl[F::Kind == QuantCodebook ? 256 : 1]; threadgroup uint arrival; \
    if constexpr (F::Kind == QuantCodebook) gguf_init_lut(tl, simd_group * 32 + simd_lane, 64);               \
    gguf_splitk_tile<F, R>(input, w0, w1, meta, partials, counters, output, aux, p, group, simd_lane, simd_group, stage, tl, &arrival); }
#define GGUF_SPLITK_SET(F, f) GGUF_SPLITK_K(F, f, 8) GGUF_SPLITK_K(F, f, 16) GGUF_SPLITK_K(F, f, 24) GGUF_SPLITK_K(F, f, 32)
GGUF_SPLITK_SET(FmtQ4K, q4k) GGUF_SPLITK_SET(FmtIQ4XS, iq4xs) GGUF_SPLITK_SET(FmtIQ4NL, iq4nl) GGUF_SPLITK_SET(FmtQ5K, q5k)
GGUF_SPLITK_SET(FmtQ6K, q6k) GGUF_SPLITK_SET(FmtQ3K, q3k) GGUF_SPLITK_SET(FmtQ80, q80) GGUF_SPLITK_SET(FmtIQ3S, iq3s)

#define PROD_SET(F, f)                                                                                    \
  SG_K(F, f, bfloat, a, 8, 32, 2, 32, 2, 1) SG_K(F, f, bfloat, a, 16, 32, 2, 32, 2, 1) SG_K(F, f, bfloat, a, 24, 32, 2, 32, 2, 1) SG_K(F, f, bfloat, a, 32, 32, 2, 32, 2, 1) \
  SGE_K(F, f, EpResidual, r, 8, 32, 2, 32, 2, 1) SGE_K(F, f, EpResidual, r, 16, 32, 2, 32, 2, 1) SGE_K(F, f, EpResidual, r, 24, 32, 2, 32, 2, 1) SGE_K(F, f, EpResidual, r, 32, 32, 2, 32, 2, 1) \
  SGE_K(F, f, EpUpWithGate, g, 8, 32, 2, 32, 2, 1) SGE_K(F, f, EpUpWithGate, g, 16, 32, 2, 32, 2, 1) SGE_K(F, f, EpUpWithGate, g, 24, 32, 2, 32, 2, 1) SGE_K(F, f, EpUpWithGate, g, 32, 32, 2, 32, 2, 1) \
  PF_K(F, f, bfloat, a, 32, 4, 64, 64, 1) PFE_K(F, f, EpResidual, r, 32, 4, 64, 64, 1) PFE_K(F, f, EpUpWithGate, g, 32, 4, 64, 64, 1)
PROD_SET(FmtQ4K, q4k)
PROD_SET(FmtIQ4XS, iq4xs)
PROD_SET(FmtIQ4NL, iq4nl)
PROD_SET(FmtQ5K, q5k)
PROD_SET(FmtQ6K, q6k)
PROD_SET(FmtQ3K, q3k)
PROD_SET(FmtQ80, q80)
PROD_SET(FmtIQ3S, iq3s)

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
// permute the K columns (in 128-wide head blocks) of a bf16 activation: out[row][h*128+e] = in[row][perm[h]*128+e]
kernel void gguf_permute_heads(device const bfloat *input [[buffer(0)]], device bfloat *output [[buffer(1)]], device const uint *perm [[buffer(2)]],
                          constant GgufPermuteParams &p [[buffer(3)]], uint index [[thread_position_in_grid]]) {
  if (index >= p.rows * p.width) return;
  const uint row = index / p.width, col = index % p.width, h = col / p.block, e = col % p.block;
  output[index] = input[ulong(row) * p.width + perm[h] * p.block + e];
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

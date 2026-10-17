#pragma once
#include "metal/kernels/common/quant_formats.h"

// The GGUF tiles' dequantization (kernels/shared/gguf_linear.metal and kernels/shared/moe_gguf.metal, through
// kernels/common/gguf_staged_tile.h and kernels/common/gguf_column_tile.h), shared with the dequantization test.
// Includers set `#pragma clang fp reassociate(off)` first, so the source order of float operations holds here.
// One thread decodes one column's group of 32 (kernels/common/quant_formats.h) to half, each value rounded once, and
// hands it to sink(c, lo, hi) by chunk: lo holds chunk c's pairs 0, 1 (elements 4c..4c+3), hi its pairs 2, 3
// (elements 16+4c..16+4c+3). The staged tiles store the group into their stage (dequant32), the column tile keeps it
// in registers.
template <class F>
inline half2 staged_linear(uint pair, float s, float m) {
  // 0x6400 is half 1024, whose ulp is 1: or-ing a code into its mantissa makes 1024 + code, exactly.
  const float2 code = float2(as_type<half2>(pair | 0x64006400u) - half2(half(1024 + F::Zero)));
  if constexpr (F::Zero) return half2(code * s);
  else return half2(fma(code, float2(s), float2(m)));
}
template <class F, class Sink>
inline void dequant32_chunks(typename F::Payload w, typename F::Meta meta, ushort j, threadgroup half2 *tl, Sink sink) {
  QuantCoef k;
  if constexpr (F::ScaleInChunk) k = F::coef(meta, F::chunk(w, 0)); else k = F::coef(meta, j);
#pragma unroll
  for (ushort c = 0; c < 4; ++c) {
    const typename F::Chunk q = F::chunk(w, c);
    half4 lo, hi;
    if constexpr (F::Kind == QuantLinear) {
      const uint4 p = F::codes(q);
      lo = half4(staged_linear<F>(p.x, k.s.x, k.m.x), staged_linear<F>(p.y, k.s.x, k.m.x));
      hi = half4(staged_linear<F>(p.z, k.s.y, k.m.y), staged_linear<F>(p.w, k.s.y, k.m.y));
    } else if constexpr (F::Kind == QuantCodebook) {   // value * s in Scale, then to half, as the reference rounds it
      typedef typename F::Scale S;
      const uchar4 b = as_type<uchar4>(F::indices(q));
      lo = half4(half2(vec<S, 2>(tl[b.x]) * S(k.s.x)), half2(vec<S, 2>(tl[b.y]) * S(k.s.x)));
      hi = half4(half2(vec<S, 2>(tl[b.z]) * S(k.s.y)), half2(vec<S, 2>(tl[b.w]) * S(k.s.y)));
    } else if constexpr (F::Kind == QuantInt8) {
      typedef typename F::Scale S;
      const uint2 v = F::values(q);
      lo = half4(vec<S, 4>(as_type<char4>(v.x)) * S(k.s.x));
      hi = half4(vec<S, 4>(as_type<char4>(v.y)) * S(k.s.y));
    } else if constexpr (F::Kind == QuantFloat8) {   // exact halves times s in float, then to half, as the reference rounds it
      const uint2 v = F::values(q);
      lo = half4(float4(quant_e4m3_word(v.x)) * k.s.x);
      hi = half4(float4(quant_e4m3_word(v.y)) * k.s.y);
    } else if constexpr (F::Kind == QuantFloat16) {   // the values times s in float, then to half
      typedef vec<typename F::Value, 2> V2;
      const uint4 v = F::values(q);
      lo = half4(float4(float2(as_type<V2>(v.x)), float2(as_type<V2>(v.y))) * k.s.x);
      hi = half4(float4(float2(as_type<V2>(v.z)), float2(as_type<V2>(v.w))) * k.s.y);
    } else {
      const uint2 g = F::grid(q); const uint s = F::signs(q);
      lo = half4(float4(as_type<uchar4>(g.x)) * k.s.x);
      hi = half4(float4(as_type<uchar4>(g.y)) * k.s.y);
      lo = select(lo, -lo, bool4(s & 1, s & 2, s & 4, s & 8));
      hi = select(hi, -hi, bool4(s & 16, s & 32, s & 64, s & 128));
    }
    sink(c, lo, hi);
  }
}
// The group into a stage column: chunk c's pairs 0, 1 at dst + 4c, pairs 2, 3 at dst + 16 + 4c.
template <class F>
inline void dequant32(typename F::Payload w, typename F::Meta meta, ushort j, threadgroup half2 *tl,
                      threadgroup half *dst) {
  dequant32_chunks<F>(w, meta, j, tl, [&](ushort c, half4 lo, half4 hi) __attribute__((always_inline)) {
    *((threadgroup half4 *)(dst + 4 * c)) = lo; *((threadgroup half4 *)(dst + 16 + 4 * c)) = hi;
  });
}

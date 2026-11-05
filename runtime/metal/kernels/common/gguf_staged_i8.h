#pragma once
#include "metal/abi/MoE.h"
#include "metal/kernels/common/gguf_staged_tile.h"

// Int8 activations for the staged decode expert passes (ops::MoeInt8; splash serve --moe-int8, off by default).
// Includers set `#pragma clang fp reassociate(off)` first.
//
// Activations: a row (or a 32-column block of the down input) is s * q with s = max |x| / 127 in fp32 and
// q = clamp(rint(x / s), -127, 127) as int8; MOE_A8_POW2 (tests) takes the power of two at or above max |x| / 127
// instead, so that s * q is exact in bf16 and a bf16 pass fed bf16(s q) multiplies the same values as the int8 pass.
inline float moe_a8_scale(float amax, uint mode) {
  const float s = amax / 127.0f;
  if (!(mode & MOE_A8_POW2) || s == 0.0f) return s;
  int e;
  static_cast<void>(frexp(s, e));   // s = m 2^e, m in [0.5, 1): 2^e >= s
  return ldexp(1.0f, e);
}
inline int moe_a8_code(float x, float s) { return s > 0.0f ? int(clamp(rint(x / s), -127.0f, 127.0f)) : 0; }

// The formats whose groups decode to int8 exactly (kernels/common/quant_formats.h): value = s * c + m with c an int8
// (QuantGrid's signed grid magnitude, a codebook format's integer table value, QuantInt8's byte, QuantLinear's
// code - Zero) and m nonzero only for QuantLinear formats without a zero point (a min, applied as m * the activations'
// sum over its 16 inputs). The host enables them one by one (MoeInt8::formats).
template <class F> constexpr bool moe_i8_format() {
  return F::Id == GGUF_FMT_IQ2S || F::Id == GGUF_FMT_IQ2XS || F::Id == GGUF_FMT_IQ3S || F::Id == GGUF_FMT_IQ3XXS
      || F::Id == GGUF_FMT_IQ4XS
      || F::Id == GGUF_FMT_Q6K || F::Id == GGUF_FMT_Q80
      || F::Id == GGUF_FMT_Q2K || F::Id == GGUF_FMT_Q4K || F::Id == GGUF_FMT_Q5K
      ;
}
template <class F> constexpr bool moe_i8_min() { return F::Kind == QuantLinear && F::Zero == 0; }
// The inputs one int8 coefficient covers: the format's Group, but 32 for Q2_K, whose 4-bit sub-scales the stage holds
// (code * sc <= 45), leaving d per 256 and the two 16-input mins.
template <class F> constexpr ushort moe_i8_group() { return F::Id == GGUF_FMT_Q2K ? 32 : (F::Group >= 32 ? 32 : 16); }

// One column's group of 32 as int8 in element order (dequant32's order, without its half rounding) and the group's
// coefficients: value of element e = k.s.x * c + k.m.x for e < 16, k.s.y * c + k.m.y for the rest.
template <class F, class Sink>
inline void dequant32_i8_chunks(typename F::Payload w, typename F::Meta meta, ushort j, threadgroup half2 *tl,
                                thread QuantCoef &k, Sink sink) {
  if constexpr (F::ScaleInChunk) k = F::coef(meta, F::chunk(w, 0)); else k = F::coef(meta, j);
  short2 sc = short2(1);
  if constexpr (F::Id == GGUF_FMT_Q2K) {   // integer sub-scales into the stage, d alone into the coefficient
    const uint pair = quant_scale_pair(meta.sc, j);
    sc = short2(short(pair & 0xF), short((pair >> 8) & 0xF));
    k.s = float2(float(as_type<half>(ushort(meta.dm & 0xFFFF))));
  }
#pragma unroll
  for (ushort c = 0; c < 4; ++c) {
    const typename F::Chunk q = F::chunk(w, c);
    char4 lo, hi;
    if constexpr (F::Kind == QuantGrid) {
      const uint2 g = F::grid(q); const uint s = F::signs(q);
      lo = char4(as_type<uchar4>(g.x));
      hi = char4(as_type<uchar4>(g.y));
      lo = select(lo, -lo, bool4(s & 1, s & 2, s & 4, s & 8));
      hi = select(hi, -hi, bool4(s & 16, s & 32, s & 64, s & 128));
    } else if constexpr (F::Kind == QuantCodebook) {
      const uchar4 b = as_type<uchar4>(F::indices(q));
      lo = char4(half4(tl[b.x], tl[b.y]));
      hi = char4(half4(tl[b.z], tl[b.w]));
    } else if constexpr (F::Kind == QuantInt8) {
      const uint2 v = F::values(q);
      lo = as_type<char4>(v.x);
      hi = as_type<char4>(v.y);
    } else {
      static_assert(F::Kind == QuantLinear, "QuantFloat8 and QuantFloat16 have no int8 form");
      const uint4 p = F::codes(q);
      lo = char4((short4(ushort4(p.x & 0xFFFFu, p.x >> 16, p.y & 0xFFFFu, p.y >> 16)) - short(F::Zero)) * sc.x);
      hi = char4((short4(ushort4(p.z & 0xFFFFu, p.z >> 16, p.w & 0xFFFFu, p.w >> 16)) - short(F::Zero)) * sc.y);
    }
    sink(c, lo, hi);
  }
}

// Threadgroup bytes of one simdgroup's int8 walk (within its kStagedSimdgroupStage halves): two int8 stages of
// 32 x 32, and in three slots (a step's fold runs during the next step) the columns' coefficients s and mins m (float2
// per column: the two 16-input halves), the activation sums of each row's halves (min formats) and the rows' step
// scales (the down input's).
constant constexpr uint kI8Stage = 2 * 32 * 32, kI8Coef = 3 * 32 * 8, kI8Sums = 3 * 16 * 4;
static_assert(kI8Stage + 2 * kI8Coef + kI8Sums + 3 * 16 * 4 <= kStagedSimdgroupStage * 2, "the int8 walk fits a stage");

// The elements a lane holds of a Rows x 32 int destination: at most 2 Rows.
template <ushort Rows> struct I8Elements {
  float fa[2 * Rows];
  ushort er[2 * Rows], ec[2 * Rows];
  bool ev[2 * Rows];
};

// One simdgroup's 32 columns (lane l stages column origin + l) of a Rows-row tile over all of K, on int8 rows x8 (K
// apart): per 32-input step one int8 stage, one K = 32 int8 matmul2d into a fresh int32 accumulator (two of K = 16
// where the format's halves have their own coefficients), folded into fp32 as acc * s (+ m * the rows' sums) per
// column, times the row's step scale (StepScales: ascale[row * K / 32 + step], the down input's per-32 scales). The
// row's whole-K scale, if any, is the caller's. Stages alternate per step behind one simdgroup barrier, as the half
// tile's; both halves' matmuls run before the step's fold. (Folding a step after the next step's matmuls, on
// alternating accumulators, ran 1.4-2.3 ms per 35B B1 cycle slower on the M6.) |acc| <= 32 * 127 * 127 < 2^19, so each
// fold is exact in fp32 where the coefficients are short.
template <class F, ushort Rows, bool StepScales>
inline void staged_i8_accumulate(device int8_t *x8, device const float *ascale, uint K, device uchar *w0,
                                 device uchar *w1, device uchar *meta, uint origin, threadgroup uchar *mem,
                                 threadgroup half2 *tl, uint lane, thread I8Elements<Rows> &e) {
  constexpr ushort Cols = 32, CAP = 2 * Rows, KS = moe_i8_group<F>(), Halves = 32 / KS;
  constexpr bool Min = moe_i8_min<F>();
  threadgroup int8_t *stage = reinterpret_cast<threadgroup int8_t *>(mem);
  threadgroup float2 *coef = reinterpret_cast<threadgroup float2 *>(mem + kI8Stage);
  threadgroup float2 *mins = reinterpret_cast<threadgroup float2 *>(mem + kI8Stage + kI8Coef);
  threadgroup int *sums = reinterpret_cast<threadgroup int *>(mem + kI8Stage + 2 * kI8Coef);
  threadgroup float *steps_scale = reinterpret_cast<threadgroup float *>(mem + kI8Stage + 2 * kI8Coef + kI8Sums);
  auto at = tensor<device int8_t, dextents<int, 2>, tensor_inline>(x8, dextents<int, 2>{int(K), Rows},
                                                                   array<int, 2>{1, int(K)});
  auto bt0 = tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline>(stage, dextents<int, 2>{32, Cols},
                                                                         array<int, 2>{1, 32});
  auto bt1 = tensor<threadgroup int8_t, dextents<int, 2>, tensor_inline>(stage + 32 * Cols, dextents<int, 2>{32, Cols},
                                                                         array<int, 2>{1, 32});
  // Each step's matmuls overwrite their accumulators (mode multiply): the fold never clears them.
  constexpr auto descriptor = matmul2d_descriptor(Rows, Cols, KS, false, true, false,
                                                  matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<1>> op;
  auto a0 = at.template slice<KS, Rows>(0, 0);
  auto b0 = bt0.template slice<KS, Cols>(0, 0), b1 = bt1.template slice<KS, Cols>(0, 0);
  auto b0h = bt0.template slice<KS, Cols>(Halves == 2 ? KS : 0, 0), b1h = bt1.template slice<KS, Cols>(Halves == 2 ? KS : 0, 0);
  // Accumulators: [step parity][half].
  auto p0 = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), int>();
  auto p0h = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), int>();
#pragma unroll
  for (ushort i = 0; i < CAP; ++i) {
    e.fa[i] = 0.0f; e.er[i] = 0; e.ec[i] = 0; e.ev[i] = false;
    if (i < p0.get_capacity()) {
      p0[i] = 0; p0h[i] = 0;
      if (p0.is_valid_element(i)) {
        const auto ix = p0.get_multidimensional_index(i);
        e.er[i] = ix[1]; e.ec[i] = ix[0]; e.ev[i] = true;
      }
    }
  }
  // Whether the lane's elements come in pairs of one row's even and odd column (then one 16-byte load serves both
  // columns' coefficients).
  bool paired = p0.get_capacity() % 2 == 0;
#pragma unroll
  for (ushort i = 0; i + 1 < CAP; i += 2)
    if (i + 1 < p0.get_capacity())
      paired = paired && e.ev[i] && e.ev[i + 1] && e.er[i] == e.er[i + 1] && e.ec[i + 1] == e.ec[i] + 1 && !(e.ec[i] & 1);
  threadgroup float4 *coef4 = reinterpret_cast<threadgroup float4 *>(coef);
  // Folds the accumulators of a step whose coefficients are in slot `slot`.
  const auto fold = [&](thread auto &acc, thread auto &acch, uint slot) __attribute__((always_inline)) {
    float4 cc = 0.0f;
#pragma unroll
    for (ushort i = 0; i < CAP; ++i) {
      if (i >= acc.get_capacity()) break;
      if (paired && !(i & 1)) cc = coef4[(slot * 32 + e.ec[i]) >> 1];
      const float2 c = paired ? ((i & 1) ? cc.zw : cc.xy) : coef[slot * 32 + e.ec[i]];
      float v;
      if constexpr (Halves == 2) {
        float lo = float(acc[i]) * c.x, hi = float(acch[i]) * c.y;
        if constexpr (Min) {
          const float2 mn = mins[slot * 32 + e.ec[i]];
          lo = lo + mn.x * float(sums[slot * 16 + 2 * e.er[i]]);
          hi = hi + mn.y * float(sums[slot * 16 + 2 * e.er[i] + 1]);
        }
        v = lo + hi;
      } else {
        v = float(acc[i]) * c.x;
        if constexpr (Min) {
          const float2 mn = mins[slot * 32 + e.ec[i]];
          v = v + mn.x * float(sums[slot * 16 + 2 * e.er[i]]) + mn.y * float(sums[slot * 16 + 2 * e.er[i] + 1]);
        }
      }
      if constexpr (StepScales) v = v * steps_scale[slot * 16 + e.er[i]];
      e.fa[i] = e.fa[i] + v;
    }
  };
  const uint groups = K / 32, units = groups / F::MetaGroups;
  const uint plane_tile = origin / QUANT_TILE_ROWS, plane_row = origin % QUANT_TILE_ROWS + lane;
  device uchar *tw0 = w0 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P0;
  device uchar *tw1 = w1 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P1;
  device uchar *tmeta = meta + (ulong(plane_tile) * units * QUANT_TILE_ROWS + plane_row) * F::MetaBytes;
  typename F::Payload packed = F::load(tw0, tw1);
  typename F::Meta hdr = F::loadMeta(tmeta);
  uint slot = 0;
  for (uint step = 0; step < groups; ++step) {
    const uint b = step & 1;
    threadgroup int8_t *dst = stage + b * (32 * Cols) + lane * 32;
    QuantCoef k;
    dequant32_i8_chunks<F>(packed, hdr, ushort(step % F::MetaGroups), tl, k,
                           [&](ushort c, char4 lo, char4 hi) __attribute__((always_inline)) {
                             *((threadgroup char4 *)(dst + 4 * c)) = lo;
                             *((threadgroup char4 *)(dst + 16 + 4 * c)) = hi;
                           });
    coef[slot * 32 + lane] = k.s;
    if constexpr (Min) {
      mins[slot * 32 + lane] = k.m;
      if (lane < 2 * Rows) {
        const uint4 v = *((device const uint4 *)(x8 + ulong(lane >> 1) * K + step * 32 + (lane & 1) * 16));
        int s = 0;
#pragma unroll
        for (ushort i = 0; i < 4; ++i) {
          const int4 c = int4(as_type<char4>(v[i]));
          s += c.x + c.y + c.z + c.w;
        }
        sums[slot * 16 + lane] = s;
      }
    }
    if constexpr (StepScales) {
      if (lane < Rows) steps_scale[slot * 16 + lane] = ascale[ulong(lane) * groups + step];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (step + 1 < groups) {
      packed = F::load(tw0 + ulong(step + 1) * QUANT_TILE_ROWS * F::P0, tw1 + ulong(step + 1) * QUANT_TILE_ROWS * F::P1);
      if ((step + 1) / F::MetaGroups != step / F::MetaGroups)
        hdr = F::loadMeta(tmeta + ulong((step + 1) / F::MetaGroups) * QUANT_TILE_ROWS * F::MetaBytes);
    }
    auto a_lo = at.template slice<KS, Rows>(step * 32, 0);
    auto a_hi = at.template slice<KS, Rows>(step * 32 + (Halves == 2 ? KS : 0), 0);
    if (b) {
      op.run(a_lo, b1, p0);
      if constexpr (Halves == 2) op.run(a_hi, b1h, p0h);
    } else {
      op.run(a_lo, b0, p0);
      if constexpr (Halves == 2) op.run(a_hi, b0h, p0h);
    }
    fold(p0, p0h, slot);
    slot = slot == 2 ? 0 : slot + 1;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);   // the stage may be reused
}

// GGUF decode on Apple9's register simdgroup_matrix path. The 8x8x8 MMA runs
// on the FP32 pipe there, so the kernel keeps every other FP32 operation to
// the minimum a group scale needs, and does the rest on the integer pipe:
// - weights enter the MMA as exact bf16: 128 + code for linear codes with a
//   min, 160 + code - zero for linear codes with a zero point (one add), or
//   the codebook, int8 or grid value;
// - one MMA chain per coefficient group, closed by the fp32 epilogue
//   s * chain + b * sum (b = m - 128 s, formats with a min) or s * chain with
//   the chain seeded by -160 * sum from the table (formats with a zero point);
// - coefficients decoded once per simdgroup into threadgroup memory;
// - L request lanes per threadgroup share the weight operands and coefficients;
//   each lane's chains and epilogue run as with L = 1, so its result does not
//   depend on the batch width.
// Weights: the GGUF image (metal/abi/QuantFormat.h), read by chunk: lane c of
// a fragment reads chunk c of each group. Activations: the Table16 table and
// row sums (kernels/common/gguf_sgmatrix.h).
#pragma clang fp reassociate(off)
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/gguf_sgmatrix.h"
#include "metal/kernels/common/moe_expert_slab.h"
#include "metal/kernels/common/quant_formats.h"
#include "metal/kernels/common/split_reduce.h"

namespace gguf_sg {

// The code offset of a linear format (value formats have none).
template <QuantKind K, class F> struct CodeZero { enum : uint { Value = 0 }; };
template <class F> struct CodeZero<QuantLinear, F> { enum : uint { Value = F::Zero }; };

template <class F> struct Shape {
  enum : uint {
    Linear = F::Kind == QuantLinear,
    HasMin = Linear && CodeZero<F::Kind, F>::Value == 0,      // Q4_K, Q5_K: s code + m
    ZeroPoint = Linear && !HasMin,                            // Q6_K, Q3_K: s (code - zero)
    // bf16 bits of the operand of code 0: 128, or 160 - zero with a zero point
    Operand = 0x4300 + (ZeroPoint ? q16sg::kZeroPointOffset - 128 - CodeZero<F::Kind, F>::Value : 0),
    CG = ZeroPoint ? 2 : 1,                                   // coefficient groups per 32 inputs
    UnitSpans = F::MetaGroups == 8 ? 4 : 1,                   // spans decoded per coefficient unit
    J = 2 * UnitSpans * CG,                                   // coefficients per column and unit
  };
};
template <class F> using Coef = metal::conditional_t<Shape<F>::HasMin != 0, float2, float>;

// The exact bf16 operand of pair f of a chunk.
template <class F>
inline bfloat2 operand(typename F::Chunk ch, uint f, threadgroup const bfloat2 *lut) {
  typedef Shape<F> S;
  if constexpr (F::Kind == QuantLinear) {
    return as_type<bfloat2>(F::codes(ch)[f] + S::Operand * 0x00010001u);
  } else if constexpr (F::Kind == QuantCodebook) {
    return lut[(F::indices(ch) >> (8 * f)) & 0xFFu];
  } else if constexpr (F::Kind == QuantInt8) {
    const uint2 v = F::values(ch);
    return bfloat2(float2(as_type<char2>(ushort((f < 2 ? v.x : v.y) >> (16 * (f & 1))))));
  } else {
    const uint2 g = F::grid(ch);
    const uchar4 m = as_type<uchar4>(f < 2 ? g.x : g.y);
    const uint s = F::signs(ch) >> (2 * f);
    const float2 v = float2(m[2 * (f & 1)], m[2 * (f & 1) + 1]);
    return bfloat2(select(v, -v, bool2(s & 1u, s & 2u)));
  }
}

// What coefficient j of a column in coefficient unit u is decoded from: its
// meta unit and, for IQ3_S, chunk 0 of its group (which holds the scale).
template <class F> struct CoefSource {
  typename F::Meta meta;
  typename F::Chunk chunk;
};
template <class F>
inline CoefSource<F> coefficient_source(device uchar *w0, device uchar *w1, device uchar *meta, uint tile,
                                        uint groups, uint column, uint u, uint j) {
  typedef Shape<F> S;
  const uint g = u * 2 * S::UnitSpans + j / S::CG;
  const uint units = groups / F::MetaGroups;
  CoefSource<F> src;
  src.meta = F::loadMeta(meta + ((ulong(tile) * units + g / F::MetaGroups) * 256 + column) * F::MetaBytes);
  if constexpr (F::Kind == QuantGrid) {
    const ulong at = (ulong(tile) * groups + g) * 256 + column;
    src.chunk = F::loadChunk(w0 + at * F::P0, w1 + at * F::P1, 0);
  }
  return src;
}
// Coefficient j of a column in coefficient unit u: group gi = j / CG of the
// unit, 16-group half h = j % CG. Formats with a min return (s, m - 128 s).
template <class F> inline Coef<F> coefficient(CoefSource<F> src, uint u, uint j) {
  typedef Shape<F> S;
  const uint g = u * 2 * S::UnitSpans + j / S::CG, h = j % S::CG;
  QuantCoef k;
  if constexpr (F::Kind == QuantGrid) k = F::coef(src.meta, src.chunk);
  else k = F::coef(src.meta, ushort(g % F::MetaGroups));
  if constexpr (S::HasMin) return float2(k.s.x, fma(-128.0f, k.s.x, k.m));
  else return h ? k.s.y : k.s.x;
}

inline float silu(float g) { return g / (1.0f + fast::exp2(-1.44269504089f * g)); }

// One threadgroup: 4 simdgroups x 16 columns = 64 columns of one segment,
// L request lanes of eight rows, one K partition (tg.y) of `splits`.
template <class F, uint L, uint Ep>
inline void decode(device const bfloat *table, device const float *sums, device uchar *w0,
                   device uchar *w1, device uchar *meta, device bfloat *out,
                   device coherent(device) float *partials, device atomic_uint *counters,
                   device const bfloat *aux, const GgufDecodeParams p, uint2 tg, uint tid, uint sg,
                   uint lane, threadgroup const bfloat2 *lut, threadgroup Coef<F> *coefs,
                   threadgroup uint *arrival) {
  typedef Shape<F> S;
  typedef Coef<F> C;
  constexpr uint FG = 4 / S::CG;  // fragments per coefficient group
  const uint K = p.input_size, spans = K / 64, groups = K / 32, splits = p.splits;
  const ulong tileSums = q16sg::sums_per_tile(K);
  const uint units = spans / S::UnitSpans;
  const uint u0 = tg.y * units / splits, u1 = (tg.y + 1) * units / splits;
  const q4sg::Lane l = q4sg::lane_map(lane);
  const uint fm = l.fm, fn = l.fn, c = fn / 2;
  const uint base = tg.x * 64 + sg * 16, tile = base / 256, col0 = base % 256;
  threadgroup C *cu = coefs + sg * 16 * S::J;

  // Weight streams: lane c reads chunk c of each group; a span's two groups
  // of one column are 256 payloads apart.
  const uint first = u0 * S::UnitSpans;
  device uchar *p0[2], *p1[2];
#pragma unroll
  for (uint nf = 0; nf < 2; ++nf) {
    const ulong at = (ulong(tile) * groups + 2 * first) * 256 + col0 + nf * 8 + fm;
    p0[nf] = w0 + at * F::P0;
    p1[nf] = w1 + at * F::P1;
  }
  typename F::Chunk cur[2][2], nxt[2][2];
  const auto load = [&](thread typename F::Chunk (&ch)[2][2]) __attribute__((always_inline)) {
#pragma unroll
    for (uint nf = 0; nf < 2; ++nf) {
      ch[nf][0] = F::loadChunk(p0[nf], p1[nf], c);
      ch[nf][1] = F::loadChunk(p0[nf] + 256 * F::P0, p1[nf] + 256 * F::P1, c);
    }
  };
  device const vec<bfloat, 8> *xt =
      reinterpret_cast<device const vec<bfloat, 8> *>(table + ulong(first) * q4sg::kXtPerGroup);
  device const float *seeds = sums + ulong(first) * 32;
  device const float *s32 = sums + q16sg::sums32_offset(K) + ulong(first) * 16;

  float2 acc[L][2];
#pragma unroll
  for (uint r = 0; r < L; ++r) acc[r][0] = acc[r][1] = float2(0);
  load(cur);
  for (uint u = u0; u < u1; ++u) {
    // This unit's coefficients for the simdgroup's 16 columns, each decoded
    // once. Every source load is issued before the first decode waits for one:
    // on the 40-core M3 the zero-point formats (16 coefficients per column and
    // unit) gain at one to four lanes, Q6_K 5120x17408 2.5/3.8/3.1/1.1% and
    // Q3_K 7.6/4.3/3.3/2.2%; the other formats stay within 0.7%.
    constexpr uint I = (16 * S::J + 31) / 32;
    CoefSource<F> src[I];
#pragma unroll
    for (uint i = 0; i < I; ++i) {
      const uint e = lane + 32 * i;
      if (e < 16 * S::J) src[i] = coefficient_source<F>(w0, w1, meta, tile, groups, col0 + (e & 15), u, e >> 4);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint i = 0; i < I; ++i) {
      const uint e = lane + 32 * i;
      if (e < 16 * S::J) cu[e] = coefficient<F>(src[i], u, e >> 4);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint us = 0; us < S::UnitSpans; ++us) {
      const bool more = us + 1 < S::UnitSpans || u + 1 < u1;
#pragma unroll
      for (uint nf = 0; nf < 2; ++nf) {
        p0[nf] += 512 * F::P0;
        p1[nf] += 512 * F::P1;
      }
      if (more) load(nxt);
#pragma unroll
      for (uint q = 0; q < 2; ++q) {
        bfloat2 a[2][4];
        C cs[S::CG][2];
#pragma unroll
        for (uint nf = 0; nf < 2; ++nf) {
#pragma unroll
          for (uint f = 0; f < 4; ++f) a[nf][f] = operand<F>(cur[nf][q], f, lut);
#pragma unroll
          for (uint h = 0; h < S::CG; ++h) cs[h][nf] = cu[((us * 2 + q) * S::CG + h) * 16 + nf * 8 + fm];
        }
#pragma unroll
        for (uint r = 0; r < L; ++r) {
          const vec<bfloat, 8> bq = xt[ulong(r) * K + (8 * q + fm) * 4 + c];
          float2 sum = float2(0), seed[S::CG];
          if (S::HasMin) sum = *(device const float2 *)(s32 + ulong(r) * tileSums + q * 8 + fn);
#pragma unroll
          for (uint h = 0; h < S::CG; ++h)
            seed[h] = S::ZeroPoint ? *(device const float2 *)(seeds + ulong(r) * tileSums + (2 * q + h) * 8 + fn)
                                   : float2(0);
#pragma unroll
          for (uint h = 0; h < S::CG; ++h) {
#pragma unroll
            for (uint nf = 0; nf < 2; ++nf) {
              float2 dot = seed[h];
#pragma unroll
              for (uint f = h * FG; f < (h + 1) * FG; ++f)
                q4sg::mma_acc<bfloat>(dot, a[nf][f], reinterpret_cast<thread const bfloat2 *>(&bq)[f]);
              if constexpr (S::HasMin) {
                acc[r][nf] = fma(dot, cs[h][nf].x, acc[r][nf]);
                acc[r][nf] = fma(sum, cs[h][nf].y, acc[r][nf]);
              } else {
                acc[r][nf] = fma(dot, float(cs[h][nf]), acc[r][nf]);
              }
            }
          }
        }
      }
      xt += 64;
      seeds += 32;
      s32 += 16;
      if (more) {
#pragma unroll
        for (uint nf = 0; nf < 2; ++nf) cur[nf][0] = nxt[nf][0], cur[nf][1] = nxt[nf][1];
      }
    }
  }
  if (splits > 1) {
    // Partials [lane][split][8 rows][destination columns], one counter per
    // 64 destination columns: the segments of a projection never share them.
    const uint stride = p.out_stride;
    device atomic_uint *counter = counters + (p.out_offset + base) / 64;
#pragma unroll
    for (uint r = 0; r < L; ++r)
#pragma unroll
      for (uint nf = 0; nf < 2; ++nf) {
        device coherent(device) float *slot =
            partials + ((ulong(r) * splits + tg.y) * 8 + fn) * stride + p.out_offset + base + nf * 8 + fm;
        slot[0] = acc[r][nf].x;
        slot[stride] = acc[r][nf].y;
      }
    if (!split_arrive_last(counter, splits, tid, arrival)) return;
#pragma unroll
    for (uint r = 0; r < L; ++r)
#pragma unroll
      for (uint nf = 0; nf < 2; ++nf)
        acc[r][nf] = split_sum(acc[r][nf], tg.y, splits, [&](uint s) {
          device coherent(device) float *slot =
              partials + ((ulong(r) * splits + s) * 8 + fn) * stride + p.out_offset + base + nf * 8 + fm;
          return float2(slot[0], slot[stride]);
        });
    split_release(counter, tid);
  }
#pragma unroll
  for (uint r = 0; r < L; ++r)
#pragma unroll
    for (uint nf = 0; nf < 2; ++nf) {
      const uint column = p.out_offset + base + nf * 8 + fm;
#pragma unroll
      for (uint i = 0; i < 2; ++i) {
        const ulong at = ulong(r * 8 + fn + i) * p.out_stride + column;
        float v = acc[r][nf][i];
        if (Ep == GGUF_EPILOGUE_RESIDUAL) v += float(aux[at]);
        if (Ep == GGUF_EPILOGUE_UP_WITH_GATE) v = float(bfloat(v)) * silu(float(aux[at]));
        out[at] = bfloat(v);
      }
    }
}

} // namespace gguf_sg

// Table16 for `lanes` eight-row tiles of `width` inputs: one simdgroup per (span, row).
kernel void decode_linear_gguf_prepare(device const bfloat *input [[buffer(0)]],
                                       device bfloat *table [[buffer(1)]], device float *sums [[buffer(2)]],
                                       constant uint &width [[buffer(3)]],
                                       uint2 tg [[threadgroup_position_in_grid]],
                                       uint sg [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
  const uint span = (tg.x * 4 + sg) / 8, row = (tg.x * 4 + sg) % 8;
  input += ulong(tg.y) * width * 8;
  const uint k = span * 64 + 2 * lane;
  q16sg::write_input(table + ulong(tg.y) * width * 8, sums + ulong(tg.y) * q16sg::sums_per_tile(width), width,
                     span, row, lane, input[row * width + k], input[row * width + k + 1]);
}

// The IQ4 codebook pair table of the codebook formats: 128 threads fill it,
// then every simdgroup of the threadgroup reads it.
template <class F> inline void codebook_lut(threadgroup bfloat2 *lut, uint tid) {
  if constexpr (F::Kind == QuantCodebook) {
    for (uint i = tid; i < 256; i += 128)
      lut[i] = bfloat2(float2(float(kIQ4NLValues[i & 15]), float(kIQ4NLValues[i >> 4])));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

#define GGUF_SG_KERNEL(Name, F, L, EP)                                                                        \
  kernel void Name(device const bfloat *table [[buffer(0)]], device const float *sums [[buffer(1)]],       \
                   device uchar *w0 [[buffer(2)]], device uchar *w1 [[buffer(3)]],                          \
                   device uchar *meta [[buffer(4)]], device bfloat *out [[buffer(5)]],                      \
                   device coherent(device) float *partials [[buffer(6)]],                                   \
                   device atomic_uint *counters [[buffer(7)]], device const bfloat *aux [[buffer(8)]],      \
                   constant GgufDecodeParams &p [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],       \
                   uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],    \
                   uint lane [[thread_index_in_simdgroup]]) {                                               \
    threadgroup bfloat2 lut[F::Kind == QuantCodebook ? 256 : 1];                                            \
    threadgroup gguf_sg::Coef<F> coefs[4 * 16 * gguf_sg::Shape<F>::J];                                     \
    threadgroup uint arrival;                                                                               \
    codebook_lut<F>(lut, tid);                                                                              \
    gguf_sg::decode<F, L, EP>(table, sums, w0, w1, meta, out, partials, counters, aux, p, tg, tid, sg, lane, \
                              lut, coefs, &arrival);                                                        \
  }
#define GGUF_SG_EPILOGUES(F, f, L)                                                                            \
  GGUF_SG_KERNEL(decode_linear_gguf_sg_##f##_l##L##_a, F, L, GGUF_EPILOGUE_NONE)                            \
  GGUF_SG_KERNEL(decode_linear_gguf_sg_##f##_l##L##_r, F, L, GGUF_EPILOGUE_RESIDUAL)                        \
  GGUF_SG_KERNEL(decode_linear_gguf_sg_##f##_l##L##_g, F, L, GGUF_EPILOGUE_UP_WITH_GATE)
#define GGUF_SG_FORMAT(F, f) \
  GGUF_SG_EPILOGUES(F, f, 1) GGUF_SG_EPILOGUES(F, f, 2) GGUF_SG_EPILOGUES(F, f, 3) GGUF_SG_EPILOGUES(F, f, 4)
GGUF_SG_FORMAT(FmtQ4K, q4k)
GGUF_SG_FORMAT(FmtQ5K, q5k)
GGUF_SG_FORMAT(FmtQ6K, q6k)
GGUF_SG_FORMAT(FmtQ3K, q3k)
GGUF_SG_FORMAT(FmtIQ4XS, iq4xs)
GGUF_SG_FORMAT(FmtIQ4NL, iq4nl)
GGUF_SG_FORMAT(FmtQ80, q80)
GGUF_SG_FORMAT(FmtIQ3S, iq3s)
#undef GGUF_SG_FORMAT
#undef GGUF_SG_EPILOGUES
#undef GGUF_SG_KERNEL

// Fused projections (qkv|z|ab, q|k|v): up to three column segments of any
// formats in one dispatch, so the small segments do not run as dispatches of
// their own. The threadgroup's tile picks its segment, and the segment's
// format picks the decode; every segment takes the same K splits.
#define GGUF_SG_SEGMENT(i, w0, w1, m) \
  device uchar *w0 [[buffer(i)]], device uchar *w1 [[buffer(i + 1)]], device uchar *m [[buffer(i + 2)]]
template <uint L>
inline void gguf_sg_fused(device const bfloat *table, device const float *sums, device uchar *w0a, device uchar *w1a,
                          device uchar *ma, device uchar *w0b, device uchar *w1b, device uchar *mb, device uchar *w0c,
                          device uchar *w1c, device uchar *mc, device bfloat *out, device coherent(device) float *partials,
                          device atomic_uint *counters, constant GgufDecodeFusedParams &p, uint2 tg, uint tid, uint sg,
                          uint lane, threadgroup bfloat2 *lut, threadgroup float2 *coefs, threadgroup uint &arrival) {
  const uint t0 = p.cols[0] / 64, t1 = t0 + p.cols[1] / 64;
  const uint s = tg.x < t0 ? 0 : tg.x < t1 ? 1 : 2;
  device uchar *w0 = s == 0 ? w0a : s == 1 ? w0b : w0c;
  device uchar *w1 = s == 0 ? w1a : s == 1 ? w1b : w1c;
  device uchar *meta = s == 0 ? ma : s == 1 ? mb : mc;
  const GgufDecodeParams q{p.input_size, p.splits, p.out_stride, p.offset[s]};
  const uint2 local(tg.x - (s == 0 ? 0 : s == 1 ? t0 : t1), tg.y);
  quant_format_switch(p.fmt[s], [&](auto format) {
    typedef decltype(format) F;
    codebook_lut<F>(lut, tid);
    gguf_sg::decode<F, L, GGUF_EPILOGUE_NONE>(table, sums, w0, w1, meta, out, partials, counters, out, q, local,
                                              tid, sg, lane, lut,
                                              reinterpret_cast<threadgroup gguf_sg::Coef<F> *>(coefs), &arrival);
  });
}
#define GGUF_SG_FUSED(L)                                                                                          \
  kernel void decode_linear_gguf_sg_fused_l##L(                                                                   \
      device const bfloat *table [[buffer(0)]], device const float *sums [[buffer(1)]],                          \
      GGUF_SG_SEGMENT(2, w0a, w1a, ma), GGUF_SG_SEGMENT(5, w0b, w1b, mb), GGUF_SG_SEGMENT(8, w0c, w1c, mc),       \
      device bfloat *out [[buffer(11)]], device coherent(device) float *partials [[buffer(12)]],                  \
      device atomic_uint *counters [[buffer(13)]], constant GgufDecodeFusedParams &p [[buffer(14)]],                  \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                       \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                     \
    threadgroup bfloat2 lut[256];                                                                                 \
    threadgroup float2 coefs[4 * 16 * 8];                                                                         \
    threadgroup uint arrival;                                                                                     \
    gguf_sg_fused<L>(table, sums, w0a, w1a, ma, w0b, w1b, mb, w0c, w1c, mc, out, partials, counters, p, tg, tid, sg, \
                     lane, lut, coefs, arrival);                                                                  \
  }
GGUF_SG_FUSED(1)
GGUF_SG_FUSED(2)
GGUF_SG_FUSED(3)
GGUF_SG_FUSED(4)
#undef GGUF_SG_FUSED
#undef GGUF_SG_SEGMENT

// MoE experts (ops/MoE.cpp): threadgroup (x, y) computes 64 columns of grouped 8-row tile y from its Table16 tile
// (kernels/shared/moe.metal) with the weights of the tile's expert (moe_gguf_segment), in the format the tile picks
// at run time: on the 40-core M3 Max one run-time-format kernel is within +1.6% of the per-format kernels
// (time-sg at 23040x2048 Q4_K and 92160x512 Q5_K, one to four lanes). No K splits, so no partials or counters.
// aux is the gate of the up pass. Grid (N / 64, tiles), 128 threads.
template <uint Ep>
inline void gguf_sg_expert(device const bfloat *table, device const float *sums, device const MoeTileDescriptor *tiles,
                           device const uint *tile_count, device uchar *w0, device uchar *w1, device uchar *meta,
                           device uchar *sw0, device uchar *sw1, device uchar *smeta, device bfloat *out,
                           device const bfloat *aux, constant MoeGgufExpertParams &p, uint2 tg, uint tid, uint sg,
                           uint lane, threadgroup bfloat2 *lut, threadgroup float2 *coefs, threadgroup uint *arrival) {
  if (tg.y >= *tile_count) return;
  const MoeGgufSegment s = moe_gguf_segment(tiles[tg.y].expert, p, w0, w1, meta, sw0, sw1, smeta);
  const uint K = p.input_size, N = p.output_size;
  const ulong rows = ulong(tg.y) * 8;
  const GgufDecodeParams q{K, 1, N, 0};
  quant_format_switch(s.format, [&](auto format) {
    typedef decltype(format) F;
    codebook_lut<F>(lut, tid);
    gguf_sg::decode<F, 1, Ep>(
        table + rows * K, sums + tg.y * q16sg::sums_per_tile(K), s.w0, s.w1, s.meta, out + rows * N, nullptr, nullptr,
        aux + rows * N, q, uint2(tg.x, 0), tid, sg, lane, lut, reinterpret_cast<threadgroup gguf_sg::Coef<F> *>(coefs),
        arrival);
  });
}
#define GGUF_SG_EXPERT(Name, EP)                                                                                    \
  kernel void Name(device const bfloat *table [[buffer(0)]], device const float *sums [[buffer(1)]],              \
                   device const MoeTileDescriptor *tiles [[buffer(2)]], device const uint *tile_count [[buffer(3)]], \
                   device uchar *w0 [[buffer(4)]], device uchar *w1 [[buffer(5)]], device uchar *meta [[buffer(6)]], \
                   device uchar *sw0 [[buffer(7)]], device uchar *sw1 [[buffer(8)]], device uchar *smeta [[buffer(9)]], \
                   device bfloat *out [[buffer(10)]], device const bfloat *aux [[buffer(11)]],                      \
                   constant MoeGgufExpertParams &p [[buffer(12)]], uint2 tg [[threadgroup_position_in_grid]],       \
                   uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],            \
                   uint lane [[thread_index_in_simdgroup]]) {                                                       \
    threadgroup bfloat2 lut[256];                                                                                   \
    threadgroup float2 coefs[4 * 16 * 8];                                                                           \
    threadgroup uint arrival;                                                                                       \
    gguf_sg_expert<EP>(table, sums, tiles, tile_count, w0, w1, meta, sw0, sw1, smeta, out, aux, p, tg, tid, sg, lane, \
                       lut, coefs, &arrival);                                                                       \
  }
GGUF_SG_EXPERT(moe_expert_gguf_sg, GGUF_EPILOGUE_NONE)
GGUF_SG_EXPERT(moe_expert_gguf_sg_up, GGUF_EPILOGUE_UP_WITH_GATE)
#undef GGUF_SG_EXPERT

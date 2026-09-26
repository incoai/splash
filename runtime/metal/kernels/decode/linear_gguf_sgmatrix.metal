// GGUF decode on Apple9's register simdgroup_matrix path. The 8x8x8 MMA runs
// on the FP32 pipe there, so the kernel keeps every other FP32 operation to
// the minimum a group scale needs, and does the rest on the integer pipe:
// - weights enter the MMA as exact bf16: 128 + code for linear codes with a
//   min, 160 + code - zero for seeded linear codes (one add), or the
//   codebook, int8 or grid value;
// - one MMA chain per coefficient group, closed by the fp32 epilogue
//   s * chain + b * sum (b = m - 128 s, formats with a min per 32 inputs) or
//   s * chain with the chain seeded by -160 * sum from the table (formats
//   with a zero point, and Q2_K, whose min per 16 inputs adds b * seed with
//   b = m / -160);
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
#include "metal/kernels/common/gguf_tile.h"
#include "metal/kernels/common/moe_expert_slab.h"
#include "metal/kernels/common/quant_formats.h"
#include "metal/kernels/common/sgmatrix.h"
#include "metal/kernels/common/split_reduce.h"

namespace gguf_sg {

// The coefficients of a threadgroup's simdgroups for one unit fit the storage of a unit of Q4_K's, eight groups of
// one (s, b) per column; the run-time-format kernels hold every format's in it.
constant constexpr uint kCoefFloat2s = GGUF_TILE_COLUMNS * 8;
template <class F> struct Shape {
  enum : uint {
    Linear = F::Kind == QuantLinear,
    HasMin = Linear && F::Zero == 0,                          // Q4_K, Q5_K, Q4_1, Q2_K: s code + m
    // Linear codes chained from the table's seed of their 16 inputs: s (code - zero) with a zero point (Q6_K, Q3_K,
    // Q4_0, IQ1), and Q2_K's min per 16 inputs
    Seeded = Linear && (!HasMin || F::Group == 16),
    // bf16 bits of the operand of code 0: 128, or 160 - zero when seeded
    Operand = 0x4300 + (Seeded ? kZeroPointOffset - 128 - F::Zero : 0),
    CG = Seeded || F::Group == 16 ? 2 : 1,                    // coefficient groups per 32 inputs
    // Spans decoded per coefficient unit: every span of a meta unit of eight groups when their coefficients fit,
    // else half of them (Q2_K's (s, b) per 16 inputs)
    UnitSpans = F::MetaGroups == 1                                                   ? 1
                : GGUF_TILE_COLUMNS * 8 * CG * (HasMin ? 2 : 1) <= kCoefFloat2s * 2 ? 4
                                                                                     : 2,
    J = 2 * UnitSpans * CG,                                   // coefficients per column and unit
  };
  // Each coefficient reads its own group's meta unit (coefficient_source), so a unit of four groups (PQ2_0's 128
  // elements) splits a coefficient unit of four spans in two.
  static_assert(F::MetaGroups == 8 || F::MetaGroups == 4 || F::MetaGroups == 1, "a meta unit is one, four or eight groups");
};
template <class F> using Coef = metal::conditional_t<Shape<F>::HasMin != 0, float2, float>;
template <class F> constant constexpr uint kCoefs = GGUF_TILE_COLUMNS * Shape<F>::J;
static_assert(GGUF_REGISTER_COLUMNS == 2 * 8, "a simdgroup's columns are its two 8-column MMA fragments");
static_assert(kCoefs<FmtQ4K> == kCoefFloat2s, "the storage holds a unit of Q4_K's coefficients");
#define GGUF_SG_COEF_BYTES(F, f) static_assert(kCoefs<F> * sizeof(Coef<F>) <= kCoefFloat2s * sizeof(float2), #f " coefficients fit");
QUANT_FORMATS(GGUF_SG_COEF_BYTES)
#undef GGUF_SG_COEF_BYTES

// The exact bf16 operand of pair f of a chunk.
template <class F>
inline bfloat2 operand(typename F::Chunk ch, uint f, threadgroup const bfloat2 *lut) {
  typedef Shape<F> S;
  if constexpr (F::Kind == QuantLinear) {
    return as_type<bfloat2>(F::codes(ch)[f] + S::Operand * 0x00010001u);   // Operand in both halves
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
// meta unit and, with ScaleInChunk, chunk 0 of its group (which holds the
// scale).
template <class F> struct CoefSource {
  typename F::Meta meta;
  typename F::Chunk chunk;
};
template <class F>
inline CoefSource<F> coefficient_source(device uchar *w0, device uchar *w1, device uchar *meta, uint plane_tile,
                                        uint groups, uint column, uint u, uint j) {
  typedef Shape<F> S;
  const uint g = u * 2 * S::UnitSpans + j / S::CG;
  const uint units = groups / F::MetaGroups;
  CoefSource<F> src;
  src.meta = F::loadMeta(meta + ((ulong(plane_tile) * units + g / F::MetaGroups) * QUANT_TILE_ROWS + column) * F::MetaBytes);
  if constexpr (F::ScaleInChunk) {
    const ulong at = (ulong(plane_tile) * groups + g) * QUANT_TILE_ROWS + column;
    src.chunk = F::loadChunk(w0 + at * F::P0, w1 + at * F::P1, 0);
  }
  return src;
}
// Coefficient j of a column in coefficient unit u: group gi = j / CG of the
// unit, 16-group half h = j % CG. Formats with a min return (s, m - 128 s),
// or with a seeded min (s, m / -160), whose product with the seed is m times
// the sum of the inputs.
template <class F> inline Coef<F> coefficient(CoefSource<F> src, uint u, uint j) {
  typedef Shape<F> S;
  const uint g = u * 2 * S::UnitSpans + j / S::CG, h = j % S::CG;
  QuantCoef k;
  if constexpr (F::ScaleInChunk) k = F::coef(src.meta, src.chunk);
  else k = F::coef(src.meta, ushort(g % F::MetaGroups));
  const float s = h ? k.s.y : k.s.x, m = h ? k.m.y : k.m.x;
  if constexpr (S::HasMin && S::Seeded) return float2(s, m * (-1.0f / kZeroPointOffset));
  else if constexpr (S::HasMin) return float2(s, fma(-128.0f, s, m));
  else return s;
}

// One threadgroup: 4 simdgroups x GGUF_REGISTER_COLUMNS = GGUF_TILE_COLUMNS columns of one segment,
// L request lanes of eight rows, one K partition (tg.y) of `splits`, into a destination of type Out.
template <class F, uint L, GgufEpilogue Ep, class Out>
inline void decode(device const bfloat *table, device const float *sums, device uchar *w0,
                   device uchar *w1, device uchar *meta, device Out *out,
                   device coherent(device) float *partials, device atomic_uint *counters,
                   device const bfloat *aux, const GgufDecodeParams p, uint2 tg, uint tid, uint sg,
                   uint lane, threadgroup const bfloat2 *lut, threadgroup Coef<F> *coefs,
                   threadgroup uint *arrival) {
  typedef Shape<F> S;
  typedef Coef<F> C;
  constexpr uint FG = 4 / S::CG;  // fragments per coefficient group
  const uint K = p.input_size, spans = K / GGUF_TABLE16_SPAN_INPUTS, groups = K / 32, splits = p.splits;
  const ulong tileSums = table16_sums_per_tile(K);
  const uint units = spans / S::UnitSpans;
  const uint u0 = tg.y * units / splits, u1 = (tg.y + 1) * units / splits;
  const sgmatrix::Lane l = sgmatrix::lane_map(lane);
  const uint fm = l.fm, fn = l.fn, c = fn / 2;
  const uint base = tg.x * GGUF_TILE_COLUMNS + sg * GGUF_REGISTER_COLUMNS, plane_tile = base / QUANT_TILE_ROWS,
             plane_row = base % QUANT_TILE_ROWS;
  constexpr uint NC = GGUF_REGISTER_COLUMNS;
  threadgroup C *cu = coefs + sg * NC * S::J;

  // Weight streams: lane c reads chunk c of each group; a span's two groups
  // of one column are a plane tile of payloads apart.
  const uint first = u0 * S::UnitSpans;
  device uchar *p0[2], *p1[2];
#pragma unroll
  for (uint nf = 0; nf < 2; ++nf) {
    const ulong at = (ulong(plane_tile) * groups + 2 * first) * QUANT_TILE_ROWS + plane_row + nf * 8 + fm;
    p0[nf] = w0 + at * F::P0;
    p1[nf] = w1 + at * F::P1;
  }
  typename F::Chunk cur[2][2], nxt[2][2];
  const auto load = [&](thread typename F::Chunk (&ch)[2][2]) __attribute__((always_inline)) {
#pragma unroll
    for (uint nf = 0; nf < 2; ++nf) {
      ch[nf][0] = F::loadChunk(p0[nf], p1[nf], c);
      ch[nf][1] = F::loadChunk(p0[nf] + QUANT_TILE_ROWS * F::P0, p1[nf] + QUANT_TILE_ROWS * F::P1, c);
    }
  };
  device const vec<bfloat, 8> *xt =
      reinterpret_cast<device const vec<bfloat, 8> *>(table + ulong(first) * GGUF_TABLE16_SPAN_VALUES);
  device const float *seeds = sums + ulong(first) * GGUF_TABLE16_SPAN_SEEDS;
  device const float *s32 = sums + table16_sums32_offset(K) + ulong(first) * GGUF_TABLE16_SPAN_SUMS;

  float2 acc[L][2];
#pragma unroll
  for (uint r = 0; r < L; ++r) acc[r][0] = acc[r][1] = float2(0);
  load(cur);
  for (uint u = u0; u < u1; ++u) {
    // This unit's coefficients for the simdgroup's NC columns, each decoded
    // once. Every source load is issued before the first decode waits for one:
    // on the 40-core M3 the zero-point formats (16 coefficients per column and
    // unit) gain at one to four lanes, Q6_K 5120x17408 2.5/3.8/3.1/1.1% and
    // Q3_K 7.6/4.3/3.3/2.2%; the other formats stay within 0.7%.
    constexpr uint I = (NC * S::J + 31) / 32;
    CoefSource<F> src[I];
#pragma unroll
    for (uint i = 0; i < I; ++i) {
      const uint e = lane + 32 * i;
      if (e < NC * S::J) src[i] = coefficient_source<F>(w0, w1, meta, plane_tile, groups, plane_row + e % NC, u, e / NC);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint i = 0; i < I; ++i) {
      const uint e = lane + 32 * i;
      if (e < NC * S::J) cu[e] = coefficient<F>(src[i], u, e / NC);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint us = 0; us < S::UnitSpans; ++us) {
      const bool more = us + 1 < S::UnitSpans || u + 1 < u1;
#pragma unroll
      for (uint nf = 0; nf < 2; ++nf) {
        p0[nf] += 2 * QUANT_TILE_ROWS * F::P0;
        p1[nf] += 2 * QUANT_TILE_ROWS * F::P1;
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
          for (uint h = 0; h < S::CG; ++h) cs[h][nf] = cu[((us * 2 + q) * S::CG + h) * NC + nf * 8 + fm];
        }
#pragma unroll
        for (uint r = 0; r < L; ++r) {
          const vec<bfloat, 8> bq = xt[ulong(r) * K + (8 * q + fm) * 4 + c];
          float2 sum = float2(0), seed[S::CG];
          if (S::HasMin && !S::Seeded) sum = *(device const float2 *)(s32 + ulong(r) * tileSums + q * 8 + fn);
#pragma unroll
          for (uint h = 0; h < S::CG; ++h)
            seed[h] = S::Seeded ? *(device const float2 *)(seeds + ulong(r) * tileSums + (2 * q + h) * 8 + fn)
                                : float2(0);
#pragma unroll
          for (uint h = 0; h < S::CG; ++h) {
#pragma unroll
            for (uint nf = 0; nf < 2; ++nf) {
              float2 dot = seed[h];
#pragma unroll
              for (uint f = h * FG; f < (h + 1) * FG; ++f)
                sgmatrix::mma_acc<bfloat>(dot, a[nf][f], reinterpret_cast<thread const bfloat2 *>(&bq)[f]);
              if constexpr (S::HasMin) {
                acc[r][nf] = fma(dot, cs[h][nf].x, acc[r][nf]);
                acc[r][nf] = fma(S::Seeded ? seed[h] : sum, cs[h][nf].y, acc[r][nf]);
              } else {
                acc[r][nf] = fma(dot, float(cs[h][nf]), acc[r][nf]);
              }
            }
          }
        }
      }
      xt += GGUF_TABLE16_SPAN_VALUES / 8;
      seeds += GGUF_TABLE16_SPAN_SEEDS;
      s32 += GGUF_TABLE16_SPAN_SUMS;
      if (more) {
#pragma unroll
        for (uint nf = 0; nf < 2; ++nf) cur[nf][0] = nxt[nf][0], cur[nf][1] = nxt[nf][1];
      }
    }
  }
  if (splits > 1) {
    // Partials [lane][split][8 rows][destination columns], one counter per
    // tile of destination columns: the segments of a projection never share them.
    const uint stride = p.out_stride;
    device atomic_uint *counter = counters + (p.out_offset + base) / GGUF_TILE_COLUMNS;
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
        out[at] = gguf_epilogue<Ep, Out>(acc[r][nf][i], aux, at);
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
  const uint k = span * GGUF_TABLE16_SPAN_INPUTS + 2 * lane;
  gguf_sg::write_input(table + ulong(tg.y) * width * 8, sums + ulong(tg.y) * table16_sums_per_tile(width), width,
                     span, row, lane, input[row * width + k], input[row * width + k + 1]);
}

// The pair table of a codebook format (quant_pair_table's, as bf16): the
// threadgroup fills it, then every simdgroup reads it.
template <class F> inline void codebook_lut(threadgroup bfloat2 *lut, uint tid) {
  if constexpr (F::Kind == QuantCodebook) {
    for (uint i = tid; i < kQuantPairTableEntries; i += GGUF_REGISTER_THREADS)
      lut[i] = bfloat2(float2(F::value(i & 15), F::value(i >> 4)));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

#define GGUF_SG_KERNEL(Name, F, L, EP, Out)                                                                   \
  kernel void Name(device const bfloat *table [[buffer(0)]], device const float *sums [[buffer(1)]],       \
                   device uchar *w0 [[buffer(2)]], device uchar *w1 [[buffer(3)]],                          \
                   device uchar *meta [[buffer(4)]], device Out *out [[buffer(5)]],                         \
                   device coherent(device) float *partials [[buffer(6)]],                                   \
                   device atomic_uint *counters [[buffer(7)]], device const bfloat *aux [[buffer(8)]],      \
                   constant GgufDecodeParams &p [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],       \
                   uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],    \
                   uint lane [[thread_index_in_simdgroup]]) {                                               \
    threadgroup bfloat2 lut[F::Kind == QuantCodebook ? kQuantPairTableEntries : 1];                         \
    threadgroup gguf_sg::Coef<F> coefs[gguf_sg::kCoefs<F>];                                                 \
    threadgroup uint arrival;                                                                               \
    codebook_lut<F>(lut, tid);                                                                              \
    gguf_sg::decode<F, L, EP>(table, sums, w0, w1, meta, out, partials, counters, aux, p, tg, tid, sg, lane, \
                              lut, coefs, &arrival);                                                        \
  }
// Each epilogue into bf16, and the plain one also into fp32 (_a_f32: the logits, ops::Projection::destination).
#define GGUF_SG_EPILOGUES(F, f, L)                                                                            \
  GGUF_SG_KERNEL(gguf_decode_sg_##f##_l##L##_a, F, L, EpNone, bfloat)                                \
  GGUF_SG_KERNEL(gguf_decode_sg_##f##_l##L##_a_f32, F, L, EpNone, float)                             \
  GGUF_SG_KERNEL(gguf_decode_sg_##f##_l##L##_r, F, L, EpResidual, bfloat)                            \
  GGUF_SG_KERNEL(gguf_decode_sg_##f##_l##L##_g, F, L, EpUpWithGate, bfloat)
#define GGUF_SG_FORMAT(F, f) \
  GGUF_SG_EPILOGUES(F, f, 1) GGUF_SG_EPILOGUES(F, f, 2) GGUF_SG_EPILOGUES(F, f, 3) GGUF_SG_EPILOGUES(F, f, 4)
QUANT_FORMATS(GGUF_SG_FORMAT)
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
  const uint t0 = p.cols[0] / GGUF_TILE_COLUMNS, t1 = t0 + p.cols[1] / GGUF_TILE_COLUMNS;
  const uint s = tg.x < t0 ? 0 : tg.x < t1 ? 1 : 2;
  device uchar *w0 = s == 0 ? w0a : s == 1 ? w0b : w0c;
  device uchar *w1 = s == 0 ? w1a : s == 1 ? w1b : w1c;
  device uchar *meta = s == 0 ? ma : s == 1 ? mb : mc;
  const GgufDecodeParams q{p.input_size, p.splits, p.out_stride, p.offset[s]};
  const uint2 local(tg.x - (s == 0 ? 0 : s == 1 ? t0 : t1), tg.y);
  quant_format_switch(p.fmt[s], [&](auto format) {
    typedef decltype(format) F;
    codebook_lut<F>(lut, tid);
    gguf_sg::decode<F, L, EpNone>(table, sums, w0, w1, meta, out, partials, counters, out, q, local,
                                              tid, sg, lane, lut,
                                              reinterpret_cast<threadgroup gguf_sg::Coef<F> *>(coefs), &arrival);
  });
}
#define GGUF_SG_FUSED(L)                                                                                          \
  kernel void gguf_decode_sg_fused_l##L(                                                                   \
      device const bfloat *table [[buffer(0)]], device const float *sums [[buffer(1)]],                          \
      GGUF_SG_SEGMENT(2, w0a, w1a, ma), GGUF_SG_SEGMENT(5, w0b, w1b, mb), GGUF_SG_SEGMENT(8, w0c, w1c, mc),       \
      device bfloat *out [[buffer(11)]], device coherent(device) float *partials [[buffer(12)]],                  \
      device atomic_uint *counters [[buffer(13)]], constant GgufDecodeFusedParams &p [[buffer(14)]],                  \
      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                       \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {                     \
    threadgroup bfloat2 lut[kQuantPairTableEntries];                                                              \
    threadgroup float2 coefs[gguf_sg::kCoefFloat2s];                                                              \
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
// aux is the gate of the up pass. Grid (column tiles, expert tiles), GGUF_REGISTER_THREADS threads.
template <GgufEpilogue Ep>
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
        table + rows * K, sums + tg.y * table16_sums_per_tile(K), s.w0, s.w1, s.meta, out + rows * N, nullptr, nullptr,
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
    threadgroup bfloat2 lut[kQuantPairTableEntries];                                                                \
    threadgroup float2 coefs[gguf_sg::kCoefFloat2s];                                                                \
    threadgroup uint arrival;                                                                                       \
    gguf_sg_expert<EP>(table, sums, tiles, tile_count, w0, w1, meta, sw0, sw1, smeta, out, aux, p, tg, tid, sg, lane, \
                       lut, coefs, &arrival);                                                                       \
  }
GGUF_SG_EXPERT(moe_expert_gguf_sg_a, EpNone)
GGUF_SG_EXPERT(moe_expert_gguf_sg_g, EpUpWithGate)
#undef GGUF_SG_EXPERT

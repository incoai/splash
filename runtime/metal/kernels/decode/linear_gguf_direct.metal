// Decode projections of up to GGUF_DIRECT_ROWS live rows (ops::LinearTile::GgufDirect): the weights read once into
// registers and multiplied against every live row by 8x8 simdgroup MMAs, the rows read from the producer's bf16
// output as they are (no activation table, so no prepare dispatch). A design after llama.cpp's Metal few-row kernels
// kernel_mul_mv_ext and kernel_mul_mv_mma_blk (MIT notice in THIRD_PARTY_NOTICES): a weight is loaded once and shared
// by the batch's rows, one MMA per 8 inputs x 8 columns x 8 rows rather than a scalar FMA per weight and row, which
// bounds the column tile (kernels/common/gguf_column_tile.h) past two rows.
//
// A simdgroup holds Fragments 8-column fragments; the MMA's A operand is the fragment's weights (row m = column fm of
// the lane map), B the rows' inputs transposed (column n = row fn), C fp32 [column][row]. A lane reads chunk fn / 2
// of each group of its column (metal/abi/QuantFormat.h): its pair f of the chunk is A's k' = fn, fn + 1 of fragment
// f, so B's k' = fm of fragment f is element 16 (f >> 1) + 4 (fm >> 1) + 2 (f & 1) + (fm & 1), which a lane reads as
// two 8-byte loads of each of its two rows per group.
// It takes the formats of codebook, int8, E4M3 and bf16 operands (QUANT_DIRECT_FORMATS, quant_direct_format), whose
// weights enter the MMA exactly as themselves (the register tile's operands,
// kernels/decode/linear_gguf_sgmatrix.metal). One fp32 chain of MMAs per coefficient (32 inputs, 16 for 16-input
// scales) is closed by fp32 s * chain.
//
// One threadgroup: `simdgroups` simdgroups over the same 8 Fragments columns, simdgroup s summing groups
// [s G / S, (s + 1) G / S) of K, then the threads adding the parts in simdgroup order, so a column's value depends
// neither on the grid nor on the other columns, and storing it as the tiles do (kernels/common/gguf_tile.h).
#pragma clang fp reassociate(off)
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/gguf_tile.h"
#include "metal/kernels/common/quant_formats.h"
#include "metal/kernels/common/sgmatrix.h"

namespace gguf_direct {

// The exact bf16 operand of pair f of a chunk.
template <class F> inline bfloat2 operand(typename F::Chunk ch, uint f, threadgroup const bfloat2 *lut) {
  static_assert(F::Kind != QuantLinear && F::Kind != QuantGrid, "the direct tile takes no offsets or grids");
  if constexpr (F::Kind == QuantCodebook) {
    return lut[(F::indices(ch) >> (8 * f)) & 0xFFu];
  } else if constexpr (F::Kind == QuantInt8) {
    const uint2 v = F::values(ch);
    return bfloat2(float2(as_type<char2>(ushort((f < 2 ? v.x : v.y) >> (16 * (f & 1))))));
  } else if constexpr (F::Kind == QuantFloat8) {   // exact: an E4M3 value has four significant bits
    const uint2 v = F::values(ch);
    const uint pair = ((f < 2 ? v.x : v.y) >> (16 * (f & 1))) & 0xFFFFu;
    return bfloat2(float2(quant_e4m3_pair(quant_byte_pairs(pair).x)));
  } else {
    static_assert(is_same_v<typename F::Value, bfloat>, "the direct tile takes bf16 values only");
    return as_type<bfloat2>(F::values(ch)[f]);
  }
}

// What a threadgroup's columns read: the planes, and the pair table of a codebook format.
struct Planes {
  device uchar *w0, *w1, *meta;
};

// The sums of the 8 Fragments columns from `column0` over groups [g_begin, g_end) of K, of rows r0 and r1 of the
// lane's B operands, into acc[nf] = C[fm][fn, fn + 1] of fragment nf.
template <class F, uint Fragments>
inline void sums(device const bfloat *x, uint K, Planes w, uint column0, uint g_begin, uint g_end, uint r0, uint r1,
                 threadgroup const bfloat2 *lut, uint lane, thread float2 (&acc)[Fragments]) {
  constexpr uint CG = F::Group == 16 ? 2 : 1;   // chains of MMAs per 32 inputs, one per coefficient
  const sgmatrix::Lane l = sgmatrix::lane_map(lane);
  const uint fm = l.fm, fn = l.fn, c = fn / 2, kb = 4 * (fm >> 1), half_b = 16 * (fm & 1);
  const uint groups = K / 32, units = groups / F::MetaGroups;
  device uchar *p0[Fragments], *p1[Fragments], *pm[Fragments];
#pragma unroll
  for (uint nf = 0; nf < Fragments; ++nf) {
    const uint column = column0 + nf * 8 + fm, tile = column / QUANT_TILE_ROWS, row = column % QUANT_TILE_ROWS;
    const ulong at = (ulong(tile) * groups + g_begin) * QUANT_TILE_ROWS + row;
    p0[nf] = w.w0 + at * F::P0;
    p1[nf] = w.w1 + at * F::P1;
    pm[nf] = w.meta + (ulong(tile) * units * QUANT_TILE_ROWS + row) * F::MetaBytes;
    acc[nf] = float2(0);
  }
  device const uint2 *x0 = reinterpret_cast<device const uint2 *>(x + ulong(r0) * K + kb);
  device const uint2 *x1 = reinterpret_cast<device const uint2 *>(x + ulong(r1) * K + kb);
  typename F::Chunk ch[Fragments], ch_next[Fragments];
  typename F::Meta mt[Fragments], mt_next[Fragments];
  uint2 lo0, hi0, lo1, hi1;
  const auto load = [&](uint g, thread typename F::Chunk (&chs)[Fragments], thread typename F::Meta (&mts)[Fragments])
      __attribute__((always_inline)) {
#pragma unroll
        for (uint nf = 0; nf < Fragments; ++nf) {
          const ulong step = ulong(g - g_begin) * QUANT_TILE_ROWS;
          chs[nf] = F::loadChunk(p0[nf] + step * F::P0, p1[nf] + step * F::P1, c);
          mts[nf] = F::loadMeta(pm[nf] + ulong(g / F::MetaGroups) * QUANT_TILE_ROWS * F::MetaBytes);
        }
      };
  if (g_begin < g_end) load(g_begin, ch, mt);
  for (uint g = g_begin; g < g_end; ++g) {
    lo0 = x0[g * 8], hi0 = x0[g * 8 + 4], lo1 = x1[g * 8], hi1 = x1[g * 8 + 4];
    if (g + 1 < g_end) load(g + 1, ch_next, mt_next);
    // B of fragment f: elements 16 (f >> 1) + kb + 2 (f & 1) + (fm & 1) of rows r0 and r1.
    bfloat2 b[4];
#pragma unroll
    for (uint f = 0; f < 4; ++f) {
      const uint v0 = (f < 2 ? lo0 : hi0)[f & 1], v1 = (f < 2 ? lo1 : hi1)[f & 1];
      b[f] = as_type<bfloat2>(((v0 >> half_b) & 0xFFFFu) | (((v1 >> half_b) & 0xFFFFu) << 16));
    }
#pragma unroll
    for (uint nf = 0; nf < Fragments; ++nf) {
      QuantCoef k;
      if constexpr (F::ScaleInChunk) k = F::coef(mt[nf], ch[nf]);
      else k = F::coef(mt[nf], ushort(g % F::MetaGroups));
#pragma unroll
      for (uint h = 0; h < CG; ++h) {
        float2 dot = float2(0);
#pragma unroll
        for (uint f = h * 4 / CG; f < (h + 1) * 4 / CG; ++f)
          sgmatrix::mma_acc<bfloat>(dot, operand<F>(ch[nf], f, lut), b[f]);
        acc[nf] = fma(dot, h ? k.s.y : k.s.x, acc[nf]);
      }
    }
    if (g + 1 < g_end) {
#pragma unroll
      for (uint nf = 0; nf < Fragments; ++nf) ch[nf] = ch_next[nf], mt[nf] = mt_next[nf];
    }
  }
}

// The threadgroup's parts of its columns' sums: [simdgroup][fragment][column of 8][row of 8].
constant constexpr uint kParts = GGUF_DIRECT_PARTS * 64;

template <class F, uint Fragments, GgufEpilogue Ep, class Out>
inline void project(device const bfloat *input, Planes w, device const bfloat *aux, device Out *output,
                    constant GgufColumnParams &p, uint tile, uint tid, uint sg, uint simdgroups, uint lane,
                    threadgroup bfloat2 *lut, threadgroup float *parts) {
  quant_pair_table<F>(lut, tid, simdgroups * 32);
  const uint K = p.input_size, groups = K / 32, column0 = tile * 8 * Fragments;
  const sgmatrix::Lane l = sgmatrix::lane_map(lane);
  const uint r0 = min(uint(l.fn), p.rows - 1), r1 = min(uint(l.fn) + 1, p.rows - 1);
  float2 acc[Fragments];
  sums<F, Fragments>(input, K, w, column0, sg * groups / simdgroups, (sg + 1) * groups / simdgroups, r0, r1, lut,
                     lane, acc);
#pragma unroll
  for (uint nf = 0; nf < Fragments; ++nf) {
    threadgroup float *at = parts + (sg * Fragments + nf) * 64 + l.fm * 8 + l.fn;
    at[0] = acc[nf].x;
    at[1] = acc[nf].y;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Output (row, column) of the threadgroup, columns fastest.
  for (uint o = tid; o < p.rows * 8 * Fragments; o += simdgroups * 32) {
    const uint row = o / (8 * Fragments), column = o % (8 * Fragments);
    const uint part = (column / 8) * 64 + (column % 8) * 8 + row;
    float total = 0.0f;
    for (uint s = 0; s < simdgroups; ++s) total += parts[s * Fragments * 64 + part];
    const ulong at = ulong(row) * p.out_stride + p.out_offset + column0 + column;
    output[at] = gguf_epilogue<Ep, Out>(total, aux, at);
  }
}

} // namespace gguf_direct

// Grid (the segment's columns / (8 Fragments)), GGUF_DIRECT threads of 32 per simdgroup. Bindings: input, the
// tensor's planes (w0, w1, meta), output, then the residual (_r); then GgufColumnParams.
#define GGUF_DIRECT_ARGS                                                                                       \
  device const bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]],   \
      device uchar *meta [[buffer(3)]]
#define GGUF_DIRECT_THREAD_ARGS                                                                                \
  uint tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                        \
      uint sg [[simdgroup_index_in_threadgroup]], uint simdgroups [[simdgroups_per_threadgroup]],              \
      uint lane [[thread_index_in_simdgroup]]
#define GGUF_DIRECT_SHARED(F)                                                                                  \
  threadgroup bfloat2 lut[F::Kind == QuantCodebook ? kQuantPairTableEntries : 1];                              \
  threadgroup float parts[gguf_direct::kParts];
#define GGUF_DIRECT_KERNEL(Name, F, FR, Ep, Out, Aux, AuxArg, ParamsSlot)                                      \
  kernel void Name(GGUF_DIRECT_ARGS, device Out *output [[buffer(4)]] AuxArg,                                   \
                   constant GgufColumnParams &p [[buffer(ParamsSlot)]], GGUF_DIRECT_THREAD_ARGS) {             \
    GGUF_DIRECT_SHARED(F)                                                                                      \
    gguf_direct::project<F, FR, Ep, Out>(input, {w0, w1, meta}, Aux, output, p, tile, tid, sg, simdgroups, lane, \
                                         lut, parts);                                                          \
  }
#define GGUF_DIRECT_RESIDUAL , device const bfloat *residual [[buffer(5)]]
#define GGUF_DIRECT_FRAGMENTS_OF(F, f, FR)                                                                     \
  GGUF_DIRECT_KERNEL(gguf_direct_##f##_f##FR##_a, F, FR, EpNone, bfloat, input, , 5)                            \
  GGUF_DIRECT_KERNEL(gguf_direct_##f##_f##FR##_a_f32, F, FR, EpNone, float, input, , 5)                         \
  GGUF_DIRECT_KERNEL(gguf_direct_##f##_f##FR##_r, F, FR, EpResidual, bfloat, residual, GGUF_DIRECT_RESIDUAL, 6)
#define GGUF_DIRECT_FORMAT(F, f)                                                                               \
  GGUF_DIRECT_FRAGMENTS_OF(F, f, 1) GGUF_DIRECT_FRAGMENTS_OF(F, f, 2)
QUANT_DIRECT_FORMATS(GGUF_DIRECT_FORMAT)
#undef GGUF_DIRECT_FORMAT
#undef GGUF_DIRECT_FRAGMENTS_OF
#undef GGUF_DIRECT_RESIDUAL
#undef GGUF_DIRECT_KERNEL
#undef GGUF_DIRECT_SHARED
#undef GGUF_DIRECT_THREAD_ARGS
#undef GGUF_DIRECT_ARGS

// MoE expert passes of a decode step on its live rows (ops/MoE.cpp, Apple9). A decode step groups each expert's
// routes into an 8-row tile, but a step routes an expert one to three of its rows: the register tile's 8-row MMA
// (kernels/decode/linear_gguf_sgmatrix.metal) multiplies every weight with all eight, which makes the pass compute-
// bound at a third of the memory bandwidth on an M3 Ultra. Here each thread dequantizes a column's groups of 32
// (gguf_staged.h's dequant32, into registers) and multiplies them with the tile's live rows alone.
//
// Threadgroup (column tile, expert tile): 64 columns, kMoeRowsParts simdgroups, one per part of K; part p walks the
// groups g = p mod kMoeRowsParts in order, part 0 then adds the other parts' sums in part order. A row's result
// therefore depends neither on the other rows of its tile nor on the batch: every row's sum runs in one fixed order.
// It does differ from the prefill tile's, whose MMA sums in another order. Input: the grouped bf16 rows
// (moe_gather_rows, or the up pass's output for the down pass). aux is the gate of the up pass.
#pragma clang fp reassociate(off)
#include "metal/abi/Gguf.h"
#include "metal/abi/MoE.h"
#include "metal/kernels/common/gguf_staged.h"
#include "metal/kernels/common/gguf_tile.h"
#include "metal/kernels/common/moe_expert_slab.h"

constant constexpr uint kMoeRowsParts = GGUF_EXPERT_ROWS_THREADS / 32;
constant constexpr uint kMoeRowsMax = 8;

// dequant32 into registers: v[c] holds values 4c..4c+3 of the group and v[4 + c] values 16 + 4c..16 + 4c + 3.
template <class F>
inline void dequant32_values(typename F::Payload w, typename F::Meta meta, ushort j, threadgroup half2 *tl,
                             thread half4 (&v)[8]) {
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
    } else if constexpr (F::Kind == QuantCodebook) {
      typedef typename F::Scale S;
      const uchar4 b = as_type<uchar4>(F::indices(q));
      lo = half4(half2(vec<S, 2>(tl[b.x]) * S(k.s.x)), half2(vec<S, 2>(tl[b.y]) * S(k.s.x)));
      hi = half4(half2(vec<S, 2>(tl[b.z]) * S(k.s.y)), half2(vec<S, 2>(tl[b.w]) * S(k.s.y)));
    } else if constexpr (F::Kind == QuantInt8) {
      typedef typename F::Scale S;
      const uint2 x = F::values(q);
      lo = half4(vec<S, 4>(as_type<char4>(x.x)) * S(k.s.x));
      hi = half4(vec<S, 4>(as_type<char4>(x.y)) * S(k.s.y));
    } else if constexpr (F::Kind == QuantFloat8) {
      const uint2 x = F::values(q);
      lo = half4(float4(quant_e4m3_word(x.x)) * k.s.x);
      hi = half4(float4(quant_e4m3_word(x.y)) * k.s.y);
    } else {
      const uint2 g = F::grid(q); const uint s = F::signs(q);
      lo = half4(float4(as_type<uchar4>(g.x)) * k.s.x);
      hi = half4(float4(as_type<uchar4>(g.y)) * k.s.y);
      lo = select(lo, -lo, bool4(s & 1, s & 2, s & 4, s & 8));
      hi = select(hi, -hi, bool4(s & 16, s & 32, s & 64, s & 128));
    }
    v[c] = lo;
    v[4 + c] = hi;
  }
}

// Each simdgroup walks one part of K for all 64 columns of the tile: lane l the columns l and l + 32, whose groups
// sit 32 payloads apart, so a simdgroup's loads of a group cover 32 consecutive columns. The lane dequantizes both
// columns' group once into floats and reads each live row's 32 inputs once for both.
constant constexpr uint kMoeRowsColumnsPerLane = GGUF_TILE_COLUMNS / 32;

template <class F, GgufEpilogue Ep>
inline void moe_rows_pass(device const bfloat *input, MoeGgufSegment seg, uint K, uint N, uint column0, uint live,
                          ulong row0, device bfloat *out, device const bfloat *aux, threadgroup half2 *tl,
                          threadgroup float *partial, uint tid) {
  const uint lane = tid % 32, part = tid / 32;
  const uint groups = K / 32;
  // The tile's columns start at row plane_row of their plane tile (QUANT_TILE_ROWS may hold several column tiles).
  const uint plane_tile = column0 / QUANT_TILE_ROWS, plane_row = column0 % QUANT_TILE_ROWS;
  device uchar *tw0 = seg.w0 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row + lane) * F::P0;
  device uchar *tw1 = seg.w1 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row + lane) * F::P1;
  device uchar *tmeta = seg.meta + (ulong(plane_tile) * (groups / F::MetaGroups) * QUANT_TILE_ROWS + plane_row + lane) *
                                       F::MetaBytes;
  float acc[kMoeRowsMax][kMoeRowsColumnsPerLane];
#pragma unroll
  for (uint r = 0; r < kMoeRowsMax; ++r)
#pragma unroll
    for (uint c = 0; c < kMoeRowsColumnsPerLane; ++c) acc[r][c] = 0.0f;
  for (uint g = part; g < groups; g += kMoeRowsParts) {
    float4 v[kMoeRowsColumnsPerLane][8];
#pragma unroll
    for (uint c = 0; c < kMoeRowsColumnsPerLane; ++c) {
      const ulong at = ulong(g) * QUANT_TILE_ROWS + 32 * c;
      const typename F::Payload w = F::load(tw0 + at * F::P0, tw1 + at * F::P1);
      const typename F::Meta hdr =
          F::loadMeta(tmeta + (ulong(g / F::MetaGroups) * QUANT_TILE_ROWS + 32 * c) * F::MetaBytes);
      half4 h[8];
      dequant32_values<F>(w, hdr, ushort(g % F::MetaGroups), tl, h);
#pragma unroll
      for (uint i = 0; i < 8; ++i) v[c][i] = float4(h[i]);
    }
#pragma unroll
    for (uint r = 0; r < kMoeRowsMax; ++r) {
      if (r >= live) break;
      device const vec<bfloat, 4> *x4 =
          reinterpret_cast<device const vec<bfloat, 4> *>(input + (row0 + r) * K + ulong(g) * 32);
      float s[kMoeRowsColumnsPerLane];
#pragma unroll
      for (uint c = 0; c < kMoeRowsColumnsPerLane; ++c) s[c] = 0.0f;
#pragma unroll
      for (uint i = 0; i < 8; ++i) {
        const float4 x = float4(x4[i]);
#pragma unroll
        for (uint c = 0; c < kMoeRowsColumnsPerLane; ++c) {
          s[c] = fma(v[c][i].x, x.x, s[c]);
          s[c] = fma(v[c][i].y, x.y, s[c]);
          s[c] = fma(v[c][i].z, x.z, s[c]);
          s[c] = fma(v[c][i].w, x.w, s[c]);
        }
      }
#pragma unroll
      for (uint c = 0; c < kMoeRowsColumnsPerLane; ++c) acc[r][c] += s[c];
    }
  }
  // Parts 1.. publish their sums, part 0 adds them in part order.
  if (part)
#pragma unroll
    for (uint r = 0; r < kMoeRowsMax; ++r) {
      if (r >= live) break;
#pragma unroll
      for (uint c = 0; c < kMoeRowsColumnsPerLane; ++c)
        partial[((part - 1) * kMoeRowsMax + r) * GGUF_TILE_COLUMNS + lane + 32 * c] = acc[r][c];
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (part) return;
#pragma unroll
  for (uint r = 0; r < kMoeRowsMax; ++r) {
    if (r >= live) break;
#pragma unroll
    for (uint c = 0; c < kMoeRowsColumnsPerLane; ++c) {
      const uint column = lane + 32 * c;
      float sum = acc[r][c];
      for (uint q = 1; q < kMoeRowsParts; ++q) sum += partial[((q - 1) * kMoeRowsMax + r) * GGUF_TILE_COLUMNS + column];
      const ulong at = (row0 + r) * N + column0 + column;
      out[at] = gguf_epilogue<Ep>(sum, aux, at);
    }
  }
}

template <GgufEpilogue Ep>
inline void moe_rows(device const bfloat *input, device const MoeTileDescriptor *tiles, device const uint *tile_count,
                     device uchar *w0, device uchar *w1, device uchar *meta, device uchar *sw0, device uchar *sw1,
                     device uchar *smeta, device bfloat *out, device const bfloat *aux,
                     constant MoeGgufExpertParams &p, uint2 tg, uint tid, threadgroup half2 *tl,
                     threadgroup float *partial) {
  if (tg.y >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[tg.y];
  const MoeGgufSegment seg = moe_gguf_segment(tile.expert, p, w0, w1, meta, sw0, sw1, smeta);
  const uint live = min(tile.rows, kMoeRowsMax);
  quant_format_switch(seg.format, [&](auto format) {
    typedef decltype(format) F;
    quant_pair_table<F>(tl, tid, GGUF_EXPERT_ROWS_THREADS);
    moe_rows_pass<F, Ep>(input, seg, p.input_size, p.output_size, tg.x * GGUF_TILE_COLUMNS, live,
                         ulong(tg.y) * kMoeRowsMax, out, aux, tl, partial, tid);
  });
}

#define MOE_ROWS_KERNEL(Name, EP)                                                                                    \
  kernel void Name(device const bfloat *input [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],  \
                   device const uint *tile_count [[buffer(2)]], device uchar *w0 [[buffer(3)]],                   \
                   device uchar *w1 [[buffer(4)]], device uchar *meta [[buffer(5)]], device uchar *sw0 [[buffer(6)]], \
                   device uchar *sw1 [[buffer(7)]], device uchar *smeta [[buffer(8)]], device bfloat *out [[buffer(9)]], \
                   device const bfloat *aux [[buffer(10)]], constant MoeGgufExpertParams &p [[buffer(11)]],       \
                   uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {          \
    threadgroup half2 tl[kQuantPairTableEntries];                                                                    \
    threadgroup float partial[(kMoeRowsParts - 1) * kMoeRowsMax * GGUF_TILE_COLUMNS];                                \
    moe_rows<EP>(input, tiles, tile_count, w0, w1, meta, sw0, sw1, smeta, out, aux, p, tg, tid, tl, partial);       \
  }
MOE_ROWS_KERNEL(moe_expert_gguf_rows_a, EpNone)
MOE_ROWS_KERNEL(moe_expert_gguf_rows_g, EpUpWithGate)
#undef MOE_ROWS_KERNEL

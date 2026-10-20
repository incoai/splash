#pragma once
#include "metal/kernels/common/gguf_staged.h"

// A register GEMV of a GGUF weight image over a tile's live rows alone (the Fast tier's MoE expert tiles,
// kernels/shared/moe_gguf.metal): each thread dequantizes a column's groups of 32 into registers, rounding each value
// once to half as dequant32 stages it, and multiplies them with every live row in fp32, where the half and bf16
// products are exact. The register form of dequant32 and the walk over K by parts are those of incoai/splash#367's
// live-row passes (dequant32_values, moe_rows_pass), held here for any caller: a part's thread may take several
// columns of one tensor (streams), the same column of several tensors that read one input (gate and up), and live
// rows anywhere in the input (in place), with the grid of a grid format in threadgroup memory. Includers set
// `#pragma clang fp reassociate(off)` first, so the source order of float operations holds here.
//
// The sum of a row and a column runs in one order, whatever the other rows and whatever threads walk it: group g of K
// goes to part g % Parts (kLiveRowsParts in the Fast tier); a part sums its groups in ascending order into a float4,
// chunk by chunk (fma of the chunk's low half, then of its high half), and reduces it to (x + y) + (z + w); the
// parts add in part order. One thread may walk every part (ThreadParts = Parts), or one simdgroup each part
// (ThreadParts = 1) and a threadgroup reduction adds them.
constant constexpr ushort kLiveRowsParts = 2;

// The grid of a grid format as 32-bit words in threadgroup memory, which every thread of the threadgroup fills: lanes
// index the grid with divergent indices, which the constant address space serializes. An eight-element entry (IQ2)
// takes two words, of which a chunk reads the half it needs. Words 0: the format reads its own tables.
template <class F> struct LiveRowsGrid { static constexpr constant uint Words = 0; };
// An IQ2 grid of eight-element entries as two words each, and quant_grid8 of its entries a and b, half h.
inline void live_rows_fill_grid8(threadgroup uint *t, constant const uint64_t *g, uint entries, uint i, uint n) {
  for (; i < entries; i += n) {
    const uint2 v = as_type<uint2>(g[i]);
    t[2 * i] = v.x;
    t[2 * i + 1] = v.y;
  }
}
inline uint2 live_rows_grid8(threadgroup const uint *t, uint a, uint b, uint h) { return uint2(t[2 * a + h], t[2 * b + h]); }
template <> struct LiveRowsGrid<FmtIQ2S> {
  static constexpr constant uint Words = 2048;
  static void fill(threadgroup uint *t, uint i, uint n) { live_rows_fill_grid8(t, kIQ2SGrid, Words / 2, i, n); }
  static uint2 grid(threadgroup const uint *t, FmtIQ2S::Chunk q) { return live_rows_grid8(t, q.x & 0xFFFF, q.x >> 16, (q.y >> 16) & 1); }
};
template <> struct LiveRowsGrid<FmtIQ2XS> {
  static constexpr constant uint Words = 1024;
  static void fill(threadgroup uint *t, uint i, uint n) { live_rows_fill_grid8(t, kIQ2XSGrid, Words / 2, i, n); }
  static uint2 grid(threadgroup const uint *t, FmtIQ2XS::Chunk q) { return live_rows_grid8(t, q.x & 0xFFFF, q.x >> 16, (q.y >> 16) & 1); }
};
template <> struct LiveRowsGrid<FmtIQ2XXS> {
  static constexpr constant uint Words = 512;
  static void fill(threadgroup uint *t, uint i, uint n) { live_rows_fill_grid8(t, kIQ2XXSGrid, Words / 2, i, n); }
  static uint2 grid(threadgroup const uint *t, FmtIQ2XXS::Chunk q) { return live_rows_grid8(t, q & 0xFF, (q >> 8) & 0xFF, (q >> 24) & 1); }
};
template <> struct LiveRowsGrid<FmtIQ3S> {
  static constexpr constant uint Words = 512;
  static void fill(threadgroup uint *t, uint i, uint n) { for (; i < Words; i += n) t[i] = kIQ3SGrid[i]; }
  static uint2 grid(threadgroup const uint *t, FmtIQ3S::Chunk q) {
    return uint2(t[(q & 0xFF) | ((q >> 16) & 0x100)], t[((q >> 8) & 0xFF) | ((q >> 17) & 0x100)]);
  }
};
template <> struct LiveRowsGrid<FmtIQ3XXS> {
  static constexpr constant uint Words = 256;
  static void fill(threadgroup uint *t, uint i, uint n) { for (; i < Words; i += n) t[i] = kIQ3XXSGrid[i]; }
  static uint2 grid(threadgroup const uint *t, FmtIQ3XXS::Chunk q) { return uint2(t[q & 0xFF], t[(q >> 8) & 0xFF]); }
};

// Chunk q of a group with the group's coefficients k as dequant32 stages it: lo = elements 4c..4c+3 of the group,
// hi = elements 16+4c..16+4c+3, each value rounded once to half. TgGrid: a grid format reads LiveRowsGrid<F>'s copy
// at `grid`.
template <class F, bool TgGrid>
inline void live_rows_chunk(typename F::Chunk q, QuantCoef k, threadgroup half2 *tl, threadgroup const uint *grid,
                            thread half4 &lo, thread half4 &hi) {
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
    const uint2 v = F::values(q);
    lo = half4(vec<S, 4>(as_type<char4>(v.x)) * S(k.s.x));
    hi = half4(vec<S, 4>(as_type<char4>(v.y)) * S(k.s.y));
  } else if constexpr (F::Kind == QuantFloat8) {
    const uint2 v = F::values(q);
    lo = half4(float4(quant_e4m3_word(v.x)) * k.s.x);
    hi = half4(float4(quant_e4m3_word(v.y)) * k.s.y);
  } else if constexpr (F::Kind == QuantFloat16) {
    typedef vec<typename F::Value, 2> V2;
    const uint4 v = F::values(q);
    lo = half4(float4(float2(as_type<V2>(v.x)), float2(as_type<V2>(v.y))) * k.s.x);
    hi = half4(float4(float2(as_type<V2>(v.z)), float2(as_type<V2>(v.w))) * k.s.y);
  } else {
    uint2 g;
    if constexpr (TgGrid) g = LiveRowsGrid<F>::grid(grid, q);
    else g = F::grid(q);
    const uint s = F::signs(q);
    lo = half4(float4(as_type<uchar4>(g.x)) * k.s.x);
    hi = half4(float4(as_type<uchar4>(g.y)) * k.s.y);
    lo = select(lo, -lo, bool4(s & 1, s & 2, s & 4, s & 8));
    hi = select(hi, -hi, bool4(s & 16, s & 32, s & 64, s & 128));
  }
}

// The coefficients of group j of a meta unit, or of the payload's group for formats with the scale in every chunk.
template <class F>
inline QuantCoef live_rows_coef(typename F::Payload w, typename F::Meta meta, ushort j) {
  if constexpr (F::ScaleInChunk) return F::coef(meta, F::chunk(w, 0));
  else return F::coef(meta, j);
}

// dequant32 into registers (#367): v[c] holds values 4c..4c+3 of the group and v[4 + c] values 16 + 4c..16 + 4c + 3.
template <class F>
inline void dequant32_values(typename F::Payload w, typename F::Meta meta, ushort j, threadgroup half2 *tl,
                             thread half4 (&v)[8]) {
  const QuantCoef k = live_rows_coef<F>(w, meta, j);
#pragma unroll
  for (ushort c = 0; c < 4; ++c) live_rows_chunk<F, false>(F::chunk(w, c), k, tl, nullptr, v[c], v[4 + c]);
}

// One weight column of a GGUF image (QUANT_TILE_ROWS rows per plane tile, as the staged tiles address it).
template <class F> struct LiveRowsColumn {
  device uchar *w0, *w1, *meta;
  void begin(device uchar *p0, device uchar *p1, device uchar *m, uint input_size, uint column) thread {
    const uint groups = input_size / 32, units = groups / F::MetaGroups;
    const uint plane_tile = column / QUANT_TILE_ROWS, plane_row = column % QUANT_TILE_ROWS;
    w0 = p0 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P0;
    w1 = p1 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P1;
    meta = m + (ulong(plane_tile) * units * QUANT_TILE_ROWS + plane_row) * F::MetaBytes;
  }
  typename F::Payload load(uint g) const thread {
    return F::load(w0 + ulong(g) * QUANT_TILE_ROWS * F::P0, w1 + ulong(g) * QUANT_TILE_ROWS * F::P1);
  }
  typename F::Meta loadMeta(uint g) const thread {
    return F::loadMeta(meta + ulong(g / F::MetaGroups) * QUANT_TILE_ROWS * F::MetaBytes);
  }
};

// Group g of Streams columns (payload w[s], meta unit m[s]) into the float4 sums acc[s][r] of R rows, row r at
// x + rows[r].
template <class F, ushort Streams, ushort R, bool TgGrid>
inline void live_rows_group(thread const typename F::Payload (&w)[Streams], thread const typename F::Meta (&m)[Streams],
                            uint g, device const bfloat *x, thread const ulong (&rows)[R], threadgroup half2 *tl,
                            threadgroup const uint *grid, thread float4 (&acc)[Streams][R]) {
  QuantCoef k[Streams];
#pragma unroll
  for (ushort s = 0; s < Streams; ++s) k[s] = live_rows_coef<F>(w[s], m[s], g % F::MetaGroups);
  // A row's inputs of chunks 2 j and 2 j + 1, eight low then eight high, in two 16-byte loads.
#pragma unroll
  for (ushort j = 0; j < 2; ++j) {
    uint4 xl[R], xh[R];
#pragma unroll
    for (ushort r = 0; r < R; ++r) {
      device const bfloat *xr = x + rows[r] + ulong(g) * 32 + 8 * j;
      xl[r] = *((device const uint4 *)xr);
      xh[r] = *((device const uint4 *)(xr + 16));
    }
#pragma unroll
    for (ushort c = 2 * j; c < 2 * j + 2; ++c) {
      half4 lo[Streams], hi[Streams];
#pragma unroll
      for (ushort s = 0; s < Streams; ++s) live_rows_chunk<F, TgGrid>(F::chunk(w[s], c), k[s], tl, grid, lo[s], hi[s]);
#pragma unroll
      for (ushort r = 0; r < R; ++r) {
        const float4 xlo = float4(as_type<bfloat4>(c & 1 ? xl[r].zw : xl[r].xy));
        const float4 xhi = float4(as_type<bfloat4>(c & 1 ? xh[r].zw : xh[r].xy));
#pragma unroll
        for (ushort s = 0; s < Streams; ++s) {
          acc[s][r] = fma(float4(lo[s]), xlo, acc[s][r]);
          acc[s][r] = fma(float4(hi[s]), xhi, acc[s][r]);
        }
      }
    }
  }
}

// The sums of parts [first_part, first_part + ThreadParts) of the Parts parts of K (input_size / 32 groups, a
// multiple of Parts) for Streams columns and R rows: sum[p][s][r] is part first_part + p's reduced float4. The
// payloads and meta units of the next groups load while the current ones multiply. The Fast tier's kernels take
// kLiveRowsParts parts, and #367's decode passes would take eight, one per simdgroup.
template <class F, ushort Parts, ushort ThreadParts, ushort Streams, ushort R, bool TgGrid>
inline void live_rows_sums(thread const LiveRowsColumn<F> (&col)[Streams], uint groups, uint first_part,
                           device const bfloat *x, thread const ulong (&rows)[R], threadgroup half2 *tl,
                           threadgroup const uint *grid, thread float (&sum)[ThreadParts][Streams][R]) {
  float4 acc[ThreadParts][Streams][R];
  typename F::Payload w[ThreadParts][Streams];
  typename F::Meta m[ThreadParts][Streams];
#pragma unroll
  for (ushort p = 0; p < ThreadParts; ++p)
#pragma unroll
    for (ushort s = 0; s < Streams; ++s) {
#pragma unroll
      for (ushort r = 0; r < R; ++r) acc[p][s][r] = 0.0f;
      w[p][s] = col[s].load(first_part + p);
      m[p][s] = col[s].loadMeta(first_part + p);
    }
  for (uint g = first_part; g < groups; g += Parts) {
    typename F::Payload cw[ThreadParts][Streams];
    typename F::Meta cm[ThreadParts][Streams];
#pragma unroll
    for (ushort p = 0; p < ThreadParts; ++p)
#pragma unroll
      for (ushort s = 0; s < Streams; ++s) {
        cw[p][s] = w[p][s];
        cm[p][s] = m[p][s];
      }
    if (g + Parts < groups) {
#pragma unroll
      for (ushort p = 0; p < ThreadParts; ++p) {
        const uint next = g + Parts + p;
        // The meta unit only where the next group enters a new one.
        const bool unit = F::MetaGroups == 1 || next / F::MetaGroups != (next - Parts) / F::MetaGroups;
#pragma unroll
        for (ushort s = 0; s < Streams; ++s) {
          w[p][s] = col[s].load(next);
          if (unit) m[p][s] = col[s].loadMeta(next);
        }
      }
    }
#pragma unroll
    for (ushort p = 0; p < ThreadParts; ++p) live_rows_group<F, Streams, R, TgGrid>(cw[p], cm[p], g + p, x, rows, tl, grid, acc[p]);
  }
#pragma unroll
  for (ushort p = 0; p < ThreadParts; ++p)
#pragma unroll
    for (ushort s = 0; s < Streams; ++s)
#pragma unroll
      for (ushort r = 0; r < R; ++r) sum[p][s][r] = (acc[p][s][r].x + acc[p][s][r].y) + (acc[p][s][r].z + acc[p][s][r].w);
}

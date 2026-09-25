#pragma once
#include "metal/abi/QuantFormat.h"
#include "metal/abi/QuantTables.h"
#include <metal_stdlib>
using namespace metal;

// Per-format decoding of one (row, group of 32) of the MDGG0001 image, shared
// by the GEMM kernels, to the values of llama.cpp's dequantize.h (MIT notice
// in THIRD_PARTY_NOTICES). A group is read by chunk (metal/abi/QuantFormat.h):
// chunk c holds pairs p = 0..3; pairs 0, 1 are elements 4c..4c+3 of the first
// 16-group and pairs 2, 3 are elements 16+4c..16+4c+3 of the second. A format
// has its id, its sizes P0, P1, MetaBytes and MetaGroups from kQuantFormats,
// its code offset Zero (0 but for the linear formats with a zero point), the
// Group of elements one coefficient covers (32, or 16 when the halves of a
// group have their own) and
//   load(plane0, plane1) -> Payload, loadMeta(meta) -> Meta
//   chunk(Payload, c) -> Chunk
//   loadChunk(plane0, plane1, c) -> Chunk, the same chunk read on its own
//   coef(Meta, j) -> QuantCoef of group j of the meta unit
//     (ScaleInChunk: coef(Meta, Chunk), the group's scale is in every chunk
//     of it rather than in the meta unit)
// and one element accessor, by Kind:
//   QuantLinear    codes(Chunk) -> uint4, pair p in component p with e0 at
//                  bit 0 and e1 at bit 16; value = s * (code - Zero) + m
//   QuantCodebook  indices(Chunk) -> uint, byte p indexes pair p in the
//                  format's pair table (quant_pair_table, of its value(i));
//                  value = s * table value
//   QuantInt8      values(Chunk) -> uint2, the int8 values of pairs 0, 1 (x)
//                  and 2, 3 (y); value = s * int8
//   (both with Scale, the narrowest type that holds s exactly)
//   QuantGrid      grid(Chunk) -> uint2, the grid magnitudes of pairs 0, 1
//                  (x) and 2, 3 (y); signs(Chunk) bit 2p + i negates element
//                  i of pair p; value = s * signed magnitude
enum QuantKind : ushort { QuantLinear, QuantCodebook, QuantInt8, QuantGrid };

// s.x scales and m.x offsets pairs 0, 1, s.y and m.y pairs 2, 3 (equal for
// 32-element coefficient groups).
struct QuantCoef {
  float2 s;
  float2 m = float2(0.0f);
};

// The id, sizes and traits of format F of kind K with code offset Z, whose
// coefficients cover G elements and, with InChunk, whose scales are in every
// chunk of a group.
#define QUANT_FORMAT(F, K, Z, G, InChunk)                                                                        \
  enum : uint { Id = F, P0 = kQuantFormats[F].plane0_bytes, P1 = kQuantFormats[F].plane1_bytes, MetaBytes = kQuantFormats[F].meta_bytes }; \
  enum : ushort { MetaGroups = kQuantFormats[F].meta_groups, Zero = Z, Group = G };                              \
  static constexpr constant QuantKind Kind = K;                                                                  \
  static constexpr constant bool ScaleInChunk = InChunk

// Entries of a codebook format's pair table: one per index byte.
constant constexpr uint kQuantPairTableEntries = 256;
// Fills the threadgroup table of format F's value pairs, as half2 (staged tiles) or bfloat2 (register tiles), entry
// b = (F::value(b & 15), F::value(b >> 4)) for the index byte b of a pair, when F is a codebook format; called by all
// threads of the threadgroup.
template <class F, class T>
inline void quant_pair_table(threadgroup T *table, uint thread_index, uint threads) {
  if constexpr (F::Kind == QuantCodebook) {
    for (uint i = thread_index; i < kQuantPairTableEntries; i += threads) table[i] = T(float2(F::value(i & 15), F::value(i >> 4)));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// The pair words of a word of 4-bit codes: pair p's e0 at bits 4p, e1 at 16 + 4p.
inline uint4 quant_nibble_pairs(uint word) { return (uint4(word) >> uint4(0, 4, 8, 12)) & 0x000F000Fu; }
// Pair-word order of bit fields: the 1-bit fields of the two chunk bytes of a halfword (bit 2p + i of byte b)
// go to bits 8b + 2p (e0) and 16 + 8b + 2p (e1); the 2-bit fields of a chunk halfword (bits 4p + 2i) to bits
// 4p (e0) and 16 + 4p (e1).
inline uint quant_spread1(uint bits) { return (bits & 0x5555u) | ((bits & 0xAAAAu) << 15); }
inline uint quant_spread2(uint bits) { return (bits & 0x3333u) | ((bits & 0xCCCCu) << 14); }

// block_q4_K / block_q5_K header: s = d * sc and m = -dmin * mn with the 6-bit sc, mn of group j. The 12 scale
// bytes are hdr.y (0-3), hdr.z (4-7) and hdr.w (8-11), taken with shifts: the group index is not a compile-time
// constant, and indexing a thread-local byte array or vector by it costs ~8% of the eight-row staged kernel on
// Apple10 (Yesheng Liang's measurement in incoai/splash 77beaed).
inline QuantCoef quant_k4_coef(uint4 hdr, ushort j) {
  uint sc, m;
  if (j < 4) {
    const uint sh = 8u * j;
    sc = (hdr.y >> sh) & 63u;
    m = (hdr.z >> sh) & 63u;
  } else {
    const uint sh = 8u * (j - 4), w = hdr.w >> sh;
    sc = (w & 0xFu) | (((hdr.y >> sh) >> 6) & 3u) << 4;
    m = ((w >> 4) & 0xFu) | (((hdr.z >> sh) >> 6) & 3u) << 4;
  }
  const half d = as_type<half>(ushort(hdr.x & 0xFFFF)), dmin = as_type<half>(ushort(hdr.x >> 16));
  return {float2(float(d) * float(sc)), float2(-float(dmin) * float(m))};
}

// Scale bytes 2j, 2j + 1 of 16 bytes: bytes 2(j & 1), 2(j & 1) + 1 of word j >> 1, in bits 0..15 (shifts, as for
// Q4_K).
inline uint quant_scale_pair(packed_uint4 bytes, ushort j) {
  const uint word = j < 2 ? bytes.x : j < 4 ? bytes.y : j < 6 ? bytes.z : bytes.w;
  return word >> (16u * (j & 1));
}

// Q4_K: plane0 4-bit codes; meta the 16-byte block header.
struct FmtQ4K {
  QUANT_FORMAT(GGUF_FMT_Q4K, QuantLinear, 0, 32, false);
  struct Payload { uint4 a; }; typedef uint Chunk; typedef uint4 Meta;
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *((device uint4 *)m); }
  static Chunk chunk(Payload w, ushort c) { return w.a[c]; }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint *)(p0 + 4 * c)); }
  static uint4 codes(Chunk q) { return quant_nibble_pairs(q); }
  static QuantCoef coef(Meta hdr, ushort j) { return quant_k4_coef(hdr, j); }
};
// Q5_K: plane0 low 4 bits, plane1 the fifth bits (byte c = chunk c); meta as Q4_K.
struct FmtQ5K {
  QUANT_FORMAT(GGUF_FMT_Q5K, QuantLinear, 0, 32, false);
  struct Payload { uint4 a; uint b; }; typedef uint2 Chunk; typedef uint4 Meta;
  static Payload load(device uchar *p0, device uchar *p1) { return {*((device uint4 *)p0), *((device uint *)p1)}; }
  static Meta loadMeta(device uchar *m) { return *((device uint4 *)m); }
  static Chunk chunk(Payload w, ushort c) { return uint2(w.a[c], quant_spread1(w.b >> (16 * (c >> 1))) >> (8 * (c & 1))); }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) {
    return uint2(*((device uint *)(p0 + 4 * c)), quant_spread1(p1[c]));
  }
  static uint4 codes(Chunk q) { return quant_nibble_pairs(q.x) | (((uint4(q.y) >> uint4(0, 2, 4, 6)) & 0x00010001u) << 4); }
  static QuantCoef coef(Meta hdr, ushort j) { return quant_k4_coef(hdr, j); }
};
// Q6_K: plane0 low 4 bits, plane1 the high 2 bits (halfword c = chunk c); meta 16 int8 scales, then half d.
// value = d * sc * (q - 32) with one scale per 16-group.
struct FmtQ6K {
  QUANT_FORMAT(GGUF_FMT_Q6K, QuantLinear, 32, 16, false);
  struct Payload { uint4 a; uint2 b; }; typedef uint2 Chunk; struct Meta { packed_uint4 sc; uint d; };
  static Payload load(device uchar *p0, device uchar *p1) { return {*((device uint4 *)p0), *((device uint2 *)p1)}; }
  static Meta loadMeta(device uchar *m) { Meta r; r.sc = *((device packed_uint4 *)m); r.d = *((device uint *)(m + 16)); return r; }
  static Chunk chunk(Payload w, ushort c) { return uint2(w.a[c], quant_spread2(w.b[c >> 1] >> (16 * (c & 1)))); }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) {
    return uint2(*((device uint *)(p0 + 4 * c)), quant_spread2(*((device ushort *)(p1 + 2 * c))));
  }
  static uint4 codes(Chunk q) { return quant_nibble_pairs(q.x) | (((uint4(q.y) >> uint4(0, 4, 8, 12)) & 0x00030003u) << 4); }
  static QuantCoef coef(Meta mt, ushort j) {
    const float d = float(as_type<half>(ushort(mt.d & 0xFFFF)));
    const uint pair = quant_scale_pair(mt.sc, j);   // the int8 scales of both 16-groups
    return {float2(d * float(as_type<char>(uchar(pair & 0xFFu))), d * float(as_type<char>(uchar(pair >> 8))))};
  }
};
// Q3_K: plane0 low 2 bits (halfword c = chunk c), plane1 the hmask bits (byte c = chunk c); meta half d, 2 zero
// bytes, the 12 packed scale bytes. value = d * (sc - 32) * (q - 4) with one scale per 16-group.
struct FmtQ3K {
  QUANT_FORMAT(GGUF_FMT_Q3K, QuantLinear, 4, 16, false);
  struct Payload { uint2 a; uint b; }; typedef uint2 Chunk; typedef uint4 Meta;
  static Payload load(device uchar *p0, device uchar *p1) { return {*((device uint2 *)p0), *((device uint *)p1)}; }
  static Meta loadMeta(device uchar *m) { return *((device uint4 *)m); }
  static Chunk chunk(Payload w, ushort c) {
    return uint2(quant_spread2(w.a[c >> 1] >> (16 * (c & 1))), quant_spread1(w.b >> (16 * (c >> 1))) >> (8 * (c & 1)));
  }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) {
    return uint2(quant_spread2(*((device ushort *)(p0 + 2 * c))), quant_spread1(p1[c]));
  }
  static uint4 codes(Chunk q) { return ((uint4(q.x) >> uint4(0, 4, 8, 12)) & 0x00030003u) | (((uint4(q.y) >> uint4(0, 2, 4, 6)) & 0x00010001u) << 2); }
  static QuantCoef coef(Meta mt, ushort j) {
    const float d = float(as_type<half>(ushort(mt.x & 0xFFFF)));
    const uint t0 = mt.y, t1 = mt.z, t2 = mt.w;   // scales[0..3], [4..7], [8..11]
    uint aux;
    switch (j >> 1) {
      case 0: aux = (t0 & 0x0f0f0f0fu) | (((t2 >> 0) & 0x03030303u) << 4); break;
      case 1: aux = (t1 & 0x0f0f0f0fu) | (((t2 >> 2) & 0x03030303u) << 4); break;
      case 2: aux = ((t0 >> 4) & 0x0f0f0f0fu) | (((t2 >> 4) & 0x03030303u) << 4); break;
      default: aux = ((t1 >> 4) & 0x0f0f0f0fu) | (((t2 >> 6) & 0x03030303u) << 4); break;
    }
    const uint pair = aux >> (16u * (j & 1));   // 6-bit scales 2j, 2j + 1 (shifts, as for Q4_K)
    return {float2(d * float(int(pair & 0xFFu) - 32), d * float(int((pair >> 8) & 0xFFu) - 32))};
  }
};
// IQ4_XS: plane0 codebook indices; meta half d, scales_h, scales_l[4]. value = d * (ls - 32) * codebook.
struct FmtIQ4XS {
  QUANT_FORMAT(GGUF_FMT_IQ4XS, QuantCodebook, 0, 32, false);
  struct Payload { uint4 a; }; typedef uint Chunk; typedef uint2 Meta; typedef float Scale;
  static half value(uint i) { return half(kIQ4NLValues[i]); }
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *((device uint2 *)m); }
  static Chunk chunk(Payload w, ushort c) { return w.a[c]; }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint *)(p0 + 4 * c)); }
  static uint indices(Chunk q) { return q; }
  static QuantCoef coef(Meta mt, ushort j) {
    const half d = as_type<half>(ushort(mt.x & 0xFFFF)); const uint sh = mt.x >> 16;
    const int ls = int((mt.y >> (4 * j)) & 0xF) | int(((sh >> (2 * j)) & 3) << 4);
    return {float2(float(d) * float(ls - 32))};
  }
};
// IQ4_NL: plane0 codebook indices; meta half d per group. value = d * codebook.
struct FmtIQ4NL {
  QUANT_FORMAT(GGUF_FMT_IQ4NL, QuantCodebook, 0, 32, false);
  struct Payload { uint4 a; }; typedef uint Chunk; typedef ushort Meta; typedef half Scale;
  static half value(uint i) { return half(kIQ4NLValues[i]); }
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static Chunk chunk(Payload w, ushort c) { return w.a[c]; }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint *)(p0 + 4 * c)); }
  static uint indices(Chunk q) { return q; }
  static QuantCoef coef(Meta mt, ushort) { return {float2(float(as_type<half>(mt)))}; }
};
// Q8_0: plane0 int8 values (bytes 8c..8c+7 = chunk c); meta half d per group.
struct FmtQ80 {
  QUANT_FORMAT(GGUF_FMT_Q80, QuantInt8, 0, 32, false);
  struct Payload { uint4 a; uint4 b; }; typedef uint2 Chunk; typedef ushort Meta; typedef half Scale;
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0), *((device uint4 *)(p0 + 16))}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static Chunk chunk(Payload w, ushort c) { const uint4 h = c < 2 ? w.a : w.b; return (c & 1) ? h.zw : h.xy; }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint2 *)(p0 + 8 * c)); }
  static uint2 values(Chunk q) { return q; }
  static QuantCoef coef(Meta mt, ushort) { return {float2(float(as_type<half>(mt)))}; }
};
// IQ3_S: plane0 word c = the 8-bit grid indices of pairs 0, 1 and 2, 3, chunk c's sign bits, the two ninth index
// bits and the group's 4-bit scale; meta half d per super-block. value = d * (1 + 2 * scale) * signed grid value.
struct FmtIQ3S {
  QUANT_FORMAT(GGUF_FMT_IQ3S, QuantGrid, 0, 32, true);
  struct Payload { uint4 a; }; typedef uint Chunk; typedef ushort Meta;
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static Chunk chunk(Payload w, ushort c) { return w.a[c]; }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint *)(p0 + 4 * c)); }
  static uint2 grid(Chunk q) { return uint2(kIQ3SGrid[(q & 0xFF) | ((q >> 16) & 0x100)], kIQ3SGrid[((q >> 8) & 0xFF) | ((q >> 17) & 0x100)]); }
  static uint signs(Chunk q) { return (q >> 16) & 0xFF; }
  static QuantCoef coef(Meta mt, Chunk q) { return {float2(float(as_type<half>(mt)) * float(1 + 2 * ((q >> 26) & 0xF)))}; }
};
// Q2_K: plane0 2-bit codes (halfword c = chunk c); meta half d, half dmin, the 16 scale bytes. value = d * sc * q -
// dmin * mn with the low (sc) and high (mn) nibble of one scale byte per 16-group.
struct FmtQ2K {
  QUANT_FORMAT(GGUF_FMT_Q2K, QuantLinear, 0, 16, false);
  struct Payload { uint2 a; }; typedef uint Chunk; struct Meta { uint dm; packed_uint4 sc; };
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint2 *)p0)}; }
  static Meta loadMeta(device uchar *m) { Meta r; r.dm = *((device uint *)m); r.sc = *((device packed_uint4 *)(m + 4)); return r; }
  static Chunk chunk(Payload w, ushort c) { return quant_spread2(w.a[c >> 1] >> (16 * (c & 1))); }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return quant_spread2(*((device ushort *)(p0 + 2 * c))); }
  static uint4 codes(Chunk q) { return (uint4(q) >> uint4(0, 4, 8, 12)) & 0x00030003u; }
  static QuantCoef coef(Meta mt, ushort j) {
    const float d = float(as_type<half>(ushort(mt.dm & 0xFFFF))), dmin = float(as_type<half>(ushort(mt.dm >> 16)));
    const uint pair = quant_scale_pair(mt.sc, j);   // the scale bytes of both 16-groups
    return {float2(d * float(pair & 0xF), d * float((pair >> 8) & 0xF)),
            float2(-dmin * float((pair >> 4) & 0xF), -dmin * float((pair >> 12) & 0xF))};
  }
};
// The eight signs of a 7-bit sign index (IQ2_XXS, IQ2_XS, IQ3_XXS): its bits, and as bit 7 their parity, which
// makes the count of negated elements even (llama.cpp's ksigns_iq2xs).
inline uint quant_signs7(uint index) { return index | (popcount(index) & 1) << 7; }
// The sign bits of chunk c from a word of four 7-bit sign indices, index l at bits 7l covering elements 8l..8l+7
// (IQ2_XXS, IQ3_XXS): elements 4c..4c+3 are half c & 1 of index c >> 1, 16+4c..16+4c+3 of index 2 + (c >> 1).
inline uint quant_chunk_signs7(uint word, ushort c) {
  const uint l = 7u * (c >> 1), h = 4u * (c & 1);
  return ((quant_signs7((word >> l) & 127) >> h) & 0xF) | ((quant_signs7((word >> (l + 14)) & 127) >> h) & 0xF) << 4;
}
// The magnitudes of half h of two eight-element grid entries (IQ2): elements 4h..4h+3 of each.
inline uint2 quant_grid8(uint64_t a, uint64_t b, uint h) {
  return h ? uint2(as_type<uint2>(a).y, as_type<uint2>(b).y) : uint2(as_type<uint2>(a).x, as_type<uint2>(b).x);
}
// The coefficients of an IQ2 group: d * (1 + 2 * scale) / 8, of the low (pairs 0, 1) and high (pairs 2, 3) nibble
// of `scales`, which are equal where one scale covers the group.
inline QuantCoef quant_iq2_coef(ushort d, uint scales) {
  const float s = float(as_type<half>(d));
  return {float2(s * float(1 + 2 * (scales & 0xF)) * 0.125f, s * float(1 + 2 * ((scales >> 4) & 0xF)) * 0.125f)};
}
// IQ3_XXS: plane0 the group's 8-bit grid indices (byte t: elements 4t..4t+3), plane1 its word of four 7-bit sign
// indices (index l at bit 7l: elements 8l..8l+7) and the 4-bit scale (bits 28..31); meta half d. value = d * (1 + 2 *
// scale) / 4 * signed grid value. A chunk is IQ3_S's word without ninth index bits.
struct FmtIQ3XXS {
  QUANT_FORMAT(GGUF_FMT_IQ3XXS, QuantGrid, 0, 32, true);
  struct Payload { uint2 a; uint b; }; typedef uint Chunk; typedef ushort Meta;
  static Payload load(device uchar *p0, device uchar *p1) { return {*((device uint2 *)p0), *((device uint *)p1)}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static Chunk chunk(Payload w, ushort c) {
    return ((w.a.x >> (8 * c)) & 0xFF) | ((w.a.y >> (8 * c)) & 0xFF) << 8 | quant_chunk_signs7(w.b, c) << 16 | (w.b >> 28) << 26;
  }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) { return chunk(load(p0, p1), c); }
  static uint2 grid(Chunk q) { return uint2(kIQ3XXSGrid[q & 0xFF], kIQ3XXSGrid[(q >> 8) & 0xFF]); }
  static uint signs(Chunk q) { return (q >> 16) & 0xFF; }
  static QuantCoef coef(Meta mt, Chunk q) { return {float2(float(as_type<half>(mt)) * float(1 + 2 * ((q >> 26) & 0xF)) * 0.25f)}; }
};
// IQ2_XXS: plane0 the group's native 8 bytes, word 0 the 8-bit grid indices of its four eight-element entries (byte
// l: elements 8l..8l+7) and word 1 their 7-bit sign indices (bits 7l) and the 4-bit scale (bits 28..31); meta half
// d. value = d * (1 + 2 * scale) / 8 * signed grid value. A chunk holds half c & 1 of entries c >> 1 (pairs 0, 1) and
// 2 + (c >> 1) (pairs 2, 3): their indices at bits 0..7 and 8..15, the chunk's sign bits at 16..23, the half at 24
// and the scale at 28..31.
struct FmtIQ2XXS {
  QUANT_FORMAT(GGUF_FMT_IQ2XXS, QuantGrid, 0, 32, true);
  struct Payload { uint2 a; }; typedef uint Chunk; typedef ushort Meta;
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint2 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static Chunk chunk(Payload w, ushort c) {
    const uint l = 8u * (c >> 1);
    return ((w.a.x >> l) & 0xFF) | ((w.a.x >> (l + 16)) & 0xFF) << 8 | quant_chunk_signs7(w.a.y, c) << 16 | uint(c & 1) << 24 |
           (w.a.y & 0xF0000000u);
  }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) { return chunk(load(p0, p1), c); }
  static uint2 grid(Chunk q) { return quant_grid8(kIQ2XXSGrid[q & 0xFF], kIQ2XXSGrid[(q >> 8) & 0xFF], (q >> 24) & 1); }
  static uint signs(Chunk q) { return (q >> 16) & 0xFF; }
  static QuantCoef coef(Meta mt, Chunk q) { return quant_iq2_coef(mt, (q >> 28) * 0x11u); }
};
// IQ2_XS: plane0 the group's four 16-bit entries, entry l (elements 8l..8l+7) a 9-bit grid index and a 7-bit sign
// index; plane1 its scales byte, the scale of the first 16 elements in bits 0..3 and of the others in 4..7; meta half
// d. value = d * (1 + 2 * scale) / 8 * signed grid value. A chunk: x the grid indices of half c & 1 of entries c >> 1
// (bits 0..15) and 2 + (c >> 1) (16..31), y the chunk's sign bits, the scales byte at bit 8 and the half at 16.
struct FmtIQ2XS {
  QUANT_FORMAT(GGUF_FMT_IQ2XS, QuantGrid, 0, 16, true);
  struct Payload { uint2 a; uchar b; }; typedef uint2 Chunk; typedef ushort Meta;
  static Payload load(device uchar *p0, device uchar *p1) { return {*((device uint2 *)p0), *p1}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static Chunk chunk(Payload w, ushort c) {
    const uint l = 16u * (c >> 1), h = 4u * (c & 1), e0 = (w.a.x >> l) & 0xFFFF, e1 = (w.a.y >> l) & 0xFFFF;
    const uint signs = ((quant_signs7(e0 >> 9) >> h) & 0xF) | ((quant_signs7(e1 >> 9) >> h) & 0xF) << 4;
    return uint2((e0 & 511) | (e1 & 511) << 16, signs | uint(w.b) << 8 | uint(c & 1) << 16);
  }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) { return chunk(load(p0, p1), c); }
  static uint2 grid(Chunk q) { return quant_grid8(kIQ2XSGrid[q.x & 0xFFFF], kIQ2XSGrid[q.x >> 16], (q.y >> 16) & 1); }
  static uint signs(Chunk q) { return q.y & 0xFF; }
  static QuantCoef coef(Meta mt, Chunk q) { return quant_iq2_coef(mt, q.y >> 8); }
};
// IQ2_S: plane0 the group's four low grid index bytes (word 0, byte l: entry l, elements 8l..8l+7) and their four
// sign bytes (word 1); plane1 the two high index bits of entry l at bits 2l, then the scales byte as IQ2_XS's; meta
// half d. value = d * (1 + 2 * scale) / 8 * signed grid value. A chunk is IQ2_XS's, with 10-bit grid indices.
struct FmtIQ2S {
  QUANT_FORMAT(GGUF_FMT_IQ2S, QuantGrid, 0, 16, true);
  struct Payload { uint2 a; ushort b; }; typedef uint2 Chunk; typedef ushort Meta;
  static Payload load(device uchar *p0, device uchar *p1) { return {*((device uint2 *)p0), *((device ushort *)p1)}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static Chunk chunk(Payload w, ushort c) {
    const uint l = c >> 1, h = 4u * (c & 1), qh = w.b;
    const uint i0 = ((w.a.x >> (8 * l)) & 0xFF) | ((qh >> (2 * l)) & 3) << 8;
    const uint i1 = ((w.a.x >> (8 * l + 16)) & 0xFF) | ((qh >> (2 * l + 4)) & 3) << 8;
    const uint signs = ((w.a.y >> (8 * l + h)) & 0xF) | ((w.a.y >> (8 * l + 16 + h)) & 0xF) << 4;
    return uint2(i0 | i1 << 16, signs | (qh >> 8) << 8 | uint(c & 1) << 16);
  }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) { return chunk(load(p0, p1), c); }
  static uint2 grid(Chunk q) { return quant_grid8(kIQ2SGrid[q.x & 0xFFFF], kIQ2SGrid[q.x >> 16], (q.y >> 16) & 1); }
  static uint signs(Chunk q) { return q.y & 0xFF; }
  static QuantCoef coef(Meta mt, Chunk q) { return quant_iq2_coef(mt, q.y >> 8); }
};
// Bytes 0, 1 and 2, 3 of a word as the pair words of two pairs.
inline uint2 quant_byte_pairs(uint v) { return uint2((v & 0xFFu) | (v & 0xFF00u) << 8, ((v >> 16) & 0xFFu) | ((v >> 8) & 0xFF0000u)); }
// The linear codes of an IQ1 chunk, 8 (grid value + 1) + 1 + delta with delta = +-1 (Zero 9), from the kIQ1SGrid
// indices of pairs 0, 1 (bits 0..10) and pairs 2, 3 (11..21), the half of their entries (22) and their negative
// deltas (23, 24).
inline uint4 quant_iq1_codes(uint q) {
  const uint h = 4 * ((q >> 22) & 1);
  const uint2 lo = quant_byte_pairs((kIQ1SGrid[q & 0x7FF] >> h) & 0x0F0F0F0Fu);
  const uint2 hi = quant_byte_pairs((kIQ1SGrid[(q >> 11) & 0x7FF] >> h) & 0x0F0F0F0Fu);
  const uint2 delta = select(uint2(0x00020002u), uint2(0u), bool2((q >> 23) & 1, (q >> 24) & 1));
  return uint4(lo << 3 | delta.x, hi << 3 | delta.y);
}

// IQ1_S: plane0 the group's four low grid index bytes (entry l: elements 8l..8l+7), plane1 its qh, the three high
// index bits of entry l at bits 3l, the 3-bit scale at 12 and a negative delta at 15; meta half d. value = d * (2 *
// scale + 1) * (grid value + delta), delta = +-1/8, which is s * (code - 9) with s = d * (2 * scale + 1) / 8 and the
// codes of quant_iq1_codes. A chunk: the indices of half c & 1 of entries c >> 1 and 2 + (c >> 1), that half, the
// deltas and the scale at bits 25..27.
struct FmtIQ1S {
  QUANT_FORMAT(GGUF_FMT_IQ1S, QuantLinear, 9, 32, true);
  struct Payload { uint a; ushort b; }; typedef uint Chunk; typedef ushort Meta;
  static Payload load(device uchar *p0, device uchar *p1) { return {*((device uint *)p0), *((device ushort *)p1)}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static Chunk chunk(Payload w, ushort c) {
    const uint l = c >> 1, qh = w.b;
    const uint i0 = ((w.a >> (8 * l)) & 0xFF) | ((qh >> (3 * l)) & 7) << 8;
    const uint i1 = ((w.a >> (8 * l + 16)) & 0xFF) | ((qh >> (3 * l + 6)) & 7) << 8;
    return i0 | i1 << 11 | uint(c & 1) << 22 | ((qh >> 15) & 1) * 3 << 23 | ((qh >> 12) & 7) << 25;
  }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) { return chunk(load(p0, p1), c); }
  static uint4 codes(Chunk q) { return quant_iq1_codes(q); }
  static QuantCoef coef(Meta mt, Chunk q) {
    return {float2(float(as_type<half>(mt)) * float(2 * ((q >> 25) & 7) + 1) * 0.125f)};
  }
};
// IQ1_M: plane0 the group's four low grid index bytes, plane1 its two qh bytes, entry l's three high index bits at
// bit 4l and its negative delta at 4l + 3; meta the block's 8 scale bytes, four halfwords, halfword k holding the
// 3-bit scales of the halves of groups 2k (bits 0..5) and 2k + 1 (6..11) and bits 4k..4k+3 of the half d at its
// bits 12..15. value = d * (2 * scale + 1) * (grid value + delta) as IQ1_S's, with a scale per 16 elements and a
// delta per 8.
struct FmtIQ1M {
  QUANT_FORMAT(GGUF_FMT_IQ1M, QuantLinear, 9, 16, false);
  struct Payload { uint a; ushort b; }; typedef uint Chunk; typedef uint2 Meta;
  static Payload load(device uchar *p0, device uchar *p1) { return {*((device uint *)p0), *((device ushort *)p1)}; }
  static Meta loadMeta(device uchar *m) { return *((device uint2 *)m); }
  static Chunk chunk(Payload w, ushort c) {
    const uint l = c >> 1, qh = w.b;
    const uint i0 = ((w.a >> (8 * l)) & 0xFF) | ((qh >> (4 * l)) & 7) << 8;
    const uint i1 = ((w.a >> (8 * l + 16)) & 0xFF) | ((qh >> (4 * l + 8)) & 7) << 8;
    return i0 | i1 << 11 | uint(c & 1) << 22 | ((qh >> (4 * l + 3)) & 1) << 23 | ((qh >> (4 * l + 11)) & 1) << 24;
  }
  static Chunk loadChunk(device uchar *p0, device uchar *p1, ushort c) { return chunk(load(p0, p1), c); }
  static uint4 codes(Chunk q) { return quant_iq1_codes(q); }
  static QuantCoef coef(Meta mt, ushort j) {
    const uint d = ((mt.x >> 12) & 0xF) | ((mt.x >> 24) & 0xF0) | ((mt.y >> 4) & 0xF00) | ((mt.y >> 16) & 0xF000);
    const uint scales = ((j < 4 ? mt.x : mt.y) >> (16 * ((j >> 1) & 1) + 6 * (j & 1)));   // shifts, as for Q4_K
    const float s = float(as_type<half>(ushort(d)));
    return {float2(s * float(2 * (scales & 7) + 1) * 0.125f, s * float(2 * ((scales >> 3) & 7) + 1) * 0.125f)};
  }
};
// Q4_0: plane0 4-bit codes; meta half d. value = d * (q - 8).
struct FmtQ40 {
  QUANT_FORMAT(GGUF_FMT_Q40, QuantLinear, 8, 32, false);
  struct Payload { uint4 a; }; typedef uint Chunk; typedef ushort Meta;
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *((device ushort *)m); }
  static Chunk chunk(Payload w, ushort c) { return w.a[c]; }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint *)(p0 + 4 * c)); }
  static uint4 codes(Chunk q) { return quant_nibble_pairs(q); }
  static QuantCoef coef(Meta mt, ushort) { return {float2(float(as_type<half>(mt)))}; }
};
// Q4_1: plane0 4-bit codes; meta half d, half m. value = d * q + m.
struct FmtQ41 {
  QUANT_FORMAT(GGUF_FMT_Q41, QuantLinear, 0, 32, false);
  struct Payload { uint4 a; }; typedef uint Chunk; typedef uint Meta;
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *((device uint *)m); }
  static Chunk chunk(Payload w, ushort c) { return w.a[c]; }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint *)(p0 + 4 * c)); }
  static uint4 codes(Chunk q) { return quant_nibble_pairs(q); }
  static QuantCoef coef(Meta mt, ushort) {
    return {float2(float(as_type<half>(ushort(mt & 0xFFFF)))), float2(float(as_type<half>(ushort(mt >> 16))))};
  }
};
// MXFP4: plane0 codebook indices; meta the E8M0 exponent e per group. value = 2^(e - 128) * kFP4Values, whose
// entries are twice the E2M1 values (llama.cpp's GGML_E8M0_TO_FP32_HALF, 2^-128 and 2^-127 subnormal).
struct FmtMXFP4 {
  QUANT_FORMAT(GGUF_FMT_MXFP4, QuantCodebook, 0, 32, false);
  struct Payload { uint4 a; }; typedef uint Chunk; typedef uchar Meta; typedef float Scale;
  static half value(uint i) { return half(kFP4Values[i]); }
  static Payload load(device uchar *p0, device uchar *) { return {*((device uint4 *)p0)}; }
  static Meta loadMeta(device uchar *m) { return *m; }
  static Chunk chunk(Payload w, ushort c) { return w.a[c]; }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) { return *((device uint *)(p0 + 4 * c)); }
  static uint indices(Chunk q) { return q; }
  static QuantCoef coef(Meta e, ushort) { return {float2(as_type<float>(e < 2 ? 0x00200000u << e : uint(e - 1) << 23))}; }
};

#undef QUANT_FORMAT

// Every format as X(format type, kernel name token), the token being its kQuantFormats name.
#define QUANT_FORMATS(X)                                                                                            \
  X(FmtQ4K, q4k) X(FmtIQ4XS, iq4xs) X(FmtIQ4NL, iq4nl) X(FmtQ5K, q5k) X(FmtQ6K, q6k) X(FmtQ3K, q3k) X(FmtQ80, q80) \
  X(FmtIQ3S, iq3s) X(FmtQ2K, q2k) X(FmtIQ3XXS, iq3xxs) X(FmtIQ2XXS, iq2xxs) X(FmtIQ2XS, iq2xs) X(FmtIQ2S, iq2s)      \
  X(FmtIQ1S, iq1s) X(FmtIQ1M, iq1m) X(FmtQ40, q40) X(FmtQ41, q41) X(FmtMXFP4, mxfp4)

// Runs body(F()) with the format type of run-time format id `format` (GGUF_FMT_*), for kernels whose tiles pick
// their tensor, and so its format, at run time. The branch is uniform in a threadgroup. The host passes known ids
// only; any other decodes as the first format, so every id runs one body (with a path that runs none, the staged
// fused kernel failed its test under shader validation).
template <class Body>
inline void quant_format_switch(uint format, Body body) {
#define QUANT_FORMAT_CASE(F, f) case F::Id: body(F()); break;
  switch (format) {
  default:
    QUANT_FORMATS(QUANT_FORMAT_CASE)
  }
#undef QUANT_FORMAT_CASE
}

// QUANT_FORMATS lists every format, by its kQuantFormats name, and so does the switch above.
#define QUANT_FORMAT_ONE(F, f) +1
static_assert(0 QUANT_FORMATS(QUANT_FORMAT_ONE) == GGUF_FMT_COUNT, "QUANT_FORMATS lists every format");
#undef QUANT_FORMAT_ONE
template <uint N> constexpr bool quant_format_named(uint id, const constant char (&token)[N]) {
  for (uint i = 0; i < N; ++i)
    if (kQuantFormats[id].name[i] != token[i]) return false;
  return true;
}
#define QUANT_FORMAT_NAME(F, f) static_assert(quant_format_named(F::Id, #f), #f " is not its kQuantFormats name");
QUANT_FORMATS(QUANT_FORMAT_NAME)
#undef QUANT_FORMAT_NAME

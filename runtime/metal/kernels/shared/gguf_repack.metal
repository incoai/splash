// Editing this file re-prepares every GGUF model.
#pragma clang fp reassociate(off)
#include "metal/abi/GgufRepack.h"
#include <metal_stdlib>

using namespace metal;

// ---- weight preparation: native GGUF rows -> MDGG0001 planes (metal/abi/QuantFormat.h) ----
// One thread per (row n, 32-wide K group g) of a chunk the host staged in image order.
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
  const uint n = t / G, g = t % G;
  constant QuantFormat &f = kQuantFormats[p.fmt];
  const uint b = g / f.meta_groups, j = g % f.meta_groups;   // native block b holds meta unit b
  device const uchar *blk = src + ulong(n) * p.src_row_bytes + ulong(b) * f.block_bytes;
  device uchar *out0 = dst + quant_tile_index(n, g, G) * f.plane0_bytes;
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
    case GGUF_FMT_Q2K: {   // block_q2_K {scales[16], qs[64], d, dmin}: group j is bits 2 (j % 4) of qs[32 (j / 4)..]
      for (uint e = 0; e < 32; ++e) lo[quant_slot(e)] = (blk[16 + 32 * (j / 4) + e] >> (2 * (j % 4))) & 3;
      gguf_store_bits(lo, 2, out0);
      if (j == 0) { for (uint i = 0; i < 4; ++i) meta[i] = blk[80 + i]; for (uint i = 0; i < 16; ++i) meta[4 + i] = blk[i]; }
      break;
    }
    case GGUF_FMT_PQ20: {   // block_pq2_0 {d, qs[32]}: element e of group j is bits 2 (e % 4) of qs[8 j + e / 4]
      for (uint e = 0; e < 32; ++e) lo[quant_slot(e)] = (blk[2 + 8 * j + e / 4] >> (2 * (e % 4))) & 3;
      gguf_store_bits(lo, 2, out0);
      if (j == 0) { meta[0] = blk[0]; meta[1] = blk[1]; }
      break;
    }
    case GGUF_FMT_Q40: case GGUF_FMT_Q41: case GGUF_FMT_MXFP4: {   // the meta unit (d, d and m, or e), then qs[16]
      for (uint l = 0; l < 16; ++l) { const uchar q = blk[f.meta_bytes + l]; lo[quant_slot(l)] = q & 15; lo[quant_slot(16 + l)] = q >> 4; }
      if (p.fmt == GGUF_FMT_MXFP4) gguf_store_bits(lo, 4, out0); else gguf_store_pairs(lo, out0);
      for (uint i = 0; i < f.meta_bytes; ++i) meta[i] = blk[i];
      break;
    }
    // The IQ3_XXS, IQ2 and IQ1 groups keep their native bytes (metal/abi/QuantFormat.h): block_*'s fields for the
    // group, copied into plane0, plane1 and, once per block, the meta unit.
    case GGUF_FMT_IQ3XXS: {   // {d, qs[64], signs and scales[32]}
      for (uint i = 0; i < 8; ++i) out0[i] = blk[2 + 8 * j + i];
      for (uint i = 0; i < 4; ++i) out1[i] = blk[66 + 4 * j + i];
      if (j == 0) { meta[0] = blk[0]; meta[1] = blk[1]; }
      break;
    }
    case GGUF_FMT_IQ2XXS: case GGUF_FMT_IQ2XS: {   // {d, qs[32] (, scales[8])}
      for (uint i = 0; i < 8; ++i) out0[i] = blk[2 + 8 * j + i];
      if (p.fmt == GGUF_FMT_IQ2XS) out1[0] = blk[66 + j];
      if (j == 0) { meta[0] = blk[0]; meta[1] = blk[1]; }
      break;
    }
    case GGUF_FMT_IQ2S: {   // {d, qs[32], signs[32], qh[8], scales[8]}
      for (uint i = 0; i < 4; ++i) { out0[i] = blk[2 + 4 * j + i]; out0[4 + i] = blk[34 + 4 * j + i]; }
      out1[0] = blk[66 + j]; out1[1] = blk[74 + j];
      if (j == 0) { meta[0] = blk[0]; meta[1] = blk[1]; }
      break;
    }
    case GGUF_FMT_IQ1S: {   // {d, qs[32], qh[8] (16-bit)}
      for (uint i = 0; i < 4; ++i) out0[i] = blk[2 + 4 * j + i];
      out1[0] = blk[34 + 2 * j]; out1[1] = blk[35 + 2 * j];
      if (j == 0) { meta[0] = blk[0]; meta[1] = blk[1]; }
      break;
    }
    case GGUF_FMT_IQ1M: {   // {qs[32], qh[16], scales[8]}
      for (uint i = 0; i < 4; ++i) out0[i] = blk[4 * j + i];
      out1[0] = blk[32 + 2 * j]; out1[1] = blk[33 + 2 * j];
      if (j == 0) for (uint i = 0; i < 8; ++i) meta[i] = blk[48 + i];
      break;
    }
    case GGUF_FMT_IQ3S:
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

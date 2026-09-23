#pragma once

// CPU reference for the GGUF image formats (metal/abi/QuantFormat.h): native
// blocks, their fp32 values with llama.cpp's dequantize_row_* semantics and
// the planes the load-time repack writes. Shared by the GGUF Metal tests.

#include "metal/abi/QuantFormat.h"

#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <string>
#include <vector>

namespace gguf_reference {

enum Fmt { Q4K = GGUF_FMT_Q4K, IQ4XS = GGUF_FMT_IQ4XS, IQ4NL = GGUF_FMT_IQ4NL, Q5K = GGUF_FMT_Q5K, Q6K = GGUF_FMT_Q6K,
           Q3K = GGUF_FMT_Q3K, Q80 = GGUF_FMT_Q80, IQ3S = GGUF_FMT_IQ3S, FMT_COUNT = GGUF_FMT_COUNT };
inline const char *fmtName(uint32_t f) { return kQuantFormats[f].name; }
inline uint16_t f2h(float f) { __fp16 h = (__fp16)f; uint16_t u; memcpy(&u, &h, 2); return u; }
inline float h2f(uint16_t u) { __fp16 h; memcpy(&h, &u, 2); return (float)h; }
inline const float kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
inline const uint32_t iq3s_grid[512] = {
#include "iq3s_grid.inc"
};
inline uint32_t rowBytes(Fmt f, uint32_t K) { const QuantFormat &i = kQuantFormats[f]; return K / i.block_elements * i.block_bytes; }
// N native rows of random bytes whose half scales (d, and dmin for Q4_K/Q5_K) are scale().
template <class Scale>
std::vector<uint8_t> makeNative(Fmt f, uint32_t N, uint32_t K, std::mt19937 &rng, Scale scale) {
  const QuantFormat &fi = kQuantFormats[f]; std::vector<uint8_t> v((size_t)N * rowBytes(f, K)); for (auto &b : v) b = (uint8_t)rng();
  const uint32_t off = f == Q6K ? 208 : f == Q3K ? 108 : 0;
  for (size_t b = 0; b < v.size() / fi.block_bytes; ++b) { uint8_t *blk = v.data() + b * fi.block_bytes;
    const uint16_t d = scale(); memcpy(blk + off, &d, 2);
    if (f == Q4K || f == Q5K) { const uint16_t m = scale(); memcpy(blk + 2, &m, 2); } }
  return v;
}
inline void scale_min_k4(const uint8_t *sc, int j, uint8_t &s, uint8_t &m) { if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; } else { s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); } }
// Planes in chunk order (metal/abi/QuantFormat.h): 4-bit linear codes pair-interleaved in chunk words, every
// other field a little-endian bit string over the slots.
inline void packBits(const uint8_t *slots, uint32_t bits, uint8_t *dst) {
  for (uint32_t w = 0, per = 32 / bits; w < bits; ++w) { uint32_t v = 0; for (uint32_t i = 0; i < per; ++i) v |= uint32_t(slots[w * per + i]) << (bits * i); memcpy(dst + 4 * w, &v, 4); } }
inline void packPairs(const uint8_t *slots, uint8_t *dst) {
  for (uint32_t c = 0; c < 4; ++c) { uint32_t v = 0; for (uint32_t i = 0; i < 8; ++i) v |= uint32_t(slots[8 * c + i]) << ((i & 1) * 16 + 4 * (i >> 1)); memcpy(dst + 4 * c, &v, 4); } }
// reference values (llama.cpp dequantize_row_* semantics) + plane bytes for group g of one row
inline void groupPack(Fmt f, const uint8_t *row, uint32_t g, float vals[32], uint8_t p0[32], uint8_t p1[8]) {
  const QuantFormat &fi = kQuantFormats[f]; const uint8_t *blk = row + (g * 32 / fi.block_elements) * fi.block_bytes; const uint32_t j = (g * 32 % fi.block_elements) / 32;
  uint8_t lo[32], hi[32];   // per slot: the (low) code and its high bits
  switch (f) {
    case Q4K: { const uint8_t *q = blk + 16 + (j / 2) * 32; int sh = (j % 2) * 4; uint16_t d16, m16; memcpy(&d16, blk, 2); memcpy(&m16, blk + 2, 2); uint8_t sc, mn; scale_min_k4(blk + 4, j, sc, mn);
      for (int k = 0; k < 32; ++k) { const uint8_t code = (q[k] >> sh) & 15; lo[quant_slot(k)] = code; vals[k] = h2f(d16) * sc * code - h2f(m16) * mn; }
      packPairs(lo, p0); return; }
    case IQ4XS: { const uint8_t *qs = blk + 8 + j * 16; uint16_t d16, shh; memcpy(&d16, blk, 2); memcpy(&shh, blk + 2, 2); const uint8_t *sl = blk + 4;
      int ls = ((sl[j / 2] >> 4 * (j % 2)) & 0xf) | (((shh >> 2 * j) & 3) << 4); float s = h2f(d16) * (ls - 32);
      for (int k = 0; k < 32; ++k) { const uint8_t code = k < 16 ? (qs[k] & 15) : (qs[k - 16] >> 4); lo[quant_slot(k)] = code; vals[k] = s * kv_iq4nl[code]; }
      packBits(lo, 4, p0); return; }
    case IQ4NL: { const uint8_t *qs = blk + 2; uint16_t d16; memcpy(&d16, blk, 2); float s = h2f(d16);
      for (int k = 0; k < 32; ++k) { const uint8_t code = k < 16 ? (qs[k] & 15) : (qs[k - 16] >> 4); lo[quant_slot(k)] = code; vals[k] = s * kv_iq4nl[code]; }
      packBits(lo, 4, p0); return; }
    case Q5K: { const uint8_t *qh = blk + 16, *ql = blk + 48 + (j / 2) * 32; int sh = (j % 2) * 4; uint16_t d16, m16; memcpy(&d16, blk, 2); memcpy(&m16, blk + 2, 2); uint8_t sc, mn; scale_min_k4(blk + 4, j, sc, mn);
      for (int k = 0; k < 32; ++k) { uint8_t l4 = (ql[k] >> sh) & 15, h1 = (qh[k] >> j) & 1; lo[quant_slot(k)] = l4; hi[quant_slot(k)] = h1; vals[k] = h2f(d16) * sc * (l4 + 16 * h1) - h2f(m16) * mn; }
      packPairs(lo, p0); packBits(hi, 1, p1); return; }
    case Q6K: { const uint32_t n = j / 4, r = j % 4; const uint8_t *ql = blk + 64 * n, *qh = blk + 128 + 32 * n; const int8_t *sc = (const int8_t *)(blk + 192) + 8 * n; uint16_t d16; memcpy(&d16, blk + 208, 2); float d = h2f(d16);
      for (int k = 0; k < 32; ++k) { uint8_t l4 = (ql[k + 32 * (r & 1)] >> (4 * (r >> 1))) & 15, h2 = (qh[k] >> (2 * r)) & 3; int q6 = (l4 | (h2 << 4)) - 32; lo[quant_slot(k)] = l4; hi[quant_slot(k)] = h2; vals[k] = d * sc[2 * r + k / 16] * q6; }
      packPairs(lo, p0); packBits(hi, 2, p1); return; }
    case Q3K: { const uint32_t n = j / 4, jj = j % 4; const uint8_t *hm = blk, *q = blk + 32 + 32 * n; uint32_t aux[4]; memcpy(aux, blk + 96, 12); uint32_t tmp = aux[2];
      const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
      aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4); aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
      aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4); aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
      const int8_t *scales = (const int8_t *)aux; uint16_t d16; memcpy(&d16, blk + 108, 2); float d = h2f(d16);
      for (int k = 0; k < 32; ++k) { int c2 = (q[k] >> (2 * jj)) & 3, hb1 = (hm[k] >> j) & 1; lo[quant_slot(k)] = uint8_t(c2); hi[quant_slot(k)] = uint8_t(hb1); vals[k] = d * (scales[2 * j + k / 16] - 32) * (c2 - (hb1 ? 0 : 4)); }
      packBits(lo, 2, p0); packBits(hi, 1, p1); return; }
    case Q80: { uint16_t d16; memcpy(&d16, blk, 2); const int8_t *qs = (const int8_t *)(blk + 2); for (int k = 0; k < 32; ++k) { vals[k] = h2f(d16) * qs[k]; lo[quant_slot(k)] = uint8_t(qs[k]); }
      packBits(lo, 8, p0); return; }
    default: { const uint8_t *qs = blk + 2 + 8 * j, *qh = blk + 66, *signs = blk + 74 + 4 * j, *scales = blk + 106; uint16_t d16; memcpy(&d16, blk, 2);
      const uint32_t sc = (scales[j / 2] >> (4 * (j % 2))) & 0xf; const float db = h2f(d16) * (1 + 2 * sc);
      for (int l = 0; l < 4; ++l) { const uint8_t *g1 = (const uint8_t *)(iq3s_grid + (qs[2 * l] | ((qh[j] << (8 - 2 * l)) & 256))), *g2 = (const uint8_t *)(iq3s_grid + (qs[2 * l + 1] | ((qh[j] << (7 - 2 * l)) & 256)));
        for (int k = 0; k < 4; ++k) { vals[8 * l + k] = db * g1[k] * ((signs[l] & (1 << k)) ? -1.f : 1.f); vals[8 * l + 4 + k] = db * g2[k] * ((signs[l] & (1 << (4 + k))) ? -1.f : 1.f); } }
      // word c: grid indices of elements 4c.. and 16+4c.. (qs[c], qs[4 + c]), chunk c's signs, their ninth bits, the scale
      for (int k = 0; k < 32; ++k) hi[quant_slot(k)] = (signs[k / 8] >> (k % 8)) & 1;
      uint8_t sign[4]; packBits(hi, 1, sign);
      for (int c = 0; c < 4; ++c) { const uint32_t v = qs[c] | uint32_t(qs[4 + c]) << 8 | uint32_t(sign[c]) << 16 | ((qh[j] >> c) & 1u) << 24 | ((qh[j] >> (4 + c)) & 1u) << 25 | sc << 26; memcpy(p0 + 4 * c, &v, 4); }
      return; }
  }
}
inline void metaPack(Fmt f, const uint8_t *row, uint32_t unit, uint8_t *dst) {
  const uint8_t *blk = row + unit * kQuantFormats[f].block_bytes;
  switch (f) { case Q4K: case Q5K: memcpy(dst, blk, 16); break; case IQ4XS: memcpy(dst, blk, 8); break; case IQ4NL: case Q80: case IQ3S: memcpy(dst, blk, 2); break;
    case Q6K: memcpy(dst, blk + 192, 16); memcpy(dst + 16, blk + 208, 2); dst[18] = dst[19] = 0; break;
    case Q3K: memcpy(dst, blk + 108, 2); dst[2] = dst[3] = 0; memcpy(dst + 4, blk + 96, 12); break; default: break; }
}
struct Packed { std::vector<uint8_t> w0, w1, meta; };
inline Packed repack(Fmt f, const std::vector<uint8_t> &native, uint32_t N, uint32_t K, std::vector<float> *Wf, std::vector<float> *Ws = nullptr) {
  const QuantFormat &fi = kQuantFormats[f]; const uint32_t G = K / 32, rb = rowBytes(f, K), units = G / fi.meta_groups;
  Packed p; p.w0.assign((size_t)N * G * fi.plane0_bytes, 0); p.w1.assign(fi.plane1_bytes ? (size_t)N * G * fi.plane1_bytes : 16, 0); p.meta.assign((size_t)N * units * fi.meta_bytes, 0); if (Wf) Wf->assign((size_t)N * K, 0.f); if (Ws) Ws->assign((size_t)N * K, 0.f);
  for (uint32_t n = 0; n < N; ++n) { const uint8_t *row = native.data() + (size_t)n * rb;
    for (uint32_t g = 0; g < G; ++g) { float vals[32]; uint8_t p0[32], p1[8]; groupPack(f, row, g, vals, p0, p1);
      memcpy(p.w0.data() + quant_tile_index(n, g, G) * fi.plane0_bytes, p0, fi.plane0_bytes); if (fi.plane1_bytes) memcpy(p.w1.data() + quant_tile_index(n, g, G) * fi.plane1_bytes, p1, fi.plane1_bytes);
      if (Wf) for (int k = 0; k < 32; ++k) (*Wf)[(size_t)n * K + g * 32 + k] = vals[k];
      if (Ws) for (int k = 0; k < 32; ++k) (*Ws)[(size_t)n * K + g * 32 + k] = h2f(f2h(vals[k])); }
    for (uint32_t u = 0; u < units; ++u) metaPack(f, row, u, p.meta.data() + quant_tile_index(n, u, units) * fi.meta_bytes); }
  return p;
}
// Upstream GGML's fp32 dequantization of native rows from an unmodified libggml-base (for example
// llama.cpp 7ab4ee7) loaded with dlopen; false and a message when it lacks the format's symbol.
inline bool ggmlDequantize(void *ggml, Fmt f, const std::vector<uint8_t> &native, std::vector<float> &values, std::string &error) {
  static const char *symbols[FMT_COUNT] = {
    "dequantize_row_q4_K", "dequantize_row_iq4_xs", "dequantize_row_iq4_nl", "dequantize_row_q5_K",
    "dequantize_row_q6_K", "dequantize_row_q3_K", "dequantize_row_q8_0", "dequantize_row_iq3_s"};
  using Dequantize = void (*)(const void *, float *, int64_t);
  auto decode = reinterpret_cast<Dequantize>(dlsym(ggml, symbols[f]));
  if (!decode) { error = dlerror(); return false; }
  values.assign(native.size() / kQuantFormats[f].block_bytes * kQuantFormats[f].block_elements, 0.f);
  decode(native.data(), values.data(), (int64_t)values.size());
  return true;
}

} // namespace gguf_reference

#pragma once

// CPU reference for the GGUF image formats (metal/abi/QuantFormat.h): native
// blocks, their fp32 values with llama.cpp's dequantize_row_* semantics (MIT
// notice in THIRD_PARTY_NOTICES) and the planes the load-time repack writes,
// and the fp64 bound a GGUF projection's result lies within. Shared by the
// GGUF tests.

#include "metal/abi/QuantFormat.h"
#include "metal/abi/QuantTables.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace gguf_reference {

enum Fmt { Q4K = GGUF_FMT_Q4K, IQ4XS = GGUF_FMT_IQ4XS, IQ4NL = GGUF_FMT_IQ4NL, Q5K = GGUF_FMT_Q5K, Q6K = GGUF_FMT_Q6K,
           Q3K = GGUF_FMT_Q3K, Q80 = GGUF_FMT_Q80, IQ3S = GGUF_FMT_IQ3S, Q2K = GGUF_FMT_Q2K, IQ3XXS = GGUF_FMT_IQ3XXS,
           IQ2XXS = GGUF_FMT_IQ2XXS, IQ2XS = GGUF_FMT_IQ2XS, IQ2S = GGUF_FMT_IQ2S, IQ1S = GGUF_FMT_IQ1S,
           IQ1M = GGUF_FMT_IQ1M, Q40 = GGUF_FMT_Q40, Q41 = GGUF_FMT_Q41, MXFP4 = GGUF_FMT_MXFP4,
           FMT_COUNT = GGUF_FMT_COUNT };
inline const char *fmtName(uint32_t f) { return kQuantFormats[f].name; }
// The format of kQuantFormats name `name`, FMT_COUNT for none.
inline Fmt fmtNamed(const std::string &name) {
  int f = 0;
  while (f < FMT_COUNT && name != fmtName(f)) ++f;
  return Fmt(f);
}
inline uint16_t f2h(float f) { __fp16 h = (__fp16)f; uint16_t u; memcpy(&u, &h, 2); return u; }
inline float h2f(uint16_t u) { __fp16 h; memcpy(&h, &u, 2); return (float)h; }
inline uint32_t rowBytes(Fmt f, uint32_t K) {
  const QuantFormat &i = kQuantFormats[f];
  return K / i.block_elements * i.block_bytes;
}
[[noreturn]] inline void unknownFormat(Fmt f) {
  throw std::invalid_argument("no GGUF reference for format " + std::to_string(int(f)));
}

// Field offsets of the GGML blocks whose half scale d is not their first field:
// block_q6_K {ql[128], qh[64], scales[16], d}, block_q3_K {hmask[32], qs[64],
// scales[12], d} and block_q2_K {scales[16], qs[64], d, dmin}; block_iq1_m
// {qs[32], qh[16], scales[8]} spreads its d over the top nibbles of the four
// halfwords of its scales.
constexpr uint32_t kQ6KScales = 128 + 64, kQ6KD = kQ6KScales + 16;
constexpr uint32_t kQ3KScales = 32 + 64, kQ3KD = kQ3KScales + 12;
constexpr uint32_t kQ2KD = 16 + 64, kIQ1MScales = 32 + 16;

// llama.cpp's GGML_E8M0_TO_FP32_HALF: 2^(e - 128), subnormal below e = 2.
inline float e8m0Half(uint8_t e) {
  const uint32_t bits = e < 2 ? 0x00200000u << e : uint32_t(e - 1) << 23;
  float value; memcpy(&value, &bits, 4); return value;
}

// N native rows of random bytes whose half scales (d, and dmin or m for Q4_K, Q5_K, Q2_K and Q4_1) are scale(), and
// MXFP4's exponents the power of two at or below it.
template <class Scale>
std::vector<uint8_t> makeNative(Fmt f, uint32_t N, uint32_t K, std::mt19937 &rng, Scale scale) {
  const QuantFormat &fi = kQuantFormats[f];
  std::vector<uint8_t> v((size_t)N * rowBytes(f, K));
  for (auto &b : v) b = (uint8_t)rng();
  const uint32_t off = f == Q6K ? kQ6KD : f == Q3K ? kQ3KD : f == Q2K ? kQ2KD : 0;
  for (size_t b = 0; b < v.size() / fi.block_bytes; ++b) {
    uint8_t *blk = v.data() + b * fi.block_bytes;
    const uint16_t d = scale();
    if (f == IQ1M) {
      uint16_t sc[4]; memcpy(sc, blk + kIQ1MScales, 8);
      for (int k = 0; k < 4; ++k) sc[k] = uint16_t((sc[k] & 0x0FFF) | ((d >> (4 * k)) & 0xF) << 12);
      memcpy(blk + kIQ1MScales, sc, 8);
    } else if (f == MXFP4) {
      blk[0] = uint8_t(std::ilogb(h2f(d)) + 128);
    } else {
      memcpy(blk + off, &d, 2);
    }
    if (f == Q4K || f == Q5K || f == Q2K || f == Q41) { const uint16_t m = scale(); memcpy(blk + off + 2, &m, 2); }
  }
  return v;
}
// The range of a format's half scales that gives its weights the magnitudes a
// model's have (a few hundredths), so the GEMM checks see realistic sums.
inline std::uniform_real_distribution<float> scaleRange(Fmt f) {
  switch (f) {
    case Q4K: case Q5K: case Q3K: case IQ4NL: case Q80: case Q2K: case Q40: case Q41: case MXFP4:
      return std::uniform_real_distribution<float>(0.0005f, 0.004f);
    case IQ4XS: return std::uniform_real_distribution<float>(0.00002f, 0.00015f);
    case Q6K: return std::uniform_real_distribution<float>(0.00002f, 0.0001f);
    case IQ3S: case IQ3XXS: return std::uniform_real_distribution<float>(0.0001f, 0.0005f);
    case IQ2XXS: case IQ2XS: case IQ2S: return std::uniform_real_distribution<float>(0.0002f, 0.001f);
    case IQ1S: case IQ1M: return std::uniform_real_distribution<float>(0.001f, 0.008f);
    case FMT_COUNT: break;
  }
  unknownFormat(f);
}
// N native rows with scales in the format's realistic range.
inline std::vector<uint8_t> makeNative(Fmt f, uint32_t N, uint32_t K, std::mt19937 &rng) {
  std::uniform_real_distribution<float> range = scaleRange(f);
  return makeNative(f, N, K, rng, [&] { return f2h(range(rng)); });
}

inline void scale_min_k4(const uint8_t *sc, int j, uint8_t &s, uint8_t &m) {
  if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; }
  else { s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
}
// Planes in chunk order (metal/abi/QuantFormat.h): 4-bit linear codes pair-interleaved in chunk words, every
// other field a little-endian bit string over the slots.
inline void packBits(const uint8_t *slots, uint32_t bits, uint8_t *dst) {
  for (uint32_t w = 0, per = 32 / bits; w < bits; ++w) {
    uint32_t v = 0;
    for (uint32_t i = 0; i < per; ++i) v |= uint32_t(slots[w * per + i]) << (bits * i);
    memcpy(dst + 4 * w, &v, 4);
  }
}
inline void packPairs(const uint8_t *slots, uint8_t *dst) {
  for (uint32_t c = 0; c < 4; ++c) {
    uint32_t v = 0;
    for (uint32_t i = 0; i < 8; ++i) v |= uint32_t(slots[8 * c + i]) << ((i & 1) * 16 + 4 * (i >> 1));
    memcpy(dst + 4 * c, &v, 4);
  }
}
// llama.cpp's ksigns_iq2xs: the eight signs of a 7-bit sign index, bit 7 their parity.
inline uint8_t signs7(uint32_t index) { return uint8_t(index | (__builtin_popcount(index) & 1) << 7); }
// The grid value (-1, 0 or 1) of element k of a kIQ1SGrid entry.
inline int iq1Value(uint32_t entry, int k) { return int((entry >> (8 * (k % 4) + 4 * (k / 4))) & 0xF) - 1; }
// reference values (llama.cpp dequantize_row_* semantics) + plane bytes for group g of one row
inline void groupPack(Fmt f, const uint8_t *row, uint32_t g, float vals[32], uint8_t p0[32], uint8_t p1[8]) {
  const QuantFormat &fi = kQuantFormats[f];
  const uint8_t *blk = row + (g * 32 / fi.block_elements) * fi.block_bytes;
  const uint32_t j = (g * 32 % fi.block_elements) / 32;
  uint8_t lo[32], hi[32];   // per slot: the (low) code and its high bits
  switch (f) {
    case Q4K: {
      const uint8_t *q = blk + 16 + (j / 2) * 32;
      const int sh = (j % 2) * 4;
      uint16_t d16, m16; memcpy(&d16, blk, 2); memcpy(&m16, blk + 2, 2);
      uint8_t sc, mn; scale_min_k4(blk + 4, j, sc, mn);
      for (int k = 0; k < 32; ++k) {
        const uint8_t code = (q[k] >> sh) & 15;
        lo[quant_slot(k)] = code;
        vals[k] = h2f(d16) * sc * code - h2f(m16) * mn;
      }
      packPairs(lo, p0); return;
    }
    case IQ4XS: {
      const uint8_t *qs = blk + 8 + j * 16, *sl = blk + 4;
      uint16_t d16, shh; memcpy(&d16, blk, 2); memcpy(&shh, blk + 2, 2);
      const int ls = ((sl[j / 2] >> 4 * (j % 2)) & 0xf) | (((shh >> 2 * j) & 3) << 4);
      const float s = h2f(d16) * (ls - 32);
      for (int k = 0; k < 32; ++k) {
        const uint8_t code = k < 16 ? (qs[k] & 15) : (qs[k - 16] >> 4);
        lo[quant_slot(k)] = code;
        vals[k] = s * kIQ4NLValues[code];
      }
      packBits(lo, 4, p0); return;
    }
    case IQ4NL: {
      const uint8_t *qs = blk + 2;
      uint16_t d16; memcpy(&d16, blk, 2);
      const float s = h2f(d16);
      for (int k = 0; k < 32; ++k) {
        const uint8_t code = k < 16 ? (qs[k] & 15) : (qs[k - 16] >> 4);
        lo[quant_slot(k)] = code;
        vals[k] = s * kIQ4NLValues[code];
      }
      packBits(lo, 4, p0); return;
    }
    case Q5K: {
      const uint8_t *qh = blk + 16, *ql = blk + 48 + (j / 2) * 32;
      const int sh = (j % 2) * 4;
      uint16_t d16, m16; memcpy(&d16, blk, 2); memcpy(&m16, blk + 2, 2);
      uint8_t sc, mn; scale_min_k4(blk + 4, j, sc, mn);
      for (int k = 0; k < 32; ++k) {
        const uint8_t l4 = (ql[k] >> sh) & 15, h1 = (qh[k] >> j) & 1;
        lo[quant_slot(k)] = l4;
        hi[quant_slot(k)] = h1;
        vals[k] = h2f(d16) * sc * (l4 + 16 * h1) - h2f(m16) * mn;
      }
      packPairs(lo, p0); packBits(hi, 1, p1); return;
    }
    case Q6K: {
      const uint32_t n = j / 4, r = j % 4;
      const uint8_t *ql = blk + 64 * n, *qh = blk + 128 + 32 * n;
      const int8_t *sc = (const int8_t *)(blk + kQ6KScales) + 8 * n;
      uint16_t d16; memcpy(&d16, blk + kQ6KD, 2);
      const float d = h2f(d16);
      for (int k = 0; k < 32; ++k) {
        const uint8_t l4 = (ql[k + 32 * (r & 1)] >> (4 * (r >> 1))) & 15, h2 = (qh[k] >> (2 * r)) & 3;
        const int q6 = (l4 | (h2 << 4)) - 32;
        lo[quant_slot(k)] = l4;
        hi[quant_slot(k)] = h2;
        vals[k] = d * sc[2 * r + k / 16] * q6;
      }
      packPairs(lo, p0); packBits(hi, 2, p1); return;
    }
    case Q3K: {
      const uint32_t n = j / 4, jj = j % 4;
      const uint8_t *hm = blk, *q = blk + 32 + 32 * n;
      uint32_t aux[4]; memcpy(aux, blk + kQ3KScales, 12);
      const uint32_t tmp = aux[2], kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
      aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
      aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
      aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
      aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
      const int8_t *scales = (const int8_t *)aux;
      uint16_t d16; memcpy(&d16, blk + kQ3KD, 2);
      const float d = h2f(d16);
      for (int k = 0; k < 32; ++k) {
        const int c2 = (q[k] >> (2 * jj)) & 3, hb1 = (hm[k] >> j) & 1;
        lo[quant_slot(k)] = uint8_t(c2);
        hi[quant_slot(k)] = uint8_t(hb1);
        vals[k] = d * (scales[2 * j + k / 16] - 32) * (c2 - (hb1 ? 0 : 4));
      }
      packBits(lo, 2, p0); packBits(hi, 1, p1); return;
    }
    case Q80: {
      uint16_t d16; memcpy(&d16, blk, 2);
      const int8_t *qs = (const int8_t *)(blk + 2);
      for (int k = 0; k < 32; ++k) { vals[k] = h2f(d16) * qs[k]; lo[quant_slot(k)] = uint8_t(qs[k]); }
      packBits(lo, 8, p0); return;
    }
    case IQ3S: {
      const uint8_t *qs = blk + 2 + 8 * j, *qh = blk + 66, *signs = blk + 74 + 4 * j, *scales = blk + 106;
      uint16_t d16; memcpy(&d16, blk, 2);
      const uint32_t sc = (scales[j / 2] >> (4 * (j % 2))) & 0xf;
      const float db = h2f(d16) * (1 + 2 * sc);
      for (int l = 0; l < 4; ++l) {
        const uint8_t *g1 = (const uint8_t *)(kIQ3SGrid + (qs[2 * l] | ((qh[j] << (8 - 2 * l)) & 256)));
        const uint8_t *g2 = (const uint8_t *)(kIQ3SGrid + (qs[2 * l + 1] | ((qh[j] << (7 - 2 * l)) & 256)));
        for (int k = 0; k < 4; ++k) {
          vals[8 * l + k] = db * g1[k] * ((signs[l] & (1 << k)) ? -1.f : 1.f);
          vals[8 * l + 4 + k] = db * g2[k] * ((signs[l] & (1 << (4 + k))) ? -1.f : 1.f);
        }
      }
      // word c: grid indices of elements 4c.. and 16+4c.. (qs[c], qs[4 + c]), chunk c's signs, their ninth bits,
      // the scale
      for (int k = 0; k < 32; ++k) hi[quant_slot(k)] = (signs[k / 8] >> (k % 8)) & 1;
      uint8_t sign[4]; packBits(hi, 1, sign);
      for (int c = 0; c < 4; ++c) {
        const uint32_t v = qs[c] | uint32_t(qs[4 + c]) << 8 | uint32_t(sign[c]) << 16 | ((qh[j] >> c) & 1u) << 24 |
                           ((qh[j] >> (4 + c)) & 1u) << 25 | sc << 26;
        memcpy(p0 + 4 * c, &v, 4);
      }
      return;
    }
    case Q2K: {
      const uint8_t *q = blk + 16 + 32 * (j / 4), *sc = blk + 2 * j;
      uint16_t d16, m16; memcpy(&d16, blk + kQ2KD, 2); memcpy(&m16, blk + kQ2KD + 2, 2);
      for (int k = 0; k < 32; ++k) {
        const uint8_t code = (q[k] >> (2 * (j % 4))) & 3, s = sc[k / 16];
        lo[quant_slot(k)] = code;
        vals[k] = h2f(d16) * (s & 0xF) * code - h2f(m16) * (s >> 4);
      }
      packBits(lo, 2, p0); return;
    }
    case Q40: case Q41: case MXFP4: {
      const uint8_t *qs = blk + fi.meta_bytes;
      uint16_t d16, m16 = 0; memcpy(&d16, blk, 2); if (f == Q41) memcpy(&m16, blk + 2, 2);
      for (int k = 0; k < 32; ++k) {
        const uint8_t code = k < 16 ? (qs[k] & 15) : (qs[k - 16] >> 4);
        lo[quant_slot(k)] = code;
        vals[k] = f == Q40 ? (code - 8) * h2f(d16) : f == Q41 ? code * h2f(d16) + h2f(m16) : kFP4Values[code] * e8m0Half(blk[0]);
      }
      if (f == MXFP4) packBits(lo, 4, p0); else packPairs(lo, p0);
      return;
    }
    // The IQ3_XXS, IQ2 and IQ1 planes are the group's native bytes (metal/abi/QuantFormat.h).
    case IQ3XXS: {
      const uint8_t *qs = blk + 2 + 8 * j;
      uint16_t d16; memcpy(&d16, blk, 2);
      uint32_t aux; memcpy(&aux, blk + 66 + 4 * j, 4);
      const float db = h2f(d16) * (0.5f + (aux >> 28)) * 0.5f;
      for (int l = 0; l < 4; ++l) {
        const uint8_t signs = signs7((aux >> 7 * l) & 127);
        const uint8_t *g1 = (const uint8_t *)(kIQ3XXSGrid + qs[2 * l]), *g2 = (const uint8_t *)(kIQ3XXSGrid + qs[2 * l + 1]);
        for (int k = 0; k < 4; ++k) {
          vals[8 * l + k] = db * g1[k] * (signs & (1 << k) ? -1.f : 1.f);
          vals[8 * l + 4 + k] = db * g2[k] * (signs & (1 << (4 + k)) ? -1.f : 1.f);
        }
      }
      memcpy(p0, qs, 8); memcpy(p1, &aux, 4); return;
    }
    case IQ2XXS: {
      uint16_t d16; memcpy(&d16, blk, 2);
      uint32_t aux[2]; memcpy(aux, blk + 2 + 8 * j, 8);
      const float db = h2f(d16) * (0.5f + (aux[1] >> 28)) * 0.25f;
      for (int l = 0; l < 4; ++l) {
        const uint8_t *grid = (const uint8_t *)(kIQ2XXSGrid + ((aux[0] >> 8 * l) & 0xFF));
        const uint8_t signs = signs7((aux[1] >> 7 * l) & 127);
        for (int k = 0; k < 8; ++k) vals[8 * l + k] = db * grid[k] * (signs & (1 << k) ? -1.f : 1.f);
      }
      memcpy(p0, aux, 8); return;
    }
    case IQ2XS: case IQ2S: {
      uint16_t d16; memcpy(&d16, blk, 2);
      const uint8_t sc = blk[(f == IQ2XS ? 66 : 74) + j];
      const float db[2] = {h2f(d16) * (0.5f + (sc & 0xf)) * 0.25f, h2f(d16) * (0.5f + (sc >> 4)) * 0.25f};
      uint16_t qs[4]; memcpy(qs, blk + 2 + 8 * j, 8);   // IQ2_XS
      const uint8_t *qs8 = blk + 2 + 4 * j, *signs = qs8 + 32, qh = blk[66 + j];   // IQ2_S
      for (int l = 0; l < 4; ++l) {
        const uint8_t *grid = f == IQ2XS ? (const uint8_t *)(kIQ2XSGrid + (qs[l] & 511))
                                         : (const uint8_t *)(kIQ2SGrid + (qs8[l] | (qh << (8 - 2 * l) & 0x300)));
        const uint8_t sign = f == IQ2XS ? signs7(qs[l] >> 9) : signs[l];
        for (int k = 0; k < 8; ++k) vals[8 * l + k] = db[l / 2] * grid[k] * (sign & (1 << k) ? -1.f : 1.f);
      }
      if (f == IQ2XS) { memcpy(p0, qs, 8); p1[0] = sc; }
      else { memcpy(p0, qs8, 4); memcpy(p0 + 4, signs, 4); p1[0] = qh; p1[1] = sc; }
      return;
    }
    case IQ1S: {
      const uint8_t *qs = blk + 2 + 4 * j;
      uint16_t d16, qh; memcpy(&d16, blk, 2); memcpy(&qh, blk + 34 + 2 * j, 2);
      const float dl = h2f(d16) * (2 * ((qh >> 12) & 7) + 1), delta = qh & 0x8000 ? -0.125f : 0.125f;
      for (int l = 0; l < 4; ++l) {
        const uint32_t entry = kIQ1SGrid[qs[l] | (((qh >> 3 * l) & 7) << 8)];
        for (int k = 0; k < 8; ++k) vals[8 * l + k] = dl * (iq1Value(entry, k) + delta);
      }
      memcpy(p0, qs, 4); memcpy(p1, &qh, 2); return;
    }
    case IQ1M: {
      const uint8_t *qs = blk + 4 * j, *qh = blk + 32 + 2 * j;
      uint16_t sc[4]; memcpy(sc, blk + kIQ1MScales, 8);
      const uint16_t d16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
      const float dl[2] = {h2f(d16) * (2 * ((sc[j / 2] >> (6 * (j % 2))) & 7) + 1),
                           h2f(d16) * (2 * ((sc[j / 2] >> (6 * (j % 2) + 3)) & 7) + 1)};
      for (int l = 0; l < 4; ++l) {
        const uint8_t h = qh[l / 2] >> (4 * (l % 2));
        const uint32_t entry = kIQ1SGrid[qs[l] | ((h & 7) << 8)];
        const float delta = h & 8 ? -0.125f : 0.125f;
        for (int k = 0; k < 8; ++k) vals[8 * l + k] = dl[l / 2] * (iq1Value(entry, k) + delta);
      }
      memcpy(p0, qs, 4); memcpy(p1, qh, 2); return;
    }
    case FMT_COUNT: break;
  }
  unknownFormat(f);
}
inline void metaPack(Fmt f, const uint8_t *row, uint32_t unit, uint8_t *dst) {
  const QuantFormat &fi = kQuantFormats[f];
  const uint8_t *blk = row + unit * fi.block_bytes;
  switch (f) {
    // The block's leading fields.
    case Q4K: case Q5K: case IQ4XS: case IQ4NL: case Q80: case IQ3S: case IQ3XXS: case IQ2XXS: case IQ2XS: case IQ2S:
    case IQ1S: case Q40: case Q41: case MXFP4:
      memcpy(dst, blk, fi.meta_bytes); return;
    case Q6K: memcpy(dst, blk + kQ6KScales, 16); memcpy(dst + 16, blk + kQ6KD, 2); dst[18] = dst[19] = 0; return;
    case Q3K: memcpy(dst, blk + kQ3KD, 2); dst[2] = dst[3] = 0; memcpy(dst + 4, blk + kQ3KScales, 12); return;
    case Q2K: memcpy(dst, blk + kQ2KD, 4); memcpy(dst + 4, blk, 16); return;
    case IQ1M: memcpy(dst, blk + kIQ1MScales, 8); return;
    case FMT_COUNT: break;
  }
  unknownFormat(f);
}
// GGML's fp32 values of one native row of K weights.
inline void rowValues(Fmt f, const uint8_t *row, uint32_t K, float *values) {
  uint8_t p0[32], p1[8];
  for (uint32_t g = 0; g < K / 32; ++g) groupPack(f, row, g, values + g * 32, p0, p1);
}
struct Packed { std::vector<uint8_t> w0, w1, meta; };
inline Packed repack(Fmt f, const std::vector<uint8_t> &native, uint32_t N, uint32_t K, std::vector<float> *Wf,
                     std::vector<float> *Ws = nullptr) {
  const QuantFormat &fi = kQuantFormats[f];
  const uint32_t G = K / 32, rb = rowBytes(f, K), units = G / fi.meta_groups;
  Packed p;
  p.w0.assign((size_t)N * G * fi.plane0_bytes, 0);
  p.w1.assign(fi.plane1_bytes ? (size_t)N * G * fi.plane1_bytes : 16, 0);
  p.meta.assign((size_t)N * units * fi.meta_bytes, 0);
  if (Wf) Wf->assign((size_t)N * K, 0.f);
  if (Ws) Ws->assign((size_t)N * K, 0.f);
  for (uint32_t n = 0; n < N; ++n) {
    const uint8_t *row = native.data() + (size_t)n * rb;
    for (uint32_t g = 0; g < G; ++g) {
      float vals[32]; uint8_t p0[32], p1[8];
      groupPack(f, row, g, vals, p0, p1);
      memcpy(p.w0.data() + quant_tile_index(n, g, G) * fi.plane0_bytes, p0, fi.plane0_bytes);
      if (fi.plane1_bytes) memcpy(p.w1.data() + quant_tile_index(n, g, G) * fi.plane1_bytes, p1, fi.plane1_bytes);
      if (Wf) for (int k = 0; k < 32; ++k) (*Wf)[(size_t)n * K + g * 32 + k] = vals[k];
      if (Ws) for (int k = 0; k < 32; ++k) (*Ws)[(size_t)n * K + g * 32 + k] = h2f(f2h(vals[k]));
    }
    for (uint32_t u = 0; u < units; ++u)
      metaPack(f, row, u, p.meta.data() + quant_tile_index(n, u, units) * fi.meta_bytes);
  }
  return p;
}
// Upstream GGML's fp32 dequantization of native rows from an unmodified libggml-base (for example
// llama.cpp 7ab4ee7) loaded with dlopen; false and a message when it lacks the format's symbol.
inline bool ggmlDequantize(void *ggml, Fmt f, const std::vector<uint8_t> &native, std::vector<float> &values,
                           std::string &error) {
  static const char *const symbols[] = {
    "dequantize_row_q4_K", "dequantize_row_iq4_xs", "dequantize_row_iq4_nl", "dequantize_row_q5_K",
    "dequantize_row_q6_K", "dequantize_row_q3_K", "dequantize_row_q8_0", "dequantize_row_iq3_s",
    "dequantize_row_q2_K", "dequantize_row_iq3_xxs", "dequantize_row_iq2_xxs", "dequantize_row_iq2_xs",
    "dequantize_row_iq2_s", "dequantize_row_iq1_s", "dequantize_row_iq1_m", "dequantize_row_q4_0",
    "dequantize_row_q4_1", "dequantize_row_mxfp4"};
  static_assert(std::size(symbols) == FMT_COUNT, "a GGML dequantize_row_* symbol per format");
  using Dequantize = void (*)(const void *, float *, int64_t);
  auto decode = reinterpret_cast<Dequantize>(dlsym(ggml, symbols[f]));
  if (!decode) { error = dlerror(); return false; }
  values.assign(native.size() / kQuantFormats[f].block_bytes * kQuantFormats[f].block_elements, 0.f);
  decode(native.data(), values.data(), (int64_t)values.size());
  return true;
}

// The fp64 dot product of a bf16 row with a weight row and the magnitudes its
// error bounds scale with.
struct Dot {
  double value = 0, magnitude = 0, inputs = 0, largest = 0;
};
inline Dot dot(const float *x, const float *w, uint32_t K) {
  Dot d;
  for (uint32_t k = 0; k < K; ++k) {
    d.value += double(x[k]) * w[k];
    d.magnitude += std::fabs(double(x[k]) * w[k]);
    d.inputs += std::fabs(x[k]);
    d.largest = std::max(d.largest, double(std::fabs(w[k])));
  }
  return d;
}

// The distance from the fp64 dot product within which a GGUF projection's
// fp32 result lies, before its epilogue and bf16 rounding. Both tiles
// multiply bf16 inputs by exactly represented weight operands and accumulate
// in fp32: 2^-16 sum|x| max|w| = 256 u sum|x| max|w| (u = 2^-24) bounds a
// chain of 256 fp32 additions over sum|x w| <= sum|x| max|w|; the longer
// chains of a large K round independently, so their error grows with the
// square root of their length. The register tile (Apple9) is exact up to that
// accumulation, but for Q2_K's mins, which it adds as m / -160 times the seed
// (-160 times the sum of 16 inputs): two more roundings of |m| sum|x| <=
// max|w| sum|x|. The staged tile rounds every weight once to half: 2^-11 of
// each product, and 2^-25 of each |x| for weights below half's normal range.
inline double projectionBound(const Dot &d, bool staged) {
  const double accumulation = std::ldexp(d.inputs * d.largest, -16);
  return staged ? accumulation + std::ldexp(d.magnitude, -11) + std::ldexp(d.inputs, -25) : accumulation;
}

} // namespace gguf_reference

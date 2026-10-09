#pragma once

#include "metal/abi/QuantTables.h"
#include <metal_stdlib>

// The native GGUF blocks of the token tables (ops/Embedding.cpp), one struct
// per format: a row is hidden / Weights blocks of Bytes bytes, each laid out as
// the format's ggml block_* (an MLX affine table: the loader's block of
// metal/abi/QuantFormat.h) at the byte offsets its struct names, and value is
// element dim as bf16. Every format's value keeps the source order of its
// float operations (reassociate(off)), as the GGUF GEMMs do.

// The little-endian half at byte `at` of a block.
inline half gguf_half(device const uchar *block, uint at) {
  return as_type<half>(ushort(block[at] | (block[at + 1] << 8)));
}

// block_q4_K: half d | half dmin | uchar scales[12] | uchar qs[128], eight
// 32-weight groups with 6-bit scales and mins; block_q5_K (Fifth) holds
// uchar qh[32] before qs, a fifth bit per weight.
template <bool Fifth> struct GgufEmbedK {
  enum : uint { Weights = 256, Bytes = Fifth ? 176 : 144, D = 0, DMin = 2, Scales = 4, High = 16, Codes = Fifth ? 48 : 16 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint j = (dim % Weights) / 32, l = dim % 32;
    const half d = gguf_half(block, D), dmin = gguf_half(block, DMin);
    device const uchar *sc = block + Scales; uchar m, s;
    if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; } else { s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
    uchar q = (block[Codes + (j / 2) * 32 + l] >> ((j % 2) * 4)) & 15;
    if (Fifth) q |= ((block[High + l] >> j) & 1) << 4;
    return bfloat(float(d) * float(s) * float(q) - float(dmin) * float(m));
  }
};
using GgufEmbedQ4K = GgufEmbedK<false>;
using GgufEmbedQ5K = GgufEmbedK<true>;
// block_q6_K: uchar ql[128] | uchar qh[64] | int8 scales[16] | half d; codes
// are 6-bit with zero point 32.
struct GgufEmbedQ6K {
  enum : uint { Weights = 256, Bytes = 210, Low = 0, High = 128, Scales = 192, D = 208, Zero = 32 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint l = dim % Weights, n = l / 128, r = l % 128, quarter = r / 32, pos = r % 32;
    const uchar lo = (block[Low + n * 64 + (quarter & 1) * 32 + pos] >> ((quarter >> 1) * 4)) & 15;
    const uchar hi = (block[High + n * 32 + pos] >> (2 * quarter)) & 3;
    const char sc = as_type<char>(block[Scales + n * 8 + 2 * quarter + pos / 16]);
    const half d = gguf_half(block, D);
    return bfloat(float(d) * float(sc) * float(int(lo | (hi << 4)) - int(Zero)));
  }
};
// block_q8_0: half d | int8 qs[32].
struct GgufEmbedQ80 {
  enum : uint { Weights = 32, Bytes = 34, D = 0, Codes = 2 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const half d = gguf_half(block, D);
    return bfloat(float(d) * float(as_type<char>(block[Codes + dim % Weights])));
  }
};
// The 2-bit code of weight l of a block_q3_K or block_q2_K's qs[64]: bits 2j of qs[32n + pos] for l = 128n + 32j + pos.
inline uchar gguf_k2_code(device const uchar *qs, uint l) { return (qs[32 * (l / 128) + l % 32] >> (2 * (l % 128 / 32))) & 3; }
// block_q3_K: uchar hmask[32] | uchar qs[64] | uchar scales[12] | half d; 16-element groups with a 6-bit scale
// (offset 32), codes with zero point 4 when their hmask bit is clear.
struct GgufEmbedQ3K {
  enum : uint { Weights = 256, Bytes = 110, High = 0, Codes = 32, Scales = 96, D = 108 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint l = dim % Weights, n = l / 128, j = (l % 128) / 32, pos = l % 32, is = l / 16;
    const uchar q = gguf_k2_code(block + Codes, l), h = (block[High + pos] >> (4 * n + j)) & 1;
    const uchar sc = (is < 8 ? block[Scales + is] & 0xF : block[Scales + is - 8] >> 4) |
                     ((block[Scales + 8 + is % 4] >> (2 * (is / 4))) & 3) << 4;
    const half d = gguf_half(block, D);
    return bfloat(float(d) * float(int(sc) - 32) * float(int(q) - (h ? 0 : 4)));
  }
};
// block_q2_K: uchar scales[16] | uchar qs[64] | half d | half dmin; 16-element groups whose scale byte holds a
// 4-bit scale and min.
struct GgufEmbedQ2K {
  enum : uint { Weights = 256, Bytes = 84, Scales = 0, Codes = 16, D = 80, DMin = 82 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint l = dim % Weights;
    const uchar q = gguf_k2_code(block + Codes, l), sc = block[Scales + l / 16];
    const half d = gguf_half(block, D), dmin = gguf_half(block, DMin);
    return bfloat(float(d) * float(sc & 0xF) * float(q) - float(dmin) * float(sc >> 4));
  }
};
// block_q4_0: half d | uchar qs[16], element j in the low nibble of qs[j] and j + 16 in the high one, with zero
// point 8; block_q4_1 (Min) holds half m after d, the value's offset.
template <bool Min> struct GgufEmbedQ4 {
  enum : uint { Weights = 32, Bytes = Min ? 20 : 18, D = 0, M = 2, Codes = Min ? 4 : 2 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint l = dim % Weights;
    const uchar q = (block[Codes + l % 16] >> (4 * (l / 16))) & 15;
    if (Min) return bfloat(float(q) * float(gguf_half(block, D)) + float(gguf_half(block, M)));
    return bfloat(float(int(q) - 8) * float(gguf_half(block, D)));
  }
};
using GgufEmbedQ40 = GgufEmbedQ4<false>;
using GgufEmbedQ41 = GgufEmbedQ4<true>;

// block_pq2_0: half d | uchar qs[32], element l in bits 2 (l % 4) of qs[l / 4]; zero point 1. decode is the
// element in fp32, which the rotated gather (shared/gguf_rotation.metal) transforms before it rounds.
struct GgufEmbedPQ20 {
  enum : uint { Weights = 128, Bytes = 34, D = 0, Codes = 2 };
  __attribute__((always_inline)) static float decode(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint l = dim % Weights;
    const uchar q = (block[Codes + l / 4] >> (2 * (l % 4))) & 3;
    return float(int(q) - 1) * float(gguf_half(block, D));
  }
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
    return bfloat(decode(block, dim));
  }
};
// block_iq4_nl: block_q4_0's layout, whose codes index the IQ4_NL values.
struct GgufEmbedIQ4NL {
  enum : uint { Weights = 32, Bytes = 18, D = 0, Codes = 2 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint l = dim % Weights;
    const uchar q = (block[Codes + l % 16] >> (4 * (l / 16))) & 15;
    return bfloat(float(gguf_half(block, D)) * float(kIQ4NLValues[q]));
  }
};
// block_iq4_xs: half d | ushort scales_h | uchar scales_l[4] | uchar qs[128], eight 32-weight groups of IQ4_NL codes
// with a 6-bit scale (offset 32): group j's element l in nibble l / 16 of qs[16 j + l % 16].
struct GgufEmbedIQ4XS {
  enum : uint { Weights = 256, Bytes = 136, D = 0, High = 2, Low = 4, Codes = 8 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint j = (dim % Weights) / 32, l = dim % 32;
    const uint high = block[High] | (block[High + 1] << 8);
    const int ls = int((block[Low + j / 2] >> (4 * (j % 2))) & 15) | int(((high >> (2 * j)) & 3) << 4);
    const uchar q = (block[Codes + 16 * j + l % 16] >> (4 * (l / 16))) & 15;
    return bfloat(float(gguf_half(block, D)) * float(ls - 32) * float(kIQ4NLValues[q]));
  }
};
// block_iq3_s: half d | uchar qs[64] | uchar qh[8] | uchar signs[32] | uchar scales[4], eight 32-weight groups with
// a 4-bit scale s worth 1 + 2 s: group j's element l is IQ3_S grid entry qs[8 j + l / 4] (ninth index bit l / 4 of
// qh[j]) at byte l % 4, negated by bit l % 8 of signs[4 j + l / 8].
struct GgufEmbedIQ3S {
  enum : uint { Weights = 256, Bytes = 110, D = 0, Codes = 2, High = 66, Signs = 74, Scales = 106 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint j = (dim % Weights) / 32, l = dim % 32;
    const uint index = block[Codes + 8 * j + l / 4] | (((block[High + j] >> (l / 4)) & 1) << 8);
    const uint magnitude = (kIQ3SGrid[index] >> (8 * (l % 4))) & 0xFF;
    const uint s = (block[Scales + j / 2] >> (4 * (j % 2))) & 15;
    const float sign = (block[Signs + 4 * j + l / 8] >> (l % 8)) & 1 ? -1.0f : 1.0f;
    return bfloat(float(gguf_half(block, D)) * float(1 + 2 * s) * float(magnitude) * sign);
  }
};
// block_mxfp4: uchar e | uchar qs[16], element l in nibble l / 16 of qs[l % 16], an E2M1 code; value =
// kFP4Values[code] (twice the E2M1 value) times 2^(e - 128), llama.cpp's GGML_E8M0_TO_FP32_HALF.
struct GgufEmbedMXFP4 {
  enum : uint { Weights = 32, Bytes = 17, E = 0, Codes = 1 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint l = dim % Weights, e = block[E];
    const uchar q = (block[Codes + l % 16] >> (4 * (l / 16))) & 15;
    return bfloat(float(kFP4Values[q]) * as_type<float>(e < 2 ? 0x00200000u << e : (e - 1) << 23));
  }
};
// MLX affine {bf16 s, bf16 z, codes}: element l of the group is code l of the codes' little-endian bit string of
// Bits-bit codes; value = s * code + z, rounded once to bf16.
template <uint Bits, uint Group> struct GgufEmbedAffine {
  enum : uint { Weights = Group, Bytes = 4 + Group * Bits / 8, S = 0, Z = 2, Codes = 4 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint at = (dim % Weights) * Bits, shift = at % 8;
    device const uchar *codes = block + Codes + at / 8;
    const uint word = uint(codes[0]) | (shift + Bits > 8 ? uint(codes[1]) << 8 : 0u);
    const float code = float((word >> shift) & ((1u << Bits) - 1));
    const float s = float(as_type<bfloat>(ushort(block[S] | (block[S + 1] << 8))));
    const float z = float(as_type<bfloat>(ushort(block[Z] | (block[Z + 1] << 8))));
    return bfloat(code * s + z);
  }
};
#define GGUF_EMBED_AFFINE(B)                                                                                       \
  typedef GgufEmbedAffine<B, 32> GgufEmbedAF##B##G32; typedef GgufEmbedAffine<B, 64> GgufEmbedAF##B##G64;        \
  typedef GgufEmbedAffine<B, 128> GgufEmbedAF##B##G128;
GGUF_EMBED_AFFINE(2) GGUF_EMBED_AFFINE(3) GGUF_EMBED_AFFINE(4) GGUF_EMBED_AFFINE(5) GGUF_EMBED_AFFINE(6)
GGUF_EMBED_AFFINE(8)
#undef GGUF_EMBED_AFFINE

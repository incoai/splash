#pragma once

#include "metal/abi/QuantTables.h"
#include "metal/kernels/common/quant_formats.h"
#include <metal_stdlib>

// The native blocks of every format (ops/Embedding.cpp), one struct per format:
// a row is hidden / Weights blocks of Bytes bytes, each laid out as the format's
// ggml block_* (an MLX affine, nvfp4 or fp8 tensor: the loader's block of
// metal/abi/QuantFormat.h) at the byte offsets its struct names, and decode is
// element dim's fp32 value, which the token gathers round once to bf16 and
// load-time widening writes as F32 (kernels/shared/embedding.metal). Every
// format's decode keeps the source order of its float operations
// (reassociate(off)), as the GGUF GEMMs do.

// The little-endian ushort at byte `at` of a block, and the half it holds.
inline ushort gguf_u16(device const uchar *block, uint at) {
  return ushort(block[at] | (block[at + 1] << 8));
}
inline half gguf_half(device const uchar *block, uint at) {
  return as_type<half>(gguf_u16(block, at));
}
// The little-endian uint at byte `at` of a block.
inline uint gguf_u32(device const uchar *block, uint at) {
  return uint(gguf_u16(block, at)) | uint(gguf_u16(block, at + 2)) << 16;
}

// block_q4_K: half d | half dmin | uchar scales[12] | uchar qs[128], eight
// 32-weight groups with 6-bit scales and mins; block_q5_K (Fifth) holds
// uchar qh[32] before qs, a fifth bit per weight. value = (d sc) q - dmin mn
// in one fused multiply-add, rounded as llama.cpp's compiled d1 * q - m1 is;
// Q2_K's and Q4_1's mins round the same way.
template <bool Fifth> struct GgufEmbedK {
  enum : uint { Weights = 256, Bytes = Fifth ? 176 : 144, D = 0, DMin = 2, Scales = 4, High = 16, Codes = Fifth ? 48 : 16 };
  __attribute__((always_inline)) static float decode(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint j = (dim % Weights) / 32, l = dim % 32;
    const half d = gguf_half(block, D), dmin = gguf_half(block, DMin);
    device const uchar *sc = block + Scales; uchar m, s;
    if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; } else { s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
    uchar q = (block[Codes + (j / 2) * 32 + l] >> ((j % 2) * 4)) & 15;
    if (Fifth) q |= ((block[High + l] >> j) & 1) << 4;
    return fma(float(d) * float(s), float(q), -(float(dmin) * float(m)));
  }
};
using GgufEmbedQ4K = GgufEmbedK<false>;
using GgufEmbedQ5K = GgufEmbedK<true>;
// block_q6_K: uchar ql[128] | uchar qh[64] | int8 scales[16] | half d; codes
// are 6-bit with zero point 32.
struct GgufEmbedQ6K {
  enum : uint { Weights = 256, Bytes = 210, Low = 0, High = 128, Scales = 192, D = 208, Zero = 32 };
  __attribute__((always_inline)) static float decode(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint l = dim % Weights, n = l / 128, r = l % 128, quarter = r / 32, pos = r % 32;
    const uchar lo = (block[Low + n * 64 + (quarter & 1) * 32 + pos] >> ((quarter >> 1) * 4)) & 15;
    const uchar hi = (block[High + n * 32 + pos] >> (2 * quarter)) & 3;
    const char sc = as_type<char>(block[Scales + n * 8 + 2 * quarter + pos / 16]);
    const half d = gguf_half(block, D);
    return float(d) * float(sc) * float(int(lo | (hi << 4)) - int(Zero));
  }
};
// block_q8_0: half d | int8 qs[32].
struct GgufEmbedQ80 {
  enum : uint { Weights = 32, Bytes = 34, D = 0, Codes = 2 };
  __attribute__((always_inline)) static float decode(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const half d = gguf_half(block, D);
    return float(d) * float(as_type<char>(block[Codes + dim % Weights]));
  }
};
// The 2-bit code of weight l of a block_q3_K or block_q2_K's qs[64]: bits 2j of qs[32n + pos] for l = 128n + 32j + pos.
inline uchar gguf_k2_code(device const uchar *qs, uint l) { return (qs[32 * (l / 128) + l % 32] >> (2 * (l % 128 / 32))) & 3; }
// block_q3_K: uchar hmask[32] | uchar qs[64] | uchar scales[12] | half d; 16-element groups with a 6-bit scale
// (offset 32), codes with zero point 4 when their hmask bit is clear.
struct GgufEmbedQ3K {
  enum : uint { Weights = 256, Bytes = 110, High = 0, Codes = 32, Scales = 96, D = 108 };
  __attribute__((always_inline)) static float decode(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint l = dim % Weights, n = l / 128, j = (l % 128) / 32, pos = l % 32, is = l / 16;
    const uchar q = gguf_k2_code(block + Codes, l), h = (block[High + pos] >> (4 * n + j)) & 1;
    const uchar sc = (is < 8 ? block[Scales + is] & 0xF : block[Scales + is - 8] >> 4) |
                     ((block[Scales + 8 + is % 4] >> (2 * (is / 4))) & 3) << 4;
    const half d = gguf_half(block, D);
    return float(d) * float(int(sc) - 32) * float(int(q) - (h ? 0 : 4));
  }
};
// block_q2_K: uchar scales[16] | uchar qs[64] | half d | half dmin; 16-element groups whose scale byte holds a
// 4-bit scale and min.
struct GgufEmbedQ2K {
  enum : uint { Weights = 256, Bytes = 84, Scales = 0, Codes = 16, D = 80, DMin = 82 };
  __attribute__((always_inline)) static float decode(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint l = dim % Weights;
    const uchar q = gguf_k2_code(block + Codes, l), sc = block[Scales + l / 16];
    const half d = gguf_half(block, D), dmin = gguf_half(block, DMin);
    return fma(float(d) * float(sc & 0xF), float(q), -(float(dmin) * float(sc >> 4)));
  }
};
// block_q4_0: half d | uchar qs[16], element j in the low nibble of qs[j] and j + 16 in the high one, with zero
// point 8; block_q4_1 (Min) holds half m after d, the value's offset. block_q5_0 and block_q5_1 (Fifth) hold
// uchar qh[4] before qs, element j's fifth bit at bit j, which doubles the zero point.
template <bool Min, bool Fifth> struct GgufEmbedQ4 {
  enum : uint { Weights = 32, D = 0, M = 2, High = Min ? 4 : 2, Codes = High + (Fifth ? 4 : 0), Bytes = Codes + 16,
                Zero = Fifth ? 16 : 8 };
  __attribute__((always_inline)) static float decode(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint l = dim % Weights;
    uint q = (block[Codes + l % 16] >> (4 * (l / 16))) & 15;
    if (Fifth) q |= ((block[High + l / 8] >> (l % 8)) & 1) << 4;
    if (Min) return fma(float(q), float(gguf_half(block, D)), float(gguf_half(block, M)));
    return float(int(q) - int(Zero)) * float(gguf_half(block, D));
  }
};
using GgufEmbedQ40 = GgufEmbedQ4<false, false>;
using GgufEmbedQ41 = GgufEmbedQ4<true, false>;
using GgufEmbedQ50 = GgufEmbedQ4<false, true>;
using GgufEmbedQ51 = GgufEmbedQ4<true, true>;

// block_pq2_0: half d | uchar qs[32], element l in bits 2 (l % 4) of qs[l / 4]; zero point 1. The rotated gather
// (shared/gguf_rotation.metal) transforms its decode before it rounds.
struct GgufEmbedPQ20 {
  enum : uint { Weights = 128, Bytes = 34, D = 0, Codes = 2 };
  __attribute__((always_inline)) static float decode(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint l = dim % Weights;
    const uchar q = (block[Codes + l / 4] >> (2 * (l % 4))) & 3;
    return float(int(q) - 1) * float(gguf_half(block, D));
  }
};
// block_iq4_nl: block_q4_0's layout, whose codes index the IQ4_NL values.
struct GgufEmbedIQ4NL {
  enum : uint { Weights = 32, Bytes = 18, D = 0, Codes = 2 };
  __attribute__((always_inline)) static float decode(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint l = dim % Weights;
    const uchar q = (block[Codes + l % 16] >> (4 * (l / 16))) & 15;
    return float(gguf_half(block, D)) * float(kIQ4NLValues[q]);
  }
};
// block_iq4_xs: half d | ushort scales_h | uchar scales_l[4] | uchar qs[128], eight 32-weight groups of IQ4_NL codes
// with a 6-bit scale (offset 32): group j's element l in nibble l / 16 of qs[16 j + l % 16].
struct GgufEmbedIQ4XS {
  enum : uint { Weights = 256, Bytes = 136, D = 0, High = 2, Low = 4, Codes = 8 };
  __attribute__((always_inline)) static float decode(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint j = (dim % Weights) / 32, l = dim % 32;
    const uint high = block[High] | (block[High + 1] << 8);
    const int ls = int((block[Low + j / 2] >> (4 * (j % 2))) & 15) | int(((high >> (2 * j)) & 3) << 4);
    const uchar q = (block[Codes + 16 * j + l % 16] >> (4 * (l / 16))) & 15;
    return float(gguf_half(block, D)) * float(ls - 32) * float(kIQ4NLValues[q]);
  }
};
// block_iq3_s: half d | uchar qs[64] | uchar qh[8] | uchar signs[32] | uchar scales[4], eight 32-weight groups with
// a 4-bit scale s worth 1 + 2 s: group j's element l is IQ3_S grid entry qs[8 j + l / 4] (ninth index bit l / 4 of
// qh[j]) at byte l % 4, negated by bit l % 8 of signs[4 j + l / 8].
struct GgufEmbedIQ3S {
  enum : uint { Weights = 256, Bytes = 110, D = 0, Codes = 2, High = 66, Signs = 74, Scales = 106 };
  __attribute__((always_inline)) static float decode(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint j = (dim % Weights) / 32, l = dim % 32;
    const uint index = block[Codes + 8 * j + l / 4] | (((block[High + j] >> (l / 4)) & 1) << 8);
    const uint magnitude = (kIQ3SGrid[index] >> (8 * (l % 4))) & 0xFF;
    const uint s = (block[Scales + j / 2] >> (4 * (j % 2))) & 15;
    const float sign = (block[Signs + 4 * j + l / 8] >> (l % 8)) & 1 ? -1.0f : 1.0f;
    return float(gguf_half(block, D)) * float(1 + 2 * s) * float(magnitude) * sign;
  }
};
// The IQ2 and IQ3_XXS grid formats: a block's leading half d, eight 32-weight groups, each element a grid magnitude
// times a scale of 4-bit s, d (0.5 + s) / 4 (IQ2) or / 2 (IQ3_XXS), negated by its sign bit: value = d (0.5 + s) unit
// grid sign, llama.cpp's operation order.
inline float gguf_grid_value(device const uchar *block, uint s, float unit, uint magnitude, bool negative) {
#pragma clang fp reassociate(off)
  return float(gguf_half(block, 0)) * (0.5f + float(s)) * unit * float(magnitude) * (negative ? -1.0f : 1.0f);
}
// Byte k of an eight-weight IQ2 grid entry.
inline uint gguf_iq2_magnitude(uint64_t entry, uint k) { return uint(entry >> (8 * k)) & 0xFF; }
// block_iq2_xxs: half d | uchar qs[64], eight 32-weight groups of eight bytes: bytes 0..3 the kIQ2XXSGrid indices of
// the group's four 8-weight entries, bytes 4..7 a word of their 7-bit sign indices (entry e's at bit 7 e, its eight
// signs quant_signs7 of it) and the scale s at bits 28..31. Element k of entry e is the entry's byte k, unit 1/4.
struct GgufEmbedIQ2XXS {
  enum : uint { Weights = 256, Bytes = 66, D = 0, Codes = 2 };
  __attribute__((always_inline)) static float decode(device const uchar *block, uint dim) {
    const uint j = (dim % Weights) / 32, e = (dim % 32) / 8, k = dim % 8;
    const uint aux = gguf_u32(block, Codes + 8 * j + 4);
    const uint signs = quant_signs7((aux >> (7 * e)) & 127);
    return gguf_grid_value(block, aux >> 28, 0.25f, gguf_iq2_magnitude(kIQ2XXSGrid[block[Codes + 8 * j + e]], k),
                           (signs >> k) & 1);
  }
};
// block_iq2_xs: half d | ushort qs[32] | uchar scales[8], eight 32-weight groups of four 8-weight entries: entry e of
// group j is qs[4 j + e], a 9-bit kIQ2XSGrid index and a 7-bit sign index above it. Entries 0 and 1 take the low
// nibble s of scales[j], 2 and 3 the high one; unit 1/4.
struct GgufEmbedIQ2XS {
  enum : uint { Weights = 256, Bytes = 74, D = 0, Codes = 2, Scales = 66 };
  __attribute__((always_inline)) static float decode(device const uchar *block, uint dim) {
    const uint j = (dim % Weights) / 32, e = (dim % 32) / 8, k = dim % 8;
    const uint entry = gguf_u16(block, Codes + 8 * j + 2 * e);
    const uint s = (block[Scales + j] >> (4 * (e / 2))) & 15;
    return gguf_grid_value(block, s, 0.25f, gguf_iq2_magnitude(kIQ2XSGrid[entry & 511], k),
                           (quant_signs7(entry >> 9) >> k) & 1);
  }
};
// block_iq2_s: half d | uchar qs[32] | uchar signs[32] | uchar qh[8] | uchar scales[8], eight 32-weight groups of
// four 8-weight entries: entry e of group j is kIQ2SGrid[qs[4 j + e] | (bits 2e of qh[j]) << 8], element k negated by
// bit k of signs[4 j + e]; scales as IQ2_XS's, unit 1/4.
struct GgufEmbedIQ2S {
  enum : uint { Weights = 256, Bytes = 82, D = 0, Codes = 2, Signs = 34, High = 66, Scales = 74 };
  __attribute__((always_inline)) static float decode(device const uchar *block, uint dim) {
    const uint j = (dim % Weights) / 32, e = (dim % 32) / 8, k = dim % 8;
    const uint index = block[Codes + 4 * j + e] | ((uint(block[High + j]) << (8 - 2 * e)) & 0x300);
    const uint s = (block[Scales + j] >> (4 * (e / 2))) & 15;
    return gguf_grid_value(block, s, 0.25f, gguf_iq2_magnitude(kIQ2SGrid[index], k),
                           (block[Signs + 4 * j + e] >> k) & 1);
  }
};
// block_iq3_xxs: half d | uchar qs[64] | uchar scales_and_signs[32], eight 32-weight groups: group j's element l is
// byte l % 4 of kIQ3XXSGrid[qs[8 j + l / 4]], its sign bit l % 8 of quant_signs7 of the 7-bit index at bit 7 (l / 8)
// of word j of scales_and_signs, whose bits 28..31 hold s; unit 1/2.
struct GgufEmbedIQ3XXS {
  enum : uint { Weights = 256, Bytes = 98, D = 0, Codes = 2, Signs = 66 };
  __attribute__((always_inline)) static float decode(device const uchar *block, uint dim) {
    const uint j = (dim % Weights) / 32, l = dim % 32;
    const uint aux = gguf_u32(block, Signs + 4 * j);
    const uint magnitude = (kIQ3XXSGrid[block[Codes + 8 * j + l / 4]] >> (8 * (l % 4))) & 0xFF;
    const uint signs = quant_signs7((aux >> (7 * (l / 8))) & 127);
    return gguf_grid_value(block, aux >> 28, 0.5f, magnitude, (signs >> (l % 8)) & 1);
  }
};
// The IQ1 formats: element k of an 8-weight kIQ1SGrid entry is nibble k / 4 of the entry's byte k % 4, less 1, plus
// the entry's delta, -1/8 when `negative` (else +1/8), times d (1 + 2 s) for the half d and 3-bit scale s of its
// weights: value = d (1 + 2 s) (grid + delta), llama.cpp's operation order.
inline float gguf_iq1_value(half d, uint s, uint index, uint k, bool negative) {
#pragma clang fp reassociate(off)
  const int grid = int((kIQ1SGrid[index] >> (8 * (k % 4) + 4 * (k / 4))) & 15) - 1;
  return float(d) * float(1 + 2 * s) * (float(grid) + (negative ? -0.125f : 0.125f));
}
// block_iq1_s: half d | uchar qs[32] | ushort qh[8], eight 32-weight groups of four 8-weight entries: entry e of
// group j is qs[4 j + e] with the three bits at 3 e of qh[j] above it; qh[j] holds the group's s at bits 12..14 and
// its negative delta at bit 15.
struct GgufEmbedIQ1S {
  enum : uint { Weights = 256, Bytes = 50, D = 0, Codes = 2, High = 34 };
  __attribute__((always_inline)) static float decode(device const uchar *block, uint dim) {
    const uint j = (dim % Weights) / 32, e = (dim % 32) / 8, k = dim % 8;
    const uint qh = gguf_u16(block, High + 2 * j);
    return gguf_iq1_value(gguf_half(block, D), (qh >> 12) & 7, block[Codes + 4 * j + e] | ((qh >> (3 * e)) & 7) << 8,
                          k, (qh >> 15) & 1);
  }
};
// block_iq1_m: uchar qs[32] | uchar qh[16] | ushort scales[4], eight 32-weight groups of four 8-weight entries:
// entry e of group j is qs[4 j + e] with the low three bits of nibble e % 2 of qh[2 j + e / 2] above it, whose bit 3
// is its negative delta. Entries e and e + 1 (e even) take s, the three bits at 6 (j % 2) + 3 (e / 2) of
// scales[j / 2], and the half d whose nibble w is the top nibble of scales[w].
struct GgufEmbedIQ1M {
  enum : uint { Weights = 256, Bytes = 56, Codes = 0, High = 32, Scales = 48 };
  __attribute__((always_inline)) static float decode(device const uchar *block, uint dim) {
    const uint j = (dim % Weights) / 32, e = (dim % 32) / 8, k = dim % 8;
    const ushort d = (gguf_u16(block, Scales) >> 12) | ((gguf_u16(block, Scales + 2) >> 8) & 0x00F0) |
                     ((gguf_u16(block, Scales + 4) >> 4) & 0x0F00) | (gguf_u16(block, Scales + 6) & 0xF000);
    const uint s = (gguf_u16(block, Scales + 2 * (j / 2)) >> (6 * (j % 2) + 3 * (e / 2))) & 7;
    const uint h = block[High + 2 * j + e / 2] >> (4 * (e % 2));
    return gguf_iq1_value(as_type<half>(d), s, block[Codes + 4 * j + e] | ((h & 7) << 8), k, h & 8);
  }
};
// block_mxfp4: uchar e | uchar qs[16], element l in nibble l / 16 of qs[l % 16], an E2M1 code; value =
// kFP4Values[code] (twice the E2M1 value) times 2^(e - 128), llama.cpp's GGML_E8M0_TO_FP32_HALF.
struct GgufEmbedMXFP4 {
  enum : uint { Weights = 32, Bytes = 17, E = 0, Codes = 1 };
  __attribute__((always_inline)) static float decode(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint l = dim % Weights, e = block[E];
    const uchar q = (block[Codes + l % 16] >> (4 * (l / 16))) & 15;
    return float(kFP4Values[q]) * as_type<float>(e < 2 ? 0x00200000u << e : (e - 1) << 23);
  }
};
// The loader's nvfp4 block: uchar scales[16] | float g | uchar codes[128], element l in nibble l % 2 of codes[l / 2]
// and its 16-group's E4M3 scale scales[l / 16]; value = kFP4Values[code] (twice the E2M1 value) times the scale the
// projections' coefficient rounds once (kernels/common/quant_formats.h), (e4m3 / 2^8) (128 g).
struct GgufEmbedNVFP4 {
  enum : uint { Weights = 256, Bytes = 148, Scales = 0, G = QUANT_NVFP4_G, Codes = QUANT_NVFP4_CODES };
  __attribute__((always_inline)) static float decode(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint l = dim % Weights;
    const uchar q = (block[Codes + l / 2] >> (4 * (l % 2))) & 15;
    const float g = as_type<float>(uint(block[G]) | uint(block[G + 1]) << 8 | uint(block[G + 2]) << 16 |
                                   uint(block[G + 3]) << 24);
    return float(kFP4Values[q]) * (float(quant_e4m3_pair(block[Scales + l / 16]).x) * (128.0f * g));
  }
};
// The loader's fp8 block: float g | uchar values[256], E4M3 values; value = (e4m3 / 2^8) (256 g), as the projections'
// coefficient scales it.
struct GgufEmbedFP8 {
  enum : uint { Weights = 256, Bytes = 260, G = 0, Values = QUANT_FP8_VALUES };
  __attribute__((always_inline)) static float decode(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    return float(quant_e4m3_pair(block[Values + dim % Weights]).x) * (256.0f * as_type<float>(gguf_u32(block, G)));
  }
};
// MLX affine {bf16 s, bf16 z, codes}: element l of the group is code l of the codes' little-endian bit string of
// Bits-bit codes; value = s * code + z, fused as the projections compute it.
template <uint Bits, uint Group> struct GgufEmbedAffine {
  enum : uint { Weights = Group, Bytes = 4 + Group * Bits / 8, S = 0, Z = 2, Codes = 4 };
  __attribute__((always_inline)) static float decode(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint at = (dim % Weights) * Bits, shift = at % 8;
    device const uchar *codes = block + Codes + at / 8;
    const uint word = uint(codes[0]) | (shift + Bits > 8 ? uint(codes[1]) << 8 : 0u);
    const float code = float((word >> shift) & ((1u << Bits) - 1));
    const float s = float(as_type<bfloat>(ushort(block[S] | (block[S + 1] << 8))));
    const float z = float(as_type<bfloat>(ushort(block[Z] | (block[Z + 1] << 8))));
    return fma(code, s, z);
  }
};
// BF16 and F16 values, which a GGUF keeps unquantized (a safetensors checkpoint its bf16 token table): each value is
// its own block, ggml's; the token gathers round an F16 value to bf16.
struct GgufEmbedBF16 {
  enum : uint { Weights = 1, Bytes = 2 };
  __attribute__((always_inline)) static float decode(device const uchar *block, uint) {
    return float(as_type<bfloat>(ushort(block[0] | (block[1] << 8))));
  }
};
struct GgufEmbedF16 {
  enum : uint { Weights = 1, Bytes = 2 };
  __attribute__((always_inline)) static float decode(device const uchar *block, uint) {
    return float(gguf_half(block, 0));
  }
};
#define GGUF_EMBED_AFFINE(B)                                                                                       \
  typedef GgufEmbedAffine<B, 32> GgufEmbedAF##B##G32; typedef GgufEmbedAffine<B, 64> GgufEmbedAF##B##G64;        \
  typedef GgufEmbedAffine<B, 128> GgufEmbedAF##B##G128;
GGUF_EMBED_AFFINE(2) GGUF_EMBED_AFFINE(3) GGUF_EMBED_AFFINE(4) GGUF_EMBED_AFFINE(5) GGUF_EMBED_AFFINE(6)
GGUF_EMBED_AFFINE(8)
#undef GGUF_EMBED_AFFINE

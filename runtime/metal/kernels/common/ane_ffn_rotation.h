#pragma once
#include "metal/abi/AneFfn.h"
#include "metal/kernels/common/quant_formats.h"

// The Hadamard rotation of the ANE FFN split (kernels/prefill/ane_ffn.metal), which the GPU's int8 FFN
// (kernels/prefill/int8_ffn.metal) applies alike: a block-diagonal rotation v -> diag(sign) H v / sqrt(n)
// of blocks of n = K * 128 values (ANE_FFN_ROTATION_UNIT), orthogonal, so it keeps the dot products of
// rows it rotates alike and spreads their outliers before int8 scales; and the values of a weight row
// in any format of metal/abi/QuantFormat.h, which both rotate.

// One block of K * 128 values held four values per lane in each 128 (value
// k * 128 + lane * 4 + e): two butterfly stages in registers and five across
// the simdgroup within each 128, then log2(K) in registers across them.
template <uint K>
inline void ane_ffn_rotate_block(thread float (&value)[K][4], uint lane, device const float *sign) {
  for (uint k = 0; k < K; ++k) {
    thread float (&v)[4] = value[k];
    for (uint half_span = 1; half_span < 4; half_span <<= 1)
      for (uint e = 0; e < 4; ++e)
        if (!(e & half_span)) {
          const float a = v[e], b = v[e + half_span];
          v[e] = a + b;
          v[e + half_span] = a - b;
        }
    for (uint mask = 1; mask < 32; mask <<= 1)
      for (uint e = 0; e < 4; ++e) {
        const float other = simd_shuffle_xor(v[e], mask);
        v[e] = (lane & mask) ? other - v[e] : v[e] + other;
      }
  }
  for (uint half_span = 1; half_span < K; half_span <<= 1)
    for (uint k = 0; k < K; ++k)
      if (!(k & half_span))
        for (uint e = 0; e < 4; ++e) {
          const float a = value[k][e], b = value[k + half_span][e];
          value[k][e] = a + b;
          value[k + half_span][e] = a - b;
        }
  const float norm = rsqrt(float(K * ANE_FFN_ROTATION_UNIT));
  for (uint k = 0; k < K; ++k)
    for (uint e = 0; e < 4; ++e) value[k][e] *= sign[k * ANE_FFN_ROTATION_UNIT + lane * 4 + e] * norm;
}

// The four values of `row` at inputs [input, input + 4) of a GGUF image
// tensor of format F with `groups` groups of 32 inputs per row: pairs 2h and
// 2h + 1 of chunk c of their group, h = input bit 4 and c = input bits 2, 3
// (metal/abi/QuantFormat.h), as kernels/common/gguf_staged.h decodes them.
template <class F>
inline void ane_ffn_gguf_values(thread float (&value)[4], device uchar *plane0, device uchar *plane1,
                                device uchar *meta, uint groups, uint row, uint input) {
  const uint group = input >> 5, h = (input >> 4) & 1;
  const ushort c = (input >> 2) & 3;
  const ulong unit = quant_tile_index(row, group, groups);
  const typename F::Payload w = F::load(plane0 + unit * F::P0, plane1 + unit * F::P1);
  const typename F::Meta header =
      F::loadMeta(meta + quant_tile_index(row, group / F::MetaGroups, groups / F::MetaGroups) * F::MetaBytes);
  const typename F::Chunk q = F::chunk(w, c);
  QuantCoef k;
  if constexpr (F::ScaleInChunk) k = F::coef(header, F::chunk(w, 0));
  else k = F::coef(header, ushort(group % F::MetaGroups));
  const float s = h ? k.s.y : k.s.x, m = h ? k.m.y : k.m.x;
  float4 v;
  if constexpr (F::Kind == QuantLinear) {
    const uint4 pairs = F::codes(q);
    const uint a = h ? pairs.z : pairs.x, b = h ? pairs.w : pairs.y;
    const float4 code = float4(a & 0xFFFFu, a >> 16, b & 0xFFFFu, b >> 16) - float(F::Zero);
    v = F::Zero ? code * s : fma(code, float4(s), float4(m));
  } else if constexpr (F::Kind == QuantCodebook) {
    const uchar4 bytes = as_type<uchar4>(F::indices(q));
    const uint a = h ? bytes.z : bytes.x, b = h ? bytes.w : bytes.y;
    v = float4(F::value(a & 15), F::value(a >> 4), F::value(b & 15), F::value(b >> 4)) * s;
  } else if constexpr (F::Kind == QuantInt8) {
    const uint2 codes = F::values(q);
    v = float4(as_type<char4>(h ? codes.y : codes.x)) * s;
  } else if constexpr (F::Kind == QuantFloat8) {
    const uint2 codes = F::values(q);
    v = float4(quant_e4m3_word(h ? codes.y : codes.x)) * s;
  } else {
    const uint2 grid = F::grid(q);
    const uint signs = F::signs(q) >> (4 * h);
    v = float4(as_type<uchar4>(h ? grid.y : grid.x)) * s;
    v = select(v, -v, bool4(signs & 1, signs & 2, signs & 4, signs & 8));
  }
  for (uint e = 0; e < 4; ++e) value[e] = v[e];
}

#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/quant_formats.h"

// The GPU side of a dense FFN split with the ANE (ops/AneFfn.cpp). The ANE
// multiplies int8 weights and activations after a block-diagonal Hadamard
// rotation R = diag(sign) H / sqrt(n) of blocks of n = K * 128 values:
// (x R)(W R)^T = x W^T, and the rotation spreads outliers before the
// per-token and per-row int8 scales. Weights are rotated here from each
// chunk's layer: affine Q4 planes, or a GGUF image tensor in any format.

constant constexpr uint kAneFfnBlock = 128;

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
  const float norm = rsqrt(float(K * kAneFfnBlock));
  for (uint k = 0; k < K; ++k)
    for (uint e = 0; e < 4; ++e) value[k][e] *= sign[k * kAneFfnBlock + lane * 4 + e] * norm;
}

// The four Q4 values of `row` at inputs [input, input + 4).
inline void ane_ffn_q4_values(thread float (&value)[4], device const uchar *weights,
                              device const bfloat *scales, device const bfloat *biases,
                              uint groups, uint row, uint input) {
  const uint unit = ((row >> 8) * groups + (input >> 6)) * 256 + (row & 255);
  const float scale = float(scales[unit]), bias = float(biases[unit]);
  device const uchar *nibbles = weights + ulong(unit) * 32 + ((input & 63) >> 1);
  for (uint e = 0; e < 4; e += 2) {
    const uchar pair = nibbles[e >> 1];
    value[e] = float(pair & 15) * scale + bias;
    value[e + 1] = float(pair >> 4) * scale + bias;
  }
}

// One threadgroup per token: its rotated row scaled to [-127, 127] and the
// scale back, times the 128 of the ANE's 2^-7 dequantization.
kernel void ane_ffn_rotate(device const bfloat *input [[buffer(0)]],
                           device const float *sign [[buffer(1)]],
                           device half *rotated [[buffer(2)]],
                           device half *token_scale [[buffer(3)]],
                           constant AneFfnRotateParams &params [[buffer(4)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint simd_group [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint simd_groups [[simdgroups_per_threadgroup]]) {
  constexpr uint MaximumBlocks = 8;
  threadgroup float peaks[32];
  const uint blocks = params.hidden / kAneFfnBlock / simd_groups;
  float value[MaximumBlocks][1][4];
  float peak = 0.0f;
  for (uint block = 0; block < blocks; ++block) {
    const uint origin = row * params.hidden + (simd_group * blocks + block) * kAneFfnBlock + lane * 4;
    for (uint e = 0; e < 4; ++e) value[block][0][e] = float(input[origin + e]);
    ane_ffn_rotate_block<1>(value[block], lane, sign);
    for (uint e = 0; e < 4; ++e) peak = max(peak, fabs(value[block][0][e]));
  }
  peak = simd_max(peak);
  if (lane == 0) peaks[simd_group] = peak;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  peak = 0.0f;
  for (uint group = 0; group < simd_groups; ++group) peak = max(peak, peaks[group]);
  const float scale = max(peak / 127.0f, 1e-8f), inverse = 1.0f / scale;
  for (uint block = 0; block < blocks; ++block) {
    const uint origin = row * params.hidden + (simd_group * blocks + block) * kAneFfnBlock + lane * 4;
    for (uint e = 0; e < 4; ++e) rotated[origin + e] = half(value[block][0][e] * inverse);
  }
  if (simd_group == 0 && lane == 0) token_scale[row] = half(scale * 128.0f);
}

// 32 x 32 tiles of rows x channels, 32 x 8 threads.
kernel void ane_ffn_pack(device const half *rotated [[buffer(0)]],
                         device char *packed [[buffer(1)]],
                         constant AneFfnPackParams &params [[buffer(2)]],
                         uint2 tile [[threadgroup_position_in_grid]],
                         uint2 position [[thread_position_in_threadgroup]]) {
  threadgroup half staged[32][33];
  const uint row = tile.x * 32, channel = tile.y * 32;
  for (uint j = position.y; j < 32; j += 8)
    staged[j][position.x] = rotated[(row + j) * params.hidden + params.channel + channel + position.x];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = position.y; j < 32; j += 8)
    packed[(channel + j) * params.stride + row + position.x] =
        char(clamp(rint(float(staged[position.x][j])), -127.0f, 127.0f));
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
  } else {
    const uint2 grid = F::grid(q);
    const uint signs = F::signs(q) >> (4 * h);
    v = float4(as_type<uchar4>(h ? grid.y : grid.x)) * s;
    v = select(v, -v, bool4(signs & 1, signs & 2, signs & 4, signs & 8));
  }
  for (uint e = 0; e < 4; ++e) value[e] = v[e];
}

// A projection's weight planes a, b, c: the affine Q4 weights, scales and
// biases (groups = inputs / 64), or a GGUF image tensor's plane0, plane1 and
// meta in format F (groups = inputs / 32).
struct AneFfnAffine {
  static void values(thread float (&value)[4], device uchar *a, device uchar *b, device uchar *c, uint groups, uint row,
                     uint input) {
    ane_ffn_q4_values(value, a, (device const bfloat *)b, (device const bfloat *)c, groups, row, input);
  }
};
template <class F>
struct AneFfnGguf {
  static void values(thread float (&value)[4], device uchar *a, device uchar *b, device uchar *c, uint groups, uint row,
                     uint input) {
    ane_ffn_gguf_values<F>(value, a, b, c, groups, row, input);
  }
};

// One simdgroup per weight row, eight rows x one rotation block of K * 128
// inputs per threadgroup: the int8 rows under their shared scale, which the
// first block also copies into the ANE's scale surface.
template <uint K, class Source>
inline void ane_ffn_weight_rows(device uchar *a, device uchar *b, device uchar *c, device const half *row_scale,
                                device char *output, device half *scale, device const float *sign,
                                constant AneFfnWeightParams &params, uint2 tile, uint simd_group, uint lane) {
  const uint row = tile.x * 8 + simd_group, block = tile.y * K * kAneFfnBlock;
  float value[K][4];
  for (uint k = 0; k < K; ++k)
    Source::values(value[k], a, b, c, params.groups, params.row + row, params.input + block + k * kAneFfnBlock + lane * 4);
  ane_ffn_rotate_block<K>(value, lane, sign);
  const float inverse = 128.0f / float(row_scale[row]);
  for (uint k = 0; k < K; ++k)
    *(device char4 *)(output + ulong(row) * params.stride + block + k * kAneFfnBlock + lane * 4) =
        char4(clamp(rint(float4(value[k][0], value[k][1], value[k][2], value[k][3]) * inverse), -127.0f, 127.0f));
  if (tile.y == 0 && lane == 0) scale[row * params.scale_stride] = row_scale[row];
}

// Each row's largest weight over inputs [input, input + width) rotated in
// blocks of K * 128, as the shared scale times 128 of ane_ffn_weights and the
// ANE.
template <uint K, class Source>
inline void ane_ffn_row_scales(device uchar *a, device uchar *b, device uchar *c, device half *row_scale,
                               device const float *sign, constant AneFfnWeightParams &params, uint tile,
                               uint simd_group, uint lane) {
  const uint row = tile * 8 + simd_group;
  float peak = 0.0f;
  for (uint block = 0; block < params.width; block += K * kAneFfnBlock) {
    float value[K][4];
    for (uint k = 0; k < K; ++k)
      Source::values(value[k], a, b, c, params.groups, params.row + row, params.input + block + k * kAneFfnBlock + lane * 4);
    ane_ffn_rotate_block<K>(value, lane, sign);
    for (uint k = 0; k < K; ++k)
      for (uint e = 0; e < 4; ++e) peak = max(peak, fabs(value[k][e]));
  }
  peak = simd_max(peak);
  if (lane == 0) row_scale[row] = half(max(peak, 1e-8f) / 127.0f * 128.0f);
}

template <uint K>
kernel void ane_ffn_weights(device uchar *a [[buffer(0)]], device uchar *b [[buffer(1)]], device uchar *c [[buffer(2)]],
    device const half *row_scale [[buffer(3)]], device char *output [[buffer(4)]], device half *scale [[buffer(5)]],
    device const float *sign [[buffer(6)]], constant AneFfnWeightParams &params [[buffer(7)]],
    uint2 tile [[threadgroup_position_in_grid]], uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  ane_ffn_weight_rows<K, AneFfnAffine>(a, b, c, row_scale, output, scale, sign, params, tile, simd_group, lane);
}
// The GGUF variants run the body of the dispatch's format.
template <uint K>
kernel void ane_ffn_weights_gguf(device uchar *a [[buffer(0)]], device uchar *b [[buffer(1)]], device uchar *c [[buffer(2)]],
    device const half *row_scale [[buffer(3)]], device char *output [[buffer(4)]], device half *scale [[buffer(5)]],
    device const float *sign [[buffer(6)]], constant AneFfnWeightParams &params [[buffer(7)]],
    uint2 tile [[threadgroup_position_in_grid]], uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  quant_format_switch(params.format, [&](auto format) {
    ane_ffn_weight_rows<K, AneFfnGguf<decltype(format)>>(a, b, c, row_scale, output, scale, sign, params, tile,
                                                         simd_group, lane);
  });
}
template <uint K>
kernel void ane_ffn_row_scale(device uchar *a [[buffer(0)]], device uchar *b [[buffer(1)]], device uchar *c [[buffer(2)]],
    device half *row_scale [[buffer(3)]], device const float *sign [[buffer(4)]],
    constant AneFfnWeightParams &params [[buffer(5)]], uint tile [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  ane_ffn_row_scales<K, AneFfnAffine>(a, b, c, row_scale, sign, params, tile, simd_group, lane);
}
template <uint K>
kernel void ane_ffn_row_scale_gguf(device uchar *a [[buffer(0)]], device uchar *b [[buffer(1)]], device uchar *c [[buffer(2)]],
    device half *row_scale [[buffer(3)]], device const float *sign [[buffer(4)]],
    constant AneFfnWeightParams &params [[buffer(5)]], uint tile [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  quant_format_switch(params.format, [&](auto format) {
    ane_ffn_row_scales<K, AneFfnGguf<decltype(format)>>(a, b, c, row_scale, sign, params, tile, simd_group, lane);
  });
}

using AneFfnWeightsKernel = void(device uchar *, device uchar *, device uchar *, device const half *, device char *,
                                 device half *, device const float *, constant AneFfnWeightParams &, uint2, uint, uint);
using AneFfnRowScaleKernel = void(device uchar *, device uchar *, device uchar *, device half *, device const float *,
                                  constant AneFfnWeightParams &, uint, uint, uint);
template [[host_name("ane_ffn_weights_128")]] kernel AneFfnWeightsKernel ane_ffn_weights<1>;
template [[host_name("ane_ffn_weights_512")]] kernel AneFfnWeightsKernel ane_ffn_weights<4>;
template [[host_name("ane_ffn_weights_gguf_128")]] kernel AneFfnWeightsKernel ane_ffn_weights_gguf<1>;
template [[host_name("ane_ffn_weights_gguf_512")]] kernel AneFfnWeightsKernel ane_ffn_weights_gguf<4>;
template [[host_name("ane_ffn_row_scale_128")]] kernel AneFfnRowScaleKernel ane_ffn_row_scale<1>;
template [[host_name("ane_ffn_row_scale_512")]] kernel AneFfnRowScaleKernel ane_ffn_row_scale<4>;
template [[host_name("ane_ffn_row_scale_gguf_128")]] kernel AneFfnRowScaleKernel ane_ffn_row_scale_gguf<1>;
template [[host_name("ane_ffn_row_scale_gguf_512")]] kernel AneFfnRowScaleKernel ane_ffn_row_scale_gguf<4>;

// 32 x 32 tiles of rows x channels, 32 x 8 threads.
kernel void ane_ffn_join(device bfloat *output [[buffer(0)]],
                         device const half *partial [[buffer(1)]],
                         constant AneFfnJoinParams &params [[buffer(2)]],
                         uint2 tile [[threadgroup_position_in_grid]],
                         uint2 position [[thread_position_in_threadgroup]]) {
  threadgroup float staged[32][33];
  const uint row = tile.x * 32, channel = tile.y * 32;
  for (uint j = position.y; j < 32; j += 8)
    staged[j][position.x] = float(partial[(channel + j) * params.stride + row + position.x]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = position.y; j < 32; j += 8) {
    if (row + j >= params.rows) continue;
    const uint index = (row + j) * params.hidden + channel + position.x;
    output[index] = bfloat(float(output[index]) + staged[position.x][j]);
  }
}

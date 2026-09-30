#include "metal/abi/KernelABI.h"

// The GPU side of a dense FFN split with the ANE (ops/AneFfn.cpp). The ANE
// multiplies int8 weights and activations after a block-diagonal Hadamard
// rotation R = diag(sign) H / sqrt(n) of blocks of n = K * 128 values:
// (x R)(W R)^T = x W^T, and the rotation spreads outliers before the
// per-token and per-row int8 scales. Weights are rotated here from the affine
// Q4 planes of each chunk's layer.

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

// One simdgroup per weight row, eight rows x one rotation block of K * 128
// inputs per threadgroup: the int8 rows under their shared scale, which the
// first block also copies into the ANE's scale surface.
template <uint K>
kernel void ane_ffn_weights(device const uchar *weights [[buffer(0)]],
                            device const bfloat *scales [[buffer(1)]],
                            device const bfloat *biases [[buffer(2)]],
                            device const half *row_scale [[buffer(3)]],
                            device char *output [[buffer(4)]],
                            device half *scale [[buffer(5)]],
                            device const float *sign [[buffer(6)]],
                            constant AneFfnWeightParams &params [[buffer(7)]],
                            uint2 tile [[threadgroup_position_in_grid]],
                            uint simd_group [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tile.x * 8 + simd_group, block = tile.y * K * kAneFfnBlock;
  float value[K][4];
  for (uint k = 0; k < K; ++k)
    ane_ffn_q4_values(value[k], weights, scales, biases, params.groups, params.row + row,
                      params.input + block + k * kAneFfnBlock + lane * 4);
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
template <uint K>
kernel void ane_ffn_row_scale(device const uchar *weights [[buffer(0)]],
                              device const bfloat *scales [[buffer(1)]],
                              device const bfloat *biases [[buffer(2)]],
                              device half *row_scale [[buffer(3)]],
                              device const float *sign [[buffer(4)]],
                              constant AneFfnWeightParams &params [[buffer(5)]],
                              uint tile [[threadgroup_position_in_grid]],
                              uint simd_group [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tile * 8 + simd_group;
  float peak = 0.0f;
  for (uint block = 0; block < params.width; block += K * kAneFfnBlock) {
    float value[K][4];
    for (uint k = 0; k < K; ++k)
      ane_ffn_q4_values(value[k], weights, scales, biases, params.groups, params.row + row,
                        params.input + block + k * kAneFfnBlock + lane * 4);
    ane_ffn_rotate_block<K>(value, lane, sign);
    for (uint k = 0; k < K; ++k)
      for (uint e = 0; e < 4; ++e) peak = max(peak, fabs(value[k][e]));
  }
  peak = simd_max(peak);
  if (lane == 0) row_scale[row] = half(max(peak, 1e-8f) / 127.0f * 128.0f);
}

using AneFfnWeightsKernel = void(device const uchar *, device const bfloat *, device const bfloat *,
                                 device const half *, device char *, device half *, device const float *,
                                 constant AneFfnWeightParams &, uint2, uint, uint);
using AneFfnRowScaleKernel = void(device const uchar *, device const bfloat *, device const bfloat *, device half *,
                                  device const float *, constant AneFfnWeightParams &, uint, uint, uint);
template [[host_name("ane_ffn_weights_128")]] kernel AneFfnWeightsKernel ane_ffn_weights<1>;
template [[host_name("ane_ffn_weights_512")]] kernel AneFfnWeightsKernel ane_ffn_weights<4>;
template [[host_name("ane_ffn_row_scale_128")]] kernel AneFfnRowScaleKernel ane_ffn_row_scale<1>;
template [[host_name("ane_ffn_row_scale_512")]] kernel AneFfnRowScaleKernel ane_ffn_row_scale<4>;

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

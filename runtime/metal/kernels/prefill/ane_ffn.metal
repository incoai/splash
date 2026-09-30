#include "metal/abi/KernelABI.h"

// The GPU side of a dense FFN split with the ANE (ops/AneFfn.cpp). The ANE
// multiplies int8 weights and activations after a block-diagonal Hadamard
// rotation R = H diag(sign) / sqrt(128): (x R)(W R)^T = x W^T, and the
// rotation spreads outliers before the per-token and per-row int8 scales.
// Weights are rotated here from the affine Q4 planes of each chunk's layer.

constant constexpr uint kAneFfnBlock = 128;

// One 128-value block held four values per lane (value lane * 4 + e): two
// butterfly stages in registers, five across the simdgroup.
inline void ane_ffn_rotate_block(thread float (&value)[4], uint lane, device const float *sign) {
  for (uint half_span = 1; half_span < 4; half_span <<= 1)
    for (uint e = 0; e < 4; ++e)
      if (!(e & half_span)) {
        const float a = value[e], b = value[e + half_span];
        value[e] = a + b;
        value[e + half_span] = a - b;
      }
  for (uint mask = 1; mask < 32; mask <<= 1)
    for (uint e = 0; e < 4; ++e) {
      const float other = simd_shuffle_xor(value[e], mask);
      value[e] = (lane & mask) ? other - value[e] : value[e] + other;
    }
  for (uint e = 0; e < 4; ++e)
    value[e] *= sign[lane * 4 + e] * 0.08838834765f;
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
  float value[MaximumBlocks][4];
  float peak = 0.0f;
  for (uint block = 0; block < blocks; ++block) {
    const uint origin = row * params.hidden + (simd_group * blocks + block) * kAneFfnBlock + lane * 4;
    thread float (&v)[4] = value[block];
    for (uint e = 0; e < 4; ++e) v[e] = float(input[origin + e]);
    ane_ffn_rotate_block(v, lane, sign);
    for (uint e = 0; e < 4; ++e) peak = max(peak, fabs(v[e]));
  }
  peak = simd_max(peak);
  if (lane == 0) peaks[simd_group] = peak;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  peak = 0.0f;
  for (uint group = 0; group < simd_groups; ++group) peak = max(peak, peaks[group]);
  const float scale = max(peak / 127.0f, 1e-8f), inverse = 1.0f / scale;
  for (uint block = 0; block < blocks; ++block) {
    const uint origin = row * params.hidden + (simd_group * blocks + block) * kAneFfnBlock + lane * 4;
    for (uint e = 0; e < 4; ++e) rotated[origin + e] = half(value[block][e] * inverse);
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

// One simdgroup per weight row, eight rows x one rotation block per
// threadgroup: the int8 rows under their shared scale, which the first block
// also copies into the ANE's scale surface.
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
  const uint row = tile.x * 8 + simd_group, local = tile.y * kAneFfnBlock + lane * 4;
  float value[4];
  ane_ffn_q4_values(value, weights, scales, biases, params.groups, params.row + row, params.input + local);
  ane_ffn_rotate_block(value, lane, sign);
  const float inverse = 128.0f / float(row_scale[row]);
  *(device char4 *)(output + ulong(row) * params.stride + local) =
      char4(clamp(rint(float4(value[0], value[1], value[2], value[3]) * inverse), -127.0f, 127.0f));
  if (tile.y == 0 && lane == 0) scale[row * params.scale_stride] = row_scale[row];
}

// Each row's largest rotated weight over inputs [input, input + width), as
// the shared scale times 128 of ane_ffn_weights and the ANE.
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
  for (uint block = 0; block < params.width / kAneFfnBlock; ++block) {
    float value[4];
    ane_ffn_q4_values(value, weights, scales, biases, params.groups, params.row + row,
                      params.input + block * kAneFfnBlock + lane * 4);
    ane_ffn_rotate_block(value, lane, sign);
    for (uint e = 0; e < 4; ++e) peak = max(peak, fabs(value[e]));
  }
  peak = simd_max(peak);
  if (lane == 0) row_scale[row] = half(max(peak, 1e-8f) / 127.0f * 128.0f);
}

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

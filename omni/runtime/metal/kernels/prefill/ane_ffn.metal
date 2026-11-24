// The GPU side of the prefill FFN's split with the Neural Engine
// (ops/AneFfn.cpp), ported from Splash for affine Q4 weights. The ANE
// multiplies int8 weights and activations after a block-diagonal Hadamard
// rotation v -> diag(sign) H v / sqrt(n) of blocks of n = K * 128 values
// (ANE_FFN_ROTATION_UNIT): orthogonal, it keeps the dot products of rows it
// rotates alike, and spreads their outliers before the per-token and per-row
// int8 scales. The weights are rotated and quantized here once, when the
// split is built; each chunk's rows are rotated and packed per layer.

#include "metal/abi/AneFfn.h"
#include "metal/abi/Linear.h"
#include <metal_stdlib>

using namespace metal;

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

// The four Q4 values of `row` at inputs [input, input + 4) of an affine
// projection of `groups` quant groups per row, its planes in tiles of
// SPLASH_AFFINE_TILE_ROWS rows, [N / T][groups][T] (metal/abi/Linear.h).
inline void ane_ffn_q4_values(thread float (&value)[4], device const uchar *weights,
                              device const bfloat *scales, device const bfloat *biases,
                              uint groups, uint row, uint input) {
  const ulong unit =
      (ulong(row / SPLASH_AFFINE_TILE_ROWS) * groups + input / 64) * SPLASH_AFFINE_TILE_ROWS +
      row % SPLASH_AFFINE_TILE_ROWS;
  const float scale = float(scales[unit]), bias = float(biases[unit]);
  device const uchar *nibbles = weights + unit * 32 + ((input & 63) >> 1);
  for (uint e = 0; e < 4; e += 2) {
    const uchar pair = nibbles[e >> 1];
    value[e] = float(pair & 15) * scale + bias;
    value[e + 1] = float(pair >> 4) * scale + bias;
  }
}

// One threadgroup of ANE_FFN_ROTATE_THREADS per token, its simdgroups taking
// the row's input blocks in turn: the rotated row scaled to
// +-ANE_FFN_INT8_PEAK, and the scale back times the ANE's ANE_FFN_INT8_UNIT.
// The rotation runs twice, for the row's peak and then for its values, so
// that no simdgroup holds more than one block.
constant constexpr uint kRotateUnits = ANE_FFN_INPUT_BLOCK / ANE_FFN_ROTATION_UNIT;
kernel void ane_ffn_rotate(device const bfloat *input [[buffer(0)]],
                           device const float *sign [[buffer(1)]],
                           device half *rotated [[buffer(2)]],
                           device half *token_scale [[buffer(3)]],
                           constant AneFfnRotateParams &params [[buffer(4)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint simd_group [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint simd_groups [[simdgroups_per_threadgroup]]) {
  threadgroup float peaks[32];
  const uint blocks = params.hidden / ANE_FFN_INPUT_BLOCK;
  const auto load = [&](thread float (&value)[kRotateUnits][4], uint block) {
    const uint origin = row * params.hidden + block * ANE_FFN_INPUT_BLOCK + lane * 4;
    for (uint k = 0; k < kRotateUnits; ++k)
      for (uint e = 0; e < 4; ++e) value[k][e] = float(input[origin + k * ANE_FFN_ROTATION_UNIT + e]);
    ane_ffn_rotate_block<kRotateUnits>(value, lane, sign);
  };
  float peak = 0.0f;
  for (uint block = simd_group; block < blocks; block += simd_groups) {
    float value[kRotateUnits][4];
    load(value, block);
    for (uint k = 0; k < kRotateUnits; ++k)
      for (uint e = 0; e < 4; ++e) peak = max(peak, fabs(value[k][e]));
  }
  peak = simd_max(peak);
  if (lane == 0) peaks[simd_group] = peak;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  peak = 0.0f;
  for (uint group = 0; group < simd_groups; ++group) peak = max(peak, peaks[group]);
  const float scale = max(peak, ANE_FFN_PEAK_FLOOR) / ANE_FFN_INT8_PEAK, inverse = 1.0f / scale;
  for (uint block = simd_group; block < blocks; block += simd_groups) {
    float value[kRotateUnits][4];
    load(value, block);
    const uint origin = row * params.hidden + block * ANE_FFN_INPUT_BLOCK + lane * 4;
    for (uint k = 0; k < kRotateUnits; ++k)
      for (uint e = 0; e < 4; ++e) rotated[origin + k * ANE_FFN_ROTATION_UNIT + e] = half(value[k][e] * inverse);
  }
  if (simd_group == 0 && lane == 0) token_scale[row] = half(scale * ANE_FFN_INT8_UNIT);
}

// ANE_FFN_TILE x ANE_FFN_TILE tiles of rows x channels, ANE_FFN_TILE x
// ANE_FFN_TILE_ROWS threads: the rotated rows as channel-major int8.
kernel void ane_ffn_pack(device const half *rotated [[buffer(0)]],
                         device char *packed [[buffer(1)]],
                         constant AneFfnPackParams &params [[buffer(2)]],
                         uint2 tile [[threadgroup_position_in_grid]],
                         uint2 position [[thread_position_in_threadgroup]]) {
  threadgroup half staged[ANE_FFN_TILE][ANE_FFN_TILE + 1];
  const uint row = tile.x * ANE_FFN_TILE, channel = tile.y * ANE_FFN_TILE;
  for (uint j = position.y; j < ANE_FFN_TILE; j += ANE_FFN_TILE_ROWS)
    staged[j][position.x] = rotated[(row + j) * params.hidden + channel + position.x];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = position.y; j < ANE_FFN_TILE; j += ANE_FFN_TILE_ROWS)
    packed[(channel + j) * params.stride + row + position.x] =
        char(clamp(rint(float(staged[position.x][j])), -ANE_FFN_INT8_PEAK, ANE_FFN_INT8_PEAK));
}

// One simdgroup per weight row, ANE_FFN_WEIGHT_ROWS rows x one rotation block
// of K * 128 inputs per threadgroup: the int8 rows under their shared scale,
// which the first block also copies into the ANE's scale surface.
template <uint K, uint Rotation>
kernel void ane_ffn_weights(device const uchar *weights [[buffer(0)]], device const bfloat *scales [[buffer(1)]],
                            device const bfloat *biases [[buffer(2)]], device const half *row_scale [[buffer(3)]],
                            device char *output [[buffer(4)]], device half *scale [[buffer(5)]],
                            device const float *sign [[buffer(6)]], constant AneFfnWeightParams &params [[buffer(7)]],
                            uint2 tile [[threadgroup_position_in_grid]],
                            uint simd_group [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tile.x * ANE_FFN_WEIGHT_ROWS + simd_group, block = tile.y * K * ANE_FFN_ROTATION_UNIT;
  float value[K][4];
  for (uint k = 0; k < K; ++k)
    ane_ffn_q4_values(value[k], weights, scales, biases, params.groups, params.row + row,
                      params.input + block + k * ANE_FFN_ROTATION_UNIT + lane * 4);
  ane_ffn_rotate_block<K>(value, lane, sign);
  const float inverse = ANE_FFN_INT8_UNIT / float(row_scale[row]);
  for (uint k = 0; k < K; ++k)
    *(device char4 *)(output + ulong(row) * params.stride + block + k * ANE_FFN_ROTATION_UNIT + lane * 4) =
        char4(clamp(rint(float4(value[k][0], value[k][1], value[k][2], value[k][3]) * inverse), -ANE_FFN_INT8_PEAK,
                    ANE_FFN_INT8_PEAK));
  if (tile.y == 0 && lane == 0) scale[row * params.scale_stride] = row_scale[row];
}

// Each row's largest weight over inputs [input, input + width) rotated in
// blocks of K * 128, as the shared scale, times the ANE's ANE_FFN_INT8_UNIT,
// of ane_ffn_weights and the ANE.
template <uint K, uint Rotation>
kernel void ane_ffn_row_scale(device const uchar *weights [[buffer(0)]], device const bfloat *scales [[buffer(1)]],
                              device const bfloat *biases [[buffer(2)]], device half *row_scale [[buffer(3)]],
                              device const float *sign [[buffer(4)]],
                              constant AneFfnWeightParams &params [[buffer(5)]],
                              uint tile [[threadgroup_position_in_grid]],
                              uint simd_group [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tile * ANE_FFN_WEIGHT_ROWS + simd_group;
  float peak = 0.0f;
  for (uint block = 0; block < params.width; block += K * ANE_FFN_ROTATION_UNIT) {
    float value[K][4];
    for (uint k = 0; k < K; ++k)
      ane_ffn_q4_values(value[k], weights, scales, biases, params.groups, params.row + row,
                        params.input + block + k * ANE_FFN_ROTATION_UNIT + lane * 4);
    ane_ffn_rotate_block<K>(value, lane, sign);
    for (uint k = 0; k < K; ++k)
      for (uint e = 0; e < 4; ++e) peak = max(peak, fabs(value[k][e]));
  }
  peak = simd_max(peak);
  if (lane == 0) row_scale[row] = half(max(peak, ANE_FFN_PEAK_FLOOR) / ANE_FFN_INT8_PEAK * ANE_FFN_INT8_UNIT);
}

using AneFfnWeightsKernel = void(device const uchar *, device const bfloat *, device const bfloat *,
                                 device const half *, device char *, device half *, device const float *,
                                 constant AneFfnWeightParams &, uint2, uint, uint);
using AneFfnRowScaleKernel = void(device const uchar *, device const bfloat *, device const bfloat *, device half *,
                                  device const float *, constant AneFfnWeightParams &, uint, uint, uint);
// _inputs: gate's and up's rows over the hidden inputs; _intermediate: down's
// rows over the ANE's intermediate channels.
constant constexpr uint kInputUnits = ANE_FFN_INPUT_BLOCK / ANE_FFN_ROTATION_UNIT;
constant constexpr uint kIntermediateUnits = ANE_FFN_INTERMEDIATE_BLOCK / ANE_FFN_ROTATION_UNIT;
template [[host_name("ane_ffn_weights_inputs")]] kernel AneFfnWeightsKernel ane_ffn_weights<kInputUnits, 0>;
template [[host_name("ane_ffn_weights_intermediate")]] kernel AneFfnWeightsKernel ane_ffn_weights<kIntermediateUnits, 1>;
template [[host_name("ane_ffn_row_scale_inputs")]] kernel AneFfnRowScaleKernel ane_ffn_row_scale<kInputUnits, 0>;
template [[host_name("ane_ffn_row_scale_intermediate")]] kernel AneFfnRowScaleKernel
    ane_ffn_row_scale<kIntermediateUnits, 1>;

// Whether `value` is an infinity or a NaN, by its exponent bits: fast math
// may fold isfinite() to true.
inline bool ane_ffn_not_finite(half value) { return (as_type<ushort>(value) & 0x7c00) == 0x7c00; }

// ANE_FFN_TILE x ANE_FFN_TILE tiles of rows x channels, ANE_FFN_TILE x
// ANE_FFN_TILE_ROWS threads. Each row's partial values take their token's
// intermediate scale (row `hidden` of the partial) and input scale in fp32. A
// partial value or token scale of a row of the chunk that is not finite sets
// `status`, which the host reads once the command completes. Each output
// rounds to bf16 once more than on the GPU alone, after the GPU part's
// residual epilogue: half a bf16 unit.
kernel void ane_ffn_join(device bfloat *output [[buffer(0)]],
                         device const half *partial [[buffer(1)]],
                         device const half *token_scale [[buffer(2)]],
                         device atomic_uint *status [[buffer(3)]],
                         constant AneFfnJoinParams &params [[buffer(4)]],
                         uint2 tile [[threadgroup_position_in_grid]],
                         uint2 position [[thread_position_in_threadgroup]]) {
  threadgroup float staged[ANE_FFN_TILE][ANE_FFN_TILE + 1];
  threadgroup float scale[ANE_FFN_TILE];
  const uint row = tile.x * ANE_FFN_TILE, channel = tile.y * ANE_FFN_TILE, token = row + position.x;
  const bool chunk = token < params.rows;
  bool finite = true;
  for (uint j = position.y; j < ANE_FFN_TILE; j += ANE_FFN_TILE_ROWS) {
    const half value = partial[(channel + j) * params.stride + token];
    finite &= !(chunk && ane_ffn_not_finite(value));
    staged[j][position.x] = float(value);
  }
  if (position.y == 0) {
    const half intermediate = partial[params.hidden * params.stride + token], input = token_scale[token];
    finite &= !(chunk && (ane_ffn_not_finite(intermediate) || ane_ffn_not_finite(input)));
    scale[position.x] = float(intermediate) * float(input);
  }
  if (!finite) atomic_store_explicit(status, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = position.y; j < ANE_FFN_TILE; j += ANE_FFN_TILE_ROWS) {
    if (row + j >= params.rows) continue;
    const uint index = (row + j) * params.hidden + channel + position.x;
    output[index] = bfloat(float(output[index]) + staged[position.x][j] * scale[j]);
  }
}

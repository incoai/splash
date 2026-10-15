#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/ane_ffn_rotation.h"

// The GPU side of a dense FFN split with the ANE (ops/AneFfn.cpp). The ANE
// multiplies int8 weights and activations after a block-diagonal Hadamard
// rotation v -> diag(sign) H v / sqrt(n) of blocks of n = K * 128 values
// (ANE_FFN_ROTATION_UNIT): orthogonal, it keeps the dot products of rows it
// rotates alike, and spreads their outliers before the per-token and per-row
// int8 scales. Weights are rotated here from each chunk's layer: an image
// tensor in any format of metal/abi/QuantFormat.h.

// One threadgroup of ANE_FFN_ROTATE_THREADS per token, each simdgroup
// rotating at most ANE_FFN_ROTATE_BLOCKS of its input blocks: the rotated row
// scaled to +-ANE_FFN_INT8_PEAK, and the scale back times the ANE's
// ANE_FFN_INT8_UNIT.
static_assert(ANE_FFN_INPUT_BLOCK == ANE_FFN_ROTATION_UNIT, "ane_ffn_rotate rotates an input block per unit");
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
  const uint blocks = params.hidden / ANE_FFN_INPUT_BLOCK / simd_groups;
  float value[ANE_FFN_ROTATE_BLOCKS][1][4];
  float peak = 0.0f;
  for (uint block = 0; block < blocks; ++block) {
    const uint origin = row * params.hidden + (simd_group * blocks + block) * ANE_FFN_INPUT_BLOCK + lane * 4;
    for (uint e = 0; e < 4; ++e) value[block][0][e] = float(input[origin + e]);
    ane_ffn_rotate_block<1>(value[block], lane, sign);
    for (uint e = 0; e < 4; ++e) peak = max(peak, fabs(value[block][0][e]));
  }
  peak = simd_max(peak);
  if (lane == 0) peaks[simd_group] = peak;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  peak = 0.0f;
  for (uint group = 0; group < simd_groups; ++group) peak = max(peak, peaks[group]);
  const float scale = max(peak, ANE_FFN_PEAK_FLOOR) / ANE_FFN_INT8_PEAK, inverse = 1.0f / scale;
  for (uint block = 0; block < blocks; ++block) {
    const uint origin = row * params.hidden + (simd_group * blocks + block) * ANE_FFN_INPUT_BLOCK + lane * 4;
    for (uint e = 0; e < 4; ++e) rotated[origin + e] = half(value[block][0][e] * inverse);
  }
  if (simd_group == 0 && lane == 0) token_scale[row] = half(scale * ANE_FFN_INT8_UNIT);
}

// ANE_FFN_TILE x ANE_FFN_TILE tiles of rows x channels, ANE_FFN_TILE x
// ANE_FFN_TILE_ROWS threads.
kernel void ane_ffn_pack(device const half *rotated [[buffer(0)]],
                         device char *packed [[buffer(1)]],
                         constant AneFfnPackParams &params [[buffer(2)]],
                         uint2 tile [[threadgroup_position_in_grid]],
                         uint2 position [[thread_position_in_threadgroup]]) {
  threadgroup half staged[ANE_FFN_TILE][ANE_FFN_TILE + 1];
  const uint row = tile.x * ANE_FFN_TILE, channel = tile.y * ANE_FFN_TILE;
  for (uint j = position.y; j < ANE_FFN_TILE; j += ANE_FFN_TILE_ROWS)
    staged[j][position.x] = rotated[(row + j) * params.hidden + params.channel + channel + position.x];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = position.y; j < ANE_FFN_TILE; j += ANE_FFN_TILE_ROWS)
    packed[(channel + j) * params.stride + row + position.x] =
        char(clamp(rint(float(staged[position.x][j])), -ANE_FFN_INT8_PEAK, ANE_FFN_INT8_PEAK));
}

// One simdgroup per weight row, ANE_FFN_WEIGHT_ROWS rows x one rotation block
// of K * 128 inputs per threadgroup: the int8 rows under their shared scale,
// which the first block also copies into the ANE's scale surface.
template <uint K, class F>
inline void ane_ffn_weight_rows(device uchar *a, device uchar *b, device uchar *c, device const half *row_scale,
                                device char *output, device half *scale, device const float *sign,
                                constant AneFfnWeightParams &params, uint2 tile, uint simd_group, uint lane) {
  const uint row = tile.x * ANE_FFN_WEIGHT_ROWS + simd_group, block = tile.y * K * ANE_FFN_ROTATION_UNIT;
  float value[K][4];
  for (uint k = 0; k < K; ++k)
    ane_ffn_gguf_values<F>(value[k], a, b, c, params.groups, params.row + row,
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
// blocks of K * 128, as the shared scale, times the ANE's ANE_FFN_INT8_UNIT, of
// ane_ffn_weights and the ANE.
template <uint K, class F>
inline void ane_ffn_row_scales(device uchar *a, device uchar *b, device uchar *c, device half *row_scale,
                               device const float *sign, constant AneFfnWeightParams &params, uint tile,
                               uint simd_group, uint lane) {
  const uint row = tile * ANE_FFN_WEIGHT_ROWS + simd_group;
  float peak = 0.0f;
  for (uint block = 0; block < params.width; block += K * ANE_FFN_ROTATION_UNIT) {
    float value[K][4];
    for (uint k = 0; k < K; ++k)
      ane_ffn_gguf_values<F>(value[k], a, b, c, params.groups, params.row + row,
                             params.input + block + k * ANE_FFN_ROTATION_UNIT + lane * 4);
    ane_ffn_rotate_block<K>(value, lane, sign);
    for (uint k = 0; k < K; ++k)
      for (uint e = 0; e < 4; ++e) peak = max(peak, fabs(value[k][e]));
  }
  peak = simd_max(peak);
  if (lane == 0) row_scale[row] = half(max(peak, ANE_FFN_PEAK_FLOOR) / ANE_FFN_INT8_PEAK * ANE_FFN_INT8_UNIT);
}

// A projection's planes a, b, c: an image tensor's plane0, plane1 and meta
// (groups = inputs / 32). Each kernel runs the body of the dispatch's format.
template <uint K>
kernel void ane_ffn_weights_gguf(device uchar *a [[buffer(0)]], device uchar *b [[buffer(1)]],
    device uchar *c [[buffer(2)]], device const half *row_scale [[buffer(3)]], device char *output [[buffer(4)]],
    device half *scale [[buffer(5)]], device const float *sign [[buffer(6)]],
    constant AneFfnWeightParams &params [[buffer(7)]],
    uint2 tile [[threadgroup_position_in_grid]], uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  quant_format_switch(params.format, [&](auto format) {
    ane_ffn_weight_rows<K, decltype(format)>(a, b, c, row_scale, output, scale, sign, params, tile, simd_group, lane);
  });
}
template <uint K>
kernel void ane_ffn_row_scale_gguf(device uchar *a [[buffer(0)]], device uchar *b [[buffer(1)]],
    device uchar *c [[buffer(2)]], device half *row_scale [[buffer(3)]], device const float *sign [[buffer(4)]],
    constant AneFfnWeightParams &params [[buffer(5)]], uint tile [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  quant_format_switch(params.format, [&](auto format) {
    ane_ffn_row_scales<K, decltype(format)>(a, b, c, row_scale, sign, params, tile, simd_group, lane);
  });
}

using AneFfnWeightsKernel = void(device uchar *, device uchar *, device uchar *, device const half *, device char *,
                                 device half *, device const float *, constant AneFfnWeightParams &, uint2, uint, uint);
using AneFfnRowScaleKernel = void(device uchar *, device uchar *, device uchar *, device half *, device const float *,
                                  constant AneFfnWeightParams &, uint, uint, uint);
// _inputs: gate's and up's rows over the hidden inputs; _intermediate: down's
// rows over the ANE's intermediate channels.
constant constexpr uint kInputUnits = ANE_FFN_INPUT_BLOCK / ANE_FFN_ROTATION_UNIT;
constant constexpr uint kIntermediateUnits = ANE_FFN_INTERMEDIATE_BLOCK / ANE_FFN_ROTATION_UNIT;
template [[host_name("ane_ffn_weights_gguf_inputs")]] kernel AneFfnWeightsKernel ane_ffn_weights_gguf<kInputUnits>;
template [[host_name("ane_ffn_weights_gguf_intermediate")]] kernel AneFfnWeightsKernel
    ane_ffn_weights_gguf<kIntermediateUnits>;
template [[host_name("ane_ffn_row_scale_gguf_inputs")]] kernel AneFfnRowScaleKernel ane_ffn_row_scale_gguf<kInputUnits>;
template [[host_name("ane_ffn_row_scale_gguf_intermediate")]] kernel AneFfnRowScaleKernel
    ane_ffn_row_scale_gguf<kIntermediateUnits>;

// Whether `value` is an infinity or a NaN, by its exponent bits: fast math
// may fold isfinite() to true.
inline bool ane_ffn_not_finite(half value) { return (as_type<ushort>(value) & 0x7c00) == 0x7c00; }

// ANE_FFN_TILE x ANE_FFN_TILE tiles of rows x channels, ANE_FFN_TILE x
// ANE_FFN_TILE_ROWS threads. Each row's partial values take their token's
// intermediate scale (row `hidden` of the partial) and input scale in fp32. A
// partial value or token scale of a row of the chunk that is not finite sets
// `status`, which the host reads once the command completes. Each output
// rounds to bf16 once more than on the GPU alone, after the GPU part's
// residual epilogue: half a bf16 unit, which the split accepts rather than
// fusing the join into that epilogue.
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

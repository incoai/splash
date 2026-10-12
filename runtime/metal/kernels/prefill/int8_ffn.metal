#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/ane_ffn_rotation.h"

using namespace mpp::tensor_ops;

// The GPU's int8 prefill FFN (ops/Int8Ffn.cpp), the ANE FFN split's W8A8
// algorithm on the GPU's int8 matmul: inputs rotated in blocks of
// ANE_FFN_INPUT_BLOCK, the intermediate rows in blocks of
// ANE_FFN_INTERMEDIATE_BLOCK, each row then scaled per token to
// +-ANE_FFN_INT8_PEAK as int8; weights rotated alike and scaled per row to
// int8 from their planes. MPP matmul2d multiplies int8 by int8 into int32 over
// a whole row, and each output takes its token's and its weight row's scales
// in fp32.

// One threadgroup of INT8_FFN_QUANTIZE_THREADS per row of `width` values,
// rotated in blocks of K * 128, each simdgroup its blocks in turn: the int8
// row scaled by the peak of its rotated values, and that scale. Two passes
// over the row rotate it twice rather than hold it: the first finds the
// peak, the second writes the codes.
template <uint K>
kernel void int8_ffn_quantize(device const bfloat *input [[buffer(0)]], device const float *sign [[buffer(1)]],
                              device char *output [[buffer(2)]], device float *token_scale [[buffer(3)]],
                              constant Int8FfnQuantizeParams &params [[buffer(4)]],
                              uint row [[threadgroup_position_in_grid]],
                              uint simd_group [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simd_groups [[simdgroups_per_threadgroup]]) {
  constexpr uint block = K * ANE_FFN_ROTATION_UNIT;
  threadgroup float peaks[32];
  const uint blocks = params.width / block;
  device const bfloat *source = input + ulong(row) * params.width;
  const auto rotated = [&](uint b, thread float (&value)[K][4]) {
    for (uint k = 0; k < K; ++k)
      for (uint e = 0; e < 4; ++e) value[k][e] = float(source[b * block + k * ANE_FFN_ROTATION_UNIT + lane * 4 + e]);
    ane_ffn_rotate_block<K>(value, lane, sign);
  };
  float peak = 0.0f;
  for (uint b = simd_group; b < blocks; b += simd_groups) {
    float value[K][4];
    rotated(b, value);
    for (uint k = 0; k < K; ++k)
      for (uint e = 0; e < 4; ++e) peak = max(peak, fabs(value[k][e]));
  }
  peak = simd_max(peak);
  if (lane == 0) peaks[simd_group] = peak;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  peak = 0.0f;
  for (uint group = 0; group < simd_groups; ++group) peak = max(peak, peaks[group]);
  const float scale = max(peak, ANE_FFN_PEAK_FLOOR) / ANE_FFN_INT8_PEAK, inverse = 1.0f / scale;
  device char *codes = output + ulong(row) * params.width;
  for (uint b = simd_group; b < blocks; b += simd_groups) {
    float value[K][4];
    rotated(b, value);
    for (uint k = 0; k < K; ++k)
      *(device char4 *)(codes + b * block + k * ANE_FFN_ROTATION_UNIT + lane * 4) =
          char4(clamp(rint(float4(value[k][0], value[k][1], value[k][2], value[k][3]) * inverse),
                      -ANE_FFN_INT8_PEAK, ANE_FFN_INT8_PEAK));
  }
  if (simd_group == 0 && lane == 0) token_scale[row] = scale;
}

// The K * 128 values of weight row `row` at rotation block `at`, rotated.
template <uint K, class F>
inline void int8_ffn_weight_block(device uchar *a, device uchar *b, device uchar *c, uint groups, uint row, uint at,
                                  thread float (&value)[K][4], uint lane) {
  for (uint k = 0; k < K; ++k)
    ane_ffn_gguf_values<F>(value[k], a, b, c, groups, row, (at * K + k) * ANE_FFN_ROTATION_UNIT + lane * 4);
}

// One simdgroup per weight row, INT8_FFN_SIMDGROUPS rows per threadgroup: the
// row's scale, the peak of its values over `width` inputs rotated in blocks of
// K * 128, over ANE_FFN_INT8_PEAK.
template <uint K>
kernel void int8_ffn_row_scale(device uchar *a [[buffer(0)]], device uchar *b [[buffer(1)]],
                               device uchar *c [[buffer(2)]], device float *row_scale [[buffer(3)]],
                               device const float *sign [[buffer(4)]],
                               constant Int8FfnRowScaleParams &params [[buffer(5)]],
                               uint tile [[threadgroup_position_in_grid]],
                               uint simd_group [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tile * INT8_FFN_SIMDGROUPS + simd_group;
  quant_format_switch(params.format, [&](auto format) {
    float peak = 0.0f;
    for (uint at = 0; at < params.width / (K * ANE_FFN_ROTATION_UNIT); ++at) {
      float value[K][4];
      int8_ffn_weight_block<K, decltype(format)>(a, b, c, params.groups, row, at, value, lane);
      ane_ffn_rotate_block<K>(value, lane, sign);
      for (uint k = 0; k < K; ++k)
        for (uint e = 0; e < 4; ++e) peak = max(peak, fabs(value[k][e]));
    }
    peak = simd_max(peak);
    if (lane == 0) row_scale[row] = max(peak, ANE_FFN_PEAK_FLOOR) / ANE_FFN_INT8_PEAK;
  });
}
using Int8FfnRowScaleKernel = void(device uchar *, device uchar *, device uchar *, device float *,
                                   device const float *, constant Int8FfnRowScaleParams &, uint, uint, uint);
constant constexpr uint kInt8InputUnits = ANE_FFN_INPUT_BLOCK / ANE_FFN_ROTATION_UNIT;
constant constexpr uint kInt8IntermediateUnits = ANE_FFN_INTERMEDIATE_BLOCK / ANE_FFN_ROTATION_UNIT;
template [[host_name("int8_ffn_row_scale_inputs")]] kernel Int8FfnRowScaleKernel int8_ffn_row_scale<kInt8InputUnits>;
template [[host_name("int8_ffn_row_scale_intermediate")]] kernel Int8FfnRowScaleKernel
    int8_ffn_row_scale<kInt8IntermediateUnits>;

using Int8FfnQuantizeKernel = void(device const bfloat *, device const float *, device char *, device float *,
                                   constant Int8FfnQuantizeParams &, uint, uint, uint, uint);
template [[host_name("int8_ffn_quantize_inputs")]] kernel Int8FfnQuantizeKernel int8_ffn_quantize<kInt8InputUnits>;
template [[host_name("int8_ffn_quantize_intermediate")]] kernel Int8FfnQuantizeKernel
    int8_ffn_quantize<kInt8IntermediateUnits>;

// One simdgroup converts weight row `row` of `width` inputs: its values
// rotated in blocks of K * 128 as int8 under its scale, into `codes`, rows of
// `width` bytes. It loads each block's values while it rotates and stores the
// one before.
template <uint K, class F>
inline void int8_ffn_convert_row(device uchar *a, device uchar *b, device uchar *c, uint groups, uint width,
                                 uint row, float scale, device char *codes, device const float *sign, uint lane) {
  constexpr uint block = K * ANE_FFN_ROTATION_UNIT;
  const uint blocks = width / block;
  const float inverse = 1.0f / scale;
  float value[K][4], next[K][4];
  int8_ffn_weight_block<K, F>(a, b, c, groups, row, 0, value, lane);
  for (uint at = 0; at < blocks; ++at) {
    if (at + 1 < blocks) int8_ffn_weight_block<K, F>(a, b, c, groups, row, at + 1, next, lane);
    ane_ffn_rotate_block<K>(value, lane, sign);
    for (uint k = 0; k < K; ++k)
      *(device char4 *)(codes + ulong(row) * width + at * block + k * ANE_FFN_ROTATION_UNIT + lane * 4) =
          char4(clamp(rint(float4(value[k][0], value[k][1], value[k][2], value[k][3]) * inverse),
                      -ANE_FFN_INT8_PEAK, ANE_FFN_INT8_PEAK));
    for (uint k = 0; k < K; ++k)
      for (uint e = 0; e < 4; ++e) value[k][e] = next[k][e];
  }
}

// A threadgroup of a GEMM dispatch (Int8FfnGemmParams), on a grid of
// row_tiles + conversion_columns by column tiles: the GEMM's tiles in the
// first row_tiles columns, as Metal runs a 2D grid in an order that keeps a
// weight tile's rows in cache (a 1D grid of the same tiles ran 21% slower on
// an M5 Max), and conversion threadgroups in the rest of each row, which run
// beside them: on an M5 Max with Qwen3.8-27B a fifth of the conversions' time
// goes (8.4 ms for 2048 rows of gate and up converting down, against 8.0 and
// 0.5 apart; 4.9 for down converting the next gate and up, against 4.1 and
// 1.0). Spreading them among the tiles of each row lost the cached order. The
// conversion reads the planes four values per lane, as ane_ffn_weights does:
// sixteen per lane, or a group of 32 per thread through threadgroup memory,
// ran it two to three times slower. True for a conversion threadgroup, with
// `task` its index among them; false for a GEMM tile.
inline bool int8_ffn_converts(uint2 group, constant Int8FfnGemmParams &params, thread uint &task) {
  if (group.x < params.row_tiles) return false;
  task = group.y * params.conversion_columns + group.x - params.row_tiles;
  return true;
}

// The GEMMs: each simdgroup INT8_FFN_SIMDGROUP_ROWS rows of the int8 input
// against INT8_FFN_COLUMNS int8 weight rows, one matmul2d over every input,
// both read from device memory: 93 TOPS on an M5 Max, about what larger tiles
// reach (64 x 32 rows by columns 95, 64 x 64 64). A simdgroup past the chunk's
// rows returns, as does a conversion threadgroup past the conversions: the
// kernels hold no barrier. The input holds whole tiles of rows; rows past the
// chunk's are computed from whatever they hold and not stored.
constexpr constant auto kInt8Gemm =
    matmul2d_descriptor(INT8_FFN_SIMDGROUP_ROWS, INT8_FFN_COLUMNS, dynamic_length_v<int>, false, true, false);
inline auto int8_ffn_rows(device char *values, uint width, uint row, uint rows) {
  return tensor((device int8_t *)values + ulong(row) * width, dextents<int, 2>{int(width), int(rows)},
                array<int, 2>{1, int(width)});
}

// gate_up: h = silu(gate) * up of the chunk's rows into `intermediate`
// (bf16), and down's int8 rows over the intermediate inputs, which this
// layer's down then multiplies.
kernel void int8_ffn_gate_up(device char *input [[buffer(0)]], device const float *token_scale [[buffer(1)]],
                             device char *gate [[buffer(2)]], device char *up [[buffer(3)]],
                             device const float *gate_scale [[buffer(4)]], device const float *up_scale [[buffer(5)]],
                             device bfloat *intermediate [[buffer(6)]], device uchar *down_a [[buffer(7)]],
                             device uchar *down_b [[buffer(8)]], device uchar *down_c [[buffer(9)]],
                             device const float *down_scale [[buffer(10)]], device char *down [[buffer(11)]],
                             device const float *sign [[buffer(12)]],
                             constant Int8FfnGemmParams &params [[buffer(13)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint simd_group [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
  uint task;
  if (int8_ffn_converts(group, params, task)) {
    if (task >= params.conversions) return;
    const uint row = task * INT8_FFN_SIMDGROUPS + simd_group;
    quant_format_switch(params.formats[0], [&](auto format) {
      int8_ffn_convert_row<kInt8IntermediateUnits, decltype(format)>(down_a, down_b, down_c, params.groups[0],
                                                                     params.width, row, down_scale[row], down, sign,
                                                                     lane);
    });
    return;
  }
  const uint row0 = group.x * INT8_FFN_ROWS + simd_group * INT8_FFN_SIMDGROUP_ROWS, column0 = group.y * INT8_FFN_COLUMNS;
  if (row0 >= params.rows) return;
  matmul2d<kInt8Gemm, execution_simdgroups<1>> operation;
  auto a = int8_ffn_rows(input, params.inputs, row0, INT8_FFN_SIMDGROUP_ROWS);
  auto g = int8_ffn_rows(gate, params.inputs, column0, INT8_FFN_COLUMNS);
  auto u = int8_ffn_rows(up, params.inputs, column0, INT8_FFN_COLUMNS);
  auto gate_sum = operation.get_destination_cooperative_tensor<decltype(a), decltype(g), int>();
  auto up_sum = operation.get_destination_cooperative_tensor<decltype(a), decltype(u), int>();
  operation.run(a, g, gate_sum);
  operation.run(a, u, up_sum);
  for (ushort i = 0; i < gate_sum.get_capacity(); ++i) {
    if (!gate_sum.is_valid_element(i)) continue;
    const auto index = gate_sum.get_multidimensional_index(i);
    const uint row = row0 + index[1], column = column0 + index[0];
    if (row >= params.rows) continue;
    const float scale = token_scale[row];
    const float g_value = float(gate_sum[i]) * scale * gate_scale[column];
    const float u_value = float(up_sum[i]) * scale * up_scale[column];
    intermediate[ulong(row) * params.outputs + column] = bfloat(splash_silu(g_value) * u_value);
  }
}

// down: output = residual + down(h) of the chunk's rows (bf16), and the next
// layer's gate and up int8 rows over the hidden inputs (conversions over
// both, gate's first), which that layer's gate_up then multiplies. A dispatch
// of no tiles converts alone: a command's first layer's gate and up.
kernel void int8_ffn_down(device char *input [[buffer(0)]], device const float *token_scale [[buffer(1)]],
                          device char *down [[buffer(2)]], device const float *down_scale [[buffer(3)]],
                          device const bfloat *residual [[buffer(4)]], device bfloat *output [[buffer(5)]],
                          device uchar *gate_a [[buffer(6)]], device uchar *gate_b [[buffer(7)]],
                          device uchar *gate_c [[buffer(8)]], device uchar *up_a [[buffer(9)]],
                          device uchar *up_b [[buffer(10)]], device uchar *up_c [[buffer(11)]],
                          device const float *gate_scale [[buffer(12)]], device const float *up_scale [[buffer(13)]],
                          device char *gate [[buffer(14)]], device char *up [[buffer(15)]],
                          device const float *sign [[buffer(16)]],
                          constant Int8FfnGemmParams &params [[buffer(17)]],
                          uint2 group [[threadgroup_position_in_grid]],
                          uint simd_group [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  uint task;
  if (int8_ffn_converts(group, params, task)) {
    if (task >= params.conversions) return;
    uint row = task * INT8_FFN_SIMDGROUPS + simd_group;
    const bool second = row >= params.converted;
    if (second) row -= params.converted;
    quant_format_switch(params.formats[second], [&](auto format) {
      int8_ffn_convert_row<kInt8InputUnits, decltype(format)>(
          second ? up_a : gate_a, second ? up_b : gate_b, second ? up_c : gate_c, params.groups[second],
          params.width, row, (second ? up_scale : gate_scale)[row], second ? up : gate, sign, lane);
    });
    return;
  }
  const uint row0 = group.x * INT8_FFN_ROWS + simd_group * INT8_FFN_SIMDGROUP_ROWS, column0 = group.y * INT8_FFN_COLUMNS;
  if (row0 >= params.rows) return;
  matmul2d<kInt8Gemm, execution_simdgroups<1>> operation;
  auto a = int8_ffn_rows(input, params.inputs, row0, INT8_FFN_SIMDGROUP_ROWS);
  auto d = int8_ffn_rows(down, params.inputs, column0, INT8_FFN_COLUMNS);
  auto sum = operation.get_destination_cooperative_tensor<decltype(a), decltype(d), int>();
  operation.run(a, d, sum);
  for (ushort i = 0; i < sum.get_capacity(); ++i) {
    if (!sum.is_valid_element(i)) continue;
    const auto index = sum.get_multidimensional_index(i);
    const uint row = row0 + index[1], column = column0 + index[0];
    if (row >= params.rows) continue;
    const ulong at = ulong(row) * params.outputs + column;
    output[at] = bfloat(float(residual[at]) + float(sum[i]) * token_scale[row] * down_scale[column]);
  }
}

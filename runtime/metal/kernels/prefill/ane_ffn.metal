#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/quant_formats.h"

// The GPU side of a dense FFN split with the ANE (ops/AneFfn.cpp). The ANE
// multiplies int8 weights and activations after a block-diagonal Hadamard
// rotation v -> diag(sign) H v / sqrt(n) of blocks of n = K * 128 values
// (ANE_FFN_ROTATION_UNIT): orthogonal, it keeps the dot products of rows it
// rotates alike, and spreads their outliers before the per-token and per-row
// int8 scales. Weights are rotated here from each chunk's layer: an image
// tensor in any format of metal/abi/QuantFormat.h.

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

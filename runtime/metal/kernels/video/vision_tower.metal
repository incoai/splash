// The kernels of the video model's text encoder's vision tower:
// Qwen3-VL-32B's 27-block ViT with its merger and three deepstack mergers,
// fp16 weights and activations with fp32 accumulation, an fp32 residual
// stream and fp32 LayerNorm statistics. A port of the language models' bf16
// tower kernels (kernels/shared/vision.metal, Qwen3.5's tower of the same
// geometry) to fp16; the head-major attention output is gathered back to rows
// by the shared vision_attention_gather, a 16-bit copy.
//
// Token order everywhere is spatial-merge-block-major: each consecutive group
// of four patches is one 2x2 spatial unit, so the mergers read contiguous rows
// (the deepstack mergers normalize the 4 x 1152 values of a merged token as one
// row) and the merged token index is row-major over the merged grid.
//
// Also the text side of an image's rows: the vision rows' splice and deepstack
// adds into the bf16 hidden rows, and the Q/K norm + interleaved M-RoPE of a
// presentation with vision blocks (the text-only encode keeps
// text_encoder_qk_norm_rope, whose positions are the row index on every axis).

#include "metal/abi/Vision.h"
#include "metal/abi/video/TextEncoderABI.h"
#include "metal/abi/video/VisionTowerABI.h"
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

namespace h3vision {

constant ushort kHeads = 16;
constant ushort kHeadDim = 72;
constant ushort kQkDim = 80;  // Q/K padded to 80 dims for the matmul's K extent
constant ushort kHidden = 1152;
constant ushort kPatch = 16;
constant ushort kMerge = 2;
constant ushort kPatchDim = 1536;  // 3 channels x 2 temporal x 16 x 16
constant ushort kGridSide = 48;    // learned position table 48 x 48
constant float kRopeTheta = 10000.0f;

inline uint2 patch_position(uint token, uint grid_width) {
  uint blocks_w = grid_width / kMerge;
  uint block = token / (kMerge * kMerge);
  uint inner = token % (kMerge * kMerge);
  return uint2((block / blocks_w) * kMerge + inner / kMerge, (block % blocks_w) * kMerge + inner % kMerge);
}

inline void bilinear_taps(uint index, uint size, thread uint *tap, thread float *weight) {
  float source = size > 1 ? float(index) * float(kGridSide - 1) / float(size - 1) : 0.0f;
  float lower = floor(source);
  for (uint offset = 0; offset < 2; ++offset) {
    float position = lower + float(offset);
    tap[offset] = uint(clamp(position, 0.0f, float(kGridSide - 1)));
    weight[offset] = max(1.0f - fabs(source - position), 0.0f);
  }
}

// Abramowitz & Stegun 7.1.26 (max error ~1.5e-7).
inline float erf_approx(float x) {
  float sign = x < 0.0f ? -1.0f : 1.0f;
  x = fabs(x);
  float t = 1.0f / (1.0f + 0.3275911f * x);
  float polynomial =
      t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
  return sign * (1.0f - polynomial * precise::exp(-x * x));
}

enum Activation { None, GeluTanh, GeluErf };

template <ushort TileM, ushort TileN, Activation Act, bool AddResidual, typename Output>
inline void gemm_tile(device half *input, device half *weights, device half *bias, device Output *output,
                      device Output *residual, uint output_size, uint input_size, uint output_origin,
                      uint row_origin) {
  device half *rows = input + ulong(row_origin) * input_size;
  device half *tile_weights = weights + ulong(output_origin) * input_size;
  auto a = tensor(rows, dextents<int, 2>{int(input_size), TileM}, array<int, 2>{1, int(input_size)});
  auto b = tensor(tile_weights, dextents<int, 2>{int(input_size), TileN}, array<int, 2>{1, int(input_size)});
  auto c = tensor(output + ulong(row_origin) * output_size, dextents<int, 2>{int(output_size), TileM},
                  array<int, 2>{1, int(output_size)});
  constexpr ushort BK = 128;
  constexpr auto descriptor =
      matmul2d_descriptor(TileM, TileN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<8>> operation;
  auto a0 = a.template slice<BK, TileM>(0, 0);
  auto b0 = b.template slice<BK, TileN>(0, 0);
  auto accumulated = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i) accumulated[i] = 0.0f;
  for (uint chunk = 0; chunk < input_size / BK; ++chunk) {
    auto a_slice = a.template slice<BK, TileM>(chunk * BK, 0);
    auto b_slice = b.template slice<BK, TileN>(chunk * BK, 0);
    operation.run(a_slice, b_slice, accumulated);
    threadgroup_barrier(mem_flags::mem_none);
  }
  auto converted = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), Output>();
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
    auto index = accumulated.get_multidimensional_index(i);
    float value = accumulated[i] + float(bias[output_origin + index[0]]);
    if constexpr (Act == GeluTanh) {
      value = 0.5f * value * (1.0f + precise::tanh(0.7978845608028654f * (value + 0.044715f * value * value * value)));
    } else if constexpr (Act == GeluErf) {
      value = 0.5f * value * (1.0f + erf_approx(value * 0.7071067811865476f));
    }
    if constexpr (AddResidual) {
      value += float(residual[ulong(row_origin + index[1]) * output_size + output_origin + index[0]]);
    }
    converted[i] = Output(value);
  }
  converted.store(c.template slice<TileN, TileM>(output_origin, 0));
}

} // namespace h3vision

using namespace h3vision;

// uint8 RGB pixels (H, W, 3) -> fp16 patch rows (tokens, 1536), block-major, features (channel, temporal, row,
// col), the single frame duplicated over the temporal patch. (x / 255 - 0.5) / 0.5 is rounded to bf16 first, as the
// reference pipeline casts pixel_values to the conditioner's bf16; every such value is exact in fp16.
kernel void h3_vision_patchify(device const uchar *pixels [[buffer(0)]], device half *patches [[buffer(1)]],
                               constant VisionGridParams &params [[buffer(2)]],
                               uint token [[threadgroup_position_in_grid]],
                               uint thread_index [[thread_index_in_threadgroup]]) {
  uint2 position = patch_position(token, params.grid_width);
  uint image_width = params.grid_width * kPatch;
  device half *row = patches + ulong(token) * kPatchDim;
  for (uint feature = thread_index; feature < kPatchDim; feature += 256) {
    uint channel = feature / (2 * kPatch * kPatch);
    uint within = feature % (kPatch * kPatch);
    uint patch_row = within / kPatch;
    uint patch_col = within % kPatch;
    ulong pixel =
        (ulong(position.x * kPatch + patch_row) * image_width + position.y * kPatch + patch_col) * 3 + channel;
    float value = (float(pixels[pixel]) * (1.0f / 255.0f) - 0.5f) / 0.5f;
    row[feature] = half(float(bfloat(value)));
  }
}

// A video block's pair of frames (Qwen3-VL's video processor: two consecutive sampled frames over the temporal
// patch of 2): as h3_vision_patchify, temporal index 0 from `first` and 1 from `second`.
kernel void h3_vision_patchify_pair(device const uchar *first [[buffer(0)]], device const uchar *second [[buffer(1)]],
                                    device half *patches [[buffer(2)]], constant VisionGridParams &params [[buffer(3)]],
                                    uint token [[threadgroup_position_in_grid]],
                                    uint thread_index [[thread_index_in_threadgroup]]) {
  uint2 position = patch_position(token, params.grid_width);
  uint image_width = params.grid_width * kPatch;
  device half *row = patches + ulong(token) * kPatchDim;
  for (uint feature = thread_index; feature < kPatchDim; feature += 256) {
    uint channel = feature / (2 * kPatch * kPatch);
    uint temporal = (feature / (kPatch * kPatch)) % 2;
    uint within = feature % (kPatch * kPatch);
    uint patch_row = within / kPatch;
    uint patch_col = within % kPatch;
    ulong pixel =
        (ulong(position.x * kPatch + patch_row) * image_width + position.y * kPatch + patch_col) * 3 + channel;
    float value = (float((temporal ? second : first)[pixel]) * (1.0f / 255.0f) - 0.5f) / 0.5f;
    row[feature] = half(float(bfloat(value)));
  }
}

// The bilinear (align-corners) resample of the 48 x 48 position table into the fp32 residual rows the patch
// embedding adds to, and the 2D rope tables (18 frequencies each for row and column, both rotation halves).
kernel void h3_vision_prepare_positions(device const half *table [[buffer(0)]], device float *hidden [[buffer(1)]],
                                        device float *rope_cos [[buffer(2)]], device float *rope_sin [[buffer(3)]],
                                        constant VisionGridParams &params [[buffer(4)]],
                                        uint token [[threadgroup_position_in_grid]],
                                        uint thread_index [[thread_index_in_threadgroup]]) {
  device float *out = hidden + ulong(token) * kHidden;
  if (token >= params.grid_height * params.grid_width) {
    for (uint dim = thread_index; dim < kHidden; dim += 256) out[dim] = 0.0f;
    return;
  }
  uint2 position = patch_position(token, params.grid_width);
  uint row_taps[2], col_taps[2];
  float row_weights[2], col_weights[2];
  bilinear_taps(position.x, params.grid_height, row_taps, row_weights);
  bilinear_taps(position.y, params.grid_width, col_taps, col_weights);
  for (uint dim = thread_index; dim < kHidden; dim += 256) {
    float blended = 0.0f;
    for (uint h = 0; h < 2; ++h)
      for (uint w = 0; w < 2; ++w)
        blended += row_weights[h] * col_weights[w] *
                   float(table[(ulong(row_taps[h]) * kGridSide + col_taps[w]) * kHidden + dim]);
    out[dim] = blended;
  }
  if (thread_index < kHeadDim) {
    constexpr uint frequencies = kHeadDim / 4;  // 18
    uint frequency = thread_index % frequencies;
    bool column = (thread_index / frequencies) & 1;
    float inverse = pow(kRopeTheta, -float(2 * frequency) / float(kHeadDim / 2));
    float angle = float(column ? position.y : position.x) * inverse;
    rope_cos[ulong(token) * kHeadDim + thread_index] = cos(angle);
    rope_sin[ulong(token) * kHeadDim + thread_index] = sin(angle);
  }
}

#define H3_VISION_GEMM(name, tile_m, tile_n, act, add_residual, output_type)                                   \
  kernel void name(device half *input [[buffer(0)]], device half *weights [[buffer(1)]],                       \
                   device half *bias [[buffer(2)]], device output_type *output [[buffer(3)]],                  \
                   device output_type *residual [[buffer(4)]], constant VisionGemmParams &params [[buffer(5)]], \
                   uint2 group [[threadgroup_position_in_grid]]) {                                             \
    gemm_tile<tile_m, tile_n, act, add_residual, output_type>(input, weights, bias, output, residual,           \
                                                              params.output_size, params.input_size,            \
                                                              group.y * tile_n, group.x * tile_m);              \
  }

H3_VISION_GEMM(h3_vision_gemm_m64n128, 64, 128, None, false, half)
H3_VISION_GEMM(h3_vision_gemm_m64n128_residual, 64, 128, None, true, float)
H3_VISION_GEMM(h3_vision_gemm_m64n128_gelu_tanh, 64, 128, GeluTanh, false, half)
H3_VISION_GEMM(h3_vision_gemm_m32n256_gelu_erf, 32, 256, GeluErf, false, half)
// The mergers' language-space rows stay fp32 until the text encoder rounds them into its bf16 rows.
H3_VISION_GEMM(h3_vision_gemm_m32n256_f32, 32, 256, None, false, float)

// LayerNorm with bias of fp32 rows of `width` values (1152 a patch, 4608 a deepstack merger's merged token), fp32
// statistics, fp16 output. One threadgroup of 256 per row.
kernel void h3_vision_layer_norm(device const float *input [[buffer(0)]], device const half *weight [[buffer(1)]],
                                 device const half *bias [[buffer(2)]], device half *output [[buffer(3)]],
                                 constant VisionNormParams &params [[buffer(4)]],
                                 uint group [[threadgroup_position_in_grid]],
                                 uint thread_index [[thread_index_in_threadgroup]],
                                 uint simd_lane [[thread_index_in_simdgroup]],
                                 uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float reductions[8];
  device const float *row = input + ulong(group) * params.width;
  device half *out = output + ulong(group) * params.width;
  float sum = 0.0f;
  for (uint column = thread_index; column < params.width; column += 256) sum += row[column];
  sum = simd_sum(sum);
  if (simd_lane == 0) reductions[simd_group] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
#pragma unroll
  for (ushort i = 0; i < 8; ++i) total += reductions[i];
  float mean = total / float(params.width);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float squares = 0.0f;
  for (uint column = thread_index; column < params.width; column += 256) {
    float centered = row[column] - mean;
    squares += centered * centered;
  }
  squares = simd_sum(squares);
  if (simd_lane == 0) reductions[simd_group] = squares;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float variance = 0.0f;
#pragma unroll
  for (ushort i = 0; i < 8; ++i) variance += reductions[i];
  variance /= float(params.width);
  float inverse = rsqrt(variance + SPLASH_VISION_NORM_EPSILON);
  for (uint column = thread_index; column < params.width; column += 256) {
    float normalized = (row[column] - mean) * inverse;
    out[column] = half(normalized * float(weight[column]) + float(bias[column]));
  }
}

// 2D rope on Q and K, scattered head-major: Q, K (heads, padded_tokens, 80), V (heads, key_tile, 128, 72); padded
// tokens zero.
kernel void h3_vision_qkv_prepare(device const half *qkv [[buffer(0)]], device const float *rope_cos [[buffer(1)]],
                                  device const float *rope_sin [[buffer(2)]], device half *queries [[buffer(3)]],
                                  device half *keys [[buffer(4)]], device half *values [[buffer(5)]],
                                  constant VisionQkvParams &params [[buffer(6)]],
                                  uint token [[threadgroup_position_in_grid]],
                                  uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr ushort half_dim = kHeadDim / 2;
  const bool valid = token < params.tokens;
  device const half *row = qkv + ulong(token) * 3 * kHeads * kHeadDim;
  device const float *cos_row = rope_cos + ulong(token) * kHeadDim;
  device const float *sin_row = rope_sin + ulong(token) * kHeadDim;
  uint key_tile = token / 128, within = token % 128;
  uint tiles = params.padded_tokens / 128;
  for (uint index = thread_index; index < uint(kHeads) * kQkDim; index += 256) {
    uint head = index / kQkDim;
    uint dim = index % kQkDim;
    ulong plane = (ulong(head) * params.padded_tokens + token) * kQkDim + dim;
    if (!valid || dim >= kHeadDim) {
      queries[plane] = half(0.0f);
      keys[plane] = half(0.0f);
      if (dim < kHeadDim) values[(((ulong(head) * tiles + key_tile) * 128) + within) * kHeadDim + dim] = half(0.0f);
      continue;
    }
    uint base = head * kHeadDim + dim;
    uint rotated = head * kHeadDim + (dim < half_dim ? dim + half_dim : dim - half_dim);
    float sign = dim < half_dim ? -1.0f : 1.0f;
    float cos_value = cos_row[dim];
    float sin_value = sin_row[dim];
    queries[plane] = half(float(row[base]) * cos_value + sign * float(row[rotated]) * sin_value);
    keys[plane] = half(float(row[kHidden + base]) * cos_value + sign * float(row[kHidden + rotated]) * sin_value);
    values[(((ulong(head) * tiles + key_tile) * 128) + within) * kHeadDim + dim] = row[2 * kHidden + base];
  }
}

// Non-causal flash attention over the image's tokens (Qwen3-VL attends within one frame), the shared tower's
// kernel with fp16 Q, K, V and probabilities: fp32 scores, softmax statistics and PV accumulation.
kernel void h3_vision_attention(device half *queries [[buffer(0)]], device half *keys [[buffer(1)]],
                                device half *values [[buffer(2)]], device half *output [[buffer(3)]],
                                constant VisionAttentionParams &params [[buffer(4)]],
                                uint2 group [[threadgroup_position_in_grid]],
                                uint thread_index [[thread_index_in_threadgroup]]) {
  constexpr ushort M = SPLASH_VISION_ATTENTION_ROWS, N = 128, D = kHeadDim;
  constexpr ushort DQ = kQkDim;
  constexpr ushort Simdgroups = SPLASH_VISION_ATTENTION_THREADS / 32;
  constexpr ushort LanesPerRow = Simdgroups * 32 / M;
  constexpr ushort Runs = N / (LanesPerRow * 4);
  static_assert(Simdgroups * 32 == SPLASH_VISION_ATTENTION_THREADS && LanesPerRow * M == Simdgroups * 32,
                "whole simdgroups whose lanes own whole rows");
  static_assert(LanesPerRow == 4, "two xor shuffles reduce a row");
  alignas(16) threadgroup float score_storage[M * N];
  alignas(16) threadgroup half probabilities[M * N];
  threadgroup float row_sum[M];
  threadgroup float previous_scale[M];
  const ushort row = thread_index / LanesPerRow;
  const ushort part = thread_index % LanesPerRow;
  float running_max = -INFINITY;
  float running_sum = 0.0f;

  uint head = group.y;
  uint query_start = group.x * M;
  ulong qk_plane = ulong(head) * params.padded_tokens * DQ;
  device half *q = queries + qk_plane + ulong(query_start) * DQ;
  device half *k = keys + qk_plane;
  device half *v = values + ulong(head) * D * params.padded_tokens;
  device half *out = output + ulong(head) * params.padded_tokens * D + ulong(query_start) * D;
  uint context = params.tokens;
  uint tiles = (context + N - 1) / N;

  auto qt = tensor(q, dextents<int, 2>{DQ, M}, array<int, 2>{1, DQ});
  constexpr auto qk_descriptor = matmul2d_descriptor(M, N, DQ, false, true, false);
  matmul2d<qk_descriptor, execution_simdgroups<Simdgroups>> qk;
  auto q0 = qt.slice<DQ, M>(0, 0);
  auto pt = tensor(probabilities, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto p0 = pt.slice<N, M>(0, 0);
  auto st = tensor(score_storage, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto s0 = st.slice<N, M>(0, 0);
  auto first_v = tensor(v, dextents<int, 2>{D, N}, array<int, 2>{1, D});
  auto first_v0 = first_v.slice<D, N>(0, 0);
  constexpr auto pv_descriptor =
      matmul2d_descriptor(M, D, N, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv_descriptor, execution_simdgroups<Simdgroups>> pv;
  auto running = pv.template get_destination_cooperative_tensor<decltype(p0), decltype(first_v0), float>();
  const bool full = uint(running.get_capacity()) * (Simdgroups * 32u) == uint(M) * D;
  for (ushort i = 0; i < running.get_capacity(); ++i)
    if (full || running.is_valid_element(i)) running[i] = 0.0f;
  threadgroup const float4 *row_scores = reinterpret_cast<threadgroup const float4 *>(score_storage + row * N);
  threadgroup half4 *row_probabilities = reinterpret_cast<threadgroup half4 *>(probabilities + row * N);
  for (uint tile = 0; tile < tiles; ++tile) {
    device half *tile_k = k + ulong(tile) * N * DQ;
    auto kt = tensor(tile_k, dextents<int, 2>{DQ, N}, array<int, 2>{1, DQ});
    auto k0 = kt.slice<DQ, N>(0, 0);
    auto scores = qk.template get_destination_cooperative_tensor<decltype(q0), decltype(k0), float>();
    qk.run(q0, k0, scores);
    scores.store(s0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int limit = int(context) - int(tile * N);
    float logits[Runs * 4];
    float tile_max = -INFINITY;
    for (ushort r = 0; r < Runs; ++r) {
      const ushort run = r * LanesPerRow + part;
      const float4 score = row_scores[run] * params.scale;
      for (ushort j = 0; j < 4; ++j) {
        logits[r * 4 + j] = int(run * 4 + j) < limit ? score[j] : -INFINITY;
        tile_max = max(tile_max, logits[r * 4 + j]);
      }
    }
    tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
    tile_max = max(tile_max, simd_shuffle_xor(tile_max, 2));
    const float next_max = max(running_max, tile_max);
    const float scale = fast::exp(running_max - next_max);
    float tile_sum = 0.0f;
    for (ushort r = 0; r < Runs; ++r) {
      float4 exponents;
      for (ushort j = 0; j < 4; ++j) {
        exponents[j] = fast::exp(logits[r * 4 + j] - next_max);
        tile_sum += exponents[j];
      }
      row_probabilities[r * LanesPerRow + part] = half4(exponents);
    }
    tile_sum += simd_shuffle_xor(tile_sum, 1);
    tile_sum += simd_shuffle_xor(tile_sum, 2);
    running_sum = running_sum * scale + tile_sum;
    running_max = next_max;
    if (part == 0) {
      previous_scale[row] = scale;
      row_sum[row] = running_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort i = 0; i < running.get_capacity(); ++i) {
      if (!full && !running.is_valid_element(i)) continue;
      auto index = running.get_multidimensional_index(i);
      running[i] *= previous_scale[index[1]];
    }
    device half *tile_v = v + ulong(tile) * N * D;
    auto vt = tensor(tile_v, dextents<int, 2>{D, N}, array<int, 2>{1, D});
    auto v0 = vt.slice<D, N>(0, 0);
    pv.run(p0, v0, running);
  }
  auto ot = tensor(out, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto converted = pv.template get_destination_cooperative_tensor<decltype(p0), decltype(first_v0), half>();
  for (ushort i = 0; i < running.get_capacity(); ++i) {
    if (!full && !running.is_valid_element(i)) continue;
    auto index = running.get_multidimensional_index(i);
    converted[i] = half(running[i] / row_sum[index[1]]);
  }
  converted.store(ot.slice<D, M>(0, 0));
}

// ---------------------------------------------------------------- text side

// hidden[index[r]] = bf16(features[r]) (mode 0, the merger's rows into the image pad rows) or
// bf16(hidden[index[r]] + features[r]) (mode 1, a deepstack merger's rows after decoder layers 0 / 1 / 2).
kernel void text_encoder_vision_rows(device bfloat *hidden [[buffer(0)]], device const uint *index [[buffer(1)]],
                                     device const float *features [[buffer(2)]],
                                     constant TextEncoderVisionRowsParams &params [[buffer(3)]],
                                     uint r [[threadgroup_position_in_grid]],
                                     uint thread_index [[thread_index_in_threadgroup]]) {
  device bfloat *row = hidden + ulong(index[r]) * params.width;
  device const float *feature = features + ulong(r) * params.width;
  for (uint column = thread_index; column < params.width; column += 256) {
    const float value = params.mode ? float(row[column]) + feature[column] : feature[column];
    row[column] = bfloat(value);
  }
}

// text_encoder_qk_norm_rope with Qwen3-VL's interleaved M-RoPE: positions [rows][4] (t, h, w, unused) as the
// conditioner's get_rope_index gives them (text rows on all three axes; an image's pad rows at (s, s + row,
// s + column) of its merged grid). One simdgroup per (row, head), four dimensions a lane.
kernel void text_encoder_qk_norm_mrope(device const bfloat *packed [[buffer(0)]],
                                       device const bfloat *query_norm [[buffer(1)]],
                                       device const bfloat *key_norm [[buffer(2)]],
                                       device const float *inverse_frequencies [[buffer(3)]],
                                       device bfloat *queries [[buffer(4)]], device bfloat *keys [[buffer(5)]],
                                       device bfloat *values [[buffer(6)]],
                                       device const uint4 *positions [[buffer(7)]],
                                       constant TextEncoderHeadParams &params [[buffer(8)]],
                                       uint2 group [[threadgroup_position_in_grid]],
                                       uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint kHead = SPLASH_TEXT_ENCODER_HEAD_DIMENSION;
  const uint row = group.x;
  const uint head = group.y;
  const uint heads = params.query_heads + 2 * params.kv_heads;
  const float4 x = float4(reinterpret_cast<device const bfloat4 *>(packed + (ulong(row) * heads + head) * kHead)[lane]);
  if (head >= params.query_heads + params.kv_heads) {
    const uint value_head = head - params.query_heads - params.kv_heads;
    reinterpret_cast<device bfloat4 *>(values + (ulong(value_head) * params.rows + row) * kHead)[lane] = bfloat4(x);
    return;
  }
  const bool query = head < params.query_heads;
  const float4 weight = float4(reinterpret_cast<device const bfloat4 *>(query ? query_norm : key_norm)[lane]);
  const float inverse_rms = rsqrt(simd_sum(dot(x, x)) / float(kHead) + params.value);
  const float4 normalized = float4(bfloat4(x * inverse_rms * weight));
  const float4 partner = simd_shuffle_xor(normalized, 16);
  const uint4 position = positions[row];
  const float4 inverse = reinterpret_cast<device const float4 *>(inverse_frequencies)[lane & 15];
  float4 angle;
  for (uint j = 0; j < 4; ++j) {
    const uint f = (lane & 15) * 4 + j;
    const uint axis = (f % 3 == 1 && f < 3 * SPLASH_TEXT_ENCODER_MROPE_HEIGHT)  ? 1u
                      : (f % 3 == 2 && f < 3 * SPLASH_TEXT_ENCODER_MROPE_WIDTH) ? 2u
                                                                               : 0u;
    angle[j] = float(position[axis]) * inverse[j];
  }
  const float4 cosine = cos(angle);
  const float4 sine = sin(angle);
  const float4 rotated = lane < 16 ? normalized * cosine - partner * sine : normalized * cosine + partner * sine;
  device bfloat *destination = query ? queries + (ulong(head) * params.rows + row) * kHead
                                     : keys + (ulong(head - params.query_heads) * params.rows + row) * kHead;
  reinterpret_cast<device bfloat4 *>(destination)[lane] = bfloat4(rotated);
}

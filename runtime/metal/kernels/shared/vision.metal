// Vision encoder kernels for the Qwen3.5 vision tower: pixel patchify,
// position/rope preparation, dense bf16 GEMMs with fused epilogues,
// LayerNorm, 2D-rope QKV preparation and non-causal flash attention; and the
// kernels Nemotron 3.5 Bolt's RADIO tower runs in their place (_radio, _d80).
//
// Conventions match the text-model kernels: 256 threads (8 simdgroups) per
// threadgroup (128 for attention), MPP cooperative-tensor matmuls with fp32
// accumulation and hand-written epilogues, grid {row_tile, output_tile}.
// Attention runs on the native head dimension (72); token counts are padded
// to the 128-key tile so K/V tiles read in full, with padded keys zeroed and
// masked by the softmax.
//
// Token order everywhere is spatial-merge-block-major: each consecutive group
// of four patches is one 2x2 spatial unit, so the merger reads contiguous
// rows and the merged token index is row-major over the merged grid.

#include "metal/abi/Vision.h"
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

constant ushort kVisionHeads = 16;
constant ushort kVisionHeadDim = 72;
// Q/K are stored padded to 80 dims (matmul K extents must be multiples of
// 16); the zeroed tail contributes nothing to the dot products.
constant ushort kVisionQkDim = 80;
constant ushort kVisionHidden = 1152;
constant ushort kVisionPatch = 16;
constant ushort kVisionMerge = 2;
constant ushort kVisionPatchDim = 1536;  // 3 channels x 2 temporal x 16 x 16
constant ushort kVisionGridSide = 48;    // learned position table is 48 x 48
constant float kVisionRopeTheta = 10000.0f;

// RADIO (C-RADIO ViT-H/16): 1280 wide, 16 heads of 80, single-frame patches
// normalized by the CLIP mean and std, a learned 128x128 position table and
// ten prefix tokens (3 cls, 7 registers) that take no position. No rope.
constant ushort kRadioHidden = 1280;
constant ushort kRadioHeadDim = 80;
constant ushort kRadioPatchDim = 768;  // 3 channels x 16 x 16
constant ushort kRadioGridSide = 128;
constant ushort kRadioPrefix = 10;
constant float kRadioMean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
constant float kRadioStd[3] = {0.26862954f, 0.26130258f, 0.27577711f};

// Block-major (row, col) of one patch token.
inline uint2 vision_patch_position(uint token, uint grid_width) {
    uint blocks_w = grid_width / kVisionMerge;
    uint block = token / (kVisionMerge * kVisionMerge);
    uint inner = token % (kVisionMerge * kVisionMerge);
    return uint2((block / blocks_w) * kVisionMerge + inner / kVisionMerge,
                 (block % blocks_w) * kVisionMerge + inner % kVisionMerge);
}

// Patchify: uint8 RGB pixels (H, W, 3) -> bf16 patch rows (tokens, 1536) in
// block-major token order with per-patch feature order (channel, temporal,
// row, col). Normalization is mean 0.5 / std 0.5, and the single frame is
// duplicated across the temporal patch dimension.

kernel void vision_patchify(
    device const uchar *pixels [[buffer(0)]],
    device bfloat *patches [[buffer(1)]],
    constant VisionGridParams &params [[buffer(2)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    uint2 position = vision_patch_position(token, params.grid_width);
    uint image_width = params.grid_width * kVisionPatch;
    device bfloat *row = patches + ulong(token) * kVisionPatchDim;
    for (uint feature = thread_index; feature < kVisionPatchDim;
         feature += 256) {
        uint channel = feature / (2 * kVisionPatch * kVisionPatch);
        uint within = feature % (kVisionPatch * kVisionPatch);
        uint patch_row = within / kVisionPatch;
        uint patch_col = within % kVisionPatch;
        ulong pixel = (ulong(position.x * kVisionPatch + patch_row) *
                           image_width +
                       position.y * kVisionPatch + patch_col) * 3 + channel;
        row[feature] = bfloat(float(pixels[pixel]) * (1.0f / 127.5f) - 1.0f);
    }
}

// RADIO's patchify: `frames` frames, one after the other in pixels, into
// rows of per-patch feature order (frame, channel, row, col), each channel
// normalized by its CLIP mean and std. An image is one frame; a video's
// tubelet two, the order of its video patch embedding (v.patch_embd.weight.1).
template <uint frames>
kernel void vision_patchify_radio(
    device const uchar *pixels [[buffer(0)]],
    device bfloat *patches [[buffer(1)]],
    constant VisionGridParams &params [[buffer(2)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    uint2 position = vision_patch_position(token, params.grid_width);
    uint image_width = params.grid_width * kVisionPatch;
    ulong frame_bytes = ulong(params.grid_height) * kVisionPatch * image_width * 3;
    device bfloat *row = patches + ulong(token) * (frames * kRadioPatchDim);
    for (uint feature = thread_index; feature < frames * kRadioPatchDim;
         feature += 256) {
        uint frame = feature / kRadioPatchDim;
        uint channel = feature % kRadioPatchDim / (kVisionPatch * kVisionPatch);
        uint within = feature % (kVisionPatch * kVisionPatch);
        ulong pixel = frame * frame_bytes +
                      (ulong(position.x * kVisionPatch + within / kVisionPatch) *
                           image_width +
                       position.y * kVisionPatch + within % kVisionPatch) * 3 + channel;
        row[feature] = bfloat((float(pixels[pixel]) * (1.0f / 255.0f) -
                               kRadioMean[channel]) / kRadioStd[channel]);
    }
}

template [[host_name("vision_patchify_radio")]] kernel void
vision_patchify_radio<1>(device const uchar *, device bfloat *, constant VisionGridParams &, uint, uint);
template [[host_name("vision_patchify_radio_video")]] kernel void
vision_patchify_radio<2>(device const uchar *, device bfloat *, constant VisionGridParams &, uint, uint);

// Grid-derived inputs: the bilinear (align-corners) resample of the learned
// 48x48 position table into the fp32 residual rows the patch embedding adds
// to, and the 2D rope tables with 18 frequencies each for the row and column
// coordinate, duplicated across both rotation halves.

inline void vision_bilinear_taps(uint index, uint size, thread uint *tap,
                                 thread float *weight) {
    float source = size > 1
        ? float(index) * float(kVisionGridSide - 1) / float(size - 1)
        : 0.0f;
    float lower = floor(source);
    for (uint offset = 0; offset < 2; ++offset) {
        float position = lower + float(offset);
        tap[offset] = uint(clamp(position, 0.0f, float(kVisionGridSide - 1)));
        weight[offset] = max(1.0f - fabs(source - position), 0.0f);
    }
}

kernel void vision_prepare_positions(
    device const bfloat *table [[buffer(0)]],
    device float *hidden [[buffer(1)]],
    device float *rope_cos [[buffer(2)]],
    device float *rope_sin [[buffer(3)]],
    constant VisionGridParams &params [[buffer(4)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    device float *out = hidden + ulong(token) * kVisionHidden;
    // The padding rows of the last GEMM tile start each image from zero: the
    // residual GEMMs add to these rows in place, and the constructor's memset
    // clears them only before the first image.
    if (token >= params.grid_height * params.grid_width) {
        for (uint dim = thread_index; dim < kVisionHidden; dim += 256)
            out[dim] = 0.0f;
        return;
    }
    uint2 position = vision_patch_position(token, params.grid_width);
    uint row_taps[2], col_taps[2];
    float row_weights[2], col_weights[2];
    vision_bilinear_taps(position.x, params.grid_height, row_taps, row_weights);
    vision_bilinear_taps(position.y, params.grid_width, col_taps, col_weights);

    for (uint dim = thread_index; dim < kVisionHidden; dim += 256) {
        float blended = 0.0f;
        for (uint h = 0; h < 2; ++h) {
            for (uint w = 0; w < 2; ++w) {
                blended += row_weights[h] * col_weights[w] *
                    float(table[(ulong(row_taps[h]) * kVisionGridSide +
                                 col_taps[w]) * kVisionHidden + dim]);
            }
        }
        out[dim] = blended;
    }

    if (thread_index < kVisionHeadDim) {
        constexpr uint frequencies = kVisionHeadDim / 4;  // 18
        uint frequency = thread_index % frequencies;
        bool column = (thread_index / frequencies) & 1;
        float inverse = pow(kVisionRopeTheta,
                            -float(2 * frequency) / float(kVisionHeadDim / 2));
        float angle = float(column ? position.y : position.x) * inverse;
        rope_cos[ulong(token) * kVisionHeadDim + thread_index] = cos(angle);
        rope_sin[ulong(token) * kVisionHeadDim + thread_index] = sin(angle);
    }
}

// RADIO's positions, as HF's cropped position embedding computes them: the
// 128x128 table resampled bilinearly (torch's align_corners=False, no
// antialias) to the square of the grid's longer side, cropped to the grid's
// top-left rows x cols. Grid {patches}.

inline void radio_bilinear_taps(uint index, uint size, thread uint *tap,
                                thread float *weight) {
    float source = max((float(index) + 0.5f) *
                           (float(kRadioGridSide) / float(size)) - 0.5f,
                       0.0f);
    uint lower = min(uint(source), uint(kRadioGridSide - 1));
    tap[0] = lower;
    tap[1] = min(lower + 1, uint(kRadioGridSide - 1));
    weight[1] = source - float(lower);
    weight[0] = 1.0f - weight[1];
}

kernel void vision_prepare_positions_radio(
    device const bfloat *table [[buffer(0)]],
    device float *hidden [[buffer(1)]],
    constant VisionGridParams &params [[buffer(2)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    uint2 position = vision_patch_position(token, params.grid_width);
    uint side = max(params.grid_height, params.grid_width);
    uint row_taps[2], col_taps[2];
    float row_weights[2], col_weights[2];
    radio_bilinear_taps(position.x, side, row_taps, row_weights);
    radio_bilinear_taps(position.y, side, col_taps, col_weights);
    device float *out = hidden + ulong(token) * kRadioHidden;
    for (uint dim = thread_index; dim < kRadioHidden; dim += 256) {
        float blended = 0.0f;
        for (uint h = 0; h < 2; ++h) {
            for (uint w = 0; w < 2; ++w) {
                blended += row_weights[h] * col_weights[w] *
                    float(table[(ulong(row_taps[h]) * kRadioGridSide + col_taps[w]) *
                                kRadioHidden + dim]);
            }
        }
        out[dim] = blended;
    }
}

// RADIO's prefix tokens follow the patches (attention without rope is
// indifferent to order, and the merger then reads the patches from row 0).
// It runs after the patch embedding, whose residual GEMM writes whole 64-row
// tiles over the rows past the patches: it writes the ten prefix rows
// and zeroes the rest up to the GEMM tile past them.
// Grid {roundUp(patches + 10, 64) - patches}.
kernel void vision_prefix_radio(
    device const bfloat *prefix [[buffer(0)]],
    device float *hidden [[buffer(1)]],
    constant VisionGridParams &params [[buffer(2)]],
    uint index [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    device float *out = hidden +
        (ulong(params.grid_height) * params.grid_width + index) * kRadioHidden;
    for (uint dim = thread_index; dim < kRadioHidden; dim += 256)
        out[dim] = index < kRadioPrefix ? float(prefix[ulong(index) * kRadioHidden + dim]) : 0.0f;
}

// Dense bf16 GEMM: output[M, N] = input[M, K] * weights[N, K]^T + bias[N]

enum VisionActivation { VisionNone, VisionGeluTanh, VisionGeluErf, VisionReluSquared };

// Abramowitz & Stegun 7.1.26 (max error ~1.5e-7, far below bf16 rounding).
inline float vision_erf(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    x = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float polynomial = t * (0.254829592f + t * (-0.284496736f +
        t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
    return sign * (1.0f - polynomial * precise::exp(-x * x));
}

template <ushort TileM, ushort TileN, VisionActivation Activation,
          bool AddResidual, typename Output>
inline void vision_gemm_tile(
    device bfloat *input,
    device bfloat *weights,
    device bfloat *bias,
    device Output *output,
    device Output *residual,
    uint output_size,
    uint input_size,
    uint output_origin,
    uint row_origin)
{
    device bfloat *rows = input + ulong(row_origin) * input_size;
    device bfloat *tile_weights = weights + ulong(output_origin) * input_size;
    auto a = tensor(rows,
                    dextents<int, 2>{int(input_size), TileM},
                    array<int, 2>{1, int(input_size)});
    auto b = tensor(tile_weights,
                    dextents<int, 2>{int(input_size), TileN},
                    array<int, 2>{1, int(input_size)});
    auto c = tensor(output + ulong(row_origin) * output_size,
                    dextents<int, 2>{int(output_size), TileM},
                    array<int, 2>{1, int(output_size)});
    // Static BK chunks with multiply_accumulate mode: static extents avoid
    // per-load bounds checks, accumulation stays in the destination
    // cooperative tensor, and a mem_none barrier per chunk keeps simdgroups
    // cache-adjacent along K.
    constexpr ushort BK = 128;
    constexpr auto descriptor = matmul2d_descriptor(
        TileM, TileN, BK, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<8>> operation;
    auto a0 = a.template slice<BK, TileM>(0, 0);
    auto b0 = b.template slice<BK, TileN>(0, 0);
    auto accumulated = operation.template get_destination_cooperative_tensor<
        decltype(a0), decltype(b0), float>();
    #pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
        accumulated[i] = 0.0f;
    }
    for (uint chunk = 0; chunk < input_size / BK; ++chunk) {
        auto a_slice = a.template slice<BK, TileM>(chunk * BK, 0);
        auto b_slice = b.template slice<BK, TileN>(chunk * BK, 0);
        operation.run(a_slice, b_slice, accumulated);
        threadgroup_barrier(mem_flags::mem_none);
    }

    auto converted = operation.template get_destination_cooperative_tensor<
        decltype(a0), decltype(b0), Output>();
    #pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
        auto index = accumulated.get_multidimensional_index(i);
        float value = accumulated[i] +
            float(bias[output_origin + index[0]]);
        if constexpr (Activation == VisionGeluTanh) {
            value = 0.5f * value *
                (1.0f + precise::tanh(
                    0.7978845608028654f *
                    (value + 0.044715f * value * value * value)));
        } else if constexpr (Activation == VisionGeluErf) {
            value = 0.5f * value *
                (1.0f + vision_erf(value * 0.7071067811865476f));
        } else if constexpr (Activation == VisionReluSquared) {
            value = max(value, 0.0f);
            value *= value;
        }
        if constexpr (AddResidual) {
            value += float(residual[
                ulong(row_origin + index[1]) * output_size +
                output_origin + index[0]]);
        }
        converted[i] = Output(value);
    }
    converted.store(c.template slice<TileN, TileM>(output_origin, 0));
}

#define VISION_GEMM_KERNEL(name, tile_m, tile_n, activation, add_residual,    \
                          output_type)                                       \
kernel void name(                                                             \
    device bfloat *input [[buffer(0)]],                                       \
    device bfloat *weights [[buffer(1)]],                                     \
    device bfloat *bias [[buffer(2)]],                                        \
    device output_type *output [[buffer(3)]],                                 \
    device output_type *residual [[buffer(4)]],                               \
    constant VisionGemmParams &params [[buffer(5)]],                          \
    uint2 group [[threadgroup_position_in_grid]])                             \
{                                                                             \
    vision_gemm_tile<tile_m, tile_n, activation, add_residual, output_type>(  \
        input, weights, bias, output, residual,                               \
        params.output_size, params.input_size,                                \
        group.y * tile_n, group.x * tile_m);                                  \
}

VISION_GEMM_KERNEL(vision_gemm_m64n128, 64, 128, VisionNone, false, bfloat)
// The residual stream, from the positions and patch embedding through the 27
// blocks, stays fp32: rounded to bf16 at each of its 55 updates, it carried
// most of the tower's drift from the fp32 reference.
VISION_GEMM_KERNEL(vision_gemm_m64n128_residual, 64, 128, VisionNone, true,
                   float)
VISION_GEMM_KERNEL(vision_gemm_m64n128_gelu_tanh, 64, 128, VisionGeluTanh, false,
                   bfloat)
// The merger's 256-divisible output sizes use the wider tile.
VISION_GEMM_KERNEL(vision_gemm_m32n256, 32, 256, VisionNone, false, bfloat)
VISION_GEMM_KERNEL(vision_gemm_m32n256_gelu_erf, 32, 256, VisionGeluErf,
                   false, bfloat)
// RADIO's MLP (erf GELU) and its merger's ReLU-squared up projection.
VISION_GEMM_KERNEL(vision_gemm_m64n128_gelu_erf, 64, 128, VisionGeluErf, false,
                   bfloat)
VISION_GEMM_KERNEL(vision_gemm_m32n256_relu2, 32, 256, VisionReluSquared,
                   false, bfloat)

// LayerNorm with bias of an fp32 residual row (fp32 statistics), one
// threadgroup per row. The fp32 output may be the input (RADIO's final norm,
// in place): each thread writes only the columns it last read.

template <typename Output>
kernel void vision_layer_norm(
    device const float *input [[buffer(0)]],
    device const bfloat *weight [[buffer(1)]],
    device const bfloat *bias [[buffer(2)]],
    device Output *output [[buffer(3)]],
    constant VisionNormParams &params [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    threadgroup float reductions[8];
    device const float *row = input + ulong(group) * params.width;
    device Output *out = output + ulong(group) * params.width;

    float sum = 0.0f;
    for (uint column = thread_index; column < params.width; column += 256) {
        sum += row[column];
    }
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
        out[column] = Output(normalized * float(weight[column]) +
                             float(bias[column]));
    }
}

#define VISION_LAYER_NORM_KERNEL(name, Output)                                 \
  template [[host_name(#name)]] kernel void vision_layer_norm<Output>(         \
      device const float *, device const bfloat *, device const bfloat *,      \
      device Output *, constant VisionNormParams &, uint, uint, uint, uint);
VISION_LAYER_NORM_KERNEL(vision_layer_norm, bfloat)
VISION_LAYER_NORM_KERNEL(vision_layer_norm_f32, float)

// RADIO's merger norm: the RMS of a pixel-shuffled row of four consecutive
// block-major fp32 residual rows (params.width, 5120) times its weight, fp32
// statistics, one threadgroup per merged row.

kernel void vision_rms_norm(
    device const float *input [[buffer(0)]],
    device const bfloat *weight [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant VisionNormParams &params [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    threadgroup float reductions[8];
    device const float *row = input + ulong(group) * params.width;
    device bfloat *out = output + ulong(group) * params.width;
    float squares = 0.0f;
    for (uint column = thread_index; column < params.width; column += 256)
        squares += row[column] * row[column];
    squares = simd_sum(squares);
    if (simd_lane == 0) reductions[simd_group] = squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    #pragma unroll
    for (ushort i = 0; i < 8; ++i) total += reductions[i];
    float inverse = rsqrt(total / float(params.width) +
                          SPLASH_VISION_MERGER_RMS_EPSILON);
    for (uint column = thread_index; column < params.width; column += 256)
        out[column] = bfloat(row[column] * inverse * float(weight[column]));
}

// QKV preparation: apply 2D rope to Q and K, scatter into per-head layouts.
// Q, K: (heads, padded_tokens, 80); V: (heads, key_tile, 128, 72). Padded
// tokens are written as zeros so full 128-key tiles are always readable and
// hold finite values under the softmax mask.

kernel void vision_qkv_prepare(
    device const bfloat *qkv [[buffer(0)]],
    device const float *rope_cos [[buffer(1)]],
    device const float *rope_sin [[buffer(2)]],
    device bfloat *queries [[buffer(3)]],
    device bfloat *keys [[buffer(4)]],
    device bfloat *values [[buffer(5)]],
    constant VisionQkvParams &params [[buffer(6)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    constexpr ushort half_dim = kVisionHeadDim / 2;
    const bool valid = token < params.tokens;
    device const bfloat *row =
        qkv + ulong(token) * 3 * kVisionHeads * kVisionHeadDim;
    device const float *cos_row = rope_cos + ulong(token) * kVisionHeadDim;
    device const float *sin_row = rope_sin + ulong(token) * kVisionHeadDim;
    uint key_tile = token / 128, within = token % 128;
    uint tiles = params.padded_tokens / 128;

    for (uint index = thread_index;
         index < uint(kVisionHeads) * kVisionQkDim; index += 256) {
        uint head = index / kVisionQkDim;
        uint dim = index % kVisionQkDim;
        ulong plane = (ulong(head) * params.padded_tokens + token) *
            kVisionQkDim + dim;
        if (!valid || dim >= kVisionHeadDim) {
            queries[plane] = bfloat(0.0f);
            keys[plane] = bfloat(0.0f);
            if (dim < kVisionHeadDim) {
                values[(((ulong(head) * tiles + key_tile) * 128) + within) *
                       kVisionHeadDim + dim] = bfloat(0.0f);
            }
            continue;
        }
        uint base = head * kVisionHeadDim + dim;
        uint rotated = head * kVisionHeadDim +
            (dim < half_dim ? dim + half_dim : dim - half_dim);
        float sign = dim < half_dim ? -1.0f : 1.0f;
        float cos_value = cos_row[dim];
        float sin_value = sin_row[dim];
        queries[plane] = bfloat(float(row[base]) * cos_value +
            sign * float(row[rotated]) * sin_value);
        keys[plane] = bfloat(float(row[kVisionHidden + base]) * cos_value +
            sign * float(row[kVisionHidden + rotated]) * sin_value);
        values[(((ulong(head) * tiles + key_tile) * 128) + within) *
               kVisionHeadDim + dim] =
            bfloat(float(row[2 * kVisionHidden + base]));
    }
}

// RADIO's QKV preparation: no rope, heads of 80 (Q and K need no padding),
// the same layouts and zeroed padding tokens.

kernel void vision_qkv_prepare_radio(
    device const bfloat *qkv [[buffer(0)]],
    device bfloat *queries [[buffer(1)]],
    device bfloat *keys [[buffer(2)]],
    device bfloat *values [[buffer(3)]],
    constant VisionQkvParams &params [[buffer(4)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    const bool valid = token < params.tokens;
    device const bfloat *row = qkv + ulong(token) * 3 * kRadioHidden;
    uint key_tile = token / 128, within = token % 128;
    uint tiles = params.padded_tokens / 128;
    for (uint index = thread_index; index < kRadioHidden; index += 256) {
        uint head = index / kRadioHeadDim;
        uint dim = index % kRadioHeadDim;
        ulong plane = (ulong(head) * params.padded_tokens + token) *
            kRadioHeadDim + dim;
        ulong value = (((ulong(head) * tiles + key_tile) * 128) + within) *
            kRadioHeadDim + dim;
        if (!valid) {
            queries[plane] = keys[plane] = values[value] = bfloat(0.0f);
            continue;
        }
        queries[plane] = row[index];
        keys[plane] = row[kRadioHidden + index];
        values[value] = row[2 * kRadioHidden + index];
    }
}

// Non-causal flash attention over the per-head layout.
// Grid: {ceil(tokens / 32), heads} of 128 threads, the query rows and threads
// of metal/abi/Vision.h. Output: (heads, padded_tokens, 72).
//
// A threadgroup takes 32 query rows on four simdgroups and 24.25 KiB of
// threadgroup memory, which keeps several resident per core. Scores, bf16
// probabilities and the row statistics each have their own storage, so a
// key tile needs two barriers: one before the softmax reads its scores and
// one before PV reads its probabilities. The next tile stores its scores
// past the second, and writes its probabilities past its first, which no
// thread reaches before finishing this tile's PV. PV accumulates into the
// running output, which each tile first rescales by its rows' maximum
// change, so one M x D fp32 cooperative tensor is live.

template <ushort D, ushort DQ>
kernel void vision_attention(
    device bfloat *queries [[buffer(0)]],
    device bfloat *keys [[buffer(1)]],
    device bfloat *values [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant VisionAttentionParams &params [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    constexpr ushort M = SPLASH_VISION_ATTENTION_ROWS, N = 128;
    constexpr ushort Simdgroups = SPLASH_VISION_ATTENTION_THREADS / 32;
    // Four lanes own one row, four consecutive keys of every 16, so the row
    // maximum and sum are two xor shuffles and scores and probabilities move
    // as vectors.
    constexpr ushort LanesPerRow = Simdgroups * 32 / M;
    constexpr ushort Runs = N / (LanesPerRow * 4);
    static_assert(Simdgroups * 32 == SPLASH_VISION_ATTENTION_THREADS &&
                      LanesPerRow * M == Simdgroups * 32,
                  "whole simdgroups whose lanes own whole rows");
    static_assert(LanesPerRow == 4, "two xor shuffles reduce a row");
    alignas(16) threadgroup float score_storage[M * N];
    alignas(16) threadgroup bfloat probabilities[M * N];
    threadgroup float row_sum[M];
    threadgroup float previous_scale[M];
    const ushort row = thread_index / LanesPerRow;
    const ushort part = thread_index % LanesPerRow;
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    uint head = group.y;
    uint query_start = group.x * M;
    ulong qk_plane = ulong(head) * params.padded_tokens * DQ;
    device bfloat *q = queries + qk_plane + ulong(query_start) * DQ;
    device bfloat *k = keys + qk_plane;
    device bfloat *v = values + ulong(head) * D * params.padded_tokens;
    device bfloat *out = output +
        ulong(head) * params.padded_tokens * D + ulong(query_start) * D;
    uint context = params.tokens;
    uint tiles = (context + N - 1) / N;

    auto qt = tensor(q, dextents<int, 2>{DQ, M}, array<int, 2>{1, DQ});
    constexpr auto qk_descriptor =
        matmul2d_descriptor(M, N, DQ, false, true, false);
    matmul2d<qk_descriptor, execution_simdgroups<Simdgroups>> qk;
    auto q0 = qt.slice<DQ, M>(0, 0);

    auto pt = tensor(probabilities, dextents<int, 2>{N, M},
                     array<int, 2>{1, N});
    auto p0 = pt.slice<N, M>(0, 0);
    auto st = tensor(score_storage, dextents<int, 2>{N, M},
                     array<int, 2>{1, N});
    auto s0 = st.slice<N, M>(0, 0);
    auto first_v = tensor(v, dextents<int, 2>{D, N},
                          array<int, 2>{1, D});
    auto first_v0 = first_v.slice<D, N>(0, 0);
    constexpr auto pv_descriptor = matmul2d_descriptor(
        M, D, N, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_descriptor, execution_simdgroups<Simdgroups>> pv;
    auto running = pv.template get_destination_cooperative_tensor<
        decltype(p0), decltype(first_v0), float>();
    // Uniform capacity partitions the M x D output across the simdgroups'
    // lanes, and equality proves no padding. Otherwise (Apple10 lays the 72
    // columns out over 96) only valid elements may be accessed: a padding
    // element's coordinates need not name a row of the statistics.
    const bool full =
        uint(running.get_capacity()) * (Simdgroups * 32u) == uint(M) * D;
    for (ushort i = 0; i < running.get_capacity(); ++i) {
        if (full || running.is_valid_element(i))
            running[i] = 0.0f;
    }
    threadgroup const float4 *row_scores =
        reinterpret_cast<threadgroup const float4 *>(score_storage + row * N);
    threadgroup bfloat4 *row_probabilities =
        reinterpret_cast<threadgroup bfloat4 *>(probabilities + row * N);
    for (uint tile = 0; tile < tiles; ++tile) {
        device bfloat *tile_k = k + ulong(tile) * N * DQ;
        auto kt = tensor(tile_k, dextents<int, 2>{DQ, N},
                         array<int, 2>{1, DQ});
        auto k0 = kt.slice<DQ, N>(0, 0);
        auto scores = qk.template get_destination_cooperative_tensor<
            decltype(q0), decltype(k0), float>();
        qk.run(q0, k0, scores);
        scores.store(s0);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Keys at or past `limit` lie beyond the image and are masked.
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
            row_probabilities[r * LanesPerRow + part] = bfloat4(exponents);
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
            if (!full && !running.is_valid_element(i))
                continue;
            auto index = running.get_multidimensional_index(i);
            running[i] *= previous_scale[index[1]];
        }
        device bfloat *tile_v = v + ulong(tile) * N * D;
        auto vt = tensor(tile_v, dextents<int, 2>{D, N},
                         array<int, 2>{1, D});
        auto v0 = vt.slice<D, N>(0, 0);
        pv.run(p0, v0, running);
    }

    auto ot = tensor(out, dextents<int, 2>{D, M}, array<int, 2>{1, D});
    auto converted = pv.template get_destination_cooperative_tensor<
        decltype(p0), decltype(first_v0), bfloat>();
    for (ushort i = 0; i < running.get_capacity(); ++i) {
        if (!full && !running.is_valid_element(i))
            continue;
        auto index = running.get_multidimensional_index(i);
        converted[i] = bfloat(running[i] / row_sum[index[1]]);
    }
    converted.store(ot.slice<D, M>(0, 0));
}

// Qwen3.5's heads of 72 (Q and K padded to 80) and RADIO's of 80.
template [[host_name("vision_attention")]] kernel void
vision_attention<kVisionHeadDim, kVisionQkDim>(
    device bfloat *, device bfloat *, device bfloat *, device bfloat *,
    constant VisionAttentionParams &, uint2, uint);
template [[host_name("vision_attention_d80")]] kernel void
vision_attention<kRadioHeadDim, kRadioHeadDim>(
    device bfloat *, device bfloat *, device bfloat *, device bfloat *,
    constant VisionAttentionParams &, uint2, uint);

// Gather padded attention output back to (tokens, hidden) rows.

template <ushort Hidden, ushort HeadDim>
kernel void vision_attention_gather(
    device const bfloat *padded [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    constant VisionQkvParams &params [[buffer(2)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    device bfloat *row = output + ulong(token) * Hidden;
    for (uint index = thread_index; index < Hidden; index += 256) {
        uint head = index / HeadDim;
        uint dim = index % HeadDim;
        row[index] = padded[
            (ulong(head) * params.padded_tokens + token) * HeadDim +
            dim];
    }
}

template [[host_name("vision_attention_gather")]] kernel void
vision_attention_gather<kVisionHidden, kVisionHeadDim>(
    device const bfloat *, device bfloat *, constant VisionQkvParams &, uint, uint);
template [[host_name("vision_attention_gather_d80")]] kernel void
vision_attention_gather<kRadioHidden, kRadioHeadDim>(
    device const bfloat *, device bfloat *, constant VisionQkvParams &, uint, uint);

// EVS (ops::VideoPruning), as the processor's compute_retention_mask runs on
// bf16 embeddings: each tubelet row's dissimilarity to the same row of the
// tubelet before, 1 - cosine_similarity, each step rounded to bf16 as torch
// rounds it (norms, the normalized rows, their products, the sum); 255 for
// the first tubelet, which it always keeps. One threadgroup per row.
kernel void video_pruning_dissimilarity(
    device const bfloat *encoded [[buffer(0)]],
    device float *dissimilarity [[buffer(1)]],
    constant VideoPruningParams &p [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    const uint tubelet = group / p.rows, row = group % p.rows;
    if (tubelet == 0) {
        if (thread_index == 0) dissimilarity[group] = 255.0f;
        return;
    }
    threadgroup float reductions[2][8];
    device const bfloat *x = encoded + ulong(tubelet * p.stride + row) * p.width;
    device const bfloat *y = encoded + ulong((tubelet - 1) * p.stride + row) * p.width;
    float xx = 0.0f, yy = 0.0f;
    for (uint column = thread_index; column < p.width; column += 256) {
        const float a = float(x[column]), b = float(y[column]);
        xx += a * a;
        yy += b * b;
    }
    xx = simd_sum(xx);
    yy = simd_sum(yy);
    if (simd_lane == 0) {
        reductions[0][simd_group] = xx;
        reductions[1][simd_group] = yy;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    xx = yy = 0.0f;
    for (ushort i = 0; i < 8; ++i) {
        xx += reductions[0][i];
        yy += reductions[1][i];
    }
    const float xNorm = max(float(bfloat(sqrt(xx))), 1e-8f);
    const float yNorm = max(float(bfloat(sqrt(yy))), 1e-8f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float dot = 0.0f;
    for (uint column = thread_index; column < p.width; column += 256)
        dot += float(bfloat(float(bfloat(float(x[column]) / xNorm)) *
                            float(bfloat(float(y[column]) / yNorm))));
    dot = simd_sum(dot);
    if (simd_lane == 0) reductions[0][simd_group] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
        float total = 0.0f;
        for (ushort i = 0; i < 8; ++i) total += reductions[0][i];
        dissimilarity[group] = float(bfloat(1.0f - float(bfloat(total))));
    }
}

// Whether EVS keeps each row: its place in the rows sorted by dissimilarity,
// descending and stable, is among the first `kept`. One thread per row,
// which counts the rows ahead of it. NaN sorts first, as in torch, so the
// order stays total and exactly `kept` rows are kept.
// ponytail: quadratic in the video's rows (32K at the 256-frame cap is ~1G
// comparisons, milliseconds); a GPU sort if the frame cap grows.
kernel void video_pruning_keep(
    device const float *dissimilarity [[buffer(0)]],
    device uint *keep [[buffer(1)]],
    constant VideoPruningParams &p [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    const uint total = p.tubelets * p.rows;
    if (index >= total) return;
    const float value = dissimilarity[index];
    const bool unordered = isnan(value);
    uint ahead = 0;
    for (uint other = 0; other < total; ++other) {
        const float d = dissimilarity[other];
        const bool before = isnan(d) ? !unordered || other < index
                                     : !unordered && (d > value || (d == value && other < index));
        ahead += before ? 1u : 0u;
    }
    keep[index] = ahead < p.kept ? 1u : 0u;
}

// The span's rows in the processor's order: each kept tubelet row after the
// text before its tubelet and the kept rows before it; each text row after
// the kept rows of the tubelets its text follows. One threadgroup per
// tubelet row, then one per text row.
kernel void video_pruning_place(
    device const bfloat *encoded [[buffer(0)]],
    device const bfloat *text [[buffer(1)]],
    device const uint *cuts [[buffer(2)]],
    device const uint *keep [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant VideoPruningParams &p [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    const uint total = p.tubelets * p.rows;
    const bool tubeletRow = group < total;
    if (tubeletRow && !keep[group]) return;
    // The tubelet rows before this one, or before this text row.
    uint before = group;
    uint destination = 0;
    if (!tubeletRow) {
        const uint token = group - total;
        uint preceded = 0;
        for (uint tubelet = 0; tubelet < p.tubelets; ++tubelet)
            preceded += cuts[tubelet] <= token ? 1u : 0u;
        before = preceded * p.rows;
        destination = token;
    } else {
        destination = cuts[group / p.rows];
    }
    threadgroup uint reductions[8];
    uint count = 0;
    for (uint index = thread_index; index < before; index += 256) count += keep[index];
    count = simd_sum(count);
    if (simd_lane == 0) reductions[simd_group] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort i = 0; i < 8; ++i) destination += reductions[i];
    device const bfloat *source =
        tubeletRow ? encoded + ulong((group / p.rows) * p.stride + group % p.rows) * p.width
                   : text + ulong(group - total) * p.width;
    device bfloat *out = output + ulong(destination) * p.width;
    for (uint column = thread_index; column < p.width; column += 256) out[column] = source[column];
}

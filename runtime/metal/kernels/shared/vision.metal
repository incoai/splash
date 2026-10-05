// Vision encoder kernels for the Qwen3.5 vision tower: pixel patchify,
// position/rope preparation, dense bf16 GEMMs with fused epilogues,
// LayerNorm, 2D-rope QKV preparation and non-causal flash attention.
//
// Conventions match the text-model kernels: 256 threads (8 simdgroups) per
// threadgroup, MPP cooperative-tensor matmuls with fp32 accumulation and
// hand-written epilogues, grid {row_tile, output_tile}. Attention runs on the
// native head dimension (72); token counts are padded to the 128-key tile so
// K/V tiles read in full, with padded keys zeroed and masked by the softmax.
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

// Dense bf16 GEMM: output[M, N] = input[M, K] * weights[N, K]^T + bias[N]

enum VisionActivation { VisionNone, VisionGeluTanh, VisionGeluErf };

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

// LayerNorm with bias of an fp32 residual row (fp32 statistics), one
// threadgroup per row.

kernel void vision_layer_norm(
    device const float *input [[buffer(0)]],
    device const bfloat *weight [[buffer(1)]],
    device const bfloat *bias [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant VisionNormParams &params [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    threadgroup float reductions[8];
    device const float *row = input + ulong(group) * params.width;
    device bfloat *out = output + ulong(group) * params.width;

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
        out[column] = bfloat(normalized * float(weight[column]) +
                             float(bias[column]));
    }
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

// Non-causal flash attention over the per-head layout.
// Grid: {ceil(tokens / 64), heads}. Output: (heads, padded_tokens, 72).

kernel void vision_attention(
    device bfloat *queries [[buffer(0)]],
    device bfloat *keys [[buffer(1)]],
    device bfloat *values [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant VisionAttentionParams &params [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    constexpr ushort M = 64, N = 128, D = kVisionHeadDim;
    constexpr ushort DQ = kVisionQkDim;
    constexpr ushort RowsPerSimdgroup = M / 8;
    // Preserve fp32 QK scores through softmax. After every lane has read the
    // scores, the first half holds bf16 probabilities and the second half
    // exposes row scales to PV. Running softmax state stays in registers, so
    // the complete threadgroup allocation is exactly 32 KiB.
    threadgroup float score_storage[M * N];
    threadgroup bfloat *probabilities =
        reinterpret_cast<threadgroup bfloat *>(score_storage);
    threadgroup float *row_sum = score_storage + M * N / 2;
    threadgroup float *previous_scale = row_sum + M;
    float running_max[RowsPerSimdgroup];
    float running_sum[RowsPerSimdgroup];
    for (ushort i = 0; i < RowsPerSimdgroup; ++i) {
        running_max[i] = -INFINITY;
        running_sum[i] = 0.0f;
    }

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
    matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
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
    constexpr auto pv_descriptor =
        matmul2d_descriptor(M, D, N, false, false, false);
    matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
    auto running = pv.template get_destination_cooperative_tensor<
        decltype(p0), decltype(first_v0), float>();
    for (ushort i = 0; i < running.get_capacity(); ++i) running[i] = 0.0f;
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

        // Each simdgroup owns eight rows. Phase one moves scores into
        // registers and computes the running softmax state; after one
        // barrier, phase two overwrites the (aliased) storage with bf16
        // probabilities.
        float exponents[RowsPerSimdgroup][4];
        float scales[RowsPerSimdgroup];
        for (ushort i = 0; i < RowsPerSimdgroup; ++i) {
            ushort row = simd_group * RowsPerSimdgroup + i;
            float values_local[4];
            float tile_max = -INFINITY;
            for (ushort j = 0; j < 4; ++j) {
                uint column = simd_lane + j * 32;
                uint key_position = tile * N + column;
                float score = score_storage[row * N + column] * params.scale;
                values_local[j] =
                    key_position < context ? score : -INFINITY;
                tile_max = max(tile_max, values_local[j]);
            }
            tile_max = simd_max(tile_max);
            float next_max = max(running_max[i], tile_max);
            float scale = fast::exp(running_max[i] - next_max);
            float tile_sum = 0.0f;
            for (ushort j = 0; j < 4; ++j) {
                exponents[i][j] = fast::exp(values_local[j] - next_max);
                tile_sum += exponents[i][j];
            }
            tile_sum = simd_sum(tile_sum);
            scales[i] = scale;
            running_sum[i] = running_sum[i] * scale + tile_sum;
            running_max[i] = next_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort i = 0; i < RowsPerSimdgroup; ++i) {
            ushort row = simd_group * RowsPerSimdgroup + i;
            if (simd_lane == 0) {
                previous_scale[row] = scales[i];
                row_sum[row] = running_sum[i];
            }
            for (ushort j = 0; j < 4; ++j) {
                uint column = simd_lane + j * 32;
                probabilities[row * N + column] = bfloat(exponents[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device bfloat *tile_v = v + ulong(tile) * N * D;
        auto vt = tensor(tile_v, dextents<int, 2>{D, N},
                         array<int, 2>{1, D});
        auto v0 = vt.slice<D, N>(0, 0);
        auto partial_output = pv.template get_destination_cooperative_tensor<
            decltype(p0), decltype(v0), float>();
        pv.run(p0, v0, partial_output);
        for (ushort i = 0; i < running.get_capacity(); ++i) {
            auto index = running.get_multidimensional_index(i);
            running[i] = running[i] * previous_scale[index[1]] +
                partial_output[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto ot = tensor(out, dextents<int, 2>{D, M}, array<int, 2>{1, D});
    auto converted = pv.template get_destination_cooperative_tensor<
        decltype(p0), decltype(first_v0), bfloat>();
    for (ushort i = 0; i < running.get_capacity(); ++i) {
        auto index = running.get_multidimensional_index(i);
        converted[i] = bfloat(running[i] / row_sum[index[1]]);
    }
    converted.store(ot.slice<D, M>(0, 0));
}

// Gather padded attention output back to (tokens, 1152) rows.

kernel void vision_attention_gather(
    device const bfloat *padded [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    constant VisionQkvParams &params [[buffer(2)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    device bfloat *row = output + ulong(token) * kVisionHidden;
    for (uint index = thread_index; index < kVisionHidden; index += 256) {
        uint head = index / kVisionHeadDim;
        uint dim = index % kVisionHeadDim;
        row[index] = padded[
            (ulong(head) * params.padded_tokens + token) * kVisionHeadDim +
            dim];
    }
}

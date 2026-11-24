// Inkling 2's vision tower's own kernels (ops/InklingVision.cpp): pixel
// patchify, the position and class rows with the rotary tables, the query,
// key and value preparation of its axial 2D RoPE, and the merger's input
// rows. Its projections, LayerNorms and attention are the shared dense
// operators (shared/dense_linear.metal, shared/normalization.metal,
// shared/dense_attention.metal), as Qwen's tower's are.
//
// Token order: the image's patches first, spatial-merge-block-major as
// Qwen's tower orders them (each group of four consecutive rows is one 2x2
// unit, so the merger reads them as one row of four), then its class token
// at row `patches`. Attention is non-causal and every row carries its own
// position, so where the class token sits changes nothing but which rows
// the merger reads.

#include "metal/abi/Vision.h"
#include <metal_stdlib>

using namespace metal;

constant ushort kInklingHeads = 16;
constant ushort kInklingHeadDim = 64;
constant ushort kInklingHidden = 1024;
constant ushort kInklingPatch = 20;
constant ushort kInklingMerge = 2;
// A patch's 1200 values [channel, patch-row, patch-col], as the GGUF's patch
// embedding orders its input, padded with zeros to the GEMM's 1280.
constant ushort kInklingPatchValues = 3 * 20 * 20;
constant ushort kInklingPatchDim = 1280;
constant ushort kInklingGridSide = 32; // learned position table is 32 x 32
constant float kInklingRopeTheta = SPLASH_INKLING_VISION_ROPE_THETA;

// Block-major (row, col) of one patch token.
inline uint2 inkling_patch_position(uint token, uint grid_width) {
    uint blocks_w = grid_width / kInklingMerge;
    uint block = token / (kInklingMerge * kInklingMerge);
    uint inner = token % (kInklingMerge * kInklingMerge);
    return uint2((block / blocks_w) * kInklingMerge + inner / kInklingMerge,
                 (block % blocks_w) * kInklingMerge + inner % kInklingMerge);
}

// Patchify: uint8 RGB pixels (H, W, 3), H and W 20 per patch, into bf16
// patch rows (patches, 1280) normalized x / 127.5 - 1, the columns past a
// patch's 1200 values zero, so the patch embedding reads finite rows. A pixel
// past the image's content is its padding, -1 as its reference pads it.
kernel void inkling_vision_patchify(
    device const uchar *pixels [[buffer(0)]],
    device bfloat *patches [[buffer(1)]],
    constant InklingPatchifyParams &params [[buffer(2)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    device bfloat *row = patches + ulong(token) * kInklingPatchDim;
    uint2 position = inkling_patch_position(token, params.grid_width);
    uint image_width = params.grid_width * kInklingPatch;
    for (uint feature = thread_index; feature < kInklingPatchDim;
         feature += 256) {
        if (feature >= kInklingPatchValues) {
            row[feature] = bfloat(0.0f);
            continue;
        }
        uint channel = feature / (kInklingPatch * kInklingPatch);
        uint within = feature % (kInklingPatch * kInklingPatch);
        uint y = position.x * kInklingPatch + within / kInklingPatch;
        uint x = position.y * kInklingPatch + within % kInklingPatch;
        float value = y < params.content_height && x < params.content_width
                          ? float(pixels[(ulong(y) * image_width + x) * 3 + channel])
                          : -1.0f;
        row[feature] = bfloat(value * (1.0f / 127.5f) - 1.0f);
    }
}

// The fp32 rows the patch embedding adds to: each patch's position, the
// learned 32x32 table resampled to the grid as bilinear interpolation
// without aligned corners does (half-pixel centres, edge taps clamped); and
// the class token's embedding plus its position at row `patches`.
// And each row's rotary tables, 64 cos and sin per token: a head's first 32
// values rotate in interleaved pairs by the column, the last 32 by the row,
// pair i at the frequency theta^(-2i / 32), the class token at (0, 0) and a
// patch at its 1-based (row, column).

inline void inkling_bilinear_taps(uint index, uint size, thread uint *tap,
                                  thread float *weight) {
    float source = (float(index) + 0.5f) * float(kInklingGridSide) /
        float(size) - 0.5f;
    float lower = floor(source);
    for (uint offset = 0; offset < 2; ++offset) {
        float position = lower + float(offset);
        tap[offset] = uint(clamp(position, 0.0f, float(kInklingGridSide - 1)));
        weight[offset] = max(1.0f - fabs(source - position), 0.0f);
    }
}

kernel void inkling_vision_prepare_positions(
    device const bfloat *table [[buffer(0)]],
    device const bfloat *class_token [[buffer(1)]],
    device float *hidden [[buffer(2)]],
    device float *rope_cos [[buffer(3)]],
    device float *rope_sin [[buffer(4)]],
    constant VisionGridParams &params [[buffer(5)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    const uint patches = params.grid_height * params.grid_width;
    device float *out = hidden + ulong(token) * kInklingHidden;
    uint2 position = uint2(0);
    if (token == patches) {
        for (uint dim = thread_index; dim < kInklingHidden; dim += 256)
            out[dim] = float(class_token[dim]) +
                float(class_token[kInklingHidden + dim]);
    } else {
        position = inkling_patch_position(token, params.grid_width);
        uint row_taps[2], col_taps[2];
        float row_weights[2], col_weights[2];
        inkling_bilinear_taps(position.x, params.grid_height, row_taps,
                              row_weights);
        inkling_bilinear_taps(position.y, params.grid_width, col_taps,
                              col_weights);
        for (uint dim = thread_index; dim < kInklingHidden; dim += 256) {
            float blended = 0.0f;
            for (uint h = 0; h < 2; ++h) {
                for (uint w = 0; w < 2; ++w) {
                    blended += row_weights[h] * col_weights[w] *
                        float(table[(ulong(row_taps[h]) * kInklingGridSide +
                                     col_taps[w]) * kInklingHidden + dim]);
                }
            }
            out[dim] = blended;
        }
        position += 1;
    }

    if (thread_index < kInklingHeadDim) {
        constexpr uint half_dim = kInklingHeadDim / 2;
        uint pair = (thread_index % half_dim) / 2;
        uint axis = thread_index < half_dim ? position.y : position.x;
        float inverse = precise::pow(kInklingRopeTheta,
                                     -float(2 * pair) / float(half_dim));
        float angle = float(axis) * inverse;
        rope_cos[ulong(token) * kInklingHeadDim + thread_index] =
            precise::cos(angle);
        rope_sin[ulong(token) * kInklingHeadDim + thread_index] =
            precise::sin(angle);
    }
}

// QKV preparation: the rotary embedding of interleaved pairs (x0, x1) ->
// (x0 cos - x1 sin, x1 cos + x0 sin) on queries and keys, and the scatter of
// q, k and v (each row's [0, 1024), [1024, 2048), [2048, 3072), head-major)
// into (heads, padded_tokens, 64). Rows from `tokens` on are zero, so whole
// 128-key tiles read finite values under the softmax mask.
kernel void inkling_vision_qkv_prepare(
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
    const bool valid = token < params.tokens;
    device const bfloat *row = qkv + ulong(token) * 3 * kInklingHidden;
    device const float *cos_row = rope_cos + ulong(token) * kInklingHeadDim;
    device const float *sin_row = rope_sin + ulong(token) * kInklingHeadDim;
    for (uint index = thread_index; index < kInklingHidden; index += 256) {
        uint head = index / kInklingHeadDim;
        uint dim = index % kInklingHeadDim;
        ulong plane = (ulong(head) * params.padded_tokens + token) *
            kInklingHeadDim + dim;
        if (!valid) {
            queries[plane] = bfloat(0.0f);
            keys[plane] = bfloat(0.0f);
            values[plane] = bfloat(0.0f);
            continue;
        }
        uint partner = index ^ 1;
        float sign = (dim & 1) ? 1.0f : -1.0f;
        float cos_value = cos_row[dim];
        float sin_value = sin_row[dim];
        queries[plane] = bfloat(float(row[index]) * cos_value +
            sign * float(row[partner]) * sin_value);
        keys[plane] = bfloat(float(row[kInklingHidden + index]) * cos_value +
            sign * float(row[kInklingHidden + partner]) * sin_value);
        values[plane] = row[2 * kInklingHidden + index];
    }
}

// The merger's input: the patch rows of the fp32 residual stream as bf16
// rows, which it reads as (patches / 4, 4096), one row per merged token,
// without the class token's row after them.
kernel void inkling_vision_patch_rows(
    device const float *hidden [[buffer(0)]],
    device bfloat *rows [[buffer(1)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    device const float *in = hidden + ulong(token) * kInklingHidden;
    device bfloat *out = rows + ulong(token) * kInklingHidden;
    for (uint dim = thread_index; dim < kInklingHidden; dim += 256)
        out[dim] = bfloat(in[dim]);
}

static_assert(kInklingHeads * kInklingHeadDim == kInklingHidden,
              "Inkling's heads cover its width");

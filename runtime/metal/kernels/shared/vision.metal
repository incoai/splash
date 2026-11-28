// The Qwen3.5 vision tower's own kernels (ops/Vision.cpp): pixel patchify,
// position and rope preparation, and 2D-rope QKV preparation into the
// per-head layout the dense attention reads (shared/dense_attention.metal);
// the tower's projections, LayerNorms and attention are the shared dense
// operators. 256 threads (8 simdgroups) per threadgroup. Attention runs on
// the native head dimension (72); token counts are padded to the 128-key
// tile so K/V tiles read in full, with padded keys zeroed and masked by the
// softmax.
//
// Token order everywhere is spatial-merge-block-major: each consecutive group
// of four patches is one 2x2 spatial unit, so the merger reads contiguous
// rows and the merged token index is row-major over the merged grid.

#include "metal/abi/DenseAttention.h"
#include "metal/abi/Vision.h"
#include <metal_stdlib>

using namespace metal;

constant ushort kVisionHeads = 16;
constant ushort kVisionHeadDim = 72;
// Q/K are stored padded as the dense attention reads them (80 dimensions).
constant ushort kVisionQkDim = splash_dense_attention_padded(kVisionHeadDim);
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

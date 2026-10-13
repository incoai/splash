#pragma once

// Parameter layouts and constants of the GPU side of the ANE FFN split
// (ops/AneFfn.cpp).
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The Hadamard rotation blocks: the inputs gate and up multiply rotate in
// blocks of ANE_FFN_INPUT_BLOCK values, the ANE's intermediate rows, which
// down multiplies, in blocks of ANE_FFN_INTERMEDIATE_BLOCK, whose larger span
// spreads their larger outliers further. A simdgroup rotates 128 values at a
// time, four per lane.
#define ANE_FFN_ROTATION_UNIT 128u
#define ANE_FFN_INPUT_BLOCK 128u
#define ANE_FFN_INTERMEDIATE_BLOCK 512u
// ane_ffn_rotate: ANE_FFN_ROTATE_THREADS threads per row, each simdgroup
// rotating at most ANE_FFN_ROTATE_BLOCKS input blocks of the row.
#define ANE_FFN_ROTATE_THREADS 256u
#define ANE_FFN_ROTATE_BLOCKS 8u
// ane_ffn_weights and ane_ffn_row_scale: one simdgroup per weight row,
// ANE_FFN_WEIGHT_ROWS rows per threadgroup.
#define ANE_FFN_WEIGHT_ROWS 8u
#define ANE_FFN_WEIGHT_THREADS (ANE_FFN_WEIGHT_ROWS * 32u)
// ane_ffn_pack and ane_ffn_join: tiles of ANE_FFN_TILE rows by ANE_FFN_TILE
// channels, ANE_FFN_TILE x ANE_FFN_TILE_ROWS threads each.
#define ANE_FFN_TILE 32u
#define ANE_FFN_TILE_ROWS 8u
// int8 values span +-ANE_FFN_INT8_PEAK. The ANE dequantizes them by
// 1 / ANE_FFN_INT8_UNIT, so that fp16 cannot overflow, which every scale
// carries back.
#define ANE_FFN_INT8_PEAK 127.0f
#define ANE_FFN_INT8_UNIT 128.0f
// The least peak a token's or a weight row's int8 scale is taken over:
// fp16's least normal value, so that every scale is a normal half, its
// inverse finite, and an all-zero row takes codes of 0.
#define ANE_FFN_PEAK_FLOOR 0x1p-14f

struct AneFfnRotateParams {
  uint32_t hidden;
};

// Rotated rows [rows][hidden] as the channel-major int8 inputs
// [channel, channel + 32 * grid.y) of the ANE, stride bytes per channel.
struct AneFfnPackParams {
  uint32_t hidden;
  uint32_t channel;
  uint32_t stride;
};

// Rotated int8 of a projection: its output rows [row, row + N) and inputs
// [input, input + width), where groups is its input count / 32 (an image
// tensor of GGUF_FMT_* format), into rows of stride bytes and their scales
// scale_stride halves apart.
struct AneFfnWeightParams {
  uint32_t groups;
  uint32_t row;
  uint32_t input;
  uint32_t width;
  uint32_t stride;
  uint32_t scale_stride;
  uint32_t format;
};

// output[t][c] += partial[c][t] * partial[hidden][t] * token_scale[t], in
// fp32, for rows t < rows, setting a status word if one of those values is
// not finite; stride halves per channel of the partial.
struct AneFfnJoinParams {
  uint32_t hidden;
  uint32_t stride;
  uint32_t rows;
};

static_assert(sizeof(AneFfnRotateParams) == 4, "ANE FFN rotate parameters are 4 bytes");
static_assert(sizeof(AneFfnPackParams) == 12, "ANE FFN pack parameters are 12 bytes");
static_assert(sizeof(AneFfnWeightParams) == 28, "ANE FFN weight parameters are 28 bytes");
static_assert(sizeof(AneFfnJoinParams) == 12, "ANE FFN join parameters are 12 bytes");

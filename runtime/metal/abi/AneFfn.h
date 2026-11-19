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

// Rotated int8 of a projection: its output rows [row, row + rows) and inputs
// [input, input + width), where groups is its input count / 32 (an image
// tensor of GGUF_FMT_* format), into rows of stride bytes and their scales
// scale_stride halves apart; the ANE's channels, gate's and up's rows or
// down's inputs, in descending units (ane_ffn_descending).
struct AneFfnWeightParams {
  uint32_t groups;
  uint32_t row;
  uint32_t input;
  uint32_t width;
  uint32_t stride;
  uint32_t scale_stride;
  uint32_t format;
  uint32_t rows;
};

// output[t][c] += partial[c][t] * partial[scale_row][t] * token_scale[t], in
// fp32, for rows t < rows, setting a status word if one of those values is
// not finite; stride halves per channel of the partial.
struct AneFfnJoinParams {
  uint32_t hidden;
  uint32_t stride;
  uint32_t rows;
  uint32_t scale_row;
};

// LoRA updates in the split (ops::AneFfn, ops::ProjectionLora). The ANE adds
// B (A x) to gate's and up's sums of its channels from the rows t = A x the
// GPU gives it per token scale, times ANE_FFN_LORA_UNIT, and B's rows over
// it, both in fp16. Down's rotated A is further int8 rows of down's weights,
// whose products with the intermediate rows the ANE writes after down's,
// before the intermediate scale; the GPU adds them to its own part of down's
// t before it multiplies B.
#define ANE_FFN_LORA_UNIT 64.0f
// Rows t [rows][rank] in bf16 as the channel-major fp16 input of the ANE,
// each value times scale over its token's scale: stride halves per channel.
struct AneFfnLoraInputParams {
  uint32_t rows;
  uint32_t rank;
  uint32_t stride;
  uint32_t source_stride;  // values of a row of t
  uint32_t source_offset;  // the update's first value in a row of t
  float scale;
};
// `rows` rows of `width` values, source_stride values apart from value
// source_offset of each, into rows [row, row + rows) at columns
// [column, column + width) of rows stride bytes apart: int8 codes as they
// are (ane_ffn_lora_codes), or bf16 values times scale as fp16
// (ane_ffn_lora_rows). With a span, the source's rows are a projection's
// span output channels and row r takes the channel at place r of the ANE's
// surfaces (ane_ffn_descending): gate's and up's rows of B.
struct AneFfnLoraRowsParams {
  uint32_t rows;
  uint32_t width;
  uint32_t source_stride;
  uint32_t source_offset;
  uint32_t stride;
  uint32_t row;
  uint32_t column;
  float scale;
  uint32_t span;
};
// t[token][j] = gpu[token][j] + partial[hidden + j][token] *
// partial[hidden + rank][token] * token_scale[token] * scale, as bf16, for
// tokens < rows; stride halves per channel of the partial.
struct AneFfnLoraDownParams {
  uint32_t hidden;
  uint32_t stride;
  uint32_t rows;
  uint32_t rank;
  float scale;
};

static_assert(sizeof(AneFfnLoraInputParams) == 24, "ANE FFN LoRA input parameters are 24 bytes");
static_assert(sizeof(AneFfnLoraRowsParams) == 36, "ANE FFN LoRA row parameters are 36 bytes");
static_assert(sizeof(AneFfnLoraDownParams) == 20, "ANE FFN LoRA down parameters are 20 bytes");
static_assert(sizeof(AneFfnRotateParams) == 4, "ANE FFN rotate parameters are 4 bytes");
static_assert(sizeof(AneFfnPackParams) == 12, "ANE FFN pack parameters are 12 bytes");
static_assert(sizeof(AneFfnWeightParams) == 32, "ANE FFN weight parameters are 32 bytes");
static_assert(sizeof(AneFfnJoinParams) == 16, "ANE FFN join parameters are 16 bytes");

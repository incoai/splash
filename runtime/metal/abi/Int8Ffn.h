#pragma once

// Parameter layouts and constants of the GPU's int8 prefill FFN
// (ops/Int8Ffn.cpp): int8 activations and weights multiplied by MPP matmul2d
// into int32, both rotated as the ANE FFN split rotates its own
// (metal/abi/AneFfn.h).
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// int8_ffn_gate_up and int8_ffn_down: threadgroups of INT8_FFN_SIMDGROUPS
// simdgroups, each GEMM threadgroup INT8_FFN_SIMDGROUP_ROWS rows per simdgroup
// over INT8_FFN_COLUMNS columns, each conversion threadgroup one weight row
// per simdgroup.
#define INT8_FFN_SIMDGROUP_ROWS 32u
#define INT8_FFN_SIMDGROUPS 4u
#define INT8_FFN_ROWS (INT8_FFN_SIMDGROUP_ROWS * INT8_FFN_SIMDGROUPS)
#define INT8_FFN_THREADS (INT8_FFN_SIMDGROUPS * 32u)
#define INT8_FFN_COLUMNS 64u
// int8_ffn_quantize: one threadgroup of INT8_FFN_QUANTIZE_THREADS per row.
#define INT8_FFN_QUANTIZE_THREADS 256u

// Rotates and quantizes `width` values per row (int8_ffn_quantize).
struct Int8FfnQuantizeParams {
  uint32_t width;
};

// The row scales of a projection's rows over `width` inputs, read from an
// image tensor of GGUF_FMT_* `format` with `groups` groups of 32 inputs per
// row (int8_ffn_row_scale).
struct Int8FfnRowScaleParams {
  uint32_t groups;
  uint32_t width;
  uint32_t format;
};

// A GEMM dispatch over `rows` rows of `inputs` int8 values against `outputs`
// int8 weight rows, on a grid of row_tiles + conversion_columns by
// outputs / INT8_FFN_COLUMNS threadgroups: `row_tiles` tiles of INT8_FFN_ROWS
// rows by INT8_FFN_COLUMNS columns per column tile, then
// `conversion_columns` threadgroups of which the first `conversions` in grid
// order convert INT8_FFN_SIMDGROUPS weight rows each of one projection or two:
// rows of `width` inputs, `converted` rows per projection, from image tensors
// of `groups[i]` groups of 32 inputs per row in format `formats[i]`.
struct Int8FfnGemmParams {
  uint32_t rows;
  uint32_t inputs;
  uint32_t outputs;
  uint32_t row_tiles;
  uint32_t conversion_columns;
  uint32_t conversions;
  uint32_t width;
  uint32_t converted;
  uint32_t groups[2];
  uint32_t formats[2];
};

static_assert(sizeof(Int8FfnQuantizeParams) == 4, "int8 FFN quantize parameters are 4 bytes");
static_assert(sizeof(Int8FfnRowScaleParams) == 12, "int8 FFN row scale parameters are 12 bytes");
static_assert(sizeof(Int8FfnGemmParams) == 48, "int8 FFN GEMM parameters are 48 bytes");

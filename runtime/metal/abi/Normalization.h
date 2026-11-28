#pragma once

// Parameter layout of the LayerNorm (kernels/shared/normalization.metal,
// norm_layer) and its host dispatch (ops/Normalization.hpp).
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// Rows of `width` values normalized over their fp32 mean and variance plus
// `epsilon`, then scaled by the weight and offset by the bias.
struct LayerNormParams {
  uint32_t width;
  float epsilon;
};

static_assert(sizeof(LayerNormParams) == 8,
              "LayerNorm parameters are 8 bytes on both sides");

// A producer's RMS norm epilogue (kernels/common/rms_norm_epilogue.h,
// ops::RmsNormEpilogue): a dispatch of `tiles` output tiles, each storing
// some columns of every one of `rows` bf16 rows of `width` values (a
// multiple of four) and its columns' sum of squares of each row, whose last
// tile writes the rows' weighted RMS norm, the norm's weights F32
// (weights_f32, a GGUF's) or bf16. A decode step's rows at most.
#define RMS_NORM_EPILOGUE_MAXIMUM_ROWS 128u
struct RmsNormEpilogueParams {
  uint32_t rows;
  uint32_t width;
  uint32_t tiles;
  uint32_t weights_f32;
};

static_assert(sizeof(RmsNormEpilogueParams) == 16,
              "RMS norm epilogue parameters are 16 bytes on both sides");

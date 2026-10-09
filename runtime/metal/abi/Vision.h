#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The epsilon of the vision tower's LayerNorms, which a vision GGUF must
// state; unsuffixed, as SPLASH_RMS_EPSILON.
#define SPLASH_VISION_NORM_EPSILON 1e-6

struct VisionGemmParams {
  uint32_t output_size;
  uint32_t input_size;
};

static_assert(sizeof(VisionGemmParams) == 8,
              "Vision GEMM parameters are 8 bytes on both sides");

struct VisionNormParams {
  uint32_t width;
};

static_assert(sizeof(VisionNormParams) == 4,
              "Vision LayerNorm parameters are 4 bytes on both sides");

struct VisionGridParams {
  uint32_t grid_height;
  uint32_t grid_width;
};

static_assert(sizeof(VisionGridParams) == 8,
              "Vision grid parameters are 8 bytes on both sides");

struct VisionQkvParams {
  uint32_t tokens;
  uint32_t padded_tokens;
};

static_assert(sizeof(VisionQkvParams) == 8,
              "Vision QKV parameters are 8 bytes on both sides");

struct VisionAttentionParams {
  uint32_t tokens;
  uint32_t padded_tokens;
  float scale;
};

static_assert(sizeof(VisionAttentionParams) == 12,
              "Vision attention parameters are 12 bytes on both sides");

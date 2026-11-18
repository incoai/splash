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
// The epsilon of RADIO's merger RMS norm (HF's projector default; the
// mmproj does not state it).
#define SPLASH_VISION_MERGER_RMS_EPSILON 1e-5

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

// Query rows of one head a vision_attention threadgroup takes, and its
// threads: four simdgroups whose softmax gives each row four lanes; a
// static_assert in shared/vision.metal ties the two literals together.
#define SPLASH_VISION_ATTENTION_ROWS 32u
#define SPLASH_VISION_ATTENTION_THREADS 128u

struct VisionAttentionParams {
  uint32_t tokens;
  uint32_t padded_tokens;
  float scale;
};

static_assert(sizeof(VisionAttentionParams) == 12,
              "Vision attention parameters are 12 bytes on both sides");

// A pruned video's EVS (ops::VideoPruning): `tubelets` tubelets of `rows`
// merged rows each, `stride` rows apart in the encoded rows, `width` bf16
// values wide; it keeps `kept` rows and lays them out among `text_tokens`
// text rows.
struct VideoPruningParams {
  uint32_t tubelets;
  uint32_t rows;
  uint32_t stride;
  uint32_t width;
  uint32_t kept;
  uint32_t text_tokens;
};

static_assert(sizeof(VideoPruningParams) == 24,
              "Video pruning parameters are 24 bytes on both sides");

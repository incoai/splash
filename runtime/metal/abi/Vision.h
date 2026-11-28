#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The epsilon of each vision tower's LayerNorms, which its GGUF must state:
// Qwen3.5's and Inkling 2's; unsuffixed, as SPLASH_RMS_EPSILON.
#define SPLASH_VISION_NORM_EPSILON 1e-6
#define SPLASH_INKLING_VISION_NORM_EPSILON 1e-5
// The base of Inkling 2's axial RoPE, transformers' default, which a
// config states only as rope_parameters.rope_theta.
#define SPLASH_INKLING_VISION_ROPE_THETA 10000.0

struct VisionGridParams {
  uint32_t grid_height;
  uint32_t grid_width;
};

static_assert(sizeof(VisionGridParams) == 8,
              "Vision grid parameters are 8 bytes on both sides");

// Inkling 2's patchify: its grid in patches and the image's own pixels at its
// top left; the rest is its padding, which the encoder takes as the reference
// pads it (-1 on the 0-255 scale), where the frame's bytes hold zero.
struct InklingPatchifyParams {
  uint32_t grid_height;
  uint32_t grid_width;
  uint32_t content_height;
  uint32_t content_width;
};

static_assert(sizeof(InklingPatchifyParams) == 16,
              "Inkling patchify parameters are 16 bytes on both sides");

struct VisionQkvParams {
  uint32_t tokens;
  uint32_t padded_tokens;
};

static_assert(sizeof(VisionQkvParams) == 8,
              "Vision QKV parameters are 8 bytes on both sides");

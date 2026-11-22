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

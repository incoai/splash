#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct FlashMoERouteParams {
  uint32_t rows;
  uint32_t experts;
  uint32_t selections;
  uint32_t normalize_top_k;
};

struct FlashMoEPointwiseParams {
  uint32_t rows;
  uint32_t width;
  uint32_t selections;
  uint32_t experts;
};

static_assert(sizeof(FlashMoERouteParams) == 16,
              "Flash MoE routing parameters are four uint32 values");
static_assert(sizeof(FlashMoEPointwiseParams) == 16,
              "Flash MoE pointwise parameters are four uint32 values");

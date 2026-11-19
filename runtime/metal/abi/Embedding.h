#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#include "metal/abi/ExecutionGeometry.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct DraftMaskRowsParams {
  uint32_t width;
  uint32_t rows;
};

static_assert(sizeof(DraftMaskRowsParams) == 8,
              "Draft mask row parameters are 8 bytes on both sides");

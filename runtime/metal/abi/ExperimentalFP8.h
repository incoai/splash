#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct ExperimentalFP8Params {
  uint32_t output_size;
  uint32_t input_size;
  uint32_t rows;
  uint32_t scale_row_stride;
  uint32_t epilogue;
};
static_assert(sizeof(ExperimentalFP8Params) == 20);

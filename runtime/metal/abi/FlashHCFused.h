#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct FlashHCFusedMatrix {
  uint32_t input_size;
  uint32_t output_size;
  uint32_t bits;
  uint32_t group_size;
  uint64_t weight_row_stride_bytes;
  uint64_t parameter_row_stride_bytes;
};

struct FlashHCFusedParams {
  uint32_t rows;
  uint32_t width;
  uint32_t streams;
  uint32_t lowrank;
  uint32_t has_injection;
  uint32_t write_raw_up;
  uint32_t arithmetic_mode; // 0 literal coefficients; 1 experimental grouped.
  uint32_t simdgroups;
  uint32_t norm_is_float;
  uint32_t norm_convention;
  uint32_t reserved0;
  uint32_t reserved1;
  float norm_epsilon;
  uint32_t reserved2;
  uint32_t reserved3;
  uint32_t reserved4;
  FlashHCFusedMatrix down;
  FlashHCFusedMatrix injection;
  FlashHCFusedMatrix up;
};

static_assert(sizeof(FlashHCFusedMatrix) == 32);
static_assert(sizeof(FlashHCFusedParams) == 160);

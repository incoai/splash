#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The loader audits which convention the checkpoint actually stores. Scale
// formation remains FP32 even when the raw norm weight is BF16.
enum FlashHCNormConvention : uint32_t {
  FlashHCOnePlusWeight = 0,
  FlashHCDirectGamma = 1,
};

struct FlashHCParams {
  uint32_t rows;
  uint32_t width;
  uint32_t streams;
  uint32_t norm_convention;
  float epsilon;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
};

static_assert(sizeof(FlashHCParams) == 32,
              "Flash HC parameters are 32 bytes on host and Metal");

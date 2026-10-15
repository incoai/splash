#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// A verify of `rows` rows per lane, lane by lane.
struct VerifyInputBatchParams {
  uint32_t vocabulary;
  uint32_t rows;
};

static_assert(sizeof(VerifyInputBatchParams) == 8,
              "Verify input parameters are 8 bytes on both sides");

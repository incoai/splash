#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct VerifyInputBatchParams {
  uint32_t vocabulary;
};

static_assert(sizeof(VerifyInputBatchParams) == 4,
              "Verify input parameters are 4 bytes on both sides");

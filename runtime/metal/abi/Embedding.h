#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct VerifyInputBatchParams {
  uint32_t vocabulary;
  uint32_t draft_rows;
};

static_assert(sizeof(VerifyInputBatchParams) == 8,
              "Verify input parameters are 8 bytes on both sides");

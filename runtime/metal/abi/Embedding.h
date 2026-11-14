#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct VerifyInputBatchParams {
  uint32_t vocabulary;
  uint32_t rows;
  // The batch's lanes, for the sub-pass-ordered copy of the same rows: a
  // 16-row verify's target layers run two 8-row sub-passes, each gathering
  // its rows from a contiguous slice.
  uint32_t lanes;
};

static_assert(sizeof(VerifyInputBatchParams) == 12,
              "Verify input parameters are 12 bytes on both sides");

struct DraftMaskRowsParams {
  uint32_t width;
  uint32_t rows;
};

static_assert(sizeof(DraftMaskRowsParams) == 8,
              "Draft mask row parameters are 8 bytes on both sides");

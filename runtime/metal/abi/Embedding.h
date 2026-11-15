#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#include "metal/abi/ExecutionGeometry.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// A verify step's input tokens, lane l's rows[l] rows from row first_row[l]:
// its anchor, row 0 of its draft input rows, then as many of its draft's
// proposals as follow, each clamped into the vocabulary.
struct VerifyInputBatchParams {
  uint32_t vocabulary;
  uint32_t first_row[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t rows[SPLASH_MAXIMUM_BATCH_WIDTH];
};

static_assert(sizeof(VerifyInputBatchParams) == 36,
              "Verify input parameters are 36 bytes on both sides");

struct DraftMaskRowsParams {
  uint32_t width;
};

static_assert(sizeof(DraftMaskRowsParams) == 4,
              "Draft mask row parameters are 4 bytes on both sides");

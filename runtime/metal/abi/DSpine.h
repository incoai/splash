#pragma once
#include "metal/abi/Sampling.h"

// The experimental backend executes eight or sixteen rows of a trained block.
// Bridge weights and activations stay BF16; refinement accumulates in FP32.
struct DSpineParams {
  uint32_t hidden;
  uint32_t code;
  uint32_t rank;
  uint32_t rows;
};

struct DSpineRefineParams {
  DSpineParams shape;
  SelectorBatchParams selection;
  uint32_t position;
};

struct DSpineGatherParams {
  uint32_t rows;
  uint32_t width;
};

static_assert(sizeof(DSpineParams) == 16, "DSpine bridge parameters are 16 bytes");
static_assert(sizeof(DSpineRefineParams) == 64, "DSpine refinement parameters are 64 bytes");
static_assert(sizeof(DSpineGatherParams) == 8, "DSpine gather parameters are 8 bytes");

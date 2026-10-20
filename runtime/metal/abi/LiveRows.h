#pragma once

#include "metal/abi/ExecutionGeometry.h"
#ifndef __METAL_VERSION__
#include <stdint.h>
#endif

// GPU-produced budget of one lane's fixed eight-row verify block. Row zero
// is always live. Confidence is a heuristic, never an acceptance probability.
struct VerifyLiveRows {
  uint32_t count;
  float prefix_confidence[SPLASH_DRAFT_PROPOSAL_TOKENS];
};
static_assert(sizeof(VerifyLiveRows) == 32,
              "Verify live rows are 32 bytes on both sides");

struct DraftLiveRowsParams {
  uint32_t lanes;
  uint32_t sampling_mask;
  float threshold;
};
static_assert(sizeof(DraftLiveRowsParams) == 12,
              "Draft live-row parameters are 12 bytes on both sides");

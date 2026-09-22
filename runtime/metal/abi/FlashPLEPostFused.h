#pragma once

#include "metal/abi/FlashPLE.h"
#include "metal/abi/FlashHC.h"

// Three raw gamma dtypes are independent. Every normalized operand and gate
// array boundary remains BF16; zero-centered 1+w formation remains F32.
enum FlashPLEPostFusedFlags : uint32_t {
  FlashPLEPostFusedKeyF32 = 1u,
  FlashPLEPostFusedQueryF32 = 2u,
  FlashPLEPostFusedConvolutionF32 = 4u,
};

struct FlashPLEPostFusedParams {
  FlashPLEPostParams geometry;
  float epsilon;
  uint32_t norm_convention;
  uint32_t norm_weight_flags;
  uint32_t norm_threads;
  uint32_t gate_threads;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
};

static_assert(sizeof(FlashPLEPostFusedParams) == 64,
              "Flash PLE fused post parameters are 64 bytes");

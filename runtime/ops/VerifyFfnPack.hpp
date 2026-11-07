#pragma once
#include "metal/abi/VerifyFfnPack.h"
#include "ops/Linear.hpp"
#ifndef SPLASH_EXPERIMENTAL_FFN_PACK_MODE
#define SPLASH_EXPERIMENTAL_FFN_PACK_MODE 0
#endif
static_assert(SPLASH_EXPERIMENTAL_FFN_PACK_MODE >= 0 && SPLASH_EXPERIMENTAL_FFN_PACK_MODE <= 2);
namespace splash::ops {
// Experimental complete FFN workspace; rows keep the target's original layout.
struct VerifyFfnPackBuffers final {
  metal::MetalBuffer control, input, residual, intermediate, output;
};
// Returns false without dispatching when the original plans/weights are unsupported.
// Both GPU-selected paths preserve the supplied wide K splits and walk.
[[nodiscard]] bool addVerifyFfnPack(metal::CommandGraph &graph, const SwiGluProjections &ffn,
                                    metal::MetalBuffer normalized, metal::MetalBuffer residual,
                                    metal::MetalBuffer output, metal::MetalBuffer gateScratch,
                                    metal::MetalBuffer wideIntermediate, LinearScratch scratch,
                                    metal::MetalBuffer verifyRows, const VerifyFfnPackBuffers &packed,
                                    const LinearPlan &upPlan, const LinearPlan &downPlan, uint32_t lanes);
} // namespace splash::ops

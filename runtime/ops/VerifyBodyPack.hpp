#pragma once
#include "ops/Linear.hpp"
#ifndef SPLASH_EXPERIMENTAL_VERIFY_BODY
#define SPLASH_EXPERIMENTAL_VERIFY_BODY 0
#endif
static_assert(SPLASH_EXPERIMENTAL_VERIFY_BODY == 0 || SPLASH_EXPERIMENTAL_VERIFY_BODY == 1);
namespace splash::ops {
struct VerifyBodyPackBuffers final {
  metal::MetalBuffer input, residual, output;
};
// The host-only contract used before restricting any proposal's prefix.
[[nodiscard]] bool supportsVerifyBodyPack(const Projection &projection, const LinearPlan &widePlan,
                                          bool residual, uint32_t lanes);
// Private target-verify path: caller must cap selector maximumRows to four.
// No dispatch is added when the host projection/plan is unsupported.
[[nodiscard]] bool addVerifyBodyPack(const Linear &linear, metal::CommandGraph &graph,
                                     const Projection &projection, const LinearPlan &widePlan,
                                     metal::MetalBuffer input, metal::MetalBuffer residual,
                                     metal::MetalBuffer output, LinearScratch scratch,
                                     metal::MetalBuffer verifyRows, const VerifyBodyPackBuffers &packed,
                                     uint32_t lanes);
} // namespace splash::ops

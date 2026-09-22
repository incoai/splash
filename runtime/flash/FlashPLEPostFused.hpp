#pragma once

#include "flash/FlashPLE.hpp"

namespace splash::flash {

inline constexpr const char *kFlashPLEPostFusedSemantics =
    "source-rms-partitions-bf16-array-boundaries-ordered-ple-state-v1";

// Isolated post-projection candidate. The four existing scratch arrays retain
// the source's BF16 boundaries and normalizedConvolution remains available for
// retained-prefix restoration. Only exact injectedOutput==hyperInput aliasing
// is supported. State updates remain a separate ordered dispatch.
void addPLEPostProjectFused(
    metal::CommandGraph &graph, const FlashPLEWeights &weights,
    metal::MetalBuffer hyperInput, metal::MetalBuffer keyProjected,
    metal::MetalBuffer valueProjected, const FlashPLEPostScratch &scratch,
    metal::MetalBuffer convolutionState, metal::MetalBuffer pleOutput,
    metal::MetalBuffer injectedOutput, metal::MetalBuffer diagnostics,
    FlashPLEGeometry geometry, metal::MetalBuffer mask = {});

} // namespace splash::flash

#pragma once

#include "FlashGDN.hpp"

#include <array>

namespace splash::flash {

// Exactly one token/request for1..4request lanes. Projected BF16 input/output
// planes retain [lane,1,width] layout, but each convolution/recurrent view is
// the original request's layer state. No state plane, pointer table, pack or
// scatter is allocated. Unused array entries may be empty.
void addGDNFusedSeparateStates(
    metal::CommandGraph &graph, const FlashGDNWeights &weights,
    const FlashGDNBuffers &buffers,
    const std::array<FlashGDNState, 4> &states, uint32_t lanes,
    float normEpsilon = 1e-6f);

} // namespace splash::flash

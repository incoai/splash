#pragma once

// The slot tables a MoE target's blocks read their routed experts through
// (ops::BlockMoeWeights::slotTable): the kernels locate routed expert e at
// slot slotTable[e] of the block's routed segments.

#include "metal/MetalBackend.hpp"

#include <cstdint>

namespace splash::model {

// The slot table of a block whose routed experts are all resident: expert e
// in slot e.
[[nodiscard]] metal::MetalBuffer residentExpertSlots(metal::MetalBackend &backend, uint32_t experts);

} // namespace splash::model

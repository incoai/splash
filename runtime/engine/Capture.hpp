#pragma once

#include <cstdint>

namespace splash::engine {

// A state boundary a prefill command captures
// (ModelBatchItem::stateCaptures), or the one it ends at whose state apply
// takes from the lane after it (fromLane).
struct LaneCapture final {
  uint32_t tokens = 0;
  // Room for the state's buffers came from reclaiming cached memory; its
  // publication counts as recycled (recycledStatePublications).
  bool recycled = false;
  // Its buffers were refused before the command while the lane held a
  // checkpoint (reserveCapture): the command ends at it without capturing
  // it.
  bool fromLane = false;
};

} // namespace splash::engine

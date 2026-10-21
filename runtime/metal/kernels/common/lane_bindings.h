#pragma once

#include "metal/abi/ExecutionGeometry.h"

// A batched kernel binds a per-lane tensor once per lane
// (ops::appendLaneBindings): lane `lane`'s binding of the four. A macro, not
// a function: the compiler keeps the chain as selects only where it is
// written in place, and turns a function's into branches.
static_assert(SPLASH_MAXIMUM_BATCH_WIDTH == 4, "one binding per lane");
#define SPLASH_LANE_BINDING(lane, lane0, lane1, lane2, lane3)                  \
  ((lane) == 0 ? (lane0)                                                       \
               : ((lane) == 1 ? (lane1) : ((lane) == 2 ? (lane2) : (lane3))))

#pragma once

#include "metal/abi/ExecutionGeometry.h"

// A batched kernel binds a per-lane tensor once per lane
// (ops::appendLaneBindings): lane `lane`'s binding of the eight. A macro, not
// a function: the compiler keeps the chain as selects only where it is
// written in place, and turns a function's into branches. Eight lanes are
// the most this binding form holds: the kernel that binds the most per-lane
// tensors (block_indexer_commit, three) reaches buffer index 28 of Metal's 31.
static_assert(SPLASH_MAXIMUM_BATCH_WIDTH == 8, "one binding per lane");
#define SPLASH_LANE_BINDING(lane, lane0, lane1, lane2, lane3, lane4, lane5, lane6, lane7)                      \
  ((lane) == 0   ? (lane0)                                                                                     \
   : (lane) == 1 ? (lane1)                                                                                     \
   : (lane) == 2 ? (lane2)                                                                                     \
   : (lane) == 3 ? (lane3)                                                                                     \
   : (lane) == 4 ? (lane4)                                                                                     \
   : (lane) == 5 ? (lane5)                                                                                     \
   : (lane) == 6 ? (lane6)                                                                                     \
                 : (lane7))

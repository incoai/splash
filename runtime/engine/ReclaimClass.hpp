#pragma once

#include <cstdint>

namespace splash::engine {

// The highest class a reclaim step may take (the class rule). Ordinary:
// checkpoints, ordinary states and KV, for growth that can wait or yield
// instead: a start a resident lane holds back, and the lane that would
// yield first (Engine::laneToYield) while another lane is resident, and
// for a pressure pass that nothing waits for. InUse: after all of those,
// the states unfinished requests use and the KV they restore through, for
// other running work (any other resident lane, or a start no resident lane
// precedes) and for a pressure pass a request waits for
// (MemoryReclaimDirective::upTo).
enum class ReclaimClass : uint8_t { Ordinary, InUse };

} // namespace splash::engine

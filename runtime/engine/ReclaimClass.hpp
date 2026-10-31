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
// (MemoryReclaimDirective::upTo). The expert store's chunks above its floor
// rank between the two, just above a Resident lane's conversion to Spill on
// a target whose KV can live on the SSD cache (Engine::escalate): a step
// takes them once nothing Ordinary is left and no lane converts, and before
// anything InUse, whatever its class, and so does the prompt's replay point
// before another request's state in use; a pressure pass takes them
// only while the host is short (MemoryReclaimDirective::hostShort), so that a
// pass a request waits for under macOS's warning alone takes what is in use
// while the store keeps them. Growth that only shortens a command or leaves
// memory for later (an optional allocation) converts no lane and takes
// neither them nor anything InUse, and a junction or a checkpoint never
// takes them.
enum class ReclaimClass : uint8_t { Ordinary, InUse };

} // namespace splash::engine

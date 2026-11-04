#pragma once

#include <cstdint>

namespace splash::engine {

// A claim's reach (ClaimSpec::reach): the highest class of cached memory
// its free() may take. Ordinary: checkpoints, ordinary states and KV no
// request in service restores through. InUse: those too, then the states
// unfinished requests use and the KV they restore through, oldest first. A
// request in service reaches InUse unless its lane is the one that would
// yield first beside another resident lane (Engine::growthClass), and so
// does the prompt's replay point; a start, an optional allocation, growth
// and the host claim short of macOS's critical pressure reach only Ordinary.
// The reach bounds the cache alone: whether a pool's unit, a conversion or
// the pass command may go is the claim's class's rule, and what goes first
// among what it may take is the ledger's fixed order (MemoryLedger::free).
enum class ReclaimClass : uint8_t { Ordinary, InUse };

} // namespace splash::engine

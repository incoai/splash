#pragma once

// The control pass's host claim as a test drives it on its engine's governor
// (MemoryGovernor::controlPass): its rights follow the pressure (takeRights),
// and it takes in its scope, as a lane's release allocates the Index extents
// its pages move into.

#include "engine/MemoryGovernor.hpp"
#include "model/SlotFile.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>

namespace splash::test {

inline engine::FreeResult hostPass(engine::MemoryGovernor &governor,
                                   uint64_t bytes = std::numeric_limits<uint64_t>::max(), bool critical = false) {
  engine::MemoryLedger &ledger = governor.ledger();
  const engine::ClaimSpec spec{.cls = engine::ClaimClass::Host};
  const uint64_t found = ledger.available(spec, {.bytes = bytes}, 0, critical ? engine::MemoryPressure::Critical : engine::MemoryPressure::Normal);
  const engine::ClaimId claim = ledger.open(spec);
  engine::FreeResult result;
  {
    const engine::MemoryGovernor::Scope scope(governor, claim);
    result = ledger.free(claim, {.bytes = std::min(bytes, found)}, 0, critical ? engine::MemoryPressure::Critical : engine::MemoryPressure::Normal);
  }
  ledger.close(claim);
  return result;
}

// The pass under macOS's critical pressure: the margin and what the host
// refused, of everything the host claim may take.
inline engine::FreeResult criticalPass(engine::MemoryGovernor &governor) {
  return hostPass(governor, engine::kHostMarginBytes + governor.ledger().hostRefusedWaitingBytes(), true);
}

// A disk quota far past what any test writes, for a cache with a tier, as
// every production one has (model::DiskBudget).
inline std::shared_ptr<model::DiskBudget> ampleDiskBudget() {
  return std::make_shared<model::DiskBudget>(uint64_t{1} << 40);
}

} // namespace splash::test

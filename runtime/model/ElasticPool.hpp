#pragma once

#include "metal/Allocation.hpp"

#include <cstdint>
#include <string_view>
#include <vector>

namespace splash::model {

struct PoolState final {
  std::string_view name;          // a literal, for /status and logs; nothing branches on it
  uint64_t floorBytes = 0;        // never released below now: lanes' floors, a held staging, the store's floor,
                                  // entries in use
  uint64_t currentBytes = 0;      // allocated now
  uint64_t capBytes = 0;          // the bound grow() approaches; == currentBytes for a pool that grows only as it
                                  // is used
  uint64_t releaseUnitBytes = 0;  // what release(1) frees now; 0 at the floor
  std::vector<uint64_t> releaseUnits;  // what each release(1) frees from now to the floor, in turn, where the
                                       // units differ (the store's chunks); empty: each releaseUnitBytes
  uint64_t growUnitBytes = 0;     // what grow() takes next; 0 when the engine cannot grow it now (at its cap,
                                  // released, a pool that grows only as it is used)
  bool rebuildOnly = false;       // a unit it gives back costs only its rebuild once something uses it again: it
                                  // grows only as it is used (the vision arena, images' rows, an idle staging),
                                  // and what live requests hold is its floor; false for a pool the lanes in
                                  // service read through (the store, the hot block cache)
};

// A memory of the model's that follows the engine's memory. The model owns the mechanism; the engine alone
// decides when a unit goes or comes.
// Threads: state() on the engine thread at any time, a command in flight included: they read
// only what the mechanism publishes and never wait behind a load. release() and grow() on the engine thread
// with no command in flight. A failure inside a mechanism throws to the engine's failure boundary.
class ElasticPool {
public:
  virtual ~ElasticPool() = default;
  [[nodiscard]] virtual PoolState state() const = 0;
  // Whole units, the fewest whose bytes reach `bytes`, never below the floor; the bytes given back, 0 at the
  // floor. 1 gives one unit; UINT64_MAX everything above the floor (the store in one compaction per size).
  [[nodiscard]] virtual uint64_t release(uint64_t bytes) = 0;
  // One unit under `admit`; a refusal changes nothing and is returned. Granted with nothing taken when
  // growUnitBytes is 0.
  [[nodiscard]] virtual metal::AllocationResult grow(const metal::AllocationAdmission &admit) = 0;
};

} // namespace splash::model

#pragma once

#include "engine/MemoryGovernor.hpp"
#include "engine/NativeRuntime.hpp"
#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace splash::engine {

// How long after reclaim last gave back a chunk of the expert store the
// control pass waits before it takes one back: a host short once is likely
// short again soon, and each chunk that goes copies its experts down and
// evicts some, and each that comes back is refilled from the SSD. A chunk
// taken back that goes again within the cool-down was taken into memory the
// engine still needed, and the cache the engine then wrote to the SSD cache
// for it is read back too: the next cool-down is twice as long, up to the
// limit. One that stays for a whole cool-down restores it.
inline constexpr double kExpertGrowthCooldownMilliseconds = 30000.0;
inline constexpr double kExpertGrowthCooldownLimitMilliseconds = 8 * kExpertGrowthCooldownMilliseconds;

// Emits only transitions; retry counts and queue depth do not produce logs.
class MemoryStatusReporter final {
public:
  [[nodiscard]] std::string update(const ResourceWaitSnapshot &wait,
                                    bool hostGrowthAllowed);
private:
  unsigned state_ = 0;
};

// The engine's memory control between commands: it releases the weights of
// an engine left without a request (NativeRuntime::releaseIdleWeights),
// records the system's pressure, logs a change in what requests wait for,
// and runs the reclaim pass the pressure policy asks for, telling the
// governor what it found. A pass the policy asks nothing of takes back one
// chunk of the expert store toward its cap (NativeRuntime::growExperts)
// while memory allows it: no request waits for memory, the host lets growth
// go on (MemoryGovernorSnapshot::hostGrowthAllowed), and the cool-down has
// passed since the pass first found a chunk given back
// (kExpertGrowthCooldownMilliseconds, longer after a chunk taken back went
// again). The chunk takes from the host only what its headroom
// holds beyond the recovery margin, so that the chunk does not bring the
// host back to the margin where reclaim would take it again, and from the
// engine only the room its limit has free. With no request in the engine
// the pass exchanges cache for what the chunk lacks of either, the store
// ranking above the cache nothing in use holds, but only cache that pays
// for it at once: the engine returns that much to the system, or nothing
// when what leaves without a copy to the SSD cache falls short
// (NativeRuntime::reclaimForExperts), and the governor's admission then
// counts what it returned as the host's already
// (MemoryGovernor::setGrowthCredit). Beside any request the chunk takes
// only free room.
class MemoryControl final {
public:
  MemoryControl(MemoryGovernor &governor, metal::MetalBackend &backend,
                NativeRuntime &loop) noexcept
      : governor_(governor), backend_(backend), loop_(loop) {}

  // One memory-control pass at a command-free point. True while transfers
  // in flight hold back part of the target: the transport runs it again at
  // the next command-free point.
  [[nodiscard]] bool run(MemoryPressure pressure);

private:
  // Records when a pass first finds a chunk of the expert store that reclaim
  // gave back.
  void noteExpertReleases(double nowMilliseconds);
  void growExperts(const MemoryGovernorSnapshot &memory, bool requestWaiting,
                   double nowMilliseconds);

  MemoryGovernor &governor_;
  metal::MetalBackend &backend_;
  NativeRuntime &loop_;
  MemoryStatusReporter reporter_;
  MemoryPressurePolicy policy_;
  uint64_t expertReleases_ = 0;
  std::optional<double> expertReleaseMilliseconds_;
  // The cool-down now, and when the pass last took a chunk back, until a
  // chunk goes or it has stayed for the cool-down.
  double expertCooldownMilliseconds_ = kExpertGrowthCooldownMilliseconds;
  std::optional<double> expertGrowthMilliseconds_;
};

} // namespace splash::engine

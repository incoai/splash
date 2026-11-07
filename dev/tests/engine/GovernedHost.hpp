#pragma once

// A test's own engine under the real MemoryGovernor over FakeMemory's host,
// for tests that build the engine and its fakes themselves (EngineHarness
// builds its own).

#include "FakeMemory.hpp"
#include "TestEngine.hpp"
#include "TestExecutor.hpp"
#include "TestKvPool.hpp"
#include "engine/Engine.hpp"
#include "engine/MemoryGovernor.hpp"
#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <optional>
#include <utility>

namespace splash::test {

// The governor of `limits` over FakeMemory's host and the stand-in backend.
// connect() completes an engine config wired to it (connectToGovernor),
// through() has a KV storage allocate its extents through it, bookStates()
// has the model allocate its states through it, and reserve() asks it to
// admit bytes as one of the model's allocations would, holding none, for a
// fake that takes its refusals from it. The host is ample until
// pause(): other applications then leave the host less spare memory than the
// governor's margin, where it refuses what no request in service needs and
// growth pauses. setPressure() hands the governor macOS's level, as the
// control pass does.
class GovernedHost final {
public:
  explicit GovernedHost(const GovernorLimits &limits = {})
      : memory(FakeMemory::Config{}), backend(""),
        governor(backend, limits.limitBytes, limits.hostReserveBytes, limits.hostSpareReserveBytes,
                 memory.hostMemory(), limits.untrackedReserveBytes),
        spareReserveBytes_(limits.hostSpareReserveBytes), ampleOtherBytes_(memory.otherBytes()) {}
  GovernedHost(const GovernedHost &) = delete;
  GovernedHost &operator=(const GovernedHost &) = delete;

  [[nodiscard]] engine::EngineConfig connect(engine::EngineConfig config) {
    engine::connectToGovernor(config, governor);
    return test::engineConfig(std::move(config));
  }
  void through(TestKvStorage &storage) { storage.through(memory, governor.allocationAdmission()); }
  [[nodiscard]] metal::AllocationResult reserve(uint64_t bytes) {
    return governor.allocationAdmission()(bytes, [] {});
  }
  // The model allocates its states' buffers through the governor, held as
  // "states" until a reclaim gives them back (Executor::memory), so that
  // the governor sees the memory a cached state's eviction frees.
  void bookStates(Executor &executor) {
    executor.memory = &memory;
    executor.admission = governor.allocationAdmission();
  }
  // Each start of the model (Executor::begin) allocates a lane's `bytes`
  // through the governor, refused with its cause.
  void admitBegins(Executor &executor, uint64_t bytes) {
    executor.beginGrowthBlocked = [this, &executor, bytes] {
      executor.beginAllocationFailure = reserve(bytes).failure;
      return executor.beginAllocationFailure != metal::AllocationFailure::None;
    };
  }
  // Paused, other applications leave half the margin above the spare
  // reserve; else what they held at the start.
  void pause(bool paused) {
    memory.setHost(paused ? memory.hostBytes() - spareReserveBytes_ - engine::kHostMarginBytes / 2
                          : ampleOtherBytes_,
                   memory.pressure());
    paused_ = paused;
    static_cast<void>(governor.evaluate());
  }
  [[nodiscard]] bool paused() const noexcept { return paused_; }
  void setPressure(engine::MemoryPressure pressure) {
    memory.setHost(memory.otherBytes(), pressure);
    governor.setPressure(pressure);
    static_cast<void>(governor.evaluate());
  }
  // From now on the limit leaves `bytes` of room beside what the engine
  // holds, as if the engine held the rest; none: the limit leaves what it
  // did. A fake's refusal that asks the governor (reserve) then lacks what
  // it asks beyond that room, which is what the ledger frees for it.
  void leaveRoom(std::optional<uint64_t> bytes) {
    memory.release("limit", memory.held("limit"));
    if (bytes)
      memory.hold("limit", governor.snapshot().headroomBytes - *bytes);
  }

  FakeMemory memory;
  metal::MetalBackend backend;
  engine::MemoryGovernor governor;

private:
  uint64_t spareReserveBytes_ = 0;
  uint64_t ampleOtherBytes_ = 0;
  bool paused_ = false;
};

} // namespace splash::test

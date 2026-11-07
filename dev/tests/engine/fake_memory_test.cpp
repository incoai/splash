// The test fakes the memory tests stand on: FakeMemory and its host, the fake elastic pool, the fakes that allocate
// through FakeMemory, and the checks the harness and the loop fixture run.

#include "EngineHarness.hpp"
#include "FakeElasticPool.hpp"
#include "FakeMemory.hpp"
#include "LoopFixture.hpp"
#include "TestAdmission.hpp"
#include "TestChecks.hpp"
#include "TestExecutor.hpp"
#include "TestKvPool.hpp"
#include "TestMetalMemory.hpp"
#include "TestRegistry.hpp"
#include "engine/KvPool.hpp"
#include "engine/MemoryGovernor.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <vector>

using namespace splash;

namespace {

using test::FakeElasticPool;
using test::FakeMemory;
using test::HostEvent;
using test::rejects;
using test::require;

constexpr uint64_t kGiB = 1ULL << 30;
constexpr uint64_t kMiB = 1ULL << 20;

// A host of 16 GiB of which other applications hold 4.
FakeMemory::Config sixteen(std::vector<HostEvent> script = {}) {
  return {.hostBytes = 16 * kGiB, .otherBytes = 4 * kGiB, .script = std::move(script)};
}

std::optional<engine::HostMemory> host(FakeMemory &memory) { return memory.hostMemory()(); }

// Over seeded sequences of allocations, holds and releases, FakeMemory's holders, total, peak, record and the
// backend's statistics stay consistent.
void testTheBooksBalanceOverSeededSequences() {
  const std::array<const char *, 4> holders{"kv", "states", "experts", "spilled_reads"};
  const metal::AllocationAdmission refuse = [](uint64_t, const std::function<void()> &) {
    return metal::AllocationResult(metal::AllocationFailure::EngineBudget);
  };
  for (uint64_t seed = 1; seed <= 10'000; ++seed) {
    std::mt19937_64 random(seed);
    FakeMemory memory(sixteen());
    metal::MetalBackend backend("");
    std::optional<engine::MemoryGovernor> governor = test::governorOver(backend, memory, {.limitBytes = 6 * kGiB});
    const test::ClaimScope claim(*governor, engine::ClaimClass::InService);
    std::map<std::string, uint64_t> expected;
    uint64_t peak = 0;
    for (int step = 0; step < 40; ++step) {
      const std::string holder = holders[random() % holders.size()];
      const uint64_t bytes = (1 + random() % 512) * kMiB;
      const size_t changes = memory.changes().size();
      switch (random() % 5) {
      case 0:
      case 1: {
        const uint64_t choice = random() % 4;
        const metal::AllocationAdmission admission = choice == 0   ? refuse
                                                     : choice == 1 ? metal::AllocationAdmission(test::admitAll)
                                                                   : governor->allocationAdmission();
        if (memory.allocate(admission, holder, bytes))
          expected[holder] += bytes;
        else
          require(memory.changes().size() == changes, "a refused allocation changed the books");
        break;
      }
      case 2:
        memory.hold(holder, bytes);
        expected[holder] += bytes;
        break;
      default:
        if (bytes <= expected[holder]) {
          memory.release(holder, bytes);
          expected[holder] -= bytes;
        } else {
          rejects([&] { memory.release(holder, bytes); }, "released more", "a release past a holder's bytes passed");
          require(memory.changes().size() == changes, "a refused release changed the books");
        }
      }
      uint64_t total = 0;
      for (const char *name : holders) {
        require(memory.held(name) == expected[name], "seed " + std::to_string(seed) + ": a holder's bytes drifted");
        total += expected[name];
      }
      peak = std::max(peak, total);
      require(memory.total() == total && memory.peak() == peak, "the total is not the holders' sum");
      memory.check();
      require(memory.changes().empty() || memory.changes().back().totalAfter == total,
              "a change does not record the total after it");
    }
  }
}

// A refusal through the real governor, at the engine's limit or the host's, holds nothing and leaves no
// reservation.
void testARefusalThroughTheGovernorHoldsNothing() {
  FakeMemory memory(sixteen({{100.0, 10 * kGiB, engine::MemoryPressure::Normal}}));
  metal::MetalBackend backend("");
  std::optional<engine::MemoryGovernor> governor = test::governorOver(backend, memory, {.limitBytes = 4 * kGiB});
  const test::ClaimScope claim(*governor, engine::ClaimClass::InService);
  require(static_cast<bool>(memory.allocate(governor->allocationAdmission(), "kv", 3 * kGiB)),
          "an allocation within the limit and the host's room was refused");
  const auto refused = [&](uint64_t bytes, metal::AllocationFailure cause) {
    const size_t changes = memory.changes().size();
    const metal::AllocationResult result = memory.allocate(governor->allocationAdmission(), "kv", bytes);
    require(!result && result.failure == cause, "the governor did not refuse with its cause");
    require(memory.held("kv") == 3 * kGiB && memory.total() == 3 * kGiB && memory.changes().size() == changes &&
                test::metalStatistics().allocatedBytes == 3 * kGiB,
            "a refused allocation holds something");
    require(governor->snapshot().headroomBytes == 1 * kGiB, "a refused allocation left a reservation behind");
  };
  refused(2 * kGiB, metal::AllocationFailure::EngineBudget);
  // Other applications take 10 GiB: 3 GiB of the host's spare memory are
  // left, 1 GiB above its spare reserve, inside the warning margin.
  memory.advanceTo(100.0);
  refused(kGiB / 2, metal::AllocationFailure::HostPressure);
}

// The host script applies each event at its instant and never goes back; the scripts the scenarios use say what
// they name.
void testTheHostScriptMovesTheHostAtItsInstants() {
  FakeMemory memory(sixteen({{0.0, 6 * kGiB, engine::MemoryPressure::Normal},
                             {250.0, 9 * kGiB, engine::MemoryPressure::Warning},
                             {400.0, 2 * kGiB, engine::MemoryPressure::Normal}}));
  memory.hold("kv", kGiB);
  require(host(memory)->available == 9 * kGiB && host(memory)->spare == 9 * kGiB &&
              memory.pressure() == engine::MemoryPressure::Normal && memory.nextEvent() == 250.0,
          "an event at the start did not apply at construction");
  memory.advanceTo(249.0);
  require(host(memory)->available == 9 * kGiB && memory.nextEvent() == 250.0, "an event applied before its instant");
  memory.advanceTo(250.0);
  require(host(memory)->available == 6 * kGiB && memory.pressure() == engine::MemoryPressure::Warning &&
              memory.nextEvent() == 400.0,
          "an event did not apply at its instant");
  memory.advanceTo(1000.0);
  require(host(memory)->available == 13 * kGiB && memory.pressure() == engine::MemoryPressure::Normal &&
              !memory.nextEvent(),
          "the last event did not apply");
  rejects([&] { memory.advanceTo(999.0); }, "went back", "the host clock went back");
  // Other applications and the engine beyond the host: none available, and
  // the excess is recorded.
  FakeMemory full(sixteen({{0.0, 14 * kGiB, engine::MemoryPressure::Normal}}));
  full.hold("kv", 3 * kGiB);
  require(host(full)->available == 0 && full.peakOverHost() == kGiB, "a host past its memory has memory available");

  FakeMemory warning(sixteen(test::persistentWarning(5 * kGiB)));
  warning.advanceTo(1e9);
  require(warning.pressure() == engine::MemoryPressure::Warning && warning.otherBytes() == 5 * kGiB,
          "persistent warning ended");
  FakeMemory low(sixteen(test::lowHostAtStart(13 * kGiB, 4 * kGiB, 60'000.0)));
  require(host(low)->available == 3 * kGiB, "the host was not short at the start");
  low.advanceTo(60'000.0);
  require(host(low)->available == 12 * kGiB, "the host did not free memory at its instant");
}

// A write to the backend's statistics behind FakeMemory's back fails its check.
void testAWriteBehindItsBackFailsTheCheck() {
  FakeMemory memory(sixteen());
  memory.hold("kv", kMiB);
  memory.check();
  test::metalStatistics().allocatedBytes += 4096;
  rejects([&] { memory.check(); }, "not what fake memory counts", "a write behind fake memory's back passed");
}

// The fake pool gives back the fewest whole units above its floor that reach the bytes asked, grows a unit under
// the admission up to its cap, and a refusal changes nothing.
void testThePoolReleasesAndGrowsByUnits() {
  FakeMemory memory(sixteen());
  metal::MetalBackend backend("");
  std::optional<engine::MemoryGovernor> governor = test::governorOver(backend, memory, {.limitBytes = 6 * kGiB});
  FakeElasticPool pool("experts", memory, kGiB, 1, 5, 8, 4);
  require(memory.held("experts") == 4 * kGiB && pool.residentShare() == 0.5, "the pool did not hold its units");
  model::PoolState state = pool.state();
  require(state.name == "experts" && state.floorBytes == kGiB && state.currentBytes == 4 * kGiB &&
              state.capBytes == 5 * kGiB && state.releaseUnitBytes == kGiB && state.growUnitBytes == kGiB,
          "the pool's state is not its units");
  require(pool.release(1) == kGiB && pool.release(kGiB + 1) == 2 * kGiB && pool.state().currentBytes == kGiB,
          "the pool did not give back the fewest units that reach the bytes");
  require(pool.release(1) == 0 && !pool.state().releaseUnitBytes, "the pool went below its floor");
  require(pool.shrinks == 3, "the pool did not count its releases");
  {
    const test::ClaimScope growth(*governor, engine::ClaimClass::Growth);
    for (int unit = 0; unit < 4; ++unit)
      require(static_cast<bool>(pool.grow(governor->allocationAdmission())), "the pool's growth was refused");
    require(pool.grows == 4 && memory.held("experts") == 5 * kGiB && !pool.state().growUnitBytes,
            "the pool did not grow to its cap by units");
    require(static_cast<bool>(pool.grow(governor->allocationAdmission())) && pool.grows == 4,
            "the pool grew past its cap");
  }
  require(pool.release(std::numeric_limits<uint64_t>::max()) == 4 * kGiB && pool.units() == 1,
          "the pool did not give back everything above its floor");
  const metal::AllocationAdmission refuse = [](uint64_t, const std::function<void()> &) {
    return metal::AllocationResult(metal::AllocationFailure::HostPressure);
  };
  const metal::AllocationResult refused = pool.grow(refuse);
  require(!refused && refused.failure == metal::AllocationFailure::HostPressure && pool.units() == 1 &&
              memory.held("experts") == kGiB,
          "a refused growth changed the pool");
}

// The KV storage allocates its extents, runway included, through FakeMemory under the governor and gives them back
// as it releases them.
void testTheStorageAllocatesThroughMemory() {
  FakeMemory memory(sixteen());
  metal::MetalBackend backend("");
  std::optional<engine::MemoryGovernor> governor =
      test::governorOver(backend, memory, {.limitBytes = 28 * 4096 + 1});
  test::TestKvStorage storage(32, 4096, 4);
  std::optional<test::ClaimScope> start = test::startScope(governor);
  engine::KvPool pool(storage.through(memory, governor->allocationAdmission()), 8);
  start.reset();
  const test::ClaimScope claim(*governor, engine::ClaimClass::InService);
  require(memory.held("kv") == 8 * 4096 && storage.allocatedBytes() == 8 * 4096, "the runway was not held");
  for (uint32_t extent = 2; extent < 4; ++extent)
    require(static_cast<bool>(storage.allocateExtent(extent, kv::PageForm::Full)), "growth was refused");
  require(memory.held("kv") == 16 * 4096, "growth was not held");
  storage.releaseExtent(3);
  require(memory.held("kv") == 12 * 4096 && memory.held("kv") == storage.allocatedBytes(),
          "a release was not given back");
  for (uint32_t extent = 3; extent < 7; ++extent)
    require(static_cast<bool>(storage.allocateExtent(extent, kv::PageForm::Full)), "growth was refused");
  const metal::AllocationResult refused = storage.allocateExtent(7, kv::PageForm::Full);
  require(!refused && refused.failure == metal::AllocationFailure::EngineBudget && !storage.allocated(7) &&
              memory.held("kv") == storage.allocatedBytes(),
          "growth past the limit was not refused or held something");
}

// The test model allocates lane buffers, state buffers, floors and staging through FakeMemory, reuses pooled
// buffers first, and a refusal holds nothing; a floor takes idle hot extents before it allocates, which its
// shortfall still counts, as a start may take them.
void testTheModelAllocatesThroughMemory() {
  FakeMemory memory(sixteen());
  metal::MetalBackend backend("");
  std::optional<engine::MemoryGovernor> governor = test::governorOver(backend, memory, {.limitBytes = 6 * kGiB});
  test::Executor model;
  model.memory = &memory;
  model.admission = governor->allocationAdmission();
  model.laneFootprintBytes = 256 * kMiB;
  model.spilledReadFloorBytes = 64 * kMiB;
  model.stagingPageBytes = kMiB;
  const test::ClaimScope claim(*governor, engine::ClaimClass::InService);
  ModelRequest request;
  request.id = 1;
  request.maxNewTokens = 1;
  require(model.begin(request).granted() && memory.held("states") == 256 * kMiB &&
              model.heldStateBytes == 256 * kMiB,
          "a lane's state buffers were not allocated");
  metal::AllocationFailure refusal;
  model.requests.at(1).position = 64;
  require(model.reserveState(1, 64, refusal) && memory.held("states") == 256 * kMiB + 64,
          "a state's buffers were not allocated");
  model.staleHotBytes = 16 * kMiB;
  const uint64_t id = 1;
  require(model.hotFloorShortfall(std::span(&id, 1)) == 64 * kMiB && model.stagingShortfall(1, 10) == 10 * kMiB,
          "the shortfall is not the whole floor and staging");
  require(model.reserveSpilledReads(1, {}).granted() &&
              model.reserveSpilledReads(1, {.kind = WorkKind::Prefill, .pages = 10}).granted() &&
              model.reserveSpilledReads(1, {.kind = WorkKind::Prefill, .pages = 6}).granted() &&
              memory.held("spilled_reads") == 58 * kMiB && !model.staleHotBytes,
          "the reads' memory was not allocated past the idle extents");
  require(model.hotFloorShortfall(std::span(&id, 1)) == 0 && model.stagingShortfall(1, 12) == 2 * kMiB,
          "held reads still fall short");
  model.end(1);
  require(memory.held("spilled_reads") == 0 && model.spilledReadBytes.empty() && model.pooledLaneBytes == 256 * kMiB,
          "an ended lane kept its reads' memory or its buffers left the pool");
  request.id = 2;
  require(model.begin(request).granted() && memory.held("states") == 256 * kMiB + 64 && !model.pooledLaneBytes,
          "a lane did not start from the pooled buffers");
  model.end(2);
  require(model.releaseIdleBuffer(false) == 256 * kMiB &&
              memory.held("states") == 64 && model.heldStateBytes == 64,
          "a reclaim did not give the pooled buffers back");
  model.laneFootprintBytes = 7 * kGiB;
  request.id = 3;
  const StateAdmission refused = model.begin(request);
  require(!refused.granted() && refused.failure == StateFailure::MemoryPressure &&
              refused.allocationFailure == metal::AllocationFailure::EngineBudget && memory.held("states") == 64,
          "a refused lane held something");

  FakeElasticPool experts("experts", memory, kGiB, 1, 3, 6, 3);
  model.experts = &experts;
  require(model.pools().size() == 1 && model.pools()[0] == &experts, "the model does not report its pool");
}

// Ended lanes leave one lane's buffers pooled; the rest are idle, a reclaim gives them back, and the next lane
// starts from those kept.
void testEndedLanesLeaveOneLanesBuffersPooled() {
  FakeMemory memory(sixteen());
  metal::MetalBackend backend("");
  std::optional<engine::MemoryGovernor> governor = test::governorOver(backend, memory, {.limitBytes = 6 * kGiB});
  test::Executor model;
  model.memory = &memory;
  model.admission = governor->allocationAdmission();
  model.laneFootprintBytes = 256 * kMiB;
  const test::ClaimScope claim(*governor, engine::ClaimClass::InService);
  ModelRequest request;
  request.maxNewTokens = 1;
  for (request.id = 1; request.id <= 4; ++request.id)
    require(model.begin(request).granted(), "a lane did not begin");
  require(memory.held("states") == kGiB, "four lanes do not hold their buffers");
  for (uint64_t id = 1; id <= 4; ++id)
    model.end(id);
  require(model.idleBufferBytes(0, false, true) == 768 * kMiB,
          "ended lanes' buffers beyond one lane's are not idle: " +
              std::to_string(model.idleBufferBytes(0, false, true)));
  require(model.idleBufferBytes(0, false, false) == kGiB, "ended lanes' buffers without keeping a lane are not idle");
  require(model.releaseIdleBuffer(true) == 768 * kMiB &&
              memory.held("states") == 256 * kMiB && model.pooledLaneBytes == 256 * kMiB,
          "a reclaim that keeps the lane did not give back all but one lane's buffers");
  request.id = 5;
  require(model.begin(request).granted() && memory.held("states") == 256 * kMiB && !model.pooledLaneBytes,
          "the next lane did not start from the buffers kept");
}

// The harness checks every tick that FakeMemory agrees with each allocator: KV held beside the storage fails the
// next tick.
void testTheHarnessChecksTheBooksEveryTick() {
  test::EngineHarness harness({.memory = sixteen(), .limits = {.limitBytes = 6 * kGiB}});
  harness.engine.submit(harness.request(1, std::vector<uint32_t>(65, 1)));
  harness.runUntilIdle();
  require(harness.memory->held("kv") == harness.storage.allocatedBytes() && harness.memory->held("kv") > 0,
          "the engine's KV was not allocated through fake memory");
  harness.memory->hold("kv", 4096);
  rejects([&] { harness.tick(); }, "for kv", "bytes held behind the storage passed the check");
}

// A tick that left the engine unchanged spins only when it claims progress or names a wakeup already due.
void testATickThatChangesNothingSpinsOnlyWhenItWouldRunAgain() {
  using test::EngineHarness;
  require(EngineHarness::spins(true, true, std::nullopt, 5.0) && EngineHarness::spins(true, false, 5.0, 5.0) &&
              EngineHarness::spins(true, false, 4.0, 5.0),
          "a tick that changed nothing and would run again did not spin");
  require(!EngineHarness::spins(true, false, std::nullopt, 5.0) && !EngineHarness::spins(true, false, 6.0, 5.0) &&
              !EngineHarness::spins(false, true, 5.0, 5.0),
          "a tick that sleeps or changed something spun");
  EngineHarness harness;
  harness.tick();
  harness.tick();
  require(harness.now() == 1.0, "the harness clock moved");
}

// The loop fixture allocates weights, KV and states through FakeMemory under the governor, runs the control pass,
// and balances its books every tick.
void testTheLoopFixtureRunsOverMemory() {
  test::LoopFixture fixture({}, {}, {}, metal::kResidencyKeepAliveSeconds,
                            test::LoopOptions{.memory = sixteen(), .limits = {.limitBytes = 6 * kGiB}});
  require(fixture.memory->held("weights") == 3 * kGiB && fixture.memory->held("kv") == fixture.storage.allocatedBytes(),
          "the loop's weights or KV were not held");
  require(fixture.loop.receive(protocol::peer::serialize(test::request(1, 4))), "the request was refused");
  fixture.run([&] { return test::idle(fixture.loop); }, "the loop did not finish the request");
  require(fixture.control.has_value(), "the control pass was not wired");
}

} // namespace

int main(int argc, char **argv) {
  static constexpr test::TestCase kTests[] = {
      {"theBooksBalanceOverSeededSequences", testTheBooksBalanceOverSeededSequences},
      {"aRefusalThroughTheGovernorHoldsNothing", testARefusalThroughTheGovernorHoldsNothing},
      {"theHostScriptMovesTheHostAtItsInstants", testTheHostScriptMovesTheHostAtItsInstants},
      {"aWriteBehindItsBackFailsTheCheck", testAWriteBehindItsBackFailsTheCheck},
      {"thePoolReleasesAndGrowsByUnits", testThePoolReleasesAndGrowsByUnits},
      {"theStorageAllocatesThroughMemory", testTheStorageAllocatesThroughMemory},
      {"theModelAllocatesThroughMemory", testTheModelAllocatesThroughMemory},
      {"endedLanesLeaveOneLanesBuffersPooled", testEndedLanesLeaveOneLanesBuffersPooled},
      {"theHarnessChecksTheBooksEveryTick", testTheHarnessChecksTheBooksEveryTick},
      {"aTickThatChangesNothingSpinsOnlyWhenItWouldRunAgain", testATickThatChangesNothingSpinsOnlyWhenItWouldRunAgain},
      {"theLoopFixtureRunsOverMemory", testTheLoopFixtureRunsOverMemory},
  };
  const test::TestGroup groups[] = {{"fake-memory", kTests}};
  return test::runTests("fake-memory", groups, argc, argv, [] { test::metalStatistics() = {}; });
}

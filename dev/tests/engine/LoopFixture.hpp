#pragma once

// What the native loop tests of several areas share: the loop fixture, its
// requests and the loops that tick it.

#include "FakeMemory.hpp"
#include "MemoryChecks.hpp"
#include "ProtocolPeer.hpp"
#include "TestChecks.hpp"
#include "TestEngine.hpp"
#include "TestExecutor.hpp"
#include "TestKvPool.hpp"
#include "TestKvTier.hpp"
#include "TestStatus.hpp"
#include "TimingProfile.hpp"
#include "engine/Cache.hpp"
#include "engine/MemoryControl.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/NativeRuntime.hpp"
#include "metal/MetalBackend.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace splash::test {

// A host whose spare memory is all its available memory.
inline std::optional<engine::HostMemory> hostMemory(std::optional<uint64_t> bytes) {
  if (!bytes)
    return std::nullopt;
  return engine::HostMemory{*bytes, *bytes};
}

// Every submitted request has ended and no command is in flight.
inline bool idle(const engine::NativeRuntime &loop) {
  const engine::EngineSnapshot counts = loop.statusSnapshot();
  return counts.submitted == counts.completed + counts.cancelled + counts.failed &&
         !loop.commandInFlight();
}

inline protocol::RequestFrame request(uint64_t id, uint32_t maxOutputTokens = 1) {
  protocol::RequestFrame result;
  result.requestId = id;
  result.priority = engine::RequestPriority::Foreground;
  result.constraint = ConstraintMode::None;
  result.absoluteDeadlineUnixMicros = 2'000'000;
  result.remainingDeadlineMicros = 1'000'000;
  result.logicalMaxOutputTokens = maxOutputTokens;
  result.promptTokens.resize(65);
  for (uint32_t i = 0; i < result.promptTokens.size(); ++i) {
    result.promptTokens[i] = i + 1;
  }
  return result;
}

inline void runUntilIdle(engine::NativeRuntime &loop) {
  for (uint32_t step = 0; step < 32 && !idle(loop); ++step) {
    static_cast<void>(loop.tick());
  }
  require(idle(loop), "native loop did not become idle");
}

// The KV pool under the loop: `pages` pages in extents of `extentPages`, of
// which the first `runwayPages` are allocated up front.
struct PoolShape {
  uint32_t pages = 32;
  uint32_t extentPages = 4;
  uint32_t runwayPages = pages;
  // A Full page's bytes, and an Index page's (none without them).
  uint64_t pageBytes = 4096, indexPageBytes = 0;
};

// What a loop fixture adds to the loop. With memory: FakeMemory, the
// stand-in backend and a real governor of `limits` wired by
// connectToGovernor, the fixture built inside a start's Floor claim
// (startScope), whose admissions the KV storage, the model and the
// weights allocate through, and with `control` the production control pass
// (MemoryControl::run at FakeMemory's pressure) between commands. With
// timing: the model takes the profile's time and the awake clock moves only
// as run() moves it. With a tier, the cache's KV tier and its quota, its
// transfers landing on the awake clock, and the cache's spill policy. `shape` sets what one
// of the model's commands may hold before the loop's engine takes it.
struct LoopOptions final {
  std::optional<FakeMemory::Config> memory;
  GovernorLimits limits{};
  const TimingProfile *timing = nullptr;
  bool control = true;
  std::optional<TierOptions> tier;
  std::optional<engine::CacheSpillPolicy> cacheSpill;
  std::function<void(Executor &)> shape;
};

// A native loop over the fake model, collecting the bytes it writes. Its
// config is completed with the engine's (test::engineConfig), and the
// fixture's weights (EngineConfig::weights), each weight
// allocation admitted (test::admitAll, or the governor's with memory), where
// the test gave none; the weights' idle release is production's unless the
// test gives one. Its unix clock stands still and its awake clock reads
// `monotonic`, advanced by `clockStep` on every read, or with timing by
// run().
struct LoopFixture {
  explicit LoopFixture(engine::NativeLoopConfig config = {}, PoolShape shape = {},
                       protocol::ProtocolLimits limits = {},
                       double idleReleaseSeconds = metal::kResidencyKeepAliveSeconds,
                       std::optional<LoopOptions> loopOptions = std::nullopt)
      : options(std::move(loopOptions)), memory(fakeMemory(options)), backend(standIn(memory)),
        governor(memoryGovernor(options, memory, backend)), startup_(startScope(governor)),
        storage(shape.pages, shape.pageBytes, shape.extentPages, shape.indexPageBytes),
        pool(memory ? storage.through(*memory, governor->allocationAdmission()) : storage, shape.runwayPages),
        tier(kvTier(options)),
        budget(tier ? std::make_shared<model::DiskBudget>(uint64_t{options->tier->capacity} * tier->slotBytes(),
                                                          [this] { return monotonic; })
                    : nullptr),
        cache(pool, tier ? &*tier : nullptr, budget, options ? options->cacheSpill : std::nullopt),
        loop(
            clocked(std::move(config), idleReleaseSeconds), cache, executor,
            [this](std::span<const uint8_t> bytes) {
              if (writeFailure)
                throw *writeFailure;
              output.insert(output.end(), bytes.begin(), bytes.end());
            },
            limits) {
    loop.serveStatus([this] { return status(); });
    storage.commandInFlight = [this] { return loop.commandInFlight(); };
    if (memory) {
      memory->hold("weights", uint64_t{weights.held()} * Weights::kImageBytes);
      weights.memory = &*memory;
      executor.memory = &*memory;
      executor.admission = governor->allocationAdmission();
      if (options->control)
        control.emplace(*governor, *backend, loop);
    }
    if (options && options->timing) {
      executor.time(*options->timing);
      executor.clock = [this] { return monotonic; };
    }
    if (tier) {
      tier->budget = budget;
      tier->capacity = options->tier->capacity;
      tier->transferLimit = options->tier->transferLimit;
      tier->bytesPerMillisecond = options->tier->bytesPerMillisecond;
      tier->clock = [this] { return monotonic; };
    }
    startup_.reset();
  }

  // Every event the loop has written.
  std::vector<protocol::EngineEvent> events() const {
    return protocol::peer::decodeEvents(output);
  }

  engine::NativeLoopConfig clocked(engine::NativeLoopConfig config, double idleReleaseSeconds) {
    if (options && options->shape)
      options->shape(executor);
    if (governor)
      engine::connectToGovernor(config.engine, *governor);
    config.engine = test::engineConfig(std::move(config.engine));
    if (!config.engine.weights)
      config.engine.weights.emplace();
    engine::WeightResidencyConfig &residency = *config.engine.weights;
    if (!residency.weights)
      residency.weights = &weights;
    if (!residency.admit)
      residency.admit = governor ? governor->allocationAdmission() : test::admitAll;
    residency.idleReleaseMilliseconds = 1000.0 * idleReleaseSeconds;
    config.unixMicros = [] { return uint64_t{1'000'000}; };
    if (options && options->timing)
      config.monotonicMilliseconds = [this] { return monotonic; };
    else
      config.monotonicMilliseconds = [this] { return monotonic += clockStep; };
    return config;
  }

  // Ticks, with memory running the control pass between commands, and
  // after each tick the checks of EngineHarness::check: the memory checks
  // (I1) with memory, the extents' activity following the pages requests
  // hold (I4) and, between commands, the cache's invariants (I5); I3 holds
  // inside the storage. Again at once after a tick that made progress, else
  // once the awake clock reaches the earliest of the command in flight's
  // end, the loop's next wakeup, the tier's next landing and the host's next
  // event, or 1 ms on; until done() or `steps` ticks, then requires done().
  void run(const std::function<bool()> &done, std::string_view what, uint32_t steps = 100'000) {
    for (uint32_t step = 0; step < steps && !done(); ++step) {
      const bool progressed = loop.tick();
      if (control && !loop.commandInFlight())
        static_cast<void>(control->run(memory->pressure()));
      if (memory)
        checkMemory({.memory = *memory,
                     .storage = storage,
                     .model = executor,
                     .weights = &weights,
                     .governor = &*governor});
      if (!loop.commandInFlight())
        cache.checkInvariants();
      if (progressed)
        continue;
      const std::optional<double> wakeup = loop.millisecondsUntilNextWakeup();
      double next = std::numeric_limits<double>::infinity();
      for (const std::optional<double> &instant :
           {executor.commandEnd(), wakeup ? std::optional<double>(monotonic + *wakeup) : std::nullopt,
            tier ? tier->nextLanding() : std::nullopt, memory ? memory->nextEvent() : std::nullopt})
        if (instant)
          next = std::min(next, *instant);
      monotonic = next == std::numeric_limits<double>::infinity() ? monotonic + 1.0 : std::max(monotonic, next);
      if (memory)
        memory->advanceTo(monotonic);
      if (tier)
        tier->land(monotonic);
      executor.land(monotonic);
    }
    require(done(), what);
  }

  std::optional<LoopOptions> options;
  std::optional<FakeMemory> memory;
  std::optional<metal::MetalBackend> backend;
  std::optional<engine::MemoryGovernor> governor;

private:
  // While the fixture is built, with memory.
  std::optional<ClaimScope> startup_;

public:
  test::TestKvStorage storage;
  engine::KvPool pool;
  std::optional<TestKvTier> tier;
  // The tier's quota, its slots' bytes, refilled on the awake clock.
  std::shared_ptr<model::DiskBudget> budget;
  engine::Cache cache;
  Executor executor;
  std::vector<uint8_t> output;
  // Thrown by every write while set, as by a closed output pipe.
  std::optional<std::system_error> writeFailure;
  // What the loop answers a status request with.
  std::function<std::string()> status = test::readyStatusJson;
  double monotonic = 100.0;
  double clockStep = 0.0;
  Weights weights;
  engine::NativeRuntime loop;
  std::optional<engine::MemoryControl> control;

private:
  static std::optional<FakeMemory> fakeMemory(const std::optional<LoopOptions> &options) {
    if (!options || !options->memory)
      return std::nullopt;
    return std::optional<FakeMemory>(std::in_place, *options->memory);
  }
  static std::optional<TestKvTier> kvTier(const std::optional<LoopOptions> &options) {
    if (!options || !options->tier)
      return std::nullopt;
    return std::optional<TestKvTier>(std::in_place, options->tier->persistent);
  }
  static std::optional<metal::MetalBackend> standIn(const std::optional<FakeMemory> &memory) {
    if (!memory)
      return std::nullopt;
    return std::optional<metal::MetalBackend>(std::in_place, "");
  }
  static std::optional<engine::MemoryGovernor> memoryGovernor(const std::optional<LoopOptions> &options,
                                                              std::optional<FakeMemory> &memory,
                                                              std::optional<metal::MetalBackend> &backend) {
    if (!memory)
      return std::nullopt;
    return governorOver(*backend, *memory, options->limits);
  }
};

inline bool answeredStatus(const LoopFixture &fixture, uint64_t correlationId) {
  const auto events = fixture.events();
  return std::any_of(events.begin(), events.end(), [&](const auto &message) {
    const auto *event = std::get_if<protocol::StatusJsonEvent>(&message);
    return event && event->correlationId == correlationId;
  });
}

// Sends a release request and returns the weights' snapshot its answer reports:
// the status document that answers it is written from the loop's snapshot of the
// moment (ServingRuntime::statusJson).
inline engine::WeightsSnapshot releaseAnswer(LoopFixture &fixture, uint64_t correlationId) {
  std::optional<engine::WeightsSnapshot> reported;
  fixture.status = [&] {
    reported = fixture.loop.weightsSnapshot();
    return test::readyStatusJson();
  };
  require(fixture.loop.receive(
              protocol::peer::serialize(protocol::ReleaseRequestFrame{correlationId})) &&
              answeredStatus(fixture, correlationId) && reported,
          "a release request was not answered with a status document");
  fixture.status = test::readyStatusJson;
  return *reported;
}

} // namespace splash::test

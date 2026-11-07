#pragma once

// An engine on the fake model with its cache, KV storage, optional tier and
// optional governed FakeMemory, on a clock that moves to the next instant
// anything is due; it records the commands and token times and checks the
// invariants after every tick.

#include "EngineTestSupport.hpp"
#include "FakeMemory.hpp"
#include "MemoryChecks.hpp"
#include "TestChecks.hpp"
#include "TestEngine.hpp"
#include "TestExecutor.hpp"
#include "TestKvPool.hpp"
#include "TestKvTier.hpp"
#include "TimingProfile.hpp"
#include "engine/Cache.hpp"
#include "engine/Engine.hpp"
#include "engine/MemoryGovernor.hpp"
#include "metal/MetalBackend.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace splash::test {

struct HarnessOptions final {
  uint32_t pages = 64;
  uint64_t pageBytes = 4096;
  uint32_t extentPages = 4;
  engine::EngineConfig engine{};  // completed by test::engineConfig
  std::optional<TierOptions> tier;
  // With it, the storage and the model allocate through FakeMemory under a
  // real MemoryGovernor of `limits`.
  std::optional<FakeMemory::Config> memory;
  GovernorLimits limits{};
  const TimingProfile *timing = nullptr;  // with it, commands take the profile's time on the clock
  // The model's command limits (Executor::groupRows, passes, maximumLanes),
  // set before the engine reads them.
  std::function<void(Executor &)> shape;
};

class EngineHarness final {
public:
  struct Command final {
    WorkKind kind = WorkKind::Decode;
    // Each lane and its rows.
    std::vector<uint64_t> lanes;
    std::vector<uint32_t> rows;
    uint32_t pass = 0;
    double startMilliseconds = 0, wallMilliseconds = 0;
  };

  explicit EngineHarness(HarnessOptions options = {})
      : options_(std::move(options)), memory(fakeMemory(options_)), backend(standIn(options_)),
        governor(memoryGovernor(options_, memory, backend)), startup_(startScope(governor)),
        storage(options_.pages, options_.pageBytes, options_.extentPages),
        pool(memory ? storage.through(*memory, governor->allocationAdmission()) : storage, 0),
        tier(kvTier(options_)), cache(pool, tier ? &*tier : nullptr, nullptr), model(),
        engine((options_.shape ? options_.shape(model) : void(), config(options_, governor)), cache, model, events) {
    guardReleases(storage, engine);
    // The accounts a start books (RuntimeResources::create).
    if (governor) {
      governor->ledger().addAccount("kv", [this] { return storage.allocatedBytes(); });
      governor->ledger().addAccount("states", [this] { return model.heldStateBytes; });
    }
    const std::function<double()> clock = [this] { return now_; };
    events.clock = clock;
    if (tier) {
      tier->capacity = options_.tier->capacity;
      tier->transferLimit = options_.tier->transferLimit;
      tier->bytesPerMillisecond = options_.tier->bytesPerMillisecond;
      tier->clock = clock;
    }
    if (memory) {
      model.memory = &*memory;
      model.admission = governor->allocationAdmission();
    }
    if (options_.timing) {
      model.time(*options_.timing);
      model.clock = clock;
    }
    model.commandTimed = [this](const BatchPlan &plan, const CommandTimes &times) {
      Command command{.kind = plan.kind, .pass = plan.pass, .startMilliseconds = now_,
                      .wallMilliseconds = times.wallMilliseconds};
      for (const BatchItem &item : plan.items) {
        command.lanes.push_back(item.requestId);
        command.rows.push_back(item.tokenCount);
      }
      commands_.push_back(std::move(command));
    };
    startup_.reset();
  }
  EngineHarness(const EngineHarness &) = delete;
  EngineHarness &operator=(const EngineHarness &) = delete;

  // A request of `prompt` for one token, its deadline 10 s from now.
  [[nodiscard]] engine::EngineRequest request(uint64_t id, std::vector<uint32_t> prompt) const {
    engine::EngineRequest result = test::request(id, prompt);
    result.deadlineMilliseconds = now_ + 10'000.0;
    return result;
  }
  // engine.tick(now()), then the checks of a tick.
  bool tick() {
    const bool progressed = engine.tick(now_);
    check(progressed);
    return progressed;
  }
  // Ticks as the native loop does, again at once after progress, else at the
  // next instant anything is due, until done() or `steps` ticks; requires
  // done().
  void runUntil(const std::function<bool()> &done, std::string_view what, uint32_t steps = 100'000) {
    for (uint32_t step = 0; step < steps && !done(); ++step)
      if (!tick())
        advanceTo(nextInstant());
    require(done(), what);
  }
  void runUntilIdle(uint32_t steps = 100'000) {
    runUntil([&] { return idle(engine); }, "the engine did not become idle", steps);
  }
  [[nodiscard]] double now() const noexcept { return now_; }
  // Moves the clock on, landing what is due by then.
  void advanceTo(double milliseconds) {
    if (milliseconds < now_)
      throw std::logic_error("the harness clock went back");
    now_ = milliseconds;
    if (memory)
      memory->advanceTo(now_);
    if (tier)
      tier->land(now_);
    model.land(now_);
  }
  // The longest time between consecutive token events of one request, or of
  // any.
  [[nodiscard]] double longestDecodeGap(std::optional<uint64_t> id = {}) const {
    double longest = 0.0;
    for (const auto &[request, times] : events.tokenTimes) {
      if (id && request != *id)
        continue;
      for (size_t index = 1; index < times.size(); ++index)
        longest = std::max(longest, times[index] - times[index - 1]);
    }
    return longest;
  }
  // I7: a tick at the instant of the one before it that left the engine as it
  // was spins when it still claims progress or names no future wakeup, as
  // the native loop would tick again at once.
  [[nodiscard]] static bool spins(bool unchanged, bool progressed, std::optional<double> wakeup,
                                  double nowMilliseconds) noexcept {
    return unchanged && (progressed || (wakeup && *wakeup <= nowMilliseconds));
  }
  // The commands that took their times (CommandTimes), as Model::submit
  // received them.
  [[nodiscard]] const std::vector<Command> &commands() const noexcept { return commands_; }

private:
  HarnessOptions options_;

public:
  // In construction order.
  std::optional<FakeMemory> memory;
  std::optional<metal::MetalBackend> backend;
  std::optional<engine::MemoryGovernor> governor;

private:
  // While the harness is built, with memory.
  std::optional<ClaimScope> startup_;

public:
  TestKvStorage storage;
  engine::KvPool pool;
  std::optional<TestKvTier> tier;
  engine::Cache cache;
  Executor model;
  Events events;
  engine::Engine engine;

private:
  static std::optional<FakeMemory> fakeMemory(const HarnessOptions &options) {
    if (!options.memory)
      return std::nullopt;
    return std::optional<FakeMemory>(std::in_place, *options.memory);
  }
  static std::optional<metal::MetalBackend> standIn(const HarnessOptions &options) {
    if (!options.memory)
      return std::nullopt;
    return std::optional<metal::MetalBackend>(std::in_place, "");
  }
  static std::optional<engine::MemoryGovernor> memoryGovernor(const HarnessOptions &options,
                                                              std::optional<FakeMemory> &memory,
                                                              std::optional<metal::MetalBackend> &backend) {
    if (!memory)
      return std::nullopt;
    return governorOver(*backend, *memory, options.limits);
  }
  static std::optional<TestKvTier> kvTier(const HarnessOptions &options) {
    if (!options.tier)
      return std::nullopt;
    return std::optional<TestKvTier>(std::in_place, options.tier->persistent);
  }
  static engine::EngineConfig config(const HarnessOptions &options, std::optional<engine::MemoryGovernor> &governor) {
    engine::EngineConfig result = options.engine;
    if (governor)
      engine::connectToGovernor(result, *governor);
    return test::engineConfig(std::move(result));
  }

  // What a tick may change: the engine's counters and requests, the memory
  // charged and held, the KV allocated and the transfers.
  [[nodiscard]] std::vector<uint64_t> fingerprint() const {
    const engine::EngineSnapshot snapshot = engine.snapshot();
    std::vector<uint64_t> result{snapshot.submitted,
                                 snapshot.completed,
                                 snapshot.cancelled,
                                 snapshot.failed,
                                 snapshot.replans,
                                 snapshot.captureCuts,
                                 snapshot.stateCaptures,
                                 snapshot.resourceSuspensions,
                                 snapshot.resourceResumptions,
                                 snapshot.checkpointPublications,
                                 snapshot.replayStatePublications,
                                 snapshot.scheduler.prefillBatches,
                                 snapshot.scheduler.prefillRows,
                                 snapshot.scheduler.decodeBatches,
                                 events.emitted,
                                 events.completedCount,
                                 events.failedCount,
                                 events.startIds.size(),
                                 events.maskRequests.size(),
                                 events.cycles.size(),
                                 snapshot.writeBehind.waiting,
                                 snapshot.writeBehind.durable,
                                 snapshot.writeBehind.unneeded,
                                 snapshot.writeBehind.refused,
                                 snapshot.spill.converted,
                                 snapshot.resources.pool.pagesAllocated,
                                 snapshot.resources.pool.pagesActive,
                                 snapshot.resources.pool.pagesPrefix,
                                 snapshot.resources.stateCache.entries,
                                 snapshot.resources.stateCache.bytes,
                                 snapshot.resources.kvTier.usedBytes,
                                 engine.commandInFlight(),
                                 cache.transfersInFlight(),
                                 cache.copyingAheadBytes(),
                                 storage.allocatedBytes(),
                                 model.heldStateBytes,
                                 tier ? tier->transfers.size() : 0,
                                 memory ? memory->total() : 0,
                                 governor ? governor->snapshot().chargedBytes : 0};
    for (const engine::ActiveRequestSnapshot &request : snapshot.activeRequests)
      result.insert(result.end(), {request.id, static_cast<uint64_t>(request.phase), request.promptProcessed,
                                   request.generatedTokens, static_cast<uint64_t>(request.kvMode)});
    return result;
  }
  // I1 and I2 with memory (checkMemory); I3 holds inside the storage
  // (guardReleases); I4 the extents' activity follows the pool's held
  // pages; I5 the cache's invariants between commands; I6 inside the model
  // (commandViolation); I7 no spin (spins).
  void check(bool progressed) {
    if (memory) {
      checkMemory({.memory = *memory,
                   .storage = storage,
                   .model = model,
                   .governor = &*governor,
                   .cache = &cache,
                   .engine = &engine});
    }
    if (!engine.commandInFlight())
      cache.checkInvariants();
    std::vector<uint64_t> current = fingerprint();
    if (spins(lastTick_ == now_ && current == lastFingerprint_, progressed, engine.nextWakeupMilliseconds(), now_))
      throw std::logic_error("tick changed nothing and named no future wakeup");
    lastTick_ = now_;
    lastFingerprint_ = std::move(current);
  }
  [[nodiscard]] double nextInstant() const {
    double next = std::numeric_limits<double>::infinity();
    for (const std::optional<double> &instant :
         {model.commandEnd(), engine.nextWakeupMilliseconds(), tier ? tier->nextLanding() : std::nullopt,
          memory ? memory->nextEvent() : std::nullopt})
      if (instant)
        next = std::min(next, *instant);
    return next == std::numeric_limits<double>::infinity() ? now_ + 1.0 : std::max(now_, next);
  }

  double now_ = 1.0;
  std::optional<double> lastTick_;
  std::vector<uint64_t> lastFingerprint_;
  std::vector<Command> commands_;
};

} // namespace splash::test

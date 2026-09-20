#include "TestImmediateTicket.hpp"
#include "engine/Engine.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>

using namespace splash;
using namespace splash::engine;

namespace {

class Backing final : public KvBacking {
public:
  explicit Backing(uint32_t pages) : resident_(pages) {}
  uint32_t pageCount() const noexcept override { return resident_.size(); }
  uint64_t bytesPerPage() const noexcept override { return 4096; }
  bool isResident(uint32_t page) const override { return resident_.at(page); }
  splash::metal::AllocationResult ensureResident(uint32_t page) override {
    ++growthAttempts;
    if (growthBlocked || (growthAllowed && !growthAllowed()))
      return allocationFailure;
    uint32_t first = extentFirstPage(page);
    for (uint32_t i = first; i < first + extentPageCount(page); ++i) {
      resident_.at(i) = true;
    }
    return true;
  }
  bool releaseBackingForPage(uint32_t page) override {
    uint32_t first = extentFirstPage(page);
    for (uint32_t i = first; i < first + extentPageCount(page); ++i) {
      resident_.at(i) = false;
    }
    releasePending = deferRelease;
    return true;
  }
  bool releaseReady() const noexcept override {
    if (completeReleaseOnPoll)
      releasePending = false;
    return !releasePending;
  }
  uint32_t extentFirstPage(uint32_t page) const override {
    return page - page % 4;
  }
  uint32_t extentPageCount(uint32_t page) const override {
    return std::min<uint32_t>(4, resident_.size() - extentFirstPage(page));
  }
  bool deferRelease = false;
  mutable bool releasePending = false;
  bool completeReleaseOnPoll = false;
  bool growthBlocked = false;
  metal::AllocationFailure allocationFailure = metal::AllocationFailure::Capacity;
  uint32_t growthAttempts = 0;
  std::function<bool()> growthAllowed;

private:
  std::vector<bool> resident_;
};

class State final : public CompositeState {
public:
  uint64_t bytes() const noexcept override { return 64; }
};

struct MaskOverlapState final {
  uint64_t requestId = 0;
  bool emitted = false;
  bool provided = false;
  bool abandoned = false;
};

class MaskOverlapTicket final : public ModelBatchTicket {
public:
  explicit MaskOverlapTicket(std::shared_ptr<MaskOverlapState> state)
      : state_(std::move(state)) {}

  std::vector<ModelMaskRequest> takeMaskRequests() override {
    if (state_->emitted)
      return {};
    state_->emitted = true;
    return {{state_->requestId, {10, 11, 12, 13, 14, 15, 16, 17}}};
  }
  bool ownsMaskWait(uint64_t id) const noexcept override {
    return state_->emitted && id == state_->requestId;
  }
  void abandonMask(uint64_t id) noexcept override {
    if (id == state_->requestId)
      state_->abandoned = true;
  }
  bool ready() const noexcept override {
    return state_->provided || state_->abandoned;
  }
  std::vector<ModelStepResult> wait() override {
    if (!ready())
      throw std::logic_error("mask-overlap ticket completed without input");
    return {{state_->requestId, 0, {42}, true, DecodeStage::Regular, 7, 0}};
  }
  double wallMilliseconds() const noexcept override { return 1.0; }

private:
  std::shared_ptr<MaskOverlapState> state_;
};

// A decode command that stays in flight until the test releases it.
class HeldTicket final : public ModelBatchTicket {
public:
  HeldTicket(std::vector<ModelStepResult> results,
             std::shared_ptr<bool> released)
      : results_(std::move(results)), released_(std::move(released)) {}
  bool ready() const noexcept override { return *released_; }
  std::vector<ModelStepResult> wait() override { return std::move(results_); }
  double wallMilliseconds() const noexcept override { return 1.0; }

private:
  std::vector<ModelStepResult> results_;
  std::shared_ptr<bool> released_;
};

class Executor final : public model::Model {
public:
  explicit Executor(
      uint32_t maximumCells = model::ExecutionLimits::maximumBatchWidth)
      : maximumCells(maximumCells) {}

  StateAdmission begin(const ModelRequest &request) override {
    ++beginAttempts;
    if (beginObserver) beginObserver();
    if (beginGrowthBlocked && beginGrowthBlocked())
      return {{}, StateFailure::MemoryPressure, beginAllocationFailure};
    if (deniedBegins) {
      --deniedBegins;
      return {{}, StateFailure::MemoryPressure};
    }
    for (uint32_t slot = 0; slot < maximumCells; ++slot) {
      const bool used = std::any_of(
          requests.begin(), requests.end(), [slot](const auto &entry) {
            return entry.second.resident && entry.second.slot == slot;
          });
      if (!used) {
        requests.emplace(request.id, Request{slot, 0, true});
        return {slot, StateFailure::None};
      }
    }
    return {{}, StateFailure::ConcurrencyLimit};
  }
  void suspend(uint64_t id) override {
    Request &entry = requests.at(id);
    if (!entry.resident)
      throw std::logic_error("request is already suspended");
    entry.resident = false;
    entry.position = 0;
    ++suspensions;
    if (physicalGrowthBlocked && unblockGrowthOnSuspend)
      *physicalGrowthBlocked = false;
  }
  StateAdmission resume(const ModelRequest &request) override {
    ++resumeAttempts;
    const uint64_t id = request.id;
    Request &entry = requests.at(id);
    for (uint32_t slot = 0; slot < maximumCells; ++slot) {
      const bool used = std::any_of(
          requests.begin(), requests.end(), [&](const auto &candidate) {
            return candidate.first != id && candidate.second.resident &&
                   candidate.second.slot == slot;
          });
      if (!used) {
        entry.slot = slot;
        entry.resident = true;
        entry.position = 0;
        entry.replaying = true;
        resumedPrompts.emplace_back(request.prompt.begin(), request.prompt.end());
        ++resumptions;
        return {slot, StateFailure::None};
      }
    }
    return {{}, StateFailure::ConcurrencyLimit};
  }
  void restore(uint64_t id, uint32_t length,
               std::shared_ptr<const CompositeState> state,
               bool restoreDraftState) override {
    if (!state)
      throw std::runtime_error("empty restore state");
    requests.at(id).position = length;
    restored += length;
    restoredDraft = restoreDraftState;
  }
  void setDraftContextPlan(uint64_t id, DraftContextPlan plan) override {
    plans[id] = std::move(plan);
  }
  std::vector<ModelStepResult> prefill(const BatchPlan &,
                                       std::span<const ModelBatchItem> items) {
    prefillWidths.push_back(static_cast<uint32_t>(items.size()));
    std::vector<ModelStepResult> result;
    for (const auto &item : items) {
      requests.at(item.requestId).position += item.tokenCount;
      prefillRows += item.tokenCount;
      ModelStepResult step{item.requestId, item.tokenCount, {}, false,
                           requests.at(item.requestId).replaying
                               ? replayDecodeStage : DecodeStage::Regular,
                           0, 0};
      if (prefillAnchor && !requests.at(item.requestId).replaying) {
        // Prefill selected a stop token or the last budgeted token: emitted
        // now, without a KV row, and the request never decodes.
        step.outputTokens = {42};
        step.outputTokensWithoutKv = 1;
        step.finished = *prefillAnchor;
      }
      result.push_back(std::move(step));
    }
    return result;
  }
  std::vector<ModelStepResult> decode(const BatchPlan &plan,
                                      std::span<const ModelBatchItem> items) {
    std::vector<ModelStepResult> result;
    for (const auto &item : items) {
      // The production runtime stores eight verify rows from the lane's
      // position; the engine must have covered them with page-table entries.
      if (uint64_t{item.pageTable.size()} * KvCache::pageTokens <
          item.logicalPosition + model::ExecutionLimits::targetVerifyRows) {
        throw std::invalid_argument("page_table_too_short");
      }
      if (plan.cohort == BatchCohort::Constrained &&
          plan.decodeStage == DecodeStage::RequestInitialMask) {
        result.push_back({item.requestId,
                          0,
                          {},
                          false,
                          DecodeStage::ApplyInitialMask,
                          0,
                          0});
      } else {
        ModelStepResult step{item.requestId, 0, {42}, decodeFinishes,
                             DecodeStage::Regular, 0, 0};
        step.outputTokensWithoutKv = decodeTokensWithoutKv;
        result.push_back(std::move(step));
      }
    }
    return result;
  }
  std::unique_ptr<ModelBatchTicket>
  submit(const BatchPlan &plan, std::span<const ModelBatchItem> items,
         std::function<void()> completion) override {
    if (plan.kind == WorkKind::Decode &&
        plan.cohort == BatchCohort::Constrained &&
        plan.decodeStage == DecodeStage::ApplyInitialMask) {
      overlap = std::make_shared<MaskOverlapState>();
      overlap->requestId = items.front().requestId;
      return std::make_unique<MaskOverlapTicket>(overlap);
    }
    if (plan.kind == WorkKind::Decode && holdDecodeUntil) {
      return std::make_unique<HeldTicket>(decode(plan, items), holdDecodeUntil);
    }
    if (plan.kind == WorkKind::Prefill && holdPrefillUntil) {
      return std::make_unique<HeldTicket>(prefill(plan, items), holdPrefillUntil);
    }
    return test::immediateTicket(plan.kind == WorkKind::Prefill
                                     ? prefill(plan, items)
                                     : decode(plan, items),
                                 completion);
  }
  // The production model copies the lane's state into a cache slot at its
  // current page-aligned boundary and returns nullptr when no slot is free
  // and the governor denies a new one. The fake denies the next
  // `deniedSnapshots` calls, or every call made at `denySnapshotAtBoundary`.
  std::shared_ptr<const CompositeState> snapshot(uint64_t id) override {
    ++snapshotAttempts;
    if (snapshotObserver)
      snapshotObserver();
    if (deniedSnapshots) {
      --deniedSnapshots;
      return nullptr;
    }
    if (denySnapshotAtBoundary &&
        *denySnapshotAtBoundary == requests.at(id).position) {
      return nullptr;
    }
    ++snapshots;
    return std::make_shared<State>();
  }
  uint64_t reclaimIdleState() noexcept override {
    const uint64_t released = reclaimableIdleStateBytes;
    reclaimableIdleStateBytes = 0;
    reclaimedIdleStateBytes += released;
    if (released && physicalGrowthBlocked)
      *physicalGrowthBlocked = false;
    return released;
  }
  void provideMask(uint64_t id, std::span<const uint32_t>) override {
    if (overlap && overlap->emitted && overlap->requestId == id)
      overlap->provided = true;
  }
  void end(uint64_t id) override { requests.erase(id); }

  struct Request {
    uint32_t slot = 0;
    uint32_t position = 0;
    bool resident = false;
    bool replaying = false;
  };
  std::unordered_map<uint64_t, Request> requests;
  std::unordered_map<uint64_t, DraftContextPlan> plans;
  uint32_t prefillRows = 0;
  uint32_t restored = 0;
  uint32_t snapshots = 0;
  uint32_t snapshotAttempts = 0;
  uint32_t deniedSnapshots = 0;
  std::optional<uint32_t> denySnapshotAtBoundary;
  uint32_t beginAttempts = 0;
  uint32_t deniedBegins = 0;
  uint32_t suspensions = 0;
  uint32_t resumptions = 0;
  uint32_t resumeAttempts = 0;
  uint32_t maximumCells = model::ExecutionLimits::maximumBatchWidth;
  std::function<void()> beginObserver;
  std::function<void()> snapshotObserver;
  std::function<bool()> beginGrowthBlocked;
  std::vector<std::vector<uint32_t>> resumedPrompts;
  std::vector<uint32_t> prefillWidths;
  bool restoredDraft = false;
  metal::AllocationFailure beginAllocationFailure = metal::AllocationFailure::None;
  uint64_t reclaimableIdleStateBytes = 0;
  uint64_t reclaimedIdleStateBytes = 0;
  bool *physicalGrowthBlocked = nullptr;
  bool unblockGrowthOnSuspend = true;
  bool decodeFinishes = true;
  DecodeStage replayDecodeStage = DecodeStage::Regular;
  uint32_t decodeTokensWithoutKv = 0;
  // Set when prefill itself ends the request: the value is `finished` (stop).
  std::optional<bool> prefillAnchor;
  std::shared_ptr<bool> holdDecodeUntil;
  std::shared_ptr<bool> holdPrefillUntil;
  std::shared_ptr<MaskOverlapState> overlap;
};

class Events final : public EngineEventSink {
public:
  void batchCompleted(WorkKind, uint32_t, uint32_t, uint32_t, uint32_t,
                      uint32_t, double) override {}
  void started(uint64_t requestId, EngineCacheStatus cache, uint32_t matched,
               uint32_t) override {
    startIds.push_back(requestId);
    starts.emplace_back(std::move(cache), matched);
  }
  void promptProgress(uint64_t id, uint32_t processed) override {
    progress[id].push_back(processed);
  }
  void tokens(uint64_t id, std::span<const uint32_t> values) override {
    emitted += values.size();
    auto &transcript = outputs[id];
    transcript.insert(transcript.end(), values.begin(), values.end());
  }
  void maskRequested(uint64_t requestId,
                     std::span<const uint32_t> simulation) override {
    maskRequests.emplace_back(
        requestId, std::vector<uint32_t>(simulation.begin(), simulation.end()));
  }
  void completed(uint64_t id, EngineFinishReason, uint32_t prompt,
                 uint32_t completion) override {
    ++completedCount;
    usage[id] = {prompt, completion};
  }
  void failed(uint64_t, std::string code, std::string message,
              bool retryable) override {
    ++failedCount;
    failures.push_back(std::move(code));
    failureDetails.emplace_back(std::move(message), retryable);
  }
  void capacityExhausted(uint64_t, uint32_t, uint32_t, uint64_t) override {
    ++capacityExhaustedCount;
  }

  std::unordered_map<uint64_t, std::vector<uint32_t>> progress;
  std::vector<std::pair<EngineCacheStatus, uint32_t>> starts;
  std::vector<uint64_t> startIds;
  std::unordered_map<uint64_t, std::vector<uint32_t>> outputs;
  std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> usage;
  std::vector<std::string> failures;
  std::vector<std::pair<std::string, bool>> failureDetails;
  uint32_t emitted = 0;
  uint32_t completedCount = 0;
  uint32_t failedCount = 0;
  uint32_t capacityExhaustedCount = 0;
  std::vector<std::pair<uint64_t, std::vector<uint32_t>>> maskRequests;
};

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

EngineRequest request(uint64_t id, const std::vector<uint32_t> &prompt) {
  EngineRequest result;
  result.id = id;
  result.prompt = prompt;
  result.maxNewTokens = 1;
  result.deadlineMilliseconds = 10000;
  return result;
}

void runUntilIdle(engine::Engine &engine) {
  for (uint32_t step = 0; step < 32 && !engine.idle(); ++step) {
    static_cast<void>(engine.tick(step + 1));
  }
  require(engine.idle(), "engine did not reach idle");
}

void testConcurrentColdPrefixesComputeOnce() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache cache(pool, CacheNamespace{});
  Executor model;
  Events events;
  engine::Engine engine({}, cache, model, events);
  for (uint32_t id = 1; id <= 4; ++id) {
    std::vector<uint32_t> prompt(193, 7);
    std::fill(prompt.begin() + 160, prompt.end(), id + 10);
    engine.submit(request(id, prompt));
  }
  static_cast<void>(engine.tick(0));
  require(model.requests.size() == 1,
          "shared cold prefix allocated redundant active state cells");
  runUntilIdle(engine);
  require(model.prefillRows == 160 + 4 * 33 && model.restored == 3 * 160,
          "concurrent cold requests recomputed their shared prefix");
  require(events.completedCount == 4 && events.failedCount == 0 &&
              events.emitted == 4,
          "shared prefill lost independent completions");
}

void testSharedPrefillRebuildsTheMissingJunctionOnce() {
  Backing backing(512);
  KvPool pool(backing);
  engine::Cache cache(pool, CacheNamespace{});
  Executor model;
  Events events;
  engine::Engine engine({}, cache, model, events);
  engine.submit(request(1, std::vector<uint32_t>(6530, 7)));
  runUntilIdle(engine);
  std::vector<uint32_t> branch(6575, 7);
  std::fill(branch.begin() + 6517, branch.end(), 8);
  {
    auto hit = cache.lookup(branch);
    require(hit.kvBoundary == 6496 && hit.resumeBoundary() == 0,
            "fixture did not recreate a KV-only internal branch");
  }
  const uint32_t before = model.prefillRows;
  for (uint32_t id = 2; id <= 5; ++id) {
    std::fill(branch.begin() + 6517, branch.end(), id + 10);
    engine.submit(request(id, branch));
  }
  runUntilIdle(engine);
  require(model.prefillRows - before == 6496 + 4 * (6575 - 6496) &&
              model.restored == 3 * 6496 && events.completedCount == 5,
          "concurrent internal branches each rebuilt the missing GDN state");
}

void testSharedPrefillReleasesDifferentJunctionsIndependently() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache cache(pool, CacheNamespace{});
  Executor model;
  Events events;
  engine::Engine engine({}, cache, model, events);
  engine.submit(request(1, std::vector<uint32_t>(193, 7)));
  for (uint32_t id = 2; id <= 3; ++id) {
    std::vector<uint32_t> branch(193, 7);
    std::fill(branch.begin() + (id == 2 ? 96 : 160), branch.end(), id + 10);
    engine.submit(request(id, branch));
  }
  static_cast<void>(engine.tick(0));
  static_cast<void>(engine.tick(1));
  static_cast<void>(engine.tick(2));
  require(model.requests.size() == 2 && events.startIds.back() == 2,
          "a ready junction waited for the producer's longer shared prefix");
  runUntilIdle(engine);
  require(model.prefillRows == 193 + 97 + 33 && events.completedCount == 3,
          "different shared boundaries were not restored independently");
}

void testSharedPrefillEvictedPublicationFallsBack() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache cache(pool, CacheNamespace{});
  Executor model;
  Events events;
  engine::Engine engine({}, cache, model, events);
  engine.submit(request(1, std::vector<uint32_t>(193, 7)));
  engine.submit(request(2, std::vector<uint32_t>(193, 7)));
  static_cast<void>(engine.tick(0));
  static_cast<void>(engine.tick(1));
  require(cache.reclaimOneState(),
          "published shared prefix was pinned against pressure reclamation");
  runUntilIdle(engine);
  require(events.completedCount == 2 && model.prefillRows == 386 &&
              cache.snapshot().activeRequests == 0,
          "evicted shared publication stranded its waiter");
}

void testSharedPrefillProducerFailureReleasesWaiters() {
  for (bool cancelled : {false, true}) {
    Backing backing(64);
    KvPool pool(backing);
    engine::Cache cache(pool, CacheNamespace{});
    Executor model;
    Events events;
    engine::Engine engine({}, cache, model, events);
    engine.submit(request(1, std::vector<uint32_t>(193, 7)));
    engine.submit(request(2, std::vector<uint32_t>(193, 7)));
    static_cast<void>(engine.tick(0));
    require(engine.snapshot().scheduler.waitingPrefix == 1 &&
                engine.resourceWaitSnapshot(0).memory == 0 &&
                model.requests.size() == 1,
            "shared prefill waiter was admitted before its prefix was ready");
    if (cancelled)
      engine.cancel(1);
    else
      engine.failRequest(1, "test_failure", "producer failed");
    runUntilIdle(engine);
    require(events.outputs[2] == std::vector<uint32_t>{42} &&
                cache.snapshot().activeRequests == 0 && model.requests.empty(),
            "failed prefix producer stranded a waiter or leaked resources");
  }
}

void testSharedPrefillWaiterCancellationAndDeadline() {
  for (bool cancelled : {false, true}) {
    Backing backing(64);
    KvPool pool(backing);
    engine::Cache cache(pool, CacheNamespace{});
    Executor model;
    Events events;
    engine::Engine engine({}, cache, model, events);
    engine.submit(request(1, std::vector<uint32_t>(193, 7)));
    auto waiter = request(2, std::vector<uint32_t>(193, 7));
    waiter.deadlineMilliseconds = 1;
    engine.submit(std::move(waiter));
    static_cast<void>(engine.tick(0));
    if (cancelled)
      engine.cancel(2);
    runUntilIdle(engine);
    require(model.beginAttempts == 1 && events.outputs[2].empty() &&
                events.outputs[1] == std::vector<uint32_t>{42} &&
                engine.snapshot().scheduler.waitingPrefix == 0,
            "expired prefix waiter allocated a cell or interrupted its producer");
  }
}

void testSharedPrefillFailedPublicationFallsBack() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache cache(pool, CacheNamespace{});
  Executor model;
  model.deniedSnapshots = 100;
  Events events;
  engine::Engine engine({}, cache, model, events);
  for (uint32_t id = 1; id <= 4; ++id)
    engine.submit(request(id, std::vector<uint32_t>(193, 7)));
  runUntilIdle(engine);
  require(events.completedCount == 4 && events.failedCount == 0 &&
              model.prefillRows == 4 * 193 && cache.snapshot().activeRequests == 0,
          "a missing prefix snapshot stranded dependent requests");
}

void testSharedPrefillDoesNotBlockUnrelatedWork() {
  for (bool images : {false, true}) {
    Backing backing(64);
    KvPool pool(backing);
    engine::Cache cache(pool, CacheNamespace{});
    Executor model;
    Events events;
    engine::Engine engine({}, cache, model, events);
    for (uint32_t id = 1; id <= 4; ++id) {
      auto value = request(id, std::vector<uint32_t>(193, images ? 7 : id));
      if (images) {
        value.images = {{0, 16, 8, 8, id, id}};
        value.imagePixels.assign(value.images.front().pixelBytes(), 1);
      }
      engine.submit(std::move(value));
    }
    static_cast<void>(engine.tick(0));
    require(model.requests.size() == 4 &&
                engine.snapshot().scheduler.waitingPrefix == 0,
            "unrelated token or image prefixes were serialized");
    runUntilIdle(engine);
    require(model.prefillRows == 4 * 193 && events.completedCount == 4,
            "unrelated work incorrectly reused a shared state");
  }
}

void testSharedPrefillHonorsPriorityAndLateArrival() {
  for (bool foreground : {false, true}) {
    Backing backing(512);
    KvPool pool(backing);
    engine::Cache cache(pool, CacheNamespace{});
    Executor model;
    Events events;
    engine::Engine engine({}, cache, model, events);
    auto producer = request(1, std::vector<uint32_t>(5001, 7));
    producer.priority = RequestPriority::Background;
    engine.submit(std::move(producer));
    static_cast<void>(engine.tick(0));
    auto waiter = request(2, std::vector<uint32_t>(5001, 7));
    waiter.priority = foreground ? RequestPriority::Foreground
                                : RequestPriority::Background;
    engine.submit(std::move(waiter));
    static_cast<void>(engine.tick(1));
    static_cast<void>(engine.tick(2));
    require(model.requests.size() == (foreground ? 2u : 1u),
            "prefix admission ignored priority or a late arrival");
    runUntilIdle(engine);
    require(events.completedCount == 2 && events.failedCount == 0,
            "late prefix waiter did not finish");
    if (!foreground)
      require(model.prefillRows == 5010 && model.restored == 4992,
              "late arrival missed the producer's planned replay point");
  }
}

void testLateSharedPrefillExtendsTheProducerPlan() {
  for (bool denyCheckpoint : {false, true}) {
    Backing backing(512);
    KvPool pool(backing);
    engine::Cache cache(pool, CacheNamespace{});
    Executor model;
    if (denyCheckpoint)
      model.denySnapshotAtBoundary = 4096;
    Events events;
    engine::Engine engine({}, cache, model, events);
    engine.submit(request(1, std::vector<uint32_t>(6601, 7)));
    static_cast<void>(engine.tick(0));
    for (uint32_t id = 2; id <= 4; ++id) {
      std::vector<uint32_t> branch(6601, 7);
      std::fill(branch.begin() + 6500, branch.end(), id + 10);
      engine.submit(request(id, branch));
    }
    runUntilIdle(engine);
    require(events.completedCount == 4 && events.failedCount == 0 &&
                model.prefillRows == 6601 + 3 * (6601 - 6496) &&
                model.restored == 3 * 6496,
            "late siblings recomputed the prefix after the producer's checkpoint");
  }
}

void testSharedPrefillCapacityFailureDoesNotDeadlock() {
  Backing backing(4);
  KvPool pool(backing);
  engine::Cache cache(pool, CacheNamespace{});
  Executor model;
  Events events;
  engine::Engine engine({}, cache, model, events);
  for (uint32_t id = 1; id <= 4; ++id)
    engine.submit(request(id, std::vector<uint32_t>(193, 7)));
  for (uint32_t step = 0; step < 64 && !engine.idle(); ++step)
    static_cast<void>(engine.tick(step * 1000));
  require(engine.idle() && cache.snapshot().activeRequests == 0 &&
              model.requests.empty() && events.emitted == 0,
          "capacity failure stranded a shared prefix producer or waiter");
  engine.submit(request(5, std::vector<uint32_t>(33, 9)));
  runUntilIdle(engine);
  require(events.outputs[5] == std::vector<uint32_t>{42},
          "capacity failure prevented subsequent service");
}

void testColdPublishesReplayStateAndLazyJunctionCanRebuildIt() {
  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  Events events;
  engine::Engine engine({.maxContext = 102400}, resources, executor, events);
  require(engine.snapshot().maximumContextTokens == 102400,
          "snapshot lost the configured context limit");
  std::vector<uint32_t> prompt(65);
  for (uint32_t i = 0; i < prompt.size(); ++i)
    prompt[i] = i + 1;

  engine.submit(request(1, prompt));
  runUntilIdle(engine);
  require(executor.prefillRows == 65 && executor.snapshots == 1 &&
              engine.snapshot().replayStatePublications == 1,
          "cold request did not retain its latest Page32 replay state");
  require(events.starts.size() == 1 &&
              events.starts[0].first == EngineCacheStatus::Miss,
          "cold request reported a cache hit");

  require(resources.reclaimCache(1, false) != 0,
          "test could not remove the latest replay state");
  engine.submit(request(2, prompt));
  runUntilIdle(engine);
  require(executor.restored == 0 && executor.prefillRows == 130 &&
              executor.snapshots == 2 &&
              engine.snapshot().junctionMaterializations == 1,
          "second request did not lazily materialize its proven KV junction");
  require(events.starts.size() == 2 &&
              events.starts[1].first == EngineCacheStatus::Miss,
          "KV-only junction replay was reported as a state hit");

  engine.submit(request(3, prompt));
  runUntilIdle(engine);
  require(executor.restored == 64 && executor.prefillRows == 131,
          "cache hit did not replay exactly one real input token");
  require(events.starts.size() == 3 &&
              events.starts[2] ==
                  std::pair<EngineCacheStatus, uint32_t>{
                      EngineCacheStatus::PrefixHit, 64},
          "state-backed hit accounting is wrong");
  require(events.completedCount == 3 && events.failedCount == 0 &&
              events.emitted == 3,
          "request lifecycle did not complete cleanly");
}

void testConcurrentDuplicateStateSkipsSnapshotCapture() {
  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  std::vector<uint32_t> prompt(65);
  for (uint32_t index = 0; index < prompt.size(); ++index)
    prompt[index] = index + 1;

  engine.submit(request(101, prompt));
  engine.submit(request(102, prompt));
  runUntilIdle(engine);

  // The second request restores the first publication without reserving a
  // redundant state cell or capturing the same state again.
  const auto snapshot = engine.snapshot();
  require(executor.snapshotAttempts == 1 && executor.snapshots == 1,
          "duplicate state publication performed a second snapshot capture");
  require(snapshot.resources.stateCache.entries == 1 &&
              snapshot.resources.stateCache.publications == 1 &&
              snapshot.cacheHits == 1 && executor.prefillRows == 66 &&
              snapshot.replayStatePublications == 1 &&
              snapshot.replayStatePublicationFailures == 0,
          "duplicate state publication was not reused and accounted");
}

void testImageSpansKeyPrefixIdentity() {
  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  // Image runs render as one placeholder id, so two prompts with different
  // images are token-identical; only the span digests differ.
  std::vector<uint32_t> prompt(65, 248056);
  auto withImage = [&](uint64_t id, uint64_t digest) {
    EngineRequest result = request(id, prompt);
    result.images = {{8, 16, 8, 8, digest, digest ^ 0xabcdULL}};
    result.imagePixels.assign(result.images[0].pixelBytes(), 1);
    return result;
  };

  engine.submit(withImage(1, 0x1111));
  runUntilIdle(engine);
  require(events.starts.size() == 1 &&
              events.starts[0].first == EngineCacheStatus::Miss,
          "image producer unexpectedly hit the cache");
  engine.submit(withImage(2, 0x2222));
  runUntilIdle(engine);
  require(events.starts.size() == 2 &&
              events.starts[1] ==
                  std::pair<EngineCacheStatus, uint32_t>{
                      EngineCacheStatus::Miss, 0},
          "a different image falsely matched the cached prefix");
  engine.submit(withImage(3, 0x1111));
  runUntilIdle(engine);
  require(events.starts.size() == 3 &&
              events.starts[2] ==
                  std::pair<EngineCacheStatus, uint32_t>{
                      EngineCacheStatus::PrefixHit, 64},
          "an identical image did not reuse the cached prefix");
  engine.submit(request(4, prompt));
  runUntilIdle(engine);
  require(events.starts.size() == 4 &&
              events.starts[3] ==
                  std::pair<EngineCacheStatus, uint32_t>{
                      EngineCacheStatus::Miss, 0},
          "a text-only prompt matched an image-keyed prefix");

  EngineRequest malformed = withImage(5, 0x1111);
  malformed.images[0].tokens = 15;
  bool rejected = false;
  try {
    engine.submit(std::move(malformed));
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected,
          "image span with the wrong merged token count was admitted");
  require(events.completedCount == 4 && events.failedCount == 0,
          "image request lifecycle did not complete cleanly");
}

void testOneRequestPublishesJunctionAndLatestReplayState() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);

  std::vector<uint32_t> prompt(97);
  for (uint32_t index = 0; index < prompt.size(); ++index)
    prompt[index] = index + 1;
  engine.submit(request(5, prompt));
  runUntilIdle(engine);
  while (resources.snapshot().stateCache.entries != 0) {
    require(resources.reclaimCache(1, false) != 0,
            "test could not leave a KV-only shared prefix");
  }

  prompt.resize(161, 777);
  engine.submit(request(6, prompt));
  runUntilIdle(engine);

  const DraftContextPlan &plan = executor.plans.at(6);
  const auto snapshot = engine.snapshot();
  require(plan.boundaries.size() == 3 && plan.boundaries[0].boundary == 96 &&
              plan.boundaries[1].boundary == 160 &&
              plan.boundaries[2].boundary == 161,
          "junction, latest replay state, and active end were not ordered");
  require(executor.snapshots == 3 && snapshot.junctionMaterializations == 1 &&
              snapshot.replayStatePublications == 2 &&
              snapshot.resources.stateCache.entries == 2,
          "one request did not retain both sparse composite states");
}

// A denied snapshot at the latest replay boundary recycles the least recently
// used cached state, which is an older unrelated state, not the junction state
// this same request published one command earlier.
void testLatestReplayDenialRecyclesOlderStateNotTheJunction() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);

  std::vector<uint32_t> prompt(97);
  for (uint32_t index = 0; index < prompt.size(); ++index)
    prompt[index] = index + 1;
  engine.submit(request(7, prompt));
  runUntilIdle(engine);
  while (resources.snapshot().stateCache.entries != 0) {
    require(resources.reclaimCache(1, false) != 0,
            "test could not remove the old composite state");
  }
  engine.submit(request(70, std::vector<uint32_t>(65, 7000)));
  runUntilIdle(engine);
  const auto older = resources.snapshot();
  require(older.stateCache.entries == 1 && older.stateCache.evictions == 1,
          "older unrelated state was not left in the cache");

  prompt.resize(161, 888);
  engine.submit(request(8, prompt));
  require(engine.tick(1) && engine.tick(2),
          "request did not publish its junction state");
  require(engine.snapshot().junctionMaterializations == 1 &&
              resources.snapshot().stateCache.entries == 2,
          "lazy junction was not materialized in the first command");
  executor.deniedSnapshots = 1;
  runUntilIdle(engine);

  const DraftContextPlan &plan = executor.plans.at(8);
  const auto snapshot = engine.snapshot();
  require(plan.boundaries.size() == 3 && plan.boundaries[0].boundary == 96 &&
              plan.boundaries[1].boundary == 160 &&
              plan.boundaries[2].boundary == 161,
          "boundaries were not armed without reservation");
  require(executor.deniedSnapshots == 0 &&
              snapshot.recycledStatePublications == 1 &&
              snapshot.replayStatePublications == 3 &&
              snapshot.replayStatePublicationFailures == 0 &&
              snapshot.resources.stateCache.evictions == 2 &&
              snapshot.resources.stateCache.entries == 2,
          "latest-state denial did not recycle exactly one older state");

  // The junction state survived the recycle: the original prompt resumes
  // from it, while the recycled prompt is back to a KV-only junction.
  const uint32_t restored = executor.restored;
  prompt.resize(97);
  engine.submit(request(9, prompt));
  runUntilIdle(engine);
  require(executor.restored == restored + 96 &&
              events.starts.back() ==
                  std::pair<EngineCacheStatus, uint32_t>{
                      EngineCacheStatus::PrefixHit, 96},
          "latest-state denial discarded the independent junction state");
  engine.submit(request(71, std::vector<uint32_t>(65, 7000)));
  runUntilIdle(engine);
  require(events.starts.back().first == EngineCacheStatus::Miss &&
              engine.snapshot().junctionMaterializations == 2,
          "recycled state was not the older unrelated one");
}

void testCancellationAfterJunctionDiscardsLaterState() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);

  std::vector<uint32_t> prompt(97);
  for (uint32_t index = 0; index < prompt.size(); ++index)
    prompt[index] = index + 1;
  engine.submit(request(9, prompt));
  runUntilIdle(engine);
  while (resources.snapshot().stateCache.entries != 0) {
    require(resources.reclaimCache(1, false) != 0,
            "test could not remove the old composite state");
  }

  prompt.resize(161, 999);
  engine.submit(request(10, prompt));
  require(engine.tick(1) && engine.tick(2),
          "request did not publish its first sparse state");
  const auto published = engine.snapshot();
  require(published.resources.stateCache.entries == 1 &&
              published.junctionMaterializations == 1 &&
              executor.snapshotAttempts == 2,
          "junction publication did not land in the first command");
  engine.cancel(10);
  runUntilIdle(engine);
  // The armed 160 boundary is dropped with the lane: no snapshot, no
  // failure counted, and the junction state stays cached.
  const auto cancelled = engine.snapshot();
  require(executor.snapshotAttempts == 2 &&
              cancelled.resources.stateCache.entries == 1 &&
              cancelled.resources.stateCache.publications == 2 &&
              cancelled.replayStatePublications == 1 &&
              cancelled.replayStatePublicationFailures == 0 &&
              cancelled.cancelled == 1 && executor.requests.empty() &&
              cancelled.resources.activeRequests == 0,
          "cancellation leaked or removed the wrong sparse state");
}

// Nothing is reserved at admission, so a boundary the model cannot snapshot
// costs exactly one attempt: the failure is counted under its purpose, the
// request is served normally, and with an empty cache nothing is recycled.
void testDeniedSnapshotCostsOnlyThatAttempt() {
  Backing backing(16);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  executor.deniedSnapshots = 100;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  std::vector<uint32_t> prompt(65, 7);

  engine.submit(request(3, prompt));
  runUntilIdle(engine);
  const DraftContextPlan &plan = executor.plans.at(3);
  require(plan.boundaries.size() == 2 && plan.boundaries[0].boundary == 64 &&
              plan.boundaries[1].boundary == 65 &&
              plan.draftContextRows() == prompt.size(),
          "replay boundary was not armed without reservation");
  require(executor.snapshotAttempts == 1 && executor.snapshots == 0 &&
              engine.snapshot().replayStatePublicationFailures == 1 &&
              engine.snapshot().junctionMaterializationFailures == 0 &&
              engine.snapshot().recycledStatePublications == 0 &&
              resources.snapshot().stateCache.evictions == 0 &&
              events.completedCount == 1,
          "latest replay-state denial was counted incorrectly");

  // The first pass left a usable KV chain but no state. The second lookup is
  // therefore a real lazy junction; its denied snapshot is counted as such
  // while the request replays through the KV prefix normally.
  engine.submit(request(4, prompt));
  runUntilIdle(engine);
  require(executor.snapshotAttempts == 2 && executor.snapshots == 0 &&
              engine.snapshot().junctionMaterializationFailures == 1 &&
              engine.snapshot().replayStatePublicationFailures == 1 &&
              events.starts.back().first == EngineCacheStatus::Miss &&
              executor.prefillRows == 130 && events.completedCount == 2 &&
              events.failedCount == 0,
          "denied lazy junction failed the request or was not counted");

  // The denial is transient: the next lane over the same prefix lands it.
  executor.deniedSnapshots = 0;
  engine.submit(request(5, prompt));
  runUntilIdle(engine);
  require(executor.snapshots == 1 &&
              engine.snapshot().junctionMaterializations == 1 &&
              resources.snapshot().stateCache.entries == 1,
          "junction was not materialized once snapshots were possible");
}

// A snapshot the model denies once at a replay boundary lands by recycling
// the least recently used cached state: one eviction, the same number of
// entries, and the recycled slot now holds the new lane's state.
void testDeniedSnapshotRecyclesLruStateAndRetries() {
  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> older(65, 1);
  const std::vector<uint32_t> newer(65, 2);
  engine.submit(request(1, older));
  runUntilIdle(engine);
  const auto cached = resources.snapshot();
  require(cached.stateCache.entries == 1 && cached.stateCache.evictions == 0,
          "recycle fixture did not cache the older state");

  executor.deniedSnapshots = 1;
  engine.submit(request(2, newer));
  runUntilIdle(engine);
  const auto recycled = engine.snapshot();
  require(executor.snapshotAttempts == 3 && executor.snapshots == 2 &&
              recycled.recycledStatePublications == 1 &&
              recycled.replayStatePublications == 2 &&
              recycled.replayStatePublicationFailures == 0 &&
              events.completedCount == 2,
          "denied snapshot was not retried after recycling a state");
  require(recycled.resources.stateCache.evictions == 1 &&
              recycled.resources.stateCache.entries == 1 &&
              recycled.resources.stateCache.publications == 2 &&
              recycled.resources.stateCache.bytes == cached.stateCache.bytes,
          "recycling changed the cached state footprint");

  // The surviving state belongs to the new lane: re-sending its prompt is a
  // state hit, while the older prompt is back to a KV-only junction.
  engine.submit(request(3, newer));
  runUntilIdle(engine);
  require(executor.restored == 64 &&
              events.starts.back() ==
                  std::pair<EngineCacheStatus, uint32_t>{
                      EngineCacheStatus::PrefixHit, 64},
          "recycled slot does not hold the new lane's state");
  engine.submit(request(4, older));
  runUntilIdle(engine);
  require(executor.restored == 64 &&
              events.starts.back().first == EngineCacheStatus::Miss &&
              engine.snapshot().junctionMaterializations == 1,
          "recycled state was not the least recently used one");
}

// When the retry after recycling is denied as well, the boundary fails and
// the engine has paid exactly one cached state for it. Persistent denial
// never drains the rest of the cache.
void testPersistentSnapshotDenialRecyclesAtMostOneState() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  for (uint64_t id : {1, 2, 3}) {
    engine.submit(request(id, std::vector<uint32_t>(65, id)));
    runUntilIdle(engine);
  }
  require(resources.snapshot().stateCache.entries == 3,
          "persistent-denial fixture did not cache three states");

  executor.deniedSnapshots = 100;
  engine.submit(request(4, std::vector<uint32_t>(65, 4)));
  runUntilIdle(engine);
  const auto after = engine.snapshot();
  require(executor.snapshotAttempts == 5 && executor.snapshots == 3 &&
              executor.deniedSnapshots == 98 &&
              after.replayStatePublicationFailures == 1 &&
              after.recycledStatePublications == 0 &&
              events.completedCount == 4 && events.failedCount == 0,
          "persistently denied snapshot was retried more than once");
  require(after.resources.stateCache.evictions == 1 &&
              after.resources.stateCache.entries == 2,
          "persistent snapshot denial recycled more than one cached state");
}

void testLongSuffixSkipsDraftRestore() {
  Backing backing(256);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  std::vector<uint32_t> prefix(65);
  for (uint32_t index = 0; index < prefix.size(); ++index) {
    prefix[index] = index + 1;
  }
  engine.submit(request(10, prefix));
  runUntilIdle(engine);

  std::vector<uint32_t> extended = prefix;
  extended.resize(4097, 99);
  engine.submit(request(11, extended));
  runUntilIdle(engine);
  require(!executor.restoredDraft &&
              executor.plans.at(11).draftStateRestoreSkipped,
          "long suffix copied a draft ring that its final window overwrites");
}

// A lane cancelled while the command that ends at its armed boundary is in
// flight publishes nothing when that command drains: no snapshot is taken,
// no failure is counted, and the blocks committed earlier stay cached.
void testCancellationInFlightAtBoundaryPublishesNoState() {
  Backing backing(256);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  std::vector<uint32_t> prompt(4097, 11);

  engine.submit(request(4, prompt));
  require(engine.tick(1) && engine.tick(2),
          "long request did not complete its first prefill command");
  require(resources.snapshot().kvCache.blocks == 64 &&
              executor.snapshotAttempts == 0,
          "first prefill command changed the state cache");
  require(engine.tick(3) && engine.commandInFlight(),
          "second prefill command was not launched");
  engine.cancel(4);
  require(engine.commandInFlight() && executor.requests.size() == 1,
          "in-flight cancellation released resources owned by the command");
  runUntilIdle(engine);
  const auto snapshot = engine.snapshot();
  require(executor.snapshotAttempts == 0 &&
              snapshot.resources.stateCache.entries == 0 &&
              snapshot.resources.stateCache.publications == 0 &&
              snapshot.replayStatePublications == 0 &&
              snapshot.replayStatePublicationFailures == 0 &&
              snapshot.cancelled == 1,
          "cancelled command snapshotted or counted its boundary");
  require(snapshot.resources.kvCache.blocks == 64 &&
              snapshot.resources.pool.pagesActive == 0 &&
              snapshot.resources.activeRequests == 0 &&
              executor.requests.empty(),
          "cancellation leaked active resources or dropped committed KV");
}

void testActiveCellGrowthReclaimsCachedStateAndRetries() {
  for (bool hostPressure : {false, true}) {
    Backing backing(16);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(1);
    Events events;
    EngineConfig config;
    config.growthPaused = [] { return false; };
    engine::Engine engine(config, resources, executor, events);

    engine.submit(request(20, std::vector<uint32_t>(65, 7)));
    runUntilIdle(engine);
    const auto cached = resources.snapshot();
    require(cached.stateCache.entries == 1,
            "state-reclaim setup did not publish a composite state");

    if (hostPressure) {
      backing.growthBlocked = true;
      executor.physicalGrowthBlocked = &backing.growthBlocked;
      executor.beginGrowthBlocked = [&] { return backing.growthBlocked; };
      executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
      executor.reclaimableIdleStateBytes = 350'224'384;
    } else {
      executor.deniedBegins = 1;
    }
    const uint32_t attempts = executor.beginAttempts;
    engine.submit(request(21, {8}));
    runUntilIdle(engine);
    const auto after = resources.snapshot();
    require(executor.beginAttempts == attempts + 2 &&
                events.completedCount == 2,
            "active-cell growth did not reclaim memory and retry");
    if (hostPressure) {
      require(executor.reclaimedIdleStateBytes == 350'224'384 &&
                  !backing.growthBlocked &&
                  after.stateCache.entries == cached.stateCache.entries &&
                  after.stateCache.evictions == cached.stateCache.evictions,
              "host-pressure state admission evicted cache or ignored idle memory");
    } else {
      require(after.stateCache.entries == 0,
              "active-cell growth did not reclaim cached state");
    }
  }
}

void testKvGrowthReclaimsIdleStateBeforeCache() {
  for (auto failure : {metal::AllocationFailure::Capacity,
                       metal::AllocationFailure::HostPressure}) {
    Backing backing(12);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(1);
    Events events;
    EngineConfig config;
    config.growthPaused = [] { return false; };
    engine::Engine engine(config, resources, executor, events);

    engine.submit(request(25, std::vector<uint32_t>(65, 25)));
    runUntilIdle(engine);
    const auto cached = resources.snapshot();
    require(cached.stateCache.entries == 1,
            "idle-state reclaim setup did not retain cache state");

    backing.growthBlocked = true;
    backing.allocationFailure = failure;
    executor.physicalGrowthBlocked = &backing.growthBlocked;
    executor.unblockGrowthOnSuspend = false;
    executor.reclaimableIdleStateBytes = 350'224'384;
    engine.submit(request(26, std::vector<uint32_t>(161, 26)));
    runUntilIdle(engine);

    const auto after = resources.snapshot();
    require(executor.reclaimedIdleStateBytes == 350'224'384 &&
                !backing.growthBlocked && events.completedCount == 2 &&
                engine.snapshot().resourceSuspensions == 0,
            "KV growth did not reclaim model-owned idle state before suspension");
    // The growth denial neither evicts the cached state nor touches the growing
    // lane's own armed boundary: its replay state still lands.
    require(after.stateCache.evictions == cached.stateCache.evictions &&
                after.stateCache.entries == cached.stateCache.entries + 1 &&
                engine.snapshot().replayStatePublications == 2 &&
                engine.snapshot().recycledStatePublications == 0,
            "KV growth evicted useful composite state before idle state memory");
  }
}

// Two lanes prefill toward their replay boundaries in one command. The
// physical KV growth denial the first lane meets is answered by reclaiming
// idle model state, never by dropping any lane's armed boundary: both replay
// states land.
void testKvGrowthDenialKeepsEveryLaneReplayState() {
  Backing backing(16);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  backing.growthBlocked = true;
  executor.physicalGrowthBlocked = &backing.growthBlocked;
  executor.reclaimableIdleStateBytes = 1U << 20;

  engine.submit(request(300, std::vector<uint32_t>(161, 300)));
  engine.submit(request(301, std::vector<uint32_t>(161, 301)));
  runUntilIdle(engine);
  const auto after = engine.snapshot();
  require(executor.reclaimedIdleStateBytes == (1U << 20) &&
              !backing.growthBlocked && executor.suspensions == 0 &&
              events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0,
          "physical growth denial suspended or failed a lane");
  require(!executor.prefillWidths.empty() && executor.prefillWidths[0] == 2,
          "both lanes did not prefill to their boundaries in one command");
  require(executor.snapshots == 2 && after.replayStatePublications == 2 &&
              after.replayStatePublicationFailures == 0 &&
              after.recycledStatePublications == 0 &&
              after.resources.stateCache.entries == 2 &&
              after.resources.stateCache.evictions == 0,
          "KV growth denial cost a lane its replay state");
}

void testPressureReclaimRespectsStateLifetimes() {
  Backing backing(8);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);

  engine.submit(request(27, std::vector<uint32_t>(65, 27)));
  runUntilIdle(engine);
  const auto cached = resources.snapshot();
  require(cached.stateCache.entries == 1 && cached.kvCache.blocks == 2,
          "pressure setup did not retain KV and composite state");

  // A shrink that nothing is waiting for stops at the resume point. Freeing
  // its cell gains the host a little; the next request pays a full replay.
  static_cast<void>(engine.reclaimMemory({.reclaimEmptyKvExtents = true,
                                          .targetBytes = 64,
                                          .keepResumePoint = true}));
  require(resources.snapshot().stateCache.entries == 1,
          "a speculative shrink discarded the only resume point");

  executor.reclaimableIdleStateBytes = 128;
  const uint64_t first = engine.reclaimMemory(
      {.reclaimEmptyKvExtents = true, .targetBytes = 64});
  require(first >= 128 && executor.reclaimedIdleStateBytes == 128,
          "pressure reclaim did not release idle active-state backing first");
  require(
      resources.snapshot().stateCache.entries == 1,
      "pressure reclaim evicted a cached state before satisfying its target");

  // Drain any initially resident but unused test extents, then prove cached
  // state is the next lifecycle selected while its parent KV remains usable.
  static_cast<void>(engine.reclaimMemory({.reclaimEmptyKvExtents = true}));
  require(engine.reclaimMemory({.reclaimEmptyKvExtents = true,
                                .targetBytes = 64}) >= 64,
          "pressure reclaim did not release immutable cached state");
  const auto stateEvicted = resources.snapshot();
  require(stateEvicted.stateCache.entries == 0 &&
              stateEvicted.kvCache.blocks == 2,
          "cached-state eviction incorrectly removed target KV");

  static_cast<void>(
      engine.reclaimMemory({.reclaimEmptyKvExtents = true,
                            .evictAllUnpinnedPrefixes = true,
                            .targetBytes = std::numeric_limits<uint64_t>::max()}));
  const auto critical = resources.snapshot();
  require(critical.stateCache.entries == 0 && critical.kvCache.blocks == 0 &&
              critical.pool.residentBackingBytes == 0,
          "critical pressure left evictable cached state or KV backing");
}

void testConcurrencyLimitDoesNotEvictCache() {
  Backing backing(256);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);

  engine.submit(request(22, std::vector<uint32_t>(65, 22)));
  runUntilIdle(engine);
  const auto cached = resources.snapshot();
  require(cached.stateCache.entries == 1,
          "concurrency-limit setup did not retain a cache state");

  engine.submit(request(23, std::vector<uint32_t>(4097, 23)));
  require(engine.tick(1) && engine.tick(2),
          "resident request did not start long prefill");
  engine.submit(request(24, {24}));
  require(engine.tick(3), "saturated engine made no resident progress");
  const auto waiting = engine.resourceWaitSnapshot(13.0);
  require(waiting.concurrency == 1 && waiting.memory == 0 &&
              waiting.suspended == 0 && waiting.oldestWaitMilliseconds == 10.0,
          "concurrency wait diagnostics were confused with memory pressure");

  const auto saturated = resources.snapshot();
  require(saturated.stateCache.entries == cached.stateCache.entries &&
              saturated.stateCache.evictions == cached.stateCache.evictions,
          "active-cell saturation was mistaken for memory pressure");

  engine.cancel(23);
  engine.cancel(24);
  runUntilIdle(engine);
}

void testHostPressureDoesNotDrainCacheOnStateAdmission() {
  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  MemoryPressure pressure = MemoryPressure::Normal;
  EngineConfig config;
  config.growthPaused = [&] { return pressure != MemoryPressure::Normal; };
  engine::Engine engine(config, resources, executor, events);
  engine.submit(request(200, std::vector<uint32_t>(65, 200)));
  runUntilIdle(engine);
  const auto cached = resources.snapshot();
  const uint64_t coldMisses = engine.snapshot().coldMisses;
  const uint32_t attempts = executor.beginAttempts;

  pressure = MemoryPressure::Warning;
  executor.deniedBegins = 100;
  engine.submit(request(201, {201}));
  static_cast<void>(engine.tick(20.0));
  static_cast<void>(engine.tick(50.0));
  require(executor.beginAttempts == attempts + 1 &&
              resources.snapshot().stateCache.entries ==
                  cached.stateCache.entries &&
              resources.snapshot().kvCache.blocks == cached.kvCache.blocks &&
              resources.snapshot().lookup.lookups == cached.lookup.lookups &&
              engine.snapshot().coldMisses == coldMisses &&
              engine.snapshot().scheduler.waitingResources == 1,
          "paused state admission drained the cache or retried without backoff");

  pressure = MemoryPressure::Normal;
  executor.deniedBegins = 0;
  for (double now = 120.0; now < 140.0 && !engine.idle(); ++now)
    static_cast<void>(engine.tick(now));
  require(engine.idle() && events.completedCount == 2 &&
              resources.snapshot().lookup.lookups == cached.lookup.lookups + 1 &&
              engine.snapshot().coldMisses == coldMisses + 1 &&
              events.capacityExhaustedCount == 0 && events.failedCount == 0,
          "state admission did not recover after host pressure cleared");
}

// Host pressure pauses foreground cache eviction for KV growth, but landing a
// denied snapshot by recycling one cached state is memory-neutral and still
// happens: the state footprint does not grow, and KV is untouched.
void testHostPressureStillRecyclesLruStateForDeniedSnapshot() {
  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  MemoryPressure pressure = MemoryPressure::Normal;
  EngineConfig config;
  config.growthPaused = [&] { return pressure != MemoryPressure::Normal; };
  engine::Engine engine(config, resources, executor, events);
  engine.submit(request(210, std::vector<uint32_t>(65, 210)));
  runUntilIdle(engine);
  const auto cached = resources.snapshot();
  const uint32_t attempts = executor.snapshotAttempts;

  pressure = MemoryPressure::Warning;
  executor.deniedSnapshots = 1;
  engine.submit(request(211, std::vector<uint32_t>(65, 211)));
  runUntilIdle(engine);
  const auto after = engine.snapshot();
  require(engine.idle() && events.completedCount == 2 &&
              executor.snapshotAttempts == attempts + 2 &&
              after.recycledStatePublications == 1 &&
              after.replayStatePublications == 2 &&
              after.replayStatePublicationFailures == 0,
          "denied snapshot under host pressure was not landed by recycling");
  require(after.resources.stateCache.evictions ==
                  cached.stateCache.evictions + 1 &&
              after.resources.stateCache.entries == cached.stateCache.entries &&
              after.resources.stateCache.bytes == cached.stateCache.bytes &&
              after.resources.kvCache.blocks == cached.kvCache.blocks + 2,
          "recycling under host pressure grew state or drained KV cache");
}

// A lone request short of pages under host pressure takes idle cached pages
// that are already resident instead of being suspended and replaying its
// prefix later. Reuse only happens when it can cover the shortfall.
void testSingletonHostPressureReusesIdleCacheInsteadOfSuspending() {
  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  MemoryPressure pressure = MemoryPressure::Normal;
  EngineConfig config;
  config.growthPaused = [&] { return pressure != MemoryPressure::Normal; };
  engine::Engine engine(config, resources, executor, events);
  engine.submit(request(230, std::vector<uint32_t>(65, 230)));
  runUntilIdle(engine);
  const auto cached = resources.snapshot();
  require(cached.kvCache.blocks >= 1 && cached.pool.pagesPrefix >= 2 &&
              cached.pool.pagesFreeResident + cached.pool.pagesPrefix >= 3,
          "idle-cache reuse fixture geometry changed");

  pressure = MemoryPressure::Warning;
  backing.growthBlocked = true;
  engine.submit(request(231, std::vector<uint32_t>(65, 231)));
  runUntilIdle(engine);
  require(engine.idle() && events.completedCount == 2 &&
              events.failedCount == 0 && events.capacityExhaustedCount == 0,
          "lone request under host pressure did not complete on idle cache");
  require(executor.suspensions == 0 &&
              engine.snapshot().resourceSuspensions == 0 &&
              backing.growthAttempts > 0,
          "lone request was suspended although idle cached pages could serve it");
  require(resources.snapshot().pool.pagesResident == cached.pool.pagesResident,
          "idle-cache reuse grew the resident footprint");
}

void testSingletonHostPressureWaitRecoversOrTerminates() {
  for (uint32_t outcome = 0; outcome < 4; ++outcome) {
    Backing backing(32);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(1);
    Events events;
    MemoryPressure pressure = MemoryPressure::Normal;
    EngineConfig config;
    config.growthPaused = [&] { return pressure != MemoryPressure::Normal; };
    config.resourceWaitTimeoutMilliseconds = outcome == 3 ? 100.0 : 30000.0;
    engine::Engine engine(config, resources, executor, events);
    engine.submit(request(220, std::vector<uint32_t>(65, 220)));
    runUntilIdle(engine);
    const auto cached = resources.snapshot();

    pressure = MemoryPressure::Warning;
    backing.growthBlocked = true;
    auto value = request(221, std::vector<uint32_t>(161, 221));
    value.deadlineMilliseconds = 300.0;
    engine.submit(std::move(value));
    static_cast<void>(engine.tick(20.0));
    static_cast<void>(engine.tick(21.0));
    // Suspension drops the lane's armed boundary silently: it is neither a
    // snapshot attempt nor a publication failure.
    require(executor.suspensions == 1 && events.capacityExhaustedCount == 0 &&
                resources.snapshot().kvCache.blocks == cached.kvCache.blocks &&
                resources.snapshot().stateCache.evictions ==
                    cached.stateCache.evictions &&
                executor.snapshotAttempts == 1 &&
                engine.snapshot().replayStatePublicationFailures == 0,
            "singleton host pressure killed active work or drained its cache");
    const uint32_t attempts = executor.resumeAttempts;
    static_cast<void>(engine.tick(50.0));
    require(executor.resumeAttempts == attempts,
            "suspended host-pressure request ignored resource backoff");
    const auto waiting = engine.resourceWaitSnapshot(50.0);
    require(waiting.memory == 1 && waiting.concurrency == 0 &&
                waiting.suspended == 1 && waiting.oldestWaitMilliseconds >= 29.0,
            "resource diagnostics lost suspension reason or original wait time");

    if (outcome == 0) {
      pressure = MemoryPressure::Normal;
      backing.growthBlocked = false;
      for (double now = 121.0; now < 140.0 && !engine.idle(); ++now)
        static_cast<void>(engine.tick(now));
      // The resumed lane re-arms its replay boundary and publishes normally.
      require(events.completedCount == 2 && events.failedCount == 0 &&
                  engine.snapshot().replayStatePublications == 2 &&
                  engine.snapshot().replayStatePublicationFailures == 0,
              "singleton request did not recover after pressure cleared");
    } else if (outcome == 1) {
      engine.cancel(221);
      static_cast<void>(engine.tick(60.0));
      require(engine.snapshot().cancelled == 1,
              "host-pressure wait ignored cancellation");
    } else if (outcome == 3) {
      static_cast<void>(engine.tick(150.0));
      require(events.failures == std::vector<std::string>{"resource_timeout"} &&
                  events.failureDetails.back().second,
              "persistent memory pressure did not fail with a retryable timeout");
    } else {
      static_cast<void>(engine.tick(301.0));
      require(events.completedCount + events.failedCount == 2,
              "host-pressure wait ignored deadline");
    }
    require(engine.idle() && executor.requests.empty() &&
                resources.snapshot().activeRequests == 0 &&
                resources.snapshot().pool.pagesActive == 0 &&
                events.capacityExhaustedCount == 0 &&
                resources.snapshot().stateCache.entries ==
                    cached.stateCache.entries + (outcome == 0 ? 1U : 0U),
            "host-pressure wait leaked request state or active KV");
    const auto cleared = engine.resourceWaitSnapshot(400.0);
    require(cleared.memory == 0 && cleared.concurrency == 0 &&
                cleared.suspended == 0 && cleared.oldestWaitMilliseconds == 0.0,
            "terminal work retained resource wait diagnostics");
  }
}

void testKvPressureNarrowsTheRealBatch() {
  Backing backing(3);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  Events events;
  engine::Engine engine({}, resources, executor, events);

  engine.submit(request(30, std::vector<uint32_t>(33, 30)));
  engine.submit(request(31, std::vector<uint32_t>(33, 31)));
  runUntilIdle(engine);

  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0,
          "KV pressure failed a lane that could run in a narrower batch");
  require(std::find(executor.prefillWidths.begin(),
                    executor.prefillWidths.end(),
                    1) != executor.prefillWidths.end(),
          "resource pressure did not degrade B2 prefill to real B1 commands");
}

// Cached state and KV share one physical budget. Required KV growth for a
// live lane takes the least recently used cached state through the unified
// reclaim order instead of failing or suspending, and the growing lane still
// publishes its own replay state afterwards.
void testKvGrowthReclaimsCachedStateWhenBudgetIsShared() {
  Backing backing(8);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  backing.growthAllowed = [&] {
    return resources.snapshot().stateCache.entries == 0;
  };
  Events events;
  engine::Engine engine({}, resources, executor, events);
  engine.submit(request(230, std::vector<uint32_t>(65, 230)));
  runUntilIdle(engine);
  const auto cached = resources.snapshot();
  require(cached.stateCache.entries == 1 && cached.pool.pagesResident == 4,
          "shared-budget fixture did not cache one state in one extent");

  engine.submit(request(231, std::vector<uint32_t>(161, 231)));
  runUntilIdle(engine);
  const auto after = engine.snapshot();
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0 &&
              executor.suspensions == 0 && executor.prefillRows == 226,
          "required KV growth failed or suspended behind cached state");
  require(after.resources.stateCache.evictions == 1 &&
              after.resources.stateCache.entries == 1 &&
              after.replayStatePublications == 2 &&
              after.recycledStatePublications == 0 &&
              after.resources.pool.pagesResident == 8,
          "KV growth did not take exactly the older cached state");
  require(after.resources.pool.pagesActive == 0 &&
              after.resources.activeRequests == 0,
          "grown request leaked its active resources");
}

void testRequiredWorkDoesNotReserveAnExtraPage() {
  for (uint32_t promptTokens : {1U, 24U}) {
    Backing backing(1);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(1);
    Events events;
    engine::Engine engine({}, resources, executor, events);
    engine.submit(request(231, std::vector<uint32_t>(promptTokens, 231)));
    runUntilIdle(engine);
    require(events.completedCount == 1 && events.emitted == 1 &&
                events.capacityExhaustedCount == 0 &&
                engine.snapshot().scheduler.decodeBatchesByWidth[0] == 1,
            "unused KV runway rejected a request whose full verify fits");
  }
}

void testAdmissionPinsDesiredStateAndCountsOnlySuccess() {
  Backing backing(16);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> desired(65, 240);
  engine.submit(request(240, desired));
  runUntilIdle(engine);
  engine.submit(request(241, std::vector<uint32_t>(65, 241)));
  runUntilIdle(engine);
  const auto before = engine.snapshot();
  require(before.resources.stateCache.entries == 2,
          "pinning test did not publish both resident states");
  uint32_t pinnedAttempts = 0;
  executor.beginObserver = [&] {
    if (resources.snapshot().stateCache.pinned == 1) ++pinnedAttempts;
  };
  executor.deniedBegins = 2;
  engine.submit(request(242, desired));
  runUntilIdle(engine);
  const auto after = engine.snapshot();
  require(pinnedAttempts == 3 && events.completedCount == 3 &&
              after.resources.stateCache.entries == 1 &&
              after.resources.stateCache.pinned == 0 &&
              events.starts.back().first == EngineCacheStatus::PrefixHit &&
              events.starts.back().second == 64 &&
              after.cacheHits == before.cacheHits + 1 &&
              after.resources.lookup.lookups == before.resources.lookup.lookups + 1,
          "successful retry failed to restore/count the protected cache state");
}

void testAdmissionCanDropItsOwnCachePinToMakeProgress() {
  Backing backing(8);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> prompt(65, 245);
  engine.submit(request(245, prompt));
  runUntilIdle(engine);
  require(resources.snapshot().stateCache.entries == 1,
          "self-pin test did not retain its initial state");
  // Under the new ceiling, one active request fits only after releasing the
  // sole cached state. Its lookup lease must not create a permanent deadlock.
  executor.beginGrowthBlocked = [&] {
    return resources.snapshot().stateCache.entries != 0;
  };
  engine.submit(request(246, prompt));
  runUntilIdle(engine);
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0 && executor.restored == 0 &&
              executor.prefillRows == 130 &&
              events.starts.back().first == EngineCacheStatus::Miss &&
              engine.snapshot().coldMisses == 2 && engine.snapshot().cacheHits == 0 &&
              resources.snapshot().lookup.lookups == 2,
          "admission waited on its own cache pin instead of recomputing cold");
}

void testSingletonCapacityFailureTerminatesCleanly() {
  Backing backing(1);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);

  engine.submit(request(40, std::vector<uint32_t>(33, 40)));
  runUntilIdle(engine);
  require(events.completedCount == 0 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 1 && engine.idle(),
          "B1 capacity failure did not emit exactly one terminal event");
  require(resources.snapshot().activeRequests == 0 && executor.requests.empty(),
          "B1 capacity failure leaked backend resources");
}

void testGrowthKeepsPrefillProgressWhenAnUnstartedPeerCanYield() {
  for (RequestPriority priority : {RequestPriority::Normal,
                                   RequestPriority::Background}) {
    Backing backing(512);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(2);
    Events events;
    engine::Engine engine({}, resources, executor, events);
    backing.growthAllowed = [&] {
      const auto started = executor.requests.find(48);
      return started == executor.requests.end() ||
             started->second.position < 2048 ||
             std::count_if(executor.requests.begin(), executor.requests.end(),
                           [](const auto &entry) { return entry.second.resident; }) < 2;
    };
    engine.submit(request(48, std::vector<uint32_t>(4096, 48)));
    auto peer = request(49, std::vector<uint32_t>(8192, 49));
    peer.priority = priority;
    engine.submit(std::move(peer));
    require(engine.tick(1) && engine.tick(2) &&
                executor.requests.at(48).position == 2048 &&
                executor.requests.at(49).position == 0,
            "growth fixture did not isolate a started and an unstarted prefill");
    require(engine.tick(3) && executor.suspensions == 1 &&
                executor.requests.at(48).resident &&
                !executor.requests.at(49).resident,
            "KV growth discarded completed prefill instead of its unstarted peer");
    runUntilIdle(engine);
    require(events.completedCount == 2 && events.failedCount == 0 &&
                events.capacityExhaustedCount == 0 && executor.prefillRows == 12288,
            "yielding an unstarted peer lost work or failed to resume");
  }
}

void testGrowthYieldsLowerPriorityResidentOutsideBatch() {
  Backing backing(512);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  backing.growthAllowed = [&] {
    return std::count_if(executor.requests.begin(), executor.requests.end(),
                         [](const auto &entry) { return entry.second.resident; }) < 2;
  };
  engine.submit(request(48, std::vector<uint32_t>(4096, 48)));
  require(engine.tick(1) && engine.tick(2) &&
              executor.requests.at(48).position == 2048,
          "priority fixture did not advance the initial request");
  auto foreground = request(49, std::vector<uint32_t>(8192, 49));
  foreground.priority = RequestPriority::Foreground;
  engine.submit(std::move(foreground));
  require(engine.tick(3) && executor.suspensions == 1 &&
              !executor.requests.at(48).resident &&
              executor.requests.at(49).resident,
          "resource recovery preempted foreground work for a lower-priority peer");
  runUntilIdle(engine);
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0,
          "priority preemption failed to finish both requests");
}

void testPrefillGrowthPreservesAnActiveDecodePeer() {
  Backing backing(512);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  engine.submit(request(48, std::vector<uint32_t>(8192, 48)));
  require(engine.tick(1) && engine.tick(2), "prefill setup did not progress");
  auto decoding = request(49, {49});
  decoding.maxNewTokens = 32;
  engine.submit(std::move(decoding));
  require(engine.tick(3) && engine.tick(4) &&
              executor.requests.at(49).position == 1,
          "short peer did not enter decode");
  backing.growthBlocked = true;
  for (uint32_t step = 5; step < 15 && !executor.suspensions; ++step)
    static_cast<void>(engine.tick(step));
  require(executor.suspensions == 1 && !executor.requests.at(48).resident &&
              executor.requests.at(49).resident,
          "prefill growth interrupted an equal-priority decode stream");
  backing.growthBlocked = false;
  executor.decodeFinishes = true;
  runUntilIdle(engine);
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0,
          "mixed-phase pressure failed to finish both requests");
}

void testPhysicalKvPressureSuspendsInsteadOfKillingActiveWork() {
  Backing backing(8);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  executor.physicalGrowthBlocked = &backing.growthBlocked;
  Events events;
  engine::Engine engine({}, resources, executor, events);

  backing.growthBlocked = true;
  engine.submit(request(50, {50}));
  engine.submit(request(51, {51}));
  runUntilIdle(engine);

  require(executor.suspensions == 1 && executor.resumptions == 1 &&
              engine.snapshot().resourceSuspensions == 1 &&
              engine.snapshot().resourceResumptions == 1,
          "physical KV pressure did not suspend and resume one lane");
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0,
          "recoverable physical pressure killed an active request");
}

void testPhysicalPressureRetryIsBackedOffWithoutProgress() {
  Backing backing(8);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  executor.physicalGrowthBlocked = &backing.growthBlocked;
  executor.unblockGrowthOnSuspend = false;
  Events events;
  MemoryPressure pressure = MemoryPressure::Warning;
  EngineConfig config;
  config.growthPaused = [&] { return pressure != MemoryPressure::Normal; };
  engine::Engine engine(config, resources, executor, events);

  backing.growthBlocked = true;
  engine.submit(request(52, {52}));
  engine.submit(request(53, {53}));
  require(engine.tick(1.0) && engine.tick(2.0),
          "physical pressure did not suspend both blocked lanes");
  const uint32_t attemptsBeforeBackoff = executor.resumeAttempts;
  require(executor.suspensions == 2 && attemptsBeforeBackoff == 0,
          "suspended lanes were retried before a resource wake-up");
  require(!engine.tick(50.0) &&
              executor.resumeAttempts == attemptsBeforeBackoff,
          "resource wait retried on an unrelated scheduler tick");

  backing.growthBlocked = false;
  pressure = MemoryPressure::Normal;
  require(engine.tick(102.0), "resource retry timer did not wake the engine");
  for (double now = 103.0; now < 120.0 && !engine.idle(); ++now)
    static_cast<void>(engine.tick(now));
  require(engine.idle() && executor.resumptions == 2 &&
              events.completedCount == 2 && events.failedCount == 0,
          "backed-off resource requests did not recover cleanly");
}

void testRecoveryDrainDoesNotConsumeResourceWaitBudget() {
  for (bool expireRequest : {false, true}) {
    Backing backing(2);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(2);
    executor.decodeFinishes = false;
    Events events;
    engine::Engine engine({}, resources, executor, events);
    for (uint64_t id : {250, 251}) {
      auto value = request(id, std::vector<uint32_t>(24, id));
      value.maxNewTokens = 4;
      value.deadlineMilliseconds = expireRequest ? 30000 : 1000000;
      engine.submit(std::move(value));
    }
    for (double now = 1; now <= 5; ++now)
      require(engine.tick(now), "drain fixture made no progress");
    require(executor.suspensions == 1, "drain fixture did not preempt a lane");
    executor.holdDecodeUntil = std::make_shared<bool>(false);
    require(engine.tick(6) && engine.commandInFlight(),
            "resident peer did not start its command");
    static_cast<void>(engine.tick(30006));
    require(executor.resumeAttempts == 0,
            "recovery retried while a resident peer was still running");
    if (expireRequest) {
      require(!events.failures.empty() &&
                  std::all_of(events.failures.begin(), events.failures.end(),
                              [](const auto &code) { return code == "deadline_exceeded"; }),
              "draining suppressed the request deadline");
    } else {
      require(events.failedCount == 0 && engine.nextWakeupMilliseconds() > 30006,
              "deliberate drain consumed the resource wait limit");
      const auto wait = engine.resourceWaitSnapshot(30006);
      require(wait.draining && wait.suspended == 1 && wait.memory == 1 &&
                  wait.oldestWaitMilliseconds >= 30000,
              "draining hid the elapsed resource wait from diagnostics");
    }
    *executor.holdDecodeUntil = true;
    for (double now = 30007; now < 30100 && !engine.idle(); ++now)
      static_cast<void>(engine.tick(now));
    require(engine.idle() && resources.snapshot().activeRequests == 0,
            "drain fixture did not release its resources");
    if (!expireRequest)
      require(events.completedCount == 2 && events.failedCount == 0 &&
                  executor.resumptions == 1,
              "suspended work did not resume after its peer completed");
  }
}

void testDecodePreemptionReplaysCommittedHistoryWithoutRepeatingOutput() {
  Backing backing(2);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  executor.decodeFinishes = false;
  // A replay result may carry a previously prepared consumer stage. It is
  // metadata, not a request to ask the frontend for that initial mask again.
  executor.replayDecodeStage = DecodeStage::ApplyInitialMask;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  for (uint64_t id : {250, 251}) {
    auto value = request(id, std::vector<uint32_t>(24, id));
    value.maxNewTokens = 4;
    value.returnProgress = true;
    engine.submit(std::move(value));
  }
  for (double now = 1; now <= 4; ++now)
    require(engine.tick(now), "decode replay setup made no progress");
  require(events.emitted == 2 && engine.tick(5) && executor.suspensions == 1,
          "full active KV did not preempt a lane at the verify boundary");
  uint64_t preempted = 0;
  for (const auto &[id, state] : executor.requests) {
    if (!state.resident) preempted = id;
  }
  require(preempted && resources.snapshot().activeRequests == 1 &&
              resources.snapshot().pool.pagesActive == 1,
          "preempted decode retained its active KV ownership");

  // New admission and the elapsed retry timer must not disturb the resident
  // lane while it uses the released headroom to finish.
  executor.holdDecodeUntil = std::make_shared<bool>(false);
  engine.submit(request(252, {252}));
  require(engine.tick(6) && engine.commandInFlight(),
          "surviving lane could not launch after preemption");
  require(!engine.tick(150) && engine.nextWakeupMilliseconds() == 1150.0 &&
              events.startIds.size() == 2 && executor.resumeAttempts == 0,
          "draining recovery busy-woke or admitted a competing request");
  *executor.holdDecodeUntil = true;
  for (double now = 151; now < 210 && !engine.idle(); ++now)
    static_cast<void>(engine.tick(now));
  require(engine.idle() && executor.resumptions == 2 &&
              events.completedCount == 3 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0,
          "decode preemption did not finish all original/new requests");
  require(events.progress.at(250) == std::vector<uint32_t>({0, 24}) &&
              events.progress.at(251) == std::vector<uint32_t>({0, 24}),
          "decode recovery reported generated history as prompt progress");
  std::vector<uint32_t> committedHistory(24, preempted);
  committedHistory.push_back(42);
  // The newly admitted request also yields once because the recovering
  // continuation has reserved both pages. Neither replays generated output.
  require(executor.resumedPrompts ==
              std::vector<std::vector<uint32_t>>{committedHistory, {252}} &&
              engine.snapshot().resourceReplayTokens == committedHistory.size() + 1,
          "replay omitted or duplicated committed generated history");
  require(events.outputs.at(250) == std::vector<uint32_t>(4, 42) &&
              events.outputs.at(251) == std::vector<uint32_t>(4, 42) &&
              events.usage.at(250) == std::pair<uint32_t, uint32_t>{24, 4} &&
              events.usage.at(251) == std::pair<uint32_t, uint32_t>{24, 4} &&
              events.startIds.size() == 3 &&
              events.maskRequests.empty() &&
              engine.snapshot().resources.lookup.lookups == 3 &&
              engine.snapshot().coldMisses == 3,
          "resource replay emitted prior output or changed request usage/hits");
  require(executor.requests.empty() && executor.snapshotAttempts == 0 &&
              resources.snapshot().pool.pagesActive == 0 &&
              resources.snapshot().activeRequests == 0,
          "decode replay leaked active resources");
}

void testLongDecodePreemptionPlansTheCurrentReplayBoundary() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  executor.decodeFinishes = false;
  executor.deniedSnapshots = std::numeric_limits<uint32_t>::max();
  Events events;
  MemoryPressure pressure = MemoryPressure::Normal;
  EngineConfig config;
  // Keep the old 4096-token replay boundary distinct from checkpoints.
  config.prefillCheckpointTokens = 8192;
  config.growthPaused = [&] { return pressure != MemoryPressure::Normal; };
  engine::Engine engine(config, resources, executor, events);

  constexpr uint64_t id = 255;
  const std::vector<uint32_t> prompt(4097, 5);
  auto value = request(id, prompt);
  value.maxNewTokens = 10'000;
  value.deadlineMilliseconds = 100'000;
  engine.submit(std::move(value));
  double now = 1;
  for (; now < 20'000 && executor.suspensions == 0; ++now) {
    if (events.emitted >= 4200) {
      backing.growthBlocked = true;
      pressure = MemoryPressure::Warning;
    }
    static_cast<void>(engine.tick(now));
  }
  require(executor.suspensions == 1 && !engine.commandInFlight(),
          "long decode did not suspend at KV growth pressure");
  const std::vector<uint32_t> emitted = events.outputs.at(id);
  std::vector<uint32_t> history = prompt;
  history.insert(history.end(), emitted.begin(), emitted.end());
  require(resources.snapshot().stateCache.entries == 0,
          "snapshot denial left a composite state to restore");

  // Retain a KV junction one draft window past the original prompt boundary.
  // The old prompt boundary, this junction, and the generated history's end
  // would require three capture spans if replay still used the old boundary.
  constexpr uint32_t retainedKvTokens = 6144;
  while (resources.snapshot().kvCache.blocks >
         retainedKvTokens / KvCache::pageTokens) {
    require(resources.reclaimOne(CacheReclaimMode::ReuseBacking).madeProgress,
            "could not reclaim the suspended decode's KV tail");
  }
  {
    auto lookup = resources.lookup(history);
    require(lookup.kvBoundary == retainedKvTokens &&
                lookup.resumeBoundary() == 0,
            "long replay did not retain the intended stateless KV junction");
  }

  backing.growthBlocked = false;
  pressure = MemoryPressure::Normal;
  executor.deniedSnapshots = 0;
  executor.decodeFinishes = true;
  now += 101;
  require(engine.tick(now++), "long replay did not resume after pressure eased");
  const auto &plan = executor.plans.at(id);
  require(plan.replayEnd == history.size() && plan.captureSpans.size() == 2,
          "resumed draft plan did not cover generated history in two spans");
  const double finishBy = now + 100;
  for (; now < finishBy && !engine.idle(); ++now)
    static_cast<void>(engine.tick(now));
  require(engine.idle() && executor.resumptions == 1 &&
              events.completedCount == 1 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0,
          "long decode did not complete after partial KV reclamation");
  require(executor.resumedPrompts ==
              std::vector<std::vector<uint32_t>>{history} &&
              engine.snapshot().resourceReplayTokens == history.size(),
          "long replay omitted or duplicated committed history");
  auto expectedOutput = emitted;
  expectedOutput.push_back(42);
  require(events.outputs.at(id) == expectedOutput &&
              events.usage.at(id) == std::pair<uint32_t, uint32_t>{
                                         prompt.size(), expectedOutput.size()} &&
              events.startIds == std::vector<uint64_t>{id},
          "long replay repeated output or changed original request usage");
  require(executor.requests.empty() &&
              resources.snapshot().pool.pagesActive == 0 &&
              resources.snapshot().activeRequests == 0,
          "long replay leaked active resources");
}

void testPreemptedDecodeRestoresItsResidentCompositeState() {
  Backing backing(6);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  for (uint64_t id : {260, 261}) {
    auto value = request(id, std::vector<uint32_t>(65, id));
    value.maxNewTokens = 30;
    engine.submit(std::move(value));
  }
  for (double now = 1; now < 300 && !engine.idle(); ++now)
    static_cast<void>(engine.tick(now));
  require(engine.idle() && executor.suspensions == 1 &&
              executor.resumptions == 1 && executor.restored == 64 &&
              executor.resumedPrompts.size() == 1 &&
              executor.resumedPrompts.front().size() == 89 &&
              engine.snapshot().resourceReplayTokens == 25,
          "preempted decode did not use its cached state plus exact suffix");
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0 && events.startIds.size() == 2 &&
              events.outputs.at(260) == std::vector<uint32_t>(30, 42) &&
              events.outputs.at(261) == std::vector<uint32_t>(30, 42) &&
              events.usage.at(260) == std::pair<uint32_t, uint32_t>{65, 30} &&
              engine.snapshot().resources.lookup.lookups == 2 &&
              engine.snapshot().cacheHits == 0 &&
              engine.snapshot().coldMisses == 2,
          "internal cache restore changed output or request accounting");
}

void testRepeatedPreemptionRespectsBackoffAndCancellation() {
  for (bool cancel : {false, true}) {
    Backing backing(4);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(1);
    executor.physicalGrowthBlocked = &backing.growthBlocked;
    executor.unblockGrowthOnSuspend = false;
    Events events;
    EngineConfig config;
    config.growthPaused = [] { return true; };
    engine::Engine engine(config, resources, executor, events);
    backing.growthBlocked = true;
    auto value = request(270, {270});
    value.deadlineMilliseconds = 350;
    engine.submit(std::move(value));
    require(engine.tick(1) && executor.suspensions == 1,
            "singleton host pressure did not preempt safely");
    require(!engine.tick(50) && !engine.tick(101) &&
                executor.resumeAttempts == 1 &&
                !engine.tick(150) && !engine.tick(201) &&
                executor.resumeAttempts == 2 &&
                engine.snapshot().resourceSuspensions == 1 &&
                engine.snapshot().resourceResumptions == 0 &&
                executor.prefillRows == 0,
            "failed recovery replayed work or ignored retry backoff");
    require(engine.nextWakeupMilliseconds() == 301 &&
                events.startIds.size() == 1 &&
                engine.snapshot().resources.lookup.lookups == 1 &&
                resources.snapshot().pool.pagesActive == 0 &&
                resources.snapshot().activeRequests == 0,
            "repeated preemption retained KV or duplicated admission counters");
    if (cancel) {
      engine.cancel(270);
      static_cast<void>(engine.tick(202));
    } else {
      static_cast<void>(engine.tick(350));
    }
    require(engine.idle() && executor.requests.empty() &&
                events.completedCount == (cancel ? 1U : 0U) &&
                events.failedCount == (cancel ? 0U : 1U) &&
                events.capacityExhaustedCount == 0 && events.emitted == 0,
            "repeated preemption ignored its terminal deadline/cancel");
  }
}

void testStateAdmissionWaitsForKvRelease() {
  enum class Outcome { Recover, CompleteOnPoll, NoCapacity, Cancel, Timeout };
  for (uint32_t pageCount : {4U, 8U}) {
    for (auto outcome : {Outcome::Recover, Outcome::CompleteOnPoll,
                         Outcome::NoCapacity, Outcome::Cancel, Outcome::Timeout}) {
      if (pageCount != 4 && outcome == Outcome::CompleteOnPoll)
        continue;
      Backing backing(pageCount);
      KvPool pool(backing);
      engine::Cache cache(pool, CacheNamespace{});
      auto pages = pool.acquirePages(pageCount, false);
      require(pages.granted(), "could not seed resident KV backing");
      for (uint32_t page : pages.pages)
        pool.releasePage(page, false);
      backing.deferRelease = true;
      backing.completeReleaseOnPoll = outcome == Outcome::CompleteOnPoll;
      Executor executor(1);
      executor.beginGrowthBlocked = [&] {
        return pool.snapshot().pagesResident != 0 || backing.releasePending ||
               outcome == Outcome::NoCapacity;
      };
      executor.beginAllocationFailure = metal::AllocationFailure::EngineBudget;
      Events events;
      EngineConfig config;
      config.resourceWaitTimeoutMilliseconds = 500;
      engine::Engine engine(config, cache, executor, events);
      engine.submit(request(284, {284}));
      static_cast<void>(engine.tick(1));
      require(!engine.idle() && events.failedCount == 0 &&
                  executor.prefillRows == 0,
              "in-flight KV release became terminal state admission failure");
      const uint32_t attempts = executor.beginAttempts;
      static_cast<void>(engine.tick(50));
      require(executor.beginAttempts == attempts,
              "pending release bypassed resource retry backoff");
      if (outcome == Outcome::Recover || outcome == Outcome::CompleteOnPoll ||
          outcome == Outcome::NoCapacity) {
        backing.releasePending = false;
        backing.deferRelease = false;
        for (double now = 101; now < 400 && !engine.idle(); ++now)
          static_cast<void>(engine.tick(now));
        if (outcome == Outcome::NoCapacity)
          require(events.failures == std::vector<std::string>{"capacity_exhausted"},
                  "true capacity exhaustion kept waiting after reclamation");
        else
          require(events.completedCount == 1 && events.failedCount == 0,
                  "state admission did not resume after KV release");
      } else if (outcome == Outcome::Cancel) {
        engine.cancel(284);
        static_cast<void>(engine.tick(51));
      } else {
        static_cast<void>(engine.tick(502));
        require(events.failures == std::vector<std::string>{"resource_timeout"},
                "pending release bypassed resource wait deadline");
      }
      require(engine.idle() && cache.snapshot().activeRequests == 0 &&
                  executor.requests.empty(),
              "pending-release state admission leaked ownership");
    }
  }
}

void testAdmissionsWaitForBackgroundRelease() {
  enum class Outcome { Recover, CompleteOnPoll, NoCapacity, Cancel, Timeout };
  enum class Path { State, RunningKv, ResumeKv };
  for (auto path : {Path::State, Path::RunningKv, Path::ResumeKv}) {
    const bool stateAdmission = path == Path::State;
    const bool resuming = path == Path::ResumeKv;
    for (auto outcome : {Outcome::Recover, Outcome::CompleteOnPoll,
                         Outcome::NoCapacity, Outcome::Cancel, Outcome::Timeout}) {
      Backing backing(8);
      KvPool pool(backing);
      engine::Cache cache(pool, CacheNamespace{});
      Executor executor(1);
      Events events;
      EngineConfig config;
      config.resourceWaitTimeoutMilliseconds = 500;
      engine::Engine engine(config, cache, executor, events);
      engine.submit(request(285, {285}));
      if (resuming) {
        backing.growthBlocked = true;
        backing.allocationFailure = metal::AllocationFailure::HostPressure;
        static_cast<void>(engine.tick(1));
        require(executor.suspensions == 1, "fixture did not suspend request");
        backing.growthBlocked = false;
      }
      auto pages = pool.acquirePages(4, false);
      require(pages.granted(), "could not seed background KV release");
      for (uint32_t page : pages.pages)
        pool.releasePage(page, false);
      backing.deferRelease = true;
      require(pool.reclaimEmptyExtents(false) == 1,
              "fixture did not start background release");
      backing.completeReleaseOnPoll = outcome == Outcome::CompleteOnPoll;
      backing.allocationFailure = metal::AllocationFailure::EngineBudget;
      backing.growthAllowed = [&] {
        return !backing.releasePending && outcome != Outcome::NoCapacity;
      };
      if (stateAdmission) {
        executor.beginAllocationFailure = metal::AllocationFailure::EngineBudget;
        executor.beginGrowthBlocked = [&] { return !backing.growthAllowed(); };
      }
      const double start = resuming ? 101 : 1;
      static_cast<void>(engine.tick(start));
      require(events.failedCount == 0 && events.capacityExhaustedCount == 0,
              "pending/just-completed release became terminal KV failure");
      if (outcome != Outcome::CompleteOnPoll) {
        const auto attempts = backing.growthAttempts + executor.beginAttempts;
        static_cast<void>(engine.tick(start + 50));
        require(backing.growthAttempts + executor.beginAttempts == attempts,
                "KV release wait bypassed retry backoff");
      }
      if (outcome == Outcome::Cancel) {
        engine.cancel(285);
        static_cast<void>(engine.tick(start + 51));
        require(events.completedCount == 1, "KV release wait ignored cancel");
      } else if (outcome == Outcome::Timeout) {
        static_cast<void>(engine.tick(start + 501));
        require(events.failures == std::vector<std::string>{"resource_timeout"},
                "KV release wait ignored timeout");
      } else {
        backing.releasePending = false;
        for (double now = start + 100; now < start + 400 && !engine.idle(); ++now)
          static_cast<void>(engine.tick(now));
        if (outcome == Outcome::NoCapacity)
          require(events.failures == std::vector<std::string>{"capacity_exhausted"},
                  "true KV exhaustion kept waiting after release");
        else
          require(events.completedCount == 1 && events.failedCount == 0,
                  "KV allocation did not recover after release");
      }
      require(engine.idle() && executor.requests.empty() &&
                  cache.snapshot().activeRequests == 0 &&
                  cache.snapshot().pool.pagesActive == 0,
              "KV release wait leaked request ownership");
    }
  }
}

void testAllocationCausesRemainRetryableAndDistinct() {
  for (bool stateAllocation : {false, true}) {
    for (auto reason : {metal::AllocationFailure::HostPressure,
                        metal::AllocationFailure::EngineBudget,
                        metal::AllocationFailure::DriverRejected}) {
      Backing backing(8);
      KvPool pool(backing);
      engine::Cache cache(pool, CacheNamespace{});
      Executor executor(1);
      Events events;
      engine::Engine engine({}, cache, executor, events);
      backing.growthBlocked = !stateAllocation;
      backing.allocationFailure = reason;
      executor.beginGrowthBlocked = [stateAllocation] { return stateAllocation; };
      executor.beginAllocationFailure = reason;
      engine.submit(request(285, {285}));
      static_cast<void>(engine.tick(1));
      if (reason == metal::AllocationFailure::HostPressure) {
        require(!engine.idle() && events.failedCount == 0 &&
                    executor.prefillRows == 0,
                "temporary host admission failure became a terminal error");
        engine.cancel(285);
        static_cast<void>(engine.tick(2));
      } else {
        require(engine.idle() && events.failures ==
                    std::vector<std::string>{"capacity_exhausted"} &&
                    events.failureDetails.size() == 1 &&
                    events.failureDetails[0].second &&
                    events.failureDetails[0].first.find(
                        metal::allocationFailureName(reason)) != std::string::npos,
                "allocation failure lost its cause or became nonretryable");
      }
      require(engine.idle() && executor.requests.empty() &&
                  cache.snapshot().activeRequests == 0,
              "allocation denial leaked request ownership");
    }
  }
}

void testRecoveryAdmitsFailedKvTargetBeforeReplaying() {
  for (bool sharePrefix : {false, true}) {
    Backing backing(64);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(2);
    executor.unblockGrowthOnSuspend = false;
    executor.physicalGrowthBlocked = &backing.growthBlocked;
    Events events;
    EngineConfig config;
    // Models the request-sized headroom denial while global pressure is Normal.
    config.growthPaused = [] { return false; };
    engine::Engine engine(config, resources, executor, events);
    const std::vector<uint32_t> prompt(129, 280);
    if (sharePrefix) {
      engine.submit(request(279, std::vector<uint32_t>(65, 280)));
      runUntilIdle(engine);
    }
    backing.allocationFailure = metal::AllocationFailure::HostPressure;
    backing.growthAllowed = [&] { return !backing.growthBlocked; };
    auto input = request(280, prompt);
    input.returnProgress = true;
    engine.submit(std::move(input));
    // First dispatch reaches the page-aligned replay state (128 tokens).
    require(engine.tick(1) && engine.tick(2), "recovery fixture did not prefill");
    const auto cached = resources.snapshot();
    const uint32_t restored = executor.restored;
    {
      auto lookup = resources.lookup(prompt);
      require(lookup.resumeBoundary() == 128 &&
                  cached.pool.pagesResident == cached.pool.pagesActive,
              "recovery fixture needs a state backed only by active prefix pages");
    }
    backing.growthBlocked = true;
    require(engine.tick(3) && engine.snapshot().resourceSuspensions == 1,
            "request-sized host denial was treated as permanent capacity");
    require(resources.snapshot().stateCache.entries == cached.stateCache.entries &&
                resources.snapshot().stateCache.evictions == cached.stateCache.evictions,
            "active prefix pages were mistaken for idle backing and lost their state");
    const uint64_t rows = executor.prefillRows;
    const uint64_t replay = engine.snapshot().resourceReplayTokens;
    for (double now : {103.0, 203.0, 303.0}) {
      require(!engine.tick(now) && !engine.commandInFlight() &&
                  executor.prefillRows == rows && executor.restored == restored &&
                  engine.snapshot().resourceReplayTokens == replay &&
                  engine.snapshot().resourceResumptions == 0 &&
                  resources.snapshot().activeRequests == 0 &&
                  resources.snapshot().pool.pagesActive == 0,
              "insufficient KV replayed committed rows or retained active leases");
    }
    const auto reported = events.progress.at(280);
    require(reported.back() == 128, "prefill recovery lost completed progress");
    backing.growthBlocked = false;
    require(engine.tick(403), "recovery did not resume when KV became available");
    for (double now = 404; now < 425 && !engine.idle(); ++now)
      static_cast<void>(engine.tick(now));
    auto expected = reported;
    expected.push_back(129);
    require(events.progress.at(280) == expected,
            "prefill recovery duplicated or regressed progress");
    require(engine.idle() && engine.snapshot().resourceResumptions == 1 &&
                executor.restored == restored + 128 &&
                engine.snapshot().resourceReplayTokens == 1 &&
                events.outputs.at(280) == std::vector<uint32_t>{42} &&
                resources.snapshot().activeRequests == 0 &&
                resources.snapshot().pool.pagesActive == 0 &&
                events.failedCount == 0 && events.capacityExhaustedCount == 0,
            "KV recovery failed to continue cleanly after headroom recovered");
  }
}

void testAdmissionReopensAfterLastSuspendedRequestResumes() {
  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  MemoryPressure pressure = MemoryPressure::Warning;
  EngineConfig config;
  config.growthPaused = [&] { return pressure != MemoryPressure::Normal; };
  engine::Engine engine(config, resources, executor, events);

  backing.growthBlocked = true;
  auto longRequest = request(271, {271});
  longRequest.maxNewTokens = 1000;
  engine.submit(std::move(longRequest));
  require(engine.tick(1) && executor.suspensions == 1,
          "recovery admission fixture did not suspend its only lane");
  pressure = MemoryPressure::Normal;
  backing.growthBlocked = false;
  require(engine.tick(101) && engine.tick(102) && executor.resumptions == 1,
          "recovery admission fixture did not resume its last waiting lane");

  engine.submit(request(272, {272}));
  for (double now = 103; now < 115 && events.startIds.size() < 2; ++now)
    static_cast<void>(engine.tick(now));
  require(events.startIds == std::vector<uint64_t>({271, 272}) &&
              executor.requests.contains(271) &&
              engine.snapshot().resourceSuspensions == 1,
          "finished resource recovery blocked new work until decode completed");
  engine.cancel(271);
  engine.cancel(272);
  for (double now = 115; now < 125 && !engine.idle(); ++now)
    static_cast<void>(engine.tick(now));
  require(engine.idle() && resources.snapshot().activeRequests == 0,
          "recovery admission fixture did not release its lanes");
}

void testAdmissionRespectsPriorityBeforeHashOrder() {
  Backing backing(8);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);

  EngineRequest background = request(100, {1});
  background.priority = RequestPriority::Background;
  EngineRequest foreground = request(1, {2});
  foreground.priority = RequestPriority::Foreground;
  engine.submit(std::move(background));
  engine.submit(std::move(foreground));
  require(engine.tick(2), "priority admission made no progress");
  require(!events.startIds.empty() && events.startIds.front() == 1,
          "unordered request storage admitted background work first");
  runUntilIdle(engine);
}

void advanceToOverlappedVerify(engine::Engine &engine, Executor &executor,
                               Events &events, uint64_t requestId) {
  require(engine.tick(1) && engine.tick(2) && engine.tick(3) && engine.tick(4),
          "constrained request did not reach its initial mask");
  require(events.maskRequests.size() == 1 &&
              events.maskRequests[0].first == requestId &&
              events.maskRequests[0].second.empty(),
          "initial constrained mask request is malformed");
  const std::array<uint32_t, 1> initialMask{1};
  engine.provideMask(requestId, initialMask);
  require(engine.tick(5) && engine.tick(6),
          "constrained request did not launch overlapped verification");
  require(engine.commandInFlight() && executor.overlap &&
              events.maskRequests.size() == 2 &&
              events.maskRequests[1].first == requestId &&
              events.maskRequests[1].second.size() == 8,
          "target forward did not retain its batch while requesting a mask");
}

EngineRequest constrainedRequest(uint64_t id, double deadline = 10'000.0) {
  EngineRequest value = request(id, {1});
  value.maxNewTokens = 2;
  value.cohort = BatchCohort::Constrained;
  value.constraint = ConstraintMode::TokenMask;
  value.deadlineMilliseconds = deadline;
  return value;
}

void testConstraintMaskOverlapsInsideOneSchedulerBatch() {
  Backing backing(8);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);

  engine.submit(constrainedRequest(200));
  advanceToOverlappedVerify(engine, executor, events, 200);
  const std::array<uint32_t, 1> verifyMask{1};
  engine.provideMask(200, verifyMask);
  require(engine.tick(7) && engine.idle() && events.completedCount == 1 &&
              events.failedCount == 0 && events.emitted == 1,
          "overlapped verify mask did not complete the owning batch");
}

void testConstraintMaskWaitHonorsCancelAndDeadline() {
  {
    Backing backing(8);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(1);
    Events events;
    engine::Engine engine({}, resources, executor, events);
    engine.submit(constrainedRequest(201));
    advanceToOverlappedVerify(engine, executor, events, 201);
    engine.cancel(201);
    const std::array<uint32_t, 1> lateMask{1};
    engine.provideMask(201, lateMask);
    engine.failRequest(201, "invalid_mask_response", "late mask");
    require(executor.overlap->abandoned && engine.tick(7) && engine.idle() &&
                events.completedCount == 1 && events.failedCount == 0 &&
                events.emitted == 0,
            "cancelled mask wait left an active scheduler batch");
  }

  {
    Backing backing(8);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(1);
    Events events;
    engine::Engine engine({}, resources, executor, events);
    engine.submit(constrainedRequest(202, 100.0));
    advanceToOverlappedVerify(engine, executor, events, 202);
    require(engine.tick(100.0) && executor.overlap->abandoned &&
                engine.idle() && events.failedCount == 1,
            "deadline did not terminate an in-flight host mask wait");
  }
}

void testDecodeNearContextCeilingCoversVerifyRows() {
  Backing backing(8);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  executor.decodeFinishes = false;
  Events events;
  EngineConfig config;
  config.maxContext = 64;
  engine::Engine engine(config, resources, executor, events);

  // prompt + max_tokens fill a context that is a whole number of pages; the
  // last decode cycles store verify rows past the logical ceiling.
  EngineRequest value = request(10, std::vector<uint32_t>(57, 10));
  value.maxNewTokens = 7;
  engine.submit(value);
  runUntilIdle(engine);
  require(events.completedCount == 1 && events.failedCount == 0 &&
              events.emitted == 7,
          "decode at the context ceiling lost its verify-row page coverage");
}

void testExpiredMaskWaitFinalizesWhileAnotherCommandRuns() {
  Backing backing(8);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  Events events;
  engine::Engine engine({}, resources, executor, events);

  engine.submit(constrainedRequest(300, 100.0));
  require(engine.tick(1) && engine.tick(2) && engine.tick(3) && engine.tick(4),
          "constrained request did not reach its initial mask wait");
  require(events.maskRequests.size() == 1, "initial mask was not requested");

  executor.holdDecodeUntil = std::make_shared<bool>(false);
  engine.submit(request(301, {2}));
  require(engine.tick(5) && engine.tick(6) && engine.tick(7) &&
              engine.commandInFlight(),
          "peer decode command was not held in flight");

  // The waiting request expires while the peer command is still running. It
  // must be finalized now; its late mask then finds no live request.
  require(engine.tick(150.0) && events.failedCount == 1 &&
              engine.commandInFlight(),
          "expired mask wait was not finalized behind an in-flight command");
  *executor.holdDecodeUntil = true;
  for (double now = 151.0; now < 160.0 && !engine.idle(); ++now)
    static_cast<void>(engine.tick(now));
  require(engine.idle() && events.completedCount == 1 &&
              events.failedCount == 1,
          "peer request did not complete after the expired wait was removed");
}

void testOrdinaryInFlightDeadlineDrainsWithoutPublishingOrOutput() {
  for (WorkKind heldKind : {WorkKind::Prefill, WorkKind::Decode}) {
    Backing backing(8);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(1);
    Events events;
    engine::Engine engine({}, resources, executor, events);
    auto released = std::make_shared<bool>(false);
    if (heldKind == WorkKind::Prefill)
      executor.holdPrefillUntil = released;
    else
      executor.holdDecodeUntil = released;
    auto value = request(310, std::vector<uint32_t>(65, 310));
    value.deadlineMilliseconds = 100;
    engine.submit(std::move(value));
    const uint32_t launchTick = heldKind == WorkKind::Prefill ? 1 : 5;
    for (uint32_t tick = 1; tick <= launchTick; ++tick)
      require(engine.tick(tick), "request did not launch the held command");
    require(engine.commandInFlight(), "test command did not remain in flight");
    const auto before = resources.snapshot();
    require(engine.tick(100) && engine.commandInFlight() &&
                executor.requests.size() == 1 && events.failedCount == 0 &&
                resources.snapshot().pool.pagesActive == before.pool.pagesActive,
            "ordinary deadline released resources owned by in-flight Metal");
    require(engine.nextWakeupMilliseconds() == 1100.0 && !engine.tick(101),
            "expired in-flight request lost its bounded health wakeup");
    engine.cancel(310); // A later cancel must not replace the deadline failure.
    *released = true;
    // A held prefill expires at its armed boundary and never snapshots; a
    // held decode follows the prefill that already published at 64.
    const uint32_t expectedSnapshots = heldKind == WorkKind::Prefill ? 0 : 1;
    require(engine.tick(102) && engine.idle() && events.failedCount == 1 &&
                events.failures.front() == "deadline_exceeded" &&
                events.completedCount == 0 && events.emitted == 0 &&
                executor.snapshotAttempts == expectedSnapshots &&
                executor.requests.empty(),
            "expired ordinary command emitted output or lost its deadline");
    const auto after = resources.snapshot();
    require(after.stateCache.publications == before.stateCache.publications &&
                after.kvCache.blocks == before.kvCache.blocks &&
                after.pool.pagesActive == 0 && after.activeRequests == 0,
            "expired ordinary command published cache state or leaked resources");
  }
}

void testStalledSuspensionFailsWithCapacity() {
  Backing backing(8);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  executor.physicalGrowthBlocked = &backing.growthBlocked;
  executor.unblockGrowthOnSuspend = false;
  Events events;
  engine::Engine engine({}, resources, executor, events);

  backing.growthBlocked = true;
  engine.submit(request(60, {60}));
  engine.submit(request(61, {61}));
  require(engine.tick(1.0) && engine.tick(2.0) && executor.suspensions == 1 &&
              events.capacityExhaustedCount == 1,
          "preemption did not identify that the remaining singleton cannot fit");

  // Nothing completes between suspension and retry, so suspending again would
  // only rotate the same two lanes forever.
  for (double now = 102.0; now < 140.0 && !engine.idle(); ++now)
    static_cast<void>(engine.tick(now));
  require(engine.idle() && events.capacityExhaustedCount == 2 &&
              engine.snapshot().resourceSuspensions == 1 &&
              engine.snapshot().resourceResumptions == 0 && executor.prefillRows == 0,
          "stalled suspension did not converge to a capacity failure");
}

void testTerminalAnchorWithoutKvIsNotCached() {
  for (uint32_t withoutKv : {1U, 0U}) {
    Backing backing(8);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(1);
    executor.decodeTokensWithoutKv = withoutKv;
    Events events;
    engine::Engine engine({}, resources, executor, events);

    // 31 prompt tokens plus the emitted token complete one Page32 block only
    // when that token has a stored KV row.
    engine.submit(request(70, std::vector<uint32_t>(31, 70)));
    runUntilIdle(engine);
    require(events.completedCount == 1 && events.emitted == 1,
            "terminal anchor request did not complete");
    require(resources.snapshot().kvCache.blocks == (withoutKv ? 0U : 1U),
            "token without a KV row changed the cached block count");
  }
}

void testPrefillCanCompleteTheRequest() {
  for (bool stop : {false, true}) {
    Backing backing(8);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(1);
    executor.prefillAnchor = stop;
    Events events;
    engine::Engine engine({}, resources, executor, events);

    EngineRequest value = request(90, std::vector<uint32_t>(40, 90));
    value.maxNewTokens = stop ? 8 : 1;
    engine.submit(std::move(value));
    runUntilIdle(engine);
    require(events.completedCount == 1 && events.emitted == 1 &&
                events.failedCount == 0,
            "request finished by prefill did not complete cleanly");
    require(engine.snapshot().scheduler.decodeBatchesByWidth[0] == 0,
            "request finished by prefill was decoded");
  }
}

const uint32_t defaultCheckpointTokens = EngineConfig{}.prefillCheckpointTokens;

void runUntilCheckpoint(engine::Engine &engine, uint64_t publications) {
  for (uint32_t step = 0; step < 128; ++step) {
    static_cast<void>(engine.tick(step + 1));
    if (engine.snapshot().checkpointPublications >= publications)
      return;
  }
  throw std::runtime_error("engine did not publish the expected checkpoint");
}

void testCancelledColdPrefillResumesItsLatestCheckpoint() {
  Backing backing(2048);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> prompt(30001, 19);
  engine.submit(request(400, prompt));
  runUntilCheckpoint(engine, 2);
  require(resources.snapshot().stateCache.entries == 1 &&
              resources.snapshot().stateCache.bytes == 64,
          "cold prefill accumulated superseded progress states");
  engine.cancel(400);
  runUntilIdle(engine);
  const uint32_t computed = executor.prefillRows;
  engine.submit(request(401, prompt));
  runUntilIdle(engine);
  const uint32_t restored = 2 * defaultCheckpointTokens;
  require(events.starts.back().second == restored &&
              executor.prefillRows - computed == prompt.size() - restored &&
              events.failedCount == 0,
          "cancelled cold prefill was recomputed before its completed checkpoint");
  require(resources.snapshot().stateCache.entries == 1 &&
              resources.snapshot().stateCache.checkpointEntries == 0,
          "successful retry retained a temporary recovery point");
}

void testConcurrentProgressRetainsAtMostOnePointPerLane() {
  Backing backing(2048);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> prompt(25001, 20);
  engine.submit(request(410, prompt));
  engine.submit(request(411, prompt));
  uint32_t maximumEntries = 0;
  for (uint32_t step = 0; step < 128 && !engine.idle(); ++step) {
    static_cast<void>(engine.tick(step + 1));
    maximumEntries =
        std::max(maximumEntries, resources.snapshot().stateCache.entries);
  }
  const uint32_t checkpoints = prompt.size() / defaultCheckpointTokens;
  require(engine.idle() && events.completedCount == 2 &&
              engine.snapshot().checkpointPublications >= checkpoints &&
              engine.snapshot().checkpointPublications <= 2 * checkpoints &&
              engine.snapshot().cacheHits == 1 &&
              executor.prefillRows == prompt.size() * 2 - 24992 &&
              maximumEntries <= 2 &&
              resources.snapshot().stateCache.entries == 1,
          "concurrent prompts accumulated progress states beyond their active lanes");
}

void testSharedCheckpointSurvivesPeerRollingReplacement() {
  Backing backing(2048);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  std::vector<uint32_t> original(25001, 20);
  auto background = request(412, original);
  background.priority = RequestPriority::Background;
  engine.submit(std::move(background));
  runUntilCheckpoint(engine, 1);
  std::vector<uint32_t> branch = original;
  std::fill(branch.begin() + defaultCheckpointTokens, branch.end(), 21);
  auto foreground = request(413, branch);
  foreground.priority = RequestPriority::Foreground;
  engine.submit(std::move(foreground));
  runUntilCheckpoint(engine, 2);
  require(resources.lookup(original).resumeBoundary() == defaultCheckpointTokens,
          "rolling a shared checkpoint retired the paused peer's recovery point");
  engine.cancel(412);
  engine.cancel(413);
  runUntilIdle(engine);
  require(events.failedCount == 0, "shared checkpoint cancellation failed");
}

void testRepeatedRetriesRollTheRestoredCheckpoint() {
  Backing backing(2048);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> prompt(40001, 31);
  for (uint32_t attempt = 0; attempt < 3; ++attempt) {
    engine.submit(request(500 + attempt, prompt));
    runUntilCheckpoint(engine, attempt + 1);
    require(events.starts.back().second == attempt * defaultCheckpointTokens &&
                resources.snapshot().stateCache.entries == 1 &&
                resources.snapshot().stateCache.checkpointEntries == 1,
            "retry promoted or accumulated intermediate checkpoints");
    engine.cancel(500 + attempt);
    runUntilIdle(engine);
  }
  engine.submit(request(503, prompt));
  runUntilIdle(engine);
  require(events.starts.back().second == 3 * defaultCheckpointTokens &&
              executor.prefillRows == prompt.size() &&
              resources.snapshot().stateCache.entries == 1 &&
              resources.snapshot().stateCache.checkpointEntries == 0,
          "successful retry recomputed or retained superseded recovery states");
}

void testRetryCancelledBeforeNextCheckpointKeepsItsSource() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> prompt(25001, 32);
  engine.submit(request(510, prompt));
  runUntilCheckpoint(engine, 1);
  engine.cancel(510);
  runUntilIdle(engine);
  for (uint32_t id : {511U, 512U}) {
    engine.submit(request(id, prompt));
    require(engine.tick(1) && engine.commandInFlight() &&
                events.starts.back().second == defaultCheckpointTokens,
            "retry did not restore the previous progress point");
    engine.cancel(id);
    runUntilIdle(engine);
    require(resources.lookup(prompt).resumeBoundary() ==
                    defaultCheckpointTokens &&
                resources.snapshot().stateCache.checkpointEntries == 1,
            "cancelled retry lost or promoted its unchanged recovery point");
  }
}

void testRestoredCheckpointAtReplayEndBecomesOrdinary() {
  for (uint32_t suffix : {1U, 31U}) {
    Backing backing(1024);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(1);
    Events events;
    engine::Engine engine({}, resources, executor, events);
    const std::vector<uint32_t> prompt(25001, 33);
    engine.submit(request(520, prompt));
    runUntilCheckpoint(engine, 1);
    engine.cancel(520);
    runUntilIdle(engine);
    const uint32_t snapshots = executor.snapshots;
    std::vector<uint32_t> shorter(
        prompt.begin(), prompt.begin() + defaultCheckpointTokens + suffix);
    engine.submit(request(521, shorter));
    runUntilIdle(engine);
    require(events.starts.back().second == defaultCheckpointTokens &&
                executor.snapshots == snapshots &&
                engine.snapshot().deduplicatedStatePublications == 1 &&
                resources.snapshot().stateCache.entries == 1 &&
                resources.snapshot().stateCache.checkpointEntries == 0 &&
                resources.lookup(shorter).resumeBoundary() ==
                    defaultCheckpointTokens,
            "restored replay endpoint was copied or retired as temporary");
  }
}

void testRetryRetiresCheckpointAtDeeperJunction() {
  Backing backing(2048);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({.prefillCheckpointTokens = 8192}, resources, executor,
                        events);
  const std::vector<uint32_t> prompt(30001, 34);
  engine.submit(request(530, prompt));
  runUntilCheckpoint(engine, 2);
  require(engine.tick(50) && engine.tick(51) && !engine.commandInFlight() &&
              executor.requests.at(530).position == 18432,
          "fixture did not commit past the last recovery point");
  engine.cancel(530);
  runUntilIdle(engine);
  engine.submit(request(531, prompt));
  require(engine.tick(1) && engine.tick(2) &&
              events.starts.back().second == 16384 &&
              engine.snapshot().junctionMaterializations == 1 &&
              resources.snapshot().stateCache.entries == 1 &&
              resources.snapshot().stateCache.checkpointEntries == 0,
          "retry kept an earlier recovery point after publishing its junction");
  runUntilIdle(engine);
  require(resources.snapshot().stateCache.entries == 2 &&
              resources.snapshot().stateCache.checkpointEntries == 0,
          "retry did not retain its normal junction and replay states");
}

void testPinnedCheckpointSkipsReplacementButNotOrdinaryState() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> prompt(18001, 35);
  engine.submit(request(540, prompt));
  runUntilCheckpoint(engine, 1);
  auto pinned = resources.lookup(prompt);
  require(pinned.resumeBoundary() == defaultCheckpointTokens,
          "fixture did not pin its checkpoint");
  runUntilIdle(engine);
  require(engine.snapshot().checkpointPublications == 1 &&
              engine.snapshot().checkpointPublicationFailures ==
                  prompt.size() / defaultCheckpointTokens - 1 &&
              executor.snapshotAttempts == 2 &&
              resources.snapshot().stateCache.entries == 2 &&
              resources.snapshot().stateCache.checkpointEntries == 1 &&
              resources.lookup(prompt).resumeBoundary() == 17984,
          "pinned recovery point was overwritten or blocked ordinary publication");
  pinned = {};
  require(resources.reclaimOneState() &&
              resources.snapshot().stateCache.checkpointEntries == 0 &&
              resources.lookup(prompt).resumeBoundary() == 17984,
          "released recovery pin did not rejoin the lower-priority queue");
}

void testFailedReplacementContinuesWithoutRecoveryPoint() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> prompt(25001, 36);
  engine.submit(request(550, prompt));
  runUntilCheckpoint(engine, 1);
  executor.denySnapshotAtBoundary = 2 * defaultCheckpointTokens;
  for (uint32_t step = 0; step < 32 &&
       !engine.snapshot().checkpointPublicationFailures; ++step)
    static_cast<void>(engine.tick(step + 1));
  require(engine.snapshot().checkpointPublicationFailures == 1 &&
              resources.snapshot().stateCache.entries == 0 &&
              events.failedCount == 0,
          "denied replacement kept the retired checkpoint or failed inference");
  runUntilIdle(engine);
  require(events.completedCount == 1 && executor.prefillRows == prompt.size() &&
              resources.snapshot().stateCache.entries == 1 &&
              resources.snapshot().stateCache.checkpointEntries == 0,
          "inference did not recover from a skipped checkpoint publication");
}

void testRollingHandleCannotRetirePromotedState() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> prompt(25001, 37);
  engine.submit(request(560, prompt));
  runUntilCheckpoint(engine, 1);
  {
    auto existing = resources.lookup(prompt);
    require(resources.reuseCompositeState(existing.state->kvBlock()),
            "shared ordinary boundary could not reuse its checkpoint");
  }
  runUntilIdle(engine);
  const std::vector<uint32_t> prefix(
      prompt.begin(), prompt.begin() + defaultCheckpointTokens + 1);
  require(resources.snapshot().stateCache.entries == 2 &&
              resources.snapshot().stateCache.checkpointEntries == 0 &&
              resources.lookup(prefix).resumeBoundary() ==
                  defaultCheckpointTokens,
          "old rolling handle deleted a state promoted by another request");
}

void testCheckpointDenialPreservesUnrelatedHotState() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> hot(65, 21);
  engine.submit(request(420, hot));
  runUntilIdle(engine);
  executor.denySnapshotAtBoundary = defaultCheckpointTokens;
  engine.submit(request(421, std::vector<uint32_t>(18001, 22)));
  for (uint32_t step = 0; step < 32 &&
       !engine.snapshot().checkpointPublicationFailures; ++step)
    static_cast<void>(engine.tick(step + 1));
  require(engine.snapshot().checkpointPublicationFailures == 1 &&
              resources.snapshot().stateCache.evictions == 0 &&
              resources.lookup(hot).resumeBoundary() == 64,
          "optional progress allocation displaced an unrelated hot prefix");
  engine.cancel(421);
  runUntilIdle(engine);
}

void testCheckpointRecyclesItsBufferBeforeReplacement() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  engine.submit(request(430, std::vector<uint32_t>(25001, 23)));
  runUntilCheckpoint(engine, 1);
  executor.snapshotObserver = [&] {
    require(resources.snapshot().stateCache.entries == 0,
            "checkpoint replacement allocated before retiring its old state");
  };
  runUntilCheckpoint(engine, 2);
  require(resources.snapshot().stateCache.entries == 1 &&
              resources.snapshot().stateCache.evictions == 0 &&
              resources.snapshot().stateCache.checkpointRetirements == 1 &&
              engine.snapshot().checkpointPublicationFailures == 0,
          "progress could not replace its old state within the allocation budget");
  engine.cancel(430);
  runUntilIdle(engine);
}

void testCancelAtCheckpointDoesNotPublishDrainingCommand() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({.prefillCheckpointTokens = 8192}, resources, executor,
                        events);
  engine.submit(request(440, std::vector<uint32_t>(18001, 24)));
  for (uint32_t step = 0; step < 32; ++step) {
    static_cast<void>(engine.tick(step + 1));
    if (!engine.commandInFlight() && executor.requests.at(440).position == 6144)
      break;
  }
  executor.holdPrefillUntil = std::make_shared<bool>(false);
  static_cast<void>(engine.tick(40));
  require(engine.commandInFlight() && executor.requests.at(440).position == 8192,
          "fixture did not stop inside the checkpoint command");
  engine.cancel(440);
  *executor.holdPrefillUntil = true;
  runUntilIdle(engine);
  require(executor.snapshotAttempts == 0 &&
              engine.snapshot().checkpointPublications == 0 &&
              resources.snapshot().kvCache.blocks == 6144 / KvCache::pageTokens,
          "cancellation published a checkpoint from the draining command");
}

void testFinalStateRecyclesItsCheckpointBeforeUnrelatedHotState() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> hot(65, 27);
  engine.submit(request(460, hot));
  runUntilIdle(engine);
  const std::vector<uint32_t> prompt(18001, 28);
  engine.submit(request(461, prompt));
  runUntilCheckpoint(engine, prompt.size() / defaultCheckpointTokens);
  executor.snapshotObserver = [&] {
    require(resources.snapshot().stateCache.entries == 1 &&
                resources.lookup(hot).resumeBoundary() == 64,
            "final state did not recycle its checkpoint before allocating");
  };
  runUntilIdle(engine);
  require(resources.snapshot().stateCache.entries == 2 &&
              resources.lookup(hot).resumeBoundary() == 64,
          "final state evicted unrelated hot state before its own checkpoint");
}

void testFinalJunctionRetiresEarlierProgressPoint() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> prompt(20001, 29);
  resources.beginRequest(470);
  require(resources.ensureTokens(470, 20000).granted(),
          "junction fixture could not allocate its KV prefix");
  static_cast<void>(resources.publishCommittedBlocks(470, prompt, 20000));
  resources.endRequest(470);
  engine.submit(request(471, prompt));
  runUntilIdle(engine);
  require(engine.snapshot().checkpointPublications ==
                  prompt.size() / defaultCheckpointTokens &&
              engine.snapshot().junctionMaterializations == 1 &&
              resources.snapshot().stateCache.entries == 1 &&
              resources.lookup(prompt).resumeBoundary() == 20000,
          "prompt-end junction left a superseded progress checkpoint resident");
}

void testShortSuffixContinuesCheckpointDraftState() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> prompt(18001, 30);
  engine.submit(request(480, prompt));
  runUntilCheckpoint(engine, 1);
  engine.cancel(480);
  runUntilIdle(engine);
  std::vector<uint32_t> shorter(
      prompt.begin(), prompt.begin() + defaultCheckpointTokens + 209);
  engine.submit(request(481, shorter));
  runUntilIdle(engine);
  require(events.starts.back().second == defaultCheckpointTokens &&
              executor.restoredDraft &&
              executor.plans.at(481).draftContextRows() == 209 &&
              !executor.plans.at(481).draftStateRestoreSkipped,
          "short checkpoint suffix discarded or rebuilt its restored draft window");
}

void testDefaultCheckpointRestoresLatestCommittedPrefix() {
  Backing backing(2048);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> donor(32769, 61);
  engine.submit(request(600, donor));
  for (uint32_t step = 0; step < 128; ++step) {
    static_cast<void>(engine.tick(step + 1));
    if (!engine.commandInFlight() &&
        executor.requests.at(600).position == 22528)
      break;
  }
  const auto committed = engine.snapshot();
  require(!engine.commandInFlight() && executor.prefillRows == 22528 &&
              executor.requests.at(600).position == 22528 &&
              committed.resources.kvCache.blocks == 22528 / KvCache::pageTokens,
          "fixture did not commit 22528 prompt tokens before cancellation");
  require(committed.checkpointPublications == 5 &&
              committed.checkpointPublicationFailures == 0 &&
              committed.resources.stateCache.entries == 1 &&
              committed.resources.stateCache.checkpointEntries == 1 &&
              committed.resources.stateCache.checkpointRetirements == 4,
          "default checkpoints accumulated or lost temporary states");
  {
    const auto lookup = resources.lookup(donor);
    require(lookup.kvBoundary == 22528 && lookup.resumeBoundary() == 20480,
            "completed KV tail did not retain its preceding 4K checkpoint");
  }
  engine.cancel(600);
  runUntilIdle(engine);
  require(engine.snapshot().cancelled == 1 &&
              resources.snapshot().stateCache.checkpointEntries == 1,
          "cancellation removed its completed recovery point");

  std::vector<uint32_t> branch(donor.begin(), donor.begin() + 22529);
  branch.back() = 62;
  engine.submit(request(601, branch));
  runUntilIdle(engine);
  const auto finished = engine.snapshot();
  require(events.startIds.back() == 601 &&
              events.starts.back().second == 20480 &&
              executor.prefillRows - 22528 == 2049 &&
              finished.completed == 1 && finished.cancelled == 1 &&
              events.failedCount == 0,
          "branch replayed before its latest checkpoint or failed completion");
  require(finished.checkpointPublications == 5 &&
              finished.junctionMaterializations == 1 &&
              finished.resources.stateCache.entries == 1 &&
              finished.resources.stateCache.checkpointEntries == 0 &&
              finished.resources.stateCache.checkpointRetirements == 5 &&
              resources.lookup(branch).resumeBoundary() == 22528,
          "branch completion retained its superseded temporary checkpoint");
}

void testCheckpointIntervalValidationAndDisable() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  for (uint32_t invalid : {1U, 2047U, 2049U}) {
    bool rejected = false;
    try {
      engine::Engine engine({.prefillCheckpointTokens = invalid}, resources,
                            executor, events);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "invalid checkpoint interval was accepted");
  }
  engine::Engine engine({.prefillCheckpointTokens = 0}, resources, executor, events);
  engine.submit(request(450, std::vector<uint32_t>(18001, 25)));
  runUntilIdle(engine);
  require(engine.snapshot().checkpointPublications == 0 && executor.snapshots == 1 &&
              resources.snapshot().stateCache.entries == 1,
          "disabling checkpoints changed existing replay-state behavior");
}

} // namespace

int main() {
  try {
    testConcurrentColdPrefixesComputeOnce();
    testSharedPrefillRebuildsTheMissingJunctionOnce();
    testSharedPrefillReleasesDifferentJunctionsIndependently();
    testSharedPrefillEvictedPublicationFallsBack();
    testSharedPrefillProducerFailureReleasesWaiters();
    testSharedPrefillWaiterCancellationAndDeadline();
    testSharedPrefillFailedPublicationFallsBack();
    testSharedPrefillDoesNotBlockUnrelatedWork();
    testSharedPrefillHonorsPriorityAndLateArrival();
    testLateSharedPrefillExtendsTheProducerPlan();
    testSharedPrefillCapacityFailureDoesNotDeadlock();
    testCancelledColdPrefillResumesItsLatestCheckpoint();
    testConcurrentProgressRetainsAtMostOnePointPerLane();
    testSharedCheckpointSurvivesPeerRollingReplacement();
    testRepeatedRetriesRollTheRestoredCheckpoint();
    testRetryCancelledBeforeNextCheckpointKeepsItsSource();
    testRestoredCheckpointAtReplayEndBecomesOrdinary();
    testRetryRetiresCheckpointAtDeeperJunction();
    testPinnedCheckpointSkipsReplacementButNotOrdinaryState();
    testFailedReplacementContinuesWithoutRecoveryPoint();
    testRollingHandleCannotRetirePromotedState();
    testCheckpointDenialPreservesUnrelatedHotState();
    testCheckpointRecyclesItsBufferBeforeReplacement();
    testCancelAtCheckpointDoesNotPublishDrainingCommand();
    testFinalStateRecyclesItsCheckpointBeforeUnrelatedHotState();
    testFinalJunctionRetiresEarlierProgressPoint();
    testShortSuffixContinuesCheckpointDraftState();
    testDefaultCheckpointRestoresLatestCommittedPrefix();
    testCheckpointIntervalValidationAndDisable();
    testColdPublishesReplayStateAndLazyJunctionCanRebuildIt();
    testConcurrentDuplicateStateSkipsSnapshotCapture();
    testImageSpansKeyPrefixIdentity();
    testOneRequestPublishesJunctionAndLatestReplayState();
    testLatestReplayDenialRecyclesOlderStateNotTheJunction();
    testCancellationAfterJunctionDiscardsLaterState();
    testDeniedSnapshotCostsOnlyThatAttempt();
    testDeniedSnapshotRecyclesLruStateAndRetries();
    testPersistentSnapshotDenialRecyclesAtMostOneState();
    testLongSuffixSkipsDraftRestore();
    testCancellationInFlightAtBoundaryPublishesNoState();
    testActiveCellGrowthReclaimsCachedStateAndRetries();
    testKvGrowthReclaimsIdleStateBeforeCache();
    testKvGrowthDenialKeepsEveryLaneReplayState();
    testPressureReclaimRespectsStateLifetimes();
    testConcurrencyLimitDoesNotEvictCache();
    testHostPressureDoesNotDrainCacheOnStateAdmission();
    testHostPressureStillRecyclesLruStateForDeniedSnapshot();
    testSingletonHostPressureReusesIdleCacheInsteadOfSuspending();
    testSingletonHostPressureWaitRecoversOrTerminates();
    testKvPressureNarrowsTheRealBatch();
    testKvGrowthReclaimsCachedStateWhenBudgetIsShared();
    testRequiredWorkDoesNotReserveAnExtraPage();
    testAdmissionPinsDesiredStateAndCountsOnlySuccess();
    testAdmissionCanDropItsOwnCachePinToMakeProgress();
    testSingletonCapacityFailureTerminatesCleanly();
    testGrowthKeepsPrefillProgressWhenAnUnstartedPeerCanYield();
    testGrowthYieldsLowerPriorityResidentOutsideBatch();
    testPrefillGrowthPreservesAnActiveDecodePeer();
    testPhysicalKvPressureSuspendsInsteadOfKillingActiveWork();
    testRecoveryDrainDoesNotConsumeResourceWaitBudget();
    testPhysicalPressureRetryIsBackedOffWithoutProgress();
    testDecodePreemptionReplaysCommittedHistoryWithoutRepeatingOutput();
    testLongDecodePreemptionPlansTheCurrentReplayBoundary();
    testPreemptedDecodeRestoresItsResidentCompositeState();
    testRepeatedPreemptionRespectsBackoffAndCancellation();
    testAdmissionReopensAfterLastSuspendedRequestResumes();
    testRecoveryAdmitsFailedKvTargetBeforeReplaying();
    testStateAdmissionWaitsForKvRelease();
    testAdmissionsWaitForBackgroundRelease();
    testAllocationCausesRemainRetryableAndDistinct();
    testAdmissionRespectsPriorityBeforeHashOrder();
    testConstraintMaskOverlapsInsideOneSchedulerBatch();
    testConstraintMaskWaitHonorsCancelAndDeadline();
    testDecodeNearContextCeilingCoversVerifyRows();
    testExpiredMaskWaitFinalizesWhileAnotherCommandRuns();
    testOrdinaryInFlightDeadlineDrainsWithoutPublishingOrOutput();
    testStalledSuspensionFailsWithCapacity();
    testTerminalAnchorWithoutKvIsNotCached();
    testPrefillCanCompleteTheRequest();
    std::cout << "KV-first engine tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "KV-first engine tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

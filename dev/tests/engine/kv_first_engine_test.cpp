#include "ScopedTestConfig.hpp"
#include "TestCache.hpp"
#include "TestChecks.hpp"
#include "TestImmediateTicket.hpp"
#include "TestKvPool.hpp"
#include "TestKvTier.hpp"
#include "benchmarks/PrefillWork.hpp"
#include "engine/Engine.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>

using namespace splash;
using namespace splash::engine;
using benchmark::draftContextRows;

namespace {

class State final : public CompositeState {
public:
  State() = default;
  // A lane's snapshot: once the cache lets go of it, its buffers return to
  // the model's pool, as QwenStateStorage's do.
  explicit State(std::shared_ptr<uint64_t> pool) : pool_(std::move(pool)) {}
  ~State() override {
    if (pool_)
      *pool_ += bytes();
  }
  uint64_t bytes() const noexcept override { return 64; }

private:
  std::shared_ptr<uint64_t> pool_;
};

class DiskState final : public CompositeState {
public:
  uint64_t bytes() const noexcept override { return 64; }
  uint64_t residentBytes() const noexcept override { return 0; }
};

// A state that holds rows its model can rebuild, as a replay point inside an
// image holds that image's rows: once the cache drops the state, the rows
// are one more of the model's cache units.
class RowsHoldingState final : public CompositeState {
public:
  RowsHoldingState(std::vector<uint64_t> &cacheUnits, uint64_t rows)
      : cacheUnits_(cacheUnits), rows_(rows) {}
  ~RowsHoldingState() override { cacheUnits_.push_back(rows_); }
  uint64_t bytes() const noexcept override { return 64; }

private:
  std::vector<uint64_t> &cacheUnits_;
  uint64_t rows_;
};

struct OffloadControl {
  bool ready = false;
  bool released = false;
};

// A state write in flight; its disk copy is a DiskState.
class OffloadTicket final : public StateOffload {
public:
  explicit OffloadTicket(std::shared_ptr<OffloadControl> control)
      : control_(std::move(control)) {}
  bool ready() const noexcept override { return control_->ready; }
  bool finish() override { return true; }
  const std::shared_ptr<const CompositeState> &state() const noexcept override {
    return disk_;
  }
private:
  std::shared_ptr<OffloadControl> control_;
  std::shared_ptr<const CompositeState> disk_ = std::make_shared<DiskState>();
};

// With a pool, it is a lane's snapshot whose buffers return to the model's
// pool once the cache lets go of them, like State's.
class OffloadState final : public CompositeState {
public:
  explicit OffloadState(std::shared_ptr<OffloadControl> control,
                        std::shared_ptr<uint64_t> pool = nullptr)
      : control_(std::move(control)), pool_(std::move(pool)) {}
  ~OffloadState() override {
    control_->released = true;
    if (pool_)
      *pool_ += bytes();
  }
  uint64_t bytes() const noexcept override { return 64; }
  bool canOffload() const noexcept override { return true; }
  std::unique_ptr<StateOffload> offload(std::function<void()>) const override {
    return std::make_unique<OffloadTicket>(control_);
  }
private:
  std::shared_ptr<OffloadControl> control_;
  std::shared_ptr<uint64_t> pool_;
};

// A lane's snapshot whose copy a persistent tier writes while it stays in
// RAM (persist); the copy is a DiskState.
class KeptState final : public CompositeState {
public:
  explicit KeptState(std::shared_ptr<OffloadControl> control) : control_(std::move(control)) {}
  uint64_t bytes() const noexcept override { return 64; }
  bool canOffload() const noexcept override { return true; }
  std::unique_ptr<StateOffload> persist(std::function<void()>) const override {
    return std::make_unique<OffloadTicket>(control_);
  }

private:
  std::shared_ptr<OffloadControl> control_;
};

struct RestoreControl {
  bool ready = false;
  bool success = true;
  bool cancelled = false;
  // No cache slot for the restored state's RAM copy.
  bool promotionDenied = false;
};

class RestoreTicket final : public StateRestore {
public:
  std::shared_ptr<RestoreControl> control;
  std::function<void()> commit;
  bool ready() const noexcept override { return control->ready; }
  bool finish() override {
    if (!control->success || control->cancelled) return false;
    commit();
    return true;
  }
  void cancel() noexcept override { control->cancelled = true; }
  std::shared_ptr<const CompositeState> snapshot() override {
    if (control->promotionDenied)
      return nullptr;
    return std::make_shared<State>();
  }
};

struct MaskOverlapState final {
  uint64_t requestId = 0;
  bool finishes = true;
  bool emitted = false;
  bool provided = false;
  bool abandoned = false;
  // Keeps the command running after its mask wait ended.
  bool held = false;
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
    return !state_->held && (state_->provided || state_->abandoned);
  }
  std::vector<ModelStepResult> wait() override {
    if (!ready())
      throw std::logic_error("mask-overlap ticket completed without input");
    return {{state_->requestId, 0, {42}, state_->finishes,
             DecodeStage::Regular, 7, 0}};
  }
  double wallMilliseconds() const noexcept override { return 1.0; }

private:
  std::shared_ptr<MaskOverlapState> state_;
};

class Executor final : public model::Model {
public:
  explicit Executor(
      uint32_t maximumLanes = model::ExecutionLimits::maximumBatchWidth)
      : maximumLanes(maximumLanes) {}

  StateAdmission begin(const ModelRequest &request) override {
    ++beginAttempts;
    lastBeginId = request.id;
    lastRestoredTokens = request.restoredTokens;
    if (beginObserver) beginObserver();
    if (beginGrowthBlocked && beginGrowthBlocked())
      return {{}, StateFailure::MemoryPressure, beginAllocationFailure};
    if (deniedBegins) {
      --deniedBegins;
      return {{}, StateFailure::MemoryPressure, metal::AllocationFailure::EngineBudget};
    }
    for (uint32_t lane = 0; lane < maximumLanes; ++lane) {
      const bool used = std::any_of(
          requests.begin(), requests.end(), [lane](const auto &entry) {
            return entry.second.resident && entry.second.lane == lane;
          });
      if (!used) {
        requests.emplace(
            request.id,
            Request{.lane = lane,
                    .resident = true,
                    .constrained =
                        request.constraint == ConstraintMode::TokenMask});
        prompts[request.id].assign(request.prompt.begin(), request.prompt.end());
        return {lane, StateFailure::None};
      }
    }
    return {{}, StateFailure::ConcurrencyLimit};
  }
  void suspend(uint64_t id) override {
    Request &entry = requests.at(id);
    if (!entry.resident)
      throw std::logic_error("request is already suspended");
    pageTables.erase(id);
    entry.resident = false;
    entry.position = 0;
    ++suspensions;
    if (kvGrowthBlocked && unblockGrowthOnSuspend)
      *kvGrowthBlocked = false;
  }
  StateAdmission resume(const ModelRequest &request) override {
    ++resumeAttempts;
    lastRestoredTokens = request.restoredTokens;
    if (resumeDenied)
      return {{}, StateFailure::MemoryPressure, metal::AllocationFailure::HostPressure};
    const uint64_t id = request.id;
    Request &entry = requests.at(id);
    for (uint32_t lane = 0; lane < maximumLanes; ++lane) {
      const bool used = std::any_of(
          requests.begin(), requests.end(), [&](const auto &candidate) {
            return candidate.first != id && candidate.second.resident &&
                   candidate.second.lane == lane;
          });
      if (!used) {
        entry.lane = lane;
        entry.resident = true;
        entry.position = 0;
        entry.replaying = true;
        resumedPrompts.emplace_back(request.prompt.begin(), request.prompt.end());
        ++resumptions;
        return {lane, StateFailure::None};
      }
    }
    return {{}, StateFailure::ConcurrencyLimit};
  }
  std::unique_ptr<StateRestore> beginRestore(
      uint64_t id, uint32_t length, std::shared_ptr<const CompositeState> state,
      bool restoreDraft, std::function<void()>) override {
    if (state->residentBytes()) {
      applyRestore(id, length, std::move(state), restoreDraft);
      return {};
    }
    ++diskReads;
    auto ticket = std::make_unique<RestoreTicket>();
    ticket->control = restoreControl;
    ticket->commit = [this, id, length, state, restoreDraft] {
      applyRestore(id, length, state, restoreDraft);
    };
    return ticket;
  }
  void setDraftContextPlan(uint64_t id, DraftContextPlan plan) override {
    plans[id] = std::move(plan);
  }
  std::vector<ModelStepResult> prefill(const BatchPlan &,
                                       std::span<const ModelBatchItem> items) {
    prefillWidths.push_back(static_cast<uint32_t>(items.size()));
    std::vector<ModelStepResult> result;
    for (const auto &item : items) {
      requireRows(item, item.logicalPosition);
      if (kv) {
        const uint64_t end = item.logicalPosition + item.tokenCount;
        for (uint64_t row = (item.logicalPosition + KvCache::pageTokens - 1) /
                            KvCache::pageTokens * KvCache::pageTokens;
             row < end; row += KvCache::pageTokens) {
          kv->content.at(item.pageTable[row / KvCache::pageTokens]) =
              rows(row / KvCache::pageTokens, item.inputTokens[row - item.logicalPosition]);
        }
      }
      Request &state = requests.at(item.requestId);
      state.position += item.tokenCount;
      prefillRows += item.tokenCount;
      // A constrained prompt's end waits for the first mask; a replay never
      // asks for it again.
      const bool awaitsMask = state.constrained && !state.replaying &&
                              state.position == prompts.at(item.requestId).size();
      ModelStepResult step{item.requestId, item.tokenCount, {}, false,
                           awaitsMask ? DecodeStage::ApplyInitialMask
                                      : DecodeStage::Regular,
                           0, 0};
      if (prefillAnchor && !state.replaying) {
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
      requireRows(item, item.logicalPosition);
      // The production runtime stores eight verify rows from the lane's
      // position; the engine must have covered them with page-table entries.
      if (uint64_t{item.pageTable.size()} * KvCache::pageTokens <
          item.logicalPosition + model::ExecutionLimits::targetVerifyRows) {
        throw std::invalid_argument("page_table_too_short");
      }
      if (plan.decodeStage == DecodeStage::ApplyInitialMask) {
        // Selects the first token under its mask and drafts nothing.
        result.push_back({item.requestId, 0, {}, false, DecodeStage::Regular,
                          0, 0});
      } else {
        const std::vector<uint32_t> tokens =
            item.requestId == poisonRequest && poisonToken
                ? std::vector<uint32_t>{*poisonToken}
                : std::vector<uint32_t>{42};
        ModelStepResult step{item.requestId, 0, tokens, decodeFinishes,
                             DecodeStage::Regular, 0, 0};
        step.outputTokensWithoutKv = decodeTokensWithoutKv;
        result.push_back(std::move(step));
      }
    }
    return result;
  }
  // What the runtime's GPU page tables rely on: a revision names one page
  // list, starts at one and only moves forward, and the list at the next
  // revision keeps the pages below its first changed one. The cache starts
  // a request's revisions again once the engine suspends or ends it.
  void checkPageTables(std::span<const ModelBatchItem> items) {
    for (const ModelBatchItem &item : items) {
      if (!item.pageTableRevision)
        throw std::logic_error("a page table has no revision");
      const auto [found, inserted] = pageTables.try_emplace(item.requestId);
      PageTableShadow &shadow = found->second;
      const auto unchangedBelow = [&](size_t first) {
        return first <= shadow.pages.size() && first <= item.pageTable.size() &&
               std::equal(shadow.pages.begin(), shadow.pages.begin() + first,
                          item.pageTable.begin());
      };
      if (!inserted &&
          (item.pageTableRevision < shadow.revision ||
           (item.pageTableRevision == shadow.revision &&
            !std::ranges::equal(shadow.pages, item.pageTable)) ||
           (item.pageTableRevision == shadow.revision + 1 &&
            !unchangedBelow(item.pageTableFirstChanged))))
        throw std::logic_error("a page table revision does not describe its pages");
      shadow = {item.pageTableRevision, {item.pageTable.begin(), item.pageTable.end()}};
    }
  }
  std::unique_ptr<ModelBatchTicket>
  submit(const BatchPlan &plan, std::span<const ModelBatchItem> items,
         std::function<void()> completion) override {
    checkPageTables(items);
    // Every constrained cycle after the first token waits for its mask
    // inside the ticket, as the production constrained ticket does.
    if (plan.kind == WorkKind::Decode && plan.constrained &&
        plan.decodeStage == DecodeStage::Regular) {
      overlap = std::make_shared<MaskOverlapState>();
      overlap->requestId = items.front().requestId;
      overlap->finishes = decodeFinishes;
      return std::make_unique<MaskOverlapTicket>(overlap);
    }
    if (plan.kind == WorkKind::Decode && holdDecodeUntil) {
      return std::make_unique<test::HeldTicket>(decode(plan, items),
                                                holdDecodeUntil, 1.0);
    }
    if (plan.kind == WorkKind::Prefill && holdPrefillUntil) {
      return std::make_unique<test::HeldTicket>(prefill(plan, items),
                                                holdPrefillUntil, 1.0);
    }
    return test::immediateTicket(plan.kind == WorkKind::Prefill
                                     ? prefill(plan, items)
                                     : decode(plan, items),
                                 completion);
  }
  uint64_t snapshotBytes() const noexcept override { return stateBytes; }
  // The production model copies the lane's state into a cache slot at its
  // current page-aligned boundary and returns nullptr when no slot is free
  // and the governor denies a new one. The fake denies the next
  // `deniedSnapshots` calls, every call made at `denySnapshotAtBoundary`,
  // and every call while `snapshotRoom` reports no memory.
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
    if (snapshotRoom && !snapshotRoom())
      return nullptr;
    ++snapshots;
    if (keptTier)
      return std::make_shared<KeptState>(keptTier);
    if (stateHeldRows)
      return std::make_shared<RowsHoldingState>(cacheUnits, stateHeldRows);
    return std::make_shared<State>(evictedStateBytes);
  }
  // Without a cache slot the production model writes the lane's state to
  // the disk tier; the fake has one when `stateTier` is set, with quota for
  // every state.
  bool canSnapshotToDisk() const noexcept override { return stateTier != nullptr; }
  std::unique_ptr<StateOffload> snapshotToDisk(uint64_t, std::function<void()>) override {
    ++diskSnapshots;
    return std::make_unique<OffloadTicket>(stateTier);
  }
  uint32_t statesToActivate() const noexcept override {
    return statesLacked ? statesLacked() : 0;
  }
  uint64_t reclaimIdleState(bool keepLane, model::IdleMemory scope) noexcept override {
    keptLane = keepLane;
    // The buffers of states the cache let go of refill the lane's footprint;
    // the rest is idle.
    const uint64_t refill = std::min(
        *evictedStateBytes, laneFootprintBytes - std::min(laneFootprintBytes, pooledLaneBytes));
    pooledLaneBytes += refill;
    reclaimableIdleStateBytes += *evictedStateBytes - refill;
    *evictedStateBytes = 0;
    if (!keepLane && pooledLaneBytes)
      return std::exchange(pooledLaneBytes, 0);
    const uint64_t released = reclaimableIdleStateBytes;
    reclaimableIdleStateBytes = 0;
    reclaimedIdleStateBytes += released;
    if (released && kvGrowthBlocked)
      *kvGrowthBlocked = false;
    if (released || scope == model::IdleMemory::Buffers)
      return released;
    ++cacheReclaims;
    if (cacheUnits.empty())
      return 0;
    const uint64_t unit = cacheUnits.back();
    cacheUnits.pop_back();
    return unit;
  }
  std::optional<std::string> provideMask(uint64_t id,
                                         std::span<const uint32_t>) override {
    if (overlap && overlap->emitted && overlap->requestId == id)
      overlap->provided = true;
    return std::nullopt;
  }
  void end(uint64_t id) override {
    requests.erase(id);
    pageTables.erase(id);
  }

  // What a page holds once prefill has written a block's first row to it.
  static uint64_t rows(uint32_t block, uint32_t firstToken) {
    return (uint64_t{block} << 32) | firstToken;
  }
  // With `kv`, a step finds every prompt block before `position` through
  // the page table the engine handed it.
  void requireRows(const ModelBatchItem &item, uint64_t position) const {
    if (!kv)
      return;
    const std::vector<uint32_t> &prompt = prompts.at(item.requestId);
    const uint64_t blocks =
        std::min<uint64_t>(position, prompt.size()) / KvCache::pageTokens;
    for (uint32_t block = 0; block < blocks; ++block) {
      if (kv->content.at(item.pageTable[block]) !=
          rows(block, prompt[block * KvCache::pageTokens]))
        throw std::logic_error("a page does not hold its block's rows");
    }
  }

  struct Request {
    uint32_t lane = 0;
    uint32_t position = 0;
    bool resident = false;
    bool replaying = false;
    bool constrained = false;
  };
  std::unordered_map<uint64_t, Request> requests;
  // The page list each request's last item named, at its revision.
  struct PageTableShadow {
    uint64_t revision = 0;
    std::vector<uint32_t> pages;
  };
  std::unordered_map<uint64_t, PageTableShadow> pageTables;
  // The storage whose pages the fake writes and checks: prefill marks the
  // page of every block it starts, as the production model writes its rows
  // there, and each later step requires the marks of the blocks before it.
  test::TestKvStorage *kv = nullptr;
  std::unordered_map<uint64_t, std::vector<uint32_t>> prompts;
  std::unordered_map<uint64_t, DraftContextPlan> plans;
  std::shared_ptr<RestoreControl> restoreControl = std::make_shared<RestoreControl>();
  uint32_t diskReads = 0;
  uint32_t prefillRows = 0;
  uint32_t restored = 0;
  uint32_t snapshots = 0;
  uint32_t snapshotAttempts = 0;
  uint32_t diskSnapshots = 0;
  std::shared_ptr<OffloadControl> stateTier;
  // With it, snapshots are KeptStates whose copies land when it says so.
  std::shared_ptr<OffloadControl> keptTier;
  // What a snapshot allocates; State::bytes() unless a test needs a state
  // larger than an extent.
  uint64_t stateBytes = 64;
  uint32_t deniedSnapshots = 0;
  std::optional<uint32_t> denySnapshotAtBoundary;
  uint32_t beginAttempts = 0;
  // The request of the latest begin(), for hooks that refuse only some.
  uint64_t lastBeginId = 0;
  // The restored prefix the latest begin() or resume() activated with.
  uint32_t lastRestoredTokens = 0;
  // Begins the budget refuses, and resumes the host refuses.
  uint32_t deniedBegins = 0;
  uint32_t suspensions = 0;
  uint32_t resumptions = 0;
  uint32_t resumeAttempts = 0;
  bool resumeDenied = false;
  uint32_t maximumLanes = model::ExecutionLimits::maximumBatchWidth;
  std::function<void()> beginObserver;
  std::function<void()> snapshotObserver;
  std::function<bool()> snapshotRoom;
  std::function<bool()> beginGrowthBlocked;
  std::vector<std::vector<uint32_t>> resumedPrompts;
  std::vector<uint32_t> prefillWidths;
  bool restoredDraft = false;
  metal::AllocationFailure beginAllocationFailure =
      metal::AllocationFailure::EngineBudget;
  uint64_t reclaimableIdleStateBytes = 0;
  // The pooled buffers a lane starts from; only a reclaim that does not keep
  // the lane releases them. A reclaim first refills them, up to
  // laneFootprintBytes, from the buffers of evicted states.
  uint64_t pooledLaneBytes = 0;
  uint64_t laneFootprintBytes = 0;
  // The buffers of snapshots the cache has let go of, until reclaimIdleState
  // takes them into the pool.
  std::shared_ptr<uint64_t> evictedStateBytes = std::make_shared<uint64_t>(0);
  // The cached states whose buffers the pool lacks for one activation.
  std::function<uint32_t()> statesLacked;
  uint64_t reclaimedIdleStateBytes = 0;
  // Caches the model can rebuild, one released per reclaim step that may
  // take them once no buffer is idle; cacheReclaims counts those steps.
  std::vector<uint64_t> cacheUnits;
  uint32_t cacheReclaims = 0;
  // Rows each state snapshotted from then on holds (RowsHoldingState); the
  // executor must outlive the cache that keeps those states.
  uint64_t stateHeldRows = 0;
  bool keptLane = false;
  bool *kvGrowthBlocked = nullptr;
  bool unblockGrowthOnSuspend = true;
  bool decodeFinishes = true;
  uint32_t decodeTokensWithoutKv = 0;
  // When set, the decode step emits poisonToken (instead of 42) for
  // poisonRequest, exercising the engine's output validation.
  uint64_t poisonRequest = 0;
  std::optional<uint32_t> poisonToken;
  // Set when prefill itself ends the request: the value is `finished` (stop).
  std::optional<bool> prefillAnchor;
  std::shared_ptr<std::atomic<bool>> holdDecodeUntil;
  std::shared_ptr<std::atomic<bool>> holdPrefillUntil;
  std::shared_ptr<MaskOverlapState> overlap;

private:
  void applyRestore(uint64_t id, uint32_t length,
                    std::shared_ptr<const CompositeState> state,
                    bool restoreDraftState) {
    if (!state)
      throw std::runtime_error("empty restore state");
    requests.at(id).position = length;
    restored += length;
    restoredDraft = restoreDraftState;
  }
};

class Events final : public EngineEventSink {
public:
  void batchCompleted(WorkKind kind, uint32_t, uint32_t, uint32_t, uint32_t,
                      uint32_t, double, double cycleMilliseconds) override {
    cycles.emplace_back(kind, cycleMilliseconds);
  }
  void started(uint64_t requestId, uint32_t matched, uint32_t) override {
    startIds.push_back(requestId);
    starts.push_back(matched);
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
                 uint32_t completion, std::span<const float>) override {
    ++completedCount;
    usage[id] = {prompt, completion};
  }
  void failed(uint64_t, LaneOutcome outcome, std::string message) override {
    ++failedCount;
    if (outcome == LaneOutcome::CapacityExhausted)
      ++capacityExhaustedCount;
    const LaneOutcomeWire wire = laneOutcomeWire(outcome);
    failures.emplace_back(wire.code);
    failureDetails.emplace_back(std::move(message), wire.retryable);
  }

  std::unordered_map<uint64_t, std::vector<uint32_t>> progress;
  // The matched prompt tokens of each start, zero for a cold one.
  std::vector<uint32_t> starts;
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
  // Each completed command's kind and engine cycle.
  std::vector<std::pair<WorkKind, double>> cycles;
};

using splash::test::require;

// Every submitted request has ended and no command is in flight.
bool idle(const engine::Engine &engine) {
  const EngineSnapshot counts = engine.snapshot();
  return counts.submitted == counts.completed + counts.cancelled + counts.failed &&
         !engine.commandInFlight();
}

EngineRequest request(uint64_t id, const std::vector<uint32_t> &prompt) {
  EngineRequest result;
  result.id = id;
  result.prompt = prompt;
  result.maxNewTokens = 1;
  result.deadlineMilliseconds = 10000;
  return result;
}

// Every engine test holds the engine to PageStorage's rule: no extent is
// released while a command is in flight.
void guardReleases(test::TestKvStorage &storage, const engine::Engine &engine) {
  storage.commandInFlight = [&engine] { return engine.commandInFlight(); };
}

// Ticks until every prompt (65 tokens of its request id) has published a
// state, then holds a lookup on each, which keeps its state resident.
std::vector<CacheLookup> runUntilStatesHeld(engine::Engine &engine, engine::Cache &cache,
                                            std::initializer_list<uint64_t> ids) {
  for (double now = 1; now < 100 && cache.snapshot().stateCache.entries < ids.size(); ++now)
    static_cast<void>(engine.tick(now));
  require(cache.snapshot().stateCache.entries == ids.size(), "the prompts did not publish states");
  std::vector<CacheLookup> held;
  for (uint64_t id : ids)
    held.push_back(cache.lookup(std::vector<uint32_t>(65, static_cast<uint32_t>(id)), {}));
  return held;
}

void runUntilIdle(engine::Engine &engine) {
  for (uint32_t step = 0; step < 32 && !idle(engine); ++step) {
    static_cast<void>(engine.tick(step + 1));
  }
  require(idle(engine), "engine did not reach idle");
}

// Ticks on from `now` until done() holds.
void tickUntil(engine::Engine &engine, double &now, const std::function<bool()> &done,
               const char *message) {
  for (uint32_t step = 0; step < 1000 && !done(); ++step)
    static_cast<void>(engine.tick(now++));
  require(done(), message);
}

void testConcurrentColdPrefixesComputeOnce() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor model;
  Events events;
  engine::Engine engine({}, cache, model, events);
  guardReleases(storage, engine);
  for (uint32_t id = 1; id <= 4; ++id) {
    std::vector<uint32_t> prompt(193, 7);
    std::fill(prompt.begin() + 160, prompt.end(), id + 10);
    engine.submit(request(id, prompt));
  }
  static_cast<void>(engine.tick(0));
  require(model.requests.size() == 1,
          "shared cold prefix allocated redundant lanes");
  runUntilIdle(engine);
  require(model.prefillRows == 160 + 4 * 33 && model.restored == 3 * 160,
          "concurrent cold requests recomputed their shared prefix");
  require(events.completedCount == 4 && events.failedCount == 0 &&
              events.emitted == 4,
          "shared prefill lost independent completions");
}

void testSharedPrefillRebuildsTheMissingJunctionOnce() {
  test::TestKvStorage storage(512, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor model;
  Events events;
  engine::Engine engine({}, cache, model, events);
  guardReleases(storage, engine);
  engine.submit(request(1, std::vector<uint32_t>(6530, 7)));
  runUntilIdle(engine);
  std::vector<uint32_t> branch(6575, 7);
  std::fill(branch.begin() + 6517, branch.end(), 8);
  {
    auto hit = cache.lookup(branch, {});
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
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor model;
  Events events;
  engine::Engine engine({}, cache, model, events);
  guardReleases(storage, engine);
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
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor model;
  Events events;
  engine::Engine engine({}, cache, model, events);
  guardReleases(storage, engine);
  engine.submit(request(1, std::vector<uint32_t>(193, 7)));
  engine.submit(request(2, std::vector<uint32_t>(193, 7)));
  static_cast<void>(engine.tick(0));
  static_cast<void>(engine.tick(1));
  // The producer's replay point is in use, so pressure takes it, not an
  // ordinary publication.
  static_cast<void>(cache.evictAll());
  require(cache.snapshot().stateCache.entries == 0,
          "published shared prefix was pinned against pressure reclamation");
  runUntilIdle(engine);
  require(events.completedCount == 2 && model.prefillRows == 386 &&
              cache.snapshot().activeRequests == 0,
          "evicted shared publication stranded its waiter");
}

void testSharedPrefillProducerFailureReleasesWaiters() {
  for (bool cancelled : {false, true}) {
    test::TestKvStorage storage(64, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor model;
    Events events;
    engine::Engine engine({}, cache, model, events);
    guardReleases(storage, engine);
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
      engine.failRequest(1, LaneOutcome::InvalidMask, "producer failed");
    runUntilIdle(engine);
    require(events.outputs[2] == std::vector<uint32_t>{42} &&
                cache.snapshot().activeRequests == 0 && model.requests.empty(),
            "failed prefix producer stranded a waiter or leaked resources");
  }
}

void testSharedPrefillWaiterCancellationAndDeadline() {
  for (bool cancelled : {false, true}) {
    test::TestKvStorage storage(64, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor model;
    Events events;
    engine::Engine engine({}, cache, model, events);
    guardReleases(storage, engine);
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
            "expired prefix waiter allocated a lane or interrupted its producer");
    require(!cancelled || events.usage.at(2) == std::pair<uint32_t, uint32_t>{193, 0},
            "a request cancelled before it started reported other usage");
  }
}

void testSharedPrefillFailedPublicationFallsBack() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor model;
  model.deniedSnapshots = 100;
  Events events;
  engine::Engine engine({}, cache, model, events);
  guardReleases(storage, engine);
  for (uint32_t id = 1; id <= 4; ++id)
    engine.submit(request(id, std::vector<uint32_t>(193, 7)));
  runUntilIdle(engine);
  require(events.completedCount == 4 && events.failedCount == 0 &&
              model.prefillRows == 4 * 193 && cache.snapshot().activeRequests == 0,
          "a missing prefix snapshot stranded dependent requests");
}

void testSharedPrefillDoesNotBlockUnrelatedWork() {
  for (bool images : {false, true}) {
    test::TestKvStorage storage(64, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor model;
    Events events;
    engine::Engine engine({}, cache, model, events);
    guardReleases(storage, engine);
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
    test::TestKvStorage storage(512, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor model;
    Events events;
    engine::Engine engine({}, cache, model, events);
    guardReleases(storage, engine);
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
    test::TestKvStorage storage(512, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor model;
    if (denyCheckpoint)
      model.denySnapshotAtBoundary = 4096;
    Events events;
    engine::Engine engine({}, cache, model, events);
    guardReleases(storage, engine);
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

// A sibling's shared boundary on one of the producer's checkpoints makes that
// checkpoint a state the sibling resumes from: it is published as a junction
// and outlives the producer's later progress instead of rolling with it.
void testSharedJunctionAtACheckpointIsReusable() {
  test::TestKvStorage storage(512, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  // The producer checkpoints at 4096; the sibling shares its first 4100
  // tokens.
  std::vector<uint32_t> prompt(8193);
  std::iota(prompt.begin(), prompt.end(), 1);
  auto producer = request(1, prompt);
  // Admitted first, it plans the sibling's junction on its own checkpoint.
  producer.priority = RequestPriority::Foreground;
  engine.submit(std::move(producer));
  std::vector<uint32_t> sibling(prompt.begin(), prompt.begin() + 4100);
  sibling.resize(4200, 0);
  engine.submit(request(2, sibling));
  runUntilIdle(engine);
  const auto counters = engine.snapshot();
  require(events.starts.at(1) == 4096 &&
              executor.restored == 4096,
          "the sibling did not resume from the producer's shared boundary");
  require(counters.junctionMaterializations == 1 && counters.checkpointPublications == 0 &&
              counters.resources.stateCache.checkpointRetirements == 0 &&
              counters.resources.stateCache.checkpointEntries == 0,
          "the shared boundary was published as a rolling checkpoint");
  std::vector<uint32_t> next(prompt.begin(), prompt.begin() + 4100);
  next.resize(4140, 1);
  require(resources.lookup(next, {}).resumeBoundary() == 4096,
          "the producer's later progress retired the shared junction");
}

void testSharedPrefillCapacityFailureDoesNotDeadlock() {
  test::TestKvStorage storage(8, 4096, 4);
  storage.budgetPages = 4;
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor model;
  Events events;
  engine::Engine engine({}, cache, model, events);
  guardReleases(storage, engine);
  for (uint32_t id = 1; id <= 4; ++id)
    engine.submit(request(id, std::vector<uint32_t>(193, 7)));
  for (uint32_t step = 0; step < 64 && !idle(engine); ++step)
    static_cast<void>(engine.tick(step * 1000));
  require(idle(engine) && cache.snapshot().activeRequests == 0 &&
              model.requests.empty() && events.emitted == 0,
          "capacity failure stranded a shared prefix producer or waiter");
  engine.submit(request(5, std::vector<uint32_t>(33, 9)));
  runUntilIdle(engine);
  require(events.outputs[5] == std::vector<uint32_t>{42},
          "capacity failure prevented subsequent service");
}

// A request that fails in an admission pass stays listed until the pass
// ends; a sibling admitted after it in the same pass plans no junction for
// it.
void testSiblingFailedInTheAdmissionPassGetsNoJunction() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({}, cache, executor, events);
  guardReleases(storage, engine);
  // 64 shared tokens, then 33 of each one's own.
  std::vector<uint32_t> prompt(97);
  std::iota(prompt.begin(), prompt.end(), 1);
  // Alone, the first cannot get its state: it fails at once.
  executor.deniedBegins = 1;
  engine.submit(request(1, prompt));
  std::fill(prompt.begin() + 64, prompt.end(), 7);
  engine.submit(request(2, prompt));
  static_cast<void>(engine.tick(1));
  require(events.capacityExhaustedCount == 1 && events.startIds == std::vector<uint64_t>{2},
          "the two requests were not admitted in one pass");
  runUntilIdle(engine);
  const auto &boundaries = executor.plans.at(2).boundaries;
  require(std::none_of(boundaries.begin(), boundaries.end(),
                       [](const DraftBoundaryPlan &boundary) {
                         return boundary.boundary == 64;
                       }) &&
              engine.snapshot().junctionMaterializations == 0 && events.completedCount == 1,
          "a sibling planned a junction for a request that failed before it");
}

// A cold request publishes its replay state; once that state is gone, the
// next request over the same prompt rebuilds it where the KV still matches,
// as a replay state, and the one after resumes from it.
void testColdPublishesReplayStateAndRebuildsALostOne() {
  test::TestKvStorage storage(32, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({.maxContext = 102400}, resources, executor, events);
  guardReleases(storage, engine);
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
  require(events.starts.size() == 1 && events.starts[0] == 0,
          "cold request reported a cache hit");

  require(resources.reclaimStateForLane(ReclaimClass::InUse).madeProgress,
          "test could not remove the latest replay state");
  engine.submit(request(2, prompt));
  runUntilIdle(engine);
  require(executor.restored == 0 && executor.prefillRows == 130 &&
              executor.snapshots == 2 &&
              engine.snapshot().replayStatePublications == 2 &&
              engine.snapshot().junctionMaterializations == 0,
          "second request did not rebuild the lost replay state");
  require(events.starts.size() == 2 &&
              events.starts[1] == 0,
          "KV-only replay was reported as a state hit");

  engine.submit(request(3, prompt));
  runUntilIdle(engine);
  require(executor.restored == 64 && executor.prefillRows == 131,
          "cache hit did not replay exactly one real input token");
  require(events.starts.size() == 3 && events.starts[2] == 64,
          "state-backed hit accounting is wrong");
  require(events.completedCount == 3 && events.failedCount == 0 &&
              events.emitted == 3,
          "request lifecycle did not complete cleanly");
}

void testConcurrentDuplicateStateSkipsSnapshotCapture() {
  test::TestKvStorage storage(32, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  std::vector<uint32_t> prompt(65);
  for (uint32_t index = 0; index < prompt.size(); ++index)
    prompt[index] = index + 1;

  engine.submit(request(101, prompt));
  engine.submit(request(102, prompt));
  runUntilIdle(engine);

  // The second request restores the first publication without reserving a
  // redundant lane or capturing the same state again.
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

// A prompt's replay state is its last whole page before its generation
// prompt, or before its last token when that is unknown.
void testReplayStateEndsBeforeTheGenerationPrompt() {
  test::TestKvStorage storage(32, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  struct Case {
    uint32_t tokens;
    uint32_t generationPromptTokens;
    uint32_t replayBoundary;
  };
  uint64_t id = 0;
  for (const Case &test : {Case{97, 0, 96}, Case{97, 7, 64}, Case{103, 7, 96},
                           Case{33, 7, 0}}) {
    std::vector<uint32_t> prompt(test.tokens);
    std::iota(prompt.begin(), prompt.end(), static_cast<uint32_t>(++id * 1000));
    EngineRequest value = request(id, prompt);
    value.generationPromptTokens = test.generationPromptTokens;
    engine.submit(std::move(value));
    runUntilIdle(engine);
    require(resources.lookup(prompt, {}).resumeBoundary() == test.replayBoundary,
            "the replay state did not end before the generation prompt");
  }
  require(executor.snapshots == 3,
          "a generation prompt within the first page left a replay state");
}

// A next turn that renders the generation prompt differently diverges inside
// it (at N-4, as Qwen3.6 without reasoning does). Only a replay state before
// the generation prompt serves it.
void testFollowUpResumesBeforeTheGenerationPrompt() {
  const auto followUp = [](uint32_t generationPromptTokens) {
    test::TestKvStorage storage(32, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor;
    Events events;
    engine::Engine engine({}, resources, executor, events);
    guardReleases(storage, engine);
    std::vector<uint32_t> prompt(97);
    std::iota(prompt.begin(), prompt.end(), 1);
    EngineRequest turn = request(1, prompt);
    turn.generationPromptTokens = generationPromptTokens;
    engine.submit(std::move(turn));
    runUntilIdle(engine);
    prompt.resize(93);
    prompt.resize(133, 500);
    engine.submit(request(2, prompt));
    runUntilIdle(engine);
    return std::pair{executor.plans.at(1), events.starts.at(1)};
  };
  const auto [plan, start] = followUp(7);
  require(plan.boundaries.size() == 2 && plan.boundaries[0].boundary == 64 &&
              plan.boundaries[1].boundary == 97,
          "the replay state was not planned before the generation prompt");
  require(start == 64,
          "the follow-up did not resume before the generation prompt");
  require(followUp(0).second == 0,
          "a replay state past the divergence served the follow-up");
}

// An identical retry, in sequence or concurrently, resumes from the replay
// state and publishes no second state inside the generation prompt.
void testRetryPublishesNoStateInsideTheGenerationPrompt() {
  for (bool concurrent : {false, true}) {
    test::TestKvStorage storage(32, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor;
    Events events;
    engine::Engine engine({}, resources, executor, events);
    guardReleases(storage, engine);
    std::vector<uint32_t> prompt(97);
    std::iota(prompt.begin(), prompt.end(), 1);
    for (uint64_t id : {1, 2}) {
      EngineRequest value = request(id, prompt);
      value.generationPromptTokens = 7;
      engine.submit(std::move(value));
      if (!concurrent)
        runUntilIdle(engine);
    }
    runUntilIdle(engine);
    const auto snapshot = engine.snapshot();
    require(executor.snapshotAttempts == 1 &&
                snapshot.junctionMaterializations == 0 &&
                snapshot.resources.stateCache.entries == 1 &&
                events.starts.at(1) == 64,
            "a retry published a state inside the generation prompt");
  }
}

// A shared junction serves the waiter's next turn too, so it also ends before
// the waiter's generation prompt.
void testSharedJunctionEndsBeforeTheGenerationPrompt() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  std::vector<uint32_t> prompt(200);
  std::iota(prompt.begin(), prompt.end(), 1);
  EngineRequest producer = request(1, prompt);
  // Admitted first, it plans the junction for the waiter's whole prompt.
  producer.priority = RequestPriority::Foreground;
  producer.generationPromptTokens = 7;
  engine.submit(std::move(producer));
  prompt.resize(97);
  EngineRequest waiter = request(2, prompt);
  waiter.generationPromptTokens = 7;
  engine.submit(std::move(waiter));
  runUntilIdle(engine);
  prompt.resize(93);
  prompt.resize(133, 500);
  engine.submit(request(3, prompt));
  runUntilIdle(engine);
  require(events.starts.size() == 3 && events.starts[1] == 64 &&
              events.starts[2] == 64,
          "the shared junction was not the waiter's reusable state");
}

void testImageSpansKeyPrefixIdentity() {
  test::TestKvStorage storage(32, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
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
  require(events.starts.size() == 1 && events.starts[0] == 0,
          "image producer unexpectedly hit the cache");
  engine.submit(withImage(2, 0x2222));
  runUntilIdle(engine);
  require(events.starts.size() == 2 && events.starts[1] == 0,
          "a different image falsely matched the cached prefix");
  engine.submit(withImage(3, 0x1111));
  runUntilIdle(engine);
  require(events.starts.size() == 3 && events.starts[2] == 64,
          "an identical image did not reuse the cached prefix");
  engine.submit(request(4, prompt));
  runUntilIdle(engine);
  require(events.starts.size() == 4 && events.starts[3] == 0,
          "a text-only prompt matched an image-keyed prefix");
  require(events.completedCount == 4 && events.failedCount == 0,
          "image request lifecycle did not complete cleanly");
}

// Two token-identical prompts share their prefix up to the block that holds
// the start of the first image on which they differ: the later one waits for
// the junction there and resumes from it.
void testSharedPrefillBoundaryStopsAtTheFirstDifferentImage() {
  const ImageSpan a{40, 16, 8, 8, 0x1111, 0x2222};
  const ImageSpan b{200, 16, 8, 8, 0x3333, 0x4444};
  const ImageSpan c{200, 16, 8, 8, 0x5555, 0x6666};
  ImageSpan otherA = a;
  otherA.digestLo ^= 1;
  struct Shape {
    std::vector<ImageSpan> producer;
    std::vector<ImageSpan> waiter;
    uint32_t shared = 0;
  };
  for (const Shape &shape : {Shape{{a, b}, {a, c}, 192}, Shape{{a}, {otherA}, 32}}) {
    test::TestKvStorage storage(64, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor model;
    Events events;
    engine::Engine engine({}, cache, model, events);
    guardReleases(storage, engine);
    const auto withImages = [](uint64_t id, const std::vector<ImageSpan> &images) {
      EngineRequest result = request(id, std::vector<uint32_t>(289, 248056));
      result.images = images;
      for (const ImageSpan &image : images)
        result.imagePixels.resize(result.imagePixels.size() + image.pixelBytes(), 1);
      return result;
    };
    engine.submit(withImages(1, shape.producer));
    engine.submit(withImages(2, shape.waiter));
    static_cast<void>(engine.tick(0));
    require(model.requests.size() == 1 && engine.snapshot().scheduler.waitingPrefix == 1,
            "the later request did not wait for the shared prefix");
    runUntilIdle(engine);
    require(events.starts.size() == 2 &&
                events.starts[1] == shape.shared &&
                model.prefillRows == 289 + 289 - shape.shared && events.completedCount == 2,
            "the shared prefix did not end at the block of the first different image");
  }
}

// One prefill chunk that every caller's prompt shares, then 65 tokens of
// `token`: the replay state lands at 2112. A prompt branching off a cached
// one at the end of the chunk plans a junction there, a draft window past
// the start.
std::vector<uint32_t> branchPrompt(uint32_t token) {
  std::vector<uint32_t> prompt(model::ExecutionLimits::prefillTokenBudget);
  std::iota(prompt.begin(), prompt.end(), 1);
  prompt.resize(prompt.size() + 65, token);
  return prompt;
}

// A request branching off a cached conversation that goes on to a state of
// its own publishes the junction at the branch point, then its own latest
// replay state.
void testOneRequestPublishesJunctionAndLatestReplayState() {
  test::TestKvStorage storage(256, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);

  engine.submit(request(5, branchPrompt(777)));
  runUntilIdle(engine);
  engine.submit(request(6, branchPrompt(888)));
  runUntilIdle(engine);

  const DraftContextPlan &plan = executor.plans.at(6);
  const auto snapshot = engine.snapshot();
  require(plan.boundaries.size() == 3 && plan.boundaries[0].boundary == 2048 &&
              plan.boundaries[1].boundary == 2112 &&
              plan.boundaries[2].boundary == 2113,
          "junction, latest replay state, and active end were not ordered");
  require(executor.snapshots == 3 && snapshot.junctionMaterializations == 1 &&
              snapshot.replayStatePublications == 2 &&
              snapshot.resources.stateCache.entries == 3,
          "one request did not retain both sparse composite states");
}

// A junction costs a snapshot, a command split and a draft window of
// capture, so it is planned only where it saves a later request at least a
// draft window of prefill: at a branch point that far past the state the
// request resumes from.
void testLazyJunctionNeedsADraftWindowOfGain() {
  test::TestKvStorage storage(256, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  std::vector<uint32_t> shared(2112);
  std::iota(shared.begin(), shared.end(), 1);
  // The shared prefix, then 161 tokens of the conversation's own from
  // `first` on: its replay state lands at 2272.
  const auto conversation = [&](uint32_t first) {
    std::vector<uint32_t> prompt = shared;
    prompt.resize(shared.size() + 161);
    std::iota(prompt.begin() + shared.size(), prompt.end(), first);
    return prompt;
  };
  const std::vector<uint32_t> cached = conversation(10'000);
  engine.submit(request(1, cached));
  runUntilIdle(engine);
  engine.submit(request(2, conversation(20'000)));
  runUntilIdle(engine);
  require(engine.snapshot().junctionMaterializations == 1,
          "a branch off the cached conversation did not publish its junction");
  const uint32_t restored = executor.restored;
  engine.submit(request(3, conversation(30'000)));
  runUntilIdle(engine);
  require(executor.restored == restored + 2112 &&
              events.starts.back() == 2112,
          "a third branch did not resume from the junction");
  // This one follows the cached conversation 96 tokens past the junction,
  // where that one goes on to its state, and then branches.
  std::vector<uint32_t> follower = cached;
  std::iota(follower.begin() + shared.size() + 96, follower.end(), 40'000);
  engine.submit(request(4, follower));
  runUntilIdle(engine);
  const auto &boundaries = executor.plans.at(4).boundaries;
  require(events.starts.back() == 2112 &&
              std::none_of(boundaries.begin(), boundaries.end(),
                           [](const DraftBoundaryPlan &boundary) {
                             return boundary.boundary == 2112 + 96;
                           }) &&
              engine.snapshot().junctionMaterializations == 1,
          "a branch point 96 tokens past the resumed state planned a junction");
}

// A denied snapshot at the latest replay boundary recycles the least recently
// used cached state, which is an older unrelated state, not the junction state
// this same request published one command earlier.
void testLatestReplayDenialRecyclesOlderStateNotTheJunction() {
  test::TestKvStorage storage(256, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  bool paused = false;
  EngineConfig config;
  config.growthPaused = [&] { return paused; };
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);

  engine.submit(request(70, std::vector<uint32_t>(65, 7000)));
  runUntilIdle(engine);
  engine.submit(request(7, branchPrompt(777)));
  runUntilIdle(engine);
  const auto older = resources.snapshot();
  require(older.stateCache.entries == 2 && older.stateCache.evictions == 0,
          "older unrelated state was not left in the cache");

  engine.submit(request(8, branchPrompt(888)));
  require(engine.tick(1) && engine.tick(2),
          "request did not publish its junction state");
  require(engine.snapshot().junctionMaterializations == 1 &&
              resources.snapshot().stateCache.entries == 3,
          "lazy junction was not materialized in the first command");
  // The pages this request held for the cached prefix are free; while the
  // engine may not grow, a snapshot takes none of them and recycles a state.
  paused = true;
  executor.deniedSnapshots = 1;
  runUntilIdle(engine);
  paused = false;

  const DraftContextPlan &plan = executor.plans.at(8);
  const auto snapshot = engine.snapshot();
  require(plan.boundaries.size() == 3 && plan.boundaries[0].boundary == 2048 &&
              plan.boundaries[1].boundary == 2112 &&
              plan.boundaries[2].boundary == 2113,
          "boundaries were not armed without reservation");
  require(executor.deniedSnapshots == 0 &&
              snapshot.recycledStatePublications == 1 &&
              snapshot.replayStatePublications == 3 &&
              snapshot.replayStatePublicationFailures == 0 &&
              snapshot.resources.stateCache.evictions == 1 &&
              snapshot.resources.stateCache.entries == 3,
          "latest-state denial did not recycle exactly one older state");

  // The junction state survived the recycle: another branch resumes from
  // it, while the recycled prompt is back to KV without a state and
  // publishes its replay state again.
  const uint32_t restored = executor.restored;
  engine.submit(request(9, branchPrompt(999)));
  runUntilIdle(engine);
  require(executor.restored == restored + 2048 &&
              events.starts.back() == 2048,
          "latest-state denial discarded the independent junction state");
  engine.submit(request(71, std::vector<uint32_t>(65, 7000)));
  runUntilIdle(engine);
  require(events.starts.back() == 0 &&
              engine.snapshot().replayStatePublications == 5 &&
              engine.snapshot().junctionMaterializations == 1,
          "recycled state was not the older unrelated one");
}

void testCancellationAfterJunctionDiscardsLaterState() {
  test::TestKvStorage storage(256, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);

  engine.submit(request(9, branchPrompt(777)));
  runUntilIdle(engine);
  engine.submit(request(10, branchPrompt(999)));
  require(engine.tick(1) && engine.tick(2),
          "request did not publish its first sparse state");
  const auto published = engine.snapshot();
  require(published.resources.stateCache.entries == 2 &&
              published.junctionMaterializations == 1 &&
              executor.snapshotAttempts == 2,
          "junction publication did not land in the first command");
  engine.cancel(10);
  runUntilIdle(engine);
  // The armed 2112 boundary is dropped with the lane: no snapshot, no
  // failure counted, and the junction state stays cached.
  const auto cancelled = engine.snapshot();
  require(executor.snapshotAttempts == 2 &&
              cancelled.resources.stateCache.entries == 2 &&
              cancelled.resources.stateCache.publications == 2 &&
              cancelled.replayStatePublications == 1 &&
              cancelled.replayStatePublicationFailures == 0 &&
              cancelled.cancelled == 1 && executor.requests.empty() &&
              cancelled.resources.activeRequests == 0,
          "cancellation leaked or removed the wrong sparse state");
}

// A publication's expected refusals are values (a null snapshot, a refused
// disk write). An exception is a broken invariant: it ends the engine
// instead of counting as a failed publication.
void testPublicationInvariantFailureIsFatal() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  executor.snapshotObserver = [] { throw std::logic_error("snapshot invariant"); };
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  engine.submit(request(3, std::vector<uint32_t>(65, 7)));
  bool threw = false;
  try {
    for (double now = 1; now < 32; ++now)
      static_cast<void>(engine.tick(now));
  } catch (const std::logic_error &error) {
    threw = std::string(error.what()) == "snapshot invariant";
  }
  require(threw && engine.snapshot().replayStatePublicationFailures == 0,
          "a broken publication invariant was counted as a failed publication");
}

// Nothing is reserved at admission, so a boundary the model cannot snapshot
// costs exactly one attempt: the failure is counted under its purpose, the
// request is served normally, and with an empty cache nothing is recycled.
void testDeniedSnapshotCostsOnlyThatAttempt() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  executor.deniedSnapshots = 100;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  std::vector<uint32_t> prompt(65, 7);

  engine.submit(request(3, prompt));
  runUntilIdle(engine);
  const DraftContextPlan &plan = executor.plans.at(3);
  require(plan.boundaries.size() == 2 && plan.boundaries[0].boundary == 64 &&
              plan.boundaries[1].boundary == 65 &&
              draftContextRows(plan) == prompt.size(),
          "replay boundary was not armed without reservation");
  require(executor.snapshotAttempts == 1 && executor.snapshots == 0 &&
              engine.snapshot().replayStatePublicationFailures == 1 &&
              engine.snapshot().junctionMaterializationFailures == 0 &&
              engine.snapshot().recycledStatePublications == 0 &&
              resources.snapshot().stateCache.evictions == 0 &&
              events.completedCount == 1,
          "latest replay-state denial was counted incorrectly");

  // The first pass left a usable KV chain but no state. The second request
  // plans its replay state there again; its denied snapshot is counted as a
  // replay-state failure while the request replays through the KV prefix
  // normally.
  engine.submit(request(4, prompt));
  runUntilIdle(engine);
  require(executor.snapshotAttempts == 2 && executor.snapshots == 0 &&
              engine.snapshot().junctionMaterializationFailures == 0 &&
              engine.snapshot().replayStatePublicationFailures == 2 &&
              events.starts.back() == 0 &&
              executor.prefillRows == 130 && events.completedCount == 2 &&
              events.failedCount == 0,
          "denied replay state failed the request or was not counted");

  // The denial is transient: the next lane over the same prefix lands it.
  executor.deniedSnapshots = 0;
  engine.submit(request(5, prompt));
  runUntilIdle(engine);
  require(executor.snapshots == 1 &&
              engine.snapshot().replayStatePublications == 1 &&
              resources.snapshot().stateCache.entries == 1,
          "replay state was not published once snapshots were possible");
}

// A snapshot the model denies once at a replay boundary lands by recycling
// the least recently used cached state: one eviction, the same number of
// entries, and the recycled slot now holds the new lane's state.
void testDeniedSnapshotRecyclesLruStateAndRetries() {
  test::TestKvStorage storage(32, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
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
  // state hit, while the older prompt is back to KV without a state.
  engine.submit(request(3, newer));
  runUntilIdle(engine);
  require(executor.restored == 64 && events.starts.back() == 64,
          "recycled slot does not hold the new lane's state");
  engine.submit(request(4, older));
  runUntilIdle(engine);
  require(executor.restored == 64 &&
              events.starts.back() == 0 &&
              engine.snapshot().replayStatePublications == 3,
          "recycled state was not the least recently used one");
}

// When the retry after recycling is denied as well, the boundary fails and
// the engine has paid exactly one cached state for it. Persistent denial
// never drains the rest of the cache.
void testPersistentSnapshotDenialRecyclesAtMostOneState() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
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
  test::TestKvStorage storage(256, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
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
              !executor.plans.at(11).restoresDraftState,
          "long suffix copied a draft ring that its final window overwrites");
}

// A lane cancelled while the command that ends at its armed boundary is in
// flight publishes nothing when that command drains: no snapshot is taken,
// no failure is counted, and the blocks committed earlier stay cached.
void testCancellationInFlightAtBoundaryPublishesNoState() {
  test::TestKvStorage storage(256, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  std::vector<uint32_t> prompt(4097, 11);

  engine.submit(request(4, prompt));
  require(engine.tick(1) && engine.tick(2),
          "long request did not complete its first prefill command");
  require(resources.snapshot().pool.pagesPrefix == 64 &&
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
  require(snapshot.resources.pool.pagesPrefix == 64 &&
              snapshot.resources.pool.pagesActive == 0 &&
              snapshot.resources.activeRequests == 0 &&
              executor.requests.empty(),
          "cancellation leaked active resources or dropped committed KV");
}

void testLaneStateGrowthReclaimsCachedStateAndRetries() {
  for (bool hostPressure : {false, true}) {
    test::TestKvStorage storage(16, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(1);
    Events events;
    EngineConfig config;
    config.growthPaused = [] { return false; };
    engine::Engine engine(config, resources, executor, events);
    guardReleases(storage, engine);

    engine.submit(request(20, std::vector<uint32_t>(65, 7)));
    runUntilIdle(engine);
    const auto cached = resources.snapshot();
    require(cached.stateCache.entries == 1,
            "state-reclaim setup did not publish a composite state");

    if (hostPressure) {
      storage.growthBlocked = true;
      executor.kvGrowthBlocked = &storage.growthBlocked;
      executor.beginGrowthBlocked = [&] { return storage.growthBlocked; };
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
            "lane state growth did not reclaim memory and retry");
    if (hostPressure) {
      require(executor.reclaimedIdleStateBytes == 350'224'384 &&
                  !storage.growthBlocked &&
                  after.stateCache.entries == cached.stateCache.entries &&
                  after.stateCache.evictions == cached.stateCache.evictions,
              "host-pressure state admission evicted cache or ignored idle memory");
    } else {
      require(after.stateCache.entries == 0,
              "lane state growth did not reclaim cached state");
    }
  }
}

void testKvGrowthReclaimsIdleStateBeforeCache() {
  for (auto failure : {metal::AllocationFailure::EngineBudget,
                       metal::AllocationFailure::HostPressure}) {
    test::TestKvStorage storage(12, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(1);
    Events events;
    EngineConfig config;
    config.growthPaused = [] { return false; };
    engine::Engine engine(config, resources, executor, events);
    guardReleases(storage, engine);

    engine.submit(request(25, std::vector<uint32_t>(65, 25)));
    runUntilIdle(engine);
    const auto cached = resources.snapshot();
    require(cached.stateCache.entries == 1,
            "idle-state reclaim setup did not retain cache state");

    storage.growthBlocked = true;
    storage.allocationFailure = failure;
    executor.kvGrowthBlocked = &storage.growthBlocked;
    executor.unblockGrowthOnSuspend = false;
    executor.reclaimableIdleStateBytes = 350'224'384;
    engine.submit(request(26, std::vector<uint32_t>(161, 26)));
    runUntilIdle(engine);

    const auto after = resources.snapshot();
    require(executor.reclaimedIdleStateBytes == 350'224'384 &&
                !storage.growthBlocked && events.completedCount == 2 &&
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
// KV growth denial the first lane meets is answered by reclaiming
// idle model state, never by dropping any lane's armed boundary: both replay
// states land.
void testKvGrowthDenialKeepsEveryLaneReplayState() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  storage.growthBlocked = true;
  executor.kvGrowthBlocked = &storage.growthBlocked;
  executor.reclaimableIdleStateBytes = 1U << 20;

  engine.submit(request(300, std::vector<uint32_t>(161, 300)));
  engine.submit(request(301, std::vector<uint32_t>(161, 301)));
  runUntilIdle(engine);
  const auto after = engine.snapshot();
  require(executor.reclaimedIdleStateBytes == (1U << 20) &&
              !storage.growthBlocked && executor.suspensions == 0 &&
              events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0,
          "a growth denial suspended or failed a lane");
  require(!executor.prefillWidths.empty() && executor.prefillWidths[0] == 2,
          "both lanes did not prefill to their boundaries in one command");
  require(executor.snapshots == 2 && after.replayStatePublications == 2 &&
              after.replayStatePublicationFailures == 0 &&
              after.recycledStatePublications == 0 &&
              after.resources.stateCache.entries == 2 &&
              after.resources.stateCache.evictions == 0,
          "KV growth denial cost a lane its replay state");
}

// A pressure pass without a byte target keeps the caches the model can
// rebuild and returns only its idle buffers; a targeted pass takes the
// caches one at a time and stops once its target is met.
void testPressurePassKeepsCachesWithoutATarget() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  executor.reclaimableIdleStateBytes = 64;
  executor.cacheUnits = {100, 100, 100};
  static_cast<void>(engine.reclaimMemory({}));
  require(executor.reclaimedIdleStateBytes == 64 && executor.cacheReclaims == 0 &&
              executor.cacheUnits.size() == 3,
          "a pass without a target took the model's caches");
  const MemoryReclaimResult targeted = engine.reclaimMemory({.targetBytes = 150});
  require(executor.cacheReclaims == 2 && executor.cacheUnits.size() == 1 &&
              targeted.releasedBytes >= 200,
          "a targeted pass did not stop taking caches at its target");
}

// Rows only a cached state held become a cache unit once a pass evicts that
// state, and a pass whose target is still unmet takes them as well.
void testPressurePassTakesTheRowsItsEvictionsLeave() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  Executor executor(1);
  engine::Cache resources(pool, nullptr, nullptr);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  executor.stateHeldRows = 100;
  engine.submit(request(29, std::vector<uint32_t>(65, 29)));
  runUntilIdle(engine);
  require(resources.snapshot().stateCache.entries == 1 && executor.cacheUnits.empty(),
          "setup did not cache a state holding rows");
  const MemoryReclaimResult critical = engine.reclaimMemory({.critical = true});
  require(resources.snapshot().stateCache.entries == 0 && executor.cacheUnits.empty() &&
              critical.releasedBytes >= 164,
          "a critical pass left the rows of the state it evicted");
}

void testPressureReclaimRespectsStateLifetimes() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);

  engine.submit(request(27, std::vector<uint32_t>(65, 27)));
  runUntilIdle(engine);
  const auto cached = resources.snapshot();
  require(cached.stateCache.entries == 1 && cached.pool.pagesPrefix == 2,
          "pressure setup did not retain KV and composite state");

  // A shrink that nothing is waiting for stops at the resume point. Freeing
  // its cell gains the host a little; the next request pays a full replay.
  static_cast<void>(engine.reclaimMemory({.targetBytes = 64, .keepResumePoint = true}));
  require(resources.snapshot().stateCache.entries == 1,
          "a speculative shrink discarded the only resume point");

  executor.reclaimableIdleStateBytes = 128;
  const MemoryReclaimResult first = engine.reclaimMemory({.targetBytes = 64});
  require(first.releasedBytes >= 128 && first.outcome == ReclaimOutcome::Met &&
              executor.reclaimedIdleStateBytes == 128,
          "pressure reclaim did not release idle state buffers first");
  require(
      resources.snapshot().stateCache.entries == 1,
      "pressure reclaim evicted a cached state before satisfying its target");

  // Drain the extents the test starts with but does not use, then prove cached
  // state is the next lifecycle selected while its parent KV remains usable.
  require(engine.reclaimMemory({}).outcome == ReclaimOutcome::Untargeted,
          "a pass without a target reported one");
  const MemoryReclaimResult state = engine.reclaimMemory({.targetBytes = 64});
  require(state.releasedBytes >= 64 && state.outcome == ReclaimOutcome::Met,
          "pressure reclaim did not release immutable cached state");
  const auto stateEvicted = resources.snapshot();
  require(stateEvicted.stateCache.entries == 0 &&
              stateEvicted.pool.pagesPrefix == 2,
          "cached-state eviction incorrectly removed target KV");

  require(engine.reclaimMemory({.critical = true}).outcome == ReclaimOutcome::Exhausted,
          "evicting everything left something to reclaim");
  const auto critical = resources.snapshot();
  require(critical.stateCache.entries == 0 && critical.pool.pagesPrefix == 0 &&
              critical.pool.allocatedBytes == 0,
          "critical pressure left evictable cached state or KV extents");
  const MemoryReclaimResult empty = engine.reclaimMemory({.targetBytes = 64});
  require(!empty.releasedBytes && empty.outcome == ReclaimOutcome::Exhausted,
          "an empty cache did not report reclaim exhausted");
}

// Pressure short of critical takes cached state and KV but keeps what a
// request starts from without allocating: a lane's pooled buffers and one
// empty extent, for the request that arrives while the host is short.
// Critical pressure takes those too.
void testWarningReclaimKeepsTheServingFootprint() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);

  engine.submit(request(28, std::vector<uint32_t>(65, 28)));
  runUntilIdle(engine);
  constexpr uint64_t extentBytes = 4 * 4096;
  require(resources.snapshot().pool.allocatedBytes == extentBytes,
          "footprint setup did not keep one allocated extent of cached KV");

  static_cast<void>(
      engine.reclaimMemory({.targetBytes = std::numeric_limits<uint64_t>::max()}));
  const auto warning = resources.snapshot();
  require(executor.keptLane && warning.stateCache.entries == 0 &&
              warning.pool.pagesPrefix == 0 &&
              warning.pool.allocatedBytes == extentBytes,
          "warning pressure did not keep only the serving footprint");

  static_cast<void>(engine.reclaimMemory({.critical = true}));
  require(!executor.keptLane &&
              resources.snapshot().pool.allocatedBytes == 0,
          "critical pressure kept the serving footprint");
}

// A pass counts only memory that leaves the engine. A warning pass that
// evicts a cached state while the lane's pooled buffers are short gains the
// host nothing: the state's buffers refill them. The pass goes on to the
// cached KV and releases the extent that empties, keeping the empty one the
// next request starts from.
void testWarningReclaimCountsOnlyWhatReachesTheHost() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  engine.submit(request(1, std::vector<uint32_t>(129, 1)));
  runUntilIdle(engine);
  constexpr uint64_t extentBytes = 4 * 4096;
  require(resources.snapshot().stateCache.bytes == 64 &&
              resources.snapshot().pool.allocatedBytes == 2 * extentBytes &&
              resources.snapshot().pool.reclaimableBytes == extentBytes,
          "fixture did not cache a state and an extent of KV beside the runway");
  // The lane's pooled buffers went back to the host earlier.
  executor.laneFootprintBytes = 64;
  const MemoryReclaimResult result = engine.reclaimMemory({.targetBytes = 64});
  require(result.releasedBytes == extentBytes && result.outcome == ReclaimOutcome::Met &&
              executor.pooledLaneBytes == 64 &&
              resources.snapshot().stateCache.entries == 0 &&
              resources.snapshot().pool.allocatedBytes == extentBytes,
          "a warning pass counted a state whose buffers stayed in the engine");
}

// Memory a pass frees but keeps still wakes the requests waiting for memory.
// A start the host refused can start from the lane's pooled buffers; a
// warning pass that refills them from an evicted state releases nothing,
// and the start retries at once rather than after its backoff.
void testWarningReclaimWakesARefusedStart() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  engine.submit(request(1, std::vector<uint32_t>(65, 1)));
  runUntilIdle(engine);
  require(resources.snapshot().stateCache.bytes == 64, "fixture did not cache a state");
  executor.laneFootprintBytes = 64;
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  executor.beginGrowthBlocked = [&] { return executor.pooledLaneBytes == 0; };
  engine.submit(request(2, std::vector<uint32_t>(65, 2)));
  static_cast<void>(engine.tick(100));
  require(engine.resourceWaitSnapshot(100).memory == 1 &&
              engine.nextWakeupMilliseconds() == 200.0,
          "the host did not refuse the start");
  const MemoryReclaimResult result = engine.reclaimMemory({.targetBytes = 64});
  require(!result.releasedBytes && executor.pooledLaneBytes == 64 &&
              resources.snapshot().stateCache.entries == 0 &&
              engine.nextWakeupMilliseconds() == 0.0,
          "a pass that refilled the lane's buffers did not wake the refused start");
  for (double now = 101; now < 120 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && events.completedCount == 2 && events.failedCount == 0,
          "the start did not run from the refilled buffers");
}

// A KV chain gives up one leaf at a time, each after its copy is written. A
// pass reports that transfers hold back the rest of its target, and passes
// with that rest (MemoryPressurePolicy continues it) take the chain as the
// copies land, each pass first collecting the copies that landed since the
// last, each leaf after its child, and stop at the target. The first extent
// the copies empty stays allocated as the runway.
void testPressureReclaimFollowsTheChain() {
  test::TestKvStorage storage(4, 100, 1);
  KvPool pool(storage, 4);
  test::TestKvTier tier;
  engine::Cache cache(pool, &tier, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({}, cache, executor, events);
  guardReleases(storage, engine);
  std::vector<uint32_t> prompt(129);
  std::iota(prompt.begin(), prompt.end(), 1000);
  cache.beginRequest(1);
  require(cache.ensureTokens(1, 128).granted(), "fixture KV failed");
  const uint64_t leaf = test::publishBlocks(cache, 1, prompt, 128);
  auto transfer = std::make_shared<OffloadControl>();
  transfer->ready = true;
  cache.publishCompositeState(
      leaf, std::make_shared<OffloadState>(transfer, executor.evictedStateBytes), false);
  cache.endRequest(1);

  const auto reclaim = [&](uint64_t target) {
    return engine.reclaimMemory({.targetBytes = target});
  };
  // The state's buffers, which the model returns once the cache has written
  // the state, then the chain's leaf, whose parent waits for its copy.
  MemoryReclaimResult result = reclaim(264);
  require(result.releasedBytes == 64 && result.outcome == ReclaimOutcome::Pending &&
              tier.demotions == 1,
          "the leaf's copy did not hold back the rest of the target");
  const uint64_t target = 264 - result.releasedBytes;
  tier.complete();
  result = reclaim(target);
  require(result.releasedBytes == 0 && result.outcome == ReclaimOutcome::Pending &&
              tier.demotions == 2,
          "the first extent the copies emptied did not stay as the runway");
  tier.complete();
  result = reclaim(target);
  require(result.releasedBytes == 100 && result.outcome == ReclaimOutcome::Met &&
              tier.demotions == 3,
          "the rest of the target did not take the chain leaf by leaf");
  tier.complete();
  result = engine.reclaimMemory({});
  require(result.releasedBytes == 100 && result.outcome == ReclaimOutcome::Untargeted &&
              tier.demotions == 3 && pool.snapshot().pagesAllocated == 2,
          "reclaim went past its target");
}

// A request starts only in a free lane. While every lane is resident, a
// waiting request is recorded as a concurrency wait without another cache
// probe or admission attempt, and, as after a failed attempt, a memory wait
// it was in no longer counts against the resource wait limit.
void testFullLanesSkipAdmissionAttempts() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  // The first pass starts the three short prompts and tries request 5, the
  // fourth of its batch, which meets pressure. Request 4 arrived before it
  // with a long prompt, so it is not held back and takes the last lane.
  executor.beginGrowthBlocked = [&] { return executor.beginAttempts == 4; };
  Events events;
  const test::ScopedTestConfig seam({.resourceWaitTimeoutMilliseconds = 1000.0});
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  const auto submit = [&](uint64_t id, uint32_t promptTokens) {
    auto value = request(id, std::vector<uint32_t>(promptTokens, id));
    value.maxNewTokens = 100'000;
    value.deadlineMilliseconds = 100'000;
    engine.submit(std::move(value));
  };
  for (uint64_t id = 1; id <= 3; ++id)
    submit(id, 33);
  submit(4, 129);
  submit(5, 64);
  require(engine.tick(1) && engine.resourceWaitSnapshot(1).memory == 1 &&
              executor.lastBeginId == 5,
          "fixture did not leave one request waiting for memory");
  for (double now = 2; now <= 6; ++now)
    static_cast<void>(engine.tick(now));
  require(executor.requests.size() == 4 && executor.requests.contains(4) &&
              executor.beginAttempts == 5,
          "fixture did not make every lane resident");

  for (double now : {200.0, 201.0, 1200.0, 1201.0})
    static_cast<void>(engine.tick(now));
  const auto waiting = engine.resourceWaitSnapshot(1201);
  require(executor.beginAttempts == 5 && events.failedCount == 0 &&
              waiting.concurrency == 1 && waiting.memory == 0,
          "full lanes retried admission or kept the memory wait limit");
  engine.cancel(1);
  require(engine.tick(1202) && executor.beginAttempts == 6 &&
              events.startIds.back() == 5,
          "a released lane did not admit the waiting request");
  for (uint64_t id : {2, 3, 4, 5})
    engine.cancel(id);
  for (double now = 1203; now < 1220 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && resources.snapshot().activeRequests == 0,
          "full-lane fixture leaked its lanes");
}

void testConcurrencyLimitDoesNotEvictCache() {
  test::TestKvStorage storage(256, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);

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
          "lane saturation was mistaken for memory pressure");

  engine.cancel(23);
  engine.cancel(24);
  runUntilIdle(engine);
}

// A lane the host refuses even as the only one in service, as under critical
// pressure, waits for the host: one ordinary attempt and one as a request in
// service per retry, and nothing evicted for an allocator that refuses all
// the same.
void testHostPressureDoesNotDrainCacheOnStateAdmission() {
  test::TestKvStorage storage(32, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  MemoryPressure pressure = MemoryPressure::Normal;
  EngineConfig config;
  config.growthPaused = [&] { return pressure != MemoryPressure::Normal; };
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  engine.submit(request(200, std::vector<uint32_t>(65, 200)));
  runUntilIdle(engine);
  const auto cached = resources.snapshot();
  const uint64_t cacheHits = engine.snapshot().cacheHits;
  const uint64_t coldMisses = engine.snapshot().coldMisses;
  const uint32_t attempts = executor.beginAttempts;

  pressure = MemoryPressure::Critical;
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  executor.beginGrowthBlocked = [&] { return pressure != MemoryPressure::Normal; };
  engine.submit(request(201, {201}));
  static_cast<void>(engine.tick(20.0));
  static_cast<void>(engine.tick(50.0));
  require(executor.beginAttempts == attempts + 2 &&
              resources.snapshot().stateCache.entries ==
                  cached.stateCache.entries &&
              resources.snapshot().pool.pagesPrefix == cached.pool.pagesPrefix &&
              engine.snapshot().cacheHits == cacheHits &&
              engine.snapshot().coldMisses == coldMisses &&
              engine.snapshot().scheduler.waitingResources == 1,
          "paused state admission drained the cache or retried without backoff");

  pressure = MemoryPressure::Normal;
  for (double now = 120.0; now < 140.0 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && events.completedCount == 2 &&
              engine.snapshot().cacheHits == cacheHits &&
              engine.snapshot().coldMisses == coldMisses + 1 &&
              events.capacityExhaustedCount == 0 && events.failedCount == 0,
          "state admission did not recover after host pressure cleared");
}

// Host pressure holds back growth that no request in service needs, but
// landing a denied snapshot by recycling one cached state is memory-neutral
// and still happens: the state footprint does not grow, and KV is untouched.
void testHostPressureStillRecyclesLruStateForDeniedSnapshot() {
  test::TestKvStorage storage(32, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  MemoryPressure pressure = MemoryPressure::Normal;
  EngineConfig config;
  config.growthPaused = [&] { return pressure != MemoryPressure::Normal; };
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  engine.submit(request(210, std::vector<uint32_t>(65, 210)));
  runUntilIdle(engine);
  const auto cached = resources.snapshot();
  const uint32_t attempts = executor.snapshotAttempts;

  pressure = MemoryPressure::Warning;
  executor.deniedSnapshots = 1;
  engine.submit(request(211, std::vector<uint32_t>(65, 211)));
  runUntilIdle(engine);
  const auto after = engine.snapshot();
  require(idle(engine) && events.completedCount == 2 &&
              executor.snapshotAttempts == attempts + 2 &&
              after.recycledStatePublications == 1 &&
              after.replayStatePublications == 2 &&
              after.replayStatePublicationFailures == 0,
          "denied snapshot under host pressure was not landed by recycling");
  require(after.resources.stateCache.evictions ==
                  cached.stateCache.evictions + 1 &&
              after.resources.stateCache.entries == cached.stateCache.entries &&
              after.resources.stateCache.bytes == cached.stateCache.bytes &&
              after.resources.pool.pagesPrefix == cached.pool.pagesPrefix + 2,
          "recycling under host pressure grew state or drained KV cache");
}

// The governor's host pause as the engine meets it: KV growth and a lane's
// state are refused for the host unless the engine marks them as memory a
// request in service needs.
struct HostPause final {
  bool paused = false;
  bool serving = false;
  // The pages no request held each time the pause let a request grow.
  std::vector<uint32_t> reusableAtGrowth;

  void attach(EngineConfig &config, test::TestKvStorage &storage, KvPool &pool) {
    config.growthPaused = [this] { return paused; };
    config.serving = [this](bool value) { serving = value; };
    storage.allocationFailure = metal::AllocationFailure::HostPressure;
    storage.growthAllowed = [this, &pool](uint32_t) {
      if (paused && !serving)
        return false;
      if (paused) {
        const KvPoolSnapshot pages = pool.snapshot();
        reusableAtGrowth.push_back(pages.pagesAllocated - pages.pagesActive);
      }
      return true;
    };
  }
  void attach(Executor &executor) {
    executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
    executor.beginGrowthBlocked = [this] { return paused && !serving; };
  }
};

// Host pressure pauses growth, not a request in service. Short of pages, it
// first takes the cached pages no request holds; once none is left it grows
// as it would without the pause, and it is never suspended for the host.
void testRequestInServiceGrowsThroughTheHostPause() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  HostPause host;
  EngineConfig config;
  host.attach(config, storage, pool);
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  // A finished prompt leaves two cached blocks under a state.
  engine.submit(request(1, std::vector<uint32_t>(65, 1)));
  runUntilIdle(engine);
  require(resources.snapshot().pool.pagesPrefix == 2 &&
              resources.snapshot().stateCache.entries == 1,
          "host pause fixture did not cache the first prompt");
  executor.decodeFinishes = false;
  auto running = request(2, std::vector<uint32_t>(65, 2));
  running.maxNewTokens = 400;
  running.deadlineMilliseconds = 1'000'000;
  engine.submit(std::move(running));
  double now = 1;
  while (now < 100 && events.outputs[2].empty())
    static_cast<void>(engine.tick(now++));
  require(!events.outputs[2].empty(), "the request did not start decoding");

  host.paused = true;
  const uint32_t allocated = pool.snapshot().pagesAllocated;
  while (now < 2000 && events.completedCount < 2)
    static_cast<void>(engine.tick(now++));
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.outputs[2].size() == 400 && executor.suspensions == 0 &&
              engine.snapshot().resourceSuspensions == 0,
          "host pressure stopped a request in service");
  require(pool.snapshot().pagesAllocated > allocated && !host.reusableAtGrowth.empty() &&
              std::all_of(host.reusableAtGrowth.begin(), host.reusableAtGrowth.end(),
                          [](uint32_t pages) { return pages == 0; }),
          "the request grew while cached pages no request held could serve it");
  require(resources.lookup(std::vector<uint32_t>(65, 1), {}).kvBoundary == 0,
          "the paused request did not take the idle cached pages first");
}

// A lane short of pages under the pause keeps the pooled buffers the next
// lane starts from. Releasing them would not let it grow: the host refuses
// its ordinary attempt all the same, and it grows as a request in service.
void testPausedPageShortfallKeepsTheLaneRunway() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  HostPause host;
  EngineConfig config;
  host.attach(config, storage, pool);
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  engine.submit(request(1, std::vector<uint32_t>(65, 1)));
  runUntilIdle(engine);
  // The finished lane's buffers wait in the pool for the next one.
  executor.pooledLaneBytes = 4096;
  executor.decodeFinishes = false;
  auto running = request(2, std::vector<uint32_t>(65, 2));
  running.maxNewTokens = 400;
  running.deadlineMilliseconds = 1'000'000;
  engine.submit(std::move(running));
  double now = 1;
  while (now < 100 && events.outputs[2].empty())
    static_cast<void>(engine.tick(now++));
  require(!events.outputs[2].empty(), "the request did not start decoding");

  host.paused = true;
  while (now < 2000 && events.completedCount < 2)
    static_cast<void>(engine.tick(now++));
  require(events.completedCount == 2 && events.failedCount == 0 &&
              !host.reusableAtGrowth.empty(),
          "the request did not grow through the host pause");
  require(executor.pooledLaneBytes == 4096 && executor.keptLane,
          "a page shortfall under the pause released the next lane's buffers");
}

// With nothing in service, the first request starts through the pause: its
// lane and its pages grow. A request that arrives beside it waits for the
// host without evicting anything, and starts once the first has finished.
void testFirstRequestStartsThroughTheHostPause() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  HostPause host;
  host.paused = true;
  EngineConfig config;
  host.attach(config, storage, pool);
  host.attach(executor);
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  auto first = request(1, std::vector<uint32_t>(65, 1));
  first.maxNewTokens = 40;
  first.deadlineMilliseconds = 1'000'000;
  engine.submit(std::move(first));
  double now = 1;
  while (now < 100 && events.outputs[1].empty())
    static_cast<void>(engine.tick(now++));
  require(!events.outputs[1].empty() && executor.beginAttempts == 2 &&
              pool.snapshot().pagesAllocated >= 4,
          "the first request did not start through the host pause");

  auto second = request(2, std::vector<uint32_t>(65, 2));
  second.deadlineMilliseconds = 1'000'000;
  engine.submit(std::move(second));
  const CacheSnapshot before = resources.snapshot();
  static_cast<void>(engine.tick(now++));
  static_cast<void>(engine.tick(now++));
  require(!executor.requests.contains(2) && events.startIds.size() == 1 &&
              engine.resourceWaitSnapshot(now).memory == 1 &&
              resources.snapshot().pool.pagesPrefix >= before.pool.pagesPrefix &&
              resources.snapshot().stateCache.evictions == before.stateCache.evictions,
          "a second request grew, or evicted for the host, beside one in service");
  while (now < 1000 && events.completedCount < 2)
    static_cast<void>(engine.tick(now++));
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.startIds == std::vector<uint64_t>({1, 2}) &&
              executor.suspensions == 0,
          "the waiting request did not start once nothing was in service");
}

// A request whose start needs no new memory still starts beside one in
// service, as before, and then grows like it.
void testRequestThatNeedsNoGrowthStartsUnderTheHostPause() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  HostPause host;
  EngineConfig config;
  host.attach(config, storage, pool);
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  auto first = request(1, std::vector<uint32_t>(33, 1));
  first.maxNewTokens = 300;
  first.deadlineMilliseconds = 1'000'000;
  engine.submit(std::move(first));
  double now = 1;
  while (now < 100 && events.outputs[1].empty())
    static_cast<void>(engine.tick(now++));
  host.paused = true;
  // The lane's buffers are pooled and the first extent has free pages.
  auto second = request(2, std::vector<uint32_t>(33, 2));
  second.maxNewTokens = 300;
  second.deadlineMilliseconds = 1'000'000;
  engine.submit(std::move(second));
  while (now < 100 && events.outputs[2].empty())
    static_cast<void>(engine.tick(now++));
  require(!events.outputs[2].empty() && host.reusableAtGrowth.empty(),
          "a request that needed no new memory waited under the host pause");
  while (now < 2000 && events.completedCount < 2)
    static_cast<void>(engine.tick(now++));
  require(events.completedCount == 2 && events.failedCount == 0 &&
              executor.suspensions == 0 && !host.reusableAtGrowth.empty(),
          "requests in service did not grow through the host pause");
}

// A suspended request resumes like a new one: beside a request in service
// its pages wait for the host, and it resumes once that request has finished.
void testSuspendedRequestWaitsForTheHostBesideOneInService() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  HostPause host;
  EngineConfig config;
  const test::ScopedTestConfig seam({.resourceWaitTimeoutMilliseconds = 100.0});
  host.attach(config, storage, pool);
  // Until the host runs short, the engine's limit is two extents.
  storage.allocationFailure = metal::AllocationFailure::EngineBudget;
  storage.growthAllowed = [&](uint32_t) {
    return host.paused ? host.serving : pool.snapshot().pagesAllocated < 8;
  };
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  for (uint64_t id : {1, 2}) {
    auto value = request(id, std::vector<uint32_t>(id == 1 ? 65 : 33, id));
    value.maxNewTokens = 150;
    value.deadlineMilliseconds = 1'000'000;
    engine.submit(std::move(value));
  }
  double now = 1;
  while (now < 1000 && !executor.suspensions)
    static_cast<void>(engine.tick(now += 5));
  require(engine.snapshot().resourceSuspensions == 1 &&
              !executor.requests.at(2).resident && executor.requests.at(1).resident,
          "the limit did not suspend the shorter request");

  host.paused = true;
  storage.allocationFailure = metal::AllocationFailure::HostPressure;
  while (now < 5000 && events.completedCount < 1) {
    require(engine.snapshot().resourceResumptions == 0,
            "a suspended request grew through the pause beside one in service");
    static_cast<void>(engine.tick(now += 5));
  }
  // Its lane resumed and gave way again each time its pages were refused.
  require(events.completedCount == 1 && executor.resumeAttempts > 0 &&
              events.failedCount == 0 && events.outputs[1].size() == 150,
          "the request in service did not finish, or the suspended one never retried");
  while (now < 10000 && events.completedCount < 2)
    static_cast<void>(engine.tick(now += 5));
  require(events.completedCount == 2 && events.failedCount == 0 &&
              engine.snapshot().resourceResumptions == 1 &&
              events.outputs[2].size() == 150,
          "the suspended request did not resume once nothing was in service");
}

// The engine's own limit still binds a request in service under the pause:
// it reclaims as without the pause, and alone with nothing left to reclaim
// it has hit the capacity, at once and not after a wait for the host.
void testEngineLimitBindsThroughTheHostPause() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  bool paused = false;
  bool serving = false;
  EngineConfig config;
  config.growthPaused = [&] { return paused; };
  config.serving = [&](bool value) { serving = value; };
  // As the governor does, the host refuses first unless a request in service
  // needs the pages, then the engine's limit of two extents.
  storage.growthAllowed = [&](uint32_t) {
    if (paused && !serving) {
      storage.allocationFailure = metal::AllocationFailure::HostPressure;
      return false;
    }
    storage.allocationFailure = metal::AllocationFailure::EngineBudget;
    return pool.snapshot().pagesAllocated < 8;
  };
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  auto value = request(1, std::vector<uint32_t>(65, 1));
  value.maxNewTokens = 1000;
  value.deadlineMilliseconds = 1'000'000;
  engine.submit(std::move(value));
  double now = 1;
  while (now < 100 && events.outputs[1].empty())
    static_cast<void>(engine.tick(now++));
  paused = true;
  while (now < 1000 && !events.failedCount && !events.completedCount)
    static_cast<void>(engine.tick(now++));
  require(events.failures == std::vector<std::string>{"capacity_exhausted"} &&
              executor.suspensions == 0 && pool.snapshot().pagesAllocated == 8,
          "a lone request at the engine's limit waited for the host instead of failing");
}

// While the host refuses growth, a lane in service short of a page takes
// cached KV, not the cached states, whose buffers would give it no page: the
// checkpoint and the ordinary state stay.
void testPausedPageShortageKeepsCachedStates() {
  test::TestKvStorage storage(64, 4096, 1);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  HostPause host;
  EngineConfig config;
  host.attach(config, storage, pool);
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  // Cached: a checkpoint, an ordinary state below it, and KV below both.
  resources.beginRequest(900);
  require(resources.ensureTokens(900, 128).granted(), "fixture KV failed");
  resources.publishCommittedBlocks(900, std::vector<uint32_t>(129, 9), 128, {});
  resources.publishCompositeState(resources.blockAt(900, 32), std::make_shared<State>(), true);
  resources.publishCompositeState(resources.blockAt(900, 64), std::make_shared<State>(), false);
  resources.endRequest(900);
  // The decode needs one more page, at its 57th token.
  auto running = request(1, std::vector<uint32_t>(33, 1));
  running.maxNewTokens = 30;
  engine.submit(std::move(running));
  double now = 1;
  tickUntil(engine, now, [&] { return !events.outputs[1].empty(); },
            "the request did not decode");
  host.paused = true;
  const StateCacheSnapshot before = resources.snapshot().stateCache;
  tickUntil(engine, now, [&] { return idle(engine); }, "the request did not finish");
  const StateCacheSnapshot after = resources.snapshot().stateCache;
  require(events.completedCount == 1 && engine.snapshot().resourceSuspensions == 0 &&
              host.reusableAtGrowth.empty(),
          "the lane did not take a cached page under the pause");
  require(after.entries == before.entries && after.evictions == before.evictions &&
              after.checkpointEvictions == 0,
          "a page shortage under the pause evicted cached states");
}

// A lone request short of pages under host pressure takes idle cached pages
// of allocated extents instead of being suspended and replaying its prefix
// later. Reuse only happens when it can cover the shortfall.
void testSingletonHostPressureReusesIdleCacheInsteadOfSuspending() {
  test::TestKvStorage storage(32, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  MemoryPressure pressure = MemoryPressure::Normal;
  EngineConfig config;
  config.growthPaused = [&] { return pressure != MemoryPressure::Normal; };
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  engine.submit(request(230, std::vector<uint32_t>(65, 230)));
  runUntilIdle(engine);
  const auto cached = resources.snapshot();
  require(cached.pool.pagesPrefix >= 2 &&
              cached.pool.pagesFree + cached.pool.pagesPrefix >= 3,
          "idle-cache reuse fixture geometry changed");

  pressure = MemoryPressure::Warning;
  storage.growthBlocked = true;
  storage.allocationFailure = metal::AllocationFailure::HostPressure;
  engine.submit(request(231, std::vector<uint32_t>(65, 231)));
  runUntilIdle(engine);
  require(idle(engine) && events.completedCount == 2 &&
              events.failedCount == 0 && events.capacityExhaustedCount == 0,
          "lone request under host pressure did not complete on idle cache");
  require(executor.suspensions == 0 &&
              engine.snapshot().resourceSuspensions == 0 &&
              storage.allocationAttempts > 0,
          "lone request was suspended although idle cached pages could serve it");
  require(resources.snapshot().pool.pagesAllocated == cached.pool.pagesAllocated,
          "idle-cache reuse allocated another extent");
}

// A request that cannot start while an earlier lane holds memory waits for
// that lane however long it runs: the wait's limit restarts whenever the lane
// has work in flight. Once the lane is done, the limit runs out as before if
// memory still does not come.
void testAdmissionWaitsOutEarlierLanes() {
  for (const bool hostRecovers : {true, false}) {
    test::TestKvStorage storage(32, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor;
    executor.decodeFinishes = false;
    Events events;
    EngineConfig config;
    const test::ScopedTestConfig seam({.resourceWaitTimeoutMilliseconds = 100.0});
    engine::Engine engine(config, resources, executor, events);
    guardReleases(storage, engine);
    auto running = request(310, {310});
    running.maxNewTokens = 1000;
    engine.submit(std::move(running));
    static_cast<void>(engine.tick(1));
    // The host has no memory for a second lane while the first runs.
    executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
    executor.beginGrowthBlocked = [&] { return !hostRecovers || !events.completedCount; };
    engine.submit(request(311, {311}));
    double now = 1;
    while (now < 500)
      static_cast<void>(engine.tick(now += 50));
    require(!events.completedCount && events.failedCount == 0,
            "a request waiting for a resident lane's memory timed out while it ran");
    executor.decodeFinishes = true;
    while (!events.completedCount && now < 1000)
      static_cast<void>(engine.tick(now += 10));
    for (const double end = now + 150; now < end && !idle(engine);)
      static_cast<void>(engine.tick(now += 10));
    if (hostRecovers)
      require(events.completedCount == 2 && events.failedCount == 0,
              "the waiting request did not start once the lane finished");
    else
      require(events.completedCount == 1 &&
                  events.failures == std::vector<std::string>{"resource_timeout"},
              "a wait that no resident lane could end did not expire");
    require(idle(engine) && executor.requests.empty(), "the resource wait leaked a request");
  }
}

// A lane submitted after a waiting request does not extend its wait, or
// requests that keep arriving could hold it until its deadline. With
// admission closed behind the waiting request only a higher priority starts
// after it; the wait fails at its limit, retryably, while that lane still
// runs. The host refused its memory, so the failure says so.
void testLaterLanesDoNotExtendAResourceWait() {
  test::TestKvStorage storage(32, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  // The host has memory for the later request's state, not the waiting one's.
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  executor.beginGrowthBlocked = [&] { return executor.lastBeginId == 320; };
  Events events;
  EngineConfig config;
  // Shorter than the retry backoff: the limit runs out before the first
  // retry, which a decoding higher priority would defer for scheduling.
  const test::ScopedTestConfig seam({.resourceWaitTimeoutMilliseconds = 50.0});
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  engine.submit(request(320, {320}));
  static_cast<void>(engine.tick(1));
  require(engine.resourceWaitSnapshot(1).memory == 1 && !executor.requests.contains(320),
          "the first request did not wait for memory");
  // A higher priority is ahead of the waiting request in admission order and
  // is not held back.
  auto later = request(321, {321});
  later.priority = RequestPriority::Foreground;
  later.maxNewTokens = 1000;
  engine.submit(std::move(later));
  double now = 1;
  while (!events.failedCount && now < 1000) {
    static_cast<void>(engine.tick(now += 10));
    require(executor.requests.contains(321), "the higher priority did not run beside the wait");
  }
  require(now == 51 && events.failures == std::vector<std::string>{"resource_timeout"} &&
              events.failureDetails.back().second && !events.completedCount,
          "a lane submitted after a waiting request extended its wait");
  require(events.failureDetails.back().first ==
              "memory did not become available within the resource wait limit: "
              "macOS is short of memory; close memory-heavy applications",
          "a wait the host refused did not name the shortage");
  engine.cancel(321);
  for (const double end = now + 100; now < end && !idle(engine);)
    static_cast<void>(engine.tick(now += 10));
  require(idle(engine) && executor.requests.empty(), "the resource wait leaked a request");
}

// The scheduler admits the shortest prompt first, so a request submitted
// later can start in the pass that refuses an earlier one memory, also once
// both have waited for a free lane. Admitted before the refusal, that lane
// holds memory the wait is for: its work restarts the wait's limit, and the
// limit runs out only once it is gone.
void testLaneAdmittedBeforeARefusalHoldsTheWaitOpen() {
  for (const bool lanesFull : {false, true}) {
    test::TestKvStorage storage(64, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor;
    executor.decodeFinishes = false;
    executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
    executor.beginGrowthBlocked = [&] { return executor.lastBeginId == 330; };
    Events events;
    EngineConfig config;
    const test::ScopedTestConfig seam({.resourceWaitTimeoutMilliseconds = 50.0});
    engine::Engine engine(config, resources, executor, events);
    guardReleases(storage, engine);
    double now = 1;
    // Occupants in every lane: both requests wait for one until they are
    // cancelled.
    const uint64_t occupants = lanesFull ? model::ExecutionLimits::maximumBatchWidth : 0;
    for (uint64_t id = 1; id <= occupants; ++id) {
      auto occupant = request(id, {static_cast<uint32_t>(id)});
      occupant.maxNewTokens = 1000;
      engine.submit(std::move(occupant));
    }
    tickUntil(engine, now, [&] { return executor.requests.size() == occupants; },
              "the occupying lanes did not start");
    engine.submit(request(330, std::vector<uint32_t>(1025, 330)));
    auto later = request(331, {331});
    later.maxNewTokens = 1000;
    engine.submit(std::move(later));
    if (lanesFull) {
      tickUntil(engine, now, [&] { return engine.resourceWaitSnapshot(now).concurrency == 2; },
                "the requests did not wait for a lane");
      for (uint64_t id = 1; id <= occupants; ++id)
        engine.cancel(id);
    }
    tickUntil(engine, now, [&] { return executor.requests.contains(331); },
              "the later request did not start");
    require(!executor.requests.contains(330) && engine.resourceWaitSnapshot(now).memory == 1,
            "the later request did not start in the pass that refused the first");
    for (const double end = now + 300; now < end;)
      static_cast<void>(engine.tick(now += 10));
    require(events.failedCount == 0,
            "a lane admitted before the refusal did not hold the wait open");
    engine.cancel(331);
    const double cancelledAt = now;
    while (!events.failedCount && now < cancelledAt + 1000)
      static_cast<void>(engine.tick(now += 10));
    require(now <= cancelledAt + 70 &&
                events.failures == std::vector<std::string>{"resource_timeout"},
            "the wait did not expire once the lane admitted before it was gone");
  }
}

void testSingletonHostPressureWaitRecoversOrTerminates() {
  for (uint32_t outcome = 0; outcome < 4; ++outcome) {
    test::TestKvStorage storage(32, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(1);
    Events events;
    MemoryPressure pressure = MemoryPressure::Normal;
    EngineConfig config;
    config.growthPaused = [&] { return pressure != MemoryPressure::Normal; };
    const test::ScopedTestConfig seam(
        {.resourceWaitTimeoutMilliseconds = outcome == 3 ? 100.0 : 30000.0});
    engine::Engine engine(config, resources, executor, events);
    guardReleases(storage, engine);
    engine.submit(request(220, std::vector<uint32_t>(65, 220)));
    runUntilIdle(engine);
    const auto cached = resources.snapshot();

    pressure = MemoryPressure::Warning;
    storage.growthBlocked = true;
    storage.allocationFailure = metal::AllocationFailure::HostPressure;
    auto value = request(221, std::vector<uint32_t>(161, 221));
    value.deadlineMilliseconds = 300.0;
    engine.submit(std::move(value));
    static_cast<void>(engine.tick(20.0));
    static_cast<void>(engine.tick(21.0));
    // Suspension drops the lane's armed boundary silently: it is neither a
    // snapshot attempt nor a publication failure.
    require(executor.suspensions == 1 && events.capacityExhaustedCount == 0 &&
                resources.snapshot().pool.pagesPrefix == cached.pool.pagesPrefix &&
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
      storage.growthBlocked = false;
      for (double now = 121.0; now < 140.0 && !idle(engine); ++now)
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
                  events.failureDetails.back().second &&
                  events.failureDetails.back().first ==
                      "memory did not become available within the resource wait "
                      "limit: macOS is short of memory; close memory-heavy "
                      "applications",
              "persistent memory pressure did not fail with a retryable timeout "
              "that names the shortage");
    } else {
      static_cast<void>(engine.tick(301.0));
      require(events.completedCount + events.failedCount == 2,
              "host-pressure wait ignored deadline");
    }
    require(idle(engine) && executor.requests.empty() &&
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
  test::TestKvStorage storage(4, 4096, 1);
  storage.budgetPages = 3;
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);

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

// Cached state and KV share one budget. Required KV growth for a
// live lane takes the least recently used cached state through the unified
// reclaim order instead of failing or suspending, and the growing lane still
// publishes its own replay state afterwards.
void testKvGrowthReclaimsCachedStateWhenBudgetIsShared() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  storage.growthAllowed = [&](uint32_t) {
    return resources.snapshot().stateCache.entries == 0;
  };
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  engine.submit(request(230, std::vector<uint32_t>(65, 230)));
  runUntilIdle(engine);
  const auto cached = resources.snapshot();
  require(cached.stateCache.entries == 1 && cached.pool.pagesAllocated == 4,
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
              after.resources.pool.pagesAllocated == 8,
          "KV growth did not take exactly the older cached state");
  require(after.resources.pool.pagesActive == 0 &&
              after.resources.activeRequests == 0,
          "grown request leaked its active resources");
}

void testRequiredWorkDoesNotReserveAnExtraPage() {
  for (uint32_t promptTokens : {1U, 24U}) {
    test::TestKvStorage storage(2, 4096, 1);
    storage.budgetPages = 1;
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(1);
    Events events;
    engine::Engine engine({}, resources, executor, events);
    guardReleases(storage, engine);
    engine.submit(request(231, std::vector<uint32_t>(promptTokens, 231)));
    runUntilIdle(engine);
    require(events.completedCount == 1 && events.emitted == 1 &&
                events.capacityExhaustedCount == 0 &&
                engine.snapshot().scheduler.decodeBatchesByWidth[0] == 1,
            "unused KV runway rejected a request whose full verify fits");
  }
}

void testAdmissionPinsDesiredStateAndCountsOnlySuccess() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
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
              events.starts.back() == 64 &&
              after.cacheHits == before.cacheHits + 1 &&
              after.coldMisses == before.coldMisses,
          "successful retry failed to restore/count the protected cache state");
}

void testAdmissionCanDropItsOwnCachePinToMakeProgress() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
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
              events.starts.back() == 0 &&
              engine.snapshot().coldMisses == 2 && engine.snapshot().cacheHits == 0,
          "admission waited on its own cache pin instead of recomputing cold");
}

// Activation learns how much of the prompt a restore brings back, so the
// model can leave out the images inside it; an attempt that gives up its
// prefix to fit activates with nothing restored.
void testActivationReceivesTheRestoreBoundary() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(65, 247);
  engine.submit(request(247, prompt));
  runUntilIdle(engine);
  require(executor.lastRestoredTokens == 0, "a cold start activated with a restored prefix");
  engine.submit(request(248, prompt));
  runUntilIdle(engine);
  require(executor.lastRestoredTokens == 64 && executor.restored == 64,
          "a prefix hit did not activate with its restore boundary");
  std::vector<uint32_t> attempts;
  executor.beginObserver = [&] { attempts.push_back(executor.lastRestoredTokens); };
  executor.beginGrowthBlocked = [&] {
    return resources.snapshot().stateCache.entries != 0;
  };
  engine.submit(request(249, prompt));
  runUntilIdle(engine);
  require(attempts.size() >= 2 && attempts.front() == 64 && attempts.back() == 0 &&
              events.starts.back() == 0,
          "the retry without its prefix activated with the dropped restore boundary");
}

// A start refused memory beside a resident lane is held back by that lane
// (judge() yields): it waits with the state its prompt resumes from still
// cached, rather than release its lease to the reclaim and evict that state.
void testRefusedStartKeepsItsLeaseBesideAResidentLane() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  double now = 1;
  const std::vector<uint32_t> prompt(65, 1);
  engine.submit(request(1, prompt));
  tickUntil(engine, now, [&] { return idle(engine); }, "the first request did not finish");
  executor.decodeFinishes = false;
  auto lane = request(2, std::vector<uint32_t>(65, 2));
  lane.maxNewTokens = 1000;
  engine.submit(std::move(lane));
  tickUntil(engine, now, [&] { return events.outputs[2].size() >= 2; },
            "the resident lane did not decode");
  // The budget refuses the start while the lane is resident.
  executor.beginGrowthBlocked = [&] {
    return executor.lastBeginId == 3 && executor.requests.contains(2);
  };
  const uint32_t prefillRows = executor.prefillRows;
  engine.submit(request(3, prompt));
  for (int step = 0; step < 5; ++step)
    static_cast<void>(engine.tick(now++));
  require(resources.probe(prompt, {}).cachedTokens() == 64 &&
              engine.resourceWaitSnapshot(now).memory == 1 &&
              executor.prefillRows == prefillRows,
          "a start held back by a resident lane gave up its own cached state");
  executor.decodeFinishes = true;
  const uint32_t restored = executor.restored;
  tickUntil(engine, now, [&] { return idle(engine); }, "the held start did not run");
  require(events.starts.back() == 64 &&
              executor.restored == restored + 64 && events.failedCount == 0,
          "the held start did not resume from its cached state");
}

// Memory a state write in flight holds comes back by itself (judge() waits):
// a start refused memory meanwhile keeps the state its prompt resumes from.
void testRefusedStartKeepsItsLeaseWhileMemoryIsPending() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  auto write = std::make_shared<OffloadControl>();
  resources.beginRequest(999);
  require(resources.ensureTokens(999, 64).granted(), "fixture KV failed");
  resources.publishCompositeState(
      test::publishBlocks(resources, 999, std::vector<uint32_t>(64, 9), 64),
      std::make_shared<OffloadState>(write), false);
  resources.endRequest(999);
  require(resources.reclaimOneState(false, 0, false) &&
              resources.snapshot().stateCache.bytes == 0,
          "the fixture state was not written");
  double now = 1;
  const std::vector<uint32_t> prompt(65, 1);
  engine.submit(request(1, prompt));
  tickUntil(engine, now, [&] { return idle(engine); }, "the first request did not finish");
  executor.beginGrowthBlocked = [&] { return !write->ready; };
  engine.submit(request(2, prompt));
  for (int step = 0; step < 5; ++step)
    static_cast<void>(engine.tick(now++));
  require(resources.probe(prompt, {}).cachedTokens() == 64 &&
              engine.resourceWaitSnapshot(now).memory == 1 && events.startIds.size() == 1,
          "a start waiting for a write in flight gave up its own cached state");
  write->ready = true;
  tickUntil(engine, now, [&] { return idle(engine); }, "the waiting start did not run");
  require(events.starts.back() == 64 &&
              events.failedCount == 0,
          "the waiting start did not resume from its cached state");
}

// A start alone with nothing else to free releases its own lease, and the
// reclaim writes the state to disk; holding its lane, it looks up again and
// restores that copy instead of recomputing the prefix.
void testDroppedLeaseLooksUpAgain() {
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({.maxContext = 102400}, cache, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(65, 17);
  auto write = std::make_shared<OffloadControl>();
  write->ready = true;
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 64).granted(), "fixture KV failed");
  cache.publishCompositeState(test::publishBlocks(cache, 999, prompt, 64),
                              std::make_shared<OffloadState>(write), false);
  cache.endRequest(999);
  executor.beginGrowthBlocked = [&] { return cache.snapshot().stateCache.bytes != 0; };
  executor.restoreControl->ready = true;
  engine.submit(request(1, prompt));
  runUntilIdle(engine);
  require(executor.diskReads == 1 && executor.restored == 64 && executor.prefillRows == 1 &&
              events.starts.back() == 64 &&
              events.failedCount == 0,
          "a start that dropped its lease recomputed the state the reclaim wrote");
}

void testSingletonCapacityFailureTerminatesCleanly() {
  test::TestKvStorage storage(2, 4096, 1);
  storage.budgetPages = 1;
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);

  engine.submit(request(40, std::vector<uint32_t>(33, 40)));
  runUntilIdle(engine);
  require(events.completedCount == 0 && events.failedCount == 1 &&
              events.capacityExhaustedCount == 1 && idle(engine),
          "B1 capacity failure did not emit exactly one terminal event");
  require(resources.snapshot().activeRequests == 0 && executor.requests.empty(),
          "B1 capacity failure leaked backend resources");
}

void testQueuedLongPrefillsLeaveRoomForShortWork() {
  test::TestKvStorage storage(2048, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  for (uint64_t id = 1; id <= 4; ++id)
    engine.submit(request(id, std::vector<uint32_t>(8193, id)));
  require(engine.tick(1) && engine.tick(2) &&
              executor.requests.size() == 1 && executor.prefillRows == 2048,
          "long prefills reserved lanes without executable work");
  engine.submit(request(5, std::vector<uint32_t>(65, 5)));
  require(engine.tick(3) && executor.requests.contains(5) &&
              executor.requests.at(5).position > 0 && events.completedCount == 0,
          "short arrival waited for a long prefill to finish");
  for (uint32_t now = 4; now < 200 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && events.completedCount == 5 &&
              events.failedCount == 0 && events.capacityExhaustedCount == 0 &&
              executor.prefillRows == 4 * 8193 + 65 && executor.suspensions == 0,
          "admission lost work, introduced replay, or stranded queued requests");
}

void testAdmissionUsesCachedRemainingWork() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> warm(4097, 47);
  engine.submit(request(1, warm));
  runUntilIdle(engine);
  engine.submit(request(2, std::vector<uint32_t>(8193, 48)));
  require(engine.tick(1) && engine.tick(2), "cold prefill did not start");
  engine.submit(request(3, warm));
  require(engine.tick(3) && executor.requests.contains(3) &&
              executor.restored == 4096 && executor.requests.at(3).position == 4097,
          "cached prompt was scheduled by total length instead of remaining work");
  runUntilIdle(engine);
  require(events.completedCount == 3 && events.failedCount == 0,
          "cache-aware admission failed to finish");
}

// While a higher priority decodes, a waiting request cannot be selected:
// admission queues it without hashing its prompt. Once the higher priority
// is gone, one probe finds its cached prefix and it starts.
void testUnselectableCandidatesAreNotProbed() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  const auto hashed = [&] { return resources.snapshot().lookup.probeHashedBlocks; };
  const std::vector<uint32_t> cached(129, 5);
  engine.submit(request(1, cached));
  runUntilIdle(engine);
  executor.decodeFinishes = false;
  auto urgent = request(2, std::vector<uint32_t>(33, 6));
  urgent.priority = RequestPriority::Foreground;
  urgent.maxNewTokens = 1000;
  engine.submit(std::move(urgent));
  double now = 100;
  tickUntil(engine, now, [&] { return events.outputs.contains(2); },
            "the higher priority did not decode");
  std::vector<uint32_t> prompt(cached.begin(), cached.begin() + 128);
  prompt.resize(161, 8);
  engine.submit(request(3, prompt));
  const uint64_t before = hashed();
  for (uint32_t step = 0; step < 5; ++step)
    static_cast<void>(engine.tick(now++));
  require(hashed() == before && !executor.requests.contains(3) &&
              engine.snapshot().scheduler.queued == 1,
          "admission probed a request it could not select");
  engine.cancel(2);
  tickUntil(engine, now, [&] { return executor.requests.contains(3); },
            "the waiting request did not start once the higher priority left");
  require(hashed() - before == 5 && events.startIds.back() == 3 &&
              events.starts.back() == 128,
          "the request was not probed once, or missed its cached prefix");
}

// A request that waits beside a shorter resident prefill keeps its probe
// from pass to pass. Every resident command publishes blocks, a change of
// the KV graph, yet each pass hashes only the page past the match again.
void testWaitingCandidateProbeIsRefreshedNotRepeated() {
  test::TestKvStorage storage(4096, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  const auto hashed = [&] { return resources.snapshot().lookup.probeHashedBlocks; };
  const std::vector<uint32_t> cached(257, 3);
  engine.submit(request(1, cached));
  runUntilIdle(engine);
  engine.submit(request(2, std::vector<uint32_t>(20'000, 2)));
  double now = 100;
  tickUntil(engine, now, [&] { return executor.requests.contains(2); },
            "the resident prefill did not start");
  std::vector<uint32_t> prompt(cached.begin(), cached.begin() + 256);
  prompt.resize(40'000, 4);
  engine.submit(request(3, prompt));
  const uint64_t before = hashed();
  tickUntil(engine, now, [&] { return hashed() != before; },
            "the waiting request was not probed");
  require(hashed() - before == 9, "the first probe did not hash its prefix and the page past it");
  const uint64_t probed = hashed();
  const size_t commands = executor.prefillWidths.size();
  for (uint32_t step = 0; step < 6; ++step)
    static_cast<void>(engine.tick(now++));
  const size_t passes = executor.prefillWidths.size() - commands;
  require(passes >= 2 && hashed() - probed == passes && !executor.requests.contains(3),
          "a kept probe hashed more than the page past its match");
}

// A request that waits for memory closes admission behind it: the request
// that arrived after it is not tried, and starts once the first has.
void testMemoryWaitHoldsBackLaterArrivals() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  // The host refuses the first request twice: as an ordinary allocation and
  // as that of the only request in service.
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  executor.beginGrowthBlocked = [&] { return executor.beginAttempts <= 2; };
  engine.submit(request(1, std::vector<uint32_t>(4097, 47)));
  engine.submit(request(2, std::vector<uint32_t>(8193, 48)));
  static_cast<void>(engine.tick(1));
  require(executor.beginAttempts == 2 && executor.requests.empty() &&
              engine.resourceWaitSnapshot(1).memory == 1,
          "a later arrival was tried while an earlier request waits for memory");
  double now = 102;
  tickUntil(engine, now, [&] { return executor.requests.contains(2); },
            "the later arrival did not start after the request it waited behind");
  require(events.startIds == std::vector<uint64_t>{1, 2},
          "the requests did not start in the order they arrived");
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  require(events.completedCount == 2 && events.failedCount == 0,
          "a request held back behind a memory wait did not finish");
}

// Beside a resident lane a request waits for memory. Nothing that arrived
// after it starts before it does, however little it needs; a higher priority
// is ahead of it and starts. The requests held back queue without a memory
// wait of their own, and start in order once the waiting request has.
void testMemoryWaitClosesAdmissionBesideAResidentLane() {
  test::TestKvStorage storage(2048, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  bool refused = true;
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  executor.beginGrowthBlocked = [&] { return refused && executor.lastBeginId == 2; };
  double now = 1;
  engine.submit(request(1, std::vector<uint32_t>(16385, 1)));
  require(engine.tick(now++) && executor.requests.contains(1), "the first request did not start");
  engine.submit(request(2, std::vector<uint32_t>(4097, 2)));
  tickUntil(engine, now, [&] { return engine.resourceWaitSnapshot(now).memory == 1; },
            "the refused request is not waiting for memory");
  engine.submit(request(3, std::vector<uint32_t>(65, 3)));
  for (uint32_t step = 0; step < 4; ++step)
    static_cast<void>(engine.tick(now++));
  require(!executor.requests.contains(3) && !events.usage.contains(3) &&
              engine.resourceWaitSnapshot(now).memory == 1 && events.failedCount == 0,
          "a later arrival started ahead of a request that waits for memory");
  auto urgent = request(4, std::vector<uint32_t>(65, 4));
  urgent.priority = RequestPriority::Foreground;
  engine.submit(std::move(urgent));
  tickUntil(engine, now, [&] { return executor.requests.contains(4) || events.usage.contains(4); },
            "closed admission held back a higher priority");
  require(!executor.requests.contains(3) && !events.usage.contains(3),
          "admission reopened for a later arrival of the same priority");
  refused = false;
  now += 100;
  tickUntil(engine, now, [&] { return executor.requests.contains(2); },
            "the waiting request did not start once its memory was there");
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  const auto started = [&](uint64_t id) {
    return std::find(events.startIds.begin(), events.startIds.end(), id) - events.startIds.begin();
  };
  require(events.completedCount == 4 && events.failedCount == 0 && started(2) < started(3) &&
              executor.prefillRows == 16385 + 4097 + 65 + 65,
          "closed admission lost work or started it out of order");
}

// The request that closed admission is cancelled: what queued behind it
// starts without it.
void testClosedAdmissionReopensWhenTheWaitEnds() {
  test::TestKvStorage storage(2048, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  executor.beginGrowthBlocked = [&] { return executor.lastBeginId == 2; };
  double now = 1;
  engine.submit(request(1, std::vector<uint32_t>(16385, 1)));
  require(engine.tick(now++) && executor.requests.contains(1), "the first request did not start");
  engine.submit(request(2, std::vector<uint32_t>(4097, 2)));
  tickUntil(engine, now, [&] { return engine.resourceWaitSnapshot(now).memory == 1; },
            "the refused request is not waiting for memory");
  engine.submit(request(3, std::vector<uint32_t>(65, 3)));
  for (uint32_t step = 0; step < 4; ++step)
    static_cast<void>(engine.tick(now++));
  require(!executor.requests.contains(3) && !events.usage.contains(3),
          "admission did not close behind the request that waits for memory");
  engine.cancel(2);
  tickUntil(engine, now, [&] { return executor.requests.contains(3) || events.usage.contains(3); },
            "admission stayed closed after the waiting request was cancelled");
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  // The cancelled request is reported as ended, without a row of its own.
  require(events.completedCount == 3 && events.failedCount == 0 &&
              engine.snapshot().cancelled == 1 && executor.prefillRows == 16385 + 65,
          "a cancelled wait left work behind");
}

// A request refused memory waits next for a prefix that a lane admitted in
// the same pass is about to compute. A prefix wait is not an attempt: the
// refusal stands, and a short request that arrived after it stays queued
// until the refused request starts from the producer's junction.
void testPrefixWaitKeepsAdmissionClosedBehindARefusal() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  // Two lanes: the first two requests take both, so the producer waits for
  // a lane while the host refuses request 4.
  Executor executor(2);
  bool hostRefuses = true;
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  executor.beginGrowthBlocked = [&] { return hostRefuses && executor.lastBeginId == 4; };
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  // The producer and request 4 share 960 tokens.
  std::vector<uint32_t> producer(1000, 7);
  std::vector<uint32_t> refused(1100, 7);
  std::fill(producer.begin() + 960, producer.end(), 8);
  std::fill(refused.begin() + 960, refused.end(), 9);
  engine.submit(request(1, {1}));
  engine.submit(request(2, {2}));
  engine.submit(request(3, producer));
  engine.submit(request(4, refused));
  static_cast<void>(engine.tick(1));
  const auto waits = engine.resourceWaitSnapshot(1);
  require(executor.requests.size() == 2 && waits.memory == 1 && waits.concurrency == 1,
          "the producer did not wait for a lane beside a refused request");
  hostRefuses = false;
  require(engine.tick(2), "the first requests did not prefill");
  engine.cancel(2);
  engine.submit(request(5, {5}));
  // The producer starts in the freed lane with a junction for request 4,
  // which then waits for it; a decode runs before the producer's prefill.
  require(engine.tick(3) && executor.requests.contains(3) &&
              engine.snapshot().scheduler.waitingPrefix == 1,
          "the refused request did not wait for the producer's prefix");
  double now = 4;
  while (!executor.requests.contains(4) && now < 100) {
    require(!executor.requests.contains(5) && !events.usage.contains(5),
            "a later arrival started while a refused request waited for a prefix");
    static_cast<void>(engine.tick(now++));
  }
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  const auto started = [&](uint64_t id) {
    return std::find(events.startIds.begin(), events.startIds.end(), id) - events.startIds.begin();
  };
  require(events.completedCount == 5 && events.failedCount == 0 && started(4) < started(5) &&
              executor.restored == 960,
          "the refused request did not start first from the producer's junction");
}

// After a suspension, suspended requests resume one at a time in admission
// order. The first one's resume is refused memory: the one after it, whose
// pages would fit, stays suspended, and once the resident lane finishes the
// first resumes first.
void testRefusedResumeHoldsBackLaterSuspendedLanes() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  const test::ScopedTestConfig seam({.resourceWaitTimeoutMilliseconds = 1000.0});
  engine::Engine engine({}, resources, executor,
                        events);
  guardReleases(storage, engine);
  // The resident lane fills the first extent and decodes within it.
  auto resident = request(1, std::vector<uint32_t>(97, 1));
  resident.maxNewTokens = 1000;
  engine.submit(std::move(resident));
  double now = 1;
  tickUntil(engine, now, [&] { return events.emitted != 0; }, "the resident lane did not decode");
  // The host refuses growth: both later requests are suspended at their
  // first prefill, request 2 to resume with five pages, request 3 with one.
  storage.growthBlocked = true;
  storage.allocationFailure = metal::AllocationFailure::HostPressure;
  const std::vector<uint32_t> first(161, 2);
  const std::vector<uint32_t> second(33, 3);
  engine.submit(request(2, first));
  engine.submit(request(3, second));
  tickUntil(engine, now, [&] { return executor.suspensions == 2; },
            "the later requests were not suspended");
  // One more extent: room for request 3's page, not for request 2's. The
  // refusal of request 2's resume drains the resident lane for the wait
  // limit; its retries after the drain are refused too.
  storage.growthBlocked = false;
  storage.allocationFailure = metal::AllocationFailure::EngineBudget;
  storage.budgetPages = 8;
  for (const double end = now + 1500; now < end; now += 101) {
    static_cast<void>(engine.tick(now));
    require(engine.snapshot().resourceResumptions == 0 && !executor.requests.at(3).resident &&
                events.failedCount == 0,
            "a suspended request resumed ahead of an earlier one refused memory");
  }
  require(executor.resumeAttempts >= 3 && executor.requests.at(1).resident,
          "the refused resume was not retried beside the resident lane");
  executor.decodeFinishes = true;
  for (const double end = now + 2000; now < end && !idle(engine); now += 101)
    static_cast<void>(engine.tick(now));
  // The executor records every state it grants, request 2's refused
  // resumes included: request 3's is the last.
  const auto &resumed = executor.resumedPrompts;
  require(idle(engine) && events.completedCount == 3 && events.failedCount == 0 &&
              engine.snapshot().resourceResumptions == 2 &&
              std::find(resumed.begin(), resumed.end(), second) == resumed.end() - 1,
          "the suspended requests did not resume in admission order");
}

// A suspended request held behind one whose resume is refused memory is not
// tried, so it neither keeps a retry time nor runs out its wait. With
// nothing resident, the native loop sleeps until the refused request's next
// retry; when the wait limit ends that request, the held one resumes.
void testHeldSuspendedRequestNeitherWakesNorExpires() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.kvGrowthBlocked = &storage.growthBlocked;
  executor.unblockGrowthOnSuspend = false;
  Events events;
  const test::ScopedTestConfig seam({.resourceWaitTimeoutMilliseconds = 300.0});
  engine::Engine engine({}, resources, executor,
                        events);
  guardReleases(storage, engine);
  // The host refuses growth, so both requests are suspended at their first
  // prefill, and then it refuses every resume.
  storage.growthBlocked = true;
  storage.allocationFailure = metal::AllocationFailure::HostPressure;
  engine.submit(request(1, {1}));
  engine.submit(request(2, {2}));
  require(engine.tick(1) && engine.tick(2) && executor.suspensions == 2 &&
              !engine.commandInFlight(),
          "the requests were not suspended");
  storage.growthBlocked = false;
  executor.resumeDenied = true;
  // As the native loop does, sleep until the next wake-up.
  double now = 2;
  while (!events.failedCount) {
    const std::optional<double> wakeup = engine.nextWakeupMilliseconds();
    require(wakeup && *wakeup > now,
            "a request held behind a refused resume woke the loop with nothing to do");
    now = *wakeup;
    static_cast<void>(engine.tick(now));
  }
  require(events.failures == std::vector<std::string>{"resource_timeout"},
          "the refused resume did not end at its wait limit alone");
  executor.resumeDenied = false;
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  require(events.completedCount == 1 && events.failedCount == 1 &&
              engine.snapshot().resourceResumptions == 1,
          "a request held behind a refused resume ran out its wait meanwhile");
}

// /status sees the requests a refused one holds back. While a pass defers
// the refused request itself for scheduling, it counts among them: it still
// waits for memory. Its wait age runs from its first refusal across such
// passes.
void testStatusCountsRequestsHeldBehindARefusal() {
  test::TestKvStorage storage(2048, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  bool refused = true;
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  executor.beginGrowthBlocked = [&] { return refused && executor.lastBeginId == 2; };
  double now = 1;
  engine.submit(request(1, std::vector<uint32_t>(16385, 1)));
  require(engine.tick(now++) && executor.requests.contains(1), "the first request did not start");
  engine.submit(request(2, std::vector<uint32_t>(4097, 2)));
  tickUntil(engine, now, [&] { return engine.resourceWaitSnapshot(now).memory == 1; },
            "the refused request is not waiting for memory");
  const double refusedBy = now;
  engine.submit(request(3, std::vector<uint32_t>(65, 3)));
  engine.submit(request(4, std::vector<uint32_t>(65, 4)));
  static_cast<void>(engine.tick(now++));
  auto wait = engine.resourceWaitSnapshot(now);
  require(wait.heldBehindRefusal == 2 && wait.memory == 1,
          "the requests behind a refusal were not counted");
  // A higher priority arrives by the refused request's next retry and starts
  // in that pass, which defers the refused request for scheduling: it leaves
  // its memory wait.
  auto urgent = request(5, std::vector<uint32_t>(65, 5));
  urgent.priority = RequestPriority::Foreground;
  engine.submit(std::move(urgent));
  now += 100;
  tickUntil(engine, now, [&] { return engine.resourceWaitSnapshot(now).memory == 0; },
            "the refused request was not deferred for scheduling");
  wait = engine.resourceWaitSnapshot(now);
  require(wait.heldBehindRefusal == 3 && !executor.requests.contains(3),
          "a refused request deferred for scheduling, or what it holds back, was not counted");
  tickUntil(engine, now, [&] { return engine.resourceWaitSnapshot(now).memory == 1; },
            "the refused request did not wait for memory again");
  wait = engine.resourceWaitSnapshot(now);
  require(wait.heldBehindRefusal == 2 && wait.oldestWaitMilliseconds >= now - refusedBy,
          "the refused request's wait age restarted when a pass deferred it");
  refused = false;
  now += 100;
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  wait = engine.resourceWaitSnapshot(now);
  require(events.completedCount == 5 && events.failedCount == 0 && wait.heldBehindRefusal == 0,
          "the requests held behind a refusal did not finish");
}

// During recovery admission tries only suspended requests, and the first one
// refused memory holds back the suspended ones after it. /status counts
// those, not the suspended requests behind an earlier one refused its start:
// recovery tries them all the same.
void testStatusCountsSuspendedRequestsHeldDuringRecovery() {
  test::TestKvStorage storage(512, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  bool hostRefuses = true;
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  executor.beginGrowthBlocked = [&] { return hostRefuses && executor.lastBeginId == 2; };
  Events events;
  const test::ScopedTestConfig seam({.resourceWaitTimeoutMilliseconds = 1000.0});
  engine::Engine engine({}, resources, executor,
                        events);
  guardReleases(storage, engine);
  // The resident lane fills the first extent and decodes within it.
  auto resident = request(1, std::vector<uint32_t>(97, 1));
  resident.maxNewTokens = 1000;
  engine.submit(std::move(resident));
  double now = 1;
  tickUntil(engine, now, [&] { return events.emitted != 0; }, "the resident lane did not decode");
  // The host refuses request 2's state and all growth. Requests 3 and 4 start
  // in the pass that refuses request 2 and are suspended at their first
  // prefill. Request 3's resume is refused too, and request 4 is held behind
  // it without an attempt.
  storage.growthBlocked = true;
  storage.allocationFailure = metal::AllocationFailure::HostPressure;
  engine.submit(request(2, std::vector<uint32_t>(4097, 2)));
  engine.submit(request(3, std::vector<uint32_t>(33, 3)));
  engine.submit(request(4, std::vector<uint32_t>(33, 4)));
  tickUntil(engine, now, [&] { return executor.suspensions == 2; },
            "the later requests were not suspended");
  for (const double end = now + 1500;
       now < end && !engine.resourceWaitSnapshot(now).heldBehindRefusal; now += 101)
    static_cast<void>(engine.tick(now));
  const auto wait = engine.resourceWaitSnapshot(now);
  require(wait.suspended == 2 && wait.heldBehindRefusal == 1,
          "recovery counted requests it does not hold back behind a refused resume");
  hostRefuses = false;
  storage.growthBlocked = false;
  executor.decodeFinishes = true;
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  require(events.completedCount == 4 && events.failedCount == 0,
          "the requests did not finish once memory was there");
}

void testSchedulingWaitDoesNotConsumeMemoryTimeout() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  const test::ScopedTestConfig seam({.resourceWaitTimeoutMilliseconds = 1000.0});
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  // The host refuses the first request twice: as an ordinary allocation and
  // as that of the only request in service.
  executor.beginGrowthBlocked = [&] { return executor.beginAttempts <= 2; };
  engine.submit(request(1, std::vector<uint32_t>(8193, 47)));
  static_cast<void>(engine.tick(1));
  require(executor.beginAttempts == 2 && !executor.requests.contains(1) &&
              engine.resourceWaitSnapshot(1).memory == 1,
          "the first request did not wait for memory");
  auto urgent = request(2, std::vector<uint32_t>(4097, 48));
  urgent.priority = RequestPriority::Foreground;
  engine.submit(std::move(urgent));
  // Prefill admission packs only the top priority tier: the first request
  // waits for scheduling alone.
  require(engine.tick(102) && engine.resourceWaitSnapshot(102).memory == 0 &&
              executor.requests.contains(2) && !executor.requests.contains(1),
          "scheduler delay retained a stale memory wait");
  for (uint32_t now = 1100; now < 1200 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && events.completedCount == 2 && events.failedCount == 0,
          "scheduling delay triggered the memory wait timeout");
}

void testUnadmittedRequestsHonorCancellationAndDeadline() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  engine.submit(request(1, std::vector<uint32_t>(8193, 1)));
  engine.submit(request(2, std::vector<uint32_t>(8193, 2)));
  auto expiring = request(3, std::vector<uint32_t>(8193, 3));
  expiring.deadlineMilliseconds = 3;
  engine.submit(std::move(expiring));
  require(engine.tick(1) && engine.tick(2), "long prefill did not start");
  engine.cancel(2);
  static_cast<void>(engine.tick(3));
  runUntilIdle(engine);
  require(engine.snapshot().cancelled == 1 && events.failedCount == 1 &&
              events.completedCount == 2 && executor.beginAttempts == 1,
          "queued cancellation or deadline allocated resources or failed cleanup");
  require(events.failures == std::vector<std::string>{"deadline_exceeded"} &&
              events.failureDetails.front().first == "request deadline exceeded",
          "a queued request's deadline lost its outcome or message");
}

void testGrowthKeepsPrefillProgressWhenAnUnstartedPeerCanYield() {
  for (RequestPriority priority : {RequestPriority::Normal,
                                   RequestPriority::Background}) {
    test::TestKvStorage storage(512, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(2);
    Events events;
    engine::Engine engine({}, resources, executor, events);
    guardReleases(storage, engine);
    storage.growthAllowed = [&](uint32_t) {
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
                !executor.requests.contains(49),
            "unstarted prefill reserved a lane before it had scheduled work");
    require(engine.tick(3) && executor.requests.at(48).resident,
            "KV growth discarded completed prefill");
    if (priority == RequestPriority::Normal) {
      // The final prefill slice admits a peer into its remaining row budget;
      // if that lane prevents KV growth, the unstarted peer must yield.
      require(executor.suspensions == 1 && !executor.requests.at(49).resident,
              "KV growth did not yield the unstarted peer");
    } else {
      require(executor.suspensions == 0 && !executor.requests.contains(49),
              "lower-priority work reserved a lane before its dispatch");
    }
    runUntilIdle(engine);
    require(events.completedCount == 2 && events.failedCount == 0 &&
                events.capacityExhaustedCount == 0 && executor.prefillRows == 12288,
            "work-aware admission lost prefill work or failed to complete");
  }
}

void testGrowthYieldsLowerPriorityResidentOutsideBatch() {
  test::TestKvStorage storage(512, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  storage.growthAllowed = [&](uint32_t) {
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
  test::TestKvStorage storage(512, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  engine.submit(request(48, std::vector<uint32_t>(8192, 48)));
  require(engine.tick(1) && engine.tick(2), "prefill setup did not progress");
  auto decoding = request(49, {49});
  decoding.maxNewTokens = 32;
  engine.submit(std::move(decoding));
  require(engine.tick(3) && engine.tick(4) &&
              executor.requests.at(49).position == 1,
          "short peer did not enter decode");
  storage.growthBlocked = true;
  for (uint32_t step = 5; step < 15 && !executor.suspensions; ++step)
    static_cast<void>(engine.tick(step));
  require(executor.suspensions == 1 && !executor.requests.at(48).resident &&
              executor.requests.at(49).resident,
          "prefill growth interrupted an equal-priority decode stream");
  storage.growthBlocked = false;
  executor.decodeFinishes = true;
  runUntilIdle(engine);
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0,
          "mixed-phase pressure failed to finish both requests");
}

void testKvPressureSuspendsInsteadOfKillingActiveWork() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.kvGrowthBlocked = &storage.growthBlocked;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);

  storage.growthBlocked = true;
  engine.submit(request(50, {50}));
  engine.submit(request(51, {51}));
  runUntilIdle(engine);

  require(executor.suspensions == 1 && executor.resumptions == 1 &&
              engine.snapshot().resourceSuspensions == 1 &&
              engine.snapshot().resourceResumptions == 1,
          "KV pressure did not suspend and resume one lane");
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0,
          "recoverable pressure killed an active request");
}

void testPressureRetryIsBackedOffWithoutProgress() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.kvGrowthBlocked = &storage.growthBlocked;
  executor.unblockGrowthOnSuspend = false;
  Events events;
  MemoryPressure pressure = MemoryPressure::Warning;
  EngineConfig config;
  config.growthPaused = [&] { return pressure != MemoryPressure::Normal; };
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);

  storage.growthBlocked = true;
  storage.allocationFailure = metal::AllocationFailure::HostPressure;
  engine.submit(request(52, {52}));
  engine.submit(request(53, {53}));
  require(engine.tick(1.0) && engine.tick(2.0),
          "pressure did not suspend both blocked lanes");
  const uint32_t attemptsBeforeBackoff = executor.resumeAttempts;
  require(executor.suspensions == 2 && attemptsBeforeBackoff == 0,
          "suspended lanes were retried before a resource wake-up");
  require(!engine.tick(50.0) &&
              executor.resumeAttempts == attemptsBeforeBackoff,
          "resource wait retried on an unrelated scheduler tick");

  storage.growthBlocked = false;
  pressure = MemoryPressure::Normal;
  require(engine.tick(102.0), "resource retry timer did not wake the engine");
  for (double now = 103.0; now < 120.0 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && executor.resumptions == 2 &&
              events.completedCount == 2 && events.failedCount == 0,
          "backed-off resource requests did not recover cleanly");
}

// The native loop blocks until the next wake-up. A waiting request's retry
// time wakes it only when tick() can retry that request: admission runs
// between commands, and after a suspension only for suspended requests.
// Otherwise the retry time passes and the loop polls without blocking.
void testAdmissionRetryWakesOnlyWhenTickCanRetry() {
  {
    test::TestKvStorage storage(8, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(1);
    executor.holdDecodeUntil = std::make_shared<std::atomic<bool>>(false);
    Events events;
    engine::Engine engine({}, resources, executor, events);
    guardReleases(storage, engine);
    engine.submit(request(1, {1}));
    require(engine.tick(1) && engine.tick(2), "resident request did not prefill");
    engine.submit(request(2, {2}));
    require(engine.tick(3) && engine.commandInFlight() &&
                engine.resourceWaitSnapshot(3).concurrency == 1,
            "second request did not wait for the resident lane's command");
    require(!engine.tick(150) &&
                engine.nextWakeupMilliseconds() == 1150.0,
            "a lane wait asked for a wake-up while a command was in flight");
    *executor.holdDecodeUntil = true;
    for (double now = 151; now < 170 && !idle(engine); ++now)
      static_cast<void>(engine.tick(now));
    require(idle(engine) && events.completedCount == 2,
            "lane wait did not complete after the command");
  }
  {
    test::TestKvStorage storage(8, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(1);
    Events events;
    bool paused = true;
    EngineConfig config;
    config.growthPaused = [&] { return paused; };
    engine::Engine engine(config, resources, executor, events);
    guardReleases(storage, engine);
    storage.growthBlocked = true;
    storage.allocationFailure = metal::AllocationFailure::HostPressure;
    engine.submit(request(1, {1}));
    engine.submit(request(2, {2}));
    require(engine.tick(1) && executor.suspensions == 1 &&
                engine.resourceWaitSnapshot(1).concurrency == 1,
            "fixture did not suspend one request behind a lane wait");
    require(!engine.tick(101) && executor.resumeAttempts == 1 &&
                engine.nextWakeupMilliseconds() == 201.0,
            "a lane wait asked for a wake-up only the suspended request gets");
    paused = false;
    storage.growthBlocked = false;
    for (double now = 201; now < 220 && !idle(engine); ++now)
      static_cast<void>(engine.tick(now));
    require(idle(engine) && events.completedCount == 2,
            "lane wait did not complete after the suspended request");
  }
}

void testRecoveryDrainDoesNotConsumeResourceWaitBudget() {
  for (bool expireRequest : {false, true}) {
    test::TestKvStorage storage(3, 4096, 1);
    storage.budgetPages = 2;
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(2);
    executor.decodeFinishes = false;
    Events events;
    engine::Engine engine({}, resources, executor, events);
    guardReleases(storage, engine);
    // The resident's held command is its last one.
    for (uint64_t id : {250, 251}) {
      auto value = request(id, std::vector<uint32_t>(24, id));
      value.maxNewTokens = 2;
      value.deadlineMilliseconds = expireRequest ? 30000 : 1000000;
      engine.submit(std::move(value));
    }
    for (double now = 1; now <= 5; ++now)
      require(engine.tick(now), "drain fixture made no progress");
    require(executor.suspensions == 1, "drain fixture did not preempt a lane");
    executor.holdDecodeUntil = std::make_shared<std::atomic<bool>>(false);
    require(engine.tick(6) && engine.commandInFlight(),
            "resident peer did not start its command");
    static_cast<void>(engine.tick(20006));
    require(engine.resourceWaitSnapshot(20006).draining,
            "recovery did not drain behind the resident peer");
    // The drain ends at the resource wait limit, and its time did not count
    // against the suspended request's own resource wait.
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
      require(!wait.draining && wait.suspended == 1 && wait.memory == 1 &&
                  wait.oldestWaitMilliseconds >= 30000,
              "drain outlived its limit or hid the elapsed resource wait");
    }
    *executor.holdDecodeUntil = true;
    for (double now = 30007; now < 30100 && !idle(engine); ++now)
      static_cast<void>(engine.tick(now));
    require(idle(engine) && resources.snapshot().activeRequests == 0,
            "drain fixture did not release its resources");
    if (!expireRequest)
      require(events.completedCount == 2 && events.failedCount == 0 &&
                  executor.resumptions == 1,
              "suspended work did not resume after its peer completed");
  }
}

void testDecodePreemptionReplaysCommittedHistoryWithoutRepeatingOutput() {
  test::TestKvStorage storage(3, 4096, 1);
  storage.budgetPages = 2;
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
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
  executor.holdDecodeUntil = std::make_shared<std::atomic<bool>>(false);
  engine.submit(request(252, {252}));
  require(engine.tick(6) && engine.commandInFlight(),
          "surviving lane could not launch after preemption");
  require(!engine.tick(150) && engine.nextWakeupMilliseconds() == 1150.0 &&
              events.startIds.size() == 2 && executor.resumeAttempts == 0,
          "draining recovery busy-woke or admitted a competing request");
  *executor.holdDecodeUntil = true;
  for (double now = 151; now < 210 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && executor.resumptions == 2 &&
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
              engine.snapshot().cacheHits == 0 && engine.snapshot().coldMisses == 3,
          "resource replay emitted prior output or changed request usage/hits");
  require(executor.requests.empty() && executor.snapshotAttempts == 0 &&
              resources.snapshot().pool.pagesActive == 0 &&
              resources.snapshot().activeRequests == 0,
          "decode replay leaked active resources");
}

void testLongDecodePreemptionPlansTheCurrentReplayBoundary() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  executor.decodeFinishes = false;
  executor.deniedSnapshots = std::numeric_limits<uint32_t>::max();
  Events events;
  MemoryPressure pressure = MemoryPressure::Normal;
  EngineConfig config;
  // Keep the old 4096-token replay boundary distinct from checkpoints.
  const test::ScopedTestConfig seam({.prefillCheckpointTokens = 8192});
  config.growthPaused = [&] { return pressure != MemoryPressure::Normal; };
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);

  constexpr uint64_t id = 255;
  const std::vector<uint32_t> prompt(4097, 5);
  auto value = request(id, prompt);
  value.maxNewTokens = 10'000;
  value.deadlineMilliseconds = 100'000;
  engine.submit(std::move(value));
  double now = 1;
  for (; now < 20'000 && executor.suspensions == 0; ++now) {
    if (events.emitted >= 4200) {
      storage.growthBlocked = true;
      storage.allocationFailure = metal::AllocationFailure::HostPressure;
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

  // Retain the lane's KV one draft window past the prompt's replay boundary,
  // whose state is gone. That KV ends there, so it is no junction: the
  // resumed lane rebuilds the state where the conversation's next turn
  // resumes, then captures the generated history's end.
  constexpr uint32_t retainedKvTokens = 6144;
  while (resources.snapshot().pool.pagesPrefix >
         retainedKvTokens / KvCache::pageTokens) {
    require(resources.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse).madeProgress,
            "could not reclaim the suspended decode's KV tail");
  }
  {
    auto lookup = resources.lookup(history, {});
    require(lookup.kvBoundary == retainedKvTokens &&
                lookup.resumeBoundary() == 0,
            "long replay did not retain the intended stateless KV");
  }

  storage.growthBlocked = false;
  pressure = MemoryPressure::Normal;
  executor.deniedSnapshots = 0;
  executor.decodeFinishes = true;
  now += 101;
  require(engine.tick(now++), "long replay did not resume after pressure eased");
  const auto &plan = executor.plans.at(id);
  require(plan.replayEnd == history.size() && plan.captureSpans.size() == 2 &&
              plan.boundaries.size() == 3 && plan.boundaries.front().boundary == 4096,
          "resumed draft plan did not rebuild the prompt's replay point first");
  const double finishBy = now + 100;
  for (; now < finishBy && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && executor.resumptions == 1 &&
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
  // The conversation's next turn resumes from the rebuilt point.
  std::vector<uint32_t> next = prompt;
  next.resize(next.size() + 40, 9);
  require(resources.lookup(next, {}).resumeBoundary() == 4096 &&
              resources.snapshot().stateCache.inUse == 0,
          "the resumed lane did not rebuild the prompt's replay point");
}

// A sibling that arrives while a producer is suspended waits for it once
// it resumes, as for any producer, and the resumed producer plans the
// junction at their shared boundary as any producer does: the sibling
// resumes there instead of prefilling up to a checkpoint interval of the
// prefix it shares.
void testResumedProducerPlansJunctionsForSiblings() {
  test::TestKvStorage storage(2048, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  Events events;
  bool paused = false;
  EngineConfig config;
  config.growthPaused = [&] { return paused; };
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  std::vector<uint32_t> prompt(20001);
  std::iota(prompt.begin(), prompt.end(), 1);
  engine.submit(request(1, prompt));
  double now = 1;
  tickUntil(engine, now,
            [&] { return !engine.commandInFlight() && executor.prefillRows >= 10'000; },
            "the producer did not pass its checkpoint at 8192");
  // The host refuses the producer's next pages: it yields its lane.
  paused = true;
  storage.growthBlocked = true;
  storage.allocationFailure = metal::AllocationFailure::HostPressure;
  tickUntil(engine, now, [&] { return executor.suspensions == 1; },
            "the producer was not suspended");
  std::vector<uint32_t> sibling(prompt.begin(), prompt.begin() + 18'000);
  sibling.resize(18'065, 0);
  engine.submit(request(2, sibling));
  const uint32_t rows = executor.prefillRows;
  paused = false;
  storage.growthBlocked = false;
  now += 101;
  tickUntil(engine, now, [&] { return engine.snapshot().scheduler.waitingPrefix == 1; },
            "the sibling did not wait for the resumed producer");
  const auto &boundaries = executor.plans.at(1).boundaries;
  require(executor.resumptions == 1 && executor.restored == 8192 &&
              std::any_of(boundaries.begin(), boundaries.end(),
                          [](const DraftBoundaryPlan &boundary) {
                            return boundary.boundary == 17'984;
                          }),
          "the resumed producer planned no junction at the shared boundary");
  tickUntil(engine, now, [&] { return idle(engine); }, "the requests did not finish");
  require(events.starts.back() == 17'984 &&
              executor.prefillRows - rows == (prompt.size() - 8192) + (sibling.size() - 17'984) &&
              events.completedCount == 2 && events.failedCount == 0,
          "the sibling did not resume from the resumed producer's junction");
}

// The field case of a lane suspended mid-decode: it resumes with its
// prompt's replay point lost, and the only memory left is another
// conversation's cached KV, two whole extents with no state to recycle.
struct LostReplayPoint {
  LostReplayPoint() {
    executor.decodeFinishes = false;
    executor.deniedSnapshots = std::numeric_limits<uint32_t>::max();
    guardReleases(storage, engine);
    auto value = request(id, prompt);
    value.maxNewTokens = 10'000;
    value.deadlineMilliseconds = 100'000;
    engine.submit(std::move(value));
    for (; now < 20'000 && executor.suspensions == 0; ++now) {
      if (events.emitted >= 200) {
        storage.growthBlocked = true;
        storage.allocationFailure = metal::AllocationFailure::HostPressure;
        pressure = MemoryPressure::Warning;
      }
      static_cast<void>(engine.tick(now));
    }
    require(executor.suspensions == 1 && !engine.commandInFlight() &&
                resources.snapshot().stateCache.entries == 0,
            "the decode was not suspended without its point");
    // Its KV goes too, and so do the extents it held.
    while (resources.reclaimOne(CacheReclaimMode::ReleaseExtents, ReclaimClass::InUse)
               .madeProgress) {
    }
    require(resources.snapshot().pool.pagesPrefix == 0 && pool.snapshot().pagesAllocated == 0,
            "the suspended lane's KV stayed");
    storage.growthBlocked = false;
    pressure = MemoryPressure::Normal;
    // Another conversation's KV, with no state, fills two whole extents.
    resources.beginRequest(900);
    require(resources.ensureTokens(900, 256).granted(), "the other conversation got no pages");
    resources.publishCommittedBlocks(900, other, 256, {});
    resources.endRequest(900);
    releases = storage.releasedExtents;
  }

  static EngineConfig config(const MemoryPressure &pressure) {
    EngineConfig result;
    result.growthPaused = [&pressure] { return pressure != MemoryPressure::Normal; };
    return result;
  }

  // Resumes the lane, whose replay point is planned again, and runs it to
  // its end. A snapshot needs memory that only released extents give, and
  // an extent goes only while no command is in flight (guardReleases).
  void finish() {
    executor.deniedSnapshots = 0;
    executor.decodeFinishes = true;
    now += 101;
    for (const double end = now + 1000; now < end && !idle(engine); ++now)
      static_cast<void>(engine.tick(now));
  }

  static constexpr uint64_t id = 256;
  const std::vector<uint32_t> prompt = std::vector<uint32_t>(4097, 5);
  const std::vector<uint32_t> other = std::vector<uint32_t>(257, 6);
  test::TestKvStorage storage{1024, 4096, 4};
  KvPool pool{storage, 0};
  engine::Cache resources{pool, nullptr, nullptr};
  Executor executor{1};
  Events events;
  MemoryPressure pressure = MemoryPressure::Normal;
  const test::ScopedTestConfig seam{{.prefillCheckpointTokens = 8192}};
  engine::Engine engine{config(pressure), resources, executor, events};
  double now = 1;
  // The extents released before the lane resumes.
  uint32_t releases = 0;
};

// The rebuilt point is in use, so its publication takes the other
// conversation's KV, oldest first, until the free pages fill an extent,
// which is emptied and goes at once, between commands, and the
// conversation's next turn resumes from it. An extent may hold less than a
// state needs: the publication then takes a second one the same way.
void testResumedLaneRebuildsItsPointFromCachedKv(uint32_t extentsPerState) {
  LostReplayPoint fixture;
  test::TestKvStorage &storage = fixture.storage;
  engine::Engine &engine = fixture.engine;
  engine::Cache &resources = fixture.resources;
  const uint32_t releases = fixture.releases;
  fixture.executor.stateBytes = extentsPerState * 4 * 4096;
  fixture.executor.snapshotRoom = [&] {
    return storage.releasedExtents >= releases + extentsPerState;
  };
  fixture.finish();
  const Events &events = fixture.events;
  const Executor &executor = fixture.executor;
  const auto after = engine.snapshot();
  require(idle(engine) && executor.resumptions == 1 && events.completedCount == 1 &&
              events.failedCount == 0 && after.recycledStatePublications == 1 &&
              after.resources.stateCache.inUseEvictions == 0 &&
              after.resources.stateCache.inUse == 0,
          "the resumed lane did not publish its rebuilt point");
  // For one extent the older KV paid one page: the lane's last page, alone
  // in its extent, moved there and its extent went. A second extent costs
  // the older KV four pages more. The next turn resumes from the point.
  std::vector<uint32_t> next = fixture.prompt;
  next.resize(next.size() + 40, 9);
  require(resources.lookup(fixture.other, {}).kvBoundary == (extentsPerState == 1 ? 224u : 96u) &&
              storage.copies.size() == extentsPerState &&
              storage.releasedExtents == releases + extentsPerState &&
              resources.lookup(next, {}).resumeBoundary() == 4096,
          "the rebuilt point did not come from the older KV");
}

// A snapshot that extents do not let fit is denied for a reason other than
// the budget: once the extents a publication released cover one snapshot,
// that publication ends. The resumed lane's prompt replay point, in use,
// costs the other conversation one extent, its last page moved into the
// free pages; the point its history reaches is a checkpoint, which takes
// only checkpoints. Neither point is published.
void testDeniedSnapshotTakesAtMostOneSnapshotOfExtents() {
  LostReplayPoint fixture;
  fixture.executor.stateBytes = 4 * 4096;
  fixture.executor.snapshotRoom = [] { return false; };
  fixture.finish();
  std::vector<uint32_t> next = fixture.prompt;
  next.resize(next.size() + 40, 9);
  require(idle(fixture.engine) && fixture.events.completedCount == 1 &&
              fixture.storage.releasedExtents == fixture.releases + 1 &&
              fixture.resources.lookup(fixture.other, {}).kvBoundary == 256 - 32 &&
              fixture.resources.probe(next, {}).cachedTokens() == 0,
          "a denied snapshot took more than one snapshot's worth of extents");
}

// A shared prefill's junction is an ordinary publication: with no cached
// state to recycle, it makes room from another conversation's cached KV, an
// extent at a time, and the sibling resumes from it rather than recompute
// the shared prefix.
void testSharedJunctionPublishesFromCachedKv() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  // Another conversation's KV, with no state, fills two whole extents.
  resources.beginRequest(900);
  require(resources.ensureTokens(900, 256).granted(), "the other conversation got no pages");
  resources.publishCommittedBlocks(900, std::vector<uint32_t>(257, 6), 256, {});
  resources.endRequest(900);
  // A snapshot needs memory that only a released extent gives.
  const uint32_t releases = storage.releasedExtents;
  executor.snapshotRoom = [&] { return storage.releasedExtents > releases; };
  for (uint32_t id = 1; id <= 2; ++id) {
    std::vector<uint32_t> prompt(193, 7);
    std::fill(prompt.begin() + 160, prompt.end(), id + 10);
    engine.submit(request(id, prompt));
  }
  runUntilIdle(engine);
  const auto snapshot = engine.snapshot();
  require(snapshot.junctionMaterializations == 1 &&
              snapshot.junctionMaterializationFailures == 0 && executor.restored == 160 &&
              executor.prefillRows == 193 + 33 && events.completedCount == 2,
          "the shared junction did not make room from cached KV");
}

// While growth is paused a released extent gives a snapshot nothing, so a
// replay point's publication that finds no state to recycle goes unpublished
// and leaves another conversation's cached KV where it is.
void testPausedPublicationInUseTakesNoKv() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  bool paused = false;
  EngineConfig config;
  config.growthPaused = [&] { return paused; };
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> other(257, 6);
  resources.beginRequest(900);
  require(resources.ensureTokens(900, 256).granted(), "the other conversation got no pages");
  resources.publishCommittedBlocks(900, other, 256, {});
  resources.endRequest(900);

  executor.snapshotRoom = [] { return false; };
  const uint32_t releases = storage.releasedExtents;
  engine.submit(request(1, std::vector<uint32_t>(4097, 5)));
  double now = 1;
  require(engine.tick(now++) && executor.requests.contains(1), "the request did not start");
  paused = true;
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  require(events.completedCount == 1 && events.failedCount == 0 &&
              engine.snapshot().replayStatePublicationFailures == 1 &&
              storage.releasedExtents == releases && storage.copies.empty() &&
              resources.lookup(other, {}).kvBoundary == 256,
          "a paused publication took cached KV for a snapshot it could not allocate");
}

void testPreemptedDecodeRestoresItsResidentCompositeState() {
  test::TestKvStorage storage(8, 4096, 2);
  storage.budgetPages = 6;
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  for (uint64_t id : {260, 261}) {
    auto value = request(id, std::vector<uint32_t>(65, id));
    value.maxNewTokens = 30;
    engine.submit(std::move(value));
  }
  // Lookups keep both prompts' states at 64 resident while the lanes contend
  // for pages: the preempted lane then resumes from its own.
  const std::vector<CacheLookup> held = runUntilStatesHeld(engine, resources, {260, 261});
  for (double now = 100; now < 400 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && executor.suspensions == 1 &&
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
              engine.snapshot().cacheHits == 0 &&
              engine.snapshot().coldMisses == 2,
          "internal cache restore changed output or request accounting");
}

// Once a resumed lane replays generated history, the generation prompt lies
// inside it: the lane's own recovery point stays its last whole page.
void testPreemptedDecodeReplayBoundaryIgnoresTheGenerationPrompt() {
  test::TestKvStorage storage(8, 4096, 2);
  storage.budgetPages = 6;
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  for (uint64_t id : {262, 263}) {
    auto value = request(id, std::vector<uint32_t>(65, id));
    // Long enough to move the boundary if it applied to the 89-token history
    // too: the prompt's state lands at 32, the history's at 64.
    value.generationPromptTokens = 30;
    value.maxNewTokens = 30;
    engine.submit(std::move(value));
  }
  const std::vector<CacheLookup> held = runUntilStatesHeld(engine, resources, {262, 263});
  for (double now = 100; now < 400 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && executor.suspensions == 1 &&
              executor.resumedPrompts.size() == 1 &&
              executor.resumedPrompts.front().size() == 89 &&
              events.completedCount == 2,
          "the decode was not preempted after 24 generated tokens");
  // Prompt tokens are the request id.
  const DraftContextPlan &plan =
      executor.plans.at(executor.resumedPrompts.front().front());
  require(executor.restored == 32 && plan.replayEnd == 89 &&
              plan.boundaries.size() == 2 && plan.boundaries[0].boundary == 64,
          "the resumed decode's replay boundary applied its generation prompt");
}

void testRepeatedPreemptionRespectsBackoffAndCancellation() {
  for (bool cancel : {false, true}) {
    test::TestKvStorage storage(4, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(1);
    executor.kvGrowthBlocked = &storage.growthBlocked;
    executor.unblockGrowthOnSuspend = false;
    Events events;
    EngineConfig config;
    config.growthPaused = [] { return true; };
    engine::Engine engine(config, resources, executor, events);
    guardReleases(storage, engine);
    storage.growthBlocked = true;
    storage.allocationFailure = metal::AllocationFailure::HostPressure;
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
                engine.snapshot().cacheHits + engine.snapshot().coldMisses == 1 &&
                resources.snapshot().pool.pagesActive == 0 &&
                resources.snapshot().activeRequests == 0,
            "repeated preemption retained KV or duplicated admission counters");
    if (cancel) {
      engine.cancel(270);
      static_cast<void>(engine.tick(202));
    } else {
      static_cast<void>(engine.tick(350));
    }
    require(idle(engine) && executor.requests.empty() &&
                events.completedCount == (cancel ? 1U : 0U) &&
                events.failedCount == (cancel ? 0U : 1U) &&
                events.capacityExhaustedCount == 0 && events.emitted == 0,
            "repeated preemption ignored its terminal deadline/cancel");
  }
}

// A budget denial releases one empty extent per step and retries after each;
// if that admits the request it runs, and if reclaim has nothing more to free
// the lone request fails as exhausted capacity in the same tick, since a
// release takes effect when it is made.
void testBudgetDenialRetriesAfterRelease() {
  for (bool recovers : {true, false}) {
    test::TestKvStorage storage(8, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    auto pages = pool.acquirePages(8);
    require(pages.granted(), "could not seed the KV extents");
    for (uint32_t page : pages.pages)
      pool.releasePage(page, false);
    Executor executor(1);
    executor.beginGrowthBlocked = [&] {
      return pool.snapshot().pagesAllocated != 0 || !recovers;
    };
    executor.beginAllocationFailure = metal::AllocationFailure::EngineBudget;
    Events events;
    EngineConfig config;
    const test::ScopedTestConfig seam({.resourceWaitTimeoutMilliseconds = 500.0});
    engine::Engine engine(config, cache, executor, events);
    guardReleases(storage, engine);
    engine.submit(request(284, {284}));
    static_cast<void>(engine.tick(1));
    // The denial released one empty extent per step, retrying after each:
    // the storage's two extents take two steps.
    require(executor.beginAttempts == 3,
            "a denied admission did not retry after each extent it released");
    if (recovers) {
      require(executor.requests.size() == 1 && events.failedCount == 0,
              "the retry after the release did not admit the request");
      for (double now = 2; now < 400 && !idle(engine); ++now)
        static_cast<void>(engine.tick(now));
      require(events.completedCount == 1 && events.failedCount == 0,
              "state admission did not proceed once its release made room");
    } else {
      require(pool.snapshot().pagesAllocated == 0 && idle(engine) &&
                  events.failures == std::vector<std::string>{"capacity_exhausted"},
              "a budget that nothing more frees did not fail as exhausted "
              "capacity at once");
    }
    require(idle(engine) && cache.snapshot().activeRequests == 0 &&
                executor.requests.empty(),
            "a budget denial leaked request ownership");
  }
}

// A lane's admission keeps the pooled buffers a lane starts from: its
// activation takes them, so releasing them would only make the retry allocate
// them again. KV growth has no use for them and releases them.
void testStateAdmissionKeepsThePooledLaneBuffers() {
  for (bool state : {true, false}) {
    test::TestKvStorage storage(8, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor executor(1);
    executor.pooledLaneBytes = 4096;
    Events events;
    engine::Engine engine({}, cache, executor, events);
    guardReleases(storage, engine);
    if (state) {
      executor.beginGrowthBlocked = [] { return true; };
      executor.beginAllocationFailure = metal::AllocationFailure::EngineBudget;
    } else {
      storage.growthBlocked = true;
    }
    engine.submit(request(286, {286}));
    static_cast<void>(engine.tick(1));
    require(idle(engine) &&
                events.failures == std::vector<std::string>{"capacity_exhausted"} &&
                (executor.pooledLaneBytes != 0) == state,
            state ? "a lane's admission released the pooled buffers it starts from"
                  : "KV growth kept idle state buffers it cannot use");
  }
}

// While growth is paused, a lane short of state buffers takes a cached
// state's: the oldest in RAM goes when those in RAM cover what the pool
// lacks, and none goes when they do not.
void testPausedStateAdmissionReusesCachedStates() {
  for (uint32_t lacked : {1U, 2U}) {
    test::TestKvStorage storage(64, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor executor;
    Events events;
    bool paused = false;
    EngineConfig config;
    config.growthPaused = [&] { return paused; };
    engine::Engine engine(config, cache, executor, events);
    guardReleases(storage, engine);
    engine.submit(request(287, std::vector<uint32_t>(65, 287)));
    runUntilIdle(engine);
    const auto cached = cache.snapshot().stateCache;
    require(cached.entries == 1, "fixture did not cache one state");

    paused = true;
    // Each state the cache gives up returns what the pool lacks of one.
    executor.statesLacked = [&] {
      const uint64_t given = cache.snapshot().stateCache.evictions - cached.evictions;
      return lacked > given ? static_cast<uint32_t>(lacked - given) : 0;
    };
    executor.beginGrowthBlocked = [&] { return executor.statesLacked() != 0; };
    executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
    engine.submit(request(288, {288}));
    for (double now = 10; now < 20 && !idle(engine); ++now)
      static_cast<void>(engine.tick(now));
    const auto after = cache.snapshot().stateCache;
    if (lacked == 1) {
      require(idle(engine) && events.completedCount == 2 && events.failedCount == 0 &&
                  after.evictions == cached.evictions + 1,
              "a paused admission did not take the cached state's buffers");
    } else {
      require(!idle(engine) && events.completedCount == 1 &&
                  engine.snapshot().scheduler.waitingResources == 1 &&
                  after.entries == 1 && after.evictions == cached.evictions,
              "a paused admission evicted a state that could not cover its lane");
    }
  }
}

// A prefill chunk that needs more new extents than the budget has room for
// takes that room once: its retry, after one reclaim step evicts the cached
// prefixes it lacks, reuses the extents the denied attempt allocated rather
// than releasing and allocating them again.
void testDeniedGrowthAllocatesEachExtentOnce() {
  constexpr uint32_t cachedExtents = 24;
  constexpr uint32_t budgetExtents = 32;
  test::TestKvStorage storage(4 * 128, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({}, cache, executor, events);
  guardReleases(storage, engine);
  for (uint32_t chain = 0; chain < cachedExtents; ++chain) {
    const uint64_t id = 5000 + chain;
    cache.beginRequest(id);
    require(cache.ensureTokens(id, 128).granted(), "could not seed a cached prefix");
    cache.publishCommittedBlocks(id, std::vector<uint32_t>(129, 7000 + chain), 128, {});
    cache.endRequest(id);
  }
  storage.growthAllowed = [&](uint32_t) {
    return pool.snapshot().pagesAllocated / 4 < budgetExtents;
  };
  const KvPoolSnapshot before = pool.snapshot();
  const uint32_t attemptsBefore = storage.allocationAttempts;
  const uint32_t blocksBefore = cache.snapshot().pool.pagesPrefix;
  // The first chunk's 64 pages need 16 new extents; the budget has room for
  // 8, and evicting cached prefixes frees the other 32 pages.
  engine.submit(request(286, std::vector<uint32_t>(16 * 4 * 32 + 1, 286)));
  static_cast<void>(engine.tick(1));
  const KvPoolSnapshot after = pool.snapshot();
  require(executor.prefillRows == 2048 && events.failedCount == 0,
          "the chunk was not admitted once evictions made room");
  require(after.extentAllocations - before.extentAllocations ==
                  budgetExtents - cachedExtents &&
              after.extentReleases == before.extentReleases,
          "denied attempts allocated and released the same extents again");
  // One denied attempt, one reclaim step that evicts the shortfall and no
  // more, and a retry the free pages cover.
  require(storage.allocationAttempts - attemptsBefore ==
                  budgetExtents - cachedExtents + 1 &&
              blocksBefore - cache.snapshot().pool.pagesPrefix == 32,
          "the denial was retried per evicted page or evicted more than it lacked");
}

// A lone request whose chunk needs more extents than the budget holds, with
// nothing to evict, fails as exhausted capacity at once: the extents its
// denied growth allocated stay for later requests, and keeping them is no
// release after which the budget may recover.
void testGrowthBeyondTheBudgetFailsAtOnce() {
  test::TestKvStorage storage(4 * 32, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({}, cache, executor, events);
  guardReleases(storage, engine);
  storage.growthAllowed = [&](uint32_t) { return pool.snapshot().pagesAllocated / 4 < 8; };
  engine.submit(request(287, std::vector<uint32_t>(16 * 4 * 32 + 1, 287)));
  static_cast<void>(engine.tick(1));
  const KvPoolSnapshot after = pool.snapshot();
  require(idle(engine) && executor.prefillRows == 0 &&
              events.failures == std::vector<std::string>{"capacity_exhausted"},
          "a request beyond the budget did not fail at once");
  require(after.extentAllocations == 8 && after.extentReleases == 0 &&
              after.reclaimableBytes == 8 * 4 * 4096,
          "the denied growth did not keep the extents it allocated");
}

// One reclaim pass releases every empty extent, however many there are,
// and reports what is left: nothing after a critical pass, no target
// otherwise. A pass short of critical keeps one extent as the runway.
void testReclaimPassReleasesEveryEmptyExtent() {
  constexpr uint32_t extents = 200;
  for (bool critical : {true, false}) {
    test::TestKvStorage storage(4 * extents, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    auto pages = pool.acquirePages(4 * extents);
    require(pages.granted(), "could not seed the KV extents");
    for (uint32_t page : pages.pages)
      pool.releasePage(page, false);
    Executor executor(1);
    Events events;
    engine::Engine engine({}, cache, executor, events);
    guardReleases(storage, engine);
    const MemoryReclaimResult result = engine.reclaimMemory({.critical = critical});
    const uint32_t kept = critical ? 0 : 1;
    require(result.outcome == (critical ? ReclaimOutcome::Exhausted
                                        : ReclaimOutcome::Untargeted) &&
                result.releasedBytes == uint64_t{extents - kept} * 4 * 4096 &&
                pool.snapshot().pagesAllocated == kept * 4 &&
                pool.snapshot().reclaimableBytes == kept * 4 * 4096,
            "a pass did not release every empty extent or report what was left");
  }
}

// A reclaim runs only between commands: while a prefill is in flight it
// throws and reclaims nothing, neither idle model state nor an extent
// nothing holds. Once the command has completed, the same reclaim takes
// both.
void testReclaimRefusesACommandInFlight() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  auto pages = pool.acquirePages(8);
  require(pages.granted(), "could not seed the KV extents");
  for (uint32_t page : pages.pages)
    pool.releasePage(page, false);
  Executor executor(1);
  executor.prefillAnchor = true;
  executor.holdPrefillUntil = std::make_shared<std::atomic<bool>>(false);
  executor.reclaimableIdleStateBytes = 64;
  Events events;
  engine::Engine engine({}, cache, executor, events);
  guardReleases(storage, engine);
  engine.submit(request(289, {289}));
  require(engine.tick(1) && engine.commandInFlight() &&
              pool.snapshot().reclaimableBytes == 4 * 4096,
          "the prefill did not stay in flight beside an empty extent");

  const MemoryReclaimDirective directive{.critical = true};
  bool refused = false;
  try {
    static_cast<void>(engine.reclaimMemory(directive));
  } catch (const std::logic_error &) {
    refused = true;
  }
  require(refused && executor.reclaimableIdleStateBytes == 64 &&
              pool.snapshot().extentReleases == 0 &&
              pool.snapshot().reclaimableBytes == 4 * 4096,
          "a reclaim ran while a command was in flight");

  *executor.holdPrefillUntil = true;
  require(engine.tick(2) && !engine.commandInFlight() &&
              events.completedCount == 1,
          "the held prefill did not complete");
  static_cast<void>(engine.reclaimMemory(directive));
  require(executor.reclaimedIdleStateBytes == 64 &&
              pool.snapshot().extentReleases == 2 &&
              pool.snapshot().pagesAllocated == 0,
          "the reclaim after the command did not take the idle memory");
}

// Three prompts of three blocks each fill pages 0 to 8, and the first
// prompt's state and blocks go, so extent 0 keeps one page. A fourth request
// continues the third prompt and keeps decoding; its new rows fill extent 0
// again, so extent 2 holds its third block alone. The fake model marks the
// pages it writes and checks them on every step (Executor::kv).
struct ScatteredPages {
  ScatteredPages() {
    model.kv = &storage;
    guardReleases(storage, engine);
    for (uint32_t id = 1; id <= 3; ++id) {
      engine.submit(request(id, prompt(1000 * id, 97)));
      runUntilIdle(engine);
    }
    for (uint32_t victim = 0; victim < 4; ++victim) {
      require(cache.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse).madeProgress,
              "the first prompt was not evicted");
    }
    EngineRequest running = request(4, prompt(3000, 129));
    running.maxNewTokens = 4;
    model.decodeFinishes = false;
    engine.submit(running);
    for (; now < 16 && events.outputs[4].empty(); ++now)
      static_cast<void>(engine.tick(now));
    const PageTableView table = cache.pageTable(4);
    const CacheSnapshot cached = cache.snapshot();
    require(!events.outputs[4].empty() &&
                std::vector<uint32_t>(table.pages.begin(), table.pages.end()) ==
                    std::vector<uint32_t>{6, 7, 8, 0, 1} &&
                cached.pool.pagesAllocated == 12 && cached.pool.pagesFree == 4,
            "compaction setup did not leave extent 2 with one held page");
  }

  static std::vector<uint32_t> prompt(uint32_t first, uint32_t tokens) {
    std::vector<uint32_t> result(tokens);
    std::iota(result.begin(), result.end(), first);
    return result;
  }

  // A fifth lane that fits only once the pool has given an extent back.
  void submitLaneTheBudgetRefuses() {
    model.beginGrowthBlocked = [this] {
      return model.lastBeginId == 5 && pool.snapshot().pagesAllocated > 8;
    };
    model.beginAllocationFailure = metal::AllocationFailure::EngineBudget;
    engine.submit(request(5, prompt(5000, 33)));
  }

  test::TestKvStorage storage{16, 4096, 4};
  KvPool pool{storage, 0};
  engine::Cache cache{pool, nullptr, nullptr};
  Executor model;
  Events events;
  engine::Engine engine{EngineConfig{}, cache, model, events};
  // The time of the next tick.
  double now = 1;
};

// A lane the budget refuses takes the pool's free pages before any cached
// block: the extent that holds the fewest pages is emptied into the others
// and released. The request whose page moved finds its rows through the
// table of its next step.
void testStateStartGathersFreePagesBeforeEvicting() {
  ScatteredPages fixture;
  engine::Cache &cache = fixture.cache;
  Events &events = fixture.events;
  const uint64_t revision = cache.pageTable(4).revision;
  const CacheSnapshot cached = cache.snapshot();
  fixture.submitLaneTheBudgetRefuses();
  for (double now = fixture.now; now < 32 && events.startIds.back() != 5; ++now)
    static_cast<void>(fixture.engine.tick(now));
  const CacheSnapshot started = cache.snapshot();
  require(events.startIds.back() == 5 && fixture.storage.copies.size() == 1 &&
              fixture.storage.copies[0].from == 8 && fixture.storage.copies[0].to == 2 &&
              started.pool.extentCompactions == 1 && started.pool.extentReleases == 1 &&
              started.pool.pagesPrefix >= cached.pool.pagesPrefix &&
              started.stateCache.entries >= cached.stateCache.entries,
          "the lane's start evicted instead of gathering the free pages");
  const PageTableView moved = cache.pageTable(4);
  require(moved.pages[2] == 2 && moved.pages[0] == 6 && moved.pages[4] == 1 &&
              moved.revision > revision,
          "the running request did not follow its moved page");
  runUntilIdle(fixture.engine);
  require(events.completedCount == 5 && events.failedCount == 0 &&
              events.outputs[4].size() == 4,
          "a request did not complete after its page moved");
}

// Pages move only between commands: a command reaches them through its page
// table and may still write them, so a copy while one is in flight throws.
// The lane the budget refuses beside a held decode waits for it, and the
// pool empties an extent for it once the decode has completed.
void testCompactionWaitsForTheCommandInFlight() {
  ScatteredPages fixture;
  engine::Engine &engine = fixture.engine;
  Events &events = fixture.events;
  double now = fixture.now;
  fixture.model.holdDecodeUntil = std::make_shared<std::atomic<bool>>(false);
  for (; now < 32 && !engine.commandInFlight(); ++now)
    static_cast<void>(engine.tick(now));
  fixture.submitLaneTheBudgetRefuses();
  for (const double end = now + 10; now < end; ++now)
    static_cast<void>(engine.tick(now));
  require(engine.commandInFlight() && fixture.storage.copies.empty() &&
              fixture.pool.snapshot().extentCompactions == 0 &&
              events.startIds.back() != 5,
          "pages moved while a command was in flight");
  *fixture.model.holdDecodeUntil = true;
  for (const double end = now + 16; now < end && events.startIds.back() != 5; ++now)
    static_cast<void>(engine.tick(now));
  require(events.startIds.back() == 5 && fixture.storage.copies.size() == 1 &&
              fixture.pool.snapshot().extentCompactions == 1,
          "the lane did not start by emptying an extent once the decode completed");
  runUntilIdle(engine);
  require(events.completedCount == 5 && events.failedCount == 0,
          "a request did not complete after the held decode");
}

void testAllocationCausesRemainDistinct() {
  for (bool stateAllocation : {false, true}) {
    for (auto reason : {metal::AllocationFailure::HostPressure,
                        metal::AllocationFailure::EngineBudget,
                        metal::AllocationFailure::DriverRejected}) {
      test::TestKvStorage storage(8, 4096, 4);
      KvPool pool(storage, 0);
      engine::Cache cache(pool, nullptr, nullptr);
      Executor executor(1);
      Events events;
      engine::Engine engine({}, cache, executor, events);
      guardReleases(storage, engine);
      storage.growthBlocked = !stateAllocation;
      storage.allocationFailure = reason;
      executor.beginGrowthBlocked = [stateAllocation] { return stateAllocation; };
      executor.beginAllocationFailure = reason;
      engine.submit(request(285, {285}));
      static_cast<void>(engine.tick(1));
      if (reason == metal::AllocationFailure::HostPressure) {
        require(!idle(engine) && events.failedCount == 0 &&
                    executor.prefillRows == 0,
                "temporary host admission failure became a terminal error");
        engine.cancel(285);
        static_cast<void>(engine.tick(2));
      } else {
        require(idle(engine) && events.failures ==
                    std::vector<std::string>{"capacity_exhausted"} &&
                    events.failureDetails.size() == 1 &&
                    !events.failureDetails[0].second &&
                    events.failureDetails[0].first.find(
                        metal::allocationFailureName(reason)) != std::string::npos,
                "allocation failure lost its cause or became retryable");
      }
      require(idle(engine) && executor.requests.empty() &&
                  cache.snapshot().activeRequests == 0,
              "allocation denial leaked request ownership");
    }
  }
}

void testRecoveryAdmitsFailedKvTargetBeforeReplaying() {
  for (bool sharePrefix : {false, true}) {
    test::TestKvStorage storage(64, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(2);
    executor.unblockGrowthOnSuspend = false;
    executor.kvGrowthBlocked = &storage.growthBlocked;
    Events events;
    EngineConfig config;
    // Models the request-sized headroom denial while global pressure is Normal.
    config.growthPaused = [] { return false; };
    engine::Engine engine(config, resources, executor, events);
    guardReleases(storage, engine);
    const std::vector<uint32_t> prompt(129, 280);
    if (sharePrefix) {
      engine.submit(request(279, std::vector<uint32_t>(65, 280)));
      runUntilIdle(engine);
    }
    storage.allocationFailure = metal::AllocationFailure::HostPressure;
    storage.growthAllowed = [&](uint32_t) { return !storage.growthBlocked; };
    auto input = request(280, prompt);
    input.returnProgress = true;
    engine.submit(std::move(input));
    // First dispatch reaches the page-aligned replay state (128 tokens).
    require(engine.tick(1) && engine.tick(2), "recovery fixture did not prefill");
    const auto cached = resources.snapshot();
    const uint32_t restored = executor.restored;
    {
      auto lookup = resources.lookup(prompt, {});
      require(lookup.resumeBoundary() == 128 &&
                  cached.pool.pagesAllocated == cached.pool.pagesActive,
              "recovery fixture needs a state backed only by active prefix pages");
    }
    storage.growthBlocked = true;
    require(engine.tick(3) && engine.snapshot().resourceSuspensions == 1,
            "request-sized host denial was treated as permanent capacity");
    require(resources.snapshot().stateCache.entries == cached.stateCache.entries &&
                resources.snapshot().stateCache.evictions == cached.stateCache.evictions,
            "active prefix pages were mistaken for idle pages and lost their state");
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
    storage.growthBlocked = false;
    require(engine.tick(403), "recovery did not resume when KV became available");
    for (double now = 404; now < 425 && !idle(engine); ++now)
      static_cast<void>(engine.tick(now));
    auto expected = reported;
    expected.push_back(129);
    require(events.progress.at(280) == expected,
            "prefill recovery duplicated or regressed progress");
    require(idle(engine) && engine.snapshot().resourceResumptions == 1 &&
                executor.restored == restored + 128 &&
                engine.snapshot().resourceReplayTokens == 1 &&
                events.outputs.at(280) == std::vector<uint32_t>{42} &&
                resources.snapshot().activeRequests == 0 &&
                resources.snapshot().pool.pagesActive == 0 &&
                events.failedCount == 0 && events.capacityExhaustedCount == 0,
            "KV recovery failed to continue cleanly after headroom recovered");
  }
}

// A resumption whose disk state fails to load goes back to waiting with the
// same KV target: its next admission, from the prefix that remains, still
// needs the pages of the dispatch that suspended it before replay runs.
void testFailedResumeRestoreKeepsTheKvTarget() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  executor.deniedSnapshots = 1000;
  executor.stateTier = std::make_shared<OffloadControl>();
  executor.stateTier->ready = true;
  Events events;
  engine::Engine engine({}, cache, executor, events);
  guardReleases(storage, engine);
  storage.allocationFailure = metal::AllocationFailure::HostPressure;
  const std::vector<uint32_t> prompt(129, 290);
  engine.submit(request(290, prompt));
  // The first dispatch reaches the replay state at 128, which goes to disk.
  require(engine.tick(1) && engine.tick(2) && executor.prefillRows == 128 &&
              engine.snapshot().diskStatePublications == 1,
          "fixture did not write the replay state to disk");
  storage.growthBlocked = true;
  require(engine.tick(3) && engine.snapshot().resourceSuspensions == 1,
          "the last prompt token's page did not suspend the request");
  storage.growthBlocked = false;
  static_cast<void>(engine.tick(103));
  require(executor.diskReads == 1 && executor.prefillRows == 128 &&
              engine.snapshot().resourceResumptions == 0,
          "resumption did not wait for its disk state");
  // The read fails and the host refuses the retry its lane; the pressure
  // controller's pass then empties the cache and releases the KV extents.
  // Afterwards one extent (128 tokens) can come back: room for replay to
  // start, not for the dispatch that suspended the request.
  executor.resumeDenied = true;
  executor.restoreControl->ready = true;
  executor.restoreControl->success = false;
  static_cast<void>(engine.tick(104));
  static_cast<void>(engine.reclaimMemory({.critical = true}));
  require(cache.snapshot().stateCache.entries == 0 && pool.snapshot().pagesAllocated == 0 &&
              engine.snapshot().resourceResumptions == 0,
          "fixture did not release the failed resumption's memory");
  executor.resumeDenied = false;
  storage.growthAllowed = [&](uint32_t) { return pool.snapshot().pagesAllocated == 0; };
  for (double now : {204.0, 304.0, 404.0}) {
    static_cast<void>(engine.tick(now));
    require(!engine.commandInFlight() && executor.prefillRows == 128 &&
                engine.snapshot().resourceResumptions == 0 &&
                engine.snapshot().resourceSuspensions == 1 &&
                cache.snapshot().activeRequests == 0,
            "replay started without the KV of the dispatch that suspended it");
  }
  storage.growthAllowed = {};
  for (double now = 504; now < 530 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && executor.prefillRows == 128 + 129 &&
              engine.snapshot().resourceResumptions == 1 &&
              events.outputs.at(290) == std::vector<uint32_t>{42} &&
              events.failedCount == 0 && cache.snapshot().activeRequests == 0,
          "request did not replay cleanly once its KV target fit");
}

void testAdmissionReopensAfterLastSuspendedRequestResumes() {
  test::TestKvStorage storage(32, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  MemoryPressure pressure = MemoryPressure::Warning;
  EngineConfig config;
  config.growthPaused = [&] { return pressure != MemoryPressure::Normal; };
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);

  storage.growthBlocked = true;
  storage.allocationFailure = metal::AllocationFailure::HostPressure;
  auto longRequest = request(271, {271});
  longRequest.maxNewTokens = 1000;
  engine.submit(std::move(longRequest));
  require(engine.tick(1) && executor.suspensions == 1,
          "recovery admission fixture did not suspend its only lane");
  pressure = MemoryPressure::Normal;
  storage.growthBlocked = false;
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
  for (double now = 115; now < 125 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && resources.snapshot().activeRequests == 0,
          "recovery admission fixture did not release its lanes");
}

void testAdmissionRespectsPriorityBeforeHashOrder() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);

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
  require(engine.tick(1) && engine.tick(2),
          "constrained request did not reach its initial mask");
  require(events.maskRequests.size() == 1 &&
              events.maskRequests[0].first == requestId &&
              events.maskRequests[0].second.empty(),
          "initial constrained mask request is malformed");
  const std::array<uint32_t, 1> initialMask{1};
  engine.provideMask(requestId, initialMask);
  // Its first token is selected, then its first cycle drafts.
  require(engine.tick(3) && engine.tick(4) && engine.tick(5) && engine.tick(6),
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
  value.constraint = ConstraintMode::TokenMask;
  value.deadlineMilliseconds = deadline;
  return value;
}

// Two Background lanes: one waits for its initial mask without a command in
// flight; the other fills the first extent and is suspended at 7, when it
// must allocate another. The host refuses its growth even as a request in
// service's or, at the hard limit, the engine's budget refuses it. The
// resource wait limit, and so the drain, is 1000 ms.
struct SuspendedBesideAMaskWait {
  explicit SuspendedBesideAMaskWait(bool hostPause) {
    guardReleases(storage, engine);
    auto resident = constrainedRequest(1, 100'000);
    resident.priority = RequestPriority::Background;
    engine.submit(std::move(resident));
    auto growing = request(2, std::vector<uint32_t>(95, 2));
    growing.priority = RequestPriority::Background;
    growing.maxNewTokens = 1000;
    growing.deadlineMilliseconds = 100'000;
    engine.submit(std::move(growing));
    for (double now = 1; now <= 4; ++now)
      static_cast<void>(engine.tick(now));
    require(events.maskRequests.size() == 1 &&
                engine.snapshot().scheduler.decoding == 1,
            "fixture lanes did not reach their mask wait and decode");
    paused = hostPause;
    storage.growthBlocked = true;
    storage.allocationFailure = paused ? metal::AllocationFailure::HostPressure
                                       : metal::AllocationFailure::EngineBudget;
    require(engine.tick(7) && executor.suspensions == 1 &&
                executor.requests.at(1).resident,
            "denied growth did not suspend the lane beside the resident");
  }

  test::TestKvStorage storage{64, 4096, 4};
  KvPool pool{storage, 0};
  engine::Cache resources{pool, nullptr, nullptr};
  Executor executor;
  Events events;
  bool paused = false;
  const test::ScopedTestConfig seam{{.resourceWaitTimeoutMilliseconds = 1000.0}};
  engine::Engine engine{{.growthPaused = [this] { return paused; }}, resources, executor,
                        events};
};

// After a suspension, admission waits for resident lanes only while memory
// stays short. Once host pressure clears with nothing failing since, the
// suspended request resumes and new work starts beside the residents. While
// pressure persists, an allocation fails again, or growth met a limit that
// only freed memory lifts, the drain lasts at most the resource wait limit;
// the suspended request then waits beside the residents, within its own
// resource wait.
void testRecoveryDrainEndsWithItsCause() {
  enum class Cause {
    PressureClears,
    KvStillShort,
    StateStillShort,
    PressurePersists,
    HardLimit
  };
  for (Cause cause : {Cause::PressureClears, Cause::KvStillShort,
                      Cause::StateStillShort, Cause::PressurePersists,
                      Cause::HardLimit}) {
    SuspendedBesideAMaskWait fixture(cause != Cause::HardLimit);
    test::TestKvStorage &storage = fixture.storage;
    engine::Cache &resources = fixture.resources;
    Executor &executor = fixture.executor;
    Events &events = fixture.events;
    bool &paused = fixture.paused;
    engine::Engine &engine = fixture.engine;
    // A request of the suspended lane's priority waits behind it.
    auto later = request(3, {3});
    later.priority = RequestPriority::Background;
    later.deadlineMilliseconds = 100'000;
    engine.submit(std::move(later));

    const bool pauseLifts = cause == Cause::PressureClears ||
                            cause == Cause::KvStillShort ||
                            cause == Cause::StateStillShort;
    if (pauseLifts)
      paused = false;
    if (cause == Cause::PressureClears || cause == Cause::StateStillShort)
      storage.growthBlocked = false;
    executor.resumeDenied = cause == Cause::StateStillShort;
    if (cause == Cause::PressureClears) {
      for (double now = 8; now < 120 && events.startIds.size() < 3; ++now)
        static_cast<void>(engine.tick(now));
      require(events.startIds == std::vector<uint64_t>({1, 2, 3}) &&
                  executor.resumptions == 1 && executor.requests.at(1).resident,
              "cleared pressure kept new work waiting for the residents");
    } else {
      // Only a lifted pause lets the suspended request retry at 107; a
      // failure there resumes the drain.
      for (double now = 8; now <= 107; ++now)
        static_cast<void>(engine.tick(now));
      const uint32_t retried = executor.resumeAttempts;
      require((retried != 0) == pauseLifts,
              "suspended request retried while the drain held");
      for (double now = 108; now < 1007; ++now)
        static_cast<void>(engine.tick(now));
      require(engine.resourceWaitSnapshot(1006).draining &&
                  executor.resumeAttempts == retried &&
                  engine.nextWakeupMilliseconds() == 1007.0,
              "drain did not hold while memory stayed short");
      static_cast<void>(engine.tick(1007));
      require(!engine.resourceWaitSnapshot(1007).draining &&
                  executor.resumeAttempts > retried &&
                  events.startIds.size() == 2,
              "drain outlasted the resource wait limit");
      for (double now = 1008; now < 2010 && events.startIds.size() < 3; ++now)
        static_cast<void>(engine.tick(now));
      require(events.failures == std::vector<std::string>{"resource_timeout"} &&
                  events.capacityExhaustedCount == 0 &&
                  events.startIds.back() == 3,
              "suspended request's wait was not bounded after the drain");
    }
    for (uint64_t id : {1, 2, 3})
      engine.cancel(id);
    for (double now = 2010; now < 2030 && !idle(engine); ++now)
      static_cast<void>(engine.tick(now));
    require(idle(engine) && resources.snapshot().activeRequests == 0,
            "recovery drain fixture leaked its lanes");
  }
}

// At the engine's limit a prefill is suspended beside two lanes that wait
// for their masks. When one of them ends, the drain stops waiting for the
// memory it gave back: the suspended request retries at once and resumes
// when that memory is enough. When it is not, its refusal starts the drain
// again.
void testRecoveryDrainEndsWhenAResidentReleasesMemory() {
  for (const bool enough : {true, false}) {
    test::TestKvStorage storage(32, 4096, 4);
    storage.budgetPages = 16;
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor;
    Events events;
    EngineConfig config;
    const test::ScopedTestConfig seam({.resourceWaitTimeoutMilliseconds = 1000.0});
    engine::Engine engine(config, resources, executor, events);
    guardReleases(storage, engine);
    // The residents hold one page and four: the large one's pages go to the
    // suspended request once it ends, the small one's page is too little.
    auto small = constrainedRequest(1, 100'000);
    auto large = constrainedRequest(3, 100'000);
    large.prompt.assign(100, 3);
    engine.submit(std::move(small));
    engine.submit(std::move(large));
    double now = 1;
    tickUntil(engine, now, [&] { return events.maskRequests.size() == 2; },
              "the residents did not wait for their masks");
    // Its first command, 416 rows, needs thirteen pages of the eleven left
    // under the engine's limit.
    auto suspended = request(2, std::vector<uint32_t>(420, 2));
    suspended.deadlineMilliseconds = 100'000;
    engine.submit(std::move(suspended));
    tickUntil(engine, now, [&] { return executor.suspensions == 1; },
              "the prefill was not suspended at the limit");
    require(engine.resourceWaitSnapshot(now).draining && executor.resumeAttempts == 0,
            "the suspension did not start the drain");
    now += 2;
    engine.cancel(enough ? 3 : 1);
    for (uint32_t step = 0; step < 4; ++step)
      static_cast<void>(engine.tick(now++));
    if (enough) {
      require(engine.snapshot().resourceResumptions == 1 && executor.requests.at(2).resident &&
                  !engine.resourceWaitSnapshot(now).draining,
              "the drain held after a resident released the memory it waited for");
    } else {
      require(executor.resumeAttempts == 1 && engine.snapshot().resourceResumptions == 0 &&
                  engine.resourceWaitSnapshot(now).draining,
              "a refusal after a resident ended did not resume the drain");
    }
    for (uint64_t id : {1, 2, 3})
      engine.cancel(id);
    tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  }
}

// A request of a strictly higher priority than every suspended one waits
// for neither the recovery drain nor the suspended lanes: it starts at once
// and finishes while the suspended lane stays suspended.
void testHigherPriorityArrivalIsNotHeldByRecovery() {
  for (const bool hostPause : {true, false}) {
    SuspendedBesideAMaskWait fixture(hostPause);
    engine::Engine &engine = fixture.engine;
    auto urgent = request(3, {3});
    urgent.priority = RequestPriority::Foreground;
    urgent.deadlineMilliseconds = 100'000;
    engine.submit(std::move(urgent));
    double now = 8;
    tickUntil(engine, now, [&] { return fixture.events.usage.contains(3); },
              "the higher priority did not finish");
    require(now < 20 && fixture.events.startIds == std::vector<uint64_t>({1, 2, 3}) &&
                fixture.events.failedCount == 0 && !fixture.executor.requests.at(2).resident,
            "recovery held back a request above every suspended one");
    for (uint64_t id : {1, 2})
      engine.cancel(id);
    tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  }
}

// Normal requests hold every lane. A Foreground arrival suspends the one that
// has done the least, without a drain, and starts in its place; once it has
// finished, the suspended lane resumes. An arrival of the residents' own
// priority suspends nothing and waits for a lane.
void testForegroundArrivalPreemptsWhenLanesAreFull() {
  for (const RequestPriority arrival : {RequestPriority::Foreground, RequestPriority::Normal}) {
    test::TestKvStorage storage(512, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor;
    executor.decodeFinishes = false;
    Events events;
    engine::Engine engine({}, resources, executor, events);
    guardReleases(storage, engine);
    // Lane 3's prompt is the shortest, and the lanes decode together.
    for (uint64_t id = 1; id <= 4; ++id) {
      auto lane = request(id, std::vector<uint32_t>(id == 3 ? 33 : 65, static_cast<uint32_t>(id)));
      lane.maxNewTokens = 1000;
      engine.submit(std::move(lane));
    }
    double now = 1;
    tickUntil(engine, now, [&] { return engine.snapshot().scheduler.decoding == 4; },
              "the lanes did not decode");
    auto arriving = request(5, std::vector<uint32_t>(9, 5));
    arriving.priority = arrival;
    engine.submit(std::move(arriving));
    if (arrival == RequestPriority::Normal) {
      for (uint32_t step = 0; step < 8; ++step)
        static_cast<void>(engine.tick(now++));
      require(engine.snapshot().prioritySuspensions == 0 && !executor.requests.contains(5) &&
                  engine.resourceWaitSnapshot(now).concurrency == 1,
              "an arrival of the residents' priority took a lane");
    } else {
      tickUntil(engine, now, [&] { return events.usage.contains(5); },
                "the Foreground arrival did not finish");
      require(engine.snapshot().prioritySuspensions == 1 &&
                  engine.snapshot().resourceSuspensions == 0 && !executor.requests.at(3).resident &&
                  !engine.resourceWaitSnapshot(now).draining,
              "the arrival did not take the lane of the least work");
      tickUntil(engine, now, [&] { return engine.snapshot().resourceResumptions == 1; },
                "the suspended lane did not resume after the arrival finished");
    }
    for (uint64_t id = 1; id <= 5; ++id)
      engine.cancel(id);
    tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
    require(events.failedCount == 0, "a lane failed");
  }
}

// While a Background lane holds memory, a Normal request's state is refused:
// the lane is suspended for it, and it starts at the next pass. Once it has
// finished, the lane resumes.
void testRefusedHigherPriorityStartPreemptsLowerResident() {
  test::TestKvStorage storage(512, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  executor.beginGrowthBlocked = [&] {
    return executor.lastBeginId == 2 && executor.requests.at(1).resident;
  };
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  auto background = request(1, std::vector<uint32_t>(65, 1));
  background.priority = RequestPriority::Background;
  background.maxNewTokens = 1000;
  engine.submit(std::move(background));
  double now = 1;
  tickUntil(engine, now, [&] { return events.outputs.contains(1); },
            "the Background lane did not decode");
  engine.submit(request(2, std::vector<uint32_t>(9, 2)));
  tickUntil(engine, now, [&] { return executor.requests.contains(2); },
            "the refused start did not take the lower lane's memory");
  require(executor.beginAttempts == 3 && engine.snapshot().prioritySuspensions == 1 &&
              !executor.requests.at(1).resident && !engine.resourceWaitSnapshot(now).draining,
          "the refused start did not suspend the lower lane");
  tickUntil(engine, now, [&] { return engine.snapshot().resourceResumptions == 1; },
            "the suspended lane did not resume");
  engine.cancel(1);
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  require(events.completedCount == 2 && events.failedCount == 0, "a request failed");
}

// Background lanes hold three lanes and a Normal producer the fourth, with
// a junction three commands in for a Normal request that shares its prefix.
// While the request waits for that prefix it takes no lane: the request it
// suspended would resume into the freed lane first and replay for nothing.
// Once the junction has landed, it takes a Background lane and starts from
// it.
void testPrefixWaiterTakesNoLaneUntilThePrefixLands() {
  test::TestKvStorage storage(512, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  for (uint64_t id = 1; id <= 3; ++id) {
    auto lane = request(id, {static_cast<uint32_t>(id)});
    lane.priority = RequestPriority::Background;
    lane.maxNewTokens = 1000;
    engine.submit(std::move(lane));
  }
  double now = 1;
  tickUntil(engine, now, [&] { return engine.snapshot().scheduler.decoding == 3; },
            "the Background lanes did not decode");
  std::vector<uint32_t> producer(5000, 7);
  std::vector<uint32_t> waiter(5000, 7);
  std::fill(producer.begin() + 4800, producer.end(), 8);
  std::fill(waiter.begin() + 4800, waiter.end(), 9);
  auto producing = request(4, producer);
  producing.maxNewTokens = 1000;
  engine.submit(std::move(producing));
  engine.submit(request(5, waiter));
  // The command that ends at the junction is the last before it lands.
  tickUntil(engine, now, [&] { return executor.prefillRows == 3 + 4800; },
            "the producer did not reach the junction");
  require(engine.snapshot().prioritySuspensions == 0 && !executor.requests.contains(5),
          "the request took a lane while it waited for a prefix");
  tickUntil(engine, now, [&] { return events.startIds.back() == 5; },
            "the request did not start once the prefix landed");
  require(engine.snapshot().prioritySuspensions == 1 &&
              events.starts.back() == 4800,
          "the request did not take a lane to start from the prefix");
  for (uint64_t id = 1; id <= 5; ++id)
    engine.cancel(id);
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  require(events.failedCount == 0, "a lane failed");
}

// A request below the priority of a running lane cannot run before that
// priority is done, so it takes no lane: with every lane taken it suspends
// no lower resident, and during recovery it leaves a free lane to the
// suspended request. It starts once the higher priority is done.
void testRequestBelowARunningPriorityTakesNoLane() {
  for (const bool recovering : {false, true}) {
    test::TestKvStorage storage(512, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor;
    executor.decodeFinishes = false;
    Events events;
    engine::Engine engine({}, resources, executor, events);
    guardReleases(storage, engine);
    // Lane 3's prompt is the shortest.
    const uint64_t lanes = recovering ? 4 : 3;
    for (uint64_t id = 1; id <= lanes; ++id) {
      auto lane = request(id, std::vector<uint32_t>(id == 3 ? 33 : 65, static_cast<uint32_t>(id)));
      lane.priority = RequestPriority::Background;
      lane.maxNewTokens = 1000;
      engine.submit(std::move(lane));
    }
    double now = 1;
    tickUntil(engine, now, [&] { return engine.snapshot().scheduler.decoding == lanes; },
              "the Background lanes did not decode");
    // A Foreground stream takes the last lane, during recovery lane 3's.
    auto stream = request(5, {5});
    stream.priority = RequestPriority::Foreground;
    stream.maxNewTokens = 1000;
    engine.submit(std::move(stream));
    tickUntil(engine, now, [&] { return events.outputs.contains(5); },
              "the Foreground stream did not decode");
    const uint64_t suspended = recovering ? 1 : 0;
    require(engine.snapshot().prioritySuspensions == suspended,
            "the Foreground stream did not take its lane");
    engine.submit(request(6, {6}));
    if (recovering)
      engine.cancel(1);
    for (uint32_t step = 0; step < 8; ++step)
      static_cast<void>(engine.tick(now++));
    require(!executor.requests.contains(6) &&
                engine.snapshot().prioritySuspensions == suspended &&
                engine.snapshot().resourceResumptions == suspended,
            "a request below a running priority took a lane");
    engine.cancel(5);
    tickUntil(engine, now, [&] { return events.outputs.contains(6); },
              "the request did not start once the higher priority was done");
    require(engine.snapshot().prioritySuspensions == suspended,
            "the request took a lane from a lower resident");
    for (uint64_t id = 1; id <= 6; ++id)
      engine.cancel(id);
    tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
    require(events.failedCount == 0, "a lane failed");
  }
}

void testConstraintMaskOverlapsInsideOneSchedulerBatch() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);

  engine.submit(constrainedRequest(200));
  advanceToOverlappedVerify(engine, executor, events, 200);
  const std::array<uint32_t, 1> verifyMask{1};
  engine.provideMask(200, verifyMask);
  require(engine.tick(7) && idle(engine) && events.completedCount == 1 &&
              events.failedCount == 0 && events.emitted == 1,
          "overlapped verify mask did not complete the owning batch");
}

// A command's engine cycle runs from the previous command's retirement while
// the engine stays busy, so it covers the host work between commands, and
// from the command's plan after a tick that found nothing to do.
void testDecodeCycleCoversHostWork() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);

  EngineRequest greedy = request(1, {1});
  greedy.maxNewTokens = 2;
  engine.submit(std::move(greedy));
  // Each command is planned 5 ms after the previous one retires and
  // retires 10 ms after its plan.
  for (double now : {10.0, 20.0, 25.0, 35.0, 40.0, 50.0})
    require(engine.tick(now), "the greedy request stalled");
  using Cycles = std::vector<std::pair<WorkKind, double>>;
  require(events.completedCount == 1 &&
              events.cycles == Cycles{{WorkKind::Prefill, 10.0},
                                      {WorkKind::Decode, 15.0},
                                      {WorkKind::Decode, 15.0}},
          "back-to-back commands were not timed from retirement to retirement");
  require(!engine.tick(60.0), "a finished request kept the engine busy");

  engine.submit(constrainedRequest(2));
  require(engine.tick(100.0) && engine.tick(110.0) &&
              events.maskRequests.size() == 1 && !engine.tick(130.0),
          "a lane waiting for its first mask kept the engine busy");
  const std::array<uint32_t, 1> mask{1};
  engine.provideMask(2, mask);
  require(engine.tick(140.0) && engine.tick(150.0) &&
              events.cycles.back() == std::pair{WorkKind::Decode, 10.0},
          "a decode planned after an idle tick was not timed from its plan");
}

// A constrained prompt asks for its first mask when its prefill completes
// and takes no decode slot until the mask arrives; its first token is then
// selected by a plan of its own, and drafting starts with the next.
void testConstrainedPrefillRequestsMaskWithoutDecodeSlot() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);

  engine.submit(constrainedRequest(1));
  require(engine.tick(1) && engine.tick(2) &&
              events.maskRequests.size() == 1 &&
              events.maskRequests[0].first == 1 &&
              events.maskRequests[0].second.empty(),
          "the prefill completion did not ask for the first mask");
  require(events.cycles == std::vector<std::pair<WorkKind, double>>{
                               {WorkKind::Prefill, 1.0}},
          "the first mask cost a command");
  for (double now = 3; now < 10; ++now)
    require(!engine.tick(now) && !engine.commandInFlight(),
            "a decode plan ran before the first mask arrived");
  const std::array<uint32_t, 1> mask{1};
  engine.provideMask(1, mask);
  require(engine.tick(10) && engine.tick(11) && events.emitted == 0 &&
              events.cycles.back().first == WorkKind::Decode &&
              !executor.overlap,
          "the first token's selection drafted or emitted a token");
  require(engine.tick(12) && executor.overlap && engine.commandInFlight(),
          "the first draft did not follow the selection");
}

void testConstraintMaskWaitHonorsCancelAndDeadline() {
  {
    test::TestKvStorage storage(8, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(1);
    Events events;
    engine::Engine engine({}, resources, executor, events);
    guardReleases(storage, engine);
    engine.submit(constrainedRequest(201));
    advanceToOverlappedVerify(engine, executor, events, 201);
    engine.cancel(201);
    const std::array<uint32_t, 1> lateMask{1};
    engine.provideMask(201, lateMask);
    engine.failRequest(201, LaneOutcome::InvalidMask, "late mask");
    require(executor.overlap->abandoned && engine.tick(7) && idle(engine) &&
                events.completedCount == 1 && events.failedCount == 0 &&
                events.emitted == 0,
            "cancelled mask wait left an active scheduler batch");
  }

  {
    test::TestKvStorage storage(8, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(1);
    Events events;
    engine::Engine engine({}, resources, executor, events);
    guardReleases(storage, engine);
    engine.submit(constrainedRequest(202, 100.0));
    advanceToOverlappedVerify(engine, executor, events, 202);
    require(engine.tick(100.0) && executor.overlap->abandoned &&
                idle(engine) && events.failedCount == 1,
            "deadline did not terminate an in-flight host mask wait");
  }
}

// A verify mask the server leaves unanswered fails its request alone once
// the limit passes: the command commits without the mask, a mask that comes
// after the limit is ignored, and the lane goes to the request waiting for
// it.
void testUnansweredVerifyMaskFailsOnlyItsRequest() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  engine.submit(constrainedRequest(203, 100'000.0));
  // The verify mask is requested at tick 6.
  advanceToOverlappedVerify(engine, executor, events, 203);
  engine.submit(request(204, {2}));
  static_cast<void>(engine.tick(5005.0));
  require(engine.commandInFlight() && !executor.overlap->abandoned &&
              events.failedCount == 0,
          "a verify mask wait ended before its limit");
  // The command still runs after the limit, so a late mask finds the
  // request in flight.
  executor.overlap->held = true;
  require(engine.tick(5006.0) && engine.commandInFlight() &&
              executor.overlap->abandoned && events.failedCount == 0,
          "an unanswered verify mask wait did not end at its limit");
  const std::array<uint32_t, 1> lateMask{1};
  engine.provideMask(203, lateMask);
  require(!executor.overlap->provided,
          "a verify mask that came after the limit reached the model");
  executor.overlap->held = false;
  require(engine.tick(5007.0) && !engine.commandInFlight() &&
              events.failures == std::vector<std::string>{"mask_timeout"} &&
              events.failureDetails.front().second && events.emitted == 0 &&
              events.completedCount == 0,
          "an unanswered verify mask did not fail its request as retryable");
  require(engine.tick(5008.0) && events.startIds.back() == 204,
          "the timed-out request's lane did not go to the waiting request");
  runUntilIdle(engine);
  require(events.completedCount == 1 && events.failedCount == 1,
          "the waiting request did not complete after the mask timeout");
}

// An initial mask the server leaves unanswered fails its request when the
// limit passes, which the engine's next wakeup names; an answered mask
// clears the limit.
void testUnansweredInitialMaskFails() {
  for (const bool answered : {false, true}) {
    test::TestKvStorage storage(8, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(1);
    Events events;
    engine::Engine engine({}, resources, executor, events);
    guardReleases(storage, engine);
    engine.submit(constrainedRequest(205, 100'000.0));
    require(engine.tick(1) && engine.tick(2) &&
                events.maskRequests.size() == 1,
            "constrained request did not reach its initial mask");
    require(engine.nextWakeupMilliseconds() == 5002.0,
            "the initial mask wait did not schedule its limit");
    if (answered) {
      const std::array<uint32_t, 1> initialMask{1};
      engine.provideMask(205, initialMask);
      require(engine.nextWakeupMilliseconds() == 100'000.0 &&
                  engine.tick(5002.0) && events.failedCount == 0,
              "an answered initial mask kept its limit");
      continue;
    }
    static_cast<void>(engine.tick(5001.0));
    require(events.failedCount == 0, "an initial mask wait ended early");
    require(engine.tick(5002.0) &&
                events.failures == std::vector<std::string>{"mask_timeout"} &&
                events.failureDetails.front().second &&
                executor.requests.empty() && idle(engine),
            "an unanswered initial mask did not fail its request");
  }
}

void testDecodeNearContextCeilingCoversVerifyRows() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  executor.decodeFinishes = false;
  Events events;
  EngineConfig config;
  config.maxContext = 64;
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);

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
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);

  engine.submit(constrainedRequest(300, 100.0));
  require(engine.tick(1) && engine.tick(2),
          "constrained request did not reach its initial mask wait");
  require(events.maskRequests.size() == 1, "initial mask was not requested");

  executor.holdDecodeUntil = std::make_shared<std::atomic<bool>>(false);
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
  for (double now = 151.0; now < 160.0 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && events.completedCount == 1 &&
              events.failedCount == 1,
          "peer request did not complete after the expired wait was removed");
}

void testOrdinaryInFlightDeadlineDrainsWithoutPublishingOrOutput() {
  for (WorkKind heldKind : {WorkKind::Prefill, WorkKind::Decode}) {
    test::TestKvStorage storage(8, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(1);
    Events events;
    engine::Engine engine({}, resources, executor, events);
    guardReleases(storage, engine);
    auto released = std::make_shared<std::atomic<bool>>(false);
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
    require(engine.tick(102) && idle(engine) && events.failedCount == 1 &&
                events.failures.front() == "deadline_exceeded" &&
                events.failureDetails.front().first ==
                    "request deadline exceeded" &&
                events.completedCount == 0 && events.emitted == 0 &&
                executor.snapshotAttempts == expectedSnapshots &&
                executor.requests.empty(),
            "expired ordinary command emitted output or lost its deadline");
    const auto after = resources.snapshot();
    require(after.stateCache.publications == before.stateCache.publications &&
                after.pool.pagesPrefix == before.pool.pagesPrefix &&
                after.pool.pagesActive == 0 && after.activeRequests == 0,
            "expired ordinary command published cache state or leaked resources");
  }
}

void testStalledSuspensionFailsWithCapacity() {
  test::TestKvStorage storage(8, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.kvGrowthBlocked = &storage.growthBlocked;
  executor.unblockGrowthOnSuspend = false;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);

  storage.growthBlocked = true;
  engine.submit(request(60, {60}));
  engine.submit(request(61, {61}));
  require(engine.tick(1.0) && engine.tick(2.0) && executor.suspensions == 1 &&
              events.capacityExhaustedCount == 1,
          "preemption did not identify that the remaining singleton cannot fit");

  // Nothing completes between suspension and retry, so suspending again would
  // only rotate the same two lanes forever.
  for (double now = 102.0; now < 140.0 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && events.capacityExhaustedCount == 2 &&
              engine.snapshot().resourceSuspensions == 1 &&
              engine.snapshot().resourceResumptions == 0 && executor.prefillRows == 0,
          "stalled suspension did not converge to a capacity failure");
}

void testTerminalAnchorWithoutKvIsNotCached() {
  for (uint32_t withoutKv : {1U, 0U}) {
    test::TestKvStorage storage(8, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(1);
    executor.decodeTokensWithoutKv = withoutKv;
    Events events;
    engine::Engine engine({}, resources, executor, events);
    guardReleases(storage, engine);

    // 31 prompt tokens plus the emitted token complete one Page32 block only
    // when that token has a stored KV row.
    engine.submit(request(70, std::vector<uint32_t>(31, 70)));
    runUntilIdle(engine);
    require(events.completedCount == 1 && events.emitted == 1,
            "terminal anchor request did not complete");
    require(resources.snapshot().pool.pagesPrefix == (withoutKv ? 0U : 1U),
            "token without a KV row changed the cached block count");
  }
}

void testPrefillCanCompleteTheRequest() {
  for (bool stop : {false, true}) {
    test::TestKvStorage storage(8, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor(1);
    executor.prefillAnchor = stop;
    Events events;
    engine::Engine engine({}, resources, executor, events);
    guardReleases(storage, engine);

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

void testOutOfVocabularyOutputFailsLaneOnly() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor model;
  Events events;
  EngineConfig config;
  config.vocabularySize = 1000;
  engine::Engine engine(config, cache, model, events);
  guardReleases(storage, engine);
  // The sampling kernels leave 0xffffffff when a logit row is entirely
  // non-finite; the engine must fail that lane before the sentinel reaches
  // the token history while the peer request completes normally.
  model.poisonRequest = 1;
  model.poisonToken = 0xFFFFFFFFU;
  engine.submit(request(1, {7, 7, 7}));
  engine.submit(request(2, {7, 7, 7}));
  runUntilIdle(engine);
  require(events.failedCount == 1,
          "out-of-vocabulary model output did not fail the lane");
  require(events.failures.size() == 1 &&
              events.failures[0] == "model_result_invalid",
          "out-of-vocabulary failure carried the wrong code");
  require(events.completedCount == 1,
          "poisoned lane took down the rest of the batch");
  require(events.outputs[1].empty() &&
              events.outputs[2] == std::vector<uint32_t>{42},
          "out-of-vocabulary token reached the event sink");
  require(events.usage.count(2) == 1,
          "clean peer request did not complete");
}

void runUntilCheckpoint(engine::Engine &engine, uint64_t publications) {
  for (uint32_t step = 0; step < 128; ++step) {
    static_cast<void>(engine.tick(step + 1));
    if (engine.snapshot().checkpointPublications >= publications)
      return;
  }
  throw std::runtime_error("engine did not publish the expected checkpoint");
}

// An engine with no test configuration plans checkpoints every
// kPrefillCheckpointTokens: the retry resumes from the second.
void testCancelledColdPrefillResumesItsLatestCheckpoint() {
  test::TestKvStorage storage(2048, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
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
  const uint32_t restored = 2 * kPrefillCheckpointTokens;
  require(events.starts.back() == restored &&
              executor.prefillRows - computed == prompt.size() - restored &&
              events.failedCount == 0,
          "cancelled cold prefill was recomputed before its completed checkpoint");
  require(resources.snapshot().stateCache.entries == 1 &&
              resources.snapshot().stateCache.checkpointEntries == 0,
          "successful retry retained a temporary recovery point");
}

void testConcurrentProgressRetainsAtMostOnePointPerLane() {
  test::TestKvStorage storage(2048, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(25001, 20);
  engine.submit(request(410, prompt));
  engine.submit(request(411, prompt));
  uint32_t maximumEntries = 0;
  for (uint32_t step = 0; step < 128 && !idle(engine); ++step) {
    static_cast<void>(engine.tick(step + 1));
    maximumEntries =
        std::max(maximumEntries, resources.snapshot().stateCache.entries);
  }
  // 24576 lies within one prefill chunk of the replay boundary at 24992.
  const uint32_t checkpoints = 5;
  require(idle(engine) && events.completedCount == 2 &&
              engine.snapshot().checkpointPublications >= checkpoints &&
              engine.snapshot().checkpointPublications <= 2 * checkpoints &&
              engine.snapshot().cacheHits == 1 &&
              executor.prefillRows == prompt.size() * 2 - 24992 &&
              maximumEntries <= 2 &&
              resources.snapshot().stateCache.entries == 1,
          "concurrent prompts accumulated progress states beyond their active lanes");
}

void testSharedCheckpointSurvivesPeerRollingReplacement() {
  test::TestKvStorage storage(2048, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  std::vector<uint32_t> original(25001, 20);
  auto background = request(412, original);
  background.priority = RequestPriority::Background;
  engine.submit(std::move(background));
  runUntilCheckpoint(engine, 1);
  std::vector<uint32_t> branch = original;
  std::fill(branch.begin() + kPrefillCheckpointTokens, branch.end(), 21);
  auto foreground = request(413, branch);
  foreground.priority = RequestPriority::Foreground;
  engine.submit(std::move(foreground));
  runUntilCheckpoint(engine, 2);
  require(resources.lookup(original, {}).resumeBoundary() == kPrefillCheckpointTokens,
          "rolling a shared checkpoint retired the paused peer's recovery point");
  engine.cancel(412);
  engine.cancel(413);
  runUntilIdle(engine);
  require(events.failedCount == 0, "shared checkpoint cancellation failed");
}

void testRepeatedRetriesRollTheRestoredCheckpoint() {
  test::TestKvStorage storage(2048, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(40001, 31);
  for (uint32_t attempt = 0; attempt < 3; ++attempt) {
    engine.submit(request(500 + attempt, prompt));
    runUntilCheckpoint(engine, attempt + 1);
    require(events.starts.back() == attempt * kPrefillCheckpointTokens &&
                resources.snapshot().stateCache.entries == 1 &&
                resources.snapshot().stateCache.checkpointEntries == 1,
            "retry promoted or accumulated intermediate checkpoints");
    engine.cancel(500 + attempt);
    runUntilIdle(engine);
  }
  engine.submit(request(503, prompt));
  runUntilIdle(engine);
  require(events.starts.back() == 3 * kPrefillCheckpointTokens &&
              executor.prefillRows == prompt.size() &&
              resources.snapshot().stateCache.entries == 1 &&
              resources.snapshot().stateCache.checkpointEntries == 0,
          "successful retry recomputed or retained superseded recovery states");
}

void testRetryCancelledBeforeNextCheckpointKeepsItsSource() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(25001, 32);
  engine.submit(request(510, prompt));
  runUntilCheckpoint(engine, 1);
  engine.cancel(510);
  runUntilIdle(engine);
  for (uint32_t id : {511U, 512U}) {
    engine.submit(request(id, prompt));
    require(engine.tick(1) && engine.commandInFlight() &&
                events.starts.back() == kPrefillCheckpointTokens,
            "retry did not restore the previous progress point");
    engine.cancel(id);
    runUntilIdle(engine);
    require(resources.lookup(prompt, {}).resumeBoundary() ==
                    kPrefillCheckpointTokens &&
                resources.snapshot().stateCache.checkpointEntries == 1,
            "cancelled retry lost or promoted its unchanged recovery point");
  }
}

// A restored endpoint becomes ordinary where it is, on disk too when no
// cache slot takes its RAM copy, and its loss is then a lost state.
void testRestoredCheckpointAtReplayEndBecomesOrdinary() {
  for (bool disk : {false, true}) {
    for (uint32_t suffix : {1U, 31U}) {
      test::TestKvStorage storage(1024, 4096, 4);
      KvPool pool(storage, 0);
      engine::Cache resources(pool, nullptr, nullptr);
      Executor executor(1);
      if (disk) {
        executor.deniedSnapshots = 1000;
        executor.stateTier = std::make_shared<OffloadControl>();
        executor.stateTier->ready = true;
        executor.restoreControl->ready = true;
        executor.restoreControl->promotionDenied = true;
      }
      Events events;
      engine::Engine engine({}, resources, executor, events);
      guardReleases(storage, engine);
      const std::vector<uint32_t> prompt(25001, 33);
      engine.submit(request(520, prompt));
      runUntilCheckpoint(engine, 1);
      engine.cancel(520);
      runUntilIdle(engine);
      const uint32_t snapshots = executor.snapshots + executor.diskSnapshots;
      std::vector<uint32_t> shorter(
          prompt.begin(), prompt.begin() + kPrefillCheckpointTokens + suffix);
      engine.submit(request(521, shorter));
      runUntilIdle(engine);
      const auto states = resources.snapshot().stateCache;
      require(events.starts.back() == kPrefillCheckpointTokens &&
                  executor.snapshots + executor.diskSnapshots == snapshots &&
                  engine.snapshot().deduplicatedStatePublications == 1 &&
                  states.entries == 1 && states.checkpointEntries == 0 &&
                  states.promotionsSkipped == (disk ? 1 : 0) &&
                  resources.lookup(shorter, {}).resumeBoundary() ==
                      kPrefillCheckpointTokens,
              "restored replay endpoint was copied or retired as temporary");
      auto held = resources.lookup(shorter, {});
      const uint64_t block = held.state->kvBlock();
      const CompositeState *copy = held.state->state().get();
      held = {};
      resources.discardState(block, copy);
      require(resources.lookup(shorter, {}).lostState,
              "the loss of a restored replay endpoint was not a lost state");
    }
  }
}

void testRetryRetiresCheckpointAtDeeperJunction() {
  test::TestKvStorage storage(2048, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  const test::ScopedTestConfig seam({.prefillCheckpointTokens = 8192});
  engine::Engine engine({}, resources, executor,
                        events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(30001, 34);
  engine.submit(request(530, prompt));
  runUntilCheckpoint(engine, 2);
  require(engine.tick(50) && engine.tick(51) && !engine.commandInFlight() &&
              executor.requests.at(530).position == 18432,
          "fixture did not commit past the last recovery point");
  engine.cancel(530);
  runUntilIdle(engine);
  // Another conversation leaves the prompt at 18432 and goes on to a state
  // of its own, so the retry's matched KV ends at a branch point.
  std::vector<uint32_t> branch(prompt.begin(), prompt.begin() + 18432);
  branch.resize(18432 + KvCache::pageTokens, 35);
  resources.beginRequest(999);
  require(resources.ensureTokens(999, branch.size()).granted(), "branch fixture KV failed");
  resources.publishCompositeState(
      test::publishBlocks(resources, 999, branch, static_cast<uint32_t>(branch.size())),
      std::make_shared<State>(), false);
  resources.endRequest(999);
  engine.submit(request(531, prompt));
  require(engine.tick(1) && engine.tick(2) &&
              events.starts.back() == 16384 &&
              engine.snapshot().junctionMaterializations == 1 &&
              resources.snapshot().stateCache.entries == 2 &&
              resources.snapshot().stateCache.checkpointEntries == 0,
          "retry kept an earlier recovery point after publishing its junction");
  runUntilIdle(engine);
  require(resources.snapshot().stateCache.entries == 3 &&
              resources.snapshot().stateCache.checkpointEntries == 0,
          "retry did not retain its normal junction and replay states");
}

void testPinnedCheckpointSkipsReplacementButNotOrdinaryState() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(18001, 35);
  engine.submit(request(540, prompt));
  runUntilCheckpoint(engine, 1);
  auto pinned = resources.lookup(prompt, {});
  require(pinned.resumeBoundary() == kPrefillCheckpointTokens,
          "fixture did not pin its checkpoint");
  runUntilIdle(engine);
  // The checkpoints at 8192 and 12288 cannot retire the pinned one; 16384
  // lies within one prefill chunk of the replay boundary at 17984.
  require(engine.snapshot().checkpointPublications == 1 &&
              engine.snapshot().checkpointPublicationFailures == 2 &&
              executor.snapshotAttempts == 2 &&
              resources.snapshot().stateCache.entries == 2 &&
              resources.snapshot().stateCache.checkpointEntries == 1 &&
              resources.lookup(prompt, {}).resumeBoundary() == 17984,
          "pinned recovery point was overwritten or blocked ordinary publication");
  pinned = {};
  require(resources.reclaimOneState(false, 0, false) &&
              resources.snapshot().stateCache.checkpointEntries == 0 &&
              resources.lookup(prompt, {}).resumeBoundary() == 17984,
          "released recovery pin did not rejoin the lower-priority queue");
}

void testFailedReplacementContinuesWithoutRecoveryPoint() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(25001, 36);
  engine.submit(request(550, prompt));
  runUntilCheckpoint(engine, 1);
  executor.denySnapshotAtBoundary = 2 * kPrefillCheckpointTokens;
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
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(25001, 37);
  engine.submit(request(560, prompt));
  runUntilCheckpoint(engine, 1);
  {
    auto existing = resources.lookup(prompt, {});
    require(resources.reuseCompositeState(existing.state->kvBlock(), false),
            "shared ordinary boundary could not reuse its checkpoint");
  }
  runUntilIdle(engine);
  const std::vector<uint32_t> prefix(
      prompt.begin(), prompt.begin() + kPrefillCheckpointTokens + 1);
  require(resources.snapshot().stateCache.entries == 2 &&
              resources.snapshot().stateCache.checkpointEntries == 0 &&
              resources.lookup(prefix, {}).resumeBoundary() ==
                  kPrefillCheckpointTokens,
          "old rolling handle deleted a state promoted by another request");
}

void testCheckpointDenialPreservesUnrelatedHotState() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> hot(65, 21);
  engine.submit(request(420, hot));
  runUntilIdle(engine);
  executor.denySnapshotAtBoundary = kPrefillCheckpointTokens;
  engine.submit(request(421, std::vector<uint32_t>(18001, 22)));
  for (uint32_t step = 0; step < 32 &&
       !engine.snapshot().checkpointPublicationFailures; ++step)
    static_cast<void>(engine.tick(step + 1));
  require(engine.snapshot().checkpointPublicationFailures == 1 &&
              resources.snapshot().stateCache.evictions == 0 &&
              resources.lookup(hot, {}).resumeBoundary() == 64,
          "optional progress allocation displaced an unrelated hot prefix");
  engine.cancel(421);
  runUntilIdle(engine);
}

void testCheckpointRecyclesItsBufferBeforeReplacement() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
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
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  const test::ScopedTestConfig seam({.prefillCheckpointTokens = 8192});
  engine::Engine engine({}, resources, executor,
                        events);
  guardReleases(storage, engine);
  engine.submit(request(440, std::vector<uint32_t>(18001, 24)));
  for (uint32_t step = 0; step < 32; ++step) {
    static_cast<void>(engine.tick(step + 1));
    if (!engine.commandInFlight() && executor.requests.at(440).position == 6144)
      break;
  }
  executor.holdPrefillUntil = std::make_shared<std::atomic<bool>>(false);
  static_cast<void>(engine.tick(40));
  require(engine.commandInFlight() && executor.requests.at(440).position == 8192,
          "fixture did not stop inside the checkpoint command");
  engine.cancel(440);
  *executor.holdPrefillUntil = true;
  runUntilIdle(engine);
  require(executor.snapshotAttempts == 0 &&
              engine.snapshot().checkpointPublications == 0 &&
              resources.snapshot().pool.pagesPrefix == 6144 / KvCache::pageTokens,
          "cancellation published a checkpoint from the draining command");
}

void testFinalStateRecyclesItsCheckpointBeforeUnrelatedHotState() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> hot(65, 27);
  engine.submit(request(460, hot));
  runUntilIdle(engine);
  const std::vector<uint32_t> prompt(18001, 28);
  engine.submit(request(461, prompt));
  // The last checkpoint is at 12288: 16384 lies within one prefill chunk of
  // the replay boundary at 17984.
  runUntilCheckpoint(engine, 3);
  executor.snapshotObserver = [&] {
    require(resources.snapshot().stateCache.entries == 1 &&
                resources.lookup(hot, {}).resumeBoundary() == 64,
            "final state did not recycle its checkpoint before allocating");
  };
  runUntilIdle(engine);
  require(resources.snapshot().stateCache.entries == 2 &&
              resources.lookup(hot, {}).resumeBoundary() == 64,
          "final state evicted unrelated hot state before its own checkpoint");
}

// A junction is an ordinary state: publishing one retires the lane's
// earlier progress point, as its replay state does.
void testJunctionRetiresEarlierProgressPoint() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(20001, 29);
  // Another conversation leaves the prompt at 18432 and goes on to a state
  // of its own: the request plans a junction at that branch point, past its
  // checkpoint at 16384.
  std::vector<uint32_t> branch(prompt.begin(), prompt.begin() + 18432);
  branch.resize(18432 + KvCache::pageTokens, 30);
  resources.beginRequest(470);
  require(resources.ensureTokens(470, branch.size()).granted(),
          "junction fixture could not allocate its KV prefix");
  resources.publishCompositeState(
      test::publishBlocks(resources, 470, branch, static_cast<uint32_t>(branch.size())),
      std::make_shared<State>(), false);
  resources.endRequest(470);
  engine.submit(request(471, prompt));
  double now = 1;
  tickUntil(engine, now, [&] { return engine.snapshot().junctionMaterializations == 1; },
            "the request did not publish its junction");
  require(engine.snapshot().checkpointPublications == 4 &&
              resources.snapshot().stateCache.checkpointEntries == 0 &&
              resources.snapshot().stateCache.entries == 2,
          "the junction left a superseded progress checkpoint resident");
  runUntilIdle(engine);
  require(resources.snapshot().stateCache.entries == 3 &&
              resources.lookup(prompt, {}).resumeBoundary() == 20000,
          "the request did not publish its replay state after the junction");
}

void testShortSuffixContinuesCheckpointDraftState() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(18001, 30);
  engine.submit(request(480, prompt));
  runUntilCheckpoint(engine, 1);
  engine.cancel(480);
  runUntilIdle(engine);
  std::vector<uint32_t> shorter(
      prompt.begin(), prompt.begin() + kPrefillCheckpointTokens + 209);
  engine.submit(request(481, shorter));
  runUntilIdle(engine);
  require(events.starts.back() == kPrefillCheckpointTokens &&
              executor.restoredDraft &&
              draftContextRows(executor.plans.at(481)) == 209 &&
              executor.plans.at(481).restoresDraftState,
          "short checkpoint suffix discarded or rebuilt its restored draft window");
}

void testDefaultCheckpointRestoresLatestCommittedPrefix() {
  test::TestKvStorage storage(2048, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
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
              committed.resources.pool.pagesPrefix == 22528 / KvCache::pageTokens,
          "fixture did not commit 22528 prompt tokens before cancellation");
  require(committed.checkpointPublications == 5 &&
              committed.checkpointPublicationFailures == 0 &&
              committed.resources.stateCache.entries == 1 &&
              committed.resources.stateCache.checkpointEntries == 1 &&
              committed.resources.stateCache.checkpointRetirements == 4,
          "default checkpoints accumulated or lost temporary states");
  {
    const auto lookup = resources.lookup(donor, {});
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
              events.starts.back() == 20480 &&
              executor.prefillRows - 22528 == 2049 &&
              finished.completed == 1 && finished.cancelled == 1 &&
              events.failedCount == 0,
          "branch replayed before its latest checkpoint or failed completion");
  require(finished.checkpointPublications == 5 &&
              finished.replayStatePublications == 1 &&
              finished.resources.stateCache.entries == 1 &&
              finished.resources.stateCache.checkpointEntries == 0 &&
              finished.resources.stateCache.checkpointRetirements == 5 &&
              resources.lookup(branch, {}).resumeBoundary() == 22528,
          "branch completion retained its superseded temporary checkpoint");
}

// A checkpoint costs a command split and a snapshot, so none is planned
// within one prefill chunk of where a request resumes or of its replay
// boundary: not for a follow-up resuming at 2976 whose replay boundary is
// 4192, nor for one resuming at 4064.
void testCheckpointsSkipNearResumeAndReplayBoundaries() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  const auto checkpointPlanned = [&](uint64_t id) {
    const auto &boundaries = executor.plans.at(id).boundaries;
    return std::any_of(boundaries.begin(), boundaries.end(),
                       [](const DraftBoundaryPlan &boundary) {
                         return boundary.boundary == kPrefillCheckpointTokens;
                       });
  };
  std::vector<uint32_t> prompt(3009);
  std::iota(prompt.begin(), prompt.end(), 1);
  EngineRequest first = request(1, prompt);
  // Its replay point lands at 2976, before the generation prompt.
  first.generationPromptTokens = 7;
  engine.submit(std::move(first));
  runUntilIdle(engine);
  const uint32_t snapshots = executor.snapshots;
  prompt.resize(3002);
  prompt.resize(4201, 500);
  engine.submit(request(2, prompt));
  runUntilIdle(engine);
  require(events.starts.back() == 2976 &&
              !checkpointPlanned(2) && engine.snapshot().checkpointPublications == 0 &&
              executor.snapshots == snapshots + 1,
          "a follow-up checkpointed within a chunk of its resume point or replay boundary");

  std::vector<uint32_t> resumed(4065);
  std::iota(resumed.begin(), resumed.end(), 10'000);
  engine.submit(request(3, resumed));
  runUntilIdle(engine);
  resumed.resize(8193, 600);
  engine.submit(request(4, resumed));
  runUntilIdle(engine);
  require(events.starts.back() == 4064 &&
              !checkpointPlanned(4) && engine.snapshot().checkpointPublications == 0,
          "a request checkpointed within a chunk of the state it resumed from");
}

void testDecodeShareValidation() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(1);
  Events events;
  for (double invalid : {-0.5, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    bool rejected = false;
    try {
      engine::Engine engine({.decodeShare = invalid}, resources, executor,
                            events);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "invalid decode share was accepted");
  }
}

} // namespace

// With no cache slot and nothing to recycle, a lane writes its state straight
// to disk and the next lane over the prefix restores it. While that write is
// in flight another lane's boundary is refused; the first lane still decodes
// then, so its KV is no cached KV that boundary could make room with.
// Long-prefill checkpoints can use free disk quota until a later state
// replaces them.
void testStateWithoutACacheSlotGoesToDisk() {
  test::TestKvStorage storage(1024, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  executor.deniedSnapshots = 1000;
  executor.stateTier = std::make_shared<OffloadControl>();
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine({}, cache, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> first(65, 1);
  auto decoding = request(1, first);
  decoding.maxNewTokens = 4;
  engine.submit(std::move(decoding));
  double now = 1;
  tickUntil(engine, now, [&] { return engine.snapshot().diskStatePublications == 1; },
            "the first lane did not publish its state");
  auto counters = engine.snapshot();
  auto states = cache.snapshot().stateCache;
  require(executor.diskSnapshots == 1 && counters.replayStatePublications == 1 &&
              counters.diskStatePublications == 1 &&
              counters.replayStatePublicationFailures == 0 &&
              counters.recycledStatePublications == 0 && states.entries == 1 &&
              states.bytes == 0 && states.diskBytes == 64 && states.offloads == 1,
          "the state did not go straight to disk");
  // The write is still in flight: the next boundary finds the staging
  // buffer busy and is not retried.
  engine.submit(request(2, std::vector<uint32_t>(65, 2)));
  tickUntil(engine, now,
            [&] { return engine.snapshot().replayStatePublicationFailures == 1; },
            "the second lane did not reach its boundary");
  require(events.completedCount == 0, "the first lane ended before the second boundary");
  runUntilIdle(engine);
  counters = engine.snapshot();
  require(executor.diskSnapshots == 1 && counters.diskStatePublications == 1 &&
              counters.replayStatePublicationFailures == 1 &&
              events.completedCount == 2 && events.failedCount == 0,
          "a second write started beside the one in flight");
  executor.stateTier->ready = true;
  static_cast<void>(engine.tick(1000));
  states = cache.snapshot().stateCache;
  require(states.entries == 1 && states.diskBytes == 64 && states.offloadFailures == 0,
          "the write did not land");
  // The prefix comes back from disk.
  engine.submit(request(3, first));
  static_cast<void>(engine.tick(1001));
  require(executor.diskReads == 1 && executor.prefillRows == 130,
          "the disk state was not read");
  executor.restoreControl->ready = true;
  runUntilIdle(engine);
  require(executor.restored == 64 && executor.prefillRows == 131 &&
              events.starts.back() == 64 &&
              events.failedCount == 0,
          "the prefix was not restored from the disk state");
  // A rolling checkpoint denied a cache slot goes to disk like any state,
  // and the replay boundary of the same prompt retires it from there.
  engine.submit(request(4, std::vector<uint32_t>(2 * kPrefillCheckpointTokens + 1, 4)));
  runUntilIdle(engine);
  counters = engine.snapshot();
  require(counters.checkpointPublications == 1 &&
              counters.checkpointPublicationFailures == 0 &&
              counters.diskStatePublications == 3 && executor.diskSnapshots == 3 &&
              counters.resources.stateCache.checkpointEntries == 0 &&
              counters.resources.stateCache.checkpointRetirements == 1 &&
              events.completedCount == 4 && events.failedCount == 0,
          "the checkpoint did not go to disk and come back out");
}

// A foreground arrival does not wait for a background producer of the same
// prompt, so both lanes compute it. Without a cache slot the first replay
// state to land goes to disk, and the other lane's publication at the same
// block finds it there: it is deduplicated, not counted as a write.
void testStateAlreadyOnDiskIsDeduplicated() {
  test::TestKvStorage storage(512, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  executor.deniedSnapshots = 1000;
  executor.stateTier = std::make_shared<OffloadControl>();
  executor.stateTier->ready = true;
  Events events;
  engine::Engine engine({}, cache, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(5001, 7);
  auto producer = request(1, prompt);
  producer.priority = RequestPriority::Background;
  engine.submit(std::move(producer));
  static_cast<void>(engine.tick(0));
  auto duplicate = request(2, prompt);
  duplicate.priority = RequestPriority::Foreground;
  engine.submit(std::move(duplicate));
  runUntilIdle(engine);
  require(events.completedCount == 2 && events.failedCount == 0 &&
              executor.prefillRows == 2 * prompt.size(),
          "both lanes did not compute the prompt");
  const auto counters = engine.snapshot();
  require(executor.diskSnapshots == 1 && counters.diskStatePublications == 1 &&
              counters.replayStatePublications == 1 &&
              counters.deduplicatedStatePublications == 1,
          "a state already on disk was counted as written");
}

// With no cache slot, a long prefill's rolling checkpoint goes to disk and
// outlives the request that made it: a branch off the same prompt restores
// it from disk instead of replaying from the start.
void testCancelledPrefillRecoversFromItsDiskCheckpoint() {
  test::TestKvStorage storage(512, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor(1);
  executor.deniedSnapshots = 1000;
  executor.stateTier = std::make_shared<OffloadControl>();
  executor.stateTier->ready = true;
  Events events;
  engine::Engine engine({}, cache, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> donor(2 * kPrefillCheckpointTokens + 1, 61);
  engine.submit(request(600, donor));
  for (uint32_t step = 0; step < 128; ++step) {
    static_cast<void>(engine.tick(step + 1));
    if (!engine.commandInFlight() && executor.requests.at(600).position == 6144)
      break;
  }
  auto counters = engine.snapshot();
  require(!engine.commandInFlight() && executor.requests.at(600).position == 6144 &&
              counters.checkpointPublications == 1 && counters.diskStatePublications == 1 &&
              counters.resources.stateCache.checkpointEntries == 1 &&
              counters.resources.stateCache.bytes == 0,
          "the checkpoint did not go to disk");
  engine.cancel(600);
  runUntilIdle(engine);
  require(engine.snapshot().cancelled == 1 &&
              cache.snapshot().stateCache.checkpointEntries == 1,
          "cancellation removed the disk checkpoint");
  std::vector<uint32_t> branch(donor.begin(), donor.begin() + 6145);
  branch.back() = 62;
  engine.submit(request(601, branch));
  static_cast<void>(engine.tick(1000));
  require(executor.diskReads == 1, "the disk checkpoint was not read");
  executor.restoreControl->ready = true;
  runUntilIdle(engine);
  counters = engine.snapshot();
  require(events.starts.back() == 4096 &&
              executor.restored == 4096 && executor.prefillRows == 6144 + 2049 &&
              counters.completed == 1 && events.failedCount == 0 &&
              counters.resources.stateCache.checkpointEntries == 0 &&
              counters.resources.stateCache.checkpointRetirements == 1,
          "the branch did not resume from the disk checkpoint");
}

// A disk checkpoint frees no cache slot for the final state, so it stays the
// lane's recovery point until that state is published. When another lane's
// write holds the staging buffer at the final boundary, the request keeps
// its checkpoint, and the next turn resumes from it whether the request was
// cancelled or completed.
void testFailedFinalStateKeepsTheDiskCheckpoint() {
  for (bool cancel : {true, false}) {
    test::TestKvStorage storage(512, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor executor;
    executor.deniedSnapshots = 1000;
    executor.stateTier = std::make_shared<OffloadControl>();
    executor.stateTier->ready = true;
    Events events;
    engine::Engine engine({}, cache, executor, events);
    guardReleases(storage, engine);
    const std::vector<uint32_t> prompt(2 * kPrefillCheckpointTokens + 1, 65);
    EngineRequest decoding = request(610, prompt);
    decoding.maxNewTokens = 64;
    engine.submit(decoding);
    for (uint32_t step = 0; step < 128; ++step) {
      static_cast<void>(engine.tick(step + 1));
      if (!engine.commandInFlight() && executor.requests.at(610).position == 6144)
        break;
    }
    auto counters = engine.snapshot();
    require(!engine.commandInFlight() && executor.requests.at(610).position == 6144 &&
                counters.checkpointPublications == 1 && counters.diskStatePublications == 1 &&
                counters.resources.stateCache.checkpointEntries == 1,
            "the checkpoint did not go to disk");
    executor.stateTier->ready = false;
    engine.submit(request(611, std::vector<uint32_t>(65, 66)));
    for (uint32_t step = 0; step < 128 && !engine.snapshot().replayStatePublicationFailures;
         ++step)
      static_cast<void>(engine.tick(1000 + step));
    counters = engine.snapshot();
    require(counters.replayStatePublications == 1 &&
                counters.replayStatePublicationFailures == 1 &&
                counters.diskStatePublications == 2 && executor.diskSnapshots == 2 &&
                events.failedCount == 0,
            "the final state was not refused beside the other lane's write");
    require(counters.resources.stateCache.checkpointEntries == 1 &&
                counters.resources.stateCache.checkpointRetirements == 0,
            "the final boundary retired the checkpoint it could not replace");
    if (cancel)
      engine.cancel(610);
    executor.stateTier->ready = true;
    runUntilIdle(engine);
    require(engine.snapshot().completed == (cancel ? 1U : 2U) &&
                cache.snapshot().stateCache.checkpointEntries == 1,
            "the request's end removed the checkpoint its final state did not replace");
    engine.submit(request(612, prompt));
    static_cast<void>(engine.tick(2000));
    executor.restoreControl->ready = true;
    runUntilIdle(engine);
    counters = engine.snapshot();
    require(events.starts.back() == 4096 &&
                executor.restored == 4096 && counters.cancelled == (cancel ? 1U : 0U) &&
                counters.completed == (cancel ? 2U : 3U) && events.failedCount == 0 &&
                counters.resources.stateCache.checkpointEntries == 0 &&
                counters.resources.stateCache.checkpointRetirements == 1,
            "the next turn did not resume from the disk checkpoint and replace it");
  }
}

// No checkpoint is planned within one prefill chunk of the replay boundary,
// whether a cache slot would take it or only the disk: the replay state lands
// in the next command. One a chunk or more away is published, on disk when no
// cache slot takes it, and the replay state retires it.
void testNoCheckpointWithinAChunkOfTheReplayBoundary() {
  const uint32_t chunk = model::ExecutionLimits::prefillTokenBudget;
  for (bool denyRam : {false, true}) {
    for (uint32_t remaining : {32u, chunk - 32, chunk}) {
      test::TestKvStorage storage(512, 4096, 4);
      KvPool pool(storage, 0);
      engine::Cache cache(pool, nullptr, nullptr);
      Executor executor;
      executor.deniedSnapshots = denyRam ? 1000 : 0;
      executor.stateTier = std::make_shared<OffloadControl>();
      executor.stateTier->ready = true;
      Events events;
      engine::Engine engine({}, cache, executor, events);
      guardReleases(storage, engine);
      const std::vector<uint32_t> prompt(kPrefillCheckpointTokens + remaining + 1, 71);
      engine.submit(request(1, prompt));
      runUntilIdle(engine);
      const auto counters = engine.snapshot();
      const uint64_t checkpoint = remaining >= chunk;
      require(counters.checkpointPublications == checkpoint &&
                  counters.checkpointPublicationFailures == 0 &&
                  counters.replayStatePublications == 1 &&
                  counters.replayStatePublicationFailures == 0 &&
                  executor.diskSnapshots == (denyRam ? 1 + checkpoint : 0) &&
                  counters.resources.stateCache.entries == 1 &&
                  counters.resources.stateCache.checkpointEntries == 0 &&
                  events.completedCount == 1 && events.failedCount == 0,
              "a checkpoint within a chunk of the replay boundary was planned or the final "
              "state was lost");
      executor.restoreControl->ready = true;
      engine.submit(request(2, prompt));
      runUntilIdle(engine);
      require(events.starts.back() == prompt.size() - 1 &&
                  events.completedCount == 2 && events.failedCount == 0,
              "skipping a checkpoint lost the final reusable prefix");
    }
  }
}

// The multiple of the interval within one prefill chunk of the replay
// boundary is not planned: a lane cancelled past it, in RAM or with its
// checkpoint on disk, resumes from the checkpoint before it.
void testSkippedCheckpointKeepsPreviousRecoveryPoint() {
  for (bool disk : {false, true}) {
    test::TestKvStorage storage(512, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor executor(1);
    executor.deniedSnapshots = disk ? 1000 : 0;
    executor.stateTier = std::make_shared<OffloadControl>();
    executor.stateTier->ready = true;
    executor.restoreControl->ready = true;
    Events events;
    engine::Engine engine({}, cache, executor, events);
    guardReleases(storage, engine);
    const std::vector<uint32_t> prompt(2 * kPrefillCheckpointTokens + 33, 73);
    engine.submit(request(1, prompt));
    for (uint32_t step = 0; step < 128; ++step) {
      static_cast<void>(engine.tick(step + 1));
      if (!engine.commandInFlight() &&
          executor.requests.at(1).position == 2 * kPrefillCheckpointTokens)
        break;
    }
    const auto before = engine.snapshot();
    require(!engine.commandInFlight() &&
                executor.requests.at(1).position == 2 * kPrefillCheckpointTokens &&
                std::none_of(executor.plans.at(1).boundaries.begin(),
                             executor.plans.at(1).boundaries.end(),
                             [](const DraftBoundaryPlan &boundary) {
                               return boundary.boundary == 2 * kPrefillCheckpointTokens;
                             }) &&
                before.checkpointPublications == 1 &&
                before.checkpointPublicationFailures == 0 &&
                before.resources.stateCache.checkpointEntries == 1 &&
                before.resources.stateCache.checkpointRetirements == 0 &&
                executor.diskSnapshots == (disk ? 1 : 0),
            "a near-final checkpoint was planned or replaced its predecessor");
    engine.cancel(1);
    runUntilIdle(engine);
    engine.submit(request(2, prompt));
    runUntilIdle(engine);
    require(events.starts.back() == kPrefillCheckpointTokens &&
                engine.snapshot().cancelled == 1 && engine.snapshot().completed == 1 &&
                events.failedCount == 0 &&
                cache.snapshot().stateCache.checkpointEntries == 0,
            "cancellation after a skipped checkpoint lost the earlier recovery point");
  }
}

// A lane alone in the pool whose growth needs an extent the budget refuses is
// not out of capacity while the tier is busy: the one state write in flight
// holds the staging buffer, so the cached state that would free memory waits
// for it, and the lane waits with it instead of failing or yielding. Once
// the write lands the cached state is written, its block gives up its pages,
// and the lane runs. Under host pressure the lane reuses the same idle
// cached pages, so it waits for them the same way.
void testGrowthWaitsForTheStateWriteInFlight() {
  for (bool paused : {false, true}) {
    test::TestKvStorage storage(16, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor executor;
    Events events;
    engine::Engine engine({.maxContext = 102400, .growthPaused = [paused] { return paused; }},
                          cache, executor, events);
    guardReleases(storage, engine);
    // A write in flight from a lane still running: its block is no leaf to evict.
    auto writing = std::make_shared<OffloadControl>();
    cache.beginRequest(999);
    require(cache.ensureTokens(999, 64).granted(), "fixture KV failed");
    const auto held = test::publishBlocks(cache, 999, std::vector<uint32_t>(64, 12), 64);
    require(cache.publishStateToDisk(held, [&](std::function<void()>) {
              return std::make_unique<OffloadTicket>(writing);
            }, false),
            "fixture write did not start");
    // A cached state in RAM whose eviction must wait for that write.
    auto cached = std::make_shared<OffloadControl>();
    cache.beginRequest(998);
    require(cache.ensureTokens(998, 64).granted(), "fixture KV failed");
    const auto cachedBlock = test::publishBlocks(cache, 998, std::vector<uint32_t>(64, 13), 64);
    cache.publishCompositeState(cachedBlock, std::make_shared<OffloadState>(cached), false);
    cache.endRequest(998);
    // Every further page needs an extent the budget, or the host, refuses.
    storage.allocationFailure = paused ? metal::AllocationFailure::HostPressure
                                       : metal::AllocationFailure::EngineBudget;
    storage.growthBlocked = true;
    engine.submit(request(1, std::vector<uint32_t>(33, 17)));
    static_cast<void>(engine.tick(1));
    static_cast<void>(engine.tick(2));
    require(events.failedCount == 0 && executor.prefillRows == 0 && executor.suspensions == 0,
            "the lane failed or yielded while the tier was busy");
    writing->ready = true;
    static_cast<void>(engine.tick(3));
    require(cache.snapshot().stateCache.offloads == 2 && cache.snapshot().stateCache.bytes == 0 &&
                cached->released,
            "the cached state was not written once the staging buffer was free");
    cached->ready = true;
    for (uint32_t step = 4; step < 20 && !idle(engine); ++step)
      static_cast<void>(engine.tick(step));
    require(events.completedCount == 1 && events.failedCount == 0 && executor.suspensions == 0 &&
                executor.prefillRows == 33,
            "the lane did not run on the pages the written state gave up");
  }
}

// Caches `count` one-page blocks, each under a state written to disk: their
// pages stay resident, and evicting one starts a demotion whose page comes
// back when the copy lands.
void cacheBlocksUnderDiskStates(engine::Cache &cache, uint32_t count) {
  auto transfer = std::make_shared<OffloadControl>();
  transfer->ready = true;
  for (uint64_t id = 900; id < 900 + count; ++id) {
    std::vector<uint32_t> prompt(32, static_cast<uint32_t>(id));
    cache.beginRequest(id);
    require(cache.ensureTokens(id, 32).granted(), "fixture KV failed");
    const auto block = test::publishBlocks(cache, id, prompt, 32);
    cache.publishCompositeState(block, std::make_shared<OffloadState>(transfer), false);
    cache.endRequest(id);
    require(cache.reclaimOneState(false, 0, false) && cache.pollTransfers(),
            "state was not demoted");
  }
}

// A lane that cannot run must never leave the engine without a wakeup: the
// transfer it waits for wakes it, and if that wake is missed the retry
// deadline does once no command is in flight (a command's completion wakes
// the loop itself). Here the pages come back but nothing reports their
// arrival, so only the deadline can end the wait.
void testWaitingLaneAlwaysNamesAWakeup() {
  test::TestKvStorage storage(12, 4096, 4);
  storage.budgetPages = 8;
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.transferLimit = 8;
  engine::Cache cache(pool, &tier, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({.maxContext = 102400}, cache, executor, events);
  guardReleases(storage, engine);
  cacheBlocksUnderDiskStates(cache, 6);
  engine.submit(request(1, std::vector<uint32_t>(97, 7)));
  static_cast<void>(engine.tick(1));
  require(tier.demotions == 1 && executor.prefillRows == 0,
          "the lane did not wait for a page");
  const auto wakeup = engine.nextWakeupMilliseconds();
  require(wakeup.has_value() && *wakeup <= 1.0 + 100.0,
          "a waiting lane left the engine without a bounded wakeup");
  // The wait ends on its own once the pages are back.
  for (uint32_t step = 2; step < 40 && !idle(engine); ++step) {
    tier.complete();
    static_cast<void>(engine.tick(step));
  }
  require(idle(engine) && events.outputs.contains(1) && events.failedCount == 0,
          "the lane never ran");
}

// A plan whose lanes all wait for pages on their way back leaves the tick
// to the other lanes: a decode that needs no page runs while a prefill
// waits, and a prefill that needs none runs while a decode waits. The
// waiting lanes run once their pages have landed.
void testWaitingPlanDoesNotIdleRunnableLanes() {
  const auto runWithTransfers = [](engine::Engine &engine, test::TestKvTier &tier,
                                   double &now, const std::function<bool()> &done) {
    for (uint32_t step = 0; step < 200 && !done(); ++step) {
      tier.complete();
      static_cast<void>(engine.tick(now++));
    }
    require(done(), "the waiting lane did not run once its pages landed");
  };
  {
    test::TestKvStorage storage(12, 4096, 4);
    storage.budgetPages = 8;
    KvPool pool(storage, 0);
    test::TestKvTier tier;
    tier.transferLimit = 8;
    engine::Cache cache(pool, &tier, nullptr);
    Executor executor;
    executor.decodeFinishes = false;
    Events events;
    engine::Engine engine({.maxContext = 102400}, cache, executor, events);
    guardReleases(storage, engine);
    cacheBlocksUnderDiskStates(cache, 6);
    // A stream decodes within its one page, beside a prefill short of two.
    auto stream = request(2, std::vector<uint32_t>(1, 5));
    stream.maxNewTokens = 1000;
    engine.submit(std::move(stream));
    double now = 1;
    tickUntil(engine, now, [&] { return events.outputs.contains(2); },
              "the stream did not decode");
    engine.submit(request(1, std::vector<uint32_t>(97, 7)));
    const uint64_t decodes = engine.snapshot().scheduler.decodeBatches;
    tickUntil(engine, now, [&] { return tier.demotions > 0; },
              "the prefill did not wait for pages");
    require(engine.commandInFlight() && engine.snapshot().scheduler.decodeBatches == decodes + 1 &&
                executor.prefillRows == 1,
            "the stream idled while a prefill waited for its pages");
    runWithTransfers(engine, tier, now, [&] { return events.outputs.contains(1); });
    engine.cancel(2);
    tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
    require(events.failedCount == 0 && executor.prefillRows == 98,
            "the lanes did not finish their prompts");
  }
  {
    test::TestKvStorage storage(72, 4096, 4);
    storage.budgetPages = 68;
    KvPool pool(storage, 0);
    test::TestKvTier tier;
    tier.transferLimit = 8;
    engine::Cache cache(pool, &tier, nullptr);
    Executor executor;
    Events events;
    engine::Engine engine({.maxContext = 102400}, cache, executor, events);
    guardReleases(storage, engine);
    cacheBlocksUnderDiskStates(cache, 3);
    // One command takes every page left: the short prompt's 30 rows and
    // 2018 of the long one's, which stops inside the page that ends at its
    // replay boundary, 2048. The short prompt's first decode needs a page;
    // the long one's next 30 rows need none.
    engine.submit(request(1, std::vector<uint32_t>(2060, 7)));
    engine.submit(request(2, std::vector<uint32_t>(30, 8)));
    double now = 1;
    static_cast<void>(engine.tick(now++));
    static_cast<void>(engine.tick(now++));
    require(executor.prefillRows == 2048 && pool.freePageCount() == 0 && !engine.commandInFlight(),
            "fixture pages are off");
    const uint64_t prefills = engine.snapshot().scheduler.prefillBatches;
    static_cast<void>(engine.tick(now++));
    require(tier.demotions > 0 && engine.commandInFlight() &&
                engine.snapshot().scheduler.prefillBatches == prefills + 1 &&
                !events.outputs.contains(2),
            "the long prompt idled while a decode waited for its page");
    runWithTransfers(engine, tier, now, [&] { return idle(engine); });
    require(events.completedCount == 2 && events.failedCount == 0 &&
                executor.prefillRows == 2090,
            "the lanes did not finish");
  }
}

// In a persistent tier a request's replay point is written behind it: once
// it has waited its delay, the pages of its chain are copied as earlier
// copies land, a few at a time, then its state, and all of it stays in RAM.
// A clean stop writes a point that has not waited yet at once. A temporary
// tier writes nothing behind.
void testReplayPointsAreWrittenBehind() {
  struct Fixture final {
    explicit Fixture(bool persistent)
        : tier(persistent), cache(pool, &tier, nullptr),
          engine({.maxContext = 102400}, cache, executor, events) {
      tier.transferLimit = 64;
      tier.capacity = 128;
      executor.keptTier = std::make_shared<OffloadControl>();
      executor.keptTier->ready = true;
      guardReleases(storage, engine);
      engine.submit(request(1, std::vector<uint32_t>(2060, 7)));
      tickUntil(engine, now, [this] { return idle(engine); }, "the request did not finish");
    }
    // The chain's 64 pages and the state, written while they stay in RAM.
    [[nodiscard]] bool durableInRam() const {
      const CacheSnapshot resources = cache.snapshot();
      return engine.snapshot().writeBehind.durable == 1 && tier.demotions == 64 &&
             resources.kvTier.copies == 64 && resources.pool.pagesPrefix == 64 &&
             resources.stateCache.bytes == 64 && resources.stateCache.diskBytes == 64;
    }
    test::TestKvStorage storage{72, 4096, 4};
    KvPool pool{storage, 0};
    test::TestKvTier tier;
    engine::Cache cache;
    Executor executor;
    Events events;
    engine::Engine engine;
    double now = 1;
  };
  {
    Fixture fixture(true);
    const std::optional<double> due = fixture.engine.nextWakeupMilliseconds();
    require(fixture.engine.snapshot().writeBehind.waiting == 1 && !fixture.tier.demotions &&
                due && *due > fixture.now &&
                *due <= fixture.now + WriteBehind::kDelayMilliseconds,
            "the replay point did not wait for its delay");
    static_cast<void>(fixture.engine.tick(*due - 1));
    require(!fixture.tier.demotions, "the replay point was written before its delay");
    double now = *due;
    static_cast<void>(fixture.engine.tick(now));
    require(fixture.tier.demotions == 16, "the replay point's copies did not start a few at a time");
    for (uint32_t step = 0; step < 16 && !fixture.durableInRam(); ++step) {
      fixture.tier.complete();
      static_cast<void>(fixture.engine.tick(++now));
    }
    require(fixture.durableInRam(), "the replay point did not become durable in RAM");
  }
  {
    Fixture fixture(true);
    require(!fixture.engine.flushRestorePoints() && fixture.tier.demotions == 16,
            "a clean stop did not write the point that had not waited");
    bool flushed = false;
    for (uint32_t step = 0; step < 16 && !flushed; ++step) {
      fixture.tier.complete();
      flushed = fixture.engine.flushRestorePoints();
    }
    require(flushed && fixture.durableInRam(), "a clean stop did not leave the point durable");
  }
  {
    Fixture fixture(false);
    require(!fixture.engine.snapshot().writeBehind.waiting && fixture.engine.flushRestorePoints() &&
                !fixture.tier.demotions,
            "a temporary tier wrote a replay point behind");
  }
}

// Pending means a transfer is in flight. With the tier unwritable and
// nothing moving, a lane that cannot get pages must be answered, not parked.
void testNothingInFlightIsNotPending() {
  test::TestKvStorage storage(12, 4096, 4);
  storage.budgetPages = 8;
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.writableFile = false;
  engine::Cache cache(pool, &tier, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({.maxContext = 102400}, cache, executor, events);
  guardReleases(storage, engine);
  auto transfer = std::make_shared<OffloadControl>();
  transfer->ready = true;
  for (uint64_t id = 900; id < 908; ++id) {
    std::vector<uint32_t> prompt(32, static_cast<uint32_t>(id));
    cache.beginRequest(id);
    if (!cache.ensureTokens(id, 32).granted()) {
      cache.endRequest(id);
      break;
    }
    cache.publishCommittedBlocks(id, prompt, 32, {});
    cache.endRequest(id);
  }
  engine.submit(request(1, std::vector<uint32_t>(97, 7)));
  runUntilIdle(engine);
  require(idle(engine) && events.outputs.contains(1) && events.failedCount == 0 &&
              tier.demotions == 0 && executor.prefillRows == 97,
          "the lane did not run on leaves the unwritable tier had to drop");
}

// A shortfall of pages is covered in one pass: when the pool cannot
// allocate another extent, the reclaim between attempts demotes as many
// leaves as the shortfall needs, within the tier's share, and the lane waits
// once for their pages instead of once per page.
void testPageShortfallDemotesInBulk() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.transferLimit = 8;
  tier.capacity = 8;
  engine::Cache cache(pool, &tier, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({.maxContext = 102400}, cache, executor, events);
  guardReleases(storage, engine);
  cacheBlocksUnderDiskStates(cache, 6);
  // The cached blocks fill two extents but two of their pages: those two
  // are free, no other extent is allocated, and the budget admits nothing
  // more. The first command, 160 tokens, falls three pages short; the last
  // token one more.
  require(pool.freePageCount() == 2 && pool.snapshot().pagesAllocated == 8,
          "fixture storage geometry changed");
  storage.growthBlocked = true;
  const uint32_t attemptsBefore = storage.allocationAttempts;
  engine.submit(request(1, std::vector<uint32_t>(161, 7)));
  static_cast<void>(engine.tick(1));
  require(tier.demotions == 3 && executor.prefillRows == 0 && executor.suspensions == 0 &&
              events.failedCount == 0,
          "a shortfall of three pages did not start three demotions in one pass");
  require(storage.allocationAttempts - attemptsBefore == 2,
          "the demotions did not start in the one reclaim step after the denial");
  static_cast<void>(engine.tick(2));
  require(tier.demotions == 3, "waiting demoted more");
  for (uint32_t step = 3; step < 40 && !idle(engine); ++step) {
    tier.complete();
    static_cast<void>(engine.tick(step));
  }
  require(idle(engine) && events.outputs.contains(1) && executor.prefillRows == 161 &&
              executor.suspensions == 0 && events.failedCount == 0 && tier.demotions == 4 &&
              cache.snapshot().kvTier.diskBlocks == 4,
          "the lane did not run on the pages the demotions gave back");
}

void testKvGrowthProceedsThroughDemotion() {
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({.maxContext = 102400}, cache, executor, events);
  guardReleases(storage, engine);
  auto transfer = std::make_shared<OffloadControl>();
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 32).granted(), "offload fixture KV failed");
  const auto block = test::publishBlocks(cache, 999, std::vector<uint32_t>(32, 12), 32);
  cache.publishCompositeState(block, std::make_shared<OffloadState>(transfer), false);
  cache.endRequest(999);
  {
    auto lookup = cache.lookup(std::vector<uint32_t>(33, 12), {});
    cache.recordLookup(lookup);
  }
  storage.growthAllowed = [&](uint32_t) { return transfer->released; };
  engine.submit(request(1, std::vector<uint32_t>(256, 17)));
  runUntilIdle(engine);
  // The demoted state's RAM served KV growth before its write finished.
  require(transfer->released && !transfer->ready && executor.prefillRows == 256 &&
              executor.suspensions == 0 && events.failedCount == 0,
          "KV growth waited for the write or replayed the lane");
  require(cache.snapshot().stateCache.diskBytes == 64, "demoted state lost its disk copy");
  transfer->ready = true;
  static_cast<void>(engine.tick(1000));
  const auto stats = cache.snapshot().stateCache;
  require(stats.offloads == 1 && stats.offloadFailures == 0 && stats.diskBytes == 64,
          "completed write was not consumed");
}

// States reach the disk tier by demotion only: the fixture publishes a RAM
// state and reclaims it while nothing else is in RAM.
void demoteState(engine::Cache &cache, uint64_t block) {
  auto transfer = std::make_shared<OffloadControl>();
  transfer->ready = true;
  cache.publishCompositeState(block, std::make_shared<OffloadState>(transfer), false);
  require(cache.reclaimOneState(false, 0, false) && cache.pollTransfers() &&
              cache.snapshot().stateCache.bytes == 0,
          "fixture state did not move to disk");
}

void publishDiskState(engine::Cache &cache, const std::vector<uint32_t> &prompt) {
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 64).granted(), "disk fixture KV allocation failed");
  demoteState(cache, test::publishBlocks(cache, 999, prompt, 64));
  cache.endRequest(999);
}

void testAsyncRestoreLifecycle() {
  for (int outcome = 0; outcome < 4; ++outcome) {
    test::TestKvStorage storage(128, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor executor;
    Events events;
    engine::Engine engine({.maxContext = 102400}, cache, executor, events);
    guardReleases(storage, engine);
    std::vector<uint32_t> prompt(65, 17);
    publishDiskState(cache, prompt);
    engine.submit(request(1, prompt));
    static_cast<void>(engine.tick(1));
    require(executor.diskReads == 1 && executor.prefillRows == 0 && events.starts.empty(),
            "disk restore ran before IO completion");
    require(cache.snapshot().stateCache.pinned == 1 && executor.requests.contains(1),
            "restore did not retain admitted state and KV");
    // A separate lane can run while this restore is pending.
    engine.submit(request(2, std::vector<uint32_t>(3, 99)));
    for (int tick = 2; tick < 8; ++tick) static_cast<void>(engine.tick(tick));
    require(events.outputs.contains(2), "restore blocked independent model work");
    if (outcome == 1) engine.cancel(1);
    if (outcome == 2) static_cast<void>(engine.tick(10001));
    if (outcome == 3) executor.restoreControl->success = false;
    require(executor.requests.contains(1), "cancellation reused IO destination before drain");
    executor.restoreControl->ready = true;
    runUntilIdle(engine);
    require(cache.snapshot().stateCache.pinned == 0 && cache.snapshot().activeRequests == 0 &&
                executor.requests.empty(), "restore leaked active resources");
    if (outcome == 0)
      require(executor.restored == 64 && executor.prefillRows == 4, "disk hit recomputed prefix");
    else if (outcome == 3)
      require(executor.restored == 0 && executor.prefillRows == 68 && executor.diskReads == 1,
              "failed read did not fall back once to cold prefill");
    else
      require(executor.restored == 0 && !events.outputs.contains(1),
              "cancelled or expired restore emitted output");
    if (outcome == 2)
      require(events.failures == std::vector<std::string>{"deadline_exceeded"} &&
                  events.failureDetails.front().first == "request deadline exceeded",
              "an expired restore lost its outcome or message");
  }
}

void testDiskHitWithNoActiveMemory() {
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({.maxContext = 102400}, cache, executor, events);
  guardReleases(storage, engine);
  std::vector<uint32_t> prompt(65, 17);
  publishDiskState(cache, prompt);
  executor.beginAllocationFailure = metal::AllocationFailure::EngineBudget;
  executor.beginGrowthBlocked = [&] { return cache.snapshot().stateCache.entries != 0; };
  engine.submit(request(1, prompt));
  runUntilIdle(engine);
  require(executor.diskReads == 0 && executor.prefillRows == 65 && events.failedCount == 0,
          "disk hit bypassed admission or deadlocked on its own pin");
}

void testRepeatedDiskHitPromotesToMemory() {
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({.maxContext = 102400}, cache, executor, events);
  guardReleases(storage, engine);
  std::vector<uint32_t> prompt(65, 17);
  publishDiskState(cache, prompt);
  executor.restoreControl->ready = true;
  for (uint64_t id = 1; id <= 4; ++id) {
    engine.submit(request(id, prompt));
    runUntilIdle(engine);
  }
  require(executor.diskReads == 1 && cache.snapshot().stateCache.promotions == 1 &&
              cache.snapshot().lookup.stateDiskHits == 1 && events.failedCount == 0,
          "hot restored prefix continued to read from disk");
}

// A failed disk read falls back to the next cache match, not to a cold start:
// the invalidated copy is gone from the lookup, the shallower state is not.
void testFailedDiskRestoreKeepsShallowerState() {
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({.maxContext = 102400}, cache, executor, events);
  guardReleases(storage, engine);
  std::vector<uint32_t> prompt(65, 17);
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 64).granted(), "fixture KV failed");
  cache.publishCommittedBlocks(999, prompt, 64, {});
  demoteState(cache, cache.blockAt(999, 64));
  cache.publishCompositeState(cache.blockAt(999, 32), std::make_shared<State>(), false);
  cache.endRequest(999);
  executor.restoreControl->ready = true;
  executor.restoreControl->success = false;
  engine.submit(request(1, prompt));
  runUntilIdle(engine);
  require(executor.diskReads == 1 && events.failedCount == 0 && executor.restored == 32 &&
              executor.prefillRows == 33,
          "failed disk restore discarded the shallower RAM state");
}

// A prefix whose state and KV moved to disk comes back before the lane runs:
// the request waits for the state read and the page restores, then starts
// as a prefix hit on the restored pages.
void testDiskKvPrefixIsRestoredBeforeTheLaneRuns() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  engine::Cache cache(pool, &tier, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({.maxContext = 102400}, cache, executor, events);
  guardReleases(storage, engine);
  std::vector<uint32_t> prompt(65, 17);
  auto transfer = std::make_shared<OffloadControl>();
  transfer->ready = true;
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 64).granted(), "fixture KV failed");
  const auto block = test::publishBlocks(cache, 999, prompt, 64);
  cache.publishCompositeState(block, std::make_shared<OffloadState>(transfer), false);
  cache.endRequest(999);
  require(cache.reclaimOne(reuse, ReclaimClass::InUse).evictedState && cache.pollTransfers(),
          "state was not demoted");
  for (uint32_t written = 1; written <= 2; ++written) {
    require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress && tier.demotions == written,
            "KV block was not written");
    tier.complete();
    require(cache.pollTransfers(), "written block did not land");
  }
  require(cache.snapshot().pool.pagesPrefix == 0 && cache.snapshot().kvTier.diskBlocks == 2,
          "prefix did not move to disk whole");

  engine.submit(request(1, prompt));
  static_cast<void>(engine.tick(1));
  require(executor.diskReads == 1 && tier.restores == 2 && executor.prefillRows == 0 &&
              events.starts.empty(),
          "lane ran before its pages came back");
  // The state read lands first; the lane still waits for its pages.
  executor.restoreControl->ready = true;
  static_cast<void>(engine.tick(2));
  static_cast<void>(engine.tick(3));
  require(executor.prefillRows == 0 && events.starts.empty(),
          "lane ran with pages still on the way");
  tier.complete();
  runUntilIdle(engine);
  require(executor.restored == 64 && executor.prefillRows == 1 && events.starts.size() == 1 &&
              events.starts[0] == 64 && events.failedCount == 0,
          "restored prefix was not used");
  const auto stats = cache.snapshot();
  require(stats.kvTier.restores == 2 &&
              stats.pool.pagesPrefix == 2 && stats.kvTier.diskBlocks == 2,
          "restore accounting is off");
}

// Siblings that share a prefix whose deepest state is on disk wait for the
// one admitted first while it is still restoring: it planned the junction at
// their shared boundary at admission, so the state is read once and the
// shared span prefilled once.
void testSharedPrefillWaitsForARestoringProducer() {
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({.maxContext = 102400}, cache, executor, events);
  guardReleases(storage, engine);
  // 64 tokens whose state is on disk, 128 more both share, 34 of their own.
  std::vector<uint32_t> prompt(226);
  std::iota(prompt.begin(), prompt.end(), 1);
  publishDiskState(cache, prompt);
  executor.restoreControl->ready = false;
  engine.submit(request(1, prompt));
  std::fill(prompt.begin() + 192, prompt.end(), 7);
  engine.submit(request(2, prompt));
  static_cast<void>(engine.tick(1));
  require(executor.diskReads == 1 && engine.snapshot().scheduler.waitingPrefix == 1,
          "the sibling did not wait for the restoring producer");
  executor.restoreControl->ready = true;
  runUntilIdle(engine);
  require(executor.diskReads == 1 &&
              events.starts.back() == 192 &&
              executor.prefillRows == (192 - 64) + 2 * 34 && events.completedCount == 2,
          "the sibling read the state again or prefilled the shared span");
}

// A restoring producer that is cancelled keeps its lane until its read has
// drained, but publishes nothing: its waiting sibling starts at once and
// restores the prefix itself.
void testCancelledRestoringProducerReleasesItsWaiter() {
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({.maxContext = 102400}, cache, executor, events);
  guardReleases(storage, engine);
  std::vector<uint32_t> prompt(226);
  std::iota(prompt.begin(), prompt.end(), 1);
  publishDiskState(cache, prompt);
  const std::shared_ptr<RestoreControl> producerRead = executor.restoreControl;
  engine.submit(request(1, prompt));
  std::fill(prompt.begin() + 192, prompt.end(), 7);
  engine.submit(request(2, prompt));
  static_cast<void>(engine.tick(1));
  require(executor.diskReads == 1 && engine.snapshot().scheduler.waitingPrefix == 1,
          "the sibling did not wait for the restoring producer");
  engine.cancel(1);
  executor.restoreControl = std::make_shared<RestoreControl>();
  static_cast<void>(engine.tick(2));
  require(executor.diskReads == 2 && engine.resourceWaitSnapshot(2).restoring == 2 &&
              executor.requests.contains(1),
          "the sibling waited for a cancelled producer's read to drain");
  producerRead->ready = true;
  executor.restoreControl->ready = true;
  runUntilIdle(engine);
  require(events.starts.back() == 64 &&
              engine.snapshot().cancelled == 1 && events.completedCount == 2,
          "the sibling did not restore the prefix itself");
}

// A request whose restore could not fit ignores the cache until it starts:
// it uses no producer's state, so it does not wait for one, even for one
// restoring the same prefix with a boundary planned inside it, and no
// producer plans a junction for it.
void testSkipCacheRequestDoesNotWaitForAProducer() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  engine::Cache cache(pool, &tier, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({.maxContext = 102400}, cache, executor, events);
  guardReleases(storage, engine);
  // The state at 64 and both blocks of KV under it are on disk.
  std::vector<uint32_t> prompt(226);
  std::iota(prompt.begin(), prompt.end(), 1);
  auto transfer = std::make_shared<OffloadControl>();
  transfer->ready = true;
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 64).granted(), "fixture KV failed");
  const auto block = test::publishBlocks(cache, 999, prompt, 64);
  cache.publishCompositeState(block, std::make_shared<OffloadState>(transfer), false);
  cache.endRequest(999);
  require(cache.reclaimOne(reuse, ReclaimClass::InUse).evictedState && cache.pollTransfers(),
          "state was not demoted");
  for (uint32_t written = 1; written <= 2; ++written) {
    require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress && tier.demotions == written,
            "KV block was not written");
    tier.complete();
    require(cache.pollTransfers(), "written block did not land");
  }
  while (pool.snapshot().pagesAllocated)
    require(cache.reclaimOne(CacheReclaimMode::ReleaseExtents, ReclaimClass::InUse).madeProgress,
            "an empty extent was not released");
  // Alone, with no page the budget grants, its restore fails: it skips the
  // cache from then on.
  storage.growthBlocked = true;
  engine.submit(request(1, prompt));
  static_cast<void>(engine.tick(1));
  require(engine.resourceWaitSnapshot(1).memory == 1 && executor.diskReads == 0,
          "the restore that could not fit did not wait for memory");
  storage.growthBlocked = false;
  // A foreground request sharing its first 192 tokens is admitted first and
  // restores the prefix. It plans a junction at 160 for a sibling sharing
  // that much with it, and none at 192 for the request that skips the cache.
  std::vector<uint32_t> producer = prompt;
  std::fill(producer.begin() + 192, producer.end(), 7);
  auto foreground = request(2, producer);
  foreground.priority = RequestPriority::Foreground;
  engine.submit(std::move(foreground));
  std::vector<uint32_t> sibling = producer;
  std::fill(sibling.begin() + 160, sibling.end(), 9);
  engine.submit(request(3, sibling));
  double now = 101;
  tickUntil(engine, now,
            [&] {
              return std::find(events.startIds.begin(), events.startIds.end(), 1) !=
                     events.startIds.end();
            },
            "the request that skips the cache did not start");
  require(events.starts.front() == 0 &&
              executor.diskReads == 1 && executor.restored == 0,
          "the request that skips the cache waited for the restoring producer");
  executor.restoreControl->ready = true;
  for (; now < 300 && !idle(engine); ++now) {
    static_cast<void>(engine.tick(now));
    tier.complete();
  }
  require(idle(engine) && events.completedCount == 3 && events.failedCount == 0,
          "the requests did not finish");
  const auto &boundaries = executor.plans.at(2).boundaries;
  require(events.starts.back() == 160 &&
              std::none_of(boundaries.begin(), boundaries.end(),
                           [](const DraftBoundaryPlan &boundary) {
                             return boundary.boundary == 192;
                           }),
          "the producer planned a junction for the request that skips the cache");
}

// A request whose prefix restore cannot fit even alone ignores the cache
// until it starts: admission no longer probes it, and it starts cold.
void testSkipCacheCandidateIsNotProbed() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  engine::Cache cache(pool, &tier, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({.maxContext = 102400}, cache, executor, events);
  guardReleases(storage, engine);
  const auto hashed = [&] { return cache.snapshot().lookup.probeHashedBlocks; };
  const std::vector<uint32_t> prompt(65, 17);
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 64).granted(), "fixture KV failed");
  demoteState(cache, test::publishBlocks(cache, 999, prompt, 64));
  cache.endRequest(999);
  for (uint32_t written = 1; written <= 2; ++written) {
    require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress, "KV block was not written");
    tier.complete();
    require(cache.pollTransfers(), "written block did not land");
  }
  require(cache.reclaimOne(CacheReclaimMode::ReleaseExtents, ReclaimClass::InUse).madeProgress &&
              storage.allocatedPages() == 0 && cache.snapshot().kvTier.diskBlocks == 2,
          "the prefix did not move to disk whole");
  // No page can be allocated until the request starts again.
  storage.growthBlocked = true;
  executor.beginObserver = [&] { storage.growthBlocked = executor.beginAttempts < 2; };
  engine.submit(request(1, prompt));
  static_cast<void>(engine.tick(1));
  require(hashed() == 2 && events.starts.empty() && engine.resourceWaitSnapshot(1).memory == 1,
          "the restore that cannot fit alone did not wait");
  double now = 2;
  tickUntil(engine, now, [&] { return !events.starts.empty(); },
            "the request did not start without its prefix");
  require(hashed() == 2 && events.starts[0] == 0,
          "a request that ignores the cache was probed or ranked by its prefix");
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  require(events.completedCount == 1 && executor.prefillRows == 65 && tier.restores == 0,
          "the request did not finish cold");
}

// A Background lane holds the page a Normal request needs to read its
// prefix back from disk. With nothing else to free, the refused pages
// suspend the lane, and the request starts at its next attempt. When the
// page is on its way back, a cached block being written to disk, the
// request waits for the write and no lane yields.
void testRefusedPrefixRestorePreemptsLowerResident() {
  for (const bool written : {false, true}) {
    test::TestKvStorage storage(64, 4096, 1);
    KvPool pool(storage, 0);
    test::TestKvTier tier;
    engine::Cache cache(pool, written ? &tier : nullptr, nullptr);
    Executor executor;
    executor.decodeFinishes = false;
    Events events;
    engine::Engine engine({.maxContext = 102400}, cache, executor, events);
    guardReleases(storage, engine);
    // The prefix's state is on disk and its two pages in memory. Beside it,
    // another state on disk keeps the page of its block, which a reclaim
    // writes to disk before it frees it.
    const std::vector<uint32_t> prompt(65, 17);
    publishDiskState(cache, prompt);
    if (written) {
      cache.beginRequest(998);
      require(cache.ensureTokens(998, 32).granted(), "fixture KV failed");
      demoteState(cache, test::publishBlocks(cache, 998, std::vector<uint32_t>(33, 5), 32));
      cache.endRequest(998);
    }
    // The Background lane's three pages fill the engine's limit.
    storage.budgetPages = storage.allocatedPages() + 3;
    auto background = request(1, std::vector<uint32_t>(65, 1));
    background.priority = RequestPriority::Background;
    background.maxNewTokens = 1000;
    engine.submit(std::move(background));
    double now = 1;
    tickUntil(engine, now, [&] { return events.outputs.contains(1); },
              "the Background lane did not decode");
    // Its state read from disk, the request needs a page beyond its prefix.
    engine.submit(request(2, prompt));
    tickUntil(engine, now, [&] { return executor.beginAttempts == 2; },
              "the request did not try to start");
    if (written) {
      require(tier.demotions == 1 && engine.snapshot().prioritySuspensions == 0 &&
                  executor.requests.at(1).resident,
              "a lane yielded memory on its way back");
      tier.complete();
    } else {
      require(engine.snapshot().prioritySuspensions == 1 && !executor.requests.at(1).resident,
              "the refused restore did not suspend the lower lane");
    }
    tickUntil(engine, now, [&] { return executor.diskReads == 1; },
              "the request was not admitted into its restore");
    require(executor.beginAttempts == 3, "the request did not start at its next attempt");
    executor.restoreControl->ready = true;
    tickUntil(engine, now, [&] { return events.startIds.back() == 2; },
              "the restored request did not start");
    require(events.starts.back() == 64 &&
                engine.snapshot().prioritySuspensions == (written ? 0U : 1U),
            "the request did not start from its restored prefix");
    for (uint64_t id : {1, 2})
      engine.cancel(id);
    tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
    require(events.failedCount == 0, "a request failed");
  }
}

// A request admitted into a restore waits for its disk reads, not for
// memory, although the host refused its previous attempt: /status counts it
// as restoring.
void testRestoringRequestIsNotWaitingForMemory() {
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor executor;
  bool hostRefuses = true;
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  executor.beginGrowthBlocked = [&] { return hostRefuses; };
  Events events;
  engine::Engine engine({.maxContext = 102400}, cache, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(65, 17);
  publishDiskState(cache, prompt);
  engine.submit(request(1, prompt));
  static_cast<void>(engine.tick(1));
  require(engine.resourceWaitSnapshot(1).memory == 1, "the host's refusal did not wait for memory");
  hostRefuses = false;
  double now = 101;
  static_cast<void>(engine.tick(now));
  const auto restoring = engine.resourceWaitSnapshot(now);
  require(executor.diskReads == 1 && restoring.restoring == 1 && restoring.memory == 0,
          "a request reading its prefix from disk was counted as waiting for memory");
  executor.restoreControl->ready = true;
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  require(events.completedCount == 1 && executor.restored == 64 &&
              engine.resourceWaitSnapshot(now).restoring == 0,
          "the restored request did not finish");
}

void testCancelledDiskPrefixStopsQueuedReads() {
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.transferLimit = 1;
  engine::Cache cache(pool, &tier, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({.maxContext = 102400}, cache, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(129, 17);
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 128).granted(), "fixture KV failed");
  const auto block = test::publishBlocks(cache, 999, prompt, 128);
  demoteState(cache, block);
  cache.endRequest(999);
  for (int page = 0; page < 4; ++page) {
    require(cache.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse).madeProgress,
            "fixture KV did not demote");
    tier.complete();
    static_cast<void>(cache.pollTransfers());
  }
  engine.submit(request(1, prompt));
  static_cast<void>(engine.tick(1));
  require(tier.restores == 1 && executor.diskReads == 1,
          "restore did not start within the tier's limit");
  engine.cancel(1);
  executor.restoreControl->ready = true;
  static_cast<void>(engine.tick(2));
  for (int tick = 3; tick < 16; ++tick) {
    tier.complete();
    static_cast<void>(engine.tick(tick));
  }
  const auto stats = cache.snapshot();
  require(tier.restores == 1 && !tier.inFlight() && idle(engine) &&
              stats.stateCache.pinned == 0 && stats.activeRequests == 0 &&
              executor.requests.empty() && !events.outputs.contains(1),
          "cancelled prefix read unused pages or retained active resources");
  require(stats.kvTier.diskBlocks == 4 && stats.pool.pagesPrefix == 1,
          "cancellation destroyed reusable disk pages");
  executor.restoreControl = std::make_shared<RestoreControl>();
  executor.restoreControl->ready = true;
  engine.submit(request(2, prompt));
  for (int tick = 16; tick < 64; ++tick) {
    tier.complete();
    static_cast<void>(engine.tick(tick));
  }
  require(tier.restores == 4 && executor.restored == 128 &&
              events.outputs.contains(2) && events.failedCount == 0 &&
              cache.snapshot().activeRequests == 0,
          "retry did not reuse the cancelled request's disk prefix");
}

// A lane short of pages waits for the copies of demoted blocks to land and
// runs on their pages: no lane yields, and no more is written than needed.
void testPagesReturnFromDemotionWithoutSuspending() {
  test::TestKvStorage storage(12, 4096, 4);
  storage.budgetPages = 8;
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.transferLimit = 8;
  engine::Cache cache(pool, &tier, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({.maxContext = 102400}, cache, executor, events);
  guardReleases(storage, engine);
  // Six cached blocks, each under a state on disk, hold six of eight pages.
  cacheBlocksUnderDiskStates(cache, 6);
  require(pool.freePageCount() == 2 && tier.demotions == 0, "fixture pages are off");

  // The first prefill command ends at the replay boundary, 96 tokens: three
  // pages against two free ones.
  engine.submit(request(1, std::vector<uint32_t>(97, 7)));
  static_cast<void>(engine.tick(1));
  require(tier.demotions == 1 && executor.suspensions == 0 &&
              executor.prefillRows == 0 && events.failedCount == 0,
          "the lane yielded or more than one block was written");
  static_cast<void>(engine.tick(2));
  static_cast<void>(engine.tick(3));
  require(tier.demotions == 1 && executor.prefillRows == 0, "waiting demoted more");
  // The lane lacks two pages in all: one for its first command, one for the
  // last token. Exactly two blocks are written.
  for (uint32_t step = 4; step < 40 && !idle(engine); ++step) {
    tier.complete();
    static_cast<void>(engine.tick(step));
  }
  require(idle(engine), "engine did not reach idle");
  require(events.outputs.contains(1) && executor.prefillRows == 97 && executor.suspensions == 0 &&
              events.failedCount == 0 && tier.demotions == 2,
          "lane did not run on the returned pages, or more was written than it lacked");
  const auto stats = cache.snapshot();
  require(stats.kvTier.diskBlocks == 2 && stats.kvTier.pendingPages == 0 &&
              stats.stateCache.diskBytes == 6 * 64,
          "tier accounting after the wait is off");
}

// A request whose prefix is on disk is refused the pages to restore it into.
// Like a refused state, that closes admission behind it, so a later, short
// request does not take the pages it waits for. Refused by the engine's limit
// beside a resident lane, it starts first once the lane finishes; waiting for
// the demotions that free its pages, it holds the short request back until it
// is cancelled.
void testRefusedRestoreClosesAdmission() {
  enum class Cause { GrowthBlocked, Pending };
  for (const Cause cause : {Cause::GrowthBlocked, Cause::Pending}) {
    test::TestKvStorage storage(16, 4096, 4);
    storage.budgetPages = 8;
    KvPool pool(storage, 0);
    test::TestKvTier tier;
    tier.transferLimit = 8;
    engine::Cache cache(pool, &tier, nullptr);
    Executor executor;
    executor.decodeFinishes = false;
    executor.restoreControl->ready = true;
    Events events;
    engine::Engine engine({.maxContext = 102400}, cache, executor, events);
    guardReleases(storage, engine);
    // Request 2's prompt: its state and both of its KV blocks are on disk.
    const std::vector<uint32_t> prompt(65, 17);
    cache.beginRequest(999);
    require(cache.ensureTokens(999, 64).granted(), "fixture KV failed");
    demoteState(cache, test::publishBlocks(cache, 999, prompt, 64));
    cache.endRequest(999);
    for (uint32_t written = 1; written <= 2; ++written) {
      require(cache.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse).madeProgress,
              "KV block was not written");
      tier.complete();
      require(cache.pollTransfers(), "written block did not land");
    }
    // Pending: cached blocks under states on disk take the first extent, whose
    // pages come back only once their copies are written.
    if (cause == Cause::Pending) {
      for (uint64_t id = 900; id < 904; ++id) {
        cache.beginRequest(id);
        require(cache.ensureTokens(id, 32).granted(), "fixture KV failed");
        demoteState(cache, test::publishBlocks(
                               cache, id, std::vector<uint32_t>(32, static_cast<uint32_t>(id)), 32));
        cache.endRequest(id);
      }
    }
    // Lane 1 takes the remaining pages and decodes within them.
    auto resident = request(1, std::vector<uint32_t>(97, 1));
    resident.maxNewTokens = 1000;
    engine.submit(std::move(resident));
    double now = 1;
    tickUntil(engine, now, [&] { return events.emitted != 0; }, "the resident lane did not decode");
    if (cause == Cause::GrowthBlocked) {
      storage.growthBlocked = true;
      storage.allocationFailure = metal::AllocationFailure::EngineBudget;
    }
    require(pool.freePageCount() == 0, "the fixture left pages free");
    engine.submit(request(2, prompt));
    tickUntil(engine, now, [&] { return engine.resourceWaitSnapshot(now).memory == 1; },
              "the restore's pages were not refused");
    require(cause == Cause::GrowthBlocked ? tier.demotions == 2 : tier.inFlight() != 0,
            "the refusal did not come from the intended cause");
    engine.submit(request(3, {3}));
    for (uint32_t step = 0; step < 10; ++step, now += 101) {
      static_cast<void>(engine.tick(now));
      require(!executor.requests.contains(3) && !events.usage.contains(3) &&
                  engine.resourceWaitSnapshot(now).heldBehindRefusal == 1 &&
                  events.failedCount == 0,
              "a later arrival started ahead of a restore refused its pages");
    }
    if (cause == Cause::Pending) {
      engine.cancel(2);
      tickUntil(engine, now, [&] { return executor.requests.contains(3) || events.usage.contains(3); },
                "admission stayed closed after the refused restore was cancelled");
    }
    executor.decodeFinishes = true;
    storage.growthBlocked = false;
    for (uint32_t step = 0; step < 100 && !idle(engine); ++step) {
      tier.complete();
      static_cast<void>(engine.tick(now++));
    }
    // A cancelled request is reported as ended.
    require(idle(engine) && events.completedCount == 3 && events.failedCount == 0,
            "the requests did not finish");
    if (cause == Cause::GrowthBlocked)
      require(events.startIds == std::vector<uint64_t>{1, 2, 3} && executor.restored == 64,
              "the refused restore did not start from its prefix before the request behind it");
  }
}

// A lane whose pages keep landing is never failed for waiting: the resource
// limit measures time without progress. With one transfer at a time every
// round moves one page, so the restore takes many rounds and several times
// the limit, and still completes as a prefix hit.
void testWaitWithProgressOutlivesTheResourceLimit() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvStorage storage(12, 4096, 4);
  storage.budgetPages = 8;
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.transferLimit = 8;
  engine::Cache cache(pool, &tier, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({.maxContext = 102400}, cache, executor, events);
  guardReleases(storage, engine);
  // A three-block prefix and its state move to disk entirely.
  std::vector<uint32_t> prompt(97, 17);
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 96).granted(), "fixture KV failed");
  demoteState(cache, test::publishBlocks(cache, 999, prompt, 96));
  cache.endRequest(999);
  for (uint32_t written = 1; written <= 3; ++written) {
    require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress && tier.demotions == written,
            "prefix block was not written");
    tier.complete();
    require(cache.pollTransfers(), "prefix block did not land");
  }
  // Six cached blocks under disk states hold six of the eight pages.
  auto transfer = std::make_shared<OffloadControl>();
  transfer->ready = true;
  for (uint64_t id = 900; id < 906; ++id) {
    std::vector<uint32_t> filler(32, static_cast<uint32_t>(id));
    cache.beginRequest(id);
    require(cache.ensureTokens(id, 32).granted(), "filler KV failed");
    cache.publishCompositeState(test::publishBlocks(cache, id, filler, 32),
                                std::make_shared<OffloadState>(transfer), false);
    cache.endRequest(id);
    require(cache.reclaimOneState(false, 0, false) && cache.pollTransfers(),
            "filler state was not demoted");
  }
  require(pool.freePageCount() == 2, "fixture pages are off");

  tier.transferLimit = 1;
  executor.restoreControl->ready = true;
  EngineRequest waiting = request(1, prompt);
  waiting.deadlineMilliseconds = 1e9;
  engine.submit(std::move(waiting));
  double now = 1.0;
  for (int round = 0; round < 40 && !idle(engine); ++round) {
    static_cast<void>(engine.tick(now));
    tier.complete();
    now += 20000.0;
  }
  require(idle(engine), "lane did not finish");
  require(now > 90000.0, "the wait did not cross the resource limit");
  require(events.failedCount == 0 && events.starts.size() == 1 &&
              events.starts[0] == 96 && executor.restored == 96 &&
              executor.prefillRows == 1 && executor.suspensions == 0,
          "a lane whose pages kept landing was failed or lost its prefix");
  require(cache.snapshot().kvTier.restores == 3,
          "resource retries lost or repeated completed KV restores");
}

// An admission that gives back the lane and leases it took in the same pass
// returns nothing waiting requests lacked before, so it is no progress: a
// restore whose pages wait behind a demotion that never lands fails at the
// limit of its first refused attempt instead of moving it out on every
// retry.
void testRolledBackAdmissionKeepsItsWaitLimit() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvStorage storage(12, 4096, 4);
  storage.budgetPages = 8;
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.transferLimit = 8;
  engine::Cache cache(pool, &tier, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({.maxContext = 102400}, cache, executor, events);
  guardReleases(storage, engine);
  // A three-block prefix and its state move to disk entirely.
  std::vector<uint32_t> prompt(97, 17);
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 96).granted(), "fixture KV failed");
  demoteState(cache, test::publishBlocks(cache, 999, prompt, 96));
  cache.endRequest(999);
  for (uint32_t written = 1; written <= 3; ++written) {
    require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress && tier.demotions == written,
            "prefix block was not written");
    tier.complete();
    require(cache.pollTransfers(), "prefix block did not land");
  }
  // Six cached blocks under disk states hold six of the eight pages.
  auto transfer = std::make_shared<OffloadControl>();
  transfer->ready = true;
  for (uint64_t id = 900; id < 906; ++id) {
    std::vector<uint32_t> filler(32, static_cast<uint32_t>(id));
    cache.beginRequest(id);
    require(cache.ensureTokens(id, 32).granted(), "filler KV failed");
    cache.publishCompositeState(test::publishBlocks(cache, id, filler, 32),
                                std::make_shared<OffloadState>(transfer), false);
    cache.endRequest(id);
    require(cache.reclaimOneState(false, 0, true) && cache.pollTransfers(),
            "filler state was not demoted");
  }
  require(pool.freePageCount() == 2, "fixture pages are off");

  // The demotion that would free the restore's third page never lands.
  tier.transferLimit = 1;
  EngineRequest waiting = request(1, prompt);
  waiting.deadlineMilliseconds = 1e9;
  engine.submit(std::move(waiting));
  constexpr double firstAttempt = 1.0;
  double failedAt = 0.0;
  for (double now = firstAttempt; now < firstAttempt + 40000.0 && !events.failedCount;
       now += 100.0) {
    static_cast<void>(engine.tick(now));
    failedAt = now;
  }
  require(events.failures == std::vector<std::string>{"resource_timeout"} &&
              failedAt - firstAttempt >= 30000.0 && failedAt - firstAttempt <= 30200.0 &&
              executor.suspensions == 0,
          "a rolled-back admission moved its own wait limit out");
}

// Pages that land while a command runs keep a waiting lane past its limit
// until its next attempt, after the command. Until then that limit is no
// wake-up: tick() does not act on it, and the native loop would poll
// without blocking for as long as the command runs.
void testLimitOutlivedByProgressDoesNotWakeTheLoop() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvStorage storage(16, 4096, 2);
  storage.budgetPages = 14;
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.capacity = 64;
  tier.transferLimit = 8;
  engine::Cache cache(pool, &tier, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine({.maxContext = 102400}, cache, executor, events);
  guardReleases(storage, engine);
  // An eight-block prefix and its state move to disk entirely.
  std::vector<uint32_t> prompt(257, 17);
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 256).granted(), "fixture KV failed");
  demoteState(cache, test::publishBlocks(cache, 999, prompt, 256));
  cache.endRequest(999);
  for (uint32_t written = 1; written <= 8; ++written) {
    require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress && tier.demotions == written,
            "prefix block was not written");
    tier.complete();
    require(cache.pollTransfers(), "prefix block did not land");
  }
  // Six cached blocks under disk states.
  auto transfer = std::make_shared<OffloadControl>();
  transfer->ready = true;
  for (uint64_t id = 900; id < 906; ++id) {
    std::vector<uint32_t> filler(32, static_cast<uint32_t>(id));
    cache.beginRequest(id);
    require(cache.ensureTokens(id, 32).granted(), "filler KV failed");
    cache.publishCompositeState(test::publishBlocks(cache, id, filler, 32),
                                std::make_shared<OffloadState>(transfer), false);
    cache.endRequest(id);
    require(cache.reclaimOneState(false, 0, false) && cache.pollTransfers(),
            "filler state was not demoted");
  }
  EngineRequest running = request(1, std::vector<uint32_t>(33, 5));
  running.deadlineMilliseconds = 1e9;
  engine.submit(std::move(running));
  double now = 1.0;
  for (int step = 0; step < 4; ++step, now += 100.0)
    static_cast<void>(engine.tick(now));

  // The running lane's next decode command is held, and the lane restoring
  // the prefix waits for pages the tier is still writing.
  executor.restoreControl->ready = true;
  auto hold = std::make_shared<std::atomic<bool>>(false);
  executor.holdDecodeUntil = hold;
  EngineRequest waiting = request(2, prompt);
  waiting.deadlineMilliseconds = 1e9;
  engine.submit(std::move(waiting));
  for (int step = 0; step < 3; ++step, now += 100.0)
    static_cast<void>(engine.tick(now));
  require(engine.commandInFlight() && engine.resourceWaitSnapshot(now).memory == 1 &&
              tier.demotions == 10 && events.starts.size() == 1,
          "the lane did not wait for pages behind the command");
  // The pages land, and the command is still held when the limit passes.
  tier.complete();
  static_cast<void>(engine.tick(now));
  now += 31000.0;
  require(!engine.tick(now) && events.failedCount == 0 && engine.commandInFlight(),
          "the lane failed although pages landed since its last attempt");
  const auto wakeup = engine.nextWakeupMilliseconds();
  require(wakeup.has_value() && *wakeup > now,
          "a limit tick() does not act on woke the loop");

  // The command's completion wakes the loop, and the lane's next attempt
  // takes the pages.
  *hold = true;
  executor.decodeFinishes = true;
  for (int step = 0; step < 40 && !idle(engine); ++step, now += 1.0) {
    static_cast<void>(engine.tick(now));
    tier.complete();
  }
  require(idle(engine) && events.completedCount == 2 && events.failedCount == 0 &&
              events.starts.size() == 2 &&
              events.starts[1] == 256 && executor.restored == 256,
          "the lane did not run on its prefix after the command");
}

// A lane waits to grow into pages whose demotion is in flight, and a request
// admitted meanwhile starts and decodes. Once the pages land the waiting
// lane still lacks one and is suspended. The later lane was resident at its
// suspension and holds memory its resume waits for: while it works, the
// wait's limit restarts, and once it finishes the suspended lane resumes.
void testLaneAdmittedBeforeASuspensionHoldsTheWaitOpen() {
  test::TestKvStorage storage(16, 4096, 4);
  storage.budgetPages = 8;
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.transferLimit = 8;
  engine::Cache cache(pool, &tier, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  const test::ScopedTestConfig seam({.resourceWaitTimeoutMilliseconds = 50.0});
  engine::Engine engine({.maxContext = 102400}, cache,
                        executor, events);
  guardReleases(storage, engine);
  // Cached blocks under states on disk take the first extent, whose pages
  // come back only once their copies are written.
  for (uint64_t id = 900; id < 904; ++id) {
    cache.beginRequest(id);
    require(cache.ensureTokens(id, 32).granted(), "fixture KV failed");
    demoteState(cache, test::publishBlocks(
                           cache, id, std::vector<uint32_t>(32, static_cast<uint32_t>(id)), 32));
    cache.endRequest(id);
  }
  // Request 1's first command needs seven of the eight pages; its last needs
  // all eight.
  engine.submit(request(1, std::vector<uint32_t>(225, 1)));
  double now = 1;
  static_cast<void>(engine.tick(now++));
  require(executor.requests.contains(1) && tier.inFlight() != 0 && executor.prefillRows == 0,
          "the lane did not wait for the pages being demoted");
  // Request 2 takes two of the pages left free.
  auto later = request(2, std::vector<uint32_t>(40, 2));
  later.maxNewTokens = 1000;
  engine.submit(std::move(later));
  tickUntil(engine, now, [&] { return events.emitted != 0; },
            "the later request did not run beside the waiting lane");
  require(tier.inFlight() != 0 && executor.prefillRows == 40 && executor.suspensions == 0,
          "the waiting lane ran or yielded before its pages landed");
  for (uint32_t step = 0; step < 100 && !executor.suspensions; ++step) {
    tier.complete();
    static_cast<void>(engine.tick(now++));
  }
  require(executor.suspensions == 1 && executor.prefillRows == 40,
          "the waiting lane was not suspended once the pages landed");
  for (const double end = now + 400; now < end;) {
    static_cast<void>(engine.tick(now += 10));
    require(events.failedCount == 0 && engine.snapshot().resourceResumptions == 0,
            "a lane resident at a suspension did not hold the wait open");
  }
  executor.decodeFinishes = true;
  tickUntil(engine, now, [&] { return idle(engine); }, "engine did not reach idle");
  require(events.completedCount == 2 && events.failedCount == 0 &&
              engine.snapshot().resourceResumptions == 1,
          "the suspended lane did not resume once the lane beside it finished");
}

// A restoring lane whose pages are all held by a resident lane waits for
// that lane instead of giving up its prefix or failing for capacity.
void testRestoringLaneWaitsForResidentLanes() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvStorage storage(12, 4096, 4);
  storage.budgetPages = 8;
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  tier.transferLimit = 8;
  engine::Cache cache(pool, &tier, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({.maxContext = 102400}, cache, executor, events);
  guardReleases(storage, engine);
  // A two-block prefix and its state move to disk.
  std::vector<uint32_t> prompt(65, 17);
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 64).granted(), "fixture KV failed");
  demoteState(cache, test::publishBlocks(cache, 999, prompt, 64));
  cache.endRequest(999);
  for (uint32_t written = 1; written <= 2; ++written) {
    require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress && tier.demotions == written,
            "block was not written");
    tier.complete();
    require(cache.pollTransfers(), "block did not land");
  }
  // A long lane held in decode occupies the pool.
  auto hold = std::make_shared<std::atomic<bool>>(false);
  executor.holdDecodeUntil = hold;
  EngineRequest holder = request(2, std::vector<uint32_t>(200, 5));
  holder.deadlineMilliseconds = 1e9;
  engine.submit(std::move(holder));
  for (int step = 1; step < 6; ++step) static_cast<void>(engine.tick(step));
  require(pool.freePageCount() < 3 && engine.commandInFlight(), "the holder did not take the pool");

  executor.restoreControl->ready = true;
  EngineRequest waiting = request(1, prompt);
  waiting.deadlineMilliseconds = 1e9;
  engine.submit(std::move(waiting));
  for (int step = 10; step < 16; ++step) static_cast<void>(engine.tick(step));
  require(events.failedCount == 0 && executor.suspensions == 0 && events.starts.size() == 1,
          "the restoring lane failed or yielded instead of waiting");
  *hold = true;
  for (int step = 20; step < 60 && !idle(engine); ++step) {
    static_cast<void>(engine.tick(step));
    tier.complete();
  }
  require(idle(engine) && events.failedCount == 0 && executor.suspensions == 0 &&
              events.starts.size() == 2 && events.starts[1] == 64 &&
              executor.restored == 64,
          "the restoring lane did not run on its prefix once pages returned");
}

// A tool-using lane is runnable on every tick and keeps the model busy. A
// restore still completes beside it: its reads land whatever the model runs,
// and the restoring lane starts on its prefix while the other decodes.
void testRestoreCompletesWhileAConstrainedLaneDecodes() {
  constexpr auto reuse = CacheReclaimMode::KeepExtents;
  test::TestKvStorage storage(128, 4096, 4);
  KvPool pool(storage, 0);
  test::TestKvTier tier;
  engine::Cache cache(pool, &tier, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine({.maxContext = 102400}, cache, executor, events);
  guardReleases(storage, engine);
  // A two-block prefix and its state move to disk.
  std::vector<uint32_t> prompt(65, 17);
  cache.beginRequest(999);
  require(cache.ensureTokens(999, 64).granted(), "fixture KV failed");
  demoteState(cache, test::publishBlocks(cache, 999, prompt, 64));
  cache.endRequest(999);
  for (uint32_t written = 1; written <= 2; ++written) {
    require(cache.reclaimOne(reuse, ReclaimClass::InUse).madeProgress && tier.demotions == written,
            "block was not written");
    tier.complete();
    require(cache.pollTransfers(), "block did not land");
  }

  // The frontend answers every mask request on the tick that made it, and
  // transfers land between ticks.
  const std::array<uint32_t, 1> mask{1};
  uint32_t answered = 0;
  auto step = [&](double now) {
    static_cast<void>(engine.tick(now));
    for (; answered < events.maskRequests.size(); ++answered)
      engine.provideMask(events.maskRequests[answered].first, mask);
    tier.complete();
  };
  EngineRequest decoding = constrainedRequest(1, 1e9);
  decoding.maxNewTokens = 1000;
  engine.submit(std::move(decoding));
  double now = 1;
  for (; now < 20; ++now) step(now);
  require(events.emitted > 1 && events.completedCount == 0, "the constrained lane is not decoding");

  executor.restoreControl->ready = true;
  EngineRequest restoring = request(2, prompt);
  restoring.deadlineMilliseconds = 1e9;
  engine.submit(std::move(restoring));
  for (; now < 100 && events.startIds.size() < 2; ++now) step(now);
  require(events.startIds.size() == 2 && events.starts[1] == 64 &&
              executor.restored == 64,
          "the restore waited for the constrained lane to stop decoding");
  require(events.completedCount == 0 && events.failedCount == 0 &&
              cache.snapshot().kvTier.restores == 2,
          "the constrained lane stopped, or the restore did not land whole");
}

// The field failure: a long decode's own replay point was the oldest cache
// entry while another request prefilled, and growth reclaim took it before
// newer state and KV; the conversation's next turn replayed its prompt.
void testRunningRequestKeepsItsReplayPoint() {
  test::TestKvStorage storage(16, 4096, 4);
  storage.budgetPages = 12;
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  double now = 1;

  const std::vector<uint32_t> first(65, 1);
  auto conversation = request(1, first);
  conversation.maxNewTokens = 60;
  engine.submit(std::move(conversation));
  tickUntil(engine, now, [&] { return events.outputs[1].size() >= 4; },
            "the conversation did not decode");
  require(resources.snapshot().stateCache.inUse == 1,
          "the decode did not use its replay point");
  // A newer request runs to completion: its state and KV are newer than the
  // decode's replay point.
  engine.submit(request(2, std::vector<uint32_t>(97, 2)));
  tickUntil(engine, now, [&] { return events.usage.contains(2); },
            "the newer request did not finish");
  // Growth now needs cached pages back.
  storage.growthBlocked = true;
  executor.kvGrowthBlocked = &storage.growthBlocked;
  executor.unblockGrowthOnSuspend = false;
  engine.submit(request(3, std::vector<uint32_t>(97, 3)));
  tickUntil(engine, now, [&] { return events.usage.contains(1) && events.usage.contains(3); },
            "the conversation or the growing request did not finish");
  require(executor.suspensions == 0 && events.failedCount == 0,
          "the fixture suspended or failed a lane");

  std::vector<uint32_t> next = first;
  next.insert(next.end(), events.outputs[1].begin(), events.outputs[1].end());
  next.resize(next.size() + 40, 7);
  engine.submit(request(4, next));
  tickUntil(engine, now, [&] { return idle(engine); }, "the next turn did not finish");
  require(events.starts.back() == 64,
          "the next turn lost the replay point its predecessor ran from");
  const auto state = resources.snapshot().stateCache;
  require(state.inUse == 0 && state.inUseEvictions == 0,
          "a finished request still used its replay point");
}

// A lane suspended under host pressure keeps using its replay point, the
// oldest cache entry once a peer finishes. The growth of another peer that
// still runs reuses cached pages: it takes the finished peer's newer state
// and KV, not the point the suspended lane and its next turn resume from.
void testSuspendedRequestKeepsItsReplayPoint() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(3);
  executor.decodeFinishes = false;
  Events events;
  bool paused = false;
  EngineConfig config;
  config.growthPaused = [&] { return paused; };
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  double now = 1;
  // Every lane needs its next page at the same step: the shortest history
  // yields, the next one finishes first.
  const std::vector<uint32_t> first(81, 1);
  const std::vector<std::pair<std::vector<uint32_t>, uint32_t>> lanes{
      {first, 40}, {std::vector<uint32_t>(113, 2), 20}, {std::vector<uint32_t>(145, 3), 210}};
  for (uint64_t id = 1; id <= lanes.size(); ++id) {
    auto value = request(id, lanes[id - 1].first);
    value.maxNewTokens = lanes[id - 1].second;
    engine.submit(std::move(value));
  }
  tickUntil(engine, now, [&] { return !events.outputs[3].empty(); }, "the lanes did not decode");
  paused = true;
  storage.growthBlocked = true;
  storage.allocationFailure = metal::AllocationFailure::HostPressure;
  tickUntil(engine, now, [&] { return executor.suspensions == 1; }, "no lane was suspended");
  // The pool grows again while the engine still drains.
  storage.growthBlocked = false;
  tickUntil(engine, now, [&] { return events.usage.contains(2); }, "the peer did not finish");
  require(executor.resumptions == 0 && resources.snapshot().stateCache.inUse == 2,
          "the suspended lane and the running peer did not both use their points");
  storage.growthBlocked = true;
  tickUntil(engine, now, [&] { return events.usage.contains(3); },
            "the growing peer did not finish");
  std::vector<uint32_t> history = first;
  history.insert(history.end(), events.outputs[1].begin(), events.outputs[1].end());
  const auto state = resources.snapshot().stateCache;
  require(executor.resumptions == 0 && state.evictions == 1 && state.inUseEvictions == 0 &&
              resources.probe(history, {}).cachedTokens() == 64,
          "the growing peer took the suspended lane's replay point");
  paused = false;
  storage.growthBlocked = false;
  tickUntil(engine, now, [&] { return idle(engine); }, "the suspended lane did not finish");
  require(executor.restored == 64, "the suspended lane did not resume from its replay point");
  std::vector<uint32_t> next = first;
  next.insert(next.end(), events.outputs[1].begin(), events.outputs[1].end());
  next.resize(next.size() + 40, 7);
  engine.submit(request(4, next));
  tickUntil(engine, now, [&] { return idle(engine); }, "the next turn did not finish");
  require(events.starts.back() == 64 &&
              resources.snapshot().stateCache.inUse == 0,
          "the next turn lost the replay point its predecessor resumed from");
}

// A resumed lane keeps using its prompt's replay point. The point its
// generated history reaches lies inside the generation prompt, which the
// conversation's next turn renders anew: that state is the lane's own
// progress, a checkpoint, and once the lane has finished, reclaim takes it
// before the prompt's point.
void testResumedLaneKeepsThePromptReplayPoint() {
  test::TestKvStorage storage(8, 4096, 2);
  storage.budgetPages = 6;
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  for (uint64_t id : {264, 265}) {
    auto value = request(id, std::vector<uint32_t>(65, id));
    // The prompt's replay point lands at 32, the history's at 64.
    value.generationPromptTokens = 30;
    value.maxNewTokens = 30;
    engine.submit(std::move(value));
  }
  // Lookups keep both prompts' points resident while the lanes contend for
  // pages: the reclaim that ends in a yield takes what is in use too.
  std::vector<CacheLookup> held = runUntilStatesHeld(engine, resources, {264, 265});
  double now = 100;
  for (; now < 200 && executor.suspensions == 0; ++now)
    static_cast<void>(engine.tick(now));
  held.clear();
  // Each prompt's point, then the resumed lane's history point.
  for (; now < 400 && resources.snapshot().stateCache.entries < 3; ++now)
    static_cast<void>(engine.tick(now));
  require(executor.resumptions == 1 && executor.restored == 32 &&
              resources.snapshot().stateCache.entries == 3 &&
              resources.snapshot().stateCache.checkpointEntries == 1,
          "the resumed lane did not publish its history's point as a checkpoint");
  // Prompt tokens are the request id.
  const std::vector<uint32_t> history = executor.resumedPrompts.front();
  const uint64_t id = history.front();
  require(!events.usage.contains(id) && resources.probe(history, {}).cachedTokens() == 64,
          "the resumed lane finished early");
  for (; now < 400 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine), "the resumed lane did not finish");
  // The first state reclaim takes is the history's checkpoint.
  while (resources.snapshot().stateCache.entries == 3) {
    require(resources.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse).madeProgress,
            "the cache could not be reclaimed");
  }
  require(resources.snapshot().stateCache.checkpointEntries == 0 &&
              resources.probe(history, {}).cachedTokens() == 32,
          "the history's point outlasted the prompt's");
  // A shrink that keeps the resume point keeps the prompt's point.
  while (resources.reclaimOne(CacheReclaimMode::ReleaseExtents, ReclaimClass::InUse, true)
             .madeProgress) {
  }
  // The next turn keeps the text before the generation prompt.
  std::vector<uint32_t> next(35, static_cast<uint32_t>(id));
  next.resize(80, 7);
  require(resources.snapshot().stateCache.entries == 1 &&
              resources.probe(next, {}).cachedTokens() == 32,
          "the shrink did not keep the prompt's replay point");
  engine.submit(request(266, next));
  for (; now < 500 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && events.starts.back() == 32,
          "the next turn did not resume from the prompt's replay point");
}

// A lane suspended right after publishing the end of its generated history
// resumes from that checkpoint, which stays one: it is the lane's own
// progress, not where the conversation's next turn resumes, so once the lane
// has finished, reclaim takes it before the prompt's replay point.
void testRestoredHistoryCheckpointStaysDisposable() {
  test::TestKvStorage storage(64, 4096, 1);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  bool paused = false;
  EngineConfig config;
  config.growthPaused = [&] { return paused; };
  engine::Engine engine(config, resources, executor, events);
  guardReleases(storage, engine);
  const auto hostPressure = [&](bool refused) {
    paused = refused;
    storage.growthBlocked = refused;
    storage.allocationFailure = refused ? metal::AllocationFailure::HostPressure
                                        : metal::AllocationFailure::EngineBudget;
  };
  auto value = request(1, std::vector<uint32_t>(65, 5));
  // The prompt's replay point lands at 32; a history of 89 tokens ends at 64.
  value.generationPromptTokens = 30;
  value.maxNewTokens = 30;
  value.priority = RequestPriority::Background;
  engine.submit(std::move(value));
  double now = 1;
  tickUntil(engine, now, [&] { return !events.outputs[1].empty(); }, "the lane did not decode");
  // The page its 24th token needs is refused: the lane yields with a history
  // of 89 tokens, resumes from the prompt's point and checkpoints at 64.
  hostPressure(true);
  tickUntil(engine, now, [&] { return executor.suspensions == 1; }, "the lane was not suspended");
  hostPressure(false);
  tickUntil(engine, now, [&] { return engine.snapshot().checkpointPublications == 1; },
            "the resumed lane did not checkpoint its history's end");
  // A foreground start refused memory makes it yield again at once.
  hostPressure(true);
  auto foreground = request(2, std::vector<uint32_t>(65, 9));
  foreground.priority = RequestPriority::Foreground;
  engine.submit(std::move(foreground));
  tickUntil(engine, now, [&] { return executor.suspensions == 2; },
            "the lane did not yield to the foreground start");
  hostPressure(false);
  tickUntil(engine, now, [&] { return idle(engine); }, "the requests did not finish");
  const std::vector<uint32_t> history = executor.resumedPrompts.back();
  require(executor.resumptions == 2 && executor.restored == 32 + 64 && history.size() == 89 &&
              events.completedCount == 2 &&
              resources.snapshot().stateCache.checkpointEntries == 1,
          "the lane did not resume from its history's checkpoint, or made it ordinary");
  const uint32_t entries = resources.snapshot().stateCache.entries;
  while (resources.snapshot().stateCache.entries == entries) {
    require(resources.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse).madeProgress,
            "the cache could not be reclaimed");
  }
  require(resources.snapshot().stateCache.checkpointEntries == 0 &&
              resources.probe(history, {}).cachedTokens() == 32,
          "the history's checkpoint outlasted the prompt's replay point");
}

// Requests with the same prompt resume from the same replay point and each
// use it: the first to end leaves it in use.
void testSharedReplayPointCountsEachRequest() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  for (uint64_t id : {1, 2}) {
    auto value = request(id, std::vector<uint32_t>(65, 7));
    value.maxNewTokens = id == 1 ? 10 : 40;
    engine.submit(std::move(value));
  }
  double now = 1;
  for (; now < 100 && !events.usage.contains(1); ++now)
    static_cast<void>(engine.tick(now));
  require(events.startIds.size() == 2 && executor.restored == 64 &&
              events.usage.contains(1) && !events.usage.contains(2),
          "the fixture did not overlap two requests on one replay point");
  require(resources.snapshot().stateCache.inUse == 1,
          "the first request to end released its peer's replay point");
  for (; now < 200 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && resources.snapshot().stateCache.inUse == 0,
          "the last request to end kept its replay point in use");
}

// A prompt sent again restores the replay point its first run published and
// uses it while it runs.
void testRestoredEndpointIsInUse() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  const std::vector<uint32_t> prompt(65, 7);
  engine.submit(request(1, prompt));
  runUntilIdle(engine);
  executor.decodeFinishes = false;
  auto again = request(2, prompt);
  again.maxNewTokens = 10;
  engine.submit(std::move(again));
  double now = 100;
  for (; now < 120 && events.outputs[2].size() < 2; ++now)
    static_cast<void>(engine.tick(now));
  require(events.starts.back() == 64 &&
              resources.snapshot().stateCache.inUse == 1,
          "a restored replay point was not in use");
  for (; now < 200 && !idle(engine); ++now)
    static_cast<void>(engine.tick(now));
  require(idle(engine) && resources.snapshot().stateCache.inUse == 0,
          "the restored replay point stayed in use");
}

// A request's replay point is in use from the moment it is reached. With no
// cache slot free and only another running request's point cached, its
// publication recycles that older point, as no ordinary publication may.
void testReplayPointRecyclesAnOlderPointInUse() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor;
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  double now = 1;
  const auto decode = [&](uint64_t id) {
    auto value = request(id, std::vector<uint32_t>(65, id));
    value.maxNewTokens = 1000;
    engine.submit(std::move(value));
    tickUntil(engine, now, [&] { return events.outputs[id].size() >= 2; },
              "a request did not decode");
  };
  decode(1);
  require(resources.snapshot().stateCache.entries == 1 &&
              resources.snapshot().stateCache.inUse == 1,
          "the older request did not cache its replay point");
  // No snapshot slot is free for the newer request's point.
  executor.deniedSnapshots = 1;
  decode(2);
  const auto snapshot = engine.snapshot();
  require(snapshot.recycledStatePublications == 1 &&
              snapshot.replayStatePublicationFailures == 0 &&
              snapshot.resources.stateCache.entries == 1 &&
              snapshot.resources.stateCache.inUse == 2 &&
              snapshot.resources.stateCache.inUseEvictions == 1,
          "the newer replay point did not recycle the older point in use");
}

// The pressure warning's speculative shrink keeps the point it kept before
// states could be in use: a request that just finished outranks an older
// running request's replay point, which goes, after the older ordinary
// state, like any cache the shrink may take.
void testWarningShrinkKeepsTheFinishedPoint() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  double now = 1;
  engine.submit(request(1, std::vector<uint32_t>(65, 1)));
  tickUntil(engine, now, [&] { return idle(engine); }, "the older conversation did not finish");
  executor.decodeFinishes = false;
  auto running = request(2, std::vector<uint32_t>(65, 2));
  running.maxNewTokens = 1000;
  engine.submit(std::move(running));
  tickUntil(engine, now, [&] { return events.outputs[2].size() >= 2; },
            "the long decode did not start");
  auto finished = request(3, std::vector<uint32_t>(65, 3));
  finished.maxNewTokens = 2;
  engine.submit(std::move(finished));
  tickUntil(engine, now, [&] { return events.usage.contains(3); },
            "the newer request did not finish");
  require(resources.snapshot().stateCache.entries == 3 &&
              resources.snapshot().stateCache.inUse == 1,
          "the fixture did not cache three replay points, one in use");

  static_cast<void>(engine.reclaimMemory(
      {.targetBytes = std::numeric_limits<uint64_t>::max(), .keepResumePoint = true}));
  const auto cachedTokens = [&](uint32_t token) {
    std::vector<uint32_t> next(65, token);
    next.resize(80, 9);
    return resources.probe(next, {}).cachedTokens();
  };
  const auto state = resources.snapshot().stateCache;
  require(state.entries == 1 && state.inUseEvictions == 1 && cachedTokens(3) == 64 &&
              cachedTokens(1) == 0 && cachedTokens(2) == 0,
          "the shrink did not keep the finished request's point over the running one");
}

// A start beside a resident lane is held back by it (judge() yields), so the
// reclaim for its memory takes nothing in use: the running request's replay
// point outlasts the start's state allocation, which the budget refuses, or
// the host, so that the lane would take a cached state's buffers, and the
// restore of its cached prefix, whose pages the budget refuses. The start
// waits for the lane and runs once it has finished.
void testHeldBackStartTakesNothingInUse() {
  enum class Refused { State, PausedState, Restore };
  for (const Refused refused : {Refused::State, Refused::PausedState, Refused::Restore}) {
    test::TestKvStorage storage(16, 4096, 4);
    KvPool pool(storage, 0);
    test::TestKvTier tier;
    engine::Cache resources(pool, &tier, nullptr);
    Executor executor;
    executor.decodeFinishes = false;
    Events events;
    engine::Engine engine({}, resources, executor, events);
    guardReleases(storage, engine);
    const std::vector<uint32_t> held(65, 2);
    if (refused == Refused::Restore) {
      // The start's prompt matches a prefix only the disk holds, and the
      // budget holds no extent beyond the one the running lane fills.
      resources.beginRequest(999);
      require(resources.ensureTokens(999, 64).granted(), "fixture KV failed");
      demoteState(resources, test::publishBlocks(resources, 999, held, 64));
      resources.endRequest(999);
      for (uint32_t written = 1; written <= 2; ++written) {
        require(resources.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::InUse)
                        .madeProgress &&
                    tier.demotions == written,
                "block was not written");
        tier.complete();
        require(resources.pollTransfers(), "block did not land");
      }
      storage.budgetPages = 4;
    }
    double now = 1;
    const std::vector<uint32_t> first(65, 1);
    auto running = request(1, first);
    running.maxNewTokens = 1000;
    engine.submit(std::move(running));
    tickUntil(engine, now, [&] { return events.outputs[1].size() >= 2; },
              "the running request did not decode");
    // Nothing but what is in use is left to reclaim.
    while (resources.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::Ordinary)
               .madeProgress) {
    }
    // The one state in RAM is the running request's replay point.
    require(resources.snapshot().stateCache.inUse == 1 &&
                resources.snapshot().stateCache.bytes == 64,
            "the running request did not use its replay point");
    if (refused == Refused::State)
      executor.deniedBegins = 1000;
    // Refused by the host, the lane takes a cached state's buffers instead
    // (reuseCachedStateWhilePaused); the one cached state is in use.
    bool hostRefuses = refused == Refused::PausedState;
    executor.statesLacked = [&] { return hostRefuses ? 1U : 0U; };
    executor.beginGrowthBlocked = [&] { return hostRefuses; };
    executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
    engine.submit(request(2, held));
    for (int step = 0; step < 20; ++step)
      static_cast<void>(engine.tick(now++));
    std::vector<uint32_t> next = first;
    next.insert(next.end(), events.outputs[1].begin(), events.outputs[1].end());
    next.resize(next.size() + 40, 7);
    require(resources.snapshot().stateCache.inUseEvictions == 0 &&
                resources.probe(next, {}).cachedTokens() == 64 &&
                engine.snapshot().scheduler.waitingResources == 1 &&
                !executor.requests.contains(2),
            "a start held back by a resident lane took the lane's replay point");
    executor.decodeFinishes = true;
    executor.deniedBegins = 0;
    hostRefuses = false;
    executor.restoreControl->ready = true;
    for (const double end = now + 200; now < end && !idle(engine); ++now) {
      static_cast<void>(engine.tick(now));
      tier.complete();
    }
    require(idle(engine) && events.completedCount == 2 && events.failedCount == 0 &&
                events.startIds.back() == 2 &&
                executor.diskReads == (refused == Refused::Restore ? 1U : 0U),
            "the held-back start did not run once the lane finished");
  }
}

// A lane that has just started beside a decoding one is the first to yield
// (laneToYield): when its first chunk's pages are refused, it is suspended.
// The reclaim before that takes nothing in use, so the decoding
// conversation's replay point is not spent on a lane that does not run.
void testLaneThatWouldYieldTakesNothingInUse() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache resources(pool, nullptr, nullptr);
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  guardReleases(storage, engine);
  double now = 1;
  const std::vector<uint32_t> first(65, 1);
  auto conversation = request(1, first);
  conversation.maxNewTokens = 1000;
  engine.submit(std::move(conversation));
  tickUntil(engine, now, [&] { return events.outputs[1].size() >= 2; },
            "the conversation did not decode");
  require(resources.snapshot().stateCache.inUse == 1,
          "the decode did not use its replay point");
  while (resources.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::Ordinary)
             .madeProgress) {
  }
  storage.growthBlocked = true;
  engine.submit(request(2, std::vector<uint32_t>(65, 2)));
  tickUntil(engine, now, [&] { return executor.suspensions == 1; },
            "the starting lane was not suspended");
  std::vector<uint32_t> next = first;
  next.resize(first.size() + 40, 7);
  require(resources.snapshot().stateCache.inUseEvictions == 0 &&
              resources.probe(next, {}).cachedTokens() == 64,
          "the lane that yielded took the decoding conversation's replay point");
  require(!executor.requests.at(2).resident && executor.requests.at(1).resident &&
              !events.usage.contains(1),
          "the starting lane did not yield to the decoding one");
  executor.decodeFinishes = true;
  storage.growthBlocked = false;
  tickUntil(engine, now, [&] { return idle(engine); }, "the suspended lane did not finish");
  require(events.completedCount == 2 && events.failedCount == 0,
          "the lanes did not both complete");
}

// Every end of a decoding request releases its replay point: cancellation
// and failure at once, before the request leaves the engine, the deadline
// and capacity exhaustion when the engine ends it.
void testEveryEndReleasesTheReplayPoint() {
  enum class End { Cancel, Failure, Deadline, Capacity };
  for (const End end : {End::Cancel, End::Failure, End::Deadline, End::Capacity}) {
    // Without a peer or cached KV to take, the decode's growth past three
    // pages exhausts the capacity, after it took the request's own point as
    // the last thing left.
    test::TestKvStorage storage(64, 4096, 1);
    if (end == End::Capacity)
      storage.budgetPages = 3;
    KvPool pool(storage, 0);
    engine::Cache resources(pool, nullptr, nullptr);
    Executor executor;
    executor.decodeFinishes = false;
    Events events;
    engine::Engine engine({}, resources, executor, events);
    guardReleases(storage, engine);
    auto value = request(1, std::vector<uint32_t>(65, 7));
    value.maxNewTokens = 1000;
    value.deadlineMilliseconds = end == End::Deadline ? 50 : 10000;
    engine.submit(std::move(value));
    double now = 1;
    for (; now < 40 && (events.outputs[1].size() < 2 || engine.commandInFlight()); ++now)
      static_cast<void>(engine.tick(now));
    require(resources.snapshot().stateCache.inUse == 1,
            "the decode did not use its replay point");
    if (end == End::Cancel)
      engine.cancel(1);
    else if (end == End::Failure)
      engine.failRequest(1, LaneOutcome::InvalidMask, "the request failed");
    require(end == End::Deadline || end == End::Capacity ||
                resources.snapshot().stateCache.inUse == 0,
            "the request's end did not release its replay point at once");
    for (now = 100; now < 200 && !idle(engine); ++now)
      static_cast<void>(engine.tick(now));
    const auto state = resources.snapshot().stateCache;
    require(idle(engine) && state.inUse == 0 &&
                state.entries == (end == End::Capacity ? 0U : 1U) &&
                events.capacityExhaustedCount == (end == End::Capacity ? 1U : 0U),
            "a request's end did not release its replay point");
  }
}

// A suspended lane keeps its replay point while it waits for the disk copy
// and after the copy fails to load. Cancellation or the deadline during the
// read, and cancellation while it waits again, release the point.
void testWaitingEndsReleaseTheReplayPoint() {
  enum class End { CancelDuringRead, DeadlineDuringRead, CancelAfterFailedRead };
  for (const End end : {End::CancelDuringRead, End::DeadlineDuringRead,
                        End::CancelAfterFailedRead}) {
    test::TestKvStorage storage(64, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor executor;
    executor.deniedSnapshots = 1000;
    executor.stateTier = std::make_shared<OffloadControl>();
    executor.stateTier->ready = true;
    Events events;
    engine::Engine engine({}, cache, executor, events);
    guardReleases(storage, engine);
    storage.allocationFailure = metal::AllocationFailure::HostPressure;
    auto value = request(290, std::vector<uint32_t>(129, 290));
    value.deadlineMilliseconds = 1000;
    engine.submit(std::move(value));
    // The replay state at 128 goes to disk; the last prompt token's page
    // suspends the request, and its resumption waits for the read.
    require(engine.tick(1) && engine.tick(2) &&
                engine.snapshot().diskStatePublications == 1,
            "fixture did not write the replay state to disk");
    storage.growthBlocked = true;
    require(engine.tick(3) && engine.snapshot().resourceSuspensions == 1,
            "the last prompt token's page did not suspend the request");
    storage.growthBlocked = false;
    static_cast<void>(engine.tick(103));
    require(executor.diskReads == 1 && cache.snapshot().stateCache.inUse == 1,
            "the resumption did not wait for its replay point");
    executor.restoreControl->ready = true;
    if (end == End::CancelDuringRead) {
      engine.cancel(290);
      static_cast<void>(engine.tick(104));
    } else if (end == End::DeadlineDuringRead) {
      static_cast<void>(engine.tick(1000));
    } else {
      executor.resumeDenied = true;
      executor.restoreControl->success = false;
      static_cast<void>(engine.tick(104));
      require(!idle(engine) && !engine.commandInFlight() &&
                  cache.snapshot().stateCache.inUse == 1,
              "a failed read ended the replay point's use");
      engine.cancel(290);
      require(cache.snapshot().stateCache.inUse == 0,
              "cancelling the waiting lane did not release its replay point");
      static_cast<void>(engine.tick(105));
    }
    require(idle(engine) && cache.snapshot().stateCache.inUse == 0 &&
                events.completedCount + events.failedCount == 1,
            "a waiting request's end did not release its replay point");
  }
}

int main() {
  try {
    testRunningRequestKeepsItsReplayPoint();
    testSuspendedRequestKeepsItsReplayPoint();
    testResumedLaneKeepsThePromptReplayPoint();
    testRestoredHistoryCheckpointStaysDisposable();
    testSharedReplayPointCountsEachRequest();
    testRestoredEndpointIsInUse();
    testReplayPointRecyclesAnOlderPointInUse();
    testWarningShrinkKeepsTheFinishedPoint();
    testHeldBackStartTakesNothingInUse();
    testLaneThatWouldYieldTakesNothingInUse();
    testEveryEndReleasesTheReplayPoint();
    testWaitingEndsReleaseTheReplayPoint();
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
    testSharedJunctionAtACheckpointIsReusable();
    testSharedPrefillCapacityFailureDoesNotDeadlock();
    testSiblingFailedInTheAdmissionPassGetsNoJunction();
    testRestoringLaneWaitsForResidentLanes();
    testRestoreCompletesWhileAConstrainedLaneDecodes();
    testWaitWithProgressOutlivesTheResourceLimit();
    testRolledBackAdmissionKeepsItsWaitLimit();
    testLimitOutlivedByProgressDoesNotWakeTheLoop();
    testLaneAdmittedBeforeASuspensionHoldsTheWaitOpen();
    testDiskKvPrefixIsRestoredBeforeTheLaneRuns();
    testSkipCacheCandidateIsNotProbed();
    testRefusedPrefixRestorePreemptsLowerResident();
    testRestoringRequestIsNotWaitingForMemory();
    testSharedPrefillWaitsForARestoringProducer();
    testCancelledRestoringProducerReleasesItsWaiter();
    testSkipCacheRequestDoesNotWaitForAProducer();
    testCancelledDiskPrefixStopsQueuedReads();
    testPagesReturnFromDemotionWithoutSuspending();
    testRefusedRestoreClosesAdmission();
    testFailedDiskRestoreKeepsShallowerState();
    testRepeatedDiskHitPromotesToMemory();
    testGrowthWaitsForTheStateWriteInFlight();
    testPageShortfallDemotesInBulk();
    testWaitingLaneAlwaysNamesAWakeup();
    testReplayPointsAreWrittenBehind();
    testWaitingPlanDoesNotIdleRunnableLanes();
    testNothingInFlightIsNotPending();
    testKvGrowthProceedsThroughDemotion();
    testAsyncRestoreLifecycle();
    testDiskHitWithNoActiveMemory();
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
    testJunctionRetiresEarlierProgressPoint();
    testShortSuffixContinuesCheckpointDraftState();
    testDefaultCheckpointRestoresLatestCommittedPrefix();
    testCheckpointsSkipNearResumeAndReplayBoundaries();
    testDecodeShareValidation();
    testColdPublishesReplayStateAndRebuildsALostOne();
    testConcurrentDuplicateStateSkipsSnapshotCapture();
    testReplayStateEndsBeforeTheGenerationPrompt();
    testFollowUpResumesBeforeTheGenerationPrompt();
    testRetryPublishesNoStateInsideTheGenerationPrompt();
    testSharedJunctionEndsBeforeTheGenerationPrompt();
    testImageSpansKeyPrefixIdentity();
    testSharedPrefillBoundaryStopsAtTheFirstDifferentImage();
    testOneRequestPublishesJunctionAndLatestReplayState();
    testLazyJunctionNeedsADraftWindowOfGain();
    testLatestReplayDenialRecyclesOlderStateNotTheJunction();
    testCancellationAfterJunctionDiscardsLaterState();
    testPublicationInvariantFailureIsFatal();
    testDeniedSnapshotCostsOnlyThatAttempt();
    testDeniedSnapshotRecyclesLruStateAndRetries();
    testPersistentSnapshotDenialRecyclesAtMostOneState();
    testStateWithoutACacheSlotGoesToDisk();
    testStateAlreadyOnDiskIsDeduplicated();
    testCancelledPrefillRecoversFromItsDiskCheckpoint();
    testFailedFinalStateKeepsTheDiskCheckpoint();
    testNoCheckpointWithinAChunkOfTheReplayBoundary();
    testSkippedCheckpointKeepsPreviousRecoveryPoint();
    testLongSuffixSkipsDraftRestore();
    testCancellationInFlightAtBoundaryPublishesNoState();
    testLaneStateGrowthReclaimsCachedStateAndRetries();
    testKvGrowthReclaimsIdleStateBeforeCache();
    testKvGrowthDenialKeepsEveryLaneReplayState();
    testPressureReclaimRespectsStateLifetimes();
    testPressurePassKeepsCachesWithoutATarget();
    testPressurePassTakesTheRowsItsEvictionsLeave();
    testWarningReclaimKeepsTheServingFootprint();
    testWarningReclaimCountsOnlyWhatReachesTheHost();
    testWarningReclaimWakesARefusedStart();
    testPressureReclaimFollowsTheChain();
    testFullLanesSkipAdmissionAttempts();
    testConcurrencyLimitDoesNotEvictCache();
    testHostPressureDoesNotDrainCacheOnStateAdmission();
    testHostPressureStillRecyclesLruStateForDeniedSnapshot();
    testRequestInServiceGrowsThroughTheHostPause();
    testPausedPageShortfallKeepsTheLaneRunway();
    testFirstRequestStartsThroughTheHostPause();
    testRequestThatNeedsNoGrowthStartsUnderTheHostPause();
    testSuspendedRequestWaitsForTheHostBesideOneInService();
    testEngineLimitBindsThroughTheHostPause();
    testSingletonHostPressureReusesIdleCacheInsteadOfSuspending();
    testPausedPageShortageKeepsCachedStates();
    testSingletonHostPressureWaitRecoversOrTerminates();
    testAdmissionWaitsOutEarlierLanes();
    testLaterLanesDoNotExtendAResourceWait();
    testLaneAdmittedBeforeARefusalHoldsTheWaitOpen();
    testKvPressureNarrowsTheRealBatch();
    testKvGrowthReclaimsCachedStateWhenBudgetIsShared();
    testRequiredWorkDoesNotReserveAnExtraPage();
    testAdmissionPinsDesiredStateAndCountsOnlySuccess();
    testAdmissionCanDropItsOwnCachePinToMakeProgress();
    testActivationReceivesTheRestoreBoundary();
    testRefusedStartKeepsItsLeaseBesideAResidentLane();
    testRefusedStartKeepsItsLeaseWhileMemoryIsPending();
    testDroppedLeaseLooksUpAgain();
    testSingletonCapacityFailureTerminatesCleanly();
    testQueuedLongPrefillsLeaveRoomForShortWork();
    testAdmissionUsesCachedRemainingWork();
    testUnselectableCandidatesAreNotProbed();
    testWaitingCandidateProbeIsRefreshedNotRepeated();
    testMemoryWaitHoldsBackLaterArrivals();
    testMemoryWaitClosesAdmissionBesideAResidentLane();
    testClosedAdmissionReopensWhenTheWaitEnds();
    testPrefixWaitKeepsAdmissionClosedBehindARefusal();
    testRefusedResumeHoldsBackLaterSuspendedLanes();
    testHeldSuspendedRequestNeitherWakesNorExpires();
    testStatusCountsRequestsHeldBehindARefusal();
    testStatusCountsSuspendedRequestsHeldDuringRecovery();
    testSchedulingWaitDoesNotConsumeMemoryTimeout();
    testUnadmittedRequestsHonorCancellationAndDeadline();
    testGrowthKeepsPrefillProgressWhenAnUnstartedPeerCanYield();
    testGrowthYieldsLowerPriorityResidentOutsideBatch();
    testPrefillGrowthPreservesAnActiveDecodePeer();
    testKvPressureSuspendsInsteadOfKillingActiveWork();
    testRecoveryDrainDoesNotConsumeResourceWaitBudget();
    testRecoveryDrainEndsWithItsCause();
    testRecoveryDrainEndsWhenAResidentReleasesMemory();
    testHigherPriorityArrivalIsNotHeldByRecovery();
    testForegroundArrivalPreemptsWhenLanesAreFull();
    testRefusedHigherPriorityStartPreemptsLowerResident();
    testPrefixWaiterTakesNoLaneUntilThePrefixLands();
    testRequestBelowARunningPriorityTakesNoLane();
    testPressureRetryIsBackedOffWithoutProgress();
    testAdmissionRetryWakesOnlyWhenTickCanRetry();
    testDecodePreemptionReplaysCommittedHistoryWithoutRepeatingOutput();
    testLongDecodePreemptionPlansTheCurrentReplayBoundary();
    testResumedProducerPlansJunctionsForSiblings();
    testResumedLaneRebuildsItsPointFromCachedKv(1);
    testResumedLaneRebuildsItsPointFromCachedKv(2);
    testDeniedSnapshotTakesAtMostOneSnapshotOfExtents();
    testSharedJunctionPublishesFromCachedKv();
    testPausedPublicationInUseTakesNoKv();
    testPreemptedDecodeRestoresItsResidentCompositeState();
    testPreemptedDecodeReplayBoundaryIgnoresTheGenerationPrompt();
    testRepeatedPreemptionRespectsBackoffAndCancellation();
    testAdmissionReopensAfterLastSuspendedRequestResumes();
    testRecoveryAdmitsFailedKvTargetBeforeReplaying();
    testFailedResumeRestoreKeepsTheKvTarget();
    testBudgetDenialRetriesAfterRelease();
    testStateAdmissionKeepsThePooledLaneBuffers();
    testPausedStateAdmissionReusesCachedStates();
    testDeniedGrowthAllocatesEachExtentOnce();
    testGrowthBeyondTheBudgetFailsAtOnce();
    testReclaimPassReleasesEveryEmptyExtent();
    testReclaimRefusesACommandInFlight();
    testStateStartGathersFreePagesBeforeEvicting();
    testCompactionWaitsForTheCommandInFlight();
    testAllocationCausesRemainDistinct();
    testAdmissionRespectsPriorityBeforeHashOrder();
    testConstraintMaskOverlapsInsideOneSchedulerBatch();
    testConstraintMaskWaitHonorsCancelAndDeadline();
    testUnansweredVerifyMaskFailsOnlyItsRequest();
    testUnansweredInitialMaskFails();
    testDecodeCycleCoversHostWork();
    testConstrainedPrefillRequestsMaskWithoutDecodeSlot();
    testDecodeNearContextCeilingCoversVerifyRows();
    testExpiredMaskWaitFinalizesWhileAnotherCommandRuns();
    testOrdinaryInFlightDeadlineDrainsWithoutPublishingOrOutput();
    testStalledSuspensionFailsWithCapacity();
    testTerminalAnchorWithoutKvIsNotCached();
    testPrefillCanCompleteTheRequest();
    testOutOfVocabularyOutputFailsLaneOnly();
    std::cout << "KV-first engine tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "KV-first engine tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

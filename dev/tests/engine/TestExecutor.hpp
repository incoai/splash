#pragma once

// The model the engine, native loop and transport tests run their engines
// on, and the states and tickets it hands them.

#include "TestImmediateTicket.hpp"
#include "TestKvPool.hpp"
#include "engine/KvCache.hpp"
#include "model/Model.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace splash::test {

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

// A model that runs no kernels, for the engine, native loop and transport
// tests. It behaves as model::Runtime does where those tests rely on it: a
// request takes the first free one of maximumLanes lanes; the prefill step
// that completes a prompt asks for a constrained request's first mask, or
// returns a score request's option logits; every constrained decode cycle
// after the first token waits for its verify mask inside its command
// (MaskOverlapTicket); and provideMask throws for a request that waits for
// no mask and refuses one that permits no token. Its other members are the
// tests' knobs and records.
class Executor final : public model::Model {
public:
  explicit Executor(
      uint32_t maximumLanes = model::ExecutionLimits::maximumBatchWidth)
      : maximumLanes(maximumLanes) {}

  void checkHealth() override {
    if (healthCheck)
      healthCheck();
  }

  StateAdmission begin(const ModelRequest &request) override {
    ++beginAttempts;
    lastBeginId = request.id;
    lastRestoredTokens = request.restoredTokens;
    if (beginObserver) beginObserver();
    beganFlags[request.id] = request.flags;
    beganSampling[request.id] = request.sampling;
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
                        request.constraint == ConstraintMode::TokenMask,
                    .scoreTokens = {request.scoreTokens.begin(),
                                    request.scoreTokens.end()}});
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
        for (uint64_t row = (item.logicalPosition + engine::KvCache::pageTokens - 1) /
                            engine::KvCache::pageTokens * engine::KvCache::pageTokens;
             row < end; row += engine::KvCache::pageTokens) {
          kv->content.at(item.pageTable[row / engine::KvCache::pageTokens]) =
              rows(row / engine::KvCache::pageTokens, item.inputTokens[row - item.logicalPosition]);
        }
      }
      Request &state = requests.at(item.requestId);
      state.position += item.tokenCount;
      prefillRows += item.tokenCount;
      ++prefillChunks[item.requestId];
      // The step that completes the prompt selects its first token; a replay
      // never does again. A constrained prompt's end waits for the first
      // mask instead, and a score request's returns its option logits.
      const bool last = !state.replaying &&
                        state.position == prompts.at(item.requestId).size();
      const bool awaitsMask = state.constrained && last;
      // The wait outlives a suspension, as the host continuation keeps it.
      state.awaitingMask = state.awaitingMask || awaitsMask;
      ModelStepResult step{item.requestId, item.tokenCount, {}, false,
                           awaitsMask ? DecodeStage::ApplyInitialMask
                                      : DecodeStage::Regular,
                           0, 0};
      if (last && !state.scoreTokens.empty()) {
        step.finished = true;
        if (invalidScores.contains(item.requestId))
          step.failure = "score logit is not finite";
        for (size_t index = 0; step.failure.empty() && index < state.scoreTokens.size();
             ++index)
          step.scoreLogits.push_back(static_cast<float>(index) + 0.5f);
      } else if (last && prefillAnchor) {
        // Prefill selected a stop token or the last budgeted token: emitted
        // now, without a KV row, and the request never decodes.
        step.outputTokens = {42};
        step.outputTokensWithoutKv = 1;
        step.finished = *prefillAnchor;
      } else if (last && firstTokens && !state.constrained) {
        step.firstToken = 42;
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
      if (uint64_t{item.pageTable.size()} * engine::KvCache::pageTokens <
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
                : std::vector<uint32_t>(decodeTokens, 42);
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
    std::unique_ptr<ModelBatchTicket> ticket;
    if (plan.kind == WorkKind::Decode && plan.constrained &&
        plan.decodeStage == DecodeStage::Regular) {
      // Every constrained cycle after the first token waits for its mask
      // inside the ticket, as the production constrained ticket does.
      overlap = std::make_shared<MaskOverlapState>();
      overlap->requestId = items.front().requestId;
      overlap->finishes = decodeFinishes;
      ticket = std::make_unique<MaskOverlapTicket>(overlap);
    } else if (plan.kind == WorkKind::Decode && holdDecodeUntil) {
      heldCompletion = std::move(completion);
      ticket = std::make_unique<HeldTicket>(decode(plan, items), holdDecodeUntil,
                                            1.0);
    } else if (plan.kind == WorkKind::Prefill && holdPrefillUntil) {
      heldCompletion = std::move(completion);
      ticket = std::make_unique<HeldTicket>(prefill(plan, items),
                                            holdPrefillUntil, 1.0);
    } else {
      ticket = immediateTicket(plan.kind == WorkKind::Prefill ? prefill(plan, items)
                                                              : decode(plan, items),
                               completion);
    }
    if (submitObserver)
      submitObserver();
    return ticket;
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
                                         std::span<const uint32_t> words) override {
    Request &entry = requests.at(id);
    const bool verify = overlap && overlap->emitted && overlap->requestId == id &&
                        !overlap->provided && !overlap->abandoned;
    if (!verify && !entry.awaitingMask)
      throw std::logic_error("request is not waiting for a token mask");
    if (std::none_of(words.begin(), words.end(), [](uint32_t word) { return word != 0; }))
      return "token mask row permits no vocabulary token";
    if (verify)
      overlap->provided = true;
    else
      entry.awaitingMask = false;
    ++providedMasks;
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
        std::min<uint64_t>(position, prompt.size()) / engine::KvCache::pageTokens;
    for (uint32_t block = 0; block < blocks; ++block) {
      if (kv->content.at(item.pageTable[block]) !=
          rows(block, prompt[block * engine::KvCache::pageTokens]))
        throw std::logic_error("a page does not hold its block's rows");
    }
  }

  struct Request {
    uint32_t lane = 0;
    uint32_t position = 0;
    bool resident = false;
    bool replaying = false;
    bool constrained = false;
    std::vector<uint32_t> scoreTokens;
    // Waits for its first token's mask.
    bool awaitingMask = false;
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
  TestKvStorage *kv = nullptr;
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
  // Whether the step that completes an unconstrained request's prompt reports
  // its first token (ModelStepResult::firstToken), as the runtime does. It is
  // 42, which the first decode step's tokens start with.
  bool firstTokens = false;
  std::shared_ptr<std::atomic<bool>> holdDecodeUntil;
  std::shared_ptr<std::atomic<bool>> holdPrefillUntil;
  // The completion of the latest held command, for the test to call as
  // Metal's completion handler would.
  std::function<void()> heldCompletion;
  std::shared_ptr<MaskOverlapState> overlap;
  std::function<void()> healthCheck;
  // Runs once each command is submitted, a held one's completion stored.
  std::function<void()> submitObserver;
  // Score requests whose final prompt step reports a non-finite logit.
  std::unordered_set<uint64_t> invalidScores;
  // The flags and sampling each request began with.
  std::unordered_map<uint64_t, uint32_t> beganFlags;
  std::unordered_map<uint64_t, SamplingParameters> beganSampling;
  // The prefill steps each request took.
  std::unordered_map<uint64_t, uint32_t> prefillChunks;
  // The tokens a decode step emits.
  uint32_t decodeTokens = 1;
  // The masks the model took.
  uint32_t providedMasks = 0;

  // Holds every command until the flag it returns is set.
  std::shared_ptr<std::atomic<bool>> holdCommands() {
    holdPrefillUntil = holdDecodeUntil = std::make_shared<std::atomic<bool>>(false);
    return holdDecodeUntil;
  }

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

} // namespace splash::test

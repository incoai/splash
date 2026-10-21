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
  // A state a prefill command captured: once the cache lets go of it, its
  // buffers return to the model's pool, as QwenStateStorage's do.
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

// The buffers of a state a prefill command captures (Executor::reserveState):
// the command seals them into the state, and dropped unsealed they return to
// the model's pool.
class Reservation final : public StateReservation {
public:
  explicit Reservation(std::shared_ptr<uint64_t> pool) : pool_(std::move(pool)) {}
  ~Reservation() override {
    if (pool_)
      *pool_ += State().bytes();
  }
  // The pool the sealed state's buffers return to.
  std::shared_ptr<uint64_t> seal() noexcept { return std::exchange(pool_, nullptr); }

private:
  std::shared_ptr<uint64_t> pool_;
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

// With a pool, it is a captured state whose buffers return to the model's
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
  bool canOffload(bool) const noexcept override { return true; }
  std::unique_ptr<StateOffload> offload(std::function<void()>) const override {
    return std::make_unique<OffloadTicket>(control_);
  }
private:
  std::shared_ptr<OffloadControl> control_;
  std::shared_ptr<uint64_t> pool_;
};

// A captured state whose copy a persistent tier writes while it stays in
// RAM (persist); the copy is a DiskState.
class KeptState final : public CompositeState {
public:
  explicit KeptState(std::shared_ptr<OffloadControl> control) : control_(std::move(control)) {}
  uint64_t bytes() const noexcept override { return 64; }
  bool canOffload(bool) const noexcept override { return true; }
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
  bool emitted = false;
  bool provided = false;
  bool abandoned = false;
  // Keeps the command running after its mask wait ended.
  bool held = false;
};

// A decode command that waits for the verify mask of its one constrained
// lane; `results` are its lanes', in order.
class MaskOverlapTicket final : public ModelBatchTicket {
public:
  MaskOverlapTicket(std::shared_ptr<MaskOverlapState> state, std::vector<ModelStepResult> results)
      : state_(std::move(state)), results_(std::move(results)) {}

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
    return std::move(results_);
  }
  double wallMilliseconds() const noexcept override { return 1.0; }

private:
  std::shared_ptr<MaskOverlapState> state_;
  std::vector<ModelStepResult> results_;
};

// A model that runs no kernels, for the engine, native loop and transport
// tests. It behaves as model::Runtime does where those tests rely on it: a
// request takes the first free one of maximumLanes lanes; a prefill step that
// commits returns the states it captures, in their reserved buffers; the
// prefill step that completes a prompt asks for a constrained request's first
// mask, or returns a score request's option logits; every decode cycle of a
// constrained lane after its first token waits for its verify mask inside its
// command (MaskOverlapTicket), beside the command's unconstrained lanes; a
// decode step throws for a lane whose output limit, lowered by a resume after
// its anchor was selected, leaves it only that anchor, which the runtime emits
// only as it selects it; and provideMask throws for a request that waits for
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
    outputLimits[request.id] = request.maxNewTokens;
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
    spilledReadFloors.erase(id);
    historyStaging.erase(id);
    entry.resident = false;
    entry.position = 0;
    ++suspensions;
    if (kvGrowthBlocked && unblockGrowthOnSuspend)
      *kvGrowthBlocked = false;
  }
  StateAdmission resume(const ModelRequest &request) override {
    ++resumeAttempts;
    lastRestoredTokens = request.restoredTokens;
    outputLimits[request.id] = request.maxNewTokens;
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
      if (nonFinitePrefills.contains(item.requestId)) {
        // The chunk commits nothing; the engine ends the request.
        result.push_back({.requestId = item.requestId,
                          .consumedPromptTokens = item.tokenCount,
                          .failure = "target prefill row is not finite"});
        continue;
      }
      state.position += item.tokenCount;
      prefillRows += item.tokenCount;
      ++prefillChunks[item.requestId];
      const uint64_t end = item.logicalPosition + item.tokenCount;
      std::vector<std::shared_ptr<const CompositeState>> captured;
      uint64_t after = item.logicalPosition;
      // The production runtime refuses a capture its draft plan leaves no
      // whole draft window for, and the plan has one for its boundaries.
      const std::vector<DraftBoundaryPlan> &planned = plans.at(item.requestId).boundaries;
      for (const StateCapture &capture : item.stateCaptures) {
        if (capture.tokens <= after || capture.tokens > end || !capture.reservation)
          throw std::logic_error("a state capture lies outside its step or out of order");
        if (std::none_of(planned.begin(), planned.end(), [&](const DraftBoundaryPlan &boundary) {
              return boundary.boundary == capture.tokens;
            }))
          throw std::logic_error("a state capture is not a boundary of the request's draft plan");
        after = capture.tokens;
        captures.push_back({item.requestId, capture.tokens, end});
        captured.push_back(seal(static_cast<Reservation &>(*capture.reservation)));
      }
      // The step that completes the prompt selects its first token; a replay
      // never does again. A constrained prompt's end waits for the first
      // mask instead, and a score request's returns its option logits.
      const bool last = !state.replaying &&
                        state.position == prompts.at(item.requestId).size();
      const bool awaitsMask = state.constrained && last;
      // The wait outlives a suspension, as the host continuation keeps it.
      state.awaitingMask = state.awaitingMask || awaitsMask;
      if (last && !awaitsMask)
        state.anchorLimit = outputLimits.at(item.requestId);
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
      // A step that fails commits nothing.
      if (step.failure.empty())
        step.capturedStates = std::move(captured);
      step.unreadableKvBlocks = takeUnreadable(item.requestId);
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
      Request &state = requests.at(item.requestId);
      const uint32_t limit = outputLimits.at(item.requestId);
      if (plan.decodeStage == DecodeStage::ApplyInitialMask) {
        // Selects the first token under its mask and drafts nothing.
        state.anchorLimit = limit;
        result.push_back({item.requestId, 0, {}, false, DecodeStage::Regular,
                          0, 0});
      } else {
        // The fake's step at the last budgeted token stands for the
        // runtime's emission of that anchor as it selects it, under the
        // same limit; an anchor selected under a higher limit was not
        // emitted, and the runtime decodes no lane left only that anchor.
        if (limit < state.anchorLimit && limit < state.generated + 2)
          throw std::logic_error("terminal anchor was not emitted on selection");
        const auto speculative = speculativeCounts.find(item.requestId);
        const uint32_t emitted = speculative == speculativeCounts.end()
            ? decodeTokens : speculative->second.second + 1;
        const std::vector<uint32_t> tokens =
            item.requestId == poisonRequest && poisonToken
                ? std::vector<uint32_t>{*poisonToken}
                : std::vector<uint32_t>(emitted, 42);
        ModelStepResult step{item.requestId, 0, tokens, decodeFinishes,
                             DecodeStage::Regular, 0, 0};
        step.outputTokensWithoutKv = decodeTokensWithoutKv;
        if (speculative != speculativeCounts.end()) {
          step.draftedTokens = speculative->second.first;
          step.acceptedDraftTokens = speculative->second.second;
        }
        step.unreadableKvBlocks = takeUnreadable(item.requestId);
        state.generated += static_cast<uint32_t>(tokens.size());
        state.anchorLimit = limit;
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
    // A prefill command streams the history of one lane at most, whose
    // pages hold only their index, and only whole, as the production
    // runtime refuses another.
    if (plan.kind == WorkKind::Prefill) {
      const auto streams = std::ranges::count_if(items, [](const ModelBatchItem &item) {
        return item.pageSources && item.pageSources->indexPages();
      });
      if (streams > 1)
        throw std::logic_error("a prefill command streams the history of one sequence");
      if (streams && plan.passes > 1)
        throw std::invalid_argument("a prefill command's passes are not ones the runtime runs");
    }
    // A lane reads the K/V only a copy holds through memory reserved for it
    // before its command, as the production runtime's readers require: a
    // decoding lane's floor of the hot block cache, a prefill's staging of
    // the pages its command reads.
    for (const ModelBatchItem &item : items) {
      if (!item.pageSources || !item.pageSources->indexPages())
        continue;
      const auto staged = historyStaging.find(item.requestId);
      const bool reserved =
          plan.kind == WorkKind::Decode
              ? spilledReadFloors.contains(item.requestId)
              : staged != historyStaging.end() &&
                    uint64_t{staged->second} * engine::KvCache::pageTokens >=
                        item.logicalPosition + item.tokenCount;
      if (!reserved)
        throw std::logic_error("a lane reads spilled K/V without memory reserved for its reads");
    }
    std::unique_ptr<ModelBatchTicket> ticket;
    const auto constrained = [&](const ModelBatchItem &item) { return requests.at(item.requestId).constrained; };
    if (plan.kind == WorkKind::Decode && plan.decodeStage == DecodeStage::Regular &&
        std::ranges::any_of(items, constrained)) {
      // A cycle with a constrained lane after its first token waits for that
      // lane's mask inside the ticket, as the production constrained ticket
      // does, and its other lanes decode as they do in a command of their
      // own.
      if (std::ranges::count_if(items, constrained) > 1)
        throw std::logic_error("the test model takes the verify mask of one lane per command");
      overlap = std::make_shared<MaskOverlapState>();
      std::vector<ModelStepResult> results;
      for (const ModelBatchItem &item : items) {
        if (constrained(item)) {
          overlap->requestId = item.requestId;
          results.push_back({item.requestId, 0, {42}, decodeFinishes, DecodeStage::Regular, 7, 0});
        } else {
          results.push_back(std::move(decode(plan, std::span(&item, 1)).front()));
        }
      }
      ticket = std::make_unique<MaskOverlapTicket>(overlap, std::move(results));
    } else if (plan.kind == WorkKind::Decode && holdDecodeUntil) {
      heldCompletion = std::move(completion);
      ticket = std::make_unique<HeldTicket>(decode(plan, items), holdDecodeUntil,
                                            1.0);
    } else if (plan.kind == WorkKind::Prefill && holdPrefillUntil) {
      heldCompletion = std::move(completion);
      ticket = std::make_unique<HeldTicket>(prefill(plan, items),
                                            holdPrefillUntil, 1.0);
    } else if (plan.kind == WorkKind::Prefill && plan.pass + 1 < plan.passes) {
      // A pass before a command's last commits nothing, as the production
      // runtime's returns.
      std::vector<ModelStepResult> results(items.size());
      for (size_t lane = 0; lane < items.size(); ++lane) results[lane].requestId = items[lane].requestId;
      const CommandTimes times = commandTimes ? commandTimes(plan, items) : CommandTimes{};
      ticket = immediateTicket(std::move(results), completion, times);
    } else {
      const CommandTimes times = commandTimes ? commandTimes(plan, items) : CommandTimes{};
      ticket = immediateTicket(plan.kind == WorkKind::Prefill ? prefill(plan, items)
                                                              : decode(plan, items),
                               completion, times);
    }
    if (submitObserver)
      submitObserver();
    return ticket;
  }
  uint32_t prefillGroupRows(bool spills) const noexcept override {
    return spills || !groupsOnlySpilling ? groupRows : model::ExecutionLimits::prefillTokenBudget;
  }
  uint32_t prefillPasses() const noexcept override { return passes; }
  uint64_t snapshotBytes() const noexcept override { return stateBytes; }
  // The production model takes a cached state's buffers from its pool, or
  // with one admission, for a resident lane's page-aligned boundary, and
  // returns nullptr with the cause when the governor refuses them. The fake
  // refuses the next `deniedReservations` calls, every call for the
  // boundary at `denyReservationAt`, and every call while `reservationRoom`
  // reports no memory, for `reservationFailure`.
  std::shared_ptr<StateReservation> reserveState(uint64_t id, uint32_t tokens,
                                                 metal::AllocationFailure &refusal) override {
    if (!requests.at(id).resident || !tokens || tokens % engine::KvCache::pageTokens)
      throw std::logic_error("a state was reserved off a resident lane's page boundary");
    ++reservationAttempts;
    if (reservationObserver)
      reservationObserver();
    if (deniedReservations) {
      --deniedReservations;
      refusal = reservationFailure;
      return nullptr;
    }
    if (denyReservationAt == tokens || (reservationRoom && !reservationRoom())) {
      refusal = reservationFailure;
      return nullptr;
    }
    refusal = metal::AllocationFailure::None;
    ++reservations;
    return std::make_shared<Reservation>(evictedStateBytes);
  }
  // The production model copies the lane's state at its position into
  // buffers it takes as reserveState() does; so does the fake, refusing as
  // reserveState() refuses.
  std::shared_ptr<const CompositeState> snapshot(uint64_t id) override {
    metal::AllocationFailure refusal;
    const std::shared_ptr<StateReservation> buffers =
        reserveState(id, requests.at(id).position, refusal);
    if (!buffers)
      return nullptr;
    ++snapshots;
    return seal(static_cast<Reservation &>(*buffers));
  }
  // Without a cache slot the production model writes the lane's state to
  // the disk tier; the fake has one when `stateTier` is set, with quota for
  // every state.
  bool canSnapshotToDisk(bool) const noexcept override { return stateTier != nullptr; }
  std::unique_ptr<StateOffload> snapshotToDisk(uint64_t, std::function<void()>) override {
    ++diskSnapshots;
    return std::make_unique<OffloadTicket>(stateTier);
  }
  uint32_t statesToActivate() const noexcept override {
    return statesLacked ? statesLacked() : 0;
  }
  // The production model reserves a decoding lane's floor of the hot block
  // cache, which the fake refuses for the requests in refusedSpilledReads,
  // and while spilledReadsRoom reports no memory for the request's, with
  // spilledReadsRefusal's cause. The lane holds it until it is suspended or
  // ends.
  model::ModelMemoryAdmission reserveSpilledReads(uint64_t id) override {
    if (refusedSpilledReads.contains(id) || (spilledReadsRoom && !spilledReadsRoom(id)))
      return {.allocationFailure = spilledReadsRefusal};
    spilledReadFloors.insert(id);
    return {.ok = true};
  }
  // It stages the history a prefill command streams, as many pages as the
  // command reads, which the fake refuses for the requests in
  // refusedStaging.
  model::ModelMemoryAdmission reserveHistoryStaging(uint64_t id, uint32_t pages) override {
    if (refusedStaging.contains(id))
      return {.allocationFailure = metal::AllocationFailure::EngineBudget};
    uint32_t &staged = historyStaging[id];
    staged = std::max(staged, pages);
    stagings.push_back(id);
    return {.ok = true};
  }
  // It grows that cache toward its planned size, which the fake refuses
  // with refusedGrowth.
  model::ModelMemoryAdmission growSpilledReads() override {
    ++growthAttempts;
    if (refusedGrowth)
      return {.allocationFailure = *refusedGrowth};
    return {.ok = true};
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
    spilledReadFloors.erase(id);
    historyStaging.erase(id);
  }

  // What a page holds once prefill has written a block's first row to it.
  static uint64_t rows(uint32_t block, uint32_t firstToken) {
    return (uint64_t{block} << 32) | firstToken;
  }
  // With `kv`, a step names pages of allocated extents only and finds every
  // prompt block before `position` through the page table the engine handed
  // it.
  void requireRows(const ModelBatchItem &item, uint64_t position) const {
    if (!kv)
      return;
    for (uint32_t page : item.pageTable)
      if (!kv->allocated(page / kv->extentPages()))
        throw std::logic_error("a page table names a page of a released extent");
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
    // The tokens its decode steps emitted, and the output limit under which
    // its anchor was selected: by the step that completed its prompt, its
    // first mask's or its latest decode step.
    uint32_t generated = 0;
    uint32_t anchorLimit = 0;
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
  // State reservations granted, and asked for.
  uint32_t reservations = 0;
  uint32_t reservationAttempts = 0;
  // Each state a prefill step captured: its request, its boundary and the
  // step's end, past the boundary for one inside the step.
  struct Capture {
    uint64_t requestId = 0;
    uint32_t tokens = 0;
    uint64_t end = 0;
  };
  std::vector<Capture> captures;
  // States taken from a lane after its command (snapshot).
  uint32_t snapshots = 0;
  uint32_t diskSnapshots = 0;
  std::shared_ptr<OffloadControl> stateTier;
  // With it, captured states are KeptStates whose copies land when it says
  // so.
  std::shared_ptr<OffloadControl> keptTier;
  // What a cached state's buffers take; State::bytes() unless a test needs a
  // state larger than an extent.
  uint64_t stateBytes = 64;
  uint32_t deniedReservations = 0;
  std::optional<uint32_t> denyReservationAt;
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
  std::function<void()> reservationObserver;
  std::function<bool()> reservationRoom;
  metal::AllocationFailure reservationFailure = metal::AllocationFailure::EngineBudget;
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
  // The buffers of states the cache has let go of, and of reservations no
  // step sealed, until reclaimIdleState takes them into the pool.
  std::shared_ptr<uint64_t> evictedStateBytes = std::make_shared<uint64_t>(0);
  // The cached states whose buffers the pool lacks for one activation.
  std::function<uint32_t()> statesLacked;
  uint64_t reclaimedIdleStateBytes = 0;
  // Caches the model can rebuild, one released per reclaim step that may
  // take them once no buffer is idle; cacheReclaims counts those steps.
  std::vector<uint64_t> cacheUnits;
  uint32_t cacheReclaims = 0;
  // Rows each state captured from then on holds (RowsHoldingState); the
  // executor must outlive the cache that keeps those states.
  uint64_t stateHeldRows = 0;
  bool keptLane = false;
  bool *kvGrowthBlocked = nullptr;
  bool unblockGrowthOnSuspend = true;
  bool decodeFinishes = true;
  std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> speculativeCounts;
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
  // The times each command that completes at once reports, from its plan
  // and items; none without it.
  std::function<CommandTimes(const BatchPlan &, std::span<const ModelBatchItem>)> commandTimes;
  // Score requests whose final prompt step reports a non-finite logit.
  std::unordered_set<uint64_t> invalidScores;
  // Requests whose prefill steps report a row that is not finite.
  std::unordered_set<uint64_t> nonFinitePrefills;
  // The flags and sampling each request began with, and the output limit
  // (ModelRequest::maxNewTokens) its latest begin() or resume() was given.
  std::unordered_map<uint64_t, uint32_t> beganFlags;
  std::unordered_map<uint64_t, SamplingParameters> beganSampling;
  std::unordered_map<uint64_t, uint32_t> outputLimits;
  // The prefill steps each request took.
  std::unordered_map<uint64_t, uint32_t> prefillChunks;
  // The most rows of a prefill command (Model::prefillGroupRows): a chunk's,
  // or a prefill group's, for every lane, as where experts stream, or with
  // groupsOnlySpilling only for a lane whose K/V spills.
  uint32_t groupRows = model::ExecutionLimits::prefillTokenBudget;
  bool groupsOnlySpilling = false;
  // The most passes over its layers a prefill command runs in
  // (Model::prefillPasses).
  uint32_t passes = 1;
  // The tokens a decode step emits.
  uint32_t decodeTokens = 1;
  // The masks the model took.
  uint32_t providedMasks = 0;
  // Requests whose floor of the hot block cache is refused
  // (reserveSpilledReads), and those whose history's staging is
  // (reserveHistoryStaging).
  std::unordered_set<uint64_t> refusedSpilledReads;
  std::function<bool(uint64_t)> spilledReadsRoom;
  metal::AllocationFailure spilledReadsRefusal = metal::AllocationFailure::EngineBudget;
  std::unordered_set<uint64_t> refusedStaging;
  // The requests that hold their floor of the hot block cache, and the
  // pages of history staged for each; the requests the staging was reserved
  // for, in order.
  std::unordered_set<uint64_t> spilledReadFloors;
  std::unordered_map<uint64_t, uint32_t> historyStaging;
  std::vector<uint64_t> stagings;
  std::optional<metal::AllocationFailure> refusedGrowth;
  uint32_t growthAttempts = 0;
  // Blocks of KV the SSD cache holds that each request's next step reports
  // it could not read (ModelStepResult::unreadableKvBlocks).
  std::unordered_map<uint64_t, std::vector<uint64_t>> unreadableBlocks;

  // Holds every command until the flag it returns is set.
  std::shared_ptr<std::atomic<bool>> holdCommands() {
    holdPrefillUntil = holdDecodeUntil = std::make_shared<std::atomic<bool>>(false);
    return holdDecodeUntil;
  }

private:
  // The blocks the request's step reports it could not read, once.
  std::vector<uint64_t> takeUnreadable(uint64_t id) {
    const auto found = unreadableBlocks.find(id);
    if (found == unreadableBlocks.end())
      return {};
    std::vector<uint64_t> blocks = std::move(found->second);
    unreadableBlocks.erase(found);
    return blocks;
  }
  // The state the step that captured it keeps in the reserved buffers.
  std::shared_ptr<const CompositeState> seal(Reservation &reservation) {
    std::shared_ptr<uint64_t> pool = reservation.seal();
    if (keptTier)
      return std::make_shared<KeptState>(keptTier);
    if (stateHeldRows)
      return std::make_shared<RowsHoldingState>(cacheUnits, stateHeldRows);
    return std::make_shared<State>(std::move(pool));
  }
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

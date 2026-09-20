#pragma once

#include "ops/Vision.hpp"
#include "engine/Cache.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/Scheduler.hpp"
#include "engine/Types.hpp"
#include "ops/PagedKv.hpp"

#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>

namespace splash::engine {

struct EngineConfig final {
  uint32_t maxContext = kv::kMaximumLogicalTokens;
  uint32_t vocabularySize = std::numeric_limits<uint32_t>::max();
  // Two draft windows balance recovery granularity and capture work.
  // Zero disables progress checkpoints without changing reusable end states.
  uint32_t prefillCheckpointTokens =
      2 * model::ExecutionLimits::draftContextTokens;
  // Patches per image the model's vision scratch covers.
  uint32_t maxImagePatches = ops::kMaximumImagePatches;
  double resourceWaitTimeoutMilliseconds = 30000.0;
  // Host growth admission, supplied by the runtime governor. Queried only on
  // failed allocation, never on the successful decode path.
  std::function<bool()> growthPaused;
};

struct ResourceWaitSnapshot final {
  uint32_t memory = 0;
  uint32_t concurrency = 0;
  uint32_t suspended = 0;
  double oldestWaitMilliseconds = 0.0;
  bool draining = false;
};

struct EngineSnapshot final {
  SchedulerSnapshot scheduler;
  CacheSnapshot resources;
  uint32_t maximumContextTokens = 0;
  uint64_t submitted = 0;
  uint64_t completed = 0;
  uint64_t cancelled = 0;
  uint64_t failed = 0;
  uint64_t cacheHits = 0;
  uint64_t coldMisses = 0;
  uint64_t reusedTokens = 0;
  uint64_t replayStatePublications = 0;
  uint64_t deduplicatedStatePublications = 0;
  // Publications completed after reclaiming a cached state.
  uint64_t recycledStatePublications = 0;
  uint64_t replayStatePublicationFailures = 0;
  uint64_t junctionMaterializations = 0;
  uint64_t junctionMaterializationFailures = 0;
  uint64_t checkpointPublications = 0;
  uint64_t checkpointPublicationFailures = 0;
  uint64_t resourceSuspensions = 0;
  uint64_t resourceResumptions = 0;
  // All prefill rows after preemption, including an unfinished prompt suffix.
  uint64_t resourceReplayTokens = 0;
};

// KV blocks define prefix identity; composite recurrent state is attached
// at sparse progress points, replay boundaries, and shared KV junctions.
class Engine final {
public:
  Engine(EngineConfig config, Cache &cache, model::Model &model,
         EngineEventSink &events);

  void submit(EngineRequest request);
  void cancel(uint64_t requestId);
  void failRequest(uint64_t requestId, std::string code, std::string message);
  void provideMask(uint64_t requestId, std::span<const uint32_t> words);
  void setCompletionNotifier(std::function<void()> notifier);

  [[nodiscard]] bool tick(double nowMilliseconds);
  [[nodiscard]] bool idle() const noexcept;
  [[nodiscard]] bool commandInFlight() const noexcept {
    return pending_.has_value();
  }
  [[nodiscard]] std::optional<double> nextWakeupMilliseconds() const;
  [[nodiscard]] EngineSnapshot snapshot() const;
  [[nodiscard]] ResourceWaitSnapshot resourceWaitSnapshot(double nowMilliseconds) const;

  // Runs only at a command-completion safe point. Reclaim order follows
  // ownership and preserves reusable prefixes for as long as possible: idle
  // model state, unused KV backing, disposable checkpoints, then ordinary
  // state/KV in LRU order.
  // Live command buffers are never eviction candidates. Physical KV release
  // is paced one extent at a time; reclaimDeferred() reports that the pass
  // stopped behind an in-flight release and should run again shortly.
  [[nodiscard]] uint64_t reclaimMemory(const MemoryReclaimDirective &directive);
  [[nodiscard]] bool reclaimDeferred() const noexcept {
    return cache_.releaseDeferred();
  }

private:
  struct Failure final {
    std::string code;
    std::string message;
    bool retryable = false;
  };

  struct ResourceWait final {
    StateFailure reason = StateFailure::None;
    std::optional<double> startedMilliseconds;
    double deadlineMilliseconds = 0.0;
    double retryMilliseconds = 0.0;
    uint64_t epoch = 0;
  };

  struct Request final {
    struct StateBoundary final {
      enum class Purpose : uint8_t { Checkpoint, Replay, Junction };
      uint32_t tokens = 0;
      Purpose purpose = Purpose::Replay;
    };

    EngineRequest request;
    std::optional<uint32_t> stateCell;
    bool suspended = false;
    uint32_t promptTokens = 0;
    uint32_t reportedPromptTokens = 0;
    uint32_t replayTokens = 0;
    // A failed dispatch must fit before replay can consume any model work.
    uint64_t resumeKvTargetTokens = 0;
    ResourceWait resourceWait;
    std::vector<uint32_t> exactTokens;
    std::vector<StateBoundary> stateBoundaries;
    size_t stateBoundaryCursor = 0;
    StateCheckpoint latestCheckpoint;
    // The scheduler owns the terminal phase; this flag records that the
    // corresponding event was emitted and model/resource ownership ended.
    bool finalized = false;
    std::optional<Failure> failure;
    bool replaying = false;
  };

  struct Pending final {
    BatchPlan plan;
    std::unique_ptr<ModelBatchTicket> ticket;
  };
  double nextHealthCheckMilliseconds_ = 0.0;

  [[nodiscard]] Request &request(uint64_t requestId);
  [[nodiscard]] bool admitQueued(double nowMilliseconds);
  [[nodiscard]] bool admit(Request &request, double nowMilliseconds);
  [[nodiscard]] static uint32_t sharedPrefillBoundary(const Request &left,
                                                      const Request &right);
  [[nodiscard]] bool pendingSharedPrefill(const Request &request,
                                          uint32_t resumeBoundary) const;
  [[nodiscard]] DraftContextPlan
  configureDraftStatePlan(Request &request, uint32_t stateBoundary,
                          uint32_t junctionBoundary);
  [[nodiscard]] bool addSharedPrefillBoundaries(Request &request, uint32_t after);
  [[nodiscard]] DraftContextPlan
  pendingDraftStatePlan(const Request &request, uint32_t stateBoundary) const;
  void armNextStateBoundary(Request &request);
  void discardPendingStateBoundaries(Request &request) noexcept;
  [[nodiscard]] bool retireCheckpoint(Request &request);
  void publishReachedStateBoundaries(Request &request,
                                     uint32_t promptProcessed);
  [[nodiscard]] bool prepare(BatchPlan &plan,
                             std::vector<ModelBatchItem> &items,
                             double nowMilliseconds);
  [[nodiscard]] bool reclaimForGrowth(
      CacheReclaimMode mode = CacheReclaimMode::ReleaseBacking);
  [[nodiscard]] bool reclaimIdleState() noexcept;
  [[nodiscard]] bool reuseIdleBackingWhilePaused(const TokenAdmission &admission);
  [[nodiscard]] bool growthPaused() const;
  struct KvAdmission {
    TokenAdmission allocation;
    bool retryableBudget = false;
  };
  [[nodiscard]] KvAdmission admitKv(Request &request, uint64_t workEnd);
  [[nodiscard]] bool budgetMayRecover(metal::AllocationFailure failure,
                                      uint64_t generation, bool reclaimed) const;
  void suspendForGrowth(Request &request, uint64_t workEnd,
                        double nowMilliseconds);
  [[nodiscard]] bool resourceRetryReady(const Request &request,
                                        double nowMilliseconds) const noexcept;
  void deferResourceRetry(Request &request, double nowMilliseconds,
                          StateFailure reason = StateFailure::MemoryPressure) noexcept;
  void signalResourceProgress() noexcept;
  void apply(const BatchPlan &plan, std::span<const ModelStepResult> results,
             double wallMilliseconds, bool representativePrefillTiming);
  void finish(Request &request, EngineFinishReason reason);
  void finishFailure(Request &request, Failure failure);
  void finishCapacity(Request &request, const TokenAdmission &admission);
  void release(Request &request);
  void sweepTerminal();

  EngineConfig config_;
  Cache &cache_;
  model::Model &model_;
  EngineEventSink &events_;
  Scheduler scheduler_;
  std::unordered_map<uint64_t, Request> requests_;
  // Pressure preempted work and a resident lane still holds its state cell.
  [[nodiscard]] bool drainingForRecovery() const;
  std::function<void()> completionNotifier_;
  std::optional<Pending> pending_;
  uint64_t resourceEpoch_ = 1;
  bool recoveringResources_ = false;
  EngineSnapshot counters_;
};

} // namespace splash::engine

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
  // Patches per image the model's vision scratch covers; zero rejects images.
  uint32_t maxImagePatches = ops::kMaximumImagePatches;
  double resourceWaitTimeoutMilliseconds = 30000.0;
  // Host growth admission, supplied by the runtime governor. Queried only on
  // failed allocation and, after a suspension the pause caused, while
  // resident lanes drain; never on the ordinary decode path.
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
  // Publications written straight to disk because no cache slot was free.
  uint64_t diskStatePublications = 0;
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
  void observePrefill(uint32_t rows, double wallMilliseconds) {
    scheduler_.observePrefill(rows, wallMilliseconds);
  }
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
    // Memory is on its way back; the limit fires only without progress.
    bool pending = false;
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
    std::optional<CacheProbe> admissionProbe;
    std::vector<StateBoundary> stateBoundaries;
    size_t stateBoundaryCursor = 0;
    StateCheckpoint latestCheckpoint;
    // The scheduler owns the terminal phase; this flag records that the
    // corresponding event was emitted and model/resource ownership ended.
    bool finalized = false;
    std::optional<Failure> failure;
    bool replaying = false;
    // Captured once the final prompt chunk completes; emitted with Done.
    std::vector<float> scoreLogits;
    bool skipCache = false;
    // Admission that waits for its state's read, its KV pages' restores,
    // or both, before the lane runs.
    struct Restore {
      CacheLookup lookup;
      DraftContextPlan draft;
      // Null when the state was in RAM.
      std::unique_ptr<StateRestore> ticket;
    };
    std::optional<Restore> restore;
  };

  // An empty plan carries only KV copies for the disk tier.
  struct Pending final {
    BatchPlan plan;
    std::unique_ptr<ModelBatchTicket> ticket;
  };
  enum class Prepared : uint8_t {
    // Some lanes were admitted and the plan runs with them.
    Runnable,
    // Every lane was denied and one yielded its memory or failed.
    Yielded,
    // Every lane was denied while pages are on their way back; nothing
    // changed, the lanes retry when the pages land.
    Waiting,
  };
  double nextHealthCheckMilliseconds_ = 0.0;

  [[nodiscard]] Request &request(uint64_t requestId);
  [[nodiscard]] bool admitQueued(double nowMilliseconds);
  [[nodiscard]] bool admit(Request &request, double nowMilliseconds);
  [[nodiscard]] static uint32_t sharedPrefillBoundary(const Request &left,
                                                      const Request &right);
  [[nodiscard]] bool pendingSharedPrefill(const Request &request,
                                          uint32_t resumeBoundary) const;
  void completeAdmission(Request &request, CacheLookup &lookup, DraftContextPlan draft);
  [[nodiscard]] bool pollRestores(double nowMilliseconds);
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
  [[nodiscard]] Prepared prepare(BatchPlan &plan,
                                 std::vector<ModelBatchItem> &items,
                                 double nowMilliseconds);
  [[nodiscard]] CacheReclaimResult reclaimForGrowth(
      CacheReclaimMode mode = CacheReclaimMode::ReleaseBacking);
  [[nodiscard]] bool reclaimIdleState() noexcept;
  [[nodiscard]] CacheReclaimResult reuseIdleBackingWhilePaused(
      const TokenAdmission &admission);
  [[nodiscard]] bool growthPaused() const;
  // Memory a lane could not get, and what the engine knows about its return.
  struct Denial {
    metal::AllocationFailure allocationFailure = metal::AllocationFailure::None;
    // On its way back: pages whose copies are being written, or a reclaim
    // that waits for the transfer in flight. The lane waits; nobody yields.
    bool pending = false;
    // Still moving: a release or a reclaim in progress, a budget that can
    // recover, or the above. Waiting or yielding beats failing.
    bool retryable = false;
  };
  struct KvAdmission {
    TokenAdmission allocation;
    Denial denial;
  };
  // What a lane does about memory it could not get. Pending memory returns
  // by itself: the lane waits. Otherwise a lane fails only when it is alone
  // with nothing left to reclaim; while other lanes hold memory, growth is
  // paused or the budget may recover, a running lane yields its memory and
  // a lane being admitted waits.
  enum class Verdict : uint8_t { Wait, Yield, Fail };
  [[nodiscard]] Verdict judge(const Denial &denial, uint64_t requestId) const;
  [[nodiscard]] bool anotherResident(uint64_t requestId) const;
  // Runs one page admission, reclaiming cache between attempts while that
  // makes progress.
  [[nodiscard]] KvAdmission admitKv(const std::function<TokenAdmission()> &attempt);
  [[nodiscard]] bool budgetMayRecover(metal::AllocationFailure failure,
                                      uint64_t generation, bool reclaimed) const;
  void suspendForGrowth(Request &request, uint64_t workEnd,
                        metal::AllocationFailure failure,
                        double nowMilliseconds);
  [[nodiscard]] bool resourceRetryReady(const Request &request,
                                        double nowMilliseconds) const noexcept;
  // With `pending`, memory is on its way back (pages of demoted blocks land
  // within commands): the wait limit measures time without progress, so it
  // moves out with every retry that follows progress and never fires while
  // progress has been made since the last attempt.
  void deferResourceRetry(Request &request, double nowMilliseconds,
                          StateFailure reason = StateFailure::MemoryPressure,
                          bool pending = false) noexcept;
  // The wait limit tick() enforces, or zero while it enforces none: a
  // pending wait that has seen progress waits for its next attempt.
  [[nodiscard]] double resourceDeadline(const Request &request) const noexcept;
  void signalResourceProgress() noexcept;
  void apply(const BatchPlan &plan, std::span<const ModelStepResult> results,
             double wallMilliseconds, bool representativePrefillTiming);
  void finish(Request &request, EngineFinishReason reason,
              std::span<const float> optionLogits);
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
  // Pressure preempted work, a resident lane still holds its state cell, and
  // memory is still short (growth is paused or allocationFailed_), up to the
  // drain's end.
  [[nodiscard]] bool drainingForRecovery() const;
  std::function<void()> completionNotifier_;
  std::optional<Pending> pending_;
  uint64_t resourceEpoch_ = 1;
  // The resource wait limit after the latest suspension; zero once passed
  // or when no request is suspended.
  double drainEndMilliseconds_ = 0.0;
  // An allocation failed since the latest suspension, or the suspension
  // itself met a limit that only freed memory lifts, unlike a host pause.
  bool allocationFailed_ = false;
  EngineSnapshot counters_;
};

} // namespace splash::engine

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
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace splash::engine {

// Prefill checkpoints sit at multiples of this many tokens
// (plannedCheckpoints): two draft windows balance recovery granularity and
// capture work. Tests substitute another interval through TestConfig.
inline constexpr uint32_t kPrefillCheckpointTokens =
    2 * model::ExecutionLimits::draftContextTokens;
static_assert(kPrefillCheckpointTokens >= model::ExecutionLimits::draftContextTokens &&
                  kPrefillCheckpointTokens % KvCache::pageTokens == 0,
              "a prefill checkpoint interval spans a draft window and whole KV pages");
// How long a request waits for memory before it fails, and a resident
// drain lasts. Tests substitute another bound through TestConfig.
inline constexpr double kResourceWaitTimeoutMilliseconds = 30000.0;

struct EngineConfig final {
  uint32_t maxContext = kv::kMaximumLogicalTokens;
  uint32_t vocabularySize = std::numeric_limits<uint32_t>::max();
  // Patches per image the server's --max-image-pixels allows; zero rejects
  // images.
  uint32_t maxImagePatches = ops::kMaximumImagePatches;
  // Decode time owed for each unit of time a prefill runs while requests of
  // equal or higher priority decode. At 0.5 a stream keeps about a third of
  // its solo rate through a long prefill, which meanwhile takes 1.5x as long;
  // zero alternates one command of each kind.
  double decodeShare = 0.5;
  // Host growth admission, supplied by the runtime governor. Queried only
  // while resident lanes drain after a suspension; allocation reads the
  // cause of a refusal instead.
  std::function<bool()> growthPaused;
  // Marks the allocations that follow as memory a request in service needs,
  // which the pause does not hold back, and clears the mark
  // (MemoryGovernor::setServing).
  std::function<void(bool)> serving;
};

// The progress checkpoints a request plans between the point it resumes from
// and its replay boundary: the multiples of `interval` at least one prefill
// chunk past the one and before the other, none when `interval` is zero.
// They sit at multiples of the interval, so requests over the same prompt
// share them. A checkpoint within one prefill chunk of either end would cost
// a command split and a snapshot for less than a chunk of recompute.
[[nodiscard]] inline std::vector<uint32_t>
plannedCheckpoints(uint32_t resumeBoundary, uint32_t replayBoundary, uint32_t interval) {
  std::vector<uint32_t> checkpoints;
  if (!interval)
    return checkpoints;
  constexpr uint32_t chunk = model::ExecutionLimits::prefillTokenBudget;
  for (uint64_t boundary =
           (uint64_t{resumeBoundary} + chunk + interval - 1) / interval * interval;
       boundary + chunk <= replayBoundary; boundary += interval)
    checkpoints.push_back(static_cast<uint32_t>(boundary));
  return checkpoints;
}

struct ResourceWaitSnapshot final {
  uint32_t memory = 0;
  uint32_t concurrency = 0;
  // Requests that admission holds back behind the first one refused memory,
  // whatever they wait for themselves, and that request itself while a pass
  // keeps it out of its memory wait. During recovery only suspended requests
  // and those of a strictly higher priority are admitted, and only they
  // count.
  uint32_t heldBehindRefusal = 0;
  // Requests admitted into a restore of their prefix from disk, waiting for
  // its reads rather than for memory.
  uint32_t restoring = 0;
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
  // Resident lanes suspended for a request of a strictly higher priority.
  uint64_t prioritySuspensions = 0;
  uint64_t resourceResumptions = 0;
  // All prefill rows after preemption, including an unfinished prompt suffix.
  uint64_t resourceReplayTokens = 0;
};

// KV blocks define prefix identity; composite recurrent state is attached
// at sparse progress points, replay boundaries, and shared KV junctions.
//
// Failure contract. A request the engine cannot serve ends with a LaneOutcome
// (Types.hpp): its own result, which the engine reports and survives. submit()
// reports an invalid request with std::invalid_argument, which the caller
// answers with that request's error. Any other exception out of submit(), and
// every exception out of tick(), cancel(), failRequest(), provideMask() and
// reclaimMemory(), is engine-fatal: NativeRuntime reports EngineUnhealthy and
// the process exits. Code below them therefore does not roll back on an
// exception; the only cleanup on that path is RAII teardown itself needs
// (state IO drains, FileRestore, Serving).
class Engine final {
public:
  Engine(EngineConfig config, Cache &cache, model::Model &model,
         EngineEventSink &events);

  // The request passed protocol::validateRequest; submit checks only what
  // the engine knows: vocabulary, context window, and vision with the
  // server's per-image patch cap.
  void submit(EngineRequest request);
  void observePrefill(uint32_t rows, double wallMilliseconds) {
    scheduler_.observePrefill(rows, wallMilliseconds);
  }
  void cancel(uint64_t requestId);
  void failRequest(uint64_t requestId, LaneOutcome outcome, std::string message);
  void provideMask(uint64_t requestId, std::span<const uint32_t> words);
  void setCompletionNotifier(std::function<void()> notifier);

  [[nodiscard]] bool tick(double nowMilliseconds);
  [[nodiscard]] bool commandInFlight() const noexcept {
    return pending_.has_value();
  }
  [[nodiscard]] std::optional<double> nextWakeupMilliseconds() const;
  [[nodiscard]] EngineSnapshot snapshot() const;
  [[nodiscard]] ResourceWaitSnapshot resourceWaitSnapshot(double nowMilliseconds) const;

  // Runs only between commands: throws std::logic_error while a command is
  // in flight. Returns the model's idle state buffers first, then the caches
  // the model can rebuild while the directive's byte target is unmet; a
  // critical directive has no target and takes them all. Under critical
  // pressure it then evicts every unpinned cache entry (Cache::evictAll()).
  // Otherwise it releases empty KV extents and reclaims the cache one
  // Cache::reclaimOne step at a time, in the order that step's contract
  // (Cache.hpp) gives, returning the buffers an evicted state handed back to
  // the model's pool after each step. While the target is still unmet, it
  // then takes the image rows only evicted states held. A pass counts only
  // memory that leaves the engine: released KV extents, the caches and idle
  // buffers the model returns; evicting a state frees nothing by itself.
  // Pages whose copies are being written count toward the target. A pass
  // short of critical keeps one lane's pooled buffers and one empty extent.
  // Requests waiting for memory retry after any step that freed some, kept
  // or released.
  // Live command buffers are never eviction candidates. A pass first collects
  // the transfers that landed, so one that continues a reclaim they held back
  // takes what they freed. The result says whether the directive's target is
  // met, waits for transfers in flight, or finds nothing left to reclaim.
  [[nodiscard]] MemoryReclaimResult reclaimMemory(const MemoryReclaimDirective &directive);

private:
  struct LaneEnd final {
    LaneOutcome outcome;
    std::string message;
  };

  struct ResourceWait final {
    StateFailure reason = StateFailure::None;
    // What refused the memory at the latest attempt.
    metal::AllocationFailure allocationFailure = metal::AllocationFailure::None;
    std::optional<double> startedMilliseconds;
    // The admissions counted when the request was first refused memory, and
    // again when it is suspended: the lanes admitted up to then hold memory
    // it waits for. A wait for a lane does not record it.
    std::optional<uint64_t> admittedBefore;
    double deadlineMilliseconds = 0.0;
    // The latest tick at which a lane submitted before the request, or one
    // that admittedBefore counts, had work in flight; the limit restarts from
    // it.
    double earlierLaneWorkMilliseconds = 0.0;
    double retryMilliseconds = 0.0;
    uint64_t epoch = 0;
    // Memory is on its way back; the limit fires only without progress.
    bool pending = false;
  };

  struct Request final {
    struct StateBoundary final {
      uint32_t tokens = 0;
      // A rolling checkpoint, the lane's own progress, retired when the next
      // one lands; otherwise a state a later request resumes from.
      bool disposable = false;
    };

    EngineRequest request;
    // The admissions counted when admit() last gave it a lane (admissions_).
    // Lanes admitted before a request was refused memory or suspended restart
    // its wait's limit.
    uint64_t admission = 0;
    std::optional<uint32_t> lane;
    uint32_t promptTokens = 0;
    uint32_t reportedPromptTokens = 0;
    uint32_t replayTokens = 0;
    // A failed dispatch must fit before replay can consume any model work.
    uint64_t resumeKvTargetTokens = 0;
    ResourceWait resourceWait;
    // The latest attempt to start it was refused memory. Until it starts,
    // nothing that comes after it in admission order is admitted, in
    // ordinary admission or among suspended requests during recovery
    // (admitQueued); a pass that does not schedule it or a prefix wait
    // leaves it in place.
    bool refusedMemory = false;
    // The prompt from submit on, then the committed output; request.prompt
    // is empty.
    std::vector<uint32_t> exactTokens;
    // Made from exactTokens and request.images, which do not change while
    // the request waits; refreshed each pass and dropped once it starts or
    // skips the cache.
    std::optional<CacheProbe> admissionProbe;
    std::vector<StateBoundary> stateBoundaries;
    size_t stateBoundaryCursor = 0;
    StateCheckpoint latestCheckpoint;
    // The block of the prompt's replay boundary, where the conversation's
    // next turn resumes: its state is in use from the moment the request
    // reaches, reuses or restores it until the request ends, suspended or not.
    StateUse replayPoint;
    // The scheduler owns the terminal phase; this flag records that the
    // corresponding event was emitted and model/resource ownership ended.
    bool finalized = false;
    // The outcome that ends this lane once the command or restore it is in
    // drains; the first one stands.
    std::optional<LaneEnd> pendingEnd;
    // When the engine asked the server for this lane's token mask, until the
    // model takes one.
    std::optional<double> maskRequestedMilliseconds;
    bool replaying = false;
    // Captured once the final prompt chunk completes; emitted with Done.
    std::vector<float> scoreLogits;
    // A restore that could not fit alone released its prefix pin:
    // admissions ignore the cache until one succeeds.
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

  struct Pending final {
    BatchPlan plan;
    std::unique_ptr<ModelBatchTicket> ticket;
    // Where the engine's time for this command starts: the previous
    // command's retirement when the engine stayed busy, else its plan.
    double startedMilliseconds = 0.0;
  };
  enum class Prepared : uint8_t {
    // Some lanes were admitted and the plan runs with them.
    Runnable,
    // Every lane was denied and one yielded its memory or failed.
    Yielded,
    // Every lane was denied while pages are on their way back; nothing
    // changed, the lanes retry when the pages land, and other lanes run
    // meanwhile.
    Waiting,
  };
  double nextHealthCheckMilliseconds_ = 0.0;

  [[nodiscard]] Request &request(uint64_t requestId);
  [[nodiscard]] bool admitQueued(double nowMilliseconds);
  [[nodiscard]] bool admit(Request &request, double nowMilliseconds);
  // Where the lane's last state is kept: the last whole page before the
  // replay's final input token and, while the lane replays only its prompt,
  // before the prompt's generation prompt, where a later request resumes.
  // Generated history that a resumed lane replays is its own: the state
  // there is a checkpoint.
  [[nodiscard]] static uint32_t
  replayStateBoundary(const Request &request) noexcept;
  // replayStateBoundary while the lane replays only its prompt.
  [[nodiscard]] static uint32_t
  promptReplayBoundary(const Request &request) noexcept;
  [[nodiscard]] static uint32_t sharedPrefillBoundary(const Request &left,
                                                      const Request &right);
  [[nodiscard]] bool pendingSharedPrefill(const Request &request,
                                          uint32_t resumeBoundary) const;
  void completeAdmission(Request &request, CacheLookup &lookup, DraftContextPlan draft);
  [[nodiscard]] bool pollRestores(double nowMilliseconds);
  [[nodiscard]] DraftContextPlan
  configureDraftStatePlan(Request &request, uint32_t stateBoundary,
                          uint32_t junctionBoundary);
  // Plans a state at `tokens`, past `after` and no later than the replay
  // boundary, keeping the plan in order; a boundary planned both ways is
  // reusable. True when it inserted one.
  bool addStateBoundary(Request &request, uint32_t after, uint32_t tokens,
                        bool disposable);
  // Plans a junction at the boundary each queued peer of the same or lower
  // priority shares with this lane, past `after`, unless the peer ignores the
  // cache; true when one was added.
  [[nodiscard]] bool addSharedPrefillBoundaries(Request &request, uint32_t after);
  [[nodiscard]] DraftContextPlan
  pendingDraftStatePlan(const Request &request, uint32_t stateBoundary) const;
  // Arms the next planned boundary, if any, with the scheduler: after
  // admission, and after the scheduler has taken a command's progress.
  void armNextStateBoundary(Request &request);
  void discardPendingStateBoundaries(Request &request) noexcept;
  [[nodiscard]] bool retireCheckpoint(Request &request);
  void publishReachedStateBoundaries(Request &request,
                                     uint32_t promptProcessed);
  [[nodiscard]] Prepared prepare(BatchPlan &plan,
                                 std::vector<ModelBatchItem> &items,
                                 double nowMilliseconds);
  // The reclaim steps for a lane's state and for KV pages the engine's limit
  // refused. Idle memory of the kind refused stays for it to reuse; idle
  // memory of the other kind is released first. Each takes cache up to the
  // class allocate() is given.
  [[nodiscard]] CacheReclaimResult reclaimForState(ReclaimClass upTo);
  [[nodiscard]] CacheReclaimResult reclaimForKv(uint32_t pages, ReclaimClass upTo);
  [[nodiscard]] bool reclaimIdleState(bool keepLane) noexcept;
  [[nodiscard]] CacheReclaimResult reuseCachedStateWhilePaused(ReclaimClass upTo);
  [[nodiscard]] CacheReclaimResult reuseCachedPagesWhilePaused(
      const TokenAdmission &admission, ReclaimClass upTo);
  [[nodiscard]] bool growthPaused() const;
  // Memory a lane could not get, and what the engine knows about its return.
  struct Denial {
    metal::AllocationFailure allocationFailure = metal::AllocationFailure::None;
    // On its way back: pages whose copies are being written, or a reclaim
    // that waits for the transfer in flight. The lane waits; nobody yields.
    bool pending = false;
  };
  // An allocation's last attempt, and what the engine knows about the memory
  // it could not get.
  template <class Admission> struct Allocation final {
    Admission admission;
    Denial denial;
  };
  // What a lane does about memory it could not get. Pending memory returns
  // by itself: the lane waits. Otherwise a lane fails only when it is alone
  // with nothing left to reclaim; while other lanes hold memory or the host
  // refuses it, a running lane yields its memory and a lane being admitted
  // waits.
  enum class Verdict : uint8_t { Wait, Yield, Fail };
  [[nodiscard]] Verdict judge(const Denial &denial, uint64_t requestId) const;
  [[nodiscard]] bool anotherResident(uint64_t requestId) const;
  // The prompt rows a lane in prefill has processed, or the tokens a
  // decoding lane holds.
  [[nodiscard]] uint64_t completedTokens(const Request &request) const;
  // Whether lane a gives up its memory before lane b: the lower priority,
  // then, at equal priority, a lane in prefill before a decoding one, then
  // the one with fewer completed tokens.
  [[nodiscard]] bool yieldsBefore(const Request &a, const Request &b) const;
  // The resident lane in prefill or decode that yields first when every
  // lane's growth fails (yieldsBefore; on a tie, the later submission), or
  // nullptr without one. prepare() gates that lane's growth and suspends
  // that lane; preemptBelow() suspends it for a higher priority.
  [[nodiscard]] Request *laneToYield();
  // Runs one allocation of a lane's state or of KV pages, reclaiming cache
  // up to class upTo between attempts while that makes progress: what is in
  // use only for running work that would not yield for it, never for a
  // start a resident lane holds back. A refusal from the host reuses what
  // the engine holds; when that gives nothing, a request in service retries
  // as one (EngineConfig::serving), which only the engine's limit and
  // critical pressure refuse. A refusal from the engine's limit reclaims
  // cache; when that gives nothing, fallback, given the denial so far, may
  // let go of what the request itself pins, and the reclaim goes on.
  template <class Attempt>
  [[nodiscard]] auto allocate(Attempt &&attempt, bool inService, ReclaimClass upTo,
                              const std::function<bool(const Denial &)> &fallback = {})
      -> Allocation<std::invoke_result_t<Attempt &>>;
  // Takes a resident request's lane: its state and KV pages go back,
  // and it waits, for `reason`, until admission resumes it with room for
  // workEnd tokens of KV to replay its history from the cache.
  void suspendLane(Request &request, uint64_t workEnd, metal::AllocationFailure failure,
                   StateFailure reason, double nowMilliseconds);
  // A lane that could not grow yields its memory: it waits for memory, and
  // resident lanes drain before admission resumes.
  void suspendForGrowth(Request &request, uint64_t workEnd,
                        metal::AllocationFailure failure,
                        double nowMilliseconds);
  // The tokens the request's KV pages hold room for.
  [[nodiscard]] uint64_t kvCapacity(uint64_t requestId) const;
  // For a request of this priority that cannot start: suspends the lane that
  // yields first (laneToYield) if its priority is strictly lower, without a
  // drain. The lane waits as for a free one, and resumes from its current KV
  // capacity once admission reaches it again. True when one was suspended.
  // Memory on its way back makes no lane yield, as in prepare(): a request
  // refused while it is pending waits for it instead.
  [[nodiscard]] bool preemptBelow(RequestPriority priority, double nowMilliseconds);
  [[nodiscard]] bool resourceRetryReady(const Request &request,
                                        double nowMilliseconds) const noexcept;
  // The wait keeps the denial's allocation failure for its timeout message.
  // With a pending denial, memory is on its way back (pages of demoted blocks
  // land within commands): the wait limit measures time without progress, so
  // it moves out with every retry that follows progress and never fires while
  // progress has been made since the last attempt.
  void deferResourceRetry(Request &request, double nowMilliseconds,
                          const Denial &denial,
                          StateFailure reason = StateFailure::MemoryPressure) noexcept;
  // Waiting for scheduling or for a prefix does not consume the memory
  // wait limit, so the wait's record is reset. A request that holds
  // admission closed (Request::refusedMemory) keeps when its wait began and
  // which lanes were admitted before it was refused.
  void deferWait(Request &request) noexcept;
  // The wait limit tick() enforces, or zero while it enforces none: a
  // pending wait that has seen progress waits for its next attempt. The
  // limit restarts whenever a lane submitted before the request, or one
  // ResourceWait::admittedBefore counts, has work in flight.
  [[nodiscard]] double resourceDeadline(const Request &request) const noexcept;
  void signalResourceProgress() noexcept;
  void apply(const BatchPlan &plan, std::span<const ModelStepResult> results,
             double wallMilliseconds, double cycleMilliseconds,
             bool representativePrefillTiming, double nowMilliseconds);
  // Whether the command in flight holds the request's lane.
  [[nodiscard]] bool inFlight(uint64_t requestId) const;
  // Ends a request early. While its command or restore runs, the first end
  // waits there (the command's mask wait is abandoned, the restore's read
  // cancelled) and apply()/pollRestores() settle it once that work drains;
  // otherwise the request ends now: Cancelled completes it, any other
  // outcome fails it.
  void settle(Request &request, LaneEnd end);
  [[nodiscard]] static LaneEnd deadlineEnd();
  [[nodiscard]] static LaneEnd capacityExhausted(std::string_view what,
                                                 metal::AllocationFailure failure,
                                                 std::string_view detail = {});
  void finish(Request &request, EngineFinishReason reason,
              std::span<const float> optionLogits);
  void finishFailure(Request &request, LaneEnd end);
  // Gives back what a lane holds: its planned and armed state boundaries,
  // its state (keepContinuation keeps the model's host continuation of a
  // suspended request) and its KV leases. The caller signals progress
  // when the memory becomes available to waiting requests: a lane that
  // held it across passes and ends (release) or fails its restore, or one
  // suspended for a higher priority (preemptBelow). An admission returning
  // what it took this pass does not: that memory was available before, and
  // a signal would count as progress in deferResourceRetry and move the
  // refused request's own wait limit out on every retry. Nor does a growth
  // suspension (suspendForGrowth), which starts the recovery drain instead.
  void vacateLane(Request &request, bool keepContinuation);
  void release(Request &request);
  void sweepTerminal();

  EngineConfig config_;
  // kPrefillCheckpointTokens and kResourceWaitTimeoutMilliseconds, or the
  // test seam's (TestConfig).
  uint32_t checkpointTokens_;
  double resourceWaitTimeoutMilliseconds_;
  Cache &cache_;
  model::Model &model_;
  EngineEventSink &events_;
  Scheduler scheduler_;
  std::unordered_map<uint64_t, Request> requests_;
  // Pressure preempted work, a lane is still resident, and memory is still
  // short (growth is paused or allocationFailed_), up to the drain's end.
  [[nodiscard]] bool drainingForRecovery() const;
  // The highest priority among suspended requests: admission recovers from a
  // suspension while there is one.
  [[nodiscard]] std::optional<RequestPriority> suspendedTier() const;
  // Whether admission tries the request. During recovery (`tier`) only the
  // requests of a strictly higher priority than every suspended one are,
  // and, once the drain is over, the suspended ones; the others wait behind
  // the suspended lanes.
  [[nodiscard]] bool admissionTries(const Request &request,
                                    std::optional<RequestPriority> tier,
                                    bool draining) const;
  std::function<void()> completionNotifier_;
  std::optional<Pending> pending_;
  // When the latest command retired, until a tick finds nothing to do.
  std::optional<double> busySinceMilliseconds_;
  // The lanes admit() has obtained so far, including those it gave back when
  // the attempt's pages were refused (Request::admission).
  uint64_t admissions_ = 0;
  uint64_t resourceEpoch_ = 1;
  // The resource wait limit after the latest suspension; zero once passed
  // or when no request is suspended.
  double drainEndMilliseconds_ = 0.0;
  // An allocation failed since the latest suspension or since a resident
  // lane last released its memory, or the suspension itself met a limit
  // that only freed memory lifts, unlike a host pause.
  bool allocationFailed_ = false;
  EngineSnapshot counters_;
};

} // namespace splash::engine

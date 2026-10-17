#pragma once

#include "ops/Vision.hpp"
#include "engine/Cache.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/Scheduler.hpp"
#include "engine/Types.hpp"
#include "engine/WriteBehind.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace splash::engine {

// Prefill checkpoints sit at multiples of this many tokens
// (plannedCheckpoints) by default: two draft windows balance recovery
// granularity and capture work.
inline constexpr uint32_t kPrefillCheckpointTokens =
    2 * model::ExecutionLimits::draftContextTokens;
static_assert(kPrefillCheckpointTokens >= model::ExecutionLimits::draftContextTokens &&
                  kPrefillCheckpointTokens % KvCache::pageTokens == 0,
              "a prefill checkpoint interval spans a draft window and whole KV pages");
// How long a request waits for memory before it fails, and a resident
// drain lasts, by default.
inline constexpr double kResourceWaitTimeoutMilliseconds = 30000.0;
// A request refused memory retries at once when the engine frees some
// (signalResourceProgress). Memory that comes back without that, as the
// host's does, it finds by retrying this often: at most a tenth of a second
// later, without spinning the loop on attempts.
inline constexpr double kResourceRetryBackoffMilliseconds = 100.0;

// The engine's constructor refuses a config without its context window, its
// vocabulary or the governor's two hooks, which the bootstrap sets from the
// model, its memory plan and the governor (connectToGovernor).
struct EngineConfig final {
  uint32_t maxContext = 0;
  uint32_t vocabularySize = 0;
  // Patches per image the server's --max-image-pixels allows; zero rejects
  // images.
  uint32_t maxImagePatches = ops::kMaximumImagePatches;
  // Decode time owed for each unit of time a prefill runs while requests of
  // equal or higher priority decode. At 0.5 a stream keeps about a third of
  // its solo rate through a long prefill, which meanwhile takes 1.5x as long;
  // zero alternates one command of each kind.
  double decodeShare = 0.5;
  // The prefill checkpoint interval and the resource wait limit.
  uint32_t prefillCheckpointTokens = kPrefillCheckpointTokens;
  double resourceWaitTimeoutMilliseconds = kResourceWaitTimeoutMilliseconds;
  // Host growth admission, supplied by the runtime governor. Queried while
  // resident lanes drain after a suspension, and when a boundary's state
  // buffers are refused: while growth is paused, extents released for them
  // cannot hold them. Allocation reads the cause of a refusal instead.
  std::function<bool()> growthPaused;
  // Marks the allocations that follow as memory a request in service needs,
  // which the pause does not hold back, or clears the mark, and returns the
  // mark it replaced (MemoryGovernor::setServing).
  std::function<bool(bool)> serving;
};

// While an active one lives, allocations are memory a request in service
// needs (EngineConfig::serving). It restores the mark it found, so that a
// start's mark (RuntimeResources::beginServing) outlives an allocation the
// engine marks before serving begins.
class Serving final {
public:
  Serving(const std::function<bool(bool)> &mark, bool active)
      : mark_(active ? &mark : nullptr), previous_(mark_ && (*mark_)(true)) {}
  ~Serving() {
    if (mark_)
      (*mark_)(previous_);
  }
  Serving(const Serving &) = delete;
  Serving &operator=(const Serving &) = delete;

private:
  const std::function<bool(bool)> *mark_;
  bool previous_;
};

// What a request fails with once it has waited the resource wait limit for
// memory, last refused for `failure`.
[[nodiscard]] std::string resourceTimeoutMessage(metal::AllocationFailure failure);

// The progress checkpoints a request plans between the point it resumes from
// and its replay boundary: the multiples of `interval` at least one prefill
// chunk past the one and before the other, none when `interval` is zero.
// They sit at multiples of the interval, so requests over the same prompt
// share them. A checkpoint within one prefill chunk of either end would cost
// a cached state, its capture and its draft window for less than a chunk of
// recompute.
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
  // Requests the latest admission pass held back behind the first one
  // refused memory, whatever they wait for themselves, and a request refused
  // memory while a pass keeps it out of its memory wait. During recovery
  // admission tries only suspended requests and those of a strictly higher
  // priority, and holds back only those.
  uint32_t heldBehindRefusal = 0;
  // Requests admitted into a restore of their prefix from disk, waiting for
  // its reads rather than for memory.
  uint32_t restoring = 0;
  uint32_t suspended = 0;
  double oldestWaitMilliseconds = 0.0;
  bool draining = false;
};

// One live request's progress, for /status active_requests: its phase and
// priority, how many prompt tokens are encoded (cached ones included) and
// generated, and its age since it arrived, which the native runtime that
// received it adds (NativeRuntime::statusSnapshot).
struct ActiveRequestSnapshot final {
  uint64_t id = 0;
  Phase phase = Phase::Queued;
  RequestPriority priority = RequestPriority::Normal;
  uint32_t promptTokens = 0;
  uint32_t promptProcessed = 0;
  uint32_t generatedTokens = 0;
  uint32_t maxNewTokens = 0;
  double ageMilliseconds = 0.0;
};

struct EngineSnapshot final {
  SchedulerSnapshot scheduler;
  // Requests not yet finalized, in submission (id) order.
  std::vector<ActiveRequestSnapshot> activeRequests;
  CacheSnapshot resources;
  uint32_t maximumContextTokens = 0;
  uint64_t submitted = 0;
  uint64_t completed = 0;
  uint64_t cancelled = 0;
  uint64_t failed = 0;
  uint64_t cacheHits = 0;
  uint64_t coldMisses = 0;
  uint64_t reusedTokens = 0;
  // Tokens whose KV requests read from cached blocks other requests
  // computed, past the cached prefixes they started from.
  uint64_t deduplicatedKvTokens = 0;
  uint64_t replayStatePublications = 0;
  uint64_t deduplicatedStatePublications = 0;
  // Publications completed after reclaiming a cached state.
  uint64_t recycledStatePublications = 0;
  // Publications written straight to disk because no cache slot was free.
  uint64_t diskStatePublications = 0;
  // States prefill commands captured at a boundary short of their end, for
  // lanes that committed.
  uint64_t stateCaptures = 0;
  // Prefill chunks, a lane's rows in one command, that ended at a state
  // boundary before their planned rows: its buffers or the KV past it were
  // refused, or a waiting request starts from its state.
  uint64_t captureCuts = 0;
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
  // A persistent tier's newest restore points (WriteBehind).
  WriteBehindSnapshot writeBehind;
};

// KV blocks define prefix identity; composite recurrent state is attached
// at sparse progress points, replay boundaries, and shared KV junctions.
//
// Failure contract. A request the engine cannot serve ends with a LaneOutcome
// (Types.hpp): its own result, which the engine reports and survives. submit()
// reports an invalid request with std::invalid_argument, which the caller
// answers with that request's error. Any other exception out of submit(), and
// every exception out of tick(), cancel(), failRequest(), provideMask(),
// reclaimMemory() and flushRestorePoints(), is engine-fatal: NativeRuntime
// reports EngineUnhealthy and the process exits. Code below them therefore
// does not roll back on an exception; the only cleanup on that path is RAII
// teardown itself needs (state IO drains, FileRestore, Serving).
class Engine final {
public:
  Engine(EngineConfig config, Cache &cache, model::Model &model,
         EngineEventSink &events);

  // The request passed protocol::validateRequest; submit checks only what
  // the engine knows: vocabulary, context window, and vision with the
  // server's per-image patch cap.
  void submit(EngineRequest request);
  void observePrefill(uint32_t rows, double wallMilliseconds, double hostMilliseconds) {
    scheduler_.observePrefill(rows, wallMilliseconds, hostMilliseconds);
  }
  void cancel(uint64_t requestId);
  void failRequest(uint64_t requestId, LaneOutcome outcome, std::string message);
  void provideMask(uint64_t requestId, std::span<const uint32_t> words);
  void setCompletionNotifier(std::function<void()> notifier);

  [[nodiscard]] bool tick(double nowMilliseconds);
  [[nodiscard]] bool commandInFlight() const noexcept {
    return pending_.has_value();
  }
  // No request is waiting, running or draining.
  [[nodiscard]] bool idle() const noexcept {
    return requests_.empty() && !pending_;
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
  // At a clean stop, once tick() has stopped: makes every restore point
  // still waiting for its delay durable, the newest first, as the writes it
  // starts land (WriteBehind::flush). True once none is left. The requests
  // and the command in flight stay as they are, for teardown.
  [[nodiscard]] bool flushRestorePoints();

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
      // A rolling checkpoint, the lane's own progress, retired once the next
      // one or a junction is published, which takes its buffers, if it needs
      // them, only after its command has completed; the prompt's replay point
      // leaves the last one cached when its own state fits beside it.
      // Otherwise a state a later request resumes from.
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
    // The blocks it restored or published in any residency end at this
    // token. A resume after a suspension counts the blocks it restores past
    // them as KV another request computed (completeAdmission). The blocks
    // below, which it restores or its replay finds cached, count as its own,
    // though one evicted while it was suspended may hold KV another request
    // computed again.
    uint32_t publishedTokens = 0;
    // Tokens of blocks another request computed whose pages its next command
    // reads: blocks it computed and then found cached
    // (Cache::publishCommittedBlocks), and those a resume restored past
    // publishedTokens. Then those its commands read, which it reports when
    // it completes.
    uint32_t unreadDeduplicatedKvTokens = 0;
    uint32_t deduplicatedKvTokens = 0;
    // Its speculative decoding so far, which it reports when it completes.
    SpeculativeCounts speculation;
    // A failed dispatch must fit before replay can consume any model work.
    uint64_t resumeKvTargetTokens = 0;
    ResourceWait resourceWait;
    // The latest attempt to start it was refused memory. Until it starts,
    // nothing that comes after it in admission order is admitted, in
    // ordinary admission or among suspended requests during recovery
    // (admitQueued); a pass that does not schedule it or a prefix wait
    // leaves it in place.
    bool refusedMemory = false;
    // The request refused memory that the latest admission pass held it
    // back behind, without trying it. It counts as held while that request
    // still waits refused (ResourceWaitSnapshot::heldBehindRefusal).
    std::optional<uint64_t> heldBehind;
    // The prompt from submit on, then the committed output; request.prompt
    // is empty.
    std::vector<uint32_t> exactTokens;
    // The first output token, sent to the client when the prefill selected it
    // (ModelStepResult::firstToken) and until the first decode cycle commits
    // it: that cycle emits it first, and its event leaves it out.
    std::optional<uint32_t> sentFirstToken;
    // Made from exactTokens and the request's prompt identity, which do not
    // change while the request waits; refreshed each pass and dropped once it starts or
    // skips the cache.
    std::optional<CacheProbe> admissionProbe;
    std::vector<StateBoundary> stateBoundaries;
    size_t stateBoundaryCursor = 0;
    StateCheckpoint latestCheckpoint;
    // The block of the prompt's replay boundary, where the conversation's
    // next turn resumes: its state is in use from the moment the request
    // reaches, reuses or restores it until the request ends, suspended or not.
    StateUse replayPoint;
    // The block of the state it restored in this residency, until a step of
    // it computed from that state reports a finite result.
    std::optional<uint64_t> unprovenState;
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

  // A state boundary a prefill command captures
  // (ModelBatchItem::stateCaptures), or the one it ends at whose state apply
  // takes from the lane after it (fromLane).
  struct Capture final {
    uint32_t tokens = 0;
    // Room for the state's buffers came from reclaiming cached memory; its
    // publication counts as recycled (recycledStatePublications).
    bool recycled = false;
    // Its buffers were refused before the command while the lane held a
    // checkpoint (reserveCapture): the command ends at it without capturing
    // it.
    bool fromLane = false;
  };
  // The buffers reserveCapture() took for a boundary's capture, null when
  // refused, whether reclaiming cached memory made room for them, and
  // whether a refused state is taken from the lane after the command.
  struct Reservation final {
    std::shared_ptr<StateReservation> buffers;
    bool recycled = false;
    bool fromLane = false;
  };
  struct Pending final {
    BatchPlan plan;
    // Each plan item's captures, in the order of its stateCaptures, then the
    // one taken from the lane, if any.
    std::vector<std::vector<Capture>> captures;
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
  // An unstarted request waits while a resident producer has yet to publish
  // a state it would start from (awaitedSharedBoundary).
  [[nodiscard]] bool pendingSharedPrefill(const Request &request,
                                          uint32_t resumeBoundary) const;
  // The last of the producer's unpublished boundaries that a request
  // resuming at resumeBoundary would start from: past it, within the prefix
  // they share. The producer publishes its boundaries in order, so the
  // request waits until that one is published; 0 when none.
  [[nodiscard]] static uint32_t awaitedSharedBoundary(const Request &waiter,
                                                      uint32_t resumeBoundary,
                                                      const Request &producer);
  // The first boundary up to `end` that ends a request's wait for the
  // producer's prefix (awaitedSharedBoundary), else `end`.
  [[nodiscard]] uint32_t awaitedBoundary(const Request &producer, uint32_t end) const;
  void completeAdmission(Request &request, CacheLookup &lookup, DraftContextPlan draft);
  [[nodiscard]] bool pollRestores(double nowMilliseconds);
  [[nodiscard]] DraftContextPlan
  configureDraftStatePlan(Request &request, uint32_t stateBoundary,
                          uint32_t junctionBoundary);
  // Plans a state at `tokens`, past `after` and no later than the replay
  // boundary, keeping the plan in order; a boundary planned both ways is
  // reusable. A new one needs a whole draft window: unless the lane's draft
  // state is complete at `after` (completeAfter), it lies at least a draft
  // window past `after` or past a boundary planned before it. True when it
  // inserted one.
  bool addStateBoundary(Request &request, uint32_t after, uint32_t tokens,
                        bool disposable, bool completeAfter);
  // Plans a junction at the boundary each queued peer of the same or lower
  // priority shares with this lane, past `after`, unless the peer ignores the
  // cache; true when one was added.
  [[nodiscard]] bool addSharedPrefillBoundaries(Request &request, uint32_t after,
                                                bool completeAfter);
  [[nodiscard]] DraftContextPlan
  pendingDraftStatePlan(const Request &request, uint32_t stateBoundary) const;
  void discardPendingStateBoundaries(Request &request) noexcept;
  // Another unfinished lane holds this lane's recovery point as its own.
  [[nodiscard]] bool checkpointShared(const Request &request) const;
  // The lane's recovery point is still cached, in either tier, and no reader
  // pins it: a reclaim could take it.
  [[nodiscard]] bool checkpointReclaimable(const Request &request) const;
  // A reader pins the lane's recovery point, which no other lane holds: it
  // cannot retire (retireCheckpoint), so no checkpoint after it is published
  // meanwhile.
  [[nodiscard]] bool checkpointPinned(const Request &request) const;
  [[nodiscard]] bool retireCheckpoint(Request &request);
  // Publishes the request's complete blocks up to committedTokens and keeps
  // the tokens of those found cached for its next command to read.
  void publishCommittedBlocks(Request &request, uint32_t committedTokens);
  // Publishes the states of the boundaries the lane's command reached: the
  // states it captured (`captures`, `states`), and at its end one without
  // buffers, taken from the lane now that the command has completed or
  // through the disk tier.
  void publishReachedStateBoundaries(
      Request &request, uint32_t promptProcessed, std::span<const Capture> captures,
      std::span<const std::shared_ptr<const CompositeState>> states, double nowMilliseconds);
  [[nodiscard]] Prepared prepare(BatchPlan &plan, std::vector<ModelBatchItem> &items,
                                 std::vector<std::vector<Capture>> &captures,
                                 double nowMilliseconds);
  // Reserves the buffers of the states a prefill lane's command reaches,
  // boundary by boundary, and the KV up to the next one; returns the rows
  // the command keeps. It ends at the first boundary whose buffers or
  // following KV are refused, and at one a request waiting for a prefix
  // starts from. A checkpoint the command passes on its way to a state that
  // retires it once published is not captured, and that state takes
  // buffers of its own class; refused the KV up to that state, the command
  // ends at the checkpoint and captures it. The prompt's replay point
  // refused buffers of its own takes those of a checkpoint the command
  // captures before it.
  [[nodiscard]] uint32_t reserveCaptures(Request &request, const BatchItem &item,
                                         ReclaimClass upTo,
                                         std::vector<StateCapture> &states,
                                         std::vector<Capture> &captures);
  // Takes the buffers of the state of class `kind` the lane's command
  // captures at `tokens`: from the model's pool or one admission, then from
  // cache its class may take, but only while the lane holds no checkpoint a
  // reclaim could take. The lane's checkpoint stays its recovery point until
  // a state that retires it is published, so no state takes its buffers, or
  // room a reclaim could make of it, before the command: refused the pool
  // and the admission while the lane holds one, a state ends its command at
  // its boundary, and apply takes it from the lane after the command
  // (publishReachedStateBoundaries). Only the first state the command
  // captures follows that checkpoint (followsCheckpoint): a later one
  // follows the state before it in the command. A checkpoint whose
  // predecessor a reader pins takes only the pool or the admission too:
  // refused, its command ends at it, and apply publishes it from the lane
  // only if the pin has gone by then. Null buffers when refused.
  [[nodiscard]] Reservation reserveCapture(Request &request, StateClass kind, uint32_t tokens,
                                           bool followsCheckpoint);
  // Makes room for one state's buffers, a step of what its class may take
  // (Cache::reclaimOneState) at a time, until `take` gets them, nothing more
  // of the class goes, or the extents given cover one state; true once
  // `take` got them. Between commands only.
  template <class Take> [[nodiscard]] bool makeStateRoom(StateClass kind, Take &&take);
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
  // The output tokens the client was sent: the committed ones and a first
  // token that no cycle has committed yet (Request::sentFirstToken), as when
  // the request is suspended or cancelled before its first decode cycle.
  [[nodiscard]] static uint32_t generatedTokens(const Request &request) noexcept;
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
  // let go of what the request itself pins, and the reclaim goes on. An
  // optional allocation, whose refusal only shortens a command, records no
  // failure (allocationFailed_).
  template <class Attempt>
  [[nodiscard]] auto allocate(Attempt &&attempt, bool inService, ReclaimClass upTo,
                              const std::function<bool(const Denial &)> &fallback = {},
                              bool optional = false)
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
  void apply(const BatchPlan &plan, std::span<const std::vector<Capture>> captures,
             std::span<const ModelStepResult> results, double wallMilliseconds,
             double hostMilliseconds, double cycleMilliseconds,
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
  // Gives back what a lane holds: its planned state boundaries,
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
  Cache &cache_;
  WriteBehind writeBehind_;
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
  // Whether the request refused memory that a pass held another one back
  // behind (Request::heldBehind) still waits refused: it is neither
  // finalized nor started since.
  [[nodiscard]] bool holdsBack(std::optional<uint64_t> refused) const;
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

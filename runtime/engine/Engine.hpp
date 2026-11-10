#pragma once

#include "ops/Vision.hpp"
#include "engine/Cache.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/MemoryPlan.hpp"
#include "engine/PassCommand.hpp"
#include "engine/Scheduler.hpp"
#include "engine/Types.hpp"
#include "engine/WeightResidency.hpp"
#include "engine/WriteBehind.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
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
// The output a request's KV is projected to hold at admission, at most: the
// tokens it may still generate, clipped to this many. A request that
// outgrows it converts to Spill when memory refuses its growth
// (EngineSpillConfig).
inline constexpr uint32_t kProjectedOutputTokens = 4096;

// The engine's constructor refuses a config without its context window, its
// vocabulary or its governor, which the bootstrap sets from the model, its
// memory plan and the governor (connectToGovernor).
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
  // G: the longest a prefill unit (a command, or one pass of it) may take
  // while requests of equal or higher priority decode; a unit takes longer
  // only where one layer of a group, or a cut at its floor rows, does
  // (Scheduler). Finite and positive.
  double decodeGapMilliseconds = 500.0;
  // The prefill checkpoint interval and the resource wait limit.
  uint32_t prefillCheckpointTokens = kPrefillCheckpointTokens;
  double resourceWaitTimeoutMilliseconds = kResourceWaitTimeoutMilliseconds;
  // The governor that admits the engine's memory and whose ledger holds
  // the requests' claims on it (ClaimHandle), which connectToGovernor sets.
  MemoryGovernor *governor = nullptr;
  // Which requests keep their KV on the SSD cache (KvMode), from the memory
  // plan; none unless the target can spill.
  EngineSpillConfig spill;
  // The weights the engine releases while idle and restores before it runs
  // the next request (WeightResidency): the native loop's; empty where
  // nothing is released.
  std::optional<WeightResidencyConfig> weights;
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
  // Starts that wait for the SSD cache's quota of active KV, which other
  // requests' active KV holds.
  uint32_t diskQuota = 0;
  // The starts the scheduler wanted that the latest admission pass held back
  // behind one refused memory or that quota, without trying them, and during
  // recovery the suspended requests held back behind a refused resume or
  // start. Usually none: the scheduler leaves out the starts a refused one
  // leads unless their command ends their prompts.
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
// generated, where its KV lives, and its age since it arrived, which the
// native runtime that received it adds (NativeRuntime::statusSnapshot).
struct ActiveRequestSnapshot final {
  uint64_t id = 0;
  Phase phase = Phase::Queued;
  RequestPriority priority = RequestPriority::Normal;
  uint32_t promptTokens = 0;
  uint32_t promptProcessed = 0;
  uint32_t generatedTokens = 0;
  uint32_t maxNewTokens = 0;
  KvMode kvMode = KvMode::Resident;
  double ageMilliseconds = 0.0;
};

// The requests of a target that can spill (EngineConfig::spill): those that
// started in each KV mode; those that converted to Spill when memory
// refused their growth; those whose output the SSD cache's quota, or memory
// for a start the cache could not take, bounded (once per request, at its
// start or a resume), and those that ended at that bound or once the cache
// could hold no more of their KV; and those refused because the quota cannot
// hold their prompt, the daily write budget is spent or the tier takes no
// writes. Those waiting for the quota now are ResourceWaitSnapshot::diskQuota.
struct SpillRequestsSnapshot final {
  uint64_t spill = 0;
  uint64_t resident = 0;
  uint64_t converted = 0;
  uint64_t bounded = 0;
  uint64_t lengthEnds = 0;
  // Of those, the converted lanes': one whose output the quota bounded on a
  // resume, and an end at that bound or at a quota that stopped its writes.
  uint64_t convertedBounded = 0;
  uint64_t convertedLengthEnds = 0;
  uint64_t refusedQuota = 0;
  uint64_t refusedWriteBudget = 0;
  uint64_t refusedClosed = 0;
};

// The decode gaps past the decode gap target G that lanes saw between two
// of their token events, each by the first cause that held in it: a prefill
// unit ran, the lane's claim waited for memory, the decode command that
// ended it read from the SSD (experts the store missed, spilled K/V), a pool
// gave a unit back, a plan was made again, or none of these. Of the gaps a
// prefill unit caused, those units' time.
enum class GapCause : uint8_t { PrefillUnit, ClaimWait, SsdRead, PoolRelease, Replan, Other };
inline constexpr size_t kGapCauses = 6;
struct DecodeGapCausesSnapshot final {
  std::array<uint64_t, kGapCauses> gaps{};
  std::array<double, kGapCauses> milliseconds{};
  double prefillMilliseconds = 0.0;
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
  // Prefill plans the engine sent back to the scheduler while preparing
  // them: a lane's memory was refused or held back, a capture cut its rows,
  // or a lane converted to Spill.
  uint64_t replans = 0;
  DecodeGapCausesSnapshot decodeGapCauses;
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
  // The copies of the newest restore points ahead of need (WriteBehind).
  WriteBehindSnapshot writeBehind;
  SpillRequestsSnapshot spill;
  // The host time of each tick that submitted a command, from its first
  // planning call (admission) to the command's submission.
  model::LatencyStats tickHostMilliseconds;
};

// KV blocks define prefix identity; composite recurrent state is attached
// at sparse progress points, replay boundaries, and shared KV junctions.
//
// Failure contract. A request the engine cannot serve ends with a LaneOutcome
// (Types.hpp): its own result, which the engine reports and survives. submit()
// reports an invalid request with std::invalid_argument, which the caller
// answers with that request's error. Any other exception out of submit(), and
// every exception out of tick(), cancel(), stop(), failRequest(),
// provideMask(), collectLandings() and flushRestorePoints(), is
// engine-fatal: NativeRuntime reports EngineUnhealthy and the process exits.
// Code below them therefore does not roll back on an exception; the only
// cleanup on that path is RAII teardown itself needs (state IO drains,
// FileRestore, the governor's Scope).
class Engine final {
public:
  Engine(EngineConfig config, Cache &cache, model::Model &model,
         EngineEventSink &events);
  // Takes its own lanes back from the governor's ledger; the cache, which
  // outlives it and another engine may share, stays.
  ~Engine();

  // The request passed protocol::validateRequest; submit checks only what
  // the engine knows: vocabulary, context window, and vision with the
  // server's per-image patch cap.
  void submit(EngineRequest request);
  void cancel(uint64_t requestId);
  // Ends a request as cancel() does (settle), but as EngineFinishReason::Stop,
  // counted completed as one that ends at its stop token is. One that has
  // generated nothing yet is cancelled; an unknown id is ignored.
  void stop(uint64_t requestId);
  void failRequest(uint64_t requestId, LaneOutcome outcome, std::string message);
  void provideMask(uint64_t requestId, std::span<const uint32_t> words);
  void setCompletionNotifier(std::function<void()> notifier);

  // While the weights are released or being restored (weightsHeld), a tick
  // runs its timed checks, collects transfers and runs write-behind, then
  // takes back one part of the weights while a request waits, inside a
  // Floor claim's scope, which the host's margins do not refuse. A part that
  // comes back ends their waits for memory; a refused one makes them wait
  // for memory (resourceWaitSnapshot), retried at every tick, which the loop
  // runs for their retries and limits; each fails once the attempt at or
  // past its own limit is refused. Once no request waits, the restore goes
  // on inside a Start claim's scope, as the host's margins admit it, and a
  // refused part ends it:
  // the weights give back what came back. No request runs until they are
  // back. While a part is written on its thread (weightsWriting), a tick
  // only ends the requests whose deadlines passed: the write's end wakes the
  // loop (the completion notifier).
  [[nodiscard]] bool tick(double nowMilliseconds);
  // A command, or a part of the weights being written, whose writer may run
  // commands of its own: the control pass waits for its end (FdTransport).
  [[nodiscard]] bool commandInFlight() const noexcept {
    return pending_.has_value() || weightsWriting();
  }
  // No request is waiting, running or draining.
  [[nodiscard]] bool idle() const noexcept {
    return requests_.empty() && !pending_;
  }
  [[nodiscard]] std::optional<double> nextWakeupMilliseconds() const;
  [[nodiscard]] EngineSnapshot snapshot() const;
  [[nodiscard]] ResourceWaitSnapshot resourceWaitSnapshot(double nowMilliseconds) const;

  // The control pass's engine side, between commands (std::logic_error
  // while a command is in flight). collectLandings lands what transfers
  // finished and refills the Index runway (spillLanded), then gives the ledger
  // the lanes' projected growth and highest priority, and the cache the
  // chains of the requests without a lane, which their start restores
  // (Cache::setWantedChains): those admission probed, their probes brought
  // up to date, and the others from a probe of this pass alone, none for a
  // request that skips the cache.
  void collectLandings(double nowMilliseconds);
  // Releases the weights (EngineConfig::weights) while the engine is idle
  // and they are held: once the idle release has passed without a request
  // (releaseIdleWeights), or at once (releaseWeights). True when it did.
  [[nodiscard]] bool releaseIdleWeights(double nowMilliseconds);
  [[nodiscard]] bool releaseWeights(double nowMilliseconds);
  // At Ready: the idle release counts from here, not from warmup.
  void startIdleClock(double nowMilliseconds);
  // Restoring from the arrival of a request that found them released.
  [[nodiscard]] WeightsSnapshot weightsSnapshot() const;
  // Neither released nor being restored; true for an engine with no weights
  // to release.
  [[nodiscard]] bool weightsHeld() const noexcept { return !weights_ || weights_->held(); }
  // A part of the weights is being written on a thread of its own
  // (model::WeightImages), which may allocate and run commands meanwhile:
  // the engine leaves memory and the backend alone until it ends.
  [[nodiscard]] bool weightsWriting() const noexcept { return weights_ && weights_->writing(); }
  // At a clean stop, once tick() has stopped: makes every restore point a
  // persistent tier has waiting durable, the newest first, as the writes it
  // starts land (WriteBehind::flush), once no part of the weights is being
  // written. True once none is left. The requests and the command in flight
  // stay as they are, for teardown.
  [[nodiscard]] bool flushRestorePoints();

private:
  // A lane's end other than its output's, and for a retryable outcome the
  // seconds after which a retry may succeed, 0 for at once.
  struct LaneEnd final {
    LaneOutcome outcome;
    std::string message;
    uint32_t retryAfterSeconds = 0;
  };

  struct ResourceWait final {
    StateFailure reason = StateFailure::None;
    // What refused the memory at the latest attempt.
    metal::AllocationFailure allocationFailure = metal::AllocationFailure::None;
    std::optional<double> startedMilliseconds;
    // The admissions counted at the request's latest refusal while it has no
    // lane (its start's, its resume's, its suspension), or at its first
    // refusal in service: the lanes admitted up to then that compete for
    // what it was refused, any for memory and, for the SSD cache's quota,
    // those holding some of it by their own count (competesForQuota),
    // restart its limit while they work (tick, deferResourceRetry).
    // A wait for a lane does not record it.
    std::optional<uint64_t> admittedBefore;
    double deadlineMilliseconds = 0.0;
    // The latest tick at which a lane submitted before the request, or one
    // that admittedBefore counts, had work in flight; the limit restarts from
    // it.
    double earlierLaneWorkMilliseconds = 0.0;
    double retryMilliseconds = 0.0;
    uint64_t epoch = 0;
    // Its latest refusal waits for memory on its way back; the limit fires
    // only without progress.
    bool pending = false;
    // When the refusals that wait for memory on its way back began: its
    // claim holds room for it from there for the resource wait limit
    // (syncClaims), however long it waited for anything else before.
    std::optional<double> pendingSinceMilliseconds;
    // That limit, or its wait's deadline, has passed: its claim may go on
    // waiting but holds nothing, a retry's refusal too (syncClaims).
    bool holdEnded = false;
    // A plan of its kind left the lane out while it was ready to retry: it
    // waits for that plan's command, not for memory, until a plan retries
    // it (passOver).
    bool passedOver = false;
    // It was refused the SSD cache's quota of active KV, which other
    // requests' active KV holds, rather than memory: what it waits for, and
    // while its refusal stands, what it holds back (Request::refusedResources).
    bool diskQuota = false;
  };

  // A request's one claim on memory in the governor's ledger
  // (MemoryLedger::open), closed when the request leaves requests_.
  struct ClaimHandle final {
    MemoryGovernor *governor = nullptr;
    ClaimId id;
    ClaimHandle() = default;
    ClaimHandle(ClaimHandle &&other) noexcept;
    ~ClaimHandle();
    // Closes the claim now, as the request ends.
    void reset() noexcept;
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
    // Lanes admitted before a request's refusal that
    // ResourceWait::admittedBefore records, and that compete for what it was
    // refused, restart its wait's limit.
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
    // Where its committed KV lives: chosen at each attempt to start it
    // (placement), kept from its start on, across suspensions too, and
    // turned to Spill once by a conversion.
    KvMode kvMode = KvMode::Resident;
    // Turned to Spill by its own conversion (convertToSpill).
    bool converted = false;
    // Where its KV goes if it starts at the attempt admission makes in this
    // pass, placed in the memory as that attempt finds it (admitQueued):
    // whether it competes for the quota and where its start puts its KV
    // read it (competesForQuota, admit). None before its first attempt.
    std::optional<KvMode> placement;
    // The most tokens it may generate, which the model is given at each
    // start, a resume's too: maxNewTokens, or fewer when the SSD cache's
    // quota, or memory for a start the cache cannot take, bounds its output
    // (admitSpill), which each attempt to start it decides anew.
    uint32_t outputLimit = 0;
    // SpillRequestsSnapshot counted its output bounded; it stays set when a
    // later attempt leaves the output unbounded, so the request counts once.
    bool boundCounted = false;
    // The KV a Resident lane takes when it resumes, before its replay
    // consumes any model work: what it held, or the work it was refused
    // room for (suspendLane). A Spill lane takes its replay's pages a
    // command at a time instead.
    uint64_t resumeKvTargetTokens = 0;
    ResourceWait resourceWait;
    ClaimHandle claim;
    // The latest attempt to start it was refused memory, or the SSD cache's
    // quota of active KV. While the scheduler wants it, it holds back the
    // starts the scheduler wants after it that compete for it
    // (competesForQuota), and a suspended one the resumes after it
    // (admitQueued). A pass that does not want it or a prefix wait
    // leaves it in place.
    bool refusedResources = false;
    // The request refused resources that the latest admission pass held it
    // back behind, without trying it. It counts as held while that request
    // still waits refused (ResourceWaitSnapshot::heldBehindRefusal).
    std::optional<uint64_t> heldBehind;
    // The rows a prefill lane's command may take this tick: where prepare
    // cut it at a state boundary its capture was refused at, or a waiting
    // request starts from (LaneShape).
    std::optional<uint32_t> rowCap;
    // The states a prepare that sent its plan back reserved for the lane's
    // command over [start, end), with their captures, which the plan made
    // again takes if it runs the lane over the same rows; this tick only.
    struct KeptCaptures final {
      uint32_t start = 0, end = 0;
      std::vector<StateCapture> states;
      std::vector<LaneCapture> captures;
    };
    std::optional<KeptCaptures> keptCaptures;
    // The prompt from submit on, then the committed output; request.prompt
    // is empty.
    std::vector<uint32_t> exactTokens;
    // The first output token, sent to the client when the prefill selected it
    // (ModelStepResult::firstToken) and until the first decode cycle commits
    // it: that cycle emits it first, and its event leaves it out.
    std::optional<uint32_t> sentFirstToken;
    // Until its first tokens go out: what held it since the engine first saw
    // it (TtftPart, the rest left out), what holds it from `heldSince` on, as
    // the last tick left the engine (holds, holderOf), and whether they went
    // out (EngineEventSink::firstTokenParts).
    TtftParts held{};
    std::optional<double> heldSince;
    TtftPart heldBy = TtftPart::Rest;
    bool firstTokenSent = false;
    // Its last token event, and since then what a decode gap past the bound
    // may be due to (decodeGap): the prefill units that ran (their time),
    // whether its claim waited and a decode command of it read from the SSD,
    // and the pools' releases and the replans as they stood then.
    std::optional<double> lastTokenMilliseconds;
    struct GapSince final {
      double prefillMilliseconds = 0.0;
      bool claimWaited = false;
      bool ssdRead = false;
      uint64_t poolReleases = 0;
      uint64_t replans = 0;
    } gapSince;
    // Its model held an anchor when its lane was last suspended: the token
    // it selected last, which the suspension keeps (Model::suspend). A
    // decoding lane holds one once its first token is selected, a
    // constrained lane's under its first mask; a lane in prefill holds what
    // it held when its replay began.
    bool suspendedWithAnchor = false;
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
    // drains, or the command between its passes that holds it has run its
    // last or is given up (PassCommand); the first one stands (deferEnd).
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
    std::vector<std::vector<LaneCapture>> captures;
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
    // Preparing it changed what the plan was made from (PlanInputs): a lane
    // was refused or held back, a capture cut its rows, its history took
    // Index pages or it converted to Spill, or the command between its passes
    // was taken; the scheduler plans again.
    Replan,
  };
  double nextHealthCheckMilliseconds_ = 0.0;

  [[nodiscard]] Request &request(uint64_t requestId);
  [[nodiscard]] bool admitQueued(double nowMilliseconds);
  [[nodiscard]] bool admit(Request &request, double nowMilliseconds);
  // Whether the request competes for the SSD cache's quota of active KV
  // (Cache::spillRoom): a lane while it holds some of it by its own count,
  // copies another request holds too included (Cache::holdsActiveKvQuota);
  // a start if it would take some once it starts, as a Spill start's
  // commitment or the copies no request holds yet on the chain it restores
  // (Cache::restoreTakesQuota). A Resident start restoring only held copies
  // thus does not compete, and the lane it becomes does. Every request
  // competes for memory. A refused start holds back the wanted starts after
  // it that compete for what it was refused (admitQueued), and the lanes
  // admitted before its latest refusal that compete for it keep its wait
  // open while they work, as every lane submitted before it does (tick).
  [[nodiscard]] bool competesForQuota(const Request &request) const;
  // The tokens the request's KV is projected to hold: those it holds and
  // the output it may still generate under `outputLimit`, its own unless
  // given, at most kProjectedOutputTokens.
  [[nodiscard]] static uint64_t projectedTokens(const Request &request, uint32_t outputLimit) noexcept;
  [[nodiscard]] static uint64_t projectedTokens(const Request &request) noexcept {
    return projectedTokens(request, request.outputLimit);
  }
  // Where the KV of a request that starts now lives: on a target that can
  // spill, Spill only when its projection's Resident KV is more than its
  // Spill peak (its floor, and the larger of the staging its prompt's
  // history takes and its lane's hot block floor, priced as laneDemand
  // prices it), and memory for the Resident KV, less the pages in memory of
  // the chain it restores (Cache::residentChainPages), beside every lane's
  // projected growth (laneDemand) is not there for its start without the
  // expert store, as the governor finds it keeping that chain, the releases
  // of the Spill lanes of its priority or a lower one counted
  // (MemoryGovernor::available, TakeRights::lanes); for every request under
  // --kv-spill always; Resident otherwise. A start alone is priced so too,
  // keeping what only work in service takes, the pooled buffers the next
  // lane starts from among it, which Spill, taking less memory, leaves to
  // the requests that follow; a start the SSD cache cannot take has no such
  // choice and goes on as the claim its admission makes finds memory
  // (inMemory). Its outputs never depend on the mode. It projects the whole
  // output, which each attempt bounds afresh (admit): a bound an attempt
  // that did not start left counts for no placement. A placement holds for
  // the memory it was made in alone: planning places every candidate in
  // each pass (laneShapes) and each attempt places its start again
  // (admitQueued), since the room under the limit moves with the memory
  // outside the engine's buffers, which no change of the engine's own state
  // names.
  [[nodiscard]] KvMode placement(const Request &request) const;
  // What `tokens` of a start's Resident KV need now beside every lane's
  // projected growth, as placement prices it, and how much of it what `spec`
  // may take gives, keeping the chain the start restores; short of the
  // need, the tokens of KV that gives it.
  struct ResidentRoom final {
    uint64_t needed = 0;
    uint64_t available = 0;
    uint64_t heldTokens = 0;
    [[nodiscard]] bool holds() const noexcept { return available >= needed; }
  };
  [[nodiscard]] ResidentRoom residentRoom(const Request &request, const ClaimSpec &spec, uint64_t tokens) const;
  // The lanes' projected growth that has not happened yet: a Resident lane's
  // pages up to its projection beyond those it holds; a Spill lane's floor
  // up to its projection beyond its floor now, in prefill beyond its floor
  // at its next command's end, whose pages beyond those it holds count Full
  // (nextCommandPages); none for a lane that ended.
  struct LaneDemand final {
    uint32_t fullPages = 0;  // the Resident lanes' pages and the Spill lanes' next prefill commands'
    uint64_t bytes = 0;      // the Spill lanes' floors and the readers' hot block floors
  };
  [[nodiscard]] LaneDemand laneDemand() const;
  // laneDemand in bytes, the Full pages at a Full page's bytes.
  [[nodiscard]] uint64_t laneDemandBytes() const;
  // The highest priority among the lanes in service, none without one.
  [[nodiscard]] std::optional<RequestPriority> highestLanePriority() const;
  // The request's projection fits in memory alone, as it would without the
  // SSD cache (SpillGeometry::residentTokens).
  [[nodiscard]] bool fitsResidentAlone(const Request &request) const;
  // The last block of a chain of KV a request holds: the one a lookup
  // matched, which it restores, or the one it holds published as its lane
  // runs, its prompt's progress while it prefills (a replay publishes its
  // own blocks again) and what it published once it decodes; 0 for none.
  [[nodiscard]] static uint64_t chainEnd(const CacheLookup &lookup) noexcept;
  [[nodiscard]] uint64_t chainEnd(const Request &request) const;
  // The full pages of a Spill request's projection that its commitment
  // covers (Cache::commitSpill): all of them but the `copied` blocks of the
  // chain it holds that have a copy (Cache::chainCopies), whose pages it
  // writes no more. Each of its other pages is written once.
  [[nodiscard]] uint32_t spillPages(const Request &request, uint32_t copied) const;
  // What becomes of a start (or a resume, which keeps its mode) in Spill
  // mode that restores the chain ending at `chainEnd`, `spec` the claim its
  // admission makes. A fresh start needs a tier that takes writes and a
  // write budget for what Spill requests committed (Cache::spillGate);
  // refused either, it goes on in memory where memory allows (inMemory), and
  // is refused otherwise. Then the SSD cache's quota of active KV
  // (Cache::spillRoom) must hold the pages it publishes up to its
  // projection. Short of that, unless every request spills, a start goes on
  // in memory where memory allows; otherwise the quota bounds its output to
  // what it holds, if that leaves a token of it past what the request holds,
  // the anchor of a resume that holds one included; it waits while other
  // requests' active KV (the copies of the blocks they hold, and their
  // commitments) holds what it lacks; and otherwise a start is refused,
  // naming what the SSD cache has for it and, where memory was asked, the
  // memoryShortfall of its least start, and a resume ends at its length.
  enum class SpillAdmission : uint8_t {
    // The quota holds what it publishes: commit it once the lane is taken.
    Commit,
    // It starts Resident instead, under memory's bound where memory sets one.
    Resident,
    // It ended: refused, or a resume at its length.
    Settled,
    // It waits for the quota other requests' active KV holds, or for the
    // memory the lanes in service hold.
    Waiting,
  };
  [[nodiscard]] SpillAdmission admitSpill(Request &request, bool resuming, uint64_t chainEnd, const ClaimSpec &spec,
                                          double nowMilliseconds);
  // What memory does for a start the SSD cache cannot take whose projection
  // fits in memory alone (fitsResidentAlone): it starts Resident where the
  // claim its admission makes holds its Resident KV now beside the lanes'
  // projected growth (residentRoom), since a lane started where memory does
  // not hold it fails at a wall the engine already sees. Beside other lanes
  // that claim is a start's, whose test placed it Spill unless every request
  // spills: short of it, it waits for them (Waiting). Alone it is in
  // service, as its lane's growth is, which takes the pooled buffers the
  // next lane starts from too: short of it, the KV memory holds bounds its
  // output, as the quota bounds a Spill start's, where it holds the prompt
  // and a token past it with the rows the decode step verifies (`tokens`,
  // prompt and output); otherwise it goes on as a Spill start, `room`
  // naming what that least start needs and what memory has for it.
  struct InMemory final {
    std::optional<SpillAdmission> admission;
    std::optional<uint64_t> tokens;
    std::optional<ResidentRoom> room;
  };
  [[nodiscard]] InMemory inMemory(const Request &request, const ClaimSpec &spec) const;
  // Whether the lane's history may gain Index pages between a command's
  // passes: under --kv-spill always, in Spill mode, a prompt `processed`
  // rows in reaches past its dense history and tail window, where its pages
  // spill as their copies land (spillLanded runs before every pass). No
  // other lane does: a release takes no lane between passes.
  [[nodiscard]] bool gainsIndexPages(std::optional<KvMode> mode, uint64_t processed) const;
  // The shapes planning reads (PlanInputs::shapes): every resident prefill
  // lane's, with its Index pages, whether it gains them, its row cap this
  // tick, whether it reads spilled history and whether it ended
  // (Request::pendingEnd), and each candidate's, from its placement in the
  // memory as it is now.
  [[nodiscard]] std::vector<LaneShape> laneShapes(std::span<const PrefillAdmission> candidates = {}) const;
  // Whether a resident Spill request's prompt goes on: its next commands
  // take Full pages, written through, which under --kv-spill always spill
  // as their copies land.
  [[nodiscard]] bool spillingPrompt() const;
  // Whether a lane is in service.
  [[nodiscard]] bool laneInService() const;
  // What becomes of the Full extents the spill of pages whose copies landed
  // empties under --kv-spill always (Cache::spillWritten): all of them while
  // a lane that converted is in service, whose conversion was for those
  // bytes, or under critical pressure, where only those bytes pay for the
  // next Index extent; else they leave memory but the runway while a
  // spilling prompt still grows into it or the host holds growth back, when
  // it could not be allocated again; otherwise all of them.
  [[nodiscard]] SpilledExtents landedExtents() const;
  // Between commands, on a target whose KV can live on the SSD cache: the
  // Index runway a lane's release moves its pages into refills
  // (Cache::spillWritten), before admission, at a command's start and in the
  // control pass (collectLandings); under --kv-spill always the pages whose
  // copies have landed leave memory first. The Index pages are memory a
  // Spill lane in service needs (an InService claim of the engine's), and
  // with none in service an Optional one that takes only what costs nothing
  // (allocate()), which only the control pass asks: no command needs the
  // runway then, and a refusal would run at each. The Full extents the
  // moves release pay for them past the engine's limit and the host
  // (MemoryGovernor::fund).
  void spillLanded(bool controlPass);
  // Ends a request at its length outside a command: in Spill mode, once its
  // KV gives no more memory back, by the SSD cache or a release (refuse).
  void endAtLength(Request &request);
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
  [[nodiscard]] bool pollRestores();
  // The RAM copy of the state the request's restore read from disk, which
  // only saves the state's next read: an Optional claim of the request that
  // takes nothing landing later; nullptr when refused.
  [[nodiscard]] std::shared_ptr<const CompositeState> promotedCopy(Request &request, StateRestore &restore);
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
      Request &request, uint32_t promptProcessed, std::span<const LaneCapture> captures,
      std::span<const std::shared_ptr<const CompositeState>> states, double nowMilliseconds);
  // Takes what the plan's lanes need to run it as planned, and returns
  // Runnable with its items and captures; or, when that changed what the
  // plan was made from, Replan, the lanes refused added to `excluded`, a
  // capture cut's rows the lane's rowCap; or, with no lane admitted, Waiting
  // or Yielded (refuse). It never changes the plan. `supersedes`: the plan
  // supersedes the command between its passes (Scheduler::abandonsPasses),
  // whose claims may take it (ClaimSpec::supersedes).
  [[nodiscard]] Prepared prepare(const BatchPlan &plan, std::vector<ModelBatchItem> &items,
                                 std::vector<std::vector<LaneCapture>> &captures, std::vector<uint64_t> &excluded,
                                 bool supersedes, double nowMilliseconds);
  // Reserves the buffers of the states a prefill lane's command reaches,
  // boundary by boundary, and the KV up to the next one; returns the rows
  // the command keeps. It ends at the first boundary whose buffers or
  // following KV are refused, and at one a request waiting for a prefix
  // starts from. A checkpoint the command passes on its way to a state that
  // retires it once published is not captured, and that state takes
  // buffers of its own class; refused the KV up to that state, the command
  // ends at the checkpoint and captures it. The prompt's replay point
  // refused buffers of its own takes those of a checkpoint the command
  // captures before it. The KV past a boundary is an Optional claim: refused,
  // it cuts the command there, and the rows past it run in a second command.
  [[nodiscard]] uint32_t reserveCaptures(Request &request, const BatchItem &item,
                                         std::vector<StateCapture> &states,
                                         std::vector<LaneCapture> &captures, bool supersedes);
  // Takes the buffers of the state of class `kind` the lane's command
  // captures at `tokens`: from the model's pool or one admission, then what
  // the ledger frees for its claim, but only while the lane holds no
  // checkpoint a reclaim could take. The lane's checkpoint stays its
  // recovery point until a state that retires it is published, so no state
  // takes its buffers, or room a reclaim could make of it, before the
  // command: refused the pool and the admission while the lane holds one, a
  // state ends its command at its boundary, and apply takes it from the lane
  // after the command (publishReachedStateBoundaries). Only the first state
  // the command captures follows that checkpoint (followsCheckpoint): a
  // later one follows the state before it in the command. A checkpoint whose
  // predecessor a reader pins takes only the pool or the admission too:
  // refused, its command ends at it, and apply publishes it from the lane
  // only if the pin has gone by then. Null buffers when refused.
  [[nodiscard]] Reservation reserveCapture(Request &request, StateClass kind, uint32_t tokens,
                                           bool followsCheckpoint, bool supersedes);
  // The claim of a state's buffers of class `kind`: the prompt's replay
  // point, the state the request's next turn resumes from, is work in
  // service that may wait for memory on its way back and reaches what is in
  // use; a checkpoint, a junction or an ordinary state is Optional, within
  // what frees memory at once.
  [[nodiscard]] ClaimSpec stateClaim(const Request &request, StateClass kind, bool supersedes) const;
  // The victims that are the engine's decisions, in this order, each offered
  // only as the claim's take rights allow: the model's idle state buffers,
  // which cost nothing and are taken as they go (Model::releaseIdleBuffer),
  // of at least Model::idleBufferBytes of no evicted states, keeping one
  // lane's start unless the rights take it first (TakeRights::laneStart);
  // the command between its passes (TakeRights::passCommand), at no cost;
  // with no command in flight, the releases of the Spill lanes the claim's
  // rights reach (TakeRights::lanes), neither restoring, ended nor between
  // passes, the lowest-ranked lane's first: each lane's pages whose copies
  // landed (Cache::releasablePages) whose every holder is such a lane, off
  // the chain a Resident start restores, then for a claim that waits for
  // writes those whose copies land, which it releases once they have
  // (releases); the lane's conversion to Spill (TakeRights::convertLane)
  // while it is Resident, decoding or prefilling, not ended and not between
  // passes, on a target that can spill with the tier open and its quota
  // holding the lane's pages, and only where the pages it would spill free,
  // net of the first lane's hot block floor its decode reads, cover what the
  // claim lacks; its claim's retries release its pages as their copies land.
  // The command between its passes gives back the buffers of the states it
  // would capture, which the model's pool releases at once; its lanes that
  // ended are left to settleEndedBetweenPasses. Last, for in-service KV
  // (TakeRights::LaneStart::Last), the pooled buffers one lane starts from,
  // all the pool keeps released.
  class EngineVictims final : public VictimSource {
  public:
    explicit EngineVictims(Engine &engine) noexcept : engine_(engine) {}
    [[nodiscard]] std::string_view name() const noexcept override { return "engine"; }
    [[nodiscard]] std::optional<Victim> peek(const VictimQuery &query, const Victim *after) const override;
    FreeResult take(const VictimQuery &query, const Victim &victim) override;
    // A lane's release of the claim's Full pages alone, where the claim
    // counts one: the free Index pages it moves them into and the runway
    // past them (Cache::inPlaceIndexPages), which the cache's reclaims keep.
    [[nodiscard]] std::array<uint32_t, 3> movesInto(const VictimQuery &query) const override;

  private:
    // What the claim's releases may move: the pages of the Spill lanes its
    // rights reach (VictimQuery::reaches), neither restoring, ended nor
    // between their command's passes, off the chain a Resident start restores.
    [[nodiscard]] ReleaseRule releaseRule(const VictimQuery &query) const;
    // Each lane's units: for a need of Full pages, the claim's pages moved
    // in place, with the bytes of the extent more that keeps the Index runway
    // (Cache::inPlaceMoves), offered where the reads of the lanes they give
    // Index pages lack nothing, since what it frees is not sized for them;
    // then for a need of bytes the host did not refuse, whole extents of
    // pages and one more the free Full pages beside complete
    // (Cache::releaseBeside), as many as compaction can empty
    // (Cache::releaseMoves), each page freeing its Full bytes less its Index
    // page's, less what those reads take (readsShortfall), offered where
    // they free some; last, after every lane's, outside --kv-spill always,
    // for a claim that waits for writes, those whose copies land, which
    // leave it Pending (later).
    // The units for bytes count what their takes in turn free as the copies
    // stand, each page, each lane's first reads and each extent once
    // (ReleaseCount): copies that land before the takes can lower the
    // count, by 102 MB for two lanes on one conversation, as a lane's landed
    // pages then join, in its whole extents, shared pages a later lane's
    // unit counted. A key is the lane's request.
    [[nodiscard]] std::vector<Victim> releases(const VictimQuery &query) const;
    [[nodiscard]] std::vector<Victim> conversions(const VictimQuery &query) const;
    Engine &engine_;
    // The release and conversion units the pass over the order in progress
    // counted as it reached them (peek), so that a pass counts the releases
    // once, not once a unit.
    mutable std::vector<Victim> reached_;
  };
  // The governor whose ledger holds the requests' claims.
  [[nodiscard]] MemoryGovernor &governor() const;
  // The pages a request's KV admission leaves free for the waiting claims
  // ranked above its claim (MemoryLedger::heldPages); none before its claim
  // opens.
  [[nodiscard]] std::array<uint32_t, 3> heldPages(const Request &request) const;
  // Settles every lane that ended (Request::pendingEnd) that no command in
  // flight, restore or command between its passes holds any more: the lanes
  // of a command something took without settling them. It runs before every
  // plan, so a take that may not settle (a claim's, inside the ledger) leaves
  // them here and no lane's end depends on who took the command.
  void settleEndedBetweenPasses();
  // Settles those of `lanes`, a command between its passes gave up, that
  // ended.
  void settleEnded(std::span<const uint64_t> lanes);
  // The time to their first tokens of the requests that have not sent them:
  // holds(now) counts the time since the last tick
  // to what held each then; holderOf names what holds one as the engine is
  // now, by precedence: a prefill command in flight with its rows, its claim
  // waiting, a decode command, another prefill command, the rest. Before a
  // request's first tokens, firstTokens sends what held it.
  void holds(double nowMilliseconds) noexcept;
  [[nodiscard]] TtftPart holderOf(const Request &request) const;
  void firstTokens(Request &request);
  // A lane's token event at `now`: a gap past G since its last counts under
  // the first cause that held in it (DecodeGapCauses).
  void decodeGap(Request &request, const model::CommandSample &sample, double nowMilliseconds);
  // Memory a lane could not get, and what the engine knows about its return.
  struct Denial {
    metal::AllocationFailure allocationFailure = metal::AllocationFailure::None;
    // On its way back: the ledger found what transfers in flight or owed
    // return covers what was refused (FreeOutcome::Pending). The lane waits;
    // nobody yields.
    bool pending = false;
    // Not memory but the SSD cache's quota of active KV (ResourceWait).
    bool diskQuota = false;
    // Not an allocation but its projection beside the lanes' projected
    // growth (inMemory).
    bool projection = false;
    // The KV pages a refusal of pages lacked, of their form; none for the
    // model's memory.
    uint32_t pages = 0;
    kv::PageForm form = kv::PageForm::Full;
  };
  // An allocation's last attempt, and what the engine knows about the memory
  // it could not get.
  template <class Admission> struct Allocation final {
    Admission admission;
    Denial denial;
    // The ledger took memory for it (FreeOutcome::Met) before an attempt.
    bool freed = false;
  };
  // What a refused allocation leaves (allocate). Waits: the work it is for
  // waits for the memory (a lane's start, its KV, its reads), so a refusal
  // no memory on its way back ends is a failure the recovery drain waits
  // out (allocationFailed_). Settled: the caller goes on without it, which
  // records none: a command ends sooner, a state goes without RAM buffers,
  // landed Spill pages stay Full or the Index runway short. SettledOnce: as
  // Settled, in one attempt the ledger frees nothing for, the caller taking
  // buffers it holds instead (its lane's, a checkpoint's).
  enum class Refusal : uint8_t { Waits, Settled, SettledOnce };
  // Converts a Resident lane to Spill on a target whose KV can live on the
  // SSD cache (Cache::convertToSpill), committing its projection's pages
  // without a copy, the chain it holds or, at admission, the one it restores
  // (`restoring`) counted; false when it cannot (no SSD cache, the tier
  // closed, its quota short, or a lane already Spill) or must not: a lane
  // of the command between its passes, which reads all its lanes' history
  // from memory. Its claim's retries release its pages as their copies land
  // (EngineVictims::releases).
  [[nodiscard]] bool convertToSpill(Request &request);
  // What a lane does about memory it could not get. Pending memory returns
  // by itself: the lane waits (the ledger finds it, its own lanes' owed
  // copies among it). While other lanes hold memory or the host refuses it,
  // a running lane yields its memory and a lane being admitted waits. Alone,
  // with nothing on its way back, it fails.
  enum class Verdict : uint8_t { Wait, Yield, Fail };
  [[nodiscard]] Verdict judge(const Denial &denial, uint64_t requestId) const;
  [[nodiscard]] bool anotherResident(uint64_t requestId) const;
  // The prompt rows a lane in prefill has processed, the tokens a decoding
  // lane holds, or what the start of a request that runs neither would
  // restore (its admission probe's cached tokens; none before it is probed).
  [[nodiscard]] uint64_t completedTokens(const Request &request) const;
  // The output tokens the client was sent: the committed ones and a first
  // token that no cycle has committed yet (Request::sentFirstToken), as when
  // the request is suspended or cancelled before its first decode cycle.
  [[nodiscard]] static uint32_t generatedTokens(const Request &request) noexcept;
  // Who memory goes to first, the higher rank first: the higher priority,
  // then, at equal priority, a lane past its prompt (decoding, or waiting for
  // its next token's mask) before one in prefill, then the one with more
  // completed tokens, then the earlier submission.
  [[nodiscard]] ClaimRank rankOf(const Request &active) const;
  // The claim of the request's allocations of class `cls`: its rank and
  // priority, and for a lane in service the reach of its growth
  // (growthClass).
  [[nodiscard]] ClaimSpec claimFor(const Request &active, ClaimClass cls) const;
  // Whether lane a gives up its memory before lane b: its rank is lower
  // short of the submission order.
  [[nodiscard]] bool yieldsBefore(const Request &a, const Request &b) const;
  // The resident lane in prefill or decode that yields first when every
  // lane's growth fails (the lowest rank), or nullptr without one. prepare()
  // gates that lane's growth and suspends that lane; preemptBelow() suspends
  // it for a higher priority. For a lane of the command between its passes,
  // see yieldsEndedLanes, EngineVictims and preemptBelow.
  [[nodiscard]] const Request *laneToYield() const;
  [[nodiscard]] Request *laneToYield() {
    return const_cast<Request *>(std::as_const(*this).laneToYield());
  }
  // Runs one allocation in the claim `spec` describes: its
  // attempts run in the claim's scope, whose refusal records what it lacks
  // (MemoryGovernor::Scope); each refusal asks the ledger to free that
  // (MemoryGovernor::free), with what memory already allocated meets in
  // place, and runs again once free() met it. A refusal free() finds memory
  // on its way back for leaves the denial pending, and the claim waiting:
  // the request waits. A request's allocations are its one claim's
  // (Request::claim), the engine's own (requestId 0) a claim of this
  // allocation alone. `refusal` says what a refusal leaves (Refusal).
  template <class Attempt>
  [[nodiscard]] auto allocate(Attempt &&attempt, const ClaimSpec &spec, Refusal refusal)
      -> Allocation<std::invoke_result_t<Attempt &>>;
  // A lane of a command refused memory as prepare() shapes it, and the KV
  // its command takes (workEndOf).
  struct Denied final {
    uint64_t requestId = 0;
    uint64_t workEnd;
    Denial denial;
    // What it was refused: its KV target, or else the model's memory for
    // its reads of KV only the SSD cache holds, which this names.
    std::optional<TokenAdmission> kv;
    std::string_view memory;
  };
  // What prepare() asks memory for a command with: its kind, whether it
  // supersedes the command between its passes (Scheduler::abandonsPasses),
  // and when.
  struct PlanContext final {
    bool prefill = false;
    bool supersedes = false;
    double now = 0.0;
  };
  // The lane's KV up to workEnd for its command: none when granted, else
  // what was refused.
  [[nodiscard]] std::optional<Denied> growLane(Request &active, uint64_t workEnd, const PlanContext &plan);
  // The model's memory for the lane's reads of the KV only the SSD cache
  // holds (Model::reserveSpilledReads): a decoding lane's floor of the hot
  // block cache, or the staging of a prefill's history up to the command's
  // end, which takes nothing where the lane holds it.
  [[nodiscard]] std::optional<Denied> reserveReads(Request &active, const BatchItem &scheduled,
                                                   const PlanContext &plan);
  // What a command none of whose lanes runs does about the memory they were
  // refused (`denied`): waits for memory on its way back, makes the lane
  // that yields first yield, or, alone with nothing on its way back, ends
  // the lane refused.
  [[nodiscard]] Prepared refuse(std::span<const Denied> denied, const PlanContext &plan);
  // What a scheduled lane's growth reclaims up to. A scheduled lane is
  // resident: it grows as a request in service. The lane that yields first
  // if growth fails takes nothing in use, since its own suspension, not
  // another conversation's replay point, pays for it. While other lanes
  // fit, it waits for them instead, resident and denied each step, as a
  // start a resident lane holds back waits. Which lane that is is asked
  // where it matters: an allocation may give up the command between its
  // passes (EngineVictims).
  [[nodiscard]] ReclaimClass growthClass(const Request &active) const;
  // The KV a lane's command takes before it runs. A prefill command's KV
  // reaches its first state boundary first; what lies past it follows once
  // that state has its buffers (reserveCaptures).
  [[nodiscard]] static uint64_t workEndOf(const Request &active, const BatchItem &scheduled,
                                          bool prefill) noexcept;
  // Beside a lane of its command that ended, a lane gives that command up in
  // place of its suspension; the next tick plans again either way.
  [[nodiscard]] Prepared yieldLane(Request &lane, uint64_t workEnd,
                                   metal::AllocationFailure failure, double nowMilliseconds);
  // Takes a resident request's lane: its state and KV pages go back,
  // and it waits, for `reason`, until admission resumes it to replay its
  // history from the cache, a Resident lane with room for workEnd tokens of
  // KV (Request::resumeKvTargetTokens).
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
  // drain, or abandons the command between its passes that holds it if a
  // lane of that command ended, which that frees (yieldsEndedLanes). A lane
  // of that command, of a lower priority than the start, gives the command
  // up with its suspension, as the start's prompt would
  // (Scheduler::yieldsTo). The lane waits as
  // for a free one, and resumes once admission reaches it again, a Resident
  // one with its current KV capacity. True when one was suspended or freed.
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
  // A start or resume refused memory (`denial`) waits for it
  // (Request::refusedResources), and a start takes it from a lane of a lower
  // priority (preemptBelow), whose suspension makes it retry at the next
  // pass. Memory on its way back makes no lane yield: it waits for that. Nor
  // does a lane yield to a start refused for its projection
  // (Denial::projection) while one of its priority or above is in service:
  // it waits for that lane, which it cannot make yield, and a lower lane
  // suspended would wait behind it as long; with every lane lower, each
  // suspension brings it closer to alone, where it is decided (inMemory). A
  // start the SSD cache's quota of active KV refused (Denial::diskQuota) waits
  // for the requests whose active KV holds what it lacks to end, no lane
  // yielding, and holds back the starts the scheduler wants after it that
  // compete for the quota (competesForQuota, Request::refusedResources).
  void waitForResources(Request &request, double nowMilliseconds, const Denial &denial);
  // The latest refusal of the request's memory, as a wait or beside lanes
  // that run: what refused it, whether memory is on its way back, and when
  // memory on its way back was first awaited, which bounds how long its
  // claim holds room (syncClaims). Memory that came back since a refusal
  // that waited for it is progress, which restarts the limit of a wait that
  // has one.
  void noteRefusal(Request &request, double nowMilliseconds, const Denial &denial) noexcept;
  // Waiting for scheduling or for a prefix does not consume the memory
  // wait limit, so the wait's record is reset, and its claim stops waiting.
  // A request refused resources (Request::refusedResources) keeps when its
  // wait began, what refused it and which lanes were admitted before it was
  // refused.
  void deferWait(Request &request);
  // Keeps each request's claim waiting (MemoryLedger::setWaiting) while it
  // waits for memory on its way back and the engine retries it next: a lane
  // of a priority the scheduler plans (belowRunnable) that no plan of its
  // kind passed over (ResourceWait::passedOver), or one without a lane that
  // admission tries and holds behind no refusal, whose latest refusal free()
  // left pending (ResourceWait::pending), up to the resource wait limit from
  // when its refusals began waiting (ResourceWait::pendingSinceMilliseconds)
  // and within its wait's deadline (resourceDeadline). A waiting claim holds
  // room against claims ranked below it; past that limit or deadline it
  // holds none, whatever its retries find (ResourceWait::holdEnded); a wait
  // for the SSD cache's quota is no memory claim. One whose latest refusal
  // was the host's, retried within its deadline, counts in the host's
  // deficit (MemoryLedger::setHostRefusedWaiting).
  void syncClaims(double nowMilliseconds);
  // Before the plan's memory is decided, marks each lane of the plan's kind
  // that waits for memory on its way back and is ready to retry (not in
  // `excluded`) as passed over while the plan leaves it out, its claim no
  // longer waiting, and as retried while the plan holds it. Its wait's
  // record stays, so its claim's hold limit runs from its first refusal
  // that waited.
  void passOver(const BatchPlan &plan, std::span<const uint64_t> excluded);
  // What copies ahead are worth now (WriteBehind::run): none unless a
  // recent free() reached cache a copy would have let go at once
  // (MemoryGovernor::copyDemand), else the larger of twice the KV the
  // lanes' next commands add and what a pool's last growth lacked, one
  // growth unit at most (MemoryLedger::growthLackBytes).
  [[nodiscard]] uint64_t copyDemandBytes() const;
  // The pages of KV through what a lane's next command reaches: in prefill
  // as many rows as one command holds, within its prompt; in decode its
  // verify rows.
  [[nodiscard]] uint32_t nextCommandPages(const Request &request) const;
  // What the model's memory for the reads of spilled KV lacks once the lanes
  // hold Index pages: none for one that holds some already; else in prefill
  // its history's staging through its next command, in decode its floor of
  // the hot block cache, the floors counted once for them all.
  [[nodiscard]] uint64_t readsShortfall(std::span<const uint64_t> lanes) const;
  // The wait limit tick() enforces, or zero while it enforces none: a
  // pending wait that has seen progress waits for its next attempt. The
  // limit restarts whenever a lane submitted before the request, or one
  // ResourceWait::admittedBefore counts that competes for what it was
  // refused, has work in flight.
  [[nodiscard]] double resourceDeadline(const Request &request) const noexcept;
  void signalResourceProgress() noexcept;
  void apply(const Pending &command, std::span<const ModelStepResult> results,
             double cycleMilliseconds, double nowMilliseconds);
  // Whether the command in flight holds the request's lane.
  [[nodiscard]] bool inFlight(uint64_t requestId) const;
  // Where `lane`, the lane that would yield, is one of the command between
  // its passes beside a lane of it that ended, gives that command up in its
  // place, which ends the lanes that ended and gives back their lanes and
  // memory and the command's captures, and returns true, for the caller to
  // try again at its next attempt; a lane that ended is never suspended.
  // Every point a lane yields asks it first.
  [[nodiscard]] bool yieldsEndedLanes(const Request &lane);
  // Records the outcome that ends the lane once the work that holds it
  // drains (Request::pendingEnd); the first one stands.
  static void deferEnd(Request &request, LaneEnd end);
  // Ends a request early. While its command or restore runs, the first end
  // waits there (the command's mask wait is abandoned, the restore's read
  // cancelled) and apply()/pollRestores() settle it once that work drains;
  // so it does while the command between its passes holds it
  // (PassCommand::laneEnded). Otherwise the request ends now: Cancelled and
  // Stopped complete it (as Cancelled and Stop), any other outcome fails it.
  // A lane of either command ends at its deadline so (tick).
  void settle(Request &request, LaneEnd end);
  [[nodiscard]] static LaneEnd deadlineEnd();
  // What says that memory does not hold a request, in every end for want
  // of memory: its lane's KV mode, phase and tokens, which tell the limit it
  // met, the Metal memory outside the server's buffers, which the limit
  // charges beside it (MemoryGovernor::untrackedBytes), and where admission
  // priced its Resident KV (`room`, the mode then Resident) the bytes that
  // needs and those memory has for it.
  [[nodiscard]] std::string memoryShortfall(const Request &request,
                                            const std::optional<ResidentRoom> &room = {}) const;
  // The end of a request that cannot fit even alone: what it could not
  // allocate and why, and its memoryShortfall.
  [[nodiscard]] LaneEnd capacityExhausted(const Request &request,
                                          std::string_view what,
                                          metal::AllocationFailure failure,
                                          std::string_view detail = {}) const;
  void finish(Request &request, EngineFinishReason reason,
              std::span<const float> optionLogits);
  void finishFailure(Request &request, LaneEnd end);
  // Gives back what a lane holds: the command between its passes that
  // holds it, given up, its planned state boundaries, its
  // state (keepContinuation keeps the model's host continuation of a
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
  // Removes the finalized requests; the idle period begins once none is left.
  void sweepTerminal(double nowMilliseconds);
  // Takes back the next part of the weights for the requests waiting, or
  // ends a restore no request waits for (tick); true when it progressed.
  [[nodiscard]] bool advanceWeights(double nowMilliseconds);

  EngineConfig config_;
  std::optional<WeightResidency> weights_;
  Cache &cache_;
  WriteBehind writeBehind_;
  model::Model &model_;
  EngineEventSink &events_;
  Scheduler scheduler_;
  std::unordered_map<uint64_t, Request> requests_;
  // Pressure preempted work, a lane is still resident, and memory is still
  // short (the host admits no growth, or allocationFailed_), up to the
  // drain's end.
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
  // Whether the request is below the highest priority that prefills or
  // decodes: the scheduler plans none of its work, and admission starts it
  // in no lane.
  [[nodiscard]] bool belowRunnable(const Request &request) const;
  std::function<void()> completionNotifier_;
  std::optional<Pending> pending_;
  PassCommand passCommand_{scheduler_, [this](uint64_t id) { return requests_.at(id).pendingEnd.has_value(); }};
  // When the latest command retired, until a tick finds nothing to do.
  std::optional<double> busySinceMilliseconds_;
  // The lanes admit() has obtained so far, including those it gave back when
  // the attempt's pages were refused (Request::admission).
  uint64_t admissions_ = 0;
  uint64_t resourceEpoch_ = 1;
  // The resource wait limit after the latest suspension; zero once passed
  // or when no request is suspended.
  double drainEndMilliseconds_ = 0.0;
  // An allocation work waits on (Refusal::Waits) failed since the latest
  // suspension or since a resident lane last released its memory, or the
  // suspension itself met a limit that only freed memory lifts, unlike a
  // host pause.
  bool allocationFailed_ = false;
  EngineSnapshot counters_;
  // EngineSnapshot::tickHostMilliseconds.
  model::LatencyHistogram tickHosts_;
  EngineVictims victims_{*this};
};

} // namespace splash::engine

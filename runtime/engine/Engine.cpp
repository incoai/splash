#include "engine/Engine.hpp"
#include "AwakeClock.hpp"
#include "Supervision.hpp"
#include "engine/EngineAllocate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace splash::engine {
namespace {

// While a command is in flight the loop wakes at least this often to run the
// command watchdog (Model::checkHealth), so a command the backend gives up on
// fails the engine within a second of its timeout, as a ticket's own wait
// finds it (MetalBackend.mm kTicketWaitSlice).
constexpr double kHealthCheckIntervalMilliseconds = 1000.0;
// A mask request the server leaves unanswered for supervision::kMaskWait
// fails its request; the batch's command slot is not held longer.
constexpr double kMaskWaitMilliseconds = double(supervision::kMaskWait.count());
// A junction costs a cached state, its capture and up to a draft window of
// draft-context rows; a later request must save at least that much prefill.
constexpr uint32_t kMinimumJunctionGain = model::ExecutionLimits::draftContextTokens;

constexpr std::string_view kKvUnreadableMessage =
    "KV of an earlier part of the conversation could not be read from the SSD cache";
constexpr std::string_view kDiskQuotaTimeoutMessage =
    "the SSD cache's room for active KV did not become available within the resource wait "
    "limit: other long requests hold it";

// `item` reads the page table its lane's cut gave it (Cache::cutPageTable).
void setPageTable(ModelBatchItem &item, PageTableView pageTable) {
  item.pageTable = pageTable.pages;
  item.pageTableRevision = pageTable.revision;
  item.pageTableFirstChanged = pageTable.firstChanged;
  item.pageSources = std::move(pageTable.sources);
}

} // namespace

std::string resourceTimeoutMessage(metal::AllocationFailure failure) {
  std::string message = "memory did not become available within the resource wait limit";
  if (failure == metal::AllocationFailure::HostPressure)
    message += ": macOS is short of memory; close memory-heavy applications";
  return message;
}

Engine::Engine(EngineConfig config, Cache &cache, model::Model &model,
               EngineEventSink &events)
    : config_(config), cache_(cache), writeBehind_(cache), model_(model), events_(events),
      scheduler_({.decodeShare = config_.decodeShare,
                  .decodeGapMilliseconds = config_.decodeGapMilliseconds,
                  .limits = model.commandLimits()},
                 model) {
  if (!config_.maxContext || !config_.vocabularySize) {
    throw std::invalid_argument("context and vocabulary sizes must be positive");
  }
  if (!config_.governor)
    throw std::invalid_argument("the engine needs its memory governor");
  if (!std::isfinite(config_.decodeShare) || config_.decodeShare < 0.0)
    throw std::invalid_argument("decode share must be nonnegative and finite");
  if (config_.spill.capable &&
      (!config_.spill.geometry.fullPageBytes || !config_.spill.geometry.slotBytes))
    throw std::invalid_argument("a target that can spill needs its page and slot sizes");
  if (config_.weights)
    weights_.emplace(*config_.weights);
  // What the engine holds is the ledger's to take, cache before the
  // engine's own victims (the tie order of the sources), the cache tells it
  // what transfers return and what its pages hold, and the model what the
  // state buffers its units free release once they leave the pool: the
  // cache's and the model's, which outlive the engine, as another engine
  // over them (backend_benchmark's) finds them.
  MemoryLedger &ledger = config_.governor->ledger();
  ledger.addSource(cache_);
  ledger.addSource(victims_);
  ledger.setReturning([&cache = cache_] { return cache.returning(); },
                      [&cache = cache_](std::span<const uint64_t> requestIds) { return cache.returningFor(requestIds); });
  ledger.setBufferBytes([&model = model_](uint32_t states, bool afterStart, bool keepLane) {
    return model.idleBufferBytes(states, afterStart, keepLane);
  });
  ledger.setPageBytes({cache_.bytesPerPage(kv::PageForm::Full), cache_.bytesPerPage(kv::PageForm::Index),
                       cache_.bytesPerPage(kv::PageForm::Slim)});
}

Engine::~Engine() { config_.governor->ledger().removeSource(victims_); }

void Engine::submit(EngineRequest value) {
  const auto outOfVocabulary = [&](uint32_t token) {
    return token >= config_.vocabularySize;
  };
  if (value.prompt.size() + value.maxNewTokens > config_.maxContext) {
    throw std::invalid_argument("request exceeds the context window");
  }
  if (std::any_of(value.prompt.begin(), value.prompt.end(), outOfVocabulary)) {
    throw std::invalid_argument("prompt token is out of vocabulary");
  }
  if (std::any_of(value.scoreTokens.begin(), value.scoreTokens.end(),
                  outOfVocabulary)) {
    throw std::invalid_argument("score token is out of vocabulary");
  }
  if (value.thinkingBudget) {
    if (const char *reason = value.thinkingBudget->unusable())
      throw std::invalid_argument(reason);
    if (std::any_of(value.thinkingBudget->end.begin(),
                    value.thinkingBudget->end.end(), outOfVocabulary) ||
        outOfVocabulary(value.thinkingBudget->alsoEnds))
      throw std::invalid_argument("thinking end token is out of vocabulary");
  }
  if (!value.images.empty() && !config_.maxImagePatches) {
    throw std::invalid_argument("this model is serving without vision");
  }
  // The protocol bounds an image by the most patches the vision encoder
  // takes; the server's pixel cap may allow fewer.
  if (std::any_of(value.images.begin(), value.images.end(),
                  [&](const ImageSpan &image) {
                    return image.grid().patches() > config_.maxImagePatches;
                  })) {
    throw std::invalid_argument(
        "image has more patches than the server's pixel cap allows");
  }
  const uint64_t id = value.id;
  // A request the protocol saw end may still be here, finalized, until the
  // next sweepTerminal; its id is free again.
  if (const auto found = requests_.find(id); found != requests_.end()) {
    if (!found->second.finalized)
      throw std::logic_error("request id is live in the engine");
    scheduler_.remove(id);
    requests_.erase(found);
  }
  Request requestState;
  requestState.exactTokens = std::move(value.prompt);
  requestState.promptTokens = static_cast<uint32_t>(requestState.exactTokens.size());
  requestState.replayTokens = requestState.promptTokens;
  requestState.outputLimit = value.maxNewTokens;
  requestState.request = std::move(value);
  const Request &stored =
      requests_.emplace(id, std::move(requestState)).first->second;
  scheduler_.submit({.id = id, .priority = stored.request.priority, .prefillTokens = stored.promptTokens});
  ++counters_.submitted;
}

void Engine::cancel(uint64_t id) {
  if (const auto found = requests_.find(id); found != requests_.end())
    settle(found->second, {LaneOutcome::Cancelled, {}});
}

void Engine::stop(uint64_t id) {
  // A request that has generated nothing has nothing to stop on: it is cancelled.
  if (const auto found = requests_.find(id); found != requests_.end())
    settle(found->second,
           {generatedTokens(found->second) ? LaneOutcome::Stopped : LaneOutcome::Cancelled, {}});
}

void Engine::failRequest(uint64_t id, LaneOutcome outcome, std::string message) {
  settle(request(id), {outcome, std::move(message)});
}

void Engine::provideMask(uint64_t id, std::span<const uint32_t> words) {
  Request &active = request(id);
  const bool ownedByActiveBatch =
      pending_ && pending_->ticket->ownsMaskWait(id);
  // A request that already left its mask wait (cancellation, deadline, or a
  // failure raced the frontend) treats the response as stale.
  if (active.finalized || active.pendingEnd ||
      (!ownedByActiveBatch && scheduler_.phase(id) != Phase::WaitingMask)) {
    return;
  }
  if (std::optional<std::string> rejected = model_.provideMask(id, words)) {
    settle(active, {LaneOutcome::InvalidMask, std::move(*rejected)});
    return;
  }
  active.maskRequestedMilliseconds.reset();
  if (!ownedByActiveBatch)
    scheduler_.maskReady(id);
}

void Engine::setCompletionNotifier(std::function<void()> notifier) {
  completionNotifier_ = std::move(notifier);
  cache_.setCompletionNotifier(completionNotifier_);
  if (weights_)
    weights_->setCompletionNotifier(completionNotifier_);
}

bool Engine::flushRestorePoints() {
  // The restore points wait for a part of the weights being written, which
  // may run commands beside them, as a tick does.
  if (weightsWriting())
    return false;
  static_cast<void>(cache_.pollTransfers());
  return writeBehind_.flush();
}

bool Engine::tick(double now) {
  // The time since the last tick goes to what held each request then;
  // however the tick ends, what holds each from now on is as it leaves the
  // engine.
  holds(now);
  struct Holders final {
    Engine &engine;
    ~Holders() {
      for (auto &[_, active] : engine.requests_)
        if (!active.finalized && !active.firstTokenSent)
          active.heldBy = engine.holderOf(active);
    }
  } const holders{*this};
  model_.checkHealth();
  nextHealthCheckMilliseconds_ = now + kHealthCheckIntervalMilliseconds;
  // A request's deadline ends it in whatever phase it is: at once, or once
  // its command in flight or the read of the state it restores has drained
  // (settle), before a read that landed can admit it (pollRestores).
  bool progressed = false;
  for (auto &[_, active] : requests_) {
    if (!active.finalized && !active.pendingEnd && active.request.deadlineMilliseconds <= now) {
      settle(active, deadlineEnd());
      progressed = true;
    }
  }
  if (now >= drainEndMilliseconds_)
    drainEndMilliseconds_ = 0.0;
  // A part of the weights being written may allocate and run commands on the
  // backend, which takes them from one thread at a time, and a KV extent's
  // release or copy refuses any command in flight (kv::PageStorage): the rest
  // of the tick waits for the write's end, which wakes the loop, and so does
  // the control pass (commandInFlight). Frames are answered meanwhile.
  if (weightsWriting()) {
    sweepTerminal(now);
    return progressed;
  }
  if (cache_.pollTransfers()) {
    // Demoted pages and written states are back; waiting lanes retry now.
    // The tier's share for Spill requests' writes has room again.
    signalResourceProgress();
    progressed = true;
    if (config_.spill.capable)
      static_cast<void>(cache_.writeThrough());
  }
  progressed = pollRestores() || progressed;
  // The claims that wait for memory on its way back hold room for it; copies
  // ahead wait for them (WriteBehind::run).
  syncClaims(now);
  progressed = writeBehind_.run(now, copyDemandBytes(), governor().claimWaiting()) || progressed;
  // The earliest submitted of the lanes with work in flight, and the
  // earliest admitted of them and, while a request waits for the SSD cache's
  // quota, of those that compete for it (competesForQuota).
  const bool quotaWaits = std::ranges::any_of(
      requests_, [](const auto &entry) { return entry.second.resourceWait.diskQuota; });
  uint64_t earliestWorking = std::numeric_limits<uint64_t>::max();
  uint64_t earliestWorkingAdmission = std::numeric_limits<uint64_t>::max();
  uint64_t earliestQuotaAdmission = std::numeric_limits<uint64_t>::max();
  if (pending_) {
    for (const BatchItem &item : pending_->plan.items) {
      const Request &lane = request(item.requestId);
      earliestWorking =
          std::min(earliestWorking, scheduler_.submissionOrder(item.requestId));
      earliestWorkingAdmission = std::min(earliestWorkingAdmission, lane.admission);
      if (quotaWaits && competesForQuota(lane))
        earliestQuotaAdmission = std::min(earliestQuotaAdmission, lane.admission);
    }
  }
  const bool draining = drainingForRecovery();
  for (auto &[id, active] : requests_) {
    // A mask request left unanswered past the limit ends its request: a lane
    // in flight commits without the mask, any other fails now.
    if (!active.pendingEnd && active.maskRequestedMilliseconds &&
        now >= *active.maskRequestedMilliseconds + kMaskWaitMilliseconds) {
      settle(active, {LaneOutcome::MaskTimeout,
                      "the server did not answer a token-mask request within " +
                          std::to_string(supervision::kMaskWait.count()) + " ms"});
      progressed = true;
    }
    // Admission is deliberately paused while resident peers finish. Start a
    // fresh resource wait only if admission still fails after that drain.
    if (draining) {
      active.resourceWait.deadlineMilliseconds = 0.0;
      continue;
    }
    // While a lane with work in flight was submitted before the request,
    // whatever it holds, or was admitted before the refusal
    // ResourceWait::admittedBefore records (the latest refusal, or the
    // suspension, of a request without a lane; a lane in service's first)
    // and competes for what it was refused, the wait's limit restarts. For the quota a lane
    // competes by its own count (competesForQuota), copies another lane
    // holds too included. The rule is wide on purpose: such a lane may still
    // return what the request needs, and a refused Spill start needs memory
    // too, as a later attempt may place it Resident; narrower rules
    // failed requests at the limit that this one completes. The wait fails
    // once no such lane has worked for a whole limit; the request's own
    // deadline applies throughout (above).
    const ResourceWait &wait = active.resourceWait;
    const uint64_t earliestAhead = wait.diskQuota ? earliestQuotaAdmission : earliestWorkingAdmission;
    if (scheduler_.submissionOrder(id) > earliestWorking ||
        (wait.admittedBefore && earliestAhead <= *wait.admittedBefore))
      active.resourceWait.earlierLaneWorkMilliseconds = now;
    // A request waiting for the weights fails at its limit only once the
    // restore's attempt there is refused too (advanceWeights).
    const double deadline = resourceDeadline(active);
    if (!active.finalized && weightsHeld() && deadline > 0.0 && now >= deadline) {
      settle(active, {LaneOutcome::ResourceTimeout,
                      active.resourceWait.diskQuota
                          ? std::string(kDiskQuotaTimeoutMessage)
                          : resourceTimeoutMessage(active.resourceWait.allocationFailure)});
      progressed = true;
    }
  }
  // The requests the checks above ended leave the engine and the scheduler
  // now, even while a command is in flight.
  if (progressed)
    sweepTerminal(now);
  if (pending_) {
    // A mask response, cancellation, or deadline can make the commit tail
    // runnable without another Metal completion wake.
    pending_->ticket->advance();
    for (ModelMaskRequest &asked : pending_->ticket->takeMaskRequests()) {
      events_.maskRequested(asked.requestId, asked.simulationTokens);
      request(asked.requestId).maskRequestedMilliseconds = now;
      progressed = true;
    }
    if (!pending_->ticket->ready())
      return progressed;
    Pending command = std::move(*pending_);
    pending_.reset();
    std::vector<ModelStepResult> results = command.ticket->wait();
    apply(command, results, now - command.startedMilliseconds, now);
    busySinceMilliseconds_ = now;
    sweepTerminal(now);
    return true;
  }

  // No request runs until the weights are back: a tick takes back a part,
  // so that frames are answered between them.
  if (!weightsHeld()) {
    progressed = advanceWeights(now) || progressed;
    if (!weightsHeld()) {
      sweepTerminal(now);
      return progressed;
    }
  }
  // The Index runway refills before admission judges memory, and under
  // --kv-spill always the pages whose copies landed leave first.
  spillLanded(false);
  // The tick's host time runs from here, its first planning call, to the
  // submission of the command it plans (tickHostMilliseconds).
  const AwakeClock::time_point planning = AwakeClock::now();
  progressed = admitQueued(now) || progressed;
  sweepTerminal(now);
  // Lanes that wait for memory on its way back cannot run before it lands;
  // the command runs the other lanes meanwhile. A plan whose lanes all wait
  // adds them, and each plan prepare sends back adds those it refused or held
  // back, so each attempt plans without at least one more lane, or with a
  // lane's rows cut at a state boundary, or after a lane's one conversion.
  std::vector<uint64_t> excluded, refused;
  for (const auto &[id, active] : requests_)
    if (active.lane && active.resourceWait.pending && !resourceRetryReady(active, now))
      excluded.push_back(id);
  // However the tick ends, the states plans sent back kept for their lanes
  // go back, and the lanes' row caps clear: the next tick's admission plans
  // each lane's command uncut, as its planning does.
  struct TickEnd final {
    Engine &engine;
    ~TickEnd() {
      for (auto &[_, active] : engine.requests_) {
        active.keptCaptures.reset();
        active.rowCap.reset();
      }
    }
  } const tickEnd{*this};
  for (;;) {
    settleEndedBetweenPasses();
    // Planning reads the lanes' shapes as they are now, before every plan.
    const std::vector<LaneShape> shapes = laneShapes();
    const PlanInputs in{.excluded = excluded, .shapes = shapes, .passes = passCommand_.progress()};
    const std::optional<BatchPlan> plan = scheduler_.next(in);
    if (!plan)
      break;
    // A command between its passes beside a lane of it that ended (a cancel,
    // a deadline) gives way, whole, to a plan of a higher priority than its
    // own, as at every point a lane yields (yieldsEndedLanes), and the tick
    // ends there: the lane's end and its memory wait for no pass a higher
    // priority's work may hold off. Beside a plan of its priority, such as
    // the decode that follows each of its passes, or a lower one, its next
    // pass comes after that plan: the command goes on, its other lanes
    // keeping the passes they ran, and the lane ends at its last pass.
    if (passCommand_.anyEnded() && !plan->pass &&
        scheduler_.yieldsTo(requests_.at(plan->items.front().requestId).request.priority, in)) {
      settleEnded(passCommand_.yield());
      sweepTerminal(now);
      return true;
    }
    // A command between its passes yields, whole, to a prefill plan of a
    // higher priority (Scheduler::abandonsPasses), which holds none of its
    // lanes. It is abandoned once that plan runs, so that the command goes
    // on while that one waits for memory.
    const bool supersedes = scheduler_.abandonsPasses(*plan, in);
    passOver(*plan, excluded);
    std::vector<ModelBatchItem> items;
    std::vector<std::vector<LaneCapture>> captures;
    const size_t planned = excluded.size();
    const Prepared prepared = prepare(*plan, items, captures, excluded, supersedes, now);
    switch (prepared) {
    case Prepared::Runnable: {
      // Unless prepare gave the command up already: a claim for it took the
      // command (EngineVictims), or a lane yielded beside one of it that
      // ended (yieldsEndedLanes).
      if (supersedes)
        settleEnded(passCommand_.yield());
      tickHosts_.add(static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(AwakeClock::now() - planning).count()));
      std::unique_ptr<ModelBatchTicket> ticket =
          model_.submit(*plan, items, completionNotifier_);
      if (!ticket) {
        throw std::logic_error("model returned an empty command ticket");
      }
      // The lanes prepare refused or held back were planned without while
      // ready: a command in passes leaves them out, as it does a lane it had
      // no room for, and they wait for its end.
      // Those that wait for memory on its way back join it once that lands.
      std::vector<uint64_t> waiting;
      for (const uint64_t id : excluded)
        if (std::ranges::find(refused, id) == refused.end())
          waiting.push_back(id);
      const std::vector<LaneShape> committed = laneShapes();
      scheduler_.commit(*plan, {.excluded = waiting, .shapes = committed, .passes = passCommand_.progress()});
      // A command in passes keeps its items and their captures for its
      // later passes (prepare), and its progress for planning.
      if (plan->passes > 1 && !plan->pass)
        passCommand_.start(*plan, std::move(items), captures);
      // The command reads the cached pages its lanes' blocks were swapped
      // for, and those a resume restored. Selecting a constrained request's
      // first token reads none, but the cycle after it does; only a first
      // token that stops the request counts pages nothing read.
      for (const BatchItem &item : plan->items) {
        Request &lane = request(item.requestId);
        lane.deduplicatedKvTokens += lane.unreadDeduplicatedKvTokens;
        counters_.deduplicatedKvTokens += std::exchange(lane.unreadDeduplicatedKvTokens, 0);
      }
      pending_ = Pending{*plan, std::move(captures), std::move(ticket), busySinceMilliseconds_.value_or(now)};
      return true;
    }
    case Prepared::Yielded:
      sweepTerminal(now);
      return true;
    case Prepared::Waiting:
      for (const BatchItem &item : plan->items)
        excluded.push_back(item.requestId);
      continue;
    case Prepared::Replan:
      refused.insert(refused.end(), excluded.begin() + static_cast<std::ptrdiff_t>(planned), excluded.end());
      ++counters_.replans;
      continue;
    }
  }
  if (!progressed)
    busySinceMilliseconds_.reset();
  return progressed;
}

bool Engine::drainingForRecovery() const {
  return drainEndMilliseconds_ > 0.0 &&
         std::any_of(requests_.begin(), requests_.end(),
                     [](const auto &entry) {
                       return entry.second.lane.has_value();
                     }) &&
         (allocationFailed_ || !governor().admits(ClaimClass::Growth));
}

std::optional<RequestPriority> Engine::suspendedTier() const {
  std::optional<RequestPriority> tier;
  for (const auto &[id, active] : requests_) {
    // A finalized request keeps the scheduler's flag until it is swept.
    if (!active.finalized && scheduler_.suspended(id) &&
        (!tier || active.request.priority < *tier))
      tier = active.request.priority;
  }
  return tier;
}

bool Engine::admissionTries(const Request &active, std::optional<RequestPriority> tier,
                            bool draining) const {
  if (!tier)
    return !draining;
  return active.request.priority < *tier ||
         (scheduler_.suspended(active.request.id) && !draining);
}

bool Engine::belowRunnable(const Request &active) const {
  const std::optional<RequestPriority> runnable = scheduler_.highestRunnablePriority();
  return runnable && active.request.priority > *runnable;
}

void Engine::holds(double now) noexcept {
  for (auto &[_, active] : requests_) {
    if (active.finalized || active.firstTokenSent)
      continue;
    if (active.heldSince && active.heldBy != TtftPart::Rest)
      active.held[static_cast<size_t>(active.heldBy)] += now - *active.heldSince;
    active.heldSince = now;
  }
}

TtftPart Engine::holderOf(const Request &active) const {
  const bool own = pending_ && std::ranges::any_of(pending_->plan.items, [&](const BatchItem &item) {
                     return item.requestId == active.request.id;
                   });
  if (own && pending_->plan.kind == WorkKind::Prefill)
    return TtftPart::OwnPrefill;
  if (active.claim.id) {
    const ClaimState &claim = governor().ledger().claim(active.claim.id);
    if (claim.waiting || claim.hostRefusedWaiting)
      return TtftPart::ClaimWait;
  }
  if (pending_)
    return pending_->plan.kind == WorkKind::Decode ? TtftPart::Decode : TtftPart::OtherPrefill;
  return TtftPart::Rest;
}

void Engine::firstTokens(Request &active) {
  if (std::exchange(active.firstTokenSent, true))
    return;
  events_.firstTokenParts(active.request.id, active.held);
}

void Engine::decodeGap(Request &active, const model::CommandSample &sample, double now) {
  Request::GapSince &since = active.gapSince;
  // The command read from the SSD, for every lane of it alike: the experts
  // the store missed, or spilled K/V (the hot blocks a decode missed). A
  // command whose lanes found every hot block in memory read nothing.
  if (sample.work.weightReadBytes || sample.work.spilledKvReadBytes)
    since.ssdRead = true;
  const uint64_t releases = governor().ledger().poolReleases();
  if (active.lastTokenMilliseconds && now - *active.lastTokenMilliseconds > config_.decodeGapMilliseconds) {
    const GapCause cause = since.prefillMilliseconds > 0.0   ? GapCause::PrefillUnit
                           : since.claimWaited               ? GapCause::ClaimWait
                           : since.ssdRead                   ? GapCause::SsdRead
                           : releases != since.poolReleases  ? GapCause::PoolRelease
                           : counters_.replans != since.replans ? GapCause::Replan
                                                                : GapCause::Other;
    DecodeGapCausesSnapshot &causes = counters_.decodeGapCauses;
    ++causes.gaps[static_cast<size_t>(cause)];
    causes.milliseconds[static_cast<size_t>(cause)] += now - *active.lastTokenMilliseconds;
    if (cause == GapCause::PrefillUnit) {
      causes.prefillMilliseconds += since.prefillMilliseconds;
    }
  }
  active.lastTokenMilliseconds = now;
  since = {.poolReleases = releases, .replans = counters_.replans};
}

std::optional<double> Engine::nextWakeupMilliseconds() const {
  std::optional<double> result;
  // While a part of the weights is written, a tick only ends requests at
  // their deadlines (tick): the write's end wakes the loop for the rest.
  const bool writing = weightsWriting();
  const bool draining = drainingForRecovery();
  if (!writing) {
    if (pending_)
      result = nextHealthCheckMilliseconds_;
    if (draining && (!result || drainEndMilliseconds_ < *result))
      result = drainEndMilliseconds_;
    // Copies ahead wait while a claim waits, which a step of the engine shows.
    if (const std::optional<double> persist = writeBehind_.nextWakeup(); persist && (!result || *persist < *result))
      result = persist;
  }
  // Admission retries run only between commands, and only for the requests
  // admission tries; those held behind one refused memory have no retry time
  // (admitQueued). Other retry times would wake the loop with nothing to do;
  // the command completion or resumption wakes it instead.
  const std::optional<RequestPriority> tier = suspendedTier();
  for (const auto &[_, active] : requests_) {
    if (active.finalized || active.pendingEnd)
      continue;
    if (!result || active.request.deadlineMilliseconds < *result)
      result = active.request.deadlineMilliseconds;
    if (writing)
      continue;
    if (active.maskRequestedMilliseconds) {
      const double limit =
          *active.maskRequestedMilliseconds + kMaskWaitMilliseconds;
      if (!result || limit < *result)
        result = limit;
    }
    const double deadline = resourceDeadline(active);
    if (!draining && deadline > 0.0 && (!result || deadline < *result))
      result = deadline;
    if (pending_ || !admissionTries(active, tier, draining) ||
        active.resourceWait.retryMilliseconds <= 0.0)
      continue;
    const double wakeup = active.resourceWait.epoch == resourceEpoch_
                              ? active.resourceWait.retryMilliseconds
                              : 0.0;
    if (!result || wakeup < *result)
      result = wakeup;
  }
  return result;
}

EngineSnapshot Engine::snapshot() const {
  EngineSnapshot result = counters_;
  result.maximumContextTokens = config_.maxContext;
  result.scheduler = scheduler_.snapshot();
  result.resources = cache_.snapshot();
  result.writeBehind = writeBehind_.snapshot();
  result.tickHostMilliseconds = tickHosts_.snapshot();
  for (const auto &[id, active] : requests_) {
    if (active.finalized)
      continue;
    const Phase phase = scheduler_.phase(id);
    result.activeRequests.push_back(
        {.id = id,
         .phase = phase,
         .priority = active.request.priority,
         .promptTokens = active.promptTokens,
         .promptProcessed =
             std::min(scheduler_.promptProcessed(id), active.promptTokens),
         .generatedTokens = generatedTokens(active),
         .maxNewTokens = active.request.maxNewTokens,
         .kvMode = active.kvMode});
  }
  std::ranges::sort(result.activeRequests, {}, &ActiveRequestSnapshot::id);
  return result;
}

bool Engine::admitQueued(double now) {
  const std::optional<RequestPriority> tier = suspendedTier();
  // Once pressure has preempted work, let resident lanes finish while memory
  // is still short before spending their released headroom on a retry or a
  // new request; this prevents repeated B4 admission/preemption churn. The
  // drain ends when growth is no longer paused and nothing has failed since
  // the suspension or since a resident lane last released its memory, and at
  // the latest after the resource wait limit; suspended requests are then
  // admitted before new work of their priority or below, one at a time. A
  // request of a strictly higher priority than every suspended one waits for
  // neither.
  if (!tier)
    drainEndMilliseconds_ = 0.0;
  const bool draining = drainingForRecovery();
  const std::vector<uint64_t> order = scheduler_.admissionOrder();
  // A request held behind a refusal is marked with the refused request until
  // the next pass (Request::heldBehind).
  for (auto &[_, active] : requests_)
    active.heldBehind.reset();
  const auto queue = [&](uint64_t id) {
    deferWait(request(id));
    scheduler_.deferAdmission(id);
  };
  // The tokens a request's start would reuse, from its probe brought up to
  // date, or as its last probe found for a start this pass does not try:
  // none for a request that ignores the cache, which is ranked by its whole
  // prompt.
  const auto cachedTokens = [&](Request &active, bool refresh) -> uint32_t {
    if (active.skipCache)
      return 0;
    if (!active.admissionProbe)
      active.admissionProbe = cache_.probe(active.exactTokens, promptIdentity(active.request));
    else if (refresh)
      cache_.refresh(*active.admissionProbe, active.exactTokens, promptIdentity(active.request));
    return active.admissionProbe->cachedTokens();
  };
  // A request starts only in a free lane. While every lane is resident,
  // record each wait of a request the scheduler would start as a failed
  // admission would. The first wanted start that waits for a lane takes one
  // from the lowest-priority resident below it, if there is one: the order
  // is by priority, so no later request has one to take. The next pass
  // admits it, above the lane it suspended. A request waiting for a prefix
  // takes none: it would leave the freed lane to the suspended one, which
  // would resume into it and replay for nothing.
  const bool lanesFull =
      std::count_if(requests_.begin(), requests_.end(), [](const auto &entry) {
        return entry.second.lane.has_value();
      }) >= model::ExecutionLimits::maximumBatchWidth;
  bool laneSought = false;
  const auto waitForLane = [&](uint64_t id, Request &active) {
    scheduler_.waitForResources(id);
    deferResourceRetry(active, now, {}, StateFailure::ConcurrencyLimit);
  };
  // The unstarted requests admission tries: during recovery those of a
  // strictly higher priority than every suspended one (admissionTries). A
  // request below the highest priority that prefills or decodes cannot run
  // before that priority is done (belowRunnable): it waits for scheduling,
  // unprobed, and takes no lane, neither a free one nor a lower resident's.
  // A start backing off from a refusal stays a candidate, ranked by its last
  // probe; one backing off from a lane wait, or restoring its prefix, is left
  // as it is.
  std::vector<PrefillAdmission> candidates;
  for (const uint64_t id : order) {
    Request &active = request(id);
    if (active.restore || scheduler_.suspended(id) || !admissionTries(active, tier, draining))
      continue;
    const bool ready = resourceRetryReady(active, now);
    if (!ready && !active.refusedResources)
      continue;
    if (belowRunnable(active)) {
      queue(id);
      continue;
    }
    if (!ready) {
      candidates.push_back({id, cachedTokens(active, false)});
      continue;
    }
    const uint32_t cached = cachedTokens(active, true);
    if (pendingSharedPrefill(active, cached)) {
      if (lanesFull) {
        waitForLane(id, active);
      } else {
        deferWait(active);
        scheduler_.waitForPrefix(id);
      }
      continue;
    }
    candidates.push_back({id, cached});
  }
  // The starts the scheduler wants in this pass
  // (Scheduler::prefillAdmissionOrder), in its lead order, each candidate
  // shaped by its placement in the memory as it is now (laneShapes). The
  // others wait for scheduling, and hold back nothing.
  std::vector<uint64_t> wanted;
  if (!candidates.empty()) {
    const std::vector<LaneShape> shapes = laneShapes(candidates);
    std::vector<PrefillAdmission> restoring;
    for (const auto &[id, active] : requests_)
      if (active.restore && !scheduler_.suspended(id))
        restoring.push_back({id, active.restore->lookup.resumeBoundary()});
    wanted = scheduler_.prefillAdmissionOrder(
        candidates, {.shapes = shapes, .passes = passCommand_.progress(), .restoring = restoring});
  }
  bool startWaits = false;
  for (const PrefillAdmission &candidate : candidates)
    if (std::ranges::find(wanted, candidate.requestId) == wanted.end()) {
      queue(candidate.requestId);
      startWaits = true;
    }
  // The wanted starts are tried in lead order, each once its backoff is over,
  // up to the first that stands refused or is refused: the wanted starts
  // after it that compete for what it was refused are held, not tried, in
  // this pass, so what reclaim, finishing lanes and returned commitments
  // free goes to it before the starts it leads. Every start competes for
  // memory; for the SSD cache's quota only those that will hold some of it
  // (competesForQuota). A start waiting for a prefix holds back nothing.
  std::optional<uint64_t> refused;
  std::optional<uint64_t> refusedQuota;
  bool progressed = false;
  for (const uint64_t id : wanted) {
    Request &active = request(id);
    // One placement for its attempt, in the memory as it is now: since
    // planning placed it, a start before it in this pass may have taken
    // room and a candidate the pass left waiting freed some (deferWait).
    // Whether it competes for the quota and where its start puts its KV
    // both read it (competesForQuota, admit).
    active.placement = placement(active);
    std::optional<uint64_t> behind = refused;
    if (!behind && refusedQuota && competesForQuota(active))
      behind = refusedQuota;
    if (behind) {
      queue(id);
      active.heldBehind = behind;
      continue;
    }
    // A start that stood refused holds the rest even when its retry
    // succeeds: its restore or first work takes what freed it in the
    // commands that follow.
    const bool stoodRefused = active.refusedResources;
    bool quota = active.resourceWait.diskQuota;
    if (resourceRetryReady(active, now)) {
      if (lanesFull) {
        if (!laneSought) {
          laneSought = true;
          if (preemptBelow(active.request.priority, now))
            return true;
        }
        waitForLane(id, active);
      } else if (admit(active, now)) {
        // Recovery admits one request at a time.
        if (tier)
          return true;
        progressed = true;
      } else if (scheduler_.phase(id) == Phase::WaitingPrefix) {
        continue;
      } else if (active.refusedResources) {
        quota = active.resourceWait.diskQuota;
      }
    }
    if (stoodRefused || active.refusedResources)
      (quota ? refusedQuota : refused) = id;
  }
  if (!tier)
    return progressed;
  // Recovery then resumes the suspended requests one at a time, in admission
  // order, below a running priority too, into the lane a higher priority
  // took from them, whatever lane leads. A refused resume holds back the
  // ones after it, as a refused start above them does. A start above them
  // that waits for scheduling holds them back too: resumed into the lane it
  // takes once it leads, they would replay for nothing below it and yield
  // that lane again. A held resume is not tried: it keeps no retry time,
  // and its wait limit starts again at its next attempt.
  std::optional<uint64_t> holding = refused ? refused : refusedQuota;
  for (const uint64_t id : order) {
    Request &active = request(id);
    if (!scheduler_.suspended(id) || active.restore || !admissionTries(active, tier, draining))
      continue;
    if (holding || startWaits) {
      active.resourceWait.retryMilliseconds = 0.0;
      active.resourceWait.deadlineMilliseconds = 0.0;
      active.heldBehind = holding;
      continue;
    }
    if (resourceRetryReady(active, now) && admit(active, now))
      return true;
    if (active.refusedResources)
      holding = id;
  }
  return false;
}

uint32_t Engine::replayStateBoundary(const Request &active) noexcept {
  // A later request may not share the generation prompt; generated history
  // that a resumed lane replays is its own: the state there is a checkpoint.
  if (active.replayTokens == active.promptTokens)
    return promptReplayBoundary(active);
  return (active.replayTokens - 1) / KvCache::pageTokens * KvCache::pageTokens;
}

uint32_t Engine::promptReplayBoundary(const Request &active) noexcept {
  const uint32_t tail = std::max(active.request.generationPromptTokens, uint32_t{1});
  return (active.promptTokens - tail) / KvCache::pageTokens * KvCache::pageTokens;
}

uint32_t Engine::sharedPrefillBoundary(const Request &left,
                                       const Request &right) {
  // Requests of different cache salts share no block.
  if (left.request.cacheSalt != right.request.cacheSalt)
    return 0;
  const auto a = std::span<const uint32_t>(left.exactTokens).first(left.promptTokens);
  const auto b = std::span<const uint32_t>(right.exactTokens).first(right.promptTokens);
  const auto end = std::mismatch(a.begin(), a.end(), b.begin(), b.end()).first;
  uint32_t boundary = std::min<uint32_t>(
      static_cast<uint32_t>(end - a.begin()),
      std::min(replayStateBoundary(left), replayStateBoundary(right)));
  boundary -= boundary % KvCache::pageTokens;
  // Blocks below the boundary have equal image identities exactly when the
  // spans that start below it are equal; otherwise the first differing
  // block holds the earliest start of the first unequal span.
  const auto startingBelow = [&](const Request &value) {
    const std::span<const ImageSpan> images = value.request.images;
    const auto end = std::partition_point(
        images.begin(), images.end(),
        [&](const ImageSpan &span) { return span.offset < boundary; });
    return std::span<const ImageSpan>(images.begin(), end);
  };
  const auto leftImages = startingBelow(left);
  const auto rightImages = startingBelow(right);
  const auto [leftUnequal, rightUnequal] = std::mismatch(
      leftImages.begin(), leftImages.end(), rightImages.begin(), rightImages.end());
  const uint32_t start =
      std::min(leftUnequal == leftImages.end() ? boundary : leftUnequal->offset,
               rightUnequal == rightImages.end() ? boundary : rightUnequal->offset);
  return start - start % KvCache::pageTokens;
}

bool Engine::pendingSharedPrefill(const Request &active,
                                  uint32_t resumeBoundary) const {
  // A request that ignores the cache uses no producer's state.
  if (active.skipCache)
    return false;
  for (const auto &[id, peer] : requests_) {
    if (!peer.lane || peer.request.priority > active.request.priority)
      continue;
    // A peer still restoring its prefix from disk has planned its boundaries
    // at admission; siblings wait for it rather than each reading the same
    // state. A failed restore discards those boundaries (pollRestores), which
    // releases them. A restoring producer that was cancelled or expired
    // publishes nothing; it only waits for its read to drain.
    if ((scheduler_.phase(id) != Phase::Prefill && !peer.restore) || peer.pendingEnd)
      continue;
    if (awaitedSharedBoundary(active, resumeBoundary, peer))
      return true;
  }
  return false;
}

uint32_t Engine::awaitedSharedBoundary(const Request &waiter, uint32_t resumeBoundary,
                                       const Request &producer) {
  const uint32_t shared = sharedPrefillBoundary(waiter, producer);
  const auto remaining =
      std::span(producer.stateBoundaries).subspan(producer.stateBoundaryCursor);
  const auto past =
      std::ranges::upper_bound(remaining, shared, {}, &Request::StateBoundary::tokens);
  if (past == remaining.begin() || std::prev(past)->tokens <= resumeBoundary)
    return 0;
  return std::prev(past)->tokens;
}

uint32_t Engine::awaitedBoundary(const Request &producer, uint32_t end) const {
  // Most commands reach no boundary, and compare no prompts.
  if (producer.stateBoundaryCursor == producer.stateBoundaries.size() ||
      producer.stateBoundaries[producer.stateBoundaryCursor].tokens > end)
    return end;
  uint32_t awaited = end;
  for (const auto &[id, waiter] : requests_) {
    // An unstarted request of the producer's priority starts once the states
    // it waits for are published, resuming where its probe last found; one
    // of a lower priority waits for the producer's priority to be done.
    if (waiter.lane || waiter.finalized || waiter.skipCache || scheduler_.suspended(id) ||
        waiter.request.priority != producer.request.priority)
      continue;
    const uint32_t resumeBoundary =
        waiter.admissionProbe ? waiter.admissionProbe->cachedTokens() : 0;
    if (const uint32_t boundary = awaitedSharedBoundary(waiter, resumeBoundary, producer))
      awaited = std::min(awaited, boundary);
  }
  return awaited;
}

bool Engine::admit(Request &active, double now) {
  const bool resuming = scheduler_.suspended(active.request.id);
  // A start of a higher priority than the command between its passes may
  // take that command, as its prompt's command does once it runs
  // (ClaimSpec::supersedes).
  const bool supersedes = scheduler_.yieldsTo(active.request.priority, {.passes = passCommand_.progress()});
  ModelRequest modelRequest = active.request.modelView();
  modelRequest.prompt = active.exactTokens;
  CacheLookup lookup =
      active.skipCache
          ? CacheLookup{}
          : cache_.lookup(modelRequest.prompt, promptIdentity(active.request),
                          active.admissionProbe ? &*active.admissionProbe
                                                : nullptr);
  // Only unstarted requests wait for a resident producer. Recheck planned
  // boundaries each step so producer loss leaves no stale dependency or lease.
  if (!resuming && pendingSharedPrefill(active, lookup.resumeBoundary())) {
    deferWait(active);
    scheduler_.waitForPrefix(active.request.id);
    return false;
  }
  // A prefix wait is not an attempt: a refusal stands until the next one.
  active.refusedResources = false;
  // Each attempt to start a request decides where its KV lives, the
  // placement admission made for it (admitQueued), and a resume keeps it;
  // each bounds its output afresh, so a bound an attempt that did not start
  // set does not outlive it. A Resident resume meets no bound again, and
  // keeps the one memory set at its start. A Spill request commits the SSD
  // cache's quota for the pages it publishes once it holds its lane.
  if (!resuming || active.kvMode == KvMode::Spill)
    active.outputLimit = active.request.maxNewTokens;
  if (!resuming)
    active.kvMode = *active.placement;
  // A request is in service when no other lane is resident
  // (anotherResident): the host exempts it once nothing in place meets it,
  // and it takes up to what is in use. Beside a resident lane it is a start,
  // which takes only the cache nothing in use holds, the model's caches it
  // rebuilds on use among them, and waits for that lane.
  ClaimSpec spec = claimFor(active, anotherResident(active.request.id) ? ClaimClass::Start : ClaimClass::InService);
  spec.supersedes = supersedes;
  bool commits = false;
  if (active.kvMode == KvMode::Spill) {
    switch (admitSpill(active, resuming, chainEnd(lookup), spec, now)) {
    case SpillAdmission::Commit:
      commits = true;
      break;
    case SpillAdmission::Resident:
      break;
    case SpillAdmission::Settled:
      return true;
    case SpillAdmission::Waiting:
      return false;
    }
  }
  modelRequest.maxNewTokens = active.outputLimit;
  const auto begin = [&] {
    modelRequest.restoredTokens = lookup.resumeBoundary();
    return resuming ? model_.resume(modelRequest) : model_.begin(modelRequest);
  };
  auto state = allocate(begin, spec, Refusal::Waits);
  // The request's own lease goes only when nothing else can free memory
  // (judge() says Fail: no other lane is resident, nothing is pending or in
  // flight, the host is not refusing). It then starts without the pin rather
  // than wait for ever, and looks up again once it holds its lane, so a state
  // the ledger wrote to disk is restored, not recomputed.
  bool droppedLease = false;
  if (!state.admission.granted() && lookup.state && memoryDenied(state.admission) &&
      judge(state.denial, active.request.id) == Verdict::Fail) {
    lookup = {};
    droppedLease = true;
    state = allocate(begin, spec, Refusal::Waits);
  }
  if (!state.admission.granted()) {
    // Without a free lane the request waits for one; only memory it could
    // not get may fail it.
    if (state.admission.failure == StateFailure::MemoryPressure) {
      if (judge(state.denial, active.request.id) == Verdict::Fail) {
        settle(active, capacityExhausted(active, "request state",
                                         state.admission.allocationFailure));
        return true;
      }
      waitForResources(active, now, state.denial);
      return false;
    }
    scheduler_.waitForResources(active.request.id);
    deferResourceRetry(active, now, state.denial, state.admission.failure);
    return false;
  }
  const uint64_t requestId = active.request.id;
  cache_.beginRequest(requestId, active.kvMode);
  active.lane = *state.admission.lane;
  active.admission = ++admissions_;
  if (droppedLease)
    lookup = cache_.lookup(modelRequest.prompt, promptIdentity(active.request));
  // The commitment covers the chain it restores too, once the lookup is
  // final, beside the copies on it the quota counts once the start holds
  // them. Writes a reclaim started may have taken the quota admitSpill
  // found, or the chain found again may have fewer copies: the start waits.
  const Cache::ChainCopies copies = commits ? cache_.chainCopies(chainEnd(lookup)) : Cache::ChainCopies{};
  if (commits && !cache_.commitSpill(requestId, spillPages(active, copies.copied), copies)) {
    vacateLane(active, resuming);
    waitForResources(active, now, {.diskQuota = true});
    return false;
  }
  const uint32_t resumeBoundary = lookup.resumeBoundary();
  // The matched chain first, then the first work's pages for a lane that
  // will not go through ordinary prefill admission before it runs: one
  // that resumes, or one that waits for a restore. As for its lane, beside
  // a resident request its pages wait for the host.
  Allocation<TokenAdmission> kv;
  if (lookup.state)
    kv = allocate([&] { return cache_.restoreRequest(requestId, lookup, heldPages(active)); }, spec,
                  Refusal::Waits);
  const bool restoring =
      lookup.state && (!lookup.state->state()->residentBytes() ||
                       cache_.kvRestoreStatus(requestId) == KvRestoreStatus::Pending);
  if (kv.admission.granted() && (resuming || restoring)) {
    // Its new pages past the chain it restored come cleared: a resume's
    // replay writes them in chunks that may end inside any of them. A
    // Resident lane that resumes takes the KV target it was suspended with
    // at once. A Spill lane takes the first work's page, as a start that
    // waits for its restore does: its replay takes Full pages a command at a
    // time (prepare), written through as its output's were.
    const auto workEnd = [&] {
      return resuming && active.kvMode == KvMode::Resident ? active.resumeKvTargetTokens
                                                           : uint64_t{resumeBoundary} + 1;
    };
    kv = allocate([&] { return cache_.ensureTokens(requestId, workEnd(), resumeBoundary, heldPages(active)); },
                  spec, Refusal::Waits);
  }
  if (!kv.admission.granted()) {
    // The host continuation survives this failed admission. No recurrent
    // state restore or replay has run, and all temporary leases are freed.
    vacateLane(active, resuming);
    const Verdict verdict = judge(kv.denial, requestId);
    if (verdict == Verdict::Fail && !restoring) {
      settle(active, capacityExhausted(active, "KV target", kv.admission.allocationFailure,
                                       pageShortfall(kv.admission)));
      return true;
    }
    // Release the prefix pin before retrying without its memory footprint.
    if (verdict == Verdict::Fail) {
      active.skipCache = true;
      active.admissionProbe.reset();
    }
    waitForResources(active, now, kv.denial);
    return false;
  }
  // Admitted, it waits for nothing.
  active.resourceWait = {};
  governor().ledger().setWaiting(active.claim.id, false);
  DraftContextPlan draft = configureDraftStatePlan(
      active, resumeBoundary, lookup.junctionBoundary);
  std::unique_ptr<StateRestore> transfer;
  if (lookup.state) {
    transfer = model_.beginRestore(requestId, resumeBoundary, lookup.state->state(),
                                   draft.restoresDraftState,
                                   completionNotifier_);
  }
  if (transfer || cache_.kvRestoreStatus(requestId) == KvRestoreStatus::Pending) {
    active.restore.emplace(Request::Restore{
        std::move(lookup), std::move(draft), std::move(transfer)});
    return true;
  }
  completeAdmission(active, lookup, std::move(draft));
  return true;
}

void Engine::completeAdmission(Request &active, CacheLookup &lookup,
                                DraftContextPlan draft) {
  const bool resuming = scheduler_.suspended(active.request.id);
  active.skipCache = false;
  active.admissionProbe.reset();
  const uint32_t resumeBoundary = lookup.resumeBoundary();
  // The blocks a resume restores past those the request published are blocks
  // another request computed, which its replay reads; the chain a start
  // restores is its cached prefix instead. Either way the restored chain is
  // the request's from the start of this residency.
  if (resuming && resumeBoundary > active.publishedTokens)
    active.unreadDeduplicatedKvTokens += resumeBoundary - active.publishedTokens;
  active.publishedTokens = std::max(active.publishedTokens, resumeBoundary);
  if (resuming)
    scheduler_.resumeFromResources(active.request.id, resumeBoundary, active.replayTokens);
  active.latestCheckpoint = {};
  active.unprovenState.reset();
  if (lookup.state) {
    active.unprovenState = lookup.state->kvBlock();
    active.latestCheckpoint = cache_.checkpointState(lookup.state->kvBlock());
    // A restored replay point of the prompt already has the ordinary state
    // the lane needs, in whichever tier holds it: a promotion that found no
    // cache slot leaves it on disk. Other restored progress points, the end of
    // a resumed lane's generated history among them, retain their rolling
    // lifetime.
    if (resumeBoundary == promptReplayBoundary(active)) {
      if (active.latestCheckpoint) {
        if (cache_.reuseStoredState(active.latestCheckpoint.kvBlock))
          ++counters_.deduplicatedStatePublications;
        active.latestCheckpoint = {};
      }
      active.replayPoint = cache_.useState(lookup.state->kvBlock());
    }
  }
  model_.setDraftContextPlan(active.request.id, std::move(draft));
  // A request whose output a start or a resume bounds counts once. A lane
  // bounded before it converts started Spill and never converts, so its
  // first bound tells whether it counts as a converted lane's too.
  if (!active.boundCounted && active.outputLimit < active.request.maxNewTokens) {
    active.boundCounted = true;
    ++counters_.spill.bounded;
    counters_.spill.convertedBounded += active.converted;
  }
  if (resuming) {
    active.resumeKvTargetTokens = 0;
    active.replaying = true;
    ++counters_.resourceResumptions;
    return;
  }
  scheduler_.resourcesReady(active.request.id, resumeBoundary);
  cache_.recordLookup(lookup);
  events_.started(active.request.id, resumeBoundary, *active.lane);
  if (active.request.returnProgress) {
    active.reportedPromptTokens = resumeBoundary;
    events_.promptProgress(active.request.id, resumeBoundary);
  }
  if (resumeBoundary) {
    ++counters_.cacheHits;
    counters_.reusedTokens += resumeBoundary;
  } else {
    ++counters_.coldMisses;
  }
  if (config_.spill.capable)
    ++(active.kvMode == KvMode::Spill ? counters_.spill.spill : counters_.spill.resident);
}

bool Engine::pollRestores() {
  bool progressed = false;
  for (auto &[id, active] : requests_) {
    if (!active.restore) continue;
    StateRestore *ticket = active.restore->ticket.get();
    // The state's read must drain before its lane is reused; KV restores
    // belong to their blocks and outlive a request that gives up.
    if (ticket && !ticket->ready()) continue;
    const KvRestoreStatus kv = cache_.kvRestoreStatus(id);
    if (!active.pendingEnd && kv == KvRestoreStatus::Pending) continue;
    // A state read that landed beside a command in flight waits for the
    // command to end, where the claim of its RAM copy may take units:
    // its lane joins no command before that.
    if (ticket && pending_ && !active.pendingEnd) continue;
    auto restore = std::move(*active.restore);
    active.restore.reset();
    progressed = true;
    if (active.pendingEnd) {
      restore.ticket.reset();
      restore.lookup = {};
      settle(active, *std::exchange(active.pendingEnd, std::nullopt));
      continue;
    }
    const bool stateRestored = !restore.ticket || restore.ticket->finish();
    if (stateRestored && kv == KvRestoreStatus::None) {
      if (restore.ticket)
        cache_.promoteState(restore.lookup, [&] { return promotedCopy(active, *restore.ticket); });
      completeAdmission(active, restore.lookup, std::move(restore.draft));
      continue;
    }
    // The prefix could not be brought back. What failed is gone from the
    // cache, so the next attempt matches the prefix that remains.
    if (!stateRestored) {
      cache_.discardState(restore.lookup.state->kvBlock(),
                          restore.lookup.state->state().get());
    }
    restore.ticket.reset();
    restore.lookup = {};
    vacateLane(active, scheduler_.suspended(id));
    signalResourceProgress();
    scheduler_.waitForResources(id);
  }
  return progressed;
}

std::shared_ptr<const CompositeState> Engine::promotedCopy(Request &active, StateRestore &restore) {
  // Optional, with no command in flight (pollRestores): without the copy
  // the restore reads the state from disk.
  ClaimSpec spec = claimFor(active, ClaimClass::Optional);
  spec.timing = NeedTiming::Immediate;
  std::shared_ptr<const CompositeState> state;
  static_cast<void>(allocate(
      [&] {
        state = restore.snapshot();
        return StateBuffers{.ok = state != nullptr};
      },
      spec, Refusal::Settled));
  return state;
}

DraftContextPlan Engine::configureDraftStatePlan(Request &active,
                                                 uint32_t stateBoundary,
                                                 uint32_t junctionBoundary) {
  if (!active.stateBoundaries.empty() || active.stateBoundaryCursor != 0) {
    throw std::logic_error("request already has a composite-state plan");
  }
  const uint32_t latestReplayBoundary = replayStateBoundary(active);
  // Plan draft windows before prefill; arbitrary chunk ends do not carry a
  // complete draft state, the point the lane resumes from does. Progress
  // points remain disposable after restoration.
  for (const uint32_t checkpoint : plannedCheckpoints(
           stateBoundary, latestReplayBoundary, config_.prefillCheckpointTokens)) {
    addStateBoundary(active, stateBoundary, checkpoint, true, true);
  }
  if (junctionBoundary >= stateBoundary + kMinimumJunctionGain)
    addStateBoundary(active, stateBoundary, junctionBoundary, false, true);
  addStateBoundary(active, stateBoundary, latestReplayBoundary,
                   latestReplayBoundary != promptReplayBoundary(active), true);
  // A resumed lane below its prompt's replay point lost that state; it
  // rebuilds the one its conversation's next turn resumes from on the way.
  addStateBoundary(active, stateBoundary, promptReplayBoundary(active), false, true);

  static_cast<void>(addSharedPrefillBoundaries(active, stateBoundary, true));
  return pendingDraftStatePlan(active, stateBoundary);
}

bool Engine::addStateBoundary(Request &active, uint32_t after, uint32_t tokens,
                              bool disposable, bool completeAfter) {
  // A state past the replay boundary would lie inside the generation prompt,
  // which a later request may not share.
  if (tokens <= after || tokens > replayStateBoundary(active))
    return false;
  const auto remaining = active.stateBoundaries.begin() + active.stateBoundaryCursor;
  const auto found = std::lower_bound(
      remaining, active.stateBoundaries.end(), tokens,
      [](const Request::StateBoundary &point, uint32_t value) {
        return point.tokens < value;
      });
  if (found != active.stateBoundaries.end() && found->tokens == tokens) {
    found->disposable = found->disposable && disposable;
    return false;
  }
  // The draft plan captures a new boundary's window from the planned
  // boundary before it, from `after` when the lane's draft state is
  // complete there, or whole past `after` (planDraftContext).
  if (!completeAfter && found == remaining &&
      tokens - after < model::ExecutionLimits::draftContextTokens)
    return false;
  active.stateBoundaries.insert(found, {tokens, disposable});
  return true;
}

bool Engine::addSharedPrefillBoundaries(Request &active, uint32_t after, bool completeAfter) {
  bool changed = false;
  for (const auto &[id, peer] : requests_) {
    // A request finalized earlier in this admission pass is still listed
    // until the pass ends. A request that ignores the cache uses no junction.
    if (id == active.request.id || peer.lane || scheduler_.suspended(id) ||
        peer.finalized || peer.skipCache ||
        peer.request.priority < active.request.priority)
      continue;
    changed = addStateBoundary(active, after, sharedPrefillBoundary(active, peer), false,
                               completeAfter) ||
              changed;
  }
  return changed;
}

DraftContextPlan Engine::pendingDraftStatePlan(const Request &active,
                                               uint32_t stateBoundary) const {
  std::vector<uint32_t> boundaries;
  boundaries.reserve(active.stateBoundaries.size() - active.stateBoundaryCursor);
  for (size_t i = active.stateBoundaryCursor; i < active.stateBoundaries.size(); ++i)
    boundaries.push_back(active.stateBoundaries[i].tokens);
  return planDraftContext(stateBoundary, active.replayTokens, boundaries);
}

void Engine::discardPendingStateBoundaries(Request &active) noexcept {
  active.stateBoundaries.clear();
  active.stateBoundaryCursor = 0;
}

bool Engine::checkpointShared(const Request &active) const {
  const auto point = active.latestCheckpoint;
  return point && std::any_of(requests_.begin(), requests_.end(), [&](const auto &entry) {
           const auto &peer = entry.second;
           return &peer != &active && !peer.finalized &&
                  peer.latestCheckpoint.kvBlock == point.kvBlock &&
                  peer.latestCheckpoint.publication == point.publication;
         });
}

bool Engine::checkpointReclaimable(const Request &active) const {
  // A pin keeps a state out of every victim choice, in either tier.
  const StateCheckpoint point = active.latestCheckpoint;
  return point && cache_.checkpointState(point.kvBlock).publication == point.publication &&
         !cache_.checkpointPinned(point);
}

bool Engine::checkpointPinned(const Request &active) const {
  return !checkpointShared(active) && cache_.checkpointPinned(active.latestCheckpoint);
}

bool Engine::retireCheckpoint(Request &active) {
  // Shared progress points remain disposable under memory pressure, but a
  // lane's normal rolling replacement must not retire its peer's recovery point.
  if (!checkpointShared(active) && !cache_.retireCheckpointState(active.latestCheckpoint))
    return false;
  active.latestCheckpoint = {};
  return true;
}

void Engine::publishCommittedBlocks(Request &active, uint32_t committedTokens) {
  // A replay after a suspension finds the blocks the request published
  // before as its own; only those past them can be another request's.
  const uint32_t replayed = std::min(committedTokens, active.publishedTokens);
  cache_.publishCommittedBlocks(active.request.id, active.exactTokens, replayed,
                                promptIdentity(active.request));
  active.unreadDeduplicatedKvTokens += cache_.publishCommittedBlocks(
      active.request.id, active.exactTokens, committedTokens, promptIdentity(active.request));
  active.publishedTokens = std::max(active.publishedTokens, committedTokens);
}

void Engine::publishReachedStateBoundaries(
    Request &active, uint32_t promptProcessed, std::span<const LaneCapture> captures,
    std::span<const std::shared_ptr<const CompositeState>> states, double now) {
  bool materialized = false;
  // The lane's draft state is complete at its progress when the command
  // ended at a boundary.
  bool endedAtBoundary = false;
  size_t captured = 0;
  while (active.stateBoundaryCursor < active.stateBoundaries.size() &&
         active.stateBoundaries[active.stateBoundaryCursor].tokens <=
             promptProcessed) {
    const Request::StateBoundary objective =
        active.stateBoundaries[active.stateBoundaryCursor++];
    const bool checkpoint = objective.disposable;
    // The conversation's next turn resumes from the prompt's replay point. A
    // rebuilt replay point counts as a replay-state publication, not a
    // junction.
    const bool replay = !checkpoint && objective.tokens == promptReplayBoundary(active);
    uint64_t &failures = checkpoint ? counters_.checkpointPublicationFailures
                         : replay   ? counters_.replayStatePublicationFailures
                                    : counters_.junctionMaterializationFailures;
    uint64_t &publications = checkpoint ? counters_.checkpointPublications
                             : replay   ? counters_.replayStatePublications
                                        : counters_.junctionMaterializations;
    // The command captured the state of every boundary it reached, but some
    // checkpoints inside it (below) and one at its end whose buffers were
    // refused (reserveCaptures): refused while the lane held a checkpoint,
    // that state is taken from the lane now (fromLane).
    std::shared_ptr<const CompositeState> state;
    bool recycled = false;
    bool fromLane = false;
    if (captured < captures.size() && captures[captured].tokens == objective.tokens) {
      recycled = captures[captured].recycled;
      fromLane = captures[captured].fromLane;
      if (!fromLane)
        state = states[captured];
      ++captured;
      if (objective.tokens != promptProcessed)
        ++counters_.stateCaptures;
    } else if (objective.tokens != promptProcessed) {
      // A checkpoint the command did not capture: a later state of it
      // retires the checkpoint once published, or the prompt's replay point
      // took its buffers.
      if (!checkpoint)
        throw std::logic_error("prefill passed a state boundary it did not capture");
      continue;
    }
    materialized = true;
    endedAtBoundary = objective.tokens == promptProcessed;
    const uint64_t block = cache_.blockAt(active.request.id, objective.tokens);
    // A windowed layout's state restores only through a Full tail, which
    // the command that computed it, or held it in its window, leaves so.
    if (!cache_.stateTailFull(block)) {
      ++failures;
      continue;
    }
    // The state is in use before any of the ways below keeps it, so each of
    // them makes room as work in use.
    if (replay)
      active.replayPoint = cache_.useState(block);
    if (cache_.reuseCompositeState(block, checkpoint)) {
      // The captured state goes back to the model's pool.
      ++counters_.deduplicatedStatePublications;
    } else {
      // The previous recovery point retires here or after the publication.
      // Here, before a checkpoint or a junction is published, a resident one
      // goes, and gives its buffers to the state of a boundary the command
      // ended at without them (below). One only on disk gives none: it stays
      // the recovery point until this state is published, and retires after
      // the publication, as does the one a reused state leaves. No checkpoint
      // is published while a restore lease pins the one before it. The
      // prompt's replay point keeps the lane's checkpoint (below).
      if (checkpoint && checkpointPinned(active)) {
        ++failures;
        continue;
      }
      const bool resident = cache_.stateResident(active.latestCheckpoint.kvBlock);
      if (!replay && resident)
        static_cast<void>(retireCheckpoint(active));
      // The command has completed, so the state it ended at for its lane's
      // checkpoint is taken from the lane now. The replay point keeps that
      // checkpoint only beside buffers of its own; refused them, it takes
      // the checkpoint's unless a peer still holds it. Then what the ledger
      // frees for the state's claim (stateClaim).
      if (fromLane) {
        bool first = true;
        recycled = allocate(
                       [&] {
                         state = model_.snapshot(active.request.id);
                         if (!state && std::exchange(first, false) && replay && resident &&
                             !checkpointShared(active) && retireCheckpoint(active))
                           state = model_.snapshot(active.request.id);
                         return StateBuffers{.ok = state != nullptr};
                       },
                       stateClaim(active,
                                  checkpoint ? StateClass::Checkpoint
                                  : replay   ? StateClass::InUse
                                             : StateClass::Ordinary,
                                  false),
                       Refusal::Settled)
                       .freed &&
                   state;
      }
      if (state) {
        cache_.publishCompositeState(block, std::move(state), checkpoint);
        ++publications;
        if (recycled)
          ++counters_.recycledStatePublications;
      } else if (cache_.reuseStoredState(block, checkpoint)) {
        // No cache slot takes a RAM copy of a state already on disk.
        ++counters_.deduplicatedStatePublications;
      } else if (model_.canSnapshotToDisk(cache_.stateInUse(block)) &&
                 cache_.publishStateToDisk(
                     block,
                     [&](std::function<void()> completion) {
                       return model_.snapshotToDisk(active.request.id, std::move(completion));
                     },
                     checkpoint)) {
        // No cache slot holds the state; the tier takes it from the lane.
        ++counters_.diskStatePublications;
        ++publications;
      } else {
        ++failures;
        continue;
      }
    }
    // The previous recovery point, if the lane still holds one, retires now,
    // except at the prompt's replay point: there it stays cached, as
    // disposable as any checkpoint, for a later request that shares the
    // prompt up to it but not up to the replay point. The lane holds it
    // until this state is published; a refused publication leaves it the
    // lane's recovery point, unless it retired above to give this state its
    // buffers.
    if (!replay)
      static_cast<void>(retireCheckpoint(active));
    active.latestCheckpoint = checkpoint ? cache_.checkpointState(block)
                                         : StateCheckpoint{};
    if (!checkpoint)
      writeBehind_.published(block, objective.tokens, now);
  }
  if (captured != captures.size())
    throw std::logic_error("prefill captured a state boundary it did not reach");
  // Late siblings extend the remaining plan after a command that reached a
  // boundary, which bounds how often the peers are compared, and only with a
  // boundary whose whole draft window can still be captured
  // (addStateBoundary).
  if (materialized && addSharedPrefillBoundaries(active, promptProcessed, endedAtBoundary))
    model_.setDraftContextPlan(
        active.request.id, pendingDraftStatePlan(active, promptProcessed));
  if (active.stateBoundaryCursor == active.stateBoundaries.size()) {
    active.stateBoundaries.clear();
    active.stateBoundaryCursor = 0;
  }
}

Engine::Prepared Engine::prepare(const BatchPlan &plan, std::vector<ModelBatchItem> &items,
                                 std::vector<std::vector<LaneCapture>> &captures, std::vector<uint64_t> &excluded,
                                 bool supersedes, double now) {
  const bool prefill = plan.kind == WorkKind::Prefill;
  const bool spill = config_.spill.capable;
  const PlanContext context{.prefill = prefill, .supersedes = supersedes, .now = now};
  // What the plan was made from that preparing it may change: each lane's
  // KV mode and whether its history held Index pages, and the command
  // between its passes.
  std::vector<std::pair<KvMode, bool>> shapes;
  for (const BatchItem &item : plan.items)
    shapes.emplace_back(request(item.requestId).kvMode, cache_.holdsIndexPages(item.requestId));
  const bool commandBetweenPasses = !plan.pass && passCommand_.active();
  const auto reshaped = [&] {
    for (size_t index = 0; index < plan.items.size(); ++index) {
      const uint64_t id = plan.items[index].requestId;
      if (request(id).kvMode != shapes[index].first || (cache_.holdsIndexPages(id) && !shapes[index].second))
        return true;
    }
    return false;
  };
  // The Index runway refills, as at the tick's start, before any lane grows.
  spillLanded(false);
  // A windowed layout's windowed layers' K/V of the pages every window has
  // passed, the processed rows' (Cache::trimWindows), leaves too: into the
  // Slim runway and what memory allows, the rest Full for a later pass. A
  // Slim page takes a fraction of the Full page it frees for the lanes to
  // grow into, so the Slim extents are memory the lanes in service need: an
  // InService claim of the engine's while one is, else an Optional one, and
  // a long prompt's pages trim as it prefills rather than filling the
  // budget Full. Refused, they ask the ledger as the lanes' pages do, and
  // in service pass the host once nothing in place meets them; with no lane
  // in service they take only what costs nothing. A
  // refusal the ledger does not meet leaves pages Full for a later pass.
  {
    const bool inService = laneInService();
    ClaimHandle claim;
    claim.governor = &governor();
    claim.id = governor().ledger().open(
        {.cls = inService ? ClaimClass::InService : ClaimClass::Optional,
         .timing = NeedTiming::Immediate});
    const MemoryGovernor::Scope scope(governor(), claim.id);
    static_cast<void>(cache_.trimWindows());
    for (Need need = governor().need(claim.id); need.bytes; need = governor().need(claim.id)) {
      if (governor().free(claim.id, need).outcome != FreeOutcome::Met)
        break;
      governor().ledger().refused(claim.id, {});
      if (!cache_.trimWindows())
        break;
    }
  }
  // A command's later passes run its items again: their lanes hold what
  // they need, and only their page tables may have changed, as moves
  // between commands rewrite them.
  if (plan.pass) {
    passCommand_.nextPass(items, captures);
    for (ModelBatchItem &item : items)
      setPageTable(item, cache_.cutPageTable(item.requestId));
    return Prepared::Runnable;
  }
  std::vector<BatchItem> admitted;
  admitted.reserve(plan.items.size());
  std::vector<Denied> denied;
  denied.reserve(plan.items.size());
  // A claim that took the command between its passes (EngineVictims) leaves
  // a plan made beside it: planning starts again at once, the memory the
  // lanes were granted kept.
  const auto commandTaken = [&] { return commandBetweenPasses && !passCommand_.active(); };
  for (const BatchItem &scheduled : plan.items) {
    Request &active = request(scheduled.requestId);
    if (!active.lane)
      throw std::logic_error("scheduled request is not resident");
    const std::optional<Denied> kv = growLane(active, workEndOf(active, scheduled, prefill), context);
    if (commandTaken())
      return Prepared::Replan;
    if (kv)
      denied.push_back(*kv);
    else
      admitted.push_back(scheduled);
  }
  // The model's memory for reads of the KV only the SSD cache holds: a
  // decoding lane's floor of the hot block cache, or the staging of a
  // prefill's history up to the command's end, for a lane whose history
  // holds Index pages, or may gain them before a later pass of the command
  // (gainsIndexPages), as a pass streams what it holds then. It is reserved
  // once every lane has its KV, since a lane's growth may release the pages
  // of lanes admitted before it (EngineVictims::releases).
  if (spill) {
    std::vector<BatchItem> kept;
    kept.reserve(admitted.size());
    for (const BatchItem &scheduled : admitted) {
      Request &active = request(scheduled.requestId);
      const uint64_t id = active.request.id;
      if (cache_.holdsIndexPages(id) ||
          (prefill && plan.passes > 1 && gainsIndexPages(active.kvMode, scheduled.promptOffset))) {
        const std::optional<Denied> memory = reserveReads(active, scheduled, context);
        if (commandTaken())
          return Prepared::Replan;
        if (memory) {
          denied.push_back(*memory);
          continue;
        }
      }
      kept.push_back(scheduled);
    }
    admitted = std::move(kept);
  }
  // A lane that runs is waiting for nothing, its claim either, once every
  // lane's memory for the command is decided; one refused waits for what
  // refused it, beside the lanes that run or alone: memory on its way back,
  // which its claim holds against the claims ranked below it (syncClaims),
  // or memory nothing returns, for which it holds nothing.
  for (const BatchItem &scheduled : admitted) {
    Request &runs = request(scheduled.requestId);
    runs.resourceWait = {};
    governor().ledger().setWaiting(runs.claim.id, false);
  }
  for (const Denied &entry : denied)
    noteRefusal(request(entry.requestId), now, entry.denial);
  // A lane refused the model's memory for its reads of spilled KV, which no
  // transfer in flight returns, takes it from the lane that yields first, as
  // when no lane fits, if that lane holds its memory outside this command
  // and yields before the refused one. The other lanes do not share that
  // refusal: run without the lane while they fit, they would leave it
  // refused for as long as they run. A lane refused its KV waits while they
  // fit, since they grow too and are refused in turn.
  if (Request *const yielding = laneToYield();
      yielding && std::ranges::none_of(plan.items, [&](const BatchItem &item) {
        return item.requestId == yielding->request.id;
      })) {
    const auto refused = std::ranges::find_if(denied, [&](const Denied &entry) {
      return !entry.kv && !entry.denial.pending &&
             yieldsBefore(*yielding, request(entry.requestId));
    });
    if (refused != denied.end())
      return yieldLane(*yielding, kvCapacity(yielding->request.id),
                       refused->denial.allocationFailure, now);
  }
  if (admitted.empty())
    return refuse(denied, context);
  // A lane refused leaves the plan: the others are planned again without
  // it, keeping what they were granted. So does a lane that converted to
  // Spill, or took Index pages by a release, while the command took shape.
  if (!denied.empty()) {
    for (const Denied &entry : denied)
      excluded.push_back(entry.requestId);
    return Prepared::Replan;
  }
  if (reshaped())
    return Prepared::Replan;
  // A lane whose command reaches the state a waiting request starts from, or
  // a state boundary whose capture is refused, ends there: it is planned
  // again with its rows up to it, which the others may fill. The first is
  // known before any state's buffers are taken.
  bool cut = false;
  for (size_t index = 0; prefill && index < plan.items.size(); ++index) {
    const BatchItem &item = plan.items[index];
    Request &active = request(item.requestId);
    const uint32_t end = awaitedBoundary(active, item.promptOffset + item.tokenCount);
    if (end != item.promptOffset + item.tokenCount) {
      active.rowCap = end - item.promptOffset;
      ++counters_.captureCuts;
      cut = true;
    }
  }
  if (cut)
    return Prepared::Replan;
  std::vector<std::vector<StateCapture>> states(plan.items.size());
  captures.assign(plan.items.size(), {});
  std::vector<uint32_t> ends(plan.items.size());
  for (size_t index = 0; prefill && index < plan.items.size(); ++index) {
    const BatchItem &item = plan.items[index];
    Request &active = request(item.requestId);
    ends[index] = item.promptOffset + item.tokenCount;
    // What a plan sent back took for the same rows, it takes again.
    if (std::optional<Request::KeptCaptures> kept = std::exchange(active.keptCaptures, std::nullopt);
        kept && kept->start == item.promptOffset && kept->end == ends[index]) {
      states[index] = std::move(kept->states);
      captures[index] = std::move(kept->captures);
      continue;
    }
    const uint32_t rows = reserveCaptures(active, item, states[index], captures[index], supersedes);
    if (rows != item.tokenCount) {
      active.rowCap = rows;
      ends[index] = item.promptOffset + rows;
      ++counters_.captureCuts;
      cut = true;
    }
  }
  // Sent back, each lane keeps what it took for its rows, which a plan made
  // again that runs it so takes without asking for it again. A claim of a
  // capture that took the command between its passes sends it back too.
  if (cut || reshaped() || commandTaken()) {
    for (size_t index = 0; prefill && index < plan.items.size(); ++index)
      request(plan.items[index].requestId).keptCaptures = Request::KeptCaptures{
          plan.items[index].promptOffset, ends[index], std::move(states[index]), std::move(captures[index])};
    return Prepared::Replan;
  }
  // The items come last: making room for a state's buffers may move KV
  // pages out of an extent, which rewrites the page lists of the lanes
  // on them.
  items.reserve(plan.items.size());
  for (size_t index = 0; index < plan.items.size(); ++index) {
    const BatchItem &scheduled = plan.items[index];
    Request &active = request(scheduled.requestId);
    ModelBatchItem item;
    item.requestId = active.request.id;
    item.logicalPosition = prefill ? scheduled.promptOffset : active.exactTokens.size();
    item.tokenCount = scheduled.tokenCount;
    setPageTable(item, cache_.cutPageTable(active.request.id));
    if (prefill) {
      item.inputTokens =
          std::span<const uint32_t>(active.exactTokens)
              .subspan(scheduled.promptOffset, scheduled.tokenCount);
      item.stateCaptures = std::move(states[index]);
    }
    items.push_back(std::move(item));
  }
  return Prepared::Runnable;
}

bool Engine::gainsIndexPages(std::optional<KvMode> mode, uint64_t processed) const {
  const SpillGeometry &geometry = config_.spill.geometry;
  return config_.spill.always && mode == KvMode::Spill &&
         processed > uint64_t{geometry.densePages + geometry.windowPages} * KvCache::pageTokens;
}

std::vector<LaneShape> Engine::laneShapes(std::span<const PrefillAdmission> candidates) const {
  std::vector<LaneShape> shapes;
  // A lane reads spilled history where it holds Index pages, a candidate
  // where its restore comes back with them (Cache::restoresIndexPages), and
  // under --kv-spill always every Spill lane, whose pages move as their
  // copies land.
  const auto spills = [&](KvMode mode) { return config_.spill.always && mode == KvMode::Spill; };
  for (const auto &[id, active] : requests_)
    if (active.lane && scheduler_.phase(id) == Phase::Prefill)
      shapes.push_back({.requestId = id,
                        .indexPages = cache_.indexPages(id),
                        .gainsIndexPages = gainsIndexPages(active.kvMode, scheduler_.promptProcessed(id)),
                        .rowCap = active.rowCap.value_or(std::numeric_limits<uint32_t>::max()),
                        .spill = spills(active.kvMode) || cache_.holdsIndexPages(id),
                        .ended = active.pendingEnd.has_value()});
  for (const PrefillAdmission &candidate : candidates) {
    const Request &active = requests_.at(candidate.requestId);
    const KvMode mode = placement(active);
    const bool restored = mode == KvMode::Spill && active.admissionProbe && !active.skipCache &&
                          cache_.restoresIndexPages(active.admissionProbe->restoredChain());
    shapes.push_back({.requestId = candidate.requestId,
                      .gainsIndexPages = gainsIndexPages(mode, candidate.cachedTokens),
                      .spill = spills(mode) || restored});
  }
  return shapes;
}

uint64_t Engine::workEndOf(const Request &active, const BatchItem &scheduled,
                           bool prefill) noexcept {
  uint64_t workEnd = active.exactTokens.size() + model::ExecutionLimits::targetVerifyRows;
  if (prefill) {
    workEnd = scheduled.promptOffset + scheduled.tokenCount;
    if (active.stateBoundaryCursor < active.stateBoundaries.size())
      workEnd = std::min<uint64_t>(
          workEnd, active.stateBoundaries[active.stateBoundaryCursor].tokens);
  }
  return workEnd;
}

Engine::Prepared Engine::yieldLane(Request &lane, uint64_t workEnd,
                                   metal::AllocationFailure failure, double now) {
  if (!yieldsEndedLanes(lane))
    suspendForGrowth(lane, workEnd, failure, now);
  return Prepared::Yielded;
}

uint64_t Engine::completedTokens(const Request &active) const {
  const uint64_t id = active.request.id;
  switch (scheduler_.phase(id)) {
  case Phase::Prefill:
    return scheduler_.promptProcessed(id);
  case Phase::Decode:
  case Phase::WaitingMask:
    return active.exactTokens.size();
  default:
    return active.admissionProbe ? active.admissionProbe->cachedTokens() : 0;
  }
}

uint32_t Engine::generatedTokens(const Request &active) noexcept {
  return static_cast<uint32_t>(active.exactTokens.size() - active.promptTokens) +
         (active.sentFirstToken ? 1 : 0);
}

void Engine::apply(const Pending &command, std::span<const ModelStepResult> results,
                   double cycleMilliseconds, double now) {
  const BatchPlan &plan = command.plan;
  const std::span<const std::vector<LaneCapture>> captures = command.captures;
  const ModelBatchTicket &ticket = *command.ticket;
  if (results.size() != plan.items.size()) {
    throw std::logic_error("model result count changed");
  }
  const model::CommandSample sample = ticket.sample();
  // A pass before the command's last commits nothing, proves nothing and
  // ends no lane.
  const bool beforeLastPass = plan.kind == WorkKind::Prefill && plan.pass + 1 < plan.passes;
  std::vector<StepResult> schedulerResults;
  schedulerResults.reserve(results.size());
  uint32_t inputTokens = 0;
  uint32_t outputTokens = 0;
  uint32_t draftedTokens = 0;
  uint32_t acceptedDraftTokens = 0;
  // Blocks whose tier copies a lane could not read.
  std::vector<uint64_t> unreadable;
  for (size_t index = 0; index < results.size(); ++index) {
    const ModelStepResult &result = results[index];
    const BatchItem &item = plan.items[index];
    if (result.requestId != item.requestId) {
      throw std::logic_error("model result order changed");
    }
    Request &active = request(result.requestId);
    // The model rejected this lane's own numerical result. A model that
    // accepts more draft tokens than it drafted, or emits a token outside the
    // vocabulary, without reporting it fails the lane the same way (a model
    // reports its non-finite rows itself).
    std::string invalid = result.failure;
    if (invalid.empty() && result.acceptedDraftTokens > result.draftedTokens)
      invalid = "model accepted more tokens than it drafted";
    if (invalid.empty()) {
      const auto outOfVocabulary = std::find_if(
          result.outputTokens.begin(), result.outputTokens.end(),
          [&](uint32_t token) { return token >= config_.vocabularySize; });
      if (outOfVocabulary != result.outputTokens.end())
        invalid = "model emitted out-of-vocabulary token " + std::to_string(*outOfVocabulary);
    }
    // KV the lane read could not be read back from the SSD cache: it ends,
    // as a retry recomputes that KV, and its copies are given up below. Its
    // result says nothing of the state it restored. Otherwise a damaged
    // state is one way to an invalid result: the state the request restored
    // goes, unless an earlier step computed from it was finite. An earlier
    // cancellation or deadline of the same lane still stands.
    if (!result.unreadableKvBlocks.empty()) {
      unreadable.insert(unreadable.end(), result.unreadableKvBlocks.begin(),
                        result.unreadableKvBlocks.end());
      deferEnd(active, {LaneOutcome::KvUnreadable, std::string(kKvUnreadableMessage)});
    } else if (!invalid.empty()) {
      if (active.unprovenState)
        cache_.invalidateState(*active.unprovenState);
      deferEnd(active, {LaneOutcome::ModelResultInvalid, std::move(invalid)});
    }
    // A lane that ended meanwhile stays in the command to its last pass.
    if (beforeLastPass) {
      schedulerResults.push_back({active.request.id, 0, false, result.nextDecodeStage});
      continue;
    }
    active.unprovenState.reset();
    if (active.pendingEnd) {
      // An in-flight Metal command cannot be revoked safely. Its provisional
      // writes remain invisible, but a cancelled, deadline-expired or
      // model-rejected request must not publish cache state or emit output
      // when that command drains; the states it captured go back to the
      // model's pool.
      schedulerResults.push_back({active.request.id,
                                  result.consumedPromptTokens, true,
                                  result.nextDecodeStage});
      continue;
    }
    if (result.outputTokensWithoutKv > result.outputTokens.size()) {
      throw std::logic_error("model reported more uncommitted tokens than output");
    }
    if (plan.kind == WorkKind::Prefill) {
      const uint32_t promptProcessed =
          item.promptOffset + result.consumedPromptTokens;
      inputTokens += result.consumedPromptTokens;
      if (active.replaying)
        counters_.resourceReplayTokens += result.consumedPromptTokens;
      if (promptProcessed == active.replayTokens)
        active.replaying = false;
      // The model captured every state but one taken from the lane, last.
      const std::vector<LaneCapture> &reached = captures[index];
      const bool fromLane = !reached.empty() && reached.back().fromLane;
      if (result.capturedStates.size() + fromLane != reached.size())
        throw std::logic_error("model returned mismatched captured states");
      publishCommittedBlocks(active, promptProcessed);
      publishReachedStateBoundaries(active, promptProcessed, captures[index],
                                    result.capturedStates, now);
      // Recovery may replay an already reported prefix, including generated
      // history.
      const uint32_t processed = std::min(promptProcessed, active.promptTokens);
      if (active.request.returnProgress &&
          processed > active.reportedPromptTokens) {
        active.reportedPromptTokens = processed;
        events_.promptProgress(active.request.id, processed);
      }
    }
    if (!result.outputTokens.empty()) {
      active.exactTokens.insert(active.exactTokens.end(),
                                result.outputTokens.begin(),
                                result.outputTokens.end());
      outputTokens += static_cast<uint32_t>(result.outputTokens.size());
      std::span<const uint32_t> unsent = result.outputTokens;
      if (active.sentFirstToken) {
        if (unsent.front() != *active.sentFirstToken)
          throw std::logic_error(
              "model committed another first token than the one sent");
        unsent = unsent.subspan(1);
        active.sentFirstToken.reset();
      }
      if (!unsent.empty()) {
        firstTokens(active);
        events_.tokens(active.request.id, unsent);
        decodeGap(active, sample, now);
      }
    }
    if (result.firstToken) {
      if (active.sentFirstToken || !result.outputTokens.empty())
        throw std::logic_error(
            "model reported a first token twice or beside output");
      active.sentFirstToken = result.firstToken;
      firstTokens(active);
      events_.tokens(active.request.id, {&*result.firstToken, 1});
      decodeGap(active, sample, now);
    }
    if (plan.kind == WorkKind::Decode) {
      draftedTokens += result.draftedTokens;
      acceptedDraftTokens += result.acceptedDraftTokens;
      active.speculation.draftedTokens += result.draftedTokens;
      active.speculation.acceptedDraftTokens += result.acceptedDraftTokens;
      active.speculation.cycles += result.draftedTokens != 0;
      // A terminal anchor is emitted without a target KV row; it never enters
      // a cached block.
      const uint32_t storedTokens =
          static_cast<uint32_t>(active.exactTokens.size()) -
          result.outputTokensWithoutKv;
      publishCommittedBlocks(active, storedTokens);
    }
    const uint64_t completionTokens =
        active.exactTokens.size() - active.promptTokens;
    const bool scoring = !active.request.scoreTokens.empty();
    if (scoring && !result.outputTokens.empty()) {
      throw std::logic_error("score request produced output tokens");
    }
    if (!result.scoreLogits.empty()) {
      if (!scoring ||
          result.scoreLogits.size() != active.request.scoreTokens.size()) {
        throw std::logic_error("model returned mismatched score logits");
      }
      active.scoreLogits = result.scoreLogits;
    }
    // Score requests carry maxNewTokens == 0; only the model's finished flag
    // on the final prompt chunk completes them. A request whose output the
    // SSD cache's quota or memory bounded ends at that bound.
    const bool complete =
        result.finished || (!scoring && completionTokens >= active.outputLimit);
    const bool boundEnd = complete && !result.finished && active.outputLimit < active.request.maxNewTokens;
    counters_.spill.lengthEnds += boundEnd;
    counters_.spill.convertedLengthEnds += boundEnd && active.converted;
    if (scoring && complete &&
        item.promptOffset + result.consumedPromptTokens !=
            active.promptTokens) {
      throw std::logic_error("score request finished before the prompt ended");
    }
    if (result.outputTokensWithoutKv && !complete) {
      throw std::logic_error("model emitted an uncommitted token and continued");
    }
    schedulerResults.push_back({active.request.id, result.consumedPromptTokens,
                                complete, result.nextDecodeStage,
                                static_cast<uint32_t>(result.outputTokens.size())});
    if (waitsForMask(result.nextDecodeStage)) {
      events_.maskRequested(active.request.id, {});
      active.maskRequestedMilliseconds = now;
    }
  }
  // No later lookup matches a block whose copy could not be read; a lane
  // holding it keeps the copy until it ends.
  std::ranges::sort(unreadable);
  const auto repeated = std::ranges::unique(unreadable);
  unreadable.erase(repeated.begin(), repeated.end());
  for (const uint64_t block : unreadable)
    cache_.poisonCopy(block);
  // Spill requests' new pages are written as they publish them.
  if (config_.spill.capable)
    static_cast<void>(cache_.writeThrough());
  const double wallMilliseconds = sample.wallMilliseconds;
  // A prefill unit ran in the gaps of the lanes that have sent tokens.
  if (plan.kind == WorkKind::Prefill)
    for (auto &[id, active] : requests_)
      if (!active.finalized && active.lastTokenMilliseconds &&
          std::ranges::none_of(plan.items, [&](const BatchItem &item) { return item.requestId == id; })) {
        active.gapSince.prefillMilliseconds += wallMilliseconds;
      }
  scheduler_.complete(plan, schedulerResults, sample);
  if (plan.kind == WorkKind::Prefill && plan.passes > 1)
    settleEnded(passCommand_.passCompleted(plan));
  events_.batchCompleted(plan.kind, plan.width(), inputTokens, outputTokens,
                         draftedTokens, acceptedDraftTokens, wallMilliseconds,
                         cycleMilliseconds);

  for (size_t index = 0; !beforeLastPass && index < results.size(); ++index) {
    const ModelStepResult &result = results[index];
    Request &active = request(result.requestId);
    if (active.pendingEnd) {
      settle(active, *std::exchange(active.pendingEnd, std::nullopt));
    } else if (schedulerResults[index].finished) {
      finish(active, result.finished ? EngineFinishReason::Stop
                                     : EngineFinishReason::Length,
             active.scoreLogits);
    }
  }
}

bool Engine::inFlight(uint64_t id) const {
  return pending_ &&
         std::any_of(pending_->plan.items.begin(), pending_->plan.items.end(),
                     [id](const BatchItem &item) { return item.requestId == id; });
}

bool Engine::yieldsEndedLanes(const Request &lane) {
  if (!passCommand_.holds(lane.request.id) || !passCommand_.anyEnded())
    return false;
  settleEnded(passCommand_.yield());
  return true;
}

void Engine::settleEndedBetweenPasses() {
  for (auto &[id, active] : requests_)
    if (active.pendingEnd && !active.finalized && !active.restore && !inFlight(id) && !passCommand_.holds(id))
      settle(active, *std::exchange(active.pendingEnd, std::nullopt));
}

void Engine::settleEnded(std::span<const uint64_t> lanes) {
  for (const uint64_t id : lanes) {
    Request &lane = request(id);
    settle(lane, std::exchange(lane.pendingEnd, std::nullopt).value());
  }
}

void Engine::deferEnd(Request &active, LaneEnd end) {
  if (!active.pendingEnd)
    active.pendingEnd = std::move(end);
}

void Engine::settle(Request &active, LaneEnd end) {
  if (active.finalized)
    return;
  const uint64_t id = active.request.id;
  const bool running = inFlight(id);
  if (active.restore || running || passCommand_.holds(id)) {
    deferEnd(active, std::move(end));
    if (active.restore) {
      if (active.restore->ticket)
        active.restore->ticket->cancel();
    } else {
      if (running)
        pending_->ticket->abandonMask(id);
      settleEnded(passCommand_.laneEnded(id, running));
    }
    return;
  }
  if (end.outcome == LaneOutcome::Cancelled)
    finish(active, EngineFinishReason::Cancelled, {});
  else if (end.outcome == LaneOutcome::Stopped)
    finish(active, EngineFinishReason::Stop, {});
  else
    finishFailure(active, std::move(end));
}

Engine::LaneEnd Engine::deadlineEnd() {
  return {LaneOutcome::DeadlineExceeded, std::string(kDeadlineExceededMessage)};
}

std::string Engine::memoryShortfall(const Request &active, const std::optional<ResidentRoom> &room) const {
  const uint64_t id = active.request.id;
  const Phase phase = scheduler_.phase(id);
  const char *const when = phase == Phase::Prefill                                   ? "in prefill"
                           : phase == Phase::Decode || phase == Phase::WaitingMask ? "in decode"
                           : scheduler_.suspended(id)                              ? "at its resume"
                                                                                    : "at its start";
  // Less Metal memory outside the server's buffers, which the limit charges
  // past the runtime reserves, fits it too: 17 MB of it on a 64 GB M5 Pro
  // serving Flash-Next.
  const bool spill = !room && active.kvMode == KvMode::Spill;
  std::string message = "the request, a " + std::string(spill ? "Spill" : "Resident") + " lane of " +
                        std::to_string(active.exactTokens.size()) + " tokens " + when +
                        ", does not fit in the memory this server may use beside device_untracked_bytes=" +
                        std::to_string(governor().untrackedBytes()) + " of Metal memory outside the server's buffers";
  if (room)
    message += ": it needs memory_needed_bytes=" + std::to_string(room->needed) +
               ", where memory_available_bytes=" + std::to_string(room->available) + " are there for it now";
  return message;
}

Engine::LaneEnd Engine::capacityExhausted(const Request &active,
                                          std::string_view what,
                                          metal::AllocationFailure failure,
                                          std::string_view detail) const {
  std::string message = "could not allocate " + std::string(what) + ": " + metal::allocationFailureName(failure);
  if (!detail.empty())
    message += " (" + std::string(detail) + ")";
  // A lone lane judge failed: only other limits fit it. --max-memory only
  // lowers this Mac's budget (EngineMemoryPolicy::hardBudgetBytes), so a
  // larger one helps only where one lowered it.
  message += "; " + memoryShortfall(active) +
             ", even after every cached prefix was evicted and any store of streamed experts shrank to its "
             "floor: restart the server with a smaller --max-context, or with a larger --max-memory where one "
             "lowered it";
  return {LaneOutcome::CapacityExhausted, std::move(message)};
}

void Engine::finish(Request &active, EngineFinishReason reason,
                    std::span<const float> optionLogits) {
  if (reason == EngineFinishReason::Cancelled)
    scheduler_.cancel(active.request.id);
  else
    scheduler_.finish(active.request.id);
  active.finalized = true;
  const uint32_t completionTokens = generatedTokens(active);
  // Only a bound of the engine's own ends a request with Length short of
  // its output: the quota's or memory's (outputLimit), or a lone Spill
  // lane's at the wall.
  const bool bounded =
      reason == EngineFinishReason::Length && completionTokens < active.request.maxNewTokens;
  events_.completed(active.request.id, reason, bounded, active.promptTokens,
                    completionTokens, active.deduplicatedKvTokens, optionLogits,
                    active.speculation);
  if (reason == EngineFinishReason::Cancelled) {
    ++counters_.cancelled;
  } else {
    ++counters_.completed;
  }
  release(active);
}

void Engine::finishFailure(Request &active, LaneEnd end) {
  scheduler_.fail(active.request.id);
  active.finalized = true;
  events_.failed(active.request.id, end.outcome, std::move(end.message), end.retryAfterSeconds);
  ++counters_.failed;
  release(active);
}

void Engine::vacateLane(Request &active, bool keepContinuation) {
  const uint64_t id = active.request.id;
  if (passCommand_.holds(id))
    settleEnded(passCommand_.yield());
  discardPendingStateBoundaries(active);
  if (keepContinuation)
    model_.suspend(id);
  else
    model_.end(id);
  cache_.endRequest(id);
  active.lane.reset();
}

void Engine::release(Request &active) {
  active.resourceWait = {};
  // An ended request holds no claim.
  active.claim.reset();
  active.maskRequestedMilliseconds.reset();
  if (active.lane || scheduler_.suspended(active.request.id)) {
    // A resident lane gives its memory back: what failed for want of it may
    // fit now, so the recovery drain ends and the next attempt finds out.
    if (active.lane)
      allocationFailed_ = false;
    vacateLane(active, false);
    signalResourceProgress();
  }
  // Every end comes here; this request's use of its replay point ends,
  // after endRequest, so that the point is stamped newer than its KV tail.
  active.replayPoint.reset();
}

void Engine::sweepTerminal(double now) {
  bool swept = false;
  for (auto iterator = requests_.begin(); iterator != requests_.end();) {
    const uint64_t id = iterator->first;
    Request &active = iterator->second;
    if (!active.finalized) {
      ++iterator;
      continue;
    }
    scheduler_.remove(id);
    iterator = requests_.erase(iterator);
    swept = true;
  }
  if (weights_ && swept && requests_.empty())
    weights_->idleFrom(now);
}

bool Engine::releaseIdleWeights(double now) {
  return weights_.value().release(now, idle(), true);
}

bool Engine::releaseWeights(double now) {
  return weights_.value().release(now, idle(), false);
}

void Engine::startIdleClock(double now) { weights_.value().idleFrom(now); }

WeightsSnapshot Engine::weightsSnapshot() const {
  WeightsSnapshot snapshot = weights_.value().snapshot();
  // A request that finds them released restores them: from its arrival.
  snapshot.restoring = snapshot.restoring ||
                       (snapshot.released && std::ranges::any_of(requests_, [](const auto &entry) {
                          return !entry.second.finalized;
                        }));
  return snapshot;
}

bool Engine::advanceWeights(double now) {
  // Every request not finalized waits for the weights: none starts before
  // they are back.
  std::vector<Request *> waiting;
  for (auto &[_, active] : requests_)
    if (!active.finalized)
      waiting.push_back(&active);
  if (waiting.empty() && !weights_->restoring())
    return false;
  // Every tick tries: memory the host frees comes back without a signal, and
  // the loop wakes for the requests' retries and limits.
  const WeightResidency::Step step = [&] {
    // Memory the requests waiting need, a Floor claim's: the host's margins
    // do not hold it back, critical pressure and the engine's limit refuse
    // it. Without a request it is a Start claim's, which comes back only as
    // far as the margins admit it.
    ClaimHandle claim;
    claim.governor = &governor();
    claim.id = governor().ledger().open(
        {.cls = waiting.empty() ? ClaimClass::Start : ClaimClass::Floor, .timing = NeedTiming::Immediate});
    const MemoryGovernor::Scope scope(governor(), claim.id);
    return weights_->restore(now);
  }();
  if (step.kind != WeightResidency::Step::Kind::Refused) {
    // A part that came back ends each waiting request's wait for memory: its
    // limit counts from its next refusal, by the restore or, once the weights
    // are back, by its start.
    for (Request *waiter : waiting) {
      waiter->resourceWait = {};
      scheduler_.deferAdmission(waiter->request.id);
    }
    signalResourceProgress();
    return true;
  }
  // Each waiting request waits for memory on its own clock, and fails once
  // this attempt at or past its limit was refused.
  bool failed = false;
  for (Request *waiter : waiting) {
    deferResourceRetry(*waiter, now, {.allocationFailure = step.failure});
    scheduler_.waitForResources(waiter->request.id);
    const double deadline = resourceDeadline(*waiter);
    if (deadline > 0.0 && now >= deadline) {
      settle(*waiter, {LaneOutcome::ResourceTimeout, resourceTimeoutMessage(step.failure)});
      weights_->waiterFailed();
      failed = true;
    }
  }
  if (std::ranges::any_of(waiting, [](const Request *waiter) { return !waiter->finalized; }))
    return failed;
  weights_->abandon();
  return true;
}

Engine::Request &Engine::request(uint64_t id) {
  auto found = requests_.find(id);
  if (found == requests_.end())
    throw std::out_of_range("unknown request");
  return found->second;
}

} // namespace splash::engine

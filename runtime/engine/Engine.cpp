#include "engine/Engine.hpp"

#include <algorithm>
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
// A mask request the server leaves unanswered this long fails its request;
// the batch's command slot is not held longer.
constexpr int kMaskWaitLimitMilliseconds = 5000;
// A junction costs a cached state, its capture and up to a draft window of
// draft-context rows; a later request must save at least that much prefill.
constexpr uint32_t kMinimumJunctionGain = model::ExecutionLimits::draftContextTokens;

// What the request's KV blocks are cached under beside its tokens.
PromptIdentity promptIdentity(const EngineRequest &request) noexcept {
  return {request.images, request.cacheSalt};
}

// A refusal for memory, which reclaim or the host's recovery may end.
bool memoryDenied(const StateAdmission &admission) noexcept {
  return admission.failure == StateFailure::MemoryPressure;
}

bool memoryDenied(const TokenAdmission &admission) noexcept {
  return admission.failure == TokenAdmissionFailure::Denied;
}

bool memoryDenied(const model::ModelMemoryAdmission &admission) noexcept {
  return !admission.granted();
}

// Memory a transfer in flight holds, which returns by itself.
bool transferPending(const StateAdmission &) noexcept { return false; }

bool transferPending(const TokenAdmission &admission) noexcept {
  return admission.failure == TokenAdmissionFailure::Pending;
}

bool transferPending(const model::ModelMemoryAdmission &) noexcept { return false; }

// What a KV target lacked when it could not be allocated.
std::string pageShortfall(const TokenAdmission &admission) {
  return "additional_pages=" + std::to_string(admission.additionalPages) +
         ", free_pages=" + std::to_string(admission.availablePages);
}

constexpr std::string_view kKvUnreadableMessage =
    "KV of an earlier part of the conversation could not be read from the SSD cache";
constexpr std::string_view kDiskQuotaTimeoutMessage =
    "the SSD cache's room for active KV did not become available within the resource wait "
    "limit: other long requests hold it";

} // namespace

std::string resourceTimeoutMessage(metal::AllocationFailure failure) {
  std::string message = "memory did not become available within the resource wait limit";
  if (failure == metal::AllocationFailure::HostPressure)
    message += ": macOS is short of memory; close memory-heavy applications";
  return message;
}

Engine::Engine(EngineConfig config, Cache &cache, model::Model &model,
               EngineEventSink &events)
    : config_(config), cache_(cache), writeBehind_(cache), model_(model),
      events_(events), scheduler_(config_.decodeShare, config_.maximumPrefillAttentionPairs) {
  if (!config_.maxContext || !config_.vocabularySize) {
    throw std::invalid_argument("context and vocabulary sizes must be positive");
  }
  if (!config_.growthPaused || !config_.serving)
    throw std::invalid_argument("the engine needs the governor's growth pause and serving mark");
  if (!std::isfinite(config_.decodeShare) || config_.decodeShare < 0.0)
    throw std::invalid_argument("decode share must be nonnegative and finite");
  if (config_.spill.capable &&
      (!config_.spill.geometry.fullPageBytes || !config_.spill.geometry.slotBytes))
    throw std::invalid_argument("a target that can spill needs its page and slot sizes");
}

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
  scheduler_.submit(
      {.id = id,
       .priority = stored.request.priority,
       .prefillTokens = stored.promptTokens,
       .deadlineMilliseconds = stored.request.deadlineMilliseconds,
       .prefillGroupRows = model_.prefillGroupRows(false),
       .prefillPasses = model_.prefillPasses()});
  ++counters_.submitted;
}

void Engine::cancel(uint64_t id) {
  if (const auto found = requests_.find(id); found != requests_.end())
    settle(found->second, {LaneOutcome::Cancelled, {}});
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
}

bool Engine::flushRestorePoints() {
  static_cast<void>(cache_.pollTransfers());
  return writeBehind_.flush();
}

bool Engine::tick(double now) {
  model_.checkHealth();
  nextHealthCheckMilliseconds_ = now + kHealthCheckIntervalMilliseconds;
  bool progressed = scheduler_.expireDeadlines(now);
  if (now >= drainEndMilliseconds_)
    drainEndMilliseconds_ = 0.0;
  if (cache_.pollTransfers()) {
    // Demoted pages and written states are back; waiting lanes retry now.
    // The tier's share for Spill requests' writes has room again.
    signalResourceProgress();
    progressed = true;
    if (config_.spill.capable)
      static_cast<void>(cache_.writeThrough());
  }
  progressed = pollRestores(now) || progressed;
  progressed = writeBehind_.run(now) || progressed;
  // The earliest submitted and the earliest admitted of the lanes with work
  // in flight.
  uint64_t earliestWorking = std::numeric_limits<uint64_t>::max();
  uint64_t earliestWorkingAdmission = std::numeric_limits<uint64_t>::max();
  if (pending_) {
    for (const BatchItem &item : pending_->plan.items) {
      earliestWorking =
          std::min(earliestWorking, scheduler_.submissionOrder(item.requestId));
      earliestWorkingAdmission =
          std::min(earliestWorkingAdmission, request(item.requestId).admission);
    }
  }
  const bool draining = drainingForRecovery();
  for (auto &[id, active] : requests_) {
    // A mask request left unanswered past the limit ends its request: a lane
    // in flight commits without the mask, any other fails now.
    if (!active.pendingEnd && active.maskRequestedMilliseconds &&
        now >= *active.maskRequestedMilliseconds + kMaskWaitLimitMilliseconds) {
      settle(active, {LaneOutcome::MaskTimeout,
                      "the server did not answer a token-mask request within " +
                          std::to_string(kMaskWaitLimitMilliseconds) + " ms"});
      progressed = true;
    }
    // Admission is deliberately paused while resident peers finish. Start a
    // fresh resource wait only if admission still fails after that drain.
    if (draining) {
      active.resourceWait.deadlineMilliseconds = 0.0;
      continue;
    }
    // A lane submitted before the request, or admitted before it was refused
    // memory or suspended, holds memory it may wait for until it finishes;
    // while it works, the wait's limit restarts. Other lanes do not extend
    // it: requests that keep arriving would otherwise hold it until its
    // deadline.
    const std::optional<uint64_t> &admittedBefore = active.resourceWait.admittedBefore;
    if (scheduler_.submissionOrder(id) > earliestWorking ||
        (admittedBefore && earliestWorkingAdmission <= *admittedBefore))
      active.resourceWait.earlierLaneWorkMilliseconds = now;
    const double deadline = resourceDeadline(active);
    if (!active.finalized && deadline > 0.0 && now >= deadline) {
      settle(active, {LaneOutcome::ResourceTimeout,
                      active.resourceWait.diskQuota
                          ? std::string(kDiskQuotaTimeoutMessage)
                          : resourceTimeoutMessage(active.resourceWait.allocationFailure)});
      progressed = true;
    }
  }
  // Finalize expired requests now, even while a command is in flight, so a
  // late mask or cancel for them is a no-op rather than a scheduler error.
  if (progressed)
    sweepTerminal();
  if (pending_) {
    auto forwardMaskRequests = [&] {
      for (ModelMaskRequest &asked : pending_->ticket->takeMaskRequests()) {
        events_.maskRequested(asked.requestId, asked.simulationTokens);
        request(asked.requestId).maskRequestedMilliseconds = now;
        progressed = true;
      }
    };
    forwardMaskRequests();
    for (const BatchItem &item : pending_->plan.items) {
      Request &active = request(item.requestId);
      if (!active.pendingEnd && active.request.deadlineMilliseconds <= now) {
        settle(active, deadlineEnd());
        progressed = true;
      }
    }
    // A mask response, cancellation, or deadline can make the commit tail
    // runnable without another Metal completion wake.
    forwardMaskRequests();
    if (!pending_->ticket->ready())
      return progressed;
    Pending command = std::move(*pending_);
    pending_.reset();
    std::vector<ModelStepResult> results = command.ticket->wait();
    apply(command, results, now - command.startedMilliseconds, now);
    busySinceMilliseconds_ = now;
    sweepTerminal();
    return true;
  }

  progressed = admitQueued(now) || progressed;
  sweepTerminal();
  if (config_.spill.capable) {
    for (const auto &[id, active] : requests_)
      if (active.lane && scheduler_.phase(id) == Phase::Prefill)
        scheduler_.setStreamsHistory(id, streamsHistory(active));
  }
  // Lanes that wait for memory on its way back cannot run before it lands;
  // the command runs the other lanes meanwhile. A plan whose lanes all wait
  // adds them, so each attempt plans without at least one more lane.
  std::vector<uint64_t> excluded;
  for (const auto &[id, active] : requests_) {
    if (active.lane && active.resourceWait.pending && !resourceRetryReady(active, now))
      excluded.push_back(id);
  }
  while (auto plan = scheduler_.next(excluded)) {
    std::vector<ModelBatchItem> items;
    std::vector<std::vector<Capture>> captures;
    switch (prepare(*plan, items, captures, now)) {
    case Prepared::Runnable: {
      std::unique_ptr<ModelBatchTicket> ticket =
          model_.submit(*plan, items, completionNotifier_);
      if (!ticket) {
        throw std::logic_error("model returned an empty command ticket");
      }
      // A prefill lane whose page list names Index pages reads its history
      // from the SSD cache: alone in its command (streamsHistory).
      const bool streamed = plan->kind == WorkKind::Prefill && items.front().pageSources;
      scheduler_.commit(*plan, excluded);
      // The command reads the cached pages its lanes' blocks were swapped
      // for, and those a resume restored. Selecting a constrained request's
      // first token reads none, but the cycle after it does; only a first
      // token that stops the request counts pages nothing read.
      for (const BatchItem &item : plan->items) {
        Request &lane = request(item.requestId);
        lane.deduplicatedKvTokens += lane.unreadDeduplicatedKvTokens;
        counters_.deduplicatedKvTokens += std::exchange(lane.unreadDeduplicatedKvTokens, 0);
      }
      pending_ = Pending{std::move(*plan), std::move(captures), std::move(ticket),
                         busySinceMilliseconds_.value_or(now), streamed};
      return true;
    }
    case Prepared::Yielded:
      sweepTerminal();
      return true;
    case Prepared::Waiting:
      for (const BatchItem &item : plan->items)
        excluded.push_back(item.requestId);
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
         (allocationFailed_ || config_.growthPaused());
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

bool Engine::holdsBack(std::optional<uint64_t> refused) const {
  if (!refused)
    return false;
  const auto found = requests_.find(*refused);
  return found != requests_.end() && !found->second.finalized && found->second.refusedResources;
}

std::optional<double> Engine::nextWakeupMilliseconds() const {
  std::optional<double> result;
  if (pending_)
    result = nextHealthCheckMilliseconds_;
  const bool draining = drainingForRecovery();
  if (draining && (!result || drainEndMilliseconds_ < *result))
    result = drainEndMilliseconds_;
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
    if (active.maskRequestedMilliseconds) {
      const double limit =
          *active.maskRequestedMilliseconds + kMaskWaitLimitMilliseconds;
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
  if (const std::optional<double> persist = writeBehind_.nextWakeup();
      persist && (!result || *persist < *result))
    result = persist;
  return result;
}

EngineSnapshot Engine::snapshot() const {
  EngineSnapshot result = counters_;
  result.maximumContextTokens = config_.maxContext;
  result.scheduler = scheduler_.snapshot();
  result.resources = cache_.snapshot();
  result.writeBehind = writeBehind_.snapshot();
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

ResourceWaitSnapshot Engine::resourceWaitSnapshot(double now) const {
  ResourceWaitSnapshot result;
  result.draining = drainingForRecovery();
  for (const auto &[id, active] : requests_) {
    if (active.finalized)
      continue;
    // Admitted, it waits for disk reads, not for memory.
    if (active.restore) {
      ++result.restoring;
      continue;
    }
    const bool waiting = scheduler_.phase(id) == Phase::WaitingResources;
    // What the latest admission pass held back, in whatever phase it left
    // them, while the request they wait behind is still refused, and a
    // request refused resources while a pass keeps it out of its wait.
    if (holdsBack(active.heldBehind) || (active.refusedResources && !waiting))
      ++result.heldBehindRefusal;
    if (!waiting)
      continue;
    if (active.resourceWait.reason == StateFailure::ConcurrencyLimit)
      ++result.concurrency;
    else if (active.resourceWait.diskQuota)
      ++result.diskQuota;
    else
      ++result.memory;
    if (scheduler_.suspended(id))
      ++result.suspended;
    if (active.resourceWait.startedMilliseconds)
      result.oldestWaitMilliseconds = std::max(
          result.oldestWaitMilliseconds, now - *active.resourceWait.startedMilliseconds);
  }
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
  // A request this pass does not start, or that waits behind one refused
  // memory, waits for scheduling; the latter is marked with the refused
  // request until the next pass.
  for (auto &[_, active] : requests_)
    active.heldBehind.reset();
  const auto queue = [&](uint64_t id) {
    deferWait(request(id));
    scheduler_.deferAdmission(id);
  };
  const auto hold = [&](uint64_t id, uint64_t refused) {
    queue(id);
    request(id).heldBehind = refused;
  };
  // A request below the highest priority that prefills or decodes cannot
  // run before that priority is done. It waits for scheduling, unprobed, and
  // takes no lane, neither a free one nor a lower resident's: the lanes of
  // that priority free theirs as they finish.
  const std::optional<RequestPriority> runnable = scheduler_.highestRunnablePriority();
  const auto belowRunnable = [&](const Request &active) {
    return runnable && active.request.priority > *runnable;
  };
  // The tokens a request's start would reuse, from its probe brought up to
  // date: none for a request that ignores the cache, which is ranked by its
  // whole prompt.
  const auto cachedTokens = [&](Request &active) -> uint32_t {
    if (active.skipCache)
      return 0;
    if (active.admissionProbe)
      cache_.refresh(*active.admissionProbe, active.exactTokens, promptIdentity(active.request));
    else
      active.admissionProbe = cache_.probe(active.exactTokens, promptIdentity(active.request));
    return active.admissionProbe->cachedTokens();
  };
  // A request starts only in a free lane. While every lane is resident,
  // record each wait as a failed admission would. The first request that
  // waits for a lane and not for a producer's prefix takes one from the
  // lowest-priority resident below it, if there is one: the order is by
  // priority, so no later request has one to take. The next pass admits
  // it, above the lane it suspended. Only the requests up to that one are
  // probed, to tell a prefix wait: a request waiting for a prefix would
  // leave the freed lane to the suspended one, which would resume into it
  // and replay for nothing.
  const bool lanesFull =
      std::count_if(requests_.begin(), requests_.end(), [](const auto &entry) {
        return entry.second.lane.has_value();
      }) >= model::ExecutionLimits::maximumBatchWidth;
  bool laneSought = false;
  const auto waitForLane = [&](uint64_t id, Request &active) {
    if (!laneSought && !pendingSharedPrefill(active, cachedTokens(active))) {
      laneSought = true;
      if (preemptBelow(active.request.priority, now))
        return true;
    }
    scheduler_.waitForResources(id);
    deferResourceRetry(active, now, {}, StateFailure::ConcurrencyLimit);
    return false;
  };
  if (tier) {
    // Recovery admits one request at a time, in admission order: those
    // above every suspended request, then the suspended ones, which resume
    // below a running priority too. As in ordinary admission, the first one
    // refused memory holds back the ones after it. Those are not tried, so
    // they keep no retry time, and their wait limit starts again at their
    // next attempt.
    std::optional<uint64_t> refused;
    for (uint64_t id : order) {
      Request &active = request(id);
      if (active.restore || !admissionTries(active, tier, draining))
        continue;
      if (refused) {
        active.resourceWait.retryMilliseconds = 0.0;
        active.resourceWait.deadlineMilliseconds = 0.0;
        active.heldBehind = refused;
        continue;
      }
      if (resourceRetryReady(active, now)) {
        const bool suspended = scheduler_.suspended(id);
        if (!suspended && belowRunnable(active))
          queue(id);
        else if (lanesFull && !suspended ? waitForLane(id, active) : admit(active, now))
          return true;
      }
      if (active.refusedResources)
        refused = id;
    }
    return false;
  }

  // A request whose start was refused memory, or the SSD cache's quota,
  // closes admission behind it until it starts: admission is open for
  // order[0, open). What reclaim, finishing lanes and returned commitments
  // free would otherwise keep going to later arrivals that need less of it
  // at once, and it would wait for as long as they keep coming. Waiting for
  // a free lane closes nothing: nothing starts without one.
  size_t open = order.size();
  std::vector<PrefillAdmission> candidates;
  // Each candidate's place in order.
  std::unordered_map<uint64_t, size_t> positions;
  for (size_t index = 0; index < order.size(); ++index) {
    const uint64_t id = order[index];
    Request &active = request(id);
    if (index >= open) {
      if (!active.restore)
        hold(id, order[open - 1]);
      continue;
    }
    if (active.refusedResources)
      open = index + 1;
    if (active.restore || !resourceRetryReady(active, now))
      continue;
    if (belowRunnable(active)) {
      queue(id);
      continue;
    }
    if (lanesFull) {
      if (waitForLane(id, active))
        return true;
      continue;
    }
    const uint32_t cached = cachedTokens(active);
    if (pendingSharedPrefill(active, cached)) {
      deferWait(active);
      scheduler_.waitForPrefix(id);
      continue;
    }
    positions.emplace(id, index);
    candidates.push_back({id, cached});
  }
  bool progressed = false;
  while (!candidates.empty()) {
    const auto selected = scheduler_.prefillAdmissionOrder(candidates);
    if (selected.empty())
      break;
    for (uint64_t id : selected) {
      if (positions.at(id) >= open)
        continue;
      progressed = admit(request(id), now) || progressed;
      std::erase_if(candidates, [id](const auto &value) {
        return value.requestId == id;
      });
      // Refused resources in this pass, it closes admission behind it at once.
      if (request(id).refusedResources)
        open = std::min(open, positions.at(id) + 1);
    }
    std::erase_if(candidates, [&](const auto &value) {
      if (positions.at(value.requestId) < open)
        return false;
      hold(value.requestId, order[open - 1]);
      return true;
    });
    // A request that could not start holds back only what arrived after it.
    if (progressed)
      break;
  }
  for (const auto &candidate : candidates)
    queue(candidate.requestId);
  return progressed;
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
  // Each attempt to start a request decides where its KV lives, and a resume
  // keeps it; each bounds its output afresh, so a bound an attempt that did
  // not start set does not outlive it. A Spill request commits the SSD
  // cache's quota for the pages it publishes once it holds its lane.
  active.outputLimit = active.request.maxNewTokens;
  if (!resuming)
    active.kvMode = kvModeFor(active);
  bool commits = false;
  if (active.kvMode == KvMode::Spill) {
    switch (admitSpill(active, resuming, chainEnd(lookup), now)) {
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
  // A request is in service when no other lane is resident
  // (anotherResident): it starts through the host's pause once reuse gives
  // nothing, and reclaims up to what is in use. Beside a resident lane it
  // takes nothing in use and waits for that lane.
  const bool inService = !anotherResident(active.request.id);
  const ReclaimClass upTo = inService ? ReclaimClass::InUse : ReclaimClass::Ordinary;
  // The request's own lease goes only when nothing else can free memory
  // (judge() says Fail: no other lane is resident, nothing is pending, the
  // host is not refusing). It then starts without the pin rather than wait
  // for ever, and looks up again once it holds its lane, so a state the
  // reclaim wrote to disk is restored, not recomputed.
  bool droppedLease = false;
  const auto state = allocate(
      [&] {
        modelRequest.restoredTokens = lookup.resumeBoundary();
        return resuming ? model_.resume(modelRequest) : model_.begin(modelRequest);
      },
      inService, upTo, [&](const Denial &denial) {
        if (!lookup.state || judge(denial, active.request.id) != Verdict::Fail)
          return false;
        lookup = {};
        droppedLease = true;
        return true;
      });
  if (!state.admission.granted()) {
    // Without a free lane the request waits for one; only memory it could
    // not get may fail it.
    if (state.admission.failure == StateFailure::MemoryPressure &&
        judge(state.denial, active.request.id) == Verdict::Fail) {
      settle(active, capacityExhausted("request state",
                                       state.admission.allocationFailure));
      return true;
    }
    active.refusedResources = state.admission.failure == StateFailure::MemoryPressure;
    scheduler_.waitForResources(active.request.id);
    deferResourceRetry(active, now, state.denial, state.admission.failure);
    // A start refused memory takes it from a resident of a lower priority,
    // whose suspension makes it retry at the next pass. Memory on its way
    // back makes no lane yield: the start waits for it.
    if (active.refusedResources && !resuming && !state.denial.pending)
      static_cast<void>(preemptBelow(active.request.priority, now));
    return false;
  }
  const uint64_t requestId = active.request.id;
  cache_.beginRequest(requestId, active.kvMode);
  active.lane = *state.admission.lane;
  active.admission = ++admissions_;
  if (droppedLease)
    lookup = cache_.lookup(modelRequest.prompt, promptIdentity(active.request));
  // The commitment covers the chain it restores too, once the lookup is
  // final. Writes a reclaim started may have taken the quota admitSpill
  // found, or the chain found again may have fewer copies: the start waits.
  if (commits && !cache_.commitSpill(requestId, spillPages(active, chainEnd(lookup)))) {
    vacateLane(active, resuming);
    waitForDiskQuota(active, now);
    return false;
  }
  const uint32_t resumeBoundary = lookup.resumeBoundary();
  // The matched chain first, then the first work's pages for a lane that
  // will not go through ordinary prefill admission before it runs: one
  // that resumes, or one that waits for a restore. As for its lane, beside
  // a resident request its pages wait for the host. Refused them, a
  // Resident lane may convert to Spill.
  const auto convert = spillFallback(active, &lookup);
  Allocation<TokenAdmission> kv;
  if (lookup.state)
    kv = allocate([&] { return cache_.restoreRequest(requestId, lookup); }, inService, upTo,
                  convert);
  const bool restoring =
      lookup.state && (!lookup.state->state()->residentBytes() ||
                       cache_.kvRestoreStatus(requestId) == KvRestoreStatus::Pending);
  if (kv.admission.granted() && (resuming || restoring)) {
    // Its new pages past the chain it restored come cleared: a resume's
    // replay writes them in chunks that may end inside any of them. A
    // Resident lane that resumes takes the KV target it was suspended with
    // at once. A Spill lane, a Resident one that converts here included,
    // takes the first work's page, as a start that waits for its restore
    // does: its replay takes Full pages a command at a time (prepare), which
    // spill as their copies land, as its output's did.
    const auto workEnd = [&] {
      return resuming && active.kvMode == KvMode::Resident ? active.resumeKvTargetTokens
                                                           : uint64_t{resumeBoundary} + 1;
    };
    kv = allocate([&] { return cache_.ensureTokens(requestId, workEnd(), resumeBoundary); },
                  inService, upTo, convert);
  }
  if (!kv.admission.granted()) {
    // The host continuation survives this failed admission. No recurrent
    // state restore or replay has run, and all temporary leases are freed.
    vacateLane(active, resuming);
    const Verdict verdict = judge(kv.denial, requestId);
    if (verdict == Verdict::Fail && !restoring) {
      settle(active, capacityExhausted("KV target", kv.admission.allocationFailure,
                                       pageShortfall(kv.admission)));
      return true;
    }
    // Release the prefix pin before retrying without its memory footprint.
    if (verdict == Verdict::Fail) {
      active.skipCache = true;
      active.admissionProbe.reset();
    }
    active.refusedResources = true;
    scheduler_.waitForResources(requestId);
    deferResourceRetry(active, now, kv.denial);
    // As for a refused state.
    if (!resuming && !kv.denial.pending)
      static_cast<void>(preemptBelow(active.request.priority, now));
    return false;
  }
  active.resourceWait = {};
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

uint64_t Engine::projectedTokens(const Request &active) noexcept {
  const uint64_t generated = active.exactTokens.size() - active.promptTokens;
  const uint64_t remaining = active.outputLimit - std::min<uint64_t>(generated, active.outputLimit);
  return active.exactTokens.size() + std::min<uint64_t>(remaining, kProjectedOutputTokens);
}

KvMode Engine::kvModeFor(const Request &active) const {
  const EngineSpillConfig &spill = config_.spill;
  if (!spill.capable)
    return KvMode::Resident;
  if (spill.always)
    return KvMode::Spill;
  const uint64_t tokens = projectedTokens(active);
  if (tokens <= spill.denseTokens)
    return KvMode::Resident;
  uint64_t bytes = spill.geometry.residentBytes(tokens);
  for (const auto &[id, peer] : requests_) {
    if (!peer.lane || id == active.request.id)
      continue;
    const uint64_t projection = projectedTokens(peer);
    bytes += peer.kvMode == KvMode::Spill ? spill.geometry.spillFloorBytes(projection)
                                          : spill.geometry.residentBytes(projection);
  }
  return bytes <= spill.geometry.residentKvBytes ? KvMode::Resident : KvMode::Spill;
}

bool Engine::fitsResidentAlone(const Request &active) const {
  return projectedTokens(active) <= config_.spill.geometry.residentTokens;
}

uint64_t Engine::chainEnd(const CacheLookup &lookup) noexcept {
  return lookup.state ? lookup.state->kvBlock() : 0;
}

uint64_t Engine::chainEnd(const Request &active) const {
  const uint64_t id = active.request.id;
  const uint32_t held = scheduler_.phase(id) == Phase::Prefill ? scheduler_.promptProcessed(id)
                                                                : active.publishedTokens;
  return held < KvCache::pageTokens
             ? 0
             : cache_.blockAt(id, held / KvCache::pageTokens * KvCache::pageTokens);
}

uint32_t Engine::spillPages(const Request &active, uint64_t chainEnd) const {
  const uint64_t pages = projectedTokens(active) / KvCache::pageTokens;
  return static_cast<uint32_t>(pages - std::min<uint64_t>(pages, cache_.copiedBlocks(chainEnd)));
}

Engine::SpillAdmission Engine::admitSpill(Request &active, bool resuming, uint64_t chainEnd,
                                          double now) {
  const SpillGeometry &geometry = config_.spill.geometry;
  // What it would do without the SSD cache: start Resident if its
  // projection fits in memory alone, and wait for the other lanes there.
  const auto residentInstead = [&] {
    if (!fitsResidentAlone(active))
      return false;
    active.kvMode = KvMode::Resident;
    return true;
  };
  // A resume goes on spilling whatever the tier and the write budget: its
  // writes are in use, and pages a closed tier does not take stay Full.
  if (!resuming) {
    if (const SpillGate gate = cache_.spillGate(); gate != SpillGate::Open) {
      if (residentInstead())
        return SpillAdmission::Resident;
      const std::string reason = "requests past " + std::to_string(geometry.residentTokens) +
                                 " tokens keep their KV on the SSD cache, ";
      if (gate == SpillGate::WriteBudget) {
        ++counters_.spill.refusedWriteBudget;
        // It refills over hours: a retry may succeed once it covers again
        // what Spill requests committed, or sooner as they end.
        const double admitted = cache_.spillWritesAdmittedAt().value_or(now);
        settle(active, {LaneOutcome::KvWriteBudget,
                        reason + "whose daily write budget (--max-cache-disk-writes) is spent",
                        static_cast<uint32_t>(std::lround(std::max(0.0, admitted - now) / 1000.0))});
      } else {
        ++counters_.spill.refusedClosed;
        settle(active, {LaneOutcome::CapacityExhausted, reason + "which takes no writes"});
      }
      return SpillAdmission::Settled;
    }
  }
  const uint64_t room = cache_.spillRoom() / geometry.slotBytes;
  if (room >= spillPages(active, chainEnd))
    return SpillAdmission::Commit;
  // Short of its projection, a start that fits in memory alone starts
  // Resident, as without the SSD cache, unless every request spills.
  if (!resuming && !config_.spill.always && residentInstead())
    return SpillAdmission::Resident;
  // Otherwise the quota bounds its output to the pages it holds, if they
  // hold a token past the tokens the request holds; those of the chain it
  // restores that have copies need none. A resume counts among them the
  // anchor its model holds once it selected the request's first token
  // (Request::suspendedWithAnchor), which the model decodes only with
  // output left past it: it emits a last budgeted anchor as it selects it.
  const uint64_t held = active.exactTokens.size() + (active.suspendedWithAnchor ? 1 : 0);
  const uint64_t copied = cache_.copiedBlocks(chainEnd);
  const uint64_t needed = held / KvCache::pageTokens + 1 - copied;
  if (room >= needed) {
    active.outputLimit = static_cast<uint32_t>(std::min<uint64_t>(
        active.outputLimit, (copied + room) * KvCache::pageTokens - active.promptTokens));
    return SpillAdmission::Commit;
  }
  // What it lacks is other requests' active KV, the copies of the blocks
  // they hold and their commitments: it waits for them to end. The quota
  // cannot hold it even alone: a start is refused, and a resume that has no
  // room left for output ends.
  const uint64_t activeLimitBytes = cache_.spillLimit();
  if (activeLimitBytes / geometry.slotBytes >= needed) {
    waitForDiskQuota(active, now);
    return SpillAdmission::Waiting;
  }
  if (resuming) {
    endAtLength(active);
  } else {
    ++counters_.spill.refusedQuota;
    settle(active, {LaneOutcome::KvDiskQuota,
                    "this request's KV needs needed_bytes=" +
                        std::to_string(needed * geometry.slotBytes) +
                        " of the SSD cache, which may hold active_limit_bytes=" +
                        std::to_string(activeLimitBytes) +
                        " of active KV now (--max-cache-disk and the volume's free space)"});
  }
  return SpillAdmission::Settled;
}

void Engine::waitForDiskQuota(Request &active, double now) {
  active.refusedResources = true;
  scheduler_.waitForResources(active.request.id);
  deferResourceRetry(active, now, {.diskQuota = true});
}

std::function<bool(const Engine::Denial &)> Engine::spillFallback(Request &active,
                                                                  const CacheLookup *restoring) {
  if (!config_.spill.capable)
    return {};
  // A start that converts starts as a Spill request, which completeAdmission
  // counts, and one refused all the same decides its mode afresh at its next
  // attempt; a lane that ran Resident before converts once.
  const bool started = !restoring || scheduler_.suspended(active.request.id);
  // Memory on its way back comes without it.
  return [this, &active, restoring, started](const Denial &denial) {
    if (denial.pending || active.kvMode != KvMode::Resident ||
        !cache_.convertToSpill(active.request.id,
                               spillPages(active, restoring ? chainEnd(*restoring)
                                                            : chainEnd(active))))
      return false;
    active.kvMode = KvMode::Spill;
    scheduler_.setPrefillGroupRows(active.request.id, model_.prefillGroupRows(true));
    counters_.spill.converted += started;
    return true;
  };
}

bool Engine::streamsHistory(const Request &active) const {
  const SpillGeometry &geometry = config_.spill.geometry;
  return cache_.holdsIndexPages(active.request.id) ||
         (active.kvMode == KvMode::Spill &&
          scheduler_.promptProcessed(active.request.id) >
              uint64_t{geometry.densePages + geometry.windowPages} * KvCache::pageTokens);
}

bool Engine::spillingPrompt() const {
  return std::any_of(requests_.begin(), requests_.end(), [&](const auto &entry) {
    const auto &[id, active] = entry;
    return active.lane && active.kvMode == KvMode::Spill && scheduler_.phase(id) == Phase::Prefill;
  });
}

void Engine::endAtLength(Request &active) {
  ++counters_.spill.lengthEnds;
  finish(active, EngineFinishReason::Length, {});
}

void Engine::completeAdmission(Request &active, CacheLookup &lookup,
                                DraftContextPlan draft) {
  const bool resuming = scheduler_.suspended(active.request.id);
  // A lane whose history spills runs prefill groups, which read it once for
  // their chunks; a Resident one only where every lane runs them, so a
  // request that arrives beside it waits for a chunk.
  scheduler_.setPrefillGroupRows(active.request.id,
                                 model_.prefillGroupRows(active.kvMode == KvMode::Spill));
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
    // we need, in whichever tier holds it: a promotion that found no cache
    // slot leaves it on disk. Other restored progress points, the end of a
    // resumed lane's generated history among them, retain their rolling
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
  if (config_.spill.capable) {
    ++(active.kvMode == KvMode::Spill ? counters_.spill.spill : counters_.spill.resident);
    counters_.spill.bounded += active.outputLimit < active.request.maxNewTokens;
  }
}

bool Engine::pollRestores(double now) {
  bool progressed = false;
  for (auto &[id, active] : requests_) {
    if (!active.restore) continue;
    // Scheduler::expireDeadlines has already failed this lane's phase; a
    // restore landing now would otherwise admit the lane on it
    // (completeAdmission) before sweepTerminal ends it.
    if (!active.pendingEnd && active.request.deadlineMilliseconds <= now)
      settle(active, deadlineEnd());
    StateRestore *ticket = active.restore->ticket.get();
    // The state's read must drain before its lane is reused; KV restores
    // belong to their blocks and outlive a request that gives up.
    if (ticket && !ticket->ready()) continue;
    const KvRestoreStatus kv = cache_.kvRestoreStatus(id);
    if (!active.pendingEnd && kv == KvRestoreStatus::Pending) continue;
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
      if (restore.ticket) cache_.promoteState(restore.lookup, *restore.ticket);
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

bool Engine::resourceRetryReady(const Request &active,
                                double now) const noexcept {
  return active.resourceWait.retryMilliseconds <= 0.0 ||
         active.resourceWait.epoch != resourceEpoch_ ||
         now >= active.resourceWait.retryMilliseconds;
}

void Engine::deferResourceRetry(Request &active, double now,
                                const Denial &denial, StateFailure reason) noexcept {
  auto &wait = active.resourceWait;
  if (!wait.startedMilliseconds)
    wait.startedMilliseconds = now;
  // Waiting for a lane is not waiting for memory: the lanes that count are
  // those admitted before the first refusal of memory.
  if (!wait.admittedBefore && reason != StateFailure::ConcurrencyLimit)
    wait.admittedBefore = admissions_;
  const bool progressed = wait.pending && wait.epoch != resourceEpoch_;
  wait.reason = reason;
  wait.allocationFailure = denial.allocationFailure;
  wait.pending = denial.pending;
  wait.diskQuota = denial.diskQuota;
  if (reason == StateFailure::ConcurrencyLimit)
    wait.deadlineMilliseconds = 0.0;
  else if (progressed || wait.deadlineMilliseconds <= 0.0)
    wait.deadlineMilliseconds = now + config_.resourceWaitTimeoutMilliseconds;
  wait.epoch = resourceEpoch_;
  wait.retryMilliseconds = now + kResourceRetryBackoffMilliseconds;
}

void Engine::deferWait(Request &active) noexcept {
  ResourceWait kept;
  if (active.refusedResources) {
    kept.startedMilliseconds = active.resourceWait.startedMilliseconds;
    kept.admittedBefore = active.resourceWait.admittedBefore;
  }
  active.resourceWait = kept;
}

double Engine::resourceDeadline(const Request &active) const noexcept {
  const ResourceWait &wait = active.resourceWait;
  if ((wait.pending && wait.epoch != resourceEpoch_) || wait.deadlineMilliseconds <= 0.0)
    return 0.0;
  // The limit restarts whenever a lane submitted before the request, or
  // admitted before it was refused memory or suspended, works, however long
  // that takes. Other lanes do not extend it: requests that keep arriving
  // would otherwise hold it until the request's deadline.
  return std::max(wait.deadlineMilliseconds,
                  wait.earlierLaneWorkMilliseconds + config_.resourceWaitTimeoutMilliseconds);
}

void Engine::signalResourceProgress() noexcept {
  if (resourceEpoch_ != std::numeric_limits<uint64_t>::max())
    ++resourceEpoch_;
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
    Request &active, uint32_t promptProcessed, std::span<const Capture> captures,
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
      // the checkpoint's unless a peer still holds it. Then room comes from
      // cache the state's class may take.
      if (fromLane) {
        state = model_.snapshot(active.request.id);
        if (!state && replay && resident && !checkpointShared(active) &&
            retireCheckpoint(active))
          state = model_.snapshot(active.request.id);
        if (!state)
          recycled = makeStateRoom(checkpoint ? StateClass::Checkpoint
                                   : replay   ? StateClass::InUse
                                              : StateClass::Ordinary,
                                   [&] {
                                     state = model_.snapshot(active.request.id);
                                     return state != nullptr;
                                   });
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

Engine::Prepared Engine::prepare(BatchPlan &plan, std::vector<ModelBatchItem> &items,
                                 std::vector<std::vector<Capture>> &captures, double now) {
  const bool prefill = plan.kind == WorkKind::Prefill;
  const bool spill = config_.spill.capable;
  // Pages whose copies have landed leave memory before any lane grows; the
  // Index runway they draw on refills if memory allows. The Full extents
  // they empty leave memory too, but one, the runway, while a spilling
  // prompt still grows into it or the host holds growth back, when it could
  // not be allocated again. The Index pages, the runway's included, are
  // memory the resident Spill lanes need, as their Full pages are, and a
  // fraction of what the moves free: the host's pause holds them back only
  // while no such lane is in service.
  if (spill) {
    const SpilledExtents extents =
        spillingPrompt() || config_.growthPaused() ? SpilledExtents::KeepRunway : SpilledExtents::Released;
    const bool inService = std::any_of(requests_.begin(), requests_.end(), [](const auto &entry) {
      return entry.second.lane && entry.second.kvMode == KvMode::Spill;
    });
    static_cast<void>(allocate([&] { return cache_.spillWritten(extents); }, inService,
                               ReclaimClass::Ordinary, {}, true));
  }
  // A command's later passes run its item again: the lane holds what it
  // needs, and only its page table may have changed, as moves between
  // commands rewrite it.
  if (plan.pass) {
    Request &active = request(plan.items.front().requestId);
    if (!active.passCommand)
      throw std::logic_error("a prefill command's pass has no command");
    ModelBatchItem item = active.passCommand->item;
    const PageTableView pageTable = cache_.cutPageTable(active.request.id);
    item.pageTable = pageTable.pages;
    item.pageTableRevision = pageTable.revision;
    item.pageTableFirstChanged = pageTable.firstChanged;
    item.pageSources = pageTable.sources;
    items.push_back(std::move(item));
    captures.assign(1, active.passCommand->captures);
    active.resourceWait = {};
    return Prepared::Runnable;
  }
  std::vector<BatchItem> admitted;
  admitted.reserve(plan.items.size());
  struct Denied final {
    uint64_t requestId = 0;
    uint64_t workEnd;
    Denial denial;
    // What it was refused: its KV target, or else the model's memory for
    // its reads of KV only the SSD cache holds, which this names.
    std::optional<TokenAdmission> kv;
    std::string_view memory;
  };
  std::vector<Denied> denied;
  denied.reserve(plan.items.size());
  // A decoding lane reads KV only the SSD cache holds, through the hot block
  // cache.
  bool spilledReads = false;
  Request *const yielding = laneToYield();
  // A scheduled lane is resident: it grows as a request in service. The
  // lane that yields first if growth fails takes nothing in use, since its
  // own suspension, not another conversation's replay point, pays for it.
  // While other lanes fit, it waits for them instead, resident and denied
  // each step, as a start a resident lane holds back waits.
  const auto growthClass = [&](const Request &active) {
    return &active == yielding && anotherResident(active.request.id) ? ReclaimClass::Ordinary
                                                                     : ReclaimClass::InUse;
  };
  // A prefill command's KV reaches its first state boundary first; what
  // lies past it follows once that state has its buffers
  // (reserveCaptures).
  const auto workEndOf = [&](const Request &active, const BatchItem &scheduled) {
    uint64_t workEnd = active.exactTokens.size() + model::ExecutionLimits::targetVerifyRows;
    if (prefill) {
      workEnd = scheduled.promptOffset + scheduled.tokenCount;
      if (active.stateBoundaryCursor < active.stateBoundaries.size())
        workEnd = std::min<uint64_t>(
            workEnd, active.stateBoundaries[active.stateBoundaryCursor].tokens);
    }
    return workEnd;
  };
  for (const BatchItem &scheduled : plan.items) {
    Request &active = request(scheduled.requestId);
    if (!active.lane)
      throw std::logic_error("scheduled request is not resident");
    const uint64_t id = active.request.id;
    const uint64_t workEnd = workEndOf(active, scheduled);
    const auto kv = allocate([&] { return cache_.ensureTokens(id, workEnd); }, true,
                             growthClass(active), spillFallback(active));
    if (kv.admission.granted())
      admitted.push_back(scheduled);
    else
      denied.push_back(Denied{id, workEnd, kv.denial, kv.admission, {}});
  }
  // The model's memory for reads of the KV only the SSD cache holds: a
  // decoding lane's floor of the hot block cache, or the staging of a
  // prefill's history up to the command's end. It is reserved once every
  // lane has its KV, since a lane that converts to Spill moves the landed
  // pages of lanes admitted before it to the index form too
  // (Cache::convertToSpill), and only for the lanes the command keeps, in
  // the plan's order. A prefill lane that streams its history
  // (streamsHistory) runs alone and whole, as the scheduler runs one
  // (planPrefill, passesFor): kept first, it is the command, in one pass;
  // after a lane kept, it leaves for a command of its own. Admission may
  // have made it stream, converting it at once where its restored chain's
  // copies had landed, which the plan, made before, did not know. A lane
  // refused its memory leaves the command, and the lanes after it take that
  // shape without it.
  if (spill) {
    std::vector<BatchItem> kept;
    kept.reserve(admitted.size());
    for (const BatchItem &scheduled : admitted) {
      Request &active = request(scheduled.requestId);
      const uint64_t id = active.request.id;
      const bool streams = prefill && streamsHistory(active);
      if (streams && !kept.empty())
        continue;
      if (cache_.holdsIndexPages(id)) {
        const auto pages = static_cast<uint32_t>(
            (uint64_t{scheduled.promptOffset} + scheduled.tokenCount + KvCache::pageTokens - 1) /
            KvCache::pageTokens);
        const auto memory = allocate(
            [&] {
              return prefill ? model_.reserveHistoryStaging(id, pages)
                             : model_.reserveSpilledReads(id);
            },
            true, growthClass(active));
        if (!memory.admission.granted()) {
          denied.push_back(Denied{id, workEndOf(active, scheduled), memory.denial, std::nullopt,
                                  prefill ? "prefill history staging" : "hot KV cache"});
          continue;
        }
        spilledReads = spilledReads || !prefill;
      }
      kept.push_back(scheduled);
      if (streams) {
        plan.passes = 1;
        break;
      }
    }
    admitted = std::move(kept);
  }
  // A lane that runs is waiting for nothing.
  for (const BatchItem &scheduled : admitted)
    request(scheduled.requestId).resourceWait = {};
  // A lane refused the model's memory for its reads of spilled KV, which no
  // transfer in flight returns, takes it from the lane that yields first, as
  // when no lane fits, if that lane holds its memory outside this command
  // and yields before the refused one. The other lanes do not share that
  // refusal: run without the lane while they fit, they would leave it
  // refused for as long as they run. A lane refused its KV waits while they
  // fit, since they grow too and are refused in turn.
  if (yielding && std::ranges::none_of(plan.items, [&](const BatchItem &item) {
        return item.requestId == yielding->request.id;
      })) {
    const auto refused = std::ranges::find_if(denied, [&](const Denied &entry) {
      return !entry.kv && !entry.denial.pending &&
             yieldsBefore(*yielding, request(entry.requestId));
    });
    if (refused != denied.end()) {
      suspendForGrowth(*yielding, kvCapacity(yielding->request.id),
                       refused->denial.allocationFailure, now);
      return Prepared::Yielded;
    }
  }
  // The hot block cache they read through grows toward its planned size,
  // by an extent at most, into memory nothing needs now.
  if (spilledReads)
    static_cast<void>(allocate([&] { return model_.growSpilledReads(); }, false,
                               ReclaimClass::Ordinary, {}, true));
  if (!admitted.empty()) {
    plan.items = std::move(admitted);
    std::vector<std::vector<StateCapture>> states(plan.items.size());
    captures.assign(plan.items.size(), {});
    for (size_t index = 0; prefill && index < plan.items.size(); ++index) {
      BatchItem &item = plan.items[index];
      Request &active = request(item.requestId);
      const uint32_t rows =
          reserveCaptures(active, item, growthClass(active), states[index], captures[index]);
      if (rows != item.tokenCount) {
        item.tokenCount = rows;
        ++counters_.captureCuts;
      }
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
      const PageTableView pageTable = cache_.cutPageTable(active.request.id);
      item.pageTable = pageTable.pages;
      item.pageTableRevision = pageTable.revision;
      item.pageTableFirstChanged = pageTable.firstChanged;
      item.pageSources = pageTable.sources;
      if (prefill) {
        item.inputTokens =
            std::span<const uint32_t>(active.exactTokens)
                .subspan(scheduled.promptOffset, scheduled.tokenCount);
        item.stateCaptures = std::move(states[index]);
      }
      items.push_back(std::move(item));
    }
    if (plan.passes > 1)
      request(plan.items.front().requestId).passCommand = Request::PassCommand{items.front(), captures.front()};
    return Prepared::Runnable;
  }

  // Partial admissions execute at their actual width. If no lane fits, the
  // lane to yield is chosen among all runnable residents: an unstarted peer
  // can release its lane before completed prefill is discarded.
  if (denied.empty())
    throw std::logic_error("empty resource admission result");
  // Memory on its way back arrives without anyone yielding. The lanes still
  // take a retry deadline: the transfer's completion wakes the engine, and
  // the deadline is what makes the wait end if that wake is ever missed.
  if (std::any_of(denied.begin(), denied.end(),
                  [](const Denied &entry) { return entry.denial.pending; })) {
    for (const Denied &entry : denied)
      deferResourceRetry(request(entry.requestId), now, entry.denial);
    return Prepared::Waiting;
  }
  const Denied &victim = *std::min_element(
      denied.begin(), denied.end(),
      [&](const Denied &left, const Denied &right) {
        return yieldsBefore(request(left.requestId), request(right.requestId));
      });
  if (!yielding)
    throw std::logic_error("no resident lane can yield");
  Request &active = *yielding;
  const auto own = std::find_if(denied.begin(), denied.end(), [&](const Denied &entry) {
    return entry.requestId == active.request.id;
  });
  // A peer that has not failed a growth attempt resumes to its current KV
  // capacity, not to the blocked lane's requirement.
  const uint64_t resumeTarget =
      own != denied.end() ? own->workEnd : kvCapacity(active.request.id);
  if (judge(victim.denial, active.request.id) != Verdict::Fail) {
    suspendForGrowth(active, resumeTarget, victim.denial.allocationFailure, now);
    return Prepared::Yielded;
  }
  // Alone with nothing left to reclaim. A Spill lane that decodes ends at
  // its length once the SSD cache can hold no more of its KV.
  Request &lone = request(victim.requestId);
  if (!victim.kv)
    settle(lone, capacityExhausted(victim.memory, victim.denial.allocationFailure));
  else if (lone.kvMode == KvMode::Spill && !prefill)
    endAtLength(lone);
  else
    settle(lone, capacityExhausted("KV target", victim.denial.allocationFailure,
                                   pageShortfall(*victim.kv) +
                                       (spill ? "; the SSD cache took no more of its KV" : "")));
  return Prepared::Yielded;
}

uint32_t Engine::reserveCaptures(Request &active, const BatchItem &item, ReclaimClass upTo,
                                 std::vector<StateCapture> &states,
                                 std::vector<Capture> &captures) {
  const uint32_t start = item.promptOffset;
  // A request waiting for a prefix starts once the command that reaches its
  // state has ended.
  const uint32_t end = awaitedBoundary(active, start + item.tokenCount);
  const std::vector<Request::StateBoundary> &boundaries = active.stateBoundaries;
  const auto stateClass = [&](size_t index) {
    return boundaries[index].disposable                               ? StateClass::Checkpoint
           : boundaries[index].tokens == promptReplayBoundary(active) ? StateClass::InUse
                                                                      : StateClass::Ordinary;
  };
  // Takes the buffers of the state at boundaries[index]; false when refused.
  // Only the first state the command captures follows the lane's
  // checkpoint, and each later one the state before it.
  const auto reserve = [&](size_t index) {
    const uint32_t tokens = boundaries[index].tokens;
    const StateClass kind = stateClass(index);
    if (kind == StateClass::InUse && !states.empty() && boundaries[index - 1].disposable) {
      // The prompt's replay point keeps the checkpoint the command captures
      // before it beside buffers of its own, and takes that checkpoint's
      // when refused the pool and the admission, as it takes its lane's
      // after a command.
      metal::AllocationFailure refusal;
      if (std::shared_ptr<StateReservation> buffers =
              model_.reserveState(active.request.id, tokens, refusal)) {
        states.push_back({tokens, std::move(buffers)});
        captures.push_back({tokens});
      } else {
        states.back().tokens = tokens;
        captures.back().tokens = tokens;
      }
      return true;
    }
    Reservation reservation = reserveCapture(active, kind, tokens, states.empty());
    if (!reservation.buffers) {
      if (reservation.fromLane)
        captures.push_back({tokens, false, true});
      return false;
    }
    states.push_back({tokens, std::move(reservation.buffers)});
    captures.push_back({tokens, reservation.recycled});
    return true;
  };
  for (size_t index = active.stateBoundaryCursor;
       index < boundaries.size() && boundaries[index].tokens <= end; ++index) {
    const uint32_t tokens = boundaries[index].tokens;
    // Published, every state but the prompt's replay point retires the
    // checkpoint before it (apply): a checkpoint the command passes on its
    // way to such a state is not captured, and that state takes buffers of
    // its own class.
    const bool passed = boundaries[index].disposable && index + 1 < boundaries.size() &&
                        boundaries[index + 1].tokens <= end &&
                        stateClass(index + 1) != StateClass::InUse;
    if (!passed && !reserve(index))
      return tokens - start;
    if (tokens == end)
      break;
    // The KV up to the next boundary, as prepare() takes the first, but its
    // refusal only ends the command here, which then captures a checkpoint
    // it would have passed: the lane runs, and its next command asks again.
    const uint32_t next = index + 1 < boundaries.size() && boundaries[index + 1].tokens < end
                              ? boundaries[index + 1].tokens
                              : end;
    const auto kv = allocate(
        [&] { return cache_.ensureTokens(active.request.id, next); }, true, upTo, {}, true);
    if (!kv.admission.granted()) {
      if (passed)
        static_cast<void>(reserve(index));
      return tokens - start;
    }
  }
  return end - start;
}

Engine::Reservation Engine::reserveCapture(Request &active, StateClass kind, uint32_t tokens,
                                           bool followsCheckpoint) {
  const uint64_t id = active.request.id;
  metal::AllocationFailure refusal = metal::AllocationFailure::None;
  Reservation result{model_.reserveState(id, tokens, refusal)};
  if (result.buffers)
    return result;
  // The lane's checkpoint is its recovery point until a state that retires
  // it is published, after a command that completed: one cancelled, expired
  // or failed leaves it to a retry. So no state takes its buffers before the
  // command, nor room a reclaim could make of it, since checkpoints go
  // first. Refused the pool and the admission while the lane holds one, the
  // state ends its command, and apply takes it from the lane after the
  // command: a junction or a checkpoint in the buffers of the checkpoint it
  // retires (publishReachedStateBoundaries). A checkpoint whose predecessor
  // a reader pins takes no room either: apply publishes it, from the lane,
  // only once the pin has gone, which keeps the predecessor the lane's
  // recovery point until then.
  const bool predecessorPinned =
      kind == StateClass::Checkpoint && followsCheckpoint && checkpointPinned(active);
  result.fromLane = predecessorPinned || checkpointReclaimable(active);
  if (result.fromLane)
    return result;
  // Room comes from what this state's class may take: cached KV unless it
  // is an optional checkpoint, and states in use only for the prompt's
  // replay point, which is in use itself. A pin keeps the lane's
  // checkpoint, if any, out of it.
  result.recycled = makeStateRoom(kind, [&] {
    result.buffers = model_.reserveState(id, tokens, refusal);
    return result.buffers != nullptr;
  });
  // The prompt's replay point is work of the request in service, the point
  // its follow-up resumes from: once nothing its class may take is left,
  // the host refuses it no more than that request's KV (allocate()). A
  // junction or a checkpoint stays optional.
  if (!result.buffers && kind == StateClass::InUse &&
      refusal == metal::AllocationFailure::HostPressure) {
    const Serving mark(config_.serving, true);
    result.buffers = model_.reserveState(id, tokens, refusal);
  }
  return result;
}

template <class Take> bool Engine::makeStateRoom(StateClass kind, Take &&take) {
  // No command is in flight, so KV that empties an extent releases it now.
  // A recycled state hands over its buffers; an extent may hold less than a
  // state, so room is made until the buffers fit, nothing more of the class
  // goes, or the extents given cover one state: a refusal after that is not
  // the budget's.
  const bool growth = !config_.growthPaused();
  const uint64_t needed = model_.snapshotBytes();
  uint64_t released = 0;
  StateRoom room;
  do {
    room = cache_.reclaimOneState(kind, growth);
    released += room.extentBytes;
    if (room && take())
      return true;
  } while (room.extentBytes && released < needed);
  return false;
}

uint64_t Engine::completedTokens(const Request &active) const {
  return scheduler_.phase(active.request.id) == Phase::Prefill
             ? scheduler_.promptProcessed(active.request.id)
             : active.exactTokens.size();
}

uint32_t Engine::generatedTokens(const Request &active) noexcept {
  return static_cast<uint32_t>(active.exactTokens.size() - active.promptTokens) +
         (active.sentFirstToken ? 1 : 0);
}

bool Engine::yieldsBefore(const Request &a, const Request &b) const {
  if (a.request.priority != b.request.priority)
    return a.request.priority > b.request.priority;
  const Phase aPhase = scheduler_.phase(a.request.id);
  const Phase bPhase = scheduler_.phase(b.request.id);
  // At equal priority, prefer uninterrupted streaming over less replay work.
  if (aPhase != bPhase)
    return aPhase == Phase::Prefill;
  return completedTokens(a) < completedTokens(b);
}

Engine::Request *Engine::laneToYield() {
  Request *yielding = nullptr;
  for (auto &[id, candidate] : requests_) {
    if (!candidate.lane)
      continue;
    // Requests enter the scheduler before they can acquire a lane.
    const Phase phase = scheduler_.phase(id);
    if (phase != Phase::Prefill && phase != Phase::Decode)
      continue;
    if (!yielding || yieldsBefore(candidate, *yielding) ||
        (!yieldsBefore(*yielding, candidate) &&
         scheduler_.submissionOrder(id) > scheduler_.submissionOrder(yielding->request.id)))
      yielding = &candidate;
  }
  return yielding;
}

uint64_t Engine::kvCapacity(uint64_t requestId) const {
  return uint64_t{cache_.pageTable(requestId).pages.size()} * KvCache::pageTokens;
}

bool Engine::preemptBelow(RequestPriority priority, double now) {
  // The lane that yields first has the lowest priority of all residents.
  Request *const victim = laneToYield();
  if (!victim || victim->request.priority <= priority)
    return false;
  suspendLane(*victim, kvCapacity(victim->request.id), metal::AllocationFailure::None,
              StateFailure::ConcurrencyLimit, now);
  // What the lane held is free: an earlier failure may fit now.
  allocationFailed_ = false;
  ++counters_.prioritySuspensions;
  signalResourceProgress();
  return true;
}

bool Engine::anotherResident(uint64_t requestId) const {
  return std::any_of(requests_.begin(), requests_.end(), [&](const auto &entry) {
    return entry.first != requestId && entry.second.lane;
  });
}

Engine::Verdict Engine::judge(const Denial &denial, uint64_t requestId) const {
  if (denial.pending)
    return Verdict::Wait;
  // Pages held by resident lanes come back when they finish, and memory the
  // host refuses when its pressure lifts; only a lane that cannot fit on its
  // own has hit the capacity.
  if (anotherResident(requestId) ||
      denial.allocationFailure == metal::AllocationFailure::HostPressure)
    return Verdict::Yield;
  return Verdict::Fail;
}

template <class Attempt>
auto Engine::allocate(Attempt &&attempt, bool inService, ReclaimClass upTo,
                      const std::function<bool(const Denial &)> &fallback, bool optional)
    -> Allocation<std::invoke_result_t<Attempt &>> {
  using Admission = std::invoke_result_t<Attempt &>;
  // Set once a request in service has nothing held left to reuse: from then
  // on only the engine's limit and critical pressure refuse it.
  bool serving = false;
  const auto tryOnce = [&] {
    const Serving mark(config_.serving, serving);
    return attempt();
  };
  Allocation<Admission> result{tryOnce(), {}};
  Admission &admission = result.admission;
  Denial &denial = result.denial;
  while (memoryDenied(admission)) {
    const bool paused =
        admission.allocationFailure == metal::AllocationFailure::HostPressure;
    CacheReclaimResult reclaimed;
    if constexpr (std::is_same_v<Admission, StateAdmission>)
      reclaimed = paused ? reuseCachedStateWhilePaused(upTo) : reclaimForState(upTo);
    else if constexpr (std::is_same_v<Admission, TokenAdmission>)
      reclaimed = paused ? reuseCachedPagesWhilePaused(admission, upTo)
                         : reclaimForKv(admission, upTo);
    else
      // The model's memory for reads of spilled KV (a hot block cache floor,
      // a history's staging) takes what a state takes; while the host
      // refuses growth, idle buffers alone, and memory for later (that
      // cache's optional growth) nothing.
      reclaimed = paused ? CacheReclaimResult{!optional &&
                                              reclaimIdleState(true, model::IdleMemory::Buffers)}
                         : reclaimForState(upTo, optional);
    if (reclaimed.madeProgress) {
      admission = tryOnce();
      continue;
    }
    denial.allocationFailure = admission.allocationFailure;
    denial.pending = reclaimed.pending;
    if (paused && inService && !serving && !reclaimed.pending) {
      serving = true;
      admission = tryOnce();
      continue;
    }
    if (fallback && fallback(denial)) {
      admission = tryOnce();
      continue;
    }
    break;
  }
  if (!admission.granted()) {
    denial.allocationFailure = admission.allocationFailure;
    denial.pending = denial.pending || transferPending(admission);
    // Memory on its way back ends the shortage without the residents.
    if (memoryDenied(admission) && !denial.pending && !optional)
      allocationFailed_ = true;
  }
  return result;
}

// The reclaim step for a lane's state the engine's limit refused. The pooled
// buffers a lane starts from stay for its activation to take: idle model
// memory beyond them goes first (a pooled buffer, else the idle vision
// arena, else one rebuildable cache entry), then the cache gives up what
// gives memory back: an extent, or a state (Cache::reclaimForMemory), which
// makes progress, and the allocation is asked for again, only once memory
// has come back. The KV it evicts before then frees pages other KV may
// take, so it wakes the requests waiting for memory as progress does.
// Memory for later (the hot block cache's growth toward its planned size)
// takes no rebuildable cache entry, among which that very cache's slots past
// its floor are, nor the empty extent kept for the next request.
// A host refusal comes to neither step but to the reuse path (allocate()):
// the pressure controller owns that shrink, and evicting for an allocator
// that refuses all the same would drain the cache, and the caches the model
// can rebuild, before macOS can acknowledge any reclaimed bytes.
CacheReclaimResult Engine::reclaimForState(ReclaimClass upTo, bool forLater) {
  if (reclaimIdleState(true, forLater ? model::IdleMemory::Buffers
                                      : model::IdleMemory::BuffersThenCaches))
    return {true, 0};
  const CacheReclaimResult reclaimed = cache_.reclaimForMemory(upTo, forLater);
  if (reclaimed.madeProgress || reclaimed.evictedKv)
    signalResourceProgress();
  return reclaimed;
}

// The reclaim step for KV pages the engine's limit refused. Allocated extents
// stay for the pages to reuse; idle state memory goes first, then the cache
// gives up what covers the shortfall, of the pages' form, in one step.
CacheReclaimResult Engine::reclaimForKv(const TokenAdmission &admission, ReclaimClass upTo) {
  if (reclaimIdleState(false, model::IdleMemory::BuffersThenCaches))
    return {true, 0};
  const CacheReclaimResult reclaimed = cache_.reclaimForPages(
      admission.additionalPages, CacheReclaimMode::KeepExtents, upTo, admission.form);
  if (reclaimed.madeProgress)
    signalResourceProgress();
  return reclaimed;
}

bool Engine::reclaimIdleState(bool keepLane, model::IdleMemory scope) noexcept {
  if (!model_.reclaimIdleState(keepLane, scope))
    return false;
  signalResourceProgress();
  return true;
}

// While growth is paused a lane short of state buffers takes a cached
// state's, as a request short of pages takes idle cached pages below:
// evicting the state returns its cell and context window to the pool the
// lane draws from, and nothing is allocated. A state goes only when those in
// RAM cover what the pool lacks, which they never do for the draft rings no
// cached state holds; otherwise the cache survives, and the request grows if
// it is in service and waits if it is not. Idle buffers beyond the lane's go
// back to the host first, but no cache the model can rebuild: freeing one
// would not let the lane start, and the paced pass takes those caches toward
// what the host lacks.
CacheReclaimResult Engine::reuseCachedStateWhilePaused(ReclaimClass upTo) {
  if (reclaimIdleState(true, model::IdleMemory::Buffers))
    return {true, 0};
  const uint32_t lacked = model_.statesToActivate();
  if (!lacked || cache_.evictableStates(upTo) < lacked)
    return {};
  const CacheReclaimResult reused = cache_.reclaimStateForLane(upTo);
  if (reused.madeProgress)
    signalResourceProgress();
  return reused;
}

// Host pressure pauses growth, and the pressure controller owns the shrink.
// Extents that stay allocated are outside that accounting: a request short
// of pages takes idle cached pages before it grows or waits. Cache is only
// evicted when the pages of the shortfall's form the request's class may
// take can actually cover it: never those a request holds, nor, for a start
// a resident lane holds back, the idle KV that states in use restore through
// (Cache::reusablePages). Otherwise it survives for later hits, and the
// request grows if it is in service and waits if it is not. Only KV goes: a
// state's buffers give no page while the host refuses growth, unless the
// state sits on the leaf that goes next (CacheReclaimMode::ReusePages). A
// reclaim that must wait for the transfer in flight makes the request wait
// with it, as it does without the pause. Idle buffers go first, but neither
// the pooled buffers the next lane starts from nor the caches the model can
// rebuild: they would not let this request grow, and the paced pass keeps
// the buffers for the next lane and takes the caches toward what the host
// lacks.
CacheReclaimResult Engine::reuseCachedPagesWhilePaused(const TokenAdmission &admission,
                                                       ReclaimClass upTo) {
  if (reclaimIdleState(true, model::IdleMemory::Buffers))
    return {true, 0};
  if (cache_.reusablePages(upTo, admission.form) < admission.additionalPages)
    return {};
  const CacheReclaimResult reused = cache_.reclaimForPages(
      admission.additionalPages, CacheReclaimMode::ReusePages, upTo, admission.form);
  if (reused.madeProgress)
    signalResourceProgress();
  return reused;
}

void Engine::suspendLane(Request &active, uint64_t workEnd, metal::AllocationFailure failure,
                         StateFailure reason, double now) {
  if (!active.lane || scheduler_.suspended(active.request.id)) {
    throw std::logic_error("request cannot be suspended");
  }
  // A decoding lane holds an anchor unless its first mask has yet to select
  // one; a lane in prefill holds what it held when its replay began.
  if (scheduler_.phase(active.request.id) == Phase::Decode)
    active.suspendedWithAnchor =
        scheduler_.decodeStage(active.request.id) == DecodeStage::Regular;
  vacateLane(active, true);
  active.resumeKvTargetTokens = workEnd;
  active.replayTokens = static_cast<uint32_t>(active.exactTokens.size());
  scheduler_.suspendForResources(active.request.id);
  deferResourceRetry(active, now, {.allocationFailure = failure}, reason);
}

void Engine::suspendForGrowth(Request &active, uint64_t workEnd,
                              metal::AllocationFailure failure, double now) {
  suspendLane(active, workEnd, failure, StateFailure::MemoryPressure, now);
  // Without its lane it waits for the memory of every lane resident now,
  // also those admitted while it waited to grow.
  active.resourceWait.admittedBefore = admissions_;
  // Resident lanes drain before admission resumes. Growth the host refused
  // resumes when its pressure lifts; any other limit only once memory is
  // freed, so it counts as a failure the drain waits out.
  drainEndMilliseconds_ = now + config_.resourceWaitTimeoutMilliseconds;
  allocationFailed_ = failure != metal::AllocationFailure::HostPressure;
  ++counters_.resourceSuspensions;
}

MemoryReclaimResult Engine::reclaimMemory(const MemoryReclaimDirective &directive) {
  if (pending_)
    throw std::logic_error("memory reclaim requested while a command is in flight");
  if (cache_.pollTransfers())
    signalResourceProgress();

  const bool keep = !directive.critical;
  uint64_t released = 0;
  // A reclaim step may free memory that stays in the engine, pages of an
  // extent that stays or buffers that refill the lane's footprint, and a
  // waiting request may fit in it all the same.
  bool reclaimed = false;
  // Pages whose copies are being written count toward the target. A
  // critical directive has none: it takes everything it may.
  const auto targetUnmet = [&] {
    return directive.critical || released + cache_.pendingBytes() < directive.targetBytes;
  };
  // Evicted states park their buffers in the model's pool, which a pass
  // returns to the host at once, or uses to refill the buffers it keeps.
  const auto releaseIdle = [&] {
    while (const uint64_t idle = model_.reclaimIdleState(keep, model::IdleMemory::Buffers))
      released += idle;
  };
  releaseIdle();
  // Caches the model can rebuild go only toward a byte target or under
  // critical pressure: an untargeted pass keeps the embedding rows and the
  // vision encoder.
  const auto reclaimModelCaches = [&] {
    while (targetUnmet()) {
      const uint64_t cache =
          model_.reclaimIdleState(keep, model::IdleMemory::BuffersThenCaches);
      if (!cache)
        break;
      released += cache;
    }
  };
  reclaimModelCaches();
  if (directive.critical) {
    const CacheReclaimResult evicted = cache_.evictAll();
    reclaimed = evicted.madeProgress;
    released += evicted.reclaimedBytes;
    releaseIdle();
  } else {
    // Even a zero-byte directive releases completely empty KV extents.
    released += cache_.releaseEmptyExtents(keep);
    while (targetUnmet()) {
      const CacheReclaimResult step =
          cache_.reclaimOne(CacheReclaimMode::ReleaseExtents, directive.upTo,
                            directive.keepResumePoint, keep);
      if (!step.madeProgress)
        break;
      reclaimed = true;
      released += step.reclaimedBytes;
      releaseIdle();
    }
  }
  // Image rows only an evicted state held are ordinary cache entries now.
  reclaimModelCaches();
  if (released || reclaimed)
    signalResourceProgress();
  if (!directive.critical && !directive.targetBytes)
    return {released, ReclaimOutcome::Untargeted};
  if (!targetUnmet())
    return {released, ReclaimOutcome::Met};
  return {released, cache_.transfersInFlight() ? ReclaimOutcome::Pending
                                               : ReclaimOutcome::Exhausted};
}

void Engine::apply(const Pending &command, std::span<const ModelStepResult> results,
                   double cycleMilliseconds, double now) {
  const BatchPlan &plan = command.plan;
  const std::span<const std::vector<Capture>> captures = command.captures;
  const ModelBatchTicket &ticket = *command.ticket;
  if (results.size() != plan.items.size()) {
    throw std::logic_error("model result count changed");
  }
  // A pass before the command's last commits nothing and proves nothing.
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
    // vocabulary, without reporting it fails the lane the same way (the Qwen
    // runtime reports its non-finite rows itself).
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
      if (!active.pendingEnd)
        active.pendingEnd = LaneEnd{LaneOutcome::KvUnreadable, std::string(kKvUnreadableMessage)};
    } else if (!invalid.empty()) {
      if (active.unprovenState)
        cache_.invalidateState(*active.unprovenState);
      if (!active.pendingEnd)
        active.pendingEnd = LaneEnd{LaneOutcome::ModelResultInvalid, std::move(invalid)};
    }
    if (!beforeLastPass)
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
    if (beforeLastPass) {
      schedulerResults.push_back({active.request.id, 0, false, result.nextDecodeStage});
      continue;
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
      const std::vector<Capture> &reached = captures[index];
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
      if (!unsent.empty())
        events_.tokens(active.request.id, unsent);
    }
    if (result.firstToken) {
      if (active.sentFirstToken || !result.outputTokens.empty())
        throw std::logic_error(
            "model reported a first token twice or beside output");
      active.sentFirstToken = result.firstToken;
      events_.tokens(active.request.id, {&*result.firstToken, 1});
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
    // SSD cache's quota bounded ends at that bound.
    const bool complete =
        result.finished || (!scoring && completionTokens >= active.outputLimit);
    counters_.spill.lengthEnds +=
        complete && !result.finished && active.outputLimit < active.request.maxNewTokens;
    if (scoring && complete &&
        item.promptOffset + result.consumedPromptTokens !=
            active.promptTokens) {
      throw std::logic_error("score request finished before the prompt ended");
    }
    if (result.outputTokensWithoutKv && !complete) {
      throw std::logic_error("model emitted an uncommitted token and continued");
    }
    schedulerResults.push_back({active.request.id, result.consumedPromptTokens,
                                complete, result.nextDecodeStage});
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
  const double wallMilliseconds = ticket.wallMilliseconds();
  scheduler_.complete(plan, schedulerResults, wallMilliseconds,
                      ticket.prefillTimingIsRepresentative(), ticket.hostMilliseconds(),
                      command.streamed ? std::optional(ticket.streamMilliseconds())
                                       : std::nullopt);
  if (plan.passes > 1 && plan.pass + 1 == plan.passes)
    request(plan.items.front().requestId).passCommand.reset();
  events_.batchCompleted(plan.kind, plan.width(), inputTokens, outputTokens,
                         draftedTokens, acceptedDraftTokens, wallMilliseconds,
                         cycleMilliseconds);

  for (size_t index = 0; index < results.size(); ++index) {
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

void Engine::settle(Request &active, LaneEnd end) {
  if (active.finalized)
    return;
  if (active.restore || inFlight(active.request.id)) {
    if (!active.pendingEnd)
      active.pendingEnd = std::move(end);
    if (!active.restore)
      pending_->ticket->abandonMask(active.request.id);
    else if (active.restore->ticket)
      active.restore->ticket->cancel();
    return;
  }
  if (end.outcome == LaneOutcome::Cancelled)
    finish(active, EngineFinishReason::Cancelled, {});
  else
    finishFailure(active, std::move(end));
}

Engine::LaneEnd Engine::deadlineEnd() {
  return {LaneOutcome::DeadlineExceeded, std::string(kDeadlineExceededMessage)};
}

Engine::LaneEnd Engine::capacityExhausted(std::string_view what,
                                          metal::AllocationFailure failure,
                                          std::string_view detail) {
  std::string message = "could not allocate " + std::string(what) + ": " +
                        metal::allocationFailureName(failure);
  if (!detail.empty())
    message += " (" + std::string(detail) + ")";
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
  // its output: the quota's (outputLimit) or a lone lane's at the wall.
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
  // A command between its passes is left, with the states it would capture.
  active.passCommand.reset();
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

void Engine::sweepTerminal() {
  for (auto iterator = requests_.begin(); iterator != requests_.end();) {
    const uint64_t id = iterator->first;
    Request &active = iterator->second;
    // Scheduler::expireDeadlines is the only source of a Failed phase on an
    // unfinalized request.
    if (scheduler_.phase(id) == Phase::Failed && !active.finalized)
      settle(active, deadlineEnd());
    if (!active.finalized) {
      ++iterator;
      continue;
    }
    scheduler_.remove(id);
    iterator = requests_.erase(iterator);
  }
}

Engine::Request &Engine::request(uint64_t id) {
  auto found = requests_.find(id);
  if (found == requests_.end())
    throw std::out_of_range("unknown request");
  return found->second;
}

} // namespace splash::engine

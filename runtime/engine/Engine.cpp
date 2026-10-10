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
// A junction costs a snapshot, a command split and up to a draft window of
// draft-context rows; a later request must save at least that much prefill.
constexpr uint32_t kMinimumJunctionGain = model::ExecutionLimits::draftContextTokens;

// A refusal for memory, which reclaim or the host's recovery may end.
bool memoryDenied(const StateAdmission &admission) noexcept {
  return admission.failure == StateFailure::MemoryPressure;
}

bool memoryDenied(const TokenAdmission &admission) noexcept {
  return admission.failure == TokenAdmissionFailure::Denied;
}

// Memory a transfer in flight holds, which returns by itself.
bool transferPending(const StateAdmission &) noexcept { return false; }

bool transferPending(const TokenAdmission &admission) noexcept {
  return admission.failure == TokenAdmissionFailure::Pending;
}

// What a KV target lacked when it could not be allocated.
std::string pageShortfall(const TokenAdmission &admission) {
  return "additional_pages=" + std::to_string(admission.additionalPages) +
         ", free_pages=" + std::to_string(admission.availablePages);
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
    : config_(config), cache_(cache), writeBehind_(cache), model_(model),
      events_(events), scheduler_(config_.decodeShare) {
  if (!config_.maxContext || !config_.vocabularySize) {
    throw std::invalid_argument("context and vocabulary sizes must be positive");
  }
  if (!config_.growthPaused || !config_.serving)
    throw std::invalid_argument("the engine needs the governor's growth pause and serving mark");
  if (!std::isfinite(config_.decodeShare) || config_.decodeShare < 0.0)
    throw std::invalid_argument("decode share must be nonnegative and finite");
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
  requestState.request = std::move(value);
  const Request &stored =
      requests_.emplace(id, std::move(requestState)).first->second;
  scheduler_.submit(
      {.id = id,
       .priority = stored.request.priority,
       .constrained = stored.request.constraint == ConstraintMode::TokenMask,
       .prefillTokens = stored.promptTokens,
       .deadlineMilliseconds = stored.request.deadlineMilliseconds});
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
    signalResourceProgress();
    progressed = true;
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
                      resourceTimeoutMessage(active.resourceWait.allocationFailure)});
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
    apply(command.plan, results, command.ticket->wallMilliseconds(),
          now - command.startedMilliseconds,
          command.ticket->prefillTimingIsRepresentative(), now);
    busySinceMilliseconds_ = now;
    sweepTerminal();
    return true;
  }

  progressed = admitQueued(now) || progressed;
  sweepTerminal();
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
    switch (prepare(*plan, items, now)) {
    case Prepared::Runnable: {
      std::unique_ptr<ModelBatchTicket> ticket =
          model_.submit(*plan, items, completionNotifier_);
      if (!ticket) {
        throw std::logic_error("model returned an empty command ticket");
      }
      scheduler_.commit(*plan, excluded);
      pending_ = Pending{std::move(*plan), std::move(ticket),
                         busySinceMilliseconds_.value_or(now)};
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
  return found != requests_.end() && !found->second.finalized && found->second.refusedMemory;
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
    result.activeRequests.push_back(
        {.id = id,
         .phase = scheduler_.phase(id),
         .priority = active.request.priority,
         .promptTokens = active.promptTokens,
         .promptProcessed =
             std::min(scheduler_.promptProcessed(id), active.promptTokens),
         .generatedTokens = static_cast<uint32_t>(active.exactTokens.size() -
                                                  active.promptTokens),
         .maxNewTokens = active.request.maxNewTokens});
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
    // request refused memory while a pass keeps it out of its memory wait.
    if (holdsBack(active.heldBehind) || (active.refusedMemory && !waiting))
      ++result.heldBehindRefusal;
    if (!waiting)
      continue;
    if (active.resourceWait.reason == StateFailure::ConcurrencyLimit)
      ++result.concurrency;
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
      cache_.refresh(*active.admissionProbe, active.exactTokens, active.request.images);
    else
      active.admissionProbe = cache_.probe(active.exactTokens, active.request.images);
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
      if (active.refusedMemory)
        refused = id;
    }
    return false;
  }

  // A request whose start was refused memory closes admission behind it
  // until it starts: admission is open for order[0, open). What reclaim and
  // finishing lanes free would otherwise keep going to later arrivals that
  // need less of it at once, and it would wait for as long as they keep
  // coming. Waiting for a free lane closes nothing: nothing starts without
  // one.
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
    if (active.refusedMemory)
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
      // Refused memory in this pass, it closes admission behind it at once.
      if (request(id).refusedMemory)
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
    const uint32_t shared = sharedPrefillBoundary(active, peer);
    for (size_t i = peer.stateBoundaryCursor; i < peer.stateBoundaries.size(); ++i) {
      const uint32_t boundary = peer.stateBoundaries[i].tokens;
      if (boundary > resumeBoundary && boundary <= shared)
        return true;
    }
  }
  return false;
}

bool Engine::admit(Request &active, double now) {
  const bool resuming = scheduler_.suspended(active.request.id);
  ModelRequest modelRequest = active.request.modelView();
  modelRequest.prompt = active.exactTokens;
  CacheLookup lookup =
      active.skipCache
          ? CacheLookup{}
          : cache_.lookup(modelRequest.prompt, active.request.images,
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
  active.refusedMemory = false;
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
    active.refusedMemory = state.admission.failure == StateFailure::MemoryPressure;
    scheduler_.waitForResources(active.request.id);
    deferResourceRetry(active, now, state.denial, state.admission.failure);
    // A start refused memory takes it from a resident of a lower priority,
    // whose suspension makes it retry at the next pass. Memory on its way
    // back makes no lane yield: the start waits for it.
    if (active.refusedMemory && !resuming && !state.denial.pending)
      static_cast<void>(preemptBelow(active.request.priority, now));
    return false;
  }
  cache_.beginRequest(active.request.id);
  active.lane = *state.admission.lane;
  active.admission = ++admissions_;
  if (droppedLease)
    lookup = cache_.lookup(modelRequest.prompt, active.request.images);
  const uint32_t resumeBoundary = lookup.resumeBoundary();
  const uint64_t requestId = active.request.id;
  // The matched chain first, then the first work's pages for a lane that
  // will not go through ordinary prefill admission before it runs: one
  // that resumes, or one that waits for a restore. As for its lane, beside
  // a resident request its pages wait for the host.
  Allocation<TokenAdmission> kv;
  if (lookup.state)
    kv = allocate([&] { return cache_.restoreRequest(requestId, lookup); }, inService, upTo);
  const bool restoring =
      lookup.state && (!lookup.state->state()->residentBytes() ||
                       cache_.kvRestoreStatus(requestId) == KvRestoreStatus::Pending);
  if (kv.admission.granted() && (resuming || restoring)) {
    const uint64_t workEnd =
        resuming ? active.resumeKvTargetTokens : uint64_t{resumeBoundary} + 1;
    kv = allocate([&] { return cache_.ensureTokens(requestId, workEnd); }, inService, upTo);
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
    active.refusedMemory = true;
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

void Engine::completeAdmission(Request &active, CacheLookup &lookup,
                                DraftContextPlan draft) {
  const bool resuming = scheduler_.suspended(active.request.id);
  active.skipCache = false;
  active.admissionProbe.reset();
  const uint32_t resumeBoundary = lookup.resumeBoundary();
  if (resuming)
    scheduler_.resumeFromResources(active.request.id, resumeBoundary, active.replayTokens);
  active.latestCheckpoint = {};
  if (lookup.state) {
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
    armNextStateBoundary(active);
    ++counters_.resourceResumptions;
    return;
  }
  scheduler_.resourcesReady(active.request.id, resumeBoundary);
  armNextStateBoundary(active);
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
  if (reason == StateFailure::ConcurrencyLimit)
    wait.deadlineMilliseconds = 0.0;
  else if (progressed || wait.deadlineMilliseconds <= 0.0)
    wait.deadlineMilliseconds = now + config_.resourceWaitTimeoutMilliseconds;
  wait.epoch = resourceEpoch_;
  wait.retryMilliseconds = now + kResourceRetryBackoffMilliseconds;
}

void Engine::deferWait(Request &active) noexcept {
  ResourceWait kept;
  if (active.refusedMemory) {
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
  // complete draft state. Progress points remain disposable after restoration.
  for (const uint32_t checkpoint : plannedCheckpoints(
           stateBoundary, latestReplayBoundary, config_.prefillCheckpointTokens)) {
    addStateBoundary(active, stateBoundary, checkpoint, true);
  }
  if (junctionBoundary >= stateBoundary + kMinimumJunctionGain)
    addStateBoundary(active, stateBoundary, junctionBoundary, false);
  addStateBoundary(active, stateBoundary, latestReplayBoundary,
                   latestReplayBoundary != promptReplayBoundary(active));
  // A resumed lane below its prompt's replay point lost that state; it
  // rebuilds the one its conversation's next turn resumes from on the way.
  addStateBoundary(active, stateBoundary, promptReplayBoundary(active), false);

  static_cast<void>(addSharedPrefillBoundaries(active, stateBoundary));
  return pendingDraftStatePlan(active, stateBoundary);
}

bool Engine::addStateBoundary(Request &active, uint32_t after, uint32_t tokens,
                              bool disposable) {
  // A state past the replay boundary would lie inside the generation prompt,
  // which a later request may not share.
  if (tokens <= after || tokens > replayStateBoundary(active))
    return false;
  const auto found = std::lower_bound(
      active.stateBoundaries.begin() + active.stateBoundaryCursor,
      active.stateBoundaries.end(), tokens,
      [](const Request::StateBoundary &point, uint32_t value) {
        return point.tokens < value;
      });
  if (found != active.stateBoundaries.end() && found->tokens == tokens) {
    found->disposable = found->disposable && disposable;
    return false;
  }
  active.stateBoundaries.insert(found, {tokens, disposable});
  return true;
}

bool Engine::addSharedPrefillBoundaries(Request &active, uint32_t after) {
  bool changed = false;
  for (const auto &[id, peer] : requests_) {
    // A request finalized earlier in this admission pass is still listed
    // until the pass ends. A request that ignores the cache uses no junction.
    if (id == active.request.id || peer.lane || scheduler_.suspended(id) ||
        peer.finalized || peer.skipCache ||
        peer.request.priority < active.request.priority)
      continue;
    changed = addStateBoundary(active, after, sharedPrefillBoundary(active, peer), false) ||
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

void Engine::armNextStateBoundary(Request &active) {
  const std::optional<uint32_t> next =
      active.stateBoundaryCursor < active.stateBoundaries.size()
          ? std::optional<uint32_t>(
                active.stateBoundaries[active.stateBoundaryCursor].tokens)
          : std::nullopt;
  scheduler_.setPrefillBoundary(active.request.id, next);
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

bool Engine::retireCheckpoint(Request &active) {
  // Shared progress points remain disposable under memory pressure, but a
  // lane's normal rolling replacement must not retire its peer's recovery point.
  if (!checkpointShared(active) && !cache_.retireCheckpointState(active.latestCheckpoint))
    return false;
  active.latestCheckpoint = {};
  return true;
}

void Engine::publishReachedStateBoundaries(Request &active,
                                           uint32_t promptProcessed, double now) {
  bool materialized = false;
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
    // The scheduler ends a command exactly at an armed boundary
    // (Scheduler::planPrefill).
    if (objective.tokens != promptProcessed)
      throw std::logic_error("prefill crossed an armed state boundary");
    materialized = true;
    const uint64_t block = cache_.blockAt(active.request.id, objective.tokens);
    // The state is in use before any of the ways below keeps it, so each of
    // them makes room as work in use.
    if (replay)
      active.replayPoint = cache_.useState(block);
    if (cache_.reuseCompositeState(block, checkpoint)) {
      ++counters_.deduplicatedStatePublications;
    } else {
      // The previous recovery point retires here or after the publication.
      // Here, before the snapshot, a resident one hands its buffers to its
      // replacement; room made for the snapshot instead would recycle the
      // oldest state, perhaps another lane's checkpoint. A restore lease
      // can delay this optional publication. A checkpoint only on disk
      // frees no cache slot for an ordinary state, so it stays the recovery
      // point until that state is published; it retires after the
      // publication, as does the one a reused state leaves. The prompt's
      // replay point keeps the lane's checkpoint (below) only beside a
      // snapshot that fits: refused room, it hands a resident one's buffers
      // to the snapshot before making room, unless a peer still holds it.
      const bool resident = cache_.stateResident(active.latestCheckpoint.kvBlock);
      if ((checkpoint || (!replay && resident)) && !retireCheckpoint(active) && checkpoint) {
        ++failures;
        continue;
      }
      std::shared_ptr<const CompositeState> state = model_.snapshot(active.request.id);
      if (!state && replay && resident && !checkpointShared(active) &&
          retireCheckpoint(active))
        state = model_.snapshot(active.request.id);
      // Room comes from what this publication's class may take: cached KV
      // unless it is an optional checkpoint, and states in use only for a
      // block in use. A state in use is never dropped for a busy write
      // slot; this publication gives way instead.
      // The command that reached this boundary is consumed and the next one
      // not yet submitted, so KV that empties an extent releases it now. A
      // recycled state hands over its buffers; an extent may hold less
      // than a state, so room is made until the snapshot fits, nothing
      // more of the class goes, or the extents given cover one snapshot:
      // a denial after that is not the budget's.
      if (!state) {
        const bool growth = !config_.growthPaused();
        const uint64_t needed = model_.snapshotBytes();
        uint64_t released = 0;
        StateRoom room;
        do {
          room = cache_.reclaimOneState(checkpoint, block, growth);
          released += room.extentBytes;
          if (room)
            state = model_.snapshot(active.request.id);
        } while (!state && room.extentBytes && released < needed);
        if (state)
          ++counters_.recycledStatePublications;
      }
      if (state) {
        cache_.publishCompositeState(block, std::move(state), checkpoint);
        ++publications;
      } else if (cache_.reuseStoredState(block, checkpoint)) {
        // No cache slot takes a RAM copy of a state already on disk.
        ++counters_.deduplicatedStatePublications;
      } else if (model_.canSnapshotToDisk() &&
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
    // lane's recovery point, unless it gave the snapshot its buffers.
    if (!replay)
      static_cast<void>(retireCheckpoint(active));
    active.latestCheckpoint = checkpoint ? cache_.checkpointState(block)
                                         : StateCheckpoint{};
    if (!checkpoint)
      writeBehind_.published(block, objective.tokens, now);
  }
  // Late siblings can extend the remaining plan only where both target and
  // draft states are complete, never at an arbitrary in-flight chunk boundary.
  if (materialized && addSharedPrefillBoundaries(active, promptProcessed))
    model_.setDraftContextPlan(
        active.request.id, pendingDraftStatePlan(active, promptProcessed));
  if (active.stateBoundaryCursor == active.stateBoundaries.size()) {
    active.stateBoundaries.clear();
    active.stateBoundaryCursor = 0;
  }
}

Engine::Prepared Engine::prepare(BatchPlan &plan,
                                 std::vector<ModelBatchItem> &items, double now) {
  items.reserve(plan.items.size());
  std::vector<BatchItem> admitted;
  admitted.reserve(plan.items.size());
  struct Denied final {
    uint64_t requestId = 0;
    TokenAdmission admission;
    uint64_t workEnd;
    Denial denial;
  };
  std::vector<Denied> denied;
  denied.reserve(plan.items.size());
  Request *const yielding = laneToYield();
  for (const BatchItem &scheduled : plan.items) {
    Request &active = request(scheduled.requestId);
    if (!active.lane)
      throw std::logic_error("scheduled request is not resident");
    const uint64_t position = plan.kind == WorkKind::Prefill
                                  ? scheduled.promptOffset
                                  : active.exactTokens.size();
    const uint64_t workEnd =
        plan.kind == WorkKind::Prefill
            ? position + scheduled.tokenCount
            : position + model::ExecutionLimits::targetVerifyRows;
    // A scheduled lane is resident: it grows as a request in service. The
    // lane that yields first if growth fails takes nothing in use, since its
    // own suspension, not another conversation's replay point, pays for it.
    // While other lanes fit, it waits for them instead, resident and denied
    // each step, as a start a resident lane holds back waits.
    const ReclaimClass upTo = &active == yielding && anotherResident(active.request.id)
                                  ? ReclaimClass::Ordinary
                                  : ReclaimClass::InUse;
    const auto kv = allocate(
        [&] { return cache_.ensureTokens(active.request.id, workEnd); }, true, upTo);
    if (!kv.admission.granted()) {
      denied.push_back(Denied{active.request.id, kv.admission, workEnd, kv.denial});
      continue;
    }
    admitted.push_back(scheduled);
    // A lane that runs is waiting for nothing.
    active.resourceWait = {};
    ModelBatchItem item;
    item.requestId = active.request.id;
    item.logicalPosition = position;
    item.tokenCount = scheduled.tokenCount;
    const PageTableView pageTable = cache_.pageTable(active.request.id);
    item.pageTable = pageTable.pages;
    item.pageTableRevision = pageTable.revision;
    item.pageTableFirstChanged = pageTable.firstChanged;
    if (plan.kind == WorkKind::Prefill) {
      item.inputTokens =
          std::span<const uint32_t>(active.exactTokens)
              .subspan(scheduled.promptOffset, scheduled.tokenCount);
    }
    items.push_back(std::move(item));
  }
  if (!admitted.empty()) {
    plan.items = std::move(admitted);
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
  if (judge(victim.denial, active.request.id) == Verdict::Fail)
    settle(request(victim.requestId),
           capacityExhausted("KV target", victim.admission.allocationFailure,
                             pageShortfall(victim.admission)));
  else
    suspendForGrowth(active, resumeTarget, victim.admission.allocationFailure,
                     now);
  return Prepared::Yielded;
}

uint64_t Engine::completedTokens(const Request &active) const {
  return scheduler_.phase(active.request.id) == Phase::Prefill
             ? scheduler_.promptProcessed(active.request.id)
             : active.exactTokens.size();
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
                      const std::function<bool(const Denial &)> &fallback)
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
    else
      reclaimed = paused ? reuseCachedPagesWhilePaused(admission, upTo)
                         : reclaimForKv(admission.additionalPages, upTo);
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
    if (!paused && fallback && fallback(denial))
      continue;
    break;
  }
  if (!admission.granted()) {
    denial.allocationFailure = admission.allocationFailure;
    denial.pending = denial.pending || transferPending(admission);
    // Memory on its way back ends the shortage without the residents.
    if (memoryDenied(admission) && !denial.pending)
      allocationFailed_ = true;
  }
  return result;
}

// The reclaim step for a lane's state the engine's limit refused. The pooled
// buffers a lane starts from stay for its activation to take: idle model
// memory beyond them goes first (a pooled buffer, else the idle vision
// arena, else one rebuildable cache entry), then one empty extent, or else
// one victim of the cache with the extent it empties.
// A host refusal comes to neither step but to the reuse path (allocate()):
// the pressure controller owns that shrink, and evicting for an allocator
// that refuses all the same would drain the cache before macOS can
// acknowledge any reclaimed bytes.
CacheReclaimResult Engine::reclaimForState(ReclaimClass upTo) {
  if (reclaimIdleState(true))
    return {true, 0};
  const CacheReclaimResult reclaimed =
      cache_.reclaimOne(CacheReclaimMode::ReleaseExtents, upTo);
  if (reclaimed.madeProgress)
    signalResourceProgress();
  return reclaimed;
}

// The reclaim step for KV pages the engine's limit refused. Allocated extents
// stay for the pages to reuse; idle state memory goes first, then the cache
// gives up what covers the shortfall in one step.
CacheReclaimResult Engine::reclaimForKv(uint32_t pages, ReclaimClass upTo) {
  if (reclaimIdleState(false))
    return {true, 0};
  const CacheReclaimResult reclaimed =
      cache_.reclaimForPages(pages, CacheReclaimMode::KeepExtents, upTo);
  if (reclaimed.madeProgress)
    signalResourceProgress();
  return reclaimed;
}

bool Engine::reclaimIdleState(bool keepLane) noexcept {
  if (!model_.reclaimIdleState(keepLane, model::IdleMemory::BuffersThenCaches))
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
// it is in service and waits if it is not.
CacheReclaimResult Engine::reuseCachedStateWhilePaused(ReclaimClass upTo) {
  if (reclaimIdleState(true))
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
// evicted when the pages the request's class may take can actually cover
// the shortfall: never those a request holds, nor, for a start a resident
// lane holds back, the idle KV that states in use restore through
// (Cache::reusablePages). Otherwise it survives for later hits, and the
// request grows if it is in service and waits if it is not. Only KV goes: a
// state's buffers give no page while the host refuses growth, unless the
// state sits on the leaf that goes next (CacheReclaimMode::ReusePages). A
// reclaim that must wait for the transfer in flight makes the request wait
// with it, as it does without the pause. Idle model state goes first, but
// not the pooled buffers the next lane starts from: they would not let this
// request grow, and the paced pass keeps them for the next one.
CacheReclaimResult Engine::reuseCachedPagesWhilePaused(const TokenAdmission &admission,
                                                       ReclaimClass upTo) {
  if (reclaimIdleState(true))
    return {true, 0};
  if (cache_.reusablePages(upTo) < admission.additionalPages)
    return {};
  const CacheReclaimResult reused =
      cache_.reclaimForPages(admission.additionalPages, CacheReclaimMode::ReusePages, upTo);
  if (reused.madeProgress)
    signalResourceProgress();
  return reused;
}

void Engine::suspendLane(Request &active, uint64_t workEnd, metal::AllocationFailure failure,
                         StateFailure reason, double now) {
  if (!active.lane || scheduler_.suspended(active.request.id)) {
    throw std::logic_error("request cannot be suspended");
  }
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
          cache_.reclaimOne(CacheReclaimMode::ReleaseExtents, ReclaimClass::InUse,
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

void Engine::apply(const BatchPlan &plan,
                   std::span<const ModelStepResult> results,
                   double wallMilliseconds, double cycleMilliseconds,
                   bool representativePrefillTiming, double now) {
  if (results.size() != plan.items.size()) {
    throw std::logic_error("model result count changed");
  }
  std::vector<StepResult> schedulerResults;
  schedulerResults.reserve(results.size());
  uint32_t inputTokens = 0;
  uint32_t outputTokens = 0;
  uint32_t draftedTokens = 0;
  uint32_t acceptedDraftTokens = 0;
  for (size_t index = 0; index < results.size(); ++index) {
    const ModelStepResult &result = results[index];
    const BatchItem &item = plan.items[index];
    if (result.requestId != item.requestId) {
      throw std::logic_error("model result order changed");
    }
    Request &active = request(result.requestId);
    if (!result.failure.empty() && !active.pendingEnd) {
      // The model rejected this lane's own numerical result. An earlier
      // cancellation or deadline of the same lane still stands.
      active.pendingEnd = LaneEnd{LaneOutcome::ModelResultInvalid, result.failure};
    }
    if (!active.pendingEnd) {
      const auto outOfVocabulary = std::find_if(
          result.outputTokens.begin(), result.outputTokens.end(),
          [&](uint32_t token) { return token >= config_.vocabularySize; });
      if (outOfVocabulary != result.outputTokens.end()) {
        // A model that emits a token outside the vocabulary without reporting
        // it fails this lane the same way (the Qwen runtime reports its
        // non-finite rows itself).
        active.pendingEnd =
            LaneEnd{LaneOutcome::ModelResultInvalid,
                    "model emitted out-of-vocabulary token " +
                        std::to_string(*outOfVocabulary)};
      }
    }
    if (active.pendingEnd) {
      // An in-flight Metal command cannot be revoked safely. Its provisional
      // writes remain invisible, but a cancelled, deadline-expired or
      // model-rejected request must not publish cache state or emit output
      // when that command drains.
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
      cache_.publishCommittedBlocks(active.request.id, active.exactTokens,
                                    promptProcessed, active.request.images);
      publishReachedStateBoundaries(active, promptProcessed, now);
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
      // A terminal anchor is emitted without a target KV row; it never enters
      // a cached block.
      const uint32_t storedTokens =
          static_cast<uint32_t>(active.exactTokens.size()) -
          result.outputTokensWithoutKv;
      cache_.publishCommittedBlocks(active.request.id, active.exactTokens,
                                    storedTokens, active.request.images);
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
    // on the final prompt chunk completes them.
    const bool complete =
        result.finished ||
        (!scoring && completionTokens >= active.request.maxNewTokens);
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
  scheduler_.complete(plan, schedulerResults, wallMilliseconds,
                      representativePrefillTiming);
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
    } else if (plan.kind == WorkKind::Prefill &&
               scheduler_.phase(active.request.id) == Phase::Prefill) {
      armNextStateBoundary(active);
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
  if (reason == EngineFinishReason::Cancelled) {
    scheduler_.cancel(active.request.id);
  }
  active.finalized = true;
  // A first token the client was sent counts though no cycle committed it, as
  // when the request is cancelled before its first decode cycle.
  const auto completionTokens = static_cast<uint32_t>(
      active.exactTokens.size() - active.promptTokens +
      (active.sentFirstToken ? 1 : 0));
  events_.completed(active.request.id, reason, active.promptTokens,
                    completionTokens, optionLogits);
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
  events_.failed(active.request.id, end.outcome, std::move(end.message));
  ++counters_.failed;
  release(active);
}

void Engine::vacateLane(Request &active, bool keepContinuation) {
  const uint64_t id = active.request.id;
  discardPendingStateBoundaries(active);
  scheduler_.setPrefillBoundary(id, std::nullopt);
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

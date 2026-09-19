#include "engine/Engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace splash::engine {
namespace {

constexpr double kResourceRetryBackoffMilliseconds = 100.0;
constexpr double kHealthCheckIntervalMilliseconds = 1000.0;

uint32_t replayStateBoundary(uint32_t tokens) noexcept {
  return tokens > 1 ? (tokens - 1) / KvCache::pageTokens * KvCache::pageTokens
                    : 0;
}

} // namespace

Engine::Engine(EngineConfig config, Cache &cache, model::Model &model,
               EngineEventSink &events)
    : config_(config), cache_(cache), model_(model), events_(events) {
  if (!config_.maxContext || !config_.vocabularySize) {
    throw std::invalid_argument("context and vocabulary sizes must be positive");
  }
  if (!std::isfinite(config_.resourceWaitTimeoutMilliseconds) ||
      config_.resourceWaitTimeoutMilliseconds <= 0.0)
    throw std::invalid_argument("resource wait timeout must be positive and finite");
  if (config_.prefillCheckpointTokens &&
      (config_.prefillCheckpointTokens <
           model::ExecutionLimits::draftContextTokens ||
       config_.prefillCheckpointTokens % KvCache::pageTokens)) {
    throw std::invalid_argument(
        "prefill checkpoint interval must span a draft window and whole KV pages");
  }
}

void Engine::submit(EngineRequest value) {
  if (!value.id || value.prompt.empty() || !value.maxNewTokens ||
      value.prompt.size() + value.maxNewTokens > config_.maxContext ||
      !std::isfinite(value.deadlineMilliseconds) ||
      value.deadlineMilliseconds <= 0.0) {
    throw std::invalid_argument("invalid backend request");
  }
  if (std::any_of(value.prompt.begin(), value.prompt.end(), [&](uint32_t token) {
        return token >= config_.vocabularySize;
      })) {
    throw std::invalid_argument("prompt token is out of vocabulary");
  }
  uint64_t previousImageEnd = 0;
  uint64_t pixelBytes = 0;
  for (const ImageSpan &image : value.images) {
    const uint64_t patches = uint64_t{image.gridHeight} * image.gridWidth;
    if (image.gridHeight < 2 || image.gridWidth < 2 || image.gridHeight % 2 ||
        image.gridWidth % 2 || patches > config_.maxImagePatches ||
        image.tokens != (image.gridHeight / 2) * (image.gridWidth / 2) ||
        image.offset < previousImageEnd ||
        uint64_t{image.offset} + image.tokens > value.prompt.size()) {
      throw std::invalid_argument("invalid backend request image span");
    }
    previousImageEnd = image.end();
    pixelBytes += image.pixelBytes();
  }
  if (value.imagePixels.size() != pixelBytes) {
    throw std::invalid_argument("invalid backend request image pixels");
  }
  const uint64_t id = value.id;
  Request requestState;
  requestState.promptTokens = static_cast<uint32_t>(value.prompt.size());
  requestState.replayTokens = requestState.promptTokens;
  requestState.request = std::move(value);
  auto [entry, inserted] = requests_.emplace(id, std::move(requestState));
  if (!inserted)
    throw std::invalid_argument("duplicate backend request id");
  const Request &stored = entry->second;
  scheduler_.submit(
      {.id = id,
       .priority = stored.request.priority,
       .cohort = stored.request.cohort,
       .promptTokens = stored.promptTokens,
       .deadlineMilliseconds = stored.request.deadlineMilliseconds});
  ++counters_.submitted;
}

void Engine::cancel(uint64_t id) {
  auto found = requests_.find(id);
  if (found == requests_.end() || found->second.finalized)
    return;
  if (pending_) {
    for (const BatchItem &item : pending_->plan.items) {
      if (item.requestId == id) {
        // An earlier deadline failure of the same in-flight lane stands.
        if (!found->second.failure) {
          found->second.failure = Failure{"cancelled", "request cancelled"};
        }
        pending_->ticket->abandonMask(id);
        return;
      }
    }
  }
  finish(found->second, EngineFinishReason::Cancelled);
}

void Engine::failRequest(uint64_t id, std::string code, std::string message) {
  Request &active = request(id);
  if (active.finalized || active.failure)
    return;
  Failure failure{std::move(code), std::move(message)};
  if (pending_) {
    for (const BatchItem &item : pending_->plan.items) {
      if (item.requestId == id) {
        active.failure = std::move(failure);
        pending_->ticket->abandonMask(id);
        return;
      }
    }
  }
  finishFailure(active, std::move(failure));
}

void Engine::provideMask(uint64_t id, std::span<const uint32_t> words) {
  Request &active = request(id);
  const bool ownedByActiveBatch =
      pending_ && pending_->ticket->ownsMaskWait(id);
  // A request that already left its mask wait (cancellation, deadline, or a
  // failure raced the frontend) treats the response as stale.
  if (active.finalized || active.failure ||
      (!ownedByActiveBatch && scheduler_.phase(id) != Phase::WaitingMask)) {
    return;
  }
  model_.provideMask(id, words);
  if (!ownedByActiveBatch)
    scheduler_.maskReady(id);
}

void Engine::setCompletionNotifier(std::function<void()> notifier) {
  completionNotifier_ = std::move(notifier);
  cache_.setCompletionNotifier(completionNotifier_);
}

bool Engine::tick(double now) {
  model_.checkHealth();
  nextHealthCheckMilliseconds_ = now + kHealthCheckIntervalMilliseconds;
  bool progressed = scheduler_.expireDeadlines(now);
  if (cache_.pollTransfers()) {
    // Demoted pages and written states are back; waiting lanes retry now.
    signalResourceProgress();
    progressed = true;
  }
  progressed = pollRestores(now) || progressed;
  const bool draining = drainingForRecovery();
  for (auto &[_, active] : requests_) {
    // Admission is deliberately paused while resident peers finish. Start a
    // fresh resource wait only if admission still fails after that drain.
    if (draining) {
      active.resourceWait.deadlineMilliseconds = 0.0;
      continue;
    }
    const ResourceWait &wait = active.resourceWait;
    const bool stalled = !(wait.pending && wait.epoch != resourceEpoch_);
    if (!active.finalized && wait.deadlineMilliseconds > 0.0 &&
        now >= wait.deadlineMilliseconds && stalled) {
      finishFailure(active, {"resource_timeout", "memory did not become available within the resource wait limit", true});
      progressed = true;
    }
  }
  // Finalize expired requests now, even while a command is in flight, so a
  // late mask or cancel for them is a no-op rather than a scheduler error.
  if (progressed)
    sweepTerminal();
  if (pending_) {
    auto forwardMaskRequests = [&] {
      for (ModelMaskRequest &request : pending_->ticket->takeMaskRequests()) {
        events_.maskRequested(request.requestId, request.simulationTokens);
        progressed = true;
      }
    };
    forwardMaskRequests();
    for (const BatchItem &item : pending_->plan.items) {
      Request &active = request(item.requestId);
      if (!active.failure &&
          active.request.deadlineMilliseconds <= now) {
        active.failure =
            Failure{"deadline_exceeded", "request deadline exceeded"};
        pending_->ticket->abandonMask(item.requestId);
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
    if (!command.plan.empty())
      apply(command.plan, results, command.ticket->wallMilliseconds(),
            command.ticket->prefillTimingIsRepresentative());
    sweepTerminal();
    return true;
  }

  progressed = admitQueued(now) || progressed;
  sweepTerminal();
  if (auto plan = scheduler_.next()) {
    std::vector<ModelBatchItem> items;
    switch (prepare(*plan, items, now)) {
    case Prepared::Runnable: {
      std::unique_ptr<ModelBatchTicket> ticket =
          model_.submit(*plan, items, completionNotifier_);
      if (!ticket) {
        throw std::logic_error("model returned an empty command ticket");
      }
      scheduler_.commit(*plan);
      pending_ = Pending{std::move(*plan), std::move(ticket)};
      return true;
    }
    case Prepared::Yielded:
      sweepTerminal();
      return true;
    case Prepared::Waiting:
      break;
    }
  }
  // No model work runs: queued KV copies ride a command of their own, so a
  // restore or a demotion never waits for the next batch.
  if (auto ticket = model_.submitTransfers(completionNotifier_)) {
    pending_ = Pending{BatchPlan{}, std::move(ticket)};
    return true;
  }
  return progressed;
}

bool Engine::idle() const noexcept { return requests_.empty() && !pending_; }

bool Engine::drainingForRecovery() const {
  return recoveringResources_ &&
         std::any_of(requests_.begin(), requests_.end(), [](const auto &entry) {
           return entry.second.stateCell.has_value();
         });
}

std::optional<double> Engine::nextWakeupMilliseconds() const {
  std::optional<double> result;
  if (pending_ || model_.needsHealthCheck())
    result = nextHealthCheckMilliseconds_;
  const bool draining = drainingForRecovery();
  for (const auto &[_, active] : requests_) {
    if (active.finalized || active.failure)
      continue;
    if (!result || active.request.deadlineMilliseconds < *result)
      result = active.request.deadlineMilliseconds;
    if (!draining && active.resourceWait.deadlineMilliseconds > 0.0 &&
        (!result || active.resourceWait.deadlineMilliseconds < *result))
      result = active.resourceWait.deadlineMilliseconds;
    if (draining || active.resourceWait.retryMilliseconds <= 0.0)
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
  return result;
}

ResourceWaitSnapshot Engine::resourceWaitSnapshot(double now) const {
  ResourceWaitSnapshot result;
  result.draining = drainingForRecovery();
  for (const auto &[id, active] : requests_) {
    if (active.finalized || scheduler_.phase(id) != Phase::WaitingResources)
      continue;
    if (active.resourceWait.reason == StateFailure::ConcurrencyLimit)
      ++result.concurrency;
    else
      ++result.memory;
    if (active.suspended)
      ++result.suspended;
    if (active.resourceWait.startedMilliseconds)
      result.oldestWaitMilliseconds = std::max(
          result.oldestWaitMilliseconds, now - *active.resourceWait.startedMilliseconds);
  }
  return result;
}

bool Engine::admitQueued(double now) {
  const bool recovering = std::any_of(
      requests_.begin(), requests_.end(),
      [](const auto &entry) { return entry.second.suspended; });
  // Once pressure has preempted work, let resident lanes finish before
  // spending their released headroom on a retry or a new request. Recover
  // one lane at a time; this prevents repeated B4 admission/preemption churn.
  if (!recovering)
    recoveringResources_ = false;
  if (drainingForRecovery())
    return false;
  bool progressed = false;
  for (uint64_t requestId : scheduler_.admissionOrder()) {
    Request &active = request(requestId);
    if (recovering && !active.suspended)
      continue;
    if (active.restore || !resourceRetryReady(active, now))
      continue;
    progressed = admit(active, now) || progressed;
    if (recovering && progressed)
      break;
  }
  return progressed;
}

bool Engine::admit(Request &active, double now) {
  const bool resuming = active.suspended;
  ModelRequest modelRequest = active.request.modelView();
  if (resuming)
    modelRequest.prompt = active.exactTokens;
  CacheLookup lookup =
      active.skipCache ? CacheLookup{}
                       : cache_.lookup(modelRequest.prompt, active.request.images);
  bool executorStarted = false;
  bool resourcesStarted = false;
  try {
    const auto activate = [&] {
      return resuming ? model_.resume(modelRequest) : model_.begin(modelRequest);
    };
    const uint64_t releaseGeneration = cache_.releaseGeneration();
    StateAdmission admission = activate();
    bool reclaimedForAdmission = false;
    Denial denial;
    while (!admission.granted() &&
           admission.failure == StateFailure::MemoryPressure) {
      const bool hostPressure =
          admission.allocationFailure == metal::AllocationFailure::HostPressure;
      const CacheReclaimResult reclaimed =
          hostPressure ? CacheReclaimResult{reclaimIdleState()} : reclaimForGrowth();
      if (reclaimed.madeProgress) {
        reclaimedForAdmission = true;
        admission = activate();
        continue;
      }
      denial.pending = reclaimed.pending;
      if (hostPressure)
        break;
      // A useful restore remains pinned throughout ordinary eviction. If
      // that pin is the last obstacle to admitting even one lane, prefer
      // cold recomputation over waiting forever for our own cache lease.
      if (!growthPaused() && lookup.state) {
        lookup = {};
        continue;
      }
      break;
    }
    if (!admission.granted()) {
      // A cell the budget or the driver refused comes back only with a
      // release in flight; any other refusal passes by itself.
      const bool refused =
          admission.allocationFailure == metal::AllocationFailure::EngineBudget ||
          admission.allocationFailure == metal::AllocationFailure::DriverRejected;
      denial.allocationFailure = admission.allocationFailure;
      denial.retryable = !refused || budgetMayRecover(admission.allocationFailure,
                                                      releaseGeneration, reclaimedForAdmission);
      if (judge(denial, active.request.id) == Verdict::Fail) {
        finishFailure(active,
                      {"capacity_exhausted",
                       std::string("could not allocate request state: ") +
                           metal::allocationFailureName(
                               admission.allocationFailure),
                       true});
        return true;
      }
      scheduler_.waitForResources(active.request.id);
      deferResourceRetry(active, now, admission.failure, denial.pending);
      return false;
    }
    executorStarted = true;
    cache_.beginRequest(active.request.id);
    resourcesStarted = true;
    active.stateCell = *admission.cell;
    const uint32_t resumeBoundary = lookup.resumeBoundary();
    const uint64_t requestId = active.request.id;
    // The matched chain first, then the first work's pages for a lane that
    // will not go through ordinary prefill admission before it runs: one
    // that resumes, or one that waits for a restore.
    KvAdmission kv;
    if (lookup.state)
      kv = admitKv([&] { return cache_.restoreRequest(requestId, lookup); });
    const bool restoring =
        lookup.state && (!lookup.state->state()->residentBytes() ||
                         cache_.kvRestoreStatus(requestId) == KvRestoreStatus::Pending);
    if (kv.allocation.granted() && (resuming || restoring)) {
      const uint64_t workEnd =
          resuming ? active.resumeKvTargetTokens : uint64_t{resumeBoundary} + 1;
      kv = admitKv([&] { return cache_.ensureTokens(requestId, workEnd); });
    }
    if (!kv.allocation.granted()) {
      // The host continuation survives this failed admission. No recurrent
      // state restore or replay has run, and all temporary leases are freed.
      if (resuming) model_.suspend(requestId);
      else model_.end(requestId);
      cache_.endRequest(requestId);
      active.stateCell.reset();
      executorStarted = resourcesStarted = false;
      const Verdict verdict = judge(kv.denial, requestId);
      if (verdict == Verdict::Fail && restoring) {
        // Release the prefix pin before retrying without its memory footprint.
        active.skipCache = true;
        scheduler_.waitForResources(requestId);
        deferResourceRetry(active, now);
        return false;
      }
      if (verdict == Verdict::Fail) {
        finishCapacity(active, kv.allocation);
        return true;
      }
      scheduler_.waitForResources(requestId);
      deferResourceRetry(active, now, StateFailure::MemoryPressure, kv.denial.pending);
      return false;
    }
    active.resumeKvTargetTokens = 0;
    active.resourceWait = {};
    DraftContextPlan draft = configureDraftStatePlan(
        active, resumeBoundary, lookup.junctionBoundary());
    std::unique_ptr<StateRestore> transfer;
    if (lookup.state) {
      transfer = model_.beginRestore(requestId, resumeBoundary, lookup.state->state(),
                                     !draft.draftStateRestoreSkipped,
                                     completionNotifier_);
    }
    if (transfer || cache_.kvRestoreStatus(requestId) == KvRestoreStatus::Pending) {
      active.restore.emplace(Request::Restore{
          std::move(lookup), std::move(draft), std::move(transfer)});
      return true;
    }
    completeAdmission(active, lookup, std::move(draft));
    return true;
  } catch (...) {
    discardPendingStateBoundaries(active);
    if (executorStarted)
      model_.end(active.request.id);
    if (resourcesStarted)
      cache_.endRequest(active.request.id);
    active.stateCell.reset();
    throw;
  }
}

void Engine::completeAdmission(Request &active, CacheLookup &lookup,
                                DraftContextPlan draft) {
  const bool resuming = active.suspended;
  active.skipCache = false;
  const uint32_t resumeBoundary = lookup.resumeBoundary();
  if (resuming)
    scheduler_.resumeFromResources(active.request.id, resumeBoundary, active.replayTokens);
  active.latestCheckpoint = {};
  if (lookup.state) {
    active.latestCheckpoint = cache_.checkpointState(lookup.state->kvBlock());
    // A restored endpoint already has the ordinary replay state we need.
    // Other restored progress points retain their rolling lifetime.
    if (active.latestCheckpoint &&
        resumeBoundary == replayStateBoundary(active.replayTokens)) {
      static_cast<void>(
          cache_.reuseCompositeState(active.latestCheckpoint.kvBlock));
      ++counters_.deduplicatedStatePublications;
      active.latestCheckpoint = {};
    }
  }
  model_.setDraftContextPlan(active.request.id, std::move(draft));
  if (resuming) {
    active.suspended = false;
    active.replaying = true;
    armNextStateBoundary(active);
    ++counters_.resourceResumptions;
    return;
  }
  active.exactTokens = std::move(active.request.prompt);
  scheduler_.resourcesReady(active.request.id, resumeBoundary);
  armNextStateBoundary(active);
  cache_.recordLookup(lookup);
  events_.started(active.request.id,
                  resumeBoundary ? EngineCacheStatus::PrefixHit
                                 : EngineCacheStatus::Miss,
                  resumeBoundary, *active.stateCell);
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
    if (!active.failure && active.request.deadlineMilliseconds <= now)
      active.failure = Failure{"deadline_exceeded", "request deadline elapsed"};
    StateRestore *ticket = active.restore->ticket.get();
    if (active.failure && ticket) ticket->cancel();
    // The state's read must drain before its cell is reused; KV restores
    // belong to their blocks and outlive a request that gives up.
    if (ticket && !ticket->ready()) continue;
    const KvRestoreStatus kv = cache_.kvRestoreStatus(id);
    if (!active.failure && kv == KvRestoreStatus::Pending) continue;
    auto restore = std::move(*active.restore);
    active.restore.reset();
    progressed = true;
    if (active.failure) {
      restore.ticket.reset();
      restore.lookup = {};
      if (active.failure->code == "cancelled") finish(active, EngineFinishReason::Cancelled);
      else finishFailure(active, std::move(*active.failure));
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
    discardPendingStateBoundaries(active);
    if (active.suspended) model_.suspend(id);
    else model_.end(id);
    cache_.endRequest(id);
    active.stateCell.reset();
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
                                StateFailure reason, bool pending) noexcept {
  auto &wait = active.resourceWait;
  if (!wait.startedMilliseconds)
    wait.startedMilliseconds = now;
  const bool progressed = wait.pending && wait.epoch != resourceEpoch_;
  wait.reason = reason;
  wait.pending = pending;
  if (reason == StateFailure::ConcurrencyLimit)
    wait.deadlineMilliseconds = 0.0;
  else if (progressed || wait.deadlineMilliseconds <= 0.0)
    wait.deadlineMilliseconds = now + config_.resourceWaitTimeoutMilliseconds;
  wait.epoch = resourceEpoch_;
  wait.retryMilliseconds = now + kResourceRetryBackoffMilliseconds;
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

  const auto addCandidate = [&](uint32_t tokens,
                                Request::StateBoundary::Purpose purpose) {
    if (!tokens || tokens <= stateBoundary)
      return;
    for (size_t index = 0; index < active.stateBoundaries.size(); ++index) {
      if (active.stateBoundaries[index].tokens != tokens)
        continue;
      if (purpose > active.stateBoundaries[index].purpose)
        active.stateBoundaries[index].purpose = purpose;
      return;
    }
    active.stateBoundaries.push_back({tokens, purpose});
  };

  // Plan draft windows before prefill; arbitrary chunk ends do not carry a
  // complete draft state. Progress points remain disposable after restoration.
  const uint32_t latestReplayBoundary = replayStateBoundary(active.replayTokens);
  if (const uint32_t interval = config_.prefillCheckpointTokens) {
    // A checkpoint earns its state only with a full interval of prefill
    // after it; the replay state at the end covers the last stretch.
    for (uint64_t boundary = (uint64_t{stateBoundary} / interval + 1) * interval;
         boundary + interval <= latestReplayBoundary; boundary += interval) {
      addCandidate(static_cast<uint32_t>(boundary),
                   Request::StateBoundary::Purpose::Checkpoint);
    }
  }
  addCandidate(junctionBoundary, Request::StateBoundary::Purpose::Junction);
  addCandidate(latestReplayBoundary, Request::StateBoundary::Purpose::Replay);
  std::sort(active.stateBoundaries.begin(), active.stateBoundaries.end(),
            [](const Request::StateBoundary &left,
               const Request::StateBoundary &right) {
              return left.tokens < right.tokens;
            });

  try {
    std::vector<uint32_t> materializationBoundaries;
    materializationBoundaries.reserve(active.stateBoundaries.size());
    for (const auto &boundary : active.stateBoundaries)
      materializationBoundaries.push_back(boundary.tokens);
    DraftContextPlan draft = planDraftContext(
        stateBoundary, active.replayTokens,
        stateBoundary ? std::optional<uint32_t>(stateBoundary) : std::nullopt,
        materializationBoundaries);
    return draft;
  } catch (...) {
    discardPendingStateBoundaries(active);
    throw;
  }
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

bool Engine::retireCheckpoint(Request &active) {
  // Shared progress points remain disposable under memory pressure, but a
  // lane's normal rolling replacement must not retire its peer's recovery point.
  const auto point = active.latestCheckpoint;
  if (point && std::any_of(requests_.begin(), requests_.end(), [&](const auto &entry) {
        const auto &peer = entry.second;
        return &peer != &active && !peer.finalized &&
               peer.latestCheckpoint.kvBlock == point.kvBlock &&
               peer.latestCheckpoint.publication == point.publication;
      })) {
    active.latestCheckpoint = {};
    return true;
  }
  if (!cache_.retireCheckpointState(active.latestCheckpoint))
    return false;
  active.latestCheckpoint = {};
  return true;
}

void Engine::publishReachedStateBoundaries(Request &active,
                                           uint32_t promptProcessed) {
  while (active.stateBoundaryCursor < active.stateBoundaries.size() &&
         active.stateBoundaries[active.stateBoundaryCursor].tokens <=
             promptProcessed) {
    const Request::StateBoundary objective =
        active.stateBoundaries[active.stateBoundaryCursor++];
    const bool checkpoint =
        objective.purpose == Request::StateBoundary::Purpose::Checkpoint;
    const bool junction =
        objective.purpose == Request::StateBoundary::Purpose::Junction;
    uint64_t &failures = checkpoint ? counters_.checkpointPublicationFailures
                         : junction ? counters_.junctionMaterializationFailures
                                    : counters_.replayStatePublicationFailures;
    uint64_t &publications = checkpoint ? counters_.checkpointPublications
                             : junction ? counters_.junctionMaterializations
                                        : counters_.replayStatePublications;
    // The scheduler ends a command exactly at an armed boundary; a boundary
    // passed inside a command has no materialized state to copy.
    if (objective.tokens != promptProcessed) {
      ++failures;
      continue;
    }
    try {
      const uint64_t block = cache_.blockAt(active.request.id, objective.tokens);
      if (cache_.reuseCompositeState(block, checkpoint)) {
        ++counters_.deduplicatedStatePublications;
      } else {
        // Recycle the previous recovery point before allocating its replacement.
        // A restore lease can delay this optional publication.
        if (!retireCheckpoint(active) && checkpoint) {
          ++failures;
          continue;
        }
        std::shared_ptr<const CompositeState> state =
            model_.snapshot(active.request.id);
        if (!state && cache_.reclaimOneState(checkpoint)) {
          state = model_.snapshot(active.request.id);
          if (state)
            ++counters_.recycledStatePublications;
        }
        if (state) {
          cache_.publishCompositeState(block, std::move(state), checkpoint);
        } else if (model_.canSnapshotToDisk() &&
                   cache_.publishStateToDisk(
                       block,
                       [&](std::function<void()> completion) {
                         return model_.snapshotToDisk(active.request.id, std::move(completion));
                       },
                       checkpoint)) {
          // No cache slot holds the state; the tier takes it from the lane.
          ++counters_.diskStatePublications;
        } else {
          ++failures;
          continue;
        }
        ++publications;
      }
      if (active.latestCheckpoint.kvBlock != block)
        static_cast<void>(retireCheckpoint(active));
      active.latestCheckpoint = checkpoint ? cache_.checkpointState(block)
                                           : StateCheckpoint{};
    } catch (const std::exception &) {
      ++failures;
    }
  }
  if (active.stateBoundaryCursor == active.stateBoundaries.size()) {
    active.stateBoundaries.clear();
    active.stateBoundaryCursor = 0;
  }
  armNextStateBoundary(active);
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
  for (const BatchItem &scheduled : plan.items) {
    Request &active = request(scheduled.requestId);
    if (!active.stateCell)
      throw std::logic_error("scheduled request is not resident");
    const uint64_t position = plan.kind == WorkKind::Prefill
                                  ? scheduled.promptOffset
                                  : active.exactTokens.size();
    const uint64_t workEnd =
        plan.kind == WorkKind::Prefill
            ? position + scheduled.tokenCount
            : position + model::ExecutionLimits::targetVerifyRows;
    const KvAdmission kv =
        admitKv([&] { return cache_.ensureTokens(active.request.id, workEnd); });
    if (!kv.allocation.granted()) {
      denied.push_back(Denied{active.request.id, kv.allocation, workEnd, kv.denial});
      continue;
    }
    admitted.push_back(scheduled);
    // A lane that runs is waiting for nothing.
    active.resourceWait = {};
    ModelBatchItem item;
    item.requestId = active.request.id;
    item.stateSlot = *active.stateCell;
    item.logicalPosition = position;
    item.promptOffset =
        plan.kind == WorkKind::Prefill ? scheduled.promptOffset : 0;
    item.tokenCount = scheduled.tokenCount;
    const PageTableView pageTable = cache_.pageTable(active.request.id);
    item.pageTable = pageTable.pages;
    item.pageTableRevision = pageTable.revision;
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
  // A partially admitted plan executes the real narrower B1/B2/B3 shape.
  // When every lane is blocked, the lane with the lowest priority and then
  // the least completed work yields. Physical pressure suspends it so another
  // request can use its backing. Host continuation is retained; committed
  // history is replayed through normal prefill on resumption. A lone request
  // that cannot fit the hard capacity fails; transient host pressure waits
  // for recovery or the existing request deadline instead.
  if (denied.empty())
    throw std::logic_error("empty resource admission result");
  // Memory on its way back arrives without anyone yielding. The lanes still
  // take a retry deadline: the transfer's completion wakes the engine, and
  // the deadline is what makes the wait end if that wake is ever missed.
  if (std::any_of(denied.begin(), denied.end(),
                  [](const Denied &entry) { return entry.denial.pending; })) {
    for (const Denied &entry : denied)
      deferResourceRetry(request(entry.requestId), now, StateFailure::MemoryPressure,
                         entry.denial.pending);
    return Prepared::Waiting;
  }
  const Denied &victim = *std::min_element(
      denied.begin(), denied.end(),
      [&](const Denied &left, const Denied &right) {
        const Request &a = request(left.requestId);
        const Request &b = request(right.requestId);
        if (a.request.priority != b.request.priority)
          return a.request.priority > b.request.priority;
        return a.exactTokens.size() < b.exactTokens.size();
      });
  Request &active = request(victim.requestId);
  if (judge(victim.denial, victim.requestId) == Verdict::Fail)
    finishCapacity(active, victim.admission);
  else
    suspendForGrowth(active, victim.workEnd, now);
  return Prepared::Yielded;
}

bool Engine::anotherResident(uint64_t requestId) const {
  return std::any_of(requests_.begin(), requests_.end(), [&](const auto &entry) {
    return entry.first != requestId && entry.second.stateCell;
  });
}

Engine::Verdict Engine::judge(const Denial &denial, uint64_t requestId) const {
  if (denial.pending)
    return Verdict::Wait;
  // Pages held by resident lanes come back when they finish; only a lane
  // that cannot fit on its own has hit the capacity.
  if (growthPaused() || denial.retryable || anotherResident(requestId) ||
      denial.allocationFailure == metal::AllocationFailure::HostPressure)
    return Verdict::Yield;
  return Verdict::Fail;
}

Engine::KvAdmission Engine::admitKv(const std::function<TokenAdmission()> &attempt) {
  const uint64_t releaseGeneration = cache_.releaseGeneration();
  bool reclaimed = false;
  bool pendingReclaim = false;
  TokenAdmission admission = attempt();
  while (!admission.granted() &&
         admission.failure == KvPageAcquireFailure::PhysicalCapacity) {
    const bool paused = growthPaused() ||
        admission.allocationFailure == metal::AllocationFailure::HostPressure;
    const CacheReclaimResult progress = paused
        ? CacheReclaimResult{reuseIdleBackingWhilePaused(admission)}
        : reclaimForGrowth(CacheReclaimMode::ReuseBacking);
    if (!progress.madeProgress) {
      pendingReclaim = progress.pending;
      break;
    }
    reclaimed = true;
    admission = attempt();
  }
  Denial denial;
  if (!admission.granted()) {
    denial.allocationFailure = admission.allocationFailure;
    denial.pending = admission.failure == KvPageAcquireFailure::Pending || pendingReclaim;
    denial.retryable = denial.pending || budgetMayRecover(admission.allocationFailure,
                                                          releaseGeneration, reclaimed);
  }
  return {admission, denial};
}

bool Engine::budgetMayRecover(metal::AllocationFailure failure,
                             uint64_t generation, bool reclaimed) const {
  if (failure != metal::AllocationFailure::EngineBudget)
    return false;
  const bool pending = cache_.releasePending();
  return pending || reclaimed || cache_.releaseGeneration() != generation;
}

bool Engine::growthPaused() const {
  return config_.memoryPressure &&
         config_.memoryPressure() != MemoryPressure::Normal;
}

CacheReclaimResult Engine::reclaimForGrowth(CacheReclaimMode mode) {
  if (reclaimIdleState())
    return {true, 0};
  // The background pressure controller owns physical shrink. Retrying a
  // paused allocator here would drain the cache before macOS can acknowledge
  // any reclaimed bytes.
  if (growthPaused())
    return {};
  const CacheReclaimResult reclaimed = cache_.reclaimOne(mode);
  if (reclaimed.madeProgress)
    signalResourceProgress();
  return reclaimed;
}

bool Engine::reclaimIdleState() noexcept {
  if (!model_.reclaimIdleState())
    return false;
  signalResourceProgress();
  return true;
}

// Host pressure pauses growth, and the pressure controller owns physical
// shrink. Backing that stays resident is outside that accounting: a request
// short of pages may take idle cached pages instead of being suspended and
// replaying its whole prefix once the pause lifts. Cache is only evicted when
// the resident idle pages can actually cover the shortfall; otherwise the
// request yields as before and the cache survives for later hits.
bool Engine::reuseIdleBackingWhilePaused(const TokenAdmission &admission) {
  if (reclaimIdleState())
    return true;
  const KvPoolSnapshot pool = cache_.snapshot().pool;
  // Cached prefixes can also have active owners; those pages cannot be reused.
  const uint32_t reusable = pool.pagesResident - pool.pagesActive;
  if (reusable < admission.additionalPages)
    return false;
  const CacheReclaimResult reused =
      cache_.reclaimOne(CacheReclaimMode::ReuseBacking);
  if (reused.madeProgress) {
    // An evicted state parks its buffers in the model's pool; under pressure
    // that memory goes back to the host now rather than waiting for the
    // next background pass.
    while (model_.reclaimIdleState()) {
    }
    signalResourceProgress();
  }
  return reused.madeProgress;
}

void Engine::suspendForGrowth(Request &active, uint64_t workEnd, double now) {
  if (!active.stateCell || active.suspended) {
    throw std::logic_error("request cannot be suspended for growth");
  }
  if (!active.stateBoundaries.empty()) {
    discardPendingStateBoundaries(active);
    scheduler_.setPrefillBoundary(active.request.id, std::nullopt);
  }
  model_.suspend(active.request.id);
  cache_.endRequest(active.request.id);
  active.stateCell.reset();
  active.suspended = true;
  active.resumeKvTargetTokens = workEnd;
  recoveringResources_ = true;
  active.replayTokens = static_cast<uint32_t>(active.exactTokens.size());
  scheduler_.suspendForResources(active.request.id);
  deferResourceRetry(active, now);
  ++counters_.resourceSuspensions;
}

uint64_t Engine::reclaimMemory(const MemoryReclaimDirective &directive) {
  if (!directive.reclaimEmptyKvExtents)
    return 0;

  uint64_t released = 0;
  while (const uint64_t idle = model_.reclaimIdleState())
    released += idle;
  const uint64_t remaining =
      released >= directive.targetBytes ? 0 : directive.targetBytes - released;
  // Even a zero-byte directive may release completely empty KV extents.
  released += cache_.reclaimCache(remaining, directive.evictAllUnpinnedPrefixes);
  // Evicted states park their buffers in the model's pool; a pressure pass
  // returns that memory to the host now rather than keeping it warm.
  while (model_.reclaimIdleState()) {
  }
  if (released)
    signalResourceProgress();
  return released;
}

void Engine::apply(const BatchPlan &plan,
                   std::span<const ModelStepResult> results,
                   double wallMilliseconds, bool representativePrefillTiming) {
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
    if (active.failure) {
      // An in-flight Metal command cannot be revoked safely. Its provisional
      // writes remain invisible, but a cancelled/deadline-expired request
      // must not publish cache state or emit output when that command drains.
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
      static_cast<void>(cache_.publishCommittedBlocks(
          active.request.id, active.exactTokens, promptProcessed,
          active.request.images));
      publishReachedStateBoundaries(active, promptProcessed);
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
      events_.tokens(active.request.id, result.outputTokens);
    }
    if (plan.kind == WorkKind::Decode) {
      draftedTokens += result.draftedTokens;
      acceptedDraftTokens += result.acceptedDraftTokens;
      // A terminal anchor is emitted without a target KV row; it never enters
      // a cached block.
      const uint32_t storedTokens =
          static_cast<uint32_t>(active.exactTokens.size()) -
          result.outputTokensWithoutKv;
      static_cast<void>(cache_.publishCommittedBlocks(
          active.request.id, active.exactTokens, storedTokens,
          active.request.images));
    }
    const uint64_t completionTokens =
        active.exactTokens.size() - active.promptTokens;
    const bool complete =
        result.finished || completionTokens >= active.request.maxNewTokens;
    if (result.outputTokensWithoutKv && !complete) {
      throw std::logic_error("model emitted an uncommitted token and continued");
    }
    schedulerResults.push_back({active.request.id, result.consumedPromptTokens,
                                complete, result.nextDecodeStage});
    if (plan.kind == WorkKind::Decode && waitsForMask(result.nextDecodeStage)) {
      events_.maskRequested(active.request.id, {});
    }
  }
  scheduler_.complete(plan, schedulerResults, wallMilliseconds,
                      representativePrefillTiming);
  events_.batchCompleted(plan.kind, plan.width(), inputTokens, outputTokens,
                         draftedTokens, acceptedDraftTokens, wallMilliseconds);

  for (size_t index = 0; index < results.size(); ++index) {
    const ModelStepResult &result = results[index];
    Request &active = request(result.requestId);
    if (active.failure) {
      Failure failure = std::move(*active.failure);
      active.failure.reset();
      if (failure.code == "cancelled") {
        finish(active, EngineFinishReason::Cancelled);
      } else {
        finishFailure(active, std::move(failure));
      }
    } else if (schedulerResults[index].finished) {
      finish(active, result.finished ? EngineFinishReason::Stop
                                     : EngineFinishReason::Length);
    }
  }
}

void Engine::finish(Request &active, EngineFinishReason reason) {
  if (active.restore) {
    if (!active.failure) active.failure = Failure{"cancelled", "request cancelled"};
    if (active.restore->ticket) active.restore->ticket->cancel();
    return;
  }
  if (active.finalized)
    return;
  if (reason == EngineFinishReason::Cancelled) {
    scheduler_.cancel(active.request.id);
  }
  active.finalized = true;
  const uint32_t completionTokens =
      active.exactTokens.size() > active.promptTokens
          ? static_cast<uint32_t>(active.exactTokens.size() -
                                  active.promptTokens)
          : 0;
  events_.completed(active.request.id, reason, active.promptTokens,
                    completionTokens);
  if (reason == EngineFinishReason::Cancelled) {
    ++counters_.cancelled;
  } else {
    static_cast<void>(retireCheckpoint(active));
    ++counters_.completed;
  }
  release(active);
}

void Engine::finishFailure(Request &active, Failure failure) {
  if (active.restore) {
    if (!active.failure) active.failure = std::move(failure);
    if (active.restore->ticket) active.restore->ticket->cancel();
    return;
  }
  if (active.finalized)
    return;
  scheduler_.fail(active.request.id);
  active.finalized = true;
  events_.failed(active.request.id, std::move(failure.code),
                 std::move(failure.message), failure.retryable);
  ++counters_.failed;
  release(active);
}

void Engine::finishCapacity(Request &active, const TokenAdmission &admission) {
  if (admission.allocationFailure != metal::AllocationFailure::None &&
      admission.allocationFailure != metal::AllocationFailure::Capacity) {
    finishFailure(active,
                  {"capacity_exhausted",
                   std::string("could not allocate KV target: ") +
                       metal::allocationFailureName(admission.allocationFailure) +
                       " (additional_pages=" +
                       std::to_string(admission.additionalPages) +
                       ", logical_pages_free=" +
                       std::to_string(admission.availablePages) + ")",
                   true});
    return;
  }
  if (active.finalized)
    return;
  scheduler_.fail(active.request.id);
  active.finalized = true;
  events_.capacityExhausted(active.request.id, admission.additionalPages,
                            admission.availablePages, 0);
  ++counters_.failed;
  release(active);
}

void Engine::release(Request &active) {
  active.resourceWait = {};
  if (active.stateCell || active.suspended) {
    discardPendingStateBoundaries(active);
    model_.end(active.request.id);
    cache_.endRequest(active.request.id);
    active.stateCell.reset();
    active.suspended = false;
    signalResourceProgress();
  }
}

void Engine::sweepTerminal() {
  for (auto iterator = requests_.begin(); iterator != requests_.end();) {
    const uint64_t id = iterator->first;
    Request &active = iterator->second;
    const Phase phase = scheduler_.phase(id);
    if (phase == Phase::Failed && !active.finalized) {
      finishFailure(active, {"deadline_exceeded", "request deadline elapsed"});
    }
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

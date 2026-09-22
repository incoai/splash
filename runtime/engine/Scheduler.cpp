#include "engine/Scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace splash::engine {
namespace {

// A lane yields to later arrivals at most this many prefill commands in a
// row: the other lanes of a full batch, each finishing one short prompt.
constexpr uint32_t kMaximumOvertakes =
    model::ExecutionLimits::maximumBatchWidth - 1;

constexpr double kContendedPrefillMilliseconds = 500.0;
constexpr uint32_t kMinimumPrefillRows = 64;

} // namespace

void Scheduler::submit(RequestSpec request) {
  Request state;
  state.spec = std::move(request);
  state.order = ++order_;
  requests_.emplace(state.spec.id, std::move(state));
}

void Scheduler::resourcesReady(uint64_t id, uint32_t processed) {
  Request &request = get(id);
  if (request.phase != Phase::Queued &&
      request.phase != Phase::WaitingResources &&
      request.phase != Phase::WaitingPrefix) {
    throw std::logic_error("only queued work can be admitted");
  }
  if (processed > request.spec.promptTokens) {
    throw std::invalid_argument("processed prompt exceeds request");
  }
  if (request.suspendedForResources) {
    throw std::logic_error("suspended request requires resumeFromResources");
  }
  request.promptProcessed = processed;
  request.decodeStage = request.spec.cohort == BatchCohort::Constrained
                            ? DecodeStage::RequestInitialMask
                            : DecodeStage::Regular;
  request.phase =
      processed == request.spec.promptTokens ? Phase::Decode : Phase::Prefill;
}

void Scheduler::suspendForResources(uint64_t id) {
  Request &request = get(id);
  if (request.phase != Phase::Prefill && request.phase != Phase::Decode) {
    throw std::logic_error("only runnable resident work can be suspended");
  }
  request.suspendedForResources = true;
  request.phase = Phase::WaitingResources;
}

void Scheduler::resumeFromResources(uint64_t id, uint32_t processed,
                                    uint32_t replayTokens) {
  Request &request = get(id);
  if (request.phase != Phase::WaitingResources ||
      !request.suspendedForResources) {
    throw std::logic_error("request is not suspended for resources");
  }
  if (processed >= replayTokens)
    throw std::invalid_argument("resource replay must leave an input token");
  request.spec.promptTokens = replayTokens;
  request.promptProcessed = processed;
  request.phase = Phase::Prefill;
  request.suspendedForResources = false;
}

void Scheduler::deferAdmission(uint64_t id) {
  Request &request = get(id);
  if (request.suspendedForResources ||
      (request.phase != Phase::Queued &&
       request.phase != Phase::WaitingResources &&
       request.phase != Phase::WaitingPrefix))
    throw std::logic_error("only unstarted work can wait for scheduling");
  request.phase = Phase::Queued;
}

void Scheduler::waitForResources(uint64_t id) {
  Request &request = get(id);
  if (request.phase != Phase::Queued &&
      request.phase != Phase::WaitingResources &&
      request.phase != Phase::WaitingPrefix) {
    throw std::logic_error("resident request cannot wait before admission");
  }
  request.phase = Phase::WaitingResources;
}

void Scheduler::waitForPrefix(uint64_t id) {
  Request &request = get(id);
  if (request.suspendedForResources ||
      (request.phase != Phase::Queued &&
       request.phase != Phase::WaitingResources &&
       request.phase != Phase::WaitingPrefix)) {
    throw std::logic_error("only unstarted requests can wait for a prefix");
  }
  request.phase = Phase::WaitingPrefix;
}

void Scheduler::maskReady(uint64_t id) {
  Request &request = get(id);
  if (request.phase != Phase::WaitingMask ||
      !waitsForMask(request.decodeStage)) {
    throw std::logic_error("request is not waiting for a mask");
  }
  request.phase = Phase::Decode;
}

void Scheduler::cancel(uint64_t id) {
  Request &request = get(id);
  if (!terminal(request.phase))
    request.phase = Phase::Cancelled;
}

void Scheduler::fail(uint64_t id) {
  Request &request = get(id);
  if (terminal(request.phase))
    return;
  request.phase = Phase::Failed;
}

void Scheduler::remove(uint64_t id) {
  if (active_) {
    for (const BatchItem &item : active_->items) {
      if (item.requestId == id) {
        throw std::logic_error("cannot remove an active batch member");
      }
    }
  }
  auto found = requests_.find(id);
  if (found == requests_.end() || !terminal(found->second.phase)) {
    throw std::logic_error("only terminal requests can be removed");
  }
  requests_.erase(found);
}

void Scheduler::setPrefillBoundary(uint64_t id,
                                         std::optional<uint32_t> boundary) {
  Request &request = get(id);
  if (boundary && (*boundary <= request.promptProcessed ||
                   *boundary > request.spec.promptTokens)) {
    throw std::invalid_argument("invalid prefill boundary");
  }
  request.prefillBoundary = boundary;
}

bool Scheduler::expireDeadlines(double now) {
  bool changed = false;
  for (auto &[_, request] : requests_) {
    const bool inFlight =
        active_ && std::any_of(active_->items.begin(), active_->items.end(),
                               [&](const BatchItem &item) {
                                 return item.requestId == request.spec.id;
                               });
    if (!inFlight && !terminal(request.phase) &&
        request.spec.deadlineMilliseconds <= now) {
      request.phase = Phase::Failed;
      changed = true;
    }
  }
  return changed;
}

std::vector<uint64_t> Scheduler::admissionOrder() const {
  std::vector<const Request *> ready;
  ready.reserve(requests_.size());
  for (const auto &[_, request] : requests_) {
    if (request.phase == Phase::Queued ||
        request.phase == Phase::WaitingResources ||
        request.phase == Phase::WaitingPrefix) {
      ready.push_back(&request);
    }
  }
  std::sort(ready.begin(), ready.end(), byPriorityThenOrder);
  std::vector<uint64_t> result;
  result.reserve(ready.size());
  for (const Request *request : ready)
    result.push_back(request->spec.id);
  return result;
}

std::vector<uint64_t> Scheduler::prefillAdmissionOrder(
    std::span<const PrefillAdmission> candidates) const {
  std::vector<PrefillRequestView> pending;
  pending.reserve(candidates.size());
  for (const auto &candidate : candidates) {
    const Request &value = get(candidate.requestId);
    if (value.suspendedForResources ||
        (value.phase != Phase::Queued && value.phase != Phase::WaitingResources &&
         value.phase != Phase::WaitingPrefix) ||
        candidate.cachedTokens >= value.spec.promptTokens)
      throw std::logic_error("invalid pending prefill admission");
    pending.push_back({&value, candidate.cachedTokens});
  }
  std::vector<PrefillRequestView> ready;
  ready.reserve(requests_.size());
  for (const auto &[_, request] : requests_)
    if (request.phase == Phase::Prefill)
      ready.push_back({&request, request.promptProcessed});
  for (const auto &request : pending)
    ready.push_back(request);
  std::vector<uint64_t> result;
  if (const auto plan = planPrefill(std::move(ready))) {
    const auto decode = nextDecode();
    if (decode && get(decode->items.front().requestId).spec.priority <
                      get(plan->items.front().requestId).spec.priority)
      return result;
    for (const auto &item : plan->items)
      if (get(item.requestId).phase != Phase::Prefill)
        result.push_back(item.requestId);
  }
  return result;
}

std::optional<BatchPlan> Scheduler::next() const {
  if (active_)
    return std::nullopt;
  auto decode = nextDecode();
  auto prefill = nextPrefill();
  if (!decode)
    return prefill;
  if (!prefill)
    return decode;

  const RequestPriority decodePriority =
      get(decode->items.front().requestId).spec.priority;
  const RequestPriority prefillPriority =
      get(prefill->items.front().requestId).spec.priority;
  if (decodePriority != prefillPriority) {
    return decodePriority < prefillPriority ? std::move(decode)
                                             : std::move(prefill);
  }

  // Prefill and fixed-eight decode use different Metal graphs and cannot be
  // packed into one command. Honor request priority first, then alternate at
  // command boundaries so equal-priority work cannot starve.
  return lastCommittedKind_ == WorkKind::Decode ? std::move(prefill)
                                                 : std::move(decode);
}

std::optional<BatchPlan> Scheduler::nextPrefill() const {
  std::vector<PrefillRequestView> ready;
  for (const auto &[_, request] : requests_) {
    if (request.phase == Phase::Prefill)
      ready.push_back({&request, request.promptProcessed});
  }
  return planPrefill(std::move(ready));
}

std::optional<BatchPlan>
Scheduler::planPrefill(std::vector<PrefillRequestView> ready) const {
  if (ready.empty())
    return std::nullopt;
  const auto dispatchRemaining = [](const PrefillRequestView &view) {
    uint32_t end = view.request->spec.promptTokens;
    if (view.request->prefillBoundary)
      end = std::min(end, *view.request->prefillBoundary);
    return end - view.promptProcessed;
  };
  // Order by the complete remaining prompt, independently of state capture
  // boundaries. After kMaximumOvertakes consecutive skips,
  // an older lane leads the next command to prevent starvation.
  const auto overdue = [](const PrefillRequestView &view) {
    return view.request->overtaken >= kMaximumOvertakes;
  };
  std::sort(ready.begin(), ready.end(),
            [&](const PrefillRequestView &a, const PrefillRequestView &b) {
              if (a.request->spec.priority != b.request->spec.priority)
                return a.request->spec.priority < b.request->spec.priority;
              if (overdue(a) != overdue(b))
                return overdue(a);
              const uint32_t remainingA =
                  a.request->spec.promptTokens - a.promptProcessed;
              const uint32_t remainingB =
                  b.request->spec.promptTokens - b.promptProcessed;
              if (remainingA != remainingB)
                return remainingA < remainingB;
              return a.request->order < b.request->order;
            });
  const RequestPriority selectedPriority = ready.front().request->spec.priority;

  BatchPlan plan;
  plan.kind = WorkKind::Prefill;
  uint32_t budget = prefillBudget(ready.front(), ready);
  for (const PrefillRequestView &view : ready) {
    if (!budget || view.request->spec.priority != selectedPriority ||
        plan.width() == model::ExecutionLimits::maximumBatchWidth)
      break;
    const uint32_t rows = std::min(dispatchRemaining(view), budget);
    plan.items.push_back(
        {view.request->spec.id, rows, view.promptProcessed});
    budget -= rows;
  }
  return plan;
}

uint32_t Scheduler::prefillBudget(
    const PrefillRequestView &leader,
    std::span<const PrefillRequestView> ready) const {
  const uint32_t maximum = model::ExecutionLimits::prefillTokenBudget;
  if (prefillMillisecondsPerToken_ <= 0.0)
    return maximum;
  uint32_t rows = maximum;
  while (rows > kMinimumPrefillRows &&
         rows * prefillMillisecondsPerToken_ > kContendedPrefillMilliseconds)
    rows /= 2;
  const bool leaderFinishing =
      leader.request->spec.promptTokens - leader.promptProcessed <= rows;
  const bool contended = std::any_of(
      requests_.begin(), requests_.end(), [&](const auto &entry) {
        const Request &peer = entry.second;
        return (peer.phase == Phase::Decode || peer.phase == Phase::WaitingMask) &&
               peer.spec.priority <= leader.request->spec.priority;
      }) || std::any_of(ready.begin(), ready.end(), [&](
                           const PrefillRequestView &peer) {
        return peer.request->spec.id != leader.request->spec.id &&
               peer.request->spec.priority <= leader.request->spec.priority &&
               (leaderFinishing || peer.request->spec.promptTokens -
                                       peer.promptProcessed <= rows);
      });
  if (!contended)
    return maximum;

  // Keep long prefills packed. Bound commands for peers decoding or waiting
  // for a CPU mask, and for peers that can finish prefill within this slice.
  // The first sample and minimum matrix shape remain limits.
  return rows;
}

std::optional<BatchPlan> Scheduler::nextDecode() const {
  std::vector<const Request *> ready;
  for (const auto &[_, request] : requests_) {
    if (request.phase == Phase::Decode)
      ready.push_back(&request);
  }
  if (ready.empty())
    return std::nullopt;
  std::sort(ready.begin(), ready.end(), [](const Request *a, const Request *b) {
    if (a->spec.priority != b->spec.priority)
      return a->spec.priority < b->spec.priority;
    if (a->lastDecodeDispatch != b->lastDecodeDispatch)
      return a->lastDecodeDispatch < b->lastDecodeDispatch;
    return a->order < b->order;
  });
  const BatchCohort cohort = ready.front()->spec.cohort;
  const DecodeStage decodeStage = ready.front()->decodeStage;
  const RequestPriority selectedPriority = ready.front()->spec.priority;
  BatchPlan plan;
  plan.kind = WorkKind::Decode;
  plan.cohort = cohort;
  plan.decodeStage = decodeStage;
  // Applying the initial mask can terminate a request or start drafting.
  // Classify that branch one request at a time; regular decode can batch.
  const uint32_t maximumWidth =
      decodeStage == DecodeStage::ApplyInitialMask
          ? 1
          : model::ExecutionLimits::maximumBatchWidth;
  for (const Request *request : ready) {
    if (request->spec.priority != selectedPriority ||
        (request->spec.cohort == BatchCohort::Constrained) !=
            (cohort == BatchCohort::Constrained) ||
        request->decodeStage != decodeStage)
      continue;
    if (request->spec.cohort == BatchCohort::Sampling)
      plan.cohort = BatchCohort::Sampling;
    plan.items.push_back({request->spec.id, 0, 0});
    if (plan.width() == maximumWidth)
      break;
  }
  return plan;
}

void Scheduler::commit(const BatchPlan &plan) {
  if (active_ || plan.empty() ||
      plan.width() > model::ExecutionLimits::maximumBatchWidth) {
    throw std::logic_error("invalid scheduler commit");
  }
  for (const BatchItem &item : plan.items) {
    const Request &request = get(item.requestId);
    const Phase expected =
        plan.kind == WorkKind::Prefill ? Phase::Prefill : Phase::Decode;
    if (request.phase != expected ||
        (plan.kind == WorkKind::Prefill &&
         (!item.tokenCount || item.promptOffset != request.promptProcessed)) ||
        (plan.kind == WorkKind::Decode &&
         (item.tokenCount || item.promptOffset ||
          request.decodeStage != plan.decodeStage))) {
      throw std::logic_error("batch no longer matches scheduler state");
    }
  }
  active_ = plan;
  lastCommittedKind_ = plan.kind;
  if (plan.kind == WorkKind::Prefill) {
    ++counters_.prefillBatches;
    for (const BatchItem &item : plan.items)
      counters_.prefillRows += item.tokenCount;
    uint64_t youngestServed = 0;
    for (const BatchItem &item : plan.items)
      youngestServed = std::max(youngestServed, get(item.requestId).order);
    for (auto &[id, request] : requests_) {
      if (terminal(request.phase) || request.phase == Phase::Decode ||
          request.phase == Phase::WaitingMask || request.suspendedForResources)
        continue;
      const bool served = std::any_of(
          plan.items.begin(), plan.items.end(),
          [id](const BatchItem &item) { return item.requestId == id; });
      if (served)
        request.overtaken = 0;
      else if (request.order < youngestServed)
        ++request.overtaken;
    }
  } else {
    const uint64_t dispatchOrder = ++decodeDispatchOrder_;
    bool hasGreedy = false;
    bool hasSampling = false;
    for (const BatchItem &item : plan.items) {
      Request &request = get(item.requestId);
      request.lastDecodeDispatch = dispatchOrder;
      const BatchCohort cohort = request.spec.cohort;
      hasGreedy = hasGreedy || cohort == BatchCohort::Greedy;
      hasSampling = hasSampling || cohort == BatchCohort::Sampling;
    }
    ++counters_.decodeBatches;
    ++counters_.decodeBatchesByWidth[plan.width() - 1];
    if (hasGreedy && hasSampling)
      ++counters_.decodeMixedGreedySamplingBatches;
  }
}

void Scheduler::complete(const BatchPlan &plan,
                         std::span<const StepResult> results,
                         double wallMilliseconds,
                         bool representativePrefillTiming) {
  if (!active_ || active_->kind != plan.kind ||
      active_->items.size() != plan.items.size() ||
      results.size() != plan.items.size()) {
    throw std::logic_error("completion does not match active batch");
  }
  for (size_t index = 0; index < results.size(); ++index) {
    const BatchItem &item = plan.items[index];
    const StepResult &result = results[index];
    if (result.requestId != item.requestId) {
      throw std::logic_error("completion request order changed");
    }
    Request &request = get(item.requestId);
    if (plan.kind == WorkKind::Prefill) {
      if (result.consumedPromptTokens != item.tokenCount ||
          item.promptOffset != request.promptProcessed) {
        throw std::logic_error("prefill completion row count changed");
      }
      request.promptProcessed += result.consumedPromptTokens;
      if (request.prefillBoundary &&
          request.promptProcessed == *request.prefillBoundary) {
        request.prefillBoundary.reset();
      }
      // A stop token or a one-token budget is selected by prefill itself.
      request.phase = result.finished ? Phase::Completed
                      : request.promptProcessed == request.spec.promptTokens
                          ? Phase::Decode
                          : Phase::Prefill;
    } else {
      request.decodeStage = result.nextDecodeStage;
      request.phase = result.finished ? Phase::Completed
                      : waitsForMask(result.nextDecodeStage)
                          ? Phase::WaitingMask
                          : Phase::Decode;
    }
  }
  if (representativePrefillTiming && plan.kind == WorkKind::Prefill) {
    uint32_t rows = 0;
    for (const BatchItem &item : plan.items)
      rows += item.tokenCount;
    observePrefill(rows, wallMilliseconds);
  }
  active_.reset();
}

void Scheduler::observePrefill(uint32_t rows, double wallMilliseconds) {
  // Tiny tails are dominated by fixed command costs, not prefill throughput.
  if (rows < kMinimumPrefillRows || !std::isfinite(wallMilliseconds) ||
      wallMilliseconds <= 0.0)
    return;
  const double observed = wallMilliseconds / rows;
  prefillMillisecondsPerToken_ =
      prefillMillisecondsPerToken_ > 0.0
          ? 0.75 * prefillMillisecondsPerToken_ + 0.25 * observed
          : observed;
}

Phase Scheduler::phase(uint64_t id) const { return get(id).phase; }

uint32_t Scheduler::promptProcessed(uint64_t id) const {
  return get(id).promptProcessed;
}

SchedulerSnapshot Scheduler::snapshot() const noexcept {
  SchedulerSnapshot result = counters_;
  for (const auto &[_, request] : requests_) {
    switch (request.phase) {
    case Phase::Queued:
      ++result.queued;
      break;
    case Phase::WaitingResources:
      ++result.waitingResources;
      break;
    case Phase::WaitingPrefix:
      ++result.waitingPrefix;
      break;
    case Phase::Prefill:
      ++result.prefilling;
      break;
    case Phase::Decode:
      ++result.decoding;
      break;
    case Phase::WaitingMask:
      ++result.waitingMask;
      break;
    case Phase::Completed:
    case Phase::Cancelled:
    case Phase::Failed:
      ++result.terminal;
      break;
    }
  }
  return result;
}

Scheduler::Request &Scheduler::get(uint64_t id) {
  auto found = requests_.find(id);
  if (found == requests_.end())
    throw std::out_of_range("unknown request");
  return found->second;
}

const Scheduler::Request &Scheduler::get(uint64_t id) const {
  auto found = requests_.find(id);
  if (found == requests_.end())
    throw std::out_of_range("unknown request");
  return found->second;
}

bool Scheduler::terminal(Phase phase) noexcept {
  return phase == Phase::Completed || phase == Phase::Cancelled ||
         phase == Phase::Failed;
}

bool Scheduler::byPriorityThenOrder(const Request *a,
                                    const Request *b) noexcept {
  if (a->spec.priority != b->spec.priority)
    return a->spec.priority < b->spec.priority;
  return a->order < b->order;
}

} // namespace splash::engine

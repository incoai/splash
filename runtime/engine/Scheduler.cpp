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

// The slice a contended prefill command aims for: while peers of its
// priority decode, the budget halves until its estimated cost fits, so they
// wait about this long between steps. A latency target, not a deadline: the
// first command has no measurement and the budget stops at
// kMinimumPrefillRows. A command with a fixed cost (a streamed target's
// expert reads, which any chunk pays in full) gets twice that cost instead,
// so its rows take at least as long as its fixed part and its peers wait
// that long, seconds rather than this slice; where its model runs a command
// in passes over its layers (Model::prefillPasses), it keeps its rows
// whatever the contention, and each pass aims for this slice, contended or
// not, so that decode steps run between passes and a cancel waits for one
// at most (passesFor). A lane that streams its history sizes its commands
// by its own reads instead (streamingRows).
constexpr double kContendedPrefillMilliseconds = 500.0;
// The fewest rows a contended prefill command is cut to, and the fewest whose
// time measures prefill throughput: below it a command's fixed costs dominate.
constexpr uint32_t kMinimumPrefillRows = 64;
// The weight of a command's measured time in a running estimate (per row,
// fixed, or a lane's wait on its history's reads): one unusually slow or
// fast command moves it a quarter of the way, and a lasting change carries
// it within a few commands.
constexpr double kPrefillTimeWeight = 0.25;

// A command excludes at most the lanes of a full batch: a linear search.
bool listed(std::span<const uint64_t> ids, uint64_t id) noexcept {
  return std::find(ids.begin(), ids.end(), id) != ids.end();
}

// The rows of an uncontended command of a lane with `remaining` rows to
// prefill: up to a chunk, or for a lane whose commands may be prefill groups
// of up to `groupRows` rows, its remaining chunks split evenly over the
// fewest groups that hold them (a short last group would pay a whole pass
// over a target's streamed experts for a few rows).
uint32_t uncontendedRows(uint32_t remaining, uint32_t groupRows) noexcept {
  constexpr uint32_t chunk = model::ExecutionLimits::prefillTokenBudget;
  if (remaining <= chunk || groupRows <= chunk)
    return std::min(remaining, chunk);
  const uint32_t chunks = (remaining + chunk - 1) / chunk;
  const uint32_t groups = (chunks + groupRows / chunk - 1) / (groupRows / chunk);
  return std::min(remaining, (chunks + groups - 1) / groups * chunk);
}

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
  if (processed > request.spec.prefillTokens) {
    throw std::invalid_argument("processed prompt exceeds request");
  }
  if (request.suspended) {
    throw std::logic_error("suspended request requires resumeFromResources");
  }
  request.promptProcessed = processed;
  request.decodeStage = DecodeStage::Regular;
  request.phase =
      processed == request.spec.prefillTokens ? Phase::Decode : Phase::Prefill;
  if (request.phase == Phase::Decode)
    startDecodeGap(request);
}

void Scheduler::suspendForResources(uint64_t id) {
  Request &request = get(id);
  if (request.phase != Phase::Prefill && request.phase != Phase::Decode) {
    throw std::logic_error("only runnable resident work can be suspended");
  }
  // A command between its passes is left: the lane replays its rows.
  request.passes.reset();
  request.suspended = true;
  request.phase = Phase::WaitingResources;
  dropStaleDecodeDebt();
}

void Scheduler::resumeFromResources(uint64_t id, uint32_t processed,
                                    uint32_t replayTokens) {
  Request &request = get(id);
  if (request.phase != Phase::WaitingResources || !request.suspended) {
    throw std::logic_error("request is not suspended for resources");
  }
  if (processed >= replayTokens)
    throw std::invalid_argument("resource replay must leave an input token");
  request.spec.prefillTokens = replayTokens;
  request.promptProcessed = processed;
  request.phase = Phase::Prefill;
  request.suspended = false;
}

void Scheduler::deferAdmission(uint64_t id) {
  Request &request = get(id);
  if (request.suspended ||
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
  if (request.suspended ||
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
  startDecodeGap(request);
}

void Scheduler::setStreamsHistory(uint64_t id, bool streams) {
  get(id).streamsHistory = streams;
}

void Scheduler::setPrefillGroupRows(uint64_t id, uint32_t rows) {
  get(id).spec.prefillGroupRows = rows;
}

void Scheduler::cancel(uint64_t id) {
  Request &request = get(id);
  request.passes.reset();
  if (!terminal(request.phase))
    request.phase = Phase::Cancelled;
  dropStaleDecodeDebt();
}

void Scheduler::finish(uint64_t id) {
  Request &request = get(id);
  request.passes.reset();
  if (!terminal(request.phase))
    request.phase = Phase::Completed;
  dropStaleDecodeDebt();
}

void Scheduler::fail(uint64_t id) {
  Request &request = get(id);
  request.passes.reset();
  if (terminal(request.phase))
    return;
  request.phase = Phase::Failed;
  dropStaleDecodeDebt();
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
      request.passes.reset();
      changed = true;
    }
  }
  if (changed)
    dropStaleDecodeDebt();
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
    if (value.suspended ||
        (value.phase != Phase::Queued && value.phase != Phase::WaitingResources &&
         value.phase != Phase::WaitingPrefix) ||
        candidate.cachedTokens >= value.spec.prefillTokens)
      throw std::logic_error("invalid pending prefill admission");
    pending.push_back({&value, candidate.cachedTokens});
  }
  std::vector<PrefillRequestView> ready;
  ready.reserve(requests_.size());
  std::vector<uint64_t> result;
  for (const auto &[_, request] : requests_) {
    // A command between its passes runs before any new prefill.
    if (request.phase == Phase::Prefill && request.passes)
      return result;
    if (request.phase == Phase::Prefill)
      ready.push_back({&request, request.promptProcessed});
  }
  for (const auto &request : pending)
    ready.push_back(request);
  if (const auto plan = planPrefill(std::move(ready))) {
    // Admission ranks candidates against every decoder, left out of a plan
    // or not.
    const auto decode = nextDecode({});
    if (decode && get(decode->items.front().requestId).spec.priority <
                      get(plan->items.front().requestId).spec.priority)
      return result;
    for (const auto &item : plan->items)
      if (get(item.requestId).phase != Phase::Prefill)
        result.push_back(item.requestId);
  }
  return result;
}

std::optional<RequestPriority> Scheduler::highestRunnablePriority() const noexcept {
  std::optional<RequestPriority> result;
  for (const auto &[_, request] : requests_) {
    if ((request.phase == Phase::Prefill || request.phase == Phase::Decode) &&
        (!result || request.spec.priority < *result))
      result = request.spec.priority;
  }
  return result;
}

std::optional<BatchPlan> Scheduler::next(std::span<const uint64_t> excluded) const {
  if (active_)
    return std::nullopt;
  auto decode = nextDecode(excluded);
  auto prefill = nextPrefill(excluded);
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

  // Prefill and decode use different Metal graphs and cannot share one
  // command. Honor request priority first. Then decode runs while
  // prefill owes it time, and otherwise the kinds alternate at command
  // boundaries so equal-priority work cannot starve.
  if (decodeDebtMilliseconds_ > 0.0)
    return decode;
  return lastCommittedKind_ == WorkKind::Decode ? std::move(prefill)
                                                 : std::move(decode);
}

std::optional<BatchPlan> Scheduler::nextPrefill(std::span<const uint64_t> excluded) const {
  // A command between its passes goes on before any other prefill, whose
  // rows would take the model's prefill memory its rows' work is in.
  for (const auto &[id, request] : requests_) {
    if (request.phase != Phase::Prefill || !request.passes)
      continue;
    if (listed(excluded, id))
      return std::nullopt;
    BatchPlan plan;
    plan.kind = WorkKind::Prefill;
    plan.items.push_back({id, request.passes->rows, request.promptProcessed});
    plan.pass = request.passes->next;
    plan.passes = request.passes->passes;
    return plan;
  }
  std::vector<PrefillRequestView> ready;
  for (const auto &[id, request] : requests_) {
    if (request.phase == Phase::Prefill && !listed(excluded, id))
      ready.push_back({&request, request.promptProcessed});
  }
  return planPrefill(std::move(ready));
}

std::optional<BatchPlan>
Scheduler::planPrefill(std::vector<PrefillRequestView> ready) const {
  if (ready.empty())
    return std::nullopt;
  // Order by the remaining prompt. After kMaximumOvertakes consecutive skips,
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
                  a.request->spec.prefillTokens - a.promptProcessed;
              const uint32_t remainingB =
                  b.request->spec.prefillTokens - b.promptProcessed;
              if (remainingA != remainingB)
                return remainingA < remainingB;
              return a.request->order < b.request->order;
            });
  const PrefillRequestView &leader = ready.front();
  const RequestPriority selectedPriority = leader.request->spec.priority;

  BatchPlan plan;
  plan.kind = WorkKind::Prefill;
  uint32_t budget = prefillBudget(leader, ready);
  uint64_t pairs = 0;
  // The command's rows so far: a lane joins only while they lie within the
  // first chunk, and its group rows bound the command's
  // (ExecutionLimits::prefillTokenBudget).
  uint32_t planned = 0;
  for (const PrefillRequestView &view : ready) {
    if (planned > model::ExecutionLimits::prefillTokenBudget)
      break;
    budget = std::min(budget, view.request->spec.prefillGroupRows - planned);
    if (!budget || view.request->spec.priority != selectedPriority ||
        plan.width() == model::ExecutionLimits::maximumBatchWidth)
      break;
    // A lane that streams its history runs alone: nothing joins a command
    // it leads, and it joins none it does not lead; overtaken often enough,
    // it leads one.
    if (view.request->streamsHistory && &view != &leader)
      continue;
    uint32_t rows =
        std::min(view.request->spec.prefillTokens - view.promptProcessed, budget);
    // The command's attention stays within maximumAttentionPairs_: a lane's
    // rows halve until they fit, and one that does not fit at
    // kMinimumPrefillRows waits for a command of its own.
    const auto exceeds = [&](uint32_t candidate) {
      return pairs + attentionPairs(view.promptProcessed, candidate) > maximumAttentionPairs_;
    };
    while (rows > kMinimumPrefillRows && exceeds(rows))
      rows /= 2;
    if (!plan.items.empty() && exceeds(rows))
      break;
    pairs += attentionPairs(view.promptProcessed, rows);
    plan.items.push_back(
        {view.request->spec.id, rows, view.promptProcessed});
    planned += rows;
    budget -= rows;
    if (leader.request->streamsHistory)
      break;
    // A leader whose command runs in passes runs alone.
    if (&view == &leader) {
      plan.passes = passesFor(*leader.request, rows);
      if (plan.passes > 1)
        break;
    }
  }
  return plan;
}

uint32_t Scheduler::prefillBudget(
    const PrefillRequestView &leader,
    std::span<const PrefillRequestView> ready) const {
  const uint32_t leaderRemaining =
      leader.request->spec.prefillTokens - leader.promptProcessed;
  // Passes bound a command's time, which fewer rows would not shorten.
  if (leader.request->spec.prefillPasses > 1 && !leader.request->streamsHistory)
    return uncontendedRows(leaderRemaining, leader.request->spec.prefillGroupRows);
  const uint32_t maximum = model::ExecutionLimits::prefillTokenBudget;
  if (prefillMillisecondsPerToken_ <= 0.0 && prefillFixedMilliseconds_ <= 0.0)
    return maximum;
  uint32_t rows = maximum;
  const double slice = std::max(kContendedPrefillMilliseconds, 2.0 * prefillFixedMilliseconds_);
  while (rows > kMinimumPrefillRows &&
         prefillFixedMilliseconds_ + rows * prefillMillisecondsPerToken_ > slice)
    rows /= 2;
  const bool leaderFinishing = leaderRemaining <= rows;
  const bool streams = leader.request->streamsHistory;
  const bool contended = std::any_of(
      requests_.begin(), requests_.end(), [&](const auto &entry) {
        const Request &peer = entry.second;
        return (peer.phase == Phase::Decode || peer.phase == Phase::WaitingMask) &&
               peer.spec.priority <= leader.request->spec.priority;
      }) || std::any_of(ready.begin(), ready.end(), [&](
                           const PrefillRequestView &peer) {
        return peer.request->spec.id != leader.request->spec.id &&
               peer.request->spec.priority <= leader.request->spec.priority &&
               (streams || leaderFinishing ||
                peer.request->spec.prefillTokens - peer.promptProcessed <= rows);
      });
  // Bound commands for peers that decode or wait for a CPU mask, including
  // decoders left out of this command (their streams resume once their
  // memory lands), and for peers that can finish prefill within this slice,
  // or, beside a lane that streams its history and so runs alone, any ready
  // peer. The first sample and minimum matrix shape remain limits.
  if (contended)
    return streams ? streamingRows(*leader.request, rows) : rows;
  // A leader that finishes within the full budget ends the command at its
  // last row: prefill cost is linear above the slice, so shortest-first
  // sequential commands minimise first-token latency. Long prefills share a
  // command only when no lane finishes. A leader that runs prefill groups
  // takes up to a group; planPrefill gives what its rows leave to the next
  // lanes within ExecutionLimits::prefillTokenBudget's contract.
  return uncontendedRows(leaderRemaining, leader.request->spec.prefillGroupRows);
}

uint32_t Scheduler::passesFor(const Request &leader, uint32_t rows) const {
  if (leader.spec.prefillPasses <= 1 || leader.streamsHistory)
    return 1;
  const double estimate = prefillFixedMilliseconds_ + rows * prefillMillisecondsPerToken_;
  return static_cast<uint32_t>(std::clamp(std::ceil(estimate / kContendedPrefillMilliseconds), 1.0,
                                          static_cast<double>(leader.spec.prefillPasses)));
}

uint32_t Scheduler::streamingRows(const Request &leader, uint32_t contendedRows) const {
  if (!leader.streamMilliseconds)
    return contendedRows;
  uint32_t rows = std::max(leader.spec.prefillGroupRows, contendedRows);
  while (rows / 2 >= contendedRows &&
         rows / 2 * prefillMillisecondsPerToken_ >= *leader.streamMilliseconds)
    rows /= 2;
  return rows;
}

std::optional<BatchPlan> Scheduler::nextDecode(std::span<const uint64_t> excluded) const {
  std::vector<const Request *> ready;
  for (const auto &[id, request] : requests_) {
    if (request.phase == Phase::Decode && !listed(excluded, id))
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
  const DecodeStage decodeStage = ready.front()->decodeStage;
  const RequestPriority selectedPriority = ready.front()->spec.priority;
  BatchPlan plan;
  plan.kind = WorkKind::Decode;
  plan.decodeStage = decodeStage;
  for (const Request *request : ready) {
    if (request->spec.priority != selectedPriority ||
        request->decodeStage != decodeStage)
      continue;
    plan.items.push_back({request->spec.id, 0, 0});
    if (plan.width() == model::ExecutionLimits::maximumBatchWidth)
      break;
  }
  return plan;
}

void Scheduler::commit(const BatchPlan &plan, std::span<const uint64_t> excluded) {
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
  if (plan.passes > 1) {
    Request &lane = get(plan.items.front().requestId);
    if (plan.width() != 1 || plan.kind != WorkKind::Prefill || plan.pass >= plan.passes ||
        (lane.passes ? plan.pass != lane.passes->next || plan.passes != lane.passes->passes ||
                           plan.items.front().tokenCount != lane.passes->rows
                     : plan.pass != 0))
      throw std::logic_error("batch no longer matches scheduler state");
    if (!lane.passes)
      lane.passes = Request::Passes{.passes = plan.passes, .rows = plan.items.front().tokenCount};
  }
  excluded_.assign(excluded.begin(), excluded.end());
  active_ = plan;
  lastCommittedKind_ = plan.kind;
  // A command counts, and passes others over, once: with its first pass.
  if (plan.kind == WorkKind::Prefill && !plan.pass) {
    ++counters_.prefillBatches;
    for (const BatchItem &item : plan.items)
      counters_.prefillRows += item.tokenCount;
    uint64_t youngestServed = 0;
    for (const BatchItem &item : plan.items)
      youngestServed = std::max(youngestServed, get(item.requestId).order);
    for (auto &[id, request] : requests_) {
      if (terminal(request.phase) || request.phase == Phase::Decode ||
          request.phase == Phase::WaitingMask || request.suspended ||
          listed(excluded, id))
        continue;
      const bool served = std::any_of(
          plan.items.begin(), plan.items.end(),
          [id](const BatchItem &item) { return item.requestId == id; });
      if (served)
        request.overtaken = 0;
      else if (request.order < youngestServed)
        ++request.overtaken;
    }
  } else if (plan.kind == WorkKind::Decode) {
    const uint64_t dispatchOrder = ++decodeDispatchOrder_;
    for (const BatchItem &item : plan.items)
      get(item.requestId).lastDecodeDispatch = dispatchOrder;
    ++counters_.decodeBatches;
    ++counters_.decodeBatchesByWidth[plan.width() - 1];
  }
}

void Scheduler::complete(const BatchPlan &plan,
                         std::span<const StepResult> results,
                         double wallMilliseconds,
                         bool representativePrefillTiming,
                         double hostMilliseconds,
                         std::optional<double> streamMilliseconds) {
  if (!active_ || active_->kind != plan.kind ||
      active_->items.size() != plan.items.size() ||
      results.size() != plan.items.size()) {
    throw std::logic_error("completion does not match active batch");
  }
  if (streamMilliseconds && (plan.kind != WorkKind::Prefill || plan.items.size() != 1))
    throw std::logic_error("only a prefill command of one lane streams its history");
  const bool timed = std::isfinite(wallMilliseconds) && wallMilliseconds > 0.0;
  // The lane of a prefill command that streamed its history, alone in it. A
  // lane that streams may run commands that read nothing yet, before its
  // first pages spill: those say nothing of its reads.
  Request *const streaming =
      streamMilliseconds ? &get(plan.items.front().requestId) : nullptr;
  // Its wait on the reads, which its waits on host steps include; none
  // without a measurement.
  const std::optional<double> stream =
      streaming && std::isfinite(*streamMilliseconds) && std::isfinite(hostMilliseconds)
          ? std::optional(std::clamp(*streamMilliseconds, 0.0, std::max(hostMilliseconds, 0.0)))
          : std::nullopt;
  if (timed) {
    commandMilliseconds_ += wallMilliseconds;
    streamingCommands_ += streaming != nullptr;
  }
  for (size_t index = 0; index < results.size(); ++index) {
    const BatchItem &item = plan.items[index];
    const StepResult &result = results[index];
    if (result.requestId != item.requestId) {
      throw std::logic_error("completion request order changed");
    }
    Request &request = get(item.requestId);
    if (plan.kind == WorkKind::Prefill && plan.pass + 1 < plan.passes) {
      // A pass before the last consumes nothing; a request it ends leaves
      // the rest.
      if (result.finished) {
        request.phase = Phase::Completed;
        request.passes.reset();
      } else if (result.consumedPromptTokens || !request.passes) {
        throw std::logic_error("prefill completion row count changed");
      } else {
        ++request.passes->next;
      }
    } else if (plan.kind == WorkKind::Prefill) {
      if (result.consumedPromptTokens != item.tokenCount ||
          item.promptOffset != request.promptProcessed) {
        throw std::logic_error("prefill completion row count changed");
      }
      request.promptProcessed += result.consumedPromptTokens;
      // A stop token or a one-token budget is selected by prefill itself; a
      // constrained prompt waits for the mask its first token needs.
      if (result.finished) {
        request.phase = Phase::Completed;
      } else if (request.promptProcessed != request.spec.prefillTokens) {
        request.phase = Phase::Prefill;
      } else if (waitsForMask(result.nextDecodeStage)) {
        request.decodeStage = result.nextDecodeStage;
        request.phase = Phase::WaitingMask;
      } else {
        request.phase = Phase::Decode;
        startDecodeGap(request);
      }
    } else {
      // The gap since its last decode command, recorded when a streaming
      // prefill command ran within it.
      if (request.decodeGapStreams != streamingCommands_)
        decodeGaps_.add(
            static_cast<uint64_t>((commandMilliseconds_ - request.decodeGapStart) * 1e6));
      startDecodeGap(request);
      request.decodeStage = result.nextDecodeStage;
      request.phase = result.finished ? Phase::Completed : Phase::Decode;
    }
  }
  // A streaming command trains the command estimate as if it had not waited
  // on its history's reads, and its lane's own estimate with that wait. A
  // command in passes trains it once, with its passes' time together.
  double commandWall = wallMilliseconds, commandHost = hostMilliseconds;
  bool representative = representativePrefillTiming;
  if (plan.kind == WorkKind::Prefill && plan.passes > 1) {
    Request &lane = get(plan.items.front().requestId);
    if (lane.passes) {
      commandWall = lane.passes->wallMilliseconds += wallMilliseconds;
      commandHost = lane.passes->hostMilliseconds += hostMilliseconds;
      representative = lane.passes->representative = lane.passes->representative && representativePrefillTiming;
    }
    if (plan.pass + 1 < plan.passes)
      representative = false;
    else
      lane.passes.reset();
  }
  if (representative && plan.kind == WorkKind::Prefill) {
    uint32_t rows = 0;
    for (const BatchItem &item : plan.items)
      rows += item.tokenCount;
    const double waited = stream.value_or(0.0);
    observePrefill(rows, commandWall - waited, commandHost - waited);
  }
  if (stream)
    streaming->streamMilliseconds =
        streaming->streamMilliseconds
            ? (1.0 - kPrefillTimeWeight) * *streaming->streamMilliseconds +
                  kPrefillTimeWeight * *stream
            : *stream;
  // While requests of the same or a higher priority decode, lanes this
  // prefill finished included, it owes decode a share of its time; decode
  // commands work the debt off with their own. A lane waiting for its mask
  // is owed nothing: it could not have decoded meanwhile, and the slice
  // already bounds the prefill it waits behind. Nor is a decoder this
  // command was planned without: it could not run either.
  if (timed) {
    if (plan.kind == WorkKind::Decode) {
      decodeDebtMilliseconds_ =
          std::max(0.0, decodeDebtMilliseconds_ - wallMilliseconds);
    } else if (streaming || decodeShare_ > 0.0) {
      const RequestPriority priority =
          get(plan.items.front().requestId).spec.priority;
      const bool contended = std::any_of(
          requests_.begin(), requests_.end(), [&](const auto &entry) {
            const auto &[id, peer] = entry;
            return peer.phase == Phase::Decode && peer.spec.priority <= priority &&
                   !listed(excluded_, id);
          });
      counters_.contendedStreamingPrefillBatches += contended && streaming;
      if (contended && decodeShare_ > 0.0)
        decodeDebtMilliseconds_ += decodeShare_ * wallMilliseconds;
    }
  }
  dropStaleDecodeDebt();
  excluded_.clear();
  active_.reset();
}

void Scheduler::startDecodeGap(Request &request) noexcept {
  request.decodeGapStart = commandMilliseconds_;
  request.decodeGapStreams = streamingCommands_;
}

void Scheduler::dropStaleDecodeDebt() noexcept {
  const bool decoding = std::any_of(
      requests_.begin(), requests_.end(), [](const auto &entry) {
        return entry.second.phase == Phase::Decode ||
               entry.second.phase == Phase::WaitingMask;
      });
  if (!decoding)
    decodeDebtMilliseconds_ = 0.0;
}

void Scheduler::observePrefill(uint32_t rows, double wallMilliseconds,
                               double hostMilliseconds) {
  // Tiny tails are dominated by fixed command costs, not prefill throughput.
  if (rows < kMinimumPrefillRows || !std::isfinite(wallMilliseconds) ||
      wallMilliseconds <= 0.0)
    return;
  const double fixed = std::isfinite(hostMilliseconds)
                           ? std::clamp(hostMilliseconds, 0.0, wallMilliseconds)
                           : 0.0;
  const double perToken = (wallMilliseconds - fixed) / rows;
  const bool measured =
      prefillMillisecondsPerToken_ > 0.0 || prefillFixedMilliseconds_ > 0.0;
  const auto mix = [&](double estimate, double observed) {
    return measured ? (1.0 - kPrefillTimeWeight) * estimate +
                          kPrefillTimeWeight * observed
                    : observed;
  };
  prefillFixedMilliseconds_ = mix(prefillFixedMilliseconds_, fixed);
  prefillMillisecondsPerToken_ = mix(prefillMillisecondsPerToken_, perToken);
}

Phase Scheduler::phase(uint64_t id) const { return get(id).phase; }

DecodeStage Scheduler::decodeStage(uint64_t id) const { return get(id).decodeStage; }

bool Scheduler::suspended(uint64_t id) const { return get(id).suspended; }

uint64_t Scheduler::submissionOrder(uint64_t id) const { return get(id).order; }

uint32_t Scheduler::promptProcessed(uint64_t id) const {
  return get(id).promptProcessed;
}

SchedulerSnapshot Scheduler::snapshot() const noexcept {
  SchedulerSnapshot result = counters_;
  result.decodeGap = decodeGaps_.snapshot();
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

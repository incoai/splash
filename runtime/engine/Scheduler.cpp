#include "engine/Scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace splash::engine {
namespace {

constexpr uint32_t kChunk = model::ExecutionLimits::prefillTokenBudget;
// The fewest rows the pair bound halves a lane's command to.
constexpr uint32_t kMinimumPrefillRows = 64;
// The fewest rows a contended cut leaves a unit, or the model's least rows
// where more (the Neural Engine FFN split's least chunk): smaller cuts lose
// prefill throughput (17% fewer rows a second on an M6 below 512 rows).
constexpr uint32_t kFloorRows = 512;
// A request yields to later arrivals of its priority at most this many
// prefill commands in a row: the other lanes of a full batch, each finishing
// one short prompt.
constexpr uint32_t kMaximumOvertakes = model::ExecutionLimits::maximumBatchWidth - 1;

// A command excludes at most the lanes of a full batch: a linear search.
bool listed(std::span<const uint64_t> ids, uint64_t id) noexcept {
  return std::find(ids.begin(), ids.end(), id) != ids.end();
}

bool holds(std::span<const BatchItem> items, uint64_t id) noexcept {
  return std::ranges::any_of(items, [id](const BatchItem &item) { return item.requestId == id; });
}

// The rows of an uncontended command of a lane with `remaining` rows to
// prefill: up to a chunk, or for a lane whose commands are groups of up to
// `groupRows` rows, its remaining chunks split evenly over the fewest groups
// that hold them (a short last group would pay a whole fixed part for a few
// rows).
uint32_t uncontendedRows(uint32_t remaining, uint32_t groupRows) noexcept {
  if (remaining <= kChunk || groupRows <= kChunk)
    return std::min(remaining, kChunk);
  const uint32_t chunks = (remaining + kChunk - 1) / kChunk;
  const uint32_t groups = (chunks + groupRows / kChunk - 1) / (groupRows / kChunk);
  return std::min(remaining, (chunks + groups - 1) / groups * kChunk);
}

// The lanes whose rows run past a chunk as they fill the chunks in order
// (model::commandViolation's count).
uint32_t lanesPastChunk(std::span<const model::PrefillLane> lanes) noexcept {
  uint64_t rows = 0;
  uint32_t past = 0;
  for (const model::PrefillLane &lane : lanes) {
    past += lane.rows && rows / kChunk != (rows + lane.rows - 1) / kChunk;
    rows += lane.rows;
  }
  return past;
}

uint64_t rowsOf(std::span<const model::PrefillLane> lanes) noexcept {
  uint64_t rows = 0;
  for (const model::PrefillLane &lane : lanes)
    rows += lane.rows;
  return rows;
}

} // namespace

Scheduler::Scheduler(SchedulerConfig config, const model::PrefillWorkSource &work)
    : config_(config), work_(work), decodeShare_(config.decodeShare) {
  if (!std::isfinite(config_.decodeGapMilliseconds) || config_.decodeGapMilliseconds <= 0.0)
    throw std::invalid_argument("the decode gap target must be positive and finite");
  if (!config_.limits.commandRows || !config_.limits.maximumPasses)
    throw std::invalid_argument("a prefill command must hold a row in at least one pass");
  counters_.decodeGapTargetMilliseconds = config_.decodeGapMilliseconds;
}

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
  request.rowsCounted = processed;
  request.decodeStage = DecodeStage::Regular;
  request.phase =
      processed == request.spec.prefillTokens ? Phase::Decode : Phase::Prefill;
  if (request.phase == Phase::Decode)
    promptCompleted(request);
}

void Scheduler::suspendForResources(uint64_t id) {
  Request &request = get(id);
  if (request.phase != Phase::Prefill && request.phase != Phase::Decode) {
    throw std::logic_error("only runnable resident work can be suspended");
  }
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
  // The replay counts again the rows it runs that ran before it, but not
  // those it runs that were counted and never ran: an abandoned command's, or
  // the rest of a command between its passes, [promptProcessed, rowsCounted)
  // from where it starts.
  request.rowsCounted = std::max(request.rowsCounted, processed) -
                        (request.promptProcessed - std::min(request.promptProcessed, processed));
  request.promptProcessed = processed;
  request.spec.prefillTokens = replayTokens;
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
}

void Scheduler::cancel(uint64_t id) {
  Request &request = get(id);
  if (!terminal(request.phase))
    request.phase = Phase::Cancelled;
  dropStaleDecodeDebt();
}

void Scheduler::finish(uint64_t id) {
  Request &request = get(id);
  if (!terminal(request.phase))
    request.phase = Phase::Completed;
  dropStaleDecodeDebt();
}

void Scheduler::fail(uint64_t id) {
  Request &request = get(id);
  if (terminal(request.phase))
    return;
  request.phase = Phase::Failed;
  dropStaleDecodeDebt();
}

void Scheduler::remove(uint64_t id) {
  if (active_ && holds(active_->items, id))
    throw std::logic_error("cannot remove an active batch member");
  auto found = requests_.find(id);
  if (found == requests_.end() || !terminal(found->second.phase)) {
    throw std::logic_error("only terminal requests can be removed");
  }
  requests_.erase(found);
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

std::vector<uint64_t> Scheduler::prefillAdmissionOrder(std::span<const PrefillAdmission> candidates,
                                                       const PlanInputs &in) const {
  // A request admission has not started, from the tokens it starts at.
  const auto unstarted = [&](const PrefillAdmission &entry) {
    const Request &value = get(entry.requestId);
    if (value.suspended ||
        (value.phase != Phase::Queued && value.phase != Phase::WaitingResources &&
         value.phase != Phase::WaitingPrefix) ||
        entry.cachedTokens >= value.spec.prefillTokens)
      throw std::logic_error("invalid pending prefill admission");
    return lane(value, entry.cachedTokens, in);
  };
  std::vector<Lane> pending;
  pending.reserve(candidates.size());
  for (const auto &candidate : candidates)
    pending.push_back(unstarted(candidate));
  // The lanes of the candidates' first command: the resident prefill lanes,
  // those of the command between its passes past its rows, which commit at
  // its last pass before any other prefill runs (nextPrefill). One whose
  // prompt those rows end is a lane ahead of every candidate of its priority
  // or a lower one, its prompt ending there, which matters only for a Spill
  // lane (below): the most urgent such lane's priority.
  std::vector<Lane> lanes;
  lanes.reserve(requests_.size());
  std::vector<uint64_t> result;
  std::optional<RequestPriority> spillEnds;
  for (const auto &[id, request] : requests_) {
    if (request.phase != Phase::Prefill || shapeOf(id, in).ended)
      continue;
    uint32_t processed = request.promptProcessed;
    if (in.passes)
      if (const auto item = std::ranges::find(in.passes->plan.items, id, &BatchItem::requestId);
          item != in.passes->plan.items.end())
        processed += item->tokenCount;
    if (processed != request.spec.prefillTokens)
      lanes.push_back(lane(request, processed, in));
    else if (shapeOf(id, in).spill && (!spillEnds || request.spec.priority < *spillEnds))
      spillEnds = request.spec.priority;
  }
  // The requests restoring their prefix, which prefill once it lands, in no
  // command planned before then.
  std::vector<Lane> restoring;
  restoring.reserve(in.restoring.size());
  for (const auto &entry : in.restoring)
    restoring.push_back(unstarted(entry));
  // Admission starts only lanes their first command serves, overdue or left
  // waiting for no other prompt: one whose prompt the command ends, or one it
  // leads or that leads next, every lane ordered before it ending its prompt
  // there, and that no restoring start leads once it lands. A start holds
  // its memory from its first command to its last, so a prompt that would
  // join with the rows that spare it a command and then wait out another's
  // prompt stays queued until it leads. One that leads next starts beside
  // no Spill lane, as two Spill lanes prefill together only as the later one
  // leads or one is overdue: beside the last command of a Spill lane, one
  // between its passes too, it stays queued. A restoring start holds back a
  // candidate it leads once it lands: past the candidate's first command,
  // which this tick plans without it, or as the candidate stands while a
  // command is between its passes, as it may land before the first command
  // after them. Admission plans again without the candidates a plan does
  // not serve until it serves all it plans, as without them the plan can
  // take another leader.
  std::optional<Planned> planned;
  size_t count = 0;
  do {
    count = pending.size();
    std::vector<Lane> ready = lanes;
    ready.insert(ready.end(), pending.begin(), pending.end());
    planned = planPrefill(ready);
    if (!planned)
      return result;
    const std::vector<BatchItem> &items = planned->plan.items;
    const auto ends = [&](const Lane &view) {
      const auto item = std::ranges::find(items, view.request->spec.id, &BatchItem::requestId);
      return item != items.end() && item->tokenCount == view.remaining();
    };
    std::erase_if(pending, [&](const Lane &view) {
      const auto item = std::ranges::find(items, view.request->spec.id, &BatchItem::requestId);
      if (item == items.end())
        return true;
      if (view.overdue || item->tokenCount == view.remaining())
        return false;
      const Lane landed = in.passes ? view : lane(*view.request, view.processed + item->tokenCount, in);
      return (spillEnds && *spillEnds <= view.request->spec.priority) ||
             std::ranges::any_of(ready,
                                 [&](const Lane &other) {
                                   return other.request != view.request && leadsBefore(other, view) &&
                                          (!ends(other) || other.shape.spill);
                                 }) ||
             std::ranges::any_of(restoring, [&](const Lane &other) { return leadsBefore(other, landed); });
    });
  } while (pending.size() != count);
  // Admission ranks candidates against every decoder, left out of a plan or
  // not.
  const auto decode = nextDecode({});
  if (decode && get(decode->items.front().requestId).spec.priority <
                    get(planned->plan.items.front().requestId).spec.priority)
    return result;
  for (const auto &item : planned->plan.items)
    if (std::ranges::any_of(pending, [&](const Lane &view) { return view.request->spec.id == item.requestId; }))
      result.push_back(item.requestId);
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

bool Scheduler::yieldsTo(RequestPriority priority, const PlanInputs &in) const {
  return in.passes && priority < get(in.passes->plan.items.front().requestId).spec.priority;
}

bool Scheduler::abandonsPasses(const BatchPlan &plan, const PlanInputs &in) const {
  return in.passes && plan.kind == WorkKind::Prefill && !samePlan(plan, in.passes->plan);
}

std::optional<BatchPlan> Scheduler::next(const PlanInputs &in) const {
  if (active_)
    return std::nullopt;
  auto decode = nextDecode(in.excluded);
  auto prefill = nextPrefill(in);
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

std::optional<BatchPlan> Scheduler::nextPrefill(const PlanInputs &in) const {
  std::optional<Planned> planned = planPrefill(readyLanes(in));
  // Its commit counts what planning it counted.
  lastPlanned_ = planned;
  // A command between its passes goes on before any other prefill, whose
  // rows would take the model's prefill memory its rows' work is in, but one
  // of a higher priority. One of its priority waits for its end: abandoning
  // it would run its passes again for the rows it started.
  if (!in.passes || (planned && yieldsTo(get(planned->plan.items.front().requestId).spec.priority, in)))
    return planned ? std::optional(std::move(planned->plan)) : std::nullopt;
  for (const BatchItem &item : in.passes->plan.items)
    if (listed(in.excluded, item.requestId))
      return std::nullopt;
  return in.passes->plan;
}

LaneShape Scheduler::shapeOf(uint64_t id, const PlanInputs &in) const {
  const auto found =
      std::ranges::find_if(in.shapes, [id](const LaneShape &shape) { return shape.requestId == id; });
  return found == in.shapes.end() ? LaneShape{.requestId = id} : *found;
}

Scheduler::Lane Scheduler::lane(const Request &request, uint32_t processed, const PlanInputs &in) const {
  return {.request = &request,
          .processed = processed,
          .shape = shapeOf(request.spec.id, in),
          .overdue = request.overtaken >= kMaximumOvertakes};
}

std::vector<model::PrefillLane> Scheduler::prefillLanes(std::span<const BatchItem> items,
                                                        const PlanInputs &in) const {
  std::vector<model::PrefillLane> result;
  result.reserve(items.size());
  for (const BatchItem &item : items)
    result.push_back({item.promptOffset, item.tokenCount, shapeOf(item.requestId, in).indexPages});
  return result;
}

bool Scheduler::groups(const Lane &lane) const {
  return config_.limits.streamsWeights || lane.shape.indexPages || lane.shape.gainsIndexPages;
}

uint32_t Scheduler::groupRows(const Lane &lane) const {
  return groups(lane) ? config_.limits.commandRows : kChunk;
}

uint32_t Scheduler::ownRows(const Lane &lane) const {
  return std::min({uncontendedRows(lane.remaining(), groupRows(lane)), config_.limits.commandRows,
                   lane.shape.rowCap});
}

bool Scheduler::contended(RequestPriority priority) const {
  // Decoders this command is planned without count: they resume once their
  // memory lands.
  return std::ranges::any_of(requests_, [&](const auto &entry) {
    const Request &peer = entry.second;
    return (peer.phase == Phase::Decode || peer.phase == Phase::WaitingMask) && peer.spec.priority <= priority;
  });
}

uint64_t Scheduler::layerPairs(std::span<const model::PrefillLane> lanes) const {
  return work_.prefillWork(lanes, 0, 1).layerAttentionPairs;
}

uint32_t Scheduler::passesFor(std::span<const model::PrefillLane> lanes, bool group) const {
  if (!group)
    return 1;
  const uint32_t layers = config_.limits.maximumPasses;
  if (!measured_)
    return layers;
  const double scale = std::max({1.0, double(rowsOf(lanes)) / measured_->rows,
                                 measured_->pairs ? double(layerPairs(lanes)) / double(measured_->pairs) : 1.0});
  return passesOver(layers, measured_->layerMilliseconds * scale);
}

uint32_t Scheduler::passesOver(uint32_t layers, double layerMilliseconds) const {
  // Passes split the layers evenly (model::passLayers): the largest holds
  // ceil(layers / passes) of them.
  const double fits = std::floor(config_.decodeGapMilliseconds / layerMilliseconds);
  const auto fit = static_cast<uint32_t>(std::clamp(fits, 1.0, double(layers)));
  return (layers + fit - 1) / fit;
}

BatchPlan Scheduler::nextPass(const BatchPlan &plan) const {
  if (plan.kind != WorkKind::Prefill || plan.pass + 1 >= plan.passes)
    throw std::logic_error("no pass follows the last");
  const uint32_t layers = config_.limits.maximumPasses;
  BatchPlan next = followingPass(plan, layers);
  if (command_.timed && command_.attended)
    next.passes = next.pass + passesOver(layers - next.layer, command_.layerMilliseconds);
  return next;
}

bool Scheduler::withinPairs(std::span<const model::PrefillLane> lanes) const {
  return layerPairs(lanes) <= config_.limits.attentionPairs;
}

uint32_t Scheduler::cutRows(const Lane &leader) const {
  const uint32_t floor = std::max(kFloorRows, config_.limits.minimumUnitRows);
  uint32_t rows = floor;
  if (measured_) {
    // The rows the last command ran in G at its time a row, fewer where this
    // lane's layers reach more pairs than its heaviest did.
    const model::PrefillLane probe{leader.processed, measured_->rows, leader.shape.indexPages};
    const double pairs = double(layerPairs({&probe, 1}));
    const double deeper =
        measured_->pairs && pairs > double(measured_->pairs) ? double(measured_->pairs) / pairs : 1.0;
    rows = static_cast<uint32_t>(std::clamp(
        measured_->rows * config_.decodeGapMilliseconds / measured_->milliseconds * deeper, double(floor),
        double(std::max(floor, kChunk))));
  }
  // A rest below the floor runs with the unit, up to a chunk.
  const uint32_t remaining = leader.remaining();
  return remaining > rows && remaining - rows < floor ? std::max(rows, std::min(remaining, kChunk)) : rows;
}

std::vector<Scheduler::Lane> Scheduler::readyLanes(const PlanInputs &in) const {
  std::vector<Lane> ready;
  for (const auto &[id, request] : requests_)
    if (request.phase == Phase::Prefill && !listed(in.excluded, id) && !shapeOf(id, in).ended)
      ready.push_back(lane(request, request.promptProcessed, in));
  return ready;
}

std::optional<Scheduler::Planned> Scheduler::planPrefill(std::vector<Lane> ready) const {
  if (ready.empty())
    return std::nullopt;
  std::sort(ready.begin(), ready.end(), leadsBefore);
  const Lane &leader = ready.front();
  const RequestPriority tier = leader.request->spec.priority;
  // A group command runs in passes, contended or not; any other runs whole,
  // cut while a lane of its priority or a higher one decodes.
  const bool group = groups(leader);
  const bool cut = !group && contended(tier);
  Planned result;
  std::vector<model::PrefillLane> lanes{{leader.processed, ownRows(leader), leader.shape.indexPages}};
  // A layer's attention pairs bound the rows, which passes never lower.
  while (lanes.front().rows > kMinimumPrefillRows && !withinPairs(lanes))
    lanes.front().rows /= 2;
  // Joiners never take a unit past its rows: a cut's, a group's or a chunk.
  const uint32_t unitRows = cut ? cutRows(leader) : groupRows(leader);
  if (unitRows < lanes.front().rows) {
    lanes.front().rows = unitRows;
    ++result.boundCuts;
  }
  uint32_t passes = passesFor(lanes, group);
  const auto indexPageLanes = [&](const std::vector<model::PrefillLane> &with, std::span<const Lane> of,
                                  uint32_t passes) {
    uint32_t count = 0;
    for (size_t index = 0; index < with.size(); ++index)
      count += with[index].indexPages || (passes > 1 && of[index].shape.gainsIndexPages);
    return count;
  };
  std::vector<Lane> members{leader};
  for (auto view = ready.begin() + 1; view != ready.end(); ++view) {
    if (view->request->spec.priority != tier || lanes.size() == model::ExecutionLimits::maximumBatchWidth)
      break;
    // A lane that runs in passes joins only a command that does.
    if (groups(*view) && !group)
      continue;
    // The fewest rows that spare it one of its own commands, each of up to
    // groupRows rows or its row cap this tick: those past its whole commands,
    // not past ownRows' even split of them (8492 rows alone run as 6144 and
    // 2348; 300 of them spare one).
    const uint32_t rows = (view->remaining() - 1) % std::min(groupRows(*view), view->shape.rowCap) + 1;
    std::vector<model::PrefillLane> with = lanes;
    with.push_back({view->processed, rows, view->shape.indexPages});
    if (rowsOf(with) > unitRows || lanesPastChunk(with) > config_.limits.lanesPastChunk || !withinPairs(with))
      continue;
    const uint32_t joined = passesFor(with, group);
    members.push_back(*view);
    if (indexPageLanes(with, members, joined) > config_.limits.indexPageLanes) {
      members.pop_back();
      ++result.indexPageConflicts;
      continue;
    }
    lanes = std::move(with);
    passes = joined;
  }
  result.plan.kind = WorkKind::Prefill;
  for (size_t index = 0; index < lanes.size(); ++index)
    result.plan.items.push_back({members[index].request->spec.id, lanes[index].rows, lanes[index].position});
  result.plan.passes = passes;
  return result;
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

void Scheduler::commit(const BatchPlan &plan, const PlanInputs &in) {
  if (active_ || plan.empty() ||
      plan.width() > model::ExecutionLimits::maximumBatchWidth) {
    throw std::logic_error("invalid scheduler commit");
  }
  const bool prefill = plan.kind == WorkKind::Prefill;
  for (const BatchItem &item : plan.items) {
    const Request &request = get(item.requestId);
    const Phase expected = prefill ? Phase::Prefill : Phase::Decode;
    if (request.phase != expected ||
        (prefill && (!item.tokenCount || item.promptOffset != request.promptProcessed)) ||
        (!prefill && (item.tokenCount || item.promptOffset || request.decodeStage != plan.decodeStage))) {
      throw std::logic_error("batch no longer matches scheduler state");
    }
  }
  // A later pass is the command's next; no other prefill commits while a
  // command is between its passes.
  if (in.passes ? prefill && !samePlan(plan, in.passes->plan)
                : plan.pass || (plan.passes > 1 && !prefill))
    throw std::logic_error("batch no longer matches scheduler state");
  const RequestPriority priority = get(plan.items.front().requestId).spec.priority;
  if (prefill && !plan.pass) {
    counters_.passCommands += plan.passes > 1;
    ++counters_.prefillBatches;
    for (const BatchItem &item : plan.items) {
      Request &request = get(item.requestId);
      const uint32_t end = item.promptOffset + item.tokenCount;
      counters_.prefillRows += end - std::clamp(request.rowsCounted, item.promptOffset, end);
      request.rowsCounted = std::max(request.rowsCounted, end);
    }
    // What planning counted for this plan, as next() made it.
    if (lastPlanned_ && samePlan(lastPlanned_->plan, plan)) {
      counters_.boundCuts += lastPlanned_->boundCuts;
      counters_.indexPageConflicts += lastPlanned_->indexPageConflicts;
    }
    // A command, its passes one, serves its lanes, joiners too: each starts
    // its count again. It overtakes the other requests of its priority
    // submitted before a lane it serves, with a lane or waiting for one: one
    // overtaken kMaximumOvertakes times in a row is overdue until a command
    // serves it. Lanes the command is planned without are blocked, not
    // passed over.
    uint64_t youngestServed = 0;
    for (const BatchItem &item : plan.items)
      youngestServed = std::max(youngestServed, get(item.requestId).order);
    for (auto &[id, request] : requests_) {
      if (request.spec.priority != priority || request.suspended || listed(in.excluded, id) ||
          (request.phase != Phase::Queued && request.phase != Phase::WaitingResources &&
           request.phase != Phase::WaitingPrefix && request.phase != Phase::Prefill))
        continue;
      if (holds(plan.items, id))
        request.overtaken = 0;
      else if (request.order < youngestServed && request.overtaken < kMaximumOvertakes)
        ++request.overtaken;
    }
  } else if (!prefill) {
    const uint64_t dispatchOrder = ++decodeDispatchOrder_;
    for (const BatchItem &item : plan.items)
      get(item.requestId).lastDecodeDispatch = dispatchOrder;
    ++counters_.decodeBatches;
    ++counters_.decodeBatchesByWidth[plan.width() - 1];
  }
  lastPlanned_.reset();
  if (prefill && !plan.pass) {
    const std::vector<model::PrefillLane> lanes = prefillLanes(plan.items, in);
    command_ = {.rows = static_cast<uint32_t>(rowsOf(lanes)), .pairs = layerPairs(lanes)};
  }
  activeContended_ = prefill && contended(priority);
  excluded_.assign(in.excluded.begin(), in.excluded.end());
  active_ = plan;
  lastCommittedKind_ = plan.kind;
}

void Scheduler::complete(const BatchPlan &plan, std::span<const StepResult> results,
                         const model::CommandSample &sample) {
  if (!active_ || active_->kind != plan.kind ||
      active_->items.size() != plan.items.size() ||
      results.size() != plan.items.size()) {
    throw std::logic_error("completion does not match active batch");
  }
  const double wallMilliseconds = sample.wallMilliseconds;
  const bool timed = std::isfinite(wallMilliseconds) && wallMilliseconds > 0.0;
  const bool prefill = plan.kind == WorkKind::Prefill;
  for (size_t index = 0; index < results.size(); ++index) {
    const BatchItem &item = plan.items[index];
    const StepResult &result = results[index];
    if (result.requestId != item.requestId) {
      throw std::logic_error("completion request order changed");
    }
    Request &request = get(item.requestId);
    if (prefill && plan.pass + 1 < plan.passes) {
      // A pass before the last consumes nothing and ends no lane: one that
      // ends meanwhile stays in the command to its last pass.
      if (result.consumedPromptTokens || result.finished)
        throw std::logic_error("prefill completion row count changed");
    } else if (prefill) {
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
      }
      if (request.phase != Phase::Prefill)
        promptCompleted(request);
    } else {
      request.decodeStage = result.nextDecodeStage;
      request.phase = result.finished ? Phase::Completed : Phase::Decode;
    }
  }
  const RequestPriority priority = get(plan.items.front().requestId).spec.priority;
  // Whether lanes of its priority or a higher one decode beside it now, those
  // it finished included, not counting decoders it was planned without, which
  // could not run either.
  const bool owesDecode = prefill && std::ranges::any_of(requests_, [&](const auto &entry) {
                            const auto &[id, peer] = entry;
                            return peer.phase == Phase::Decode && peer.spec.priority <= priority &&
                                   !listed(excluded_, id);
                          });
  if (prefill) {
    ++counters_.prefillUnits;
    const double rowsMilliseconds =
        wallMilliseconds - std::clamp(sample.imageEncodeMilliseconds, 0.0, std::max(wallMilliseconds, 0.0));
    // A command's rows' time over its passes sizes its later passes and,
    // once its last pass ran, the next commands. An abandoned one, one with
    // a pass whose rows took no time, or one whose completion added work,
    // sizes nothing.
    const model::LayerRange ran =
        model::passLayers(plan.layer, plan.passes - plan.pass, config_.limits.maximumPasses);
    command_.milliseconds += rowsMilliseconds;
    if (rowsMilliseconds > command_.heaviestMilliseconds) {
      command_.heaviestMilliseconds = rowsMilliseconds;
      command_.layerMilliseconds = rowsMilliseconds / (ran.end - ran.first);
    }
    command_.timed = command_.timed && timed && rowsMilliseconds > 0.0 && sample.representative;
    command_.attended = command_.attended || sample.work.attentionPairs >= sample.work.layerAttentionPairs;
    if (plan.pass + 1 == plan.passes && command_.timed)
      measured_ = command_;
    if (timed && activeContended_)
      contendedPrefill_.add(static_cast<uint64_t>(wallMilliseconds * 1e6));
  }
  // While requests of the same or a higher priority decode, lanes this
  // prefill finished included, it owes decode a share of its time; decode
  // commands work the debt off with their own. A lane waiting for its mask
  // is owed nothing: it could not have decoded meanwhile, and the bound
  // already bounds the prefill it waits behind. Nor is a decoder this
  // command was planned without: it could not run either.
  if (timed) {
    if (!prefill)
      decodeDebtMilliseconds_ = std::max(0.0, decodeDebtMilliseconds_ - wallMilliseconds);
    else if (owesDecode && decodeShare_ > 0.0) {
      decodeDebtMilliseconds_ += decodeShare_ * wallMilliseconds;
    }
  }
  dropStaleDecodeDebt();
  excluded_.clear();
  active_.reset();
}

void Scheduler::abandoned() {
  // A command of other lanes may be in flight, never a pass of this one.
  if (active_ && active_->passes > 1)
    throw std::logic_error("a command in flight cannot be abandoned");
  ++counters_.abandonedPassCommands;
}

void Scheduler::promptCompleted(Request &request) noexcept {
  request.overtaken = 0;
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

Phase Scheduler::phase(uint64_t id) const { return get(id).phase; }

DecodeStage Scheduler::decodeStage(uint64_t id) const { return get(id).decodeStage; }

bool Scheduler::suspended(uint64_t id) const { return get(id).suspended; }

uint64_t Scheduler::submissionOrder(uint64_t id) const { return get(id).order; }

uint32_t Scheduler::promptProcessed(uint64_t id) const {
  return get(id).promptProcessed;
}

SchedulerSnapshot Scheduler::snapshot() const noexcept {
  SchedulerSnapshot result = counters_;
  result.contendedPrefill = contendedPrefill_.snapshot();
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

bool Scheduler::leadsBefore(const Lane &a, const Lane &b) noexcept {
  // Priority first; then overdue lanes, oldest first; then the fewest
  // remaining rows, then submission order.
  if (a.request->spec.priority != b.request->spec.priority)
    return a.request->spec.priority < b.request->spec.priority;
  if (a.overdue != b.overdue)
    return a.overdue;
  if (!a.overdue && a.remaining() != b.remaining())
    return a.remaining() < b.remaining();
  return a.request->order < b.request->order;
}

bool Scheduler::samePlan(const BatchPlan &a, const BatchPlan &b) noexcept {
  return a.kind == b.kind && a.pass == b.pass && a.passes == b.passes && a.layer == b.layer && a.items == b.items;
}

} // namespace splash::engine

#pragma once

#include "engine/Types.hpp"
#include "model/CommandCost.hpp"
#include "model/LatencyHistogram.hpp"
#include "model/Model.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace splash::engine {

enum class Phase : uint8_t {
  Queued,
  WaitingResources,
  WaitingPrefix,
  Prefill,
  Decode,
  WaitingMask,
  Completed,
  Cancelled,
  Failed,
};

// What the scheduler plans with: decode's share of contended prefill time,
// the time G a prefill unit may take while lanes of its priority or a higher
// one decode, and what one prefill command of the model may hold.
struct SchedulerConfig final {
  double decodeShare = 0.5;
  double decodeGapMilliseconds = 500.0;
  model::CommandLimits limits;
};

struct RequestSpec final {
  uint64_t id = 0;
  RequestPriority priority = RequestPriority::Normal;
  // Tokens to prefill: the prompt, and after resumeFromResources the
  // replayed history.
  uint32_t prefillTokens = 0;
};

struct PrefillAdmission final {
  uint64_t requestId = 0;
  uint32_t cachedTokens = 0;
};

// A prefill lane as the engine knows it when the scheduler plans: the pages
// of its history only the SSD cache holds now, whether it may gain such pages
// between a command's passes (a Spill lane past its dense history and
// window), the most rows a state capture prepare cut this tick leaves it,
// whether it is a Spill lane, and whether its request ended between its
// command's passes (a cancel, its deadline), so that it runs to the
// command's last pass for no one.
struct LaneShape final {
  uint64_t requestId = 0;
  uint32_t indexPages = 0;
  bool gainsIndexPages = false;
  uint32_t rowCap = std::numeric_limits<uint32_t>::max();
  bool spill = false;
  bool ended = false;
};

// The command between its passes as planning reads it: its plan at its next
// pass.
struct PassProgress final {
  BatchPlan plan;
};

// What a plan is made from, rebuilt before each: the lanes that cannot run
// in it (they wait for memory on its way back, or prepare refused or held
// them back this tick), the shapes of the resident prefill lanes and, for
// admission, of the candidates, the command between its passes, if any,
// and, for admission, the admitted requests that restore their prefix from
// the SSD cache before they run, each from the boundary it restores to.
struct PlanInputs final {
  std::span<const uint64_t> excluded;
  std::span<const LaneShape> shapes;
  const PassProgress *passes = nullptr;
  std::span<const PrefillAdmission> restoring;
};

struct SchedulerSnapshot final {
  uint32_t queued = 0;
  uint32_t waitingResources = 0;
  uint32_t waitingPrefix = 0;
  uint32_t prefilling = 0;
  uint32_t decoding = 0;
  uint32_t waitingMask = 0;
  uint32_t terminal = 0;
  uint64_t prefillBatches = 0;
  // The prefill commands' rows, a command's with its first pass (the
  // matrix's work); rows its lanes plan again after it was abandoned between
  // its passes count once.
  uint64_t prefillRows = 0;
  uint64_t decodeBatches = 0;
  std::array<uint64_t, model::ExecutionLimits::maximumBatchWidth>
      decodeBatchesByWidth{};
  // The decode gap target contended prefill work is sized by (G).
  double decodeGapTargetMilliseconds = 0.0;
  // The wall time of each prefill unit, a command or one pass of it, that ran
  // while a lane of its priority or a higher one decoded: what the target
  // bounds, its quantiles within a millisecond (model::FineLatencyHistogram).
  model::LatencyStats contendedPrefill;
  // Prefill units run; commands that ran in passes, and those a superseding
  // plan abandoned between passes; contended plans whose leader the cut gave
  // fewer rows than it would run alone; and plans that left a lane out
  // that could have joined them because a command holds at most
  // model::CommandLimits::indexPageLanes lanes whose history may hold Index
  // pages.
  uint64_t prefillUnits = 0;
  uint64_t passCommands = 0;
  uint64_t abandonedPassCommands = 0;
  uint64_t boundCuts = 0;
  uint64_t indexPageConflicts = 0;
};

// One single-owner scheduling policy.
//
// Prefill takes the ready lanes of the most urgent priority. Within it a
// request that kMaximumOvertakes prefill commands in a row served later
// arrivals of its priority and not it, with a lane or waiting for one, is
// overdue and leads, the oldest first, until a command serves it; otherwise
// the lane of the fewest remaining rows leads, ties in submission order. A
// group command, whose
// target streams experts or whose leader has or may gain Index pages, holds
// up to the model's command rows and runs in layer passes, contended or not:
// as many as the last command's measured time at these rows and attention
// pairs takes of G, every pass before any command was measured. Any other
// command runs whole: a chunk, or while a lane of its priority or a higher
// one decodes (contended), the rows the last command ran in G, never fewer
// than max(512, the model's least rows) but where a layer's attention pairs
// bound them (cutRows). A lane joins with the rows that spare it a command of
// its own where the command's rows, lanes past a chunk, Index-page lanes and
// attention pairs allow. The command between its passes runs its later
// passes before any other prefill but one of a higher priority. Decode takes
// the ready lanes of one priority and stage, constrained or not, without
// waiting for more, so it runs only the four real M8/M16/M24/M32 shapes,
// between a command's passes too.
class Scheduler final {
public:
  Scheduler(SchedulerConfig config, const model::PrefillWorkSource &work);

  void submit(RequestSpec request);
  void deferAdmission(uint64_t requestId);
  void waitForResources(uint64_t requestId);
  void waitForPrefix(uint64_t requestId);
  void resourcesReady(uint64_t requestId, uint32_t alreadyProcessed);
  void suspendForResources(uint64_t requestId);
  void resumeFromResources(uint64_t requestId, uint32_t alreadyProcessed,
                           uint32_t replayTokens);
  void maskReady(uint64_t requestId);
  void cancel(uint64_t requestId);
  // Ends a request outside a command's completion, as at its length.
  void finish(uint64_t requestId);
  void fail(uint64_t requestId);
  void remove(uint64_t requestId);

  [[nodiscard]] std::vector<uint64_t> admissionOrder() const;
  // The candidates their first command would serve, overdue or left waiting
  // for no other prompt: running them to their prompt's end, leading them now
  // or, beside no Spill lane, next, beside the resident prefill lanes (those
  // of the command between its passes past its rows), and led by none of the
  // requests restoring in `in` once they land.
  // Admission starts only those, planning again without the others until it
  // serves all it plans, with the candidates' shapes in `in`.
  [[nodiscard]] std::vector<uint64_t>
  prefillAdmissionOrder(std::span<const PrefillAdmission> candidates, const PlanInputs &in) const;
  // The highest priority among requests that prefill or decode. A waiting
  // request of a lower priority cannot be in prefillAdmissionOrder's result
  // while it lasts: the plan takes one tier, the highest among resident
  // prefill lanes and candidates, and is dropped when a higher tier decodes.
  [[nodiscard]] std::optional<RequestPriority> highestRunnablePriority() const noexcept;
  // The next command, planned without the excluded lanes. While a command is
  // between its passes (in.passes), that is its next pass or decode, as its
  // priority and the decode share order them, or a prefill plan of a higher
  // priority.
  [[nodiscard]] std::optional<BatchPlan> next(const PlanInputs &in) const;
  // Whether `plan` is a prefill plan other than the next pass of the command
  // between its passes, which is abandoned before it commits.
  [[nodiscard]] bool abandonsPasses(const BatchPlan &plan, const PlanInputs &in) const;
  // Whether work of `priority` abandons the command between its passes: a
  // higher priority than the command's. The engine asks it for a start
  // (Engine::admit) and for a plan beside a lane of the command that ended
  // (Engine::tick).
  [[nodiscard]] bool yieldsTo(RequestPriority priority, const PlanInputs &in) const;
  // The excluded lanes are those the plan was made without: blocked, not
  // passed over, so they lose nothing to it. A later pass must be the
  // command's next.
  void commit(const BatchPlan &plan, const PlanInputs &in);
  // A pass before the command's last consumes no rows and ends no lane. A
  // timed command's last pass leaves its rows' time over its passes, image
  // encodes left out, for the next commands to be sized from.
  void complete(const BatchPlan &plan, std::span<const StepResult> results,
                const model::CommandSample &sample);
  // The pass after `plan`, a completed pass before its command's last: it
  // runs from the layer `plan` ended at. Once a pass of the command ran its
  // heaviest attention layer, the layers left are split again into the
  // fewest passes that fit in G at the time a layer of its heaviest pass so
  // far took; until then, or where a pass was untimed, the planned passes
  // stay.
  [[nodiscard]] BatchPlan nextPass(const BatchPlan &plan) const;
  // The command between its passes was abandoned.
  void abandoned();

  [[nodiscard]] Phase phase(uint64_t requestId) const;
  // The stage of its decode commands, which a suspension keeps.
  [[nodiscard]] DecodeStage decodeStage(uint64_t requestId) const;
  // Suspended by suspendForResources and not resumed since; a terminal
  // phase keeps it until remove(), so the engine can end the request's
  // continuation.
  [[nodiscard]] bool suspended(uint64_t requestId) const;
  // The request's place in submission order.
  [[nodiscard]] uint64_t submissionOrder(uint64_t requestId) const;
  [[nodiscard]] uint32_t promptProcessed(uint64_t requestId) const;
  [[nodiscard]] SchedulerSnapshot snapshot() const noexcept;

private:
  struct Request final {
    RequestSpec spec;
    Phase phase = Phase::Queued;
    uint32_t promptProcessed = 0;
    // Set by suspendForResources: the request comes back through
    // resumeFromResources, never through resourcesReady.
    bool suspended = false;
    DecodeStage decodeStage = DecodeStage::Regular;
    uint64_t order = 0;
    uint64_t lastDecodeDispatch = 0;
    // The prefill commands of its priority in a row that served later
    // arrivals and not it, up to kMaximumOvertakes, at which it is overdue;
    // none once a command serves it or its prompt completes.
    uint32_t overtaken = 0;
    // The prompt position its committed rows reach in prefillRows, never
    // short of promptProcessed, which rows planned again after their command
    // was abandoned between its passes do not pass; a resume's replay counts
    // again the rows it runs that ran before it, and none it runs that were
    // counted and never ran.
    uint32_t rowsCounted = 0;
  };

  // A prefill lane as planning weighs it.
  struct Lane final {
    const Request *request = nullptr;
    uint32_t processed = 0;
    LaneShape shape;
    bool overdue = false;
    [[nodiscard]] uint32_t remaining() const noexcept { return request->spec.prefillTokens - processed; }
  };
  // A plan and what planning it counted: a leader the cut gave fewer rows,
  // and joinable lanes left out for the Index-page lanes.
  struct Planned final {
    BatchPlan plan;
    uint32_t boundCuts = 0;
    uint32_t indexPageConflicts = 0;
  };

  [[nodiscard]] Request &get(uint64_t requestId);
  [[nodiscard]] const Request &get(uint64_t requestId) const;
  [[nodiscard]] static bool terminal(Phase phase) noexcept;
  [[nodiscard]] static bool byPriorityThenOrder(const Request *a,
                                                const Request *b) noexcept;
  [[nodiscard]] static bool samePlan(const BatchPlan &a, const BatchPlan &b) noexcept;
  // Whether prefill takes lane `a` before lane `b`: the order planning leads
  // by and joins in.
  [[nodiscard]] static bool leadsBefore(const Lane &a, const Lane &b) noexcept;

  [[nodiscard]] LaneShape shapeOf(uint64_t requestId, const PlanInputs &in) const;
  [[nodiscard]] Lane lane(const Request &request, uint32_t processed, const PlanInputs &in) const;
  [[nodiscard]] std::vector<model::PrefillLane> prefillLanes(std::span<const BatchItem> items,
                                                             const PlanInputs &in) const;
  // Whether the lane's commands are groups, of up to the model's command
  // rows in layer passes: the target streams experts, or the lane has or may
  // gain Index pages, whose reads a pass bounds as rows cannot.
  [[nodiscard]] bool groups(const Lane &lane) const;
  // The most rows a command of the lane alone holds: the model's command
  // rows where it groups, else a chunk.
  [[nodiscard]] uint32_t groupRows(const Lane &lane) const;
  // The rows of an uncontended command of the lane: a chunk, or a group's.
  [[nodiscard]] uint32_t ownRows(const Lane &lane) const;
  [[nodiscard]] bool contended(RequestPriority priority) const;
  // The heaviest attention layer's pairs of `lanes` as one command.
  [[nodiscard]] uint64_t layerPairs(std::span<const model::PrefillLane> lanes) const;
  [[nodiscard]] bool withinPairs(std::span<const model::PrefillLane> lanes) const;
  // The passes of a command of `lanes` as its first pass runs: one but for a
  // group; for a group the fewest whose every pass holds no more layers than
  // fit in G at the time a layer of the last command's heaviest pass took,
  // scaled up by these rows and pairs where more, never down (a command's
  // fixed part does not shrink with its rows); every layer its own pass
  // before any command was measured. Its later passes may be sized from its
  // own passes (nextPass).
  [[nodiscard]] uint32_t passesFor(std::span<const model::PrefillLane> lanes, bool group) const;
  // The fewest passes of `layers` layers whose every pass holds no more
  // layers than fit in G at `layerMilliseconds` a layer, at least one.
  [[nodiscard]] uint32_t passesOver(uint32_t layers, double layerMilliseconds) const;
  // The rows of a contended unit the leader leads, which no joiner takes it
  // past: those the last command ran in G (measured_), fewer where the
  // leader's layers reach more attention pairs, from the floor,
  // max(kFloorRows, CommandLimits::minimumUnitRows), to a chunk; the floor
  // before any command ran; to the leader's prompt's end where the rest
  // would fall below the floor. The pair bound may leave the leader fewer.
  [[nodiscard]] uint32_t cutRows(const Lane &leader) const;
  // The resident prefill lanes a plan may serve: those it is not made
  // without and that did not end.
  [[nodiscard]] std::vector<Lane> readyLanes(const PlanInputs &in) const;
  [[nodiscard]] std::optional<Planned> planPrefill(std::vector<Lane> ready) const;
  [[nodiscard]] std::optional<BatchPlan> nextPrefill(const PlanInputs &in) const;
  [[nodiscard]] std::optional<BatchPlan> nextDecode(std::span<const uint64_t> excluded) const;
  // Debt is owed only to requests that decode or wait for a mask: once the
  // last one leaves, a later decoder starts without it.
  void dropStaleDecodeDebt() noexcept;
  // A prompt that completed leaves its aging behind.
  static void promptCompleted(Request &request) noexcept;

  SchedulerConfig config_;
  const model::PrefillWorkSource &work_;
  std::unordered_map<uint64_t, Request> requests_;
  std::optional<BatchPlan> active_;
  // The prefill plan next() made last and what planning it counted, which
  // its commit adds to the counters; the commit of any plan clears it.
  mutable std::optional<Planned> lastPlanned_;
  // Whether the prefill unit in flight was contended when it committed, as
  // planning bounds units: what contendedPrefill_ counts.
  bool activeContended_ = false;
  // The lanes the active command was planned without.
  std::vector<uint64_t> excluded_;
  uint64_t order_ = 0;
  uint64_t decodeDispatchOrder_ = 0;
  double decodeShare_;
  // Decode time that prefill still owes the lanes that decoded beside it
  // (not those waiting for a mask): equal-priority decode runs until its
  // commands' wall time has worked it off.
  double decodeDebtMilliseconds_ = 0.0;
  std::optional<WorkKind> lastCommittedKind_;
  // A prefill command as it ran: its rows, the attention pairs of its
  // heaviest layer, its rows' time over its passes and in its heaviest, and
  // the time a layer of that pass took (each pass's own cost included),
  // image encodes left out; whether every pass of it was timed and
  // representative, and whether a pass of it computed its heaviest attention
  // layer's pairs (before one did, its time a layer holds none of its
  // attention).
  struct Measured final {
    uint32_t rows = 0;
    uint64_t pairs = 0;
    double milliseconds = 0.0;
    double heaviestMilliseconds = 0.0;
    double layerMilliseconds = 0.0;
    bool timed = true;
    bool attended = false;
  };
  // The command in flight, and the last one that completed timed: what a
  // contended cut and a group's passes are sized from.
  Measured command_;
  std::optional<Measured> measured_;
  SchedulerSnapshot counters_;
  model::FineLatencyHistogram contendedPrefill_;
};

} // namespace splash::engine

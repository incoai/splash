#pragma once

#include "engine/Types.hpp"
#include "model/Model.hpp"

#include <array>
#include <cstdint>
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

struct RequestSpec final {
  uint64_t id = 0;
  RequestPriority priority = RequestPriority::Normal;
  bool constrained = false;
  // Tokens to prefill: the prompt, and after resumeFromResources the
  // replayed history.
  uint32_t prefillTokens = 0;
  double deadlineMilliseconds = 0.0;
};

struct PrefillAdmission final {
  uint64_t requestId = 0;
  uint32_t cachedTokens = 0;
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
  uint64_t prefillRows = 0;
  uint64_t decodeBatches = 0;
  std::array<uint64_t, model::ExecutionLimits::maximumBatchWidth>
      decodeBatchesByWidth{};
};

// One single-owner scheduling policy. Prefill batches the shortest remaining
// sequences first into an actual-row budget, with a bound on how often later
// arrivals may overtake a lane; decode takes the ready lanes of one priority,
// constraint mode and stage without waiting for more, so it runs only the
// four real M8/M16/M24/M32 shapes.
class Scheduler final {
public:
  // Decode time owed for each unit of time a prefill runs while requests of
  // equal or higher priority decode; zero alternates one command of each kind.
  explicit Scheduler(double decodeShare) noexcept : decodeShare_(decodeShare) {}

  void submit(RequestSpec request);
  void observePrefill(uint32_t rows, double wallMilliseconds);
  void deferAdmission(uint64_t requestId);
  void waitForResources(uint64_t requestId);
  void waitForPrefix(uint64_t requestId);
  void resourcesReady(uint64_t requestId, uint32_t alreadyProcessed);
  void suspendForResources(uint64_t requestId);
  void resumeFromResources(uint64_t requestId, uint32_t alreadyProcessed,
                           uint32_t replayTokens);
  void maskReady(uint64_t requestId);
  void cancel(uint64_t requestId);
  void fail(uint64_t requestId);
  void remove(uint64_t requestId);

  // A sparse-state materialization point can stop one sequence without
  // padding or shortening any peer in the same ragged command. The boundary
  // must lie past the request's progress; complete() consumes a boundary its
  // command reached.
  void setPrefillBoundary(uint64_t requestId,
                          std::optional<uint32_t> absoluteTokens);

  [[nodiscard]] bool expireDeadlines(double nowMilliseconds);
  [[nodiscard]] std::vector<uint64_t> admissionOrder() const;
  // Preview the dispatch row budget before allocating new lanes.
  [[nodiscard]] std::vector<uint64_t>
  prefillAdmissionOrder(std::span<const PrefillAdmission> candidates) const;
  // The highest priority among requests that prefill or decode. A waiting
  // request of a lower priority cannot be in prefillAdmissionOrder's result
  // while it lasts: the plan takes one tier, the highest among resident
  // prefill lanes and candidates, and is dropped when a higher tier decodes.
  [[nodiscard]] std::optional<RequestPriority> highestRunnablePriority() const noexcept;
  // The next command, planned without the excluded lanes: those that wait
  // for memory on its way back and cannot run before it lands.
  [[nodiscard]] std::optional<BatchPlan> next(std::span<const uint64_t> excluded) const;
  // The excluded lanes are those the plan was made without: blocked, not
  // passed over, so they lose nothing to it.
  void commit(const BatchPlan &plan, std::span<const uint64_t> excluded);
  void complete(const BatchPlan &plan, std::span<const StepResult> results,
                double wallMilliseconds, bool representativePrefillTiming);

  [[nodiscard]] Phase phase(uint64_t requestId) const;
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
    std::optional<uint32_t> prefillBoundary;
    // Set by suspendForResources: the request comes back through
    // resumeFromResources, never through resourcesReady.
    bool suspended = false;
    DecodeStage decodeStage = DecodeStage::Regular;
    uint64_t order = 0;
    uint64_t lastDecodeDispatch = 0;
    // Consecutive prefill commands that served a later arrival while this
    // lane received no rows. Reset whenever the lane is served.
    uint32_t overtaken = 0;
  };

  struct PrefillRequestView final {
    const Request *request = nullptr;
    uint32_t promptProcessed = 0;
  };

  [[nodiscard]] Request &get(uint64_t requestId);
  [[nodiscard]] const Request &get(uint64_t requestId) const;
  [[nodiscard]] static bool terminal(Phase phase) noexcept;
  [[nodiscard]] static bool byPriorityThenOrder(const Request *a,
                                                const Request *b) noexcept;
  [[nodiscard]] std::optional<BatchPlan>
  nextPrefill(std::span<const uint64_t> excluded) const;
  [[nodiscard]] std::optional<BatchPlan>
  nextDecode(std::span<const uint64_t> excluded) const;
  [[nodiscard]] std::optional<BatchPlan>
  planPrefill(std::vector<PrefillRequestView> ready) const;
  // The rows the lane can take in one command: up to its prompt's end or its
  // next state boundary.
  [[nodiscard]] static uint32_t dispatchRemaining(const PrefillRequestView &view) noexcept;
  [[nodiscard]] uint32_t
  prefillBudget(const PrefillRequestView &leader,
                std::span<const PrefillRequestView> ready) const;
  // Debt is owed only to requests that decode or wait for a mask: once the
  // last one leaves, a later decoder starts without it.
  void dropStaleDecodeDebt() noexcept;

  std::unordered_map<uint64_t, Request> requests_;
  std::optional<BatchPlan> active_;
  // The lanes the active command was planned without.
  std::vector<uint64_t> excluded_;
  uint64_t order_ = 0;
  uint64_t decodeDispatchOrder_ = 0;
  double prefillMillisecondsPerToken_ = 0.0;
  double decodeShare_;
  // Decode time that prefill still owes the lanes that decoded beside it
  // (not those waiting for a mask): equal-priority decode runs until its
  // commands' wall time has worked it off.
  double decodeDebtMilliseconds_ = 0.0;
  std::optional<WorkKind> lastCommittedKind_;
  SchedulerSnapshot counters_;
};

} // namespace splash::engine

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
  BatchCohort cohort = BatchCohort::Greedy;
  uint32_t promptTokens = 0;
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
  // Committed decode batches containing both Greedy and Sampling requests.
  // Counts scheduler dispatches, not completed GPU commands or sampled tokens.
  uint64_t decodeMixedGreedySamplingBatches = 0;
};

// One single-owner policy for the specialized backend. Prefill packs the
// shortest remaining sequences first into an actual-row budget, with a bound
// on how often later arrivals may overtake a lane; decode dispatches every
// ready lane immediately and therefore has only the four real
// M8/M16/M24/M32 shapes.
class Scheduler final {
public:
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
  // padding or shortening any peer in the same packed command.
  void setPrefillBoundary(uint64_t requestId,
                          std::optional<uint32_t> absoluteTokens);

  [[nodiscard]] bool expireDeadlines(double nowMilliseconds);
  [[nodiscard]] std::vector<uint64_t> admissionOrder() const;
  // Preview the dispatch row budget before allocating new resident cells.
  [[nodiscard]] std::vector<uint64_t>
  prefillAdmissionOrder(std::span<const PrefillAdmission> candidates) const;
  [[nodiscard]] std::optional<BatchPlan> next() const;
  void commit(const BatchPlan &plan);
  void complete(const BatchPlan &plan, std::span<const StepResult> results,
                double wallMilliseconds = 0.0,
                bool representativePrefillTiming = true);

  [[nodiscard]] Phase phase(uint64_t requestId) const;
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
    bool suspendedForResources = false;
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
  [[nodiscard]] std::optional<BatchPlan> nextPrefill() const;
  [[nodiscard]] std::optional<BatchPlan> nextDecode() const;
  [[nodiscard]] std::optional<BatchPlan>
  planPrefill(std::vector<PrefillRequestView> ready) const;
  [[nodiscard]] uint32_t
  prefillBudget(const PrefillRequestView &leader,
                std::span<const PrefillRequestView> ready) const;

  std::unordered_map<uint64_t, Request> requests_;
  std::optional<BatchPlan> active_;
  uint64_t order_ = 0;
  uint64_t decodeDispatchOrder_ = 0;
  double prefillMillisecondsPerToken_ = 0.0;
  std::optional<WorkKind> lastCommittedKind_;
  SchedulerSnapshot counters_;
};

} // namespace splash::engine

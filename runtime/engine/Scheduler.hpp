#pragma once

#include "engine/Types.hpp"
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

struct RequestSpec final {
  uint64_t id = 0;
  RequestPriority priority = RequestPriority::Normal;
  // Tokens to prefill: the prompt, and after resumeFromResources the
  // replayed history.
  uint32_t prefillTokens = 0;
  double deadlineMilliseconds = 0.0;
  // The most rows of one of its prefill commands (Model::prefillGroupRows): a
  // prefill group's, or a chunk's; the engine may set it again once it knows
  // where the request's KV lives (setPrefillGroupRows).
  uint32_t prefillGroupRows = model::ExecutionLimits::prefillTokenBudget;
  // The most passes over the model's layers one of its prefill commands may
  // run in (Model::prefillPasses, BatchPlan::passes).
  uint32_t prefillPasses = 1;
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
  // Prefill commands of a lane that streams its history from the SSD cache
  // that ran while lanes of its priority or a higher one decoded, and the
  // gaps decoding lanes saw between their commands across such a command:
  // the command time from one of a lane's decode commands to its next.
  uint64_t contendedStreamingPrefillBatches = 0;
  model::LatencyStats decodeGap;
};

// One single-owner scheduling policy. Prefill batches the shortest remaining
// sequences first into an actual-row budget, with a bound on how often later
// arrivals may overtake a lane; a lane that streams its history from the SSD
// cache runs alone, and so does one whose command runs in passes over the
// model's layers, which no other prefill interrupts. Decode takes the ready
// lanes of one priority and stage, constrained or not, without waiting for
// more, so it runs only the four real M8/M16/M24/M32 shapes, between a
// command's passes too.
class Scheduler final {
public:
  // Decode time owed for each unit of time a prefill runs while requests of
  // equal or higher priority decode; zero alternates one command of each kind.
  // A prefill command's attention computes at most maximumAttentionPairs
  // query-key pairs (attentionPairs): its lanes' rows halve until they fit.
  explicit Scheduler(double decodeShare,
                     uint64_t maximumAttentionPairs = std::numeric_limits<uint64_t>::max()) noexcept
      : decodeShare_(decodeShare), maximumAttentionPairs_(maximumAttentionPairs) {}

  // The query-key pairs the attention of `rows` prompt rows from `position`
  // computes: each row attends to its history and itself, so the pairs, and
  // a command's time, grow with the depth of its rows.
  [[nodiscard]] static constexpr uint64_t attentionPairs(uint64_t position, uint32_t rows) noexcept {
    return rows * position + uint64_t{rows} * (rows + 1) / 2;
  }

  void submit(RequestSpec request);
  // A prefill command of `rows` rows took wallMilliseconds, of which its GPU
  // work waited hostMilliseconds on host steps: a fixed part (a streamed
  // target's expert reads), the rest per row.
  void observePrefill(uint32_t rows, double wallMilliseconds, double hostMilliseconds = 0.0);
  void deferAdmission(uint64_t requestId);
  void waitForResources(uint64_t requestId);
  void waitForPrefix(uint64_t requestId);
  void resourcesReady(uint64_t requestId, uint32_t alreadyProcessed);
  void suspendForResources(uint64_t requestId);
  void resumeFromResources(uint64_t requestId, uint32_t alreadyProcessed,
                           uint32_t replayTokens);
  void maskReady(uint64_t requestId);
  // Whether the resident lane's prefill commands read its history from the
  // SSD cache (KvHistoryStream), which a command does for one lane alone and
  // whatever its rows; the engine says so before it plans.
  void setStreamsHistory(uint64_t requestId, bool streams);
  void setPrefillGroupRows(uint64_t requestId, uint32_t rows);
  void cancel(uint64_t requestId);
  // Ends a request outside a command's completion, as at its length.
  void finish(uint64_t requestId);
  void fail(uint64_t requestId);
  void remove(uint64_t requestId);

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
  // A prefill command of one lane that streamed its history from the SSD
  // cache waited streamMilliseconds of its hostMilliseconds on the reads
  // (ModelBatchTicket); none for a command that streamed nothing. A pass
  // before a command's last consumes no rows (StepResult), and the
  // command's estimate learns from its passes' time together.
  void complete(const BatchPlan &plan, std::span<const StepResult> results,
                double wallMilliseconds, bool representativePrefillTiming,
                double hostMilliseconds = 0.0,
                std::optional<double> streamMilliseconds = std::nullopt);

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
    // Consecutive prefill commands that served a later arrival while this
    // lane received no rows. Reset whenever the lane is served.
    uint32_t overtaken = 0;
    // Its prefill commands stream its history (setStreamsHistory), and the
    // running estimate of how long one waits on the reads; none before the
    // first that read any. The command estimate leaves that wait out: it is
    // this lane's own cost, which grows with its history.
    bool streamsHistory = false;
    std::optional<double> streamMilliseconds;
    // While it decodes, where its gap since its last decode command began
    // (startDecodeGap): the command time and streaming prefill commands
    // completed then.
    double decodeGapStart = 0.0;
    uint64_t decodeGapStreams = 0;
    // Between the passes of its prefill command: the command's passes, its
    // next pass and its rows, and the time its passes took so far, a sample
    // of its rows unless one was not.
    struct Passes final {
      uint32_t passes = 0;
      uint32_t next = 0;
      uint32_t rows = 0;
      double wallMilliseconds = 0.0;
      double hostMilliseconds = 0.0;
      bool representative = true;
    };
    std::optional<Passes> passes;
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
  [[nodiscard]] uint32_t
  prefillBudget(const PrefillRequestView &leader,
                std::span<const PrefillRequestView> ready) const;
  // The rows of a contended command of a lane that streams its history,
  // which reads all of it whatever its rows: the fewest, halving from its
  // prefill group's, whose time covers its estimated wait on the reads, so
  // that fewer would not end the command sooner, and no fewer than
  // `contendedRows`, those of another contended command, which it takes
  // before its first streamed command.
  [[nodiscard]] uint32_t streamingRows(const Request &leader, uint32_t contendedRows) const;
  // The passes a command of `rows` rows of the lane runs in: as many as keep
  // each within the contended slice by the command estimate, up to its
  // model's; one for a lane that streams its history, or before the first
  // estimate.
  [[nodiscard]] uint32_t passesFor(const Request &leader, uint32_t rows) const;
  // A lane's gap between decode commands starts when it begins to decode
  // and after each of its decode commands.
  void startDecodeGap(Request &request) noexcept;
  // Debt is owed only to requests that decode or wait for a mask: once the
  // last one leaves, a later decoder starts without it.
  void dropStaleDecodeDebt() noexcept;

  std::unordered_map<uint64_t, Request> requests_;
  std::optional<BatchPlan> active_;
  // The lanes the active command was planned without.
  std::vector<uint64_t> excluded_;
  uint64_t order_ = 0;
  uint64_t decodeDispatchOrder_ = 0;
  // A prefill command's estimated cost: fixed, plus per row.
  double prefillFixedMilliseconds_ = 0.0;
  double prefillMillisecondsPerToken_ = 0.0;
  double decodeShare_;
  uint64_t maximumAttentionPairs_;
  // Decode time that prefill still owes the lanes that decoded beside it
  // (not those waiting for a mask): equal-priority decode runs until its
  // commands' wall time has worked it off.
  double decodeDebtMilliseconds_ = 0.0;
  std::optional<WorkKind> lastCommittedKind_;
  SchedulerSnapshot counters_;
  // The time of the commands completed so far, and the streaming prefill
  // commands among them, which lanes' decode gaps are measured on; the gaps
  // across a streaming command.
  double commandMilliseconds_ = 0.0;
  uint64_t streamingCommands_ = 0;
  model::LatencyHistogram decodeGaps_;
};

} // namespace splash::engine

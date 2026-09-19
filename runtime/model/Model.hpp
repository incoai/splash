#pragma once

#include "ops/Vision.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace splash {

enum class BatchCohort : uint8_t { Greedy, Sampling, Constrained };
enum class WorkKind : uint8_t { Prefill, Decode };

enum class DecodeStage : uint8_t {
  Regular,
  RequestInitialMask,
  ApplyInitialMask,
};

[[nodiscard]] constexpr bool waitsForMask(DecodeStage stage) noexcept {
  return stage == DecodeStage::ApplyInitialMask;
}

enum class ConstraintMode : uint8_t { None, TokenMask };

struct SamplingParameters final {
  float temperature = 0.0F;
  float topP = 1.0F;
  uint32_t topK = 0;
  uint64_t seed = 0;
};

// Immutable view of the fields a model needs to activate a sequence.  Engine
// priority and deadline policy deliberately do not cross this boundary.
struct ModelRequest final {
  uint64_t id = 0;
  BatchCohort cohort = BatchCohort::Greedy;
  std::span<const uint32_t> prompt;
  std::span<const struct ImageSpan> images;
  std::span<const uint8_t> imagePixels;
  uint32_t maxNewTokens = 0;
  SamplingParameters sampling;
  ConstraintMode constraint = ConstraintMode::None;
  // Nonempty selects score-only mode: prefill runs to completion, no token is
  // generated, and the raw final-position logits at these ids are returned in
  // ModelStepResult::scoreLogits. maxNewTokens must be zero.
  std::span<const uint32_t> scoreTokens{};
};

struct ImageSpan final {
  uint32_t offset = 0;
  uint32_t tokens = 0;
  uint32_t gridHeight = 0;
  uint32_t gridWidth = 0;
  uint64_t digestLo = 0;
  uint64_t digestHi = 0;

  [[nodiscard]] uint32_t end() const noexcept { return offset + tokens; }
  [[nodiscard]] uint64_t pixelBytes() const noexcept {
    return ops::imagePixelBytes(gridHeight, gridWidth);
  }
  bool operator==(const ImageSpan &) const = default;
};

// Immutable target-recurrent plus draft-context state.  Concrete model
// implementations own its buffers; the engine only pins and accounts it.
class CompositeState {
public:
  virtual ~CompositeState() = default;
  // Footprint retained by the cache. A cached state owns a private copy of
  // the lane's state; dropping the reference returns that slot to the model's
  // pool, and idle-state reclaim frees it.
  [[nodiscard]] virtual uint64_t bytes() const noexcept = 0;
};

enum class DraftBoundaryPurpose : uint8_t { Active, Materialization };

struct DraftCaptureSpan final {
  uint32_t begin = 0;
  uint32_t end = 0;
  bool resetDraftState = false;
};

struct DraftBoundaryPlan final {
  uint32_t boundary = 0;
  DraftBoundaryPurpose purpose = DraftBoundaryPurpose::Active;
  uint32_t captureBegin = 0;
};

struct DraftContextPlan final {
  uint32_t replayBegin = 0;
  uint32_t replayEnd = 0;
  std::optional<uint32_t> restoredDraftBoundary;
  std::vector<DraftCaptureSpan> captureSpans;
  std::vector<DraftBoundaryPlan> boundaries;

  uint64_t targetPrefillRows = 0;
  uint64_t draftContextRowsActive = 0;
  uint64_t draftContextRowsMaterialization = 0;
  uint64_t draftContextRowsAvoided = 0;
  uint64_t draftStateRestoreSkipped = 0;
  uint64_t draftStateResets = 0;

  [[nodiscard]] uint64_t draftContextRows() const noexcept {
    return draftContextRowsActive + draftContextRowsMaterialization;
  }
  [[nodiscard]] std::span<const DraftCaptureSpan> captures() const noexcept {
    return captureSpans;
  }
  [[nodiscard]] std::span<const DraftBoundaryPlan>
  plannedBoundaries() const noexcept {
    return boundaries;
  }
};

struct DispatchDraftCaptureSpan final {
  uint32_t absoluteBegin = 0;
  uint32_t absoluteEnd = 0;
  uint32_t compactDestinationRow = 0;
  bool resetDraftState = false;
  uint32_t activeRows = 0;
  uint32_t materializationRows = 0;
};

struct DispatchDraftCapturePlan final {
  std::array<DispatchDraftCaptureSpan, 2> values{};
  uint32_t count = 0;

  [[nodiscard]] uint32_t size() const noexcept { return count; }
  [[nodiscard]] const DispatchDraftCaptureSpan &
  operator[](uint32_t index) const {
    return values.at(index);
  }
  [[nodiscard]] auto begin() const noexcept { return values.begin(); }
  [[nodiscard]] auto end() const noexcept { return values.begin() + count; }
};

[[nodiscard]] DraftContextPlan
planDraftContext(uint32_t replayBegin, uint32_t replayEnd,
                 std::optional<uint32_t> restoredDraftBoundary,
                 std::span<const uint32_t> materializationBoundaries);

[[nodiscard]] DispatchDraftCapturePlan
draftCaptureSpansForDispatch(const DraftContextPlan &plan,
                             uint32_t dispatchBegin, uint32_t dispatchEnd);

struct BatchItem final {
  uint64_t requestId = 0;
  uint32_t tokenCount = 0;
  uint32_t promptOffset = 0;
};

struct BatchPlan final {
  WorkKind kind = WorkKind::Decode;
  BatchCohort cohort = BatchCohort::Greedy;
  std::vector<BatchItem> items;
  DecodeStage decodeStage = DecodeStage::Regular;

  [[nodiscard]] bool empty() const noexcept { return items.empty(); }
  [[nodiscard]] uint32_t width() const noexcept {
    return static_cast<uint32_t>(items.size());
  }
};

enum class StateFailure : uint8_t { None, ConcurrencyLimit, MemoryPressure };

struct StateAdmission final {
  std::optional<uint32_t> cell;
  StateFailure failure = StateFailure::None;
  metal::AllocationFailure allocationFailure = metal::AllocationFailure::None;

  [[nodiscard]] bool granted() const noexcept { return cell.has_value(); }
};

struct ModelBatchItem final {
  uint64_t requestId = 0;
  uint32_t stateSlot = 0;
  uint64_t logicalPosition = 0;
  uint32_t promptOffset = 0;
  uint32_t tokenCount = 0;
  std::span<const uint32_t> pageTable;
  uint64_t pageTableRevision = 0;
  std::span<const uint32_t> inputTokens{};
};

struct ModelStepResult final {
  uint64_t requestId = 0;
  uint32_t consumedPromptTokens = 0;
  std::vector<uint32_t> outputTokens;
  // True after a stop token or the final score-only prefill chunk.
  // Generation budget exhaustion is the engine's decision.
  bool finished = false;
  DecodeStage nextDecodeStage = DecodeStage::Regular;
  uint32_t draftedTokens = 0;
  uint32_t acceptedDraftTokens = 0;
  // Trailing output tokens that have no target KV row: a stop token or the
  // last budgeted token is emitted as soon as it is selected instead of
  // spending one more verify cycle on it. They never enter a cached block.
  uint32_t outputTokensWithoutKv = 0;
  // Raw final-prompt-position logits at the request's scoreTokens, in
  // requested order. Empty for generation and for cancelled/failed scoring.
  std::vector<float> scoreLogits{};
  // Set when the model computed an unusable result for this lane alone, such
  // as a non-finite score logit. The engine fails that one request before it
  // publishes cache state or emits output, and the rest of the batch stands.
  // Broken invariants and GPU faults stay exceptions and remain engine-fatal.
  std::string failure{};

  bool operator==(const ModelStepResult &) const = default;
};

struct ModelMaskRequest final {
  uint64_t requestId = 0;
  std::vector<uint32_t> simulationTokens;
};

class ModelBatchTicket {
public:
  virtual ~ModelBatchTicket() = default;
  [[nodiscard]] virtual std::vector<ModelMaskRequest> takeMaskRequests() {
    return {};
  }
  [[nodiscard]] virtual bool ownsMaskWait(uint64_t) const noexcept {
    return false;
  }
  virtual void abandonMask(uint64_t) noexcept {}
  [[nodiscard]] virtual bool ready() const noexcept = 0;
  [[nodiscard]] virtual std::vector<ModelStepResult> wait() = 0;
  [[nodiscard]] virtual double wallMilliseconds() const noexcept = 0;
  // Fixed auxiliary work remains in wall latency but must not train the
  // text-prefill throughput estimate.
  [[nodiscard]] virtual bool prefillTimingIsRepresentative() const noexcept {
    return true;
  }
};

namespace model {

// Startup capabilities used for protocol sizing and admission independently
// of target-specific headers.
struct ModelCapabilities final {
  uint32_t vocabularySize = 0;
  uint32_t maximumContextTokens = 0;
  uint32_t maximumBatchWidth = 0;
  uint32_t prefillTokenBudget = 0;
  uint32_t draftQueryRows = 0;
  uint32_t draftProposalTokens = 0;
  uint32_t targetVerifyRows = 0;
  uint32_t draftContextTokens = 0;
};

// Compile-time ceiling of the one native DFlash execution contract. Concrete
// target/draft manifests are validated against these capabilities at startup;
// cache-page and attention-kernel geometry live with their operators.
struct ExecutionLimits final {
  static constexpr uint32_t maximumBatchWidth = 4;
  static constexpr uint32_t prefillTokenBudget = 2048;
  static constexpr uint32_t draftQueryRows = 8;
  static constexpr uint32_t draftProposalTokens = 7;
  static constexpr uint32_t targetVerifyRows = 8;
  static constexpr uint32_t draftContextTokens = 2048;
  static constexpr uint32_t speculativeScratchTokens = targetVerifyRows - 1;
  // One step emits at most its retained verify rows plus a terminal anchor
  // (a stop token or the last budgeted token) that never receives a KV row.
  static constexpr uint32_t maximumStepTokens = targetVerifyRows + 1;
  // Direct finite-option scoring (SemIf/Jev): a score-only request carries
  // 2..255 distinct token ids and returns their raw final-position logits.
  static constexpr uint32_t minimumScoreOptions = 2;
  static constexpr uint32_t maximumScoreOptions = 255;
};

static_assert(ExecutionLimits::draftQueryRows ==
              ExecutionLimits::draftProposalTokens + 1);
static_assert(ExecutionLimits::targetVerifyRows ==
              ExecutionLimits::draftQueryRows);

// Shared accounting for model-owned state allocations.  Target and draft
// implementations may allocate from separate physical pools while the engine
// observes one byte total through StateStorage.
struct StateAllocationTracker final {
  std::atomic<uint64_t> bytes{0};
};

class StateStorage {
public:
  virtual ~StateStorage() = default;
  [[nodiscard]] virtual uint64_t actualAllocatedBytes() const noexcept = 0;
  // Frees pooled idle buffers beyond the counts kept warm and returns the
  // bytes released. Active lanes and cached states are never touched.
  [[nodiscard]] virtual uint64_t releaseIdle(uint32_t keepCells,
                                             uint32_t keepRings) noexcept = 0;
};

// Startup sizing and observability are part of the concrete model runtime,
// not cache or scheduler state. The engine consumes these values without
// knowing the target or draft architecture that produced them.
struct ModelMemoryPlan final {
  uint64_t activeStateCellPlannedAllocatedBytes = 0;
  uint64_t sharedPrefillPlannedAllocatedBytes = 0;
  uint64_t sharedDecodePlannedAllocatedBytes = 0;
  uint64_t pipelineReserveBytes = 0;
  uint64_t runtimeOverheadReserveBytes = 0;

  [[nodiscard]] std::optional<std::string> validationError() const {
    if (!activeStateCellPlannedAllocatedBytes)
      return "active_state_cell_planned_allocated_bytes_required";
    if (!sharedPrefillPlannedAllocatedBytes)
      return "shared_prefill_planned_allocated_bytes_required";
    if (!sharedDecodePlannedAllocatedBytes)
      return "shared_decode_planned_allocated_bytes_required";
    if (!pipelineReserveBytes)
      return "pipeline_reserve_required";
    if (!runtimeOverheadReserveBytes)
      return "runtime_overhead_reserve_required";
    return std::nullopt;
  }
};

struct ModelMemoryActual final {
  uint64_t stateActualAllocatedBytes = 0;
  uint64_t sharedPrefillActualAllocatedBytes = 0;
  uint64_t sharedDecodeActualAllocatedBytes = 0;
};

struct ModelTelemetry final {
  uint64_t stateResidentBytes = 0;
  uint32_t warmIdleStateCells = 0;
  uint64_t targetPrefillRows = 0;
  uint64_t draftContextRowsActive = 0;
  uint64_t draftContextRowsMaterialization = 0;
  uint64_t draftContextRowsAvoided = 0;
  uint64_t draftStateRestoreSkipped = 0;
  uint64_t draftStateResets = 0;
  uint64_t imageEncodes = 0;
  uint64_t imageEmbeddingReuses = 0;
  uint32_t lastDecodeWidth = 0;
  uint64_t lastDecodeFusedOperations = 0;
  uint64_t lastDecodeM16Dispatches = 0;
  uint64_t lastDecodeM24Dispatches = 0;
  uint64_t lastDecodeM32Dispatches = 0;
  uint64_t constrainedMaskOverlapBatches = 0;
  uint64_t constrainedMaskOverlapRequests = 0;
  double lastConstrainedTargetForwardGpuSeconds = 0.0;
  double totalConstrainedTargetForwardGpuSeconds = 0.0;
  double lastConstrainedMaskWaitSeconds = 0.0;
  double totalConstrainedMaskWaitSeconds = 0.0;
  double lastPrefillGpuSeconds = 0.0;
  double lastDecodeGpuSeconds = 0.0;
  double totalPrefillGpuSeconds = 0.0;
  double totalDecodeGpuSeconds = 0.0;
  double lastPrefillWallSeconds = 0.0;
  double lastDecodeWallSeconds = 0.0;
  double totalPrefillWallSeconds = 0.0;
  double totalDecodeWallSeconds = 0.0;
};

// Observable outcome used to check repeated runs of the same configuration.
// Different configurations may round differently and choose different tokens;
// paired timings only require comparable work, not identical token IDs.
struct WarmupLaneResult final {
  ModelStepResult step;
  std::optional<uint32_t> pendingToken;
  uint64_t committedTokens = 0;

  bool operator==(const WarmupLaneResult &) const = default;

  [[nodiscard]] bool sameWorkAs(const WarmupLaneResult &other) const noexcept {
    return step.requestId == other.step.requestId &&
        step.consumedPromptTokens == other.step.consumedPromptTokens &&
        step.outputTokens.size() == other.step.outputTokens.size() &&
        step.finished == other.step.finished &&
        step.nextDecodeStage == other.step.nextDecodeStage &&
        step.draftedTokens == other.step.draftedTokens &&
        step.acceptedDraftTokens == other.step.acceptedDraftTokens &&
        step.outputTokensWithoutKv == other.step.outputTokensWithoutKv &&
        pendingToken.has_value() == other.pendingToken.has_value() &&
        committedTokens == other.committedTokens;
  }
};

struct WarmupStepResult final {
  bool completed = false;
  uint64_t estimatedPeakBytes = 0;
  std::string detail;
  // For prefill/decode-batch warmup, the synchronous production phase
  // including graph construction and result finalization, but not setup,
  // observation capture or teardown.
  double wallSeconds = 0.0;
  // Those two warmups return one observation per lane in batch-plan order.
  std::vector<WarmupLaneResult> lanes;
};

class Model {
public:
  virtual ~Model() = default;
  virtual void checkHealth() {}
  [[nodiscard]] virtual bool needsHealthCheck() const noexcept { return false; }
  [[nodiscard]] virtual StateAdmission begin(const ModelRequest &request) = 0;
  // Safe-point preemption releases execution backing, retaining only the
  // request's host-side sampling/constraint continuation. Resume replays the
  // supplied committed history through the ordinary packed-prefill path.
  virtual void suspend(uint64_t requestId) = 0;
  [[nodiscard]] virtual StateAdmission resume(const ModelRequest &request) = 0;
  virtual void restore(uint64_t requestId, uint32_t restoredPrefixLength,
                       std::shared_ptr<const CompositeState> state,
                       bool restoreDraftState) = 0;
  virtual void setDraftContextPlan(uint64_t requestId,
                                   DraftContextPlan plan) = 0;
  // Optional async wake hook; an immediately ready ticket need not call it.
  [[nodiscard]] virtual std::unique_ptr<ModelBatchTicket>
  submit(const BatchPlan &plan, std::span<const ModelBatchItem> items,
         std::function<void()> completion) = 0;
  // Copies the request's committed state at its current page-aligned
  // boundary into a cache slot. Returns nullptr when no slot is free and the
  // governor denies a new one; the caller may release a cached state and
  // retry.
  [[nodiscard]] virtual std::shared_ptr<const CompositeState>
  snapshot(uint64_t requestId) = 0;
  // Releases one unit of idle model state (an unused buffer, then caches
  // that can be rebuilt) and returns its bytes; zero when nothing is idle.
  // A denied allocation retries between calls, so it frees only what it
  // needs.
  [[nodiscard]] virtual uint64_t reclaimIdleState() noexcept = 0;
  virtual void provideMask(uint64_t requestId,
                           std::span<const uint32_t> words) = 0;
  virtual void end(uint64_t requestId) = 0;
};

// Adds startup warmup and observability to the model interface.
class RuntimeModel : public Model {
public:
  ~RuntimeModel() override = default;

  // Actual rows, not padded dispatch rows; valid range is 1..prefillTokenBudget.
  virtual WarmupStepResult warmupPrefill(uint32_t rows) = 0;
  virtual WarmupStepResult warmupDecodeBatch(uint32_t width) = 0;
  virtual WarmupStepResult warmupDraftVerifyCommit() = 0;
  virtual WarmupStepResult warmupCompositeStateRestore() = 0;
  [[nodiscard]] virtual ModelMemoryActual actualRuntimeMemory() const = 0;
  [[nodiscard]] virtual ModelTelemetry telemetry() const noexcept = 0;
};

} // namespace model
} // namespace splash

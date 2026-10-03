#pragma once

#include "ops/PagedKv.hpp"
#include "ops/Vision.hpp"
#include "model/StateTransfer.hpp"

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace splash {

enum class WorkKind : uint8_t { Prefill, Decode };

enum class DecodeStage : uint8_t {
  // Drafts and verifies.
  Regular,
  // A constrained request's prompt is done; it waits for, or holds, the mask
  // its first token is selected under.
  ApplyInitialMask,
};

[[nodiscard]] constexpr bool waitsForMask(DecodeStage stage) noexcept {
  return stage == DecodeStage::ApplyInitialMask;
}

// Values are the native request frame's constraint byte.
enum class ConstraintMode : uint8_t { None = 0, TokenMask = 1 };

// Request options, one bit each, as the native request frame's flags word
// carries them; a request with any other bit set is a request error.
enum RequestFlag : uint32_t {
  // Never select the model's stop tokens, so generation runs to its output
  // limit. Only unconstrained generation can carry it.
  RequestIgnoreEndOfSequence = 1U << 0,
};

inline constexpr uint32_t kRequestFlagBits = RequestIgnoreEndOfSequence;

// A request's token selection, its fields in the native request frame's
// order. The defaults are greedy selection with nothing changing the logits.
struct SamplingParameters final {
  // Zero selects greedily; sampling divides the logits by it.
  float temperature = 0.0F;
  float topP = 1.0F;
  // Sampling keeps the topK most likely tokens; 0 keeps every token, as does
  // a topK past the vocabulary.
  uint32_t topK = 0;
  // The sampling penalties, applied before greedy and sampled selection alike:
  // repetition scales the logits of prompt and output tokens, presence and
  // frequency lower those of output tokens. The defaults change nothing.
  float presencePenalty = 0.0F;
  float frequencyPenalty = 0.0F;
  float repetitionPenalty = 1.0F;
  // Sampling drops the tokens less likely than minP times the most likely
  // one, before top-k and top-p; 0 drops none.
  float minP = 0.0F;
  uint64_t seed = 0;

  bool operator==(const SamplingParameters &) const = default;

  // The first rule the parameters break, or none. A sampling temperature is
  // at least FLT_MIN: the kernels divide by it, and Metal flushes a subnormal
  // one to zero.
  [[nodiscard]] std::optional<std::string_view>
  validationError() const noexcept {
    const bool temperatureValid =
        temperature == 0.0F ||
        (std::isfinite(temperature) &&
         temperature >= std::numeric_limits<float>::min());
    if (!temperatureValid || !(topP > 0.0F && topP <= 1.0F) ||
        !(minP >= 0.0F) || minP > 1.0F) {
      return "sampling requires temperature 0 or at least FLT_MIN, top_p in "
             "(0,1] and min_p in [0,1]";
    }
    if (!(std::fabs(presencePenalty) <= 2.0F) ||
        !(std::fabs(frequencyPenalty) <= 2.0F) ||
        !std::isfinite(repetitionPenalty) || repetitionPenalty <= 0.0F) {
      return "sampling requires presence and frequency penalties in [-2,2] "
             "and a positive repetition penalty";
    }
    return std::nullopt;
  }

  // Greedy selection with nothing changing the logits, whatever the seed.
  [[nodiscard]] bool isNeutral() const noexcept {
    SamplingParameters neutral;
    neutral.seed = seed;
    return *this == neutral;
  }
};

// Immutable view of the fields a model needs to activate a sequence.  Engine
// priority and deadline policy deliberately do not cross this boundary.
struct ModelRequest final {
  uint64_t id = 0;
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
  // RequestFlag bits.
  uint32_t flags = 0;
  // Prompt tokens the lane takes from a cached state. Images that end at or
  // before it are neither staged nor encoded; their spans still place
  // rotary positions.
  uint32_t restoredTokens = 0;
};

// One image in the prompt: the run of placeholder tokens it occupies (one per
// merged 2x2 patch group, row-major over the merged grid), the patch grid of
// the frontend's resized pixels, and a 128-bit digest of that content.
// Placeholder token ids are identical for every image, so cache identity keys
// on the digest as well as the tokens.
struct ImageSpan final {
  uint32_t offset = 0;
  uint32_t tokens = 0;
  uint32_t gridHeight = 0;
  uint32_t gridWidth = 0;
  uint64_t digestLo = 0;
  uint64_t digestHi = 0;

  [[nodiscard]] uint32_t end() const noexcept { return offset + tokens; }
  [[nodiscard]] ops::ImageGrid grid() const noexcept {
    return {gridHeight, gridWidth};
  }
  [[nodiscard]] uint64_t pixelBytes() const noexcept {
    return grid().pixelBytes();
  }
  bool operator==(const ImageSpan &) const = default;
};

// The first rule a request's image spans break, or none: each covers a valid
// grid within the patch limit with one token per merged patch group, the
// runs are sorted, disjoint and inside the prompt, and the pixels are the
// grids' own.
[[nodiscard]] inline std::optional<std::string_view>
imageSpansValidationError(std::span<const ImageSpan> spans, size_t promptTokens,
                          uint64_t pixelBytes) noexcept {
  uint64_t previousEnd = 0;
  uint64_t gridPixelBytes = 0;
  for (const ImageSpan &span : spans) {
    const ops::ImageGrid grid = span.grid();
    if (!grid.valid() || grid.patches() > ops::kMaximumImagePatches)
      return "image grid must be even-sided and within the patch limit";
    if (span.tokens != grid.mergedTokens())
      return "image span tokens must equal the merged grid size";
    const uint64_t end = uint64_t{span.offset} + span.tokens;
    if (span.offset < previousEnd || end > promptTokens) {
      return "image spans must be sorted, non-overlapping runs inside the "
             "prompt";
    }
    previousEnd = end;
    gridPixelBytes += grid.pixelBytes();
  }
  if (pixelBytes != gridPixelBytes)
    return "image pixels do not match the image grids";
  return std::nullopt;
}

// Immutable target-recurrent plus draft-context state.  Concrete model
// implementations own its buffers; the engine only pins and accounts it.
class CompositeState {
public:
  virtual ~CompositeState() = default;
  // Footprint retained by the cache. A cached state owns a private copy of
  // the lane's state; dropping the reference returns its buffers to the
  // model's pool, and idle-state reclaim frees them.
  [[nodiscard]] virtual uint64_t bytes() const noexcept = 0;
  [[nodiscard]] virtual uint64_t residentBytes() const noexcept { return bytes(); }
  [[nodiscard]] virtual bool canOffload() const noexcept { return false; }
  // Starts writing this state to the disk tier and returns the ticket that
  // carries its disk copy; the source is free as soon as the call returns.
  // A null ticket means that the disk quota cannot admit another state.
  [[nodiscard]] virtual std::unique_ptr<StateOffload>
  offload(std::function<void()>) const { return {}; }
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
  std::vector<DraftCaptureSpan> captureSpans;
  std::vector<DraftBoundaryPlan> boundaries;
  // The first capture continues the restored draft ring: a non-zero replay
  // whose first boundary is within one draft window of it. Otherwise the
  // restore skips the ring.
  bool restoresDraftState = false;
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

// A replay from a non-zero boundary starts from the composite state restored
// there.
[[nodiscard]] DraftContextPlan
planDraftContext(uint32_t replayBegin, uint32_t replayEnd,
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
  // A decode's lanes exchange token masks with the host; constrained and
  // unconstrained lanes never share a command.
  bool constrained = false;
  std::vector<BatchItem> items;
  DecodeStage decodeStage = DecodeStage::Regular;

  [[nodiscard]] bool empty() const noexcept { return items.empty(); }
  [[nodiscard]] uint32_t width() const noexcept {
    return static_cast<uint32_t>(items.size());
  }
};

enum class StateFailure : uint8_t { None, ConcurrencyLimit, MemoryPressure };

struct StateAdmission final {
  std::optional<uint32_t> lane;
  StateFailure failure = StateFailure::None;
  metal::AllocationFailure allocationFailure = metal::AllocationFailure::None;
  // On a refused start: what the attempt matched (cached image rows, the
  // encoder it would use). The caller keeps it while it reclaims and
  // retries, so the retry finds them.
  std::shared_ptr<const void> held{};

  [[nodiscard]] bool granted() const noexcept { return lane.has_value(); }
};

struct ModelBatchItem final {
  uint64_t requestId = 0;
  uint64_t logicalPosition = 0;
  uint32_t tokenCount = 0;
  std::span<const uint32_t> pageTable;
  // pageTableRevision names the page list and is nonzero; the list equals
  // the one at revision - 1 below pageTableFirstChanged.
  uint64_t pageTableRevision = 0;
  uint32_t pageTableFirstChanged = 0;
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
  // as a non-finite score logit or a selection outside the vocabulary (the
  // sampling kernels' sentinel for a non-finite logit row). The engine fails
  // that one request before it publishes cache state or emits output, and the
  // rest of the batch stands. Broken invariants and GPU faults stay
  // exceptions and remain engine-fatal.
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
};

// A token mask row holds one bit per vocabulary token in 32-bit words.
[[nodiscard]] constexpr uint32_t
maskWordsPerToken(uint32_t vocabularySize) noexcept {
  return static_cast<uint32_t>((uint64_t{vocabularySize} + 31) / 32);
}

// Compile-time ceiling of the one native DFlash execution contract. Concrete
// target/draft manifests are validated against these limits at startup;
// cache-page and attention-kernel geometry live with their operators.
struct ExecutionLimits final {
  static constexpr uint32_t maximumBatchWidth = 4;
  static constexpr uint32_t prefillTokenBudget = 2048;
  // KV pages startup warmup runs on, from page 0: the runway the engine's KV
  // pool allocates when it is built.
  static constexpr uint32_t warmupKvPages =
      (prefillTokenBudget + kv::kPageTokens - 1) / kv::kPageTokens;
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
// observes one byte total through ModelMemoryActual::stateActualAllocatedBytes.
struct StateAllocationTracker final {
  std::atomic<uint64_t> bytes{0};
};

// Startup sizing and observability are part of the concrete model runtime,
// not cache or scheduler state. The engine consumes these values without
// knowing the target or draft architecture that produced them.
struct ModelMemoryPlan final {
  uint64_t laneStatePlannedAllocatedBytes = 0;
  uint64_t sharedPrefillPlannedAllocatedBytes = 0;
  uint64_t sharedDecodePlannedAllocatedBytes = 0;
};

// Fixed reserves the memory plan carries beside the planned arenas: Metal
// pipeline objects and encoder scratch, and the process's own runtime
// overhead. Startup counts them before a model loads.
inline constexpr uint64_t kPipelineReserveBytes = 256ULL << 20;
inline constexpr uint64_t kRuntimeOverheadReserveBytes = 512ULL << 20;

struct ModelMemoryActual final {
  uint64_t stateActualAllocatedBytes = 0;
  uint64_t sharedPrefillActualAllocatedBytes = 0;
  uint64_t sharedDecodeActualAllocatedBytes = 0;
  // The buffer a state's write to the disk tier stages through; zero without
  // a tier.
  uint64_t stateStagingBytes = 0;
};

struct ModelTelemetry final {
  uint64_t stateAllocatedBytes = 0;
  // Pooled state buffers no lane holds: GDN parity cells and draft rings.
  uint32_t idleGdnCells = 0;
  uint32_t idleDraftRings = 0;
  uint64_t targetPrefillRows = 0;
  uint64_t draftContextRowsActive = 0;
  uint64_t draftContextRowsMaterialization = 0;
  uint64_t draftContextRowsAvoided = 0;
  uint64_t draftStateRestoreSkipped = 0;
  uint64_t draftStateResets = 0;
  uint64_t imageEncodes = 0;
  uint64_t imageEmbeddingReuses = 0;
  // The vision encoder's scratch while it exists, the encoded rows the
  // embedding cache keeps for reuse, those states in RAM hold to resume
  // inside an image, and every image's rows anything holds, each counted
  // once.
  uint64_t visionArenaBytes = 0;
  uint64_t embeddingCacheBytes = 0;
  uint64_t stateHeldImageBytes = 0;
  uint64_t imageRowsBytes = 0;
  uint32_t lastDecodeWidth = 0;
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

// One warmup lane's observable outcome: its step result, the anchor the step
// left pending and the tokens it committed. The runtime oracle checks the
// warmups' tokens and their repeatability by it; the decode warmup's anchor
// is set inside the runtime, so no public request reproduces that lane.
struct WarmupLaneResult final {
  ModelStepResult step;
  std::optional<uint32_t> pendingToken;
  uint64_t committedTokens = 0;

  bool operator==(const WarmupLaneResult &) const = default;
};

struct WarmupStepResult final {
  std::string detail;
  // For prefill/decode-batch warmup, the synchronous production phase
  // including graph construction and result finalization, but not setup,
  // observation capture or teardown.
  double wallSeconds = 0.0;
  // Those two warmups return one observation per lane in batch-plan order.
  std::vector<WarmupLaneResult> lanes;
};

// What a reclaim step may release of the model's idle memory.
enum class IdleMemory : uint8_t {
  // Pooled state buffers.
  Buffers,
  // Pooled state buffers, then caches that can be rebuilt.
  BuffersThenCaches,
};

class Model {
public:
  virtual ~Model() = default;
  virtual void checkHealth() {}
  [[nodiscard]] virtual StateAdmission begin(const ModelRequest &request) = 0;
  // Safe-point preemption returns the request's state buffers, retaining
  // only its host-side sampling/constraint continuation. Resume replays the
  // supplied committed history through the ordinary packed-prefill path.
  virtual void suspend(uint64_t requestId) = 0;
  [[nodiscard]] virtual StateAdmission resume(const ModelRequest &request) = 0;
  // Restores a cached state into the request's lane at `boundary`. A RAM
  // state is copied now and the call returns null; a disk state returns the
  // read, whose finish() commits it. `completion` only wakes the engine.
  [[nodiscard]] virtual std::unique_ptr<StateRestore>
  beginRestore(uint64_t requestId, uint32_t boundary,
               std::shared_ptr<const CompositeState> state, bool restoreDraft,
               std::function<void()> completion) = 0;
  virtual void setDraftContextPlan(uint64_t requestId,
                                   DraftContextPlan plan) = 0;
  // Optional async wake hook; an immediately ready ticket need not call it.
  [[nodiscard]] virtual std::unique_ptr<ModelBatchTicket>
  submit(const BatchPlan &plan, std::span<const ModelBatchItem> items,
         std::function<void()> completion) = 0;
  // Copies the request's committed state at its current page-aligned
  // boundary into the cached state's buffers. Returns nullptr when the pool
  // has none free and the governor denies new ones; the caller may release a
  // cached state and retry.
  [[nodiscard]] virtual std::shared_ptr<const CompositeState>
  snapshot(uint64_t requestId) = 0;
  // The bytes one lane's state snapshot allocates.
  [[nodiscard]] virtual uint64_t snapshotBytes() const noexcept = 0;
  // Whether the disk tier takes a state written from a lane: a tier exists
  // and its state file accepts writes. The quota is the write's own concern.
  [[nodiscard]] virtual bool canSnapshotToDisk() const noexcept { return false; }
  // Writes the request's committed state at its current page-aligned
  // boundary to the disk tier from the lane's own buffers, for a state the
  // pool has no cached state's buffers for; the ticket carries its disk copy.
  // Null when the quota cannot admit another state: the caller may free quota
  // and retry.
  [[nodiscard]] virtual std::unique_ptr<StateOffload>
  snapshotToDisk(uint64_t, std::function<void()>) { return {}; }
  // The cached states whose buffers a lane's activation would still have to
  // allocate: each one evicted returns to the pool what a lane takes. Zero
  // when the pool holds a lane's buffers.
  [[nodiscard]] virtual uint32_t statesToActivate() const noexcept { return 0; }
  // Releases one unit of idle model memory and returns its bytes; zero when
  // nothing in scope is idle. A unit is one pooled state buffer (keepLane
  // keeps those one lane starts from); with BuffersThenCaches, once no
  // buffer is idle, the vision arena when no image waits for its encode and
  // nothing holds it, and then one embedding entry nothing else holds,
  // oldest first. A denied allocation retries between calls, so it frees
  // only what it needs.
  [[nodiscard]] virtual uint64_t reclaimIdleState(bool keepLane,
                                                  IdleMemory scope) noexcept = 0;
  // Why this request's mask is unusable, or nothing when the model took it.
  [[nodiscard]] virtual std::optional<std::string>
  provideMask(uint64_t requestId, std::span<const uint32_t> words) = 0;
  virtual void end(uint64_t requestId) = 0;
};

// Adds startup warmup and observability to the model interface.
class RuntimeModel : public Model {
public:
  ~RuntimeModel() override = default;

  // Actual rows, not padded dispatch rows; valid range is 1..prefillTokenBudget.
  virtual WarmupStepResult warmupPrefill(uint32_t rows) = 0;
  virtual WarmupStepResult warmupDecodeBatch(uint32_t width) = 0;
  virtual WarmupStepResult warmupCompositeStateRestore() = 0;
  [[nodiscard]] virtual ModelMemoryActual actualRuntimeMemory() const = 0;
  [[nodiscard]] virtual ModelTelemetry telemetry() const noexcept = 0;
};

} // namespace model
} // namespace splash

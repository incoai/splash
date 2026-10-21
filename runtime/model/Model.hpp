#pragma once

#include "ops/PagedKv.hpp"
#include "ops/Vision.hpp"
#include "model/StateTransfer.hpp"

#include <algorithm>
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
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace splash {

namespace model {
class KvPageSources;
} // namespace model

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

// A request that generates inside a thinking block may select at most
// `tokens` thinking tokens, and a constrained one more until its grammar
// admits the end; the target then selects the `end` tokens in order, the
// last of which closes the block. Thinking also ends where the request
// selects that close or `alsoEnds` first, such as a tool call's opening that
// follows thinking directly; no other end token is either.
struct ThinkingBudget final {
  uint32_t tokens = 0;
  std::vector<uint32_t> end;
  uint32_t alsoEnds = 0;

  // Why the target cannot end thinking with this budget: no end tokens, or
  // one before the last that would end thinking itself. Null when it can.
  [[nodiscard]] const char *unusable() const noexcept {
    if (end.empty())
      return "a thinking budget needs end tokens";
    if (std::any_of(end.begin(), end.end() - 1,
                    [&](uint32_t token) { return token == end.back() || token == alsoEnds; }))
      return "only the last thinking end token may end thinking";
    return nullptr;
  }

  bool operator==(const ThinkingBudget &) const = default;
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
  // The engine request's thinking budget; null for a request without one.
  const ThinkingBudget *thinkingBudget = nullptr;
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
  // Whether the disk tier takes this state's write now, for a state in use
  // or not: a RAM copy, the tier's state file takes writes, its limit holds
  // a state's slot, and the write budget admits one of the class
  // (model::SlotFile::takesWrite).
  [[nodiscard]] virtual bool canOffload(bool) const noexcept { return false; }
  // Starts writing this state to the disk tier and returns the ticket that
  // carries its disk copy; the source is free as soon as the call returns.
  // A null ticket means that the disk quota cannot admit another state.
  [[nodiscard]] virtual std::unique_ptr<StateOffload>
  offload(std::function<void()>) const { return {}; }
  // Starts writing this state to the disk tier from its own buffers, for a
  // persistent tier's copy of a state that stays in RAM: the buffers must
  // stay as they are until the ticket is ready. Null as for offload().
  [[nodiscard]] virtual std::unique_ptr<StateOffload>
  persist(std::function<void()>) const { return {}; }
  // On a disk copy in a persistent tier, keeps the label with its slot once
  // the write started before has landed (SlotFile::label). Nothing
  // otherwise.
  virtual void label(std::vector<std::byte>) const {}
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
  std::vector<BatchItem> items;
  DecodeStage decodeStage = DecodeStage::Regular;
  // A prefill of one lane may run in passes, each over the next of `passes`
  // even shares of the target's layers (Model::prefillPasses): this command
  // is pass `pass`. The first pass begins the rows, the last ends them and
  // commits; the rows' work between lives in the model's prefill memory,
  // which no other prefill touches meanwhile, while other lanes' decode
  // commands run between passes. A request that ends between passes leaves
  // the rest unrun and nothing of the rows committed.
  uint32_t pass = 0;
  uint32_t passes = 1;

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

// The buffers of one cached state, taken for a state boundary that a prefill
// command reaches (Model::reserveState). The command's completion seals them
// into the state of that boundary; whatever it still holds goes back to the
// model's pool when the last handle goes, which the command's ticket keeps
// until the GPU is done with them.
class StateReservation {
public:
  virtual ~StateReservation() = default;
};

// A state a prefill command captures: the request's composite state after
// `tokens` prompt tokens, in `reservation`'s buffers. The command writes the
// state of a boundary inside it as a command ending there would store it, and
// its completion copies the state of a boundary at its end.
struct StateCapture final {
  uint32_t tokens = 0;
  std::shared_ptr<StateReservation> reservation;
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
  // Where each page of the list reads its K/V from, cut with the list
  // (engine::Cache::cutPageTable); null while no page holds only its index.
  // The command keeps it until it completes, and lets it go before its
  // ticket's wait() returns.
  std::shared_ptr<const model::KvPageSources> pageSources{};
  std::span<const uint32_t> inputTokens{};
  // A prefill chunk's states to capture, ascending: page-aligned boundaries in
  // (logicalPosition, logicalPosition + tokenCount], each one where the
  // request's draft context plan leaves a whole draft window.
  std::vector<StateCapture> stateCaptures{};
};

struct ModelStepResult final {
  uint64_t requestId = 0;
  // A prefill's rows, which only its last pass consumes (BatchPlan::passes):
  // an earlier pass's result is the request's id and nothing else.
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
  // The states of the chunk's stateCaptures, in order, when the chunk
  // commits; empty when it does not.
  std::vector<std::shared_ptr<const CompositeState>> capturedStates{};
  // Set when the model computed an unusable result for this lane alone: a
  // prefill chunk whose last row is not finite, a non-finite score logit or
  // a selection outside the vocabulary (the sampling kernels' sentinel for a
  // logit row with no selectable token). Every step that publishes cache
  // state is checked so, which lets the engine keep what earlier steps
  // published. The engine fails that one request before it publishes cache
  // state or emits output, and the rest of the batch stands. Broken
  // invariants and GPU faults stay exceptions and remain engine-fatal.
  std::string failure{};
  // Set when the tier copies of blocks this lane read could not be read
  // (model::KvPageSource::block), on a host step or anywhere else: a failed
  // read is the lanes' that needed it, never an exception the engine dies
  // of. The engine ends that one request as retryable
  // (engine::LaneOutcome::KvUnreadable) before it publishes cache state or
  // emits output, keeps the state it restored, which the failure says
  // nothing of, and gives the copies up (engine::Cache::poisonCopy).
  std::vector<uint64_t> unreadableKvBlocks{};
  // The request's first output token, which the step that completes its
  // prompt selected and left pending: the first decode cycle writes its KV row
  // and emits it first. Reported so that the engine can send it to the client
  // now rather than with that cycle.
  std::optional<uint32_t> firstToken{};

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
  // Of wallMilliseconds, the time the command's GPU work waited on its host
  // steps: a streamed target's expert reads, which a prefill command pays
  // about whole whatever its rows.
  [[nodiscard]] virtual double hostMilliseconds() const noexcept { return 0.0; }
  // Of hostMilliseconds, the time it waited on the reads of its history's
  // spilled K/V (model::KvHistoryStream), which bound a streamed prefill
  // command's length whatever its rows.
  [[nodiscard]] virtual double streamMilliseconds() const noexcept { return 0.0; }
  // Fixed auxiliary work remains in wall latency but must not train the
  // text-prefill throughput estimate.
  [[nodiscard]] virtual bool prefillTimingIsRepresentative() const noexcept {
    return true;
  }
};

namespace model {

// The attention of a block-sparse target, which lets its committed K/V leave
// memory while each page keeps its blocks' index (kv::PageForm::Index): each
// verify row of a lane attends at most selectedBlocks blocks of blockTokens
// tokens. Rows that read the whole of a page table instead (a row with at
// most denseHistoryTokens of history, its first sinkTokens and its last
// localTokens) read only Full pages, which the cache keeps for them.
struct SparseKvShape final {
  uint32_t blockTokens = 0;
  uint32_t selectedBlocks = 0;
  uint32_t verifyRows = 0;
  uint32_t denseHistoryTokens = 0;
  uint32_t sinkTokens = 0;
  uint32_t localTokens = 0;

  // The hot block cache of spilled K/V (KvBlockCache) in slots a layer: one
  // decoding lane's rows select at most laneSlots(), and while `lanes` lanes
  // decode over Index pages it holds requiredSlots(lanes), one lane's more
  // than they select, so that two lanes' selections do not thrash it; for
  // one lane, the size past which more slots save few reads.
  [[nodiscard]] constexpr uint64_t laneSlots() const noexcept { return uint64_t{verifyRows} * selectedBlocks; }
  [[nodiscard]] constexpr uint64_t requiredSlots(uint32_t lanes) const noexcept {
    return lanes ? (uint64_t{lanes} + 1) * laneSlots() : 0;
  }
  // The history from its start that a row may read whole through its page
  // table: its dense history, or its sink tokens if more.
  [[nodiscard]] constexpr uint32_t denseTokens() const noexcept { return std::max(denseHistoryTokens, sinkTokens); }

  bool operator==(const SparseKvShape &) const = default;
};

// Startup capabilities used for protocol sizing and admission independently
// of target-specific headers.
struct ModelCapabilities final {
  uint32_t vocabularySize = 0;
  uint32_t maximumContextTokens = 0;
  // Set for a block-sparse target.
  std::optional<SparseKvShape> sparseKv;
};

// A tier copy of one block's page, as reads of spilled K/V reach it on any
// thread: engine::KvPageTier's slots, or a test's bytes.
class KvCopy {
public:
  virtual ~KvCopy() = default;
  // Unique in the process: the copy's one write (model::SlotFile::Slot::id).
  [[nodiscard]] virtual uint64_t id() const noexcept = 0;
  // `bytes` at `offset` of the copy, read into `buffer` (SlotFile::readAt
  // says what it must hold), or nothing when they cannot be read.
  [[nodiscard]] virtual std::optional<std::span<const std::byte>>
  readAt(uint64_t offset, uint64_t bytes, std::span<std::byte> buffer) const noexcept = 0;
};

// Where a request's page reads its K/V: {} for a page that holds it
// (kv::PageForm::Full), else the block on a page that holds only its index
// (kv::PageForm::Index) and the tier copy that holds its K/V.
struct KvPageSource final {
  uint64_t block = 0;
  std::shared_ptr<const KvCopy> copy;

  [[nodiscard]] bool indexed() const noexcept { return block != 0; }
};

// A request's page sources at one page-table revision, immutable, in chunks
// of kChunkPages pages that snapshots share: a cut copies only the chunks
// from its first changed page on. Chunk c holds pages [c kChunkPages,
// (c + 1) kChunkPages), the last chunk the rest, and counts its Index pages.
class KvPageSources final {
public:
  static constexpr uint32_t kChunkPages = 256;
  struct Chunk final {
    std::array<KvPageSource, kChunkPages> pages{};
    uint32_t indexPages = 0;
  };

  KvPageSources() = default;
  // std::invalid_argument unless the chunks, none null, hold `pages` pages.
  KvPageSources(std::vector<std::shared_ptr<const Chunk>> chunks, uint32_t pages)
      : chunks_(std::move(chunks)), pages_(pages) {
    if (chunks_.size() != (uint64_t{pages} + kChunkPages - 1) / kChunkPages)
      throw std::invalid_argument("KV page sources do not hold their pages");
    for (const std::shared_ptr<const Chunk> &chunk : chunks_) {
      if (!chunk) throw std::invalid_argument("KV page sources hold a null chunk");
      indexPages_ += chunk->indexPages;
    }
  }

  [[nodiscard]] uint32_t size() const noexcept { return pages_; }
  [[nodiscard]] uint32_t indexPages() const noexcept { return indexPages_; }
  // A page past them reads its own K/V: one the command writes.
  [[nodiscard]] const KvPageSource &operator[](uint32_t page) const noexcept {
    static const KvPageSource own{};
    return page < pages_ ? chunks_[page / kChunkPages]->pages[page % kChunkPages] : own;
  }
  [[nodiscard]] std::span<const std::shared_ptr<const Chunk>> chunks() const noexcept { return chunks_; }

private:
  std::vector<std::shared_ptr<const Chunk>> chunks_;
  uint32_t pages_ = 0;
  uint32_t indexPages_ = 0;
};

// Memory the model takes for a request between commands through the
// engine's ordinary allocation loop (Engine::allocate): granted, or the
// cause of the refusal, after which the engine reclaims and retries.
struct ModelMemoryAdmission final {
  bool ok = false;
  metal::AllocationFailure allocationFailure = metal::AllocationFailure::None;

  [[nodiscard]] bool granted() const noexcept { return ok; }
};

// A token mask row holds one bit per vocabulary token in 32-bit words.
[[nodiscard]] constexpr uint32_t
maskWordsPerToken(uint32_t vocabularySize) noexcept {
  return static_cast<uint32_t>((uint64_t{vocabularySize} + 31) / 32);
}

// Compile-time ceiling of the one native DFlash execution contract. A draft
// must have been trained for blocks of draftQueryRows rows over
// draftContextTokens context tokens, which inspectModelRoot checks a DFlash2
// checkpoint's config for; the other limits are the runtime's own. Cache-page
// and attention-kernel geometry live with their operators.
struct ExecutionLimits final {
  static constexpr uint32_t maximumBatchWidth = 4;
  // The rows of a prefill chunk. A prefill command's lanes fill its chunks
  // one after another, and the runtime lets at most one lane's rows run past
  // a chunk: a prefill group has one scratch GDN cell for the state between
  // a lane's parts. So every lane but a command's last lies within its first
  // chunk, and a command holds at most each of its lanes' group rows
  // (Model::prefillGroupRows; a chunk's for a lane that runs no groups). The
  // rows the engine cuts, or a lane it drops, before the command runs keep
  // both: the lanes before the last still lie within the first chunk.
  static constexpr uint32_t prefillTokenBudget = 2048;
  // KV pages startup warmup runs on, from page 0: the runway the engine's KV
  // pool allocates when it is built.
  static constexpr uint32_t warmupKvPages =
      (prefillTokenBudget + kv::kPageTokens - 1) / kv::kPageTokens;
  static constexpr uint32_t draftQueryRows = 8;
  static constexpr uint32_t draftProposalTokens = 7;
  static constexpr uint32_t targetVerifyRows = 8;
  static constexpr uint32_t draftContextTokens = 2048;
  // The most target layers a DFlash2 draft reads (its target_layer_ids).
  static constexpr uint32_t maximumDraftTargetLayers = 8;
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
  // Pooled state buffers no lane holds: GDN parity cells, draft rings and
  // context windows.
  uint32_t idleGdnCells = 0;
  uint32_t idleDraftRings = 0;
  uint32_t idleContextWindows = 0;
  uint64_t targetPrefillRows = 0;
  uint64_t draftContextRowsActive = 0;
  uint64_t draftContextRowsMaterialization = 0;
  uint64_t draftContextRowsAvoided = 0;
  uint64_t draftStateRestoreSkipped = 0;
  uint64_t draftStateResets = 0;
  uint64_t imageEncodes = 0;
  uint64_t imageEmbeddingReuses = 0;
  // Entries the embedding cache let go of, past its bytes or to a reclaim:
  // the image of one is encoded again where it is placed next, unless
  // something else still holds its rows.
  uint64_t imageEmbeddingEvictions = 0;
  // The vision encoder's scratch while it exists, the encoded rows the
  // embedding cache keeps for reuse, those states in RAM hold to resume
  // inside an image, and every image's rows anything holds, each counted
  // once.
  uint64_t visionArenaBytes = 0;
  uint64_t embeddingCacheBytes = 0;
  uint64_t stateHeldImageBytes = 0;
  uint64_t imageRowsBytes = 0;
  uint64_t constrainedMaskOverlapBatches = 0;
  uint64_t constrainedMaskOverlapRequests = 0;
  // Prefill chunks the GPU ran again alone once the Neural Engine split's
  // work for them failed (Runtime::prefillAsync).
  uint64_t aneFfnReruns = 0;
  // Whether decode cycles verify the rows the verify budget chooses
  // (budgetsVerifyRows) rather than every row.
  bool verifyBudget = false;
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
  // For prefill warmup, of wallSeconds the time its GPU work waited on host
  // steps (ModelBatchTicket::hostMilliseconds).
  double hostSeconds = 0.0;
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
  // supplied committed history through the ordinary prefill path, and takes
  // the output limit it is given (maxNewTokens), which the engine sets anew
  // at each start of a request (the SSD cache's quota may bound it). That
  // limit must leave output past the anchor the request holds once its
  // first token is selected, which the suspension keeps: the model emits a
  // last budgeted anchor as it selects it, and decodes no lane left only
  // that anchor.
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
  // Takes the buffers of one cached state for the request's state boundary
  // at `tokens`, which the prefill command reaching it fills
  // (ModelBatchItem::stateCaptures): from the pool, or one admission for
  // what the pool lacks. Null when the governor refuses, which sets
  // `refusal` to the cause; the caller may release a cached state and retry.
  [[nodiscard]] virtual std::shared_ptr<StateReservation>
  reserveState(uint64_t requestId, uint32_t tokens, metal::AllocationFailure &refusal) = 0;
  // Copies the request's committed state at its current page-aligned
  // boundary into the buffers of a cached state, taken as reserveState()
  // takes them: the state of a boundary its prefill command ended at without
  // buffers. Null when the governor refuses; the caller may release a cached
  // state and retry.
  [[nodiscard]] virtual std::shared_ptr<const CompositeState> snapshot(uint64_t requestId) = 0;
  // The bytes one cached state's buffers take.
  [[nodiscard]] virtual uint64_t snapshotBytes() const noexcept = 0;
  // Memory for reads of a request's spilled K/V, taken between commands and
  // held until it ends or is suspended; a model that reads none grants it at
  // once, taking nothing. The hot block cache's floor for the request's lane,
  // which decodes over Index pages (model::KvBlockCache::reserveLane), in
  // place of the staging once no prompt holds it; the staging of a
  // prefill's history of `pages` pages, which holds Index pages, the
  // request's while its prompt goes on, also once refused
  // (model::KvHistoryStream::reserve); and, optionally, one more
  // extent of the hot block cache toward its planned size
  // (model::KvBlockCache::growTowardTarget).
  [[nodiscard]] virtual ModelMemoryAdmission reserveSpilledReads(uint64_t) { return {.ok = true}; }
  [[nodiscard]] virtual ModelMemoryAdmission reserveHistoryStaging(uint64_t, uint32_t) { return {.ok = true}; }
  [[nodiscard]] virtual ModelMemoryAdmission growSpilledReads() { return {.ok = true}; }
  // Whether the disk tier takes a state written from a lane now, for a
  // state in use or not, as CompositeState::canOffload. Room in the quota is
  // the write's own concern.
  [[nodiscard]] virtual bool canSnapshotToDisk(bool) const noexcept { return false; }
  // Writes the request's committed state at its current page-aligned
  // boundary to the disk tier from the lane's own buffers, for a state no
  // cached state's buffers could be reserved for; the ticket carries its disk
  // copy.
  // Null when the quota cannot admit another state: the caller may free quota
  // and retry.
  [[nodiscard]] virtual std::unique_ptr<StateOffload>
  snapshotToDisk(uint64_t, std::function<void()>) { return {}; }
  // The cached states whose eviction would return to the pool what a lane's
  // activation still has to allocate: zero when the pool holds a lane's
  // buffers, and the maximum when no number of states does (a buffer no
  // cached state holds, such as Qwen's draft rings).
  [[nodiscard]] virtual uint32_t statesToActivate() const noexcept { return 0; }
  // The most rows of one lane's prefill command: a prefill group's where
  // the model runs them (one lane's chunks, layer by layer) to read
  // something once for all of them, its streamed experts for every lane or a
  // spilled history for a lane whose K/V spills (`spills`), else a chunk's.
  [[nodiscard]] virtual uint32_t prefillGroupRows(bool spills) const noexcept {
    static_cast<void>(spills);
    return ExecutionLimits::prefillTokenBudget;
  }
  // The most passes over its layers one lane's prefill command may run in
  // (BatchPlan::passes): its layers where a command pays a cost per layer
  // whatever its rows, which keeps a command long however few they are, as
  // a streamed target's expert reads; 1 where a command runs whole. A
  // command that encodes images or streams its history runs whole.
  [[nodiscard]] virtual uint32_t prefillPasses() const noexcept { return 1; }
  // Releases one unit of idle model memory and returns its bytes; zero when
  // nothing in scope is idle. A unit is one pooled state buffer (keepLane
  // keeps those one lane starts from); with BuffersThenCaches, once no
  // buffer is idle, the vision arena when no image waits for its encode and
  // nothing holds it, the staging of spilled history once no prompt streams
  // into it, an extent of the hot block cache above its lanes' floors, and
  // then one embedding entry nothing else holds, oldest first. A denied allocation
  // retries between calls, so it frees only what it needs.
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

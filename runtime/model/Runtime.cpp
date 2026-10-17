#include "model/Runtime.hpp"
#ifdef SPLASH_RUNTIME_INSTRUMENTATION
#include "model/RuntimeInstrumentation.hpp"
#endif
#include "model/Qwen4DraftContext.hpp"
#include "model/Qwen4Exp.hpp"
#include "model/Qwen4Steps.hpp"
#include "AwakeClock.hpp"
#include "model/QwenState.hpp"
#include "model/QwenTarget.hpp"
#include "model/RuntimeArenas.hpp"
#include "model/SparseRowStore.hpp"

#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/GDN.h"
#include "metal/abi/QuantFormat.h"
#include "ops/AneFfn.hpp"
#include "ops/Linear.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/PagedKv.hpp"
#include "ops/RoPE.hpp"
#include "ops/RowCopy.hpp"
#include "ops/Sampling.hpp"
#include "ops/Vision.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <list>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace splash::model {
namespace {

using metal::BufferStorage;
using metal::CommandGraph;
using metal::CommandTicket;
using metal::CommandTiming;
using metal::MetalBackend;
using metal::MetalBuffer;

class DeferredMetalTicket final : public ModelBatchTicket {
public:
  // Runs once the command has completed, and may add work of its own to its
  // timing, as a rerun of the command does.
  using Completion = std::function<std::vector<ModelStepResult>(CommandTiming &)>;

  DeferredMetalTicket(CommandTicket ticket, Completion completion,
                      bool representativePrefillTiming = true)
      : completion_(std::move(completion)), ticket_(std::move(ticket)),
        representativePrefillTiming_(representativePrefillTiming) {}

  bool ready() const noexcept override { return ticket_.ready(); }

  std::vector<ModelStepResult> wait() override {
    if (!completion_) {
      throw std::logic_error("Metal ticket was already consumed");
    }
    CommandTiming timing = ticket_.wait();
    const double commandSeconds = timing.wallSeconds;
    Completion completion = std::move(completion_);
    std::vector<ModelStepResult> results = completion(timing);
    wallMilliseconds_ = timing.wallSeconds * 1000.0;
    hostMilliseconds_ = timing.hostSeconds * 1000.0;
    // Work the completion added takes wall time, but is no sample of the
    // command's rows.
    if (timing.wallSeconds != commandSeconds)
      representativePrefillTiming_ = false;
    return results;
  }

  double wallMilliseconds() const noexcept override {
    return wallMilliseconds_;
  }
  double hostMilliseconds() const noexcept override { return hostMilliseconds_; }
  bool prefillTimingIsRepresentative() const noexcept override {
    return representativePrefillTiming_;
  }

private:
  // Before the ticket, which goes first: a ticket dropped unconsumed waits for
  // its command before the completion lets go of what the command writes,
  // such as the state reservations of its items.
  Completion completion_;
  CommandTicket ticket_;
  double wallMilliseconds_ = 0.0;
  double hostMilliseconds_ = 0.0;
  bool representativePrefillTiming_;
};

using kv::ChunkedPrefillParams;

bool isStopToken(const RuntimeGeometry &geometry, uint32_t token) noexcept {
  return token == geometry.target.stopTokens[0] ||
         token == geometry.target.stopTokens[1];
}

void requireShared(const MetalBuffer &buffer, std::string_view label) {
  if (!buffer || buffer.storage() != BufferStorage::Shared ||
      !buffer.contents()) {
    throw std::logic_error(std::string(label) + " is not CPU-visible");
  }
}

template <class T>
T *contents(const MetalBuffer &buffer, std::string_view label) {
  requireShared(buffer, label);
  return static_cast<T *>(buffer.contents());
}

void validatePlan(const BatchPlan &plan, std::span<const ModelBatchItem> items,
                  WorkKind expected) {
  if (plan.kind != expected || plan.empty() || plan.width() > kLaneCount ||
      items.size() != plan.items.size()) {
    throw std::invalid_argument("model runtime received an invalid batch plan");
  }
  for (size_t index = 0; index < items.size(); ++index) {
    if (items[index].requestId != plan.items[index].requestId ||
        (expected == WorkKind::Prefill &&
         (!plan.items[index].tokenCount ||
          plan.items[index].tokenCount != items[index].tokenCount ||
          items[index].inputTokens.size() != items[index].tokenCount)) ||
        (expected == WorkKind::Decode &&
         (plan.items[index].tokenCount || items[index].tokenCount ||
          !items[index].inputTokens.empty()))) {
      throw std::invalid_argument("batch items do not match explicit plan");
    }
  }
}

// The lane a start's admission gave it, or the cause of its refusal.
StateAdmission laneAdmission(uint32_t lane, const metal::AllocationResult &result) {
  if (result)
    return {lane, StateFailure::None};
  return {{}, StateFailure::MemoryPressure, result.failure};
}

// Any unassigned lane works: its buffers come from the storage's pool, and
// the governor is asked only for what the pool lacks.
template <class Activate>
StateAdmission admitIdleLane(const QwenStateStorage &states,
                             Activate activate) {
  for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
    if (!states.metadata(lane).assigned())
      return activate(lane);
  }
  return {{}, StateFailure::ConcurrencyLimit};
}

// A prefill arena's linear scratch. Prefill plans read plain bf16 rows, so
// there is no input table or sums.
ops::LinearScratch prefillScratch(const PrefillArena &arena) {
  return {.partials = arena.get(PrefillTensor::LinearPartials),
          .counters = arena.get(PrefillTensor::LinearCounters),
          .rotated = arena.get(PrefillTensor::LinearRotated)};
}

// A prefill arena's tensors as the draft's context prefill of chunk `chunk`
// reads them (prefillGroupTensor).
DFlashPrefillBuffers draftPrefillBuffers(const PrefillArena &arena, uint32_t chunk) {
  return {arena.get(PrefillTensor::Captured, chunk), prefillScratch(arena),
          arena.get(PrefillTensor::ContextProjected), arena.get(PrefillTensor::ContextHidden),
          arena.get(PrefillTensor::ContextKv), arena.get(PrefillTensor::DraftRopeCos, chunk),
          arena.get(PrefillTensor::DraftRopeSin, chunk)};
}

// A prefill arena's tensors as the target's prefill reads them.
// The buffers of a prefill group's chunk `chunk`: its rows of the group's
// tensors (prefillGroupTensor), and the arena's others.
QwenTargetPrefillBuffers prefillBuffers(const PrefillArena &arena, uint32_t chunk) {
  QwenTargetPrefillBuffers buffers;
  buffers.linearScratch = prefillScratch(arena);
  buffers.hidden = {arena.get(PrefillTensor::Hidden0, chunk), arena.get(PrefillTensor::Hidden1, chunk)};
  buffers.normalized = arena.get(PrefillTensor::Normalized, chunk);
  buffers.captured = arena.get(PrefillTensor::Captured, chunk);
  buffers.gdnPacked = arena.get(PrefillTensor::GdnPacked);
  buffers.gdnQueries = arena.get(PrefillTensor::GdnQueries);
  buffers.gdnKeys = arena.get(PrefillTensor::GdnKeys);
  buffers.gdnValues = arena.get(PrefillTensor::GdnValues);
  buffers.gdnDecay = arena.get(PrefillTensor::GdnDecay);
  buffers.gdnBeta = arena.get(PrefillTensor::GdnBeta);
  buffers.recurrent = arena.get(PrefillTensor::Recurrent);
  buffers.gdnHidden = arena.get(PrefillTensor::GdnHidden);
  buffers.gdnOutput = arena.get(PrefillTensor::GdnOutput, chunk);
  buffers.denseGateScratch = arena.get(PrefillTensor::GateIntermediate);
  buffers.denseIntermediate = arena.get(PrefillTensor::Intermediate);
  buffers.fullPacked = arena.get(PrefillTensor::FullPacked);
  buffers.fullQueries = arena.get(PrefillTensor::FullQueries);
  buffers.fullAttention = arena.get(PrefillTensor::FullAttention);
  buffers.attentionPartials = arena.get(PrefillTensor::AttentionPartials);
  buffers.attentionStatistics = arena.get(PrefillTensor::AttentionStatistics);
  buffers.attentionHidden = arena.get(PrefillTensor::AttentionHidden);
  buffers.attentionOutput = arena.get(PrefillTensor::AttentionOutput, chunk);
  buffers.ropeCos = arena.get(PrefillTensor::RopeCos, chunk);
  buffers.ropeSin = arena.get(PrefillTensor::RopeSin, chunk);
  buffers.ropeInverseFrequencies = arena.get(PrefillTensor::TargetInverseFrequencies);
  buffers.chunkKeys = arena.get(PrefillTensor::ChunkKeys);
  buffers.chunkValues = arena.get(PrefillTensor::ChunkValues);
  buffers.moe = arena.moeScratch(chunk);
  return buffers;
}

} // namespace

struct Runtime::Impl {
  // An image by content: the fields a placement's span identifies it by.
  struct ImageKey final {
    uint64_t digestLo = 0;
    uint64_t digestHi = 0;
    uint32_t gridHeight = 0;
    uint32_t gridWidth = 0;

    bool operator==(const ImageKey &) const = default;
  };
  struct ImageKeyHash final {
    // The digest is already a content hash.
    size_t operator()(const ImageKey &key) const noexcept {
      return static_cast<size_t>(key.digestLo ^ key.digestHi);
    }
  };

  // One image's encoded rows, shared by every placement that still has rows
  // to inject (repeated placements and concurrent requests alike) and by
  // the embedding cache. Whichever placement's chunk reaches the image
  // first encodes it and the others inject after it; the pixels go once
  // the encode has completed.
  struct ImageRows final {
    ImageKey key;
    MetalBuffer pixels;
    MetalBuffer embeddings;
    bool encoding = false;
    bool encoded = false;
    // Its entry in the embedding cache while the cache holds it.
    std::optional<std::list<std::shared_ptr<ImageRows>>::iterator> cached;
  };

  // A placement keeps its rows until its last row is injected; its span
  // stays, because rotary positions after it depend on its grid.
  struct ImageState final {
    ImageSpan span;
    std::shared_ptr<ImageRows> rows;
  };

  struct Request final {
    uint64_t id = 0;
    uint32_t stateLane = 0;
    bool resident = false;
    bool promptComplete = false;
    // Rebuild state from already-emitted tokens without sampling an initial
    // anchor, consuming RNG, or replaying output to the caller.
    bool replayingGeneration = false;
    uint32_t promptTokens = 0;
    uint32_t maxNewTokens = 0;
    uint32_t generatedTokens = 0;
    SamplingParameters sampling;
    ConstraintMode constraint = ConstraintMode::None;
    // RequestFlag bits.
    uint32_t flags = 0;
    std::optional<uint32_t> pendingToken;
    // While the request thinks under a budget (ModelRequest::thinkingBudget),
    // that budget with the thinking tokens it may still select and, once
    // they ran out, the tokens of its end it selected; none once thinking
    // ended.
    struct Thinking final {
      ThinkingBudget budget;
      uint32_t endSelected = 0;
    };
    std::optional<Thinking> thinking;
    // A constrained request's final prompt row, held from its prompt's end
    // until its first token is selected under its first mask, suspensions
    // included. Composite cache state never stores it; every cache hit
    // replays one input token and regenerates this value.
    std::vector<uint16_t> finalTargetHidden;
    std::array<float, kSamplingUniformCount> cycleUniforms{};
    // Nonempty selects score-only mode: the final prefill chunk computes raw
    // logits at these token ids instead of selecting an anchor.
    std::vector<uint32_t> scoreTokens;
    std::vector<uint32_t> maskWords;
    // Set only while the current scheduler-owned ticket overlaps grammar-mask
    // computation with target verification. This is model runtime state, not a
    // scheduler decode stage.
    bool verifyMaskInFlight = false;
    uint64_t rngCounter = 0;
    DecodeStage decodeStage = DecodeStage::Regular;
    std::optional<DraftContextPlan> draftContextPlan;
    // The draft rings a restore left to compute from its context window:
    // positions [begin, begin + rows), in the request's next prefill chunk,
    // whose commit clears it.
    struct DraftRebuild final {
      uint32_t begin = 0;
      uint32_t rows = 0;
    };
    std::optional<DraftRebuild> draftRebuild;
    std::vector<ImageState> images;
    // What its activation took from a cached state: the images that end
    // there were left out (ModelRequest::restoredTokens).
    uint32_t restoredTokens = 0;
    // A step of it found a row that is not finite (ModelStepResult::failure):
    // its lane's draft rings may hold such rows, so the lane returns them
    // cleared (QwenStateStorage::clearDraftRings).
    bool resultInvalid = false;
    // Every token the model has read or will read before the pending one:
    // the prompt (with any replayed generation) and the emitted output, which
    // the target may read (QwenTargetRequest).
    std::vector<uint32_t> tokens;
    // The MTP head's proposals (mtpStepsFor): the steps the cycle in flight
    // drafts, and decayed counts of the proposals the target accepted and
    // of those it judged (a cycle judges its proposals up to the first it
    // rejects). The prior is a rate of 0.8 worth five proposals.
    uint32_t mtpDrafted = 0;
    float mtpAccepted = 4.0F;
    float mtpJudged = 5.0F;
    // With the head's draft vocabulary: the request's context tokens beside
    // it and the count by which it falls back to the whole head
    // (Qwen4DraftContext.hpp).
    Qwen4DraftContext draftContext;
  };

  struct DecodeLaneResult final {
    Request *request = nullptr;
    uint32_t retained = 0;
    uint32_t accepted = 0;
    uint32_t currentAnchor = 0;
    uint32_t maximumRetained = 0;
    // The draft proposals the lane's thinking end replaced, which its
    // speculation counts leave out.
    uint32_t endDrafts = 0;
    // Why the lane's selection is unusable (invalidSelection), found before
    // any lane commits.
    std::string failure;
  };

  // What a lane's GPU table was last written from. Its entries stay valid
  // while the revision does: KvPool never releases the extent of a page a
  // request holds (PageStorage::releaseExtent).
  struct PageTableBinding final {
    uint64_t requestId = 0;
    uint64_t revision = 0;
  };

  MetalBackend &backend;
  const LoadedModel &model;
  const RuntimeGeometry geometry;
  const ops::ExecutionPlans &operators;
  kv::PageStorage &kvPages;
  QwenStateStorage &states;
  std::unique_ptr<PrefillArena> prefillArena;
  std::unique_ptr<DecodeArena> decodeArena;
  // Every state lane's penalty words, bound whole: a batch lane reads the row
  // of its request's state lane, which need not be its own.
  MetalBuffer penaltyTable;
  std::unordered_map<uint64_t, Request> requests;
  // Allocated for images that need an encode, sized for the largest one the
  // start that built it staged, and reclaimable once no image waits for one
  // and no refused start holds it. Injecting already encoded rows needs no
  // vision arena.
  std::shared_ptr<ops::Vision> vision;
  // Every image's rows while anything holds them, so that a placement of
  // the same image anywhere shares them. Entries of rows nothing holds any
  // more go when a lookup or a walk finds them.
  std::unordered_map<ImageKey, std::weak_ptr<ImageRows>, ImageKeyHash> imageRows;
  // Encoded rows kept for reuse once no placement has rows of them left to
  // inject, including prefix hits that land inside an image and still need
  // its remaining rows. Most recently used first, bounded by bytes; the
  // memory reclaimer drops the least recently used entry nothing else holds.
  static constexpr uint64_t kEmbeddingCacheBytes = 512ULL * 1024 * 1024;
  std::list<std::shared_ptr<ImageRows>> embeddingCache;
  uint64_t embeddingCacheBytes = 0;
  // A state in RAM that resumes inside an image, with the rows it needs: its
  // boundary lies less than a page before the image's end, so a restore
  // there injects the image's last rows. The pointer the cache keeps owns
  // both, so the rows go with the state's RAM copy, which the cache drops
  // when it evicts the state or writes it to disk. Held rows and the
  // embedding cache together keep at most kEmbeddingCacheBytes of rows.
  struct HeldState final {
    std::shared_ptr<const CompositeState> state;
    std::shared_ptr<ImageRows> rows;
  };
  std::vector<std::weak_ptr<const HeldState>> stateHolds;
  std::array<PageTableBinding, kLaneCount> pageTableBindings{};
  ModelTelemetry counters;
  ops::Sampling sampling;
  QwenTarget targetModel;
  // Null for a target without a DFlash2 draft (LoadedModel::hasDraft), whose
  // MTP head drafts the proposals (QwenTarget::addMtpDraft).
  std::unique_ptr<DFlashDraft> draftModel;
  // Proposals the MTP head drafts per cycle: at most SPLASH_MTP_DRAFTS (1 to
  // 7, default 7), fewer while a lane's proposals are rejected
  // (mtpStepsFor); SPLASH_MTP_ADAPTIVE=0 always drafts SPLASH_MTP_DRAFTS
  // (default 3). The later proposals repeat the last.
  bool mtpAdaptive = mtpAdaptiveFromEnvironment();
  uint32_t mtpDraftSteps = mtpDraftStepsFromEnvironment(mtpAdaptive);
  static bool mtpAdaptiveFromEnvironment() {
    const char *value = std::getenv("SPLASH_MTP_ADAPTIVE");
    return !value || std::string_view(value) != "0";
  }
  static uint32_t mtpDraftStepsFromEnvironment(bool adaptive) {
    const char *value = std::getenv("SPLASH_MTP_DRAFTS");
    if (!value || !*value) return adaptive ? kDraftProposalTokens : 3;
    const long steps = std::strtol(value, nullptr, 10);
    return static_cast<uint32_t>(std::clamp<long>(steps, 1, kDraftProposalTokens));
  }
  // A step drafts proposal k, which the target keeps only when it accepts
  // the k - 1 before it too: at a lane's acceptance rate r, with chance
  // about r^k. Each proposal costs a draft step and, since the verify routes
  // only its live rows to experts (verifyLiveRows), its row's experts in the
  // verify too, so a lane drafts while r^k stays at least 0.3, and at least
  // two: all seven at r = 0.9, five at 0.8 and two at 0.6.
  static constexpr float kMtpWorthwhile = 0.3F;
  // How much of a lane's acceptance counts survives each cycle.
  static constexpr float kMtpDecay = 0.8F;
  uint32_t mtpStepsFor(const Request &entry) const {
    if (!mtpAdaptive) return mtpDraftSteps;
    const float rate = entry.mtpAccepted / entry.mtpJudged;
    uint32_t steps = 1;
    for (float reach = rate * rate; steps < mtpDraftSteps && reach >= kMtpWorthwhile; reach *= rate) ++steps;
    return std::max(steps, std::min(2U, mtpDraftSteps));
  }
  // The rows of each lane a verify after the MTP head's draft routes to
  // Qwen3.8-Flash-Next's experts: the anchor and the drafted proposals. The
  // rows past them only repeat the last proposal, so they route to no expert
  // (QwenTarget::addVerify liveRows) and no token is kept from them. 0 when
  // every row routes: a lane without drafted proposals, or a full verify.
  uint32_t verifyLiveRows(std::span<Request *const> entries) const {
    if (!targetModel.hasMtp()) return 0;
    uint32_t rows = 0;
    for (const Request *entry : entries) {
      if (!entry->mtpDrafted) return 0;
      rows = std::max(rows, entry->mtpDrafted + 1);
    }
    return rows < kDecodeRows ? rows : 0;
  }
  // The outcome of the proposals a lane's cycle drafted: `accepted`
  // proposals in a row, counted past the drafted ones when a repeat of the
  // last was accepted too.
  static void countMtpOutcome(Request &entry, uint32_t accepted) {
    if (!entry.mtpDrafted) return;
    const uint32_t kept = std::min(accepted, entry.mtpDrafted);
    entry.mtpAccepted = kMtpDecay * entry.mtpAccepted + static_cast<float>(kept);
    entry.mtpJudged = kMtpDecay * entry.mtpJudged + static_cast<float>(kept < entry.mtpDrafted ? kept + 1 : kept);
    entry.mtpDrafted = 0;
  }
  // With a draft vocabulary (Qwen4DraftVocabulary.hpp), each step of the
  // MTP head reads a quarter of the vocabulary projection at most, but a
  // token outside it and the request's context is never proposed, and each
  // such token the target emits ends its cycle's accepted run. A lane
  // drafts from them while at most SPLASH_MTP_DRAFT_VOCAB_GATE (default
  // 0.06) of its recent tokens fell outside, counted from a warm start
  // unless SPLASH_MTP_DRAFT_VOCAB_WARMSTART=0 (Qwen4DraftContext), and the
  // batch while every lane does.
  float mtpVocabularyGate = [] {
    const char *value = std::getenv("SPLASH_MTP_DRAFT_VOCAB_GATE");
    if (!value || !*value) return 0.06F;
    char *end = nullptr;
    const float gate = std::strtof(value, &end);
    if (end == value || *end || !std::isfinite(gate) || gate < 0.0F || gate > 1.0F)
      throw std::invalid_argument("SPLASH_MTP_DRAFT_VOCAB_GATE must be a fraction in [0, 1]");
    return gate;
  }();
  bool mtpVocabularyWarmStart = [] {
    const char *value = std::getenv("SPLASH_MTP_DRAFT_VOCAB_WARMSTART");
    return !value || std::string_view(value) != "0";
  }();
  Qwen4DraftContextCache<kLaneCount> draftContextCache;
  // Qwen3.8-Flash-Next's PLE table when it streams from its file
  // (--ple-ssd-cache): the verify's host step stages the rows of its rows'
  // n-grams, those of the anchor and the proposals the MTP head drafts. The
  // anchor's are read ahead as the cycle is encoded and, halfway through
  // the draft, the proposals so far are copied out and an event signalled
  // (QwenTargetMtpDraft::prefetchEvent), on which a listener hashes their
  // rows and reads them ahead while the later steps and the trunk run
  // (SparseRowStore::prefetch, a hint the stage reads past). Unless
  // SPLASH_PLE_PREFETCH=0.
  SparseRowStore *pleStore = nullptr;
  bool plePrefetch = [] {
    const char *value = std::getenv("SPLASH_PLE_PREFETCH");
    return !value || std::string_view(value) != "0";
  }();
  metal::SharedEvent plePrefetchEvent;
  uint64_t plePrefetchValue = 0;
  MetalBuffer plePrefetchTokens;
  // The listener of the last cycle's signal, whose callback reads
  // plePrefetchTokens: the next cycle's copy waits for it, as does the end.
  std::future<void> plePrefetched;
  void waitPlePrefetch() {
    if (plePrefetched.valid()) plePrefetched.get();
  }
  // The PLE rows of `tokens` from `position` on of a lane whose tokens
  // before them are `history`.
  void plePrefetchRows(uint64_t position, std::span<const uint32_t> tokens, std::span<const uint32_t> history,
                       std::vector<uint32_t> &ids) const {
    const size_t base = ids.size();
    ids.resize(base + tokens.size() * geometry.target.pleHeads());
    qwen4PleRows(*targetModel.qwen4(), position, tokens, history, ids.data() + base);
  }
  // Reads ahead the anchors' rows now and arms the listener for the
  // proposals of the draft's step `step` of the lanes' cycle.
  void armPlePrefetch(QwenTargetMtpDraft &draft, std::span<Request *const> entries,
                      std::span<const ModelBatchItem> items, uint32_t step) {
    waitPlePrefetch();
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    std::vector<uint32_t> ids;
    for (uint32_t lane = 0; lane < lanes; ++lane)
      plePrefetchRows(items[lane].logicalPosition, {&*entries[lane]->pendingToken, 1}, entries[lane]->tokens, ids);
    pleStore->prefetch(ids);
    if (!plePrefetchEvent) {
      plePrefetchEvent = backend.newSharedEvent();
      plePrefetchTokens = backend.allocateBuffer(uint64_t{kLaneCount} * kDraftProposalTokens * sizeof(uint32_t),
                                                 BufferStorage::Shared, "ple-prefetch-proposals");
    }
    draft.prefetchEvent = plePrefetchEvent;
    draft.prefetchValue = ++plePrefetchValue;
    draft.prefetchStep = step;
    draft.prefetchTokens = plePrefetchTokens;
    // Each lane's position and the tokens its proposals' n-grams reach
    // back to (its anchor included), which the listener hashes with the
    // proposals it finds in the copy.
    struct Lane final {
      uint64_t position;
      std::vector<uint32_t> history;
    };
    std::vector<Lane> laneInputs;
    const size_t reach = geometry.target.pleNgram;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const Request &entry = *entries[lane];
      const uint64_t position = items[lane].logicalPosition;
      std::vector<uint32_t> history(entry.tokens.begin() + std::min<size_t>(entry.tokens.size(), position) -
                                        std::min<size_t>(position, reach),
                                    entry.tokens.begin() + std::min<size_t>(entry.tokens.size(), position));
      history.push_back(*entry.pendingToken);
      laneInputs.push_back({position + 1, std::move(history)});
    }
    auto done = std::make_shared<std::promise<void>>();
    plePrefetched = done->get_future();
    plePrefetchEvent.notify(
        draft.prefetchValue,
        [this, lanes, step, laneInputs = std::move(laneInputs), tokens = plePrefetchTokens, done]() noexcept {
          try {
            std::vector<uint32_t> ids;
            const auto *proposals = static_cast<const uint32_t *>(tokens.contents());
            for (uint32_t lane = 0; lane < lanes; ++lane)
              plePrefetchRows(laneInputs[lane].position, {proposals + lane * kDraftProposalTokens, step},
                              laneInputs[lane].history, ids);
            pleStore->prefetch(ids);
          } catch (...) {
            // A hint: the stage reads what it lacks.
          }
          done->set_value();
        });
  }
  [[nodiscard]] bool useDraftVocabulary(std::span<Request *const> entries) const {
    return targetModel.hasMtpDraftVocabulary() && std::ranges::all_of(entries, [this](const Request *entry) {
      return entry->draftContext.useVocabulary(mtpVocabularyWarmStart, mtpVocabularyGate);
    });
  }
  void countDraftVocabulary(Request &entry, std::span<const uint32_t> tokens) const {
    if (targetModel.hasMtpDraftVocabulary()) entry.draftContext.record(tokens);
  }
  // Binds each lane's context rows (copying in those it lacks) to a draft
  // from the draft vocabulary.
  void bindDraftContext(QwenTargetVerifyBuffers &buffers, std::span<Request *const> entries) {
    const uint32_t format = geometry.target.mtpDraftContextFormat;
    if (format >= GGUF_FMT_COUNT) return;
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    const uint32_t hidden = geometry.target.hiddenSize;
    buffers.draftContextLogits = decodeArena->batchSlice(DecodeTensor::DraftContextLogits, lanes);
    buffers.draftContextLogitsSlot =
        static_cast<uint32_t>(decodeArena->get(0, DecodeTensor::DraftContextLogits).sizeBytes() / sizeof(float));
    buffers.draftContextTokens = decodeArena->batchSlice(DecodeTensor::DraftContextTokens, kLaneCount);
    buffers.draftContextInfo = decodeArena->batchSlice(DecodeTensor::DraftContextInfo, lanes);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      Request &entry = *entries[lane];
      const uint32_t stateLane = entry.stateLane;
      const uint32_t count = static_cast<uint32_t>(entry.draftContext.tokens.size());
      auto *pairs = contents<uint32_t>(decodeArena->get(stateLane, DecodeTensor::DraftContextCopies), "context copies");
      auto *tokens = contents<uint32_t>(decodeArena->get(stateLane, DecodeTensor::DraftContextTokens), "context tokens");
      const uint32_t synced = draftContextCache.bind(stateLane, entry.id, count);
      for (uint32_t row = synced; row < count; ++row) {
        pairs[2 * (row - synced)] = entry.draftContext.tokens[row];
        pairs[2 * (row - synced) + 1] = row;
        tokens[row] = entry.draftContext.tokens[row];
      }
      buffers.draftContextCopies[lane] = decodeArena->get(stateLane, DecodeTensor::DraftContextCopies);
      buffers.draftContextCopyCounts[lane] = count - synced;
      buffers.draftContextCounts[lane] = count;
      const MetalBuffer plane1 = decodeArena->get(stateLane, DecodeTensor::DraftContextPlane1);
      ops::Projection head(kQwen4DraftContextTokens, hidden,
                           ops::BlockWeights{{ops::QuantizedSegment::planes(
                               format, kQwen4DraftContextTokens, hidden,
                               decodeArena->get(stateLane, DecodeTensor::DraftContextPlane0), plane1,
                               decodeArena->get(stateLane, DecodeTensor::DraftContextMeta))}});
      head.destination = ops::FloatOutput::Float32;
      buffers.draftContextHeads[lane] = std::move(head);
      auto *info = contents<uint32_t>(decodeArena->get(lane, DecodeTensor::DraftContextInfo), "context lane");
      info[0] = count;
      info[1] = stateLane;
      info[2] = info[3] = 0;
    }
  }
  ops::AneFfn *aneFfn;
  // The prefetch listener reads buffers of this.
  ~Impl() { waitPlePrefetch(); }
  explicit Impl(RuntimeContext value)
      : backend(value.backend),
        model(value.model),
        geometry(RuntimeGeometry::from(value.model, value.kvPages.layout().format,
                                       value.stateStorage.layout().target.format)),
        operators(value.operators),
        kvPages(value.kvPages),
        states(value.stateStorage),
        sampling(geometry.target.vocabularySize),
        targetModel(std::visit(
                        [&](const auto &weights) {
                          return QwenTarget(weights, geometry.target, value.backend, operators,
                                            geometry.prefillChunks);
                        },
                        value.model.target)),
        draftModel(value.model.hasDraft()
                       ? std::make_unique<DFlashDraft>(value.model.draft, value.backend, operators)
                       : nullptr),
        aneFfn(value.aneFfn) {
    pleStore = value.model.streamedTable();
    if (states.layout() != model.stateLayout(states.layout().target.format) ||
        kvPages.layout() != model.targetKvLayout(kvPages.layout().format)) {
      throw std::invalid_argument(
          "model runtime resources do not match the loaded model");
    }
    prefillArena = std::make_unique<PrefillArena>(backend, geometry, operators);
    decodeArena = std::make_unique<DecodeArena>(backend, geometry, operators);
    penaltyTable =
        decodeArena->batchSlice(DecodeTensor::PenaltyState, kLaneCount);
    preparePolicyPipelines();
  }

  // Warmup selects greedily, so the first sampled, penalized or constrained
  // request would compile the policy's kernels inside its TTFT and stall the
  // engine meanwhile; compile them now. A sampled, penalized and constrained
  // lane whose thinking budget ran out and a greedy one reach every kernel
  // the first-token and verify selections and the drafts they verify
  // dispatch.
  void preparePolicyPipelines() const {
    const std::array<uint32_t, 1> end{};
    const ops::SamplingPolicy sampled{
        .topK = 0,
        .temperature = 1.0F,
        .topP = 0.95F,
        .constrained = true,
        .penalties = {1.1F, 0.5F, 0.5F},
        .minP = 0.05F,
        .thinkingCutoff = ops::ThinkingCutoff{.end = end}};
    const std::array<ops::SamplingPolicy, 2> policies{sampled, {}};
    const std::array<uint32_t, 2> stateLanes{0, 1};
    const ops::PenaltyTable penalties{penaltyTable, stateLanes};
    CommandGraph graph;
    sampling.addInitial(graph, policies, samplingBuffers(2), 0,
                        geometry.target.stopTokens[0],
                        geometry.target.stopTokens[1], penalties);
    sampling.addThinkingDrafts(
        graph, policies,
        decodeArena->batchSlice(DecodeTensor::ProposedTokens, 2));
    sampling.addVerify(graph, policies, samplingBuffers(2),
                       geometry.target.stopTokens[0],
                       geometry.target.stopTokens[1], penalties);
    backend.preparePipelines(graph.dispatches());
  }

  Request &request(uint64_t id) {
    auto found = requests.find(id);
    if (found == requests.end())
      throw std::out_of_range("unknown request");
    return found->second;
  }

  static bool samplingEnabled(const Request &entry) noexcept {
    return samplingPolicy(entry).samples();
  }

  // Qwen3.5 M-RoPE: text rows advance one counter shared by all three axes;
  // an image's rows spread over (t, h, w) from the counter at the image start
  // and the counter then advances by max(merged height, merged width).
  static std::array<uint32_t, 3> ropePosition(const Request &entry,
                                              uint64_t logical) {
    int64_t delta = 0;
    for (const ImageState &image : entry.images) {
      const ImageSpan &span = image.span;
      if (logical < span.offset)
        break;
      const uint32_t mergedHeight = span.gridHeight / 2;
      const uint32_t mergedWidth = span.gridWidth / 2;
      const uint32_t start =
          static_cast<uint32_t>(static_cast<int64_t>(span.offset) + delta);
      if (logical < span.end()) {
        const uint32_t local = static_cast<uint32_t>(logical - span.offset);
        return {start, start + local / mergedWidth,
                start + local % mergedWidth};
      }
      delta += static_cast<int64_t>(std::max(mergedHeight, mergedWidth)) -
               static_cast<int64_t>(span.tokens);
    }
    const uint32_t position =
        static_cast<uint32_t>(static_cast<int64_t>(logical) + delta);
    return {position, position, position};
  }

  uint64_t embeddingBytes(const ImageSpan &span) const {
    return uint64_t{ops::Vision::embeddingRows(span.grid())} *
           geometry.target.hiddenSize * sizeof(uint16_t);
  }

  static ImageKey imageKey(const ImageSpan &span) noexcept {
    return {span.digestLo, span.digestHi, span.gridHeight, span.gridWidth};
  }

  // The rows of an identical image that something still holds, moved to the
  // front of the embedding cache when it is there; null otherwise.
  std::shared_ptr<ImageRows> findRows(const ImageSpan &span) {
    const auto found = imageRows.find(imageKey(span));
    if (found == imageRows.end())
      return {};
    std::shared_ptr<ImageRows> rows = found->second.lock();
    if (!rows) {
      imageRows.erase(found);
      return {};
    }
    if (rows->cached)
      embeddingCache.splice(embeddingCache.begin(), embeddingCache, *rows->cached);
    return rows;
  }

  // Keeps encoded rows for reuse as the most recently used, dropping the
  // least recently used while the rows kept for reuse exceed the cache's
  // bytes.
  void retain(const std::shared_ptr<ImageRows> &rows) {
    if (rows->cached) {
      embeddingCache.splice(embeddingCache.begin(), embeddingCache, *rows->cached);
      return;
    }
    const uint64_t bytes = rows->embeddings.sizeBytes();
    embeddingCache.push_front(rows);
    rows->cached = embeddingCache.begin();
    embeddingCacheBytes += bytes;
    while (!embeddingCache.empty() &&
           embeddingCacheBytes + heldRowsBytes(true) > kEmbeddingCacheBytes)
      static_cast<void>(uncache(std::prev(embeddingCache.end())));
  }

  // The bytes of the distinct rows states in RAM hold: all of them, or only
  // those the embedding cache does not hold as well.
  [[nodiscard]] uint64_t heldRowsBytes(bool uncachedOnly) const noexcept {
    uint64_t bytes = 0;
    for (auto hold = stateHolds.begin(); hold != stateHolds.end(); ++hold) {
      const std::shared_ptr<const HeldState> held = hold->lock();
      if (!held || (uncachedOnly && held->rows->cached))
        continue;
      const bool counted = std::any_of(
          stateHolds.begin(), hold, [&](const std::weak_ptr<const HeldState> &earlier) {
            const std::shared_ptr<const HeldState> other = earlier.lock();
            return other && other->rows == held->rows;
          });
      if (!counted)
        bytes += held->rows->embeddings.sizeBytes();
    }
    return bytes;
  }

  // A state in RAM whose boundary lies inside an image, less than a page
  // before its end, holds the image's encoded rows, unless that would take
  // the rows kept for reuse past the cache's bytes: the state returned owns
  // them. Boundaries deeper inside an image keep only the embedding cache.
  std::shared_ptr<const CompositeState>
  holdStraddledRows(const Request &entry, uint64_t boundary,
                    std::shared_ptr<const CompositeState> state) {
    std::erase_if(stateHolds, [](const std::weak_ptr<const HeldState> &hold) {
      return hold.expired();
    });
    for (const ImageState &image : entry.images) {
      // The command that reached the boundary encoded the image, and a
      // command's states are held before it releases the rows of the images
      // it finished.
      if (image.span.offset >= boundary || image.span.end() <= boundary)
        continue;
      if (image.span.end() - boundary >= kv::kPageTokens)
        break;
      const bool kept =
          image.rows->cached ||
          std::ranges::any_of(stateHolds, [&](const std::weak_ptr<const HeldState> &hold) {
            const std::shared_ptr<const HeldState> held = hold.lock();
            return held && held->rows == image.rows;
          });
      const uint64_t added = kept ? 0 : image.rows->embeddings.sizeBytes();
      if (embeddingCacheBytes + heldRowsBytes(true) + added > kEmbeddingCacheBytes)
        break;
      auto held = std::make_shared<const HeldState>(HeldState{std::move(state), image.rows});
      stateHolds.push_back(held);
      return {held, held->state.get()};
    }
    return state;
  }

  // Drops one entry of the embedding cache and returns the bytes it held.
  uint64_t uncache(std::list<std::shared_ptr<ImageRows>>::iterator entry) noexcept {
    const uint64_t bytes = (*entry)->embeddings.sizeBytes();
    (*entry)->cached.reset();
    embeddingCache.erase(entry);
    embeddingCacheBytes -= bytes;
    return bytes;
  }

  // A resident request returns its lane's buffers to the pool.
  void releaseLane(Request &entry) {
    // A target without a DFlash2 draft writes no rings.
    if (entry.resultInvalid && draftModel)
      states.clearDraftRings(entry.stateLane);
    states.releaseLane(entry.stateLane, entry.id);
    pageTableBindings[entry.stateLane] = {};
    draftContextCache.release(entry.stateLane);
  }

  // A request lets go of its images; the encoded ones stay in the cache.
  void releaseImages(Request &entry) {
    for (const ImageState &image : entry.images) {
      if (image.rows && image.rows->encoded)
        retain(image.rows);
    }
    entry.images.clear();
  }

  // Frees one cache that can be rebuilt and returns its bytes. The vision
  // arena goes first, when no image waits for its encode and nothing holds
  // it: an image whose rows are encoded never needs it, and the next start
  // that does builds one sized for its own images. Then the least recently
  // used embedding entry nothing else holds, one at a time, since only an
  // encode rebuilds it. An entry something else holds is skipped, since
  // dropping it frees nothing.
  uint64_t releaseOneCache() noexcept {
    if (vision && vision.use_count() == 1 && visionIdle()) {
      const uint64_t bytes = vision->arenaBytes();
      vision.reset();
      return bytes;
    }
    for (auto entry = embeddingCache.end(); entry != embeddingCache.begin();) {
      if ((--entry)->use_count() == 1)
        return uncache(entry);
    }
    return 0;
  }

  // Puts back the encoder a start replaced, or drops the one it built,
  // unless the start completes: an admission granted it, but a later step of
  // the start threw.
  struct VisionRollback final {
    Impl &runtime;
    std::shared_ptr<ops::Vision> previous;
    bool committed = false;
    ~VisionRollback() {
      if (!committed)
        runtime.vision = std::move(previous);
    }
  };

  // What a refused start matched (StateAdmission::held): the rows it would
  // share and, when an image still needs its encode and the live encoder
  // covers it, that encoder.
  struct Matched final {
    std::vector<std::shared_ptr<ImageRows>> rows;
    std::shared_ptr<ops::Vision> encoder;
  };

  // A request's lane with everything else its start allocates, in one
  // admission: the pixel and embedding buffers of the images nothing holds
  // yet and, when an image still needs an encode that the live encoder does
  // not cover, a vision scratch sized for the largest such image. That
  // encoder replaces the live one, which covers fewer patches, so every
  // image waiting on the old one fits the new; a command in flight keeps the
  // old arena until it completes. Rows something holds are shared, encoded
  // or not. Images the restored prefix covers are left out: only their
  // spans are kept. At the budget the engine retries a denied start after
  // each reclaim step, and a denial builds nothing, so no encoder arena,
  // image buffer or lane state is built and dropped every time. The refusal
  // keeps its cause and holds what it matched, so the reclaim before the
  // retry spares it; a grant hands the request's images to `images` and
  // counts the rows it shares as reuses, each once.
  StateAdmission activate(const ModelRequest &request, uint32_t stateLane,
                          std::vector<ImageState> &images) {
    if (request.images.empty())
      return laneAdmission(stateLane, states.tryActivateLane(stateLane, request.id));
    // The engine rejects image requests at submission when there is no vision.
    if (!model.descriptor.hasVision())
      throw std::logic_error("image request reached a model without vision");
    std::vector<ImageState> staged;
    staged.reserve(request.images.size());
    std::vector<std::shared_ptr<ImageRows>> shared;
    uint64_t bytes = 0;
    // The patches of the largest staged image that still needs its encode.
    uint32_t encodePatches = 0;
    for (const ImageSpan &span : request.images) {
      if (span.end() <= request.restoredTokens) {
        staged.push_back({span, nullptr});
        continue;
      }
      // New rows enter the registry now, so a repeated placement shares
      // them; they have no buffers until the admission allocates them.
      std::shared_ptr<ImageRows> rows = findRows(span);
      if (!rows) {
        rows = std::make_shared<ImageRows>();
        rows->key = imageKey(span);
        imageRows.insert_or_assign(rows->key, rows);
        bytes += span.pixelBytes() + embeddingBytes(span);
      } else if (rows->embeddings && std::ranges::find(shared, rows) == shared.end()) {
        shared.push_back(rows);
      }
      if (!rows->encoded)
        encodePatches = std::max(encodePatches, span.gridHeight * span.gridWidth);
      staged.push_back({span, std::move(rows)});
    }
    const uint64_t encoderBytes =
        encodePatches && !(vision && vision->maximumPatches() >= encodePatches)
            ? ops::Vision::scratchBytes(model.vision.tensors.layout, encodePatches)
            : 0;
    std::shared_ptr<ops::Vision> encoder;
    const uint8_t *pixels = request.imagePixels.data();
    const auto allocate = [&] {
      if (encoderBytes) {
        encoder = std::make_shared<ops::Vision>(
            backend, model.vision.tensors, encodePatches);
      }
      for (ImageState &image : staged) {
        const ImageSpan &span = image.span;
        if (image.rows && !image.rows->embeddings) {
          ImageRows &rows = *image.rows;
          rows.pixels = backend.allocateBuffer(
              span.pixelBytes(), BufferStorage::Shared, "image pixels");
          std::memcpy(contents<uint8_t>(rows.pixels, "image pixels"), pixels,
                      static_cast<size_t>(span.pixelBytes()));
          rows.embeddings = backend.allocateBuffer(
              embeddingBytes(span), BufferStorage::Private, "image embeddings");
        }
        pixels += span.pixelBytes();
      }
    };
    StateAdmission admission = laneAdmission(
        stateLane, states.tryActivateLane(stateLane, request.id, encoderBytes + bytes, allocate));
    if (!admission.granted()) {
      admission.held = std::make_shared<const Matched>(
          Matched{std::move(shared), encodePatches && !encoderBytes ? vision : nullptr});
      return admission;
    }
    if (encoder)
      vision = std::move(encoder);
    counters.imageEmbeddingReuses += shared.size();
    images = std::move(staged);
    return admission;
  }

  // No image waits for its encode, so the vision arena can go.
  [[nodiscard]] bool visionIdle() noexcept {
    for (auto entry = imageRows.begin(); entry != imageRows.end();) {
      const std::shared_ptr<ImageRows> rows = entry->second.lock();
      if (!rows) {
        entry = imageRows.erase(entry);
        continue;
      }
      if (!rows->encoded)
        return false;
      ++entry;
    }
    return true;
  }

  // Encodes every image whose rows first appear in this chunk and overwrites
  // the chunk's placeholder embedding rows, the item's from row `rowBegin` of
  // `hidden` on, with the image rows. Text-only requests add no dispatches.
  void addImageRows(CommandGraph &graph, Request &entry, const ModelBatchItem &item, const MetalBuffer &hidden,
                    uint32_t rowBegin) {
    const uint64_t chunkBegin = item.logicalPosition;
    const uint64_t chunkEnd = chunkBegin + item.tokenCount;
    for (ImageState &image : entry.images) {
      const uint64_t begin = std::max<uint64_t>(chunkBegin, image.span.offset);
      const uint64_t end = std::min<uint64_t>(chunkEnd, image.span.end());
      if (begin >= end)
        continue;
      if (!image.rows)
        throw std::logic_error("prefill reached an image its activation left out");
      ImageRows &rows = *image.rows;
      if (!rows.encoded && !rows.encoding) {
        if (!vision)
          throw std::logic_error("image request has no vision encoder");
        vision->encode(graph, image.span.grid(), rows.pixels, rows.embeddings);
        rows.encoding = true;
        ++counters.imageEncodes;
      }
      const uint32_t width = model.vision.tensors.layout.outputHiddenSize;
      ops::RowCopy::add(
          graph, rows.embeddings,
          {static_cast<uint32_t>(begin - image.span.offset), width, 0}, hidden,
          {rowBegin + static_cast<uint32_t>(begin - chunkBegin), width, 0},
          static_cast<uint32_t>(end - begin), width);
    }
  }

  static float nextUniform(Request &entry) noexcept {
    uint64_t value =
        entry.sampling.seed + (++entry.rngCounter) * 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return float(value >> 40) * 0x1p-24F;
  }

  static void stageSamplingCycle(Request &entry) noexcept {
    entry.cycleUniforms.fill(0.0F);
    for (uint32_t index = SPLASH_UNIFORM_PROPOSALS;
         index < SPLASH_SAMPLING_UNIFORMS; ++index) {
      entry.cycleUniforms[index] = nextUniform(entry);
    }
  }

  [[nodiscard]] MetalBuffer synchronizedPageTable(Request &entry,
                                                  const ModelBatchItem &item) {
    if (entry.stateLane >= pageTableBindings.size())
      throw std::out_of_range("request state lane is outside page tables");
    if (item.pageTable.empty() ||
        item.pageTable.size() > kMaximumPageTableEntries) {
      throw std::invalid_argument("request page table has invalid length");
    }
    if (!item.pageTableRevision)
      throw std::invalid_argument("request page table has no revision");
    PageTableBinding &binding = pageTableBindings[entry.stateLane];
    MetalBuffer destination =
        decodeArena->get(entry.stateLane, DecodeTensor::PageTable);
    // Rewrite only what changed since the table was written: nothing at the
    // same revision, the entries from the first changed page on at the next
    // one, and everything after two changes or for another request.
    const auto size = static_cast<uint32_t>(item.pageTable.size());
    uint32_t first = 0;
    if (binding.requestId == entry.id) {
      if (binding.revision == item.pageTableRevision)
        first = size;
      else if (binding.revision + 1 == item.pageTableRevision)
        first = std::min(item.pageTableFirstChanged, size);
    }
    if (first < size)
      kvPages.writeEntries(item.pageTable, first, destination);
    binding = {entry.id, item.pageTableRevision};
    return destination;
  }

  void addRopeTables(CommandGraph &graph, MetalBuffer targetPositions,
                     uint32_t targetRows, MetalBuffer draftPositions,
                     uint32_t draftRows, MetalBuffer targetCos,
                     MetalBuffer targetSin, MetalBuffer draftCos,
                     MetalBuffer draftSin) const {
    ops::RoPE::addTables(
        graph, std::move(targetPositions), std::move(draftPositions),
        prefillArena->get(PrefillTensor::TargetInverseFrequencies),
        prefillArena->get(PrefillTensor::DraftInverseFrequencies),
        std::move(targetCos), std::move(targetSin), std::move(draftCos),
        std::move(draftSin), {targetRows, draftModel ? draftRows : 0u}, kPrefillRows);
  }

  // A constrained lane keeps its final prompt row, which prefill leaves at
  // row 0 of its Hidden0 block, until its first mask arrives.
  void captureFinalHidden(Request &entry, uint32_t lane) const {
    const uint16_t *source =
        contents<uint16_t>(decodeArena->get(lane, DecodeTensor::Hidden0),
                           "target final hidden source");
    entry.finalTargetHidden.assign(source,
                                   source + geometry.target.hiddenSize);
  }

  static DispatchDraftCapturePlan
  activeDraftCaptures(const Request &entry, const ModelBatchItem &item) {
    if (!entry.draftContextPlan) {
      throw std::logic_error("prefill request has no draft context plan");
    }
    const uint64_t next = item.logicalPosition + item.tokenCount;
    return draftCaptureSpansForDispatch(
        *entry.draftContextPlan, static_cast<uint32_t>(item.logicalPosition),
        static_cast<uint32_t>(next));
  }

  // The rows of a dispatch's draft capture spans (a DispatchDraftCapturePlan,
  // or a prefill group's, which runs several dispatches' worth).
  template <class Spans> static uint32_t captureRows(const Spans &captures) {
    uint32_t rows = 0;
    for (const auto &capture : captures)
      rows += capture.absoluteEnd - capture.absoluteBegin;
    return rows;
  }

  // The lengths after the draft ring takes rows [begin, end) at target
  // length targetTokens. Unless `reset` starts a new window there, the rows
  // continue the ring, which must hold rows ending at `begin`. A prefill
  // capture also keeps the rows' context in the window (`windowed`), which
  // stays current if it was; decode commits leave it behind.
  static QwenLogicalLengths
  advanceDraftContext(const QwenLogicalLengths &previous, uint64_t targetTokens,
                      uint64_t begin, uint64_t end, bool reset, bool windowed) {
    if (!reset && (!previous.draftLength || previous.draftEnd() != begin))
      throw std::logic_error("draft capture does not continue the draft ring");
    const uint64_t combined = (reset ? 0 : previous.draftLength) + (end - begin);
    QwenLogicalLengths next = previous;
    next.targetTokens = targetTokens;
    next.draftLength =
        static_cast<uint32_t>(std::min<uint64_t>(combined, kDraftCacheStride));
    next.draftBase = end - next.draftLength;
    if (windowed && (reset || previous.hasCurrentContextWindow()))
      next.windowEnd = end;
    return next;
  }

  // Only a sampled lane's draws read its cycle's uniforms.
  void uploadSamplingUniforms(const Request &entry, uint32_t lane) const {
    std::copy(entry.cycleUniforms.begin(), entry.cycleUniforms.end(),
              contents<float>(
                  decodeArena->get(lane, DecodeTensor::SamplingUniforms),
                  "sampling uniforms"));
  }

  // Only a constrained lane's selections read its mask rows.
  std::span<uint32_t> constraintMasks(uint32_t lane) const {
    const MetalBuffer masks =
        decodeArena->get(lane, DecodeTensor::ConstraintMasks);
    return {contents<uint32_t>(masks, "constraint masks"),
            masks.sizeBytes() / sizeof(uint32_t)};
  }

  void uploadConstraintMasks(uint32_t lane,
                             std::span<const uint32_t> masks) const {
    const std::span<uint32_t> rows = constraintMasks(lane);
    if (masks.size() > rows.size())
      throw std::invalid_argument("constraint mask exceeds decode arena");
    std::ranges::copy(masks, rows.begin());
  }

  // A constrained lane whose mask was abandoned admits every token.
  void admitEveryToken(uint32_t lane) const {
    std::ranges::fill(constraintMasks(lane),
                      std::numeric_limits<uint32_t>::max());
  }

  static ops::SamplingPenalties samplingPenalties(const Request &entry) noexcept {
    return {entry.sampling.repetitionPenalty, entry.sampling.presencePenalty,
            entry.sampling.frequencyPenalty};
  }

  // A constrained request's grammar admits the end of its thinking only where
  // the thinking text may end, such as between whole characters. Until the
  // end starts, it waits for a row whose mask admits its first token, and
  // the request thinks past its budget meanwhile: a few tokens, such as the
  // rest of a character the budget's last token began. The end's other
  // tokens are text, which the grammar admits after its first.
  static bool thinkingEndWaits(const Request &entry) noexcept {
    return entry.constraint == ConstraintMode::TokenMask &&
           !entry.thinking->endSelected;
  }

  // A selection's rows select the request's next tokens in order, so with n
  // thinking tokens left, its row n starts the budget's end, and once none
  // are left, its row 0 goes on with the end's tokens not yet selected; an
  // end that waits (thinkingEndWaits) starts at the first of those rows
  // whose mask admits it.
  static ops::SamplingPolicy samplingPolicy(const Request &entry) noexcept {
    std::optional<ops::ThinkingCutoff> cutoff;
    if (entry.thinking) {
      const ThinkingBudget &budget = entry.thinking->budget;
      cutoff = ops::ThinkingCutoff{
          budget.tokens,
          std::span(budget.end).subspan(entry.thinking->endSelected),
          budget.alsoEnds, thinkingEndWaits(entry)};
    }
    return {entry.sampling.topK, entry.sampling.temperature,
            entry.sampling.topP, entry.constraint == ConstraintMode::TokenMask,
            (entry.flags & RequestIgnoreEndOfSequence) != 0,
            samplingPenalties(entry), entry.sampling.minP, cutoff};
  }

  ops::SamplingBuffers samplingBuffers(uint32_t lanes) const {
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->batchSlice(tensor, lanes);
    };
    return {d(DecodeTensor::Logits),
            d(DecodeTensor::TargetPartialMasses),
            d(DecodeTensor::TargetVocabularyRows),
            d(DecodeTensor::SamplingUniforms),
            d(DecodeTensor::ConstraintMasks),
            d(DecodeTensor::OutputTokens),
            d(DecodeTensor::ArgmaxValues),
            d(DecodeTensor::ArgmaxIndices),
            d(DecodeTensor::InputTokens),
            d(DecodeTensor::Candidates),
            d(DecodeTensor::ProposalProbs),
            d(DecodeTensor::TargetVocabularyRanges),
            d(DecodeTensor::TargetVocabularyArrivals)};
  }

  std::span<uint32_t> penaltyWords(uint32_t stateLane) const {
    return {contents<uint32_t>(
                decodeArena->get(stateLane, DecodeTensor::PenaltyState),
                "penalty words"),
            geometry.target.vocabularySize};
  }

  // Rebuilds a penalized request's penalty words when it takes a state lane,
  // at activation and at resume, from the history the lane's prefill
  // consumes. No command reads the lane's words yet.
  void bindPenalties(const Request &entry,
                     std::span<const uint32_t> history) const {
    const ops::SamplingPenalties penalties = samplingPenalties(entry);
    if (!penalties.active())
      return;
    ops::Sampling::rebuildPenaltyWords(penaltyWords(entry.stateLane), history,
                                       entry.generatedTokens,
                                       entry.pendingToken,
                                       penalties.repetition != 1.0F);
  }

  // The one place a token the target selected becomes the pending anchor:
  // tokens are one step's selections in order, the new anchor last. The
  // command that selected them has completed, and the next one that reads
  // the lane's words is encoded after this. Each token counts against the
  // thinking budget until one ends thinking, and once the budget ran out,
  // it is the end's next token, or while the end waits (thinkingEndWaits),
  // a thinking token past the budget; the cutoff (samplingPolicy) never lets
  // thinking run past it otherwise.
  void commitSelected(Request &entry, std::span<const uint32_t> tokens) {
    if (tokens.empty())
      throw std::logic_error("no selected token to commit");
    if (samplingPenalties(entry).active())
      ops::Sampling::countPenaltyTokens(penaltyWords(entry.stateLane), tokens);
    for (const uint32_t token : tokens) {
      if (!entry.thinking)
        break;
      ThinkingBudget &budget = entry.thinking->budget;
      if (token == budget.end.back() || token == budget.alsoEnds)
        entry.thinking.reset();
      else if (budget.tokens)
        --budget.tokens;
      else if (token == budget.end[entry.thinking->endSelected])
        ++entry.thinking->endSelected;
      else if (!thinkingEndWaits(entry))
        throw std::logic_error("thinking went past its budget");
    }
    // The previous anchor and every selection before the new one are now
    // read (Request::tokens).
    if (entry.pendingToken) entry.tokens.push_back(*entry.pendingToken);
    entry.tokens.insert(entry.tokens.end(), tokens.begin(), tokens.end() - 1);
    entry.pendingToken = tokens.back();
  }

  // A selection outside the vocabulary is the sampling kernels' sentinel for a
  // logit row with no selectable token, its admitted logits all NaN or -inf: a
  // numerical outcome of this request, which it reports as its lane failure
  // (ModelStepResult::failure) so the batch survives. The message counts the
  // row's non-finite logits, which the lane's completed command left in its
  // logits rows (row k selected token k).
  [[nodiscard]] std::string invalidSelection(std::span<const uint32_t> tokens,
                                             uint32_t lane) const {
    const auto found = std::ranges::find_if(tokens, [&](uint32_t token) {
      return token >= geometry.target.vocabularySize;
    });
    if (found == tokens.end())
      return {};
    const uint32_t vocabulary = geometry.target.vocabularySize;
    const std::span<const float> row(
        contents<float>(decodeArena->get(lane, DecodeTensor::Logits),
                        "unselectable logits") +
            uint64_t(found - tokens.begin()) * vocabulary,
        vocabulary);
    uint32_t nan = 0, positive = 0, negative = 0;
    for (const float logit : row) {
      nan += std::isnan(logit);
      positive += logit == std::numeric_limits<float>::infinity();
      negative += logit == -std::numeric_limits<float>::infinity();
    }
    return "target logits had no selectable token: " + std::to_string(nan) +
           " NaN, " + std::to_string(positive) + " +inf and " +
           std::to_string(negative) + " -inf of " + std::to_string(vocabulary);
  }

  // A lane of an initial selection, whose final prompt row is at row 0 of
  // its Hidden0 block: one that selects its first token, or a score lane,
  // which needs only the logits.
  struct InitialSelection final {
    Request *entry = nullptr;
    uint32_t lane = 0;
    bool select = false;
  };

  // A sampled lane draws its first token with the first uniform of a fresh
  // cycle.
  void uploadInitialUniform(Request &entry, uint32_t lane) const {
    entry.cycleUniforms.fill(0.0F);
    entry.cycleUniforms[SPLASH_UNIFORM_INITIAL] = nextUniform(entry);
    uploadSamplingUniforms(entry, lane);
  }

  // One LM head over the batch's `width` lanes, then one selection of the
  // first token of every selecting lane from its logits row 0, which
  // initialToken() reads. The head computes, and nothing reads, the other
  // rows of each lane and the lanes not listed; a lane that does not select
  // takes the argmax of its row. Sampled lanes' uniforms and constrained
  // lanes' masks are uploaded first.
  void encodeInitialSelections(CommandGraph &graph,
                               std::span<const InitialSelection> lanes,
                               uint32_t width) {
    const uint32_t storage = targetModel.decodeStorageLanes(width);
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->batchSlice(tensor, storage);
    };
    targetModel.addHeadBatch(graph, d(DecodeTensor::Hidden0),
                             d(DecodeTensor::FinalHidden),
                             d(DecodeTensor::Logits), width,
                             decodeArena->linearScratch());
    if (std::ranges::none_of(lanes, &InitialSelection::select))
      return;
    std::array<ops::SamplingPolicy, kLaneCount> policies{};
    std::array<uint32_t, kLaneCount> stateLanes{};
    for (const InitialSelection &lane : lanes) {
      if (!lane.select)
        continue;
      policies[lane.lane] = samplingPolicy(*lane.entry);
      stateLanes[lane.lane] = lane.entry->stateLane;
    }
    sampling.addInitial(graph, std::span(policies).first(width),
                        samplingBuffers(width), 0,
                        geometry.target.stopTokens[0],
                        geometry.target.stopTokens[1],
                        {penaltyTable, std::span(stateLanes).first(width)});
  }

  // The first token encodeInitialSelections() selected for a batch lane: a
  // selection writes one output token per lane, in lane order.
  uint32_t initialToken(uint32_t lane) const {
    return contents<uint32_t>(
        decodeArena->batchSlice(DecodeTensor::OutputTokens, lane + 1),
        "initial tokens")[lane];
  }

  // Selects the first token of each constrained lane of the plan under the
  // mask it was given, from its final prompt row, in one command. The lanes
  // draft and verify from their next plan on.
  std::unique_ptr<ModelBatchTicket>
  submitInitialSelection(std::span<const ModelBatchItem> items,
                         std::function<void()> completion) {
    const uint32_t width = static_cast<uint32_t>(items.size());
    std::array<InitialSelection, kLaneCount> selections{};
    for (uint32_t lane = 0; lane < width; ++lane) {
      Request &entry = request(items[lane].requestId);
      if (entry.decodeStage != DecodeStage::ApplyInitialMask ||
          entry.pendingToken ||
          entry.maskWords.size() != geometry.maskWords() ||
          entry.finalTargetHidden.size() != geometry.target.hiddenSize) {
        throw std::logic_error("initial selection state is invalid");
      }
      std::ranges::copy(entry.finalTargetHidden,
                        contents<uint16_t>(
                            decodeArena->get(lane, DecodeTensor::Hidden0),
                            "final prompt hidden"));
      if (samplingEnabled(entry))
        uploadInitialUniform(entry, lane);
      uploadConstraintMasks(lane, entry.maskWords);
      selections[lane] = {&entry, lane, true};
    }
    CommandGraph graph;
    encodeInitialSelections(graph, std::span(selections).first(width), width);
    CommandTicket command =
        backend.submitCommandAsync(graph.command(), std::move(completion));
    auto finish = [this, selections, width](CommandTiming) {
      std::vector<ModelStepResult> results;
      results.reserve(width);
      for (const InitialSelection &selection :
           std::span(selections).first(width)) {
        Request &entry = *selection.entry;
        ModelStepResult &result = results.emplace_back();
        result.requestId = entry.id;
        const uint32_t token = initialToken(selection.lane);
        // A failed selection leaves no anchor: the engine ends the request
        // before any output, and the other lanes go on.
        result.failure = invalidSelection({&token, 1}, selection.lane);
        if (!result.failure.empty()) {
          entry.resultInvalid = true;
          continue;
        }
        commitSelected(entry, {&token, 1});
        entry.maskWords.clear();
        entry.finalTargetHidden.clear();
        entry.decodeStage = DecodeStage::Regular;
        emitTerminalAnchor(entry, result);
      }
      return results;
    };
    return std::make_unique<DeferredMetalTicket>(std::move(command),
                                                 std::move(finish));
  }

  ChunkedPrefillParams chunkParams(uint64_t logicalPosition,
                                   uint32_t chunkTokens, uint32_t chunkStride,
                                   std::span<const uint32_t> pages) const {
    return ops::PagedAttention::prefillParams(
        logicalPosition, chunkTokens, chunkStride,
        static_cast<uint32_t>(pages.size()));
  }

  // A state a lane's prefill command captures
  // (ModelBatchItem::stateCaptures): the buffers it is written to, which the
  // command's items hold, its row among the lane's rows in its chunk, whether
  // it lies at the end of the lane's rows in the command, where the
  // completion copies the lane's state rather than the command writing it,
  // and the lengths it is sealed with.
  struct PrefillStateCapture final {
    QwenStateReservation *reservation = nullptr;
    uint32_t row = 0;
    bool atEnd = false;
    QwenLogicalLengths lengths;
  };

  // A lane's rows in a chunk of a prefill command (PrefillChunk): `part`, the
  // rows of the lane's item the chunk runs, with the states they capture;
  // where they sit among the chunk's rows and in its attention scratch; the
  // lane's page table; the draft context rows they capture (from
  // captureBegin on among the chunk's) and their states.
  struct PrefillSequence final {
    Request *entry = nullptr;
    ModelBatchItem part;
    uint32_t lane = 0;
    uint32_t rowBegin = 0;
    uint32_t attentionStride = 0;
    uint64_t queryOffset = 0;
    uint64_t kvOffset = 0;
    uint32_t captureBegin = 0;
    ChunkedPrefillParams chunk;
    MetalBuffer pageTable;
    DispatchDraftCapturePlan captures;
    std::vector<PrefillStateCapture> stateCaptures;
  };

  // A chunk of a prefill command: at most kPrefillRows rows, its lanes' in
  // lane order, which a command of them alone would run. A command of more
  // rows runs its chunks as a prefill group (QwenTarget::addPrefill), each
  // in its own slot of the prefill arena.
  struct PrefillChunk final {
    std::vector<PrefillSequence> sequences;
    uint32_t rows = 0;
    uint32_t capturedRows = 0;
  };

  MetalBuffer prefillU16(const MetalBuffer &tensor, uint32_t begin,
                         uint32_t rows, uint32_t width) const {
    return backend.view(tensor, bytesFor<uint16_t>(uint64_t{begin} * width),
                        bytesFor<uint16_t>(uint64_t{rows} * width));
  }

  // The states a lane's command captures, ascending in (logicalPosition,
  // logicalPosition + tokenCount]. Each one's lengths fold the command's
  // draft captures below it into the lane's, as a command ending there would
  // commit them, and must be a cached state's: page-aligned with the whole
  // draft window, which a plan that does not give one fails here, before
  // anything is submitted.
  std::vector<PrefillStateCapture>
  stateCaptures(const ModelBatchItem &item, const DispatchDraftCapturePlan &draftCaptures,
                const QwenLogicalLengths &lengths) const {
    std::vector<PrefillStateCapture> result;
    result.reserve(item.stateCaptures.size());
    const uint64_t end = item.logicalPosition + item.tokenCount;
    uint64_t previous = item.logicalPosition;
    for (const StateCapture &capture : item.stateCaptures) {
      auto *reservation = dynamic_cast<QwenStateReservation *>(capture.reservation.get());
      if (!reservation || capture.tokens <= previous || capture.tokens > end)
        throw std::invalid_argument("invalid prefill state capture");
      QwenLogicalLengths captured = lengths;
      captured.targetTokens = capture.tokens;
      for (const DispatchDraftCaptureSpan &span : draftCaptures)
        if (span.absoluteBegin < capture.tokens)
          captured = advanceDraftContext(captured, capture.tokens, span.absoluteBegin,
                                         std::min(span.absoluteEnd, capture.tokens),
                                         span.resetDraftState, true);
      states.validateLengths(captured, true);
      result.push_back({reservation, static_cast<uint32_t>(capture.tokens - item.logicalPosition),
                        capture.tokens == end, captured});
      previous = capture.tokens;
    }
    return result;
  }

  // The states a lane's committed command captured, in order: one at the
  // command's end takes the lane's new cell and window, as snapshot() does,
  // and the others the command wrote. Each holds its straddled image rows
  // before the command releases the rows of the images it finished.
  std::vector<std::shared_ptr<const CompositeState>>
  sealStates(const Request &entry, std::span<const PrefillStateCapture> captures) {
    std::vector<std::shared_ptr<const CompositeState>> sealed;
    sealed.reserve(captures.size());
    for (const PrefillStateCapture &capture : captures) {
      if (capture.atEnd)
        states.copyState(entry.stateLane, *capture.reservation);
      sealed.push_back(holdStraddledRows(entry, capture.lengths.targetTokens,
                                         states.seal(*capture.reservation, capture.lengths)));
    }
    return sealed;
  }

  // The lane of a prefill command's item, as the command reads it: resident,
  // the item inside its prompt, and the lane's state exactly at the item's
  // start in its current cell, which a decode step leaves without the rows
  // it retained. A cold start clears the lane first.
  Request &prefillLane(const ModelBatchItem &item) {
    Request &entry = request(item.requestId);
    if (item.logicalPosition > entry.promptTokens || item.tokenCount > entry.promptTokens - item.logicalPosition ||
        !entry.resident)
      throw std::invalid_argument("invalid Qwen prefill item");
    const QwenLaneMetadata &metadata = states.metadata(entry.stateLane);
    if (metadata.requestId != entry.id || metadata.lengths.targetTokens != item.logicalPosition)
      throw std::logic_error("prefill state length is not exact");
    if (metadata.pendingRows) throw std::logic_error("prefill lane has decoded rows pending");
    if (item.logicalPosition == 0) states.clearForColdStart(entry.stateLane);
    return entry;
  }

  // An item's rows of the prefill arena's inputs from row `row` (of chunk
  // `chunk`'s rows, in a prefill group): its tokens and their RoPE positions.
  void writePromptRows(const Request &entry, const ModelBatchItem &item, uint32_t row, uint32_t chunk = 0) {
    auto *tokens = contents<uint32_t>(prefillArena->get(PrefillTensor::InputTokens, chunk), "prefill input tokens");
    auto *positions =
        contents<uint32_t>(prefillArena->get(PrefillTensor::TargetPositions, chunk), "target RoPE positions");
    for (uint32_t index = 0; index < item.tokenCount; ++index) {
      const uint32_t token = item.inputTokens[index];
      if (token >= geometry.target.vocabularySize) throw std::invalid_argument("prompt token is out of vocabulary");
      tokens[row + index] = token;
      const std::array<uint32_t, 3> rotary = ropePosition(entry, item.logicalPosition + index);
      std::copy(rotary.begin(), rotary.end(), positions + (uint64_t{row} + index) * 3);
    }
  }

  // A request as the target reads it beyond a step's rows: its tokens, a
  // prefill's its prompt, and their RoPE positions.
  static QwenTargetRequest targetRequest(const Request &entry) {
    return {entry.tokens, [&entry](uint64_t logical) { return ropePosition(entry, logical); }};
  }

  // The states a sequence of `rows` rows captures inside them, in their
  // reservations' cells; one at its end is the cell the sequence writes.
  static std::vector<QwenTargetStateCapture> insideStateCaptures(std::span<const PrefillStateCapture> captures,
                                                                 uint32_t rows) {
    std::vector<QwenTargetStateCapture> inside;
    for (const PrefillStateCapture &capture : captures) {
      if (capture.row == rows) continue;
      const GdnParityBuffers &cell = capture.reservation->buffers().gdn->buffers();
      inside.push_back({capture.row, cell.convolutionLayers, cell.recurrentLayers, cell.auxiliary});
    }
    return inside;
  }

  // The chunks of a prefill command, in order: its items' rows fill them
  // from the first with room on, so that a lane's rows past a chunk's
  // continue in the next, up to the arena's chunks (a prefill group's). A
  // lane's part of a chunk captures the draft context rows and states inside
  // it as a command of the lane's rows before it would, with the lengths
  // that command's commit leaves; a state at the end of the lane's rows in
  // the command is `atEnd`. Each chunk's rows go into its slot of the arena:
  // their tokens and RoPE positions, and the draft positions of the rows the
  // DFlash2 draft captures.
  std::vector<PrefillChunk> preparePrefill(std::span<const ModelBatchItem> items,
                                           std::array<Request *, kLaneCount> &entries) {
    uint64_t rows = 0;
    for (const ModelBatchItem &item : items) rows += item.tokenCount;
    if (!rows || rows > uint64_t{kPrefillRows} * prefillArena->chunks())
      throw std::invalid_argument("a prefill command's rows exceed its chunks");
    std::vector<PrefillChunk> chunks(1);
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      const ModelBatchItem &item = items[lane];
      Request &entry = prefillLane(item);
      entries[lane] = &entry;
      const MetalBuffer pageTable = synchronizedPageTable(entry, item);
      const uint64_t end = item.logicalPosition + item.tokenCount;
      QwenLogicalLengths lengths = states.metadata(entry.stateLane).lengths;
      size_t nextCapture = 0;
      for (uint32_t offset = 0; offset < item.tokenCount;) {
        if (chunks.back().rows == kPrefillRows) chunks.emplace_back();
        PrefillChunk &chunk = chunks.back();
        PrefillSequence &sequence = chunk.sequences.emplace_back();
        ModelBatchItem &part = sequence.part;
        part = item;
        part.logicalPosition = item.logicalPosition + offset;
        part.tokenCount = std::min(kPrefillRows - chunk.rows, item.tokenCount - offset);
        part.inputTokens = item.inputTokens.subspan(offset, part.tokenCount);
        part.stateCaptures.clear();
        const uint64_t partEnd = part.logicalPosition + part.tokenCount;
        while (nextCapture < item.stateCaptures.size() && item.stateCaptures[nextCapture].tokens <= partEnd)
          part.stateCaptures.push_back(item.stateCaptures[nextCapture++]);
        sequence.entry = &entry;
        sequence.lane = lane;
        sequence.rowBegin = chunk.rows;
        sequence.attentionStride = (part.tokenCount + kTileRows - 1) / kTileRows * kTileRows;
        sequence.captureBegin = chunk.capturedRows;
        sequence.chunk = chunkParams(part.logicalPosition, part.tokenCount, sequence.attentionStride, item.pageTable);
        sequence.pageTable = pageTable;
        sequence.captures = activeDraftCaptures(entry, part);
        sequence.stateCaptures = stateCaptures(part, sequence.captures, lengths);
        for (PrefillStateCapture &capture : sequence.stateCaptures) capture.atEnd = capture.lengths.targetTokens == end;
        chunk.rows += part.tokenCount;
        chunk.capturedRows += captureRows(sequence.captures);
        offset += part.tokenCount;
        // A next part starts from the lengths a commit of this part's rows
        // would leave; the completion folds the last part's (prefillAsync).
        if (offset < item.tokenCount) {
          lengths.targetTokens = partEnd;
          for (const DispatchDraftCaptureSpan &span : sequence.captures)
            lengths = advanceDraftContext(lengths, partEnd, span.absoluteBegin, span.absoluteEnd, span.resetDraftState,
                                          true);
        }
      }
      if (nextCapture != item.stateCaptures.size())
        throw std::logic_error("a prefill command left a state capture behind");
    }
    for (uint32_t c = 0; c < chunks.size(); ++c) {
      PrefillChunk &chunk = chunks[c];
      uint64_t queryOffset = 0, kvOffset = 0;
      for (PrefillSequence &sequence : chunk.sequences) {
        sequence.queryOffset = queryOffset;
        sequence.kvOffset = kvOffset;
        const uint64_t headRows = uint64_t{sequence.attentionStride} * geometry.target.attentionHeadDimension;
        queryOffset += bytesFor<uint16_t>(geometry.target.attentionQueryHeads * headRows);
        kvOffset += bytesFor<uint16_t>(geometry.target.attentionKvHeads * headRows);
        writePromptRows(*sequence.entry, sequence.part, sequence.rowBegin, c);
      }
      if (queryOffset > prefillArena->get(PrefillTensor::FullQueries).sizeBytes() ||
          kvOffset > prefillArena->get(PrefillTensor::ChunkKeys).sizeBytes())
        throw std::logic_error("prefill scratch geometry overflowed");
      auto *draftPositions =
          contents<uint32_t>(prefillArena->get(PrefillTensor::DraftPositions, c), "draft RoPE positions");
      for (const PrefillSequence &sequence : chunk.sequences)
        for (const DispatchDraftCaptureSpan &capture : sequence.captures)
          for (uint32_t row = capture.absoluteBegin; row < capture.absoluteEnd; ++row)
            draftPositions[sequence.captureBegin + capture.compactDestinationRow + row - capture.absoluteBegin] = row;
    }
    return chunks;
  }

  // Computes a restored lane's draft rings from its context window.
  void addDraftRebuild(CommandGraph &graph, uint32_t lane,
                       const Request::DraftRebuild &rebuild) const {
    auto p = [&](PrefillTensor tensor) { return prefillArena->get(tensor, 0); };
    ops::RoPE::addDraftRangeTables(graph, p(PrefillTensor::DraftInverseFrequencies),
                                   p(PrefillTensor::DraftRopeCos), p(PrefillTensor::DraftRopeSin),
                                   rebuild.rows, rebuild.begin, kPrefillRows);
    draftModel->addContextRebuild(graph, draftPrefillBuffers(*prefillArena, 0), states.window(lane),
                                  states.draft(lane), rebuild.rows, rebuild.begin);
  }

  // The draft's context of chunk `index` of a command's chunks: the rows its
  // parts capture go into their lanes' rings and windows. A state captured
  // inside the command takes into its window the rows its lane's parts
  // capture below its boundary, in whichever chunk they are, over the lane's
  // older rows its window held (prefillAsync); the lane's window then goes
  // on to rows that overwrite them. A state at the end of the lane's rows
  // takes the lane's window.
  void addDraftContext(CommandGraph &graph, std::span<const PrefillChunk> chunks, uint32_t index) {
    const PrefillChunk &chunk = chunks[index];
    if (!chunk.capturedRows || !draftModel) return;
    std::array<DFlashPrefillSpan, kLaneCount * 2> spans{};
    uint32_t spanCount = 0;
    std::vector<DFlashWindowCapture> windowCaptures;
    for (const PrefillSequence &sequence : chunk.sequences) {
      for (const DispatchDraftCaptureSpan &capture : sequence.captures) {
        DFlashPrefillSpan &span = spans.at(spanCount++);
        span.compactRow = sequence.captureBegin + capture.compactDestinationRow;
        span.rows = capture.absoluteEnd - capture.absoluteBegin;
        span.startPosition = capture.absoluteBegin;
        span.ring = states.draft(sequence.entry->stateLane);
        span.window = states.window(sequence.entry->stateLane);
      }
      for (const PrefillChunk &other : chunks)
        for (const PrefillSequence &part : other.sequences) {
          if (part.lane != sequence.lane) continue;
          for (const PrefillStateCapture &state : part.stateCaptures) {
            if (state.atEnd) continue;
            const uint64_t boundary = state.lengths.targetTokens;
            for (const DispatchDraftCaptureSpan &draft : sequence.captures)
              if (draft.absoluteBegin < boundary)
                windowCaptures.push_back(
                    {sequence.captureBegin + draft.compactDestinationRow,
                     static_cast<uint32_t>(std::min<uint64_t>(draft.absoluteEnd, boundary) - draft.absoluteBegin),
                     draft.absoluteBegin, state.reservation->buffers().window->buffer()});
          }
        }
    }
    draftModel->addContextPrefill(graph, draftPrefillBuffers(*prefillArena, index), chunk.capturedRows,
                                  std::span(spans).first(spanCount), windowCaptures);
  }

  // What a prefill command leaves its completion, by lane, indexed like
  // `entries`: its draft captures, the states it captures, and the last row
  // of the target's final hidden it computed, which no later work in the
  // command writes.
  struct PrefillOutputs final {
    std::array<std::vector<DispatchDraftCaptureSpan>, kLaneCount> captures{};
    std::array<std::vector<PrefillStateCapture>, kLaneCount> stateCaptures{};
    std::array<MetalBuffer, kLaneCount> lastRows{};
  };

  // Whether every value of a BF16 row is finite.
  static bool finiteRow(const MetalBuffer &row) {
    const uint16_t *bits = contents<uint16_t>(row, "final prefill row");
    return std::none_of(bits, bits + row.sizeBytes() / sizeof(uint16_t),
                        [](uint16_t value) { return (value & 0x7F80) == 0x7F80; });
  }

  // A lane whose command finishes its prompt copies the prompt's row of the
  // final hidden, `lastRow`, to row 0 of its Hidden0 block. A constrained
  // lane's completion captures that row into finalTargetHidden
  // (captureFinalHidden), which holds it until the first mask. The others
  // share one head: a score lane reads raw logits at the final prompt
  // position, and a policy lane selects its first token; their selection is
  // returned.
  std::optional<InitialSelection> finishPrompt(CommandGraph &graph, Request &entry, const ModelBatchItem &item,
                                               const MetalBuffer &lastRow, uint32_t lane) {
    if (entry.replayingGeneration || item.logicalPosition + item.tokenCount != entry.promptTokens)
      return std::nullopt;
    const uint32_t hidden = geometry.target.hiddenSize;
    ops::RowCopy::add(graph, lastRow, {0, hidden, 0},
                      decodeArena->get(lane, DecodeTensor::Hidden0), {0, hidden, 0}, 1, hidden);
    if (entry.constraint != ConstraintMode::None) return std::nullopt;
    const bool scoring = !entry.scoreTokens.empty();
    if (!scoring && samplingEnabled(entry)) uploadInitialUniform(entry, lane);
    return InitialSelection{&entry, lane, !scoring};
  }

  // A prefill command (preparePrefill): a restored lane's draft rings
  // (DFlash2) first; each chunk's RoPE tables, embedding and image rows in
  // its own tensors; the target's layers over the chunks
  // (QwenTarget::addPrefill); the draft's context of each chunk's captured
  // rows; and the first token of each lane whose rows end its prompt. A
  // lane's GDN and auxiliary state pass from its current cell through its
  // parts to its next cell, which its last part writes; a part that ends at
  // a state it captures writes that state's cell, another the next cell or
  // the arena's scratch cell, whichever the part after it does not write. So
  // no part writes the cell it reads, and none the lane's current cell; with
  // one scratch cell, at most one lane's rows run past a chunk.
  PrefillOutputs encodePrefillGraph(CommandGraph &graph, std::span<const ModelBatchItem> items,
                                    std::array<Request *, kLaneCount> &entries) {
    const std::vector<PrefillChunk> chunks = preparePrefill(items, entries);
    const uint32_t lanes = static_cast<uint32_t>(items.size());
    // Each lane's parts, (chunk, sequence) in order.
    std::array<std::vector<std::pair<uint32_t, uint32_t>>, kLaneCount> parts;
    for (uint32_t c = 0; c < chunks.size(); ++c)
      for (uint32_t s = 0; s < chunks[c].sequences.size(); ++s) parts[chunks[c].sequences[s].lane].push_back({c, s});
    std::vector<std::vector<const GdnParityBuffers *>> written(chunks.size());
    for (uint32_t c = 0; c < chunks.size(); ++c) written[c].resize(chunks[c].sequences.size());
    uint32_t spanning = 0;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const GdnParityBuffers *const next = &states.next(entries[lane]->stateLane);
      // The cell the part after this one writes; none after the last part,
      // whose end is the lane's rows' (a state there is `atEnd`).
      const GdnParityBuffers *after = nullptr;
      spanning += parts[lane].size() > 1;
      for (size_t index = parts[lane].size(); index-- > 0;) {
        const auto [c, s] = parts[lane][index];
        const PrefillSequence &sequence = chunks[c].sequences[s];
        const PrefillStateCapture *boundary = nullptr;
        for (const PrefillStateCapture &capture : sequence.stateCaptures)
          if (!capture.atEnd && capture.row == sequence.part.tokenCount) boundary = &capture;
        const GdnParityBuffers *cell = next;
        if (boundary)
          cell = &boundary->reservation->buffers().gdn->buffers();
        else if (after == next)
          cell = prefillArena->groupCell();
        if (!cell) throw std::logic_error("a prefill group has no scratch cell");
        written[c][s] = after = cell;
      }
    }
    if (spanning > 1) throw std::invalid_argument("several lanes' rows run past a prefill chunk");

    // A restored lane's rings first: they pass through the context buffers
    // and the draft RoPE tables before the command's rows do. Once the window
    // is full, the rows' window store and ring writes reuse the slots of the
    // oldest restored positions, so the rebuild comes before them too.
    // Without a draft (the MTP head) nothing reads the rings.
    for (uint32_t lane = 0; lane < lanes; ++lane)
      if (const auto &rebuild = entries[lane]->draftRebuild; rebuild && draftModel)
        addDraftRebuild(graph, entries[lane]->stateLane, *rebuild);

    std::array<const GdnParityBuffers *, kLaneCount> from{};
    for (uint32_t lane = 0; lane < lanes; ++lane) from[lane] = &states.current(entries[lane]->stateLane);
    std::vector<std::vector<QwenTargetPrefillSequence>> sequences(chunks.size());
    // Each part's states inside its rows, in their reservations' cells.
    std::vector<std::vector<QwenTargetStateCapture>> inside;
    inside.reserve(uint64_t{chunks.size()} * kLaneCount);
    std::vector<QwenTargetPrefillChunk> modelChunks;
    modelChunks.reserve(chunks.size());
    for (uint32_t c = 0; c < chunks.size(); ++c) {
      const PrefillChunk &chunk = chunks[c];
      const auto p = [&](PrefillTensor tensor) { return prefillArena->get(tensor, c); };
      QwenTargetPrefillChunk &model = modelChunks.emplace_back();
      model.buffers = prefillBuffers(*prefillArena, c);
      model.rows = chunk.rows;
      addRopeTables(graph, p(PrefillTensor::TargetPositions), chunk.rows, p(PrefillTensor::DraftPositions),
                    chunk.capturedRows, model.buffers.ropeCos, model.buffers.ropeSin, p(PrefillTensor::DraftRopeCos),
                    p(PrefillTensor::DraftRopeSin));
      targetModel.addEmbedding(graph, p(PrefillTensor::InputTokens), model.buffers.hidden[0], chunk.rows);
      for (uint32_t s = 0; s < chunk.sequences.size(); ++s) {
        const PrefillSequence &sequence = chunk.sequences[s];
        const ModelBatchItem &part = sequence.part;
        addImageRows(graph, *sequence.entry, part, model.buffers.hidden[0], sequence.rowBegin);
        const GdnParityBuffers &in = *from[sequence.lane], &out = *written[c][s];
        from[sequence.lane] = &out;
        QwenTargetPrefillSequence &destination = sequences[c].emplace_back();
        destination.lane = sequence.lane;
        destination.rowBegin = sequence.rowBegin;
        destination.rows = part.tokenCount;
        destination.tokens = part.inputTokens;
        destination.request = targetRequest(*sequence.entry);
        destination.attentionStride = sequence.attentionStride;
        destination.queryOffset = sequence.queryOffset;
        destination.kvOffset = sequence.kvOffset;
        destination.chunk = sequence.chunk;
        destination.pageTable = sequence.pageTable;
        destination.convolutionIn = in.convolutionLayers;
        destination.convolutionOut = out.convolutionLayers;
        destination.recurrentIn = in.recurrentLayers;
        destination.recurrentOut = out.recurrentLayers;
        destination.auxiliaryIn = in.auxiliary;
        destination.auxiliaryOut = out.auxiliary;
        // A state at the part's end is the cell it writes.
        destination.stateCaptures = inside.emplace_back(insideStateCaptures(sequence.stateCaptures, part.tokenCount));
        // The rows the draft's context reads.
        destination.captureCount = static_cast<uint32_t>(sequence.captures.size());
        for (uint32_t index = 0; index < sequence.captures.size(); ++index) {
          const DispatchDraftCaptureSpan &capture = sequence.captures[index];
          destination.captures[index] = {
              sequence.rowBegin + static_cast<uint32_t>(capture.absoluteBegin - part.logicalPosition),
              sequence.captureBegin + capture.compactDestinationRow, capture.absoluteEnd - capture.absoluteBegin};
        }
      }
      model.sequences = sequences[c];
    }
    const std::vector<MetalBuffer> finalHidden = targetModel.addPrefill(graph, modelChunks, kvPages, aneFfn);
    for (uint32_t c = 0; c < chunks.size(); ++c) addDraftContext(graph, chunks, c);

    PrefillOutputs outputs;
    std::array<InitialSelection, kLaneCount> selections{};
    uint32_t selectionCount = 0;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      // The lane's last part ends its rows, as a command of the part's rows
      // would.
      const auto [c, s] = parts[lane].back();
      const PrefillSequence &last = chunks[c].sequences[s];
      const MetalBuffer lastRow =
          prefillU16(finalHidden[c], last.rowBegin + last.part.tokenCount - 1, 1, geometry.target.hiddenSize);
      if (const auto selection = finishPrompt(graph, *entries[lane], items[lane], lastRow, lane))
        selections[selectionCount++] = *selection;
      for (const auto [partChunk, partSequence] : parts[lane]) {
        const PrefillSequence &part = chunks[partChunk].sequences[partSequence];
        outputs.captures[lane].insert(outputs.captures[lane].end(), part.captures.begin(), part.captures.end());
        outputs.stateCaptures[lane].insert(outputs.stateCaptures[lane].end(), part.stateCaptures.begin(),
                                           part.stateCaptures.end());
      }
      outputs.lastRows[lane] = lastRow;
    }
    if (selectionCount) encodeInitialSelections(graph, std::span(selections).first(selectionCount), lanes);
    return outputs;
  }

  // Writes a lane's inputs that the CPU makes: the anchor at row 0 of the
  // lane's draft input rows (whose other rows the draft's block fills with
  // mask tokens, and which the verify reads its anchor from), the RoPE
  // positions of its rows, and the target's own inputs of its verify rows
  // (QwenTarget::prepareVerifyLane).
  void prepareDecodeLane(Request &entry, const ModelBatchItem &item,
                         uint32_t lane) {
    if (!entry.resident || !entry.promptComplete || !entry.pendingToken) {
      throw std::logic_error("decode request is not ready");
    }
    const QwenLaneMetadata &metadata = states.metadata(entry.stateLane);
    if (metadata.lengths.targetTokens != item.logicalPosition ||
        !metadata.lengths.hasCompleteDraftWindow(kDraftCacheStride)) {
      throw std::logic_error("decode state length is not exact");
    }
    static_cast<void>(synchronizedPageTable(entry, item));
    auto *draftInput = contents<uint32_t>(
        decodeArena->get(lane, DecodeTensor::DraftInputTokens),
        "draft input tokens");
    draftInput[0] = *entry.pendingToken;
    std::fill(draftInput + 1, draftInput + kDecodeRows,
              geometry.target.maskToken);
    if (!draftModel) {
      // The MTP head writes each proposal on the GPU (QwenTarget::addMtpDraft)
      // as a certain draw: its sparse candidates the token alone, at
      // probability one, which acceptance reads as a draft's.
      auto *candidates = contents<uint32_t>(decodeArena->get(lane, DecodeTensor::Candidates), "candidates");
      auto *probabilities =
          contents<float>(decodeArena->get(lane, DecodeTensor::ProposalProbs), "proposal probabilities");
      for (uint32_t i = 0; i < kDraftProposalTokens; ++i)
        for (uint32_t c = 0; c < SPLASH_DRAFT_CANDIDATES; ++c) {
          candidates[i * SPLASH_DRAFT_CANDIDATES + c] = c ? geometry.target.vocabularySize : 0;
          probabilities[i * SPLASH_DRAFT_CANDIDATES + c] = c ? 0.0F : 1.0F;
        }
    }
    targetModel.prepareVerifyLane(lane, item.logicalPosition, targetRequest(entry));

    auto *positions =
        contents<uint32_t>(decodeArena->get(lane, DecodeTensor::Positions),
                           "decode RoPE positions");
    auto *draftPositions =
        contents<uint32_t>(decodeArena->get(lane, DecodeTensor::DraftPositions),
                           "decode draft RoPE positions");
    for (uint32_t row = 0; row < kDecodeRows; ++row) {
      const std::array<uint32_t, 3> rotary =
          ropePosition(entry, item.logicalPosition + row);
      std::copy(rotary.begin(), rotary.end(), positions + row * 3);
      // The draft is a text model over logical positions.
      draftPositions[row] = static_cast<uint32_t>(item.logicalPosition + row);
    }
  }

  // Batch lanes beyond the active width replay the last active request so
  // every padded M32 lane binds valid state.
  static Request &laneEntry(std::span<Request *const> entries, uint32_t lane) {
    Request *entry = entries[std::min<size_t>(lane, entries.size() - 1)];
    if (!entry)
      throw std::invalid_argument("empty decode batch lane");
    return *entry;
  }

  void bindDraftRings(
      std::span<Request *const> entries,
      std::vector<std::array<MetalBuffer, kLaneCount>> &keys,
      std::vector<std::array<MetalBuffer, kLaneCount>> &values) const {
    keys.resize(geometry.draft.layers);
    values.resize(geometry.draft.layers);
    for (uint32_t layer = 0; layer < geometry.draft.layers; ++layer) {
      for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
        const auto &ring =
            states.draft(laneEntry(entries, lane).stateLane)[layer];
        keys[layer][lane] = ring.keys;
        values[layer][lane] = ring.values;
      }
    }
  }

  void encodeDraftBatchGraph(CommandGraph &graph,
                             std::span<Request *const> entries,
                             std::span<const uint64_t> logicalPositions) {
    if (entries.empty() || entries.size() > kLaneCount ||
        entries.size() != logicalPositions.size()) {
      throw std::invalid_argument("invalid draft decode batch");
    }
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    // The draft shares the target's vocabulary head and its storage rows.
    const uint32_t storage = targetModel.decodeStorageLanes(lanes);
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->batchSlice(tensor, storage);
    };
    std::array<uint32_t, kLaneCount> cacheLengths{};
    for (uint32_t lane = 0; lane < lanes; ++lane)
      cacheLengths[lane] = static_cast<uint32_t>(logicalPositions[lane]);

    DFlashDecodeBuffers buffers;
    buffers.linearScratch = decodeArena->linearScratch();
    for (uint32_t hidden = 0; hidden < buffers.hidden.size(); ++hidden) {
      buffers.hidden[hidden] = d(static_cast<DecodeTensor>(
          static_cast<uint32_t>(DecodeTensor::DraftHidden0) + hidden));
    }
    buffers.normalized = d(DecodeTensor::DraftNormalized);
    buffers.dynamic = d(DecodeTensor::DraftDynamic);
    buffers.convolved = d(DecodeTensor::DraftConvolved);
    buffers.proposalQkv = d(DecodeTensor::DraftProposalQkv);
    buffers.attention = d(DecodeTensor::DraftAttention);
    buffers.projected = d(DecodeTensor::DraftProjected);
    buffers.residual = d(DecodeTensor::DraftResidual);
    buffers.intermediate = d(DecodeTensor::DraftIntermediate);
    buffers.finalHidden = d(DecodeTensor::DraftFinalHidden);
    buffers.logits = d(DecodeTensor::Logits);
    buffers.selectorHidden = d(DecodeTensor::SelectorHidden);
    buffers.queryKeys = d(DecodeTensor::DraftQueryKeys);
    buffers.queryValues = d(DecodeTensor::DraftQueryValues);
    buffers.ropeCos = d(DecodeTensor::DraftRopeCos);
    buffers.ropeSin = d(DecodeTensor::DraftRopeSin);
    buffers.gateScratch = decodeArena->gateScratch();
    bindDraftRings(entries, buffers.persistentKeys, buffers.persistentValues);
    draftModel->addDecode(graph, std::move(buffers),
                          targetModel.vocabularyProjection(),
                          std::span(cacheLengths).first(lanes));
    std::array<uint32_t, kLaneCount> anchors{};
    std::array<ops::SamplingPolicy, kLaneCount> policies{};
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      Request &entry = laneEntry(entries, lane);
      if (!entry.pendingToken)
        throw std::invalid_argument("draft batch lane has no anchor");
      anchors[lane] = *entry.pendingToken;
      policies[lane] = samplingPolicy(entry);
    }
    draftModel->addSelection(
        graph,
        {d(DecodeTensor::Logits), d(DecodeTensor::TopPartialIds),
         d(DecodeTensor::TopPartialValues), d(DecodeTensor::Candidates),
         d(DecodeTensor::Unary), d(DecodeTensor::SelectorHidden),
         d(DecodeTensor::SamplingUniforms), d(DecodeTensor::ProposedTokens),
         d(DecodeTensor::ProposalProbs)},
        std::span(anchors).first(lanes), std::span(policies).first(lanes));
    sampling.addThinkingDrafts(graph, std::span(policies).first(lanes),
                               d(DecodeTensor::ProposedTokens));
  }

  // Without a DFlash2 draft, the proposals the MTP head drafted
  // (encodeMtpDraft) take the end tokens of each lane whose thinking cutoff
  // is at row 0, as the draft's selection's do (encodeDraftBatchGraph).
  void encodeThinkingDrafts(CommandGraph &graph, std::span<Request *const> entries) {
    if (entries.empty() || entries.size() > kLaneCount)
      throw std::invalid_argument("invalid thinking draft batch");
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    std::array<ops::SamplingPolicy, kLaneCount> policies{};
    for (uint32_t lane = 0; lane < lanes; ++lane)
      policies[lane] = samplingPolicy(laneEntry(entries, lane));
    sampling.addThinkingDrafts(graph, std::span(policies).first(lanes),
                               decodeArena->batchSlice(DecodeTensor::ProposedTokens,
                                                       targetModel.decodeStorageLanes(lanes)));
  }

  // The target's verify forward or, with `mtpDraft`, the MTP head's draft
  // steps over the same rows (QwenTarget::addMtpDraft), as deep as the
  // batch's deepest lane wants.
  void encodeTargetVerifyBatchForward(CommandGraph &graph,
                                      std::span<Request *const> entries,
                                      std::span<const ModelBatchItem> items,
                                      bool mtpDraft = false) {
    if (entries.empty() || entries.size() > kLaneCount ||
        entries.size() != items.size()) {
      throw std::invalid_argument("invalid target verify batch");
    }
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    const uint32_t storage = targetModel.decodeStorageLanes(lanes);
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->batchSlice(tensor, storage);
    };

    std::array<ChunkedPrefillParams, kLaneCount> chunks{};
    const uint32_t attentionLayers =
        geometry.target.kvLayout.attentionLayers;
    std::vector<MetalBuffer> chunkKeys(attentionLayers);
    std::vector<MetalBuffer> chunkValues(attentionLayers);
    QwenTargetVerifyBuffers buffers;
    buffers.linearScratch = decodeArena->linearScratch();
    buffers.hidden = {d(DecodeTensor::Hidden0), d(DecodeTensor::Hidden1)};
    buffers.normalized = d(DecodeTensor::Normalized);
    buffers.gdnHidden = d(DecodeTensor::GdnHidden);
    buffers.gdnOutput = d(DecodeTensor::GdnOutput);
    buffers.denseIntermediate = d(DecodeTensor::Intermediate);
    buffers.fullPacked = d(DecodeTensor::FullPacked);
    buffers.fullQueries = d(DecodeTensor::FullQueries);
    buffers.attentionPartials = d(DecodeTensor::AttentionPartials);
    buffers.attentionStatistics = d(DecodeTensor::AttentionStatistics);
    buffers.fullAttention = d(DecodeTensor::FullAttention);
    buffers.attentionHidden = d(DecodeTensor::AttentionHidden);
    buffers.attentionOutput = d(DecodeTensor::AttentionOutput);
    buffers.ropeCos = d(DecodeTensor::RopeCos);
    buffers.ropeSin = d(DecodeTensor::RopeSin);
    buffers.ropeInverseFrequencies = prefillArena->get(PrefillTensor::TargetInverseFrequencies);
    buffers.capturedTargetHidden = d(DecodeTensor::CapturedTargetHidden);
    buffers.finalHidden = d(DecodeTensor::FinalHidden);
    buffers.logits = d(DecodeTensor::Logits);
    buffers.denseGateScratch = decodeArena->gateScratch();
    buffers.gdnPacked = d(DecodeTensor::GdnPacked);
    buffers.gdnTape = decodeArena->batchSlice(DecodeTensor::GdnTape, kLaneCount);
    buffers.chunkKeys = chunkKeys;
    buffers.chunkValues = chunkValues;
    buffers.moe = decodeArena->moeScratch(storage);
    for (uint32_t lane = 0; lane < lanes; ++lane)
      chunks[lane] = ops::PagedAttention::verifyParams(
          items[lane].logicalPosition,
          static_cast<uint32_t>(items[lane].pageTable.size()));
    for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
      Request &entry = laneEntry(entries, lane);
      buffers.pageTables[lane] =
          decodeArena->get(entry.stateLane, DecodeTensor::PageTable);
      buffers.currentGdnStates[lane] = states.current(entry.stateLane).stateBase;
      buffers.nextGdnStates[lane] = states.next(entry.stateLane).stateBase;
      buffers.auxiliary[lane] = states.current(entry.stateLane).auxiliary;
    }
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint32_t stateLane = entries[lane]->stateLane;
      const QwenLaneMetadata &metadata = states.metadata(stateLane);
      buffers.gdnTapeLanes[lane] = {decodeArena->gdnTapeSlot(stateLane, metadata.activeParity),
                                    decodeArena->gdnTapeSlot(stateLane, metadata.activeParity ^ 1),
                                    metadata.pendingRows};
    }
    for (uint32_t layer = 0; layer < attentionLayers; ++layer) {
      chunkKeys[layer] = decodeArena->attentionBatchSlice(
          DecodeTensor::ChunkKeysBase, layer, storage);
      chunkValues[layer] = decodeArena->attentionBatchSlice(
          DecodeTensor::ChunkValuesBase, layer, storage);
    }
    if (mtpDraft) {
      buffers.anchors = d(DecodeTensor::DraftInputTokens);
      buffers.proposals = d(DecodeTensor::ProposedTokens);
      buffers.candidates = d(DecodeTensor::Candidates);
      buffers.proposalsPerLane = kDraftProposalTokens;
      buffers.candidatesPerProposal = SPLASH_DRAFT_CANDIDATES;
      uint32_t steps = 1;
      for (uint32_t lane = 0; lane < lanes; ++lane) steps = std::max(steps, mtpStepsFor(*entries[lane]));
      for (uint32_t lane = 0; lane < lanes; ++lane) entries[lane]->mtpDrafted = steps;
      // From the draft vocabulary and each lane's context while every lane
      // does.
      if (targetModel.hasMtpDraftVocabulary())
        for (uint32_t lane = 0; lane < lanes; ++lane)
          entries[lane]->draftContext.scan(entries[lane]->tokens, geometry.target.vocabularySize,
                                           kQwen4DraftContextTokens, targetModel.qwen4()->layout.mtpDraftTokens);
      const bool draftVocabulary = useDraftVocabulary(entries.first(lanes));
      if (draftVocabulary) bindDraftContext(buffers, entries.first(lanes));
      QwenTargetMtpDraft draft;
      draft.steps = steps;
      draft.draftVocabulary = draftVocabulary;
      if (plePrefetch && pleStore && steps > 1) armPlePrefetch(draft, entries.first(lanes), items, (steps + 1) / 2);
      targetModel.addMtpDraft(graph, std::move(buffers), kvPages, std::span(chunks).first(lanes), lanes, draft);
      return;
    }
    targetModel.addVerify(graph, std::move(buffers), kvPages,
                          std::span(chunks).first(lanes), lanes, verifyLiveRows(entries.first(lanes)));
  }

  // The MTP head's draft of a cycle's proposals, in place of a DFlash2
  // draft's: its steps, then the thinking end tokens. A lane keeps no token
  // past its drafted proposals (verifyLiveRows), so its retention is capped
  // there.
  void encodeMtpDraft(CommandGraph &graph, std::span<Request *const> entries, std::span<const ModelBatchItem> items,
                      std::span<DecodeLaneResult> lanes, std::span<uint32_t> maximumRetained) {
    if (draftModel || !targetModel.hasMtp()) throw std::logic_error("the target has no MTP head to draft with");
    encodeTargetVerifyBatchForward(graph, entries, items, true);
    if (const uint32_t liveRows = verifyLiveRows(entries))
      for (uint32_t lane = 0; lane < entries.size(); ++lane) {
        lanes[lane].maximumRetained = std::min(lanes[lane].maximumRetained, liveRows);
        maximumRetained[lane] = lanes[lane].maximumRetained;
      }
    encodeThinkingDrafts(graph, entries);
  }

  void encodeTargetVerifyBatchPolicy(CommandGraph &graph,
                                     std::span<Request *const> entries) {
    if (entries.empty() || entries.size() > kLaneCount)
      throw std::invalid_argument("invalid target policy batch");
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    std::array<ops::SamplingPolicy, kLaneCount> policies{};
    std::array<uint32_t, kLaneCount> stateLanes{};
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      if (!entries[lane])
        throw std::invalid_argument("empty target policy lane");
      policies[lane] = samplingPolicy(*entries[lane]);
      stateLanes[lane] = entries[lane]->stateLane;
    }
    sampling.addVerify(graph, std::span(policies).first(lanes),
                       samplingBuffers(lanes), geometry.target.stopTokens[0],
                       geometry.target.stopTokens[1],
                       {penaltyTable, std::span(stateLanes).first(lanes)});
  }

  void encodeDraftStateCommitBatch(CommandGraph &graph,
                                   std::span<Request *const> entries,
                                   std::span<const ModelBatchItem> items) {
    if (entries.empty() || entries.size() > kLaneCount ||
        entries.size() != items.size()) {
      throw std::invalid_argument("invalid draft state commit batch");
    }
    const uint32_t lanes = static_cast<uint32_t>(entries.size());
    // The draft's projections take the target's storage rows, as in its
    // decode (encodeDraftBatchGraph).
    const uint32_t storage = targetModel.decodeStorageLanes(lanes);
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->batchSlice(tensor, storage);
    };

    std::array<uint32_t, kLaneCount> startPositions{};
    for (uint32_t lane = 0; lane < lanes; ++lane)
      startPositions[lane] = static_cast<uint32_t>(items[lane].logicalPosition);
    DFlashContextBuffers buffers;
    buffers.linearScratch = decodeArena->linearScratch();
    buffers.capturedTargetHidden = d(DecodeTensor::CapturedTargetHidden);
    buffers.projected = d(DecodeTensor::ContextProjected);
    buffers.hidden = d(DecodeTensor::ContextHidden);
    buffers.contextKv = d(DecodeTensor::ContextKv);
    buffers.ropeCos = d(DecodeTensor::DraftRopeCos);
    buffers.ropeSin = d(DecodeTensor::DraftRopeSin);
    buffers.retainedCounts = d(DecodeTensor::RetainedCount);
    bindDraftRings(entries, buffers.persistentKeys, buffers.persistentValues);
    draftModel->addContextCommit(graph, std::move(buffers),
                                 std::span(startPositions).first(lanes));
  }

  void encodeBatchAcceptance(CommandGraph &graph,
                             std::span<Request *const> lanes,
                             std::span<const uint32_t> maximumRetained) {
    if (lanes.empty() || lanes.size() > kLaneCount ||
        lanes.size() != maximumRetained.size()) {
      throw std::invalid_argument("invalid DFlash acceptance batch");
    }
    std::array<ops::SamplingPolicy, kLaneCount> policies{};
    for (uint32_t lane = 0; lane < lanes.size(); ++lane) {
      if (!lanes[lane] || !maximumRetained[lane] ||
          maximumRetained[lane] > kDecodeRows) {
        throw std::invalid_argument("invalid DFlash acceptance lane");
      }
      policies[lane] = samplingPolicy(*lanes[lane]);
    }
    const uint32_t width = static_cast<uint32_t>(lanes.size());
    sampling.addAcceptance(
        graph,
        {decodeArena->batchSlice(DecodeTensor::ProposedTokens, width),
         decodeArena->batchSlice(DecodeTensor::Candidates, width),
         decodeArena->batchSlice(DecodeTensor::ProposalProbs, width),
         decodeArena->batchSlice(DecodeTensor::TargetVocabularyRows, width),
         decodeArena->batchSlice(DecodeTensor::SamplingUniforms, width),
         decodeArena->batchSlice(DecodeTensor::OutputTokens, width),
         decodeArena->batchSlice(DecodeTensor::RetainedCount, width),
         decodeArena->batchSlice(DecodeTensor::AcceptedCount, width)},
        maximumRetained, std::span(policies).first(width),
        geometry.target.stopTokens[0], geometry.target.stopTokens[1]);
  }

  // The target's commit after acceptance (QwenTarget::addStateCommit): each
  // lane's auxiliary state after its retained rows. The GDN state folds the
  // retained rows into the lane's next step instead.
  void encodeTargetStateCommit(CommandGraph &graph,
                               std::span<Request *const> lanes) {
    if (lanes.empty() || lanes.size() > kLaneCount)
      throw std::invalid_argument("invalid target state commit batch");
    const uint32_t width = static_cast<uint32_t>(lanes.size());
    QwenTargetCommitBuffers buffers;
    buffers.retainedCounts = decodeArena->batchSlice(DecodeTensor::RetainedCount, width);
    for (uint32_t lane = 0; lane < width; ++lane) {
      Request &entry = laneEntry(lanes, lane);
      buffers.auxiliaryIn[lane] = states.current(entry.stateLane).auxiliary;
      buffers.auxiliaryOut[lane] = states.next(entry.stateLane).auxiliary;
    }
    targetModel.addStateCommit(graph, std::move(buffers), width);
  }

  // A verify's selection, acceptance and commits, which end its cycle's last
  // command: the target's state commit and the DFlash2 draft's, whose rings
  // take the rows the lanes retained.
  void encodeVerifyCommit(CommandGraph &graph, std::span<Request *const> entries,
                          std::span<const ModelBatchItem> items,
                          std::span<const uint32_t> maximumRetained) {
    encodeTargetVerifyBatchPolicy(graph, entries);
    encodeBatchAcceptance(graph, entries, maximumRetained);
    encodeTargetStateCommit(graph, entries);
    if (draftModel) encodeDraftStateCommitBatch(graph, entries, items);
  }

  void encodeBatchEmbedding(CommandGraph &graph, DecodeTensor tokens,
                            DecodeTensor output, uint32_t lanes) {
    if (!lanes || lanes > kLaneCount)
      throw std::invalid_argument("invalid embedding batch width");
    const uint32_t rows = lanes * kDecodeRows;
    targetModel.addEmbedding(graph, decodeArena->batchSlice(tokens, lanes),
                             decodeArena->batchSlice(output, lanes), rows);
  }

  void encodeBatchVerifyInput(CommandGraph &graph, uint32_t lanes) {
    if (!lanes || lanes > kLaneCount)
      throw std::invalid_argument("invalid verify-input batch width");
    targetModel.addVerifyInput(
        graph, decodeArena->batchSlice(DecodeTensor::DraftInputTokens, lanes),
        decodeArena->batchSlice(DecodeTensor::ProposedTokens, lanes),
        decodeArena->batchSlice(DecodeTensor::InputTokens, lanes), lanes);
  }

  // A stop token or the last budgeted token needs no target work of its own:
  // the next cycle would only echo it as output. Emitting it as soon as it is
  // selected saves that cycle; the engine is told it has no KV row.
  bool emitTerminalAnchor(Request &entry, ModelStepResult &result) const {
    const bool stop = isStopToken(geometry, *entry.pendingToken);
    if (!stop && entry.maxNewTokens - entry.generatedTokens != 1)
      return false;
    result.outputTokens.push_back(*entry.pendingToken);
    result.outputTokensWithoutKv = 1;
    result.finished = stop;
    ++entry.generatedTokens;
    return true;
  }

  std::vector<ModelStepResult> finalizeDecode(
      std::span<DecodeLaneResult> lanes, std::span<const ModelBatchItem> items,
      CommandTiming timing) {
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      DecodeLaneResult &laneResult = lanes[lane];
      auto d = [&](DecodeTensor tensor) {
        return decodeArena->get(lane, tensor);
      };
      laneResult.retained = *contents<uint32_t>(d(DecodeTensor::RetainedCount),
                                                "GPU retained token count");
      laneResult.accepted = *contents<uint32_t>(d(DecodeTensor::AcceptedCount),
                                                "GPU accepted draft count");
      if (!laneResult.retained || laneResult.retained > kDecodeRows)
        throw std::runtime_error("target policy produced invalid retention");
      if (laneResult.accepted > kDraftProposalTokens)
        throw std::runtime_error(
            "target accepted more than the draft proposed");
      countMtpOutcome(*laneResult.request, laneResult.accepted);
      // The retained target tokens end with the next anchor. A non-finite
      // target row can also accept a sentinel draft proposal as an interior
      // token, so every retained token is checked.
      const uint32_t *targetTokens = contents<uint32_t>(
          d(DecodeTensor::OutputTokens), "target output tokens");
      laneResult.failure =
          invalidSelection({targetTokens, laneResult.retained}, lane);
      if (laneResult.failure.empty())
        countDraftVocabulary(*laneResult.request, {targetTokens, laneResult.retained});
    }

    std::vector<ModelStepResult> results;
    results.reserve(items.size());
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      DecodeLaneResult &laneResult = lanes[lane];
      Request &entry = *laneResult.request;
      if (!laneResult.failure.empty()) {
        // The cycle's state and tokens are not committed; the engine ends
        // the request.
        entry.resultInvalid = true;
        entry.maskWords.clear();
        entry.verifyMaskInFlight = false;
        results.push_back({.requestId = entry.id,
                           .failure = std::move(laneResult.failure)});
        continue;
      }
      const uint32_t *targetTokens =
          contents<uint32_t>(decodeArena->get(lane, DecodeTensor::OutputTokens),
                             "target output tokens");
      std::vector<uint32_t> output;
      output.reserve(laneResult.retained);
      output.push_back(laneResult.currentAnchor);
      output.insert(output.end(), targetTokens,
                    targetTokens + (laneResult.retained - 1));

      states.swapDecodeParity(entry.stateLane, laneResult.retained);
      const uint64_t nextLength =
          items[lane].logicalPosition + laneResult.retained;
      states.updateLengths(
          entry.stateLane,
          advanceDraftContext(states.metadata(entry.stateLane).lengths,
                              nextLength, items[lane].logicalPosition,
                              nextLength, false, false));
      entry.generatedTokens += laneResult.retained;
      commitSelected(entry, {targetTokens, laneResult.retained});
      entry.maskWords.clear();
      entry.verifyMaskInFlight = false;
      // The end replaced the first proposals, which acceptance keeps as far
      // as the lane retains them; the draft proposed the others.
      const uint32_t accepted =
          std::min(laneResult.accepted, laneResult.retained - 1);
      results.push_back({entry.id,
                         0,
                         std::move(output),
                         false,
                         DecodeStage::Regular,
                         kDraftProposalTokens - laneResult.endDrafts,
                         accepted - std::min(accepted, laneResult.endDrafts)});
      ModelStepResult &result = results.back();
      if (entry.generatedTokens < entry.maxNewTokens)
        emitTerminalAnchor(entry, result);
    }

    counters.lastDecodeGpuSeconds = timing.gpuSeconds;
    counters.totalDecodeGpuSeconds += timing.gpuSeconds;
    counters.lastDecodeWallSeconds = timing.wallSeconds;
    counters.totalDecodeWallSeconds += timing.wallSeconds;
    return results;
  }

  // A constrained DFlash cycle has one host dependency between three Metal
  // commands: draft proposals define the grammar simulation, while the target
  // forward is independent of the resulting mask.  This ticket keeps the
  // scheduler batch (and therefore its DecodeArena lanes) owned across that
  // dependency.  All state transitions run on the engine thread; completion
  // handlers only wake it, so they capture the wake hook and never the ticket.
  class ConstrainedDecodeTicket final : public ModelBatchTicket {
  public:
    ConstrainedDecodeTicket(Impl &impl, std::vector<DecodeLaneResult> lanes,
                            std::span<const ModelBatchItem> items,
                            const CommandGraph &draft,
                            std::function<void()> completion)
        : impl_(impl), lanes_(std::move(lanes)),
          items_(items.begin(), items.end()),
          wake_(std::move(completion)) {
      submit(draft);
    }

    std::vector<ModelMaskRequest> takeMaskRequests() override {
      std::vector<ModelMaskRequest> requests;
      if (stage_ == Stage::Draft && command_.ready()) {
        addTiming(command_.wait());
        std::array<Request *, kLaneCount> entries{};
        for (uint32_t lane = 0; lane < lanes_.size(); ++lane) {
          DecodeLaneResult &laneResult = lanes_[lane];
          Request &entry = *laneResult.request;
          const uint32_t *proposed = contents<uint32_t>(
              impl_.decodeArena->get(lane, DecodeTensor::ProposedTokens),
              "constrained draft proposals");
          entry.maskWords.clear();
          entry.verifyMaskInFlight = true;
          entries[lane] = &entry;

          if (!abandoned_[lane]) {
            ModelMaskRequest request;
            request.requestId = entry.id;
            request.simulationTokens.reserve(kDecodeRows);
            request.simulationTokens.push_back(*entry.pendingToken);
            request.simulationTokens.insert(request.simulationTokens.end(),
                                            proposed,
                                            proposed + kDraftProposalTokens);
            requests.push_back(std::move(request));
          }
        }

        CommandGraph target;
        const uint32_t width = static_cast<uint32_t>(lanes_.size());
        impl_.encodeBatchVerifyInput(target, width);
        impl_.encodeBatchEmbedding(target, DecodeTensor::InputTokens,
                                   DecodeTensor::Hidden0, width);
        impl_.encodeTargetVerifyBatchForward(
            target, {entries.data(), lanes_.size()}, items_);
        submit(target);
        stage_ = Stage::TargetForward;
      }

      if (stage_ == Stage::TargetForward && command_.ready()) {
        const CommandTiming forward = command_.wait();
        addTiming(forward);
        targetForwardGpuSeconds_ += forward.gpuSeconds;
        maskWaitStarted_ = AwakeClock::now();
        stage_ = Stage::WaitingMask;
      }

      if (stage_ == Stage::WaitingMask) {
        bool masksReady = true;
        for (uint32_t lane = 0; lane < lanes_.size(); ++lane) {
          masksReady = masksReady && (abandoned_[lane] ||
                                      !lanes_[lane].request->maskWords.empty());
        }
        if (masksReady) {
          maskWaitSeconds_ +=
              std::chrono::duration<double>(AwakeClock::now() - *maskWaitStarted_)
                  .count();
          maskWaitStarted_.reset();
          std::array<Request *, kLaneCount> entries{};
          std::array<uint32_t, kLaneCount> maximumRetained{};
          for (uint32_t lane = 0; lane < lanes_.size(); ++lane) {
            DecodeLaneResult &laneResult = lanes_[lane];
            Request &entry = *laneResult.request;
            entries[lane] = &entry;
            maximumRetained[lane] = laneResult.maximumRetained;
            if (abandoned_[lane])
              impl_.admitEveryToken(lane);
            else
              impl_.uploadConstraintMasks(lane, entry.maskWords);
          }

          CommandGraph commit;
          impl_.encodeVerifyCommit(commit, {entries.data(), lanes_.size()}, items_,
                                   {maximumRetained.data(), lanes_.size()});
          submit(commit);
          stage_ = Stage::Commit;
        }
      }
      return requests;
    }

    bool ownsMaskWait(uint64_t requestId) const noexcept override {
      if (stage_ == Stage::Draft || stage_ == Stage::Done)
        return false;
      return std::any_of(lanes_.begin(), lanes_.end(),
                         [requestId](const DecodeLaneResult &lane) {
                           return lane.request->id == requestId;
                         });
    }

    void abandonMask(uint64_t requestId) noexcept override {
      if (stage_ == Stage::Done)
        return;
      for (uint32_t lane = 0; lane < lanes_.size(); ++lane) {
        if (lanes_[lane].request->id == requestId) {
          abandoned_[lane] = true;
          lanes_[lane].request->maskWords.clear();
          return;
        }
      }
    }

    bool ready() const noexcept override {
      return stage_ == Stage::Commit && command_.ready();
    }

    std::vector<ModelStepResult> wait() override {
      if (!ready())
        throw std::logic_error("constrained decode ticket is not complete");
      addTiming(command_.wait());
      stage_ = Stage::Done;
      ModelTelemetry &counters = impl_.counters;
      ++counters.constrainedMaskOverlapBatches;
      counters.constrainedMaskOverlapRequests += lanes_.size();
      counters.lastConstrainedTargetForwardGpuSeconds =
          targetForwardGpuSeconds_;
      counters.totalConstrainedTargetForwardGpuSeconds +=
          targetForwardGpuSeconds_;
      counters.lastConstrainedMaskWaitSeconds = maskWaitSeconds_;
      counters.totalConstrainedMaskWaitSeconds += maskWaitSeconds_;
      return impl_.finalizeDecode(lanes_, items_, timing_);
    }

    double wallMilliseconds() const noexcept override {
      return timing_.wallSeconds * 1000.0;
    }

  private:
    enum class Stage : uint8_t {
      Draft,
      TargetForward,
      WaitingMask,
      Commit,
      Done
    };

    void submit(const CommandGraph &graph) {
      command_ = impl_.backend.submitCommandAsync(graph.command(), wake_);
    }

    void addTiming(CommandTiming value) noexcept {
      timing_.gpuSeconds += value.gpuSeconds;
      timing_.wallSeconds += value.wallSeconds;
    }

    Impl &impl_;
    std::vector<DecodeLaneResult> lanes_;
    std::vector<ModelBatchItem> items_;
    Stage stage_ = Stage::Draft;
    CommandTicket command_;
    CommandTiming timing_;
    std::array<bool, kLaneCount> abandoned_{};
    double targetForwardGpuSeconds_ = 0.0;
    double maskWaitSeconds_ = 0.0;
    std::optional<AwakeClock::time_point> maskWaitStarted_;
    std::function<void()> wake_;
  };
};

Runtime::Runtime(RuntimeContext context)
    : impl_(std::make_unique<Impl>(context)) {}

Runtime::~Runtime() = default;

void Runtime::checkHealth() { impl_->backend.checkHealth(); }

void Runtime::beginColdRequest(const ModelRequest &request,
                               uint32_t stateLane) {
  if (const StateAdmission admission = beginAt(request, stateLane); !admission.granted()) {
    throw metal::MetalAllocationError(
        std::string("unable to allocate a lane's state: ") +
            metal::allocationFailureName(admission.allocationFailure),
        admission.allocationFailure);
  }
  try {
    setDraftContextPlan(
        request.id,
        planDraftContext(0, static_cast<uint32_t>(request.prompt.size()), {}));
  } catch (...) {
    end(request.id);
    throw;
  }
}

std::vector<uint8_t> Runtime::pendingGdnRows(uint64_t requestId) const {
  const Impl::Request &entry = impl_->request(requestId);
  if (!entry.resident)
    throw std::logic_error("request is not resident");
  const QwenLaneMetadata &metadata = impl_->states.metadata(entry.stateLane);
  const ops::GdnShape shape = impl_->geometry.target.gdnShape();
  const uint32_t kvWidth = (shape.keyHeads + shape.valueHeads) * shape.headDimension;
  // Each field holds one row after another; the offset and row bytes of each.
  const std::array<std::pair<uint64_t, uint64_t>, 4> fields{{
      {0, uint64_t{shape.convolutionDimension} * 2},
      {gdn_tape_kv_offset(shape.convolutionDimension), uint64_t{kvWidth} * 2},
      {gdn_tape_decay_offset(shape.convolutionDimension, kvWidth), uint64_t{shape.valueHeads} * 4},
      {gdn_tape_beta_offset(shape.convolutionDimension, kvWidth, shape.valueHeads),
       uint64_t{shape.valueHeads} * 2},
  }};
  // The latest step's slot is the one the lane's next step folds from.
  const uint8_t *slot =
      contents<const uint8_t>(impl_->decodeArena->batchSlice(DecodeTensor::GdnTape, kLaneCount),
                              "GDN tape") +
      impl_->decodeArena->gdnTapeSlot(entry.stateLane, metadata.activeParity);
  std::vector<uint8_t> rows;
  for (uint32_t layer = 0; layer < impl_->geometry.target.stateLayout.layers; ++layer) {
    const uint8_t *tape = slot + layer * ops::gdnTapeLayerBytes(shape);
    for (const auto &[offset, rowBytes] : fields)
      rows.insert(rows.end(), tape + offset, tape + offset + metadata.pendingRows * rowBytes);
  }
  return rows;
}

StateAdmission Runtime::begin(const ModelRequest &request) {
  Impl::VisionRollback rollback{*impl_, impl_->vision};
  StateAdmission admission = admitIdleLane(
      impl_->states, [&](uint32_t lane) { return beginAt(request, lane); });
  rollback.committed = admission.granted();
  return admission;
}

void Runtime::suspend(uint64_t requestId) {
  Impl::Request &entry = impl_->request(requestId);
  if (!entry.resident || entry.verifyMaskInFlight) {
    throw std::logic_error("Qwen request cannot be suspended");
  }
  impl_->releaseLane(entry);
  impl_->releaseImages(entry);
  entry.draftContextPlan.reset();
  entry.draftRebuild.reset();
  entry.replayingGeneration |= entry.promptComplete;
  entry.promptComplete = false;
  entry.resident = false;
}

StateAdmission Runtime::resume(const ModelRequest &request) {
  Impl::Request &entry = impl_->request(request.id);
  if (entry.resident) {
    throw std::logic_error("Qwen request is not suspended");
  }
  if (request.prompt.size() < entry.promptTokens) {
    throw std::invalid_argument("recomputed history cannot shorten the prompt");
  }
  Impl::VisionRollback rollback{*impl_, impl_->vision};
  std::vector<Impl::ImageState> images;
  StateAdmission admission = admitIdleLane(impl_->states, [&](uint32_t lane) {
    return impl_->activate(request, lane, images);
  });
  if (admission.granted()) {
    entry.stateLane = *admission.lane;
    entry.resident = true;
    entry.promptTokens = static_cast<uint32_t>(request.prompt.size());
    entry.tokens.assign(request.prompt.begin(), request.prompt.end());
    entry.images = std::move(images);
    entry.restoredTokens = request.restoredTokens;
    impl_->bindPenalties(entry, request.prompt);
  }
  rollback.committed = admission.granted();
  return admission;
}

StateAdmission Runtime::beginAt(const ModelRequest &request, uint32_t stateLane) {
  if (!request.id || stateLane >= kLaneCount || request.prompt.empty()) {
    throw std::invalid_argument("invalid executor request activation");
  }
  if (impl_->requests.contains(request.id)) {
    throw std::logic_error("request is already active");
  }
  Impl::Request entry;
  entry.id = request.id;
  entry.promptTokens = static_cast<uint32_t>(request.prompt.size());
  entry.tokens.assign(request.prompt.begin(), request.prompt.end());
  entry.maxNewTokens = request.maxNewTokens;
  entry.sampling = request.sampling;
  entry.constraint = request.constraint;
  entry.flags = request.flags;
  if (request.thinkingBudget)
    entry.thinking = Impl::Request::Thinking{*request.thinkingBudget};
  entry.scoreTokens.assign(request.scoreTokens.begin(),
                           request.scoreTokens.end());
  std::vector<Impl::ImageState> images;
  const StateAdmission admission = impl_->activate(request, stateLane, images);
  if (!admission.granted())
    return admission;
  entry.stateLane = stateLane;
  entry.resident = true;
  entry.images = std::move(images);
  entry.restoredTokens = request.restoredTokens;
  impl_->bindPenalties(entry, request.prompt);
  auto [_, inserted] = impl_->requests.emplace(request.id, std::move(entry));
  if (!inserted) {
    throw std::logic_error("request insertion lost uniqueness");
  }
  return admission;
}

std::unique_ptr<StateRestore> Runtime::beginRestore(
    uint64_t requestId, uint32_t boundary,
    std::shared_ptr<const CompositeState> state, bool restoreDraft,
    std::function<void()> completion) {
  Impl::Request &entry = impl_->request(requestId);
  if (!entry.resident || !state || boundary >= entry.promptTokens)
    throw std::invalid_argument("invalid state restore");
  return impl_->states.beginRestore(entry.stateLane, *state, restoreDraft,
      std::move(completion), [this, requestId, boundary, restoreDraft] {
        finishRestore(requestId, boundary, restoreDraft);
      });
}

void Runtime::finishRestore(uint64_t requestId, uint32_t restoredPrefixLength,
                            bool restoreDraftState) {
  Impl::Request &entry = impl_->request(requestId);
  // A shorter restore would replay rows of images that were never staged.
  if (restoredPrefixLength < entry.restoredTokens)
    throw std::invalid_argument("restore stops before images its activation left out");
  if (!restoreDraftState)
    ++impl_->counters.draftStateRestoreSkipped;
  const QwenLogicalLengths &lengths =
      impl_->states.metadata(entry.stateLane).lengths;
  if (lengths.targetTokens != restoredPrefixLength ||
      (restoreDraftState &&
       !lengths.hasCompleteDraftWindow(kDraftCacheStride)) ||
      (!restoreDraftState && lengths.draftLength != 0)) {
    throw std::invalid_argument("prefix logical length does not match state");
  }
  // Activation left out the images ModelRequest::restoredTokens covers; a
  // restore further in releases the rest here: warmup and direct callers
  // activate with 0, as does an engine start that let its cache lease go and
  // found a state when it looked up again. Their spans stay because rotary
  // positions after them depend on their grids.
  for (Impl::ImageState &image : entry.images) {
    if (image.span.end() <= restoredPrefixLength) {
      image.rows.reset();
    }
  }
  entry.promptComplete = false;
  if (!entry.replayingGeneration) {
    entry.finalTargetHidden.clear();
    entry.pendingToken.reset();
  }
  entry.draftContextPlan.reset();
  entry.draftRebuild.reset();
  if (restoreDraftState)
    entry.draftRebuild = Impl::Request::DraftRebuild{
        static_cast<uint32_t>(lengths.draftBase), lengths.draftLength};
}

void Runtime::setDraftContextPlan(uint64_t requestId, DraftContextPlan plan) {
  Impl::Request &entry = impl_->request(requestId);
  if (!entry.resident || plan.replayEnd != entry.promptTokens) {
    throw std::invalid_argument("draft context plan does not match request");
  }
  const uint64_t current =
      impl_->states.metadata(entry.stateLane).lengths.targetTokens;
  if (plan.replayBegin != current)
    throw std::invalid_argument("draft context plan restore boundary is stale");
  entry.draftContextPlan = std::move(plan);
}

std::vector<ModelStepResult>
Runtime::prefill(const BatchPlan &plan, std::span<const ModelBatchItem> items) {
  return prefillAsync(plan, items, {})->wait();
}

std::unique_ptr<ModelBatchTicket>
Runtime::submit(const BatchPlan &plan, std::span<const ModelBatchItem> items,
                std::function<void()> completion) {
  switch (plan.kind) {
  case WorkKind::Prefill:
    return prefillAsync(plan, items, std::move(completion));
  case WorkKind::Decode:
    return decodeAsync(plan, items, std::move(completion));
  }
  throw std::logic_error("unknown model work kind");
}

std::unique_ptr<ModelBatchTicket>
Runtime::prefillAsync(const BatchPlan &plan,
                      std::span<const ModelBatchItem> items,
                      std::function<void()> completion) {
  validatePlan(plan, items, WorkKind::Prefill);
  if (plan.decodeStage != DecodeStage::Regular) {
    throw std::invalid_argument("Qwen prefill cannot resume a mask plan");
  }

  std::array<Impl::Request *, kLaneCount> entries{};
  // Each request's draws before the chunk, which a rerun of it restores.
  std::array<uint64_t, kLaneCount> draws{};
  CommandGraph graph;
  if (impl_->aneFfn) {
    for (uint32_t lane = 0; lane < items.size(); ++lane)
      draws[lane] = impl_->request(items[lane].requestId).rngCounter;
    impl_->aneFfn->begin();
  }
  const auto outputs = impl_->encodePrefillGraph(graph, items, entries);
  const bool encodesImages = std::any_of(
      entries.begin(), entries.begin() + items.size(), [](const auto *entry) {
        return std::any_of(entry->images.begin(), entry->images.end(),
                           [](const auto &image) {
                             return image.rows && image.rows->encoding;
                           });
      });
  // A state captured inside the command starts from the lane's context
  // window as the command finds it, whose older rows the command's own rows
  // overwrite. It is copied once, before the command runs: a rerun (below)
  // would find the window the failed run wrote.
  for (uint32_t lane = 0; lane < items.size(); ++lane)
    for (const Impl::PrefillStateCapture &capture : outputs.stateCaptures[lane])
      if (!capture.atEnd)
        impl_->states.copyWindow(entries[lane]->stateLane, *capture.reservation);
  // The completion's items hold the states' reservations until the ticket
  // has waited for every run of the command.
  std::vector<ModelBatchItem> copiedItems(items.begin(), items.end());
  CommandTicket command =
      impl_->aneFfn ? impl_->aneFfn->commit(graph, std::move(completion))
                    : impl_->backend.submitCommandAsync(graph.command(), std::move(completion));
  Impl *impl = impl_.get();
  auto finish = [impl, entries, outputs, draws,
                 items = std::move(copiedItems)](CommandTiming &timing) mutable {
    // A chunk whose Neural Engine work failed holds no usable outputs, and
    // the split has stopped: the GPU runs the chunk again alone, which
    // computes what it would have the first time. Nothing of the chunk is
    // committed before the code below: state parity, lengths, the selected
    // token, the states it captures and the image rows' state. The rerun
    // writes the same KV and draft rows, and the same cells and window rows of
    // the states it captures, whose older window rows were copied before the
    // first run; a restored lane's rebuild may read window slots the failed
    // run filled with its own rows, but each belongs to a position the rerun
    // captures again, and that capture overwrites the ring slot after the
    // rebuild; clearForColdStart clears the current state again, which
    // neither command writes; synchronizedPageTable writes nothing at the
    // same revision; image rows the first command encoded are still
    // `encoding` and are copied, not encoded again; and the requests' draws
    // restored, a sampled first token draws the same uniform. The first
    // ticket was released before this completion runs
    // (DeferredMetalTicket::wait), so the backend takes the rerun's command,
    // whose time counts in the chunk's.
    if (impl->aneFfn && !impl->aneFfn->finish()) {
      for (uint32_t lane = 0; lane < items.size(); ++lane)
        entries[lane]->rngCounter = draws[lane];
      CommandGraph again;
      static_cast<void>(impl->encodePrefillGraph(again, items, entries));
      const CommandTiming rerun = impl->backend.submitCommandAsync(again.command()).wait();
      timing.gpuSeconds += rerun.gpuSeconds;
      timing.wallSeconds += rerun.wallSeconds;
      timing.hostSeconds += rerun.hostSeconds;
      ++impl->counters.aneFfnReruns;
    }
    // A row that is not finite, of the chunk or of the state it computed
    // from, reaches the lane's last row through the recurrent state, and
    // through attention on a GPU whose tile weighs non-finite keys and values
    // (some families' tiles give them zero weight, for every reader). Such a
    // lane commits nothing of the chunk.
    std::array<bool, kLaneCount> finite{};
    for (uint32_t lane = 0; lane < items.size(); ++lane)
      finite[lane] = Impl::finiteRow(outputs.lastRows[lane]);

    std::vector<ModelStepResult> results;
    results.reserve(items.size());
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      Impl::Request &entry = *entries[lane];
      const ModelBatchItem &item = items[lane];
      const uint64_t nextLength = item.logicalPosition + item.tokenCount;
      if (!finite[lane]) {
        // The chunk's state is not committed; the engine ends the request.
        entry.resultInvalid = true;
        results.push_back({.requestId = entry.id,
                           .consumedPromptTokens = item.tokenCount,
                           .failure = "target prefill row is not finite"});
        continue;
      }
      // The anchor this chunk selects when it completes a generation prompt.
      std::optional<uint32_t> selected;
      if (nextLength == entry.promptTokens && !entry.replayingGeneration &&
          entry.scoreTokens.empty() && entry.constraint == ConstraintMode::None) {
        selected = impl->initialToken(lane);
        if (std::string failure = impl->invalidSelection({&*selected, 1}, lane);
            !failure.empty()) {
          // The chunk's state is not committed; the engine ends the
          // request.
          entry.resultInvalid = true;
          results.push_back({.requestId = entry.id,
                             .consumedPromptTokens = item.tokenCount,
                             .failure = std::move(failure)});
          continue;
        }
      }
      impl->states.swapParity(entry.stateLane);
      entry.draftRebuild.reset();
      QwenLogicalLengths lengths = impl->states.metadata(entry.stateLane).lengths;
      lengths.targetTokens = nextLength;
      for (const DispatchDraftCaptureSpan &capture : outputs.captures[lane]) {
        lengths = Impl::advanceDraftContext(lengths, nextLength,
                                            capture.absoluteBegin,
                                            capture.absoluteEnd,
                                            capture.resetDraftState, true);
        impl->counters.draftContextRowsActive += capture.activeRows;
        impl->counters.draftContextRowsMaterialization +=
            capture.materializationRows;
        if (capture.resetDraftState)
          ++impl->counters.draftStateResets;
      }
      impl->counters.targetPrefillRows += item.tokenCount;
      impl->counters.draftContextRowsAvoided +=
          item.tokenCount - Impl::captureRows(outputs.captures[lane]);
      impl->states.updateLengths(entry.stateLane, lengths);
      entry.promptComplete = nextLength == entry.promptTokens;
      ModelStepResult result{entry.id, item.tokenCount, {}, false,
                             DecodeStage::Regular, 0, 0};
      if (entry.promptComplete && !entry.replayingGeneration) {
        entry.pendingToken.reset();
        if (!entry.scoreTokens.empty()) {
          // Score-only: read the raw fp32 logits at the final prompt position
          // (the lane's logits row 0) in requested order.
          const float *row = contents<float>(
              impl->decodeArena->get(lane, DecodeTensor::Logits),
              "score logits");
          result.scoreLogits.reserve(entry.scoreTokens.size());
          for (uint32_t token : entry.scoreTokens) {
            const float logit = row[token];
            if (!std::isfinite(logit)) {
              // A numerical outcome for this request, not a broken invariant:
              // report it as a lane failure so the engine drops this request
              // before cache publication or output and the batch survives.
              result.scoreLogits.clear();
              result.failure = "score logit is not finite";
              entry.resultInvalid = true;
              break;
            }
            result.scoreLogits.push_back(logit);
          }
          result.finished = true;
        } else if (selected) {
          impl->commitSelected(entry, {&*selected, 1});
          // Unless it ends the request, the first token waits for the first
          // decode cycle; the engine sends it to the client now.
          if (!impl->emitTerminalAnchor(entry, result))
            result.firstToken = entry.pendingToken;
        } else {
          // The first token waits for the request's first mask. A replay
          // never gets here: it keeps its stage, and a request that holds
          // its mask asks for none.
          impl->captureFinalHidden(entry, lane);
          entry.decodeStage = DecodeStage::ApplyInitialMask;
          result.nextDecodeStage = DecodeStage::ApplyInitialMask;
        }
      }
      if (entry.promptComplete)
        entry.replayingGeneration = false;
      if (result.failure.empty())
        result.capturedStates = impl->sealStates(entry, outputs.stateCaptures[lane]);
      results.push_back(std::move(result));
    }

    // The images of the lanes whose rows are finite are encoded, and an
    // image whose last row is injected goes to the cache, after the states
    // above took the rows they hold.
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      if (!finite[lane])
        continue;
      const uint64_t chunkEnd = items[lane].logicalPosition + items[lane].tokenCount;
      for (Impl::ImageState &image : entries[lane]->images) {
        if (!image.rows)
          continue;
        Impl::ImageRows &rows = *image.rows;
        if (rows.encoding) {
          rows.encoding = false;
          rows.encoded = true;
          rows.pixels = MetalBuffer{};
        }
        // Its last row is injected: the cache owns the rows from now on, so
        // reclaim can free them while the request decodes.
        if (image.span.end() <= chunkEnd) {
          impl->retain(image.rows);
          image.rows.reset();
        }
      }
    }
    // The images the command encoded for those lanes alone stay to be
    // encoded again by their next use.
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      if (finite[lane])
        continue;
      for (Impl::ImageState &image : entries[lane]->images)
        if (image.rows && image.rows->encoding)
          image.rows->encoding = false;
    }
    impl->counters.lastPrefillWallSeconds = timing.wallSeconds;
    impl->counters.totalPrefillWallSeconds += timing.wallSeconds;
    impl->counters.lastPrefillGpuSeconds = timing.gpuSeconds;
    impl->counters.totalPrefillGpuSeconds += timing.gpuSeconds;
    return results;
  };
  return std::make_unique<DeferredMetalTicket>(
      std::move(command), std::move(finish), !encodesImages);
}

std::vector<ModelStepResult>
Runtime::decode(const BatchPlan &plan, std::span<const ModelBatchItem> items) {
  return decodeAsync(plan, items, {})->wait();
}

std::unique_ptr<ModelBatchTicket>
Runtime::decodeAsync(const BatchPlan &plan,
                     std::span<const ModelBatchItem> items,
                     std::function<void()> completion) {
  validatePlan(plan, items, WorkKind::Decode);
  const bool constrained = plan.constrained;
  if (plan.decodeStage != DecodeStage::Regular) {
    if (!constrained) {
      throw std::invalid_argument(
          "only constrained decode uses a specialized decode stage");
    }
    return impl_->submitInitialSelection(items, std::move(completion));
  }

  const uint32_t width = static_cast<uint32_t>(items.size());
  std::vector<Impl::DecodeLaneResult> lanes(width);
  std::array<Impl::Request *, kLaneCount> requests{};
  std::array<uint64_t, kLaneCount> logicalPositions{};
  std::array<uint32_t, kLaneCount> maximumRetained{};
  for (uint32_t lane = 0; lane < width; ++lane) {
    const ModelBatchItem &item = items[lane];
    Impl::Request &entry = impl_->request(item.requestId);
    if ((entry.constraint == ConstraintMode::TokenMask) != constrained) {
      throw std::invalid_argument(
          "request does not belong to the batch's constraint mode");
    }
    if (entry.decodeStage != DecodeStage::Regular) {
      throw std::logic_error("request decode stage does not match decode plan");
    }
    if (!entry.pendingToken)
      throw std::logic_error("decode request has no current anchor");
    if (entry.draftRebuild)
      throw std::logic_error("a restored lane computes its draft rings in a prefill chunk");
    const uint32_t remaining = entry.maxNewTokens - entry.generatedTokens;
    if (!remaining)
      throw std::logic_error("completed request was decoded");
    if (isStopToken(impl_->geometry, *entry.pendingToken) || remaining == 1) {
      throw std::logic_error("terminal anchor was not emitted on selection");
    }

    if (constrained && !entry.maskWords.empty())
      throw std::logic_error("constrained request has stale mask state");
    if (Impl::samplingEnabled(entry)) {
      Impl::stageSamplingCycle(entry);
      impl_->uploadSamplingUniforms(entry, lane);
    }

    // DFlash has one physical graph: anchor + seven proposal rows. A shorter
    // output budget only lowers the token-exact commit count; it never
    // changes the Metal graph shape.
    Impl::DecodeLaneResult &laneResult = lanes[lane];
    laneResult.request = &entry;
    laneResult.currentAnchor = *entry.pendingToken;
    laneResult.maximumRetained = std::min(remaining, kDecodeRows);
    if (const auto cutoff = Impl::samplingPolicy(entry).thinkingCutoff)
      laneResult.endDrafts = cutoff->replacedProposals();

    impl_->prepareDecodeLane(entry, item, lane);
    requests[lane] = &entry;
    logicalPositions[lane] = item.logicalPosition;
    maximumRetained[lane] = laneResult.maximumRetained;
  }

  const std::span<Impl::Request *const> entries(requests.data(), width);
  const uint32_t ropeRows = width * kDecodeRows;
  CommandGraph commandGraph;
  impl_->addRopeTables(
      commandGraph,
      impl_->decodeArena->batchSlice(DecodeTensor::Positions, width), ropeRows,
      impl_->decodeArena->batchSlice(DecodeTensor::DraftPositions, width),
      ropeRows, impl_->decodeArena->batchSlice(DecodeTensor::RopeCos, width),
      impl_->decodeArena->batchSlice(DecodeTensor::RopeSin, width),
      impl_->decodeArena->batchSlice(DecodeTensor::DraftRopeCos, width),
      impl_->decodeArena->batchSlice(DecodeTensor::DraftRopeSin, width));
  // Without a DFlash2 draft the MTP head drafts the proposals.
  if (impl_->draftModel) {
    impl_->encodeBatchEmbedding(commandGraph, DecodeTensor::DraftInputTokens,
                                DecodeTensor::DraftHidden0, width);
    impl_->encodeDraftBatchGraph(commandGraph, entries,
                                 {logicalPositions.data(), width});
  } else {
    impl_->encodeMtpDraft(commandGraph, entries, items, lanes, {maximumRetained.data(), width});
  }
  if (constrained) {
    return std::make_unique<Impl::ConstrainedDecodeTicket>(
        *impl_, std::move(lanes), items, commandGraph, std::move(completion));
  }
  impl_->encodeBatchVerifyInput(commandGraph, width);
  impl_->encodeBatchEmbedding(commandGraph, DecodeTensor::InputTokens,
                              DecodeTensor::Hidden0, width);
  impl_->encodeTargetVerifyBatchForward(commandGraph, entries, items);
  impl_->encodeVerifyCommit(commandGraph, entries, items, {maximumRetained.data(), width});

  std::vector<ModelBatchItem> copiedItems(items.begin(), items.end());
  Impl *impl = impl_.get();
  auto finish = [impl, lanes = std::move(lanes),
                 items = std::move(copiedItems)](CommandTiming timing) mutable {
    return impl->finalizeDecode(lanes, items, timing);
  };
  CommandTicket command = impl_->backend.submitCommandAsync(
      commandGraph.command(), std::move(completion));
  return std::make_unique<DeferredMetalTicket>(std::move(command),
                                               std::move(finish));
}

uint32_t Runtime::residentLane(uint64_t requestId) {
  Impl::Request &entry = impl_->request(requestId);
  if (!entry.resident)
    throw std::logic_error("request is not resident");
  return entry.stateLane;
}

std::shared_ptr<StateReservation> Runtime::reserveState(uint64_t requestId, uint32_t tokens) {
  static_cast<void>(residentLane(requestId));
  if (!tokens || tokens % kv::kPageTokens)
    throw std::invalid_argument("a state boundary is page-aligned");
  return impl_->states.reserve();
}

std::shared_ptr<const CompositeState> Runtime::snapshot(uint64_t requestId) {
  const uint32_t lane = residentLane(requestId);
  std::shared_ptr<const CompositeState> state = impl_->states.snapshot(lane);
  if (!state)
    return state;
  return impl_->holdStraddledRows(impl_->request(requestId),
                                  impl_->states.metadata(lane).lengths.targetTokens,
                                  std::move(state));
}

uint64_t Runtime::snapshotBytes() const noexcept {
  return impl_->states.layout().cachedBytes();
}

bool Runtime::canSnapshotToDisk(bool inUse) const noexcept {
  return impl_->states.canSnapshotToDisk(inUse);
}

std::unique_ptr<StateOffload>
Runtime::snapshotToDisk(uint64_t requestId, std::function<void()> completion) {
  return impl_->states.snapshotToDisk(residentLane(requestId), std::move(completion));
}

uint32_t Runtime::prefillGroupRows() const noexcept { return impl_->geometry.prefillChunks * kPrefillRows; }

uint32_t Runtime::statesToActivate() const noexcept {
  return impl_->states.statesToActivate();
}

uint64_t Runtime::reclaimIdleState(bool keepLane, IdleMemory scope) noexcept {
  // One unit per call, so a denied allocation frees only what it needs;
  // rebuildable caches go once the pool has nothing more to give.
  if (const uint64_t buffer = impl_->states.releaseOneIdle(keepLane))
    return buffer;
  return scope == IdleMemory::BuffersThenCaches ? impl_->releaseOneCache() : 0;
}

std::optional<std::string>
Runtime::provideMask(uint64_t requestId, std::span<const uint32_t> words) {
  Impl::Request &entry = impl_->request(requestId);
  const bool acceptsMask =
      waitsForMask(entry.decodeStage) || entry.verifyMaskInFlight;
  // Initial-mask replies can race resource preemption. They belong to the
  // host continuation, not the released device state.
  if (entry.constraint != ConstraintMode::TokenMask || !acceptsMask ||
      !entry.maskWords.empty()) {
    throw std::logic_error("request is not waiting for a token mask");
  }
  const uint32_t maskWords = impl_->geometry.maskWords();
  uint64_t expected = entry.verifyMaskInFlight
                          ? uint64_t{kDecodeRows + 1} * maskWords
                          : maskWords;
  // The native loop matches each response's word count to its request.
  if (words.size() != expected) {
    throw std::logic_error("token mask has the wrong word count");
  }
  const uint32_t rows = static_cast<uint32_t>(words.size() / maskWords);
  for (uint32_t row = 0; row < rows; ++row) {
    auto begin = words.begin() + uint64_t{row} * maskWords;
    if (std::none_of(begin, begin + maskWords,
                     [](uint32_t word) { return word != 0; })) {
      return "token mask row permits no vocabulary token";
    }
  }
  if (entry.verifyMaskInFlight) {
    if (!entry.pendingToken || (words[*entry.pendingToken / 32] &
                                (1U << (*entry.pendingToken % 32))) == 0) {
      return "verify mask is not synchronized to the pending anchor";
    }
  }
  entry.maskWords.assign(words.begin(), words.end());
  return std::nullopt;
}

void Runtime::end(uint64_t requestId) {
  auto found = impl_->requests.find(requestId);
  if (found == impl_->requests.end())
    return;
  impl_->releaseImages(found->second);
  if (found->second.resident)
    impl_->releaseLane(found->second);
  impl_->requests.erase(found);
}

namespace {

// A warmup step whose lane failed (a row or logit that is not finite) fails
// the warmup there, with the lane's reason.
void requireLanesSucceeded(std::span<const ModelStepResult> results) {
  for (const ModelStepResult &result : results)
    if (!result.failure.empty())
      throw std::runtime_error(result.failure);
}

// Warmup runs on the startup runway the engine's KV pool allocated
// (ExecutionLimits::warmupKvPages); it never allocates KV.
void requireRunwayPages(const kv::PageStorage &storage,
                        std::span<const uint32_t> pages) {
  for (uint32_t page : pages) {
    if (page >= ExecutionLimits::warmupKvPages || !storage.isAllocated(page)) {
      throw std::logic_error("warmup KV page " + std::to_string(page) +
                             " is outside the startup runway");
    }
  }
}

// A warmup request's batch item. Each warmup residency keeps one page list,
// so its revision stays 1.
ModelBatchItem warmupItem(uint64_t id, uint64_t position, uint32_t tokens,
                          std::span<const uint32_t> pages) {
  return {.requestId = id,
          .logicalPosition = position,
          .tokenCount = tokens,
          .pageTable = pages,
          .pageTableRevision = 1};
}

} // namespace

void Runtime::prepareWarmupDecode(uint64_t requestId, uint32_t anchor) {
  // Teacher-force a valid input so EOS selected by synthetic prefill cannot
  // prevent the warmup from exercising the real draft/verify/commit graph.
  while (anchor < impl_->geometry.target.vocabularySize &&
         isStopToken(impl_->geometry, anchor))
    ++anchor;
  if (anchor >= impl_->geometry.target.vocabularySize)
    throw std::logic_error("decode warmup has no non-terminal input token");
  auto &entry = impl_->request(requestId);
  entry.pendingToken = anchor;
  entry.generatedTokens = 0;
}

WarmupStepResult Runtime::warmupPrefill(uint32_t rows) {
  if (!rows || rows > kPrefillRows)
    throw std::invalid_argument("invalid prefill warmup row count");
  constexpr uint64_t id = std::numeric_limits<uint64_t>::max() - 100;
  double wallSeconds = 0.0, hostSeconds = 0.0;
  std::vector<WarmupLaneResult> lanes;
  std::vector<uint32_t> warmupPrompt(rows, 0);
  ModelRequest request;
  request.id = id;
  request.prompt = warmupPrompt;
  request.maxNewTokens = 16;
  beginColdRequest(request, 0);
  try {
    std::vector<uint32_t> pages((rows + kv::kPageTokens - 1) / kv::kPageTokens);
    std::iota(pages.begin(), pages.end(), 0u);
    requireRunwayPages(impl_->kvPages, pages);
    BatchPlan plan{.kind = WorkKind::Prefill,
                   .items = {{id, rows}},
                   .decodeStage = DecodeStage::Regular};
    ModelBatchItem item = warmupItem(id, 0, rows, pages);
    item.inputTokens = request.prompt;
    const auto phaseStart = AwakeClock::now();
    const std::unique_ptr<ModelBatchTicket> ticket = prefillAsync(plan, std::span<const ModelBatchItem>(&item, 1), {});
    auto result = ticket->wait();
    wallSeconds = std::chrono::duration<double>(AwakeClock::now() - phaseStart).count();
    hostSeconds = ticket->hostMilliseconds() / 1000.0;
    requireLanesSucceeded(result);
    if (result.size() != 1 || result[0].consumedPromptTokens != rows) {
      throw std::runtime_error("prefill warmup result mismatch");
    }
    lanes.push_back({std::move(result[0]), impl_->request(id).pendingToken,
                     impl_->states.metadata(0).lengths.targetTokens});
    end(id);
  } catch (...) {
    end(id);
    throw;
  }
  return {"real " + std::to_string(rows) +
              "-row KV target+draft prefill [M32]",
          wallSeconds, std::move(lanes), hostSeconds};
}

WarmupStepResult Runtime::warmupDecodeBatch(uint32_t width) {
  if (!width || width > kLaneCount) {
    throw std::invalid_argument("invalid decode warmup width");
  }
  constexpr uint64_t firstId = std::numeric_limits<uint64_t>::max() - 110;
  // Plan order is deliberately unrelated to state-lane order. DecodeArena
  // lanes follow the explicit BatchPlan, while recurrent and KV state stay
  // addressed by each request's state lane; batching must never assume lanes
  // 0..3.
  constexpr std::array<uint32_t, kLaneCount> stateLaneOrder{2, 0, 3, 1};
  double wallSeconds = 0.0;
  std::vector<WarmupLaneResult> lanes;
  std::array<std::vector<uint32_t>, kLaneCount> pages;
  try {
    for (uint32_t lane = 0; lane < width; ++lane) {
      std::vector<uint32_t> warmupPrompt{lane};
      ModelRequest request;
      request.id = firstId + lane;
      request.prompt = warmupPrompt;
      request.maxNewTokens = 16;
      beginColdRequest(request, stateLaneOrder[lane]);
      pages[lane] = {5 + lane};
      requireRunwayPages(impl_->kvPages, pages[lane]);
      BatchPlan prefillPlan{.kind = WorkKind::Prefill,
                            .items = {{request.id, 1}},
                            .decodeStage = DecodeStage::Regular};
      ModelBatchItem item = warmupItem(request.id, 0, 1, pages[lane]);
      item.inputTokens = request.prompt;
      requireLanesSucceeded(
          prefill(prefillPlan, std::span<const ModelBatchItem>(&item, 1)));
      prepareWarmupDecode(request.id, warmupPrompt.back());
    }
    BatchPlan plan;
    plan.kind = WorkKind::Decode;
    std::vector<ModelBatchItem> items;
    for (uint32_t lane = 0; lane < width; ++lane) {
      plan.items.push_back({firstId + lane, 0});
      items.push_back(warmupItem(firstId + lane, 1, 0, pages[lane]));
    }
    const auto phaseStart = AwakeClock::now();
    auto decoded = decode(plan, items);
    wallSeconds = std::chrono::duration<double>(AwakeClock::now() - phaseStart).count();
    requireLanesSucceeded(decoded);
    bool committedEveryLane = decoded.size() == width;
    for (uint32_t lane = 0; committedEveryLane && lane < width; ++lane) {
      const auto &lengths = impl_->states.metadata(stateLaneOrder[lane]).lengths;
      committedEveryLane = !decoded[lane].outputTokens.empty() &&
                           lengths.targetTokens > 1 &&
                           lengths.targetTokens ==
                               1 + decoded[lane].outputTokens.size() -
                                   decoded[lane].outputTokensWithoutKv &&
                           lengths.hasCompleteDraftWindow(kDraftCacheStride);
    }
    if (!committedEveryLane) {
      throw std::runtime_error(
          "decode warmup B" + std::to_string(width) +
          " mismatch [results=" + std::to_string(decoded.size()) + "]");
    }
    for (uint32_t lane = 0; lane < width; ++lane) {
      lanes.push_back({std::move(decoded[lane]),
                       impl_->request(firstId + lane).pendingToken,
                       impl_->states.metadata(stateLaneOrder[lane]).lengths.targetTokens});
      end(firstId + lane);
    }
  } catch (...) {
    for (uint32_t lane = 0; lane < width; ++lane)
      end(firstId + lane);
    throw;
  }
  return {"real B" + std::to_string(width) + " draft/verify/commit decode",
          wallSeconds, std::move(lanes)};
}

WarmupStepResult Runtime::warmupCompositeStateRestore() {
  constexpr uint64_t id = std::numeric_limits<uint64_t>::max() - 121;
  constexpr uint32_t prefixTokens = 2 * kv::kPageTokens;
  constexpr uint32_t suffixTokens = kDecodeRows;
  constexpr uint32_t promptTokens = prefixTokens + suffixTokens;
  std::vector<uint32_t> warmupPrompt(promptTokens, 2);
  ModelRequest request;
  request.id = id;
  request.prompt = warmupPrompt;
  request.maxNewTokens = 8;
  std::shared_ptr<const CompositeState> cachedState;
  double wallSeconds = 0.0;
  beginColdRequest(request, 0);
  try {
    // Deliberately non-contiguous physical ids exercise page-table lookup.
    const std::vector<uint32_t> pages{12, 10, 11};
    requireRunwayPages(impl_->kvPages, pages);
    BatchPlan plan{.kind = WorkKind::Prefill,
                   .items = {{id, prefixTokens}},
                   .decodeStage = DecodeStage::Regular};
    ModelBatchItem item = warmupItem(id, 0, prefixTokens, pages);
    item.inputTokens =
        std::span<const uint32_t>(request.prompt).first(prefixTokens);
    static_cast<void>(prefill(plan, std::span<const ModelBatchItem>(&item, 1)));
    wallSeconds = impl_->counters.lastPrefillWallSeconds;
    cachedState = snapshot(id);
    if (!cachedState)
      throw metal::MetalAllocationError("prefix warmup state allocation failed");
    end(id);
    beginColdRequest(request, 1);
    if (beginRestore(id, prefixTokens, cachedState, true, {}))
      throw std::logic_error("a resident state restore returned a read");
    setDraftContextPlan(id, planDraftContext(prefixTokens, promptTokens, {}));
    const auto &restored = impl_->states.metadata(1).lengths;
    if (restored.targetTokens != prefixTokens ||
        !restored.hasCompleteDraftWindow(kDraftCacheStride)) {
      throw std::runtime_error("prefix restore length mismatch");
    }

    // Continue from committed KV history. This M8 command teacher-forces a
    // new chunk, then the real speculative cycle overwrites its speculative
    // page suffix and advances only the accepted commit length.
    BatchPlan suffixPlan{.kind = WorkKind::Prefill,
                         .items = {{id, suffixTokens}},
                         .decodeStage = DecodeStage::Regular};
    ModelBatchItem suffix = warmupItem(id, prefixTokens, suffixTokens, pages);
    suffix.inputTokens = std::span<const uint32_t>(request.prompt)
                             .subspan(prefixTokens, suffixTokens);
    requireLanesSucceeded(
        prefill(suffixPlan, std::span<const ModelBatchItem>(&suffix, 1)));
    prepareWarmupDecode(id, warmupPrompt.back());
    const double continuationWallSeconds =
        impl_->counters.lastPrefillWallSeconds;
    wallSeconds += continuationWallSeconds;
    BatchPlan decodePlan{.kind = WorkKind::Decode,
                         .items = {{id, 0}},
                         .decodeStage = DecodeStage::Regular};
    ModelBatchItem decodeItem = warmupItem(id, promptTokens, 0, pages);
    auto decoded =
        decode(decodePlan, std::span<const ModelBatchItem>(&decodeItem, 1));
    requireLanesSucceeded(decoded);
    const double historicalDecodeWallSeconds =
        impl_->counters.lastDecodeWallSeconds;
    wallSeconds += historicalDecodeWallSeconds;
    const auto &continued = impl_->states.metadata(1).lengths;
    if (decoded.size() != 1 || decoded[0].outputTokens.empty() ||
        !continued.hasCompleteDraftWindow(kDraftCacheStride) ||
        continued.targetTokens <= promptTokens ||
        continued.targetTokens !=
            promptTokens + decoded[0].outputTokens.size() -
                decoded[0].outputTokensWithoutKv) {
      throw std::runtime_error(
          "restored historical prefix did not continue exactly");
    }
    end(id);
  } catch (...) {
    end(id);
    throw;
  }
  return {"real paged-KV state restore, arbitrary page table, lane move, "
          "bounded restore continuation, and decode",
          wallSeconds, {}};
}

ModelMemoryActual Runtime::actualRuntimeMemory() const {
  const QwenTargetScratchBytes target = impl_->targetModel.allocatedScratchBytes();
  return {impl_->states.actualAllocatedBytes(), impl_->prefillArena->bytes() + target.prefill,
          impl_->decodeArena->bytes() + target.decode, impl_->states.stagingBytes()};
}

ModelTelemetry Runtime::telemetry() const noexcept {
  ModelTelemetry result = impl_->counters;
  result.stateAllocatedBytes = impl_->states.actualAllocatedBytes();
  result.idleGdnCells = impl_->states.idleCells();
  result.idleDraftRings = impl_->states.idleRings();
  result.idleContextWindows = impl_->states.idleWindows();
  result.visionArenaBytes = impl_->vision ? impl_->vision->arenaBytes() : 0;
  result.embeddingCacheBytes = impl_->embeddingCacheBytes;
  result.stateHeldImageBytes = impl_->heldRowsBytes(false);
  for (const auto &[_, held] : impl_->imageRows) {
    if (const std::shared_ptr<const Impl::ImageRows> rows = held.lock())
      result.imageRowsBytes += rows->pixels.sizeBytes() + rows->embeddings.sizeBytes();
  }
  return result;
}

std::vector<ops::SwiGluProjections> aneFfnLayers(const LoadedModel &model) {
  std::vector<ops::SwiGluProjections> layers;
  if (const auto *dense = std::get_if<Qwen3_8Weights>(&model.target))
    for (const Qwen3_8LayerWeights &layer : dense->layers)
      layers.push_back({&layer.gateProjection, &layer.upProjection, &layer.downProjection});
  return layers;
}

void withPrefillArena(
    MetalBackend &backend, const LoadedModel &model, const ops::ExecutionPlans &operators, kv::Format format,
    const std::function<void(const ops::PrefillFfnBuffers &, const std::array<MetalBuffer, 2> &)> &use) {
  // A prefill arena holds no state cell, so either state format sizes it.
  const PrefillArena arena(backend, RuntimeGeometry::from(model, format, ops::GdnStateFormat::Float32),
                           operators);
  const QwenTargetPrefillBuffers buffers = prefillBuffers(arena, 0);
  use(buffers.ffn(), buffers.hidden);
}

ModelMemoryPlan plannedRuntimeMemory(const LoadedModel &model,
                                     const ops::ExecutionPlans &operators,
                                     kv::Format kvFormat,
                                     ops::GdnStateFormat stateFormat) {
  requireCompatibleModel(model);
  const RuntimeGeometry geometry = RuntimeGeometry::from(model, kvFormat, stateFormat);
  return {model.stateLayout(stateFormat).laneBytes(),
          plannedPrefillBytes(geometry, operators),
          plannedDecodeBytes(geometry, operators)};
}

std::unique_ptr<RuntimeModel> createRuntime(RuntimeContext context) {
  return std::make_unique<Runtime>(std::move(context));
}

#ifdef SPLASH_RUNTIME_INSTRUMENTATION
std::vector<uint32_t> RuntimeInstrumentation::lastPleRows(const Runtime &runtime, uint32_t lane) {
  const QwenTarget &target = runtime.impl_->targetModel;
  return target.qwen4Steps_ ? target.qwen4Steps_->verifyPleRows(lane) : std::vector<uint32_t>{};
}
#endif

} // namespace splash::model

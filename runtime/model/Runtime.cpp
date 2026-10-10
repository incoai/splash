#include "model/Runtime.hpp"
#include "AwakeClock.hpp"
#include "model/QwenState.hpp"
#include "model/QwenTarget.hpp"
#include "model/RuntimeArenas.hpp"

#include "metal/CommandGraph.hpp"
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
#include <cstring>
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
      : ticket_(std::move(ticket)), completion_(std::move(completion)),
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
    // Work the completion added takes wall time, but is no sample of the
    // command's rows.
    if (timing.wallSeconds != commandSeconds)
      representativePrefillTiming_ = false;
    return results;
  }

  double wallMilliseconds() const noexcept override {
    return wallMilliseconds_;
  }
  bool prefillTimingIsRepresentative() const noexcept override {
    return representativePrefillTiming_;
  }

private:
  CommandTicket ticket_;
  Completion completion_;
  double wallMilliseconds_ = 0.0;
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

// A prefill arena's tensors as the draft's context prefill reads them.
DFlashPrefillBuffers draftPrefillBuffers(const PrefillArena &arena) {
  return {arena.get(PrefillTensor::Captured), prefillScratch(arena),
          arena.get(PrefillTensor::ContextProjected), arena.get(PrefillTensor::ContextHidden),
          arena.get(PrefillTensor::ContextKv), arena.get(PrefillTensor::DraftRopeCos),
          arena.get(PrefillTensor::DraftRopeSin)};
}

// A prefill arena's tensors as the target's prefill reads them.
QwenTargetPrefillBuffers prefillBuffers(const PrefillArena &arena) {
  QwenTargetPrefillBuffers buffers;
  buffers.linearScratch = prefillScratch(arena);
  buffers.hidden = {arena.get(PrefillTensor::Hidden0), arena.get(PrefillTensor::Hidden1)};
  buffers.normalized = arena.get(PrefillTensor::Normalized);
  buffers.captured = arena.get(PrefillTensor::Captured);
  buffers.gdnPacked = arena.get(PrefillTensor::GdnPacked);
  buffers.gdnQueries = arena.get(PrefillTensor::GdnQueries);
  buffers.gdnKeys = arena.get(PrefillTensor::GdnKeys);
  buffers.gdnValues = arena.get(PrefillTensor::GdnValues);
  buffers.gdnDecay = arena.get(PrefillTensor::GdnDecay);
  buffers.gdnBeta = arena.get(PrefillTensor::GdnBeta);
  buffers.recurrent = arena.get(PrefillTensor::Recurrent);
  buffers.gdnHidden = arena.get(PrefillTensor::GdnHidden);
  buffers.gdnOutput = arena.get(PrefillTensor::GdnOutput);
  buffers.denseGateScratch = arena.get(PrefillTensor::GateIntermediate);
  buffers.denseIntermediate = arena.get(PrefillTensor::Intermediate);
  buffers.fullPacked = arena.get(PrefillTensor::FullPacked);
  buffers.fullQueries = arena.get(PrefillTensor::FullQueries);
  buffers.fullAttention = arena.get(PrefillTensor::FullAttention);
  buffers.attentionPartials = arena.get(PrefillTensor::AttentionPartials);
  buffers.attentionStatistics = arena.get(PrefillTensor::AttentionStatistics);
  buffers.attentionHidden = arena.get(PrefillTensor::AttentionHidden);
  buffers.attentionOutput = arena.get(PrefillTensor::AttentionOutput);
  buffers.ropeCos = arena.get(PrefillTensor::RopeCos);
  buffers.ropeSin = arena.get(PrefillTensor::RopeSin);
  buffers.chunkKeys = arena.get(PrefillTensor::ChunkKeys);
  buffers.chunkValues = arena.get(PrefillTensor::ChunkValues);
  buffers.moe = arena.moeScratch();
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
  };

  struct DecodeLaneResult final {
    Request *request = nullptr;
    uint32_t retained = 0;
    uint32_t accepted = 0;
    uint32_t currentAnchor = 0;
    uint32_t maximumRetained = 0;
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
  DFlashDraft draftModel;
  ops::AneFfn *aneFfn;
  explicit Impl(RuntimeContext value)
      : backend(value.backend),
        model(value.model),
        geometry(RuntimeGeometry::from(value.model, value.kvPages.layout().format)),
        operators(value.operators),
        kvPages(value.kvPages),
        states(value.stateStorage),
        sampling(geometry.target.vocabularySize),
        targetModel(std::visit(
                        [&](const auto &weights) {
                          return QwenTarget(weights, geometry.target,
                                            value.backend, operators);
                        },
                        value.model.target)),
        draftModel(value.model.draft, value.backend, operators),
        aneFfn(value.aneFfn) {
    if (states.layout() != model.stateLayout() ||
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
  // lane and a greedy one reach every kernel the first-token and verify
  // selections dispatch.
  void preparePolicyPipelines() const {
    const ops::SamplingPolicy sampled{.topK = 0,
                                      .temperature = 1.0F,
                                      .topP = 0.95F,
                                      .constrained = true,
                                      .penalties = {1.1F, 0.5F, 0.5F},
                                      .minP = 0.05F};
    const std::array<ops::SamplingPolicy, 2> policies{sampled, {}};
    const std::array<uint32_t, 2> stateLanes{0, 1};
    const ops::PenaltyTable penalties{penaltyTable, stateLanes};
    CommandGraph graph;
    sampling.addInitial(graph, policies, samplingBuffers(2), 0,
                        geometry.target.stopTokens[0],
                        geometry.target.stopTokens[1], penalties);
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
    return entry.sampling.temperature > 0.0F;
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
  holdStraddledRows(const Request &entry, std::shared_ptr<const CompositeState> state) {
    std::erase_if(stateHolds, [](const std::weak_ptr<const HeldState> &hold) {
      return hold.expired();
    });
    const uint64_t boundary = states.metadata(entry.stateLane).lengths.targetTokens;
    for (const ImageState &image : entry.images) {
      // The chunk that ended at the boundary encoded the image it reached.
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
  // the chunk's placeholder embedding rows with the image rows. Text-only
  // requests add no dispatches.
  void addImageRows(CommandGraph &graph, Request &entry,
                    const ModelBatchItem &item, uint32_t rowBegin) {
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
          {static_cast<uint32_t>(begin - image.span.offset), width, 0},
          prefillArena->get(PrefillTensor::Hidden0),
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
        std::move(draftSin), {targetRows, draftRows}, kPrefillRows);
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

  static uint32_t captureRows(const DispatchDraftCapturePlan &captures) {
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

  static ops::SamplingPolicy samplingPolicy(const Request &entry) noexcept {
    return {entry.sampling.topK, entry.sampling.temperature,
            entry.sampling.topP, entry.constraint == ConstraintMode::TokenMask,
            (entry.flags & RequestIgnoreEndOfSequence) != 0,
            samplingPenalties(entry), entry.sampling.minP};
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
  // the lane's words is encoded after this.
  void commitSelected(Request &entry, std::span<const uint32_t> tokens) {
    if (tokens.empty())
      throw std::logic_error("no selected token to commit");
    if (samplingPenalties(entry).active())
      ops::Sampling::countPenaltyTokens(penaltyWords(entry.stateLane), tokens);
    entry.pendingToken = tokens.back();
  }

  // A selection outside the vocabulary is the sampling kernels' sentinel for a
  // non-finite logit row: a numerical outcome of this request, which it reports
  // as its lane failure (ModelStepResult::failure) so the batch survives.
  [[nodiscard]] std::string invalidSelection(std::span<const uint32_t> tokens) const {
    const auto found = std::find_if(tokens.begin(), tokens.end(), [&](uint32_t token) {
      return token >= geometry.target.vocabularySize;
    });
    if (found == tokens.end())
      return {};
    return "target selected out-of-vocabulary token " + std::to_string(*found) +
           " from a non-finite logit row";
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
        backend.submitCommandAsync(graph.dispatches(), std::move(completion));
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
        result.failure = invalidSelection({&token, 1});
        if (!result.failure.empty())
          continue;
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

  struct RaggedPrefillSequence final {
    Request *entry = nullptr;
    const ModelBatchItem *item = nullptr;
    uint32_t lane = 0;
    uint32_t rowBegin = 0;
    uint32_t attentionStride = 0;
    uint64_t queryOffset = 0;
    uint64_t kvOffset = 0;
    uint32_t captureBegin = 0;
    ChunkedPrefillParams chunk;
    MetalBuffer pageTable;
    DispatchDraftCapturePlan captures;
  };

  struct RaggedPrefillBatch final {
    std::vector<RaggedPrefillSequence> sequences;
    uint32_t rows = 0;
    uint32_t capturedRows = 0;
  };

  MetalBuffer prefillU16(const MetalBuffer &tensor, uint32_t begin,
                         uint32_t rows, uint32_t width) const {
    return backend.view(tensor, bytesFor<uint16_t>(uint64_t{begin} * width),
                        bytesFor<uint16_t>(uint64_t{rows} * width));
  }

  RaggedPrefillBatch
  prepareRaggedPrefill(std::span<const ModelBatchItem> items,
                       std::array<Request *, kLaneCount> &entries) {
    RaggedPrefillBatch batch;
    batch.sequences.reserve(items.size());
    uint64_t queryOffset = 0;
    uint64_t kvOffset = 0;
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      const ModelBatchItem &item = items[lane];
      Request &entry = request(item.requestId);
      if (item.tokenCount > kPrefillRows ||
          item.logicalPosition > entry.promptTokens ||
          item.tokenCount > entry.promptTokens - item.logicalPosition ||
          !entry.resident) {
        throw std::invalid_argument("invalid ragged Qwen prefill item");
      }
      const QwenLaneMetadata &metadata = states.metadata(entry.stateLane);
      if (metadata.requestId != entry.id ||
          metadata.lengths.targetTokens != item.logicalPosition) {
        throw std::logic_error("ragged prefill state length is not exact");
      }
      if (item.logicalPosition == 0)
        states.clearForColdStart(entry.stateLane);
      if (item.tokenCount > kPrefillRows - batch.rows) {
        throw std::invalid_argument("ragged prefill exceeds actual-row budget");
      }
      auto captures = activeDraftCaptures(entry, item);
      const uint32_t capturedRows = captureRows(captures);
      const uint32_t attentionStride =
          ((item.tokenCount + kTileRows - 1) / kTileRows) * kTileRows;
      const ChunkedPrefillParams chunk =
          chunkParams(item.logicalPosition, item.tokenCount, attentionStride,
                      item.pageTable);
      MetalBuffer pageTable = synchronizedPageTable(entry, item);
      batch.sequences.push_back({&entry, &item, lane, batch.rows,
                                 attentionStride, queryOffset, kvOffset,
                                 batch.capturedRows, chunk, std::move(pageTable),
                                 std::move(captures)});
      entries[lane] = &entry;
      batch.rows += item.tokenCount;
      batch.capturedRows += capturedRows;
      queryOffset += bytesFor<uint16_t>(
          uint64_t{geometry.target.attentionQueryHeads} * attentionStride *
          geometry.target.attentionHeadDimension);
      kvOffset += bytesFor<uint16_t>(
          uint64_t{geometry.target.attentionKvHeads} * attentionStride *
          geometry.target.attentionHeadDimension);
    }
    if (!batch.rows ||
        queryOffset >
            prefillArena->get(PrefillTensor::FullQueries).sizeBytes() ||
        kvOffset > prefillArena->get(PrefillTensor::ChunkKeys).sizeBytes()) {
      throw std::logic_error("ragged prefill scratch geometry overflowed");
    }

    auto *input =
        contents<uint32_t>(prefillArena->get(PrefillTensor::InputTokens),
                           "ragged prefill input tokens");
    auto *targetPositions =
        contents<uint32_t>(prefillArena->get(PrefillTensor::TargetPositions),
                           "target RoPE positions");
    auto *draftPositions =
        contents<uint32_t>(prefillArena->get(PrefillTensor::DraftPositions),
                           "draft RoPE positions");
    for (const RaggedPrefillSequence &sequence : batch.sequences) {
      const ModelBatchItem &item = *sequence.item;
      std::copy(item.inputTokens.begin(), item.inputTokens.end(),
                input + sequence.rowBegin);
      for (uint32_t localRow = 0; localRow < item.tokenCount; ++localRow) {
        const uint32_t row = sequence.rowBegin + localRow;
        if (input[row] >= geometry.target.vocabularySize) {
          throw std::invalid_argument("prompt token is out of vocabulary");
        }
        const std::array<uint32_t, 3> rotary =
            ropePosition(*sequence.entry, item.logicalPosition + localRow);
        std::copy(rotary.begin(), rotary.end(), targetPositions + row * 3);
      }
      for (const DispatchDraftCaptureSpan &capture : sequence.captures) {
        for (uint32_t row = capture.absoluteBegin; row < capture.absoluteEnd;
             ++row) {
          const uint32_t compactRow = sequence.captureBegin +
                                      capture.compactDestinationRow + row -
                                      capture.absoluteBegin;
          draftPositions[compactRow] = row;
        }
      }
    }
    return batch;
  }

  // Computes a restored lane's draft rings from its context window.
  void addDraftRebuild(CommandGraph &graph, uint32_t lane,
                       const Request::DraftRebuild &rebuild) const {
    auto p = [&](PrefillTensor tensor) { return prefillArena->get(tensor); };
    ops::RoPE::addDraftRangeTables(graph, p(PrefillTensor::DraftInverseFrequencies),
                                   p(PrefillTensor::DraftRopeCos), p(PrefillTensor::DraftRopeSin),
                                   rebuild.rows, rebuild.begin, kPrefillRows);
    draftModel.addContextRebuild(graph, draftPrefillBuffers(*prefillArena), states.window(lane),
                                 states.draft(lane), rebuild.rows, rebuild.begin);
  }

  void addRaggedDraftContext(CommandGraph &graph,
                             const RaggedPrefillBatch &batch) {
    if (!batch.capturedRows)
      return;
    std::array<DFlashPrefillSpan, kLaneCount * 2> spans{};
    uint32_t spanCount = 0;
    for (const RaggedPrefillSequence &sequence : batch.sequences) {
      for (const DispatchDraftCaptureSpan &capture : sequence.captures) {
        DFlashPrefillSpan &span = spans.at(spanCount++);
        span.compactRow = sequence.captureBegin + capture.compactDestinationRow;
        span.rows = capture.absoluteEnd - capture.absoluteBegin;
        span.startPosition = capture.absoluteBegin;
        span.ring = states.draft(sequence.entry->stateLane);
        span.window = states.window(sequence.entry->stateLane);
      }
    }
    draftModel.addContextPrefill(graph, draftPrefillBuffers(*prefillArena), batch.capturedRows,
                                 std::span(spans).first(spanCount));
  }

  // Returns each lane's draft captures, indexed like `entries`.
  std::array<DispatchDraftCapturePlan, kLaneCount>
  encodeRaggedPrefillGraph(CommandGraph &graph,
                           std::span<const ModelBatchItem> items,
                           std::array<Request *, kLaneCount> &entries) {
    RaggedPrefillBatch batch = prepareRaggedPrefill(items, entries);
    auto p = [&](PrefillTensor tensor) { return prefillArena->get(tensor); };

    // A restored lane's rings first: they pass through the context buffers
    // and the draft RoPE tables before this chunk's rows do. Once the window
    // is full, this chunk's window store and ring writes reuse the slots of
    // the oldest restored positions, so the rebuild comes before them too.
    for (const RaggedPrefillSequence &sequence : batch.sequences)
      if (const auto &rebuild = sequence.entry->draftRebuild)
        addDraftRebuild(graph, sequence.entry->stateLane, *rebuild);

    addRopeTables(graph, p(PrefillTensor::TargetPositions), batch.rows,
                  p(PrefillTensor::DraftPositions), batch.capturedRows,
                  p(PrefillTensor::RopeCos), p(PrefillTensor::RopeSin),
                  p(PrefillTensor::DraftRopeCos),
                  p(PrefillTensor::DraftRopeSin));

    targetModel.addEmbedding(graph, p(PrefillTensor::InputTokens),
                             p(PrefillTensor::Hidden0), batch.rows);
    for (const RaggedPrefillSequence &sequence : batch.sequences) {
      addImageRows(graph, *sequence.entry, *sequence.item, sequence.rowBegin);
    }

    std::array<QwenTargetPrefillSequence, kLaneCount> modelSequences{};
    const uint32_t modelSequenceCount =
        static_cast<uint32_t>(batch.sequences.size());
    const uint64_t stateBindingCount = uint64_t{modelSequenceCount} *
                                       geometry.target.stateLayout.layers;
    std::vector<MetalBuffer> convolutionIn(stateBindingCount);
    std::vector<MetalBuffer> convolutionOut(stateBindingCount);
    std::vector<MetalBuffer> recurrentIn(stateBindingCount);
    std::vector<MetalBuffer> recurrentOut(stateBindingCount);
    for (uint32_t lane = 0; lane < batch.sequences.size(); ++lane) {
      const RaggedPrefillSequence &sequence = batch.sequences[lane];
      QwenTargetPrefillSequence &destination = modelSequences[lane];
      destination.rowBegin = sequence.rowBegin;
      destination.rows = sequence.item->tokenCount;
      destination.attentionStride = sequence.attentionStride;
      destination.queryOffset = sequence.queryOffset;
      destination.kvOffset = sequence.kvOffset;
      destination.chunk = sequence.chunk;
      destination.pageTable = sequence.pageTable;
      const uint32_t gdnLayers = geometry.target.stateLayout.layers;
      const uint64_t stateBegin = uint64_t{lane} * gdnLayers;
      destination.convolutionIn =
          std::span(convolutionIn).subspan(stateBegin, gdnLayers);
      destination.convolutionOut =
          std::span(convolutionOut).subspan(stateBegin, gdnLayers);
      destination.recurrentIn =
          std::span(recurrentIn).subspan(stateBegin, gdnLayers);
      destination.recurrentOut =
          std::span(recurrentOut).subspan(stateBegin, gdnLayers);
      const GdnParityBuffers &in = states.current(sequence.entry->stateLane);
      const GdnParityBuffers &out = states.next(sequence.entry->stateLane);
      for (uint32_t layer = 0; layer < gdnLayers; ++layer) {
        convolutionIn[stateBegin + layer] = in.convolutionLayers[layer];
        convolutionOut[stateBegin + layer] = out.convolutionLayers[layer];
        recurrentIn[stateBegin + layer] = in.recurrentLayers[layer];
        recurrentOut[stateBegin + layer] = out.recurrentLayers[layer];
      }
      destination.captureCount = sequence.captures.size();
      for (uint32_t index = 0; index < sequence.captures.size(); ++index) {
        const DispatchDraftCaptureSpan &capture = sequence.captures[index];
        destination.captures[index] = {
            sequence.rowBegin +
                static_cast<uint32_t>(capture.absoluteBegin -
                                      sequence.item->logicalPosition),
            sequence.captureBegin + capture.compactDestinationRow,
            capture.absoluteEnd - capture.absoluteBegin};
      }
    }
    QwenTargetPrefillBuffers buffers = prefillBuffers(*prefillArena);
    const MetalBuffer finalHidden = targetModel.addPrefill(
        graph, std::move(buffers),
        std::span(modelSequences).first(batch.sequences.size()), batch.rows,
        kvPages.layers(), aneFfn);
    addRaggedDraftContext(graph, batch);

    // A lane that finishes its prompt copies the prompt's last row to row 0
    // of its Hidden0 block. A constrained lane's completion captures that
    // row into finalTargetHidden (captureFinalHidden), which holds it until
    // the first mask. The others share one head: a score lane reads raw
    // logits at the final prompt position, and a policy lane selects its
    // first token.
    std::array<InitialSelection, kLaneCount> selections{};
    uint32_t selectionCount = 0;
    for (const RaggedPrefillSequence &sequence : batch.sequences) {
      Request &entry = *sequence.entry;
      const ModelBatchItem &item = *sequence.item;
      if (entry.replayingGeneration ||
          item.logicalPosition + item.tokenCount != entry.promptTokens)
        continue;
      const uint32_t hidden = geometry.target.hiddenSize;
      ops::RowCopy::add(
          graph,
          prefillU16(finalHidden, sequence.rowBegin, item.tokenCount, hidden),
          {item.tokenCount - 1, hidden, 0},
          decodeArena->get(sequence.lane, DecodeTensor::Hidden0),
          {0, hidden, 0}, 1, hidden);
      if (entry.constraint != ConstraintMode::None)
        continue;
      const bool scoring = !entry.scoreTokens.empty();
      if (!scoring && samplingEnabled(entry))
        uploadInitialUniform(entry, sequence.lane);
      selections[selectionCount++] = {&entry, sequence.lane, !scoring};
    }
    if (selectionCount) {
      encodeInitialSelections(
          graph, std::span(selections).first(selectionCount),
          static_cast<uint32_t>(batch.sequences.size()));
    }
    std::array<DispatchDraftCapturePlan, kLaneCount> captures{};
    for (const RaggedPrefillSequence &sequence : batch.sequences)
      captures[sequence.lane] = sequence.captures;
    return captures;
  }

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
    draftModel.addDecode(graph, std::move(buffers),
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
    draftModel.addSelection(
        graph,
        {d(DecodeTensor::Logits), d(DecodeTensor::TopPartialIds),
         d(DecodeTensor::TopPartialValues), d(DecodeTensor::Candidates),
         d(DecodeTensor::Unary), d(DecodeTensor::SelectorHidden),
         d(DecodeTensor::SamplingUniforms), d(DecodeTensor::ProposedTokens),
         d(DecodeTensor::ProposalProbs)},
        std::span(anchors).first(lanes), std::span(policies).first(lanes));
  }

  void encodeTargetVerifyBatchForward(CommandGraph &graph,
                                      std::span<Request *const> entries,
                                      std::span<const ModelBatchItem> items) {
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
    const uint32_t gdnLayers = geometry.target.stateLayout.layers;
    const uint32_t attentionLayers =
        geometry.target.kvLayout.attentionLayers;
    std::vector<MetalBuffer> gdnPacked(gdnLayers);
    std::vector<MetalBuffer> gdnMixed(gdnLayers);
    std::vector<MetalBuffer> gdnDecay(gdnLayers);
    std::vector<MetalBuffer> gdnBeta(gdnLayers);
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
    buffers.capturedTargetHidden = d(DecodeTensor::CapturedTargetHidden);
    buffers.finalHidden = d(DecodeTensor::FinalHidden);
    buffers.logits = d(DecodeTensor::Logits);
    buffers.denseGateScratch = decodeArena->gateScratch();
    buffers.gdnPacked = gdnPacked;
    buffers.gdnMixed = gdnMixed;
    buffers.gdnDecay = gdnDecay;
    buffers.gdnBeta = gdnBeta;
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
    }
    for (uint32_t layer = 0; layer < gdnLayers; ++layer) {
      gdnPacked[layer] = decodeArena->gdnBatchSlice(
          DecodeTensor::VerifyPackedBase, layer, storage);
      gdnMixed[layer] = decodeArena->gdnBatchSlice(
          DecodeTensor::VerifyMixedBase, layer, storage);
      gdnDecay[layer] = decodeArena->gdnBatchSlice(
          DecodeTensor::VerifyDecayBase, layer, storage);
      gdnBeta[layer] = decodeArena->gdnBatchSlice(
          DecodeTensor::VerifyBetaBase, layer, storage);
    }
    for (uint32_t layer = 0; layer < attentionLayers; ++layer) {
      chunkKeys[layer] = decodeArena->attentionBatchSlice(
          DecodeTensor::ChunkKeysBase, layer, storage);
      chunkValues[layer] = decodeArena->attentionBatchSlice(
          DecodeTensor::ChunkValuesBase, layer, storage);
    }
    targetModel.addVerify(graph, std::move(buffers), kvPages.layers(),
                          std::span(chunks).first(lanes), lanes);
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
    draftModel.addContextCommit(graph, std::move(buffers),
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

  void encodeBatchGdnCommit(CommandGraph &graph,
                            std::span<Request *const> lanes) {
    if (lanes.empty() || lanes.size() > kLaneCount)
      throw std::invalid_argument("invalid GDN commit batch");
    const uint32_t width = static_cast<uint32_t>(lanes.size());
    std::array<MetalBuffer, kLaneCount> currentStates;
    std::array<MetalBuffer, kLaneCount> nextStates;
    for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
      Request *entry = lanes[std::min(lane, width - 1)];
      if (!entry)
        throw std::invalid_argument("empty GDN commit lane");
      currentStates[lane] = states.current(entry->stateLane).stateBase;
      nextStates[lane] = states.next(entry->stateLane).stateBase;
    }
    targetModel.addStateCommit(
        graph,
        {decodeArena->gdnStorage(DecodeTensor::VerifyPackedBase),
         decodeArena->gdnStorage(DecodeTensor::VerifyMixedBase),
         decodeArena->gdnStorage(DecodeTensor::VerifyDecayBase),
         decodeArena->gdnStorage(DecodeTensor::VerifyBetaBase), currentStates,
         nextStates,
         decodeArena->batchSlice(DecodeTensor::RetainedCount, width)},
        width);
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
      // The retained target tokens end with the next anchor. A non-finite
      // target row can also accept a sentinel draft proposal as an interior
      // token, so every retained token is checked.
      const uint32_t *targetTokens = contents<uint32_t>(
          d(DecodeTensor::OutputTokens), "target output tokens");
      laneResult.failure = invalidSelection({targetTokens, laneResult.retained});
    }

    std::vector<ModelStepResult> results;
    results.reserve(items.size());
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      DecodeLaneResult &laneResult = lanes[lane];
      Request &entry = *laneResult.request;
      if (!laneResult.failure.empty()) {
        // The cycle's state and tokens are not committed; the engine ends
        // the request.
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

      states.swapParity(entry.stateLane);
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
      results.push_back({entry.id,
                         0,
                         std::move(output),
                         false,
                         DecodeStage::Regular,
                         kDraftProposalTokens,
                         std::min(laneResult.accepted, laneResult.retained - 1)});
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
          impl_.encodeTargetVerifyBatchPolicy(commit,
                                              {entries.data(), lanes_.size()});
          impl_.encodeBatchAcceptance(commit, {entries.data(), lanes_.size()},
                                      {maximumRetained.data(), lanes_.size()});
          impl_.encodeBatchGdnCommit(commit, {entries.data(), lanes_.size()});
          impl_.encodeDraftStateCommitBatch(
              commit, {entries.data(), lanes_.size()}, items_);
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
      command_ = impl_.backend.submitCommandAsync(graph.dispatches(), wake_);
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
  impl_->states.releaseLane(entry.stateLane, requestId);
  impl_->pageTableBindings[entry.stateLane] = {};
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
  entry.maxNewTokens = request.maxNewTokens;
  entry.sampling = request.sampling;
  entry.constraint = request.constraint;
  entry.flags = request.flags;
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
  const auto captures = impl_->encodeRaggedPrefillGraph(graph, items, entries);
  const bool encodesImages = std::any_of(
      entries.begin(), entries.begin() + items.size(), [](const auto *entry) {
        return std::any_of(entry->images.begin(), entry->images.end(),
                           [](const auto &image) {
                             return image.rows && image.rows->encoding;
                           });
      });
  std::vector<ModelBatchItem> copiedItems(items.begin(), items.end());
  CommandTicket command =
      impl_->aneFfn ? impl_->aneFfn->commit(graph, std::move(completion))
                    : impl_->backend.submitCommandAsync(graph.command(), std::move(completion));
  Impl *impl = impl_.get();
  auto finish = [impl, entries, captures, draws,
                 items = std::move(copiedItems)](CommandTiming &timing) mutable {
    // A chunk whose Neural Engine work failed holds no usable outputs, and
    // the split has stopped: the GPU runs the chunk again alone, which
    // computes what it would have the first time. Nothing of the chunk is
    // committed before the code below: state parity, lengths, the selected
    // token and the image rows' state. The rerun writes the same KV and draft
    // rows; a restored lane's rebuild may read window slots the failed run
    // filled with its own rows, but each belongs to a position the rerun
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
      static_cast<void>(impl->encodeRaggedPrefillGraph(again, items, entries));
      const CommandTiming rerun = impl->backend.submitCommandAsync(again.command()).wait();
      timing.gpuSeconds += rerun.gpuSeconds;
      timing.wallSeconds += rerun.wallSeconds;
      ++impl->counters.aneFfnReruns;
    }
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
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

    std::vector<ModelStepResult> results;
    results.reserve(items.size());
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      Impl::Request &entry = *entries[lane];
      const ModelBatchItem &item = items[lane];
      const uint64_t nextLength = item.logicalPosition + item.tokenCount;
      // The anchor this chunk selects when it completes a generation prompt.
      std::optional<uint32_t> selected;
      if (nextLength == entry.promptTokens && !entry.replayingGeneration &&
          entry.scoreTokens.empty() && entry.constraint == ConstraintMode::None) {
        selected = impl->initialToken(lane);
        if (std::string failure = impl->invalidSelection({&*selected, 1});
            !failure.empty()) {
          // The chunk's state is not committed; the engine ends the
          // request.
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
      for (const DispatchDraftCaptureSpan &capture : captures[lane]) {
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
          item.tokenCount - Impl::captureRows(captures[lane]);
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
      results.push_back(std::move(result));
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
  impl_->encodeBatchEmbedding(commandGraph, DecodeTensor::DraftInputTokens,
                              DecodeTensor::DraftHidden0, width);
  impl_->encodeDraftBatchGraph(commandGraph, entries,
                               {logicalPositions.data(), width});
  if (constrained) {
    return std::make_unique<Impl::ConstrainedDecodeTicket>(
        *impl_, std::move(lanes), items, commandGraph, std::move(completion));
  }
  impl_->encodeBatchVerifyInput(commandGraph, width);
  impl_->encodeBatchEmbedding(commandGraph, DecodeTensor::InputTokens,
                              DecodeTensor::Hidden0, width);
  impl_->encodeTargetVerifyBatchForward(commandGraph, entries, items);
  impl_->encodeTargetVerifyBatchPolicy(commandGraph, entries);
  impl_->encodeBatchAcceptance(commandGraph, entries,
                               {maximumRetained.data(), width});
  impl_->encodeBatchGdnCommit(commandGraph, entries);
  impl_->encodeDraftStateCommitBatch(commandGraph, entries, items);

  std::vector<ModelBatchItem> copiedItems(items.begin(), items.end());
  Impl *impl = impl_.get();
  auto finish = [impl, lanes = std::move(lanes),
                 items = std::move(copiedItems)](CommandTiming timing) mutable {
    return impl->finalizeDecode(lanes, items, timing);
  };
  CommandTicket command = impl_->backend.submitCommandAsync(
      commandGraph.dispatches(), std::move(completion));
  return std::make_unique<DeferredMetalTicket>(std::move(command),
                                               std::move(finish));
}

uint32_t Runtime::residentLane(uint64_t requestId) {
  Impl::Request &entry = impl_->request(requestId);
  if (!entry.resident)
    throw std::logic_error("request is not resident");
  return entry.stateLane;
}

std::shared_ptr<const CompositeState> Runtime::snapshot(uint64_t requestId) {
  std::shared_ptr<const CompositeState> state =
      impl_->states.snapshot(residentLane(requestId));
  if (!state)
    return state;
  return impl_->holdStraddledRows(impl_->request(requestId), std::move(state));
}

uint64_t Runtime::snapshotBytes() const noexcept {
  return impl_->states.layout().cachedBytes();
}

bool Runtime::canSnapshotToDisk() const noexcept {
  return impl_->states.canSnapshotToDisk();
}

std::unique_ptr<StateOffload>
Runtime::snapshotToDisk(uint64_t requestId, std::function<void()> completion) {
  return impl_->states.snapshotToDisk(residentLane(requestId), std::move(completion));
}

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
  if (found->second.resident) {
    impl_->states.releaseLane(found->second.stateLane, requestId);
    impl_->pageTableBindings[found->second.stateLane] = {};
  }
  impl_->requests.erase(found);
}

namespace {

// A warmup step whose lane failed (a non-finite logit row) fails the warmup
// there, with the lane's reason.
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
  double wallSeconds = 0.0;
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
    auto result = prefill(plan, std::span<const ModelBatchItem>(&item, 1));
    wallSeconds = std::chrono::duration<double>(AwakeClock::now() - phaseStart).count();
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
          wallSeconds, std::move(lanes)};
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
  return {impl_->states.actualAllocatedBytes(), impl_->prefillArena->bytes(),
          impl_->decodeArena->bytes(), impl_->states.stagingBytes()};
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
  const PrefillArena arena(backend, RuntimeGeometry::from(model, format), operators);
  const QwenTargetPrefillBuffers buffers = prefillBuffers(arena);
  use(buffers.ffn(), buffers.hidden);
}

ModelMemoryPlan plannedRuntimeMemory(const LoadedModel &model,
                                     const ops::ExecutionPlans &operators,
                                     kv::Format format) {
  requireCompatibleModel(model);
  const RuntimeGeometry geometry = RuntimeGeometry::from(model, format);
  return {model.stateLayout().laneBytes(),
          plannedPrefillBytes(geometry, operators),
          plannedDecodeBytes(geometry, operators)};
}

std::unique_ptr<RuntimeModel> createRuntime(RuntimeContext context) {
  return std::make_unique<Runtime>(std::move(context));
}

} // namespace splash::model

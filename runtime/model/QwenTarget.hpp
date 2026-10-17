#pragma once

#include "Model.hpp"
#include "QwenHybridLayout.hpp"
#include "StateLayout.hpp"
#include "WeightStore.hpp"
#include "ops/GDN.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"
#include "ops/PagedAttention.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace splash::ops {
class AneFfn;
} // namespace splash::ops

namespace splash::kv {
class PageStorage;
} // namespace splash::kv

namespace splash::model {

struct Qwen3_8Layout;
struct Qwen3_8LayerWeights;
struct Qwen3_6MoeLayout;
struct Qwen3_6MoeLayerWeights;
struct Qwen4ExpWeights;
struct Qwen4ExpLayerWeights;
struct Qwen4HyperWeights;
struct Qwen4DraftHead;
struct Qwen4Scratch;
class Qwen4Steps;

// Both supported targets bind the same mixer tensors per hybrid layer; only
// the FFN differs between them.
struct QwenGdnWeights final {
  ops::Projection inputProjection;
  metal::MetalBuffer convolutionWeights;
  metal::MetalBuffer decay;
  metal::MetalBuffer timeBias;
  ops::NormWeights mixerNorm;
  ops::Projection outputProjection;
  // The value-head order of outputProjection's input columns, in which the
  // GDN writes its output.
  ops::GdnHeadOrder outputHeadOrder = ops::GdnHeadOrder::Grouped;
};

struct QwenAttentionWeights final {
  ops::Projection inputProjection;
  ops::NormWeights queryNorm;
  ops::NormWeights keyNorm;
  ops::Projection outputProjection;
};

using QwenMixerWeights = std::variant<QwenGdnWeights, QwenAttentionWeights>;

class ExpertStore;

// A Qwen target's weights outside its layers and the record of every file
// its weights were read from. A MoE target whose routed experts stream reads
// them through `experts` (model/ExpertStore.hpp).
struct QwenTargetWeightsBase {
  ops::NormWeights finalNorm;
  ops::Projection logitsProjection;
  ops::EmbeddingWeights tokenEmbedding;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
  std::string manifestFingerprintSha256;
  std::shared_ptr<ExpertStore> experts;
};

// The weights of a target of Layout, whose layers the family keeps in Layer.
template <class Layout, class Layer> struct QwenTargetWeights final : QwenTargetWeightsBase {
  Layout layout;
  std::vector<Layer> layers;
};

// Runtime-visible tensor geometry shared by the supported Qwen hybrid
// targets: the target's dimensions, the layers the draft reads and what its
// loaded weights add. It describes semantics only; operators remain
// responsible for choosing device-specific Metal pipelines and compute tiles.
struct QwenTargetGeometry final : QwenTargetDimensions {
  QwenTargetGeometry() = default;
  // A target of `dimensions` whose DFlash2 draft reads the output rows of
  // `captureLayers` (DFlashDraftLayout::targetLayerIds); more than the
  // geometry holds leave it invalid.
  QwenTargetGeometry(const QwenTargetDimensions &dimensions, std::span<const uint32_t> captureLayers)
      : QwenTargetDimensions(dimensions), captureLayerCount(static_cast<uint32_t>(captureLayers.size())) {
    if (captureLayerCount <= captureLayerValues.size())
      std::ranges::copy(captureLayers, captureLayerValues.begin());
  }

  // The format of most of the sparse MoE blocks' routed expert weights, and
  // whether every routed and shared expert weight is in a format the register
  // tile takes (ops::MoeShape).
  uint32_t moeExpertFormat = GGUF_FMT_COUNT;
  bool moeRegisterFormats = true;
  // The layers whose output rows the DFlash2 draft reads, ascending.
  std::array<uint32_t, ExecutionLimits::maximumDraftTargetLayers> captureLayerValues{};
  uint32_t captureLayerCount = 0;
  // Qwen3.8-Flash-Next's MTP head's full-attention blocks (0 or 1), whose KV
  // is the last attention layer of kvLayout (Qwen4ExpLayout::mtpLayers).
  uint32_t mtpLayers = 0;
  // With a draft vocabulary, the format of the vocabulary projection's rows
  // each state lane keeps of its request's context tokens
  // (Qwen4DraftVocabulary.hpp); GGUF_FMT_COUNT for none.
  uint32_t mtpDraftContextFormat = GGUF_FMT_COUNT;
  // The target's KV layout in the format the runtime stores KV in, and its
  // GDN state layout.
  kv::Layout kvLayout{};
  GdnStateLayout stateLayout{};
  // Qwen3.8-Flash-Next's PLE table when it streams from its file: the bytes
  // and format (GGUF_FMT_*) of its native rows, which a step stages
  // (Qwen4Scratch::pleStaging); zero bytes when the table is in memory.
  uint32_t pleStreamRowBytes = 0;
  uint32_t pleStreamFormat = 0;
  // Distinct operator requirements, collected from the loaded weights.
  std::vector<ops::ProjectionShape> prefillProjections;
  std::vector<ops::ProjectionShape> decodeProjections;
  std::vector<ops::ProjectionShape> gateUpProjections;

  [[nodiscard]] constexpr uint32_t gdnKeyWidth() const noexcept {
    return gdnKeyHeads * gdnHeadDimension;
  }
  [[nodiscard]] constexpr uint32_t capturedHiddenSize() const noexcept {
    return hiddenSize * captureLayerCount;
  }
  [[nodiscard]] constexpr ops::MoeShape moeShape() const noexcept {
    return {hiddenSize, experts, expertsPerToken, expertIntermediateSize, moeExpertFormat, moeRegisterFormats};
  }
  [[nodiscard]] constexpr std::span<const uint32_t>
  captureLayers() const noexcept {
    return {captureLayerValues.data(), captureLayerCount};
  }
  // The trunk's attention layers, before the MTP head's in the KV pages.
  [[nodiscard]] constexpr uint32_t trunkAttentionLayers() const noexcept {
    return kvLayout.attentionLayers - mtpLayers;
  }
  // The capture slot of `layer`, whose output the draft reads.
  [[nodiscard]] constexpr std::optional<uint32_t> captureSlot(uint32_t layer) const noexcept {
    const auto layers = captureLayers();
    const auto found = std::find(layers.begin(), layers.end(), layer);
    if (found == layers.end()) return std::nullopt;
    return static_cast<uint32_t>(found - layers.begin());
  }
  // Whether a layer's mixer adds its output to the layer's input rows, which
  // its FFN then adds to; the mixers of a target whose residual is
  // hyper-connected streams write their output to a branch of the target's
  // own scratch instead (QwenTarget::scratchBytes).
  [[nodiscard]] constexpr bool mixersAddToHidden() const noexcept { return !hyperConnections; }
  [[nodiscard]] constexpr ops::GdnShape gdnShape() const noexcept {
    return {gdnKeyHeads, gdnValueHeads, gdnHeadDimension,
            convolutionDimension, packedGdnWidth, gdnSigmoidGate};
  }
  // The layout itself was checked by requireQwenLayout when the target
  // loaded. The draft reads 1 to ExecutionLimits::maximumDraftTargetLayers
  // of its layers, ascending. The projection lists hold every projection the
  // weights dispatch, which each have sizes.
  [[nodiscard]] bool valid() const noexcept {
    const auto sized = [](const std::vector<ops::ProjectionShape> &shapes) {
      return !shapes.empty() && std::all_of(shapes.begin(), shapes.end(), [](const auto &shape) {
        return shape.outputSize && shape.inputSize;
      });
    };
    const auto captured = [&] {
      if (!captureLayerCount || captureLayerCount > ExecutionLimits::maximumDraftTargetLayers) return false;
      const auto ids = captureLayers();
      return std::ranges::adjacent_find(ids, std::ranges::greater_equal{}) == ids.end() && ids.back() < layers;
    };
    return captured() && mtpLayers <= 1 && mtpLayers <= kvLayout.attentionLayers &&
           stateLayout.layers + trunkAttentionLayers() == layers &&
           gdnShape().valid() &&
           kvLayout.kvHeads == attentionKvHeads &&
           kvLayout.headDimension == attentionHeadDimension &&
           sized(prefillProjections) && sized(decodeProjections) &&
           ((ffnKind == QwenFfnKind::Dense && intermediateSize && sized(gateUpProjections)) ||
            (ffnKind == QwenFfnKind::SparseMoe && moeShape().valid()));
  }
};

struct QwenTargetPrefillCapture final {
  uint32_t sourceStart = 0;
  uint32_t destinationStart = 0;
  uint32_t rows = 0;
};

// A state a prefill command captures at a row inside it (a state boundary):
// each GDN layer's convolution and recurrent state, as a command ending at
// that row stores them, in the views of a cache cell, and the target's
// auxiliary state (GdnStateLayout::auxiliaryBytes) in the cell's.
struct QwenTargetStateCapture final {
  uint32_t row = 0;
  std::span<const metal::MetalBuffer> convolution;
  std::span<const metal::MetalBuffer> recurrent;
  metal::MetalBuffer auxiliary;
};

// A request as a target step reads it beyond its rows: its tokens as far as
// the runtime has them, tokens[p] the token at position p (a prefill's, its
// prompt), and the RoPE position (t, h, w) of each of its positions. Only a
// target that makes inputs of its own of a step's tokens reads them
// (Qwen3.8-Flash-Next: its PLE n-grams and QSA blocks).
struct QwenTargetRequest final {
  std::span<const uint32_t> tokens;
  std::function<std::array<uint32_t, 3>(uint64_t)> rotary;
};

// A lane's rows of a prefill chunk: `tokens`, of positions from
// chunk.committed_tokens on of `request`.
struct QwenTargetPrefillSequence final {
  uint32_t lane = 0;
  uint32_t rowBegin = 0;
  uint32_t rows = 0;
  std::span<const uint32_t> tokens;
  QwenTargetRequest request;
  uint32_t attentionStride = 0;
  uint64_t queryOffset = 0;
  uint64_t kvOffset = 0;
  kv::ChunkedPrefillParams chunk;
  metal::MetalBuffer pageTable;
  std::span<const metal::MetalBuffer> convolutionIn;
  std::span<const metal::MetalBuffer> convolutionOut;
  std::span<const metal::MetalBuffer> recurrentIn;
  std::span<const metal::MetalBuffer> recurrentOut;
  std::array<QwenTargetPrefillCapture, 2> captures{};
  uint32_t captureCount = 0;
  // In row order, each inside (0, rows).
  std::span<const QwenTargetStateCapture> stateCaptures;
  // The target's auxiliary state before and after the sequence's rows.
  metal::MetalBuffer auxiliaryIn;
  metal::MetalBuffer auxiliaryOut;
};

struct QwenTargetPrefillBuffers final {
  // Split projections of chunks of up to 32 rows (LinearGguf.cpp).
  ops::LinearScratch linearScratch{};
  std::array<metal::MetalBuffer, 2> hidden;
  metal::MetalBuffer normalized;
  metal::MetalBuffer captured;
  metal::MetalBuffer gdnPacked;
  metal::MetalBuffer gdnQueries;
  metal::MetalBuffer gdnKeys;
  metal::MetalBuffer gdnValues;
  metal::MetalBuffer gdnDecay;
  metal::MetalBuffer gdnBeta;
  metal::MetalBuffer recurrent;
  metal::MetalBuffer gdnHidden;
  metal::MetalBuffer gdnOutput;
  metal::MetalBuffer denseGateScratch;
  metal::MetalBuffer denseIntermediate;
  metal::MetalBuffer fullPacked;
  metal::MetalBuffer fullQueries;
  metal::MetalBuffer fullAttention;
  metal::MetalBuffer attentionPartials;
  metal::MetalBuffer attentionStatistics;
  metal::MetalBuffer attentionHidden;
  metal::MetalBuffer attentionOutput;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  // The RoPE inverse frequencies the tables were made of (rotaryPairs).
  metal::MetalBuffer ropeInverseFrequencies;
  metal::MetalBuffer chunkKeys;
  metal::MetalBuffer chunkValues;
  ops::MoeScratch moe;

  // The dense FFN's buffers among these.
  [[nodiscard]] ops::PrefillFfnBuffers ffn() const {
    return {normalized, denseGateScratch, denseIntermediate, linearScratch};
  }
};

// A chunk of a prefill command: `rows` rows of its lanes' sequences, packed
// in sequence order, in its own buffers.
struct QwenTargetPrefillChunk final {
  QwenTargetPrefillBuffers buffers;
  std::span<const QwenTargetPrefillSequence> sequences;
  uint32_t rows = 0;
};

struct QwenTargetVerifyBuffers final {
  ops::LinearScratch linearScratch{};
  std::array<metal::MetalBuffer, 2> hidden;
  metal::MetalBuffer normalized;
  metal::MetalBuffer gdnHidden;
  metal::MetalBuffer gdnOutput;
  metal::MetalBuffer denseIntermediate;
  metal::MetalBuffer fullPacked;
  metal::MetalBuffer fullQueries;
  metal::MetalBuffer attentionPartials;
  metal::MetalBuffer attentionStatistics;
  metal::MetalBuffer fullAttention;
  metal::MetalBuffer attentionHidden;
  metal::MetalBuffer attentionOutput;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  metal::MetalBuffer ropeInverseFrequencies;
  metal::MetalBuffer capturedTargetHidden;
  metal::MetalBuffer finalHidden;
  metal::MetalBuffer logits;
  metal::MetalBuffer denseGateScratch;
  metal::MetalBuffer gdnPacked;
  // The lanes' decode tapes (ops::GdnTapeLane).
  metal::MetalBuffer gdnTape;
  std::array<ops::GdnTapeLane, ExecutionLimits::maximumBatchWidth> gdnTapeLanes{};
  std::span<const metal::MetalBuffer> chunkKeys;
  std::span<const metal::MetalBuffer> chunkValues;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
      currentGdnStates;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
      nextGdnStates;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
      pageTables;
  ops::MoeScratch moe;
  // Each lane's auxiliary state before the step; the commit writes the one
  // after it (addStateCommit).
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> auxiliary;
  // An MTP draft's (addMtpDraft): each lane's anchor at row 0 of its draft
  // input rows, the proposals it writes (proposalsPerLane per lane) and the
  // sparse candidates of each (candidatesPerProposal per proposal, the
  // proposal written as the first).
  metal::MetalBuffer anchors;
  metal::MetalBuffer proposals;
  metal::MetalBuffer candidates;
  uint32_t proposalsPerLane = 0;
  uint32_t candidatesPerProposal = 0;
  // With the draft vocabulary, each lane's context rows beside it
  // (Qwen4DraftVocabulary.hpp): its head over its state lane's rows, how
  // many of them hold its context tokens, and the (token, row) pairs to copy
  // into them first; the lanes' logits of them (draftContextLogitsSlot
  // floats each), the state lanes' tokens of them and each lane's
  // Qwen4MtpContextLane. Lanes without draftContextCounts draft from the
  // draft vocabulary alone.
  std::array<ops::Projection, ExecutionLimits::maximumBatchWidth> draftContextHeads;
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> draftContextCounts{};
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> draftContextCopies;
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> draftContextCopyCounts{};
  metal::MetalBuffer draftContextLogits;
  uint32_t draftContextLogitsSlot = 0;
  metal::MetalBuffer draftContextTokens;
  metal::MetalBuffer draftContextInfo;
};

// The auxiliary state after the rows each lane of a verify retained, which
// its commit writes from the state before the step (addStateCommit). The GDN
// state folds the rows into the lane's next step instead (ops::GdnTapeLane).
struct QwenTargetCommitBuffers final {
  metal::MetalBuffer retainedCounts;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> auxiliaryIn;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> auxiliaryOut;
};

// The MTP head's draft (addMtpDraft): `steps` draft steps, the first from
// each lane's anchor; the proposals of a lane are those its head drafted,
// and, from the step that drafts the deepest lane's last, every later one
// repeats it. The verify after it keeps rows past a lane's drafted proposals
// from no expert (addVerify liveRows).
struct QwenTargetMtpDraft final {
  uint32_t steps = 0;
  // Whether the steps draft from the draft vocabulary and the lanes' context
  // rows (QwenTarget::hasMtpDraftVocabulary) rather than the whole head.
  bool draftVocabulary = false;
  // With `prefetchEvent`, step `prefetchStep` copies each lane's proposals so
  // far to `prefetchTokens` (proposalsPerLane uint32 per lane) and signals
  // the event to `prefetchValue`: a streamed PLE table reads the rows those
  // proposals' verify will gather ahead while the later steps run.
  metal::SharedEvent prefetchEvent;
  uint64_t prefetchValue = 0;
  uint32_t prefetchStep = 0;
  metal::MetalBuffer prefetchTokens;
};

// The bytes of a target's own step scratch beside the runtime's arenas, a
// prefill's and a decode's (QwenTarget::scratchBytes).
struct QwenTargetScratchBytes final {
  uint64_t prefill = 0;
  uint64_t decode = 0;
};

// The geometry of a target whose DFlash2 draft reads the output rows of
// `captureLayers` (DFlashDraftLayout::targetLayerIds).
template <class Layout, class Layer>
[[nodiscard]] QwenTargetGeometry qwenTargetGeometry(const QwenTargetWeights<Layout, Layer> &weights,
                                                    std::span<const uint32_t> captureLayers);
[[nodiscard]] QwenTargetGeometry qwenTargetGeometry(const Qwen4ExpWeights &weights,
                                                    std::span<const uint32_t> captureLayers);

// Builds the shared Qwen GDN/attention layer graph with the target's dense
// or sparse-MoE FFN. Architecture-specific loaders supply the model's tensors.
// A target that needs more of a step than the buffers the runtime binds
// (Qwen3.8-Flash-Next: model/Qwen4Steps.hpp) keeps that scratch itself and
// makes its own inputs of the step's tokens; the runtime calls the same
// functions for every target, which do nothing for one without them.
class QwenTarget final {
public:
  // A target whose prefill commands run up to `prefillChunks` chunks.
  template <class Layout, class Layer>
  QwenTarget(const QwenTargetWeights<Layout, Layer> &weights, const QwenTargetGeometry &geometry,
             metal::MetalBackend &backend, const ops::ExecutionPlans &operators, uint32_t prefillChunks);
  QwenTarget(const Qwen4ExpWeights &weights, const QwenTargetGeometry &geometry, metal::MetalBackend &backend,
             const ops::ExecutionPlans &operators, uint32_t prefillChunks);
  QwenTarget(QwenTarget &&) noexcept;
  ~QwenTarget();

  // The step scratch a target of `geometry` keeps itself (none for most),
  // and what this one allocated of it.
  [[nodiscard]] static QwenTargetScratchBytes scratchBytes(const QwenTargetGeometry &geometry,
                                                           uint32_t prefillChunks);
  [[nodiscard]] QwenTargetScratchBytes allocatedScratchBytes() const noexcept;

  [[nodiscard]] const ops::Projection &
  vocabularyProjection() const noexcept;
  // Lanes of storage the tensors of a decode step of `lanes` lanes bind: a
  // linear tile may hold more rows than the step (LinearPlan::storageRows;
  // a three-lane GGUF step on the staged tile runs its 32-row tile over four
  // lanes). Every op still processes the step's lanes; padding rows read
  // stale activations and write results no active row reads.
  [[nodiscard]] uint32_t decodeStorageLanes(uint32_t lanes) const;

  // A prefill command of one or more chunks (a prefill group), run layer by
  // layer: a layer of each chunk in turn, then the next layer, so that a
  // layer whose experts stream loads the union of the chunks' experts once.
  // A chunk's sequences read the states and KV that the chunks before it
  // wrote, in `kv`. The dense FFN of a command of one chunk the split takes
  // (AneFfn::splits) runs split with the Neural Engine on `aneFfn`, when
  // given. Returns each chunk's buffer of its last layer's output rows.
  //
  // It first writes the inputs the target makes of the chunks' tokens on
  // the CPU and stages a streamed table's rows of them, and starts reading
  // ahead the rows of the prompt rows after them: the command is submitted
  // once it returns, while no other command reading the target's prefill
  // scratch is in flight.
  [[nodiscard]] std::vector<metal::MetalBuffer> addPrefill(metal::CommandGraph &graph,
                                                           std::span<const QwenTargetPrefillChunk> chunks,
                                                           const kv::PageStorage &kv,
                                                           ops::AneFfn *aneFfn = nullptr);
  // Writes the inputs the target makes on the CPU of batch lane `lane`'s
  // verify rows, those from `position` on of `request`, before the cycle's
  // commands are encoded.
  void prepareVerifyLane(uint32_t lane, uint64_t position, const QwenTargetRequest &request);
  // With `liveRows`, Qwen3.8-Flash-Next's verify routes each lane's rows
  // from it on to no expert (ops::MoE::add lastRow): the rows past the
  // proposals its MTP head drafted, which only repeat the last one.
  void addVerify(
      metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
      const kv::PageStorage &kv,
      std::span<const kv::ChunkedPrefillParams> chunks,
      uint32_t lanes, uint32_t liveRows = 0) const;
  // Qwen3.8-Flash-Next's weights, or null for another family.
  [[nodiscard]] const Qwen4ExpWeights *qwen4() const noexcept;
  // Whether Qwen3.8-Flash-Next's MTP head drafts (Qwen4ExpLayout::mtpLayers).
  [[nodiscard]] bool hasMtp() const noexcept { return geometry_.mtpLayers != 0; }
  // Whether the MTP head drafts from a draft vocabulary when asked
  // (Qwen4ExpLayout::mtpDraftTokens).
  [[nodiscard]] bool hasMtpDraftVocabulary() const noexcept;
  // `draft.steps` draft steps of the MTP head over each lane's verify rows,
  // before the verify: step j writes proposal j - 1, the last one also every
  // later proposal. The verify then writes the head's KV at its rows.
  void addMtpDraft(metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers, const kv::PageStorage &kv,
                   std::span<const kv::ChunkedPrefillParams> chunks, uint32_t lanes,
                   QwenTargetMtpDraft draft) const;
  // The final norm and LM head over `lanes` lanes of targetVerifyRows rows,
  // as verify ends: one sweep of the vocabulary projection for every lane.
  void addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden,
                    metal::MetalBuffer finalHidden, metal::MetalBuffer logits,
                    uint32_t lanes, ops::LinearScratch scratch) const;
  // The verify input tokens addEmbedding then gathers: each lane's anchor,
  // row 0 of its draft input, and the draft's proposals; then the inputs the
  // target makes of them itself. Qwen3.8-Flash-Next's GPU hashes their PLE
  // rows, and when its table streams a host step of the command stages them,
  // between the hash and the verify that gathers them.
  void addVerifyInput(metal::CommandGraph &graph, metal::MetalBuffer draftInput,
                      metal::MetalBuffer proposals,
                      metal::MetalBuffer verifyInput, uint32_t lanes) const;
  void addEmbedding(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                    metal::MetalBuffer hidden, uint32_t rows) const;
  // The commit after acceptance of a verify of `lanes` lanes, in the same
  // command after the verify (addVerify): each lane's auxiliary state after
  // its retained rows, from what the verify computed of its rows. None for a
  // target without auxiliary state.
  void addStateCommit(metal::CommandGraph &graph, QwenTargetCommitBuffers buffers, uint32_t lanes) const;

private:
  // Tests read the target's step scratch (RuntimeInstrumentation).
  friend class RuntimeInstrumentation;
  using WeightView =
      std::variant<const QwenTargetWeights<Qwen3_8Layout, Qwen3_8LayerWeights> *,
                   const QwenTargetWeights<Qwen3_6MoeLayout, Qwen3_6MoeLayerWeights> *,
                   const Qwen4ExpWeights *>;
  struct PrefillStep;
  struct PrefillQsa;
  struct VerifyStep;

  // A decode step of `lanes` lanes over the verify rows, with its plans and
  // scratch bound, for `use` to add its work to.
  void withVerifyStep(metal::CommandGraph &graph, const QwenTargetVerifyBuffers &buffers, const kv::PageStorage &kv,
                      std::span<const kv::ChunkedPrefillParams> chunks, uint32_t lanes,
                      const std::function<void(VerifyStep &)> &use) const;

  // A prefill's layers over its chunks' steps (addPrefill): each step's
  // begin; per layer, each step through its FFN's routing (addPrefillLayer),
  // the layer's load of streamed experts, and each step's routed work
  // (addPrefillExperts); then each step's end, which returns its last
  // layer's output rows. Each family has its overloads: Qwen3.5 and Qwen3.6
  // (QwenTargetWeights: hidden rows each layer adds to) and
  // Qwen3.8-Flash-Next (Qwen4ExpWeights: residual streams, the PLE and QSA).
  template <class Weights>
  [[nodiscard]] std::vector<metal::MetalBuffer> addPrefillLayers(std::span<PrefillStep> steps,
                                                                 const Weights &weights) const;
  template <class Layout, class Layer>
  void addPrefillBegin(PrefillStep &step, const QwenTargetWeights<Layout, Layer> &weights) const;
  template <class Layout, class Layer>
  void addPrefillLayer(PrefillStep &step, const QwenTargetWeights<Layout, Layer> &weights, uint32_t index) const;
  template <class Layout, class Layer>
  void addPrefillExperts(PrefillStep &step, const QwenTargetWeights<Layout, Layer> &weights, uint32_t index) const;
  template <class Layout, class Layer>
  [[nodiscard]] metal::MetalBuffer addPrefillEnd(PrefillStep &step,
                                                 const QwenTargetWeights<Layout, Layer> &weights) const;
  void addPrefillBegin(PrefillStep &step, const Qwen4ExpWeights &weights) const;
  void addPrefillLayer(PrefillStep &step, const Qwen4ExpWeights &weights, uint32_t index) const;
  void addPrefillExperts(PrefillStep &step, const Qwen4ExpWeights &weights, uint32_t index) const;
  [[nodiscard]] metal::MetalBuffer addPrefillEnd(PrefillStep &step, const Qwen4ExpWeights &weights) const;
  // Every family's mixers, of the step's normalized rows into the rows their
  // output projection reads: a GDN (buffers.gdnHidden) and a full attention
  // (buffers.attentionHidden), which, with `qsa`, attends only the blocks
  // Qwen3.8-Flash-Next's sparse attention selects.
  [[nodiscard]] metal::MetalBuffer addPrefillGdn(PrefillStep &step, const QwenGdnWeights &mixer) const;
  [[nodiscard]] metal::MetalBuffer addPrefillAttention(PrefillStep &step, const QwenAttentionWeights &mixer,
                                                       const PrefillQsa *qsa) const;
  // Qwen3.5/3.6: the RMS norm of `input` into buffers.normalized; a mixer's
  // output projection added to `input` into `output`; the FFN of layer
  // `index` adding to the mixer's `residual` rows into `output` (a MoE's
  // routing, whose routed work addPrefillExperts adds).
  void addPrefillNorm(PrefillStep &step, metal::MetalBuffer input, const ops::NormWeights &norm) const;
  void addPrefillOutput(PrefillStep &step, metal::MetalBuffer hidden, const ops::Projection &projection,
                        metal::MetalBuffer input, metal::MetalBuffer output) const;
  void addPrefillFfn(PrefillStep &step, uint32_t index, const Qwen3_8LayerWeights &layer,
                     metal::MetalBuffer residual, metal::MetalBuffer output) const;
  void addPrefillFfn(PrefillStep &step, uint32_t index, const Qwen3_6MoeLayerWeights &layer,
                     metal::MetalBuffer residual, metal::MetalBuffer output) const;
  // Qwen3.8-Flash-Next: the mix of `streams` through `hyper` into `output`
  // and, when it injects, the injection weights into the step's scratch.
  void addQwen4PrefillMix(PrefillStep &step, const Qwen4HyperWeights &hyper, metal::MetalBuffer output,
                          const metal::MetalBuffer &streams) const;
  metal::MetalBuffer addVerifyMixer(VerifyStep &step, const QwenGdnWeights &mixer, const ops::NormWeights &norm,
                                    metal::MetalBuffer input) const;
  metal::MetalBuffer addVerifyMixer(VerifyStep &step, const QwenAttentionWeights &mixer,
                                    const ops::NormWeights &norm, metal::MetalBuffer input) const;
  void addVerifyFfn(VerifyStep &step, const Qwen3_8LayerWeights &layer, metal::MetalBuffer residual,
                    metal::MetalBuffer output) const;
  void addVerifyFfn(VerifyStep &step, const Qwen3_6MoeLayerWeights &layer, metal::MetalBuffer residual,
                    metal::MetalBuffer output) const;

  // A verify's layers and head (addVerify), each family's: Qwen3.5 and
  // Qwen3.6's hidden rows each layer adds to, then the final norm and head;
  // Qwen3.8-Flash-Next's residual streams, the PLE and QSA, then its final
  // mix, which the head reads without a norm (addQwen4*).
  template <class Layout, class Layer>
  void addVerifyLayers(VerifyStep &step, const QwenTargetWeights<Layout, Layer> &weights) const;
  void addVerifyLayers(VerifyStep &step, const Qwen4ExpWeights &weights) const;
  // The PLE layer's `count` rows into h.pleRows: from the table by
  // h.pleIndices, or, when the table streams, from the rows staged by
  // h.pleSlots, with the table's own gather.
  void addPleGather(metal::CommandGraph &graph, const Qwen4Scratch &h, uint32_t count) const;
  // A mix into `output`; its projections read the tables its norm and low
  // step write, and with `next`, the decode projection that reads `output`
  // first takes the table the mix writes (the returned input).
  ops::PreparedInput addQwen4VerifyMix(VerifyStep &step, const Qwen4HyperWeights &hyper, metal::MetalBuffer streams,
                                       metal::MetalBuffer output, const ops::Projection *next = nullptr) const;
  // A full-attention block over the step's rows into its scratch's branch.
  void addQwen4VerifyAttention(VerifyStep &step, const Qwen4ExpLayerWeights &layer,
                               const QwenAttentionWeights &mixer, uint32_t attentionLayer,
                               ops::PreparedInput input = {}) const;
  // The MTP head over each lane's rows: its inputs (`embedding` rows paired
  // with the carried streams and `history`), its attention block (writing
  // its KV) and, when `full`, its FFN, its head mix and logits. Each lane's
  // rows before `firstRow` route to no expert (ops::MoE::add).
  // With `draftHead`, the logits are of its tokens rather than the whole
  // vocabulary.
  void addQwen4MtpVerify(VerifyStep &step, const Qwen4ExpWeights &weights, metal::MetalBuffer embedding,
                         metal::MetalBuffer history, uint32_t limit, bool full, uint32_t firstRow = 0,
                         const Qwen4DraftHead *draftHead = nullptr) const;
  // The MTP head's KV at a prefill chunk's rows, and each sequence's carried
  // streams after its rows (and at each state captured inside them).
  void addQwen4MtpPrefill(PrefillStep &step, const Qwen4ExpWeights &weights) const;
  // A state cell's carried MTP streams within its auxiliary state.
  [[nodiscard]] metal::MetalBuffer mtpCarried(const metal::MetalBuffer &auxiliary) const;

  WeightView weights_;
  const QwenTargetWeightsBase &weightsBase_;
  QwenTargetGeometry geometry_;
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
  // Qwen3.8-Flash-Next's step scratch and inputs; null for another family.
  std::unique_ptr<Qwen4Steps> qwen4Steps_;
};

} // namespace splash::model

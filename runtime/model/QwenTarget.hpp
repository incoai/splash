#pragma once

#include "Model.hpp"
#include "QwenHybridLayout.hpp"
#include "RopeScaling.hpp"
#include "SpilledKvReads.hpp"
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
class SparseRowStore;

// A Qwen target's weights outside its layers and the record of every file
// its weights were read from. A MoE target whose routed experts stream reads
// them through `experts` (model/ExpertStore.hpp), and a target whose table
// streams from its file reads its rows through `streamedTable`
// (model/SparseRowStore.hpp).
struct QwenTargetWeightsBase {
  ops::Projection logitsProjection;
  ops::EmbeddingWeights tokenEmbedding;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
  std::string manifestFingerprintSha256;
  std::shared_ptr<ExpertStore> experts;
  std::shared_ptr<SparseRowStore> streamedTable;
};

// The weights of a Qwen3.5-architecture target of Layout, whose layers the
// family keeps in Layer, and the norm the head reads its last layer's output
// through.
template <class Layout, class Layer> struct QwenTargetWeights final : QwenTargetWeightsBase {
  Layout layout;
  std::vector<Layer> layers;
  ops::NormWeights finalNorm;
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
  // The target's KV layout in the format the runtime stores KV in, and its
  // GDN state layout.
  kv::Layout kvLayout{};
  GdnStateLayout stateLayout{};
  // The scaling of the attention layers' RoPE the model is served with
  // (ModelDescriptor::ropeScaling).
  RopeScaling ropeScaling{};
  // Distinct operator requirements, collected from the loaded weights
  // (qwenTargetGeometry): include adds a projection the weights dispatch,
  // finish the formats of their MoE blocks' expert weights and the head's
  // logits projection, which decode alone runs.
  std::vector<ops::ProjectionShape> prefillProjections;
  std::vector<ops::ProjectionShape> decodeProjections;
  std::vector<ops::ProjectionShape> gateUpProjections;

  void include(const ops::Projection &projection) { decodeProjections.push_back(projection.shape()); }
  void finish(std::span<const ops::BlockMoeWeights *const> moeBlocks, const ops::Projection &logits);

  // The window the target serves with its RoPE scaling, which everything
  // sized by the context follows (ModelDescriptor::capabilities).
  [[nodiscard]] uint32_t contextTokens() const noexcept { return servedContextTokens(ropeScaling); }
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
  // own scratch instead (qwenTargetScratchBytes).
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
    return captured() && stateLayout.layers + kvLayout.attentionLayers == layers &&
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
  // Its pages' block index, for a layout with one (ops::BlockIndex::tables).
  metal::MetalBuffer indexTable;
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

// The layers [first, end) of a prefill command one of its passes runs
// (BatchPlan::passes): the pass with the first layer begins the chunks'
// rows, the one with the last ends them.
struct QwenPrefillLayers final {
  uint32_t first = 0;
  uint32_t end = 0;
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
  // Each lane's pages' block index, for a layout with one
  // (ops::BlockIndex::tables).
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> indexTables;
  ops::MoeScratch moe;
  // Each lane's verify rows (ops::DraftSelector::addVerifyRows), which a MoE target's
  // steps route alone (ops::MoeBuffers::verifyRows).
  metal::MetalBuffer verifyRows;
  // Each lane's auxiliary state before the step; the commit writes the one
  // after it (addStateCommit).
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> auxiliary;
};

// The auxiliary state after the rows each lane of a verify retained, which
// its commit writes from the state before the step (addStateCommit). The GDN
// state folds the rows into the lane's next step instead (ops::GdnTapeLane).
struct QwenTargetCommitBuffers final {
  metal::MetalBuffer retainedCounts;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> auxiliaryIn;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> auxiliaryOut;
};

// The bytes of a target's own step scratch beside the runtime's arenas, a
// prefill's and a decode's (qwenTargetScratchBytes).
struct QwenTargetScratchBytes final {
  uint64_t prefill = 0;
  uint64_t decode = 0;
};

// A layer's MoE block, null for a dense FFN.
[[nodiscard]] const ops::BlockMoeWeights *qwenMoeBlock(const Qwen3_8LayerWeights &layer) noexcept;
template <class Layer> [[nodiscard]] const ops::BlockMoeWeights *qwenMoeBlock(const Layer &layer) noexcept {
  return &layer.ffn;
}

// The projections a family's layers and head dispatch beyond their mixers'
// and MoE blocks' (qwenTargetGeometry): Qwen3.8-27B's dense FFN; the
// Qwen3.6-35B-A3B's none (Qwen3.8-Flash-Next's: model/Qwen4Exp.hpp).
void includeFamilyProjections(QwenTargetGeometry &geometry,
                              const QwenTargetWeights<Qwen3_8Layout, Qwen3_8LayerWeights> &weights);
void includeFamilyProjections(QwenTargetGeometry &geometry,
                              const QwenTargetWeights<Qwen3_6MoeLayout, Qwen3_6MoeLayerWeights> &weights);

// The geometry of a target of `weights` whose DFlash2 draft reads the output
// rows of `captureLayers` (DFlashDraftLayout::targetLayerIds): its layout's,
// every projection its weights dispatch and the formats of its MoE blocks'
// expert weights.
template <class Weights>
[[nodiscard]] QwenTargetGeometry qwenTargetGeometry(const Weights &weights, std::span<const uint32_t> captureLayers) {
  QwenTargetGeometry geometry(weights.layout, captureLayers);
  geometry.kvLayout = weights.layout.kvLayout();
  geometry.stateLayout = weights.layout.gdnStateLayout();
  std::vector<const ops::BlockMoeWeights *> moeBlocks;
  for (const auto &layer : weights.layers) {
    std::visit([&](const auto &mixer) {
      geometry.include(mixer.inputProjection);
      geometry.include(mixer.outputProjection);
    }, layer.mixer);
    if (const ops::BlockMoeWeights *block = qwenMoeBlock(layer)) moeBlocks.push_back(block);
  }
  includeFamilyProjections(geometry, weights);
  geometry.finish(moeBlocks, weights.logitsProjection);
  return geometry;
}

// The target the runtime drives: a family's layers over the blocks every
// Qwen hybrid family shares, made for its weights by makeQwenTarget. The
// shared blocks are a step's checks and plans, the GDN and full-attention
// mixers (one wiring each for prefill and for verify), the draft's captures
// and the head's projection; each family composes them into its layers and
// residual: the Qwen3.5 architecture's (Qwen3.8-27B and Qwen3.6-35B-A3B,
// model/QwenTarget.cpp) and Qwen3.8-Flash-Next's (model/Qwen4ExpTarget.cpp).
// A family that needs more of a step than the buffers the runtime binds
// keeps that scratch itself (qwenTargetScratchBytes) and makes its own
// inputs of the step's tokens; the runtime calls the same functions for
// every target, which do nothing for one without them.
class QwenTarget {
public:
  QwenTarget(const QwenTarget &) = delete;
  QwenTarget &operator=(const QwenTarget &) = delete;
  virtual ~QwenTarget();

  // The step scratch the target allocated itself (qwenTargetScratchBytes).
  [[nodiscard]] virtual QwenTargetScratchBytes allocatedScratchBytes() const noexcept;

  [[nodiscard]] const ops::Projection &
  vocabularyProjection() const noexcept;
  // Lanes of storage the tensors of a decode step of `lanes` lanes bind: a
  // linear tile may hold more rows than the step (LinearPlan::storageRows;
  // a three-lane GGUF step on the staged tile runs its 32-row tile over four
  // lanes, a one-lane step may run its 16-row tile over two). Every op still
  // processes the step's lanes; padding rows read stale activations and write
  // results no active row reads.
  [[nodiscard]] uint32_t decodeStorageLanes(uint32_t lanes) const;

  // A prefill command of one or more chunks (a prefill group), run layer by
  // layer: a layer of each chunk in turn, then the next layer, so that a
  // layer whose experts stream loads the union of the chunks' experts once.
  // A chunk's sequences read the states and KV that the chunks before it
  // wrote, in `kv`, and a sequence whose history holds Index pages reads its
  // K/V through `spilled` (the command's SpilledKvReads, the null one for
  // none). The dense FFN of a command of one chunk the split takes
  // (AneFfn::splits) runs split with the Neural Engine on `aneFfn`, when
  // given. Of the command, this runs `layers`, all of them or a pass's:
  // what the layers before left in the chunks' buffers and the target's
  // prefill scratch the layers after read on, from another command, with no
  // other prefill between. With the last layer, returns each chunk's buffer
  // of its last layer's output rows, and else nothing.
  //
  // With the first layer, it first writes the inputs the target makes of
  // the chunks' tokens on the CPU and stages a streamed table's rows of
  // them, and starts reading ahead the rows of the prompt rows after them:
  // the command is submitted once it returns, while no other command
  // reading the target's prefill scratch is in flight.
  [[nodiscard]] virtual std::vector<metal::MetalBuffer> addPrefill(metal::CommandGraph &graph,
                                                                   std::span<const QwenTargetPrefillChunk> chunks,
                                                                   const kv::PageStorage &kv, SpilledKvReads &spilled,
                                                                   QwenPrefillLayers layers,
                                                                   ops::AneFfn *aneFfn = nullptr) = 0;
  // Writes the inputs the target makes on the CPU of batch lane `lane`'s
  // verify rows, those from `position` on of `request`, before the cycle's
  // commands are encoded.
  virtual void prepareVerifyLane(uint32_t lane, uint64_t position, const QwenTargetRequest &request);
  // A verify's layers and head over `lanes` lanes, whose blocks on Index
  // pages it reads through `spilled` (the command's SpilledKvReads).
  virtual void addVerify(
      metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
      const kv::PageStorage &kv, SpilledKvReads &spilled,
      std::span<const kv::ChunkedPrefillParams> chunks,
      uint32_t lanes) const = 0;
  // The head over `lanes` lanes of targetVerifyRows rows of the last layer's
  // output, as verify ends: one sweep of the vocabulary projection for every
  // lane.
  virtual void addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden,
                            metal::MetalBuffer finalHidden, metal::MetalBuffer logits,
                            uint32_t lanes, ops::LinearScratch scratch) const = 0;
  // The verify input tokens addEmbedding then gathers: each lane's anchor,
  // row 0 of its draft input, and the draft's proposals; then the inputs the
  // target makes of them itself.
  virtual void addVerifyInput(metal::CommandGraph &graph, metal::MetalBuffer draftInput,
                              metal::MetalBuffer proposals,
                              metal::MetalBuffer verifyInput, uint32_t lanes) const;
  void addEmbedding(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                    metal::MetalBuffer hidden, uint32_t rows) const;
  // The commit after acceptance of a verify of `lanes` lanes: in the
  // verify's command after it (addVerify), or for constrained decoding in
  // the next command, before any other decode step reuses the verify's
  // scratch. Each lane's auxiliary state after its retained rows, from what
  // the verify computed of its rows. None for a target without auxiliary
  // state.
  virtual void addStateCommit(metal::CommandGraph &graph, QwenTargetCommitBuffers buffers, uint32_t lanes) const;
  // The table rows batch lane `lane`'s verify rows of the last step read,
  // each row's in order, for tests (RuntimeInstrumentation::lastTableRows);
  // none for a target that reads no table.
  [[nodiscard]] virtual std::vector<uint32_t> verifyTableRows(uint32_t lane) const;

protected:
  QwenTarget(const QwenTargetWeightsBase &weights, const QwenTargetGeometry &geometry, metal::MetalBackend &backend,
             const ops::ExecutionPlans &operators);

  // The state a prefill command's layers share for one of its chunks: its
  // rows, their buffers and plans, the K/V they attend and the command's
  // reads of spilled K/V, the buffers the layer's MoE routes and runs
  // through, and the next GDN and attention layer of the step. A family
  // whose steps need more extends it.
  struct PrefillStep {
    metal::CommandGraph &graph;
    const QwenTargetPrefillBuffers &buffers;
    std::span<const QwenTargetPrefillSequence> sequences;
    uint32_t rows;
    std::span<const SplashKvLayer> kvLayers;
    SpilledKvReads &spilled;
    // Each sequence's attention plan, which every attention layer runs.
    std::vector<ops::PrefillAttentionPlan> attention{};
    std::optional<ops::MoePlan> moe{};
    // The layer's MoE buffers, which its routing (addPrefillLayer), the
    // layer's load and its routed work (addPrefillExperts) share.
    ops::MoeBuffers ffn{};
    ops::AneFfn *aneFfn = nullptr;
    uint32_t gdnLayer = 0;
    uint32_t attentionLayer = 0;
  };

  struct VerifyStep {
    metal::CommandGraph &graph;
    const QwenTargetVerifyBuffers &buffers;
    std::span<const SplashKvLayer> kvLayers;
    SpilledKvReads &spilled;
    std::span<const kv::ChunkedPrefillParams> chunks;
    uint32_t lanes;
    uint32_t rows;
    ops::VerifyAttentionPlan attention;
    std::optional<ops::MoePlan> moe{};
    uint32_t gdnLayer = 0;
    uint32_t attentionLayer = 0;
  };

  // Where a mixer's output projection writes: `rows`, adding the projection
  // to `residual` when given.
  struct BlockOutput final {
    metal::MetalBuffer rows;
    metal::MetalBuffer residual;
  };

  // The checked steps of a prefill's chunks, each with its plans, as a
  // family's Step (a PrefillStep or one extending it).
  template <class Step = PrefillStep>
  [[nodiscard]] std::vector<Step> prefillSteps(metal::CommandGraph &graph,
                                               std::span<const QwenTargetPrefillChunk> chunks,
                                               const kv::PageStorage &kv, SpilledKvReads &spilled,
                                               QwenPrefillLayers layers, ops::AneFfn *aneFfn) const {
    requirePrefill(chunks, kv, layers, aneFfn);
    std::vector<Step> steps;
    steps.reserve(chunks.size());
    for (const QwenTargetPrefillChunk &chunk : chunks)
      steps.push_back(Step{prefillStep(graph, chunk, kv, spilled, aneFfn)});
    return steps;
  }
  // A prefill's layers `range` over its steps (addPrefill), through a
  // family's `layers`: with the first layer, each step's begin
  // (addPrefillBegin), and else each step's next GDN and attention layer
  // past those before the range (attends); per layer, each step through its
  // FFN's routing (addPrefillLayer), the hand-off of an attention layer's
  // spilled history once every step attended it (SpilledKvReads::layerEnd),
  // one load of the streamed experts any step's rows route (of the layer's
  // MoE block, moe), and each step's routed work (addPrefillExperts); then,
  // with the last layer, each step's end, which returns its last layer's
  // output rows (addPrefillEnd).
  template <class Step, class Layers>
  [[nodiscard]] std::vector<metal::MetalBuffer> addPrefillLayers(std::span<Step> steps, const Layers &layers,
                                                                 QwenPrefillLayers range) const {
    if (!range.first)
      for (Step &step : steps) layers.addPrefillBegin(step);
    for (uint32_t index = 0; index < range.first; ++index)
      for (Step &step : steps) ++(layers.attends(index) ? step.attentionLayer : step.gdnLayer);
    std::vector<ops::MoeLoadPart> parts;
    if (geometry_.ffnKind == QwenFfnKind::SparseMoe)
      for (const Step &step : steps) parts.push_back({&step.ffn, &*step.moe});
    for (uint32_t index = range.first; index < range.end; ++index) {
      const uint32_t attentionLayer = steps.front().attentionLayer;
      for (Step &step : steps) layers.addPrefillLayer(step, index);
      if (steps.front().attentionLayer != attentionLayer)
        steps.front().spilled.layerEnd(steps.front().graph, attentionLayer);
      if (const ops::BlockMoeWeights *ffn = layers.moe(index); ffn && ffn->stream)
        ffn->stream->addLoad(steps.front().graph, parts, ffn->streamBlock);
      for (Step &step : steps) layers.addPrefillExperts(step, index);
    }
    std::vector<metal::MetalBuffer> outputs;
    if (range.end != geometry_.layers) return outputs;
    outputs.reserve(steps.size());
    for (Step &step : steps) {
      requireLayerPartition(step.gdnLayer, step.attentionLayer);
      outputs.push_back(layers.addPrefillEnd(step));
    }
    return outputs;
  }
  // The checked step of a verify of `lanes` lanes, with its plans.
  [[nodiscard]] VerifyStep verifyStep(metal::CommandGraph &graph, const QwenTargetVerifyBuffers &buffers,
                                      const kv::PageStorage &kv, SpilledKvReads &spilled,
                                      std::span<const kv::ChunkedPrefillParams> chunks, uint32_t lanes) const;
  // Throws unless a family's `layers` match the geometry's layers and its
  // attention layers.
  template <class Layer> void requireLayers(std::span<const Layer> layers) const {
    requireLayers(layers.size(), static_cast<size_t>(std::ranges::count_if(layers, [](const Layer &layer) {
                    return std::holds_alternative<QwenAttentionWeights>(layer.mixer);
                  })));
  }
  void requireLayers(size_t layers, size_t attentionLayers) const;
  void requireLayerPartition(uint32_t gdnLayers, uint32_t attentionLayers) const;

  // The mixers, of a step's normalized rows (buffers.normalized) into their
  // output projection's `output`: a GDN, and a full attention whose rows
  // attend, when given, only the blocks of their lists (`blocks`, a prefill's
  // per sequence, a verify's for the step), through the step's reads of
  // spilled K/V.
  void addPrefillMixer(PrefillStep &step, const QwenGdnWeights &mixer, BlockOutput output) const;
  void addPrefillMixer(PrefillStep &step, const QwenAttentionWeights &mixer, BlockOutput output,
                       std::span<const ops::AttentionBlocks> blocks = {}) const;
  // A verify's normalized rows are in the table `normalized` describes,
  // which the mixer's input projection reads (decodeInput).
  void addVerifyMixer(VerifyStep &step, const QwenGdnWeights &mixer, ops::PreparedInput normalized,
                      BlockOutput output) const;
  void addVerifyMixer(VerifyStep &step, const QwenAttentionWeights &mixer, ops::PreparedInput normalized,
                      BlockOutput output, const ops::AttentionBlocks &blocks = {}) const;
  // The input the decode plan of `projection` over `lanes` lanes reads, a
  // table its producer writes or plain rows.
  [[nodiscard]] ops::LinearInput decodeInput(const ops::Projection &projection, uint32_t lanes) const;
  // A decode projection over `lanes` lanes through its own plan; `prepared`
  // describes the table its producer wrote for the plan's input.
  void addDecodeProjection(metal::CommandGraph &graph, metal::MetalBuffer input, const ops::Projection &projection,
                           metal::MetalBuffer output, uint32_t lanes, ops::LinearScratch scratch,
                           ops::PreparedInput prepared = {}) const;
  // Each of a prefill step's captures of layer `index`'s output rows, when
  // the draft reads the layer: capture(slot, source row, destination row,
  // rows) of the captured hidden rows' slot `slot`.
  template <class Capture> void forEachPrefillCapture(const PrefillStep &step, uint32_t index, Capture capture) const {
    if (const auto slot = geometry_.captureSlot(index))
      for (const QwenTargetPrefillSequence &sequence : step.sequences)
        for (uint32_t c = 0; c < sequence.captureCount; ++c)
          capture(*slot, sequence.captures[c].sourceStart, sequence.captures[c].destinationStart,
                  sequence.captures[c].rows);
  }
  // Rows [begin, begin + count) of a row-major buffer of `width` values of T.
  template <class T>
  [[nodiscard]] metal::MetalBuffer rowsOf(const metal::MetalBuffer &buffer, uint32_t begin, uint32_t count,
                                          uint32_t width) const {
    return backend_.view(buffer, uint64_t{begin} * width * sizeof(T), uint64_t{count} * width * sizeof(T));
  }

  const QwenTargetWeightsBase &weightsBase_;
  QwenTargetGeometry geometry_;
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;

private:
  void requirePrefill(std::span<const QwenTargetPrefillChunk> chunks, const kv::PageStorage &kv,
                      QwenPrefillLayers layers, const ops::AneFfn *aneFfn) const;
  [[nodiscard]] PrefillStep prefillStep(metal::CommandGraph &graph, const QwenTargetPrefillChunk &chunk,
                                        const kv::PageStorage &kv, SpilledKvReads &spilled,
                                        ops::AneFfn *aneFfn) const;
};

// The target of a Qwen3.5-architecture family's weights (Qwen3.8-27B,
// Qwen3.6-35B-A3B), whose prefill commands run up to `prefillChunks` chunks,
// and the step scratch it keeps itself (none: it has no table to stream).
template <class Layout, class Layer>
[[nodiscard]] std::unique_ptr<QwenTarget> makeQwenTarget(const QwenTargetWeights<Layout, Layer> &weights,
                                                         const QwenTargetGeometry &geometry,
                                                         metal::MetalBackend &backend,
                                                         const ops::ExecutionPlans &operators,
                                                         uint32_t prefillChunks);
template <class Layout, class Layer>
[[nodiscard]] QwenTargetScratchBytes qwenTargetScratchBytes(const QwenTargetWeights<Layout, Layer> &,
                                                            const QwenTargetGeometry &, uint32_t, uint64_t) {
  return {};
}

} // namespace splash::model

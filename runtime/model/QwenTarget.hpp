#pragma once

#include "Model.hpp"
#include "QwenHybridLayout.hpp"
#include "RopeScaling.hpp"
#include "SpilledKvReads.hpp"
#include "StateLayout.hpp"
#include "WeightStore.hpp"
#include "ops/GDN.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/LaneRows.hpp"
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

// Every Qwen family binds these mixer tensors per layer, a GDN or an
// attention; their FFNs differ.
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
  // The layer's learned sinks, an F32 per query head, of a geometry whose
  // attention adds them (ops::AttentionPlan::sinks); none otherwise.
  metal::MetalBuffer sinks{};
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

// The weights of a Qwen target of Layout, whose layers the family keeps in
// Layer, and the norm the head reads its last layer's output through.
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

  // The format of the sparse MoE blocks' routers, which their plans follow
  // (ops::moeRouterFormat).
  uint32_t moeRouterFormat = ops::QuantizedSegment::kFloat32;
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
  // How attention layer `attention` (the a-th attention layer) attends: with
  // the RoPE of the target's pairs unless unrotatedLayers names it, and over
  // the window of kvLayout where that names it.
  [[nodiscard]] constexpr ops::AttentionVariant attentionVariant(uint32_t attention) const noexcept {
    return {.rotated = attention >= 64 || !(unrotatedLayers >> attention & 1),
            .windowTokens = kvLayout.windowed(attention) ? kvLayout.windowTokens : 0};
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
  // of its layers, ascending. A target without GDN layers has no GDN shape.
  // The projection lists hold every projection the weights dispatch, which
  // each have sizes; a target with dense FFNs, a MoE target's leading layers
  // too, holds their gate and up projections.
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
    const bool dense = intermediateSize && sized(gateUpProjections);
    return captured() && stateLayout.layers + kvLayout.attentionLayers == layers &&
           (!stateLayout.layers || gdnShape().valid()) &&
           kvLayout.kvHeads == attentionKvHeads &&
           kvLayout.headDimension == attentionHeadDimension &&
           sized(prefillProjections) && sized(decodeProjections) &&
           ((ffnKind == QwenFfnKind::Dense && dense) ||
            (ffnKind == QwenFfnKind::SparseMoe && moeBlockShape().valid() && (!leadingDenseLayers || dense)));
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
// chunk.committed_tokens on of `request`, rows rowBegin to rowBegin + rows of
// the chunk. A decode step's lane leaves them unset: the step's layout places
// its rows (ops::LayoutRows), and only a prompt's helpers read them.
struct QwenTargetPrefillSequence final {
  uint32_t lane = 0;
  uint32_t rowBegin = 0;
  uint32_t rows = 0;
  std::span<const uint32_t> tokens;
  QwenTargetRequest request;
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
  // Split projections of the chunks the staged tile runs (LinearGguf.cpp).
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
  metal::MetalBuffer attentionPartials;
  metal::MetalBuffer attentionStatistics;
  metal::MetalBuffer attentionHidden;
  metal::MetalBuffer attentionOutput;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  // The RoPE inverse frequencies the tables were made of (rotaryPairs).
  metal::MetalBuffer ropeInverseFrequencies;
  // Each sequence's rows of the chunk (SplashLaneRows), as the host writes
  // them, which the attention layers read (ops::AttentionRows).
  metal::MetalBuffer laneRows;
  ops::MoeScratch moe;

  // The dense FFN's buffers among these.
  [[nodiscard]] ops::PrefillFfnBuffers ffn() const {
    return {normalized, denseGateScratch, denseIntermediate, linearScratch};
  }
};

// What a decode step's layers read and write beside a prompt chunk's
// buffers, a decode cycle's target forward being the prompt forward over the
// step's rows (QwenTargetPrefillChunk::decode): the step's lanes and rows
// (ops::StepRows); the lanes' GDN states before and after the step and their
// decode tapes (ops::GdnTapeLane); each lane's verify rows, the first of its
// rows, which alone route a MoE layer (ops::MoeBuffers::verifyRows); the
// step's rows' places in their lanes' trees (ops::TreeRow), which the GDN and
// attention layers read; and the rows the draft reads (the captured layers'
// rows), the final norm's rows and the logits of every row of the step. A
// lane's auxiliary state before the step and its pages' block index are its
// sequence's (QwenTargetPrefillSequence::auxiliaryIn, indexTable).
struct QwenTargetDecodeBuffers final {
  ops::StepRows step{};
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> currentGdnStates;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> nextGdnStates;
  metal::MetalBuffer gdnTape;
  std::array<ops::GdnTapeLane, ExecutionLimits::maximumBatchWidth> gdnTapeLanes{};
  metal::MetalBuffer verifyRows;
  metal::MetalBuffer tree;
  metal::MetalBuffer capturedTargetHidden;
  metal::MetalBuffer finalHidden;
  metal::MetalBuffer logits;
};

// A chunk of a prefill command: `rows` rows of its lanes' sequences, packed
// in sequence order, in its own buffers; or a decode step (`decode`), whose
// sequences are its lanes, each of the cycle's lane cap of rows at most
// (ops::StepRows::laneCap), which the step's layout places
// (QwenTargetPrefillBuffers::laneRows: the draft's selector writes it on the
// device), its `rows` the step's.
struct QwenTargetPrefillChunk final {
  QwenTargetPrefillBuffers buffers;
  std::span<const QwenTargetPrefillSequence> sequences;
  uint32_t rows = 0;
  const QwenTargetDecodeBuffers *decode = nullptr;
};

// The layers [first, end) of a prefill command one of its passes runs
// (BatchPlan::passes): the pass with the first layer begins the chunks'
// rows, the one with the last ends them.
struct QwenPrefillLayers final {
  uint32_t first = 0;
  uint32_t end = 0;
};

// A decode step's verify input as its draft's selector lays it out on the
// device (ops::DraftSelector::addVerifyRows): each of the step's rows' token
// and tree row (metal/abi/VerifyTree.h), each lane's rows of the step
// (SplashLaneRows), and the step's rows.
struct QwenVerifyInput final {
  metal::MetalBuffer tokens;
  metal::MetalBuffer treeRows;
  metal::MetalBuffer laneRows;
  ops::StepRows step{};
};

// The auxiliary state after the rows each lane of a verify retained, which
// its commit writes from the state before the step (addStateCommit). The GDN
// state folds the rows into the lane's next step instead (ops::GdnTapeLane).
// For a layout with a block index, each lane's index table and a layer's
// bytes of the index plane (kv::PageStorage::indexLayerBytes).
struct QwenTargetCommitBuffers final {
  // Each lane's accepted path, a row mask of its block (metal/abi/
  // VerifyTree.h), as acceptance wrote it, and its rows of the step as the
  // device holds them (SplashLaneRows).
  metal::MetalBuffer retainedRows;
  metal::MetalBuffer laneRows;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> auxiliaryIn;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> auxiliaryOut;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> indexTables;
  uint32_t indexLayerBytes = 0;
};

// The GDN state after the first rows of the path a lane accepted of its
// verify, folded into a cache cell from the verify's tape (addStateFold): the
// lane's retained rows (the path, a row mask of its block, as acceptance
// wrote it), the cell the verify wrote, which holds the state before its
// rows, the lanes' decode tapes and the offset of the verify's slot in them,
// and the cache cell.
struct QwenTargetFoldBuffers final {
  metal::MetalBuffer retainedRows;
  metal::MetalBuffer state;
  metal::MetalBuffer gdnTape;
  uint64_t stepSlot = 0;
  metal::MetalBuffer fold;
};

// The bytes of a target's own step scratch beside the runtime's arenas, a
// prefill's and a decode's (qwenTargetScratchBytes).
struct QwenTargetScratchBytes final {
  uint64_t prefill = 0;
  uint64_t decode = 0;
};

// A layer's FFN, of one kind: its MoE block, null for a dense FFN, and its
// dense SwiGLU, empty for a MoE block. A family with dense layers declares
// its own pair beside its layer type (Qwen3_8.hpp, Flint.hpp).
template <class Layer> [[nodiscard]] const ops::BlockMoeWeights *qwenMoeBlock(const Layer &layer) noexcept {
  return &layer.ffn;
}
template <class Layer> [[nodiscard]] std::optional<ops::SwiGluProjections> denseFfn(const Layer &) noexcept {
  return std::nullopt;
}

// The projections of a layer's mixer, of either kind (QwenMixerWeights) or
// of one.
template <class Mixer> void includeMixer(QwenTargetGeometry &geometry, const Mixer &mixer) {
  geometry.include(mixer.inputProjection);
  geometry.include(mixer.outputProjection);
}
inline void includeMixer(QwenTargetGeometry &geometry, const QwenMixerWeights &mixer) {
  std::visit([&](const auto &kind) { includeMixer(geometry, kind); }, mixer);
}
// A dense FFN's projections, whose gate and up run as one gate/up plan,
// which takes them of one size and rotation in any formats.
void includeDenseFfn(QwenTargetGeometry &geometry, ops::SwiGluProjections ffn);

// The projections a family's layers and head dispatch beyond their mixers',
// dense FFNs' and MoE blocks' (qwenTargetGeometry): none, but
// Qwen3.8-Flash-Next's (model/Qwen4Exp.hpp).
template <class Weights> void includeFamilyProjections(QwenTargetGeometry &, const Weights &) {}

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
    includeMixer(geometry, layer.mixer);
    if (const ops::BlockMoeWeights *block = qwenMoeBlock(layer)) moeBlocks.push_back(block);
    if (const std::optional<ops::SwiGluProjections> dense = denseFfn(layer)) includeDenseFfn(geometry, *dense);
  }
  includeFamilyProjections(geometry, weights);
  geometry.finish(moeBlocks, weights.logitsProjection);
  return geometry;
}

// The target the runtime drives: a family's layers over the blocks every
// Qwen target shares, made for its weights by makeQwenTarget. The shared
// blocks are a step's checks and plans, the GDN and attention mixers (one
// wiring for a prompt chunk's and a decode step's rows), the FFNs of a
// residual of one row (a dense SwiGLU and a MoE block's routing and
// experts), the draft's captures and the normed head; each family composes
// them into its layers and residual: the Qwen3.5 architecture's
// (Qwen3.8-27B, Qwen3.6-35B-A3B and Flint, model/QwenTarget.cpp) and
// Qwen3.8-Flash-Next's (model/Qwen4ExpTarget.cpp).
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
  // Lanes of storage, a draft's block of rows each, the vocabulary
  // projection binds over a draft step of `lanes` lanes, which runs the
  // target's head: its tile may hold more rows than the step's
  // (LinearPlan::storageRows; a three-lane step on the staged tile runs its
  // 32-row tile over four lanes). The head still processes the step's lanes;
  // padding rows read stale activations and write results no row reads.
  [[nodiscard]] uint32_t headStorageLanes(uint32_t lanes) const;
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
  // of its last layer's output rows, and else nothing; of those only each
  // sequence's last row is sure to be computed (planOutputRows). A decode
  // step's one chunk (QwenTargetPrefillChunk::decode) ends in its own head
  // over every row and captures every row the draft reads, in the step's
  // buffers: what this returns of it is unread.
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
  // Adds what only a state captured inside a prefill sequence's rows runs:
  // for a sequence of the first chunk from auxiliary state `auxiliaryIn`,
  // capturing after its first row into `captured`. That is Flash-Next's PLE
  // history alone, as a capture's GDN scans and QSA pending keys run the
  // kernels of every prefill (Runtime::warmupPrefill creates its pipelines).
  virtual void addInsideCapture(metal::CommandGraph &graph, metal::MetalBuffer auxiliaryIn,
                                metal::MetalBuffer captured) const;
  // Writes the inputs the target makes on the CPU of batch lane `lane`'s
  // verify rows, those from `position` on of `request`, before the cycle's
  // commands are encoded.
  virtual void prepareVerifyLane(uint32_t lane, uint64_t position, const QwenTargetRequest &request);
  // The head over `rows` rows of the last layer's output: one sweep of the
  // vocabulary projection, over the rows of the first tokens after prompts
  // (Runtime's initial selections), not a decode step's, whose forward ends
  // in a head of its own (addPrefill with a decode chunk).
  virtual void addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden,
                            metal::MetalBuffer finalHidden, metal::MetalBuffer logits,
                            uint32_t rows, ops::LinearScratch scratch) const = 0;
  // The inputs the target makes itself of a step's verify input, which the
  // draft's selector writes and addEmbedding gathers: none, but a family's
  // that reads them itself.
  virtual void addVerifyInput(metal::CommandGraph &graph, const QwenVerifyInput &input) const;
  void addEmbedding(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                    metal::MetalBuffer hidden, uint32_t rows) const;
  // The commit after acceptance of a decode step (`step`, whose lanes' rows
  // the layout places, QwenTargetCommitBuffers::laneRows): in the step's
  // command after its forward, or for constrained decoding in the next
  // command, before any other decode step reuses the step's scratch. Each
  // lane's auxiliary state after its retained rows, from what the forward
  // computed of its rows. None for a target without auxiliary state.
  virtual void addStateCommit(metal::CommandGraph &graph, QwenTargetCommitBuffers buffers,
                              const ops::StepRows &step) const;
  // After a verify's acceptance, the GDN state after its first `rows` rows of
  // one lane, which the lane's next step would hold had the lane retained
  // exactly those, in a cache cell, when the lane retained that many. None
  // for a target without GDN layers; a target with auxiliary state, which
  // its commit writes after every retained row, captures no state in decode
  // (Model::capturesInDecode).
  void addStateFold(metal::CommandGraph &graph, QwenTargetFoldBuffers buffers, uint32_t rows) const;
  // The table rows a lane's rows of the last step read (`rows`, its rows of
  // the step), each row's in order, for tests
  // (RuntimeInstrumentation::lastTableRows); none for a target that reads no
  // table.
  [[nodiscard]] virtual std::vector<uint32_t> verifyTableRows(const ops::LaneRows &rows) const;

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
    // Each sequence's chunk and page table, in sequence order, as the
    // attention layers take them.
    std::array<kv::ChunkedPrefillParams, ExecutionLimits::maximumBatchWidth> chunks{};
    std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> pageTables{};
    // The sequences' attention plan, which every attention layer runs, and
    // with a sliding window (kv::Layout::windowTokens) their windowed plan,
    // which the sliding layers run instead.
    std::optional<ops::AttentionPlan> attention{};
    std::optional<ops::AttentionPlan> windowedAttention{};
    std::optional<ops::MoePlan> moe{};
    // The layer's MoE buffers, which its routing (addPrefillLayer), the
    // layer's load and its routed work (addPrefillExperts) share.
    ops::MoeBuffers ffn{};
    ops::AneFfn *aneFfn = nullptr;
    uint32_t gdnLayer = 0;
    uint32_t attentionLayer = 0;
    // The rows of the layer being added that something reads, ascending, when
    // not every row (planOutputRows): its state writes run on every row, its
    // output work on these alone, read row i as row i.
    std::vector<uint32_t> readRows{};
    // A decode step's own buffers; none for a prompt chunk.
    const QwenTargetDecodeBuffers *decode = nullptr;
    [[nodiscard]] uint32_t outputRows() const noexcept {
      return readRows.empty() ? rows : static_cast<uint32_t>(readRows.size());
    }
    [[nodiscard]] std::span<const kv::ChunkedPrefillParams> attentionSequences() const noexcept {
      return std::span(chunks).first(sequences.size());
    }
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
  // output rows (addPrefillEnd). The steps share the scratch their expert
  // passes gather into and write, all but their routing (targetGroupTensor),
  // so each step's expert work runs through its combine before the next
  // step's begins: the first step's passes over the experts resident before
  // the load run beside its reads (ops::MoE::addResident), each later step's
  // after the step before it.
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
      const ops::BlockMoeWeights *ffn = layers.moe(index);
      const bool streams = ffn && ffn->stream;
      if (streams)
        ffn->stream->addLoad(steps.front().graph, parts, ffn->streamBlock, [&] {
          ops::MoE::addResident(steps.front().graph, steps.front().ffn, *ffn, *steps.front().moe);
        });
      for (size_t at = 0; at < steps.size(); ++at) {
        if (streams && at) ops::MoE::addResident(steps[at].graph, steps[at].ffn, *ffn, *steps[at].moe);
        layers.addPrefillExperts(steps[at], index);
      }
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
  // output projection's `output`: a GDN, and an attention whose rows attend
  // as the layer's variant does (QwenTargetGeometry::attentionVariant), with
  // the layer's sinks where its geometry adds them, and, when given, only the
  // blocks of their lists (`blocks`, one per sequence), through the step's
  // reads of spilled K/V. The normalized rows are as `normalized` describes,
  // the input projection's plan's (a table its producer wrote, or plain).
  void addPrefillMixer(PrefillStep &step, const QwenGdnWeights &mixer, ops::PreparedInput normalized,
                       BlockOutput output) const;
  void addPrefillMixer(PrefillStep &step, const QwenAttentionWeights &mixer, ops::PreparedInput normalized,
                       BlockOutput output, std::span<const ops::AttentionBlocks> blocks = {}) const;
  // The FFNs of a layer whose residual is one row of hiddenSize values: each
  // reads the RMS norm `norm` of the mixer's `residual` rows and adds its
  // output to them into `output`. A dense SwiGLU, in a prefill on the
  // Neural Engine with the step's split of it (layer `index`'s, AneFfn); a
  // MoE block's routing, whose routed work a step adds beside and after the
  // layer's load (addPrefillLayers, ops::MoE::addResident and addRouted with
  // step.ffn).
  void addPrefillDenseFfn(PrefillStep &step, uint32_t index, const ops::NormWeights &norm,
                          ops::SwiGluProjections ffn, metal::MetalBuffer residual, metal::MetalBuffer output) const;
  void addPrefillMoeRouting(PrefillStep &step, const ops::NormWeights &norm, const ops::BlockMoeWeights &ffn,
                            metal::MetalBuffer residual, metal::MetalBuffer output) const;
  // The head of a target whose last layer's rows read a final RMS norm,
  // `finalNorm`, into `finalHidden`, as addHeadBatch takes them.
  void addNormalizedHead(metal::CommandGraph &graph, const ops::NormWeights &finalNorm, metal::MetalBuffer hidden,
                         metal::MetalBuffer finalHidden, metal::MetalBuffer logits, uint32_t rows,
                         ops::LinearScratch scratch) const;
  // Copies `rows` rows of a layer's output rows, from row `sourceRow` of
  // `output`, into capture slot `slot` of the captured hidden rows from row
  // `destinationRow`: the draft's capture of a layer whose output is one row
  // of hiddenSize values.
  void addCapture(metal::CommandGraph &graph, uint32_t slot, metal::MetalBuffer output, uint32_t sourceRow,
                  metal::MetalBuffer captured, uint32_t destinationRow, uint32_t rows) const;
  // Before prefill layer `index`: the target's last layer, an attention
  // layer the draft does not read, outputs each sequence's last row alone,
  // all the runtime reads of it (addPrefill), its MoE on a plan of those rows
  // (step.readRows).
  void planOutputRows(PrefillStep &step, uint32_t index) const;
  // Copies a step's read rows of its output work, rows 0.. of `compact`, to
  // their rows of `output`; nothing when it outputs every row.
  void addReadRows(const PrefillStep &step, const metal::MetalBuffer &compact, const metal::MetalBuffer &output) const;
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
// Qwen3.6-35B-A3B, Flint), whose prefill commands run up to `prefillChunks`
// chunks, and the step scratch a target of its layout keeps itself (none: it
// has no table to stream).
template <class Layout, class Layer>
[[nodiscard]] std::unique_ptr<QwenTarget> makeQwenTarget(const QwenTargetWeights<Layout, Layer> &weights,
                                                         const QwenTargetGeometry &geometry,
                                                         metal::MetalBackend &backend,
                                                         const ops::ExecutionPlans &operators,
                                                         uint32_t prefillChunks);
template <class Layout>
[[nodiscard]] QwenTargetScratchBytes qwenTargetScratchBytes(const Layout &, const QwenTargetGeometry &, uint32_t,
                                                            uint64_t) {
  return {};
}

} // namespace splash::model

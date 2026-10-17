#include "model/QwenTarget.hpp"

#include "model/Qwen3_6Moe.hpp"
#include "model/Qwen3_8.hpp"
#include "model/Qwen4DraftVocabulary.hpp"
#include "model/Qwen4Exp.hpp"
#include "model/Qwen4Steps.hpp"
#include "model/SparseRowStore.hpp"
#include "model/WeightStore.hpp"
#include "metal/abi/Qwen4.h"
#include "ops/AneFfn.hpp"
#include "ops/Embedding.hpp"
#include "ops/Normalization.hpp"
#include "ops/PageStorage.hpp"
#include "ops/Qwen4.hpp"
#include "ops/RowCopy.hpp"
#include "ops/Sampling.hpp"

#include <algorithm>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>

namespace splash::model {
namespace {

template <class Weights>
void requireWeights(const Weights &weights,
                    const QwenTargetGeometry &geometry) {
  const uint32_t attentionLayers = static_cast<uint32_t>(std::count_if(
      weights.layers.begin(), weights.layers.end(), [](const auto &layer) {
        return std::holds_alternative<QwenAttentionWeights>(layer.mixer);
      }));
  if (!geometry.valid() || weights.layers.size() != geometry.layers ||
      attentionLayers != geometry.trunkAttentionLayers()) {
    throw std::invalid_argument(
        "Qwen target weights do not match execution geometry");
  }
}

} // namespace

template <class Layout, class Layer>
QwenTarget::QwenTarget(const QwenTargetWeights<Layout, Layer> &weights,
                       const QwenTargetGeometry &geometry,
                       metal::MetalBackend &backend,
                       const ops::ExecutionPlans &operators, uint32_t)
    : weights_(&weights), weightsBase_(weights), geometry_(geometry),
      backend_(backend), operators_(operators) {
  requireWeights(weights, geometry_);
}

template QwenTarget::QwenTarget(const Qwen3_8Weights &, const QwenTargetGeometry &, metal::MetalBackend &,
                                const ops::ExecutionPlans &, uint32_t);
template QwenTarget::QwenTarget(const Qwen3_6MoeWeights &, const QwenTargetGeometry &, metal::MetalBackend &,
                                const ops::ExecutionPlans &, uint32_t);

namespace {

void includeProjection(QwenTargetGeometry &geometry, const ops::Projection &projection) {
  geometry.decodeProjections.push_back(projection.shape());
}

// The projections a dense layer's FFN dispatches; a MoE block runs its own
// plans (ops::MoeShape). Its gate and up run as one gate/up plan, which takes
// them of one size and rotation in any formats.
void includeFfn(QwenTargetGeometry &geometry, const Qwen3_8LayerWeights &layer) {
  const ops::ProjectionShape gate = layer.gateProjection.shape(), up = layer.upProjection.shape();
  if (gate.outputSize != up.outputSize || gate.inputSize != up.inputSize || gate.rotated != up.rotated)
    throw WeightStoreError("fused gate/up projections must have matching shapes");
  includeProjection(geometry, layer.gateProjection);
  includeProjection(geometry, layer.upProjection);
  includeProjection(geometry, layer.downProjection);
  geometry.gateUpProjections.push_back(up);
}
void includeFfn(QwenTargetGeometry &, const Qwen3_6MoeLayerWeights &) {}

// The format of most routed expert weights of a target's MoE blocks,
// GGUF_FMT_COUNT for none (ops::MoeShape::expertFormat).
uint32_t routedExpertFormat(std::span<const Qwen3_8LayerWeights>) { return GGUF_FMT_COUNT; }
uint32_t routedExpertFormat(std::span<const Qwen3_6MoeLayerWeights> layers) {
  std::array<uint64_t, GGUF_FMT_COUNT> weights{};
  for (const Qwen3_6MoeLayerWeights &layer : layers) {
    const ops::BlockMoeWeights &block = layer.ffn;
    for (const ops::BlockExpertProjection *projection : {&block.gate, &block.up, &block.down})
      if (!projection->routed.isFloat())
        weights[projection->routed.formatId] += uint64_t{projection->routed.outputSize} * projection->routed.inputSize;
  }
  const auto most = std::max_element(weights.begin(), weights.end());
  return *most ? uint32_t(most - weights.begin()) : GGUF_FMT_COUNT;
}
// Whether the register tile takes every routed and shared expert weight of a
// MoE block, and so of a target's MoE blocks (ops::MoeShape::registerFormats).
bool registerTakes(const ops::BlockMoeWeights &block) {
  for (const ops::BlockExpertProjection *projection : {&block.gate, &block.up, &block.down})
    for (const ops::QuantizedSegment *s : {&projection->routed, &projection->shared})
      if (!s->isFloat() && !quant_register_tensor(s->formatId, s->inputSize)) return false;
  return true;
}
bool expertRegisterFormats(std::span<const Qwen3_8LayerWeights>) { return true; }
bool expertRegisterFormats(std::span<const Qwen3_6MoeLayerWeights> layers) {
  return std::ranges::all_of(layers, [](const Qwen3_6MoeLayerWeights &layer) { return registerTakes(layer.ffn); });
}

} // namespace

template <class Layout, class Layer>
QwenTargetGeometry qwenTargetGeometry(const QwenTargetWeights<Layout, Layer> &weights,
                                      std::span<const uint32_t> captureLayers) {
  const Layout &layout = weights.layout;
  QwenTargetGeometry geometry(layout, captureLayers);
  geometry.kvLayout = layout.kvLayout();
  geometry.stateLayout = layout.gdnStateLayout();
  for (const auto &layer : weights.layers) {
    std::visit([&](const auto &mixer) {
      includeProjection(geometry, mixer.inputProjection);
      includeProjection(geometry, mixer.outputProjection);
    }, layer.mixer);
    includeFfn(geometry, layer);
  }
  geometry.moeExpertFormat = routedExpertFormat(weights.layers);
  geometry.moeRegisterFormats = expertRegisterFormats(weights.layers);
  geometry.prefillProjections = geometry.decodeProjections;
  includeProjection(geometry, weights.logitsProjection);
  for (auto *shapes : {&geometry.prefillProjections, &geometry.decodeProjections,
                       &geometry.gateUpProjections}) {
    std::sort(shapes->begin(), shapes->end());
    shapes->erase(std::unique(shapes->begin(), shapes->end()), shapes->end());
  }
  return geometry;
}

template QwenTargetGeometry qwenTargetGeometry(const Qwen3_8Weights &, std::span<const uint32_t>);
template QwenTargetGeometry qwenTargetGeometry(const Qwen3_6MoeWeights &, std::span<const uint32_t>);

QwenTargetGeometry qwenTargetGeometry(const Qwen4ExpWeights &weights, std::span<const uint32_t> captureLayers) {
  const Qwen4ExpLayout &layout = weights.layout;
  QwenTargetGeometry geometry(layout, captureLayers);
  geometry.kvLayout = layout.kvLayout();
  geometry.stateLayout = layout.gdnStateLayout();
  geometry.mtpLayers = layout.mtpLayers;
  if (static_cast<bool>(weights.mtp) != (layout.mtpLayers == 1))
    throw WeightStoreError("the MTP head does not match the layout");
  if (weights.pleStore) {
    geometry.pleStreamRowBytes = weights.pleStore->table().rowBytes;
    geometry.pleStreamFormat = weights.pleFormat;
  }
  const auto hyper = [&](const Qwen4HyperWeights &mix) {
    includeProjection(geometry, mix.down);
    includeProjection(geometry, mix.up);
  };
  std::array<uint64_t, GGUF_FMT_COUNT> expertWeights{};
  geometry.moeRegisterFormats = true;
  for (const Qwen4ExpLayerWeights &layer : weights.layers) {
    geometry.moeRegisterFormats &= registerTakes(layer.ffn);
    hyper(layer.mixerHyper);
    hyper(layer.ffnHyper);
    std::visit([&](const auto &mixer) {
      includeProjection(geometry, mixer.inputProjection);
      includeProjection(geometry, mixer.outputProjection);
    }, layer.mixer);
    if (layer.indexer) {
      includeProjection(geometry, layer.indexer->query);
      includeProjection(geometry, layer.indexer->key);
    }
    if (layer.ple) {
      includeProjection(geometry, layer.ple->key);
      includeProjection(geometry, layer.ple->value);
    }
    const ops::BlockMoeWeights &block = layer.ffn;
    for (const ops::BlockExpertProjection *projection : {&block.gate, &block.up, &block.down})
      if (!projection->routed.isFloat())
        expertWeights[projection->routed.formatId] +=
            uint64_t{projection->routed.outputSize} * projection->routed.inputSize;
  }
  hyper(weights.finalHyper);
  if (weights.mtp) {
    const Qwen4MtpWeights &mtp = *weights.mtp;
    geometry.moeRegisterFormats &= registerTakes(mtp.block.ffn);
    hyper(mtp.block.mixerHyper);
    hyper(mtp.block.ffnHyper);
    hyper(mtp.head);
    std::visit([&](const auto &mixer) {
      includeProjection(geometry, mixer.inputProjection);
      includeProjection(geometry, mixer.outputProjection);
    }, mtp.block.mixer);
    includeProjection(geometry, mtp.block.indexer->query);
    includeProjection(geometry, mtp.block.indexer->key);
    includeProjection(geometry, mtp.embedHidden);
    // The draft vocabulary's head runs at decode alone, on plans of its own
    // shape (their scratch and storage rows), and its context rows as one
    // quantized tensor of the vocabulary projection's format.
    if (mtp.draftHead) {
      includeProjection(geometry, mtp.draftHead->logits);
      const auto &segments = weights.logitsProjection.blocks().segments;
      if (segments.size() == 1 && !segments.front().isFloat() && !weights.logitsProjection.rotation) {
        geometry.mtpDraftContextFormat = segments.front().formatId;
        geometry.decodeProjections.push_back({kQwen4DraftContextTokens, layout.hiddenSize, false});
      }
    }
  }
  const auto most = std::max_element(expertWeights.begin(), expertWeights.end());
  geometry.moeExpertFormat = *most ? uint32_t(most - expertWeights.begin()) : GGUF_FMT_COUNT;
  geometry.prefillProjections = geometry.decodeProjections;
  includeProjection(geometry, weights.logitsProjection);
  for (auto *shapes : {&geometry.prefillProjections, &geometry.decodeProjections, &geometry.gateUpProjections}) {
    std::sort(shapes->begin(), shapes->end());
    shapes->erase(std::unique(shapes->begin(), shapes->end()), shapes->end());
  }
  return geometry;
}

QwenTarget::QwenTarget(const Qwen4ExpWeights &weights, const QwenTargetGeometry &geometry,
                       metal::MetalBackend &backend, const ops::ExecutionPlans &operators, uint32_t prefillChunks)
    : weights_(&weights), weightsBase_(weights), geometry_(geometry), backend_(backend), operators_(operators),
      qwen4Steps_(std::make_unique<Qwen4Steps>(weights, geometry_, backend, prefillChunks)) {
  requireWeights(weights, geometry_);
}

QwenTarget::QwenTarget(QwenTarget &&) noexcept = default;
QwenTarget::~QwenTarget() = default;

QwenTargetScratchBytes QwenTarget::scratchBytes(const QwenTargetGeometry &geometry, uint32_t prefillChunks) {
  // A hyper-connected target's streams, PLE and QSA scratch (model/Qwen4Steps.hpp).
  return geometry.hyperConnected() ? Qwen4Steps::scratchBytes(geometry, prefillChunks) : QwenTargetScratchBytes{};
}

QwenTargetScratchBytes QwenTarget::allocatedScratchBytes() const noexcept {
  return qwen4Steps_ ? qwen4Steps_->allocatedBytes() : QwenTargetScratchBytes{};
}

const Qwen4ExpWeights *QwenTarget::qwen4() const noexcept {
  const auto *weights = std::get_if<const Qwen4ExpWeights *>(&weights_);
  return weights ? *weights : nullptr;
}

const ops::Projection &QwenTarget::vocabularyProjection() const noexcept {
  return weightsBase_.logitsProjection;
}

uint32_t QwenTarget::decodeStorageLanes(uint32_t lanes) const {
  const uint32_t rows = lanes * ExecutionLimits::targetVerifyRows;
  uint32_t storageRows = rows;
  for (const auto &shape : geometry_.decodeProjections)
    storageRows = std::max(storageRows, operators_.linear().decodeStorageRows(rows, shape));
  return storageRows / ExecutionLimits::targetVerifyRows;
}

namespace {

// Rows [begin, begin + count) of a row-major buffer of `width` values of T.
template <class T>
metal::MetalBuffer rowsOf(metal::MetalBackend &backend, const metal::MetalBuffer &buffer, uint32_t begin,
                          uint32_t count, uint32_t width) {
  return backend.view(buffer, uint64_t{begin} * width * sizeof(T), uint64_t{count} * width * sizeof(T));
}

ops::HyperShape hyperShape(const QwenTargetGeometry &g) { return {g.hiddenSize, g.hyperConnections, g.hyperRank}; }
ops::PleShape pleShape(const QwenTargetGeometry &g) {
  return {g.hiddenSize, g.hyperConnections, g.pleConvolutionTaps, g.pleNgram};
}

// The QSA indexer's fp32 projections of `rows` rows: one float segment each
// (Qwen4Exp.cpp readIndexer), at any row count.
void addIndexerProjection(metal::CommandGraph &graph, const ops::Linear &linear, metal::MetalBuffer input,
                          const ops::Projection &projection, metal::MetalBuffer output, uint32_t rows) {
  const auto &segments = projection.blocks().segments;
  if (segments.size() != 1 || !segments.front().isFloat())
    throw std::invalid_argument("the QSA indexer projections are single float segments");
  ops::addGgufFloat(graph, std::move(input), segments.front(), std::move(output), rows, projection.outputSize, 0,
                    ops::FloatOutput::Float32, linear.ggufFloatTile(rows, projection.outputSize));
}

// The rows a decode step verifies of each of its lanes.
constexpr uint32_t kVerifyRows = ExecutionLimits::targetVerifyRows;

// A decode projection of a step of `lanes` lanes through its own plan;
// `prepared` describes the scratch table its producer wrote for that plan's
// input.
void addDecodeProjection(metal::CommandGraph &graph, const ops::Linear &linear, metal::MetalBuffer input,
                         const ops::Projection &projection, metal::MetalBuffer output, uint32_t lanes,
                         ops::LinearScratch scratch, ops::PreparedInput prepared = {}) {
  linear.add(graph,
             {.input = std::move(input), .output = std::move(output), .scratch = scratch,
              .prepared = std::move(prepared)},
             projection, linear.decodePlan(projection, lanes));
}

void requireLayerPartition(const QwenTargetGeometry &geometry, uint32_t gdnLayers, uint32_t attentionLayers) {
  if (gdnLayers != geometry.stateLayout.layers ||
      attentionLayers != geometry.trunkAttentionLayers())
    throw std::logic_error("Qwen target layer partition mismatch");
}

// A prefill sequence binds every GDN layer's state cells, in and out, and so
// does each state it captures.
void requirePrefillStates(const QwenTargetGeometry &geometry, const QwenTargetPrefillSequence &sequence) {
  const auto everyLayer = [&](std::span<const metal::MetalBuffer> views) {
    return views.size() == geometry.stateLayout.layers;
  };
  if (!everyLayer(sequence.convolutionIn) || !everyLayer(sequence.convolutionOut) ||
      !everyLayer(sequence.recurrentIn) || !everyLayer(sequence.recurrentOut) ||
      !std::ranges::all_of(sequence.stateCaptures, [&](const QwenTargetStateCapture &capture) {
        return everyLayer(capture.convolution) && everyLayer(capture.recurrent) &&
               (capture.auxiliary || !geometry.stateLayout.auxiliaryBytes);
      }))
    throw std::invalid_argument("Qwen prefill state layer mismatch");
}

// Copies `rows` rows of a capture layer's output, from row `sourceRow`, into
// capture slot `slot` of the captured hidden rows from row `destinationRow`.
void addCapture(metal::CommandGraph &graph, const QwenTargetGeometry &geometry, uint32_t slot,
                metal::MetalBuffer output, uint32_t sourceRow, metal::MetalBuffer captured,
                uint32_t destinationRow, uint32_t rows) {
  const uint32_t width = geometry.hiddenSize, capturedWidth = geometry.capturedHiddenSize();
  if (slot >= capturedWidth / width)
    throw std::logic_error("Qwen target capture slot past the captured hidden rows");
  ops::RowCopy::add(graph, std::move(output), {sourceRow, width, 0}, std::move(captured),
                    {destinationRow, capturedWidth, slot * width}, rows, width);
}

// Qwen3.8-Flash-Next's capture: the mean of a capture layer's output
// streams, `rows` rows from row `sourceRow` of `streams`, into capture slot
// `slot` of the captured hidden rows from row `destinationRow`.
void addStreamCapture(metal::CommandGraph &graph, metal::MetalBackend &backend, const QwenTargetGeometry &geometry,
                      uint32_t slot, const metal::MetalBuffer &streams, uint32_t sourceRow,
                      metal::MetalBuffer captured, uint32_t destinationRow, uint32_t rows) {
  const uint32_t width = geometry.hiddenSize, capturedWidth = geometry.capturedHiddenSize();
  if (slot >= capturedWidth / width)
    throw std::logic_error("Qwen target capture slot past the captured hidden rows");
  ops::Qwen4::addStreamMean(graph, rowsOf<float>(backend, streams, sourceRow, rows, geometry.streamWidth()),
                            std::move(captured), {destinationRow, capturedWidth, slot * width}, hyperShape(geometry),
                            rows);
}

// A layer's MoE block, null for a dense FFN.
const ops::BlockMoeWeights *moeOf(const Qwen3_8LayerWeights &) { return nullptr; }
template <class Layer> const ops::BlockMoeWeights *moeOf(const Layer &layer) { return &layer.ffn; }

} // namespace

// The state a prefill command's layers share for one of its chunks: its
// rows, their buffers and plans, the buffers the layer's MoE routes and runs
// through, and the next GDN and attention layer of the step.
struct QwenTarget::PrefillStep {
  metal::CommandGraph &graph;
  const QwenTargetPrefillBuffers &buffers;
  std::span<const QwenTargetPrefillSequence> sequences;
  uint32_t rows;
  std::span<const SplashKvLayer> kvLayers;
  // Each sequence's attention plan, which every attention layer runs.
  std::vector<ops::PrefillAttentionPlan> attention{};
  std::optional<ops::MoePlan> moe{};
  // The layer's MoE buffers, which its routing (addPrefillLayer), the layer's
  // load and its routed work (addPrefillExperts) share.
  ops::MoeBuffers ffn{};
  ops::AneFfn *aneFfn = nullptr;
  uint32_t gdnLayer = 0;
  uint32_t attentionLayer = 0;
  // Qwen3.8-Flash-Next: the chunk's scratch, the QSA blocks each sequence's
  // rows complete and where each attention layer's QSA index sits in the KV
  // extents.
  Qwen4Scratch qwen4{};
  std::vector<Qwen4Blocks> blocks{};
  std::span<const ops::QsaIndex> qsaIndex{};
};

// Qwen3.8-Flash-Next's sparse attention of an attention layer: its indexer
// and the blocks each row keeps.
struct QwenTarget::PrefillQsa {
  const Qwen4IndexerWeights &indexer;
  uint32_t topBlocks = 0;
};

struct QwenTarget::VerifyStep {
  metal::CommandGraph &graph;
  const QwenTargetVerifyBuffers &buffers;
  std::span<const SplashKvLayer> kvLayers;
  std::span<const kv::ChunkedPrefillParams> chunks;
  uint32_t lanes;
  uint32_t rows;
  ops::VerifyAttentionPlan attention;
  std::optional<ops::MoePlan> moe{};
  uint32_t gdnLayer = 0;
  uint32_t attentionLayer = 0;
  // Qwen3.8-Flash-Next after its MTP head's draft: each lane's rows from
  // here on route to no expert (addVerify liveRows); 0 routes every row.
  uint32_t liveRows = 0;
  // Qwen3.8-Flash-Next: as PrefillStep's, and the blocks the lanes complete.
  Qwen4Scratch qwen4{};
  Qwen4LaneBlocks blocks{};
  std::span<const ops::QsaIndex> qsaIndex{};
};

namespace {

// Where each attention layer's QSA index sits in the KV extents of `kv`.
std::vector<ops::QsaIndex> qsaIndex(const kv::PageStorage &kv, uint32_t layers) {
  std::vector<ops::QsaIndex> result;
  for (uint32_t layer = 0; layer < layers; ++layer) {
    const kv::PageStorage::IndexPlacement placement = kv.indexPlacement(layer);
    result.push_back({placement.keysOffset, placement.pooledOffset});
  }
  return result;
}

} // namespace

std::vector<metal::MetalBuffer> QwenTarget::addPrefill(metal::CommandGraph &graph,
                                                       std::span<const QwenTargetPrefillChunk> chunks,
                                                       const kv::PageStorage &kv, ops::AneFfn *aneFfn) {
  const std::span<const SplashKvLayer> kvLayers = kv.layers();
  // The Neural Engine splits a dense target's commands, which are one chunk.
  if (chunks.empty() || kvLayers.size() != geometry_.kvLayout.attentionLayers || (aneFfn && chunks.size() > 1))
    throw std::invalid_argument("invalid Qwen prefill");
  std::vector<ops::QsaIndex> indexes;
  if (qwen4Steps_) indexes = qsaIndex(kv, geometry_.kvLayout.attentionLayers);
  std::vector<PrefillStep> steps;
  steps.reserve(chunks.size());
  for (uint32_t index = 0; index < chunks.size(); ++index) {
    const QwenTargetPrefillChunk &chunk = chunks[index];
    if (chunk.sequences.empty() || chunk.sequences.size() > ExecutionLimits::maximumBatchWidth || !chunk.rows ||
        chunk.rows > ExecutionLimits::prefillTokenBudget)
      throw std::invalid_argument("invalid Qwen prefill chunk");
    PrefillStep &step = steps.emplace_back(PrefillStep{graph, chunk.buffers, chunk.sequences, chunk.rows, kvLayers});
    for (const QwenTargetPrefillSequence &sequence : chunk.sequences) {
      requirePrefillStates(geometry_, sequence);
      step.attention.push_back(
          operators_.prefillAttention(sequence.rows, geometry_.attentionQueryHeads, geometry_.kvLayout));
    }
    if (geometry_.ffnKind == QwenFfnKind::SparseMoe) step.moe = operators_.moePrefill(geometry_.moeShape(), chunk.rows);
    if (aneFfn && aneFfn->splits(chunk.rows)) step.aneFfn = aneFfn;
    if (qwen4Steps_) {
      step.qwen4 = qwen4Steps_->prefillScratch(index);
      step.blocks = qwen4Steps_->preparePrefill(index, chunk);
      step.qsaIndex = indexes;
    }
  }
  if (qwen4Steps_) qwen4Steps_->readAhead(chunks);
  return std::visit([&](const auto *weights) { return addPrefillLayers(steps, *weights); }, weights_);
}

template <class Weights>
std::vector<metal::MetalBuffer> QwenTarget::addPrefillLayers(std::span<PrefillStep> steps,
                                                             const Weights &weights) const {
  for (PrefillStep &step : steps) addPrefillBegin(step, weights);
  std::vector<ops::MoeLoadPart> parts;
  if (geometry_.ffnKind == QwenFfnKind::SparseMoe)
    for (const PrefillStep &step : steps) parts.push_back({&step.ffn, &*step.moe});
  for (uint32_t index = 0; index < geometry_.layers; ++index) {
    for (PrefillStep &step : steps) addPrefillLayer(step, weights, index);
    // Streamed experts: one load of those any step's rows route.
    if (const ops::BlockMoeWeights *ffn = moeOf(weights.layers[index]); ffn && ffn->stream)
      ffn->stream->addLoad(steps.front().graph, parts, ffn->streamBlock);
    for (PrefillStep &step : steps) addPrefillExperts(step, weights, index);
  }
  std::vector<metal::MetalBuffer> outputs;
  outputs.reserve(steps.size());
  for (PrefillStep &step : steps) {
    requireLayerPartition(geometry_, step.gdnLayer, step.attentionLayer);
    outputs.push_back(addPrefillEnd(step, weights));
  }
  return outputs;
}

metal::MetalBuffer QwenTarget::addPrefillGdn(PrefillStep &step, const QwenGdnWeights &mixer) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const uint32_t layer = step.gdnLayer++;
  operators_.linear().addPrefill(step.graph, b.normalized, mixer.inputProjection, b.gdnPacked, step.rows,
                                 b.linearScratch);
  for (const QwenTargetPrefillSequence &sequence : step.sequences) {
    const auto u16 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
      return rowsOf<uint16_t>(backend_, buffer, sequence.rowBegin, sequence.rows, width);
    };
    const auto f32 = [&](const metal::MetalBuffer &buffer, uint32_t width) {
      return rowsOf<float>(backend_, buffer, sequence.rowBegin, sequence.rows, width);
    };
    std::vector<ops::GdnPrefillCapture> captures;
    captures.reserve(sequence.stateCaptures.size());
    for (const QwenTargetStateCapture &capture : sequence.stateCaptures)
      captures.push_back({capture.row, capture.convolution[layer], capture.recurrent[layer]});
    ops::GDN::addPrefill(
        step.graph, backend_,
        {u16(b.gdnPacked, geometry_.packedGdnWidth), mixer.convolutionWeights, sequence.convolutionIn[layer],
         sequence.convolutionOut[layer], u16(b.gdnQueries, geometry_.gdnKeyWidth()),
         u16(b.gdnKeys, geometry_.gdnKeyWidth()), u16(b.gdnValues, geometry_.attentionWidth), mixer.decay,
         mixer.timeBias, f32(b.gdnDecay, geometry_.gdnValueHeads), u16(b.gdnBeta, geometry_.gdnValueHeads),
         sequence.recurrentIn[layer], sequence.recurrentOut[layer], u16(b.recurrent, geometry_.attentionWidth),
         mixer.mixerNorm, u16(b.gdnHidden, geometry_.attentionWidth)},
        geometry_.gdnShape(), sequence.rows, geometry_.stateLayout.format, mixer.outputHeadOrder, captures);
  }
  return b.gdnHidden;
}

// With `qsa`, the indexer's projections of the step's rows and each
// sequence's QSA keys and block index (writing its rows' and its completed
// blocks' entries) come first; a sequence whose rows attend more blocks than
// they keep then selects theirs, and its attention's tiles of rows attend
// through the block lists their selections make.
metal::MetalBuffer QwenTarget::addPrefillAttention(PrefillStep &step, const QwenAttentionWeights &mixer,
                                                   const PrefillQsa *qsa) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const Qwen4Scratch &h = step.qwen4;
  const ops::Linear &linear = operators_.linear();
  const uint32_t layer = step.attentionLayer++;
  constexpr uint32_t indexWidth = QWEN4_QSA_HEADS * QWEN4_QSA_DIMENSION;
  const auto u16Rows = [&](const metal::MetalBuffer &buffer, const QwenTargetPrefillSequence &sequence,
                           uint32_t width) {
    return rowsOf<uint16_t>(backend_, buffer, sequence.rowBegin, sequence.rows, width);
  };
  const auto f32Rows = [&](const metal::MetalBuffer &buffer, const QwenTargetPrefillSequence &sequence,
                           uint32_t width) {
    return rowsOf<float>(backend_, buffer, sequence.rowBegin, sequence.rows, width);
  };
  // A sequence's QSA is a dispatch of one lane.
  std::vector<uint32_t> firstPositions;
  for (const QwenTargetPrefillSequence &sequence : step.sequences)
    firstPositions.push_back(static_cast<uint32_t>(sequence.chunk.committed_tokens));
  const auto qsaRows = [&](size_t index) {
    const QwenTargetPrefillSequence &sequence = step.sequences[index];
    return ops::QsaRows{sequence.rows, std::span(firstPositions).subspan(index, 1), {&sequence.pageTable, 1},
                        qsa->topBlocks, h.maskWords, h.scoreStride};
  };
  linear.addPrefill(step.graph, b.normalized, mixer.inputProjection, b.fullPacked, step.rows, b.linearScratch);
  // Each sequence's indexer projects its rows and pools the blocks they
  // complete; a sequence whose rows attend sparsely selects their blocks, and
  // its tiles attend through the block lists of the selections, each
  // sequence's after the lists of those before it.
  std::vector<ops::AttentionBlocks> blocks(step.sequences.size());
  if (qsa) {
    addIndexerProjection(step.graph, linear, b.normalized, qsa->indexer.query, h.indexerQuery, step.rows);
    addIndexerProjection(step.graph, linear, b.normalized, qsa->indexer.key, h.indexerKey, step.rows);
    std::vector<ops::BlockListRows> sparseRows;
    uint32_t lists = 0;
    for (size_t index = 0; index < step.sequences.size(); ++index) {
      const QwenTargetPrefillSequence &sequence = step.sequences[index];
      ops::Qwen4::addQsaProject(step.graph, f32Rows(h.indexerQuery, sequence, indexWidth),
                                f32Rows(h.indexerKey, sequence, QWEN4_QSA_DIMENSION), qsa->indexer.queryNorm,
                                f32Rows(b.ropeCos, sequence, geometry_.rotaryPairs),
                                f32Rows(b.ropeSin, sequence, geometry_.rotaryPairs), step.qsaIndex[layer],
                                u16Rows(h.qsaQueries, sequence, indexWidth), qsaRows(index));
      ops::Qwen4::addQsaPool(step.graph,
                             {step.blocks[index].records, {&step.blocks[index].count, 1}, 0, {&sequence.pageTable, 1}},
                             qsa->indexer.keyNorm, b.ropeInverseFrequencies, step.qsaIndex[layer]);
      if (!ops::Qwen4::qsaSparse(firstPositions[index], sequence.rows, qsa->topBlocks)) continue;
      ops::Qwen4::addQsaSelect(step.graph, u16Rows(h.qsaQueries, sequence, indexWidth), step.qsaIndex[layer],
                               h.qsaScores, h.scoreRows,
                               rowsOf<uint32_t>(backend_, h.qsaMask, sequence.rowBegin, sequence.rows, h.maskWords),
                               qsaRows(index));
      blocks[index] = {h.qsaLists, h.qsaListCounts, h.listStride, lists};
      sparseRows.push_back({sequence.rowBegin, sequence.rows, firstPositions[index], lists});
      lists += ops::PagedAttention::prefillBlockListCount(sequence.rows);
    }
    if (!sparseRows.empty())
      ops::PagedAttention::addPrefillBlockLists(step.graph, h.qsaMask, h.maskWords, sparseRows,
                                                {h.qsaLists, h.qsaListCounts, h.listStride, 0});
  }
  for (size_t index = 0; index < step.sequences.size(); ++index) {
    const QwenTargetPrefillSequence &sequence = step.sequences[index];
    const uint64_t headBytes = uint64_t{sequence.attentionStride} * geometry_.attentionHeadDimension * sizeof(uint16_t);
    const uint64_t queryBytes = geometry_.attentionQueryHeads * headBytes;
    const uint64_t kvBytes = geometry_.attentionKvHeads * headBytes;
    const metal::MetalBuffer queries = backend_.view(b.fullQueries, sequence.queryOffset, queryBytes);
    const metal::MetalBuffer attentionRows = backend_.view(b.fullAttention, sequence.queryOffset, queryBytes);
    const metal::MetalBuffer keys = backend_.view(b.chunkKeys, sequence.kvOffset, kvBytes);
    const metal::MetalBuffer values = backend_.view(b.chunkValues, sequence.kvOffset, kvBytes);
    ops::PagedAttention::addPrefillProjection(
        step.graph, u16Rows(b.fullPacked, sequence, geometry_.packedFullWidth), mixer.queryNorm, mixer.keyNorm,
        f32Rows(b.ropeCos, sequence, geometry_.rotaryPairs), f32Rows(b.ropeSin, sequence, geometry_.rotaryPairs),
        queries, keys, values, sequence.rows, sequence.attentionStride, geometry_.attentionQueryHeads,
        geometry_.kvLayout);
    ops::PagedAttention::addPrefillStore(step.graph, step.kvLayers[layer], keys, values, sequence.pageTable,
                                         sequence.chunk, geometry_.kvLayout);
    ops::PagedAttention::addPrefill(step.graph, step.kvLayers[layer], queries, attentionRows, b.attentionPartials,
                                    b.attentionStatistics, sequence.pageTable, sequence.chunk, step.attention[index],
                                    blocks[index]);
    ops::PagedAttention::addPrefillGate(
        step.graph, u16Rows(b.fullPacked, sequence, geometry_.packedFullWidth), attentionRows,
        u16Rows(b.attentionHidden, sequence, geometry_.attentionWidth), sequence.rows, sequence.attentionStride,
        geometry_.attentionQueryHeads, geometry_.kvLayout);
  }
  return b.attentionHidden;
}

// Qwen3.5 and Qwen3.6 (model/Qwen3_8.hpp, model/Qwen3_6Moe.hpp). Layer l
// reads its input rows from buffers.hidden[l & 1] and adds its mixer's and
// its FFN's output to them into the other; the layers the draft reads copy
// their output rows into buffers.captured.

template <class Layout, class Layer>
void QwenTarget::addPrefillBegin(PrefillStep &, const QwenTargetWeights<Layout, Layer> &) const {}

template <class Layout, class Layer>
void QwenTarget::addPrefillLayer(PrefillStep &step, const QwenTargetWeights<Layout, Layer> &weights,
                                 uint32_t index) const {
  const Layer &layer = weights.layers[index];
  const QwenTargetPrefillBuffers &b = step.buffers;
  const metal::MetalBuffer input = b.hidden[index & 1];
  addPrefillNorm(step, input, layer.inputNorm);
  const metal::MetalBuffer residual = std::visit(
      [&](const auto &mixer) {
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(mixer)>, QwenGdnWeights>) {
          addPrefillOutput(step, addPrefillGdn(step, mixer), mixer.outputProjection, input, b.gdnOutput);
          return b.gdnOutput;
        } else {
          addPrefillOutput(step, addPrefillAttention(step, mixer, nullptr), mixer.outputProjection, input,
                           b.attentionOutput);
          return b.attentionOutput;
        }
      },
      layer.mixer);
  addPrefillFfn(step, index, layer, residual, b.hidden[(index & 1) ^ 1]);
}

template <class Layout, class Layer>
void QwenTarget::addPrefillExperts(PrefillStep &step, const QwenTargetWeights<Layout, Layer> &weights,
                                   uint32_t index) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  if (const ops::BlockMoeWeights *ffn = moeOf(weights.layers[index]))
    ops::MoE::addRouted(step.graph, step.ffn, *ffn, *step.moe);
  if (const auto slot = geometry_.captureSlot(index))
    for (const QwenTargetPrefillSequence &sequence : step.sequences)
      for (uint32_t capture = 0; capture < sequence.captureCount; ++capture) {
        const QwenTargetPrefillCapture &c = sequence.captures[capture];
        addCapture(step.graph, geometry_, *slot, b.hidden[(index & 1) ^ 1], c.sourceStart, b.captured,
                   c.destinationStart, c.rows);
      }
}

template <class Layout, class Layer>
metal::MetalBuffer QwenTarget::addPrefillEnd(PrefillStep &step, const QwenTargetWeights<Layout, Layer> &) const {
  return step.buffers.hidden[geometry_.layers & 1];
}

void QwenTarget::addPrefillNorm(PrefillStep &step, metal::MetalBuffer input, const ops::NormWeights &norm) const {
  ops::Normalization::addRms(step.graph, input, norm, step.buffers.normalized, geometry_.hiddenSize, step.rows);
}

void QwenTarget::addPrefillOutput(PrefillStep &step, metal::MetalBuffer hidden, const ops::Projection &projection,
                                  metal::MetalBuffer input, metal::MetalBuffer output) const {
  operators_.linear().addPrefillResidual(step.graph, hidden, projection, input, output, step.rows,
                                         step.buffers.linearScratch);
}

void QwenTarget::addPrefillFfn(PrefillStep &step, uint32_t index, const Qwen3_8LayerWeights &layer,
                               metal::MetalBuffer residual, metal::MetalBuffer output) const {
  addPrefillNorm(step, residual, layer.postAttentionNorm);
  const ops::PrefillFfnBuffers ffn = step.buffers.ffn();
  if (step.aneFfn)
    step.aneFfn->add(step.graph, index, ffn, residual, output, step.rows);
  else
    operators_.linear().addPrefillSwiGlu(step.graph, {&layer.gateProjection, &layer.upProjection,
                                                      &layer.downProjection},
                                         ffn, residual, output, step.rows);
}

void QwenTarget::addPrefillFfn(PrefillStep &step, uint32_t, const Qwen3_6MoeLayerWeights &layer,
                               metal::MetalBuffer residual, metal::MetalBuffer output) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  addPrefillNorm(step, residual, layer.postAttentionNorm);
  step.ffn = {b.normalized, residual, output, b.moe};
  ops::MoE::addRouting(step.graph, step.ffn, layer.ffn, *step.moe);
}

// A decode step over the lanes' verify rows: the verify's layers
// (addVerify) or the MTP head's draft steps (addMtpDraft).
void QwenTarget::withVerifyStep(metal::CommandGraph &graph, const QwenTargetVerifyBuffers &buffers,
                                const kv::PageStorage &kv, std::span<const kv::ChunkedPrefillParams> chunks,
                                uint32_t lanes, const std::function<void(VerifyStep &)> &use) const {
  const std::span<const SplashKvLayer> kvLayers = kv.layers();
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth || chunks.size() != lanes ||
      kvLayers.size() != geometry_.kvLayout.attentionLayers ||
      buffers.chunkKeys.size() != geometry_.kvLayout.attentionLayers ||
      buffers.chunkValues.size() != geometry_.kvLayout.attentionLayers) {
    throw std::invalid_argument("invalid Qwen verify batch");
  }
  const uint32_t rows = lanes * kVerifyRows;
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> histories{};
  for (uint32_t lane = 0; lane < lanes; ++lane)
    histories[lane] = chunks[lane].committed_tokens;
  VerifyStep step{graph, buffers, kvLayers, chunks, lanes, rows,
                  operators_.verifyAttention(lanes, geometry_.attentionQueryHeads, geometry_.kvLayout,
                                             std::span(histories).first(lanes))};
  if (geometry_.ffnKind == QwenFfnKind::SparseMoe) step.moe = operators_.moeDecode(geometry_.moeShape(), lanes);
  std::vector<ops::QsaIndex> indexes;
  if (qwen4Steps_) {
    indexes = qsaIndex(kv, geometry_.kvLayout.attentionLayers);
    step.qwen4 = qwen4Steps_->verifyScratch(decodeStorageLanes(lanes) * kVerifyRows);
    step.qsaIndex = indexes;
    std::array<uint32_t, ExecutionLimits::maximumBatchWidth> positions{};
    for (uint32_t lane = 0; lane < lanes; ++lane) positions[lane] = chunks[lane].committed_tokens;
    step.blocks = qwen4Steps_->verifyBlocks(std::span(positions).first(lanes));
  }
  use(step);
}

void QwenTarget::addVerify(
    metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
    const kv::PageStorage &kv,
    std::span<const kv::ChunkedPrefillParams> chunks, uint32_t lanes, uint32_t liveRows) const {
  if (liveRows && (!hasMtp() || liveRows >= kVerifyRows)) throw std::invalid_argument("invalid Qwen verify live rows");
  withVerifyStep(graph, buffers, kv, chunks, lanes, [&](VerifyStep &step) {
    step.liveRows = liveRows;
    std::visit([&](const auto *weights) { addVerifyLayers(step, *weights); }, weights_);
  });
}

// Qwen3.5 and Qwen3.6: layer l reads its input rows from buffers.hidden[l & 1]
// and adds its mixer's and its FFN's output to them into the other; the final
// norm and head follow.
template <class Layout, class Layer>
void QwenTarget::addVerifyLayers(VerifyStep &step, const QwenTargetWeights<Layout, Layer> &weights) const {
  const QwenTargetVerifyBuffers &buffers = step.buffers;
  for (uint32_t index = 0; index < geometry_.layers; ++index) {
    const auto &layer = weights.layers[index];
    const metal::MetalBuffer input = buffers.hidden[index & 1];
    const metal::MetalBuffer output = buffers.hidden[(index & 1) ^ 1];
    const metal::MetalBuffer residual = std::visit(
        [&](const auto &mixer) { return addVerifyMixer(step, mixer, layer.inputNorm, input); }, layer.mixer);
    addVerifyFfn(step, layer, residual, output);
    if (const auto slot = geometry_.captureSlot(index))
      addCapture(step.graph, geometry_, *slot, output, 0, buffers.capturedTargetHidden, 0, step.rows);
  }
  requireLayerPartition(geometry_, step.gdnLayer, step.attentionLayer);
  addHeadBatch(step.graph, buffers.hidden[geometry_.layers & 1], buffers.finalHidden, buffers.logits, step.lanes,
               buffers.linearScratch);
}

// Each producer emits the table (if any) its consumer's plan reads.
metal::MetalBuffer QwenTarget::addVerifyMixer(VerifyStep &step, const QwenGdnWeights &mixer,
                                              const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const uint32_t layer = step.gdnLayer++;
  const ops::LinearPlan inputPlan = linear.decodePlan(mixer.inputProjection, step.lanes);
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      step.graph, input, norm, b.normalized, geometry_.hiddenSize, step.rows, b.linearScratch, inputPlan.input());
  linear.add(step.graph,
             {.input = b.normalized, .output = b.gdnPacked, .scratch = b.linearScratch,
              .prepared = normalized},
             mixer.inputProjection, inputPlan);
  const ops::LinearPlan outputPlan =
      linear.decodePlan(mixer.outputProjection, step.lanes, ops::LinearEpilogue::Residual);
  const ops::PreparedInput hidden = ops::GDN::addDecode(
      step.graph,
      {b.gdnPacked, mixer.convolutionWeights, b.currentGdnStates, b.nextGdnStates, b.gdnTape,
       std::span(b.gdnTapeLanes).first(step.lanes), mixer.decay, mixer.timeBias, mixer.mixerNorm, b.gdnHidden,
       b.linearScratch},
      geometry_.gdnShape(), step.lanes, layer, geometry_.stateLayout.strides(),
      mixer.outputHeadOrder, outputPlan.input());
  linear.add(step.graph,
             {.input = b.gdnHidden, .output = b.gdnOutput, .residual = input, .scratch = b.linearScratch,
              .prepared = hidden},
             mixer.outputProjection, outputPlan);
  return b.gdnOutput;
}

metal::MetalBuffer QwenTarget::addVerifyMixer(VerifyStep &step, const QwenAttentionWeights &mixer,
                                              const ops::NormWeights &norm, metal::MetalBuffer input) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const uint32_t layer = step.attentionLayer++;
  const ops::LinearPlan inputPlan = linear.decodePlan(mixer.inputProjection, step.lanes);
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      step.graph, input, norm, b.normalized, geometry_.hiddenSize, step.rows, b.linearScratch, inputPlan.input());
  linear.add(step.graph,
             {.input = b.normalized, .output = b.fullPacked, .scratch = b.linearScratch, .prepared = normalized},
             mixer.inputProjection, inputPlan);
  ops::PagedAttention::addVerifyProjection(step.graph, b.fullPacked, mixer.queryNorm, mixer.keyNorm, b.ropeCos,
                                           b.ropeSin, b.fullQueries, b.chunkKeys[layer], b.chunkValues[layer],
                                           geometry_.attentionQueryHeads, geometry_.kvLayout, step.lanes);
  ops::PagedAttention::addVerify(step.graph, step.kvLayers[layer],
                                 {b.chunkKeys[layer], b.chunkValues[layer], b.fullQueries, b.attentionPartials,
                                  b.attentionStatistics, b.fullAttention, b.pageTables},
                                 step.chunks, step.attention);
  const ops::LinearPlan outputPlan =
      linear.decodePlan(mixer.outputProjection, step.lanes, ops::LinearEpilogue::Residual);
  const ops::PreparedInput hidden = ops::PagedAttention::addVerifyGate(
      step.graph, b.fullPacked, b.fullAttention, b.attentionHidden, geometry_.attentionQueryHeads,
      geometry_.kvLayout, step.lanes, b.linearScratch, outputPlan.input());
  linear.add(step.graph,
             {.input = b.attentionHidden, .output = b.attentionOutput, .residual = input,
              .scratch = b.linearScratch, .prepared = hidden},
             mixer.outputProjection, outputPlan);
  return b.attentionOutput;
}

void QwenTarget::addVerifyFfn(VerifyStep &step, const Qwen3_8LayerWeights &layer, metal::MetalBuffer residual,
                              metal::MetalBuffer output) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const ops::Linear &linear = operators_.linear();
  const ops::LinearPlan gateUpPlan =
      linear.decodePlan(layer.upProjection, step.lanes, ops::LinearEpilogue::GateUp, &layer.gateProjection);
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      step.graph, residual, layer.postAttentionNorm, b.normalized, geometry_.hiddenSize, step.rows,
      b.linearScratch, gateUpPlan.input());
  linear.add(step.graph,
             {.input = b.normalized, .output = b.denseIntermediate, .gateScratch = b.denseGateScratch,
              .scratch = b.linearScratch, .prepared = normalized},
             layer.upProjection, gateUpPlan, &layer.gateProjection);
  linear.add(step.graph,
             {.input = b.denseIntermediate, .output = output, .residual = residual, .scratch = b.linearScratch},
             layer.downProjection,
             linear.decodePlan(layer.downProjection, step.lanes, ops::LinearEpilogue::Residual));
}

void QwenTarget::addVerifyFfn(VerifyStep &step, const Qwen3_6MoeLayerWeights &layer, metal::MetalBuffer residual,
                              metal::MetalBuffer output) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  ops::Normalization::addRms(step.graph, residual, layer.postAttentionNorm, b.normalized, geometry_.hiddenSize,
                             step.rows);
  ops::MoE::add(step.graph, {b.normalized, residual, output, b.moe}, layer.ffn, *step.moe);
}

void QwenTarget::addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden,
                              metal::MetalBuffer finalHidden, metal::MetalBuffer logits, uint32_t lanes,
                              ops::LinearScratch scratch) const {
  const ops::Linear &linear = operators_.linear();
  const ops::LinearPlan logitsPlan = linear.decodePlan(vocabularyProjection(), lanes);
  // Qwen3.8-Flash-Next's hidden rows are its final mix, which the head reads
  // as they are.
  if (qwen4()) {
    linear.add(graph, {.input = std::move(hidden), .output = std::move(logits), .scratch = scratch},
               vocabularyProjection(), logitsPlan);
    return;
  }
  const ops::PreparedInput normalized = ops::Normalization::addRms(
      graph, std::move(hidden), weightsBase_.finalNorm, finalHidden, geometry_.hiddenSize,
      lanes * kVerifyRows, scratch, logitsPlan.input());
  linear.add(graph,
             {.input = std::move(finalHidden), .output = std::move(logits), .scratch = scratch,
              .prepared = normalized},
             vocabularyProjection(), logitsPlan);
}

void QwenTarget::addVerifyInput(metal::CommandGraph &graph,
                                metal::MetalBuffer draftInput,
                                metal::MetalBuffer proposals,
                                metal::MetalBuffer verifyInput,
                                uint32_t lanes) const {
  ops::Embedding::addVerifyInput(graph, std::move(draftInput), std::move(proposals), verifyInput,
                                 geometry_.vocabularySize, lanes);
  if (qwen4Steps_) qwen4Steps_->addVerifyInput(graph, std::move(verifyInput), lanes);
}

void QwenTarget::prepareVerifyLane(uint32_t lane, uint64_t position, const QwenTargetRequest &request) {
  if (qwen4Steps_) qwen4Steps_->prepareVerifyLane(lane, position, request);
}



void QwenTarget::addEmbedding(metal::CommandGraph &graph,
                              metal::MetalBuffer tokens,
                              metal::MetalBuffer hidden,
                              uint32_t rows) const {
  ops::Embedding::add(graph, std::move(tokens), weightsBase_.tokenEmbedding, std::move(hidden),
                      rows);
}

void QwenTarget::addStateCommit(metal::CommandGraph &graph, QwenTargetCommitBuffers buffers, uint32_t lanes) const {
  if (!lanes || lanes > ExecutionLimits::maximumBatchWidth)
    throw std::invalid_argument("invalid Qwen state commit batch");
  if (!qwen4Steps_) return;
  // Qwen3.8-Flash-Next: each lane's PLE history after its retained rows, from
  // the convolution inputs its verify wrote.
  const ops::PleShape shape = pleShape(geometry_);
  const metal::MetalBuffer convolution = qwen4Steps_->verifyScratch(lanes * kVerifyRows).pleConvolution;
  const uint64_t laneBytes = uint64_t{kVerifyRows} * shape.width() * sizeof(uint16_t);
  for (uint32_t lane = 0; lane < lanes; ++lane)
    ops::Qwen4::addPleCommit(graph, backend_.view(convolution, lane * laneBytes, laneBytes),
                             buffers.auxiliaryIn[lane], buffers.auxiliaryOut[lane],
                             backend_.view(buffers.retainedCounts, lane * sizeof(uint32_t), sizeof(uint32_t)), shape);
  // With the MTP head, each lane carries its last retained row's streams.
  if (hasMtp()) {
    const metal::MetalBuffer streams = qwen4Steps_->verifyScratch(lanes * kVerifyRows).streams;
    for (uint32_t lane = 0; lane < lanes; ++lane)
      ops::Qwen4::addMtpCarry(graph, rowsOf<float>(backend_, streams, lane * kVerifyRows, kVerifyRows,
                                                   geometry_.streamWidth()),
                              mtpCarried(buffers.auxiliaryOut[lane]),
                              backend_.view(buffers.retainedCounts, lane * sizeof(uint32_t), sizeof(uint32_t)),
                              hyperShape(geometry_), kVerifyRows);
  }
}


// Qwen3.8-Flash-Next (model/Qwen4Exp.hpp). Every block reads the mix of the
// fp32 residual streams and adds its output back to each stream; the PLE
// layer adds its gated n-gram values before its mixer's mix.

void QwenTarget::addQwen4PrefillMix(PrefillStep &step, const Qwen4HyperWeights &hyper, metal::MetalBuffer output,
                                    const metal::MetalBuffer &streams) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const Qwen4Scratch &h = step.qwen4;
  const ops::Linear &linear = operators_.linear();
  const ops::HyperShape hs = hyperShape(geometry_);
  ops::Qwen4::addNorm(step.graph, streams, hyper.norm, hyper.inject, h.normalized, h.weights, hs, step.rows);
  linear.addPrefill(step.graph, h.normalized, hyper.down, h.low, step.rows, b.linearScratch);
  ops::Qwen4::addLow(step.graph, h.low, hs, step.rows);
  linear.addPrefill(step.graph, h.low, hyper.up, h.gate, step.rows, b.linearScratch);
  ops::Qwen4::addMix(step.graph, h.normalized, h.gate, std::move(output), hs, step.rows);
}

// The streams start as the embedding's rows; every MoE adds its rows to
// zeros into the branch the combine adds to the streams.
void QwenTarget::addPrefillBegin(PrefillStep &step, const Qwen4ExpWeights &) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  step.ffn = {b.normalized, step.qwen4.zeros, step.qwen4.branch, b.moe};
  ops::Qwen4::addInit(step.graph, b.hidden[0], step.qwen4.streams, hyperShape(geometry_), step.rows);
}

void QwenTarget::addPrefillLayer(PrefillStep &step, const Qwen4ExpWeights &weights, uint32_t index) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const Qwen4Scratch &h = step.qwen4;
  const ops::HyperShape hs = hyperShape(geometry_);
  const uint32_t rows = step.rows;
  const auto project = [&](metal::MetalBuffer input, const ops::Projection &projection, metal::MetalBuffer output) {
    operators_.linear().addPrefill(step.graph, std::move(input), projection, std::move(output), rows,
                                   b.linearScratch);
  };
  const auto f32Rows = [&](const metal::MetalBuffer &buffer, uint32_t begin, uint32_t count, uint32_t width) {
    return rowsOf<float>(backend_, buffer, begin, count, width);
  };
  const auto u16Rows = [&](const metal::MetalBuffer &buffer, uint32_t begin, uint32_t count, uint32_t width) {
    return rowsOf<uint16_t>(backend_, buffer, begin, count, width);
  };
  const Qwen4ExpLayerWeights &layer = weights.layers[index];
  if (layer.ple) {
    const Qwen4PleWeights &ple = *layer.ple;
    const ops::PleShape ps = pleShape(geometry_);
    addPleGather(step.graph, h, rows * geometry_.pleHeads());
    project(h.pleRows, ple.key, h.gate);
    project(h.pleRows, ple.value, h.pleValue);
    ops::Qwen4::addPleGate(step.graph, h.gate, h.pleValue, h.streams, ple.keyNorm, ple.queryNorm,
                           ple.convolutionNorm, h.pleGated, h.pleConvolution, ps, rows);
    for (const QwenTargetPrefillSequence &sequence : step.sequences) {
      ops::Qwen4::addPleConvolution(
          step.graph, u16Rows(h.pleConvolution, sequence.rowBegin, sequence.rows, ps.width()),
          sequence.auxiliaryIn, sequence.auxiliaryOut, ple.convolution,
          f32Rows(h.pleGated, sequence.rowBegin, sequence.rows, ps.width()),
          f32Rows(h.streams, sequence.rowBegin, sequence.rows, ps.width()), ps, sequence.rows);
      // A state captured inside the command holds the history after its row.
      for (const QwenTargetStateCapture &capture : sequence.stateCaptures)
        ops::Qwen4::addPleHistory(step.graph,
                                  u16Rows(h.pleConvolution, sequence.rowBegin, capture.row, ps.width()),
                                  sequence.auxiliaryIn, capture.auxiliary, ps, capture.row);
    }
  }
  addQwen4PrefillMix(step, layer.mixerHyper, b.normalized, h.streams);
  std::visit(
      [&](const auto &mixer) {
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(mixer)>, QwenGdnWeights>) {
          project(addPrefillGdn(step, mixer), mixer.outputProjection, h.branch);
        } else {
          const PrefillQsa qsa{*layer.indexer, weights.layout.indexerTopBlocks};
          project(addPrefillAttention(step, mixer, &qsa), mixer.outputProjection, h.branch);
        }
      },
      layer.mixer);
  ops::Qwen4::addCombine(step.graph, h.streams, h.branch, h.weights, hs, rows);
  addQwen4PrefillMix(step, layer.ffnHyper, b.normalized, h.streams);
  ops::MoE::addRouting(step.graph, step.ffn, layer.ffn, *step.moe);
}

void QwenTarget::addPrefillExperts(PrefillStep &step, const Qwen4ExpWeights &weights, uint32_t index) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const Qwen4Scratch &h = step.qwen4;
  ops::MoE::addRouted(step.graph, step.ffn, weights.layers[index].ffn, *step.moe);
  ops::Qwen4::addCombine(step.graph, h.streams, h.branch, h.weights, hyperShape(geometry_), step.rows);
  if (const auto slot = geometry_.captureSlot(index))
    for (const QwenTargetPrefillSequence &sequence : step.sequences)
      for (uint32_t capture = 0; capture < sequence.captureCount; ++capture) {
        const QwenTargetPrefillCapture &c = sequence.captures[capture];
        addStreamCapture(step.graph, backend_, geometry_, *slot, h.streams, c.sourceStart, b.captured,
                         c.destinationStart, c.rows);
      }
}

// The final mix into buffers.hidden[1], which the head reads as it is; then
// the MTP head's KV at the rows.
metal::MetalBuffer QwenTarget::addPrefillEnd(PrefillStep &step, const Qwen4ExpWeights &weights) const {
  addQwen4PrefillMix(step, weights.finalHyper, step.buffers.hidden[1], step.qwen4.streams);
  if (weights.mtp) addQwen4MtpPrefill(step, weights);
  return step.buffers.hidden[1];
}

// Row r pairs the trunk's streams of the position before (row 0: the
// sequence's carried streams) with the row's token; each sequence carries
// its last row's streams, and a state captured inside the command its
// row's. The head's attention block writes the rows' KV and index keys; its
// output, which no draft of these rows reads, is left there.
void QwenTarget::addQwen4MtpPrefill(PrefillStep &step, const Qwen4ExpWeights &weights) const {
  const Qwen4MtpWeights &mtp = *weights.mtp;
  const QwenTargetPrefillBuffers &b = step.buffers;
  const Qwen4Scratch &h = step.qwen4;
  const ops::Linear &linear = operators_.linear();
  const ops::HyperShape hs = hyperShape(geometry_);
  const uint32_t rows = step.rows, hidden = geometry_.hiddenSize, width = geometry_.streamWidth();
  // Each stream's rows of the eh projection's input and output lie at the
  // scratch's row capacity apart, which holds the rows of any plan of the
  // step's rows (LinearPlan::storageRows) beside the step's own.
  const uint32_t stride = ExecutionLimits::prefillTokenBudget;
  for (const QwenTargetPrefillSequence &sequence : step.sequences) {
    ops::Qwen4::addMtpInput(step.graph, rowsOf<uint16_t>(backend_, b.hidden[0], sequence.rowBegin, sequence.rows, hidden),
                            mtpCarried(sequence.auxiliaryIn),
                            rowsOf<float>(backend_, h.streams, sequence.rowBegin, sequence.rows, width),
                            mtp.embedNorm, mtp.hiddenNorm, h.mtpInput, hs, sequence.rows, sequence.rows,
                            sequence.rowBegin, stride);
    ops::Qwen4::addMtpCarry(step.graph, rowsOf<float>(backend_, h.streams, sequence.rowBegin, sequence.rows, width),
                            mtpCarried(sequence.auxiliaryOut), {}, hs, sequence.rows);
    for (const QwenTargetStateCapture &capture : sequence.stateCaptures)
      ops::Qwen4::addMtpCarry(step.graph, rowsOf<float>(backend_, h.streams, sequence.rowBegin, capture.row, width),
                              mtpCarried(capture.auxiliary), {}, hs, capture.row);
  }
  for (uint32_t s = 0; s < hs.streams; ++s)
    linear.addPrefill(step.graph, rowsOf<uint16_t>(backend_, h.mtpInput, s * stride, stride, 2 * hidden),
                      mtp.embedHidden, rowsOf<uint16_t>(backend_, h.gate, s * stride, stride, hidden), rows,
                      b.linearScratch);
  ops::Qwen4::addMtpWiden(step.graph, h.gate, h.mtpStreams, hs, rows, stride);
  addQwen4PrefillMix(step, mtp.block.mixerHyper, b.normalized, h.mtpStreams);
  const PrefillQsa qsa{*mtp.block.indexer, weights.layout.indexerTopBlocks};
  static_cast<void>(addPrefillAttention(step, std::get<QwenAttentionWeights>(mtp.block.mixer), &qsa));
}

ops::PreparedInput QwenTarget::addQwen4VerifyMix(VerifyStep &step, const Qwen4HyperWeights &hyper,
                                                 metal::MetalBuffer streams, metal::MetalBuffer output,
                                                 const ops::Projection *next) const {
  const Qwen4Scratch &h = step.qwen4;
  const ops::LinearScratch scratch = step.buffers.linearScratch;
  const ops::Linear &linear = operators_.linear();
  const ops::HyperShape hs = hyperShape(geometry_);
  const auto layout = [&](const ops::Projection &projection) {
    return linear.decodePlan(projection, step.lanes).input();
  };
  const auto project = [&](metal::MetalBuffer input, const ops::Projection &projection, metal::MetalBuffer out,
                           ops::PreparedInput prepared) {
    addDecodeProjection(step.graph, linear, std::move(input), projection, std::move(out), step.lanes, scratch,
                        std::move(prepared));
  };
  ops::PreparedInput prepared = ops::Qwen4::addNorm(step.graph, std::move(streams), hyper.norm, hyper.inject,
                                                    h.normalized, h.weights, hs, step.rows, scratch,
                                                    layout(hyper.down));
  project(h.normalized, hyper.down, h.low, std::move(prepared));
  prepared = ops::Qwen4::addLow(step.graph, h.low, hs, step.rows, scratch, layout(hyper.up));
  project(h.low, hyper.up, h.gate, std::move(prepared));
  return ops::Qwen4::addMix(step.graph, h.normalized, h.gate, std::move(output), hs, step.rows, scratch,
                            next ? layout(*next) : ops::LinearInput::Plain);
}

void QwenTarget::addQwen4VerifyAttention(VerifyStep &step, const Qwen4ExpLayerWeights &layer,
                                         const QwenAttentionWeights &mixer, uint32_t attentionLayer,
                                         ops::PreparedInput input) const {
  constexpr uint32_t laneRows = kVerifyRows;
  constexpr uint32_t indexWidth = QWEN4_QSA_HEADS * QWEN4_QSA_DIMENSION;
  const QwenTargetVerifyBuffers &b = step.buffers;
  const Qwen4Scratch &h = step.qwen4;
  const ops::Linear &linear = operators_.linear();
  const auto project = [&](metal::MetalBuffer input, const ops::Projection &projection, metal::MetalBuffer output,
                           ops::PreparedInput prepared = {}) {
    addDecodeProjection(step.graph, linear, std::move(input), projection, std::move(output), step.lanes,
                        b.linearScratch, std::move(prepared));
  };
  const uint32_t topBlocks = qwen4()->layout.indexerTopBlocks;
  project(b.normalized, mixer.inputProjection, b.fullPacked, std::move(input));
  const Qwen4IndexerWeights &indexer = *layer.indexer;
  addIndexerProjection(step.graph, linear, b.normalized, indexer.query, h.indexerQuery, step.rows);
  addIndexerProjection(step.graph, linear, b.normalized, indexer.key, h.indexerKey, step.rows);
  // Every lane's QSA in one dispatch of each kernel.
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> firstPositions{};
  bool sparse = false;
  for (uint32_t lane = 0; lane < step.lanes; ++lane) {
    firstPositions[lane] = step.chunks[lane].committed_tokens;
    sparse = sparse || ops::Qwen4::qsaSparse(firstPositions[lane], laneRows, topBlocks);
  }
  const auto pageTables = std::span(b.pageTables).first(step.lanes);
  const ops::QsaRows qsa{laneRows, std::span(firstPositions).first(step.lanes), pageTables, topBlocks,
                         h.maskWords, h.scoreStride};
  const metal::MetalBuffer queries = rowsOf<uint16_t>(backend_, h.qsaQueries, 0, step.rows, indexWidth);
  ops::Qwen4::addQsaProject(step.graph, rowsOf<float>(backend_, h.indexerQuery, 0, step.rows, indexWidth),
                            rowsOf<float>(backend_, h.indexerKey, 0, step.rows, QWEN4_QSA_DIMENSION),
                            indexer.queryNorm, rowsOf<float>(backend_, b.ropeCos, 0, step.rows, geometry_.rotaryPairs),
                            rowsOf<float>(backend_, b.ropeSin, 0, step.rows, geometry_.rotaryPairs),
                            step.qsaIndex[attentionLayer], queries, qsa);
  ops::Qwen4::addQsaPool(step.graph,
                         {step.blocks.records, std::span(step.blocks.counts).first(step.lanes),
                          step.blocks.laneRecords, pageTables},
                         indexer.keyNorm, b.ropeInverseFrequencies, step.qsaIndex[attentionLayer]);
  // Once a lane's rows attend sparsely, every lane's attends through its
  // list: a lane's rows that keep every block select them all.
  const ops::AttentionBlocks blocks{h.qsaLists, h.qsaListCounts, sparse ? h.listStride : 0, 0};
  if (sparse) {
    ops::Qwen4::addQsaSelect(step.graph, queries, step.qsaIndex[attentionLayer], h.qsaScores, h.scoreRows,
                             rowsOf<uint32_t>(backend_, h.qsaMask, 0, step.rows, h.maskWords), qsa);
    std::array<ops::BlockListRows, ExecutionLimits::maximumBatchWidth> laneLists{};
    for (uint32_t lane = 0; lane < step.lanes; ++lane)
      laneLists[lane] = {lane * laneRows, laneRows, firstPositions[lane], lane};
    ops::PagedAttention::addBlockLists(step.graph, h.qsaMask, h.maskWords, std::span(laneLists).first(step.lanes),
                                       laneRows, blocks);
  }
  ops::PagedAttention::addVerifyProjection(step.graph, b.fullPacked, mixer.queryNorm, mixer.keyNorm, b.ropeCos,
                                           b.ropeSin, b.fullQueries, b.chunkKeys[attentionLayer],
                                           b.chunkValues[attentionLayer], geometry_.attentionQueryHeads,
                                           geometry_.kvLayout, step.lanes);
  ops::PagedAttention::addVerify(step.graph, step.kvLayers[attentionLayer],
                                 {b.chunkKeys[attentionLayer], b.chunkValues[attentionLayer], b.fullQueries,
                                  b.attentionPartials, b.attentionStatistics, b.fullAttention, b.pageTables},
                                 step.chunks, step.attention, blocks);
  const ops::PreparedInput hidden = ops::PagedAttention::addVerifyGate(
      step.graph, b.fullPacked, b.fullAttention, b.attentionHidden, geometry_.attentionQueryHeads,
      geometry_.kvLayout, step.lanes, b.linearScratch,
      linear.decodePlan(mixer.outputProjection, step.lanes).input());
  project(b.attentionHidden, mixer.outputProjection, h.branch, hidden);
}

// Qwen3.8-Flash-Next: the residual streams start as the embedding's rows,
// every block reads their mix and adds its branch back to them, and the head
// reads the final mix without a norm.
void QwenTarget::addVerifyLayers(VerifyStep &step, const Qwen4ExpWeights &weights) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const Qwen4Scratch &h = step.qwen4;
  const ops::Linear &linear = operators_.linear();
  const ops::HyperShape hs = hyperShape(geometry_);
  const uint32_t rows = step.rows;
  const auto project = [&](metal::MetalBuffer input, const ops::Projection &projection, metal::MetalBuffer output,
                           ops::PreparedInput prepared = {}) {
    addDecodeProjection(step.graph, linear, std::move(input), projection, std::move(output), step.lanes,
                        b.linearScratch, std::move(prepared));
  };
  ops::Qwen4::addInit(step.graph, b.hidden[0], h.streams, hs, rows);
  for (uint32_t index = 0; index < geometry_.layers; ++index) {
    const Qwen4ExpLayerWeights &layer = weights.layers[index];
    if (layer.ple) {
      const Qwen4PleWeights &ple = *layer.ple;
      const ops::PleShape ps = pleShape(geometry_);
      addPleGather(step.graph, h, rows * geometry_.pleHeads());
      project(h.pleRows, ple.key, h.gate);
      project(h.pleRows, ple.value, h.pleValue);
      ops::Qwen4::addPleGate(step.graph, h.gate, h.pleValue, h.streams, ple.keyNorm, ple.queryNorm,
                             ple.convolutionNorm, h.pleGated, h.pleConvolution, ps, rows);
      // Each lane's rows after its history; the commit writes the history
      // after its retained rows.
      constexpr uint32_t laneRows = kVerifyRows;
      for (uint32_t lane = 0; lane < step.lanes; ++lane)
        ops::Qwen4::addPleConvolution(
            step.graph, rowsOf<uint16_t>(backend_, h.pleConvolution, lane * laneRows, laneRows, ps.width()),
            b.auxiliary[lane], {}, ple.convolution,
            rowsOf<float>(backend_, h.pleGated, lane * laneRows, laneRows, ps.width()),
            rowsOf<float>(backend_, h.streams, lane * laneRows, laneRows, ps.width()), ps, laneRows);
    }
    std::visit(
        [&](const auto &mixer) {
          using Mixer = std::remove_cvref_t<decltype(mixer)>;
          ops::PreparedInput input =
              addQwen4VerifyMix(step, layer.mixerHyper, h.streams, b.normalized, &mixer.inputProjection);
          if constexpr (std::is_same_v<Mixer, QwenGdnWeights>) {
            const uint32_t gdnLayer = step.gdnLayer++;
            project(b.normalized, mixer.inputProjection, b.gdnPacked, std::move(input));
            const ops::PreparedInput hidden = ops::GDN::addDecode(
                step.graph,
                {b.gdnPacked, mixer.convolutionWeights, b.currentGdnStates, b.nextGdnStates, b.gdnTape,
                 std::span(b.gdnTapeLanes).first(step.lanes), mixer.decay, mixer.timeBias, mixer.mixerNorm,
                 b.gdnHidden, b.linearScratch},
                geometry_.gdnShape(), step.lanes, gdnLayer, geometry_.stateLayout.strides(),
                mixer.outputHeadOrder, linear.decodePlan(mixer.outputProjection, step.lanes).input());
            project(b.gdnHidden, mixer.outputProjection, h.branch, hidden);
          } else {
            addQwen4VerifyAttention(step, layer, mixer, step.attentionLayer++, std::move(input));
          }
        },
        layer.mixer);
    ops::Qwen4::addCombine(step.graph, h.streams, h.branch, h.weights, hs, rows);
    addQwen4VerifyMix(step, layer.ffnHyper, h.streams, b.normalized);
    ops::MoE::add(step.graph, {b.normalized, h.zeros, h.branch, b.moe}, layer.ffn, *step.moe, 0, step.liveRows);
    ops::Qwen4::addCombine(step.graph, h.streams, h.branch, h.weights, hs, rows);
    if (const auto slot = geometry_.captureSlot(index))
      addStreamCapture(step.graph, backend_, geometry_, *slot, h.streams, 0, b.capturedTargetHidden, 0, rows);
  }
  requireLayerPartition(geometry_, step.gdnLayer, step.attentionLayer);
  project(b.finalHidden, weights.logitsProjection, b.logits,
          addQwen4VerifyMix(step, weights.finalHyper, h.streams, b.finalHidden, &weights.logitsProjection));
  // The MTP head's KV at the step's rows: row r pairs the trunk's streams of
  // the position before with the row's token. The head reads them as its
  // context at the next cycle's draft steps; the rows past each lane's
  // accepted ones are overwritten before any later step reads them.
  if (weights.mtp) addQwen4MtpVerify(step, weights, b.hidden[0], h.streams, kVerifyRows, false);
}

metal::MetalBuffer QwenTarget::mtpCarried(const metal::MetalBuffer &auxiliary) const {
  return backend_.view(auxiliary, geometry_.pleHistoryBytes(), uint64_t{geometry_.streamWidth()} * sizeof(float));
}

bool QwenTarget::hasMtpDraftVocabulary() const noexcept {
  const Qwen4ExpWeights *weights = qwen4();
  return weights && weights->mtp && weights->mtp->draftHead;
}

void QwenTarget::addQwen4MtpVerify(VerifyStep &step, const Qwen4ExpWeights &weights, metal::MetalBuffer embedding,
                                   metal::MetalBuffer history, uint32_t limit, bool full, uint32_t firstRow,
                                   const Qwen4DraftHead *draftHead) const {
  constexpr uint32_t laneRows = kVerifyRows;
  const Qwen4MtpWeights &mtp = *weights.mtp;
  const QwenTargetVerifyBuffers &b = step.buffers;
  const Qwen4Scratch &h = step.qwen4;
  const ops::Linear &linear = operators_.linear();
  const ops::HyperShape hs = hyperShape(geometry_);
  const uint32_t rows = step.rows, hidden = geometry_.hiddenSize, width = geometry_.streamWidth();
  // Each stream's rows of the eh projection's input and output lie the
  // scratch's rows apart: the storage rows of the step's plans
  // (withVerifyStep), which a plan's tile may take beyond the step's rows.
  const uint32_t stride = decodeStorageLanes(step.lanes) * laneRows;
  for (uint32_t lane = 0; lane < step.lanes; ++lane)
    ops::Qwen4::addMtpInput(step.graph, rowsOf<uint16_t>(backend_, embedding, lane * laneRows, laneRows, hidden),
                            mtpCarried(b.auxiliary[lane]),
                            rowsOf<float>(backend_, history, lane * laneRows, laneRows, width), mtp.embedNorm,
                            mtp.hiddenNorm, h.mtpInput, hs, laneRows, limit, lane * laneRows, stride);
  // The eh projection of each stream's rows, then the fp32 streams.
  for (uint32_t s = 0; s < hs.streams; ++s)
    addDecodeProjection(step.graph, linear, rowsOf<uint16_t>(backend_, h.mtpInput, s * stride, stride, 2 * hidden),
                        mtp.embedHidden, rowsOf<uint16_t>(backend_, h.gate, s * stride, stride, hidden), step.lanes,
                        b.linearScratch);
  ops::Qwen4::addMtpWiden(step.graph, h.gate, h.mtpStreams, hs, rows, stride);
  const uint32_t attentionLayer = geometry_.trunkAttentionLayers();
  const QwenAttentionWeights &attention = std::get<QwenAttentionWeights>(mtp.block.mixer);
  addQwen4VerifyAttention(
      step, mtp.block, attention, attentionLayer,
      addQwen4VerifyMix(step, mtp.block.mixerHyper, h.mtpStreams, b.normalized, &attention.inputProjection));
  if (!full) return;
  ops::Qwen4::addCombine(step.graph, h.mtpStreams, h.branch, h.weights, hs, rows);
  addQwen4VerifyMix(step, mtp.block.ffnHyper, h.mtpStreams, b.normalized);
  ops::MoE::add(step.graph, {b.normalized, h.zeros, h.branch, b.moe}, mtp.block.ffn, *step.moe, firstRow);
  ops::Qwen4::addCombine(step.graph, h.mtpStreams, h.branch, h.weights, hs, rows);
  const ops::Projection &logits = draftHead ? draftHead->logits : weights.logitsProjection;
  addDecodeProjection(step.graph, linear, b.finalHidden, logits, b.logits, step.lanes, b.linearScratch,
                      addQwen4VerifyMix(step, mtp.head, h.mtpStreams, b.finalHidden, &logits));
}

void QwenTarget::addMtpDraft(metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers, const kv::PageStorage &kv,
                             std::span<const kv::ChunkedPrefillParams> chunks, uint32_t lanes,
                             QwenTargetMtpDraft draft) const {
  const Qwen4ExpWeights *weights = qwen4();
  constexpr uint32_t laneRows = kVerifyRows;
  if (!weights || !weights->mtp || !draft.steps || draft.steps > buffers.proposalsPerLane ||
      buffers.proposalsPerLane + 1 > laneRows || !buffers.candidatesPerProposal)
    throw std::invalid_argument("invalid MTP draft batch");
  // The draft vocabulary's logits are its tokens', ascending.
  const Qwen4DraftHead *draftHead = draft.draftVocabulary && weights->mtp->draftHead ? &*weights->mtp->draftHead : nullptr;
  if (draftHead && draftHead->logits.outputSize < weights->layout.mtpDraftTokens)
    throw std::logic_error("the draft head's rows are fewer than its tokens");
  const uint32_t vocabulary = draftHead ? weights->layout.mtpDraftTokens : geometry_.vocabularySize;
  // Each lane's context rows: copied in from the vocabulary projection's
  // rows first, then projected after the draft head at every step.
  bool context = false;
  if (draftHead)
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      if (!buffers.draftContextCounts[lane]) continue;
      context = true;
      const ops::QuantizedSegment &rows = buffers.draftContextHeads[lane].blocks().segments.front();
      if (rows.outputSize != kQwen4DraftContextTokens || buffers.draftContextCounts[lane] > rows.outputSize)
        throw std::invalid_argument("invalid MTP draft context rows");
      ops::Qwen4::addCopyHeadRows(graph, weights->logitsProjection.blocks().segments.front(), rows,
                                  buffers.draftContextCopies[lane], buffers.draftContextCopyCounts[lane]);
    }
  withVerifyStep(graph, buffers, kv, chunks, lanes, [&](VerifyStep &step) {
    const Qwen4Scratch &h = step.qwen4;
    const uint32_t width = geometry_.streamWidth();
    // Step j reads rows 0..j-1 (the anchor and the proposals so far, each
    // with the streams of the row before) and drafts proposal j - 1 from
    // row j - 1. The rows before it rewrite the KV and index keys the steps
    // before wrote, from the streams each of those steps kept of its row
    // (h.mtpHistory), so they need no experts; the later rows repeat row
    // j - 1, so they route to its experts.
    for (uint32_t j = 1; j <= draft.steps; ++j) {
      ops::Qwen4::addMtpTokens(graph, buffers.anchors, buffers.proposals, h.mtpTokens, lanes, laneRows,
                               buffers.proposalsPerLane, j);
      ops::Embedding::add(graph, h.mtpTokens, weightsBase_.tokenEmbedding, buffers.hidden[1], step.rows);
      addQwen4MtpVerify(step, *weights, buffers.hidden[1], h.mtpHistory, j, true, j - 1, draftHead);
      if (context) {
        // A lane's rows run as a one-lane plan, whose tile may store more
        // rows than the lane's: they are copied to the head of the branch
        // scratch, which holds the step's storage rows, and projected there.
        const uint32_t hidden = geometry_.hiddenSize;
        for (uint32_t lane = 0; lane < lanes; ++lane)
          if (buffers.draftContextCounts[lane]) {
            ops::RowCopy::add(graph, buffers.finalHidden, {lane * laneRows, hidden, 0}, h.branch, {0, hidden, 0},
                              laneRows, hidden);
            addDecodeProjection(graph, operators_.linear(), h.branch, buffers.draftContextHeads[lane],
                                backend_.view(buffers.draftContextLogits,
                                              uint64_t{lane} * buffers.draftContextLogitsSlot * sizeof(float),
                                              uint64_t{buffers.draftContextLogitsSlot} * sizeof(float)),
                                1, buffers.linearScratch);
          }
        ops::Qwen4::addMtpArgmaxContext(graph, buffers.logits, buffers.proposals, buffers.candidates,
                                        draftHead->tokens, buffers.draftContextLogits, buffers.draftContextTokens,
                                        buffers.draftContextInfo, lanes, laneRows, vocabulary,
                                        kQwen4DraftContextTokens, buffers.draftContextLogitsSlot,
                                        buffers.proposalsPerLane, buffers.candidatesPerProposal, j - 1,
                                        j == draft.steps);
      } else {
        ops::Qwen4::addMtpArgmax(graph, buffers.logits, buffers.proposals, buffers.candidates, lanes, laneRows,
                                 vocabulary, buffers.proposalsPerLane, buffers.candidatesPerProposal, j - 1,
                                 j == draft.steps, draftHead ? draftHead->tokens : metal::MetalBuffer{});
      }
      if (draft.prefetchEvent && j == draft.prefetchStep) {
        // The proposals so far, two halfwords each, as the listener reads
        // them; nothing later in the cycle writes the copy.
        ops::RowCopy::add(graph, buffers.proposals, {0, 2 * buffers.proposalsPerLane, 0}, draft.prefetchTokens,
                          {0, 2 * buffers.proposalsPerLane, 0}, lanes, 2 * j);
        graph.signal(draft.prefetchEvent, draft.prefetchValue);
      }
      if (j < draft.steps)
        for (uint32_t lane = 0; lane < lanes; ++lane)
          ops::Qwen4::addMtpCarry(graph, rowsOf<float>(backend_, h.mtpStreams, lane * laneRows, j, width),
                                  rowsOf<float>(backend_, h.mtpHistory, lane * laneRows + j - 1, 1, width), {},
                                  hyperShape(geometry_), j);
    }
  });
}

void QwenTarget::addPleGather(metal::CommandGraph &graph, const Qwen4Scratch &h, uint32_t count) const {
  const Qwen4ExpWeights *weights = qwen4();
  if (!weights) throw std::logic_error("only Qwen3.8-Flash-Next gathers PLE rows");
  if (!geometry_.pleStreamRowBytes) {
    ops::Embedding::add(graph, h.pleIndices, weights->pleTable, h.pleRows, count);
    return;
  }
  // At most `count` distinct rows, each slot below it: the gather of the
  // table's rows with the staging as a table of `count` rows, so a row's
  // values are computed exactly as from the table.
  const ops::EmbeddingWeights staged{count, geometry_.pleHeadDimension,
                                     ops::NativeRows(h.pleStaging, geometry_.pleStreamFormat)};
  ops::Embedding::add(graph, h.pleSlots, staged, h.pleRows, count);
}

} // namespace splash::model

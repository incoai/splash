#pragma once

// Shared Qwen prefill/decode scratch layouts and allocation owners.

#include "model/DFlashDraft.hpp"
#include "model/ModelFactory.hpp"
#include "model/QwenState.hpp"
#include "model/QwenTarget.hpp"
#include "model/WorkFormulas.hpp"

#include "Checked.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Sampling.h"
#include "ops/ExecutionPlans.hpp"
#include "ops/LaneRows.hpp"
#include "ops/PagedKv.hpp"

#include <algorithm>
#include <cmath>
#include <array>
#include <cstring>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace splash::model {

inline constexpr uint32_t kLaneCount = ExecutionLimits::maximumBatchWidth;
// The most rows of a decode lane, its verify block's, and the most packed
// rows of a decode step, kDecodeRows a lane: what the decode arena holds.
inline constexpr uint32_t kDecodeRows = ExecutionLimits::targetVerifyRows;
inline constexpr uint32_t kDecodeStepRows = kLaneCount * kDecodeRows;
// The rows of a decode lane's draft block (its query rows), the first of a
// verify block's positions.
inline constexpr uint32_t kDraftRows = ExecutionLimits::draftQueryRows;
// Per-lane sampling uniforms handed to the sampler each cycle, in the
// layout of metal/abi/Sampling.h.
inline constexpr uint32_t kSamplingUniformCount = SPLASH_SAMPLING_UNIFORMS;
inline constexpr uint32_t kDraftProposalTokens = ExecutionLimits::draftProposalTokens;
inline constexpr uint32_t kPrefillRows = ExecutionLimits::prefillTokenBudget;
// A prefill group (QwenTarget::addPrefill): a command of up to this many
// chunks of kPrefillRows rows, run layer by layer, so that a target whose
// experts stream reads each layer's once for all of them, and one whose K/V
// spills reads each attention layer's spilled history once
// (model::SpilledKvReads). A group holds the tensors its chunks carry from
// layer to layer once per chunk (prefillGroupTensor), and requests that
// arrive while it runs wait for the whole command: four chunks read a layer
// once per 8192 rows and keep both bounded.
inline constexpr uint32_t kPrefillGroupChunks = 4;
inline constexpr uint32_t kTileRows = kv::kPageTokens;
inline constexpr uint32_t kRaggedAttentionRows =
    kPrefillRows + kLaneCount * (kTileRows - 1);
inline constexpr uint32_t kDraftCacheStride = ExecutionLimits::draftContextTokens;
inline constexpr uint32_t kMaximumPageTableEntries =
    (kv::kMaximumPhysicalTokens + kv::kPageTokens - 1) / kv::kPageTokens;

struct RuntimeGeometry final {
  QwenTargetGeometry target;
  DFlashDraftLayout draft;
  // The attention of a target whose K/V spills (ModelCapabilities::sparseKv),
  // whose runtime reads it through the hot block cache and the history
  // stream (RuntimeContext); none for K/V that stays in memory.
  std::optional<SparseKvShape> spilledKv;
  // The chunks of a prefill command: a group's for a target whose experts
  // stream or whose K/V spills, otherwise one.
  uint32_t prefillChunks = 1;
  // Whether the target's experts stream from the SSD.
  bool streamedExperts = false;
  // The step scratch the target keeps itself beside the arenas.
  QwenTargetScratchBytes targetScratch{};

  // The geometry of `model` as it loaded, or with `parts` streaming
  // whatever streams of it as it loaded (plannedRuntimeMemory). Only a
  // block-sparse target with a spillable layout (kv::Layout::spillable)
  // reads spilled K/V.
  [[nodiscard]] static RuntimeGeometry from(const LoadedModel &model,
                                            kv::Format kvFormat,
                                            ops::GdnStateFormat stateFormat) {
    return from(model, kvFormat, stateFormat, StreamedParts::loaded(model));
  }
  [[nodiscard]] static RuntimeGeometry from(const LoadedModel &model, kv::Format kvFormat,
                                            ops::GdnStateFormat stateFormat, StreamedParts parts) {
    const DFlashDraftLayout &draft = model.draft.layout;
    return from(model.descriptor, draft,
                std::visit([&](const auto &weights) { return qwenTargetGeometry(weights, draft.targetLayerIds()); },
                           model.target),
                kvFormat, stateFormat, parts);
  }
  // The geometry of a model of `descriptor` whose draft is of `draft` and
  // whose target's weights give `target` (qwenTargetGeometry, as they loaded
  // or as their files plan them, plannedTargetGeometry), with `parts`
  // streaming.
  [[nodiscard]] static RuntimeGeometry from(const ModelDescriptor &descriptor, const DFlashDraftLayout &draft,
                                            QwenTargetGeometry target, kv::Format kvFormat,
                                            ops::GdnStateFormat stateFormat, StreamedParts parts) {
    RuntimeGeometry result;
    result.draft = draft;
    result.target = std::move(target);
    result.target.kvLayout = descriptor.targetKvLayout;
    result.target.kvLayout.format = kvFormat;
    result.target.stateLayout = descriptor.stateLayout.target;
    result.target.stateLayout.format = stateFormat;
    result.target.ropeScaling = descriptor.ropeScaling;
    if (parts.kv) result.spilledKv = descriptor.capabilities.sparseKv;
    result.streamedExperts = parts.experts;
    if (parts.experts || parts.kv) result.prefillChunks = kPrefillGroupChunks;
    if (!result.target.valid() || !result.draft.stateLayout().valid() ||
        (parts.kv && !result.target.kvLayout.spillable()) ||
        result.target.contextTokens() != descriptor.capabilities.maximumContextTokens ||
        result.target.hiddenSize != result.draft.hiddenSize ||
        result.target.vocabularySize != result.draft.vocabularySize ||
        result.target.capturedHiddenSize() != result.draft.targetHiddenSize()) {
      throw std::invalid_argument("invalid model runtime geometry");
    }
    // Sized from the geometry, once it is valid.
    result.targetScratch = std::visit(
        [&](const auto &layout) {
          return qwenTargetScratchBytes(layout, result.target, result.prefillChunks, parts.tableRowBytes);
        },
        descriptor.target);
    return result;
  }

  [[nodiscard]] uint32_t draftRotaryPairs() const noexcept {
    return draft.attentionHeadDimension / 2;
  }
  [[nodiscard]] uint32_t maskWords() const noexcept {
    return maskWordsPerToken(target.vocabularySize);
  }
};

// What one prefill command of a runtime of `geometry` may hold: its chunks'
// rows; passes over the target's layers, at most one a layer, or with the
// Neural Engine's FFN split one, as the split takes a chunk's layers whole
// (QwenTarget::requirePrefill), and at the rate of its larger commands from
// the least rows of a chunk the split takes (`aneFfnRows`,
// ops::AneFfn::minimumRows; 0 without the split); one lane whose rows run
// past a chunk, as a prefill group passes each lane's state between its
// parts through its one scratch cell (PrefillArena::groupCell); the history
// of one lane streamed (KvHistoryStream) where K/V spills, none where it
// stays in memory; whether its experts stream; and in one attention layer
// the pairs of a full chunk at the end of the native window.
[[nodiscard]] inline CommandLimits commandLimits(const RuntimeGeometry &geometry, uint32_t aneFfnRows) noexcept {
  return {.commandRows = geometry.prefillChunks * kPrefillRows,
          .layers = geometry.target.layers,
          .maximumPasses = aneFfnRows ? 1u : geometry.target.layers,
          .minimumUnitRows = aneFfnRows ? aneFfnRows : 1u,
          .lanesPastChunk = 1,
          .indexPageLanes = geometry.spilledKv ? 1u : 0u,
          .streamsWeights = geometry.streamedExperts,
          .attentionPairs = denseAttentionPairs(geometry.target.nativeContextTokens - kPrefillRows, kPrefillRows)};
}

// The attention layers of `target` before its layer `layer`.
[[nodiscard]] inline uint32_t attentionLayersBefore(const QwenTargetGeometry &target, uint32_t layer) noexcept {
  uint32_t attention = 0;
  for (uint32_t before = 0; before < layer; ++before)
    attention += target.isFullAttentionLayer(before);
  return attention;
}

// How far the rows of attention layer `attention` of `target` reach back: a
// sliding layer's window, the row's own key included (the window's other
// keys are kv::Layout::windowTokens); a block-sparse target's indexer
// selection; else every key.
[[nodiscard]] inline AttentionReach attentionReach(const QwenTargetGeometry &target, uint32_t attention) noexcept {
  if (const uint32_t window = target.attentionVariant(attention).windowTokens)
    return {.windowTokens = window + 1};
  if (target.kvLayout.reads == kv::Reads::SelectedBlocks)
    return {.denseHistoryTokens = target.indexerTokens(),
            .selectedTokens = target.indexerTokens(),
            .blockTokens = target.indexerBlockTokens};
  return {};
}

// What a pass from layer `first` with `passesLeft` passes left (passLayers)
// of a prefill command of `lanes` computes on `target` (Model::prefillWork):
// each attention layer's pairs over the lanes' rows at its reach, the most of
// them over every attention layer whatever the pass (a pass does not split a
// layer), and their sum over the pass's layers. What the pass reads is the
// runtime's to add.
[[nodiscard]] inline CommandWork prefillCommandWork(const QwenTargetGeometry &target,
                                                    std::span<const PrefillLane> lanes, uint32_t first,
                                                    uint32_t passesLeft) noexcept {
  const LayerRange run = passLayers(first, passesLeft, target.layers);
  CommandWork work;
  for (uint32_t layer = 0, attention = 0; layer < target.layers; ++layer) {
    if (!target.isFullAttentionLayer(layer))
      continue;
    const AttentionReach reach = attentionReach(target, attention++);
    uint64_t pairs = 0;
    for (const PrefillLane &lane : lanes)
      pairs += attentionPairs(reach, lane.position, lane.rows);
    work.layerAttentionPairs = std::max(work.layerAttentionPairs, pairs);
    if (layer >= run.first && layer < run.end)
      work.attentionPairs += pairs;
  }
  return work;
}

template <class T> constexpr uint64_t bytesFor(uint64_t elements) noexcept {
  return elements * sizeof(T);
}

// The tensors of a target step (QwenTargetPrefillBuffers,
// QwenTargetVerifyBuffers) over the rows of its command's sequences, sized by
// their rows: a prompt chunk's, or a decode step's lanes' (TargetArena).
enum class TargetTensor : uint32_t {
  Hidden0,
  Hidden1,
  Normalized,
  GdnPacked,
  GdnQueries,
  GdnKeys,
  GdnValues,
  GdnDecay,
  GdnBeta,
  Recurrent,
  GdnHidden,
  GdnOutput,
  Intermediate,
  FullPacked,
  FullQueries,
  FullAttention,
  AttentionPartials,
  AttentionStatistics,
  AttentionHidden,
  AttentionOutput,
  RopeCos,
  RopeSin,
  ChunkKeys,
  ChunkValues,
  // One tensor per ops::kMoeScratchFields entry, in its order (moeScratchTensor).
  MoeScratch,
  MoeScratchLast = MoeScratch + ops::kMoeScratchFields.size() - 1,
  Count,
};

constexpr uint32_t targetTensorCount = static_cast<uint32_t>(TargetTensor::Count);

// The arena tensor of MoE scratch field `field` (ops::kMoeScratchFields).
template <class Tensor>
constexpr Tensor moeScratchTensor(size_t field) noexcept {
  return static_cast<Tensor>(static_cast<uint32_t>(Tensor::MoeScratch) + field);
}

// The commands a TargetArena serves: up to `rows` rows a chunk, of `chunks`
// chunks (a prefill group's), whose attention's workspace is `attention` and
// whose MoE blocks' `moe` (the operators' bounds of the plans its commands
// run), each chunk of at most kLaneCount sequences.
struct TargetRows final {
  uint32_t rows = 0;
  uint32_t chunks = 1;
  ops::AttentionWorkspace attention{};
  ops::MoeWorkspace moe{};
};

// Whether a target tensor's rows outlive a layer of a prefill group, which
// then holds every chunk's: the rows the layers carry (the hidden rows and
// RoPE), those of the FFN its routed experts read after the layer's load
// (input, the mixer's output they add to, the MoE's routing and its copy of
// the slot table). The chunks share the rest of the MoE scratch, which each
// chunk's expert passes use from its gather through its combine, one chunk's
// after another's (QwenTarget::addPrefillLayers).
[[nodiscard]] constexpr bool targetGroupTensor(TargetTensor tensor) noexcept {
  switch (tensor) {
  case TargetTensor::Hidden0:
  case TargetTensor::Hidden1:
  case TargetTensor::RopeCos:
  case TargetTensor::RopeSin:
  case TargetTensor::Normalized:
  case TargetTensor::GdnOutput:
  case TargetTensor::AttentionOutput:
    return true;
  default:
    break;
  }
  for (const auto routing : {&ops::MoeScratch::selectedExperts, &ops::MoeScratch::routingWeights,
                             &ops::MoeScratch::tileDescriptors, &ops::MoeScratch::tileCount,
                             &ops::MoeScratch::groupedRoutes, &ops::MoeScratch::routeRows,
                             &ops::MoeScratch::expertSlots})
    for (size_t field = 0; field < ops::kMoeScratchFields.size(); ++field)
      if (ops::kMoeScratchFields[field].buffer == routing && tensor == moeScratchTensor<TargetTensor>(field))
        return true;
  return false;
}

// The bytes of each target tensor of commands of `shape`, a group tensor's
// holding every chunk's rows, each chunk's aligned.
[[nodiscard]] std::array<uint64_t, targetTensorCount> targetTensorBytes(const QwenTargetGeometry &target,
                                                                       const TargetRows &shape);

// The target step tensors of a command (TargetRows), views of an arena's
// allocation from `offset` on: a prompt's (the prefill arena's) or a decode
// step's (the decode arena's, which runs between a prompt's passes). Row
// tensors hold their rows in order, a decode step's lanes' packed as its
// layout places them (DecodeArena::rows), and a group tensor chunk c's rows
// from c times a chunk's bytes on.
class TargetArena final {
public:
  [[nodiscard]] static uint64_t bytes(const QwenTargetGeometry &target, const TargetRows &shape) {
    uint64_t total = 0;
    for (const uint64_t value : targetTensorBytes(target, shape))
      total = checkedAdd(total, alignUp(value), "target arena");
    return total;
  }

  TargetArena(metal::MetalBackend &backend, const metal::MetalBuffer &base, uint64_t offset,
              const QwenTargetGeometry &target, const TargetRows &shape)
      : shape_(shape) {
    const auto sizes = targetTensorBytes(target, shape);
    uint64_t cursor = offset;
    for (uint32_t index = 0; index < sizes.size(); ++index) {
      if (sizes[index]) {
        tensors_[index] = backend.view(base, cursor, sizes[index]);
        if (shape.chunks > 1 && targetGroupTensor(static_cast<TargetTensor>(index))) {
          const uint64_t chunkBytes = sizes[index] / shape.chunks;
          for (uint32_t chunk = 0; chunk < shape.chunks; ++chunk)
            chunkViews_[index].push_back(backend.view(base, cursor + chunk * chunkBytes, chunkBytes));
        }
      }
      cursor += alignUp(sizes[index]);
    }
  }

  // Chunk `chunk`'s view of a group's tensor (targetGroupTensor), or the
  // tensor of an arena of one chunk.
  [[nodiscard]] metal::MetalBuffer get(TargetTensor tensor, uint32_t chunk) const {
    const uint32_t index = static_cast<uint32_t>(tensor);
    if (chunk >= shape_.chunks) throw std::out_of_range("target arena chunk");
    return chunkViews_[index].empty() ? tensors_[index] : chunkViews_[index][chunk];
  }
  // A tensor no chunk holds its own of: a group's is bound by its chunk.
  [[nodiscard]] metal::MetalBuffer get(TargetTensor tensor) const {
    const uint32_t index = static_cast<uint32_t>(tensor);
    if (!chunkViews_[index].empty()) throw std::logic_error("a prefill group's tensor is bound by its chunk");
    return tensors_[index];
  }
  [[nodiscard]] ops::MoeScratch moeScratch(uint32_t chunk) const {
    ops::MoeScratch scratch;
    for (size_t field = 0; field < ops::kMoeScratchFields.size(); ++field)
      scratch.*ops::kMoeScratchFields[field].buffer = get(moeScratchTensor<TargetTensor>(field), chunk);
    return scratch;
  }
  [[nodiscard]] const TargetRows &shape() const noexcept { return shape_; }

private:
  TargetRows shape_;
  std::array<metal::MetalBuffer, targetTensorCount> tensors_{};
  std::array<std::vector<metal::MetalBuffer>, targetTensorCount> chunkViews_{};
};

enum class PrefillTensor : uint32_t {
  InputTokens,
  Captured,
  GateIntermediate,
  TargetPositions,
  DraftPositions,
  TargetInverseFrequencies,
  DraftInverseFrequencies,
  ContextProjected,
  ContextHidden,
  ContextKv,
  DraftRopeCos,
  DraftRopeSin,
  LinearPartials,
  LinearCounters,
  // The int8 prefill tile's codes and scales (ops::LinearScratch::input, sums).
  LinearTable,
  LinearSums,
  // The rotated input of a rotated projection (ops::LinearScratch::rotated).
  LinearRotated,
  // A prefill group's scratch GDN cell, which carries the state from a chunk
  // to the next when the lane's cells cannot (Runtime's prefill groups).
  GroupCell,
  // Each sequence's rows of the chunk (SplashLaneRows), which the host
  // writes and the attention layers read (ops::AttentionRows).
  LaneLayout,
  Count,
};

constexpr uint32_t prefillTensorCount =
    static_cast<uint32_t>(PrefillTensor::Count);

// Whether a prefill tensor's rows outlive a layer of a prefill group, which
// then holds every chunk's: the rows the layers carry (tokens and positions),
// and those the draft's context reads once every layer ran (the captured
// target rows, their draft positions and RoPE). The target's step tensors
// are its TargetArena's (targetGroupTensor); a tensor a family does not use
// holds no rows; a target's own scratch is its own (QwenTarget).
[[nodiscard]] constexpr bool prefillGroupTensor(PrefillTensor tensor) noexcept {
  switch (tensor) {
  case PrefillTensor::InputTokens:
  case PrefillTensor::LaneLayout:
  case PrefillTensor::TargetPositions:
  case PrefillTensor::Captured:
  case PrefillTensor::DraftPositions:
  case PrefillTensor::DraftRopeCos:
  case PrefillTensor::DraftRopeSin:
    return true;
  default:
    return false;
  }
}

// The target step tensors' shape of a runtime's prompt commands: its
// prefill group's chunks of kPrefillRows rows.
[[nodiscard]] TargetRows prefillTargetRows(const RuntimeGeometry &geometry, const ops::ExecutionPlans &operators);
// Sizes depend on the geometry and the device's operator plans: the
// prefill-only tensors', beside the target step tensors (prefillTargetRows).
[[nodiscard]] std::array<uint64_t, prefillTensorCount>
prefillTensorBytes(const RuntimeGeometry &geometry,
                   const ops::ExecutionPlans &operators);
[[nodiscard]] uint64_t prefillArenaBytes(const RuntimeGeometry &geometry,
                                         const ops::ExecutionPlans &operators);
// The arena's and the target's own prefill scratch (qwenTargetScratchBytes).
[[nodiscard]] uint64_t plannedPrefillBytes(const RuntimeGeometry &geometry,
                                           const ops::ExecutionPlans &operators);

// The target step tensors (target(), prefillTargetRows) and the prompt's
// own. A group's tensors (prefillGroupTensor) hold chunk c's rows from c
// times a chunk's (aligned) bytes on, which get(tensor, c) views.
class PrefillArena final {
public:
  PrefillArena(metal::MetalBackend &backend, const RuntimeGeometry &geometry,
                const ops::ExecutionPlans &operators)
      : bytes_(prefillArenaBytes(geometry, operators)), chunks_(geometry.prefillChunks),
        base_(backend.allocateBuffer(bytes_, metal::BufferStorage::Shared, "qwen-shared-prefill")),
        target_(backend, base_, 0, geometry.target, prefillTargetRows(geometry, operators)) {
    const auto sizes = prefillTensorBytes(geometry, operators);
    uint64_t cursor = TargetArena::bytes(geometry.target, target_.shape());
    for (uint32_t index = 0; index < sizes.size(); ++index) {
      if (sizes[index]) {
        tensors_[index] = backend.view(base_, cursor, sizes[index]);
        if (chunks_ > 1 && prefillGroupTensor(static_cast<PrefillTensor>(index))) {
          const uint64_t chunkBytes = sizes[index] / chunks_;
          for (uint32_t chunk = 0; chunk < chunks_; ++chunk)
            chunkViews_[index].push_back(backend.view(base_, cursor + chunk * chunkBytes, chunkBytes));
        }
      }
      cursor += alignUp(sizes[index]);
    }
    if (chunks_ > 1)
      groupCell_ = gdnCellBuffers(backend, get(PrefillTensor::GroupCell), geometry.target.stateLayout);
    if (cursor != bytes_)
      throw std::logic_error("prefill arena mismatch");
    auto *target = static_cast<float *>(
        get(PrefillTensor::TargetInverseFrequencies).contents());
    auto *draft = static_cast<float *>(
        get(PrefillTensor::DraftInverseFrequencies).contents());
    if (!target || !draft)
      throw std::logic_error("RoPE frequencies are not CPU-visible");
    rotaryInverseFrequencies(geometry.target.rotaryTheta, geometry.target.rotaryPairs,
                             geometry.target.ropeScaling,
                             {target, geometry.target.rotaryPairs});
    for (uint32_t dim = 0; dim < geometry.draftRotaryPairs(); ++dim) {
      draft[dim] = std::pow(geometry.draft.rotaryTheta,
                            -static_cast<float>(dim) / geometry.draftRotaryPairs());
    }
    // Split projections return their counters to zero; they start there.
    if (const metal::MetalBuffer counters = get(PrefillTensor::LinearCounters))
      std::memset(counters.contents(), 0, counters.sizeBytes());
  }

  // Chunk `chunk`'s view of a group's tensor (prefillGroupTensor), or the
  // tensor of an arena of one chunk.
  [[nodiscard]] metal::MetalBuffer get(PrefillTensor tensor, uint32_t chunk) const {
    const uint32_t index = static_cast<uint32_t>(tensor);
    if (chunk >= chunks_) throw std::out_of_range("prefill arena chunk");
    return chunkViews_[index].empty() ? tensors_[index] : chunkViews_[index][chunk];
  }
  // A tensor no chunk holds its own of: a group's is bound by its chunk.
  [[nodiscard]] metal::MetalBuffer get(PrefillTensor tensor) const {
    const uint32_t index = static_cast<uint32_t>(tensor);
    if (!chunkViews_[index].empty()) throw std::logic_error("a prefill group's tensor is bound by its chunk");
    return tensors_[index];
  }
  [[nodiscard]] uint32_t chunks() const noexcept { return chunks_; }
  // A group's scratch cell (PrefillTensor::GroupCell), whose views are empty
  // when cells have no bytes; null without groups.
  [[nodiscard]] const GdnParityBuffers *groupCell() const noexcept {
    return groupCell_ ? &*groupCell_ : nullptr;
  }
  [[nodiscard]] const TargetArena &target() const noexcept { return target_; }
  [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }

private:
  uint64_t bytes_ = 0;
  uint32_t chunks_ = 1;
  metal::MetalBuffer base_;
  TargetArena target_;
  std::array<metal::MetalBuffer, prefillTensorCount> tensors_{};
  std::array<std::vector<metal::MetalBuffer>, prefillTensorCount> chunkViews_{};
  std::optional<GdnParityBuffers> groupCell_;
};

// A decode step's own tensors, beside the target step tensors of its lanes'
// rows (DecodeArena::target): the step's rows' (decodeRowTensor), and a
// lane's of each of the others.
enum class DecodeTensor : uint32_t {
  InputTokens,
  DraftPositions,
  ContextProjected,
  ContextHidden,
  ContextKv,
  CapturedTargetHidden,
  DraftQueryKeys,
  DraftQueryValues,
  DraftRopeCos,
  DraftRopeSin,
  FinalHidden,
  Logits,
  ArgmaxValues,
  ArgmaxIndices,
  TargetPartialMasses,
  TargetVocabularyRows,
  TargetVocabularyRanges,
  TargetVocabularyArrivals,
  SamplingUniforms,
  ConstraintMasks,
  OutputTokens,
  RetainedCount,
  // Each lane's accepted path, the mask of its retained rows of its verify
  // block (metal/abi/VerifyTree.h), which acceptance writes (metal/abi/
  // Sampling.h AcceptBatchParams) and the K/V path commit reads.
  RetainedRows,
  AcceptedCount,
  VerifyRows,
  // Each lane's rows of a step (SplashLaneRows), which the step's selector
  // writes (ops::DraftSelector::addVerifyRows), or a first token's selection.
  LaneLayout,
  DraftInputTokens,
  DraftHidden0,
  DraftHidden1,
  DraftNormalized,
  DraftDynamic,
  DraftConvolved,
  DraftProposalQkv,
  DraftAttention,
  DraftProjected,
  DraftResidual,
  DraftIntermediate,
  DraftFinalHidden,
  SelectorHidden,
  Candidates,
  Unary,
  TopPartialIds,
  TopPartialValues,
  // A step's rows' places in their lanes' trees, a row of
  // metal/abi/VerifyTree.h per row: chains (ops::chainRow) from the arena's
  // start, then as each step's selector lays them out
  // (ops::DraftSelector::addVerifyRows), with the row of its lane's tree
  // each verify row is.
  TreeRows,
  VerifySources,
  // Each lane's draft tree as the selector grew it, best first
  // (ops::DraftSelectorBuffers): its rows' tokens, tree rows and scores (each
  // row's draft probability and estimate), the children each row drew and
  // its next position's probabilities, and its size.
  DraftTreeTokens,
  DraftTrees,
  DraftTreeScores,
  DraftTreeDraws,
  DraftTreeNext,
  DraftTreeSizes,
  // Each lane's host chain of its tree: its tokens and their estimates
  // (DraftTreeParams in metal/abi/Sampling.h), which the host writes.
  ChainTokens,
  ChainRates,
  PageTable,
  // Indexed by state lane, like PageTable, for a layout with a block index:
  // the entries of the pages' index (kv::PageStorage::writeIndexEntries),
  // which a block-sparse attention selects through (ops::BlockIndex).
  IndexTable,
  // For a target whose K/V spills: where each entry of a verify's block
  // lists reads its block (ops::AttentionBlocks::locations), a lane's list
  // in its row of the batch.
  Locations,
  // Indexed by state lane, like PageTable: a penalized request's penalty
  // words (ops::Sampling::rebuildPenaltyWords).
  PenaltyState,
  // Indexed by state lane, like PageTable: the lane's two decode tape slots
  // (ops::GdnTapeLane), one per state parity (gdnTapeSlot).
  GdnTape,
  Count,
};

constexpr uint32_t decodeTensorCount =
    static_cast<uint32_t>(DecodeTensor::Count);

// Whether a decode tensor holds a row of each of a step's packed rows, which
// lie in lane order, each lane's from its first row (DecodeTensor::
// LaneLayout): the target's inputs and outputs, its selection's workspace
// (indexed by logits row, ops::Sampling::workspace) and the draft's context
// projections. Each takes kDecodeRows rows a lane, so a step of `lanes`
// lanes finds all of its rows in their share. The others are a lane's own,
// the draft's block of kDraftRows among them, indexed by batch lane, or by
// state lane where they say so.
[[nodiscard]] constexpr bool decodeRowTensor(DecodeTensor tensor) noexcept {
  switch (tensor) {
  case DecodeTensor::InputTokens:
  case DecodeTensor::ContextProjected:
  case DecodeTensor::ContextHidden:
  case DecodeTensor::ContextKv:
  case DecodeTensor::CapturedTargetHidden:
  case DecodeTensor::FinalHidden:
  case DecodeTensor::Logits:
  case DecodeTensor::ArgmaxValues:
  case DecodeTensor::ArgmaxIndices:
  case DecodeTensor::TargetPartialMasses:
  case DecodeTensor::TargetVocabularyRows:
  case DecodeTensor::TargetVocabularyRanges:
  case DecodeTensor::TargetVocabularyArrivals:
  case DecodeTensor::OutputTokens:
  case DecodeTensor::TreeRows:
    return true;
  default:
    return false;
  }
}

// The target step tensors' shape of a runtime's decode steps: every lane's
// rows, one chunk.
[[nodiscard]] TargetRows decodeTargetRows(const RuntimeGeometry &geometry, const ops::ExecutionPlans &operators);

// One slot of a state lane's decode tape (DecodeTensor::GdnTape): one step of
// every GDN layer (ops::GdnTapeLane). A lane holds two, one per state parity.
[[nodiscard]] uint64_t gdnTapeSlotBytes(const RuntimeGeometry &geometry) noexcept;

[[nodiscard]] std::array<uint64_t, decodeTensorCount>
decodeTensorBytes(const RuntimeGeometry &geometry,
                  const ops::ExecutionPlans &operators);
// A decode tensor is one allocation of every lane's. A row tensor's B1-B4
// prefixes hold 16/32/48/64 rows, each the most rows of a step of that many
// lanes. Lanes are never separated by arena-alignment holes; only whole
// tensor boundaries are aligned. The target step tensors (decodeTargetRows)
// follow them.
[[nodiscard]] uint64_t decodeArenaBaseBytes(const RuntimeGeometry &geometry,
                                            const ops::ExecutionPlans &operators);
// The arena's base and its gate and linear scratch buffers, each in whole
// host pages.
[[nodiscard]] uint64_t decodeArenaBytes(const RuntimeGeometry &geometry,
                                        const ops::ExecutionPlans &operators);
// The arena's and the target's own decode scratch (qwenTargetScratchBytes).
[[nodiscard]] uint64_t plannedDecodeBytes(const RuntimeGeometry &geometry,
                                          const ops::ExecutionPlans &operators);

class DecodeArena final {
public:
  DecodeArena(metal::MetalBackend &backend, RuntimeGeometry geometry,
               const ops::ExecutionPlans &operators)
      : backend_(backend), geometry_(std::move(geometry)),
        base_(backend_.allocateBuffer(decodeArenaBaseBytes(geometry_, operators), metal::BufferStorage::Shared,
                                      "qwen-shared-decode")),
        target_(backend_, base_, laneTensorBytes(geometry_, operators), geometry_.target,
                decodeTargetRows(geometry_, operators)) {
    auto sizes = decodeTensorBytes(geometry_, operators);
    uint64_t cursor = 0;
    for (uint32_t tensor = 0; tensor < sizes.size(); ++tensor) {
      uint64_t stride = sizes[tensor];
      offsets_[tensor] = cursor;
      sizes_[tensor] = sizes[tensor];
      if (sizes[tensor]) {
        for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
          tensors_[lane][tensor] = backend_.view(
              base_, cursor + uint64_t{lane} * stride, sizes[tensor]);
        }
      }
      cursor +=
          alignUp(checkedMultiply(stride, kLaneCount, "decode tensor"));
    }
    if (cursor + TargetArena::bytes(geometry_.target, target_.shape()) != decodeArenaBaseBytes(geometry_, operators))
      throw std::logic_error("decode arena mismatch");
    // Sampled rows' draws return their arrival counts to zero; they start
    // there.
    const metal::MetalBuffer arrivals =
        batchSlice(DecodeTensor::TargetVocabularyArrivals, kLaneCount);
    std::memset(arrivals.contents(), 0, arrivals.sizeBytes());
    // Every block verifies a chain until its selector writes a tree.
    auto *tree = static_cast<ops::TreeRow *>(batchSlice(DecodeTensor::TreeRows, kLaneCount).contents());
    for (uint32_t row = 0; row < kDecodeStepRows; ++row)
      tree[row] = ops::chainRow(row % kDecodeRows);

    // Each scratch buffer takes whole host pages (decodeArenaBytes).
    const auto allocate = [&](uint64_t bytes, metal::BufferStorage storage, const char *label) {
      return bytes ? backend_.allocateBuffer(alignUp(bytes), storage, label) : metal::MetalBuffer{};
    };
    gateScratch_ = allocate(gateScratchBytes(geometry_, operators), metal::BufferStorage::Private,
                            "qwen-gate-scratch");
    // Each field exists only when some plan uses it (split-only plans have
    // partials and counters but no activation table).
    const auto linearSize = linearScratchSize(geometry_, operators);
    linearScratch_.input = allocate(linearSize.input, metal::BufferStorage::Private, "linear-input");
    linearScratch_.sums = allocate(linearSize.sums, metal::BufferStorage::Private, "linear-sums");
    linearScratch_.partials =
        allocate(linearSize.partials, metal::BufferStorage::Private, "linear-partials");
    linearScratch_.counters =
        allocate(linearSize.counters, metal::BufferStorage::Shared, "linear-counters");
    linearScratch_.rotated = allocate(linearSize.rotated, metal::BufferStorage::Private, "linear-rotated");
    if (linearSize.counters)
      std::memset(linearScratch_.counters.contents(), 0, linearSize.counters);
    bytes_ = decodeArenaBytes(geometry_, operators);
  }

  [[nodiscard]] metal::MetalBuffer get(uint32_t lane, DecodeTensor tensor) const {
    if (lane >= kLaneCount)
      throw std::out_of_range("invalid decode lane");
    return tensors_[lane][static_cast<uint32_t>(tensor)];
  }

  [[nodiscard]] metal::MetalBuffer batchSlice(DecodeTensor tensor, uint32_t lanes) const {
    if (!lanes || lanes > kLaneCount)
      throw std::out_of_range("invalid decode batch width");
    const uint32_t index = static_cast<uint32_t>(tensor);
    // Model-specific scratch is represented by an empty buffer.  The
    // selected target graph consumes either dense-FFN or MoE tensors, never
    // both, so no dummy allocation is needed for the inactive operator.
    if (!sizes_[index])
      return {};
    return backend_.view(base_, offsets_[index],
                         uint64_t{lanes} * sizes_[index]);
  }

  // The target step tensors of the lanes' rows.
  [[nodiscard]] const TargetArena &target() const noexcept { return target_; }
  // Rows `rows` of a step's packed rows of a row tensor, the step's own
  // (decodeRowTensor) or its target's: a lane's, as the step's layout places
  // them, or the step's, from row 0; none of a tensor the target does not
  // use.
  [[nodiscard]] metal::MetalBuffer rows(DecodeTensor tensor, ops::LaneRows rows) const {
    if (!decodeRowTensor(tensor))
      throw std::invalid_argument("a lane's own decode tensor holds no step rows");
    const uint32_t index = static_cast<uint32_t>(tensor);
    return stepRows(base_, offsets_[index], sizes_[index] * kLaneCount, rows);
  }
  [[nodiscard]] metal::MetalBuffer rows(TargetTensor tensor, ops::LaneRows rows) const {
    const metal::MetalBuffer whole = target_.get(tensor);
    return stepRows(whole, 0, whole ? whole.sizeBytes() : 0, rows);
  }
  // Lane `lane`'s rows of a target row tensor, kDecodeRows from row
  // kDecodeRows x lane, and those of the first `lanes` lanes.
  [[nodiscard]] metal::MetalBuffer laneRows(TargetTensor tensor, uint32_t lane) const {
    if (lane >= kLaneCount)
      throw std::out_of_range("invalid decode lane");
    return rows(tensor, {lane * kDecodeRows, kDecodeRows});
  }
  [[nodiscard]] metal::MetalBuffer lanesRows(TargetTensor tensor, uint32_t lanes) const {
    if (!lanes || lanes > kLaneCount)
      throw std::out_of_range("invalid decode batch width");
    return rows(tensor, {0, lanes * kDecodeRows});
  }

  [[nodiscard]] ops::LinearScratch linearScratch() const { return linearScratch_; }
  static ops::LinearScratchSize linearScratchSize(const RuntimeGeometry &geometry,
                                                 const ops::ExecutionPlans &operators);

  [[nodiscard]] metal::MetalBuffer gateScratch() const { return gateScratch_; }

  static uint64_t gateScratchBytes(const RuntimeGeometry &geometry,
                                  const ops::ExecutionPlans &operators) {
    // One private gate buffer is reused serially by the target dense FFN (when
    // present), over a step's rows, and the always-dense DFlash draft, over
    // its block's. Sparse target FFNs use their own route-major arena tensors,
    // but must not remove the draft's scratch.
    const uint64_t draft = operators.gateUpWorkspace({geometry.draft.intermediateSize, geometry.draft.hiddenSize},
                                                     kLaneCount * kDraftRows);
    uint64_t target = 0;
    for (const auto &p : geometry.target.gateUpProjections)
      target = std::max(target, operators.gateUpWorkspace(p, kDecodeStepRows));
    return std::max(target, draft);
  }

  // The offset in batchSlice(DecodeTensor::GdnTape, kLaneCount) of state
  // lane `stateLane`'s tape slot of parity `parity`.
  [[nodiscard]] uint64_t gdnTapeSlot(uint32_t stateLane, uint32_t parity) const {
    if (stateLane >= kLaneCount || parity > 1)
      throw std::out_of_range("invalid GDN tape slot");
    return (uint64_t{stateLane} * 2 + parity) * gdnTapeSlotBytes(geometry_);
  }

  [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }

private:
  // Rows `rows` of the kDecodeStepRows rows the `bytes` bytes at `offset` of
  // `buffer` hold, none of no bytes.
  [[nodiscard]] metal::MetalBuffer stepRows(const metal::MetalBuffer &buffer, uint64_t offset, uint64_t bytes,
                                            ops::LaneRows rows) const {
    if (!rows.rows || uint64_t{rows.first} + rows.rows > kDecodeStepRows)
      throw std::out_of_range("invalid decode step rows");
    if (!bytes) return {};
    const uint64_t row = bytes / kDecodeStepRows;
    return backend_.view(buffer, offset + rows.first * row, rows.rows * row);
  }

  // The lanes' tensors' bytes, each tensor's aligned, before the target
  // step tensors.
  [[nodiscard]] static uint64_t laneTensorBytes(const RuntimeGeometry &geometry, const ops::ExecutionPlans &operators) {
    uint64_t bytes = 0;
    for (const uint64_t value : decodeTensorBytes(geometry, operators))
      bytes = checkedAdd(bytes, alignUp(checkedMultiply(value, kLaneCount, "decode tensor")), "decode arena");
    return bytes;
  }

  metal::MetalBackend &backend_;
  RuntimeGeometry geometry_;
  metal::MetalBuffer base_;
  TargetArena target_;
  std::array<std::array<metal::MetalBuffer, decodeTensorCount>, kLaneCount> tensors_{};
  std::array<uint64_t, decodeTensorCount> offsets_{};
  std::array<uint64_t, decodeTensorCount> sizes_{};
  metal::MetalBuffer gateScratch_;
  ops::LinearScratch linearScratch_;
  uint64_t bytes_ = 0;
};

} // namespace splash::model

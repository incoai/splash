#pragma once

// Shared Qwen prefill/decode scratch layouts and allocation owners.

#include "model/DFlashDraft.hpp"
#include "model/ModelFactory.hpp"
#include "model/QwenState.hpp"
#include "model/QwenTarget.hpp"

#include "Checked.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Sampling.h"
#include "ops/ExecutionPlans.hpp"
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
inline constexpr uint32_t kDecodeRows = ExecutionLimits::targetVerifyRows;
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
    RuntimeGeometry result;
    result.draft = model.draft.layout;
    result.target = std::visit(
        [&](const auto &weights) { return qwenTargetGeometry(weights, result.draft.targetLayerIds()); },
        model.target);
    result.target.kvLayout = model.targetKvLayout(kvFormat);
    result.target.stateLayout = model.stateLayout(stateFormat).target;
    result.target.ropeScaling = model.descriptor.ropeScaling;
    if (parts.kv) result.spilledKv = model.descriptor.capabilities.sparseKv;
    if (parts.experts || parts.kv) result.prefillChunks = kPrefillGroupChunks;
    if (!result.target.valid() || !result.draft.stateLayout().valid() ||
        (parts.kv && (!result.spilledKv || !result.target.kvLayout.spillable())) ||
        result.target.contextTokens() != model.descriptor.capabilities.maximumContextTokens ||
        result.target.hiddenSize != result.draft.hiddenSize ||
        result.target.vocabularySize != result.draft.vocabularySize ||
        result.target.capturedHiddenSize() != result.draft.targetHiddenSize()) {
      throw std::invalid_argument("invalid model runtime geometry");
    }
    // Sized from the geometry, once it is valid.
    result.targetScratch = std::visit(
        [&](const auto &weights) {
          return qwenTargetScratchBytes(weights, result.target, result.prefillChunks, parts.tableRowBytes);
        },
        model.target);
    return result;
  }

  [[nodiscard]] uint32_t draftRotaryPairs() const noexcept {
    return draft.attentionHeadDimension / 2;
  }
  [[nodiscard]] uint32_t maskWords() const noexcept {
    return maskWordsPerToken(target.vocabularySize);
  }
};

template <class T> constexpr uint64_t bytesFor(uint64_t elements) noexcept {
  return elements * sizeof(T);
}

enum class PrefillTensor : uint32_t {
  Hidden0,
  Hidden1,
  InputTokens,
  Normalized,
  Captured,
  GdnPacked,
  GdnQueries,
  GdnKeys,
  GdnValues,
  GdnDecay,
  GdnBeta,
  Recurrent,
  GdnHidden,
  GdnOutput,
  GateIntermediate,
  Intermediate,
  FullPacked,
  FullQueries,
  FullAttention,
  AttentionPartials,
  AttentionStatistics,
  AttentionHidden,
  AttentionOutput,
  TargetPositions,
  DraftPositions,
  TargetInverseFrequencies,
  DraftInverseFrequencies,
  RopeCos,
  RopeSin,
  ContextProjected,
  ContextHidden,
  ContextKv,
  DraftRopeCos,
  DraftRopeSin,
  ChunkKeys,
  ChunkValues,
  // One tensor per ops::kMoeScratchFields entry, in its order (moeScratchTensor).
  MoeScratch,
  MoeScratchLast = MoeScratch + ops::kMoeScratchFields.size() - 1,
  LinearPartials,
  LinearCounters,
  // The rotated input of a rotated projection (ops::LinearScratch::rotated).
  LinearRotated,
  // A prefill group's scratch GDN cell, which carries the state from a chunk
  // to the next when the lane's cells cannot (Runtime's prefill groups).
  GroupCell,
  Count,
};

constexpr uint32_t prefillTensorCount =
    static_cast<uint32_t>(PrefillTensor::Count);

// The arena tensor of MoE scratch field `field` (ops::kMoeScratchFields).
template <class Tensor>
constexpr Tensor moeScratchTensor(size_t field) noexcept {
  return static_cast<Tensor>(static_cast<uint32_t>(Tensor::MoeScratch) + field);
}

// Whether a prefill tensor's rows outlive a layer of a prefill group, which
// then holds every chunk's: the rows the layers carry (tokens and positions,
// RoPE, the hidden rows), those of the FFN its routed experts read after the
// layer's load (input, the mixer's output they add to, the MoE's routing),
// and those the draft's context reads once every layer ran (the captured
// target rows, their draft positions and RoPE). A tensor a family does not
// use holds no rows; a target's own scratch is its own (QwenTarget).
[[nodiscard]] constexpr bool prefillGroupTensor(PrefillTensor tensor) noexcept {
  switch (tensor) {
  case PrefillTensor::Hidden0:
  case PrefillTensor::Hidden1:
  case PrefillTensor::InputTokens:
  case PrefillTensor::TargetPositions:
  case PrefillTensor::RopeCos:
  case PrefillTensor::RopeSin:
  case PrefillTensor::Captured:
  case PrefillTensor::DraftPositions:
  case PrefillTensor::DraftRopeCos:
  case PrefillTensor::DraftRopeSin:
  case PrefillTensor::Normalized:
  case PrefillTensor::GdnOutput:
  case PrefillTensor::AttentionOutput:
    return true;
  default:
    break;
  }
  for (const auto routing : {&ops::MoeScratch::selectedExperts, &ops::MoeScratch::routingWeights,
                             &ops::MoeScratch::tileDescriptors, &ops::MoeScratch::tileCount,
                             &ops::MoeScratch::groupedRoutes, &ops::MoeScratch::routeRows})
    for (size_t field = 0; field < ops::kMoeScratchFields.size(); ++field)
      if (ops::kMoeScratchFields[field].buffer == routing && tensor == moeScratchTensor<PrefillTensor>(field))
        return true;
  return false;
}

// Sizes depend on the geometry and the device's operator plans.
[[nodiscard]] std::array<uint64_t, prefillTensorCount>
prefillTensorBytes(const RuntimeGeometry &geometry,
                   const ops::ExecutionPlans &operators);
[[nodiscard]] uint64_t prefillArenaBytes(const RuntimeGeometry &geometry,
                                         const ops::ExecutionPlans &operators);
// The arena's and the target's own prefill scratch (qwenTargetScratchBytes).
[[nodiscard]] uint64_t plannedPrefillBytes(const RuntimeGeometry &geometry,
                                           const ops::ExecutionPlans &operators);

// A group's tensors (prefillGroupTensor) hold chunk c's rows from c times a
// chunk's (aligned) bytes on, which get(tensor, c) views.
class PrefillArena final {
public:
  PrefillArena(metal::MetalBackend &backend, const RuntimeGeometry &geometry,
                const ops::ExecutionPlans &operators)
      : bytes_(prefillArenaBytes(geometry, operators)), chunks_(geometry.prefillChunks) {
    const auto sizes = prefillTensorBytes(geometry, operators);
    base_ = backend.allocateBuffer(bytes_, metal::BufferStorage::Shared,
                                   "qwen-shared-prefill");
    uint64_t cursor = 0;
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
    if (const metal::MetalBuffer cell = get(PrefillTensor::GroupCell))
      groupCell_ = gdnCellBuffers(backend, cell, geometry.target.stateLayout);
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
  // A group's scratch cell (PrefillTensor::GroupCell); null without groups.
  [[nodiscard]] const GdnParityBuffers *groupCell() const noexcept {
    return groupCell_ ? &*groupCell_ : nullptr;
  }
  [[nodiscard]] ops::MoeScratch moeScratch(uint32_t chunk) const {
    ops::MoeScratch scratch;
    for (size_t field = 0; field < ops::kMoeScratchFields.size(); ++field)
      scratch.*ops::kMoeScratchFields[field].buffer =
          get(moeScratchTensor<PrefillTensor>(field), chunk);
    return scratch;
  }
  [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }

private:
  metal::MetalBuffer base_;
  std::array<metal::MetalBuffer, prefillTensorCount> tensors_{};
  std::array<std::vector<metal::MetalBuffer>, prefillTensorCount> chunkViews_{};
  std::optional<GdnParityBuffers> groupCell_;
  uint64_t bytes_ = 0;
  uint32_t chunks_ = 1;
};

enum class DecodeTensor : uint32_t {
  Hidden0,
  Hidden1,
  InputTokens,
  Normalized,
  GdnHidden,
  GdnOutput,
  Intermediate,
  FullPacked,
  FullQueries,
  AttentionPartials,
  AttentionStatistics,
  FullAttention,
  AttentionHidden,
  AttentionOutput,
  Positions,
  DraftPositions,
  RopeCos,
  RopeSin,
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
  AcceptedCount,
  VerifyRows,
  VerifyConfidence,
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
  ProposalProbs,
  ProposedTokens,
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
  // A GDN layer's packed input projection rows, which its mixer reads.
  GdnPacked,
  // Indexed by state lane, like PageTable: the lane's two decode tape slots
  // (ops::GdnTapeLane), one per state parity (gdnTapeSlot).
  GdnTape,
  ChunkKeysBase,
  ChunkValuesBase,
  // One tensor per ops::kMoeScratchFields entry, in its order (moeScratchTensor).
  MoeScratch,
  MoeScratchLast = MoeScratch + ops::kMoeScratchFields.size() - 1,
  Count,
};

constexpr uint32_t decodeTensorCount =
    static_cast<uint32_t>(DecodeTensor::Count);

// The attention layers' verify chunk staging, which holds a stride per lane
// for each attention layer, layer by layer.
constexpr bool isLayerMajorTensor(DecodeTensor tensor) noexcept {
  return tensor == DecodeTensor::ChunkKeysBase ||
         tensor == DecodeTensor::ChunkValuesBase;
}

// One slot of a state lane's decode tape (DecodeTensor::GdnTape): one step of
// every GDN layer (ops::GdnTapeLane). A lane holds two, one per state parity.
[[nodiscard]] uint64_t gdnTapeSlotBytes(const RuntimeGeometry &geometry) noexcept;

[[nodiscard]] std::array<uint64_t, decodeTensorCount>
decodeTensorBytes(const RuntimeGeometry &geometry,
                  const ops::ExecutionPlans &operators);
// A decode tensor is one M32 allocation.  B1/B2/B3/B4 are prefixes
// containing 8/16/24/32 rows. Lanes are never separated by arena-alignment
// holes; only whole tensor boundaries are aligned.
[[nodiscard]] uint64_t decodeArenaBaseBytes(const RuntimeGeometry &geometry,
                                            const ops::ExecutionPlans &operators);
// The arena's base, its gate and linear scratch.
[[nodiscard]] uint64_t decodeArenaBytes(const RuntimeGeometry &geometry,
                                        const ops::ExecutionPlans &operators);
// The arena's and the target's own decode scratch (qwenTargetScratchBytes).
[[nodiscard]] uint64_t plannedDecodeBytes(const RuntimeGeometry &geometry,
                                          const ops::ExecutionPlans &operators);

class DecodeArena final {
public:
  DecodeArena(metal::MetalBackend &backend, RuntimeGeometry geometry,
               const ops::ExecutionPlans &operators)
      : backend_(backend), geometry_(std::move(geometry)) {
    const uint64_t baseBytes = decodeArenaBaseBytes(geometry_, operators);
    auto sizes = decodeTensorBytes(geometry_, operators);
    base_ = backend_.allocateBuffer(baseBytes, metal::BufferStorage::Shared,
                                    "qwen-shared-decode");
    uint64_t cursor = 0;
    for (uint32_t tensor = 0; tensor < sizes.size(); ++tensor) {
      uint64_t stride = sizes[tensor];
      offsets_[tensor] = cursor;
      sizes_[tensor] = sizes[tensor];
      const DecodeTensor kind = static_cast<DecodeTensor>(tensor);
      if (sizes[tensor] && !isLayerMajorTensor(kind)) {
        for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
          tensors_[lane][tensor] = backend_.view(
              base_, cursor + uint64_t{lane} * stride, sizes[tensor]);
        }
      }
      cursor +=
          alignUp(checkedMultiply(stride, kLaneCount, "decode tensor"));
    }
    if (cursor != baseBytes)
      throw std::logic_error("decode arena mismatch");
    // Sampled rows' draws return their arrival counts to zero; they start
    // there.
    const metal::MetalBuffer arrivals =
        batchSlice(DecodeTensor::TargetVocabularyArrivals, kLaneCount);
    std::memset(arrivals.contents(), 0, arrivals.sizeBytes());

    const uint64_t denseScratchBytes = gateScratchBytes(geometry_, operators);
    if (denseScratchBytes) {
      gateScratch_ = backend_.allocateBuffer(
          denseScratchBytes, metal::BufferStorage::Private, "qwen-gate-scratch");
    }
    // Each field exists only when some plan uses it (split-only plans have
    // partials and counters but no activation table).
    const auto linearSize = linearScratchSize(geometry_, operators);
    const auto allocate = [&](uint64_t bytes, metal::BufferStorage storage, const char *label) {
      return bytes ? backend_.allocateBuffer(bytes, storage, label) : metal::MetalBuffer{};
    };
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
    if (isLayerMajorTensor(tensor)) {
      throw std::logic_error(
          "layer-major decode scratch requires a layer view");
    }
    return tensors_[lane][static_cast<uint32_t>(tensor)];
  }

  [[nodiscard]] metal::MetalBuffer batchSlice(DecodeTensor tensor, uint32_t lanes) const {
    if (!lanes || lanes > kLaneCount)
      throw std::out_of_range("invalid decode batch width");
    if (isLayerMajorTensor(tensor)) {
      throw std::logic_error(
          "layer-major decode scratch requires a layer view");
    }
    const uint32_t index = static_cast<uint32_t>(tensor);
    // Model-specific scratch is represented by an empty buffer.  The
    // selected target graph consumes either dense-FFN or MoE tensors, never
    // both, so no dummy allocation is needed for the inactive operator.
    if (!sizes_[index])
      return {};
    return backend_.view(base_, offsets_[index],
                         uint64_t{lanes} * sizes_[index]);
  }

  [[nodiscard]] ops::MoeScratch moeScratch(uint32_t lanes) const {
    ops::MoeScratch scratch;
    for (size_t field = 0; field < ops::kMoeScratchFields.size(); ++field)
      scratch.*ops::kMoeScratchFields[field].buffer =
          batchSlice(moeScratchTensor<DecodeTensor>(field), lanes);
    return scratch;
  }

  [[nodiscard]] ops::LinearScratch linearScratch() const { return linearScratch_; }
  static ops::LinearScratchSize linearScratchSize(const RuntimeGeometry &geometry,
                                                 const ops::ExecutionPlans &operators);

  [[nodiscard]] metal::MetalBuffer gateScratch() const { return gateScratch_; }

  static uint64_t gateScratchBytes(const RuntimeGeometry &geometry,
                                  const ops::ExecutionPlans &operators) {
    // One private gate buffer is reused serially by the target dense FFN (when
    // present) and the always-dense DFlash draft. Sparse target FFNs use their
    // own route-major arena tensors, but must not remove the draft's scratch.
    const uint64_t draft = operators.gateUpWorkspace(
        {geometry.draft.intermediateSize, geometry.draft.hiddenSize});
    uint64_t target = 0;
    for (const auto &p : geometry.target.gateUpProjections)
      target = std::max(target, operators.gateUpWorkspace(p));
    return std::max(target, draft);
  }

  // The offset in batchSlice(DecodeTensor::GdnTape, kLaneCount) of state
  // lane `stateLane`'s tape slot of parity `parity`.
  [[nodiscard]] uint64_t gdnTapeSlot(uint32_t stateLane, uint32_t parity) const {
    if (stateLane >= kLaneCount || parity > 1)
      throw std::out_of_range("invalid GDN tape slot");
    return (uint64_t{stateLane} * 2 + parity) * gdnTapeSlotBytes(geometry_);
  }

  [[nodiscard]] metal::MetalBuffer attentionBatchSlice(DecodeTensor base,
                                                uint32_t attentionLayer,
                                                uint32_t lanes) const {
    const uint32_t layers = geometry_.target.kvLayout.attentionLayers;
    if (!lanes || lanes > kLaneCount || attentionLayer >= layers ||
        !isLayerMajorTensor(base)) {
      throw std::out_of_range("invalid batched attention layer");
    }
    return layerBatchSlice(base, layers, attentionLayer, lanes);
  }

  [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }

private:
  // A layer-major tensor holds one stride per lane for each of its `layers`
  // layers, layer by layer; the planner sized it as layers x stride, so the
  // stride is recovered here, not supplied.
  [[nodiscard]] metal::MetalBuffer layerBatchSlice(DecodeTensor base, uint32_t layers,
                                                   uint32_t layer, uint32_t lanes) const {
    const uint32_t index = static_cast<uint32_t>(base);
    const uint64_t stride = sizes_[index] / layers;
    return backend_.view(base_, offsets_[index] + uint64_t{layer} * kLaneCount * stride,
                         uint64_t{lanes} * stride);
  }

  metal::MetalBackend &backend_;
  RuntimeGeometry geometry_;
  metal::MetalBuffer base_;
  std::array<std::array<metal::MetalBuffer, decodeTensorCount>, kLaneCount> tensors_{};
  std::array<uint64_t, decodeTensorCount> offsets_{};
  std::array<uint64_t, decodeTensorCount> sizes_{};
  metal::MetalBuffer gateScratch_;
  ops::LinearScratch linearScratch_;
  uint64_t bytes_ = 0;
};

} // namespace splash::model

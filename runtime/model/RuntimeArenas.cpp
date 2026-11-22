#include "model/RuntimeArenas.hpp"

#include "ops/DraftSelector.hpp"
#include "ops/Sampling.hpp"

namespace splash::model {

std::array<uint64_t, targetTensorCount> targetTensorBytes(const QwenTargetGeometry &target, const TargetRows &shape) {
  std::array<uint64_t, targetTensorCount> result{};
  auto put = [&](TargetTensor tensor, uint64_t bytes) {
    auto &size = result[static_cast<uint32_t>(tensor)];
    size = std::max(size, bytes);
  };
  const uint64_t rows = shape.rows;
  // Each of a chunk's sequences stages its rows in whole KV blocks
  // (ops::PagedAttention), the last of each block's rows unused.
  const uint64_t stagedRows = rows + uint64_t{shape.lanes} * (kTileRows - 1);
  put(TargetTensor::Hidden0, bytesFor<uint16_t>(rows * target.hiddenSize));
  put(TargetTensor::Hidden1, bytesFor<uint16_t>(rows * target.hiddenSize));
  put(TargetTensor::Normalized, bytesFor<uint16_t>(rows * target.hiddenSize));
  // The GDN layers' scratch, which a target without them has none of.
  const uint64_t gdnRows = target.stateLayout.layers ? rows : 0;
  put(TargetTensor::GdnPacked, bytesFor<uint16_t>(gdnRows * target.packedGdnWidth));
  put(TargetTensor::GdnQueries, bytesFor<uint16_t>(gdnRows * target.gdnKeyWidth()));
  put(TargetTensor::GdnKeys, bytesFor<uint16_t>(gdnRows * target.gdnKeyWidth()));
  put(TargetTensor::GdnValues, bytesFor<uint16_t>(gdnRows * target.attentionWidth));
  put(TargetTensor::GdnDecay, bytesFor<float>(gdnRows * target.gdnValueHeads));
  put(TargetTensor::GdnBeta, bytesFor<uint16_t>(gdnRows * target.gdnValueHeads));
  put(TargetTensor::Recurrent, bytesFor<uint16_t>(gdnRows * target.attentionWidth));
  put(TargetTensor::GdnHidden, bytesFor<uint16_t>(gdnRows * target.attentionWidth));
  put(TargetTensor::Intermediate, bytesFor<uint16_t>(rows * target.intermediateSize));
  put(TargetTensor::FullPacked, bytesFor<uint16_t>(rows * target.packedFullWidth));
  put(TargetTensor::FullQueries,
      bytesFor<uint16_t>(uint64_t{target.attentionQueryHeads} * stagedRows * target.attentionHeadDimension));
  put(TargetTensor::AttentionPartials, shape.attention.partialsBytes);
  put(TargetTensor::AttentionStatistics, shape.attention.statisticsBytes);
  put(TargetTensor::AttentionHidden, bytesFor<uint16_t>(rows * target.attentionWidth));
  // The rows a mixer adds its output to, which the FFN adds to.
  if (target.mixersAddToHidden()) {
    put(TargetTensor::GdnOutput, bytesFor<uint16_t>(gdnRows * target.hiddenSize));
    put(TargetTensor::AttentionOutput, bytesFor<uint16_t>(rows * target.hiddenSize));
  }
  put(TargetTensor::RopeCos, bytesFor<float>(rows * target.rotaryPairs));
  put(TargetTensor::RopeSin, bytesFor<float>(rows * target.rotaryPairs));
  if (target.ffnKind == QwenFfnKind::SparseMoe)
    for (size_t field = 0; field < ops::kMoeScratchFields.size(); ++field)
      put(moeScratchTensor<TargetTensor>(field), shape.moe.*ops::kMoeScratchFields[field].bytes);
  // A prefill group's tensors hold every chunk's rows, each chunk's aligned.
  if (shape.chunks > 1) {
    TargetRows chunk = shape;
    chunk.rows = shape.chunkRows;
    chunk.chunks = 1;
    const std::array<uint64_t, targetTensorCount> each = targetTensorBytes(target, chunk);
    for (uint32_t index = 0; index < result.size(); ++index)
      if (result[index] && targetGroupTensor(static_cast<TargetTensor>(index)))
        result[index] = alignUp(each[index]) * shape.chunks;
  }
  return result;
}

TargetRows prefillTargetRows(const RuntimeGeometry &geometry, const ops::ExecutionPlans &operators) {
  const QwenTargetGeometry &target = geometry.target;
  TargetRows shape{.rows = kPrefillRows,
                   .chunks = geometry.prefillChunks,
                   .chunkRows = geometry.prefillChunkRows,
                   .lanes = geometry.target.lanes,
                   // Any one sequence's splits: a chunk's sequences share the
                   // workspace in turn where they do not fit together.
                   .attention = operators.sequenceAttentionWorkspace(target.attentionQueryHeads, target.kvLayout),
                   .moe = {}};
  if (target.ffnKind == QwenFfnKind::SparseMoe)
    shape.moe = operators.moeWorkspace(target.moeShape(), target.moeRouterFormat, kPrefillRows, ops::StepKind::Prefill);
  return shape;
}

TargetRows decodeTargetRows(const RuntimeGeometry &geometry, const ops::ExecutionPlans &operators) {
  const QwenTargetGeometry &target = geometry.target;
  // Every lane's tiles at the most splits, so a step's lanes attend in one
  // dispatch whatever their histories, and the MoE plan of every step of up
  // to its most rows.
  const ops::AttentionWorkspace lane = operators.laneAttentionWorkspace(target.attentionQueryHeads, target.kvLayout);
  TargetRows shape{.rows = decodeStorageRows(geometry.target.lanes),
                   .chunks = 1,
                   .chunkRows = decodeStorageRows(geometry.target.lanes),
                   .lanes = geometry.target.lanes,
                   .attention = {lane.partialsBytes * geometry.target.lanes, lane.statisticsBytes * geometry.target.lanes},
                   .moe = {}};
  if (target.ffnKind == QwenFfnKind::SparseMoe)
    shape.moe = operators.moeWorkspace(target.moeShape(), target.moeRouterFormat, target.stepRows(),
                                     ops::StepKind::Decode);
  return shape;
}

namespace {

// The prefill tensors of a command of `rows` rows (prefillTensorBytes).
std::array<uint64_t, prefillTensorCount> prefillTensorBytes(const RuntimeGeometry &geometry,
                                                            const ops::ExecutionPlans &operators,
                                                            uint32_t rows) {
  std::array<uint64_t, prefillTensorCount> result{};
  auto put = [&](PrefillTensor tensor, uint64_t bytes) {
    auto &size = result[static_cast<uint32_t>(tensor)];
    size = std::max(size, bytes);
  };
  put(PrefillTensor::InputTokens, bytesFor<uint32_t>(rows));
  put(PrefillTensor::Captured,
      bytesFor<uint16_t>(uint64_t{rows} *
                         geometry.target.capturedHiddenSize()));
  put(PrefillTensor::GateIntermediate,
      bytesFor<uint16_t>(uint64_t{rows} *
                         geometry.target.intermediateSize));
  // Three rotary axes per row (Qwen3.5 M-RoPE); text rows repeat one value.
  put(PrefillTensor::TargetPositions,
      bytesFor<uint32_t>(uint64_t{rows} * 3));
  put(PrefillTensor::DraftPositions, bytesFor<uint32_t>(rows));
  put(PrefillTensor::LaneLayout, bytesFor<SplashLaneRows>(geometry.target.lanes));
  put(PrefillTensor::TargetInverseFrequencies,
      bytesFor<float>(geometry.target.rotaryPairs));
  put(PrefillTensor::DraftInverseFrequencies,
      bytesFor<float>(geometry.draftRotaryPairs()));
  put(PrefillTensor::ContextProjected,
      bytesFor<uint16_t>(uint64_t{rows} * geometry.draft.hiddenSize));
  put(PrefillTensor::ContextHidden,
      bytesFor<uint16_t>(uint64_t{rows} * geometry.draft.hiddenSize));
  put(PrefillTensor::ContextKv,
      bytesFor<uint16_t>(uint64_t{rows} *
                         geometry.draft.contextKvSize()));
  put(PrefillTensor::DraftRopeCos,
      bytesFor<float>(uint64_t{rows} *
                      geometry.draftRotaryPairs()));
  put(PrefillTensor::DraftRopeSin,
      bytesFor<float>(uint64_t{rows} *
                      geometry.draftRotaryPairs()));
  // The split partials and counters, the rotated rows, the register tile's
  // tables of the plans of up to 32 rows and the int8 tile's codes and scales
  // of a full chunk: the target's, and any draft's context projections' on
  // the draft's plans.
  ops::LinearScratchSize linear;
  for (const auto &projection : geometry.target.prefillProjections)
    linear.include(operators.linear().scratchSize(projection, rows));
  const auto &draft = geometry.draft;
  for (const ops::ProjectionShape projection :
       {ops::ProjectionShape{draft.hiddenSize, draft.targetHiddenSize()}, {draft.contextKvSize(), draft.hiddenSize}})
    if (draft.present()) linear.include(operators.draftLinear().scratchSize(projection, rows));
  put(PrefillTensor::LinearPartials, linear.partials);
  put(PrefillTensor::LinearCounters, linear.counters);
  put(PrefillTensor::LinearTable, linear.input);
  put(PrefillTensor::LinearSums, linear.sums);
  put(PrefillTensor::LinearRotated, linear.rotated);
  return result;
}

} // namespace

std::array<uint64_t, prefillTensorCount> prefillTensorBytes(const RuntimeGeometry &geometry,
                                                            const ops::ExecutionPlans &operators) {
  std::array<uint64_t, prefillTensorCount> result = prefillTensorBytes(geometry, operators, kPrefillRows);
  // A prefill group's tensors hold every chunk's rows, each chunk's aligned,
  // and its chain's scratch cell.
  if (geometry.prefillChunks > 1) {
    const std::array<uint64_t, prefillTensorCount> each =
        prefillTensorBytes(geometry, operators, geometry.prefillChunkRows);
    for (uint32_t index = 0; index < result.size(); ++index)
      if (result[index] && prefillGroupTensor(static_cast<PrefillTensor>(index)))
        result[index] = alignUp(each[index]) * geometry.prefillChunks;
    result[static_cast<uint32_t>(PrefillTensor::GroupCell)] = groupCellBytes(geometry.target.stateLayout);
  }
  return result;
}

uint64_t prefillArenaBytes(const RuntimeGeometry &geometry,
                          const ops::ExecutionPlans &operators) {
  uint64_t bytes = TargetArena::bytes(geometry.target, prefillTargetRows(geometry, operators));
  for (uint64_t value : prefillTensorBytes(geometry, operators)) {
    bytes = checkedAdd(bytes, alignUp(value), "prefill arena");
  }
  return bytes;
}

uint64_t plannedPrefillBytes(const RuntimeGeometry &geometry,
                            const ops::ExecutionPlans &operators) {
  return checkedAdd(prefillArenaBytes(geometry, operators),
                    geometry.targetScratch.prefill,
                    "prefill scratch");
}

uint64_t gdnTapeSlotBytes(const RuntimeGeometry &geometry) noexcept {
  return uint64_t{geometry.target.stateLayout.layers} *
         ops::gdnTapeLayerBytes(geometry.target.gdnShape());
}
std::array<uint64_t, decodeTensorCount>
decodeTensorBytes(const RuntimeGeometry &geometry,
                  const ops::ExecutionPlans &operators) {
  std::array<uint64_t, decodeTensorCount> result{};
  // A model without a draft has no draft tensors: its draft's dimensions
  // size them to nothing, and its attention none. Its selector's trees and
  // host chains stay, which a cycle's rows are laid out from.
  const auto draftWorkspace = geometry.draft.present()
                                  ? operators.draftAttentionWorkspacePerLane(geometry.draft.attentionShape())
                                  : ops::DraftAttentionWorkspace{};
  const auto samplingWorkspace = ops::Sampling::workspace(kDecodeRows);
  const auto selectorWorkspace = ops::DraftSelector::workspace(kDraftProposalTokens);
  auto put = [&](DecodeTensor tensor, uint64_t bytes) {
    auto &size = result[static_cast<uint32_t>(tensor)];
    size = std::max(size, bytes);
  };
  // A step's packed rows (decodeRowTensor) and a lane's verify block's
  // positions: kDecodeRows a lane.
  const uint64_t r = kDecodeRows;
  put(DecodeTensor::InputTokens, bytesFor<uint32_t>(r));
  put(DecodeTensor::DraftPositions, bytesFor<uint32_t>(r));
  put(DecodeTensor::ContextProjected,
      bytesFor<uint16_t>(r * geometry.draft.hiddenSize));
  put(DecodeTensor::ContextHidden,
      bytesFor<uint16_t>(r * geometry.draft.hiddenSize));
  put(DecodeTensor::ContextKv,
      bytesFor<uint16_t>(r * geometry.draft.contextKvSize()));
  put(DecodeTensor::CapturedTargetHidden,
      bytesFor<uint16_t>(r * geometry.draft.targetHiddenSize()));
  put(DecodeTensor::DraftQueryKeys, draftWorkspace.queryKeysBytes);
  put(DecodeTensor::DraftQueryValues, draftWorkspace.queryValuesBytes);
  // A lane's verify block's positions, the first kDraftRows of which its
  // draft's rows take and one a retained row of which its context commit
  // takes, so one RoPE table per lane serves both.
  put(DecodeTensor::DraftRopeCos,
      bytesFor<float>(r * geometry.draftRotaryPairs()));
  put(DecodeTensor::DraftRopeSin,
      bytesFor<float>(r * geometry.draftRotaryPairs()));
  put(DecodeTensor::FinalHidden,
      bytesFor<uint16_t>(r * geometry.target.hiddenSize));
  put(DecodeTensor::Logits,
      bytesFor<float>(r * geometry.target.vocabularySize));
  put(DecodeTensor::ArgmaxValues, samplingWorkspace.argmaxValuesBytes);
  put(DecodeTensor::ArgmaxIndices, samplingWorkspace.argmaxIndicesBytes);
  put(DecodeTensor::TargetPartialMasses, samplingWorkspace.partialMassesBytes);
  put(DecodeTensor::TargetVocabularyRows, samplingWorkspace.vocabularyRowsBytes);
  put(DecodeTensor::TargetVocabularyRanges,
      samplingWorkspace.vocabularyRangesBytes);
  put(DecodeTensor::TargetVocabularyArrivals,
      samplingWorkspace.vocabularyArrivalsBytes);
  put(DecodeTensor::SamplingUniforms, bytesFor<float>(kSamplingUniformCount));
  put(DecodeTensor::ConstraintMasks,
      bytesFor<uint32_t>(uint64_t{ExecutionLimits::maximumStepTokens} *
                         geometry.maskWords()));
  put(DecodeTensor::OutputTokens, bytesFor<uint32_t>(r));
  put(DecodeTensor::RetainedCount, sizeof(uint32_t));
  put(DecodeTensor::RetainedRows, sizeof(uint32_t));
  put(DecodeTensor::AcceptedCount, sizeof(uint32_t));
  put(DecodeTensor::LaneLayout, sizeof(ops::LaneRows));
  // The draft's own rows: its block's, kDraftRows a lane.
  const uint64_t d = kDraftRows;
  put(DecodeTensor::DraftInputTokens, bytesFor<uint32_t>(d));
  for (uint32_t index = 0; index < 2; ++index) {
    put(static_cast<DecodeTensor>(
            static_cast<uint32_t>(DecodeTensor::DraftHidden0) + index),
        bytesFor<uint16_t>(d * geometry.draft.hiddenSize));
  }
  put(DecodeTensor::DraftNormalized,
      bytesFor<uint16_t>(d * geometry.draft.hiddenSize));
  put(DecodeTensor::DraftDynamic,
      bytesFor<uint16_t>(d * geometry.draft.dynamicSize));
  put(DecodeTensor::DraftConvolved, draftWorkspace.convolutionBytes);
  put(DecodeTensor::DraftProposalQkv, draftWorkspace.qkvBytes);
  put(DecodeTensor::DraftAttention, draftWorkspace.groupedQueriesBytes);
  put(DecodeTensor::DraftProjected,
      bytesFor<uint16_t>(d * geometry.draft.hiddenSize));
  put(DecodeTensor::DraftResidual,
      bytesFor<uint16_t>(d * geometry.draft.hiddenSize));
  put(DecodeTensor::DraftIntermediate,
      bytesFor<uint16_t>(d * geometry.draft.intermediateSize));
  put(DecodeTensor::DraftFinalHidden,
      bytesFor<uint16_t>(d * geometry.draft.hiddenSize));
  put(DecodeTensor::SelectorHidden,
      bytesFor<uint16_t>(d * geometry.draft.selectorRank));
  put(DecodeTensor::Candidates, selectorWorkspace.candidatesBytes);
  put(DecodeTensor::Unary, selectorWorkspace.unaryBytes);
  put(DecodeTensor::TopPartialIds, selectorWorkspace.partialIdsBytes);
  put(DecodeTensor::TopPartialValues, selectorWorkspace.partialValuesBytes);
  put(DecodeTensor::TreeRows, bytesFor<ops::TreeRow>(r));
  put(DecodeTensor::VerifySources, bytesFor<uint32_t>(r));
  // A lane's tree takes a verify block's rows.
  const uint64_t v = ExecutionLimits::targetVerifyRows;
  put(DecodeTensor::DraftTreeTokens, bytesFor<uint32_t>(v));
  put(DecodeTensor::DraftTrees, bytesFor<ops::TreeRow>(v));
  put(DecodeTensor::DraftTreeScores, bytesFor<float>(2 * v));
  put(DecodeTensor::DraftTreeDraws, bytesFor<uint8_t>(v * SPLASH_TREE_CHILDREN));
  put(DecodeTensor::DraftTreeNext, bytesFor<float>(v * SPLASH_DRAFT_CANDIDATES));
  put(DecodeTensor::DraftTreeCaps, bytesFor<float>(v));
  put(DecodeTensor::DraftTreeSizes, sizeof(uint32_t));
  put(DecodeTensor::ChainTokens, bytesFor<uint32_t>(SPLASH_TREE_CHAIN_TOKENS));
  put(DecodeTensor::ChainRates, bytesFor<float>(uint64_t{SPLASH_TREE_CHAIN_TOKENS} * SPLASH_TREE_AGREEMENTS));
  put(DecodeTensor::PageTable, bytesFor<SplashKvPage>(kMaximumPageTableEntries));
  if (geometry.target.kvLayout.indexDimension)
    put(DecodeTensor::IndexTable, bytesFor<SplashKvPage>(kMaximumPageTableEntries));
  // A verify lane's list of a tile of its rows holds each row's selected
  // blocks before the anchor's block and, from it on, the blocks of the K/V
  // slots of the path rows it attends, no more of those than the tile has
  // rows: at most selectedBlocks + 1 entries a row of the tile
  // (Qwen4Steps.cpp, listStride), though a tree row alone may hold more.
  if (const auto &shape = geometry.spilledKv)
    put(DecodeTensor::Locations, bytesFor<uint32_t>(uint64_t{shape->verifyRows} * (shape->selectedBlocks + 1)));
  put(DecodeTensor::PenaltyState,
      bytesFor<uint32_t>(geometry.target.vocabularySize));
  put(DecodeTensor::GdnTape, 2 * gdnTapeSlotBytes(geometry));
  return result;
}

namespace {

// A draft projection of `outputSize` by `inputSize` as its checkpoint stores
// it: af4g64 planes (DraftCheckpoint).
ops::ProjectionShape draftShape(uint32_t outputSize, uint32_t inputSize) {
  return ops::Projection(outputSize, inputSize,
                         ops::BlockWeights{{ops::QuantizedSegment::planes(GGUF_FMT_AF4G64, outputSize, inputSize, {},
                                                                          {}, {})}})
      .shape();
}

} // namespace

uint32_t draftStepLanes(const RuntimeGeometry &geometry, const ops::ExecutionPlans &operators, uint32_t lanes) {
  const auto &d = geometry.draft;
  const auto &t = geometry.target;
  const uint32_t rows = lanes * kDraftRows;
  uint32_t storage = rows;
  // A model without a draft binds its vocabulary head's rows alone.
  if (d.present())
    for (const auto &[output, input] : {std::pair{d.dynamicSize, d.hiddenSize}, {d.qkvSize, d.hiddenSize},
                                        {d.hiddenSize, d.attentionSize}, {d.intermediateSize, d.hiddenSize},
                                        {d.hiddenSize, d.intermediateSize}, {d.selectorRank, d.hiddenSize},
                                        {d.hiddenSize, d.targetHiddenSize()}, {d.contextKvSize(), d.hiddenSize}})
      storage = std::max(storage, operators.draftLinear().storageRows(rows, draftShape(output, input)));
  // The target's vocabulary head and the draft's view of its subset's rows,
  // as the target's weights give them (QwenTargetGeometry::finish).
  for (const ops::ProjectionShape &shape : t.decodeProjections)
    if (shape.inputSize == t.hiddenSize &&
        (shape.outputSize == t.vocabularySize || shape.outputSize == t.draftVocabulary.columns(t.vocabularySize)))
      storage = std::max(storage, operators.decodeLinear().storageRows(rows, shape));
  return (storage + kDraftRows - 1) / kDraftRows;
}

uint32_t decodeArenaLanes(const RuntimeGeometry &geometry, const ops::ExecutionPlans &operators) {
  uint32_t held = 0;
  for (uint32_t lanes = 1; lanes <= geometry.target.lanes; ++lanes)
    held = std::max(held, draftStepLanes(geometry, operators, lanes));
  return held;
}

uint64_t decodeArenaBaseBytes(const RuntimeGeometry &geometry,
                             const ops::ExecutionPlans &operators) {
  uint64_t bytes = TargetArena::bytes(geometry.target, decodeTargetRows(geometry, operators));
  const auto sizes = decodeTensorBytes(geometry, operators);
  const uint32_t held = decodeArenaLanes(geometry, operators);
  for (uint32_t tensor = 0; tensor < sizes.size(); ++tensor) {
    const uint32_t shares = decodeTensorLanes(static_cast<DecodeTensor>(tensor), held);
    bytes = checkedAdd(bytes, alignUp(checkedMultiply(sizes[tensor], shares, "decode tensor")), "decode arena");
  }
  return bytes;
}

ops::LinearScratchSize DecodeArena::linearScratchSize(
    const RuntimeGeometry &geometry, const ops::ExecutionPlans &operators) {
  const auto &d = geometry.draft;
  ops::LinearScratchSize result;
  // A step's rows: the target's projections, the vocabulary head shared with
  // the draft among them, and the draft's context projections.
  for (const auto &p : geometry.target.decodeProjections)
    result.include(operators.decodeLinear().scratchSize(p, geometry.target.stepRows()));
  if (!d.present())
    return result;
  for (const ops::ProjectionShape shape :
       {ops::ProjectionShape{d.hiddenSize, d.targetHiddenSize()}, {d.contextKvSize(), d.hiddenSize}})
    result.include(operators.draftDecodeLinear().scratchSize(shape, geometry.target.stepRows()));
  // The draft's own block, kDraftRows a lane.
  for (const ops::ProjectionShape shape : {ops::ProjectionShape{d.dynamicSize, d.hiddenSize},
       {d.qkvSize, d.hiddenSize}, {d.hiddenSize, d.attentionSize},
       {d.intermediateSize, d.hiddenSize}, {d.hiddenSize, d.intermediateSize},
       {d.selectorRank, d.hiddenSize}})
    result.include(operators.draftLinear().scratchSize(shape, geometry.target.lanes * kDraftRows));
  return result;
}

uint64_t decodeArenaBytes(const RuntimeGeometry &geometry,
                         const ops::ExecutionPlans &operators) {
  // Metal allocates a buffer in whole host pages and the memory ledger
  // charges what it allocates, so the arena allocates each scratch buffer
  // beside its base in whole pages and the plan counts them so: counted as
  // the fields' sum, Qwen3.8-Flash-Next's 122,880 bytes of linear sums took
  // 131,072, and the 8,192 bytes the plan left out cost one request's KV a
  // whole extent, 8,192 tokens of its context at --max-memory 9.4G.
  const ops::LinearScratchSize linear = DecodeArena::linearScratchSize(geometry, operators);
  uint64_t bytes = decodeArenaBaseBytes(geometry, operators);
  for (const uint64_t scratch : {DecodeArena::gateScratchBytes(geometry, operators), linear.input, linear.sums,
                                 linear.partials, linear.counters, linear.rotated})
    bytes = checkedAdd(bytes, alignUp(scratch), "decode scratch");
  return bytes;
}

uint64_t plannedDecodeBytes(const RuntimeGeometry &geometry,
                           const ops::ExecutionPlans &operators) {
  return checkedAdd(decodeArenaBytes(geometry, operators),
                    geometry.targetScratch.decode,
                    "decode scratch");
}

} // namespace splash::model

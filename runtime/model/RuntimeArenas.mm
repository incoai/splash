#include "model/RuntimeArenas.hpp"

#include "ops/Sampling.hpp"

namespace splash::model {

std::array<uint64_t, prefillTensorCount>
prefillTensorBytes(const RuntimeGeometry &geometry,
                   const ops::ExecutionPlans &operators) {
  std::array<uint64_t, prefillTensorCount> result{};
  auto put = [&](PrefillTensor tensor, uint64_t bytes) {
    result[static_cast<uint32_t>(tensor)] = bytes;
  };
  put(PrefillTensor::Hidden0,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} * geometry.target.hiddenSize));
  put(PrefillTensor::Hidden1,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} * geometry.target.hiddenSize));
  put(PrefillTensor::InputTokens, bytesFor<uint32_t>(kPrefillRows));
  put(PrefillTensor::Normalized,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} * geometry.target.hiddenSize));
  put(PrefillTensor::Captured,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.target.capturedHiddenSize()));
  put(PrefillTensor::GdnPacked,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.target.packedGdnWidth));
  put(PrefillTensor::GdnQueries,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.target.gdnKeyWidth()));
  put(PrefillTensor::GdnKeys,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.target.gdnKeyWidth()));
  put(PrefillTensor::GdnValues,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.target.attentionWidth));
  put(PrefillTensor::GdnDecay,
      bytesFor<float>(uint64_t{kPrefillRows} *
                      geometry.target.gdnValueHeads));
  put(PrefillTensor::GdnBeta,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.target.gdnValueHeads));
  put(PrefillTensor::Recurrent,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.target.attentionWidth));
  put(PrefillTensor::GdnHidden,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.target.attentionWidth));
  put(PrefillTensor::GdnOutput,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} * geometry.target.hiddenSize));
  put(PrefillTensor::GateIntermediate,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.target.denseIntermediateSize));
  put(PrefillTensor::Intermediate,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.target.denseIntermediateSize));
  put(PrefillTensor::FullPacked,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.target.packedAttentionWidth));
  put(PrefillTensor::FullQueries,
      bytesFor<uint16_t>(uint64_t{geometry.target.attentionQueryHeads} *
                         kPackedAttentionRows *
                         geometry.target.attentionHeadDimension));
  put(PrefillTensor::FullAttention,
      bytesFor<uint16_t>(uint64_t{geometry.target.attentionQueryHeads} *
                         kPackedAttentionRows *
                         geometry.target.attentionHeadDimension));
  const ops::AttentionWorkspace attentionWorkspace =
      operators.prefillAttentionWorkspace(
          kPrefillRows, geometry.target.attentionQueryHeads,
          geometry.target.kvLayout);
  put(PrefillTensor::AttentionPartials, attentionWorkspace.partialsBytes);
  put(PrefillTensor::AttentionStatistics, attentionWorkspace.statisticsBytes);
  put(PrefillTensor::AttentionHidden,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} *
                         geometry.target.attentionWidth));
  put(PrefillTensor::AttentionOutput,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} * geometry.target.hiddenSize));
  put(PrefillTensor::ProjectionSums,
      bytesFor<float>(uint64_t{kPrefillRows} *
                      geometry.projectionSumsWidth()));
  put(PrefillTensor::DownProjectionSums,
      bytesFor<float>(uint64_t{kPrefillRows} *
                      geometry.projectionSumsWidth()));
  // Three rotary axes per row (Qwen3.5 M-RoPE); text rows repeat one value.
  put(PrefillTensor::TargetPositions,
      bytesFor<uint32_t>(uint64_t{kPrefillRows} * 3));
  put(PrefillTensor::DraftPositions, bytesFor<uint32_t>(kPrefillRows));
  put(PrefillTensor::TargetInverseFrequencies,
      bytesFor<float>(geometry.target.rotaryPairs));
  put(PrefillTensor::DraftInverseFrequencies,
      bytesFor<float>(geometry.draftState.headDimension / 2));
  put(PrefillTensor::RopeCos,
      bytesFor<float>(uint64_t{kPrefillRows} * geometry.target.rotaryPairs));
  put(PrefillTensor::RopeSin,
      bytesFor<float>(uint64_t{kPrefillRows} * geometry.target.rotaryPairs));
  put(PrefillTensor::ContextProjected,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} * geometry.draft.hiddenSize));
  put(PrefillTensor::ContextHidden,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} * geometry.draft.hiddenSize));
  put(PrefillTensor::ContextQkv,
      bytesFor<uint16_t>(uint64_t{kPrefillRows} * geometry.draft.qkvSize));
  put(PrefillTensor::DraftRopeCos,
      bytesFor<float>(uint64_t{kPrefillRows} *
                      (geometry.draftState.headDimension / 2)));
  put(PrefillTensor::DraftRopeSin,
      bytesFor<float>(uint64_t{kPrefillRows} *
                      (geometry.draftState.headDimension / 2)));
  put(PrefillTensor::ChunkKeys,
      bytesFor<uint16_t>(uint64_t{geometry.target.attentionKvHeads} *
                         kPackedAttentionRows *
                         geometry.target.attentionHeadDimension));
  put(PrefillTensor::ChunkValues,
      bytesFor<uint16_t>(uint64_t{geometry.target.attentionKvHeads} *
                         kPackedAttentionRows *
                         geometry.target.attentionHeadDimension));
  if (geometry.target.ffnKind == QwenFfnKind::SparseMoe) {
    const ops::MoeWorkspace workspace =
        operators.moePrefillWorkspace(geometry.target.moe, kPrefillRows);
    put(PrefillTensor::MoeSelectedExperts, workspace.selectedExpertsBytes);
    put(PrefillTensor::MoeRoutingWeights, workspace.routingWeightsBytes);
    put(PrefillTensor::MoeTileDescriptors, workspace.tileDescriptorsBytes);
    put(PrefillTensor::MoeTileCount, workspace.tileCountBytes);
    put(PrefillTensor::MoeGroupedRoutes, workspace.groupedRoutesBytes);
    put(PrefillTensor::MoeRouteRows, workspace.routeRowsBytes);
    put(PrefillTensor::MoeGroupedInput, workspace.groupedInputBytes);
    put(PrefillTensor::MoeExpertIntermediate, workspace.expertIntermediateBytes);
    put(PrefillTensor::MoeExpertOutput, workspace.expertOutputBytes);
  }
  return result;
}

uint64_t plannedPrefillBytes(const RuntimeGeometry &geometry,
                            const ops::ExecutionPlans &operators) {
  uint64_t bytes = 0;
  for (uint64_t value : prefillTensorBytes(geometry, operators)) {
    bytes = checkedAdd(bytes, alignArena(value), "prefill arena");
  }
  return bytes;
}

static uint64_t gdnPackedStride(const RuntimeGeometry &geometry) noexcept {
  return bytesFor<uint16_t>(uint64_t{kDecodeRows} *
                            geometry.target.packedGdnWidth);
}
static uint64_t gdnMixedStride(const RuntimeGeometry &geometry) noexcept {
  return bytesFor<uint16_t>(uint64_t{kDecodeRows} *
                            geometry.target.convolutionDimension);
}
static uint64_t gdnDecayStride(const RuntimeGeometry &geometry) noexcept {
  return bytesFor<float>(uint64_t{kDecodeRows} *
                         geometry.target.gdnValueHeads);
}
static uint64_t gdnBetaStride(const RuntimeGeometry &geometry) noexcept {
  return bytesFor<uint16_t>(uint64_t{kDecodeRows} *
                            geometry.target.gdnValueHeads);
}
uint64_t decodeChunkLayerBytes(const RuntimeGeometry &geometry) noexcept {
  return bytesFor<uint16_t>(uint64_t{geometry.target.attentionKvHeads} *
                            kTileRows *
                            geometry.target.attentionHeadDimension);
}

std::array<uint64_t, decodeTensorCount>
decodeTensorBytes(const RuntimeGeometry &geometry,
                  const ops::ExecutionPlans &operators) {
  std::array<uint64_t, decodeTensorCount> result{};
  const auto draftWorkspace =
      operators.draftAttentionWorkspacePerLane(geometry.draft.attentionShape());
  const auto samplingWorkspace = ops::Sampling::workspace(kDecodeRows);
  const auto selectorWorkspace = ops::Sampling::draftWorkspace(kDraftProposalTokens);
  auto put = [&](DecodeTensor tensor, uint64_t bytes) {
    result[static_cast<uint32_t>(tensor)] = bytes;
  };
  const uint64_t r = kDecodeRows;
  put(DecodeTensor::Hidden0,
      bytesFor<uint16_t>(r * geometry.target.hiddenSize));
  put(DecodeTensor::Hidden1,
      bytesFor<uint16_t>(r * geometry.target.hiddenSize));
  put(DecodeTensor::InputTokens, bytesFor<uint32_t>(r));
  put(DecodeTensor::Normalized,
      bytesFor<uint16_t>(r * geometry.target.hiddenSize));
  put(DecodeTensor::Recurrent,
      bytesFor<uint16_t>(r * geometry.target.attentionWidth));
  put(DecodeTensor::GdnHidden,
      bytesFor<uint16_t>(r * geometry.target.attentionWidth));
  put(DecodeTensor::GdnOutput,
      bytesFor<uint16_t>(r * geometry.target.hiddenSize));
  put(DecodeTensor::Intermediate,
      bytesFor<uint16_t>(r * geometry.target.denseIntermediateSize));
  put(DecodeTensor::FullPacked,
      bytesFor<uint16_t>(r * geometry.target.packedAttentionWidth));
  put(DecodeTensor::FullQueries,
      bytesFor<uint16_t>(uint64_t{geometry.target.attentionQueryHeads} *
                         kTileRows * geometry.target.attentionHeadDimension));
  const ops::AttentionWorkspace attentionWorkspace =
      operators.verifyAttentionWorkspacePerLane(
          geometry.target.attentionQueryHeads, geometry.target.kvLayout);
  put(DecodeTensor::AttentionPartials, attentionWorkspace.partialsBytes);
  put(DecodeTensor::AttentionStatistics, attentionWorkspace.statisticsBytes);
  put(DecodeTensor::FullAttention,
      bytesFor<uint16_t>(uint64_t{geometry.target.attentionQueryHeads} *
                         kTileRows * geometry.target.attentionHeadDimension));
  put(DecodeTensor::AttentionHidden,
      bytesFor<uint16_t>(r * geometry.target.attentionWidth));
  put(DecodeTensor::AttentionOutput,
      bytesFor<uint16_t>(r * geometry.target.hiddenSize));
  put(DecodeTensor::Positions, bytesFor<uint32_t>(r * 3));
  put(DecodeTensor::DraftPositions, bytesFor<uint32_t>(r));
  put(DecodeTensor::RopeCos,
      bytesFor<float>(r * geometry.target.rotaryPairs));
  put(DecodeTensor::RopeSin,
      bytesFor<float>(r * geometry.target.rotaryPairs));
  put(DecodeTensor::Arrived, sizeof(uint32_t));
  put(DecodeTensor::Generation, sizeof(uint32_t));
  put(DecodeTensor::ContextProjected,
      bytesFor<uint16_t>(r * geometry.draft.hiddenSize));
  put(DecodeTensor::ContextHidden,
      bytesFor<uint16_t>(r * geometry.draft.hiddenSize));
  put(DecodeTensor::ContextQkv,
      bytesFor<uint16_t>(r * geometry.draft.qkvSize));
  put(DecodeTensor::CapturedTargetHidden,
      bytesFor<uint16_t>(r * geometry.draft.targetHiddenSize));
  put(DecodeTensor::DraftQueryKeys, draftWorkspace.queryKeysBytes);
  put(DecodeTensor::DraftQueryValues, draftWorkspace.queryValuesBytes);
  // Proposal attention and accepted target-hidden injection use the same
  // eight absolute positions, so one RoPE table per lane is sufficient.
  put(DecodeTensor::DraftRopeCos,
      bytesFor<float>(r * (geometry.draftState.headDimension / 2)));
  put(DecodeTensor::DraftRopeSin,
      bytesFor<float>(r * (geometry.draftState.headDimension / 2)));
  put(DecodeTensor::FinalHidden,
      bytesFor<uint16_t>(r * geometry.target.hiddenSize));
  put(DecodeTensor::Logits,
      bytesFor<uint16_t>(r * geometry.target.vocabularySize));
  put(DecodeTensor::ArgmaxValues, samplingWorkspace.argmaxValuesBytes);
  put(DecodeTensor::ArgmaxIndices, samplingWorkspace.argmaxIndicesBytes);
  put(DecodeTensor::TargetTopPartialIds, samplingWorkspace.partialIdsBytes);
  put(DecodeTensor::TargetTopPartialValues, samplingWorkspace.partialValuesBytes);
  put(DecodeTensor::TargetTopIds, samplingWorkspace.topIdsBytes);
  put(DecodeTensor::TargetTopProbs, samplingWorkspace.topProbabilitiesBytes);
  put(DecodeTensor::SamplingUniforms, bytesFor<float>(kSamplingUniformCount));
  put(DecodeTensor::ConstraintMasks,
      bytesFor<uint32_t>(uint64_t{ExecutionLimits::maximumStepTokens} *
                         geometry.maskWords()));
  put(DecodeTensor::OutputTokens, bytesFor<uint32_t>(r));
  put(DecodeTensor::RetainedCount, sizeof(uint32_t));
  put(DecodeTensor::NextAnchor, sizeof(uint32_t));
  put(DecodeTensor::AcceptedCount, sizeof(uint32_t));
  put(DecodeTensor::DraftInputTokens, bytesFor<uint32_t>(r));
  for (uint32_t index = 0; index < 2; ++index) {
    put(static_cast<DecodeTensor>(
            static_cast<uint32_t>(DecodeTensor::DraftHidden0) + index),
        bytesFor<uint16_t>(r * geometry.draft.hiddenSize));
  }
  put(DecodeTensor::DraftNormalized,
      bytesFor<uint16_t>(r * geometry.draft.hiddenSize));
  put(DecodeTensor::DraftDynamic,
      bytesFor<uint16_t>(r * geometry.draft.dynamicSize));
  put(DecodeTensor::DraftConvolved, draftWorkspace.convolutionBytes);
  put(DecodeTensor::DraftProposalQkv, draftWorkspace.qkvBytes);
  put(DecodeTensor::DraftAttention, draftWorkspace.groupedQueriesBytes);
  put(DecodeTensor::DraftProjected,
      bytesFor<uint16_t>(r * geometry.draft.hiddenSize));
  put(DecodeTensor::DraftResidual,
      bytesFor<uint16_t>(r * geometry.draft.hiddenSize));
  put(DecodeTensor::DraftIntermediate,
      bytesFor<uint16_t>(r * geometry.draft.intermediateSize));
  put(DecodeTensor::DraftFinalHidden,
      bytesFor<uint16_t>(r * geometry.draft.hiddenSize));
  put(DecodeTensor::SelectorHidden,
      bytesFor<uint16_t>(r * geometry.draft.selectorRank));
  put(DecodeTensor::Candidates, selectorWorkspace.candidatesBytes);
  put(DecodeTensor::Unary, selectorWorkspace.unaryBytes);
  put(DecodeTensor::TopPartialIds, selectorWorkspace.partialIdsBytes);
  put(DecodeTensor::TopPartialValues, selectorWorkspace.partialValuesBytes);
  put(DecodeTensor::ProposalProbs, selectorWorkspace.proposalProbabilitiesBytes);
  put(DecodeTensor::ProposedTokens, bytesFor<uint32_t>(kDraftProposalTokens));
  put(DecodeTensor::PageTable, bytesFor<uint32_t>(kMaximumPageTableEntries));
  put(DecodeTensor::VerifyPackedBase,
      uint64_t{geometry.target.stateLayout.layers} *
          gdnPackedStride(geometry));
  put(DecodeTensor::VerifyMixedBase,
      uint64_t{geometry.target.stateLayout.layers} *
          gdnMixedStride(geometry));
  put(DecodeTensor::VerifyDecayBase,
      uint64_t{geometry.target.stateLayout.layers} *
          gdnDecayStride(geometry));
  put(DecodeTensor::VerifyBetaBase,
      uint64_t{geometry.target.stateLayout.layers} *
          gdnBetaStride(geometry));
  put(DecodeTensor::ChunkKeysBase,
      uint64_t{geometry.target.kvLayout.attentionLayers} *
          decodeChunkLayerBytes(geometry));
  put(DecodeTensor::ChunkValuesBase,
      uint64_t{geometry.target.kvLayout.attentionLayers} *
          decodeChunkLayerBytes(geometry));
  if (geometry.target.ffnKind == QwenFfnKind::SparseMoe) {
    const ops::MoeWorkspace workspace =
        operators.moeDecodeWorkspacePerLane(geometry.target.moe);
    put(DecodeTensor::MoeSelectedExperts, workspace.selectedExpertsBytes);
    put(DecodeTensor::MoeRoutingWeights, workspace.routingWeightsBytes);
    put(DecodeTensor::MoeTileDescriptors, workspace.tileDescriptorsBytes);
    put(DecodeTensor::MoeTileCount, workspace.tileCountBytes);
    put(DecodeTensor::MoeGroupedRoutes, workspace.groupedRoutesBytes);
    put(DecodeTensor::MoeRouteRows, workspace.routeRowsBytes);
    put(DecodeTensor::MoeGroupedInput, workspace.groupedInputBytes);
    put(DecodeTensor::MoeExpertIntermediate, workspace.expertIntermediateBytes);
    put(DecodeTensor::MoeExpertOutput, workspace.expertOutputBytes);
  }
  return result;
}

uint64_t decodeArenaBaseBytes(const RuntimeGeometry &geometry,
                             const ops::ExecutionPlans &operators) {
  uint64_t bytes = 0;
  for (uint64_t value : decodeTensorBytes(geometry, operators)) {
    bytes = checkedAdd(
        bytes, alignArena(checkedMultiply(value, kLaneCount, "decode tensor")),
        "decode arena");
  }
  return bytes;
}

ops::LinearScratchSize DecodeArena::linearScratchSize(
    const RuntimeGeometry &geometry, const ops::ExecutionPlans &operators) {
  const auto &t = geometry.target;
  const auto &d = geometry.draft;
  ops::LinearScratchSize result;
  const auto include = [&](ops::LinearMatrix matrix, ops::QuantFamily quant) {
    if (!matrix.outputSize || !matrix.inputSize) return;
    for (uint32_t lanes = 1; lanes <= kLaneCount; ++lanes) {
      for (auto epilogue : {ops::LinearEpilogue::None, ops::LinearEpilogue::Residual,
                            ops::LinearEpilogue::GateUp}) {
        const auto size = operators.linear().decodeScratchSize(
            {matrix, lanes * kDecodeRows, ops::LinearPhase::Decode, epilogue, quant});
        result.input = std::max(result.input, size.input);
        result.sums = std::max(result.sums, size.sums);
        result.partials = std::max(result.partials, size.partials);
        result.counters = std::max(result.counters, size.counters);
      }
    }
  };
  for (auto matrix : {ops::LinearMatrix{t.packedGdnWidth, t.hiddenSize},
       {t.packedAttentionWidth, t.hiddenSize}, {t.hiddenSize, t.attentionWidth},
       {t.denseIntermediateSize, t.hiddenSize}, {t.hiddenSize, t.denseIntermediateSize},
       {t.vocabularySize, t.hiddenSize}})
    include(matrix, t.quant);
  // The draft shares the target's vocabulary head.
  include({d.vocabularySize, d.hiddenSize}, t.quant);
  for (auto matrix : {ops::LinearMatrix{d.dynamicSize, d.hiddenSize},
       {d.qkvSize, d.hiddenSize}, {d.hiddenSize, d.attentionSize},
       {d.intermediateSize, d.hiddenSize}, {d.hiddenSize, d.intermediateSize},
       {d.selectorRank, d.hiddenSize}, {d.hiddenSize, d.targetHiddenSize}})
    include(matrix, ops::QuantFamily::Affine);
  return result;
}

uint64_t plannedDecodeBytes(const RuntimeGeometry &geometry,
                           const ops::ExecutionPlans &operators) {
  return checkedAdd(decodeArenaBaseBytes(geometry, operators),
                    checkedAdd(DecodeArena::gateScratchBytes(geometry, operators),
                               DecodeArena::linearScratchSize(geometry, operators).bytes(),
                               "Q4 decode scratch"),
                    "planned gate scratch");
}

} // namespace splash::model

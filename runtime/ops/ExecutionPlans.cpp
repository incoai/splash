#include "ops/ExecutionPlans.hpp"

#include "ops/ChoiceTable.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace splash::ops {
namespace {

constexpr uint32_t kMaximumLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kDecodeRows = SPLASH_TARGET_VERIFY_ROWS;
static_assert(kMaximumLanes == 4);

AttentionShape attentionShape(uint32_t queryHeads, kv::Layout layout) {
  return {queryHeads, layout.kvHeads, layout.headDimension, layout.format};
}
kv::Layout attentionLayout(AttentionShape shape) {
  // Layer count affects the persistent cache, not one layer's execution key.
  return {1, shape.kvHeads, shape.headDimension, shape.format};
}

void validateHistory(uint32_t history, uint32_t rows) {
  if (uint64_t{history} + rows > kv::kMaximumPhysicalTokens)
    throw std::invalid_argument("attention choice history exceeds context");
}
VerifyAttentionPolicy verifyKey(uint32_t lanes, uint32_t queryHeads,
                                 kv::Layout layout,
                                 std::span<const uint32_t> histories) {
  if (!lanes || lanes > kMaximumLanes ||
      (histories.size() != lanes && histories.size() != kMaximumLanes))
    throw std::invalid_argument("invalid verify attention history vector");
  for (uint32_t lane = 0; lane < lanes; ++lane)
    validateHistory(histories[lane], kDecodeRows);
  return {attentionShape(queryHeads, layout), lanes};
}

constexpr std::array attentionFields{
    &AttentionWorkspace::partialsBytes, &AttentionWorkspace::statisticsBytes};
constexpr std::array draftFields{
    &DraftAttentionWorkspace::convolutionBytes,
    &DraftAttentionWorkspace::qkvBytes,
    &DraftAttentionWorkspace::groupedQueriesBytes,
    &DraftAttentionWorkspace::queryKeysBytes,
    &DraftAttentionWorkspace::queryValuesBytes};

// The shipped configuration of a MoE workload's phase, the first candidate.
MoeConfig baselineMoeConfig(MoePhase phase) noexcept {
  return (phase == MoePhase::Prefill ? MoE::prefillCandidates() : MoE::decodeCandidates()).front();
}

// The operator plan of `config` for a MoE workload's phase and physical rows.
MoePlan phasePlan(const MoeWorkload &workload, const MoeConfig &config) {
  if (workload.phase == MoePhase::Prefill) return MoE::prefillPlan(workload.shape, workload.rows, config);
  if (workload.phase != MoePhase::Decode || !workload.rows || workload.rows % kDecodeRows)
    throw std::invalid_argument("invalid MoE phase or physical rows");
  return MoE::decodePlan(workload.shape, workload.rows / kDecodeRows, config);
}

template <typename Workspace, size_t N>
void include(Workspace &bound, const Workspace &required,
             const std::array<uint64_t Workspace::*, N> &fields,
             uint32_t lanes = 1) {
  for (auto field : fields) {
    const uint64_t bytes = required.*field;
    bound.*field = std::max(bound.*field,
                           bytes / lanes + uint64_t{bytes % lanes != 0});
  }
}

} // namespace

ExecutionPlans::ExecutionPlans(const DeviceCapabilities &device)
    : linear_(device), baselineLinear_(device),
      moeRouteWideRows_(moeRouteWideRows(device.gpuCoreCount)),
      moeDecodeSimdgroups_(moeDecodeSimdgroups(device.appleGpuFamily)),
      appleGpuFamily_(device.appleGpuFamily) {}

void ExecutionPlans::install(const OperatorChoices &choices) {
  OperatorChoices pending = choices;
  Linear nextLinear = baselineLinear_;
  nextLinear.setChoices(pending.linear);
  for (const auto &choice : pending.prefillAttention) {
    const auto &w = choice.workload;
    (void)PagedAttention::prefillPlan(w.rows, w.shape.queryHeads,
                                    attentionLayout(w.shape), 0,
                                    choice.configuration);
  }
  for (const auto &choice : pending.verifyAttention) {
    const auto &w = choice.workload;
    const std::array<uint32_t, kMaximumLanes> histories{};
    (void)PagedAttention::verifyPlan(w.lanes, w.shape.queryHeads,
                                   attentionLayout(w.shape), histories,
                                   choice.configuration);
  }
  for (const auto &choice : pending.draftAttention)
    (void)DraftAttention::plan(choice.workload.shape, choice.workload.lanes,
                               choice.configuration);
  for (const auto &choice : pending.moe) {
    const auto &w = choice.workload;
    if (w.shape.weightLayout == WeightLayout::Block32)
      throw std::invalid_argument("block MoE plans are not tuned");
    // The choice's own configuration: an invalid device field fails install
    // although moePlan replaces it.
    (void)phasePlan(w, choice.configuration);
  }
  sortUniqueChoices(pending.prefillAttention);
  sortUniqueChoices(pending.verifyAttention);
  sortUniqueChoices(pending.draftAttention);
  sortUniqueChoices(pending.moe);
  // All potentially throwing work is above. No partial table install can
  // affect a production lookup if validation or allocation fails.
  std::swap(linear_, nextLinear);
  std::swap(choices_, pending);
}

PrefillAttentionPlan ExecutionPlans::prefillAttention(
    uint32_t rows, uint32_t queryHeads, kv::Layout layout,
    uint32_t historyTokens) const {
  validateHistory(historyTokens, rows);
  const PrefillAttentionPolicy workload{attentionShape(queryHeads, layout), rows};
  return PagedAttention::prefillPlan(
      rows, queryHeads, layout, historyTokens,
      chosenConfiguration(choices_.prefillAttention, workload,
                       PrefillAttentionConfig{}));
}

VerifyAttentionPlan ExecutionPlans::verifyAttention(
    uint32_t lanes, uint32_t queryHeads, kv::Layout layout,
    std::span<const uint32_t> historyTokens) const {
  const auto workload = verifyKey(lanes, queryHeads, layout, historyTokens);
  return PagedAttention::verifyPlan(
      lanes, queryHeads, layout, historyTokens,
      chosenConfiguration(choices_.verifyAttention, workload, VerifyAttentionConfig{}));
}

DraftAttentionPlan ExecutionPlans::draftAttention(DraftAttentionShape shape,
                                                 uint32_t lanes) const {
  return DraftAttention::plan(
      shape, lanes,
      chosenConfiguration(choices_.draftAttention,
                       DraftAttentionWorkload{shape, lanes},
                       DraftAttentionConfiguration{}));
}

// The device's fields of a MoE plan: the router threshold, the 8-row tile
// simdgroups and, for a GGUF plan, which is not tuned, its tiles.
MoePlan ExecutionPlans::moePlan(const MoeWorkload &workload, MoeConfig config) const {
  const MoeShape shape = workload.shape;
  const bool prefill = workload.phase == MoePhase::Prefill;
  config.routeWideRows = moeRouteWideRows_;
  // The four-simdgroup 8-row tiles are measured at decode occupancy only; a
  // prefill chunk's much larger expert grid keeps the shipped tile.
  config.m8Simdgroups = prefill ? MoeExpertSimdgroups::Eight : moeDecodeSimdgroups_;
  if (shape.weightLayout == WeightLayout::Block32) {
    const MoeGgufTile tile = moeGgufTile(appleGpuFamily_, shape);
    config.expertTile = prefill ? moeGgufPrefillTile(shape, workload.rows, tile) : MoeExpertTile::M8;
    config.ggufTile = tile;
    config.ggufRouterTile = linear_.ggufFloatTile(workload.rows, shape.experts);
  }
  return phasePlan(workload, config);
}

MoePlan ExecutionPlans::moePrefill(MoeShape shape, uint32_t rows) const {
  const MoeWorkload workload{shape, rows, MoePhase::Prefill};
  return moePlan(workload, chosenConfiguration(choices_.moe, workload, baselineMoeConfig(MoePhase::Prefill)));
}

MoePlan ExecutionPlans::moeDecode(MoeShape shape, uint32_t lanes) const {
  // Validate before multiplying an untrusted width into a physical-row key.
  if (!lanes || lanes > kMaximumLanes)
    throw std::invalid_argument("invalid MoE decode width");
  const MoeWorkload workload{shape, lanes * kDecodeRows, MoePhase::Decode};
  return moePlan(workload, chosenConfiguration(choices_.moe, workload, baselineMoeConfig(MoePhase::Decode)));
}

std::array<MoePlan, 2> ExecutionPlans::moeCandidates(const MoeWorkload &workload) const {
  // GGUF plans are not tuned: both candidates are the device's plan.
  if (workload.shape.weightLayout == WeightLayout::Block32) {
    const MoePlan plan = moePlan(workload, {});
    return {plan, plan};
  }
  const std::array<MoeConfig, 2> configs =
      workload.phase == MoePhase::Prefill ? MoE::prefillCandidates() : MoE::decodeCandidates();
  return {moePlan(workload, configs[0]), moePlan(workload, configs[1])};
}

AttentionWorkspace ExecutionPlans::prefillAttentionWorkspace(
    uint32_t maximumRows, uint32_t queryHeads, kv::Layout layout) const {
  auto bound = PagedAttention::prefillWorkspace(maximumRows, queryHeads, layout);
  const auto shape = attentionShape(queryHeads, layout);
  for (const auto &choice : choices_.prefillAttention) {
    const auto &w = choice.workload;
    if (w.shape == shape && w.rows <= maximumRows)
      include(bound, PagedAttention::prefillWorkspace(
                         w.rows, queryHeads, layout, choice.configuration),
              attentionFields);
  }
  return bound;
}

AttentionWorkspace ExecutionPlans::verifyAttentionWorkspacePerLane(
    uint32_t queryHeads, kv::Layout layout) const {
  AttentionWorkspace bound;
  for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes)
    include(bound, PagedAttention::verifyWorkspace(lanes, queryHeads, layout),
            attentionFields, lanes);
  const auto shape = attentionShape(queryHeads, layout);
  for (const auto &choice : choices_.verifyAttention) {
    const auto &w = choice.workload;
    if (w.shape == shape)
      include(bound, PagedAttention::verifyWorkspace(
                         w.lanes, queryHeads, layout, choice.configuration),
              attentionFields, w.lanes);
  }
  return bound;
}

DraftAttentionWorkspace ExecutionPlans::draftAttentionWorkspacePerLane(
    DraftAttentionShape shape) const {
  DraftAttentionWorkspace bound;
  for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes)
    include(bound, DraftAttention::plan(shape, lanes).workspace(), draftFields,
            lanes);
  for (const auto &choice : choices_.draftAttention) {
    const auto &w = choice.workload;
    if (w.shape == shape)
      include(bound, DraftAttention::plan(shape, w.lanes,
                                          choice.configuration).workspace(),
              draftFields, w.lanes);
  }
  return bound;
}

MoeWorkspace ExecutionPlans::moePrefillWorkspace(MoeShape shape,
                                               uint32_t maximumRows) const {
  // Validate the bound before iterating; every row is included even if a
  // future grouped layout's largest field is not monotone in row count.
  auto bound = moePrefill(shape, maximumRows).workspace();
  for (uint32_t rows = 1; rows <= maximumRows; ++rows) {
    include(bound, moePlan({shape, rows, MoePhase::Prefill}, baselineMoeConfig(MoePhase::Prefill)).workspace(),
            kMoeWorkspaceFields);
    include(bound, moePrefill(shape, rows).workspace(), kMoeWorkspaceFields);
  }
  return bound;
}

MoeWorkspace ExecutionPlans::moeDecodeWorkspacePerLane(MoeShape shape) const {
  MoeWorkspace bound;
  for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes) {
    const MoeWorkload workload{shape, lanes * kDecodeRows, MoePhase::Decode};
    include(bound, moePlan(workload, baselineMoeConfig(MoePhase::Decode)).workspace(), kMoeWorkspaceFields, lanes);
    include(bound, moeDecode(shape, lanes).workspace(), kMoeWorkspaceFields, lanes);
  }
  return bound;
}

uint64_t ExecutionPlans::gateUpWorkspace(ProjectionShape shape) const {
  uint64_t bound = 0;
  for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes) {
    const LinearWorkload workload{{shape.outputSize, shape.inputSize}, lanes * kDecodeRows,
                                  LinearPhase::Decode, LinearEpilogue::GateUp, shape.layout};
    bound = std::max({bound, baselineLinear_.plan(workload).gateScratchBytes(),
                      linear_.plan(workload).gateScratchBytes()});
  }
  return bound;
}

} // namespace splash::ops

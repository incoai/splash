#include "ops/ExecutionPlans.hpp"

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace splash::ops {
namespace {

constexpr uint32_t kMaximumLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kDecodeRows = SPLASH_TARGET_VERIFY_ROWS;
static_assert(kMaximumLanes == 4);

auto shapeKey(DraftAttentionShape s) noexcept {
  return std::tuple{s.hiddenSize, s.dynamicSize, s.qkvSize, s.attentionSize,
                    s.queryHeads, s.kvHeads, s.headDimension};
}
auto shapeKey(MoeShape s) noexcept {
  return std::tuple{s.hiddenSize, s.experts, s.expertsPerToken,
                    s.expertIntermediateSize};
}
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

template <typename Choice>
void sortUnique(std::vector<Choice> &choices) {
  std::sort(choices.begin(), choices.end(), [](const auto &a, const auto &b) {
    return a.workload < b.workload;
  });
  if (std::adjacent_find(choices.begin(), choices.end(),
                        [](const auto &a, const auto &b) {
                          return a.workload == b.workload;
                        }) != choices.end())
    throw std::invalid_argument("duplicate operator choice");
}
template <typename Choice, typename Workload, typename Configuration>
Configuration configurationFor(const std::vector<Choice> &choices,
                               const Workload &workload,
                               Configuration baseline) {
  const auto found = std::lower_bound(
      choices.begin(), choices.end(), workload,
      [](const auto &choice, const auto &key) { return choice.workload < key; });
  return found != choices.end() && found->workload == workload
             ? found->configuration
             : baseline;
}

constexpr std::array attentionFields{
    &AttentionWorkspace::partialsBytes, &AttentionWorkspace::statisticsBytes};
constexpr std::array draftFields{
    &DraftAttentionWorkspace::convolutionBytes,
    &DraftAttentionWorkspace::qkvBytes,
    &DraftAttentionWorkspace::groupedQueriesBytes,
    &DraftAttentionWorkspace::queryKeysBytes,
    &DraftAttentionWorkspace::queryValuesBytes};
constexpr std::array moeFields{
    &MoeWorkspace::selectedExpertsBytes, &MoeWorkspace::routingWeightsBytes,
    &MoeWorkspace::tileDescriptorsBytes, &MoeWorkspace::tileCountBytes,
    &MoeWorkspace::groupedRoutesBytes, &MoeWorkspace::routeRowsBytes,
    &MoeWorkspace::groupedInputBytes, &MoeWorkspace::expertIntermediateBytes,
    &MoeWorkspace::expertOutputBytes};

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

std::strong_ordering DraftAttentionWorkload::operator<=>(
    const DraftAttentionWorkload &other) const noexcept {
  return std::tuple{shapeKey(shape), lanes} <=>
         std::tuple{shapeKey(other.shape), other.lanes};
}
bool DraftAttentionWorkload::operator==(
    const DraftAttentionWorkload &other) const noexcept {
  return (*this <=> other) == 0;
}
std::strong_ordering MoeWorkload::operator<=>(
    const MoeWorkload &other) const noexcept {
  return std::tuple{shapeKey(shape), rows, phase} <=>
         std::tuple{shapeKey(other.shape), other.rows, other.phase};
}
bool MoeWorkload::operator==(const MoeWorkload &other) const noexcept {
  return (*this <=> other) == 0;
}

ExecutionPlans::ExecutionPlans(const DeviceCapabilities &device)
    : linear_(device), baselineLinear_(device),
      moeRouteWideRows_(moeRouteWideRows(device.gpuCoreCount)),
      moeDecodeSimdgroups_(moeDecodeSimdgroups(device.appleGpuFamily)) {}

void ExecutionPlans::install(const OperatorChoices &choices) {
  OperatorChoices pending = choices;
  Q4Linear nextLinear = baselineLinear_;
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
    if (w.phase == MoePhase::Prefill)
      (void)MoE::prefillPlan(w.shape, w.rows, choice.configuration);
    else if (w.phase == MoePhase::Decode && w.rows && w.rows % kDecodeRows == 0)
      (void)MoE::decodePlan(w.shape, w.rows / kDecodeRows, choice.configuration);
    else
      throw std::invalid_argument("invalid MoE choice phase or physical rows");
  }
  sortUnique(pending.prefillAttention);
  sortUnique(pending.verifyAttention);
  sortUnique(pending.draftAttention);
  sortUnique(pending.moe);
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
      configurationFor(choices_.prefillAttention, workload,
                       PrefillAttentionConfig{}));
}

VerifyAttentionPlan ExecutionPlans::verifyAttention(
    uint32_t lanes, uint32_t queryHeads, kv::Layout layout,
    std::span<const uint32_t> historyTokens) const {
  const auto workload = verifyKey(lanes, queryHeads, layout, historyTokens);
  return PagedAttention::verifyPlan(
      lanes, queryHeads, layout, historyTokens,
      configurationFor(choices_.verifyAttention, workload, VerifyAttentionConfig{}));
}

DraftAttentionPlan ExecutionPlans::draftAttention(DraftAttentionShape shape,
                                                 uint32_t lanes) const {
  return DraftAttention::plan(
      shape, lanes,
      configurationFor(choices_.draftAttention,
                       DraftAttentionWorkload{shape, lanes},
                       DraftAttentionConfiguration{}));
}

MoePlan ExecutionPlans::moePrefill(MoeShape shape, uint32_t rows) const {
  MoeConfig config =
      configurationFor(choices_.moe, MoeWorkload{shape, rows, MoePhase::Prefill},
                       MoeConfig{MoeExpertTile::M32});
  config.routeWideRows = moeRouteWideRows_;
  // The four-simdgroup 8-row tiles are measured at decode occupancy only; a
  // prefill chunk's much larger expert grid keeps the shipped tile.
  config.m8Simdgroups = MoeExpertSimdgroups::Eight;
  return MoE::prefillPlan(shape, rows, config);
}

MoePlan ExecutionPlans::moeDecode(MoeShape shape, uint32_t lanes) const {
  // Validate before multiplying an untrusted width into a physical-row key.
  if (!lanes || lanes > kMaximumLanes)
    throw std::invalid_argument("invalid MoE decode width");
  MoeConfig config = configurationFor(
      choices_.moe, MoeWorkload{shape, lanes * kDecodeRows, MoePhase::Decode},
      MoeConfig{MoeExpertTile::M8});
  config.routeWideRows = moeRouteWideRows_;
  config.m8Simdgroups = moeDecodeSimdgroups_;
  return MoE::decodePlan(shape, lanes, config);
}

std::array<MoePlan, 2> ExecutionPlans::moeCandidates(const MoeWorkload &workload) const {
  if (workload.phase == MoePhase::Prefill)
    return MoE::prefillCandidates(workload.shape, workload.rows, moeRouteWideRows_);
  if (workload.phase != MoePhase::Decode || workload.rows % kDecodeRows)
    throw std::invalid_argument("invalid MoE candidate workload");
  return MoE::decodeCandidates(workload.shape, workload.rows / kDecodeRows,
                               moeRouteWideRows_, moeDecodeSimdgroups_);
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
  auto bound = MoE::prefillPlan(shape, maximumRows).workspace();
  for (uint32_t rows = 1; rows < maximumRows; ++rows)
    include(bound, MoE::prefillPlan(shape, rows).workspace(), moeFields);
  for (const auto &choice : choices_.moe) {
    const auto &w = choice.workload;
    if (w.phase == MoePhase::Prefill && shapeKey(w.shape) == shapeKey(shape) &&
        w.rows <= maximumRows)
      include(bound, MoE::prefillPlan(shape, w.rows,
                                     choice.configuration).workspace(), moeFields);
  }
  return bound;
}

MoeWorkspace ExecutionPlans::moeDecodeWorkspacePerLane(MoeShape shape) const {
  MoeWorkspace bound;
  for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes)
    include(bound, MoE::decodePlan(shape, lanes).workspace(), moeFields, lanes);
  for (const auto &choice : choices_.moe) {
    const auto &w = choice.workload;
    if (w.phase == MoePhase::Decode && shapeKey(w.shape) == shapeKey(shape)) {
      const uint32_t lanes = w.rows / kDecodeRows;
      include(bound, MoE::decodePlan(shape, lanes,
                                    choice.configuration).workspace(), moeFields,
              lanes);
    }
  }
  return bound;
}

uint64_t ExecutionPlans::gateUpWorkspace(LinearMatrix matrix) const {
  uint64_t bound = 0;
  for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes) {
    const LinearWorkload workload{matrix, lanes * kDecodeRows,
                                  LinearPhase::Decode, LinearEpilogue::GateUp};
    bound = std::max({bound, baselineLinear_.plan(workload).gateScratchBytes(),
                      linear_.plan(workload).gateScratchBytes()});
  }
  return bound;
}

} // namespace splash::ops

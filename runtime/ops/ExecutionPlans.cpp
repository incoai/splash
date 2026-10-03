#include "ops/ExecutionPlans.hpp"

#include <algorithm>
#include <stdexcept>

namespace splash::ops {
namespace {

constexpr uint32_t kMaximumLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kDecodeRows = SPLASH_TARGET_VERIFY_ROWS;
static_assert(kMaximumLanes == 4);

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
    : linear_(device), moeRouteWideRows_(moeRouteWideRows(plannedGpuCores(device))),
      moeDecodeSimdgroups_(moeDecodeSimdgroups(device.appleGpuFamily)),
      appleGpuFamily_(device.appleGpuFamily) {}

PrefillAttentionPlan ExecutionPlans::prefillAttention(
    uint32_t rows, uint32_t queryHeads, kv::Layout layout) const {
  return PagedAttention::prefillPlan(rows, queryHeads, layout);
}

VerifyAttentionPlan ExecutionPlans::verifyAttention(
    uint32_t lanes, uint32_t queryHeads, kv::Layout layout,
    std::span<const uint32_t> historyTokens) const {
  return PagedAttention::verifyPlan(lanes, queryHeads, layout, historyTokens);
}

DraftAttentionPlan ExecutionPlans::draftAttention(DraftAttentionShape shape,
                                                 uint32_t lanes) const {
  return DraftAttention::plan(shape, lanes);
}

// The router threshold, the expert tile, the simdgroups of a decode plan's
// 8-row tiles and, for a GGUF plan, its tiles.
MoeConfig ExecutionPlans::moeConfig(MoeShape shape, uint32_t rows, MoePhase phase) const {
  const bool prefill = phase == MoePhase::Prefill;
  MoeConfig config;
  config.routeWideRows = moeRouteWideRows_;
  config.expertTile = prefill ? MoeExpertTile::M32 : MoeExpertTile::M8;
  if (!prefill) config.m8Simdgroups = moeDecodeSimdgroups_;
  if (shape.weightLayout == WeightLayout::Block32) {
    const MoeGgufTile tile = moeGgufTile(appleGpuFamily_, shape);
    if (prefill) config.expertTile = moeGgufPrefillTile(shape, rows, tile);
    config.ggufTile = tile;
    config.ggufRouterTile = linear_.ggufFloatTile(rows, shape.experts);
  }
  return config;
}

MoePlan ExecutionPlans::moePrefill(MoeShape shape, uint32_t rows) const {
  return MoE::prefillPlan(shape, rows, moeConfig(shape, rows, MoePhase::Prefill));
}

MoePlan ExecutionPlans::moeDecode(MoeShape shape, uint32_t lanes) const {
  // Validate before multiplying an untrusted width into the plan's rows.
  if (!lanes || lanes > kMaximumLanes)
    throw std::invalid_argument("invalid MoE decode width");
  return MoE::decodePlan(shape, lanes, moeConfig(shape, lanes * kDecodeRows, MoePhase::Decode));
}

AttentionWorkspace ExecutionPlans::prefillAttentionWorkspace(
    uint32_t maximumRows, uint32_t queryHeads, kv::Layout layout) const {
  return PagedAttention::prefillWorkspace(maximumRows, queryHeads, layout);
}

// The verify bound is linear in the lanes: one lane's is every width's share.
AttentionWorkspace ExecutionPlans::verifyAttentionWorkspacePerLane(
    uint32_t queryHeads, kv::Layout layout) const {
  return PagedAttention::verifyWorkspace(1, queryHeads, layout);
}

// The draft workspace is linear in the lanes: one lane's is every width's
// share.
DraftAttentionWorkspace ExecutionPlans::draftAttentionWorkspacePerLane(
    DraftAttentionShape shape) const {
  return DraftAttention::plan(shape, 1).workspace();
}

MoeWorkspace ExecutionPlans::moePrefillWorkspace(MoeShape shape,
                                               uint32_t maximumRows) const {
  // Validate the bound before iterating; every row is included even if a
  // future grouped layout's largest field is not monotone in row count.
  auto bound = moePrefill(shape, maximumRows).workspace();
  for (uint32_t rows = 1; rows <= maximumRows; ++rows)
    include(bound, moePrefill(shape, rows).workspace(), kMoeWorkspaceFields);
  return bound;
}

MoeWorkspace ExecutionPlans::moeDecodeWorkspacePerLane(MoeShape shape) const {
  MoeWorkspace bound;
  for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes)
    include(bound, moeDecode(shape, lanes).workspace(), kMoeWorkspaceFields, lanes);
  return bound;
}

uint64_t ExecutionPlans::gateUpWorkspace(ProjectionShape shape) const {
  uint64_t bound = 0;
  for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes) {
    const LinearWorkload workload{{shape.outputSize, shape.inputSize}, lanes * kDecodeRows,
                                  LinearPhase::Decode, LinearEpilogue::GateUp, shape.layout};
    bound = std::max(bound, linear_.plan(workload).gateScratchBytes());
  }
  return bound;
}

} // namespace splash::ops

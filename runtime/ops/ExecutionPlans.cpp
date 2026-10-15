#include "ops/ExecutionPlans.hpp"

#include <algorithm>
#include <stdexcept>

namespace splash::ops {
namespace {

constexpr uint32_t kRowBlock = SPLASH_ROW_BLOCK;
constexpr uint32_t kMaximumBlocks = SPLASH_MAXIMUM_CYCLE_ROWS / SPLASH_ROW_BLOCK;
static_assert(kMaximumBlocks == 4);

template <typename Workspace, size_t N>
void include(Workspace &bound, const Workspace &required,
             const std::array<uint64_t Workspace::*, N> &fields,
             uint32_t blocks = 1) {
  for (auto field : fields) {
    const uint64_t bytes = required.*field;
    bound.*field = std::max(bound.*field,
                           bytes / blocks + uint64_t{bytes % blocks != 0});
  }
}

} // namespace

ExecutionPlans::ExecutionPlans(const DeviceCapabilities &device)
    : linear_(device), family_(gpuFamilyClass(device.appleGpuFamily)) {}

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
                                                 uint32_t lanes, uint32_t rows) const {
  return DraftAttention::plan(shape, lanes, rows);
}

// The expert tiles and their rows, and the router's tile.
MoeConfig ExecutionPlans::moeConfig(MoeShape shape, uint32_t rows, MoePhase phase) const {
  MoeConfig config;
  config.ggufTile = moeGgufTile(family_, shape);
  config.expertTile =
      phase == MoePhase::Prefill ? moeGgufPrefillTile(shape, rows, config.ggufTile) : MoeExpertTile::M8;
  config.ggufRouterTile = linear_.ggufFloatTile(rows, shape.experts);
  return config;
}

MoePlan ExecutionPlans::moePrefill(MoeShape shape, uint32_t rows) const {
  return MoE::prefillPlan(shape, rows, moeConfig(shape, rows, MoePhase::Prefill));
}

MoePlan ExecutionPlans::moeDecode(MoeShape shape, uint32_t blocks) const {
  // Validate before multiplying an untrusted width into the plan's rows.
  if (!blocks || blocks > kMaximumBlocks)
    throw std::invalid_argument("invalid MoE decode width");
  return MoE::decodePlan(shape, blocks, moeConfig(shape, blocks * kRowBlock, MoePhase::Decode));
}

AttentionWorkspace ExecutionPlans::prefillAttentionWorkspace(
    uint32_t maximumRows, uint32_t queryHeads, kv::Layout layout) const {
  return PagedAttention::prefillWorkspace(maximumRows, queryHeads, layout);
}

// The verify bound is linear in the lanes, each a row block: one lane's is
// every width's share.
AttentionWorkspace ExecutionPlans::verifyAttentionWorkspacePerRowBlock(
    uint32_t queryHeads, kv::Layout layout) const {
  return PagedAttention::verifyWorkspace(1, queryHeads, layout);
}

// The draft workspace is linear in the rows: one lane of one row block's is
// every cycle's share.
DraftAttentionWorkspace ExecutionPlans::draftAttentionWorkspacePerRowBlock(
    DraftAttentionShape shape) const {
  return DraftAttention::plan(shape, 1, kRowBlock).workspace();
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

MoeWorkspace ExecutionPlans::moeDecodeWorkspacePerRowBlock(MoeShape shape) const {
  MoeWorkspace bound;
  for (uint32_t blocks = 1; blocks <= kMaximumBlocks; ++blocks)
    include(bound, moeDecode(shape, blocks).workspace(), kMoeWorkspaceFields, blocks);
  return bound;
}

uint64_t ExecutionPlans::gateUpWorkspace(ProjectionShape shape) const {
  uint64_t bound = 0;
  for (uint32_t blocks = 1; blocks <= kMaximumBlocks; ++blocks) {
    const LinearWorkload workload{{shape.outputSize, shape.inputSize}, blocks * kRowBlock,
                                  LinearPhase::Decode, LinearEpilogue::GateUp};
    bound = std::max(bound, linear_.plan(workload).gateScratchBytes());
  }
  return bound;
}

} // namespace splash::ops

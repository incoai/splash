#include "ops/ExecutionPlans.hpp"

#include "metal/abi/Gguf.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace splash::ops {
namespace {

constexpr uint32_t kMaximumLanes = SPLASH_MAXIMUM_BATCH_WIDTH;

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

ExecutionPlans::ExecutionPlans(const DeviceCapabilities &device, DecodeGeometry geometry,
                                 uint32_t maximumHistoryTokens)
    : ExecutionPlans(device, Int8Activations::Off, geometry, maximumHistoryTokens) {}
ExecutionPlans::ExecutionPlans(const DeviceCapabilities &device, Int8Activations int8Activations,
                                 DecodeGeometry geometry, uint32_t maximumHistoryTokens)
    : linear_(device, int8Activations, geometry), draftLinear_(device, Int8Activations::Off, geometry),
      maximumHistoryTokens_(maximumHistoryTokens),
      maximumVerifySplits_(maximumHistoryTokens ? kv::verifyAttentionSplits(maximumHistoryTokens, 0, geometry.rows())
                                                : kv::kVerifyMaximumSplits) {
  if (maximumHistoryTokens > SPLASH_MAXIMUM_CONTEXT_TOKENS)
    throw std::invalid_argument("verify history bound exceeds the logical context limit");
}

// A chunk's rows pick its projections' tile, and a staged tile's height names
// its kernels (gguf_decode_<format>_m<rows>_*), so each height a chunk takes
// is a class, and the prefill tile one more. Every other kernel a chunk's
// rows pick today changes once along them (selectFloatTile, selectMoe, the
// staged norm, the hyper-connection norm's quarters, and the Neural Engine
// split, which takes no class's first chunk: AneFfn::kMinimumRows), so a
// chunk of one row and one of the most take both sides. A class no warmup ran
// created its kernels while its request waited: 19 s for the 27B's cold
// 17-row chunk on a 16-core Apple9 plan.
std::vector<uint32_t> ExecutionPlans::prefillRowClasses() const {
  std::vector<uint32_t> first;
  std::pair<LinearTile, uint32_t> last{};
  for (uint32_t rows = 1; rows <= SPLASH_PREFILL_TOKEN_BUDGET; ++rows) {
    const LinearPlan plan = linear_.plan({{GGUF_TILE_COLUMNS, GGUF_TILE_COLUMNS}, rows, LinearPhase::Prefill});
    const LinearTile tile = plan.configuration().tile;
    const std::pair key{tile, tile == LinearTile::GgufStaged ? plan.storageRows() : 0};
    if (first.empty() || key != last) first.push_back(rows);
    last = key;
  }
  return first;
}

PrefillAttentionPlan ExecutionPlans::prefillAttention(
    uint32_t rows, uint32_t historyTokens, uint32_t queryHeads, kv::Layout layout,
    AttentionVariant variant) const {
  return PagedAttention::prefillPlan(rows, historyTokens, queryHeads, layout, calibration().attention,
                                     linear_.int8Activations(), variant);
}

VerifyAttentionPlan ExecutionPlans::verifyAttention(
    uint32_t lanes, uint32_t queryHeads, kv::Layout layout,
    std::span<const uint32_t> historyTokens, AttentionVariant variant) const {
  return PagedAttention::verifyPlan(lanes, queryHeads, layout, historyTokens, calibration().attention,
                                    linear_.int8Activations(), variant, geometry(), maximumVerifySplits_);
}

DraftAttentionPlan ExecutionPlans::draftAttention(DraftAttentionShape shape,
                                                 uint32_t lanes) const {
  shape.geometry = geometry();
  return DraftAttention::plan(shape, lanes);
}

MoePlan ExecutionPlans::moePrefill(MoeShape shape, uint32_t routerFormat, uint32_t rows) const {
  return MoE::prefillPlan(shape, rows, selectMoe(calibration(), shape, routerFormat, rows, MoePhase::Prefill));
}

MoePlan ExecutionPlans::moeDecode(MoeShape shape, uint32_t routerFormat, uint32_t lanes) const {
  if (geometry().rows() != 8) throw std::invalid_argument("sixteen-row MoE execution is unsupported");
  // Validate before multiplying an untrusted width into the plan's rows.
  if (!lanes || lanes > kMaximumLanes)
    throw std::invalid_argument("invalid MoE decode width");
  return MoE::decodePlan(shape, lanes,
                         selectMoe(calibration(), shape, routerFormat, lanes * geometry().rows(), MoePhase::Decode));
}

AttentionWorkspace ExecutionPlans::prefillAttentionWorkspace(
    uint32_t maximumRows, uint32_t queryHeads, kv::Layout layout) const {
  return PagedAttention::prefillWorkspace(maximumRows, queryHeads, layout);
}

// The verify bound is linear in the lanes: one lane's is every width's share.
AttentionWorkspace ExecutionPlans::verifyAttentionWorkspacePerLane(
    uint32_t queryHeads, kv::Layout layout) const {
  return PagedAttention::verifyWorkspace(1, queryHeads, layout, geometry(), maximumVerifySplits_);
}

// The draft workspace is linear in the lanes: one lane's is every width's
// share.
DraftAttentionWorkspace ExecutionPlans::draftAttentionWorkspacePerLane(
    DraftAttentionShape shape) const {
  shape.geometry = geometry();
  return DraftAttention::plan(shape, 1).workspace();
}

MoeWorkspace ExecutionPlans::moePrefillWorkspace(MoeShape shape, uint32_t routerFormat,
                                                 uint32_t maximumRows) const {
  // Validate the bound before iterating; every row is included even if a
  // future grouped layout's largest field is not monotone in row count.
  auto bound = moePrefill(shape, routerFormat, maximumRows).workspace();
  for (uint32_t rows = 1; rows <= maximumRows; ++rows)
    include(bound, moePrefill(shape, routerFormat, rows).workspace(), kMoeWorkspaceFields);
  return bound;
}

MoeWorkspace ExecutionPlans::moeDecodeWorkspacePerLane(MoeShape shape, uint32_t routerFormat) const {
  MoeWorkspace bound;
  for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes)
    include(bound, moeDecode(shape, routerFormat, lanes).workspace(), kMoeWorkspaceFields, lanes);
  return bound;
}

uint64_t ExecutionPlans::gateUpWorkspace(ProjectionShape shape) const {
  uint64_t bound = 0;
  for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes) {
    const LinearWorkload workload{{shape.outputSize, shape.inputSize}, lanes * geometry().rows(),
                                  LinearPhase::Decode, LinearEpilogue::GateUp, geometry()};
    bound = std::max(bound, linear_.plan(workload).gateScratchBytes());
  }
  return bound;
}

} // namespace splash::ops

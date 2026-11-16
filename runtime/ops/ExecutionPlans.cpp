#include "ops/ExecutionPlans.hpp"
#include "ops/KernelSelection.hpp"

#include "metal/abi/Gguf.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace splash::ops {
namespace {

constexpr uint32_t kMaximumLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kDecodeRows = SPLASH_TARGET_VERIFY_ROWS;

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

ExecutionPlans::ExecutionPlans(const DeviceCapabilities &device, Int8Activations int8Activations)
    : linear_(device, int8Activations), draftLinear_(device, Int8Activations::Off) {}

// A chunk's projection tile and row height name its kernels, so each height
// a chunk takes is a class, including both shared prefill row tiles.
// Every other kernel a chunk's
// rows pick today changes once along them (selectFloatTile, selectMoe, the
// staged norm, the hyper-connection norm's quarters, and the Neural Engine
// split, which takes no class's first chunk: AneFfn::kMinimumRows), so a
// chunk of one row and one of the most take both sides. A class no warmup ran
// created its kernels while its request waited: 19 s for the 27B's cold
// 17-row chunk on a 16-core Apple9 plan.
std::vector<uint32_t> ExecutionPlans::prefillRowClasses() const {
  std::vector<uint32_t> first;
  // Include both format capabilities: a low-grid prefill may use the small
  // row tile, then return to the full tile before ANE splitting begins.
  const std::array<UnitFormats, 2> units{{{}, {.prefill64Takes = true}}};
  std::array<std::pair<LinearTile, uint32_t>, units.size()> last{};
  for (uint32_t rows = 1; rows <= SPLASH_PREFILL_TOKEN_BUDGET; ++rows) {
    const LinearWorkload workload{{GGUF_TILE_COLUMNS, GGUF_TILE_COLUMNS}, rows, LinearPhase::Prefill};
    bool changed = first.empty();
    for (size_t index = 0; index < units.size(); ++index) {
      const auto config = selectProjection(calibration(), linear_.int8Activations(), workload, units[index]);
      const LinearPlan plan = Linear::plan(workload, config, FloatOutput::BFloat16);
      const uint32_t height = config.tile == LinearTile::GgufStaged ? plan.storageRows()
                              : config.prefill64Rows ? GGUF_PREFILL_ROWS / 2 : GGUF_PREFILL_ROWS;
      const std::pair key{config.tile, height};
      changed |= key != last[index];
      last[index] = key;
    }
    if (changed) first.push_back(rows);
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
                                    linear_.int8Activations(), variant);
}

DraftAttentionPlan ExecutionPlans::draftAttention(DraftAttentionShape shape,
                                                 uint32_t lanes,
                                                 uint32_t rows) const {
  return DraftAttention::plan(shape, lanes, rows);
}

MoePlan ExecutionPlans::moePrefill(MoeShape shape, uint32_t routerFormat, uint32_t rows) const {
  return MoE::prefillPlan(shape, rows, selectMoe(calibration(), shape, routerFormat, rows, MoePhase::Prefill));
}

MoePlan ExecutionPlans::moeDecode(MoeShape shape, uint32_t routerFormat, uint32_t lanes) const {
  // Validate before multiplying an untrusted width into the plan's rows.
  if (!lanes || lanes > kMaximumLanes)
    throw std::invalid_argument("invalid MoE decode width");
  return MoE::decodePlan(shape, lanes,
                         selectMoe(calibration(), shape, routerFormat, lanes * kDecodeRows, MoePhase::Decode));
}

AttentionWorkspace ExecutionPlans::prefillAttentionWorkspace(
    uint32_t maximumRows, uint32_t queryHeads, kv::Layout layout) const {
  return PagedAttention::prefillWorkspace(maximumRows, queryHeads, layout);
}

// Verify attention is linear in lanes. Wider native blocks use prefill
// tiles, so their workspace follows those tiles rather than an eight-row
// verify's split count.
AttentionWorkspace ExecutionPlans::verifyAttentionWorkspacePerLane(
    uint32_t queryHeads, kv::Layout layout, uint32_t rows) const {
  return rows == SPLASH_TARGET_VERIFY_ROWS ? PagedAttention::verifyWorkspace(1, queryHeads, layout)
                                          : PagedAttention::prefillWorkspace(rows, queryHeads, layout);
}

// The draft workspace is linear in the lanes: one lane's is every width's
// share.
DraftAttentionWorkspace ExecutionPlans::draftAttentionWorkspacePerLane(
    DraftAttentionShape shape, uint32_t rows) const {
  return DraftAttention::plan(shape, 1, rows).workspace();
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

uint64_t ExecutionPlans::gateUpWorkspace(ProjectionShape shape, uint32_t rowsPerLane) const {
  uint64_t bound = 0;
  for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes) {
    const LinearWorkload workload{{shape.outputSize, shape.inputSize}, lanes * kDecodeRows,
                                  LinearPhase::Decode, LinearEpilogue::GateUp};
    bound = std::max(bound, linear_.plan(workload).gateScratchBytes());
    // A served draft's whole block runs the head's gate/up pair too.
    const LinearWorkload block{{shape.outputSize, shape.inputSize},
                               lanes * rowsPerLane,
                               LinearPhase::Decode, LinearEpilogue::GateUp};
    bound = std::max(bound, linear_.plan(block).gateScratchBytes());
  }
  return bound;
}

} // namespace splash::ops

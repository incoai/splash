#include "ops/ExecutionPlans.hpp"

#include "metal/abi/Gguf.h"

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace splash::ops {

ExecutionPlans::ExecutionPlans(const DeviceCapabilities &device, Int8Activations int8Activations)
    : linear_(device, int8Activations), draftLinear_(device, Int8Activations::Off) {}

// A chunk's rows pick its projections' tile, and a tile's height names its
// kernels (gguf_decode_<format>_m<rows>_*, the register tile's _l<tiles>), so
// each tile and height a chunk takes is a class, and the prefill tile one
// more. Every other kernel a chunk's rows pick today changes once along them
// (selectFloatTile, selectMoe, the staged norm, the hyper-connection norm's
// quarters, and the Neural Engine split, which takes no class's first chunk:
// AneFfn::kMinimumRows), so a chunk of one row and one of the most take both
// sides. A class no warmup ran created its kernels while its request waited:
// 19 s for the 27B's cold 17-row chunk on a 16-core Apple9 plan.
std::vector<uint32_t> ExecutionPlans::prefillRowClasses() const {
  std::vector<uint32_t> first;
  const std::array<UnitFormats, 2> units{{{}, {.prefill64Takes = true}}};
  std::array<std::pair<LinearTile, uint32_t>, units.size()> last{};
  for (uint32_t rows = 1; rows <= SPLASH_PREFILL_TOKEN_BUDGET; ++rows) {
    const LinearWorkload workload{{GGUF_TILE_COLUMNS, GGUF_TILE_COLUMNS}, rows};
    bool changed = first.empty();
    for (size_t index = 0; index < units.size(); ++index) {
      const auto config = selectProjection(calibration(), linear_.int8Activations(), workload, units[index]);
      const LinearPlan plan = Linear::plan(workload, config, FloatOutput::BFloat16);
      const bool named = config.tile == LinearTile::GgufStaged || config.tile == LinearTile::GgufRegister;
      const std::pair key{config.tile, named ? plan.storageRows() : config.prefill64Rows ? GGUF_PREFILL_ROWS / 2 : 0};
      changed |= key != last[index];
      last[index] = key;
    }
    if (changed) first.push_back(rows);
  }
  return first;
}

// A step's rows pick its projections' tile and height as a chunk's do
// (prefillRowClasses); a float projection's tile, which a step's rows move
// only at the accelerator tile's fewest rows, as they fill one 64-row tile
// (selectFloatTile); and a MoE target's expert tile (32-row tiles once its
// rows' routes outnumber its experts). Its router's tile changes with them
// on no device.
std::vector<uint32_t> ExecutionPlans::decodeRowClasses(uint32_t mostRows,
                                                       std::optional<std::pair<MoeShape, uint32_t>> moe) const {
  if (mostRows > SPLASH_MAXIMUM_BATCH_WIDTH * SPLASH_TARGET_VERIFY_ROWS)
    throw std::invalid_argument("a decode step holds at most a verify block a lane");
  const FloatTileCalibration &floats = calibration().floatTile;
  std::vector<uint32_t> first;
  std::tuple<LinearTile, uint32_t, bool, uint32_t> last{};
  for (uint32_t rows = 1; rows <= mostRows; ++rows) {
    const LinearPlan plan = linear_.plan({{GGUF_TILE_COLUMNS, GGUF_TILE_COLUMNS}, rows});
    const LinearTile tile = plan.configuration().tile;
    const bool named = tile == LinearTile::GgufStaged || tile == LinearTile::GgufRegister;
    MoeConfig config{};
    if (moe) config = selectMoe(calibration(), moe->first, moe->second, rows);
    const std::tuple key{tile, named ? plan.storageRows() : 0, floats.neuralAccelerator && rows >= floats.minimumRows,
                         uint32_t(config.expertTile)};
    if (first.empty() || key != last) first.push_back(rows);
    last = key;
  }
  return first;
}

AttentionPlan ExecutionPlans::attention(std::span<const kv::ChunkedPrefillParams> sequences, uint32_t queryHeads,
                                        kv::Layout layout, AttentionWorkspace capacity,
                                        AttentionVariant variant) const {
  return PagedAttention::plan(sequences, queryHeads, layout, calibration().attention, linear_.int8Activations(),
                              capacity, variant);
}

DraftAttentionPlan ExecutionPlans::draftAttention(DraftAttentionShape shape,
                                                 uint32_t lanes, uint32_t rows) const {
  return DraftAttention::plan(shape, lanes, rows);
}

MoePlan ExecutionPlans::moe(MoeShape shape, uint32_t routerFormat, uint32_t rows) const {
  return MoE::plan(shape, rows, selectMoe(calibration(), shape, routerFormat, rows));
}

AttentionWorkspace ExecutionPlans::sequenceAttentionWorkspace(uint32_t queryHeads, kv::Layout layout) const {
  return PagedAttention::workspace(kAttentionSequenceSplits, queryHeads, layout);
}

AttentionWorkspace ExecutionPlans::laneAttentionWorkspace(uint32_t queryHeads, kv::Layout layout) const {
  return PagedAttention::workspace(kv::kAttentionMaximumSplits, queryHeads, layout);
}

// The draft workspace is linear in the lanes: one lane's is every width's
// share.
DraftAttentionWorkspace ExecutionPlans::draftAttentionWorkspacePerLane(
    DraftAttentionShape shape, uint32_t rows) const {
  return DraftAttention::plan(shape, 1, rows).workspace();
}

MoeWorkspace ExecutionPlans::moeWorkspace(MoeShape shape, uint32_t routerFormat, uint32_t maximumRows) const {
  // Validate the bound before iterating; every row is included even if a
  // future grouped layout's largest field is not monotone in row count.
  MoeWorkspace bound = moe(shape, routerFormat, maximumRows).workspace();
  for (uint32_t rows = 1; rows <= maximumRows; ++rows) {
    const MoeWorkspace required = moe(shape, routerFormat, rows).workspace();
    for (const auto field : kMoeWorkspaceFields) bound.*field = std::max(bound.*field, required.*field);
  }
  return bound;
}

uint64_t ExecutionPlans::gateUpWorkspace(ProjectionShape shape, uint32_t maximumRows) const {
  uint64_t bound = 0;
  for (uint32_t rows = 1; rows <= maximumRows; ++rows) {
    const LinearWorkload workload{{shape.outputSize, shape.inputSize}, rows, LinearEpilogue::GateUp};
    bound = std::max(bound, linear_.plan(workload).gateScratchBytes());
  }
  return bound;
}

} // namespace splash::ops

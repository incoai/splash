#include "ops/MoE.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "metal/abi/MoE.h"
#include "ops/BufferExtent.hpp"

#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace splash::ops {
namespace {

// A GGUF tensor of output x input weights, and the planes of its GGUF_FMT_*
// format; what reads a float tensor checks its fp32 values.
void requireSegment(const QuantizedSegment &segment, uint32_t output, uint32_t input, bool floatWeights,
                    const std::string &name) {
  if (segment.isFloat() != floatWeights || (!floatWeights && segment.formatId >= GGUF_FMT_COUNT) ||
      segment.outputSize != output || segment.inputSize != input)
    throw std::invalid_argument(name + " does not match the plan");
  if (!floatWeights) requireSegmentPlanes(segment, name);
}

// Every weight the plan's dispatches read. The router runs through
// addGgufFloat, which checks its weights.
void validate(const BlockMoeWeights &weights, MoeShape shape) {
  const uint32_t hidden = shape.hiddenSize, intermediate = shape.expertIntermediateSize;
  requireSegment(weights.router, shape.experts, hidden, true, "MoE router");
  requireSegment(weights.sharedScalarGate, 1, hidden, true, "MoE shared scalar gate");
  // The select kernel reads the scalar gate's fp32 weights.
  requireBytes(weights.sharedScalarGate.plane0, uint64_t{hidden} * sizeof(float), "MoE shared scalar gate weight");
  for (const auto &[projection, name, output, input] :
       {std::tuple{&weights.gate, "gate", intermediate, hidden}, std::tuple{&weights.up, "up", intermediate, hidden},
        std::tuple{&weights.down, "down", hidden, intermediate}}) {
    requireSegment(projection->routed, shape.experts * output, input, false, std::string("MoE expert ") + name);
    requireSegment(projection->shared, output, input, false, std::string("MoE shared ") + name);
  }
}

MoeWorkspace workspaceFor(MoeShape shape, uint32_t rows, uint32_t tileRows, MoeGgufTile ggufTile) {
  if (!shape.valid())
    throw std::invalid_argument("invalid MoE workspace shape");
  const uint64_t routes = uint64_t{rows} * shape.routesPerToken();
  const uint32_t tiles = moeMaximumTiles(rows, shape, tileRows);
  const uint64_t groupedRows = uint64_t{tiles} * tileRows;
  const uint32_t widest = std::max(shape.hiddenSize, shape.expertIntermediateSize);
  // The router's fp32 scores, a row of expert slots per row, live in the
  // grouped input until the gather overwrites them. Register plans also hold
  // the down pass's Table16 tiles there.
  const uint64_t scoreBytes = uint64_t{rows} * SPLASH_MOE_EXPERT_SLOTS * sizeof(float);
  const bool table16 = ggufTile == MoeGgufTile::Register;
  const uint64_t sumsBytes = table16 ? tableSumsBytes(LinearInput::Table16, widest, groupedRows) : 0;
  return {routes * sizeof(uint32_t), routes * sizeof(float),
          uint64_t{tiles} * sizeof(MoeTileDescriptor), sizeof(MoeTileCount),
          groupedRows * sizeof(uint32_t), routes * sizeof(uint32_t),
          std::max(tableBytes(table16 ? widest : shape.hiddenSize, groupedRows), scoreBytes),
          groupedRows * shape.expertIntermediateSize * sizeof(uint16_t),
          groupedRows * widest * sizeof(uint16_t), sumsBytes};
}

// Whether a plan runs gate and up in one pass
// (moe_expert_gguf_m<rows>_gate_up): on the staged tile, where gate and up
// share their routed format and their shared format. The register tile runs
// two passes.
bool oneGateUpPass(const BlockMoeWeights &weights, const MoePlan &plan) noexcept {
  return plan.configuration().ggufTile == MoeGgufTile::Staged &&
         weights.gate.routed.formatId == weights.up.routed.formatId &&
         weights.gate.shared.formatId == weights.up.shared.formatId;
}

// The expert passes over grouped tiles, each on the grid of the live
// tiles that the grouping wrote (MoeTileCount), within the column tiles of
// `group` and maximumTiles(): gate and up in one pass into
// expertIntermediate (oneGateUpPass), or gate into expertOutput (the down
// pass overwrites it after the up pass consumed it) and up with silu(gate)
// into expertIntermediate; then down into expertOutput. Register prefill
// plans read Table16 tiles from groupedInput: the gather writes the gate/up
// input's and a prepare dispatch the down input's. Register decode plans run
// the live-row passes (MoePlan::liveRows) over the gathered bf16 rows and the
// up pass's output.
void addExperts(metal::CommandGraph &graph, const MoeScratch &scratch, const BlockMoeWeights &weights,
                const MoePlan &plan, const MoeGroupParams &group) {
  const MoeShape shape = plan.shape();
  const uint32_t tiles = plan.maximumTiles();
  // Decode on the register tile runs the live-row passes over bf16 rows;
  // prefill on it the register tile over Table16 tiles.
  const bool liveRows = plan.liveRows();
  const bool table16 = plan.configuration().ggufTile == MoeGgufTile::Register && !liveRows;
  const uint32_t threads =
      liveRows ? GGUF_EXPERT_ROWS_THREADS : table16 ? GGUF_REGISTER_THREADS : GGUF_STAGED_THREADS;
  const std::string kernel = liveRows  ? "moe_expert_gguf_rows"
                             : table16 ? "moe_expert_gguf_sg"
                                       : "moe_expert_gguf_m" + std::to_string(plan.tileRows());
  const metal::IndirectGrid gateUpGrid{scratch.tileCount, offsetof(MoeTileCount, gate_up_grid)};
  const metal::IndirectGrid downGrid{scratch.tileCount, offsetof(MoeTileCount, down_grid)};
  const auto pass = [&](const BlockExpertProjection &projection, bool up,
                        const metal::MetalBuffer &input,
                        const metal::MetalBuffer &output, uint32_t n, uint32_t k,
                        uint32_t columns, const metal::IndirectGrid &grid) {
    std::vector<metal::MetalBuffer> bindings{input};
    if (table16) bindings.push_back(scratch.groupedSums);
    bindings.insert(bindings.end(),
                    {scratch.tileDescriptors, scratch.tileCount,
                     projection.routed.plane0, projection.routed.plane1Slot(),
                     projection.routed.meta, projection.shared.plane0,
                     projection.shared.plane1Slot(), projection.shared.meta, output,
                     scratch.expertOutput});
    graph.add(kernel + (up ? "_g" : "_a"), std::move(bindings),
              MoeGgufExpertParams{k, n, shape.experts, projection.routed.formatId,
                                  projection.shared.formatId},
              {columns, tiles, 1}, grid, {threads, 1, 1});
  };
  const uint32_t hidden = shape.hiddenSize;
  const uint32_t intermediate = shape.expertIntermediateSize;
  if (oneGateUpPass(weights, plan)) {
    const BlockExpertProjection &gate = weights.gate, &up = weights.up;
    graph.add(kernel + "_gate_up",
              {scratch.groupedInput, scratch.tileDescriptors, scratch.tileCount,
               gate.routed.plane0, gate.routed.plane1Slot(), gate.routed.meta,
               gate.shared.plane0, gate.shared.plane1Slot(), gate.shared.meta,
               up.routed.plane0, up.routed.plane1Slot(), up.routed.meta,
               up.shared.plane0, up.shared.plane1Slot(), up.shared.meta,
               scratch.expertIntermediate},
              MoeGgufExpertParams{hidden, intermediate, shape.experts, gate.routed.formatId,
                                  gate.shared.formatId},
              {group.gate_up_columns, tiles, 1}, gateUpGrid, {threads, 1, 1});
  } else {
    pass(weights.gate, false, scratch.groupedInput, scratch.expertOutput,
         intermediate, hidden, group.gate_up_columns, gateUpGrid);
    pass(weights.up, true, scratch.groupedInput, scratch.expertIntermediate,
         intermediate, hidden, group.gate_up_columns, gateUpGrid);
  }
  if (table16)
    graph.add("moe_prepare_table16",
              {scratch.expertIntermediate, scratch.tileCount,
               scratch.groupedInput, scratch.groupedSums},
              intermediate, {tiles, intermediate / 256, 1});
  pass(weights.down, false,
       table16 ? scratch.groupedInput : scratch.expertIntermediate,
       scratch.expertOutput, hidden, intermediate, group.down_columns, downGrid);
}

} // namespace

// Plans run 8-row tiles in decode and 8- or 32-row tiles in prefill; the
// register tile runs 8-row tiles.
MoePlan::MoePlan(MoeShape shape, uint32_t rows, MoeConfig config,
                 MoePhase phase)
    : shape_(shape), rows_(rows), config_(config), phase_(phase) {
  if (phase == MoePhase::Decode && config.expertTile != MoeExpertTile::M8)
    throw std::invalid_argument("invalid MoE expert tile configuration");
  if (config.ggufTile == MoeGgufTile::Register && config.expertTile != MoeExpertTile::M8)
    throw std::invalid_argument("the register expert tile takes 8-row tiles");
  workspace_ = workspaceFor(shape, rows, tileRows(), config.ggufTile);
  maximumTiles_ = moeMaximumTiles(rows, shape, tileRows());
}

void MoE::add(metal::CommandGraph &graph, const MoeBuffers &buffers,
              const BlockMoeWeights &weights, const MoePlan &plan) {
  const MoeShape shape = plan.shape();
  const uint32_t rows = plan.rows();
  const uint32_t tileRows = plan.tileRows();
  validate(weights, shape);
  const uint32_t tiles = plan.maximumTiles();
  const MoeWorkspace &required = plan.workspace();
  const uint64_t hiddenRows = uint64_t{rows} * shape.hiddenSize * sizeof(uint16_t);
  requireBytes(buffers.input, hiddenRows, "MoE input");
  requireBytes(buffers.residual, hiddenRows, "MoE residual");
  requireBytes(buffers.output, hiddenRows, "MoE output");
  const MoeScratch &scratch = buffers.scratch;
  for (const MoeScratchField &field : kMoeScratchFields)
    requireBytes(scratch.*field.buffer, required.*field.bytes, field.name);
  const MoeRouteParams routeParams{rows, shape.hiddenSize, shape.experts,
                                   shape.expertsPerToken};
  // fp32 scores of the F32 router in rows of expert slots, as the select
  // kernel reads.
  addGgufFloat(graph, buffers.input, weights.router, scratch.groupedInput, rows,
               SPLASH_MOE_EXPERT_SLOTS, 0, FloatOutput::Float32, plan.configuration().ggufRouterTile);
  graph.add("moe_route_select_f32",
            {scratch.groupedInput, buffers.input,
             weights.sharedScalarGate.plane0, scratch.selectedExperts,
             scratch.routingWeights},
            routeParams, {rows, 1, 1}, {SPLASH_MOE_EXPERT_SLOTS, 1, 1});
  // The expert passes launch the live tiles alone, on grids the grouping
  // writes with the tile count. Over every tile a step could fill, a 35B
  // decode step of one lane launched 65 tiles' threadgroups for a median of
  // 30 live ones; the rest return at once yet cost a 40-core M5 Max 0.8-4.1%
  // of a decode cycle, a 12-core M6 0.1-1.0% and a 40-core M3 Max up to 4.8%.
  const bool oneGateUp = oneGateUpPass(weights, plan);
  const MoeGroupParams group{rows, shape.expertsPerToken, tileRows, shape.experts,
                             shape.expertIntermediateSize / (oneGateUp ? GGUF_STAGED_COLUMNS : GGUF_TILE_COLUMNS),
                             shape.hiddenSize / GGUF_TILE_COLUMNS};
  graph.add("moe_group_routes",
            {scratch.selectedExperts, scratch.tileDescriptors,
             scratch.tileCount, scratch.groupedRoutes, scratch.routeRows},
            group, {1, 1, 1}, {SPLASH_MOE_EXPERT_SLOTS, 1, 1});
  const MoeGatherParams gather{tileRows, shape.hiddenSize, shape.routesPerToken()};
  if (plan.configuration().ggufTile == MoeGgufTile::Register && !plan.liveRows())
    graph.add("moe_gather_table16",
              {buffers.input, scratch.groupedRoutes, scratch.tileCount,
               scratch.groupedInput, scratch.groupedSums},
              gather, {tiles, shape.hiddenSize / 256, 1});
  else
    graph.add("moe_gather_rows",
              {buffers.input, scratch.groupedRoutes, scratch.tileDescriptors,
               scratch.tileCount, scratch.groupedInput},
              gather, {tiles, shape.hiddenSize / 256, 1});
  addExperts(graph, scratch, weights, plan, group);
  graph.add("moe_combine",
            {scratch.expertOutput, scratch.routeRows, scratch.routingWeights,
             buffers.residual, buffers.output},
            MoeCombineParams{rows, shape.hiddenSize, shape.routesPerToken()},
            {rows, shape.hiddenSize / 256, 1});
}

MoePlan MoE::prefillPlan(MoeShape shape, uint32_t rows, MoeConfig config) {
  if (!rows || rows > SPLASH_PREFILL_TOKEN_BUDGET)
    throw std::invalid_argument("invalid MoE prefill rows");
  return MoePlan(shape, rows, config, MoePhase::Prefill);
}

MoePlan MoE::decodePlan(MoeShape shape, uint32_t lanes, MoeConfig config) {
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid MoE decode batch width");
  return MoePlan(shape, lanes * SPLASH_TARGET_VERIFY_ROWS, config, MoePhase::Decode);
}

} // namespace splash::ops

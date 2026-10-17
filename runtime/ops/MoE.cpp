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
  if (weights.sharedScalarGate.formatId != QuantizedSegment::kFloat32)
    throw std::invalid_argument("the MoE shared scalar gate is not F32");
  requireBytes(weights.sharedScalarGate.plane0, uint64_t{hidden} * sizeof(float), "MoE shared scalar gate weight");
  if (!weights.slots) throw std::invalid_argument("MoE routed experts have no slots");
  requireBytes(weights.slotTable, uint64_t{shape.experts} * sizeof(uint32_t), "MoE slot table");
  for (const auto &[projection, name, output, input] :
       {std::tuple{&weights.gate, "gate", intermediate, hidden}, std::tuple{&weights.up, "up", intermediate, hidden},
        std::tuple{&weights.down, "down", hidden, intermediate}}) {
    requireSegment(projection->routed, weights.slots * output, input, false, std::string("MoE expert ") + name);
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
  // The router's fp32 scores, a row of routerWidth() expert slots per row,
  // live in the grouped input until the gather overwrites them. Register
  // plans also hold the down pass's Table16 tiles there.
  const uint64_t scoreBytes = uint64_t{rows} * shape.routerWidth() * sizeof(float);
  const bool table16 = ggufTile == MoeGgufTile::Register;
  const uint64_t sumsBytes = table16 ? tableSumsBytes(LinearInput::Table16, widest, groupedRows) : 0;
  return {routes * sizeof(uint32_t), routes * sizeof(float),
          uint64_t{tiles} * sizeof(MoeTileDescriptor), sizeof(MoeTileCount),
          groupedRows * sizeof(uint32_t), routes * sizeof(uint32_t),
          std::max(tableBytes(table16 ? widest : shape.hiddenSize, groupedRows), scoreBytes),
          groupedRows * shape.expertIntermediateSize * sizeof(uint16_t),
          groupedRows * widest * sizeof(uint16_t), sumsBytes};
}

// A routing kernel's instance for the shape's router width: past
// SPLASH_MOE_EXPERT_SLOTS experts, its wide one.
std::string routingKernel(std::string name, MoeShape shape) {
  static_assert(SPLASH_MOE_WIDE_EXPERT_SLOTS == 512, "the wide routing kernels are the _e512 instances");
  if (shape.routerWidth() == SPLASH_MOE_WIDE_EXPERT_SLOTS) name += "_e512";
  return name;
}

// Whether a plan runs gate and up in one pass
// (moe_expert_gguf_m<rows>_gate_up[_mixed]): on the staged tile, whose simdgroups
// each decode their own pass's formats. The register tile runs two passes.
bool oneGateUpPass(const MoePlan &plan) noexcept { return plan.configuration().ggufTile == MoeGgufTile::Staged; }

// The expert passes over grouped tiles, each on the grid of the live
// tiles that the grouping wrote (MoeTileCount), within the column tiles of
// `group` and maximumTiles(): gate and up in one pass into
// expertIntermediate (oneGateUpPass), or gate into expertOutput (the down
// pass overwrites it after the up pass consumed it) and up with silu(gate)
// into expertIntermediate; then down into expertOutput. The gate/up passes
// read the grouped rows the gather writes into groupedInput, as Table16 tiles
// on register plans, whose down pass reads the tiles a prepare dispatch
// writes there.
void addExperts(metal::CommandGraph &graph, const MoeScratch &scratch, const BlockMoeWeights &weights,
                const MoePlan &plan, const MoeGroupParams &group) {
  const MoeShape shape = plan.shape();
  const uint32_t tiles = plan.maximumTiles();
  const bool table16 = plan.configuration().ggufTile == MoeGgufTile::Register;
  if (table16)
    for (const BlockExpertProjection *projection : {&weights.gate, &weights.up, &weights.down})
      for (const QuantizedSegment *s : {&projection->routed, &projection->shared})
        if (!quant_register_tensor(s->formatId, s->inputSize))
          throw std::invalid_argument(std::string("the register expert tile does not take ") + s->name() +
                                      " weights of " + std::to_string(s->inputSize) + " inputs");
  const uint32_t threads = table16 ? GGUF_REGISTER_THREADS : GGUF_STAGED_THREADS;
  const std::string kernel = table16 ? "moe_expert_gguf_sg" : "moe_expert_gguf_m" + std::to_string(plan.tileRows());
  const metal::IndirectGrid gateUpGrid{scratch.tileCount, offsetof(MoeTileCount, gate_up_grid)};
  const metal::IndirectGrid downGrid{scratch.tileCount, offsetof(MoeTileCount, down_grid)};
  const auto params = [&](const BlockExpertProjection &projection, uint32_t n, uint32_t k) {
    return MoeGgufExpertParams{k, n, shape.experts, projection.routed.formatId, projection.shared.formatId};
  };
  const auto pass = [&](const BlockExpertProjection &projection, bool up,
                        const metal::MetalBuffer &passInput,
                        const metal::MetalBuffer &output, uint32_t n, uint32_t k,
                        uint32_t columns, const metal::IndirectGrid &grid) {
    std::vector<metal::MetalBuffer> bindings{passInput};
    if (table16) bindings.push_back(scratch.groupedSums);
    bindings.insert(bindings.end(),
                    {scratch.tileDescriptors, scratch.tileCount,
                     projection.routed.plane0, projection.routed.plane1Slot(),
                     projection.routed.meta, projection.shared.plane0,
                     projection.shared.plane1Slot(), projection.shared.meta, output,
                     scratch.expertOutput, weights.slotTable});
    graph.add(kernel + (up ? "_g" : "_a"), std::move(bindings), params(projection, n, k), {columns, tiles, 1}, grid,
              {threads, 1, 1});
  };
  const uint32_t hidden = shape.hiddenSize;
  const uint32_t intermediate = shape.expertIntermediateSize;
  if (oneGateUpPass(plan)) {
    const BlockExpertProjection &gate = weights.gate, &up = weights.up;
    // One pair table where gate and up share their formats, one each otherwise.
    const bool mixed = gate.routed.formatId != up.routed.formatId || gate.shared.formatId != up.shared.formatId;
    graph.add(kernel + (mixed ? "_gate_up_mixed" : "_gate_up"),
              {scratch.groupedInput, scratch.tileDescriptors, scratch.tileCount,
               gate.routed.plane0, gate.routed.plane1Slot(), gate.routed.meta,
               gate.shared.plane0, gate.shared.plane1Slot(), gate.shared.meta,
               up.routed.plane0, up.routed.plane1Slot(), up.routed.meta,
               up.shared.plane0, up.shared.plane1Slot(), up.shared.meta,
               scratch.expertIntermediate, weights.slotTable},
              MoeGgufGateUpParams{hidden, intermediate, shape.experts, gate.routed.formatId, gate.shared.formatId,
                                  up.routed.formatId, up.shared.formatId},
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
              intermediate, {tiles, (intermediate + 255) / 256, 1});
  pass(weights.down, false,
       table16 ? scratch.groupedInput : scratch.expertIntermediate,
       scratch.expertOutput, hidden, intermediate, group.down_columns, downGrid);
}

} // namespace

// Plans run 8-row tiles in decode and 8- or 32-row tiles in prefill; the
// register tile runs 8-row tiles.
MoePlan::MoePlan(MoeShape shape, uint32_t rows, MoeConfig config,
                 MoePhase phase)
    : shape_(shape), rows_(rows), phase_(phase), config_(config) {
  if (phase == MoePhase::Decode && config.expertTile != MoeExpertTile::M8)
    throw std::invalid_argument("invalid MoE expert tile configuration");
  if (config.ggufTile == MoeGgufTile::Register && config.expertTile != MoeExpertTile::M8)
    throw std::invalid_argument("the register expert tile takes 8-row tiles");
  workspace_ = workspaceFor(shape, rows, tileRows(), config.ggufTile);
  maximumTiles_ = moeMaximumTiles(rows, shape, tileRows());
}

namespace {

// A step's grouping into tiles, which its expert passes read: the gate and up
// passes' column tiles of a staged GGUF plan that runs them in one pass are
// its staged columns.
MoeGroupParams groupParams(const MoePlan &plan) {
  const MoeShape shape = plan.shape();
  const bool oneGateUp = oneGateUpPass(plan);
  return {plan.rows(), shape.expertsPerToken, plan.tileRows(), shape.experts,
          shape.expertIntermediateSize / (oneGateUp ? GGUF_STAGED_COLUMNS : GGUF_TILE_COLUMNS),
          shape.hiddenSize / GGUF_TILE_COLUMNS};
}

} // namespace

void MoE::add(metal::CommandGraph &graph, const MoeBuffers &buffers,
              const BlockMoeWeights &weights, const MoePlan &plan) {
  addRouting(graph, buffers, weights, plan);
  if (weights.stream) {
    const MoeLoadPart part{&buffers, &plan};
    weights.stream->addLoad(graph, {&part, 1}, weights.streamBlock);
  }
  addRouted(graph, buffers, weights, plan);
}

void MoE::addRouting(metal::CommandGraph &graph, const MoeBuffers &buffers,
                     const BlockMoeWeights &weights, const MoePlan &plan) {
  const MoeShape shape = plan.shape();
  const uint32_t rows = plan.rows();
  validate(weights, shape);
  const MoeWorkspace &required = plan.workspace();
  const uint64_t hiddenRows = uint64_t{rows} * shape.hiddenSize * sizeof(uint16_t);
  requireBytes(buffers.input, hiddenRows, "MoE input");
  requireBytes(buffers.residual, hiddenRows, "MoE residual");
  requireBytes(buffers.output, hiddenRows, "MoE output");
  const MoeScratch &scratch = buffers.scratch;
  for (const MoeScratchField &field : kMoeScratchFields)
    requireBytes(scratch.*field.buffer, required.*field.bytes, field.name);
  addRoutes(graph, buffers.input, weights.router, weights.sharedScalarGate, scratch.groupedInput,
            scratch.selectedExperts, scratch.routingWeights, rows, shape, plan.configuration().ggufRouterTile);
  // The expert passes launch the live tiles alone, on grids the grouping
  // writes with the tile count. Over every tile a step could fill, a 35B
  // decode step of one lane launched 65 tiles' threadgroups for a median of
  // 30 live ones; the rest return at once yet cost a 40-core M5 Max 0.8-4.1%
  // of a decode cycle, a 12-core M6 0.1-1.0% and a 40-core M3 Max up to 4.8%.
  graph.add(routingKernel("moe_group_routes", shape),
            {scratch.selectedExperts, scratch.tileDescriptors,
             scratch.tileCount, scratch.groupedRoutes, scratch.routeRows},
            groupParams(plan), {1, 1, 1}, {shape.routerWidth(), 1, 1});
}

void MoE::addRouted(metal::CommandGraph &graph, const MoeBuffers &buffers,
                    const BlockMoeWeights &weights, const MoePlan &plan) {
  const MoeShape shape = plan.shape();
  const uint32_t rows = plan.rows();
  const uint32_t tileRows = plan.tileRows();
  const uint32_t tiles = plan.maximumTiles();
  const MoeScratch &scratch = buffers.scratch;
  const MoeGatherParams gather{tileRows, shape.hiddenSize, shape.routesPerToken()};
  if (plan.configuration().ggufTile == MoeGgufTile::Register)
    graph.add("moe_gather_table16",
              {buffers.input, scratch.groupedRoutes, scratch.tileCount,
               scratch.groupedInput, scratch.groupedSums},
              gather, {tiles, shape.hiddenSize / 256, 1});
  else
    graph.add("moe_gather_rows",
              {buffers.input, scratch.groupedRoutes, scratch.tileDescriptors,
               scratch.tileCount, scratch.groupedInput},
              gather, {tiles, shape.hiddenSize / 256, 1});
  addExperts(graph, scratch, weights, plan, groupParams(plan));
  graph.add("moe_combine",
            {scratch.expertOutput, scratch.routeRows, scratch.routingWeights,
             buffers.residual, buffers.output},
            MoeCombineParams{rows, shape.hiddenSize, shape.routesPerToken()},
            {rows, shape.hiddenSize / 256, 1});
}

void MoE::addRoutes(metal::CommandGraph &graph, const metal::MetalBuffer &input, const QuantizedSegment &router,
                    const QuantizedSegment &sharedScalarGate, const metal::MetalBuffer &scores,
                    const metal::MetalBuffer &selected, const metal::MetalBuffer &routingWeights, uint32_t rows,
                    MoeShape shape, FloatTile routerTile) {
  // fp32 scores of the F32 router in rows of routerWidth() expert slots, as
  // the select kernel of that width reads.
  const uint32_t width = shape.routerWidth();
  addGgufFloat(graph, input, router, scores, rows, width, 0, FloatOutput::Float32, routerTile);
  graph.add(routingKernel("moe_route_select_f32", shape),
            {scores, input, sharedScalarGate.plane0, selected, routingWeights},
            MoeRouteParams{rows, shape.hiddenSize, shape.experts, shape.expertsPerToken}, {rows, 1, 1},
            {width, 1, 1});
}

uint32_t MoE::expertLoadGroups(const MoeExpertLoadParams &params) noexcept {
  uint64_t groups = 0;
  for (const MoeExpertLoadProjection &projection : params.projections)
    groups = std::max(groups, (uint64_t{projection.rows} * (projection.columns / 32) + MOE_EXPERT_LOAD_THREADS - 1) /
                                  MOE_EXPERT_LOAD_THREADS);
  return static_cast<uint32_t>(groups);
}

void MoE::addExpertLoad(metal::CommandGraph &graph, const metal::MetalBuffer &staging,
                        const metal::MetalBuffer &arena, const metal::MetalBuffer &slots,
                        const metal::MetalBuffer &sources, const MoeExpertLoadParams &params, uint32_t jobs,
                        const metal::IndirectGrid &grid) {
  graph.add("moe_expert_load", {staging, arena, slots, sources}, params, {expertLoadGroups(params), jobs, 3}, grid,
            {MOE_EXPERT_LOAD_THREADS, 1, 1});
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

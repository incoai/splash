#include "ops/MoE.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "metal/abi/MoE.h"
#include "ops/BufferExtent.hpp"

#include <functional>
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
          groupedRows * widest * sizeof(uint16_t), sumsBytes, 0};
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
// into expertIntermediate; then down into expertOutput. Register plans read
// Table16 tiles from groupedInput: the gather writes the gate/up input's and
// a prepare dispatch the down input's. The gate/up passes read `input`: the
// grouped rows, or with plan.rowsInPlace() the plan's rows in place. A
// `beside` pass is encoded ahead of the first gate/up pass and overlaps it.
void addExperts(metal::CommandGraph &graph, const MoeScratch &scratch, const BlockMoeWeights &weights,
                const MoePlan &plan, const MoeGroupParams &group, const metal::MetalBuffer &input,
                const std::function<void()> &beside) {
  const MoeShape shape = plan.shape();
  const uint32_t tiles = plan.maximumTiles();
  const bool table16 = plan.configuration().ggufTile == MoeGgufTile::Register;
  if (table16)
    for (const BlockExpertProjection *projection : {&weights.gate, &weights.up, &weights.down})
      for (const QuantizedSegment *s : {&projection->routed, &projection->shared})
        if (!quant_register_format(s->formatId))
          throw std::invalid_argument(std::string("the register expert tile does not take ") + s->name() + " weights");
  const uint32_t threads = table16 ? GGUF_REGISTER_THREADS : GGUF_STAGED_THREADS;
  const std::string kernel = table16 ? "moe_expert_gguf_sg" : "moe_expert_gguf_m" + std::to_string(plan.tileRows());
  const metal::IndirectGrid gateUpGrid{scratch.tileCount, offsetof(MoeTileCount, gate_up_grid)};
  const metal::IndirectGrid downGrid{scratch.tileCount, offsetof(MoeTileCount, down_grid)};
  const auto pass = [&](const BlockExpertProjection &projection, bool up,
                        const metal::MetalBuffer &passInput, uint32_t inPlace,
                        const metal::MetalBuffer &output, uint32_t n, uint32_t k,
                        uint32_t columns, const metal::IndirectGrid &grid) {
    std::vector<metal::MetalBuffer> bindings{passInput};
    if (table16) bindings.push_back(scratch.groupedSums);
    bindings.insert(bindings.end(),
                    {scratch.tileDescriptors, scratch.tileCount,
                     projection.routed.plane0, projection.routed.plane1Slot(),
                     projection.routed.meta, projection.shared.plane0,
                     projection.shared.plane1Slot(), projection.shared.meta, output,
                     scratch.expertOutput});
    graph.add(kernel + (up ? "_g" : "_a"), std::move(bindings),
              MoeGgufExpertParams{k, n, shape.experts, projection.routed.formatId,
                                  projection.shared.formatId, inPlace},
              {columns, tiles, 1}, grid, {threads, 1, 1});
  };
  const uint32_t hidden = shape.hiddenSize;
  const uint32_t intermediate = shape.expertIntermediateSize;
  if (beside) beside();
  if (oneGateUpPass(weights, plan)) {
    const BlockExpertProjection &gate = weights.gate, &up = weights.up;
    graph.add(kernel + "_gate_up",
              {input, scratch.tileDescriptors, scratch.tileCount,
               gate.routed.plane0, gate.routed.plane1Slot(), gate.routed.meta,
               gate.shared.plane0, gate.shared.plane1Slot(), gate.shared.meta,
               up.routed.plane0, up.routed.plane1Slot(), up.routed.meta,
               up.shared.plane0, up.shared.plane1Slot(), up.shared.meta,
               scratch.expertIntermediate},
              MoeGgufExpertParams{hidden, intermediate, shape.experts, gate.routed.formatId,
                                  gate.shared.formatId, group.in_place},
              {group.gate_up_columns, tiles, 1}, gateUpGrid, {threads, 1, 1});
    if (beside) graph.concurrentWithPrevious();
  } else {
    pass(weights.gate, false, input, group.in_place, scratch.expertOutput,
         intermediate, hidden, group.gate_up_columns, gateUpGrid);
    // The gate parks in expertOutput rows of `intermediate` columns, ahead of
    // the shared rows `beside` writes while it is no wider than them.
    if (beside && intermediate <= hidden) graph.concurrentWithPrevious();
    pass(weights.up, true, input, group.in_place, scratch.expertIntermediate,
         intermediate, hidden, group.gate_up_columns, gateUpGrid);
  }
  if (table16)
    graph.add("moe_prepare_table16",
              {scratch.expertIntermediate, scratch.tileCount,
               scratch.groupedInput, scratch.groupedSums},
              intermediate, {tiles, intermediate / 256, 1});
  pass(weights.down, false,
       table16 ? scratch.groupedInput : scratch.expertIntermediate, 0,
       scratch.expertOutput, hidden, intermediate, group.down_columns, downGrid);
}

// The shared expert of a plan with a sharedRowsBase as dense GGUF prefill
// projections over the layer's rows (Linear::addGgufPrefill's kernels): gate
// into sharedIntermediate, up with silu(gate) in place, down into the shared
// rows of expertOutput, where the grouping maps the shared routes. Each pass
// is encoded ahead of the dispatch of the routing chain it overlaps (the
// router, the gather, the first gate/up pass), whose buffers it does not
// touch: its few threadgroups start first and the chain's fill the cores
// beside them.
struct SharedPasses {
  metal::CommandGraph &graph;
  const MoeBuffers &buffers;
  const BlockMoeWeights &weights;
  const MoePlan &plan;

  void plain(const QuantizedSegment &w, const metal::MetalBuffer &input, const metal::MetalBuffer &output,
             uint32_t n, uint32_t k, uint32_t offset) const {
    // One segment: the two slots past it bind its planes, which no tile reads.
    std::vector<metal::MetalBuffer> bindings{input};
    for (int slot = 0; slot < 3; ++slot) bindings.insert(bindings.end(), {w.plane0, w.plane1Slot(), w.meta});
    bindings.push_back(output);
    graph.add(std::string("gguf_prefill_") + w.name() + "_a", std::move(bindings),
              GgufPrefillSegmentsParams{k, plan.rows(), n, {n, 0, 0}, {offset, 0, 0}}, grid(n),
              {GGUF_PREFILL_THREADS, 1, 1});
  }
  metal::DispatchSize grid(uint32_t n) const {
    return {(plan.inputRows() + GGUF_PREFILL_ROWS - 1) / GGUF_PREFILL_ROWS, n / GGUF_TILE_COLUMNS, 1};
  }
  void gate() const {
    const MoeShape shape = plan.shape();
    plain(weights.gate.shared, buffers.input, buffers.scratch.sharedIntermediate, shape.expertIntermediateSize,
          shape.hiddenSize, 0);
  }
  void up() const {
    const MoeShape shape = plan.shape();
    const QuantizedSegment &w = weights.up.shared;
    const metal::MetalBuffer &h = buffers.scratch.sharedIntermediate;
    graph.add(std::string("gguf_prefill_") + w.name() + "_g", {buffers.input, w.plane0, w.plane1Slot(), w.meta, h, h},
              GgufPrefillParams{shape.hiddenSize, plan.rows(), shape.expertIntermediateSize, 0},
              grid(shape.expertIntermediateSize), {GGUF_PREFILL_THREADS, 1, 1});
  }
  // The shared rows of expertOutput, at a column offset of whole rows.
  void down() const {
    const MoeShape shape = plan.shape();
    plain(weights.down.shared, buffers.scratch.sharedIntermediate, buffers.scratch.expertOutput, shape.hiddenSize,
          shape.expertIntermediateSize, plan.sharedRowsBase() * shape.hiddenSize);
  }
};

// Encodes `shared` ahead of `chain`, the two overlapping.
template <class Shared, class Chain>
void overlap(metal::CommandGraph &graph, Shared &&shared, Chain &&chain) {
  shared();
  chain();
  graph.concurrentWithPrevious();
}

} // namespace

// Plans run 8-row tiles in decode and 8- or 32-row tiles in prefill; the
// register tile runs 8-row tiles.
MoePlan::MoePlan(MoeShape shape, uint32_t rows, MoeConfig config,
                 MoePhase phase)
    : shape_(shape), rows_(rows), config_(config) {
  if (phase == MoePhase::Decode && config.expertTile != MoeExpertTile::M8)
    throw std::invalid_argument("invalid MoE expert tile configuration");
  if (config.ggufTile == MoeGgufTile::Register && config.expertTile != MoeExpertTile::M8)
    throw std::invalid_argument("the register expert tile takes 8-row tiles");
  workspace_ = workspaceFor(shape, rows, tileRows(), config.ggufTile);
  if (sharedRowsBase())
    workspace_.sharedIntermediateBytes = uint64_t{inputRows()} * shape.expertIntermediateSize * sizeof(uint16_t);
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
  requireBytes(buffers.input, uint64_t{plan.inputRows()} * shape.hiddenSize * sizeof(uint16_t), "MoE input");
  requireBytes(buffers.residual, hiddenRows, "MoE residual");
  requireBytes(buffers.output, hiddenRows, "MoE output");
  const MoeScratch &scratch = buffers.scratch;
  for (const MoeScratchField &field : kMoeScratchFields)
    requireBytes(scratch.*field.buffer, required.*field.bytes, field.name);
  const MoeRouteParams routeParams{rows, shape.hiddenSize, shape.experts,
                                   shape.expertsPerToken};
  // The shared expert as dense projections beside the routing, or as tiles of
  // the expert passes.
  const SharedPasses shared{graph, buffers, weights, plan};
  const bool dense = plan.sharedRowsBase() != 0;
  const auto beside = [&](void (SharedPasses::*pass)() const, auto &&chain) {
    if (dense) overlap(graph, [&] { (shared.*pass)(); }, chain);
    else chain();
  };
  // fp32 scores of the F32 router in rows of expert slots, as the select
  // kernel reads.
  beside(&SharedPasses::gate, [&] {
    addGgufFloat(graph, buffers.input, weights.router, scratch.groupedInput, rows,
                 SPLASH_MOE_EXPERT_SLOTS, 0, FloatOutput::Float32, plan.configuration().ggufRouterTile);
  });
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
                             shape.hiddenSize / GGUF_TILE_COLUMNS, plan.rowsInPlace(), plan.sharedRowsBase()};
  graph.add("moe_group_routes",
            {scratch.selectedExperts, scratch.tileDescriptors,
             scratch.tileCount, scratch.groupedRoutes, scratch.routeRows},
            group, {1, 1, 1}, {SPLASH_MOE_EXPERT_SLOTS, 1, 1});
  // Gather and combine: four columns per thread, 1024 per threadgroup where
  // the width allows (256 otherwise).
  const uint32_t copyThreads = shape.hiddenSize % 1024 ? 64 : 256;
  const MoeGatherParams gather{tileRows, shape.hiddenSize, shape.routesPerToken()};
  if (plan.configuration().ggufTile == MoeGgufTile::Register)
    graph.add("moe_gather_table16",
              {buffers.input, scratch.groupedRoutes, scratch.tileCount,
               scratch.groupedInput, scratch.groupedSums},
              gather, {tiles, shape.hiddenSize / 256, 1});
  else if (!group.in_place)
    beside(&SharedPasses::up, [&] {
      graph.add("moe_gather_rows",
                {buffers.input, scratch.groupedRoutes, scratch.tileDescriptors,
                 scratch.tileCount, scratch.groupedInput},
                gather, {tiles, shape.hiddenSize / (4 * copyThreads), 1}, {copyThreads, 1, 1});
    });
  // The shared down pass reads the intermediate its up pass wrote before the
  // gather's barrier.
  addExperts(graph, scratch, weights, plan, group, group.in_place ? buffers.input : scratch.groupedInput,
             dense ? [&] { shared.down(); } : std::function<void()>{});
  graph.add("moe_combine",
            {scratch.expertOutput, scratch.routeRows, scratch.routingWeights,
             buffers.residual, buffers.output},
            MoeCombineParams{rows, shape.hiddenSize, shape.routesPerToken()},
            {rows, shape.hiddenSize / (4 * copyThreads), 1}, {copyThreads, 1, 1});
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

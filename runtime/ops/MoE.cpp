#include "ops/MoE.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "metal/abi/MoE.h"
#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace splash::ops {

namespace {
// The formats the W-A8 tile takes (kernels/common/gguf_staged_i8.h moe_i8_format).
constexpr uint32_t kInt8Formats[] = {
    GGUF_FMT_IQ2S, GGUF_FMT_IQ2XS, GGUF_FMT_IQ3S, GGUF_FMT_IQ3XXS, GGUF_FMT_IQ4XS,
    GGUF_FMT_Q6K,  GGUF_FMT_Q80,   GGUF_FMT_Q2K,  GGUF_FMT_Q4K,    GGUF_FMT_Q5K,
};
uint64_t alignUp(uint64_t bytes) { return (bytes + 255) / 256 * 256; }
} // namespace

uint32_t moeInt8Formats() noexcept {
  uint32_t mask = 0;
  for (const uint32_t id : kInt8Formats) mask |= 1u << id;
  return mask;
}

MoeInt8 moeInt8Decode() noexcept { return {.activations = 1, .formats = moeInt8Formats()}; }

MoeInt8Layout moeInt8Layout(MoeShape shape, uint64_t groupedRows) noexcept {
  MoeInt8Layout layout;
  layout.x8 = 0;
  layout.xScales = alignUp(groupedRows * shape.hiddenSize);
  layout.d8 = alignUp(layout.xScales + groupedRows * sizeof(float));
  layout.dScales = alignUp(layout.d8 + groupedRows * shape.expertIntermediateSize);
  layout.bytes = layout.dScales + groupedRows * (shape.expertIntermediateSize / 32) * sizeof(float);
  return layout;
}

namespace {

// A GGUF tensor of output x input weights, of a GGUF_FMT_* format unless
// they are float.
void requireShape(const QuantizedSegment &segment, uint32_t output, uint32_t input, bool floatWeights,
                  const std::string &name) {
  if (segment.isFloat() != floatWeights || (!floatWeights && segment.formatId >= GGUF_FMT_COUNT) ||
      segment.outputSize != output || segment.inputSize != input)
    throw std::invalid_argument(name + " does not match the plan");
}

// A GGUF tensor of output x input weights, and the planes of its GGUF_FMT_*
// format; what reads a float tensor checks its fp32 values.
void requireSegment(const QuantizedSegment &segment, uint32_t output, uint32_t input, bool floatWeights,
                    const std::string &name) {
  requireShape(segment, output, input, floatWeights, name);
  if (!floatWeights) requireSegmentPlanes(segment, name);
}

// Every weight the plan's dispatches read. The router runs through
// addGgufFloat, which checks its weights. The expert kernels reach the routed
// experts' chunks by address, where no binding's check guards them: their
// views were checked where they were made (ExpertChunks::of), for the
// block's slots and routed formats and sizes.
void validate(const BlockMoeWeights &weights, MoeShape shape) {
  const uint32_t hidden = shape.hiddenSize, intermediate = shape.expertIntermediateSize;
  requireSegment(weights.router, shape.experts, hidden, true, "MoE router");
  if (shape.sharedGate) {
    requireSegment(weights.sharedScalarGate, 1, hidden, true, "MoE shared scalar gate");
    // The select kernel reads the scalar gate's fp32 weights.
    if (weights.sharedScalarGate.formatId != QuantizedSegment::kFloat32)
      throw std::invalid_argument("the MoE shared scalar gate is not F32");
    requireBytes(weights.sharedScalarGate.plane0, uint64_t{hidden} * sizeof(float), "MoE shared scalar gate weight");
  }
  if (!weights.slots) throw std::invalid_argument("MoE routed experts have no slots");
  requireBytes(weights.slotTable, uint64_t{shape.experts} * sizeof(uint32_t), "MoE slot table");
  for (const auto &[projection, name, output, input] :
       {std::tuple{&weights.gate, "gate", intermediate, hidden}, std::tuple{&weights.up, "up", intermediate, hidden},
        std::tuple{&weights.down, "down", hidden, intermediate}}) {
    requireShape(projection->routed, weights.slots * output, input, false, std::string("MoE expert ") + name);
    requireSegment(projection->shared, shape.sharedSlices * output, input, false, std::string("MoE shared ") + name);
  }
  if (weights.chunks.empty()) throw std::invalid_argument("MoE routed experts have no chunks");
  if (!weights.chunks.checkedFor(weights.slots, weights.gate.routed, weights.up.routed, weights.down.routed))
    throw std::invalid_argument("MoE expert chunks were checked for another block");
}

MoeWorkspace workspaceFor(MoeShape shape, uint32_t rows, uint32_t tileRows, MoeGgufTile ggufTile, bool int8) {
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
  // Int8 plans (MoePlan::int8) keep their int8 rows there (moeInt8Layout).
  const uint64_t sumsBytes = table16 ? tableSumsBytes(LinearInput::Table16, widest, groupedRows)
                             : int8  ? moeInt8Layout(shape, groupedRows).bytes
                                     : 0;
  return {routes * sizeof(uint32_t), routes * sizeof(float),
          uint64_t{tiles} * sizeof(MoeTileDescriptor), sizeof(MoeTileCount),
          groupedRows * sizeof(uint32_t), routes * sizeof(uint32_t),
          std::max(tableBytes(table16 ? widest : shape.hiddenSize, groupedRows), scoreBytes),
          groupedRows * shape.expertIntermediateSize * sizeof(uint16_t),
          groupedRows * widest * sizeof(uint16_t), sumsBytes};
}

// The shape's compiled routing (MoeShape::compiledRouting).
const CompiledMoeRouting &compiledRouting(MoeShape shape) {
  if (const CompiledMoeRouting *compiled = shape.compiledRouting()) return *compiled;
  throw std::invalid_argument("no MoE kernel routes the shape's router and shared slices");
}

// A routing kernel's instance for the shape's router width (past
// SPLASH_MOE_EXPERT_SLOTS experts, its wide one) and its compiled routing's
// `suffix`.
std::string routingKernel(std::string name, MoeShape shape, std::string_view CompiledMoeRouting::*suffix) {
  static_assert(SPLASH_MOE_WIDE_EXPERT_SLOTS == 512, "the wide routing kernels are the _e512 instances");
  if (shape.routerWidth() == SPLASH_MOE_WIDE_EXPERT_SLOTS) name += "_e512";
  name += compiledRouting(shape).*suffix;
  return name;
}

// The int8 parameters of an int8 plan (MoePlan::int8): its layout in groupedSums and switches; the up epilogue
// writes the down input's int8 rows where a down segment runs int8.
MoeInt8Params int8Params(const MoePlan &plan, const BlockMoeWeights &weights) {
  const MoeInt8 &int8 = plan.configuration().int8;
  const MoeInt8Layout layout = moeInt8Layout(plan.shape(), uint64_t{plan.maximumTiles()} * plan.tileRows());
  const auto takes = [&](const QuantizedSegment &segment) {
    return segment.formatId < 32 && ((int8.formats >> segment.formatId) & 1u);
  };
  const uint32_t mode = (int8.pow2 ? MOE_A8_POW2 : 0u) |
                        (takes(weights.down.routed) || takes(weights.down.shared) ? MOE_A8_DOWN : 0u);
  return {int8.formats, mode, static_cast<uint32_t>(layout.x8), static_cast<uint32_t>(layout.xScales),
          static_cast<uint32_t>(layout.d8), static_cast<uint32_t>(layout.dScales)};
}

// Whether a fake quantizing plan (MoeInt8::activations 2) rounds the gate/up input (`down` false) or the down input:
// where the W-A8 tile would take the pass's segments (MoeInt8::formats; none named, every pass), so that a test
// compares a bf16 pass with the int8 pass of the same formats.
bool fakeRounds(const MoePlan &plan, const BlockMoeWeights &weights, bool down) {
  const MoeInt8 &int8 = plan.configuration().int8;
  if (!plan.int8() || int8.activations != 2) return false;
  if (!int8.formats) return true;
  const auto takes = [&](const QuantizedSegment &segment) {
    return segment.formatId < 32 && ((int8.formats >> segment.formatId) & 1u);
  };
  return down ? takes(weights.down.routed) || takes(weights.down.shared)
              : takes(weights.gate.routed) || takes(weights.gate.shared) || takes(weights.up.routed) ||
                    takes(weights.up.shared);
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
  // The passes' instances for the shape's shared slices.
  const std::string slices(compiledRouting(shape).slicesSuffix);
  const metal::IndirectGrid gateUpGrid{scratch.tileCount, offsetof(MoeTileCount, gate_up_grid)};
  const metal::IndirectGrid downGrid{scratch.tileCount, offsetof(MoeTileCount, down_grid)};
  const uint32_t floorSlots = weights.chunks.floorSlots(), chunkSlots = weights.chunks.chunkSlots();
  // Projection z (0 gate, 1 up, 2 down) of the routed experts, which the
  // pass reaches in the chunks.
  const auto params = [&](const BlockExpertProjection &projection, uint32_t z, uint32_t n, uint32_t k) {
    return MoeGgufExpertParams{k, n, shape.experts, projection.routed.formatId, projection.shared.formatId,
                               floorSlots, chunkSlots, z};
  };
  const auto pass = [&](const BlockExpertProjection &projection, uint32_t z, bool up,
                        const metal::MetalBuffer &passInput,
                        const metal::MetalBuffer &output, uint32_t n, uint32_t k,
                        uint32_t columns, const metal::IndirectGrid &grid) {
    std::vector<metal::MetalBuffer> bindings{passInput};
    if (table16) bindings.push_back(scratch.groupedSums);
    bindings.insert(bindings.end(),
                    {scratch.tileDescriptors, scratch.tileCount, projection.shared.plane0,
                     projection.shared.plane1Slot(), projection.shared.meta, output,
                     scratch.expertOutput, weights.slotTable});
    graph.add(kernel + (up ? "_g" : "_a") + slices, std::move(bindings), params(projection, z, n, k),
              {columns, tiles, 1}, grid, {threads, 1, 1});
    graph.addresses(weights.chunks.planes());
  };
  const uint32_t hidden = shape.hiddenSize;
  const uint32_t intermediate = shape.expertIntermediateSize;
  // Int8 plans with W-A8 formats (MoeInt8::formats) run the int8 passes (kernels/shared/moe_gguf_i8.metal); a fake
  // quantizing plan (MoeInt8::activations 2) rounds the intermediate before the down pass.
  const bool wa8 = plan.int8() && plan.configuration().int8.activations == 1 && plan.configuration().int8.formats;
  const bool fake = fakeRounds(plan, weights, true);
  if (wa8) {
    const MoeInt8Params a8 = int8Params(plan, weights);
    const BlockExpertProjection &gate = weights.gate, &up = weights.up, &down = weights.down;
    const bool mixed = gate.routed.formatId != up.routed.formatId || gate.shared.formatId != up.shared.formatId;
    graph.add(std::string("moe_expert_gguf_i8_m8_gate_up") + (mixed ? "_mixed" : "") + slices,
              {scratch.groupedInput, scratch.tileDescriptors, scratch.tileCount,
               gate.shared.plane0, gate.shared.plane1Slot(), gate.shared.meta,
               up.shared.plane0, up.shared.plane1Slot(), up.shared.meta,
               scratch.expertIntermediate, weights.slotTable, scratch.groupedSums},
              MoeGgufI8GateUpParams{{hidden, intermediate, shape.experts, gate.routed.formatId, gate.shared.formatId,
                                     up.routed.formatId, up.shared.formatId, floorSlots, chunkSlots},
                                    a8},
              {group.gate_up_columns, tiles, 1}, gateUpGrid, {threads, 1, 1});
    graph.addresses(weights.chunks.planes());
    graph.add("moe_expert_gguf_i8_m8_a" + slices,
              {scratch.expertIntermediate, scratch.tileDescriptors, scratch.tileCount, down.shared.plane0,
               down.shared.plane1Slot(), down.shared.meta, scratch.expertOutput, scratch.expertOutput,
               weights.slotTable, scratch.groupedSums},
              MoeGgufI8ExpertParams{params(down, 2, hidden, intermediate), a8}, {group.down_columns, tiles, 1},
              downGrid, {threads, 1, 1});
    graph.addresses(weights.chunks.planes());
    return;
  }
  if (oneGateUpPass(plan)) {
    const BlockExpertProjection &gate = weights.gate, &up = weights.up;
    // One pair table where gate and up share their formats, one each otherwise.
    const bool mixed = gate.routed.formatId != up.routed.formatId || gate.shared.formatId != up.shared.formatId;
    graph.add(kernel + (mixed ? "_gate_up_mixed" : "_gate_up") + slices,
              {scratch.groupedInput, scratch.tileDescriptors, scratch.tileCount,
               gate.shared.plane0, gate.shared.plane1Slot(), gate.shared.meta,
               up.shared.plane0, up.shared.plane1Slot(), up.shared.meta,
               scratch.expertIntermediate, weights.slotTable},
              MoeGgufGateUpParams{hidden, intermediate, shape.experts, gate.routed.formatId, gate.shared.formatId,
                                  up.routed.formatId, up.shared.formatId, floorSlots, chunkSlots},
              {group.gate_up_columns, tiles, 1}, gateUpGrid, {threads, 1, 1});
    graph.addresses(weights.chunks.planes());
  } else {
    pass(weights.gate, 0, false, scratch.groupedInput, scratch.expertOutput,
         intermediate, hidden, group.gate_up_columns, gateUpGrid);
    pass(weights.up, 1, true, scratch.groupedInput, scratch.expertIntermediate,
         intermediate, hidden, group.gate_up_columns, gateUpGrid);
  }
  if (fake)
    graph.add("moe_a8_fake", {scratch.expertIntermediate, scratch.tileDescriptors, scratch.tileCount},
              MoeA8FakeParams{plan.tileRows(), intermediate, 32,
                              plan.configuration().int8.pow2 ? MOE_A8_POW2 : 0u},
              {tiles, plan.tileRows(), 1}, {32, 1, 1});
  if (table16)
    graph.add("moe_prepare_table16",
              {scratch.expertIntermediate, scratch.tileCount,
               scratch.groupedInput, scratch.groupedSums},
              intermediate, {tiles, (intermediate + 255) / 256, 1});
  pass(weights.down, 2, false,
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
  workspace_ = workspaceFor(shape, rows, tileRows(), config.ggufTile, int8());
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
          shape.hiddenSize / GGUF_TILE_COLUMNS, SPLASH_TARGET_VERIFY_ROWS};
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
  std::vector<metal::MetalBuffer> grouping{scratch.selectedExperts, scratch.tileDescriptors, scratch.tileCount,
                                           scratch.groupedRoutes, scratch.routeRows};
  if (buffers.verifyRows) {
    // A verify step's rows are its lanes' blocks.
    if (plan.phase() != MoePhase::Decode)
      throw std::invalid_argument("only a verify step's MoE routes its verify rows");
    requireBytes(buffers.verifyRows, uint64_t{rows / SPLASH_TARGET_VERIFY_ROWS} * sizeof(uint32_t),
                 "MoE verify rows");
    grouping.push_back(buffers.verifyRows);
  }
  graph.add(routingKernel(buffers.verifyRows ? "moe_group_routes_verified" : "moe_group_routes", shape,
                          &CompiledMoeRouting::slicesSuffix),
            std::move(grouping), groupParams(plan), {1, 1, 1}, {shape.routerWidth(), 1, 1});
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
  else if (plan.int8() && plan.configuration().int8.activations == 1) {
    // The A8 pre-pass in the gather (MoeInt8::activations 1): the bf16 rows and the int8 rows with their scales.
    const MoeInt8Layout layout = moeInt8Layout(shape, uint64_t{tiles} * tileRows);
    graph.add("moe_gather_rows_a8",
              {buffers.input, scratch.groupedRoutes, scratch.tileDescriptors, scratch.tileCount,
               scratch.groupedInput, scratch.groupedSums},
              MoeGatherA8Params{tileRows, shape.hiddenSize, shape.routesPerToken(),
                                plan.configuration().int8.pow2 ? MOE_A8_POW2 : 0u,
                                static_cast<uint32_t>(layout.x8), static_cast<uint32_t>(layout.xScales)},
              {tiles, shape.hiddenSize / 256, 1});
  } else {
    graph.add("moe_gather_rows",
              {buffers.input, scratch.groupedRoutes, scratch.tileDescriptors,
               scratch.tileCount, scratch.groupedInput},
              gather, {tiles, shape.hiddenSize / 256, 1});
    if (fakeRounds(plan, weights, false))
      graph.add("moe_a8_fake", {scratch.groupedInput, scratch.tileDescriptors, scratch.tileCount},
                MoeA8FakeParams{tileRows, shape.hiddenSize, shape.hiddenSize,
                                plan.configuration().int8.pow2 ? MOE_A8_POW2 : 0u},
                {tiles, tileRows, 1}, {32, 1, 1});
  }
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
  // The router's fp32 scores, in the format its checkpoint holds it
  // (BlockMoeWeights), in rows of routerWidth() expert slots, as the select
  // kernel of that width reads, which weights gated shared slices by the
  // sigmoid of the input's product with the scalar gate.
  const uint32_t width = shape.routerWidth();
  addGgufFloat(graph, input, router, scores, rows, width, 0, FloatOutput::Float32, routerTile);
  std::vector<metal::MetalBuffer> bindings{scores};
  if (shape.sharedGate) bindings.insert(bindings.end(), {input, sharedScalarGate.plane0});
  bindings.insert(bindings.end(), {selected, routingWeights});
  graph.add(routingKernel("moe_route_select_f32", shape, &CompiledMoeRouting::selectSuffix), std::move(bindings),
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
                        const ExpertChunks &chunks, const metal::MetalBuffer &slots,
                        const metal::MetalBuffer &sources, const MoeExpertLoadParams &params, uint32_t jobs,
                        const metal::IndirectGrid &grid) {
  if (chunks.empty()) throw std::invalid_argument("MoE expert chunks have no slots");
  MoeExpertLoadParams load = params;
  load.floor_slots = chunks.floorSlots();
  load.chunk_slots = chunks.chunkSlots();
  graph.add("moe_expert_load", {staging, slots, sources}, load, {expertLoadGroups(params), jobs, 3}, grid,
            {MOE_EXPERT_LOAD_THREADS, 1, 1});
  graph.addresses(chunks.planes(), true);
}

ExpertChunks ExpertChunks::of(std::vector<metal::MetalBuffer> planes, uint32_t floorSlots, uint32_t chunkSlots,
                              uint32_t slots, const QuantizedSegment &gate, const QuantizedSegment &up,
                              const QuantizedSegment &down) {
  if (planes.empty() || planes.size() % MOE_EXPERT_CHUNK_ADDRESSES ||
      planes.size() > uint64_t{MOE_EXPERT_CHUNK_ADDRESSES} * MOE_EXPERT_MAX_CHUNKS)
    throw std::invalid_argument("MoE expert chunks take 1 to " + std::to_string(MOE_EXPERT_MAX_CHUNKS) + " chunks of " +
                                std::to_string(MOE_EXPERT_CHUNK_ADDRESSES) + " planes");
  const uint64_t count = planes.size() / MOE_EXPERT_CHUNK_ADDRESSES;
  const uint64_t capacity = floorSlots + (count - 1) * chunkSlots;
  if (!floorSlots || !chunkSlots || capacity < slots)
    throw std::invalid_argument("MoE expert chunks hold " + std::to_string(capacity) + " slots, the block " +
                                std::to_string(slots));
  if (!slots) throw std::invalid_argument("MoE routed experts have no slots");
  // The slots chunk `chunk` holds of the block's: the floor's, then each
  // elastic chunk's from slot floorSlots on, none past the block's.
  const auto held = [&](uint64_t chunk) {
    if (!chunk) return std::min(floorSlots, slots);
    const uint64_t first = floorSlots + (chunk - 1) * chunkSlots;
    return first < slots ? static_cast<uint32_t>(std::min<uint64_t>(chunkSlots, slots - first)) : 0u;
  };
  uint32_t z = 0;
  for (const auto &[routed, name] : {std::tuple{&gate, "gate"}, std::tuple{&up, "up"}, std::tuple{&down, "down"}}) {
    if (routed->isFloat() || routed->formatId >= GGUF_FMT_COUNT || !routed->outputSize || routed->outputSize % slots)
      throw std::invalid_argument(std::string("MoE expert ") + name + " is not planes of a GGUF format in rows of " +
                                  std::to_string(slots) + " slots");
    const QuantFormat &format = routed->format();
    const uint32_t rows = routed->outputSize / slots;
    const uint32_t groups = routed->inputSize / 32, units = groups / format.meta_groups;
    for (uint64_t chunk = 0; chunk < count; ++chunk) {
      const uint32_t chunkHeld = held(chunk);
      for (const auto &[plane, blocks, unitBytes, what] :
           {std::tuple{0u, groups, format.plane0_bytes, "plane0"},
            std::tuple{1u, groups, format.plane1_bytes, "plane1"}, std::tuple{2u, units, format.meta_bytes, "meta"}}) {
        const metal::MetalBuffer &view = planes[chunk * MOE_EXPERT_CHUNK_ADDRESSES + z * 3 + plane];
        const uint64_t bytes =
            chunkHeld && unitBytes ? (quant_tile_index(chunkHeld * rows - 1, blocks - 1, blocks) + 1) * unitBytes : 0;
        if (!view || view.sizeBytes() < bytes)
          throw std::invalid_argument(std::string("MoE expert ") + name + " chunk " + std::to_string(chunk) + " " +
                                      what + (view.sizeBytes() < bytes ? " buffer holds " +
                                                                             std::to_string(view.sizeBytes()) +
                                                                             " bytes, needs " + std::to_string(bytes)
                                                                       : std::string(" has no buffer")));
      }
    }
    ++z;
  }
  ExpertChunks chunks;
  chunks.live_ = std::make_shared<Live>();
  for (uint64_t live = 1; live <= count; ++live)
    chunks.live_->planes.push_back(std::make_shared<const std::vector<metal::MetalBuffer>>(
        planes.begin(), planes.begin() + static_cast<std::ptrdiff_t>(live * MOE_EXPERT_CHUNK_ADDRESSES)));
  chunks.live_->count.store(static_cast<uint32_t>(count), std::memory_order_release);
  chunks.floorSlots_ = floorSlots;
  chunks.chunkSlots_ = chunkSlots;
  chunks.slots_ = slots;
  chunks.routed_ = {routedOf(gate), routedOf(up), routedOf(down)};
  return chunks;
}

bool ExpertChunks::checkedFor(uint32_t slots, const QuantizedSegment &gate, const QuantizedSegment &up,
                              const QuantizedSegment &down) const noexcept {
  return live_ && slots == slots_ && routed_ == std::array{routedOf(gate), routedOf(up), routedOf(down)};
}

void ExpertChunks::keep(uint32_t chunks) {
  if (!chunks || chunks > this->chunks())
    throw std::invalid_argument("MoE expert chunks keep " + std::to_string(chunks) + " of " +
                                std::to_string(this->chunks()) + " chunks");
  live_->count.store(chunks, std::memory_order_release);
}

ExpertChunks residentExpertChunks(const BlockMoeWeights &weights) {
  std::vector<metal::MetalBuffer> planes;
  planes.reserve(MOE_EXPERT_CHUNK_ADDRESSES);
  for (const BlockExpertProjection *projection : {&weights.gate, &weights.up, &weights.down})
    planes.insert(planes.end(), {projection->routed.plane0, projection->routed.plane1Slot(), projection->routed.meta});
  return ExpertChunks::of(std::move(planes), weights.slots, weights.slots, weights.slots, weights.gate.routed,
                          weights.up.routed, weights.down.routed);
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

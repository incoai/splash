#include "ops/MoE.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "metal/abi/MoE.h"
#include "ops/BufferExtent.hpp"
#include "ops/LaneBindings.hpp"

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
  if (shape.selectionBias)
    requireBytes(weights.selectionBias, uint64_t{shape.experts} * sizeof(float), "MoE selection bias");
  requireBytes(weights.expertScales, uint64_t{3} * shape.experts * sizeof(float), "MoE expert scales");
  if (!weights.slots) throw std::invalid_argument("MoE routed experts have no slots");
  requireBytes(weights.slotTable, uint64_t{shape.experts} * sizeof(uint32_t), "MoE slot table");
  if (weights.stream)
    requireBytes(weights.loadSlotTable, uint64_t{shape.experts} * sizeof(uint32_t), "MoE load slot table");
  for (const auto &[projection, name, output, input] :
       {std::tuple{&weights.gate, "gate", intermediate, hidden}, std::tuple{&weights.up, "up", intermediate, hidden},
        std::tuple{&weights.down, "down", hidden, intermediate}}) {
    requireShape(projection->routed, weights.slots * output, input, false, std::string("MoE expert ") + name);
    if (shape.sharedSlices)
      requireSegment(projection->shared, shape.sharedSlices * output, input, false, std::string("MoE shared ") + name);
  }
  if (weights.chunks.empty()) throw std::invalid_argument("MoE routed experts have no chunks");
  if (!weights.chunks.checkedFor(weights.slots, weights.gate.routed, weights.up.routed, weights.down.routed))
    throw std::invalid_argument("MoE expert chunks were checked for another block");
}

MoeWorkspace workspaceFor(MoeShape shape, uint32_t rows, uint32_t tileRows, bool int8Passes) {
  if (!shape.valid())
    throw std::invalid_argument("invalid MoE workspace shape");
  const uint64_t routes = uint64_t{rows} * shape.routesPerToken();
  const uint32_t tiles = moeMaximumTiles(rows, shape, tileRows);
  // A grouped row per route; the grouped input and the intermediate, which
  // the passes' matmuls read, hold tileRows - 1 rows more, which a last
  // tile's matmul may read past its live rows (moe_matmul_rows).
  const uint64_t matmulRows = routes + tileRows - 1;
  // The router's fp32 scores, a row of routerWidth() expert slots per row,
  // live in the grouped input until the gather overwrites them.
  const uint64_t scoreBytes = uint64_t{rows} * shape.routerWidth() * sizeof(float);
  // The int8 passes' codes and scales (addGgufPrefillInt8Quantize): the
  // chunk's rows of the hidden size, and a grouped row per route of the
  // larger of it and the intermediate size, the gate/up input's then down's.
  // The int8 tile reads a tile's live rows alone, so neither holds a row more.
  const uint32_t widest = std::max(shape.hiddenSize, shape.expertIntermediateSize);
  return {routes * sizeof(uint32_t), routes * sizeof(float),
          uint64_t{tiles} * sizeof(MoeTileDescriptor), sizeof(MoeTileCount),
          routes * sizeof(uint32_t), routes * sizeof(uint32_t), uint64_t{shape.experts} * sizeof(uint32_t),
          std::max(tableBytes(shape.hiddenSize, matmulRows), scoreBytes),
          matmulRows * shape.expertIntermediateSize * sizeof(uint16_t),
          routes * shape.hiddenSize * sizeof(uint16_t),
          int8Passes ? gguf_prefill_int8_table_bytes(shape.hiddenSize, rows) : 0,
          int8Passes ? gguf_prefill_int8_scale_bytes(shape.hiddenSize, rows) : 0,
          int8Passes ? gguf_prefill_int8_table_bytes(widest, routes) : 0,
          int8Passes ? gguf_prefill_int8_scale_bytes(widest, routes) : 0};
}

// The shape's compiled routing (MoeShape::compiledRouting).
const CompiledMoeRouting &compiledRouting(MoeShape shape) {
  if (const CompiledMoeRouting *compiled = shape.compiledRouting()) return *compiled;
  throw std::invalid_argument("no MoE kernel routes the shape's router, shared slices and experts");
}

// A routing kernel's instance for the shape's router width (past
// SPLASH_MOE_EXPERT_SLOTS experts, the softmax routing's wide one) and its
// compiled routing's `suffix`.
std::string routingKernel(std::string name, MoeShape shape, std::string_view CompiledMoeRouting::*suffix) {
  static_assert(SPLASH_MOE_WIDE_EXPERT_SLOTS == 512, "the wide routing kernels are the _e512 instances");
  if (shape.routerWidth() == SPLASH_MOE_WIDE_EXPERT_SLOTS) name += "_e512";
  name += compiledRouting(shape).*suffix;
  return name;
}

// The formats an expert pass decodes, its pipeline's function constants:
// each once in ascending order, the last repeated past them, so passes of one
// format set share a pipeline (moe_gguf.metal, moe_formats).
template <size_t Count>
std::array<uint32_t, Count> passFormats(std::array<uint32_t, Count> formats) {
  std::sort(formats.begin(), formats.end());
  const auto distinct = std::unique(formats.begin(), formats.end());
  std::fill(distinct, formats.end(), distinct[-1]);
  return formats;
}

// The expert passes over the grouped tiles `range` names (MOE_TILES_*), each
// on the grid of those live tiles that the grouping wrote (MoeTileCount),
// within the column tiles of `group` and maximumTiles(), the routed experts in
// the slots `slots` names: gate and up in one pass
// (moe_expert_gguf_m<rows>_gate_up[_mixed], whose simdgroups each decode
// their own pass's formats) from the grouped rows the gather writes into
// groupedInput to expertIntermediate, then down to expertOutput
// (moe_expert_gguf_m<rows>_down), each times its experts' scales
// (BlockMoeWeights::expertScales).
void addExperts(metal::CommandGraph &graph, const MoeScratch &scratch, const BlockMoeWeights &weights,
                const metal::MetalBuffer &slots, const MoePlan &plan, const MoeGroupParams &group, uint32_t range) {
  const MoeShape shape = plan.shape();
  const uint32_t tiles = plan.maximumTiles();
  const std::string kernel = "moe_expert_gguf_m" + std::to_string(plan.tileRows());
  // The passes' instances for the shape's shared slices.
  const std::string slices(compiledRouting(shape).slicesSuffix);
  const size_t gateUpOffset = range == MOE_TILES_LOADED     ? offsetof(MoeTileCount, loaded_gate_up_grid)
                              : range == MOE_TILES_RESIDENT ? offsetof(MoeTileCount, resident_gate_up_grid)
                                                            : offsetof(MoeTileCount, gate_up_grid);
  const size_t downOffset = range == MOE_TILES_LOADED     ? offsetof(MoeTileCount, loaded_down_grid)
                            : range == MOE_TILES_RESIDENT ? offsetof(MoeTileCount, resident_down_grid)
                                                          : offsetof(MoeTileCount, down_grid);
  const metal::IndirectGrid gateUpGrid{scratch.tileCount, gateUpOffset};
  const metal::IndirectGrid downGrid{scratch.tileCount, downOffset};
  const uint32_t floorSlots = weights.chunks.floorSlots(), chunkSlots = weights.chunks.chunkSlots();
  const uint32_t hidden = shape.hiddenSize;
  const uint32_t intermediate = shape.expertIntermediateSize;
  // A block without a shared expert binds its slot table in the shared
  // planes' slots and decodes its routed formats alone.
  const auto sharedOf = [&](const BlockExpertProjection &p) {
    if (shape.sharedSlices) return p;
    return BlockExpertProjection{p.routed, QuantizedSegment::planes(p.routed.formatId, 0, p.routed.inputSize, slots,
                                                                    slots, slots)};
  };
  const BlockExpertProjection gate = sharedOf(weights.gate), up = sharedOf(weights.up), down = sharedOf(weights.down);
  // One pair table where gate and up share their formats, one each otherwise.
  const bool mixed = gate.routed.formatId != up.routed.formatId || gate.shared.formatId != up.shared.formatId;
  const std::string activation(compiledRouting(shape).activationSuffix);
  graph.add(kernel + (mixed ? "_gate_up_mixed" : "_gate_up") + slices + activation,
            {scratch.groupedInput, scratch.tileDescriptors, scratch.tileCount,
             gate.shared.plane0, gate.shared.plane1Slot(), gate.shared.meta,
             up.shared.plane0, up.shared.plane1Slot(), up.shared.meta,
             scratch.expertIntermediate, slots, weights.expertScales},
            MoeGgufGateUpParams{hidden, intermediate, shape.experts, gate.routed.formatId, gate.shared.formatId,
                                up.routed.formatId, up.shared.formatId, floorSlots, chunkSlots, range},
            {group.gate_up_columns, tiles, 1}, gateUpGrid, {GGUF_STAGED_THREADS, 1, 1});
  graph.specialize(passFormats<4>({gate.routed.formatId, gate.shared.formatId, up.routed.formatId, up.shared.formatId}));
  graph.addresses(weights.chunks.planes());
  // Down, projection 2 of the routed experts, which it reaches in the chunks.
  graph.add(kernel + "_down" + slices,
            {scratch.expertIntermediate, scratch.tileDescriptors, scratch.tileCount, down.shared.plane0,
             down.shared.plane1Slot(), down.shared.meta, scratch.expertOutput, slots, weights.expertScales},
            MoeGgufExpertParams{intermediate, hidden, shape.experts, down.routed.formatId, down.shared.formatId,
                                floorSlots, chunkSlots, range},
            {group.down_columns, tiles, 1}, downGrid, {GGUF_STAGED_THREADS, 1, 1});
  graph.specialize(passFormats<2>({down.routed.formatId, down.shared.formatId}));
  graph.addresses(weights.chunks.planes());
}

} // namespace

MoePlan::MoePlan(MoeShape shape, uint32_t rows, MoeConfig config)
    : shape_(shape), rows_(rows), config_(config) {
  // The int8 passes run an expert tile's rows in one int8 tile (moe_expert_int8).
  if (config.int8Passes && tileRows() > GGUF_PREFILL_INT8_ROWS)
    throw std::invalid_argument("invalid MoE int8 passes: expert tiles past the int8 tile's rows");
  workspace_ = workspaceFor(shape, rows, tileRows(), config.int8Passes);
  maximumTiles_ = moeMaximumTiles(rows, shape, tileRows());
}

namespace {

// A step's grouping into tiles, which its expert passes read: the gate/up
// pass's column tiles are its staged columns. A streamed block's grouping is
// that of a step whose passes run around its load (MoE::add).
MoeGroupParams groupParams(const MoePlan &plan, const BlockMoeWeights &weights) {
  const MoeShape shape = plan.shape();
  return {plan.rows(),
          shape.expertsPerToken,
          plan.tileRows(),
          shape.experts,
          shape.expertIntermediateSize / GGUF_STAGED_COLUMNS,
          shape.hiddenSize / GGUF_TILE_COLUMNS,
          weights.stream != nullptr,
          0};
}

void addGather(metal::CommandGraph &graph, const MoeBuffers &buffers, const MoePlan &plan) {
  const MoeShape shape = plan.shape();
  const MoeScratch &scratch = buffers.scratch;
  graph.add("moe_gather_rows",
            {buffers.input, scratch.groupedRoutes, scratch.tileDescriptors,
             scratch.tileCount, scratch.groupedInput},
            MoeGatherParams{shape.hiddenSize, shape.routesPerToken()},
            {plan.maximumTiles(), shape.hiddenSize / 256, 1});
}

// A resident block's passes on int8 activations (MoeConfig::int8Passes):
// the chunk's input rows quantized as the dense prefill quantizes them (per
// row and 64 inputs), their codes and scales gathered by route, the gate into
// expertIntermediate, up times the gate's SiLU over it (both bf16, as the
// staged gate/up pass rounds them; a thread reads each gate element before
// it writes the product there), the intermediate quantized into the grouped
// codes the gate and up passes have read, and down into expertOutput, which
// the combine reads; each pass times its experts' scales
// (BlockMoeWeights::expertScales). The grouped codes and scales hold a row
// per route, the scales' row stride. Each pass is named by its projection's
// format, which its routed and shared experts share where the plan has the
// passes (moeInt8Experts).
void addInt8Experts(metal::CommandGraph &graph, const MoeBuffers &buffers, const BlockMoeWeights &weights,
                    const MoePlan &plan) {
  const MoeShape shape = plan.shape();
  const MoeScratch &scratch = buffers.scratch;
  const uint32_t rows = plan.rows(), hidden = shape.hiddenSize, intermediate = shape.expertIntermediateSize;
  const uint32_t routes = rows * shape.routesPerToken();
  addGgufPrefillInt8Quantize(graph, buffers.input, scratch.inputCodes, scratch.inputScales, hidden, rows, rows);
  graph.add("moe_gather_int8",
            {scratch.inputCodes, scratch.inputScales, scratch.groupedRoutes, scratch.tileDescriptors,
             scratch.tileCount, scratch.groupedCodes, scratch.groupedScales},
            MoeGatherInt8Params{hidden, shape.routesPerToken(), rows, routes},
            {plan.maximumTiles(), hidden / 256, 1});
  const std::string slices(compiledRouting(shape).slicesSuffix);
  const auto pass = [&](const char *name, const BlockExpertProjection &p, uint32_t k, uint32_t n,
                        const metal::MetalBuffer &output) {
    graph.add(std::string("moe_expert_int8_") + p.routed.name() + "_" + name + slices,
              {scratch.groupedCodes, scratch.groupedScales, scratch.tileDescriptors, scratch.tileCount,
               p.shared.plane0, p.shared.plane1Slot(), p.shared.meta, output, weights.slotTable,
               weights.expertScales},
              MoeInt8ExpertParams{k, n, shape.experts, p.routed.formatId, p.shared.formatId,
                                  weights.chunks.floorSlots(), weights.chunks.chunkSlots(), MOE_TILES_ALL, routes},
              {n / GGUF_TILE_COLUMNS, plan.maximumTiles(), 1}, {GGUF_PREFILL_INT8_THREADS, 1, 1});
    graph.addresses(weights.chunks.planes());
  };
  pass("gate", weights.gate, hidden, intermediate, scratch.expertIntermediate);
  pass("up", weights.up, hidden, intermediate, scratch.expertIntermediate);
  addGgufPrefillInt8Quantize(graph, scratch.expertIntermediate, scratch.groupedCodes, scratch.groupedScales,
                             intermediate, routes, routes);
  pass("down", weights.down, intermediate, hidden, scratch.expertOutput);
}

// The combine, or with `convolution` the combine running a verify step's
// convolution of its rows (MoeCombineConvolution).
void addCombine(metal::CommandGraph &graph, const MoeBuffers &buffers, const MoePlan &plan,
                const MoeCombineConvolution *convolution) {
  const MoeShape shape = plan.shape();
  const MoeScratch &scratch = buffers.scratch;
  const MoeCombineParams params{plan.rows(), shape.hiddenSize, shape.routesPerToken()};
  if (convolution) {
    MoE::addCombineConvolution(
        graph, {scratch.expertOutput, scratch.routeRows, scratch.routingWeights, buffers.residual, params},
        *convolution);
    return;
  }
  graph.add("moe_combine",
            {scratch.expertOutput, scratch.routeRows, scratch.routingWeights,
             buffers.residual, buffers.output},
            params, {plan.rows(), shape.hiddenSize / 256, 1});
}

} // namespace

void MoE::addRouting(metal::CommandGraph &graph, const MoeBuffers &buffers, const BlockMoeWeights &weights,
                     const MoePlan &plan) {
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
  // A decode step's lanes' rows, which the selector writes on the device.
  MoeGroupParams group = groupParams(plan, weights);
  const uint32_t lanes = buffers.lanes;
  if (lanes > SPLASH_MAXIMUM_BATCH_WIDTH) throw std::invalid_argument("a MoE step has more lanes than a batch");
  if (lanes) requireBytes(buffers.laneRows, uint64_t{lanes} * sizeof(SplashLaneRows), "MoE lane rows");
  group.lanes = lanes;
  MoE::addRoutes(graph, buffers.input, weights.router, weights.sharedScalarGate, weights.selectionBias,
                 scratch.groupedInput, scratch.selectedExperts, scratch.routingWeights, rows, shape,
                 plan.configuration().ggufRouterTile);
  // The expert passes launch the live tiles alone, on grids the grouping
  // writes with the tile count: a one-lane 35B decode step fills a median of
  // 30 of the 65 tiles it could, and the idle ones' threadgroups, returning at
  // once, still cost up to 4.8% of a decode cycle (40-core M3 Max). A
  // streamed grouping puts the tiles of experts the slot table marks absent
  // last (MoeTileCount) and copies the table into expertSlots.
  std::vector<metal::MetalBuffer> grouping{scratch.selectedExperts, scratch.tileDescriptors, scratch.tileCount,
                                           scratch.groupedRoutes, scratch.routeRows, weights.slotTable,
                                           scratch.expertSlots};
  if (lanes) grouping.push_back(buffers.laneRows);
  graph.add(routingKernel(lanes ? "moe_group_routes_verified" : "moe_group_routes", shape,
                          &CompiledMoeRouting::slicesSuffix),
            std::move(grouping), group, {1, 1, 1}, {shape.routerWidth(), 1, 1});
}

// A streamed block's gather and passes over the experts resident before its
// load, on the grouping's copy of their slots.
void MoE::addResident(metal::CommandGraph &graph, const MoeBuffers &buffers,
                      const BlockMoeWeights &weights, const MoePlan &plan) {
  if (!weights.stream) throw std::invalid_argument("a resident MoE block's experts run after its routing alone");
  addGather(graph, buffers, plan);
  addExperts(graph, buffers.scratch, weights, buffers.scratch.expertSlots, plan, groupParams(plan, weights),
             MOE_TILES_RESIDENT);
}

// A streamed block's passes over the experts its load brought, on the load
// slot table, after addResident's beside the load; a resident block's gather
// and passes over every expert. Then the combine.
void MoE::addRouted(metal::CommandGraph &graph, const MoeBuffers &buffers,
                    const BlockMoeWeights &weights, const MoePlan &plan,
                    const MoeCombineConvolution *convolution) {
  const MoeGroupParams group = groupParams(plan, weights);
  if (weights.stream) {
    addExperts(graph, buffers.scratch, weights, weights.loadSlotTable, plan, group, MOE_TILES_LOADED);
  } else if (plan.configuration().int8Passes) {
    addInt8Experts(graph, buffers, weights, plan);
  } else {
    addGather(graph, buffers, plan);
    addExperts(graph, buffers.scratch, weights, weights.slotTable, plan, group, MOE_TILES_ALL);
  }
  addCombine(graph, buffers, plan, convolution);
}

void MoE::addCombineConvolution(metal::CommandGraph &graph, const MoeCombine &combine,
                                const MoeCombineConvolution &convolution) {
  const MoeCombineConvolution &c = convolution;
  // A lane's rows are a threadgroup's, its verify block's at most.
  requireLayoutRows(c.rows, MOE_COMBINE_CONVOLUTION_ROWS);
  const MoeCombineParams &sums = combine.params;
  if (c.shape.channels % MOE_COMBINE_CONVOLUTION_CHANNELS || sums.hidden_size != c.shape.channels ||
      sums.rows != c.rows.rows || !sums.routes_per_row)
    throw std::invalid_argument("invalid MoE combine of a short convolution");
  const uint64_t routes = uint64_t{c.rows.rows} * sums.routes_per_row;
  requireBytes(combine.routeRows, routes * sizeof(uint32_t), "MoE combine route rows");
  requireBytes(combine.routingWeights, routes * sizeof(float), "MoE combine routing weights");
  requireBytes(combine.residual, uint64_t{c.rows.rows} * sums.hidden_size * sizeof(uint16_t), "MoE combine residual");
  const ShortConvolutionParams p =
      ShortConvolution::verifyParams(c.buffers, c.shape, c.stateIn, c.state, c.layout, c.rows, c.tree, c.tape);
  const auto in = laneBindings(c.stateIn);
  graph.add("moe_combine_convolution",
            {combine.expertOutput, combine.routeRows, combine.routingWeights, combine.residual, in[0], in[1], in[2],
             in[3], in[4], in[5], in[6], in[7], c.buffers.taps, c.buffers.residual, c.buffers.output, c.layout, c.tree,
             c.tape},
            MoeCombineConvolutionParams{sums, p},
            {c.shape.channels / MOE_COMBINE_CONVOLUTION_CHANNELS, c.rows.sequences, 1},
            {MOE_COMBINE_CONVOLUTION_CHANNELS, c.rows.bound, 1});
}

void MoE::addRoutes(metal::CommandGraph &graph, const metal::MetalBuffer &input, const QuantizedSegment &router,
                    const QuantizedSegment &sharedScalarGate, const metal::MetalBuffer &selectionBias,
                    const metal::MetalBuffer &scores, const metal::MetalBuffer &selected,
                    const metal::MetalBuffer &routingWeights, uint32_t rows, MoeShape shape, FloatTile routerTile) {
  // The router's fp32 scores, in the format its checkpoint holds it
  // (BlockMoeWeights), in rows of routerWidth() expert slots, as the select
  // kernel of that width reads, which weights gated shared slices by the
  // sigmoid of the input's product with the scalar gate.
  const uint32_t width = shape.routerWidth();
  addGgufFloat(graph, input, router, scores, rows, width, 0, FloatOutput::Float32, routerTile);
  const MoeRouteParams route{rows, shape.hiddenSize, shape.experts, shape.expertsPerToken};
  const std::string select = routingKernel("moe_route_select_f32", shape, &CompiledMoeRouting::selectSuffix);
  if (shape.selectionBias) {
    graph.add(select, {scores, selectionBias, selected, routingWeights}, MoeBiasedRouteParams{route, shape.routedScale},
              {rows, 1, 1}, {width, 1, 1});
    return;
  }
  std::vector<metal::MetalBuffer> bindings{scores};
  if (shape.sharedGate) bindings.insert(bindings.end(), {input, sharedScalarGate.plane0});
  bindings.insert(bindings.end(), {selected, routingWeights});
  graph.add(select, std::move(bindings), route, {rows, 1, 1}, {width, 1, 1});
}

uint32_t MoE::expertLoadGroups(const MoeExpertLoadParams &params) noexcept {
  uint64_t groups = 0;
  // A row tile's spans of groups each (moe_expert_load).
  for (const MoeExpertLoadProjection &projection : params.projections)
    groups = std::max(groups, (uint64_t{projection.rows} + QUANT_TILE_ROWS - 1) / QUANT_TILE_ROWS *
                                  ((projection.columns / 32 + MOE_EXPERT_LOAD_SPAN - 1) / MOE_EXPERT_LOAD_SPAN));
  return static_cast<uint32_t>(groups);
}

void MoE::addExpertLoad(metal::CommandGraph &graph, const metal::MetalBuffer &staging,
                        const ExpertChunks &chunks, const metal::MetalBuffer &slots,
                        const metal::MetalBuffer &sources, const MoeExpertLoadParams &params, uint32_t jobs,
                        const metal::IndirectGrid &grid) {
  if (chunks.empty()) throw std::invalid_argument("MoE expert chunks have no slots");
  if (params.source_planes != 1 && params.source_planes != 3)
    throw std::invalid_argument("a MoE expert load's jobs name 1 or 3 staged planes a projection");
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

MoePlan MoE::plan(MoeShape shape, uint32_t rows, MoeConfig config) {
  if (!rows || rows > SPLASH_PREFILL_TOKEN_BUDGET)
    throw std::invalid_argument("invalid MoE rows");
  return MoePlan(shape, rows, config);
}

} // namespace splash::ops

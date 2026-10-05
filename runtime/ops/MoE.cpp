#include "ops/MoE.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "metal/abi/Linear.h"
#include "metal/abi/MoE.h"
#include "Checked.hpp"
#include "ops/BufferExtent.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace splash::ops {
namespace {

static_assert(offsetof(MoeExpertParams, expert_stride_bytes_0) == 16);

// An affine Q8 projection of output x input weights, a byte each, with a
// bf16 scale and bias per 64 of them.
void requireProjection(const Q8Projection &projection, uint32_t output, uint32_t input, const std::string &name) {
  if (projection.outputSize != output || projection.inputSize != input)
    throw std::invalid_argument(name + " does not match the plan");
  const uint64_t weights = uint64_t{output} * input;
  requireBytes(projection.planes.weights, weights, name + " weight");
  requireBytes(projection.planes.scales, weights / 32, name + " scale");
  requireBytes(projection.planes.biases, weights / 32, name + " bias");
}

// `experts` Q4 slabs of output x input weights, expertStrideBytes apart, each
// its weights, then a bf16 scale and bias per 64 of them. The kernels read a
// slab at each stride, so padding may follow every slab but the last.
void requireExperts(const ExpertProjection &projection, uint32_t experts, uint32_t output, uint32_t input,
                    const std::string &name) {
  const uint64_t weights = uint64_t{output} * input, slab = weights / 2 + weights / 16;
  const uint64_t stride = projection.expertStrideBytes;
  if (projection.experts != experts || projection.outputSize != output || projection.inputSize != input ||
      stride < slab || stride % sizeof(uint16_t))
    throw std::invalid_argument(name + " does not match the plan");
  // The end of the last slab, past any buffer when it overflows.
  uint64_t last = 0, bytes = 0;
  if (!checkedMultiply(uint64_t{experts - 1}, stride, last) || !checkedAdd(last, slab, bytes))
    bytes = std::numeric_limits<uint64_t>::max();
  requireBytes(projection.packed, bytes, name);
}

// A GGUF tensor of output x input weights, and the planes of its GGUF_FMT_*
// format; what reads a float tensor checks its fp32 values.
void requireSegment(const QuantizedSegment &segment, uint32_t output, uint32_t input, bool floatWeights,
                    const std::string &name) {
  if (segment.isFloat() != floatWeights || (!floatWeights && segment.formatId >= GGUF_FMT_COUNT) ||
      segment.outputSize != output || segment.inputSize != input)
    throw std::invalid_argument(name + " does not match the plan");
  if (!floatWeights) requireSegmentPlanes(segment, name);
}

// Every weight the plan's dispatches read. The router of a GGUF block runs
// through addGgufFloat, which checks its weights.
void validate(const MoeWeights &weights, MoeShape shape) {
  if (shape.weightLayout != weights.layout())
    throw std::invalid_argument("MoE weight layout does not match plan");
  const uint32_t hidden = shape.hiddenSize, intermediate = shape.expertIntermediateSize;
  if (shape.weightLayout == WeightLayout::Block32) {
    const BlockMoeWeights &blocks = weights.blocks();
    requireSegment(blocks.router, shape.experts, hidden, true, "MoE router");
    requireSegment(blocks.sharedScalarGate, 1, hidden, true, "MoE shared scalar gate");
    // The select kernel reads the scalar gate's fp32 weights.
    requireBytes(blocks.sharedScalarGate.plane0, uint64_t{hidden} * sizeof(float), "MoE shared scalar gate weight");
    for (const auto &[projection, name, output, input] :
         {std::tuple{&blocks.gate, "gate", intermediate, hidden}, std::tuple{&blocks.up, "up", intermediate, hidden},
          std::tuple{&blocks.down, "down", hidden, intermediate}}) {
      requireSegment(projection->routed, shape.experts * output, input, false, std::string("MoE expert ") + name);
      requireSegment(projection->shared, output, input, false, std::string("MoE shared ") + name);
    }
    return;
  }
  const AffineMoeWeights &affine = weights.affine();
  // The router's rows are the expert slots; the shared expert's scalar gate
  // pads its one row to a storage tile.
  requireProjection(affine.router, SPLASH_MOE_EXPERT_SLOTS, hidden, "MoE router");
  requireProjection(affine.sharedScalarGate, SPLASH_AFFINE_TILE_ROWS, hidden, "MoE shared scalar gate");
  requireExperts(affine.expertGate, shape.experts, intermediate, hidden, "MoE expert gate");
  requireExperts(affine.expertUp, shape.experts, intermediate, hidden, "MoE expert up");
  requireExperts(affine.expertDown, shape.experts, hidden, intermediate, "MoE expert down");
  requireExperts(affine.sharedGate, 1, intermediate, hidden, "MoE shared gate");
  requireExperts(affine.sharedUp, 1, intermediate, hidden, "MoE shared up");
  requireExperts(affine.sharedDown, 1, hidden, intermediate, "MoE shared down");
}

MoeWorkspace workspaceFor(MoeShape shape, uint32_t rows, uint32_t tileRows,
                          bool splitExperts, MoeGgufTile ggufTile) {
  if (!shape.valid())
    throw std::invalid_argument("invalid MoE workspace shape");
  const uint64_t routes = uint64_t{rows} * shape.routesPerToken();
  const uint32_t tiles = moeMaximumTiles(rows, shape, tileRows);
  const uint64_t groupedRows = uint64_t{tiles} * tileRows;
  const uint32_t widest = std::max(shape.hiddenSize, shape.expertIntermediateSize);
  const uint32_t outputWidth = splitExperts ? widest : shape.hiddenSize;
  // The router's fp32 scores, a row of expert slots per row, live in the
  // grouped input until the gather overwrites them. Register plans also hold
  // the down pass's Table16 tiles there.
  const uint64_t scoreBytes = uint64_t{rows} * SPLASH_MOE_EXPERT_SLOTS * sizeof(float);
  const bool table16 = ggufTile == MoeGgufTile::Register;
  const uint64_t sumsBytes = table16 ? tableSumsBytes(LinearInput::Table16, widest, groupedRows) : 0;
  return {routes * sizeof(uint32_t), routes * sizeof(float),
          uint64_t{tiles} * sizeof(MoeTileDescriptor), sizeof(uint32_t),
          groupedRows * sizeof(uint32_t), routes * sizeof(uint32_t),
          std::max(tableBytes(table16 ? widest : shape.hiddenSize, groupedRows), scoreBytes),
          groupedRows * shape.expertIntermediateSize * sizeof(uint16_t),
          groupedRows * outputWidth * sizeof(uint16_t), sumsBytes};
}

// Pipelines, column tiles and threadgroup width of a decode plan's fused
// gate/up and down passes; see MoeExpertSimdgroups for the four-simdgroup
// form's geometry and measurements.
struct ExpertPasses final {
  const char *gateUp;
  const char *down;
  uint32_t gateUpColumns;
  uint32_t downColumns;
  uint32_t threads;
};

ExpertPasses fusedExpertPasses(const MoeConfig &config) noexcept {
  const uint32_t threads = static_cast<uint32_t>(config.m8Simdgroups) * 32;
  if (config.m8Simdgroups == MoeExpertSimdgroups::Four)
    return {"moe_expert_gate_up_q4_m8_n128_sg4",
            "moe_expert_down_q4_m8_n256_sg4", 128, 256, threads};
  return {"moe_expert_gate_up_q4_m8", "moe_expert_down_q4_m8", 128, 128,
          threads};
}

void addAffineExperts(metal::CommandGraph &graph, const MoeScratch &scratch,
                      const AffineMoeWeights &weights, const MoePlan &plan) {
  const MoeShape shape = plan.shape();
  const uint32_t tiles = plan.maximumTiles();
  // The two expert strides are the gate and up slabs of the fused tile; a
  // single-matrix pass reads only the first, so its params repeat one stride.
  const MoeExpertParams gateUp{shape.hiddenSize, shape.expertIntermediateSize,
                               shape.experts, 0,
                               weights.expertGate.expertStrideBytes,
                               weights.expertUp.expertStrideBytes};
  const MoeExpertParams gate{shape.hiddenSize, shape.expertIntermediateSize,
                             shape.experts, 0,
                             weights.expertGate.expertStrideBytes,
                             weights.expertGate.expertStrideBytes};
  const MoeExpertParams up{shape.hiddenSize, shape.expertIntermediateSize,
                           shape.experts, 0, weights.expertUp.expertStrideBytes,
                           weights.expertUp.expertStrideBytes};
  const MoeExpertParams down{shape.expertIntermediateSize, shape.hiddenSize,
                             shape.experts, 0,
                             weights.expertDown.expertStrideBytes,
                             weights.expertDown.expertStrideBytes};
  if (plan.splitExperts()) {
    // The gate lands in expertOutput, which the down pass overwrites only
    // after the up pass has consumed it.
    graph.add("prefill_moe_expert_q4_n256_m32",
              {scratch.groupedInput, scratch.tileDescriptors,
               scratch.tileCount, weights.expertGate.packed,
               weights.sharedGate.packed, scratch.expertOutput},
              gate, {shape.expertIntermediateSize / 256, tiles, 1});
    graph.add("prefill_moe_expert_q4_n256_up_silu_m32",
              {scratch.groupedInput, scratch.tileDescriptors,
               scratch.tileCount, weights.expertUp.packed,
               weights.sharedUp.packed, scratch.expertOutput,
               scratch.expertIntermediate},
              up, {shape.expertIntermediateSize / 256, tiles, 1});
    graph.add("prefill_moe_expert_q4_n256_m32",
              {scratch.expertIntermediate, scratch.tileDescriptors,
               scratch.tileCount, weights.expertDown.packed,
               weights.sharedDown.packed, scratch.expertOutput},
              down, {shape.hiddenSize / 256, tiles, 1});
  } else {
    // The workspace holds the same grouped rows whatever the column tile;
    // only the grid's column count and the threadgroup width follow it.
    const ExpertPasses passes = fusedExpertPasses(plan.configuration());
    graph.add(passes.gateUp,
              {scratch.groupedInput, scratch.tileDescriptors,
               scratch.tileCount, weights.expertGate.packed,
               weights.expertUp.packed, weights.sharedGate.packed,
               weights.sharedUp.packed, scratch.expertIntermediate},
              gateUp,
              {shape.expertIntermediateSize / passes.gateUpColumns, tiles, 1},
              {passes.threads, 1, 1});
    graph.add(passes.down,
              {scratch.expertIntermediate, scratch.tileDescriptors,
               scratch.tileCount, weights.expertDown.packed,
               weights.sharedDown.packed, scratch.expertOutput},
              down, {shape.hiddenSize / passes.downColumns, tiles, 1},
              {passes.threads, 1, 1});
  }
}

// The three GGUF expert passes over grouped tiles: gate into expertOutput
// (the down pass overwrites it after the up pass consumed it), up with
// silu(gate) into expertIntermediate, down into expertOutput. Register plans
// read Table16 tiles from groupedInput: the gather writes the gate/up input's
// and a prepare dispatch the down input's.
void addGgufExperts(metal::CommandGraph &graph, const MoeScratch &scratch,
                    const BlockMoeWeights &weights, const MoePlan &plan) {
  const MoeShape shape = plan.shape();
  const uint32_t tiles = plan.maximumTiles();
  const bool table16 = plan.configuration().ggufTile == MoeGgufTile::Register;
  const auto pass = [&](const BlockExpertProjection &projection, bool up,
                        const metal::MetalBuffer &input,
                        const metal::MetalBuffer &output, uint32_t n, uint32_t k) {
    std::vector<metal::MetalBuffer> bindings{input};
    if (table16) bindings.push_back(scratch.groupedSums);
    bindings.insert(bindings.end(),
                    {scratch.tileDescriptors, scratch.tileCount,
                     projection.routed.plane0, projection.routed.plane1Slot(),
                     projection.routed.meta, projection.shared.plane0,
                     projection.shared.plane1Slot(), projection.shared.meta, output,
                     scratch.expertOutput});
    const std::string kernel = table16 ? "moe_expert_gguf_sg" : "moe_expert_gguf_m" + std::to_string(plan.tileRows());
    graph.add(kernel + (up ? "_g" : "_a"), std::move(bindings),
              MoeGgufExpertParams{k, n, shape.experts, projection.routed.formatId,
                                  projection.shared.formatId},
              {n / GGUF_TILE_COLUMNS, tiles, 1}, {table16 ? GGUF_REGISTER_THREADS : GGUF_STAGED_THREADS, 1, 1});
  };
  const uint32_t hidden = shape.hiddenSize;
  const uint32_t intermediate = shape.expertIntermediateSize;
  pass(weights.gate, false, scratch.groupedInput, scratch.expertOutput,
       intermediate, hidden);
  pass(weights.up, true, scratch.groupedInput, scratch.expertIntermediate,
       intermediate, hidden);
  if (table16)
    graph.add("moe_prepare_table16",
              {scratch.expertIntermediate, scratch.tileCount,
               scratch.groupedInput, scratch.groupedSums},
              intermediate, {tiles, intermediate / 256, 1});
  pass(weights.down, false,
       table16 ? scratch.groupedInput : scratch.expertIntermediate,
       scratch.expertOutput, hidden, intermediate);
}

} // namespace

// Prefill plans and GGUF plans run the three expert passes of the split
// plan, affine decode plans the fused gate/up tile.
MoePlan::MoePlan(MoeShape shape, uint32_t rows, MoeConfig config,
                 MoePhase phase)
    : shape_(shape), rows_(rows), config_(config),
      splitExperts_(shape.weightLayout == WeightLayout::Block32 || phase == MoePhase::Prefill) {
  // Affine plans have 32-row prefill and 8-row decode kernels, GGUF plans
  // 8-row kernels in both phases and 32-row prefill kernels.
  const bool gguf = shape.weightLayout == WeightLayout::Block32;
  const bool prefill = phase == MoePhase::Prefill;
  if ((!prefill && config.expertTile != MoeExpertTile::M8) ||
      (!gguf && prefill && config.expertTile != MoeExpertTile::M32))
    throw std::invalid_argument("invalid MoE expert tile configuration");
  if (config.ggufTile == MoeGgufTile::Register &&
      (shape.weightLayout != WeightLayout::Block32 || config.expertTile != MoeExpertTile::M8))
    throw std::invalid_argument("the register expert tile takes block 8-row tiles");
  workspace_ = workspaceFor(shape, rows, tileRows(), splitExperts_, config.ggufTile);
  maximumTiles_ = moeMaximumTiles(rows, shape, tileRows());
}

void MoE::add(metal::CommandGraph &graph, const MoeBuffers &buffers,
              const MoeWeights &weights, const MoePlan &plan) {
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
  const bool block = weights.layout() == WeightLayout::Block32;
  if (block) {
    // fp32 scores of the F32 router in rows of expert slots, as the select
    // kernel reads.
    addGgufFloat(graph, buffers.input, weights.blocks().router, scratch.groupedInput, rows,
                 SPLASH_MOE_EXPERT_SLOTS, 0, FloatOutput::Float32, plan.configuration().ggufRouterTile);
    graph.add("moe_route_select_f32",
              {scratch.groupedInput, buffers.input,
               weights.blocks().sharedScalarGate.plane0, scratch.selectedExperts,
               scratch.routingWeights},
              routeParams, {rows, 1, 1}, {SPLASH_MOE_EXPERT_SLOTS, 1, 1});
  } else {
    const AffineMoeWeights &affine = weights.affine();
    const MoeRouteTile route = moeRouteTile(rows, plan.configuration().routeWideRows);
    graph.add(route.rows == 8 ? "moe_route_scores_q8_m8"
                              : "moe_route_scores_q8_m32",
              {buffers.input, affine.router.planes.weights, affine.router.planes.scales,
               affine.router.planes.biases, scratch.groupedInput},
              routeParams,
              {(rows + route.rows - 1) / route.rows, SPLASH_MOE_EXPERT_SLOTS / route.experts, 1});
    graph.add("moe_route_select_q8",
              {scratch.groupedInput, buffers.input,
               affine.sharedScalarGate.planes.weights,
               affine.sharedScalarGate.planes.scales,
               affine.sharedScalarGate.planes.biases, scratch.selectedExperts,
               scratch.routingWeights},
              routeParams, {rows, 1, 1}, {SPLASH_MOE_EXPERT_SLOTS, 1, 1});
  }
  graph.add("moe_group_routes",
            {scratch.selectedExperts, scratch.tileDescriptors,
             scratch.tileCount, scratch.groupedRoutes, scratch.routeRows},
            MoeGroupParams{rows, shape.expertsPerToken, tileRows,
                           shape.experts},
            {1, 1, 1}, {SPLASH_MOE_EXPERT_SLOTS, 1, 1});
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
  if (block)
    addGgufExperts(graph, scratch, weights.blocks(), plan);
  else
    addAffineExperts(graph, scratch, weights.affine(), plan);
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

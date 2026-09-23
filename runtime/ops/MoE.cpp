#include "ops/MoE.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/MoE.h"
#include "metal/abi/QuantFormat.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace splash::ops {
namespace {

static_assert(offsetof(MoeExpertParams, expert_stride_bytes_0) == 16);

bool matches(const Q8Projection &projection, uint32_t output,
             uint32_t input) noexcept {
  const uint64_t elements = uint64_t{output} * input;
  const uint64_t parameterBytes = elements / 32;
  return projection.weights && projection.scales && projection.biases &&
         projection.outputSize == output && projection.inputSize == input &&
         projection.weights.sizeBytes() >= elements &&
         projection.scales.sizeBytes() >= parameterBytes &&
         projection.biases.sizeBytes() >= parameterBytes;
}

bool matches(const ExpertQ4Projection &projection, uint32_t experts,
             uint32_t output, uint32_t input) noexcept {
  if (!projection.packed || projection.experts != experts || !experts ||
      projection.outputSize != output || projection.inputSize != input)
    return false;
  const uint64_t elements = uint64_t{output} * input;
  const uint64_t payloadBytes = elements / 2 + elements / 16;
  const uint64_t stride = projection.expertStrideBytes;
  const uint64_t available = projection.packed.sizeBytes();
  if (!stride || stride < payloadBytes || stride % sizeof(uint16_t) ||
      available < payloadBytes)
    return false;
  // The shader reads [weights][BF16 scales][BF16 biases] at each stride.
  // Allow padding between experts, without requiring it after the last one.
  // Division proves the last payload fits without overflowing expert*stride.
  return uint64_t{experts - 1} <= (available - payloadBytes) / stride;
}

// A float segment holds [output][input] floats in plane0; a quantized one
// its planes in a GGUF_FMT_* format.
bool matches(const GgufSegment &segment, uint32_t output, uint32_t input,
             bool floatWeights) noexcept {
  if (!segment.plane0 || segment.isFloat() != floatWeights ||
      segment.outputSize != output || segment.inputSize != input)
    return false;
  return floatWeights
             ? segment.plane0.sizeBytes() >= uint64_t{output} * input * sizeof(float)
             : segment.formatId < GGUF_FMT_COUNT && segment.meta;
}

bool matches(const GgufExpertProjection &projection, uint32_t experts,
             uint32_t output, uint32_t input) noexcept {
  return matches(projection.routed, experts * output, input, false) &&
         matches(projection.shared, output, input, false);
}

void validate(const MoeWeights &weights, MoeShape shape) {
  const uint32_t hidden = shape.hiddenSize;
  const uint32_t intermediate = shape.expertIntermediateSize;
  if (shape.quant == QuantFamily::Gguf) {
    const GgufMoeWeights *gguf = weights.gguf ? &*weights.gguf : nullptr;
    if (!shape.valid() || !gguf ||
        !matches(gguf->router, shape.experts, hidden, true) ||
        !matches(gguf->sharedExpertGate, 1, hidden, true) ||
        !matches(gguf->gate, shape.experts, intermediate, hidden) ||
        !matches(gguf->up, shape.experts, intermediate, hidden) ||
        !matches(gguf->down, shape.experts, hidden, intermediate))
      throw std::invalid_argument("GGUF MoE weights do not match execution shape");
    return;
  }
  if (!shape.valid() || weights.gguf || !matches(weights.router, 256, hidden) ||
      !matches(weights.sharedExpertGate, 256, hidden) ||
      !matches(weights.expertGate, shape.experts, intermediate, hidden) ||
      !matches(weights.expertUp, shape.experts, intermediate, hidden) ||
      !matches(weights.expertDown, shape.experts, hidden, intermediate) ||
      !matches(weights.sharedGate, 1, intermediate, hidden) ||
      !matches(weights.sharedUp, 1, intermediate, hidden) ||
      !matches(weights.sharedDown, 1, hidden, intermediate)) {
    throw std::invalid_argument("MoE weights do not match execution shape");
  }
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
  // The router's rows x 256 fp32 scores live in the grouped input until the
  // gather overwrites them. Register plans also hold the down pass's Table16
  // tiles there, and one tile's row sums take 3 K / 4 fp32
  // (kernels/common/gguf_sgmatrix.h).
  const uint64_t scoreBytes = uint64_t{rows} * 256 * sizeof(float);
  const uint64_t sumsBytes = ggufTile == MoeGgufTile::Register
      ? uint64_t{tiles} * (uint64_t{widest} * 3 / 4) * sizeof(float) : 0;
  return {routes * sizeof(uint32_t), routes * sizeof(float),
          uint64_t{tiles} * sizeof(MoeTileDescriptor), sizeof(uint32_t),
          groupedRows * sizeof(uint32_t), routes * sizeof(uint32_t),
          std::max(groupedRows * (ggufTile == MoeGgufTile::Register ? widest : shape.hiddenSize) *
                       sizeof(uint16_t), scoreBytes),
          groupedRows * shape.expertIntermediateSize * sizeof(uint16_t),
          groupedRows * outputWidth * sizeof(uint16_t), sumsBytes};
}

// Pipelines, column tiles and threadgroup width of a plan's fused gate/up
// and down passes. Only the 8-row tiles have a four-simdgroup form; see
// MoeExpertSimdgroups for its geometry and measurements.
struct ExpertPasses final {
  const char *gateUp;
  const char *down;
  uint32_t gateUpColumns;
  uint32_t downColumns;
  uint32_t threads;
};

ExpertPasses fusedExpertPasses(const MoeConfig &config) noexcept {
  if (config.expertTile == MoeExpertTile::M32)
    return {"moe_expert_gate_up_q4_m32", "moe_expert_down_q4_m32", 128, 128,
            metal::CommandGraph::kDefaultThreads};
  const uint32_t threads = static_cast<uint32_t>(config.m8Simdgroups) * 32;
  if (config.m8Simdgroups == MoeExpertSimdgroups::Four)
    return {"moe_expert_gate_up_q4_m8_n128_sg4",
            "moe_expert_down_q4_m8_n256_sg4", 128, 256, threads};
  return {"moe_expert_gate_up_q4_m8", "moe_expert_down_q4_m8", 128, 128,
          threads};
}

void addAffineExperts(metal::CommandGraph &graph, const MoeBuffers &buffers,
                      const MoeWeights &weights, const MoePlan &plan) {
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
              {buffers.groupedInput, buffers.tileDescriptors,
               buffers.tileCount, weights.expertGate.packed,
               weights.sharedGate.packed, buffers.expertOutput},
              gate, {shape.expertIntermediateSize / 256, tiles, 1});
    graph.add("prefill_moe_expert_q4_n256_up_silu_m32",
              {buffers.groupedInput, buffers.tileDescriptors,
               buffers.tileCount, weights.expertUp.packed,
               weights.sharedUp.packed, buffers.expertOutput,
               buffers.expertIntermediate},
              up, {shape.expertIntermediateSize / 256, tiles, 1});
    graph.add("prefill_moe_expert_q4_n256_m32",
              {buffers.expertIntermediate, buffers.tileDescriptors,
               buffers.tileCount, weights.expertDown.packed,
               weights.sharedDown.packed, buffers.expertOutput},
              down, {shape.hiddenSize / 256, tiles, 1});
  } else {
    // The workspace holds the same grouped rows whatever the column tile;
    // only the grid's column count and the threadgroup width follow it.
    const ExpertPasses passes = fusedExpertPasses(plan.config());
    graph.add(passes.gateUp,
              {buffers.groupedInput, buffers.tileDescriptors,
               buffers.tileCount, weights.expertGate.packed,
               weights.expertUp.packed, weights.sharedGate.packed,
               weights.sharedUp.packed, buffers.expertIntermediate},
              gateUp,
              {shape.expertIntermediateSize / passes.gateUpColumns, tiles, 1},
              {passes.threads, 1, 1});
    graph.add(passes.down,
              {buffers.expertIntermediate, buffers.tileDescriptors,
               buffers.tileCount, weights.expertDown.packed,
               weights.sharedDown.packed, buffers.expertOutput},
              down, {shape.hiddenSize / passes.downColumns, tiles, 1},
              {passes.threads, 1, 1});
  }
}

// The three GGUF expert passes over grouped tiles: gate into expertOutput
// (the down pass overwrites it after the up pass consumed it), up with
// silu(gate) into expertIntermediate, down into expertOutput. Register plans
// read Table16 tiles from groupedInput: the gather writes the gate/up input's
// and a prepare dispatch the down input's.
void addGgufExperts(metal::CommandGraph &graph, const MoeBuffers &buffers,
                    const GgufMoeWeights &weights, const MoePlan &plan) {
  const MoeShape shape = plan.shape();
  const uint32_t tiles = plan.maximumTiles();
  const bool table16 = plan.config().ggufTile == MoeGgufTile::Register;
  const auto plane1 = [](const GgufSegment &s) { return s.plane1 ? s.plane1 : s.meta; };
  const auto pass = [&](const GgufExpertProjection &projection, bool up,
                        const metal::MetalBuffer &input,
                        const metal::MetalBuffer &output, uint32_t n, uint32_t k) {
    std::vector<metal::MetalBuffer> bindings{input};
    if (table16) bindings.push_back(buffers.groupedSums);
    bindings.insert(bindings.end(),
                    {buffers.tileDescriptors, buffers.tileCount,
                     projection.routed.plane0, plane1(projection.routed),
                     projection.routed.meta, projection.shared.plane0,
                     plane1(projection.shared), projection.shared.meta, output,
                     buffers.expertOutput});
    const std::string kernel = table16 ? "moe_expert_gguf_sg" : "moe_expert_gguf_m" + std::to_string(plan.tileRows());
    graph.add(kernel + (up ? "_up" : ""), std::move(bindings),
              MoeGgufExpertParams{k, n, shape.experts, projection.routed.formatId,
                                  projection.shared.formatId},
              {n / 64, tiles, 1},
              {table16 || plan.tileRows() == 64 ? 128u : 64u, 1, 1});
  };
  const uint32_t hidden = shape.hiddenSize;
  const uint32_t intermediate = shape.expertIntermediateSize;
  pass(weights.gate, false, buffers.groupedInput, buffers.expertOutput,
       intermediate, hidden);
  pass(weights.up, true, buffers.groupedInput, buffers.expertIntermediate,
       intermediate, hidden);
  if (table16)
    graph.add("moe_prepare_table16",
              {buffers.expertIntermediate, buffers.tileCount,
               buffers.groupedInput, buffers.groupedSums},
              intermediate, {tiles, intermediate / 256, 1});
  pass(weights.down, false,
       table16 ? buffers.groupedInput : buffers.expertIntermediate,
       buffers.expertOutput, hidden, intermediate);
}

} // namespace

// GGUF plans always run the three expert passes of the split plan.
MoePlan::MoePlan(MoeShape shape, uint32_t rows, MoeConfig config,
                 bool prefill)
    : shape_(shape), rows_(rows), config_(config),
      splitExperts_(shape.quant == QuantFamily::Gguf ||
                    (prefill && config.expertTile == MoeExpertTile::M32)) {
  // GGUF plans have 8-row kernels in both phases and 64-row prefill kernels;
  // affine plans 8- and 32-row kernels in both phases.
  const bool gguf = shape.quant == QuantFamily::Gguf;
  if (gguf ? config.expertTile != MoeExpertTile::M8 && (!prefill || config.expertTile != MoeExpertTile::M64)
           : config.expertTile != MoeExpertTile::M8 && config.expertTile != MoeExpertTile::M32)
    throw std::invalid_argument("invalid MoE expert tile configuration");
  if (config.m8Simdgroups != MoeExpertSimdgroups::Eight &&
      config.m8Simdgroups != MoeExpertSimdgroups::Four)
    throw std::invalid_argument("invalid MoE expert simdgroup configuration");
  if (config.ggufTile == MoeGgufTile::Register &&
      (shape.quant != QuantFamily::Gguf || config.expertTile != MoeExpertTile::M8))
    throw std::invalid_argument("the GGUF register expert tile takes GGUF 8-row tiles");
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
  const uint64_t rowBytes = uint64_t{rows} * shape.hiddenSize * sizeof(uint16_t);
  if (buffers.input.sizeBytes() < rowBytes ||
      buffers.residual.sizeBytes() < rowBytes ||
      buffers.output.sizeBytes() < rowBytes)
    throw std::invalid_argument("MoE row buffers are smaller than execution shape");
  if (buffers.selectedExperts.sizeBytes() < required.selectedExpertsBytes ||
      buffers.routingWeights.sizeBytes() < required.routingWeightsBytes ||
      buffers.tileDescriptors.sizeBytes() < required.tileDescriptorsBytes ||
      buffers.tileCount.sizeBytes() < required.tileCountBytes ||
      buffers.groupedRoutes.sizeBytes() < required.groupedRoutesBytes ||
      buffers.routeRows.sizeBytes() < required.routeRowsBytes ||
      buffers.groupedInput.sizeBytes() < required.groupedInputBytes ||
      buffers.expertIntermediate.sizeBytes() < required.expertIntermediateBytes ||
      buffers.expertOutput.sizeBytes() < required.expertOutputBytes ||
      buffers.groupedSums.sizeBytes() < required.groupedSumsBytes) {
    throw std::invalid_argument("MoE grouped scratch is smaller than its bound");
  }
  const MoeRouteParams routeParams{rows, shape.hiddenSize, shape.experts,
                                   shape.expertsPerToken};
  const GgufMoeWeights *gguf = weights.gguf ? &*weights.gguf : nullptr;
  if (gguf) {
    // fp32 scores of the F32 router in rows of 256, as the select kernel reads.
    addGgufFloat(graph, buffers.input, gguf->router, buffers.groupedInput, rows,
                 256, 0, FloatOutput::Float32);
    graph.add("moe_route_select_f32",
              {buffers.groupedInput, buffers.input,
               gguf->sharedExpertGate.plane0, buffers.selectedExperts,
               buffers.routingWeights},
              routeParams, {rows, 1, 1});
  } else {
    const MoeRouteTile route = moeRouteTile(rows, plan.config().routeWideRows);
    graph.add(route.rows == 8 ? "moe_route_scores_q8_m8"
                              : "moe_route_scores_q8_m32",
              {buffers.input, weights.router.weights, weights.router.scales,
               weights.router.biases, buffers.groupedInput},
              routeParams,
              {(rows + route.rows - 1) / route.rows, 256 / route.experts, 1});
    graph.add("moe_route_select_q8",
              {buffers.groupedInput, buffers.input,
               weights.sharedExpertGate.weights,
               weights.sharedExpertGate.scales,
               weights.sharedExpertGate.biases, buffers.selectedExperts,
               buffers.routingWeights},
              routeParams, {rows, 1, 1});
  }
  graph.add("moe_group_routes",
            {buffers.selectedExperts, buffers.tileDescriptors,
             buffers.tileCount, buffers.groupedRoutes, buffers.routeRows},
            MoeGroupParams{rows, shape.expertsPerToken, tileRows,
                           shape.experts},
            {1, 1, 1});
  const MoeGatherParams gather{tileRows, shape.hiddenSize, shape.routesPerToken()};
  if (plan.config().ggufTile == MoeGgufTile::Register)
    graph.add("moe_gather_table16",
              {buffers.input, buffers.groupedRoutes, buffers.tileCount,
               buffers.groupedInput, buffers.groupedSums},
              gather, {tiles, shape.hiddenSize / 256, 1});
  else
    graph.add("moe_gather_rows",
              {buffers.input, buffers.groupedRoutes, buffers.tileCount,
               buffers.groupedInput},
              gather, {tiles, shape.hiddenSize / 256, 1});
  if (gguf)
    addGgufExperts(graph, buffers, *gguf, plan);
  else
    addAffineExperts(graph, buffers, weights, plan);
  graph.add("moe_combine",
            {buffers.expertOutput, buffers.routeRows, buffers.routingWeights,
             buffers.residual, buffers.output},
            MoeCombineParams{rows, shape.hiddenSize, shape.routesPerToken()},
            {rows, shape.hiddenSize / 256, 1});
}

MoePlan MoE::prefillPlan(MoeShape shape, uint32_t rows) {
  return prefillPlan(shape, rows, {moePrefillTile(shape)});
}

MoePlan MoE::prefillPlan(MoeShape shape, uint32_t rows, MoeConfig config) {
  if (!rows || rows > SPLASH_PREFILL_TOKEN_BUDGET)
    throw std::invalid_argument("invalid MoE prefill rows");
  return MoePlan(shape, rows, config, true);
}

MoePlan MoE::decodePlan(MoeShape shape, uint32_t lanes, MoeConfig config) {
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid MoE decode batch width");
  return MoePlan(shape, lanes * SPLASH_TARGET_VERIFY_ROWS, config, false);
}

std::array<MoePlan, 2> MoE::prefillCandidates(MoeShape shape, uint32_t rows,
                                          uint32_t routeWideRows) {
  const MoePlan baseline = prefillPlan(shape, rows, {moePrefillTile(shape), routeWideRows});
  if (shape.quant == QuantFamily::Gguf) return {baseline, baseline};
  return {baseline, prefillPlan(shape, rows, {MoeExpertTile::M8, routeWideRows})};
}

std::array<MoePlan, 2> MoE::decodeCandidates(MoeShape shape, uint32_t lanes,
                                         uint32_t routeWideRows,
                                         MoeExpertSimdgroups m8Simdgroups,
                                         MoeGgufTile ggufTile) {
  const MoePlan baseline = decodePlan(
      shape, lanes, {MoeExpertTile::M8, routeWideRows, m8Simdgroups, ggufTile});
  if (shape.quant == QuantFamily::Gguf) return {baseline, baseline};
  return {baseline, decodePlan(shape, lanes,
                               {MoeExpertTile::M32, routeWideRows, m8Simdgroups})};
}

} // namespace splash::ops

#include "ops/Mamba2.hpp"

#include "metal/abi/Mamba2.h"
#include "ops/BufferExtent.hpp"
#include "ops/LaneBindings.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace splash::ops {

namespace {

constexpr uint32_t kRows = SPLASH_TARGET_VERIFY_ROWS;

// The geometries the kernels compile (kernels/shared/mamba2.metal) and their
// name suffix: Nemotron 3.5 Bolt's.
struct CompiledMamba2 final {
  Mamba2Shape shape;
  std::string_view suffix;
};
constexpr std::array kCompiledMamba2{CompiledMamba2{{8, 128, 64, 128}, "_g8_h128"}};

std::string_view suffixOf(const Mamba2Shape &shape) {
  const std::string_view suffix = shape.compiledSuffix();
  if (suffix.empty()) throw std::invalid_argument("no Mamba-2 kernel for the geometry");
  return suffix;
}

void requireWeights(const Mamba2Weights &w, const Mamba2Shape &shape) {
  const uint64_t heads = uint64_t{shape.heads} * 4;
  requireBytes(w.convolution, uint64_t{shape.convolutionDimension()} * SPLASH_GDN_CONVOLUTION_TAPS * 4,
               "Mamba-2 convolution weights");
  requireBytes(w.convolutionBias, uint64_t{shape.convolutionDimension()} * 4, "Mamba-2 convolution bias");
  requireBytes(w.a, heads, "Mamba-2 A");
  requireBytes(w.dtBias, heads, "Mamba-2 dt bias");
  requireBytes(w.d, heads, "Mamba-2 D");
}

} // namespace

std::string_view Mamba2Shape::compiledSuffix() const noexcept {
  for (const CompiledMamba2 &compiled : kCompiledMamba2)
    if (compiled.shape == *this) return compiled.suffix;
  return {};
}

uint64_t mamba2TapeLayerBytes(const Mamba2Shape &shape) {
  return mamba2_tape_layer_bytes(shape.convolutionDimension(), shape.groups * shape.stateDimension + shape.innerSize(),
                                 shape.heads);
}

void Mamba2::addDecode(metal::CommandGraph &graph, metal::MetalBuffer packed, const Mamba2Weights &weights,
                       std::span<const metal::MetalBuffer> currentStates,
                       std::span<const metal::MetalBuffer> nextStates, metal::MetalBuffer tape,
                       std::span<const GdnTapeLane> tapeLanes, metal::MetalBuffer laneRows, uint32_t rows,
                       metal::MetalBuffer y, Mamba2Shape shape, uint32_t layer, GdnStateStrides state) {
  const std::string_view suffix = suffixOf(shape);
  const auto lanes = uint32_t(tapeLanes.size());
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH) throw std::invalid_argument("invalid Mamba-2 decode batch width");
  if (!state.valid() || state.format != GdnStateFormat::Float32)
    throw std::invalid_argument("Mamba-2 keeps its state in fp32");
  requireWeights(weights, shape);
  if (rows < lanes || rows > uint64_t{lanes} * kRows) throw std::invalid_argument("invalid Mamba-2 decode rows");
  requireBytes(packed, uint64_t{rows} * shape.packedWidth() * 2, "Mamba-2 packed rows");
  requireBytes(y, uint64_t{rows} * shape.innerSize() * 4, "Mamba-2 gated rows");
  requireBytes(laneRows, uint64_t{lanes} * sizeof(SplashLaneRows), "Mamba-2 lane rows");
  const uint64_t tapeBytes = mamba2TapeLayerBytes(shape);
  GDNDecodeBatchParams params{0, layer, state.convolutionLayerBytes, state.recurrentLayerBytes,
                              state.convolutionStateBytes, {}, {}, {}};
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const GdnTapeLane &t = tapeLanes[lane];
    // The scan folds a chain's rows: a path of the previous step's first rows.
    if (t.pendingPath & (t.pendingPath + 1) || t.pendingPath >> kRows)
      throw std::invalid_argument("Mamba-2 folds a chain's rows alone");
    params.pending_tape[lane] = t.pendingSlot + layer * tapeBytes;
    params.step_tape[lane] = t.stepSlot + layer * tapeBytes;
    params.pending_path[lane] = t.pendingPath;
    requireBytes(tape, std::max(params.pending_tape[lane], params.step_tape[lane]) + tapeBytes, "Mamba-2 tape");
  }
  std::vector<metal::MetalBuffer> bindings{std::move(packed), weights.convolution, weights.convolutionBias};
  appendLaneBindings(bindings, currentStates, nextStates);
  bindings.insert(bindings.end(),
                  {std::move(tape), weights.a, weights.dtBias, weights.d, std::move(y), std::move(laneRows)});
  graph.add("verify_mamba2" + std::string(suffix), std::move(bindings), params, {shape.heads, lanes, 1});
}

void Mamba2::addPrefillStep(metal::CommandGraph &graph, metal::MetalBuffer packed, const Mamba2Weights &weights,
                            const Mamba2StepStates &states, metal::MetalBuffer tape, uint64_t pendingTape,
                            uint64_t stepTape, uint32_t pendingRows, metal::MetalBuffer y, Mamba2Shape shape) {
  const std::string_view suffix = suffixOf(shape);
  requireWeights(weights, shape);
  if (pendingRows > kRows) throw std::invalid_argument("invalid Mamba-2 prefill tape");
  requireBytes(packed, uint64_t{kRows} * shape.packedWidth() * 2, "Mamba-2 packed rows");
  requireBytes(y, uint64_t{kRows} * shape.innerSize() * 4, "Mamba-2 gated rows");
  const uint64_t convolution = uint64_t{SPLASH_GDN_CONVOLUTION_TAPS - 1} * shape.convolutionDimension() * 2;
  const uint64_t recurrent = uint64_t{shape.innerSize()} * shape.stateDimension * 4;
  requireBytes(states.convolutionIn, convolution, "Mamba-2 convolution state");
  requireBytes(states.convolutionOut, convolution, "Mamba-2 convolution state");
  requireBytes(states.recurrentIn, recurrent, "Mamba-2 recurrent state");
  requireBytes(states.recurrentOut, recurrent, "Mamba-2 recurrent state");
  const uint64_t tapeBytes = mamba2TapeLayerBytes(shape);
  requireBytes(tape, std::max(pendingTape, stepTape) + tapeBytes, "Mamba-2 tape");
  GDNDecodeBatchParams params{};
  params.pending_tape[0] = pendingTape;
  params.step_tape[0] = stepTape;
  params.pending_path[0] = (1u << pendingRows) - 1u;
  graph.add("prefill_mamba2" + std::string(suffix),
            {std::move(packed), weights.convolution, weights.convolutionBias, states.convolutionIn,
             states.recurrentIn, states.convolutionOut, states.recurrentOut, std::move(tape), weights.a,
             weights.dtBias, weights.d, std::move(y)},
            params, {shape.heads, 1, 1});
}

void Mamba2::addGroupNorm(metal::CommandGraph &graph, metal::MetalBuffer y, const metal::MetalBuffer &norm,
                          metal::MetalBuffer hidden, double epsilon, Mamba2Shape shape, uint32_t rows) {
  const std::string_view suffix = suffixOf(shape);
  if (!rows) throw std::invalid_argument("invalid Mamba-2 norm rows");
  requireBytes(y, uint64_t{rows} * shape.innerSize() * 4, "Mamba-2 gated rows");
  requireBytes(norm, uint64_t{shape.innerSize()} * 4, "Mamba-2 norm weights");
  requireBytes(hidden, uint64_t{rows} * shape.innerSize() * 2, "Mamba-2 normalized rows");
  graph.add("mamba2_group_norm" + std::string(suffix), {std::move(y), norm, std::move(hidden)},
            Mamba2NormParams{float(epsilon)}, {shape.groups, rows, 1});
}

} // namespace splash::ops

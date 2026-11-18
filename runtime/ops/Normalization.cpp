#include "Normalization.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "ops/BufferExtent.hpp"

#include <utility>
#include <stdexcept>

namespace splash::ops {

namespace {

// normKernel's variant of the plain norm's kernel `name` of the weights'
// epsilon: _e5 for SPLASH_RMS_EPSILON_E5, which only F32 weights compile.
std::string rmsKernel(std::string_view name, const NormWeights &weights, uint32_t width) {
  if (weights.epsilon == SPLASH_RMS_EPSILON) return normKernel(name, weights, width);
  if (weights.epsilon != SPLASH_RMS_EPSILON_E5 || !weights.float32)
    throw std::invalid_argument("no RMS norm kernel for the norm's epsilon");
  return normKernel(name, NormWeights{weights.buffer, true}, width) + "_e5";
}

} // namespace

std::string normKernel(std::string_view name, const NormWeights &weights, uint32_t width) {
  if (weights.epsilon != SPLASH_RMS_EPSILON) throw std::invalid_argument("the kernel takes the default RMS epsilon");
  requireBytes(weights.buffer, weights.bytes(width), "norm weight");
  return std::string(name) + (weights.float32 ? "_f32" : "");
}

PreparedInput Normalization::addRms(metal::CommandGraph &graph,
                                    metal::MetalBuffer input,
                                    const NormWeights &weight,
                                    metal::MetalBuffer output, uint32_t width,
                                    uint32_t rows, LinearScratch scratch,
                                    LinearInput layout) {
  const uint64_t bytes = uint64_t{rows} * width * 2;
  requireBytes(input, bytes, "norm input");
  requireBytes(output, bytes, "norm output");
  if (layout != LinearInput::Plain) {
    requireTableScratch(scratch, layout, width, rows);
    graph.add(rmsKernel(std::string("norm_rms") + tableSuffix(layout) + "_decode", weight, width),
              {input, weight.buffer, output, scratch.input, scratch.sums}, width, {rows, 1, 1});
    return {std::move(output), layout};
  }
  if (rows <= SPLASH_STAGED_NORM_ROWS && width <= SPLASH_STAGED_NORM_WIDTH && width % 4 == 0)
    graph.add(rmsKernel("norm_rms_staged", weight, width), {std::move(input), weight.buffer, output},
              width, {rows, 1, 1}, {SPLASH_STAGED_NORM_THREADS, 1, 1});
  else
    graph.add(rmsKernel("norm_rms", weight, width), {std::move(input), weight.buffer, output},
              width, {rows, 1, 1});
  return {};
}

} // namespace splash::ops

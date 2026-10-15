#include "Normalization.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <utility>
#include <stdexcept>

namespace splash::ops {

std::string normKernel(std::string_view name, const NormWeights &weights, uint32_t width) {
  requireBytes(weights.buffer, weights.bytes(width), "norm weight");
  return std::string(name) + (weights.float32 ? "_f32" : "");
}

PreparedInput Normalization::addRms(metal::MetalBackend &backend, metal::CommandGraph &graph,
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
    graph.add(normKernel(std::string("norm_rms") + tableSuffix(layout) + "_decode", weight, width),
              {input, weight.buffer, output, scratch.input, scratch.sums}, width, {rows, 1, 1});
    return {std::move(output), layout};
  }
  if (rows <= SPLASH_STAGED_NORM_ROWS && width <= SPLASH_STAGED_NORM_WIDTH && width % 4 == 0) {
    const std::string name = normKernel("norm_rms_staged", weight, width);
    // The driver caps the staged norm's total threads below the design width
    // on some families (under validation, on Apple7); the kernel strides its
    // staging/output loops by the dispatch's width, which the design width
    // keeps elsewhere.
    const uint32_t threads =
        std::min<uint32_t>(SPLASH_STAGED_NORM_THREADS, backend.pipelineMaxThreads(name));
    graph.add(name, {std::move(input), weight.buffer, output}, NormParams{width, threads},
              {rows, 1, 1}, {threads, 1, 1});
    return {};
  }
  graph.add(normKernel("norm_rms", weight, width), {std::move(input), weight.buffer, output},
            width, {rows, 1, 1});
  return {};
}

} // namespace splash::ops

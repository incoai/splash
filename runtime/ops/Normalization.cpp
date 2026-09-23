#include "Normalization.hpp"

#include <utility>
#include <stdexcept>

namespace splash::ops {

std::string normKernel(std::string_view name, const NormWeights &weights, uint32_t width) {
  if (!weights.buffer || weights.buffer.sizeBytes() < weights.bytes(width))
    throw std::invalid_argument("norm weights are below the width");
  return std::string(name) + (weights.float32 ? "_f32" : "");
}

PreparedInput Normalization::addRms(metal::CommandGraph &graph,
                                    metal::MetalBuffer input,
                                    const NormWeights &weight,
                                    metal::MetalBuffer output, uint32_t width,
                                    uint32_t rows, LinearScratch scratch,
                                    LinearInput layout) {
  if (layout != LinearInput::Plain && scratch.input && rows && rows % 8 == 0) {
    if (scratch.input.sizeBytes() < tableBytes(width, rows) ||
        scratch.sums.sizeBytes() < tableSumsBytes(layout, width, rows) || width % 64)
      throw std::invalid_argument("Q4 normalization scratch is below requirement");
    graph.add(normKernel(layout == LinearInput::Table16 ? "norm_rms_q16_decode" : "norm_rms_q4_decode",
                         weight, width),
              {input, weight.buffer, output, scratch.input, scratch.sums}, width, {rows, 1, 1});
    return {std::move(output), layout};
  }
  graph.add(normKernel("norm_rms", weight, width), {std::move(input), weight.buffer, output},
            width, {rows, 1, 1});
  return {};
}

void Normalization::addRmsWithQ4Sums(
    metal::CommandGraph &graph, metal::MetalBuffer input,
    const NormWeights &weight, metal::MetalBuffer output,
    metal::MetalBuffer sums, uint32_t width, uint32_t rows) {
  graph.add(normKernel("prefill_norm_rms_sums32", weight, width),
            {std::move(input), weight.buffer, std::move(output),
             std::move(sums)},
            width, {rows, 1, 1});
}

} // namespace splash::ops

#include "Normalization.hpp"

#include <utility>
#include <stdexcept>

namespace splash::ops {

PreparedInput Normalization::addRms(metal::CommandGraph &graph,
                                    metal::MetalBuffer input,
                                    metal::MetalBuffer weight,
                                    metal::MetalBuffer output, uint32_t width,
                                    uint32_t rows, LinearScratch scratch,
                                    LinearInput layout) {
  if (layout == LinearInput::Table64 && scratch.input && rows && rows % 8 == 0) {
    if (scratch.input.sizeBytes() < uint64_t(width) * rows * 2 ||
        scratch.sums.sizeBytes() < uint64_t(width) * rows / 16 || width % 64)
      throw std::invalid_argument("Q4 normalization scratch is below requirement");
    graph.add("norm_rms_q4_decode", {input, weight, output, scratch.input, scratch.sums},
              width, {rows, 1, 1});
    return {std::move(output), LinearInput::Table64};
  }
  graph.add("norm_rms", {std::move(input), std::move(weight), output}, width,
            {rows, 1, 1});
  return {};
}

void Normalization::addRmsWithQ4Sums(
    metal::CommandGraph &graph, metal::MetalBuffer input,
    metal::MetalBuffer weight, metal::MetalBuffer output,
    metal::MetalBuffer sums, uint32_t width, uint32_t rows) {
  graph.add("prefill_norm_rms_sums32",
            {std::move(input), std::move(weight), std::move(output),
             std::move(sums)},
            width, {rows, 1, 1});
}

} // namespace splash::ops

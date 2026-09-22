#include "Normalization.hpp"

#include "metal/abi/ExecutionGeometry.h"

#include <utility>
#include <stdexcept>

namespace splash::ops {
namespace {
// The staging array in norm_rms_staged bounds the row width it can hold. Both
// served models normalize rows of 2048 or 5120 bfloat, so the bound costs
// nothing today; wider rows keep the unstaged kernel. The kernel binds bfloat4
// views, so callers pass 8-byte aligned buffers, which every arena and weight
// section already is. The 1024-thread dispatch is a hard requirement: every
// supported GPU family admits it, and the backend throws rather than running
// a pipeline that cannot.
constexpr uint32_t kStagedWidth = SPLASH_STAGED_NORM_WIDTH;
constexpr uint32_t kStagedThreads = SPLASH_STAGED_NORM_THREADS;
} // namespace

void Normalization::addRms(metal::CommandGraph &graph,
                           metal::MetalBuffer input,
                           metal::MetalBuffer weight,
                           metal::MetalBuffer output, uint32_t width,
                           uint32_t rows, LinearScratch scratch) {
  if (scratch.input && rows && rows % 8 == 0) {
    if (scratch.input.sizeBytes() < uint64_t(width) * rows * 2 ||
        scratch.sums.sizeBytes() < uint64_t(width) * rows / 16 || width % 64)
      throw std::invalid_argument("Q4 normalization scratch is below requirement");
    graph.add("norm_rms_q4_decode", {input, weight, output, scratch.input, scratch.sums},
              width, {rows, 1, 1});
    return;
  }
  if (width <= kStagedWidth && width % 4 == 0) {
    graph.add("norm_rms_staged",
              {std::move(input), std::move(weight), std::move(output)}, width,
              {rows, 1, 1}, {kStagedThreads, 1, 1});
    return;
  }
  graph.add("norm_rms",
            {std::move(input), std::move(weight), std::move(output)}, width,
            {rows, 1, 1});
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

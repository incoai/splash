#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/Linear.hpp"

#include <cstdint>

namespace splash::ops {

// The per-channel bf16 multipliers of an RMS norm as stored. Every kernel
// that reads them widens them to fp32.
struct NormWeights final {
  metal::MetalBuffer buffer;

  [[nodiscard]] static constexpr uint64_t bytes(uint32_t width) noexcept { return uint64_t{width} * 2; }
};

class Normalization final {
public:
  // Also writes the consumer's `layout` table into `scratch` when it is not
  // Plain, and throws when `scratch` cannot hold it; returns what the
  // scratch then describes.
  static PreparedInput addRms(metal::CommandGraph &graph, metal::MetalBuffer input,
                              const NormWeights &weight, metal::MetalBuffer output,
                              uint32_t width, uint32_t rows,
                              LinearScratch scratch = {},
                              LinearInput layout = LinearInput::Plain);

  // Fused RMS normalization plus Q4 input-group sums for an affine prefill
  // projection, of whole 64-input groups.
  static void addRmsWithQ4Sums(metal::CommandGraph &graph,
                               metal::MetalBuffer input,
                               const NormWeights &weight,
                               metal::MetalBuffer output,
                               metal::MetalBuffer sums, uint32_t width,
                               uint32_t rows);
};

} // namespace splash::ops

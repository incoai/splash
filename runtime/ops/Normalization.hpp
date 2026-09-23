#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/Linear.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace splash::ops {

// The per-channel multipliers of an RMS norm as stored: bf16 in the packed
// formats, F32 in a GGUF, which keeps its norms unquantized as llama.cpp
// does. Every kernel that reads them widens them to fp32, so the type only
// selects the kernel variant that loads them (normKernel).
struct NormWeights final {
  metal::MetalBuffer buffer;
  bool float32 = false;

  [[nodiscard]] constexpr uint64_t bytes(uint32_t width) const noexcept {
    return uint64_t{width} * (float32 ? 4 : 2);
  }
};

// The variant of kernel `name` that loads `weights`: `name` for bf16,
// `name`_f32 for F32. Throws unless the buffer holds `width` multipliers.
[[nodiscard]] std::string normKernel(std::string_view name, const NormWeights &weights,
                                     uint32_t width);

class Normalization final {
public:
  // Also writes the consumer's `layout` table into `scratch` when it needs
  // one; returns what the scratch then describes.
  static PreparedInput addRms(metal::CommandGraph &graph, metal::MetalBuffer input,
                              const NormWeights &weight, metal::MetalBuffer output,
                              uint32_t width, uint32_t rows,
                              LinearScratch scratch = {},
                              LinearInput layout = LinearInput::Plain);

  // Fused RMS normalization plus Q4 input-group sums for packed prefill.
  static void addRmsWithQ4Sums(metal::CommandGraph &graph,
                               metal::MetalBuffer input,
                               const NormWeights &weight,
                               metal::MetalBuffer output,
                               metal::MetalBuffer sums, uint32_t width,
                               uint32_t rows);
};

} // namespace splash::ops

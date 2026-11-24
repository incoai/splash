#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/Linear.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace splash::ops {

// The per-channel multipliers of an RMS norm as stored: bf16 in an MLX
// target and the draft, F32 in a GGUF, which keeps its norms unquantized as
// llama.cpp does. Every kernel that reads them widens them to fp32, so the type only
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

// A LayerNorm's bf16 weight and bias of every column.
struct LayerNormWeights final {
  metal::MetalBuffer weight;
  metal::MetalBuffer bias;
};

class Normalization final {
public:
  // A LayerNorm's contract, which its owner checks once, where it takes the
  // weights: a bf16 weight and bias of each of `width` columns. Throws,
  // naming `name`, otherwise.
  static void requireLayer(const LayerNormWeights &weights, uint32_t width, const std::string &name);
  // LayerNorm with bias of `rows` fp32 rows of `width`, a tower's residual
  // stream, into bf16 rows: (x - mean) / sqrt(variance + epsilon) * weight +
  // bias, the mean and variance fp32 over the row, of weights requireLayer
  // took. Into fp32 rows instead with outputElement Float32, which may be the
  // input's own (the LayerNorm that starts Inkling 2's tower's residual
  // stream).
  static void addLayer(metal::CommandGraph &graph, const metal::MetalBuffer &input,
                       const LayerNormWeights &weights, const metal::MetalBuffer &output, uint32_t width,
                       uint32_t rows, float epsilon, FloatOutput outputElement = FloatOutput::BFloat16);
  // Also writes the consumer's `layout` table into `scratch` when it is not
  // Plain, and throws when `scratch` cannot hold it; returns what the
  // scratch then describes.
  static PreparedInput addRms(metal::CommandGraph &graph, metal::MetalBuffer input,
                              const NormWeights &weight, metal::MetalBuffer output,
                              uint32_t width, uint32_t rows,
                              LinearScratch scratch = {},
                              LinearInput layout = LinearInput::Plain);
};

} // namespace splash::ops

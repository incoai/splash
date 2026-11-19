#pragma once

#include "metal/MetalBackend.hpp"

#include <compare>
#include <cstdint>
#include <utility>

namespace splash::ops {

// Physical layout, independent of the checkpoint container and compute tile:
// affine Q4, quantized per 64 inputs.
enum class WeightLayout : uint8_t { Affine64 };

struct ProjectionShape final {
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  WeightLayout layout = WeightLayout::Affine64;
  auto operator<=>(const ProjectionShape &) const = default;
};

// The element type a projection writes: bf16, or fp32 for a vocabulary
// head's logits.
enum class FloatOutput : uint8_t { BFloat16, Float32 };
[[nodiscard]] constexpr uint64_t elementBytes(FloatOutput type) noexcept {
  return type == FloatOutput::Float32 ? sizeof(float) : sizeof(uint16_t);
}

struct AffineWeights final {
  metal::MetalBuffer weights;
  metal::MetalBuffer scales;
  metal::MetalBuffer biases;
};

// A Q8 affine projection, quantized per 64 inputs in storage tiles of
// SPLASH_AFFINE_TILE_ROWS rows (metal/abi/Linear.h): the MoE router and the
// shared expert's scalar gate.
struct Q8Projection final {
  AffineWeights planes;
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
};

// An expert-major Q4 slab holding one complete storage-tiled projection per
// expert, expertStrideBytes apart: the operator selects an expert by its
// offset, so no per-expert buffer or copy exists at run time.
struct ExpertProjection final {
  metal::MetalBuffer packed;
  uint32_t experts = 0;
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  uint64_t expertStrideBytes = 0;
};

// A projection of outputSize x inputSize affine Q4 weights. A default value
// holds empty weights, which a reader replaces.
class Projection final {
public:
  Projection() = default;
  Projection(uint32_t output, uint32_t input, AffineWeights weights)
      : outputSize(output), inputSize(input), weights_(std::move(weights)) {}

  [[nodiscard]] WeightLayout layout() const noexcept { return WeightLayout::Affine64; }
  [[nodiscard]] const AffineWeights &affine() const noexcept { return weights_; }

  [[nodiscard]] ProjectionShape shape() const noexcept { return {outputSize, inputSize, layout()}; }

  // Views of a projection's planes (Linear.cpp): over the leading `rows` rows,
  // whole SPLASH_AFFINE_TILE_ROWS tiles; or over the leading `inputs` inputs
  // of each of its rows, whole quant groups, which reads its planes as they
  // are (planeInputs()).
  [[nodiscard]] Projection leadingRows(const metal::MetalBackend &backend, uint32_t rows) const;
  [[nodiscard]] Projection leadingInputs(uint32_t inputs) const;
  // Whether those views take it: its own planes, not a view.
  [[nodiscard]] bool takesPlaneViews() const noexcept { return !planeInputs_; }
  // The inputs each row of the weight planes holds when the projection reads
  // only their first inputSize, a view of leadingInputs(); zero for
  // inputSize. Only the prefill residual tiles take such a view, on kernel
  // instances of their own (Linear::add).
  [[nodiscard]] uint32_t planeInputs() const noexcept { return planeInputs_; }
  [[nodiscard]] uint32_t planeInputSize() const noexcept { return planeInputs_ ? planeInputs_ : inputSize; }

  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  // fp32 only for plain decode plans (Linear::plan), which keep the tile of
  // the bf16 plan.
  FloatOutput destination = FloatOutput::BFloat16;

private:
  AffineWeights weights_;
  uint32_t planeInputs_ = 0;
};

// A token table of outputSize rows of inputSize affine Q4 values, which
// Embedding gathers. It is intentionally a separate type: no table may be
// bound as a projection.
class EmbeddingWeights final {
public:
  EmbeddingWeights() = default;
  EmbeddingWeights(uint32_t output, uint32_t input, AffineWeights weights)
      : outputSize(output), inputSize(input), weights_(std::move(weights)) {}

  [[nodiscard]] WeightLayout layout() const noexcept { return WeightLayout::Affine64; }
  [[nodiscard]] const AffineWeights &affine() const noexcept { return weights_; }

  uint32_t outputSize = 0;
  uint32_t inputSize = 0;

private:
  AffineWeights weights_;
};

} // namespace splash::ops

#pragma once

#include "metal/MetalBackend.hpp"

#include <compare>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

struct QuantFormat;

namespace splash::ops {

// Physical layout, independent of the checkpoint container and compute tile.
enum class WeightLayout : uint8_t { Affine64, Block32 };

struct ProjectionShape final {
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  WeightLayout layout = WeightLayout::Affine64;
  auto operator<=>(const ProjectionShape &) const = default;
};

// The element type a projection writes: bf16, or fp32 for the vocabulary
// head's logits and a float segment's router scores.
enum class FloatOutput : uint8_t { BFloat16, Float32 };
[[nodiscard]] constexpr uint64_t elementBytes(FloatOutput type) noexcept {
  return type == FloatOutput::Float32 ? sizeof(float) : sizeof(uint16_t);
}

struct AffineWeights final {
  metal::MetalBuffer weights;
  metal::MetalBuffer scales;
  metal::MetalBuffer biases;
};

// A Q8 affine projection, quantized per 64 inputs in StorageN=256 order: the
// MoE router and the shared expert's scalar gate.
struct Q8Projection final {
  AffineWeights planes;
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
};

// An expert-major Q4 slab holding one complete StorageN-packed projection per
// expert, expertStrideBytes apart: the operator selects an expert by its
// offset, so no per-expert buffer or copy exists at run time.
struct ExpertProjection final {
  metal::MetalBuffer packed;
  uint32_t experts = 0;
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  uint64_t expertStrideBytes = 0;
};

// A prepared GGUF tensor occupying a projection's output columns
// [columnOffset, columnOffset + outputSize): repacked planes in the GGUF_FMT_*
// format formatId (metal/abi/QuantFormat.h), or a float tensor the GGUF keeps
// unquantized (F32, as llama.cpp keeps the MoE router), whose plane0 holds
// [outputSize][inputSize] floats multiplied unrounded in fp32
// (kernels/shared/gguf_float.metal). The source container is not part of it.
struct QuantizedSegment final {
  // The formatId of a float segment.
  static constexpr uint32_t kFloat32 = 0xffffffff;

  [[nodiscard]] static QuantizedSegment planes(uint32_t formatId, uint32_t outputSize, uint32_t inputSize,
                                               metal::MetalBuffer plane0, metal::MetalBuffer plane1,
                                               metal::MetalBuffer meta);
  [[nodiscard]] static QuantizedSegment floats(uint32_t outputSize, uint32_t inputSize,
                                               metal::MetalBuffer values) {
    return {std::move(values), {}, {}, outputSize, inputSize, 0, kFloat32};
  }

  metal::MetalBuffer plane0;
  metal::MetalBuffer plane1;
  metal::MetalBuffer meta;
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  uint32_t columnOffset = 0;
  uint32_t formatId = kFloat32;

  [[nodiscard]] bool isFloat() const noexcept { return formatId == kFloat32; }
  // The plane geometry of a quantized segment.
  [[nodiscard]] const QuantFormat &format() const noexcept;
  // The kernel name suffix of its format ("f32" for a float segment).
  [[nodiscard]] const char *name() const noexcept;
  // The buffer bound in plane1's slot: a format without a second plane binds
  // its meta plane there, which its kernels never read as plane1.
  [[nodiscard]] const metal::MetalBuffer &plane1Slot() const noexcept { return plane1 ? plane1 : meta; }
};

struct BlockWeights final {
  std::vector<QuantizedSegment> segments;
};

// Immutable weights in one of the two layouts: Affine holds the Affine64
// form, Block the Block32 form, and layout() names the one held. Projections,
// token tables and MoE blocks share this pattern; the accessor of the layout
// not held throws std::bad_variant_access. A default value holds empty affine
// weights, which a reader replaces.
template <class Affine, class Block>
class LayoutWeights {
public:
  LayoutWeights() = default;
  LayoutWeights(Affine weights) : storage_(std::move(weights)) {}
  LayoutWeights(Block weights) : storage_(std::move(weights)) {}

  [[nodiscard]] WeightLayout layout() const noexcept {
    return std::visit([](const auto &weights) { return layoutOf(weights); }, storage_);
  }
  [[nodiscard]] const Affine &affine() const { return std::get<Affine>(storage_); }
  [[nodiscard]] const Block &blocks() const { return std::get<Block>(storage_); }

private:
  // One layout per alternative: an alternative without one does not compile.
  static constexpr WeightLayout layoutOf(const Affine &) noexcept { return WeightLayout::Affine64; }
  static constexpr WeightLayout layoutOf(const Block &) noexcept { return WeightLayout::Block32; }

  std::variant<Affine, Block> storage_;
};

// A projection of outputSize x inputSize. A block projection's segments tile
// its leading output columns in order: each takes every input and starts
// where the previous one ends; the columns past the last are padding.
class Projection final : public LayoutWeights<AffineWeights, BlockWeights> {
public:
  Projection() = default;
  Projection(uint32_t output, uint32_t input, AffineWeights weights)
      : LayoutWeights(std::move(weights)), outputSize(output), inputSize(input) {}
  Projection(uint32_t output, uint32_t input, BlockWeights weights)
      : LayoutWeights(std::move(weights)), outputSize(output), inputSize(input) {
    if (blocks().segments.empty()) throw std::invalid_argument("block projection has no segments");
    uint32_t covered = 0;
    for (const QuantizedSegment &s : blocks().segments) {
      if (s.inputSize != input || s.columnOffset != covered || !s.outputSize || s.outputSize > output - covered)
        throw std::invalid_argument("block segments do not tile the projection");
      covered += s.outputSize;
    }
  }

  [[nodiscard]] ProjectionShape shape() const noexcept {
    return {outputSize, inputSize, layout()};
  }

  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  // fp32 only for plain decode plans (Linear::plan), which keep the tile of
  // the bf16 plan.
  FloatOutput destination = FloatOutput::BFloat16;
};

// A token table's rows as the GGUF stores them, in a gguf_embedding_format
// (metal/abi/Gguf.h), gathered, never multiplied (Embedding.cpp).
struct NativeRows final {
  NativeRows(metal::MetalBuffer rows, uint32_t formatId);
  metal::MetalBuffer rows;
  uint32_t formatId;
  [[nodiscard]] const char *name() const noexcept;
};

// A token table of outputSize rows of inputSize values, which Embedding
// gathers: affine Q4 rows or native GGUF rows. It is intentionally a separate
// type: no table may be bound as a projection.
class EmbeddingWeights final : public LayoutWeights<AffineWeights, NativeRows> {
public:
  EmbeddingWeights() = default;
  EmbeddingWeights(uint32_t output, uint32_t input, AffineWeights weights)
      : LayoutWeights(std::move(weights)), outputSize(output), inputSize(input) {}
  EmbeddingWeights(uint32_t output, uint32_t input, NativeRows rows)
      : LayoutWeights(std::move(rows)), outputSize(output), inputSize(input) {}

  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
};

} // namespace splash::ops

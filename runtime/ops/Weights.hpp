#pragma once

#include "metal/MetalBackend.hpp"

#include <compare>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

struct QuantFormat;

namespace splash::ops {

struct ProjectionShape final {
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  // Its quantized segments multiply the rotated input (InputRotation), which
  // takes LinearScratch::rotated.
  bool rotated = false;
  auto operator<=>(const ProjectionShape &) const = default;
};

// Prism ML's input rotation (metal/abi/Gguf.h, GGUF_ROTATION_BLOCK): a block
// projection whose weights were stored for rotated inputs multiplies H (D x)
// of its input x in its quantized segments, its float segments x itself; a
// token table stored rotated gathers its rows as D (H r). D is one int8 sign
// per input.
struct InputRotation final {
  metal::MetalBuffer signs;
  [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(signs); }
};

// The element type a projection writes: bf16, or fp32 for the vocabulary
// head's logits and a float segment's router scores.
enum class FloatOutput : uint8_t { BFloat16, Float32 };
[[nodiscard]] constexpr uint64_t elementBytes(FloatOutput type) noexcept {
  return type == FloatOutput::Float32 ? sizeof(float) : sizeof(uint16_t);
}

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

// A projection of outputSize x inputSize. Its segments tile its leading
// output columns in order: each takes every input and starts where the
// previous one ends; the columns past the last are padding. A default value
// holds no segments, which a reader replaces.
class Projection final {
public:
  Projection() = default;
  Projection(uint32_t output, uint32_t input, BlockWeights weights)
      : outputSize(output), inputSize(input), weights_(std::move(weights)) {
    if (weights_.segments.empty()) throw std::invalid_argument("block projection has no segments");
    uint32_t covered = 0;
    for (const QuantizedSegment &s : weights_.segments) {
      if (s.inputSize != input || s.columnOffset != covered || !s.outputSize || s.outputSize > output - covered)
        throw std::invalid_argument("block segments do not tile the projection");
      covered += s.outputSize;
    }
  }

  [[nodiscard]] const BlockWeights &blocks() const noexcept { return weights_; }
  [[nodiscard]] ProjectionShape shape() const noexcept {
    return {outputSize, inputSize, static_cast<bool>(rotation)};
  }

  // Views of a projection of one unrotated quantized tensor (Linear.cpp):
  // over the leading `rows` rows of its planes, whole QUANT_TILE_ROWS tiles;
  // or over the leading `inputs` inputs of each of its rows, whole quant
  // groups and meta units, which reads its planes as they are (planeInputs()).
  [[nodiscard]] Projection leadingRows(const metal::MetalBackend &backend, uint32_t rows) const;
  [[nodiscard]] Projection leadingInputs(uint32_t inputs) const;
  // Whether those views take it: its own planes, not a view, of one
  // unrotated quantized tensor.
  [[nodiscard]] bool takesPlaneViews() const noexcept;
  // The inputs each row of the weight planes holds when the projection reads
  // only their first inputSize, a view of leadingInputs(); zero for
  // inputSize. Only the prefill residual tiles of quantized weights take such
  // a view, on kernel instances of their own (Linear::add).
  [[nodiscard]] uint32_t planeInputs() const noexcept { return planeInputs_; }
  [[nodiscard]] uint32_t planeInputSize() const noexcept { return planeInputs_ ? planeInputs_ : inputSize; }

  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  // fp32 only for plain decode plans (Linear::plan), which keep the tile of
  // the bf16 plan.
  FloatOutput destination = FloatOutput::BFloat16;
  InputRotation rotation;

private:
  BlockWeights weights_;
  uint32_t planeInputs_ = 0;
};

// A token table's rows as the GGUF stores them, in a gguf_embedding_format
// (metal/abi/Gguf.h), gathered, never multiplied (Embedding.cpp). A default
// value holds no rows, which a reader replaces.
struct NativeRows final {
  NativeRows() = default;
  NativeRows(metal::MetalBuffer rows, uint32_t formatId);
  metal::MetalBuffer rows;
  uint32_t formatId = 0;
  [[nodiscard]] const char *name() const noexcept;
};

// A token table of outputSize rows of inputSize values, which Embedding
// gathers. It is intentionally a separate type: no table may be bound as a
// projection.
class EmbeddingWeights final {
public:
  EmbeddingWeights() = default;
  EmbeddingWeights(uint32_t output, uint32_t input, NativeRows rows)
      : outputSize(output), inputSize(input), rows_(std::move(rows)) {}

  [[nodiscard]] const NativeRows &blocks() const noexcept { return rows_; }

  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  // Native rows stored rotated, gathered as D (H r) (InputRotation).
  InputRotation rotation;

private:
  NativeRows rows_;
};

} // namespace splash::ops

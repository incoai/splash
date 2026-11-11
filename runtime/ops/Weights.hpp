#pragma once

#include "metal/MetalBackend.hpp"
#include "metal/abi/QuantFormat.h"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace splash::ops {

// How selection classes a weight unit's quantized segments together (a
// projection or a gate/up pair; FormatClassifier):
// GridLookup when every segment's register-tile operands come from grid
// lookups (IQ3_XXS, the IQ2 formats, IQ1); GridLookupOrQ2K when every one is
// GridLookup or Q2_K, at least one Q2_K; otherwise FourBitsOrMore when every
// one has at least four bits per weight (quant_four_bits) or none is
// quantized; otherwise UnderFourBits.
enum class FormatClass : uint8_t { GridLookup, GridLookupOrQ2K, UnderFourBits, FourBitsOrMore };
inline constexpr size_t kFormatClasses = 4;

// The FormatClass of the quantized segments' formats it is given.
class FormatClassifier final {
public:
  constexpr void add(uint32_t format) noexcept {
    grid_ &= gridLookup(format) || format == GGUF_FMT_Q2K;
    q2k_ |= format == GGUF_FMT_Q2K;
    fourBits_ &= quant_four_bits(format);
    any_ = true;
  }
  [[nodiscard]] constexpr FormatClass formatClass() const noexcept {
    if (any_ && grid_) return q2k_ ? FormatClass::GridLookupOrQ2K : FormatClass::GridLookup;
    return fourBits_ ? FormatClass::FourBitsOrMore : FormatClass::UnderFourBits;
  }

private:
  [[nodiscard]] static constexpr bool gridLookup(uint32_t format) noexcept {
    switch (format) {
    case GGUF_FMT_IQ3XXS: case GGUF_FMT_IQ2XXS: case GGUF_FMT_IQ2XS: case GGUF_FMT_IQ2S:
    case GGUF_FMT_IQ1S: case GGUF_FMT_IQ1M: return true;
    default: return false;
    }
  }
  bool grid_ = true, q2k_ = false, fourBits_ = true, any_ = false;
};

struct ProjectionShape final {
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  // Its quantized segments multiply the rotated input (InputRotation), which
  // takes LinearScratch::rotated.
  bool rotated = false;
  // The Apple9 register tile takes all its quantized segments
  // (quant_register_tensor); a projection with one it does not take stages
  // whatever its rows (Linear::decodeStorageRows, decodeScratchSize).
  bool registerFormats = true;
  // The FormatClass of its quantized segments, whose RegisterBoundary row
  // bounds the register tile (Linear::decodeStorageRows).
  FormatClass formatClass = FormatClass::FourBitsOrMore;
  // The bits per weight its quantized segments store together, its float
  // ones aside, in sixteenths, rounded down; 0 where it has none or they are
  // unknown, which Apple10's staged split law takes for the most a format
  // stores as it weighs a split's partials against the weights. The bits,
  // with the wave law from 4.5, decide whether one lane runs the 16-row tile,
  // never at 0 (KernelSelection.cpp).
  uint32_t weightBits16 = 0;
  // The int8 prefill tile takes all its quantized segments
  // (quant_prefill_int8_format), of which it has one at least; a shape that
  // names no formats bounds a projection the tile takes
  // (Linear::prefillScratchSize).
  bool prefillInt8Formats = true;
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

// A prepared tensor occupying a projection's output columns
// [columnOffset, columnOffset + outputSize), in one of two layouts. Planes:
// repacked planes in the GGUF_FMT_* format formatId (metal/abi/QuantFormat.h),
// which the block tiles read. Rows: the tensor's rows as stored, a row of
// inputSize values per output column in plane0, which the float projection
// multiplies unrounded in fp32 (kernels/shared/gguf_float.metal): F32 values
// (formatId kFloat32, which no GGUF_FMT_* format names) or BF16 values
// (GGUF_FMT_BF16), as a GGUF keeps a tensor unquantized, or a quantized
// format's native blocks, each value the fp32 its decode gives, as the token
// gather decodes it. A segment of rows is a float segment (isFloat); the MoE
// router is one, in the format its checkpoint holds it. The source container
// is not part of it.
struct QuantizedSegment final {
  // The formatId of F32 rows.
  static constexpr uint32_t kFloat32 = 0xffffffff;
  enum class Layout : uint8_t { Planes, Rows };

  [[nodiscard]] static QuantizedSegment planes(uint32_t formatId, uint32_t outputSize, uint32_t inputSize,
                                               metal::MetalBuffer plane0, metal::MetalBuffer plane1,
                                               metal::MetalBuffer meta);
  // Rows of F32 or BF16 values or of a quantized format's native blocks,
  // whole ones each (rowBytes); throws for F16 and any other formatId.
  [[nodiscard]] static QuantizedSegment rows(uint32_t formatId, uint32_t outputSize, uint32_t inputSize,
                                             metal::MetalBuffer rows);
  // Rows of F32 values.
  [[nodiscard]] static QuantizedSegment floats(uint32_t outputSize, uint32_t inputSize, metal::MetalBuffer values) {
    return rows(kFloat32, outputSize, inputSize, std::move(values));
  }
  // The bytes of a row of `inputSize` values in formatId's rows; 0 for a
  // formatId that has none, or a row of a partial native block.
  [[nodiscard]] static uint64_t rowBytes(uint32_t formatId, uint32_t inputSize) noexcept;

  metal::MetalBuffer plane0;
  metal::MetalBuffer plane1;
  metal::MetalBuffer meta;
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  uint32_t columnOffset = 0;
  uint32_t formatId = kFloat32;
  Layout layout = Layout::Rows;

  [[nodiscard]] bool isFloat() const noexcept { return layout == Layout::Rows; }
  // Rows of a quantized format, which the float projection decodes.
  [[nodiscard]] bool quantizedRows() const noexcept {
    return isFloat() && formatId != kFloat32 && formatId != GGUF_FMT_BF16;
  }
  // The plane geometry or native block of a GGUF_FMT_* format.
  [[nodiscard]] const QuantFormat &format() const noexcept;
  // The kernel name suffix of its format: its kQuantFormats name, "f32" for
  // F32 rows.
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
    bool registerFormats = true, prefillInt8Formats = true;
    FormatClassifier classes;
    // The quantized segments' weights and the bytes of their native blocks.
    uint64_t weights = 0, bytes = 0;
    for (const QuantizedSegment &s : weights_.segments) {
      registerFormats &= s.isFloat() || quant_register_tensor(s.formatId, s.inputSize);
      if (s.isFloat()) continue;
      prefillInt8Formats &= quant_prefill_int8_format(s.formatId);
      classes.add(s.formatId);
      const uint64_t n = uint64_t{s.outputSize} * s.inputSize;
      weights += n;
      bytes += n * s.format().block_bytes / s.format().block_elements;
    }
    return {outputSize, inputSize, static_cast<bool>(rotation), registerFormats, classes.formatClass(),
            weights ? uint32_t(128 * bytes / weights) : 0, weights && prefillInt8Formats};
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

// A token table's rows as stored, gathered, never multiplied (Embedding.cpp):
// whole native blocks of a GGUF format (metal/abi/QuantFormat.h), or BF16 or
// F16 values at any width (quant_stored_row_bytes), which a GGUF or a
// safetensors checkpoint may keep unquantized. A default value holds no rows,
// which a reader replaces.
struct StoredRows final {
  StoredRows() = default;
  // Rows of a GGUF_FMT_*; throws for any other formatId.
  StoredRows(metal::MetalBuffer rows, uint32_t formatId);
  metal::MetalBuffer rows;
  uint32_t formatId = 0;
  // Its format's native block and kernel name suffix.
  [[nodiscard]] const QuantFormat &format() const noexcept;
};

// A token table of outputSize rows of inputSize values, which Embedding
// gathers. It is intentionally a separate type: no table may be bound as a
// projection.
class EmbeddingWeights final {
public:
  EmbeddingWeights() = default;
  EmbeddingWeights(uint32_t output, uint32_t input, StoredRows rows)
      : outputSize(output), inputSize(input), rows_(std::move(rows)) {}

  [[nodiscard]] const StoredRows &rows() const noexcept { return rows_; }

  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  // Rows stored rotated, gathered as D (H r) (InputRotation).
  InputRotation rotation;

private:
  StoredRows rows_;
};

} // namespace splash::ops

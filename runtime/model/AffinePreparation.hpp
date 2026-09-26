#pragma once

// The affine images of a checkpoint in the existing packed ABI, as
// model/AffineTarget.cpp plans an MLX target's and model/DraftCheckpoint.cpp a
// DFlash2 draft's: their identity and their writer. An MLX projection's
// codes, scales and biases are reordered into 256-row tiles without
// requantization, a BF16 projection is quantized into the same tiles as MLX's
// affine quantization rounds it, the GDN decay becomes
// float(-exp(double(A_log))), and every other tensor is copied as stored.

#include "model/PreparedWeights.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace splash::model::affine {

enum class SectionKind { Copy, Decay, Projection, Quantize };

// A checkpoint tensor a section reads: its name, the dtypes it is read in and
// its shape, planned from the layout; tensor is bound once the checkpoint is
// opened.
struct Input {
  std::string name;
  std::vector<std::string> dtypes;
  std::vector<uint64_t> shape;
  const SourceTensor *tensor = nullptr;
};

// Source rows of a fused projection: its MLX weight, scales and biases
// (Projection), or its BF16 weight (Quantize).
struct ProjectionPart {
  uint32_t rows = 0;
  std::vector<Input> fields;
};

struct Section {
  SectionKind kind = SectionKind::Copy;
  uint64_t offset = 0, bytes = 0;
  Input input; // Copy and Decay
  // Projection and Quantize: the parts in row order. A Projection's rows past
  // them are zero; a Quantize section has none, nor experts, and 4 bits.
  std::vector<ProjectionPart> parts;
  uint32_t rows = 0, columns = 0, experts = 1, bits = 4;
};

// A 16-byte header (magic, layer, type) in a 16 KiB block, then 16 KiB-aligned
// sections; quantized lists the affine modules it reads and their bits.
struct Image {
  std::string name, magic;
  uint32_t layer = 0, type = 0;
  uint64_t bytes = 0;
  std::vector<Section> sections;
  std::vector<std::pair<std::string, uint32_t>> quantized;
};

// The identity of an image planned from a checkpoint at `source`, the
// component directory/name.
[[nodiscard]] PreparedWeight affineImageWeight(const Image &image, std::string_view directory,
                                               const std::string &source);

// Writes an image into its preallocated, zeroed destination within the
// preparation staging bound; admit runs before each chunk.
void writeAffineImage(int destination, const Image &image, const PreparationCheck &admit);
// writeAffineImage for image, which outlives the writer.
[[nodiscard]] inline WeightWriter affineImageWriter(const Image &image) {
  return [&image](int destination, const PreparationCheck &admit) { writeAffineImage(destination, image, admit); };
}

} // namespace splash::model::affine

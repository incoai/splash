#pragma once

// The affine images of a checkpoint, laid out as a Splash package's files,
// which the target and draft loaders read (QwenTargetLoader.hpp,
// DFlashDraft.cpp): model/AffineTarget.cpp plans an MLX target's and
// model/DraftCheckpoint.cpp a DFlash2 draft's, and writeAffineImage writes
// them. An MLX projection's codes, scales and biases are reordered into
// 256-row tiles without requantization, a BF16 projection is quantized into
// the same tiles as MLX's affine quantization rounds it, the GDN decay becomes
// float(-exp(double(A_log))), and every other tensor is copied as stored.

#include "model/WeightSource.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <utility>
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
// sections; quantized lists the affine modules it reads and their bits, which
// inspection holds an MLX checkpoint's quantization to (ModelDescriptor.mm).
struct Image {
  std::string name, magic;
  uint32_t layer = 0, type = 0;
  uint64_t bytes = 0;
  std::vector<Section> sections;
  std::vector<std::pair<std::string, uint32_t>> quantized;
};

// Writes every byte of a bound image into destination, which is its size.
void writeAffineImage(std::span<uint8_t> destination, const Image &image);

} // namespace splash::model::affine

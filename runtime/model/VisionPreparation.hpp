#pragma once

// The image of a vision tower (MDFV0001, every tensor BF16), as
// model/VisionLoader.cpp plans it from an MLX checkpoint or a GGUF mmproj, and
// its writer. BF16 is copied; F32 and F16 require exact BF16 values. An MLX
// affine matrix is decoded to nearest BF16 from its packed codes and parameters.

#include "model/WeightSource.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace splash::model::vision {

// A source tensor and its name; for an affine matrix, tensor holds its U32
// codes and scales/biases supply one BF16 parameter per group of columns.
struct Input {
  std::string name;
  SourceTensor tensor;
  const SourceTensor *scales = nullptr, *biases = nullptr;
  uint32_t bits = 0, group = 0;
};

// A section of rows x columns values, stored as storedRows x storedColumns
// (padding stays zero), written from one source tensor, or from the two
// temporal frames of a GGUF patch embedding.
struct Section {
  std::string mlx, gguf; // the tensor's names in either source
  uint32_t rows, columns, storedRows, storedColumns;
  bool patch = false; // the patch embedding, whose rows the writer reorders
  uint64_t offset = 0;
  std::vector<Input> inputs{};
};

// The header (magic, block count, file kind 0) in a 16 KiB block, then
// 16 KiB-aligned sections.
struct Plan {
  uint32_t depth = 0, patchSize = 0;
  uint64_t bytes = 0;
  std::vector<Section> sections;
};

// Writes every byte of a bound plan into destination, which is its size.
void writeVision(std::span<uint8_t> destination, const Plan &plan);

} // namespace splash::model::vision

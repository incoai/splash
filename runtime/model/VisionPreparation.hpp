#pragma once

// The packed vision/model.bin (MDFV0001, every tensor BF16) of a vision
// tower, as model/VisionLoader.cpp plans it from an MLX checkpoint or a GGUF
// mmproj: its identity and its writer. A BF16 tensor is copied; an F32 or F16
// tensor is converted only when every value is exactly a BF16, and
// preparation fails otherwise.

#include "model/PreparedWeights.hpp"

#include <string>
#include <vector>

namespace splash::model::vision {

// A source tensor and the name it has in its file.
struct Input {
  std::string name;
  SourceTensor tensor;
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

// The identity of a plan of the source at `source`.
[[nodiscard]] PreparedWeight visionWeight(const Plan &plan, const std::string &source);

// Writes a plan into its preallocated, zeroed destination within the
// preparation staging bound; admit runs before each chunk.
void writeVision(int destination, const Plan &plan, const PreparationCheck &admit);

} // namespace splash::model::vision

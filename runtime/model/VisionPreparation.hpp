#pragma once

// The image of a vision tower (MDFV0001), as its tower plans it from a
// safetensors checkpoint or a GGUF mmproj (Qwen's in model/VisionLoader.cpp,
// Inkling's in model/InklingVision.cpp), and its writer. A section is BF16
// values: a BF16 tensor is copied; an F32 or F16 tensor is converted only
// when every value is exactly a BF16, and loading fails otherwise. A
// projection's matrix whose inputs are all Q8_0 (an Inkling mmproj's) is that
// Q8_0 tensor as a GGUF image holds one (model/GgufImageLayout.hpp): a
// descriptor, then its plane0 and meta sections, each block's 32 codes and
// its F16 scale byte for byte (Section::q8Planes). Any other Q8_0 tensor, a
// vector, a table or one of q, k and v beside another type, is refused.

#include "model/WeightSource.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace splash::model::vision {

// A source tensor and the name it has in its file.
struct Input {
  std::string name;
  SourceTensor tensor;
};

// A section of rows x columns values, stored as storedRows x storedColumns
// (padding stays zero), written from one source tensor, from the two
// temporal frames of a GGUF patch embedding, or, stacked, from several
// tensors of rows / inputs rows each, one after another (Inkling's q, k and
// v). A section without inputs is all zero (Inkling's absent biases).
struct Section {
  // A Qwen tower's tensor's names in either source; of an Inkling tower's,
  // gguf alone, what its section holds (its mmproj tensor's name, or a label).
  std::string mlx, gguf;
  uint32_t rows, columns, storedRows, storedColumns;
  bool patch = false; // the patch embedding, whose rows the writer reorders
  uint64_t offset = 0;
  std::vector<Input> inputs{};
  bool stacked = false;
  // A projection's matrix, which its tower reads as Q8_0 planes too
  // (ops::DenseWeights::quantized); and stored so, when its inputs are all
  // Q8_0, of whole 64-row plane tiles and unpadded columns of whole blocks.
  bool projection = false;
  bool q8Planes = false;
};

// The header (magic, block count, file kind 0) in a 16 KiB block, then
// 16 KiB-aligned sections.
struct Plan {
  uint32_t depth = 0, patchSize = 0;
  uint64_t bytes = 0;
  std::vector<Section> sections;
};

// The bytes a section takes from its offset: rows x storedColumns BF16
// values, or a Q8_0 matrix's descriptor, plane0 and meta sections, each
// 16 KiB aligned.
[[nodiscard]] uint64_t sectionBytes(const Section &s);
// The offsets of a Q8_0 matrix's plane0 and meta sections.
[[nodiscard]] std::pair<uint64_t, uint64_t> q8PlaneOffsets(const Section &s);
// Sets each section's offset, one after another from the header's 16 KiB
// block, each 16 KiB aligned, and the plan's bytes.
void placeSections(Plan &plan);
// Marks each projection whose inputs are all Q8_0 as stored so (q8Planes)
// and places the sections again; throws, naming it, for a Q8_0 input of any
// other section.
void placeQ8Matrices(Plan &plan);

// Each section of a bound plan and the format it loads in, Q8_0 planes or
// BF16 values, with its shape, one line each: what its image's record names
// (WeightFileRecord::formats).
[[nodiscard]] std::string imageFormats(const Plan &plan);

// Writes every byte of a bound plan into destination, which is its size.
void writeVision(std::span<uint8_t> destination, const Plan &plan);

} // namespace splash::model::vision

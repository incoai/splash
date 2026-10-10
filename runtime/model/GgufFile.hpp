#pragma once

// Header parser for llama.cpp GGUF files (version 3): metadata scalars,
// strings and small numeric arrays, the tensor table and where the tensor
// data starts. Tensor data is never read here.

#include "model/WeightSource.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace splash::model {

class GgufError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// The ggml type ids production names, as stored in GGUF tensor infos; the
// rest are looked up in kGgmlTypes. llama.cpp's NVFP4 (kNVFP4) loads into
// the NVFP4 format's native rows (model/GgufImage.cpp, imageFormat).
namespace ggml {
inline constexpr uint32_t kF32 = 0, kQ8_0 = 8, kBF16 = 30, kNVFP4 = 40, kPQ2_0 = 142;
}

struct GgmlTypeTraits {
  const char *name;
  uint32_t blockElements;
  uint32_t blockBytes;
};

// (id, name, block elements, block bytes) of the ggml types this parser can
// size, as ggml-common.h defines them and, for 142, PrismML-Eng/llama.cpp's
// block_pq2_0; a tensor of another type is rejected.
inline constexpr std::array<std::pair<uint32_t, GgmlTypeTraits>, 32> kGgmlTypes{{
    {0, {"F32", 1, 4}},         {1, {"F16", 1, 2}},         {2, {"Q4_0", 32, 18}},
    {3, {"Q4_1", 32, 20}},      {6, {"Q5_0", 32, 22}},      {7, {"Q5_1", 32, 24}},
    {8, {"Q8_0", 32, 34}},      {9, {"Q8_1", 32, 36}},      {10, {"Q2_K", 256, 84}},
    {11, {"Q3_K", 256, 110}},   {12, {"Q4_K", 256, 144}},   {13, {"Q5_K", 256, 176}},
    {14, {"Q6_K", 256, 210}},   {15, {"Q8_K", 256, 292}},   {16, {"IQ2_XXS", 256, 66}},
    {17, {"IQ2_XS", 256, 74}},  {18, {"IQ3_XXS", 256, 98}}, {19, {"IQ1_S", 256, 50}},
    {20, {"IQ4_NL", 32, 18}},   {21, {"IQ3_S", 256, 110}},  {22, {"IQ2_S", 256, 82}},
    {23, {"IQ4_XS", 256, 136}}, {24, {"I8", 1, 1}},         {25, {"I16", 1, 2}},
    {26, {"I32", 1, 4}},        {27, {"I64", 1, 8}},        {28, {"F64", 1, 8}},
    {29, {"IQ1_M", 256, 56}},   {30, {"BF16", 1, 2}},       {39, {"MXFP4", 32, 17}},
    {40, {"NVFP4", 64, 36}},    {142, {"PQ2_0", 128, 34}},
}};

// nullptr for type ids this parser does not know.
[[nodiscard]] constexpr const GgmlTypeTraits *ggmlTypeTraits(uint32_t type) noexcept {
  for (const auto &[id, traits] : kGgmlTypes)
    if (id == type) return &traits;
  return nullptr;
}
[[nodiscard]] std::string ggmlTypeName(uint32_t type);

struct GgufTensor {
  std::string name;
  uint32_t type = 0;
  std::vector<uint64_t> dims; // dims[0] is the fastest (row length)
  uint64_t offset = 0;        // in the file's tensor data
  uint64_t bytes = 0;
  uint32_t file = 0;          // which file of a split GGUF holds it
  [[nodiscard]] uint64_t columns() const noexcept { return dims.empty() ? 0 : dims[0]; }
  [[nodiscard]] uint64_t rows() const;
  [[nodiscard]] uint64_t elements() const;
};

// Prism ML's input rotation (prism.hadamard.* keys, metal/abi/Gguf.h): the
// tensors whose weights were stored for rotated inputs, the token tables
// stored rotated and the signs of each input width, one explicit int8 sign
// per input. The parser keeps only the one transform the kernels run and
// checks that every named tensor has the signs of its width.
struct GgufRotation {
  // The GDN value dimension of the rotated inputs is in grouped head order
  // (prism.hadamard.gdn_v_grouped), not llama.cpp's tiled one.
  bool valueHeadsGrouped = false;
  std::set<std::string, std::less<>> weights;
  std::set<std::string, std::less<>> tables;
  std::map<uint32_t, std::vector<int8_t>> signs;
};

// A GGUF's scalar metadata by key: every integer and boolean as unsigned, a
// negative integer as its two's complement, every float as a double, and the
// strings. GgufFile reads it from a header, and model-check from the
// installer's copy of the header's (install/gguf.py scalar_metadata), so that
// gguf::requireMetadata holds either to the same rules.
struct GgufMetadata {
  std::map<std::string, uint64_t, std::less<>> unsigneds;
  std::map<std::string, double, std::less<>> floats;
  std::map<std::string, std::string, std::less<>> strings;

  [[nodiscard]] std::optional<uint64_t> unsignedValue(std::string_view key) const;
  [[nodiscard]] std::optional<double> floatValue(std::string_view key) const;
  [[nodiscard]] std::optional<std::string> stringValue(std::string_view key) const;
  // general.architecture, empty when the metadata names none.
  [[nodiscard]] std::string architecture() const;
};

class GgufFile final {
public:
  // Parses the header of source and sets where its tensor data starts.
  explicit GgufFile(WeightSource &source);
  // The files of a split GGUF (llama.cpp's gguf-split), in split order: the
  // first one's metadata, whose split.* keys must describe these files, and
  // every file's tensors, each tagged with its file.
  explicit GgufFile(std::span<WeightSource *const> sources);

  [[nodiscard]] const WeightSource &source() const noexcept { return *sources_.front(); }
  [[nodiscard]] const WeightSource &source(uint32_t file) const { return *sources_.at(file); }
  [[nodiscard]] uint32_t files() const noexcept { return static_cast<uint32_t>(sources_.size()); }
  [[nodiscard]] const GgufMetadata &metadata() const noexcept { return metadata_; }
  [[nodiscard]] std::optional<std::span<const double>> numericArray(std::string_view key) const;
  // The rotation the metadata declares, if any.
  [[nodiscard]] const std::optional<GgufRotation> &rotation() const noexcept { return rotation_; }

  [[nodiscard]] const std::vector<GgufTensor> &tensors() const noexcept { return tensors_; }
  [[nodiscard]] const GgufTensor *find(std::string_view name) const noexcept;
  [[nodiscard]] const GgufTensor &require(std::string_view name) const;

private:
  void parse(WeightSource &source, uint32_t file);
  std::vector<WeightSource *> sources_;
  GgufMetadata metadata_;
  std::map<std::string, std::vector<double>, std::less<>> arrays_;
  // The string arrays of the rotation keys, the only ones kept.
  std::map<std::string, std::vector<std::string>, std::less<>> names_;
  std::optional<GgufRotation> rotation_;

  [[nodiscard]] std::optional<GgufRotation> readRotation() const;
  std::vector<GgufTensor> tensors_;
  std::map<std::string, size_t, std::less<>> index_;
};

} // namespace splash::model

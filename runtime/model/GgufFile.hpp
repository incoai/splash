#pragma once

// Header parser for llama.cpp GGUF files (version 3): metadata scalars,
// strings and small numeric arrays, the tensor table and where the tensor
// data starts. Tensor data is never read here.

#include "model/PreparedWeights.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace splash::model {

class GgufError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// ggml type ids as stored in GGUF tensor infos.
namespace ggml {
inline constexpr uint32_t kF32 = 0, kF16 = 1, kQ4_0 = 2, kQ4_1 = 3, kQ8_0 = 8,
                          kQ2_K = 10, kQ3_K = 11, kQ4_K = 12, kQ5_K = 13, kQ6_K = 14,
                          kIQ2_XXS = 16, kIQ2_XS = 17, kIQ3_XXS = 18, kIQ1_S = 19,
                          kIQ4_NL = 20, kIQ3_S = 21, kIQ2_S = 22, kIQ4_XS = 23,
                          kIQ1_M = 29, kBF16 = 30, kMXFP4 = 39, kPQ2_0 = 142;
}

struct GgmlTypeTraits {
  const char *name;
  uint32_t blockElements;
  uint32_t blockBytes;
};
// nullptr for type ids this parser does not know.
[[nodiscard]] const GgmlTypeTraits *ggmlTypeTraits(uint32_t type) noexcept;
[[nodiscard]] std::string ggmlTypeName(uint32_t type);

struct GgufTensor {
  std::string name;
  uint32_t type = 0;
  std::vector<uint64_t> dims; // dims[0] is the fastest (row length)
  uint64_t offset = 0;        // in the file's tensor data
  uint64_t bytes = 0;
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

class GgufFile final {
public:
  // Parses the header of source and sets where its tensor data starts.
  explicit GgufFile(WeightSource &source);

  [[nodiscard]] const WeightSource &source() const noexcept { return source_; }
  [[nodiscard]] const std::string &architecture() const noexcept { return architecture_; }

  [[nodiscard]] std::optional<uint64_t> unsignedValue(std::string_view key) const;
  [[nodiscard]] std::optional<std::string> stringValue(std::string_view key) const;
  [[nodiscard]] std::optional<double> floatValue(std::string_view key) const;
  [[nodiscard]] std::optional<std::span<const double>> numericArray(std::string_view key) const;
  // The rotation the metadata declares, if any.
  [[nodiscard]] const std::optional<GgufRotation> &rotation() const noexcept { return rotation_; }

  [[nodiscard]] const std::vector<GgufTensor> &tensors() const noexcept { return tensors_; }
  [[nodiscard]] const GgufTensor *find(std::string_view name) const noexcept;
  [[nodiscard]] const GgufTensor &require(std::string_view name) const;

private:
  const WeightSource &source_;
  std::string architecture_;
  std::map<std::string, uint64_t, std::less<>> unsigned_;
  std::map<std::string, std::string, std::less<>> strings_;
  std::map<std::string, double, std::less<>> floats_;
  std::map<std::string, std::vector<double>, std::less<>> arrays_;
  // The string arrays of the rotation keys, the only ones kept.
  std::map<std::string, std::vector<std::string>, std::less<>> names_;
  std::optional<GgufRotation> rotation_;

  [[nodiscard]] std::optional<GgufRotation> readRotation() const;
  std::vector<GgufTensor> tensors_;
  std::map<std::string, size_t, std::less<>> index_;
};

} // namespace splash::model

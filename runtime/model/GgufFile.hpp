#pragma once

// Header parser for llama.cpp GGUF files (version 3): metadata scalars,
// strings and small numeric arrays, the tensor table and where the tensor
// data starts. Tensor data is never read here.

#include "model/PreparedWeights.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <optional>
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

// ggml type ids as stored in GGUF tensor infos.
namespace ggml {
inline constexpr uint32_t kF32 = 0, kF16 = 1, kQ8_0 = 8, kQ3_K = 11, kQ4_K = 12,
                          kQ5_K = 13, kQ6_K = 14, kIQ4_NL = 20, kIQ3_S = 21,
                          kIQ4_XS = 23, kBF16 = 30, kPQ2_0 = 142;
}

struct GgmlTypeTraits {
  const char *name;
  uint32_t blockElements;
  uint32_t blockBytes;
};

// (id, name, block elements, block bytes) of the ggml types this parser can
// size, as ggml-common.h defines them and, for 142, PrismML-Eng/llama.cpp's
// block_pq2_0; a tensor of another type is rejected.
inline constexpr std::array<std::pair<uint32_t, GgmlTypeTraits>, 31> kGgmlTypes{{
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
    {142, {"PQ2_0", 128, 34}},
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
  [[nodiscard]] uint64_t columns() const noexcept { return dims.empty() ? 0 : dims[0]; }
  [[nodiscard]] uint64_t rows() const;
  [[nodiscard]] uint64_t elements() const;
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
  std::vector<GgufTensor> tensors_;
  std::map<std::string, size_t, std::less<>> index_;
};

} // namespace splash::model

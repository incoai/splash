#pragma once

#include "metal/MetalBackend.hpp"
#include "model/WeightLayout.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace splash::model {

class WeightStoreError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

struct WeightFileRecord final {
  std::string relativePath;
  std::string magic;
  uint32_t layer = 0;
  uint32_t type = 0;
  uint64_t declaredBytes = 0;
  // What the image was written from (ModelDescriptor::sourceIdentity).
  std::string contentIdentity{};
};

// The lowercase hex SHA-256 of bytes or text.
[[nodiscard]] std::string weightDigest(std::span<const uint8_t> bytes);
[[nodiscard]] std::string weightDigest(std::string_view text);
// A SHA-256 digest in lowercase hex.
[[nodiscard]] inline std::string digestHex(std::span<const uint8_t, 32> digest) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(2 * digest.size());
  for (uint8_t byte : digest) {
    result += digits[byte >> 4];
    result += digits[byte & 15];
  }
  return result;
}

// The file of a loaded image (WeightImages): its header checked, its
// sections read in order as aligned views of the image's buffer.
class WeightFile final {
public:
  WeightFile(metal::MetalBackend &backend, metal::MetalBuffer image,
             std::string relativePath, std::string_view expectedMagic,
             uint32_t expectedLayer, uint32_t expectedType,
             std::string contentIdentity);
  ~WeightFile();

  WeightFile(const WeightFile &) = delete;
  WeightFile &operator=(const WeightFile &) = delete;
  WeightFile(WeightFile &&) noexcept;
  WeightFile &operator=(WeightFile &&) noexcept;

  [[nodiscard]] metal::MetalBuffer section(uint64_t bytes,
                                            std::string_view label);
  // Requires the sections read to cover the whole file.
  void finish();
  [[nodiscard]] const WeightFileRecord &record() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// A norm of `width` multipliers: F32 when `float32` (a GGUF image keeps its
// norms as the GGUF stores them), bf16 otherwise.
[[nodiscard]] ops::NormWeights readNorm(WeightFile &file, uint32_t width,
                                        bool float32, std::string_view label);

// GGUF image sections: a descriptor, then plane0, optional plane1
// and metadata, each 16 KiB aligned (GgufTensorDescriptor and the layout in
// model/GgufImageLayout.hpp).
[[nodiscard]] ops::QuantizedSegment readQuantizedSegment(WeightFile &file,
                                                   std::string_view label);
// A single-tensor projection; its descriptor must hold the layout's sizes.
[[nodiscard]] ops::Projection readBlockProjection(WeightFile &file, uint32_t outputSize,
                                                  uint32_t inputSize, std::string_view label);
// Native block_q4_K, block_q6_K or block_q8_0 rows for the token table
// (gathered, never multiplied).
[[nodiscard]] ops::EmbeddingWeights readBlockEmbedding(WeightFile &file, uint32_t outputSize,
                                                       uint32_t inputSize, std::string_view label);

[[nodiscard]] std::string
weightManifestFingerprint(std::span<const WeightFileRecord> records);

} // namespace splash::model

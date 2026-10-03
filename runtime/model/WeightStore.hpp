#pragma once

#include "metal/MetalBackend.hpp"
#include "model/WeightLayout.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <filesystem>
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
  std::string contentIdentity{};
};

// A read-only mmap with one no-copy Metal base buffer.  Sections are checked,
// aligned views that retain the mapping; no model loader owns raw mmap state.
// A file with a content identity, its cache key, is a prepared file: it is
// mapped only as the cache verified it (requireVerifiedFile).
class WeightFile final {
public:
  WeightFile(metal::MetalBackend &backend, std::filesystem::path path,
             std::string relativePath, std::string_view expectedMagic,
             uint32_t expectedLayer, uint32_t expectedType,
             std::string contentIdentity = {});
  ~WeightFile();

  WeightFile(const WeightFile &) = delete;
  WeightFile &operator=(const WeightFile &) = delete;
  WeightFile(WeightFile &&) noexcept;
  WeightFile &operator=(WeightFile &&) noexcept;

  [[nodiscard]] metal::MetalBuffer section(uint64_t bytes,
                                            std::string_view label);
  // One section of the parts' total bytes, as a view of each part in order.
  [[nodiscard]] std::vector<metal::MetalBuffer> split(std::initializer_list<uint64_t> parts,
                                                      std::string_view label);
  // Requires the sections read to cover the whole file.
  void finish();
  [[nodiscard]] const WeightFileRecord &record() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] uint64_t q4PackedBytes(uint32_t outputSize,
                                     uint32_t inputSize);
void validateQ4Layout(uint32_t outputSize, uint32_t inputSize);

[[nodiscard]] ops::Projection
readAffineProjection(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                     std::string_view label);

// A norm of `width` multipliers: F32 when `float32` (a GGUF image keeps its
// norms as the GGUF stores them), bf16 otherwise.
[[nodiscard]] ops::NormWeights readNorm(WeightFile &file, uint32_t width,
                                        bool float32, std::string_view label);

// GGUF image sections: a 64-byte descriptor, then plane0, optional plane1
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

// Embedding weights, scales and biases are independently aligned sections
// so token gather can bind each table directly.
[[nodiscard]] ops::EmbeddingWeights
readAffineEmbedding(WeightFile &file, uint32_t outputSize,
                           uint32_t inputSize, std::string_view label);

[[nodiscard]] ops::Q8Projection
readAffineQ8Projection(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                       std::string_view label);

[[nodiscard]] ops::ExpertProjection
readAffineExpertProjection(WeightFile &file, uint32_t experts,
                           uint32_t outputSize, uint32_t inputSize,
                           std::string_view label);

[[nodiscard]] std::string
weightManifestFingerprint(std::span<const WeightFileRecord> records);

} // namespace splash::model

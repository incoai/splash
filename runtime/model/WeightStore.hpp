#pragma once

#include "metal/MetalBackend.hpp"
#include "ops/Linear.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::model {

inline constexpr uint32_t kQ4GroupElements = 64;
inline constexpr uint64_t kBFloat16Bytes = 2;

inline constexpr uint64_t kWeightFileAlignment = 16 * 1024;
inline constexpr uint32_t kQ4StorageN = 256;

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
};

// A read-only mmap with one no-copy Metal base buffer.  Sections are checked,
// aligned views that retain the mapping; no model loader owns raw mmap state.
class WeightFile final {
public:
  WeightFile(metal::MetalBackend &backend, std::filesystem::path path,
             std::string relativePath, std::string_view expectedMagic,
             uint32_t expectedLayer, uint32_t expectedType);
  // An image built in memory (GGUF load-time repack); same header and
  // section rules as a mapped file.
  WeightFile(metal::MetalBackend &backend, metal::MetalBuffer image,
             std::string relativePath, std::string_view expectedMagic,
             uint32_t expectedLayer, uint32_t expectedType);
  ~WeightFile();

  WeightFile(const WeightFile &) = delete;
  WeightFile &operator=(const WeightFile &) = delete;
  WeightFile(WeightFile &&) noexcept;
  WeightFile &operator=(WeightFile &&) noexcept;

  [[nodiscard]] metal::MetalBuffer section(uint64_t bytes,
                                            std::string_view label = {});
  void finish();
  [[nodiscard]] const WeightFileRecord &record() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] uint64_t checkedWeightMultiply(uint64_t left, uint64_t right,
                                             std::string_view description);
[[nodiscard]] uint64_t q4PackedBytes(uint32_t outputSize,
                                     uint32_t inputSize);
void validateQ4Layout(uint32_t outputSize, uint32_t inputSize);

[[nodiscard]] ops::Q4Projection
readQ4Projection(WeightFile &file, metal::MetalBackend &backend,
                 uint32_t outputSize, uint32_t inputSize,
                 std::string_view label);

// GGUF K-quant sections: a 64-byte descriptor, then plane0, optional plane1
// and metadata, each 16 KiB aligned (model/GgufImage.hpp; the same layout
// dev/tools/convert_gguf_to_splash.py writes to disk).
inline constexpr std::string_view kKQuantMagic = "MDKQ0001";
[[nodiscard]] ops::KQuantSegment readKQuantSegment(WeightFile &file,
                                                   std::string_view label);
[[nodiscard]] ops::Q4Projection readKQuantProjection(WeightFile &file,
                                                     std::string_view label);
// Native block_q4_K rows for the token table (gathered, never multiplied).
[[nodiscard]] ops::Q4Projection readKQuantEmbedding(WeightFile &file,
                                                    std::string_view label);

// Embedding weights, scales and biases are independently aligned sections
// so token gather can bind each table directly.
[[nodiscard]] ops::Q4Projection
readQ4ProjectionComponents(WeightFile &file, uint32_t outputSize,
                           uint32_t inputSize, std::string_view label);

[[nodiscard]] ops::Q8Projection
readQ8Projection(WeightFile &file, metal::MetalBackend &backend,
                 uint32_t outputSize, uint32_t inputSize,
                 std::string_view label);

[[nodiscard]] ops::ExpertQ4Projection
readExpertQ4Projection(WeightFile &file, uint32_t experts,
                       uint32_t outputSize, uint32_t inputSize,
                       std::string_view label);

[[nodiscard]] std::string
weightManifestFingerprint(std::span<const WeightFileRecord> records);

} // namespace splash::model

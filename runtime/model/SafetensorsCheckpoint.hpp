#pragma once

#include "model/WeightSource.hpp"

#include <filesystem>
#include <memory>
#include <string_view>

namespace splash::model {

// Checkpoint configuration and safetensors index. Opening parses only
// metadata; the images read tensor data in bounded slices, without loading
// the MLX runtime or allocating tensors.
class SafetensorsCheckpoint final {
public:
  explicit SafetensorsCheckpoint(const std::filesystem::path &directory);
  ~SafetensorsCheckpoint();
  [[nodiscard]] const SourceTensor *find(std::string_view name) const noexcept;
  [[nodiscard]] const SourceTensor &require(std::string_view name) const;
  void requireQuantization(std::string_view projection, uint32_t bits) const;
  void requireConfigNumber(std::string_view key, double expected) const;
  // `legacyKey`, when given, names the field in configurations that predate `key`.
  void requireConfigString(std::string_view key, std::string_view expected,
                           std::string_view legacyKey = {}) const;
  void requireLayerTypes(uint32_t layers, uint32_t fullAttentionPeriod) const;
  void checkUnchanged() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::model

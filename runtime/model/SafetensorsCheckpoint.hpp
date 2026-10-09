#pragma once

#include "model/WeightSource.hpp"

#include <filesystem>
#include <memory>
#include <string_view>

namespace splash::model {

// A checkpoint's safetensors index. Opening parses only its shards' headers;
// the images read tensor data in bounded slices, without loading the MLX
// runtime or allocating tensors. It reads no configuration: inspection checks
// the model's config.json, the target's quantization included
// (inspectModelRoot).
class SafetensorsCheckpoint final {
public:
  explicit SafetensorsCheckpoint(const std::filesystem::path &directory);
  ~SafetensorsCheckpoint();
  [[nodiscard]] const SourceTensor *find(std::string_view name) const noexcept;
  [[nodiscard]] const SourceTensor &require(std::string_view name) const;
  void checkUnchanged() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::model

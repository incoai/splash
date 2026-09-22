#pragma once

// Loads a Qwen3.8 target straight from a llama.cpp GGUF: each layer image is
// planned (GgufImage), allocated as one anonymous Metal buffer, filled by the
// CPU (header, descriptors, small tensors) and by the gguf_repack / gguf_copy
// kernels reading the mmapped file, then handed out as a WeightFile so the
// section readers are the ones used for packed files.

#include <filesystem>
#include <memory>

#include "metal/MetalBackend.hpp"
#include "model/GgufFile.hpp"
#include "model/GgufImage.hpp"
#include "model/WeightStore.hpp"

namespace splash::model {

// The single .gguf in a target directory (shards are not supported).
[[nodiscard]] std::filesystem::path findTargetGguf(const std::filesystem::path &directory);

class GgufTargetLoader final {
public:
  GgufTargetLoader(metal::MetalBackend &backend, std::filesystem::path path,
                   gguf::TargetGeometry geometry);
  ~GgufTargetLoader();
  GgufTargetLoader(const GgufTargetLoader &) = delete;
  GgufTargetLoader &operator=(const GgufTargetLoader &) = delete;

  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile head();
  [[nodiscard]] WeightFile embedding();
  [[nodiscard]] const GgufFile &file() const noexcept { return file_; }
  [[nodiscard]] const gguf::ImagePlanner &planner() const noexcept { return planner_; }

private:
  [[nodiscard]] WeightFile build(const gguf::Image &image, uint32_t expectedLayer,
                                 uint32_t expectedType);

  metal::MetalBackend *backend_;
  GgufFile file_;
  gguf::ImagePlanner planner_;
  int descriptor_ = -1;
};

} // namespace splash::model

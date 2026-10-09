#pragma once

// Source adapter for an MLX target: its block images (model/MlxImage.hpp) are
// written into memory, and QwenTargetLoader reads them as block-quantized
// weights (BlockTargetFormat).

#include "model/GgufImage.hpp"
#include "model/QwenHybridLayout.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightImages.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace splash::model {

class MlxTargetLoader final {
public:
  // Plans every image of the checkpoint in directory once.
  MlxTargetLoader(metal::MetalBackend &backend, WeightImages &images, const std::filesystem::path &directory,
                  const QwenTargetDimensions &geometry);
  MlxTargetLoader(const MlxTargetLoader &) = delete;
  MlxTargetLoader &operator=(const MlxTargetLoader &) = delete;

  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile head();
  [[nodiscard]] WeightFile embedding();

private:
  // The checkpoint and its images, which their writers share.
  struct Planned {
    explicit Planned(const std::filesystem::path &directory) : checkpoint(directory) {}
    SafetensorsCheckpoint checkpoint;
    std::vector<gguf::Image> images; // layers, head, embedding
  };
  [[nodiscard]] WeightFile open(size_t index);

  metal::MetalBackend &backend_;
  WeightImages &images_;
  std::shared_ptr<Planned> planned_;
};

// The bytes of every image of the MLX target in directory.
[[nodiscard]] uint64_t mlxTargetImageBytes(const std::filesystem::path &directory,
                                           const QwenTargetDimensions &geometry);

} // namespace splash::model

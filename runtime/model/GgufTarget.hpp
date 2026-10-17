#pragma once

// Source adapter for a Qwen GGUF: its images (model/GgufImageLayout.hpp) are
// written into memory, and QwenTargetLoader reads them as block-quantized
// weights (BlockTargetFormat).

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

#include "model/GgufFile.hpp"
#include "model/GgufImage.hpp"
#include "model/QwenHybridLayout.hpp"
#include "model/WeightImages.hpp"

namespace splash::model {

// The .gguf files of a target directory: its one GGUF, or every part of a
// split one (NAME-0000K-of-0000N.gguf, K = 1..N) in split order.
[[nodiscard]] std::vector<std::filesystem::path> findTargetGgufs(const std::filesystem::path &directory);

class GgufTargetLoader final {
public:
  // Plans every image from the GGUF's metadata once: of its one file, or of
  // every part of a split one in split order (findTargetGgufs).
  GgufTargetLoader(metal::MetalBackend &backend, WeightImages &images,
                   const std::vector<std::filesystem::path> &paths, const QwenTargetDimensions &geometry);
  GgufTargetLoader(const GgufTargetLoader &) = delete;
  GgufTargetLoader &operator=(const GgufTargetLoader &) = delete;

  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile head();
  [[nodiscard]] WeightFile embedding();
  // The input rotation of a Prism ML GGUF, which planImages checked names
  // every quantized tensor of the target and its token table.
  [[nodiscard]] const std::optional<GgufRotation> &rotation() const noexcept { return rotation_; }

private:
  // The GGUF's files and its images, which their writers share.
  struct Planned {
    explicit Planned(const std::vector<std::filesystem::path> &paths) : source(paths) {}
    OpenGguf source;
    std::vector<gguf::Image> images; // layers, head, embedding
  };
  [[nodiscard]] WeightFile open(size_t index);

  metal::MetalBackend &backend_;
  WeightImages &images_;
  std::shared_ptr<Planned> planned_;
  std::optional<GgufRotation> rotation_;
};

} // namespace splash::model

#pragma once

#include "model/AffinePreparation.hpp"
#include "model/PreparedFiles.hpp"

#include <memory>
#include <span>

namespace splash::model {

struct Qwen3_8Layout;
struct Qwen3_6MoeLayout;

// Native MLX affine source -> the existing packed target ABI. Both this adapter
// and the block-quantized adapter publish through PreparedWeights and serve
// through WeightFile; neither changes inference kernels. The checkpoint is
// planned once; each image is prepared when it is opened.
class AffineTargetLoader final {
public:
  AffineTargetLoader(metal::MetalBackend &backend, const std::filesystem::path &directory,
                     const Qwen3_8Layout &layout, PreparationCheck admitConversion = {});
  AffineTargetLoader(metal::MetalBackend &backend, const std::filesystem::path &directory,
                     const Qwen3_6MoeLayout &layout, PreparationCheck admitConversion = {});
  ~AffineTargetLoader();
  // Every image's cache identity and size, layers first, for the model's
  // disk check before the first image is written.
  [[nodiscard]] std::span<const PreparedWeight> weights() const noexcept;
  // Writes every missing image and maps none.
  void prepare();
  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile head();
  [[nodiscard]] WeightFile embedding();
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] uint64_t preparedAffineBytes(const Qwen3_8Layout &layout);
[[nodiscard]] uint64_t preparedAffineBytes(const Qwen3_6MoeLayout &layout);
// The planned image of target layer `layer`: its sections at their offsets.
[[nodiscard]] affine::Image affineLayerImage(const Qwen3_8Layout &layout, uint32_t layer);
[[nodiscard]] affine::Image affineLayerImage(const Qwen3_6MoeLayout &layout, uint32_t layer);

} // namespace splash::model

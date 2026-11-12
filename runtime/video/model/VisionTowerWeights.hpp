#pragma once

// The text encoder's vision tower weights (video/ops/VisionTower.hpp), read from the encoder's own MLX checkpoint:
// mlx-community's Qwen3-VL-32B checkpoints keep the tower (`vision_tower.*`, 351 tensors, 1.19 GB) unquantized in
// bf16 at every width, the same bytes in the 4- and 6-bit repositories. Each stage's tensors are read when asked
// for and converted to fp16 (round to nearest even; 183,501 of the 595 M values change, all of them below fp16's
// normal range, relRMS 7e-9), the patch embedding from MLX's (out, t, h, w, c) to the kernels' (out, c, t, h, w),
// fc1 / fc2 zero-padded from 4304 to 4352 intermediate values. The returned buffers are the caller's: a stage's
// memory lives as long as its weights, so the encode streams the tower as it streams the text layers.
#include "metal/MetalBackend.hpp"
#include "video/ops/VisionTower.hpp"

#include <cstdint>

namespace splash::model {
class SafetensorsCheckpoint;
}

namespace splash::video::model {

class VisionTowerWeights final {
public:
  // Checks every tensor of the tower in the checkpoint (names, BF16, shapes); reads none. Throws
  // std::invalid_argument for a checkpoint without the tower or with a quantized one.
  VisionTowerWeights(metal::MetalBackend &backend, const splash::model::SafetensorsCheckpoint &checkpoint);

  [[nodiscard]] ops::VisionTowerEmbedding embedding() const;
  [[nodiscard]] ops::VisionTowerBlock block(uint32_t index) const;
  // 0: the merger; 1 + j: deepstack merger j.
  [[nodiscard]] ops::VisionTowerMerger merger(uint32_t index) const;

private:
  metal::MetalBackend &backend_;
  const splash::model::SafetensorsCheckpoint &checkpoint_;
};

} // namespace splash::video::model

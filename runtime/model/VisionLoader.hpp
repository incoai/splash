#pragma once

#include "model/ModelDescriptor.hpp"
#include "model/WeightImages.hpp"
#include "ops/Vision.hpp"

#include <filesystem>
#include <memory>

namespace splash::model {

// Source adapter for the vision tower: the vision_tower.* tensors of an MLX
// checkpoint or a GGUF mmproj, both written into an image laid out as a Splash
// package's vision/model.bin (model/VisionPreparation.hpp). Construction
// validates the source's metadata and plans the image; tensor values are
// converted only when it is written.
class VisionLoader final {
public:
  VisionLoader(const std::filesystem::path &directory, VisionSource source, const ops::VisionLayout &layout);
  ~VisionLoader();
  VisionLoader(const VisionLoader &) = delete;
  VisionLoader &operator=(const VisionLoader &) = delete;

  [[nodiscard]] const ops::VisionLayout &layout() const noexcept;
  // The image of vision/model.bin.
  [[nodiscard]] ImagePlan image() const;

private:
  struct Planned;
  std::shared_ptr<const Planned> planned_;
};

// The bytes of the image a source is written into.
[[nodiscard]] uint64_t visionImageBytes(const ops::VisionLayout &layout);

// Throws unless the layout is one a vision/model.bin can hold: every size set,
// the heads covering the width, the merger's width the merged patches' and
// the patch embedding's width that of two frames of RGB patches.
void requireVisionLayout(const ops::VisionLayout &layout);

} // namespace splash::model

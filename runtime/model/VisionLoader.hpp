#pragma once

#include "model/ModelDescriptor.hpp"
#include "model/VisionPreparation.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightStore.hpp"
#include "ops/Vision.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace splash::model {

// Source adapter for the vision tower: the tower's tensors of a safetensors
// checkpoint or a GGUF mmproj, Qwen's or Inkling's (model/InklingVision.hpp),
// written into the tower's image (model/VisionPreparation.hpp). Construction
// validates the source's metadata and plans the image; tensor values are
// converted only when it is written.
class VisionLoader final {
public:
  VisionLoader(const std::filesystem::path &directory, VisionSource source, const ops::VisionLayout &layout);
  ~VisionLoader();
  VisionLoader(const VisionLoader &) = delete;
  VisionLoader &operator=(const VisionLoader &) = delete;

  [[nodiscard]] const ops::VisionLayout &layout() const noexcept;
  // The image's sections, bound to the source.
  [[nodiscard]] const vision::Plan &imagePlan() const noexcept;
  // The image of vision/model.bin.
  [[nodiscard]] ImagePlan image() const;
  // Writes the image once with its writer (image), into memory it frees
  // after: what preparation refuses of the source, a weight that is not
  // exactly a BF16 say, is refused here (the engine's vision-check, at
  // installation).
  void check() const;

private:
  struct Planned;
  std::shared_ptr<const Planned> planned_;
};

// A loaded vision tower: its weights, views of its image, the image's record
// and the bytes its load allocated.
struct VisionTowerWeights final {
  ops::VisionWeights tensors;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
};

// The tower's image, written from its upstream source.
[[nodiscard]] VisionTowerWeights loadVisionTowerWeights(metal::MetalBackend &backend, WeightImages &images,
                                                        const VisionLoader &source);

// Throws unless the layout is one a vision/model.bin can hold: every size set,
// the heads covering the width, the merger's width the merged patches' and
// the patch embedding's width that of RGB patches of the tower's frames (two
// of Qwen's, one of Inkling's).
void requireVisionLayout(const ops::VisionLayout &layout);

} // namespace splash::model

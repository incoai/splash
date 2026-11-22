#pragma once

#include "WeightStore.hpp"
#include "ops/Vision.hpp"

namespace splash::model {

// The Qwen targets share one vision-tower architecture. Its language
// projection width belongs to VisionLayout, so its image is independent of a
// particular text model; the loader validated the width the family's layout
// states. Its weights as views of the image (VisionLoader.cpp writes it), in
// the image's order.
[[nodiscard]] ops::VisionWeights readQwenVision(WeightFile &file, const ops::VisionLayout &layout);

} // namespace splash::model

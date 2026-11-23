#pragma once

// Inkling-2-Mini's images as the sections each holds (gguf::Section, by the
// names of llama.cpp's converter's GGUFs), which its GGUF plan
// (model/Inkling2LoopGguf.cpp) and its safetensors plan
// (model/Inkling2LoopSafetensors.cpp) lay out alike.

#include "model/GgufImage.hpp"
#include "model/Inkling2Loop.hpp"

#include <cstdint>
#include <vector>

namespace splash::model::inkling2 {

// Weight layer `index`'s image, in the order its loader reads it
// (model/Inkling2Loop.cpp readLayer).
[[nodiscard]] std::vector<gguf::Section> layerSections(const Inkling2LoopLayout &layout, uint32_t index);
// The head's image before the head itself (output.weight).
[[nodiscard]] std::vector<gguf::Section> headSections(const Inkling2LoopLayout &layout);

} // namespace splash::model::inkling2

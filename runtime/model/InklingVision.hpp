#pragma once

// Inkling 2's vision tower (ops::kInklingVisionLayout) from a clip inkling2v
// mmproj (GGUF, as ggml-org's and Unsloth's releases convert it) or a
// safetensors checkpoint (the release's, an MLX conversion's): its image's
// plan, a source's binding to that plan, and its weights read back from the
// image. The image (vision/model.bin, every tensor BF16) holds, in
// order:
//  - the patch embedding [1024][1200] as the GGUF orders a patch's values
//    ([channel, patch-row, patch-col]), stored with zero columns to 1280,
//    and its zero bias;
//  - the 32x32 position table, one row of 1024 per cell;
//  - the class token's embedding and position, two rows;
//  - the LayerNorm before the blocks, weight and bias;
//  - each block's LayerNorm, q, k and v stacked into one [3072][1024]
//    projection and their biases, the attention output and its bias, its
//    LayerScale, the MLP's LayerNorm, up and down projections with their
//    biases, and its LayerScale;
//  - the projector's two matrices, each with a zero bias.
// Every tensor of an F32, F16 or BF16 mmproj is exact in BF16; a Q8_0
// matrix's values are rounded to it (vision::writeVision).

#include "model/GgufFile.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/VisionPreparation.hpp"
#include "model/WeightStore.hpp"
#include "ops/Vision.hpp"

namespace splash::model {

// The image's sections, bound to no source.
[[nodiscard]] vision::Plan inklingVisionPlan(const ops::VisionLayout &layout);

// Binds every section of plan to its tensors in gguf, an inkling2v mmproj of
// the layout's tower, whose every tensor the tower uses.
void bindInklingMmproj(const GgufFile &gguf, const ops::VisionLayout &layout, vision::Plan &plan);

// Binds every section of plan to its unquantized tensors in a safetensors
// checkpoint of the model: transformers' names (model.vision_tower., as an
// MLX conversion keeps them), the release's (model.visual.), or either under
// vision_tower. in model.visual.'s place (oMLX's, an nvfp4 conversion's).
void bindInklingCheckpoint(const SafetensorsCheckpoint &checkpoint, const ops::VisionLayout &layout,
                           vision::Plan &plan);

// The tower's weights as views of its image.
[[nodiscard]] ops::VisionWeights readInklingVision(WeightFile &file, const ops::VisionLayout &layout);

} // namespace splash::model

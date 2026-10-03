#pragma once

#include "metal/MetalBackend.hpp"
#include "model/GgufImage.hpp"
#include "model/PreparedWeights.hpp"

namespace splash::model {

// The identity of a planned image of source: the preparation identity, the
// whole plan and the bytes and type of every tensor it reads.
[[nodiscard]] PreparedWeight ggufImageWeight(const WeightSource &source, const gguf::Image &image);

// Writes a planned image into its preallocated, zeroed destination file: the
// header and descriptors, the copied rows and the planes the GPU repacks.
// Staging stays within kWeightPreparationStagingBytes whatever the tensor,
// layer or expert count; admit runs before each chunk.
void writeGgufImage(metal::MetalBackend &backend, const WeightSource &source, int destination,
                    const gguf::Image &image, const PreparationCheck &admit);

} // namespace splash::model

#pragma once

#include "metal/MetalBackend.hpp"
#include "model/GgufImage.hpp"
#include "model/WeightSource.hpp"

#include <cstdint>

namespace splash::model {

// The Metal staging of the rows of one repack step, whatever the tensor,
// layer or expert count.
inline constexpr uint64_t kGgufRepackStagingBytes = 32 << 20;

// Writes every byte of a planned image into its buffer: the header and
// descriptors, the copied rows and the planes the GPU repacks into it, each
// read from the source its rows name (a GGUF's, or an MLX checkpoint's
// tensors).
void writeGgufImage(metal::MetalBackend &backend, const metal::MetalBuffer &image, const gguf::Image &plan);

} // namespace splash::model

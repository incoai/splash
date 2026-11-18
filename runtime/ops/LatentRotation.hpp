#pragma once

// Latent rotation (kernels/shared/latent_rotation.metal), DeepSeek-V4.1's:
// vectors RMS normalized or not, their last pairs rotated by their row's
// position, forward or back, and rounded to an FP8 or FP4 grid or not, as
// bf16. The model's window latents, compressed latents, indexer queries and
// keys and attention output each take a combination of them
// (metal/abi/LatentRotation.h).

#include "metal/CommandGraph.hpp"
#include "metal/abi/LatentRotation.h"

#include <cstdint>

namespace splash::ops {

enum class LatentRounding : uint32_t {
  None = LATENT_ROUND_NONE,
  Fp8 = LATENT_ROUND_FP8,
  Fp4E4m3Scale = LATENT_ROUND_FP4_E4M3_SCALE,
  Fp4Pow2Scale = LATENT_ROUND_FP4_POW2_SCALE,
};

// `vectors` vectors of `width` values per row, the last ropePairs pairs of
// each rotated, back where inverse; each RMS normalized first where
// normalize, by weights of `width` values and normEpsilon.
struct LatentRotationShape final {
  uint32_t vectors = 0;
  uint32_t width = 0;
  uint32_t ropePairs = 0;
  bool inverse = false;
  LatentRounding rounding = LatentRounding::None;
  bool normalize = false;
  float normEpsilon = 0.0f;
};

struct LatentRotation final {
  // output = bf16 [rows][vectors][width] of input (bf16, or fp32 where
  // floatInput) of the same shape: normalized by `norm` (fp32 [width],
  // unread unless shape.normalize), rotated by positions[row] (uint32
  // [rows]) times frequencies[pair] (fp32 [ropePairs]), and rounded.
  static void add(metal::CommandGraph &graph, metal::MetalBuffer input, bool floatInput, metal::MetalBuffer norm,
                  metal::MetalBuffer frequencies, metal::MetalBuffer positions, metal::MetalBuffer output,
                  LatentRotationShape shape, uint32_t rows);
};

} // namespace splash::ops

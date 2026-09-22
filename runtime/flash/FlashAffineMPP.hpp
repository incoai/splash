#pragma once

#include "FlashAffine.hpp"

namespace splash::flash {

// Explicit alternate reductions for prefill. GroupAffineF32 feeds exactly
// represented integer codes to BF16 MPP and applies the original F32 affine
// coefficients to each group/chunk. ReconstructedBF16 instead rounds each
// reconstructed coefficient to BF16 before MPP. Neither mode is the canonical
// vector reduction, and both require separate source-weight qualification.
enum class FlashAffineMPPMode : uint32_t {
  GroupAffineF32 = 0,
  ReconstructedBF16 = 1,
};

enum class FlashAffineMPPTile : uint32_t {
  M8N64 = 0,
  M16N64 = 1,
  M16N128 = 2,
  M32N64 = 3,
  M32N128 = 4,
  // Cached whole-K BF16 operands only; original affine MPP keeps its existing
  // geometry set and rejects these selectors.
  M64N64 = 5,
  M64N128 = 6,
};

inline constexpr const char *kFlashAffineMPPGroupSemantics =
    "affine-integer-bf16-mpp-group-f32coeff-f32accum-bf16-v1";
inline constexpr const char *kFlashAffineMPPReconstructedSemantics =
    "affine-f32coeff-rounded-bf16-mpp-f32accum-bf16-v1";

// Dense original-layout projection only: BF16 input[rows,K] -> output[rows,N].
// Static tiles are masked on both row and output tails. No converted model or
// full reconstructed matrix is allocated; source coefficients remain signed.
// The caller clears and checks the same sticky diagnostics as addAffine.
void addAffineMPP(metal::CommandGraph &graph, metal::MetalBuffer input,
                  const FlashAffineProjection &projection,
                  metal::MetalBuffer output, metal::MetalBuffer diagnostics,
                  uint32_t rows, FlashAffineMPPMode mode,
                  FlashAffineMPPTile tile);

} // namespace splash::flash

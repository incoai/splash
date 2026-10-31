#pragma once

// Parameters of the latent rotation kernels (kernels/shared/latent_rotation.metal,
// ops/LatentRotation.cpp), DeepSeek-V4.1's: vectors of `width` values, each
// RMS normalized by a norm's weights or not, its last rope_pairs pairs of
// values rotated as interleaved pairs, by its row's position times each
// pair's frequency or back, then rounded to the grid of the FP8 or FP4 values
// DeepSeek's kernels store them in, or not, as bf16, which holds those grids
// exactly.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The grids a vector may be rounded to (dev/tools/deepseek41_reference.py
// fp8_round and fp4_round, model.py's act_quant and fp4_act_quant): none;
// E4M3 by a power-of-two scale per 32 values (the window's latents, the
// attention's output); E2M1 by an E4M3 scale per 16 values (the compressed
// latents); E2M1 by a power-of-two scale per 32 values (the indexer's queries
// and keys).
#define LATENT_ROUND_NONE 0u
#define LATENT_ROUND_FP8 1u
#define LATENT_ROUND_FP4_E4M3_SCALE 2u
#define LATENT_ROUND_FP4_POW2_SCALE 3u
// One simdgroup per vector; the widest vector it holds (the query latent of
// 1280 values, which is RMS normalized alone).
#define LATENT_ROTATION_MAX_WIDTH 2048u

struct LatentRotationParams {
  uint32_t rows;
  // Vectors per row (heads), of `width` values each.
  uint32_t vectors;
  uint32_t width;
  // Pairs rotated at each vector's end.
  uint32_t rope_pairs;
  // Whether to rotate back, by -angle.
  uint32_t inverse;
  // LATENT_ROUND_*.
  uint32_t rounding;
  // Whether to RMS normalize each vector by the norm's weights first.
  uint32_t normalize;
  float norm_epsilon;
};

static_assert(sizeof(LatentRotationParams) == 32,
              "latent rotation parameters are 32 bytes on both sides");

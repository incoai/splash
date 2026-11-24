#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// Affine (MLX) Q4 and Q8 projections store each of their weight, scale and
// bias planes in tiles of this many output rows, [N / T][K / 64][T], every
// row of a tile holding its quant group's values. The routing kernels read an
// affine router as one tile of expert slots (SPLASH_MOE_EXPERT_SLOTS).
#define SPLASH_AFFINE_TILE_ROWS 256u

// A Q4 projection's matrix, for prefill and the decode tiles whose grid
// covers it (a K-split tile reads its split count from the grid). Separate
// from ops::LinearMatrix so host-only fields cannot change the ABI.
struct Q4Params {
  uint32_t output_size;
  uint32_t input_size;
};

static_assert(sizeof(Q4Params) == 8,
              "Q4 projection parameters are 8 bytes on both sides");

// The prefill residual kernels over a view of the leading inputs of wider
// weight rows (prefill_linear_q4_*_leading_inputs): each row of the weight,
// scale and bias planes holds plane_input_size inputs, of which the
// projection reads the first matrix.input_size.
struct Q4PrefillLeadingParams {
  Q4Params matrix;
  uint32_t plane_input_size;
};

static_assert(sizeof(Q4PrefillLeadingParams) == 12,
              "Q4 leading-input prefill parameters are 12 bytes on both sides");

// The persistent decode tiles' matrix and their grid's `groups`
// threadgroups, which stride over the column tiles. The stride is a
// parameter rather than [[threadgroups_per_grid]]: compiled at -O3, a loop
// striding by that attribute ran nondeterministically under Metal shader
// validation, which the kernel tests run with.
struct Q4PersistentParams {
  uint32_t output_size;
  uint32_t input_size;
  uint32_t groups;
};

static_assert(sizeof(Q4PersistentParams) == 12,
              "Q4 persistent decode parameters are 12 bytes on both sides");

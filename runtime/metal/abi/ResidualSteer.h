#pragma once

// Parameters of the residual steer kernel (kernels/shared/residual_steer.metal,
// ops/ResidualSteer.cpp): the bf16 residual buffer a Qwen3.5-family target
// writes, steered after a layer's FFN by a control vector
// (model/ControlVector.hpp).

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// A control vector's change to `rows` rows of `hidden` bf16 values of the
// residual after a steered layer, with row `layer` of a table of directions
// of `hidden` fp32 values: every value h becomes h + scale v (add), or
// h - scale (h . v) v (project, v of unit length). A threadgroup of
// RESIDUAL_STEER_THREADS threads steers one row.
#define RESIDUAL_STEER_THREADS 256u
#define RESIDUAL_STEER_ADD 0u
#define RESIDUAL_STEER_PROJECT 1u

struct ResidualSteerParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t layer;
  uint32_t mode;
  float scale;
};

static_assert(sizeof(ResidualSteerParams) == 20,
              "residual steer parameters are 20 bytes on both sides");

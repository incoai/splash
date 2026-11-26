#pragma once

// Parameters of the projective steer kernel (kernels/shared/steer_project.metal,
// ops/Steering.cpp): a layer's residual rows, fp32, are projected off one
// unit-norm direction, h <- h - alpha (h . d) d, a threadgroup a row.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// A threadgroup a row: the row's dot product with the direction is reduced
// over the threads, then every thread subtracts its columns' share.
#define STEER_THREADS 256u

struct SteerParams {
  uint32_t rows;
  uint32_t hidden;
  float alpha;
  // Zero: keeps the struct's size explicit on both sides.
  float reserved;
};

static_assert(sizeof(SteerParams) == 16, "Steer parameters are 16 bytes on both sides");

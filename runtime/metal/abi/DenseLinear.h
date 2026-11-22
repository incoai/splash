#pragma once

// Parameter layout of the dense bf16 projections
// (kernels/shared/dense_linear.metal) and their host dispatch
// (ops/DenseLinear.hpp).
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// output[r][n] = epilogue(sum_k input[r][k] weight[n][k] + bias[n]) for the
// rows r < rows: weights [output_size][input_size] and bias as stored, bf16.
// A dispatch reads whole row tiles of its input and writes its rows alone.
struct DenseLinearParams {
  uint32_t rows;
  uint32_t output_size;
  uint32_t input_size;
  uint32_t padding;
};

static_assert(sizeof(DenseLinearParams) == 16,
              "dense projection parameters are 16 bytes on both sides");

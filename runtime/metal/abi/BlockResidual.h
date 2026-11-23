#pragma once

// Parameters of the block residual kernel (kernels/shared/block_residual.metal,
// ops/BlockResidual.cpp), Inkling 2's depth attention: per row, `queries`
// learned queries attend over `states` state vectors of `hidden` fp32 values
// (completed blocks' states and the current partial one), each RMS normalized
// without a gain: s^_k = s_k / sqrt(mean(s_k^2) + eps), a_jk = softmax_k(q_j .
// s^_k * scale), out_j = sum_k a_jk s^_k. Rows are independent.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#define BLOCK_RESIDUAL_MAXIMUM_STATES 6u
#define BLOCK_RESIDUAL_MAXIMUM_QUERIES 4u
// Threads of the kernel's threadgroup, one per row, and the widest state its
// threads hold in registers.
#define BLOCK_RESIDUAL_THREADS 1024u
#define BLOCK_RESIDUAL_MAXIMUM_HIDDEN 4096u

// State k's rows are state_stride[k] elements apart, each query's output's
// output_stride; the queries are rows of `hidden` values.
struct BlockResidualParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t states;
  uint32_t queries;
  uint32_t state_stride[BLOCK_RESIDUAL_MAXIMUM_STATES];
  uint32_t output_stride;
  float scale;
};

static_assert(sizeof(BlockResidualParams) == 48, "block residual parameters are 48 bytes on both sides");

#pragma once

// Parameter layouts shared by ops/Mamba2.cpp and the Mamba-2 kernels
// (kernels/shared/mamba2.metal). The step kernels take GDNDecodeBatchParams:
// a Mamba-2 state cell is a GDN state cell of Mamba-2 geometry (a head's P
// state rows over N columns, in fp32), and the tape holds what a step's rows
// fold into the state.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/GDN.h"

// A step leaves its rows on the lane's tape: one layer's holds, each field
// for every row in turn, the rows' convolution inputs ([conv_dim] bf16),
// their B rows (each group's) then x rows ([bx_width] fp32), and the heads'
// decay exp(softplus(dt) A) and softplus(dt) ([heads] fp32 each).
inline constexpr uint64_t mamba2_tape_bx_offset(uint32_t conv_dim) {
  return uint64_t(SPLASH_TARGET_VERIFY_ROWS) * conv_dim * 2;
}
inline constexpr uint64_t mamba2_tape_decay_offset(uint32_t conv_dim, uint32_t bx_width) {
  return mamba2_tape_bx_offset(conv_dim) + uint64_t(SPLASH_TARGET_VERIFY_ROWS) * bx_width * 4;
}
inline constexpr uint64_t mamba2_tape_dt_offset(uint32_t conv_dim, uint32_t bx_width, uint32_t heads) {
  return mamba2_tape_decay_offset(conv_dim, bx_width) + uint64_t(SPLASH_TARGET_VERIFY_ROWS) * heads * 4;
}
// One layer's tape, a multiple of 16 bytes.
inline constexpr uint64_t mamba2_tape_layer_bytes(uint32_t conv_dim, uint32_t bx_width, uint32_t heads) {
  return (mamba2_tape_dt_offset(conv_dim, bx_width, heads) + uint64_t(SPLASH_TARGET_VERIFY_ROWS) * heads * 4 + 15) /
         16 * 16;
}

// The grouped RMS norm of the gated rows: each row's groups normalized with
// `epsilon` and scaled by the norm's weights.
struct Mamba2NormParams {
  float epsilon;
};

static_assert(sizeof(Mamba2NormParams) == 4, "Mamba-2 norm parameters are 4 bytes on both sides");

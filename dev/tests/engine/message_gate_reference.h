#pragma once

// The message gate's reference path (message_gate_test.metal, which
// convolution_mix_kernels_test.mm runs), a gate at a time: each row's and
// message's RMS norm without weights and their concatenation in bf16
// (message_gate_test_prepare), the down projection and the forget and gain
// projections through the float tile (ops::addGgufFloat), and the gated row
// (1 - tanh(forget g)) x + tanh(gain g) m in fp32 through the weighted RMS
// norm into bf16 (message_gate_test_norm). It rounds where the fused kernels
// do (metal/abi/MessageGate.h), its sums in other orders: the engine's path
// until the fused kernels took every call, which the tests check them with.
// One threadgroup of MESSAGE_GATE_REFERENCE_THREADS threads a row.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#define MESSAGE_GATE_REFERENCE_THREADS 256u

// A gate of `rows` rows of `hidden` values: the row x at column x_column of
// rows of x_stride values and its message at message_column of rows of
// message_stride values, both fp32; `rank` values a row between its
// projections.
struct MessageGateReferenceParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t x_stride;
  uint32_t x_column;
  uint32_t message_stride;
  uint32_t message_column;
  uint32_t rank;
};

static_assert(sizeof(MessageGateReferenceParams) == 28, "reference gate parameters are 28 bytes on both sides");

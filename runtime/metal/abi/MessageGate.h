#pragma once

// Parameters of the message gate kernels (kernels/shared/message_gate.metal,
// ops/MessageGate.cpp): a row x takes a message m by gates of a low-rank
// projection of both, g = down(norm(x), norm(m)), as
// (1 - tanh(forget g)) x + tanh(gain g) m, in fp32; a weighted RMS norm then
// writes the projections' bf16 input. One threadgroup of MESSAGE_GATE_THREADS
// threads a row.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#define MESSAGE_GATE_THREADS 256u

// A gate of `rows` rows of `hidden` values: the row x at column x_column of
// rows of x_stride values and its message at message_column of rows of
// message_stride values, both fp32.
struct MessageGateParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t x_stride;
  uint32_t x_column;
  uint32_t message_stride;
  uint32_t message_column;
};

static_assert(sizeof(MessageGateParams) == 24, "message gate parameters are 24 bytes on both sides");

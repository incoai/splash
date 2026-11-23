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

// The fused gates (ops::MessageGate, a decode step's rows): two kernels, each
// over groups of MESSAGE_GATE_GROUP_ROWS rows and up to
// MESSAGE_GATE_MAXIMUM_GATES gates of the same rows (a layer's q, k and v
// inputs: grid z), which read a gate's weights, F32 or BF16 rows as stored,
// once a group.
// message_gate_down normalizes and concatenates its group's rows itself and
// writes MESSAGE_GATE_DOWN_OUTPUTS of their rank values a threadgroup (32
// threadgroups a gate for a rank of 64; a simdgroup a value, 8 threadgroups,
// took 2.0 ms of a decode step's 96 gates against 1.25 ms on an M5 Pro, F32
// weights a gate a dispatch); message_gate_apply takes MESSAGE_GATE_THREADS
// of a row's values a threadgroup (a slice), a thread each: forget, gain and
// the gated row, whose squares the slices of a group add, the last to arrive
// in slice order, which then writes the group's normalized rows (slices of
// 128 and 64 values, two and four threads a value, ran no faster). A rank is
// at most MESSAGE_GATE_MAXIMUM_RANK, a multiple of four.
#define MESSAGE_GATE_GROUP_ROWS 8u
#define MESSAGE_GATE_DOWN_OUTPUTS 2u
#define MESSAGE_GATE_MAXIMUM_RANK 64u
#define MESSAGE_GATE_MAXIMUM_GATES 3u

// A gate of `rows` rows of `hidden` values: the row x at column x_column of
// rows of x_stride values and its message at message_column of rows of
// message_stride values, both fp32; `rank` values a row between its
// projections.
struct MessageGateParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t x_stride;
  uint32_t x_column;
  uint32_t message_stride;
  uint32_t message_column;
  uint32_t rank;
};

static_assert(sizeof(MessageGateParams) == 28, "message gate parameters are 28 bytes on both sides");

// The fused gates of a dispatch: `gates` gates of `rows` rows of `hidden`
// values, gate g's row x at column x_columns[g] of rows of x_stride values and
// its message at message_columns[g] of rows of message_stride values, both
// fp32, in whole float4s; `rank` values a row between each gate's projections.
struct MessageGateFusedParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t x_stride;
  uint32_t message_stride;
  uint32_t rank;
  uint32_t gates;
  uint32_t x_columns[MESSAGE_GATE_MAXIMUM_GATES];
  uint32_t message_columns[MESSAGE_GATE_MAXIMUM_GATES];
};

static_assert(sizeof(MessageGateFusedParams) == 48, "fused message gate parameters are 48 bytes on both sides");

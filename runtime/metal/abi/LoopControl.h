#pragma once

// Parameters of the loop control kernels (kernels/shared/loop_control.metal,
// ops/LoopControl.cpp), the gate through which an Inkling 2 loop pass's
// message reaches a stream of the pass's layer input: per row, with the
// stream's input u and message m of `hidden` fp32 values,
//   d   = W_down . concat(rms(u), rms(m))        (`rank` values)
//   out = (1 - tanh(W_forget . d)) u + tanh(W_gain . d) m
// rms without a gain. loop_control_down_<weights> writes d, fp32 [rows][rank];
// loop_control_gate_<weights> the output. A dispatch runs up to
// LOOP_CONTROL_MAXIMUM_STREAMS streams of one row count (grid z), each with
// its own weights and buffers; the inputs', messages' and outputs' rows are
// input_stride, message_stride and output_stride elements apart.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#define LOOP_CONTROL_MAXIMUM_STREAMS 4u
// Threads of both kernels' threadgroups: eight simdgroups.
#define LOOP_CONTROL_THREADS 256u
// Rows of a threadgroup's tile, and the columns of the gate's.
#define LOOP_CONTROL_TILE_ROWS 32u
#define LOOP_CONTROL_GATE_COLUMNS 64u
// Columns of the down projection's threadgroup.
#define LOOP_CONTROL_DOWN_COLUMNS 8u

struct LoopControlParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t rank;
  uint32_t streams;
  uint32_t input_stride;
  uint32_t message_stride;
  uint32_t output_stride;
};

static_assert(sizeof(LoopControlParams) == 28, "loop control parameters are 28 bytes on both sides");

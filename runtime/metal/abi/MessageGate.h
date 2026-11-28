#pragma once

// Parameters of the message gate kernels (kernels/shared/message_gate.metal,
// ops/MessageGate.cpp): a row x takes a message m by gates of a low-rank
// projection of both, g = down(norm(x), norm(m)), as
// (1 - tanh(forget g)) x + tanh(gain g) m, in fp32; a weighted RMS norm then
// writes the projections' bf16 input. The kernels below read the gates'
// weights themselves; the reference path they are checked against, which
// rounds at the same points, runs in the tests
// (dev/tests/engine/message_gate_reference.h).
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#define MESSAGE_GATE_THREADS 256u

// The fused gates (ops::MessageGate, a decode step's rows): two kernels over
// up to MESSAGE_GATE_MAXIMUM_GATES gates of the same rows (a layer's q, k and
// v inputs: grid z), over groups of rows, which read a gate's weights, F32 or
// BF16 rows as stored, once a group: a call's rows in one group, of the
// least power of two that holds them, up to MESSAGE_GATE_GROUP_ROWS, more in
// groups of that many (ops::MessageGate::groupRows). Each kernel computes
// every row of its group, so a group past the call's rows computes rows it
// stores nothing of (a step of 1 row took an 8-row group's time). A group
// holds at most 4 rows: a call of 5 to 16 rows as groups of 4, two to four
// times the threadgroups, each thread keeping half or a quarter of the sums,
// took Inkling-2-Mini's whole decode step 1.3-2.4% less on a 32-core M5 Max,
// 0.1-1.2% on a 12-core M6 and 0.4-1.8% on a 16-core M5 Pro than as one group
// of 8 rows, or of 16 as two halves of 8 in one threadgroup, though it reads
// each gate's weights once a group more (decode-profile A B B A at 5 to 8, 12
// and 16 rows a lane, the same bits; the M5 Max's two to four lanes of 8 rows
// 0.8-1.9% less). Groups of 2 rows took a step of 3 or 4 rows 1% less on the
// M5 Max and no less on the others. Each row of x
// and of the messages is normalized once, into the bf16 rows that each of the
// down projection's threadgroups of the row's group reads, by the mix that
// writes it, in the mix's dispatch (kernels/common/message_gate_normalize.h,
// ops::BlockResidualMixInput::normalized), where each threadgroup had
// normalized its group's rows itself: an eight-row decode step's gates took
// 2.13 ms where they took 2.64 (Inkling-2-Mini on a 16-core M5 Pro, separate
// dispatches, a kernel of a threadgroup a row normalizing them), the step
// 0.47 ms less; in the mixes, 48 dispatches fewer a step. message_gate_down
// writes MESSAGE_GATE_DOWN_OUTPUTS
// of the group's rank values a threadgroup (32 threadgroups a gate for a rank
// of 64; a simdgroup a value, 8 threadgroups, took 2.0 ms of a decode step's
// 96 gates against 1.25 ms on an M5 Pro, F32 weights a gate a dispatch);
// message_gate_apply takes MESSAGE_GATE_THREADS of a row's values a
// threadgroup (a slice), a thread each: forget, gain and the gated row, whose
// squares the slices of a group add, the last to arrive in slice order, which
// then writes the group's normalized rows (slices of 128 and 64 values, two
// and four threads a value, ran no faster). A rank is at most
// MESSAGE_GATE_MAXIMUM_RANK, a multiple of four.
#define MESSAGE_GATE_GROUP_ROWS 4u
#define MESSAGE_GATE_DOWN_OUTPUTS 2u
#define MESSAGE_GATE_MAXIMUM_RANK 64u
#define MESSAGE_GATE_MAXIMUM_GATES 3u

// The fused gates of more rows (ops::MessageGate, a prompt chunk's): one
// kernel, message_gate_rows, of a gate's MESSAGE_GATE_TILE_ROWS rows a
// threadgroup of MESSAGE_GATE_TILE_THREADS threads (grid z the gates, as
// above), which reads the tile's rows and the gate's weights itself and keeps
// its rank values in threadgroup memory: the tile's inverse RMS, its rank
// values, its gated rows, which it stores to the gate's scratch and reads
// back, and their norm, the projections on simdgroup float MMA, each
// simdgroup 8 columns of a 64-column block for the tile's rows, so that the
// tile reads each weight once, the down projection's normalized inputs staged
// in threadgroup memory. Its rows of whole 64-value steps, its rank a
// multiple of 8. A 2,583-token Inkling-2-Mini prefill's gates took 102 ms on
// it on a 16-core M5 Pro, against 141 ms as the float projections' four
// dispatches a gate and 128 ms with each simdgroup reading its own operands
// from device memory (2 x 4 tiles of 16 x 16 a threadgroup).
#define MESSAGE_GATE_TILE_ROWS 32u
#define MESSAGE_GATE_TILE_THREADS 256u

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

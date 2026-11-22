#pragma once

// Inkling 2's loop control (kernels/shared/loop_control.metal,
// metal/abi/LoopControl.h): the gate through which a loop pass's message
// reaches each stream (Q, K, V, FFN) of its core layer's input,
//   d   = W_down . concat(rms(u), rms(m))
//   out = (1 - tanh(W_forget . d)) u + tanh(W_gain . d) m,
// in fp32, its low rank (64) projections in two dispatches of its own
// (down, then forget/gain with the mix) rather than through ops::Linear: the
// rank is too small for the projection tiles, and the kernels read the fp32
// rows as they are, where ops::Linear's float projection reads bf16 input.

#include "metal/CommandGraph.hpp"
#include "metal/abi/LoopControl.h"
#include "ops/FloatRows.hpp"

#include <cstdint>
#include <span>

namespace splash::ops {

// A stream's gate weights as a GGUF stores them, rows of their inputs: down
// [rank][2 hidden] (the input's half first), forget and gain [hidden][rank],
// all F32 (ggml-org's files) or all BF16.
struct LoopControlWeights final {
  metal::MetalBuffer down;
  metal::MetalBuffer forget;
  metal::MetalBuffer gain;
  FloatElement element = FloatElement::Float32;
};

// A stream of a dispatch: its weights, its fp32 input and message rows, its
// fp32 output rows (which may be the input's or the message's: an element is
// read and written by one thread, Inkling's sliding layers writing each of
// their Q, K and V streams, which share one input, over its message), and
// `low`, the fp32 [rows][rank] scratch of its rows' d (lowBytes).
struct LoopControlStream final {
  const LoopControlWeights *weights = nullptr;
  metal::MetalBuffer input;
  metal::MetalBuffer message;
  metal::MetalBuffer output;
  metal::MetalBuffer low;
};

// The streams' width and rank, and the elements from one row to the next of
// every stream's inputs, messages and outputs.
struct LoopControlShape final {
  uint32_t hidden = 0;
  uint32_t rank = 0;
  uint32_t inputStride = 0;
  uint32_t messageStride = 0;
  uint32_t outputStride = 0;
};

struct LoopControl final {
  // The gate of each of 1 to LOOP_CONTROL_MAXIMUM_STREAMS streams (a layer's
  // Q, K and V together, or its FFN) over `rows` rows, in two dispatches
  // whatever the streams, whose weights share one element type. hidden is a
  // multiple of LOOP_CONTROL_GATE_COLUMNS, rank of 64, and every stride of 4
  // (rows from 16-byte aligned views).
  static void add(metal::CommandGraph &graph, std::span<const LoopControlStream> streams, LoopControlShape shape,
                  uint32_t rows);
  // The bytes of a stream's `low` scratch of `rows` rows.
  [[nodiscard]] static uint64_t lowBytes(uint32_t rank, uint32_t rows) noexcept;
};

} // namespace splash::ops

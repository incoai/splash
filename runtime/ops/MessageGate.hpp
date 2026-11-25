#pragma once

// The message gate (kernels/shared/message_gate.metal): a row x takes a
// message m, a row of another pass, by gates of a low-rank projection of
// both: g = down(norm(x), norm(m)), the two RMS-normalized without weights
// side by side and rounded to bf16, its rank values rounded to bf16 too (a
// bf16 projection in the source); the gated row (1 - tanh(forget g)) x +
// tanh(gain g) m in fp32; then the weighted RMS norm of the projections that
// read it, into their bf16 input. Its three projections are float segments.
// A call of up to kMaximumGates gates runs them fused, in kernels that read
// the weights themselves (metal/abi/MessageGate.h): a decode step's rows at
// most (kFusedRows) in three kernels, each row normalized once for its down
// projection, more rows, a prompt chunk's, in one tiled kernel, a tile of
// rows a threadgroup. They take weights of F32 or BF16
// rows, all of a call's of one format, a rank of 8 to MESSAGE_GATE_MAXIMUM_RANK
// in runs of 8 and a hidden of whole 64-value steps, x and the messages in
// whole float4s, and add refuses any other call (requireWeights says why).
// Inkling-2-Mini's gates are such rows whatever their checkpoint holds: its
// loaders keep F32 and BF16 tensors as stored and decode a quantized or F16
// one, as some GGUFs keep the gates, to F32 (FloatTensor sections,
// model/GgufImage.hpp), and its target checks a layer's at load. The gates'
// reference path, a gate's normalization, its projections through the float
// tile and its norm, which rounds at the same points as these kernels and
// adds its sums in other orders, runs in the tests
// (dev/tests/engine/message_gate_reference.h).

#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/MessageGate.h"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"
#include "ops/Weights.hpp"

#include <cstdint>
#include <span>

namespace splash::ops {

// A gate's projections: down [rank][2 hidden], forget and gain
// [hidden][rank], float segments.
struct MessageGateWeights final {
  QuantizedSegment down;
  QuantizedSegment forget;
  QuantizedSegment gain;
};

// One gate of a call: its projections, the norm after it (F32), the columns
// of x and of the message it reads, and its output, bf16 [rows][hidden].
struct MessageGateInput final {
  MessageGateWeights weights;
  NormWeights norm;
  uint32_t xColumn = 0;
  uint32_t messageColumn = 0;
  metal::MetalBuffer output;
};

// The rows a call's gates read: x and the messages, fp32 rows of a stride (a
// mix's output of several queries, the messages of four); and its scratch. A
// decode step's rows: the normalized concatenations, bf16
// [gates][rows][2 hidden], and then over the same bytes the gated rows, fp32
// [gates][rows][hidden] (forget), the rank values, bf16 [gates][rows][rank]
// (low), the groups' partial sums (gain) and the counters, one a group of
// MESSAGE_GATE_GROUP_ROWS rows a gate, zero before and after (kCountersBytes
// for the most). More rows: the gated rows, fp32 [rows][hidden], of gate 0 in
// forget, of gate 1 in gain and of gate 2 in concat.
struct MessageGateBuffers final {
  metal::MetalBuffer x;
  uint32_t xStride = 0;
  metal::MetalBuffer message;
  uint32_t messageStride = 0;
  metal::MetalBuffer concat;
  metal::MetalBuffer low;
  metal::MetalBuffer forget;
  metal::MetalBuffer gain;
  metal::MetalBuffer counters;
};

struct MessageGate final {
  // The most rows the fused gates take: a decode step's at the widest batch.
  static constexpr uint32_t kFusedRows = SPLASH_MAXIMUM_BATCH_WIDTH * SPLASH_TARGET_VERIFY_ROWS;
  // The most gates a call takes: a layer's q, k and v inputs'.
  static constexpr uint32_t kMaximumGates = MESSAGE_GATE_MAXIMUM_GATES;
  // The counters of the fused gates of the most gates and rows, one a group
  // a gate.
  static constexpr uint64_t kCountersBytes = uint64_t{kMaximumGates} *
                                             ((kFusedRows + MESSAGE_GATE_GROUP_ROWS - 1) / MESSAGE_GATE_GROUP_ROWS) *
                                             sizeof(uint32_t);
  // The partial sums of the fused gates of `gates` gates of `rows` rows of
  // `hidden` values (message_gate_apply).
  static constexpr uint64_t partialSumsBytes(uint32_t gates, uint32_t hidden, uint32_t rows) noexcept {
    const uint64_t groups = (uint64_t{rows} + MESSAGE_GATE_GROUP_ROWS - 1) / MESSAGE_GATE_GROUP_ROWS;
    const uint64_t slices = (uint64_t{hidden} + MESSAGE_GATE_THREADS - 1) / MESSAGE_GATE_THREADS;
    return uint64_t{gates} * groups * slices * MESSAGE_GATE_GROUP_ROWS * sizeof(float);
  }
  // The gated rows of `rows` rows of `hidden` values, each gate's through its
  // norm into its output. The projections reading the outputs prepare their
  // own tables, so the scratch describes nothing. Throws for a call the fused
  // kernels do not take.
  static PreparedInput add(metal::CommandGraph &graph, const MessageGateBuffers &buffers,
                           std::span<const MessageGateInput> gates, uint32_t hidden, uint32_t rows);
  // Throws, saying why, unless the fused kernels take a call of these gates'
  // weights of `hidden` values a row (a target checks its gates at load).
  static void requireWeights(std::span<const MessageGateWeights> gates, uint32_t hidden);
};

} // namespace splash::ops

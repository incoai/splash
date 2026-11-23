#pragma once

// The message gate (kernels/shared/message_gate.metal): a row x takes a
// message m, a row of another pass, by gates of a low-rank projection of
// both: g = down(norm(x), norm(m)), the two RMS-normalized without weights
// side by side and rounded to bf16, its rank values rounded to bf16 too (a
// bf16 projection in the source); the gated row (1 - tanh(forget g)) x +
// tanh(gain g) m in fp32; then the weighted RMS norm of the projections that
// read it, into their bf16 input. Its three projections are float segments.
// A call of at most kMaximumGates gates whose weights are F32 or BF16 rows,
// all of one format, of a rank of at most MESSAGE_GATE_MAXIMUM_RANK and a
// multiple of 4, x and the messages in whole float4s, runs its gates fused,
// in kernels that read the weights themselves (metal/abi/MessageGate.h): a
// decode step's rows at most (kFusedRows) in two kernels, more rows, a prompt
// chunk's, in one tiled kernel, a tile of rows a threadgroup, where hidden is
// a multiple of 64 and the rank of 8. Inkling-2-Mini's gates are such rows
// whatever their checkpoint holds: its loaders keep F32 and BF16 tensors as
// stored and decode a quantized one, as some GGUFs keep the gates, to F32
// (FloatTensor sections, model/GgufImage.hpp). Every other call runs each
// gate's normalization, its projections through ops::addGgufFloat and its
// norm, the float tile reading the weights once for many rows: a quantized
// format's rows, F32 and BF16 gates in one call, another rank or stride, and
// more rows than a decode step's of another hidden or rank. Every path rounds
// at the same points; their sums add in other orders.

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
// mix's output of several queries, the messages of four); and its scratch.
// Fused, a decode step's rows: the rank values, bf16 [gates][rows][rank]
// (low), the gated rows, fp32 [gates][rows][hidden] (forget), the groups'
// partial sums (gain) and the counters, one a group of
// MESSAGE_GATE_GROUP_ROWS rows a gate, zero before and after (kCountersBytes
// for the most). Fused, more rows: the gated rows, fp32 [rows][hidden], of
// gate 0 in forget, of gate 1 in gain and of gate 2 in concat. Otherwise, a
// gate at a time: the down projection's bf16 input [rows][2 hidden] (concat)
// and output [rows][rank] (low), and the forget and gain projections' fp32
// rows [rows][hidden].
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

// The kernels of a call: the fused ones where its rows and weights take them
// (Fused), or the general path for any weights (FloatProjections), which the
// precision checks compare them with.
enum class MessageGateKernels : uint8_t { Fused, FloatProjections };

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
  // The gated rows of `rows` rows of `hidden` values, each gate's through its
  // norm into its output, the float projections on `tile`. The projections
  // reading the outputs prepare their own tables, so the scratch describes
  // nothing.
  static PreparedInput add(metal::CommandGraph &graph, const MessageGateBuffers &buffers,
                           std::span<const MessageGateInput> gates, uint32_t hidden, uint32_t rows, FloatTile tile,
                           MessageGateKernels kernels = MessageGateKernels::Fused);
};

} // namespace splash::ops

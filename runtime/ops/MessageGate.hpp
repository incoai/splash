#pragma once

// The message gate (kernels/shared/message_gate.metal): a row x takes a
// message m, a row of another pass, by gates of a low-rank projection of
// both: g = down(norm(x), norm(m)), the two RMS-normalized without weights
// side by side and rounded to bf16, its rank values rounded to bf16 too (a
// bf16 projection in the source); the gated row (1 - tanh(forget g)) x +
// tanh(gain g) m in fp32; then the weighted RMS norm of the projections that
// read it, into their bf16 input. Its three projections are F32 float
// segments, which run through ops::addGgufFloat.

#include "metal/CommandGraph.hpp"
#include "metal/abi/MessageGate.h"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"
#include "ops/Weights.hpp"

#include <cstdint>

namespace splash::ops {

// A gate's projections: down [rank][2 hidden], forget and gain
// [hidden][rank], float segments.
struct MessageGateWeights final {
  QuantizedSegment down;
  QuantizedSegment forget;
  QuantizedSegment gain;
};

// A gate's rows: x and the message, fp32 at a column of rows of a stride
// (a mix's output of several queries, the messages of four); the gate's
// scratch, its down projection's bf16 input [rows][2 hidden] and output
// [rows][rank], the forget and gain projections' fp32 rows [rows][hidden];
// and the normalized output, bf16 [rows][hidden].
struct MessageGateBuffers final {
  metal::MetalBuffer x;
  uint32_t xStride = 0;
  uint32_t xColumn = 0;
  metal::MetalBuffer message;
  uint32_t messageStride = 0;
  uint32_t messageColumn = 0;
  metal::MetalBuffer concat;
  metal::MetalBuffer low;
  metal::MetalBuffer forget;
  metal::MetalBuffer gain;
  metal::MetalBuffer output;
};

struct MessageGate final {
  // The gated rows of `rows` rows of `hidden` values through `norm` (F32)
  // into buffers.output, the float projections on `tile`. The projections
  // reading the output prepare their own tables, so the scratch describes
  // nothing.
  static PreparedInput add(metal::CommandGraph &graph, const MessageGateBuffers &buffers,
                           const MessageGateWeights &weights, const NormWeights &norm, uint32_t hidden, uint32_t rows,
                           FloatTile tile);
};

} // namespace splash::ops

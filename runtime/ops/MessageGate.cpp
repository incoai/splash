#include "ops/MessageGate.hpp"

#include "metal/abi/MessageGate.h"
#include "ops/BufferExtent.hpp"

#include <array>
#include <stdexcept>

namespace splash::ops {

PreparedInput MessageGate::add(metal::CommandGraph &graph, const MessageGateBuffers &buffers,
                               const MessageGateWeights &weights, const NormWeights &norm, uint32_t hidden,
                               uint32_t rows, FloatTile tile) {
  const uint32_t rank = weights.down.outputSize;
  if (!hidden || !rows || buffers.xColumn + uint64_t{hidden} > buffers.xStride ||
      buffers.messageColumn + uint64_t{hidden} > buffers.messageStride || weights.down.inputSize != 2 * hidden ||
      weights.forget.inputSize != rank || weights.gain.inputSize != rank || weights.forget.outputSize != hidden ||
      weights.gain.outputSize != hidden)
    throw std::invalid_argument("invalid message gate");
  if (!norm.float32) throw std::invalid_argument("a message gate norm is not F32");
  requireBytes(buffers.x, rowBytes(rows, buffers.xStride, uint64_t{buffers.xColumn} + hidden, sizeof(float)),
               "message gate row");
  requireBytes(buffers.message,
               rowBytes(rows, buffers.messageStride, uint64_t{buffers.messageColumn} + hidden, sizeof(float)),
               "message gate message");
  requireBytes(buffers.concat, uint64_t{rows} * 2 * hidden * sizeof(uint16_t), "message gate concatenation");
  requireBytes(norm.buffer, norm.bytes(hidden), "message gate norm");
  requireBytes(buffers.output, uint64_t{rows} * hidden * sizeof(uint16_t), "message gate output");
  const MessageGateParams p{rows, hidden, buffers.xStride, buffers.xColumn, buffers.messageStride,
                            buffers.messageColumn};
  const metal::DispatchSize groups{rows, 1, 1}, threads{MESSAGE_GATE_THREADS, 1, 1};
  graph.add("message_gate_prepare", {buffers.x, buffers.message, buffers.concat}, p, groups, threads);
  // The rank values in bf16, the forget and gain projections' input, which
  // write fp32 (addGgufFloat checks the float projections' buffers).
  addGgufFloat(graph, buffers.concat, weights.down, buffers.low, rows, rank, 0, FloatOutput::BFloat16, tile);
  const std::array projections{FloatProjection{&weights.forget, buffers.forget, hidden, 0, tile},
                               FloatProjection{&weights.gain, buffers.gain, hidden, 0, tile}};
  addGgufFloat(graph, buffers.low, projections, rows, FloatOutput::Float32);
  graph.add("message_gate_norm",
            {buffers.x, buffers.message, buffers.forget, buffers.gain, norm.buffer, buffers.output}, p, groups,
            threads);
  return {};
}

} // namespace splash::ops

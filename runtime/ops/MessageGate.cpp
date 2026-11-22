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
  requireBytes(norm.buffer, norm.bytes(hidden), "message gate norm");
  requireBytes(buffers.output, uint64_t{rows} * hidden * sizeof(uint16_t), "message gate output");
  const MessageGateParams p{rows, hidden, buffers.xStride, buffers.xColumn, buffers.messageStride,
                            buffers.messageColumn, rank};
  const metal::DispatchSize threads{MESSAGE_GATE_THREADS, 1, 1};
  // The fused gate runs a decode step's rows of F32 weights of a rank its
  // threadgroups hold, its rows in whole float4s.
  const auto float32 = [](const QuantizedSegment &s) { return s.formatId == QuantizedSegment::kFloat32; };
  if (rows <= kFusedRows && float32(weights.down) && float32(weights.forget) && float32(weights.gain) &&
      rank <= MESSAGE_GATE_MAXIMUM_RANK && rank % 4 == 0 && hidden % 4 == 0 && buffers.xStride % 4 == 0 &&
      buffers.xColumn % 4 == 0 && buffers.messageStride % 4 == 0 && buffers.messageColumn % 4 == 0) {
    const uint32_t groups = (rows + MESSAGE_GATE_GROUP_ROWS - 1) / MESSAGE_GATE_GROUP_ROWS;
    const uint32_t slices = (hidden + MESSAGE_GATE_THREADS - 1) / MESSAGE_GATE_THREADS;
    requireBytes(buffers.low, uint64_t{rows} * rank * sizeof(uint16_t), "message gate rank values");
    requireBytes(buffers.forget, uint64_t{rows} * hidden * sizeof(float), "message gate gated rows");
    requireBytes(buffers.gain, uint64_t{groups} * slices * MESSAGE_GATE_GROUP_ROWS * sizeof(float),
                 "message gate partial sums");
    requireBytes(buffers.counters, uint64_t{groups} * sizeof(uint32_t), "message gate counters");
    requireBytes(weights.down.plane0, uint64_t{rank} * 2 * hidden * sizeof(float), "message gate down weights");
    requireBytes(weights.forget.plane0, uint64_t{hidden} * rank * sizeof(float), "message gate forget weights");
    requireBytes(weights.gain.plane0, uint64_t{hidden} * rank * sizeof(float), "message gate gain weights");
    graph.add("message_gate_down", {buffers.x, buffers.message, weights.down.plane0, buffers.low}, p,
              {(rank + MESSAGE_GATE_DOWN_OUTPUTS - 1) / MESSAGE_GATE_DOWN_OUTPUTS, groups, 1}, threads);
    graph.add("message_gate_apply",
              {buffers.x, buffers.message, weights.forget.plane0, weights.gain.plane0, buffers.low, norm.buffer,
               buffers.forget, buffers.gain, buffers.counters, buffers.output},
              p, {slices, groups, 1}, threads);
    return {};
  }
  requireBytes(buffers.concat, uint64_t{rows} * 2 * hidden * sizeof(uint16_t), "message gate concatenation");
  const metal::DispatchSize groups{rows, 1, 1};
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

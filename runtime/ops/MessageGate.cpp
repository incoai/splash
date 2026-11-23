#include "ops/MessageGate.hpp"

#include "metal/abi/MessageGate.h"
#include "metal/abi/QuantFormat.h"
#include "ops/BufferExtent.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

namespace splash::ops {
namespace {

// Rows of `values` values of a stride, from a column, in whole float4s.
bool quads(uint32_t stride, uint32_t column) { return stride % 4 == 0 && column % 4 == 0; }

// A gate's projections the fused kernels read: F32 or BF16 rows, all of
// `format`, of a rank their threadgroups hold.
bool fusedWeights(const MessageGateWeights &weights, uint32_t format, uint32_t rank) {
  const auto rows = [&](const QuantizedSegment &s) { return s.isFloat() && s.formatId == format; };
  return (format == QuantizedSegment::kFloat32 || format == GGUF_FMT_BF16) && rows(weights.down) &&
         rows(weights.forget) && rows(weights.gain) && rank <= MESSAGE_GATE_MAXIMUM_RANK && rank % 4 == 0;
}

} // namespace

PreparedInput MessageGate::add(metal::CommandGraph &graph, const MessageGateBuffers &buffers,
                               std::span<const MessageGateInput> gates, uint32_t hidden, uint32_t rows,
                               FloatTile tile) {
  if (gates.empty() || !hidden || !rows) throw std::invalid_argument("invalid message gates");
  const uint32_t rank = gates.front().weights.down.outputSize;
  for (const MessageGateInput &gate : gates) {
    const MessageGateWeights &weights = gate.weights;
    if (gate.xColumn + uint64_t{hidden} > buffers.xStride ||
        gate.messageColumn + uint64_t{hidden} > buffers.messageStride || weights.down.outputSize != rank ||
        weights.down.inputSize != 2 * hidden || weights.forget.inputSize != rank ||
        weights.gain.inputSize != rank || weights.forget.outputSize != hidden || weights.gain.outputSize != hidden)
      throw std::invalid_argument("invalid message gate");
    if (!gate.norm.float32) throw std::invalid_argument("a message gate norm is not F32");
    requireBytes(buffers.x, rowBytes(rows, buffers.xStride, uint64_t{gate.xColumn} + hidden, sizeof(float)),
                 "message gate row");
    requireBytes(buffers.message,
                 rowBytes(rows, buffers.messageStride, uint64_t{gate.messageColumn} + hidden, sizeof(float)),
                 "message gate message");
    requireBytes(gate.norm.buffer, gate.norm.bytes(hidden), "message gate norm");
    requireBytes(gate.output, uint64_t{rows} * hidden * sizeof(uint16_t), "message gate output");
  }
  const metal::DispatchSize threads{MESSAGE_GATE_THREADS, 1, 1};
  // The fused gates run a decode step's rows of F32 or BF16 weights, all of
  // one, their rows in whole float4s.
  const uint32_t format = gates.front().weights.down.formatId;
  const bool fused =
      rows <= kFusedRows && gates.size() <= kMaximumGates && hidden % 4 == 0 && buffers.xStride % 4 == 0 &&
      buffers.messageStride % 4 == 0 && std::ranges::all_of(gates, [&](const MessageGateInput &gate) {
        return fusedWeights(gate.weights, format, rank) && quads(buffers.xStride, gate.xColumn) &&
               quads(buffers.messageStride, gate.messageColumn);
      });
  if (fused) {
    const auto count = static_cast<uint32_t>(gates.size());
    const uint64_t weightBytes = format == GGUF_FMT_BF16 ? sizeof(uint16_t) : sizeof(float);
    const std::string suffix = format == GGUF_FMT_BF16 ? "_bf16" : "_f32";
    const uint32_t groups = (rows + MESSAGE_GATE_GROUP_ROWS - 1) / MESSAGE_GATE_GROUP_ROWS;
    const uint32_t slices = (hidden + MESSAGE_GATE_THREADS - 1) / MESSAGE_GATE_THREADS;
    requireBytes(buffers.low, uint64_t{count} * rows * rank * sizeof(uint16_t), "message gate rank values");
    requireBytes(buffers.forget, uint64_t{count} * rows * hidden * sizeof(float), "message gate gated rows");
    requireBytes(buffers.gain, uint64_t{count} * groups * slices * MESSAGE_GATE_GROUP_ROWS * sizeof(float),
                 "message gate partial sums");
    requireBytes(buffers.counters, uint64_t{count} * groups * sizeof(uint32_t), "message gate counters");
    MessageGateFusedParams p{rows, hidden, buffers.xStride, buffers.messageStride, rank, count, {}, {}};
    // A slot past the call's gates binds its first gate's buffers, which the
    // kernels never read.
    const auto slot = [&](uint32_t index) -> const MessageGateInput & { return gates[index < count ? index : 0]; };
    for (uint32_t index = 0; index < count; ++index) {
      const MessageGateWeights &weights = gates[index].weights;
      requireBytes(weights.down.plane0, uint64_t{rank} * 2 * hidden * weightBytes, "message gate down weights");
      requireBytes(weights.forget.plane0, uint64_t{hidden} * rank * weightBytes, "message gate forget weights");
      requireBytes(weights.gain.plane0, uint64_t{hidden} * rank * weightBytes, "message gate gain weights");
      p.x_columns[index] = gates[index].xColumn;
      p.message_columns[index] = gates[index].messageColumn;
    }
    graph.add("message_gate_down" + suffix,
              {buffers.x, buffers.message, slot(0).weights.down.plane0, slot(1).weights.down.plane0,
               slot(2).weights.down.plane0, buffers.low},
              p, {(rank + MESSAGE_GATE_DOWN_OUTPUTS - 1) / MESSAGE_GATE_DOWN_OUTPUTS, groups, count}, threads);
    graph.add("message_gate_apply" + suffix,
              {buffers.x, buffers.message, slot(0).weights.forget.plane0, slot(1).weights.forget.plane0,
               slot(2).weights.forget.plane0, slot(0).weights.gain.plane0, slot(1).weights.gain.plane0,
               slot(2).weights.gain.plane0, buffers.low, slot(0).norm.buffer, slot(1).norm.buffer,
               slot(2).norm.buffer, buffers.forget, buffers.gain, buffers.counters, slot(0).output, slot(1).output,
               slot(2).output},
              p, {slices, groups, count}, threads);
    return {};
  }
  requireBytes(buffers.concat, uint64_t{rows} * 2 * hidden * sizeof(uint16_t), "message gate concatenation");
  const metal::DispatchSize groups{rows, 1, 1};
  for (const MessageGateInput &gate : gates) {
    const MessageGateParams p{rows, hidden, buffers.xStride, gate.xColumn, buffers.messageStride, gate.messageColumn,
                              rank};
    graph.add("message_gate_prepare", {buffers.x, buffers.message, buffers.concat}, p, groups, threads);
    // The rank values in bf16, the forget and gain projections' input, which
    // write fp32 (addGgufFloat checks the float projections' buffers).
    addGgufFloat(graph, buffers.concat, gate.weights.down, buffers.low, rows, rank, 0, FloatOutput::BFloat16, tile);
    const std::array projections{FloatProjection{&gate.weights.forget, buffers.forget, hidden, 0, tile},
                                 FloatProjection{&gate.weights.gain, buffers.gain, hidden, 0, tile}};
    addGgufFloat(graph, buffers.low, projections, rows, FloatOutput::Float32);
    graph.add("message_gate_norm",
              {buffers.x, buffers.message, buffers.forget, buffers.gain, gate.norm.buffer, gate.output}, p, groups,
              threads);
  }
  return {};
}

} // namespace splash::ops

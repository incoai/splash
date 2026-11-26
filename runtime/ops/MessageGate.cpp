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

// Why the fused kernels do not take gates of these weights and `hidden`
// values a row, or nullptr where they take them: F32 or BF16 rows, a call's
// down projections of one format and its forget and gain projections of one
// format, of a rank of at most MESSAGE_GATE_MAXIMUM_RANK and a multiple of 8,
// of a hidden of whole 64-value steps (a prompt chunk's tiled kernel's; a
// decode step's kernels take any multiple of 4 of either).
const char *refusal(std::span<const MessageGateWeights> gates, uint32_t hidden) {
  if (gates.empty() || gates.size() > MessageGate::kMaximumGates) return "a call takes 1 to 3 gates";
  const uint32_t downFormat = gates.front().down.formatId, upFormat = gates.front().forget.formatId,
                 rank = gates.front().down.outputSize;
  const auto validFormat = [](uint32_t format) {
    return format == QuantizedSegment::kFloat32 || format == GGUF_FMT_BF16;
  };
  if (!validFormat(downFormat) || !validFormat(upFormat))
    return "the fused kernels read F32 or BF16 rows";
  const auto rows = [](const QuantizedSegment &s, uint32_t format) { return s.isFloat() && s.formatId == format; };
  for (const MessageGateWeights &weights : gates) {
    if (!rows(weights.down, downFormat) || !rows(weights.forget, upFormat) || !rows(weights.gain, upFormat))
      return "a call's down projections and its forget and gain projections are F32 or BF16 rows all of one format";
    if (weights.down.outputSize != rank || weights.down.inputSize != 2 * uint64_t{hidden} ||
        weights.forget.outputSize != hidden || weights.forget.inputSize != rank || weights.gain.outputSize != hidden ||
        weights.gain.inputSize != rank)
      return "a gate's projections are [rank][2 hidden], [hidden][rank] and [hidden][rank]";
  }
  if (!rank || rank > MESSAGE_GATE_MAXIMUM_RANK || rank % 8 != 0)
    return "the fused kernels take a rank of 8 to 64 in runs of 8";
  if (!hidden || hidden % 64 != 0) return "the fused kernels take rows of whole 64-value steps";
  return nullptr;
}

} // namespace

void MessageGate::requireWeights(std::span<const MessageGateWeights> gates, uint32_t hidden) {
  if (const char *reason = refusal(gates, hidden))
    throw std::invalid_argument(std::string("message gates the fused kernels do not take: ") + reason);
}

PreparedInput MessageGate::add(metal::CommandGraph &graph, const MessageGateBuffers &buffers,
                               std::span<const MessageGateInput> gates, uint32_t hidden, uint32_t rows) {
  if (gates.empty() || !hidden || !rows || !gates.front().weights.down.outputSize)
    throw std::invalid_argument("invalid message gates");
  if (gates.size() > kMaximumGates)
    throw std::invalid_argument("message gates the fused kernels do not take: a call takes 1 to 3 gates");
  const uint32_t rank = gates.front().weights.down.outputSize;
  std::array<MessageGateWeights, kMaximumGates> weights;
  for (size_t index = 0; index < gates.size(); ++index) {
    const MessageGateInput &gate = gates[index];
    weights[index] = gate.weights;
    if (gate.xColumn + uint64_t{hidden} > buffers.xStride ||
        gate.messageColumn + uint64_t{hidden} > buffers.messageStride || gate.weights.down.outputSize != rank ||
        gate.weights.down.inputSize != 2 * hidden || gate.weights.forget.inputSize != rank ||
        gate.weights.gain.inputSize != rank || gate.weights.forget.outputSize != hidden ||
        gate.weights.gain.outputSize != hidden)
      throw std::invalid_argument("invalid message gate");
    if (!gate.norm.float32) throw std::invalid_argument("a message gate norm is not F32");
    if (!quads(buffers.xStride, gate.xColumn) || !quads(buffers.messageStride, gate.messageColumn))
      throw std::invalid_argument("message gates the fused kernels do not take: x and the messages in whole float4s");
    requireBytes(buffers.x, rowBytes(rows, buffers.xStride, uint64_t{gate.xColumn} + hidden, sizeof(float)),
                 "message gate row");
    requireBytes(buffers.message,
                 rowBytes(rows, buffers.messageStride, uint64_t{gate.messageColumn} + hidden, sizeof(float)),
                 "message gate message");
    requireBytes(gate.norm.buffer, gate.norm.bytes(hidden), "message gate norm");
    requireBytes(gate.output, uint64_t{rows} * hidden * sizeof(uint16_t), "message gate output");
  }
  const auto count = static_cast<uint32_t>(gates.size());
  requireWeights(std::span(weights).first(count), hidden);
  const uint32_t downFormat = gates.front().weights.down.formatId, upFormat = gates.front().weights.forget.formatId;
  const uint64_t downWeightBytes = downFormat == GGUF_FMT_BF16 ? sizeof(uint16_t) : sizeof(float);
  const uint64_t upWeightBytes = upFormat == GGUF_FMT_BF16 ? sizeof(uint16_t) : sizeof(float);
  const std::string downSuffix = downFormat == GGUF_FMT_BF16 ? "_bf16" : "_f32";
  const std::string upSuffix = upFormat == GGUF_FMT_BF16 ? "_bf16" : "_f32";
  const std::string rowsSuffix = downFormat == upFormat ? downSuffix : downSuffix + upSuffix;
  MessageGateFusedParams p{rows, hidden, buffers.xStride, buffers.messageStride, rank, count, {}, {}};
  // A slot past the call's gates binds its first gate's buffers, which the
  // kernels never read.
  const auto slot = [&](uint32_t index) -> const MessageGateInput & { return gates[index < count ? index : 0]; };
  for (uint32_t index = 0; index < count; ++index) {
    requireBytes(weights[index].down.plane0, uint64_t{rank} * 2 * hidden * downWeightBytes, "message gate down weights");
    requireBytes(weights[index].forget.plane0, uint64_t{hidden} * rank * upWeightBytes, "message gate forget weights");
    requireBytes(weights[index].gain.plane0, uint64_t{hidden} * rank * upWeightBytes, "message gate gain weights");
    p.x_columns[index] = gates[index].xColumn;
    p.message_columns[index] = gates[index].messageColumn;
  }
  if (rows > kFusedRows) {
    // Gate g's gated rows, fp32 [rows][hidden], in forget, gain and concat.
    const std::array<metal::MetalBuffer, kMaximumGates> gated{buffers.forget, buffers.gain, buffers.concat};
    for (uint32_t index = 0; index < count; ++index)
      requireBytes(gated[index], uint64_t{rows} * hidden * sizeof(float), "message gate gated rows");
    const auto scratch = [&](uint32_t index) { return gated[index < count ? index : 0]; };
    graph.add("message_gate_rows" + rowsSuffix,
              {buffers.x, buffers.message, slot(0).weights.down.plane0, slot(1).weights.down.plane0,
               slot(2).weights.down.plane0, slot(0).weights.forget.plane0, slot(1).weights.forget.plane0,
               slot(2).weights.forget.plane0, slot(0).weights.gain.plane0, slot(1).weights.gain.plane0,
               slot(2).weights.gain.plane0, slot(0).norm.buffer, slot(1).norm.buffer, slot(2).norm.buffer,
               scratch(0), scratch(1), scratch(2), slot(0).output, slot(1).output, slot(2).output},
              p, {(rows + MESSAGE_GATE_TILE_ROWS - 1) / MESSAGE_GATE_TILE_ROWS, 1, count},
              {MESSAGE_GATE_TILE_THREADS, 1, 1});
    return {};
  }
  const metal::DispatchSize threads{MESSAGE_GATE_THREADS, 1, 1};
  const uint32_t groups = (rows + MESSAGE_GATE_GROUP_ROWS - 1) / MESSAGE_GATE_GROUP_ROWS;
  const uint32_t slices = (hidden + MESSAGE_GATE_THREADS - 1) / MESSAGE_GATE_THREADS;
  requireBytes(buffers.low, uint64_t{count} * rows * rank * sizeof(uint16_t), "message gate rank values");
  // The normalized concatenations, bf16 [gates][rows][2 hidden], which the
  // down projection reads before the apply writes the gated rows, fp32
  // [gates][rows][hidden], over the same bytes of forget.
  static_assert(2 * sizeof(uint16_t) == sizeof(float), "a gate's concatenation takes its gated rows' bytes");
  requireBytes(buffers.forget, uint64_t{count} * rows * hidden * sizeof(float), "message gate gated rows");
  requireBytes(buffers.gain, uint64_t{count} * groups * slices * MESSAGE_GATE_GROUP_ROWS * sizeof(float),
               "message gate partial sums");
  requireBytes(buffers.counters, uint64_t{count} * groups * sizeof(uint32_t), "message gate counters");
  graph.add("message_gate_normalize", {buffers.x, buffers.message, buffers.forget}, p, {rows, 1, count}, threads);
  graph.add("message_gate_down" + downSuffix,
            {buffers.forget, slot(0).weights.down.plane0, slot(1).weights.down.plane0, slot(2).weights.down.plane0,
             buffers.low},
            p, {(rank + MESSAGE_GATE_DOWN_OUTPUTS - 1) / MESSAGE_GATE_DOWN_OUTPUTS, groups, count}, threads);
  graph.add("message_gate_apply" + upSuffix,
            {buffers.x, buffers.message, slot(0).weights.forget.plane0, slot(1).weights.forget.plane0,
             slot(2).weights.forget.plane0, slot(0).weights.gain.plane0, slot(1).weights.gain.plane0,
             slot(2).weights.gain.plane0, buffers.low, slot(0).norm.buffer, slot(1).norm.buffer,
             slot(2).norm.buffer, buffers.forget, buffers.gain, buffers.counters, slot(0).output, slot(1).output,
             slot(2).output},
            p, {slices, groups, count}, threads);
  return {};
}

} // namespace splash::ops

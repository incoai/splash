#include "ops/LoopControl.hpp"

#include "ops/BufferExtent.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace splash::ops {
namespace {

// The kernels' MMA steps take 64 inputs, the gate's threadgroups 64 columns
// and the down projection's 8, and their rows load as float4 pairs.
void requireShape(const LoopControlShape &shape) {
  if (!shape.hidden || shape.hidden % LOOP_CONTROL_GATE_COLUMNS || !shape.rank || shape.rank % 64 ||
      shape.inputStride < shape.hidden || shape.messageStride < shape.hidden || shape.outputStride < shape.hidden ||
      shape.inputStride % 4 || shape.messageStride % 4 || shape.outputStride % 4)
    throw std::invalid_argument("invalid loop control shape");
}

uint32_t tiles(uint32_t rows) { return (rows + LOOP_CONTROL_TILE_ROWS - 1) / LOOP_CONTROL_TILE_ROWS; }

} // namespace

uint64_t LoopControl::lowBytes(uint32_t rank, uint32_t rows) noexcept { return uint64_t{rows} * rank * sizeof(float); }

void LoopControl::add(metal::CommandGraph &graph, std::span<const LoopControlStream> streams, LoopControlShape shape,
                      uint32_t rows) {
  requireShape(shape);
  if (streams.empty() || streams.size() > LOOP_CONTROL_MAXIMUM_STREAMS || !rows)
    throw std::invalid_argument("invalid loop control");
  const FloatElement element = streams.front().weights ? streams.front().weights->element : FloatElement::Float32;
  const uint64_t weightBytes = elementBytes(element), hidden = shape.hidden, rank = shape.rank;
  for (const LoopControlStream &stream : streams) {
    if (!stream.weights || stream.weights->element != element)
      throw std::invalid_argument("loop control streams take weights of one element type");
    requireBytes(stream.weights->down, rank * 2 * hidden * weightBytes, "loop control down weights");
    requireBytes(stream.weights->forget, hidden * rank * weightBytes, "loop control forget weights");
    requireBytes(stream.weights->gain, hidden * rank * weightBytes, "loop control gain weights");
    requireBytes(stream.input, rowBytes(rows, shape.inputStride, hidden, sizeof(float)), "loop control input");
    requireBytes(stream.message, rowBytes(rows, shape.messageStride, hidden, sizeof(float)), "loop control message");
    requireBytes(stream.output, rowBytes(rows, shape.outputStride, hidden, sizeof(float)), "loop control output");
    requireBytes(stream.low, lowBytes(shape.rank, rows), "loop control low rows");
  }
  const auto count = static_cast<uint32_t>(streams.size());
  const LoopControlParams params{
      rows, shape.hidden, shape.rank, count, shape.inputStride, shape.messageStride, shape.outputStride};
  // Each of a buffer's LOOP_CONTROL_MAXIMUM_STREAMS slots binds its stream's,
  // the slots past the streams the first stream's, which they do not touch.
  const auto bind = [&](std::vector<metal::MetalBuffer> &bindings, auto member) {
    for (uint32_t s = 0; s < LOOP_CONTROL_MAXIMUM_STREAMS; ++s) bindings.push_back(member(streams[s < count ? s : 0]));
  };
  const std::string token = elementToken(element);
  std::vector<metal::MetalBuffer> down;
  bind(down, [](const LoopControlStream &s) { return s.input; });
  bind(down, [](const LoopControlStream &s) { return s.message; });
  bind(down, [](const LoopControlStream &s) { return s.weights->down; });
  bind(down, [](const LoopControlStream &s) { return s.low; });
  graph.add("loop_control_down_" + token, std::move(down), params,
            {shape.rank / LOOP_CONTROL_DOWN_COLUMNS, tiles(rows), count}, {LOOP_CONTROL_THREADS, 1, 1});
  std::vector<metal::MetalBuffer> gate;
  bind(gate, [](const LoopControlStream &s) { return s.low; });
  bind(gate, [](const LoopControlStream &s) { return s.weights->forget; });
  bind(gate, [](const LoopControlStream &s) { return s.weights->gain; });
  bind(gate, [](const LoopControlStream &s) { return s.input; });
  bind(gate, [](const LoopControlStream &s) { return s.message; });
  bind(gate, [](const LoopControlStream &s) { return s.output; });
  graph.add("loop_control_gate_" + token, std::move(gate), params,
            {shape.hidden / LOOP_CONTROL_GATE_COLUMNS, tiles(rows), count}, {LOOP_CONTROL_THREADS, 1, 1});
}

} // namespace splash::ops

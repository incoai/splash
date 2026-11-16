#include "ops/StateCodec.hpp"

#include "ops/BufferExtent.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace splash::ops {
namespace {

constexpr uint32_t kThreads = metal::CommandGraph::kDefaultThreads;
static_assert(kThreads % 2 == 0, "a block's two threads share a threadgroup");

StateCodecParams params(const StateCodecShape &shape) {
  return {shape.stateLayerBytes, shape.codesLayerBytes, state_codec_scales_offset(shape.rows, shape.columns),
          static_cast<uint32_t>(uint64_t{shape.rows} * shape.columns / SPLASH_STATE_CODEC_THREAD_VALUES), 0};
}

// Each layer's values and codes, to the last layer's end.
void requireExtents(const StateCodecShape &shape, const metal::MetalBuffer &state, const metal::MetalBuffer &codes) {
  if (!shape.valid())
    throw std::invalid_argument("invalid state codec shape");
  requireBytes(state, uint64_t{shape.layers - 1} * shape.stateLayerBytes + shape.stateBytes(), "state codec state");
  requireBytes(codes, uint64_t{shape.layers - 1} * shape.codesLayerBytes + shape.codesBytes(), "state codec codes");
}

void add(metal::CommandGraph &graph, std::string name, metal::MetalBuffer first, metal::MetalBuffer second,
         const StateCodecShape &shape) {
  const StateCodecParams p = params(shape);
  if (shape.format == GdnStateFormat::BFloat16)
    name += "_bf16_state";
  graph.add(std::move(name), {std::move(first), std::move(second)}, p,
            {(p.threads_per_layer + kThreads - 1) / kThreads, shape.layers, 1}, {kThreads, 1, 1});
}

} // namespace

void StateCodec::addQuantize(metal::CommandGraph &graph, metal::MetalBuffer state, metal::MetalBuffer codes,
                             StateCodecShape shape) {
  requireExtents(shape, state, codes);
  add(graph, "state_quantize", std::move(state), std::move(codes), shape);
}

void StateCodec::addDequantize(metal::CommandGraph &graph, metal::MetalBuffer codes, metal::MetalBuffer state,
                               StateCodecShape shape) {
  requireExtents(shape, state, codes);
  add(graph, "state_dequantize", std::move(codes), std::move(state), shape);
}

} // namespace splash::ops

#include "ops/ValueMix.hpp"

#include "ops/BufferExtent.hpp"

#include <stdexcept>
#include <utility>

namespace splash::ops {

constexpr uint32_t kThreads = metal::CommandGraph::kDefaultThreads;

void ValueMix::add(metal::CommandGraph &graph, metal::MetalBuffer normed,
                   metal::MetalBuffer scaledEmbeddings, metal::MetalBuffer lambda,
                   metal::MetalBuffer valueInput, uint32_t rows, uint32_t width) {
  if (!rows || !width)
    throw std::invalid_argument("invalid value mix");
  const uint64_t elements = uint64_t{rows} * width;
  requireBytes(normed, elements * 2, "value mix input");
  requireBytes(scaledEmbeddings, elements * 2, "value mix embeddings");
  requireBytes(lambda, 4, "value mix lambda");
  requireBytes(valueInput, elements * 2, "value mix output");
  // One thread per value; the block's sigmoid is one exp per thread, which
  // the compiler hoists out of the element loop's only varying term.
  graph.add("ripple_value_mix",
            {std::move(normed), std::move(scaledEmbeddings), std::move(lambda),
             std::move(valueInput)},
            elements, {(elements + kThreads - 1) / kThreads, 1, 1},
            {kThreads, 1, 1});
}

} // namespace splash::ops

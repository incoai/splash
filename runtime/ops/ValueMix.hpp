#pragma once

#include "metal/CommandGraph.hpp"

#include <cstdint>

namespace splash::ops {

// The value residual's mix (model/Ripple.hpp RippleLayerWeights'
// valueResidualLambda): a block's value input is the normed input lerped to
// the scaled token embeddings by the sigmoid of the block's lambda, the same
// embedding rows every block mixes back in (the gather writes them scaled).
// `rows` rows of `width` values; the kernel computes the sigmoid of the
// block's scalar and rounds the mix once to bf16.
class ValueMix final {
public:
  static void add(metal::CommandGraph &graph, metal::MetalBuffer normed, metal::MetalBuffer scaledEmbeddings,
                  metal::MetalBuffer lambda, metal::MetalBuffer valueInput, uint32_t rows, uint32_t width);
};

} // namespace splash::ops

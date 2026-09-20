#include "ops/Embedding.hpp"

#include "metal/abi/Embedding.h"

#include <stdexcept>
#include <utility>

namespace splash::ops {
namespace {

const char *embeddingPipeline(uint32_t hiddenSize) {
  switch (hiddenSize) {
  case 5120:
    return "embedding_q4_h5120";
  case 2048:
    return "embedding_q4_h2048";
  default:
    throw std::invalid_argument("unsupported compiled Q4 embedding shape");
  }
}

const char *embeddingPipelineQ8(uint32_t hiddenSize) {
  switch (hiddenSize) {
  case 5120:
    return "embedding_q8_h5120";
  default:
    throw std::invalid_argument("unsupported compiled Q8 embedding shape");
  }
}

} // namespace

void Embedding::add(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                    const Q4Projection &table, metal::MetalBuffer output,
                    uint32_t rows) {
  if (!rows || !table.outputSize || !table.inputSize)
    throw std::invalid_argument("invalid Q4 embedding shape");
  const uint32_t hiddenGroups = (table.inputSize + 127) / 128;
  const Q4EmbeddingParams params{rows, table.outputSize};
  graph.add(embeddingPipeline(table.inputSize),
            {std::move(tokens), table.weights, table.scales, table.biases,
             std::move(output)},
            params, {hiddenGroups, 1, 1});
}

void Embedding::add(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                    const Q8Projection &table, metal::MetalBuffer output,
                    uint32_t rows) {
  if (!rows || !table.outputSize || !table.inputSize)
    throw std::invalid_argument("invalid Q8 embedding shape");
  const uint32_t hiddenGroups = (table.inputSize + 127) / 128;
  const Q4EmbeddingParams params{rows, table.outputSize};
  graph.add(embeddingPipelineQ8(table.inputSize),
            {std::move(tokens), table.weights, table.scales, table.biases,
             std::move(output)},
            params, {hiddenGroups, 1, 1});
}

} // namespace splash::ops

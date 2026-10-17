#include "ops/NgramEmbedding.hpp"

#include "metal/abi/NgramEmbedding.h"

#include <stdexcept>
#include <utility>

namespace splash::ops {
namespace {

constexpr uint32_t kThreads = HYPER_CONNECTION_ROW_THREADS;

// The gate's and convolution's kernels hold a row's per-stream values in
// arrays of the most streams a mix reads.
void requireShape(NgramShape shape) {
  if (!shape.hidden || !shape.streams || shape.streams > HYPER_CONNECTION_MAXIMUM_STREAMS || !shape.taps ||
      !shape.dilation)
    throw std::invalid_argument("invalid n-gram embedding shape");
}

uint32_t groups(uint32_t width) { return (width + kThreads - 1) / kThreads; }

NgramEmbeddingParams ngramParams(NgramShape shape, uint32_t rows, bool writeHistory) {
  return {rows, shape.hidden, shape.streams, shape.taps, shape.dilation, writeHistory ? 1u : 0u};
}

} // namespace

void NgramEmbedding::addGate(metal::CommandGraph &graph, metal::MetalBuffer key, metal::MetalBuffer value,
                             metal::MetalBuffer streams, const NormWeights &keyNorm, const NormWeights &queryNorm,
                             const NormWeights &convolutionNorm, metal::MetalBuffer gated,
                             metal::MetalBuffer convolutionInput, NgramShape shape, uint32_t rows) {
  requireShape(shape);
  graph.add("ngram_embedding_gate",
            {std::move(key), std::move(value), std::move(streams), keyNorm.buffer, queryNorm.buffer,
             convolutionNorm.buffer, std::move(gated), std::move(convolutionInput)},
            ngramParams(shape, rows, false), {rows, 1, 1}, {kThreads, 1, 1});
}

void NgramEmbedding::addConvolution(metal::CommandGraph &graph, metal::MetalBuffer convolutionInput,
                                    metal::MetalBuffer historyIn, metal::MetalBuffer historyOut,
                                    metal::MetalBuffer taps, metal::MetalBuffer gated, metal::MetalBuffer streams,
                                    NgramShape shape, uint32_t rows) {
  requireShape(shape);
  const bool writeHistory = static_cast<bool>(historyOut);
  const uint32_t gridRows = rows + (writeHistory ? shape.history() : 0);
  graph.add("ngram_embedding_conv",
            {std::move(convolutionInput), historyIn, writeHistory ? historyOut : historyIn, std::move(taps),
             std::move(gated), std::move(streams)},
            ngramParams(shape, rows, writeHistory), {gridRows, groups(shape.width()), 1}, {kThreads, 1, 1});
}

void NgramEmbedding::addHistory(metal::CommandGraph &graph, metal::MetalBuffer convolutionInput,
                                metal::MetalBuffer historyIn, metal::MetalBuffer historyOut, NgramShape shape,
                                uint32_t rows) {
  requireShape(shape);
  if (!rows) throw std::invalid_argument("invalid n-gram embedding history");
  graph.add("ngram_embedding_history", {std::move(convolutionInput), std::move(historyIn), std::move(historyOut)},
            ngramParams(shape, rows, true), {shape.history(), groups(shape.width()), 1}, {kThreads, 1, 1});
}

void NgramEmbedding::addCommit(metal::CommandGraph &graph, metal::MetalBuffer convolutionInput,
                               metal::MetalBuffer historyIn, metal::MetalBuffer historyOut, metal::MetalBuffer retained,
                               NgramShape shape) {
  requireShape(shape);
  graph.add("ngram_embedding_commit",
            {std::move(convolutionInput), std::move(historyIn), std::move(historyOut), std::move(retained)},
            ngramParams(shape, 0, true), {shape.history(), groups(shape.width()), 1}, {kThreads, 1, 1});
}

void NgramEmbedding::addHash(metal::CommandGraph &graph, metal::MetalBuffer tokens, metal::MetalBuffer before,
                             metal::MetalBuffer indices, const NgramHashParams &params, uint32_t heads) {
  if (!params.rows || heads != (params.ngram - 1) * params.heads_per_ngram || heads > NGRAM_EMBEDDING_MAXIMUM_HEADS ||
      params.ngram > NGRAM_EMBEDDING_MAXIMUM_NGRAM)
    throw std::invalid_argument("invalid n-gram hash");
  graph.add("ngram_embedding_hash", {std::move(tokens), std::move(before), std::move(indices)}, params,
            {groups(params.rows), 1, 1}, {kThreads, 1, 1});
}

} // namespace splash::ops

#include "ops/NgramEmbedding.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/NgramEmbedding.h"
#include "ops/BufferExtent.hpp"

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

// The bytes of `rows` rows of the streams' width, in bf16 (the convolution
// input) or fp32 (the streams and the gated values), and of the history, the
// bf16 convolution inputs of its rows, which a state cell holds whole and a
// kernel taking it may read all of.
uint64_t bfloat16Rows(NgramShape shape, uint32_t rows) { return uint64_t{rows} * shape.width() * sizeof(uint16_t); }
uint64_t floatRows(NgramShape shape, uint32_t rows) { return uint64_t{rows} * shape.width() * sizeof(float); }
uint64_t historyBytes(NgramShape shape) { return bfloat16Rows(shape, shape.history()); }

} // namespace

void NgramEmbedding::addGate(metal::CommandGraph &graph, metal::MetalBuffer key, metal::MetalBuffer value,
                             metal::MetalBuffer streams, const NormWeights &keyNorm, const NormWeights &queryNorm,
                             const NormWeights &convolutionNorm, metal::MetalBuffer gated,
                             metal::MetalBuffer convolutionInput, NgramShape shape, uint32_t rows) {
  requireShape(shape);
  if (!keyNorm.float32 || !queryNorm.float32 || !convolutionNorm.float32)
    throw std::invalid_argument("the n-gram embedding norms are not F32");
  requireBytes(key, bfloat16Rows(shape, rows), "n-gram embedding key");
  requireBytes(value, uint64_t{rows} * shape.hidden * sizeof(uint16_t), "n-gram embedding value");
  requireBytes(streams, floatRows(shape, rows), "n-gram embedding streams");
  requireBytes(keyNorm.buffer, keyNorm.bytes(shape.width()), "n-gram embedding key norm");
  requireBytes(queryNorm.buffer, queryNorm.bytes(shape.width()), "n-gram embedding query norm");
  requireBytes(convolutionNorm.buffer, convolutionNorm.bytes(shape.width()), "n-gram embedding convolution norm");
  requireBytes(gated, floatRows(shape, rows), "n-gram embedding gated");
  requireBytes(convolutionInput, bfloat16Rows(shape, rows), "n-gram embedding convolution input");
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
  requireBytes(convolutionInput, bfloat16Rows(shape, rows), "n-gram embedding convolution input");
  requireBytes(historyIn, historyBytes(shape), "n-gram embedding history");
  if (writeHistory) requireBytes(historyOut, historyBytes(shape), "n-gram embedding next history");
  requireBytes(taps, uint64_t{shape.width()} * shape.taps * sizeof(float), "n-gram embedding convolution taps");
  requireBytes(gated, floatRows(shape, rows), "n-gram embedding gated");
  requireBytes(streams, floatRows(shape, rows), "n-gram embedding streams");
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
  requireBytes(convolutionInput, bfloat16Rows(shape, rows), "n-gram embedding convolution input");
  requireBytes(historyIn, historyBytes(shape), "n-gram embedding history");
  requireBytes(historyOut, historyBytes(shape), "n-gram embedding next history");
  graph.add("ngram_embedding_history", {std::move(convolutionInput), std::move(historyIn), std::move(historyOut)},
            ngramParams(shape, rows, true), {shape.history(), groups(shape.width()), 1}, {kThreads, 1, 1});
}

void NgramEmbedding::addCommit(metal::CommandGraph &graph, metal::MetalBuffer convolutionInput,
                               metal::MetalBuffer historyIn, metal::MetalBuffer historyOut, metal::MetalBuffer retained,
                               NgramShape shape) {
  requireShape(shape);
  // The GPU's count is of the lane's verify rows, any of which it may retain.
  requireBytes(convolutionInput, bfloat16Rows(shape, SPLASH_TARGET_VERIFY_ROWS), "n-gram embedding convolution input");
  requireBytes(historyIn, historyBytes(shape), "n-gram embedding history");
  requireBytes(historyOut, historyBytes(shape), "n-gram embedding next history");
  requireBytes(retained, sizeof(uint32_t), "n-gram embedding retained rows");
  graph.add("ngram_embedding_commit",
            {std::move(convolutionInput), std::move(historyIn), std::move(historyOut), std::move(retained)},
            ngramParams(shape, 0, true), {shape.history(), groups(shape.width()), 1}, {kThreads, 1, 1});
}

void NgramEmbedding::addHash(metal::CommandGraph &graph, metal::MetalBuffer tokens, metal::MetalBuffer before,
                             metal::MetalBuffer indices, const NgramHashParams &params, uint32_t heads) {
  if (!params.rows || heads != (params.ngram - 1) * params.heads_per_ngram || heads > NGRAM_EMBEDDING_MAXIMUM_HEADS ||
      params.ngram > NGRAM_EMBEDDING_MAXIMUM_NGRAM)
    throw std::invalid_argument("invalid n-gram hash");
  // Each lane of SPLASH_TARGET_VERIFY_ROWS rows reads the ngram - 1 tokens
  // before it.
  const uint64_t lanes = (uint64_t{params.rows} + SPLASH_TARGET_VERIFY_ROWS - 1) / SPLASH_TARGET_VERIFY_ROWS;
  requireBytes(tokens, uint64_t{params.rows} * sizeof(uint32_t), "n-gram hash token");
  requireBytes(before, lanes * (params.ngram - 1) * sizeof(uint32_t), "n-gram hash prior token");
  requireBytes(indices, uint64_t{params.rows} * heads * sizeof(uint32_t), "n-gram hash index");
  graph.add("ngram_embedding_hash", {std::move(tokens), std::move(before), std::move(indices)}, params,
            {groups(params.rows), 1, 1}, {kThreads, 1, 1});
}

} // namespace splash::ops

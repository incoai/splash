#include "ops/NgramEmbedding.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/NgramEmbedding.h"
#include "ops/BufferExtent.hpp"
#include "ops/LaneBindings.hpp"

#include <limits>
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

NgramEmbeddingParams ngramParams(NgramShape shape, uint32_t rows, bool writeHistory, bool tree = false,
                                 uint32_t sequences = 0) {
  return {rows, shape.hidden, shape.streams, shape.taps, shape.dilation, writeHistory ? 1u : 0u, tree ? 1u : 0u,
          sequences};
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
                                    std::span<const metal::MetalBuffer> historyIn,
                                    std::span<const metal::MetalBuffer> historyOut, metal::MetalBuffer taps,
                                    metal::MetalBuffer gated, metal::MetalBuffer streams, NgramShape shape,
                                    metal::MetalBuffer layout, const LayoutRows &rows, metal::MetalBuffer tree) {
  requireShape(shape);
  const bool writeHistory = !historyOut.empty(), inTree = static_cast<bool>(tree);
  // A verify block's paths are row masks of 32 rows, and its history follows
  // the path it accepts (addCommit).
  requireLayoutRows(rows, inTree ? 32 : std::numeric_limits<uint32_t>::max());
  if (historyIn.size() != rows.sequences || (writeHistory && historyOut.size() != rows.sequences) ||
      (inTree && writeHistory))
    throw std::invalid_argument("invalid n-gram embedding convolution");
  if (inTree) requireBytes(tree, uint64_t{rows.rows} * sizeof(SplashTreeRow), "n-gram embedding tree rows");
  requireBytes(layout, uint64_t{rows.sequences} * sizeof(SplashLaneRows), "n-gram embedding layout");
  requireBytes(convolutionInput, bfloat16Rows(shape, rows.rows), "n-gram embedding convolution input");
  for (const metal::MetalBuffer &history : historyIn)
    requireBytes(history, historyBytes(shape), "n-gram embedding history");
  for (const metal::MetalBuffer &history : historyOut)
    requireBytes(history, historyBytes(shape), "n-gram embedding next history");
  requireBytes(taps, uint64_t{shape.width()} * shape.taps * sizeof(float), "n-gram embedding convolution taps");
  requireBytes(gated, floatRows(shape, rows.rows), "n-gram embedding gated");
  requireBytes(streams, floatRows(shape, rows.rows), "n-gram embedding streams");
  const auto in = laneBindings(historyIn), out = laneBindings(writeHistory ? historyOut : historyIn);
  // Without a tree the kernel binds another buffer for it, of which it reads
  // nothing.
  metal::MetalBuffer treeRows = inTree ? std::move(tree) : taps;
  graph.add("ngram_embedding_conv",
            {std::move(convolutionInput), in[0], in[1], in[2], in[3], out[0], out[1], out[2], out[3], std::move(taps),
             std::move(gated), std::move(streams), std::move(layout), std::move(treeRows)},
            ngramParams(shape, rows.rows, writeHistory, inTree, rows.sequences),
            {rows.bound + (writeHistory ? shape.history() : 0), groups(shape.width()), rows.sequences},
            {kThreads, 1, 1});
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
                               metal::MetalBuffer historyIn, metal::MetalBuffer historyOut, metal::MetalBuffer path,
                               NgramShape shape, uint32_t rows) {
  requireShape(shape);
  if (!rows || rows > 32) throw std::invalid_argument("invalid n-gram embedding commit");
  // The GPU's path is of the lane's verify rows, any of which it may hold.
  requireBytes(convolutionInput, bfloat16Rows(shape, rows), "n-gram embedding convolution input");
  requireBytes(historyIn, historyBytes(shape), "n-gram embedding history");
  requireBytes(historyOut, historyBytes(shape), "n-gram embedding next history");
  requireBytes(path, sizeof(uint32_t), "n-gram embedding accepted path");
  graph.add("ngram_embedding_commit",
            {std::move(convolutionInput), std::move(historyIn), std::move(historyOut), std::move(path)},
            ngramParams(shape, 0, true), {shape.history(), groups(shape.width()), 1}, {kThreads, 1, 1});
}

void NgramEmbedding::addHash(metal::CommandGraph &graph, NgramHashBuffers buffers, NgramHashParams params,
                             const StepRows &step, uint32_t heads) {
  if (heads != (params.ngram - 1) * params.heads_per_ngram || heads > NGRAM_EMBEDDING_MAXIMUM_HEADS ||
      params.ngram > NGRAM_EMBEDDING_MAXIMUM_NGRAM)
    throw std::invalid_argument("invalid n-gram hash");
  // The kernel finds a row's lane by the lanes' first rows on the device.
  requireStepRows(step);
  params.lanes = step.lanes;
  // Each lane reads the ngram - 1 tokens before its first row.
  const uint64_t rows = step.rows;
  requireBytes(buffers.tokens, rows * sizeof(uint32_t), "n-gram hash token");
  requireBytes(buffers.tree, rows * sizeof(TreeRow), "n-gram hash tree row");
  requireBytes(buffers.laneRows, uint64_t{params.lanes} * sizeof(LaneRows), "n-gram hash lane rows");
  requireBytes(buffers.before, uint64_t{params.lanes} * (params.ngram - 1) * sizeof(uint32_t),
               "n-gram hash prior token");
  requireBytes(buffers.indices, rows * heads * sizeof(uint32_t), "n-gram hash index");
  graph.add("ngram_embedding_hash",
            {std::move(buffers.tokens), std::move(buffers.tree), std::move(buffers.laneRows), std::move(buffers.before),
             std::move(buffers.indices)},
            params, {groups(step.rows), 1, 1}, {kThreads, 1, 1});
}

} // namespace splash::ops

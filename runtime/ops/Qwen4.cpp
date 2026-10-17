#include "ops/Qwen4.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Qwen4.h"
#include "metal/abi/RoPE.h"
#include "ops/BufferExtent.hpp"
#include "ops/LaneBindings.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::ops {
namespace {

constexpr uint32_t kThreads = QWEN4_ROW_THREADS;

void requireShape(HyperShape shape) {
  if (!shape.hidden || !shape.streams || shape.streams > QWEN4_MAXIMUM_STREAMS)
    throw std::invalid_argument("invalid hyper-connection shape");
}

uint32_t groups(uint32_t width) { return (width + kThreads - 1) / kThreads; }

Qwen4HyperParams hyperParams(HyperShape shape, uint32_t rows, bool inject = false) {
  return {rows, shape.hidden, shape.streams, shape.rank, inject ? 1u : 0u};
}

Qwen4PleParams pleParams(PleShape shape, uint32_t rows, bool writeHistory) {
  return {rows, shape.hidden, shape.streams, shape.taps, shape.dilation, writeHistory ? 1u : 0u};
}

} // namespace

void Qwen4::addInit(metal::CommandGraph &graph, metal::MetalBuffer embedding, metal::MetalBuffer streams,
                    HyperShape shape, uint32_t rows) {
  requireShape(shape);
  graph.add("qwen4_hyper_init", {std::move(embedding), std::move(streams)}, hyperParams(shape, rows),
            {rows, groups(shape.hidden), 1}, {kThreads, 1, 1});
}

PreparedInput Qwen4::addNorm(metal::CommandGraph &graph, metal::MetalBuffer streams, const NormWeights &norm,
                             metal::MetalBuffer inject, metal::MetalBuffer normalized, metal::MetalBuffer weights,
                             HyperShape shape, uint32_t rows, LinearScratch scratch, LinearInput layout) {
  requireShape(shape);
  if (shape.streams != QWEN4_MAXIMUM_STREAMS || shape.hidden != QWEN4_NORM_HIDDEN)
    throw std::invalid_argument("no hyper-connection norm kernel for the shape");
  const bool injects = static_cast<bool>(inject);
  // A mix without injection binds its norm in the unused slots.
  std::vector<metal::MetalBuffer> bindings{std::move(streams), norm.buffer, injects ? inject : norm.buffer, normalized,
                                           injects ? weights : norm.buffer};
  // An injecting mix of decode's few rows runs a threadgroup of a quarter per
  // stream each, the rest one of kThreads (kernels/shared/qwen4.metal).
  const bool quarters = injects && rows <= QWEN4_NORM_QUARTER_ROWS;
  const std::string name = quarters ? "qwen4_hyper_norm_quarters" : "qwen4_hyper_norm_loaded";
  const uint32_t threads = quarters ? QWEN4_NORM_QUARTER_THREADS : kThreads;
  if (layout == LinearInput::Plain) {
    graph.add(name, std::move(bindings), hyperParams(shape, rows, injects), {rows, 1, 1}, {threads, 1, 1});
    return {};
  }
  requireTableScratch(scratch, layout, shape.width(), rows);
  bindings.insert(bindings.end(), {scratch.input, scratch.sums});
  graph.add(name + tableSuffix(layout), std::move(bindings), hyperParams(shape, rows, injects), {rows, 1, 1},
            {threads, 1, 1});
  return {std::move(normalized), layout};
}

PreparedInput Qwen4::addLow(metal::CommandGraph &graph, metal::MetalBuffer low, HyperShape shape, uint32_t rows,
                            LinearScratch scratch, LinearInput layout) {
  requireShape(shape);
  if (layout == LinearInput::Plain) {
    graph.add("qwen4_hyper_low", {std::move(low)}, hyperParams(shape, rows), {groups(rows * shape.rank), 1, 1},
              {kThreads, 1, 1});
    return {};
  }
  requireTableScratch(scratch, layout, shape.rank, rows);
  graph.add(std::string("qwen4_hyper_low") + tableSuffix(layout), {low, scratch.input, scratch.sums},
            hyperParams(shape, rows), {rows, 1, 1}, {kThreads, 1, 1});
  return {std::move(low), layout};
}

PreparedInput Qwen4::addMix(metal::CommandGraph &graph, metal::MetalBuffer normalized, metal::MetalBuffer gate,
                            metal::MetalBuffer mixed, HyperShape shape, uint32_t rows, LinearScratch scratch,
                            LinearInput layout) {
  requireShape(shape);
  if (layout == LinearInput::Plain) {
    graph.add("qwen4_hyper_mix", {std::move(normalized), std::move(gate), std::move(mixed)},
              hyperParams(shape, rows), {rows, groups(shape.hidden), 1}, {kThreads, 1, 1});
    return {};
  }
  // Each threadgroup writes whole spans of its 256 values.
  if (shape.hidden % kThreads) throw std::invalid_argument("a mix table takes whole threadgroups of values");
  requireTableScratch(scratch, layout, shape.hidden, rows);
  graph.add(std::string("qwen4_hyper_mix") + tableSuffix(layout),
            {std::move(normalized), std::move(gate), mixed, scratch.input, scratch.sums}, hyperParams(shape, rows),
            {rows, groups(shape.hidden), 1}, {kThreads, 1, 1});
  return {std::move(mixed), layout};
}

void Qwen4::addCombine(metal::CommandGraph &graph, metal::MetalBuffer streams, metal::MetalBuffer branch,
                       metal::MetalBuffer weights, HyperShape shape, uint32_t rows) {
  requireShape(shape);
  graph.add("qwen4_hyper_combine", {std::move(streams), std::move(branch), std::move(weights)},
            hyperParams(shape, rows), {rows, groups(shape.hidden), 1}, {kThreads, 1, 1});
}

void Qwen4::addPleGate(metal::CommandGraph &graph, metal::MetalBuffer key, metal::MetalBuffer value,
                       metal::MetalBuffer streams, const NormWeights &keyNorm, const NormWeights &queryNorm,
                       const NormWeights &convolutionNorm, metal::MetalBuffer gated,
                       metal::MetalBuffer convolutionInput, PleShape shape, uint32_t rows) {
  graph.add("qwen4_ple_gate",
            {std::move(key), std::move(value), std::move(streams), keyNorm.buffer, queryNorm.buffer,
             convolutionNorm.buffer, std::move(gated), std::move(convolutionInput)},
            pleParams(shape, rows, false), {rows, 1, 1}, {kThreads, 1, 1});
}

void Qwen4::addPleConvolution(metal::CommandGraph &graph, metal::MetalBuffer convolutionInput,
                              metal::MetalBuffer historyIn, metal::MetalBuffer historyOut, metal::MetalBuffer taps,
                              metal::MetalBuffer gated, metal::MetalBuffer streams, PleShape shape, uint32_t rows) {
  const bool writeHistory = static_cast<bool>(historyOut);
  const uint32_t gridRows = rows + (writeHistory ? shape.history() : 0);
  graph.add("qwen4_ple_conv",
            {std::move(convolutionInput), historyIn, writeHistory ? historyOut : historyIn, std::move(taps),
             std::move(gated), std::move(streams)},
            pleParams(shape, rows, writeHistory), {gridRows, groups(shape.width()), 1}, {kThreads, 1, 1});
}

void Qwen4::addPleHistory(metal::CommandGraph &graph, metal::MetalBuffer convolutionInput,
                          metal::MetalBuffer historyIn, metal::MetalBuffer historyOut, PleShape shape,
                          uint32_t rows) {
  if (!rows) throw std::invalid_argument("invalid PLE history");
  graph.add("qwen4_ple_history", {std::move(convolutionInput), std::move(historyIn), std::move(historyOut)},
            pleParams(shape, rows, true), {shape.history(), groups(shape.width()), 1}, {kThreads, 1, 1});
}

void Qwen4::addPleCommit(metal::CommandGraph &graph, metal::MetalBuffer convolutionInput,
                         metal::MetalBuffer historyIn, metal::MetalBuffer historyOut, metal::MetalBuffer retained,
                         PleShape shape) {
  graph.add("qwen4_ple_commit",
            {std::move(convolutionInput), std::move(historyIn), std::move(historyOut), std::move(retained)},
            pleParams(shape, 0, true), {shape.history(), groups(shape.width()), 1}, {kThreads, 1, 1});
}

void Qwen4::addStreamMean(metal::CommandGraph &graph, metal::MetalBuffer streams, metal::MetalBuffer destination,
                          RowRegion to, HyperShape shape, uint32_t rows) {
  requireShape(shape);
  if (!rows || uint64_t{to.column} + shape.hidden > to.stride) throw std::invalid_argument("invalid stream mean");
  requireBytes(streams, uint64_t{rows} * shape.width() * sizeof(float), "stream mean input");
  requireBytes(destination, (uint64_t{to.row} + rows) * to.stride * sizeof(uint16_t), "stream mean output");
  const Qwen4StreamMeanParams params{rows, shape.hidden, shape.streams, to.row, to.stride, to.column};
  graph.add("qwen4_stream_mean", {std::move(streams), std::move(destination)}, params,
            {rows, groups(shape.hidden), 1}, {kThreads, 1, 1});
}

void Qwen4::addPleHash(metal::CommandGraph &graph, metal::MetalBuffer tokens, metal::MetalBuffer before,
                       metal::MetalBuffer indices, const Qwen4PleHashParams &params, uint32_t heads) {
  if (!params.rows || heads != (params.ngram - 1) * params.heads_per_ngram || heads > QWEN4_PLE_MAXIMUM_HEADS ||
      params.ngram > QWEN4_PLE_MAXIMUM_NGRAM)
    throw std::invalid_argument("invalid PLE hash");
  graph.add("qwen4_ple_hash", {std::move(tokens), std::move(before), std::move(indices)}, params,
            {groups(params.rows), 1, 1}, {kThreads, 1, 1});
}

namespace {

// The parameters every QSA dispatch of `rows`' lanes shares.
Qwen4QsaParams qsaParams(const QsaRows &rows, QsaIndex index) {
  if (!rows.rows || rows.firstPositions.empty() || rows.firstPositions.size() > SPLASH_MAXIMUM_BATCH_WIDTH ||
      rows.pageTables.size() != rows.firstPositions.size())
    throw std::invalid_argument("invalid QSA rows");
  Qwen4QsaParams params{.lanes = static_cast<uint32_t>(rows.firstPositions.size()),
                        .rows = rows.rows,
                        .first_position = {},
                        .counts = {},
                        .lane_records = 0,
                        .row_offset = 0,
                        .top_blocks = rows.topBlocks,
                        .mask_words = rows.maskWords,
                        .score_stride = rows.scoreStride,
                        .keys_offset = index.keysOffset,
                        .pooled_offset = index.pooledOffset};
  std::ranges::copy(rows.firstPositions, std::begin(params.first_position));
  return params;
}

} // namespace

void Qwen4::addQsaProject(metal::CommandGraph &graph, metal::MetalBuffer query, metal::MetalBuffer key,
                          const NormWeights &queryNorm, metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
                          QsaIndex index, metal::MetalBuffer queries, const QsaRows &rows) {
  const Qwen4QsaParams params = qsaParams(rows, index);
  if (!queryNorm.float32) throw std::invalid_argument("the QSA query norm is not F32");
  const uint64_t laneRows = uint64_t{params.lanes} * rows.rows;
  requireBytes(query, laneRows * QWEN4_QSA_HEADS * QWEN4_QSA_DIMENSION * sizeof(float), "QSA query");
  requireBytes(key, laneRows * QWEN4_QSA_DIMENSION * sizeof(float), "QSA key");
  requireBytes(queryNorm.buffer, QWEN4_QSA_DIMENSION * sizeof(float), "QSA query norm");
  requireBytes(ropeCos, laneRows * SPLASH_TARGET_ROPE_PAIRS * sizeof(float), "QSA RoPE cosine");
  requireBytes(ropeSin, laneRows * SPLASH_TARGET_ROPE_PAIRS * sizeof(float), "QSA RoPE sine");
  requireBytes(queries, laneRows * QWEN4_QSA_HEADS * QWEN4_QSA_DIMENSION * sizeof(uint16_t), "QSA queries");
  const auto tables = laneBindings(rows.pageTables);
  graph.add("qwen4_qsa_project",
            {std::move(query), std::move(key), queryNorm.buffer, std::move(ropeCos), std::move(ropeSin), tables[0],
             tables[1], tables[2], tables[3], std::move(queries)},
            params, {rows.rows, QWEN4_QSA_HEADS + 1, params.lanes}, {QWEN4_QSA_DIMENSION, 1, 1});
}

void Qwen4::addQsaPool(metal::CommandGraph &graph, const QsaBlocks &blocks, const NormWeights &keyNorm,
                       metal::MetalBuffer inverseFrequencies, QsaIndex index) {
  if (blocks.counts.empty() || blocks.counts.size() > SPLASH_MAXIMUM_BATCH_WIDTH ||
      blocks.pageTables.size() != blocks.counts.size() || (blocks.counts.size() > 1 && !blocks.laneRecords))
    throw std::invalid_argument("invalid QSA blocks");
  Qwen4QsaParams params{.lanes = static_cast<uint32_t>(blocks.counts.size()),
                        .rows = 0,
                        .first_position = {},
                        .counts = {},
                        .lane_records = blocks.laneRecords,
                        .row_offset = 0,
                        .top_blocks = 0,
                        .mask_words = 0,
                        .score_stride = 0,
                        .keys_offset = index.keysOffset,
                        .pooled_offset = index.pooledOffset};
  uint32_t most = 0;
  for (size_t lane = 0; lane < blocks.counts.size(); ++lane) {
    const uint32_t count = blocks.counts[lane];
    if (count && blocks.counts.size() > 1 && count > blocks.laneRecords)
      throw std::invalid_argument("a QSA lane's blocks overrun the next lane's");
    requireBytes(blocks.records, (uint64_t{blocks.laneRecords} * lane + count) * sizeof(Qwen4QsaBlock),
                 "QSA block");
    params.counts[lane] = count;
    most = std::max(most, count);
  }
  if (!most) return;
  const auto tables = laneBindings(blocks.pageTables);
  graph.add("qwen4_qsa_pool",
            {blocks.records, keyNorm.buffer, std::move(inverseFrequencies), tables[0], tables[1], tables[2],
             tables[3]},
            params, {most, params.lanes, 1}, {QWEN4_QSA_DIMENSION, 1, 1});
}

void Qwen4::addQsaSelect(metal::CommandGraph &graph, metal::MetalBuffer queries, QsaIndex index,
                         metal::MetalBuffer scores, uint32_t scoreRows, metal::MetalBuffer mask,
                         const QsaRows &rows) {
  Qwen4QsaParams params = qsaParams(rows, index);
  if (!scoreRows || !rows.topBlocks || !rows.maskWords ||
      (params.lanes > 1 && scoreRows < uint64_t{params.lanes} * rows.rows))
    throw std::invalid_argument("invalid QSA selection");
  const auto tables = laneBindings(rows.pageTables);
  uint32_t blocks = 0;
  for (const uint32_t first : rows.firstPositions)
    blocks = std::max(blocks, static_cast<uint32_t>(qsaCompleteBlocks(first, rows.rows)));
  // A row's scores hold each complete block before it, and its bitmap those
  // and its tail block: the kernels write no further.
  if (rows.scoreStride < blocks || uint64_t{rows.maskWords} * 32 <= blocks)
    throw std::invalid_argument("QSA scores or bitmaps narrower than the rows' history");
  const uint64_t laneRows = uint64_t{params.lanes} * rows.rows;
  requireBytes(queries, laneRows * QWEN4_QSA_HEADS * QWEN4_QSA_DIMENSION * sizeof(uint16_t), "QSA queries");
  requireBytes(scores, (params.lanes > 1 ? laneRows : std::min(scoreRows, rows.rows)) * uint64_t{rows.scoreStride} *
                           sizeof(float),
               "QSA score");
  requireBytes(mask, laneRows * rows.maskWords * sizeof(uint32_t), "QSA bitmap");
  // One lane's rows in batches of the scratch's rows; several lanes' at once.
  const uint32_t batchRows = params.lanes > 1 ? rows.rows : scoreRows;
  for (uint32_t offset = 0; offset < rows.rows; offset += batchRows) {
    params.row_offset = offset;
    params.rows = std::min(batchRows, rows.rows - offset);
    graph.add("qwen4_qsa_score", {queries, tables[0], tables[1], tables[2], tables[3], scores}, params,
              {(blocks + QWEN4_QSA_SCORE_BLOCKS - 1) / QWEN4_QSA_SCORE_BLOCKS,
               (params.rows + QWEN4_QSA_SCORE_ROWS - 1) / QWEN4_QSA_SCORE_ROWS, params.lanes},
              {QWEN4_QSA_SCORE_THREADS, 1, 1});
    graph.add("qwen4_qsa_select", {scores, mask}, params, {params.rows, params.lanes, 1},
              {QWEN4_QSA_SELECT_THREADS, 1, 1});
  }
}

} // namespace splash::ops

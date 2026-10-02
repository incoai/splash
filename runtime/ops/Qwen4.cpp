#include "ops/Qwen4.hpp"

#include "metal/abi/Qwen4.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

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
  return {rows, shape.hidden, shape.streams, shape.taps, shape.dilation, writeHistory ? 1u : 0u, 1, rows};
}

} // namespace

void Qwen4::addInit(metal::CommandGraph &graph, metal::MetalBuffer embedding, metal::MetalBuffer streams,
                    HyperShape shape, uint32_t rows) {
  requireShape(shape);
  graph.add("qwen4_hyper_init", {std::move(embedding), std::move(streams)}, hyperParams(shape, rows),
            {rows, groups(shape.hidden), 1}, {kThreads, 1, 1});
}

void Qwen4::addNorm(metal::CommandGraph &graph, metal::MetalBuffer streams, const NormWeights &norm,
                    metal::MetalBuffer inject, metal::MetalBuffer normalized, metal::MetalBuffer weights,
                    HyperShape shape, uint32_t rows) {
  requireShape(shape);
  const bool injects = static_cast<bool>(inject);
  // A mix without injection binds its norm in the unused slots.
  graph.add("qwen4_hyper_norm",
            {std::move(streams), norm.buffer, injects ? inject : norm.buffer, std::move(normalized),
             injects ? weights : norm.buffer},
            hyperParams(shape, rows, injects), {rows, 1, 1}, {kThreads, 1, 1});
}

void Qwen4::addLow(metal::CommandGraph &graph, metal::MetalBuffer low, HyperShape shape, uint32_t rows) {
  requireShape(shape);
  graph.add("qwen4_hyper_low", {std::move(low)}, hyperParams(shape, rows), {groups(rows * shape.rank), 1, 1},
            {kThreads, 1, 1});
}

void Qwen4::addMix(metal::CommandGraph &graph, metal::MetalBuffer normalized, metal::MetalBuffer gate,
                   metal::MetalBuffer mixed, HyperShape shape, uint32_t rows) {
  requireShape(shape);
  graph.add("qwen4_hyper_mix", {std::move(normalized), std::move(gate), std::move(mixed)},
            hyperParams(shape, rows), {rows, groups(shape.hidden), 1}, {kThreads, 1, 1});
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

void Qwen4::addPleCommit(metal::CommandGraph &graph, metal::MetalBuffer convolutionInput,
                         metal::MetalBuffer historyIn, metal::MetalBuffer historyOut, metal::MetalBuffer retained,
                         PleShape shape) {
  graph.add("qwen4_ple_commit",
            {std::move(convolutionInput), std::move(historyIn), std::move(historyOut), std::move(retained)},
            pleParams(shape, 0, true), {shape.history(), groups(shape.width()), 1}, {kThreads, 1, 1});
}

void Qwen4::addMtpTokens(metal::CommandGraph &graph, metal::MetalBuffer anchors, metal::MetalBuffer proposals,
                         metal::MetalBuffer tokens, uint32_t lanes, uint32_t laneRows, uint32_t proposalsPerLane,
                         uint32_t limit) {
  if (!limit || limit > laneRows || limit > proposalsPerLane + 1) throw std::invalid_argument("invalid MTP step");
  const uint32_t rows = lanes * laneRows;
  const Qwen4MtpParams params{rows, 0, 0, laneRows, limit, 0, proposalsPerLane, 0, 0, 0, 0, 0};
  graph.add("qwen4_mtp_tokens", {std::move(anchors), std::move(proposals), std::move(tokens)}, params,
            {groups(rows), 1, 1}, {kThreads, 1, 1});
}

void Qwen4::addMtpInput(metal::CommandGraph &graph, metal::MetalBuffer embedding, metal::MetalBuffer carried,
                        metal::MetalBuffer history, const NormWeights &embedNorm, const NormWeights &hiddenNorm,
                        metal::MetalBuffer output, HyperShape shape, uint32_t rows, uint32_t limit,
                        uint32_t rowOffset, uint32_t totalRows) {
  requireShape(shape);
  if (!rows || !limit || limit > rows || rowOffset + rows > totalRows) throw std::invalid_argument("invalid MTP input");
  const Qwen4MtpParams params{rows, shape.hidden, shape.streams, totalRows, limit, 0, 0, 0, rowOffset, 0, 0, 0};
  // A single row reads no history: it binds the carried streams instead.
  graph.add("qwen4_mtp_input",
            {std::move(embedding), carried, history ? std::move(history) : carried, embedNorm.buffer,
             hiddenNorm.buffer, std::move(output)},
            params, {rows, 1, 1}, {kThreads, 1, 1});
}

void Qwen4::addMtpWiden(metal::CommandGraph &graph, metal::MetalBuffer projected, metal::MetalBuffer streams,
                        HyperShape shape, uint32_t rows) {
  requireShape(shape);
  const Qwen4MtpParams params{rows, shape.hidden, shape.streams, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  graph.add("qwen4_mtp_widen", {std::move(projected), std::move(streams)}, params,
            {groups(rows * shape.width()), 1, 1}, {kThreads, 1, 1});
}

void Qwen4::addMtpArgmax(metal::CommandGraph &graph, metal::MetalBuffer logits, metal::MetalBuffer proposals,
                         metal::MetalBuffer candidates, uint32_t lanes, uint32_t laneRows, uint32_t vocabulary,
                         uint32_t proposalsPerLane, uint32_t candidatesPerProposal, uint32_t row, bool last) {
  if (!lanes || row >= laneRows || row >= proposalsPerLane) throw std::invalid_argument("invalid MTP argmax");
  const Qwen4MtpParams params{lanes * laneRows, 0, 0, laneRows, 0, vocabulary, proposalsPerLane,
                              candidatesPerProposal, row, last ? 1u : 0u, 0, 0};
  graph.add("qwen4_mtp_argmax", {std::move(logits), std::move(proposals), std::move(candidates)}, params,
            {lanes, 1, 1}, {1024, 1, 1});
}

void Qwen4::addMtpCarry(metal::CommandGraph &graph, metal::MetalBuffer streams, metal::MetalBuffer carried,
                        metal::MetalBuffer retained, HyperShape shape, uint32_t rows) {
  requireShape(shape);
  const bool counted = static_cast<bool>(retained);
  const Qwen4MtpParams params{rows, shape.hidden, shape.streams, rows, rows, 0, 0, 0, 0, 0, counted ? 1u : 0u, 0};
  graph.add("qwen4_mtp_carry", {streams, std::move(carried), counted ? std::move(retained) : streams}, params,
            {groups(shape.width()), 1, 1}, {kThreads, 1, 1});
}

void Qwen4::addPleHash(metal::CommandGraph &graph, metal::MetalBuffer tokens, metal::MetalBuffer before,
                       metal::MetalBuffer indices, const Qwen4PleHashParams &params, uint32_t heads) {
  if (!params.rows || heads != (params.ngram - 1) * params.heads_per_ngram || heads > QWEN4_PLE_MAXIMUM_HEADS ||
      params.ngram > 4)
    throw std::invalid_argument("invalid PLE hash");
  graph.add("qwen4_ple_hash", {std::move(tokens), std::move(before), std::move(indices)}, params,
            {groups(params.rows), 1, 1}, {kThreads, 1, 1});
}

void Qwen4::addQsaProject(metal::CommandGraph &graph, metal::MetalBuffer query, metal::MetalBuffer key,
                          const NormWeights &queryNorm, metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
                          metal::MetalBuffer pageTable, QsaIndex index, metal::MetalBuffer queries,
                          QsaRows rows) {
  if (!rows.rows) throw std::invalid_argument("invalid QSA rows");
  const Qwen4QsaParams params{rows.rows, rows.firstPosition, rows.topBlocks, rows.maskWords, rows.scoreStride, 0, 0, 0,
                              index.keysOffset, index.pooledOffset};
  graph.add("qwen4_qsa_project",
            {std::move(query), std::move(key), queryNorm.buffer, std::move(ropeCos), std::move(ropeSin),
             std::move(pageTable), std::move(queries)},
            params, {rows.rows, QWEN4_QSA_HEADS + 1, 1}, {QWEN4_QSA_DIMENSION, 1, 1});
}

void Qwen4::addQsaPool(metal::CommandGraph &graph, metal::MetalBuffer blocks, uint32_t count,
                       const NormWeights &keyNorm, metal::MetalBuffer inverseFrequencies,
                       metal::MetalBuffer pageTable, QsaIndex index) {
  if (!count) return;
  const Qwen4QsaParams params{0, 0, 0, 0, 0, count, 0, 0, index.keysOffset, index.pooledOffset};
  graph.add("qwen4_qsa_pool",
            {std::move(blocks), keyNorm.buffer, std::move(inverseFrequencies), std::move(pageTable)},
            params, {count, 1, 1}, {QWEN4_QSA_DIMENSION, 1, 1});
}

void Qwen4::addQsaSelect(metal::CommandGraph &graph, metal::MetalBuffer queries, QsaIndex index,
                         metal::MetalBuffer pageTable, metal::MetalBuffer scores, uint32_t scoreRows,
                         metal::MetalBuffer mask, QsaRows rows) {
  if (!rows.rows || !scoreRows || !rows.maskWords) throw std::invalid_argument("invalid QSA selection");
  const uint32_t blocks = static_cast<uint32_t>((uint64_t{rows.firstPosition} + rows.rows) / QWEN4_QSA_BLOCK);
  for (uint32_t offset = 0; offset < rows.rows; offset += scoreRows) {
    const uint32_t batch = std::min(scoreRows, rows.rows - offset);
    const Qwen4QsaParams params{batch, rows.firstPosition, rows.topBlocks, rows.maskWords, rows.scoreStride,
                                0, 0, offset, index.keysOffset, index.pooledOffset};
    graph.add("qwen4_qsa_score", {queries, pageTable, scores}, params,
              {(blocks + 255) / 256, (batch + 7) / 8, 1}, {256, 1, 1});
    graph.add("qwen4_qsa_select", {scores, mask}, params, {batch, 1, 1}, {QWEN4_QSA_SELECT_THREADS, 1, 1});
  }
}

} // namespace splash::ops

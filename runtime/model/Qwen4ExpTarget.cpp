#include "model/Qwen4ExpTarget.hpp"

#include "metal/abi/BlockIndexer.h"
#include "model/Qwen4Exp.hpp"
#include "model/RuntimeArenas.hpp"
#include "model/SparseRowStore.hpp"
#include "ops/BlockIndexer.hpp"
#include "ops/Embedding.hpp"
#include "ops/HyperConnection.hpp"
#include "ops/NgramEmbedding.hpp"
#include "ops/PageStorage.hpp"

#include <stdexcept>
#include <utility>
#include <variant>

namespace splash::model {
namespace {

// The rows a decode step verifies of each of its lanes.
constexpr uint32_t kVerifyRows = ExecutionLimits::targetVerifyRows;

ops::HyperShape hyperShape(const QwenTargetGeometry &g) { return {g.hiddenSize, g.hyperConnections, g.hyperRank}; }
ops::NgramShape ngramShape(const QwenTargetGeometry &g) {
  return {g.hiddenSize, g.hyperConnections, g.pleConvolutionTaps, g.pleNgram};
}

// The QSA indexer's fp32 query and key projections of `rows` rows, one float
// segment each (Qwen4Exp.cpp readIndexer), each on the float tile of its own
// width: in one dispatch when the two take the same tile.
void addIndexerProjections(metal::CommandGraph &graph, const ops::Linear &linear, metal::MetalBuffer input,
                           const Qwen4IndexerWeights &indexer, metal::MetalBuffer query, metal::MetalBuffer key,
                           uint32_t rows) {
  const auto projection = [&](const ops::Projection &weights, metal::MetalBuffer output) {
    const auto &segments = weights.blocks().segments;
    if (segments.size() != 1 || !segments.front().isFloat())
      throw std::invalid_argument("the QSA indexer projections are single float segments");
    return ops::FloatProjection{&segments.front(), std::move(output), weights.outputSize, 0,
                                linear.ggufFloatTile(rows, weights.outputSize, segments.front().formatId)};
  };
  const std::array<ops::FloatProjection, 2> projections{projection(indexer.query, std::move(query)),
                                                        projection(indexer.key, std::move(key))};
  ops::addGgufFloat(graph, std::move(input), projections, rows, ops::FloatOutput::Float32);
}

// Where each attention layer's QSA index sits in the extents of `kv` that
// index tables reach (kv::PageStorage::writeIndexEntries).
std::vector<uint32_t> indexOffsets(const kv::PageStorage &kv, uint32_t layers) {
  std::vector<uint32_t> result;
  for (uint32_t layer = 0; layer < layers; ++layer) result.push_back(kv.indexOffset(layer));
  return result;
}

// The draft's capture of a layer's output: the mean of its output streams,
// `rows` rows from row `sourceRow` of `streams`, into capture slot `slot` of
// the captured hidden rows from row `destinationRow`.
void addStreamCapture(metal::CommandGraph &graph, const QwenTargetGeometry &geometry, uint32_t slot,
                      metal::MetalBuffer streams, metal::MetalBuffer captured, uint32_t destinationRow,
                      uint32_t rows) {
  const uint32_t width = geometry.hiddenSize, capturedWidth = geometry.capturedHiddenSize();
  if (slot >= capturedWidth / width)
    throw std::logic_error("Qwen target capture slot past the captured hidden rows");
  ops::HyperConnection::addStreamMean(graph, std::move(streams), std::move(captured),
                                      {destinationRow, capturedWidth, slot * width}, hyperShape(geometry), rows);
}

} // namespace

Qwen4ExpTarget::Qwen4ExpTarget(const Qwen4ExpWeights &weights, const QwenTargetGeometry &geometry,
                               metal::MetalBackend &backend, const ops::ExecutionPlans &operators,
                               uint32_t prefillChunks)
    : QwenTarget(weights, geometry, backend, operators), weights_(weights),
      steps_(weights, geometry_, backend, prefillChunks) {
  requireLayers(std::span<const Qwen4ExpLayerWeights>(weights.layers));
}

QwenTargetScratchBytes Qwen4ExpTarget::allocatedScratchBytes() const noexcept { return steps_.allocatedBytes(); }

std::vector<uint32_t> Qwen4ExpTarget::verifyTableRows(uint32_t lane) const { return steps_.verifyPleRows(lane); }

std::vector<metal::MetalBuffer> Qwen4ExpTarget::addPrefill(metal::CommandGraph &graph,
                                                           std::span<const QwenTargetPrefillChunk> chunks,
                                                           const kv::PageStorage &kv, SpilledKvReads &spilled,
                                                           QwenPrefillLayers layers, ops::AneFfn *aneFfn) {
  std::vector<PrefillStep> steps = prefillSteps<PrefillStep>(graph, chunks, kv, spilled, layers, aneFfn);
  const std::vector<uint32_t> offsets = indexOffsets(kv, geometry_.kvLayout.attentionLayers);
  // The inputs of the chunks' rows are written once, for their first layer.
  const bool begins = !layers.first;
  for (uint32_t index = 0; index < steps.size(); ++index) {
    PrefillStep &step = steps[index];
    step.scratch = steps_.prefillScratch(index);
    step.blocks = begins ? steps_.preparePrefill(index, chunks[index]) : steps_.prefillBlocks(index, chunks[index]);
    step.indexOffsets = offsets;
    step.ffn = {step.buffers.normalized, step.scratch.zeros, step.scratch.branch, step.buffers.moe, {}};
  }
  if (begins) steps_.readAhead(chunks);
  return addPrefillLayers(std::span(steps), *this, layers);
}

void Qwen4ExpTarget::addPrefillMix(PrefillStep &step, const Qwen4HyperWeights &hyper,
                                   metal::MetalBuffer output) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const Qwen4Scratch &h = step.scratch;
  const ops::Linear &linear = operators_.linear();
  const ops::HyperShape hs = hyperShape(geometry_);
  ops::HyperConnection::addNorm(step.graph, h.streams, hyper.norm, hyper.inject, h.normalized, h.weights, hs,
                                step.rows);
  linear.addPrefill(step.graph, h.normalized, hyper.down, h.low, step.rows, b.linearScratch, b.splitFree);
  ops::HyperConnection::addLow(step.graph, h.low, hs, step.rows);
  linear.addPrefill(step.graph, h.low, hyper.up, h.gate, step.rows, b.linearScratch, b.splitFree);
  ops::HyperConnection::addMix(step.graph, h.normalized, h.gate, std::move(output), hs, step.rows);
}

void Qwen4ExpTarget::addPrefillBegin(PrefillStep &step) const {
  ops::HyperConnection::addInit(step.graph, step.buffers.hidden[0], step.scratch.streams, hyperShape(geometry_),
                                step.rows);
}

bool Qwen4ExpTarget::attends(uint32_t index) const {
  return std::holds_alternative<QwenAttentionWeights>(weights_.layers[index].mixer);
}

void Qwen4ExpTarget::addPrefillLayer(PrefillStep &step, uint32_t index) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const Qwen4Scratch &h = step.scratch;
  const uint32_t rows = step.rows;
  const auto f32Rows = [&](const metal::MetalBuffer &buffer, uint32_t begin, uint32_t count, uint32_t width) {
    return rowsOf<float>(buffer, begin, count, width);
  };
  const auto u16Rows = [&](const metal::MetalBuffer &buffer, uint32_t begin, uint32_t count, uint32_t width) {
    return rowsOf<uint16_t>(buffer, begin, count, width);
  };
  const Qwen4ExpLayerWeights &layer = weights_.layers[index];
  if (layer.ple) {
    const Qwen4PleWeights &ple = *layer.ple;
    const ops::NgramShape ps = ngramShape(geometry_);
    const ops::Linear &linear = operators_.linear();
    addPleGather(step.graph, h, rows * geometry_.pleHeads());
    linear.addPrefill(step.graph, h.pleRows, ple.key, h.gate, rows, b.linearScratch, b.splitFree);
    linear.addPrefill(step.graph, h.pleRows, ple.value, h.pleValue, rows, b.linearScratch, b.splitFree);
    ops::NgramEmbedding::addGate(step.graph, h.gate, h.pleValue, h.streams, ple.keyNorm, ple.queryNorm,
                                 ple.convolutionNorm, h.pleGated, h.pleConvolution, ps, rows);
    for (const QwenTargetPrefillSequence &sequence : step.sequences) {
      ops::NgramEmbedding::addConvolution(
          step.graph, u16Rows(h.pleConvolution, sequence.rowBegin, sequence.rows, ps.width()),
          sequence.auxiliaryIn, sequence.auxiliaryOut, ple.convolution,
          f32Rows(h.pleGated, sequence.rowBegin, sequence.rows, ps.width()),
          f32Rows(h.streams, sequence.rowBegin, sequence.rows, ps.width()), ps, sequence.rows);
      // A state captured inside the command holds the history after its row.
      for (const QwenTargetStateCapture &capture : sequence.stateCaptures)
        ops::NgramEmbedding::addHistory(step.graph,
                                        u16Rows(h.pleConvolution, sequence.rowBegin, capture.row, ps.width()),
                                        sequence.auxiliaryIn, capture.auxiliary, ps, capture.row);
    }
  }
  addPrefillMix(step, layer.mixerHyper, b.normalized);
  std::visit(
      [&](const auto &mixer) {
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(mixer)>, QwenGdnWeights>)
          addPrefillMixer(step, mixer, {h.branch, {}});
        else
          addPrefillMixer(step, mixer, {h.branch, {}}, addPrefillQsa(step, *layer.indexer));
      },
      layer.mixer);
  ops::HyperConnection::addCombine(step.graph, h.streams, h.branch, h.weights, hyperShape(geometry_), rows);
  addPrefillMix(step, layer.ffnHyper, b.normalized);
  ops::MoE::addRouting(step.graph, step.ffn, layer.ffn, *step.moe);
}

const ops::BlockMoeWeights *Qwen4ExpTarget::moe(uint32_t index) const { return &weights_.layers[index].ffn; }

void Qwen4ExpTarget::addPrefillExperts(PrefillStep &step, uint32_t index) const {
  const Qwen4Scratch &h = step.scratch;
  ops::MoE::addRouted(step.graph, step.ffn, weights_.layers[index].ffn, *step.moe);
  ops::HyperConnection::addCombine(step.graph, h.streams, h.branch, h.weights, hyperShape(geometry_), step.rows);
  forEachPrefillCapture(step, index, [&](uint32_t slot, uint32_t source, uint32_t destination, uint32_t rows) {
    addStreamCapture(step.graph, geometry_, slot, rowsOf<float>(h.streams, source, rows, geometry_.streamWidth()),
                     step.buffers.captured, destination, rows);
  });
}

metal::MetalBuffer Qwen4ExpTarget::addPrefillEnd(PrefillStep &step) const {
  addPrefillMix(step, weights_.finalHyper, step.buffers.hidden[1]);
  return step.buffers.hidden[1];
}

// Each sequence's QSA is a dispatch of one lane: its indexer projects its
// rows, pools the blocks they complete and keeps the keys of the block they
// leave open, and those of any state captured inside them; a sequence whose
// rows attend sparsely then selects their blocks, its lists after those of
// the sequences before it.
std::vector<ops::AttentionBlocks> Qwen4ExpTarget::addPrefillQsa(PrefillStep &step,
                                                                const Qwen4IndexerWeights &indexer) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const Qwen4Scratch &h = step.scratch;
  const ops::Linear &linear = operators_.linear();
  const uint32_t layer = step.attentionLayer, topBlocks = weights_.layout.indexerTopBlocks;
  const uint32_t keyWidth = geometry_.indexerHeadDimension, queryWidth = geometry_.indexerHeads * keyWidth;
  const auto u16Rows = [&](const metal::MetalBuffer &buffer, const QwenTargetPrefillSequence &sequence,
                           uint32_t width) {
    return rowsOf<uint16_t>(buffer, sequence.rowBegin, sequence.rows, width);
  };
  const auto f32Rows = [&](const metal::MetalBuffer &buffer, const QwenTargetPrefillSequence &sequence,
                           uint32_t width) {
    return rowsOf<float>(buffer, sequence.rowBegin, sequence.rows, width);
  };
  std::vector<uint32_t> firstPositions;
  for (const QwenTargetPrefillSequence &sequence : step.sequences)
    firstPositions.push_back(static_cast<uint32_t>(sequence.chunk.committed_tokens));
  const auto qsaRows = [&](size_t index) {
    const QwenTargetPrefillSequence &sequence = step.sequences[index];
    return ops::IndexerRows{sequence.rows, std::span(firstPositions).subspan(index, 1),
                            {{&sequence.indexTable, 1}, step.indexOffsets[layer]}, topBlocks, h.maskWords,
                            h.scoreStride};
  };
  addIndexerProjections(step.graph, linear, b.normalized, indexer, h.indexerQuery, h.indexerKey, step.rows);
  std::vector<ops::AttentionBlocks> blocks(step.sequences.size());
  std::vector<ops::BlockListRows> sparseRows;
  uint32_t lists = 0;
  for (size_t index = 0; index < step.sequences.size(); ++index) {
    const QwenTargetPrefillSequence &sequence = step.sequences[index];
    const metal::MetalBuffer keys = u16Rows(h.qsaKeys, sequence, keyWidth);
    const metal::MetalBuffer pending = steps_.qsaPending(sequence.auxiliaryIn, layer);
    ops::BlockIndexer::addProject(step.graph, f32Rows(h.indexerQuery, sequence, queryWidth),
                                  f32Rows(h.indexerKey, sequence, keyWidth), indexer.queryNorm,
                                  f32Rows(b.ropeCos, sequence, geometry_.rotaryPairs),
                                  f32Rows(b.ropeSin, sequence, geometry_.rotaryPairs),
                                  u16Rows(h.qsaQueries, sequence, queryWidth), keys, qsaRows(index));
    ops::BlockIndexer::addPool(step.graph,
                               {step.blocks[index].records, {&step.blocks[index].count, 1}, 0, keys, {&pending, 1}},
                               indexer.keyNorm, b.ropeInverseFrequencies, geometry_.ropeScaling.attentionFactor(),
                               qsaRows(index));
    ops::BlockIndexer::addPending(step.graph, keys, pending, steps_.qsaPending(sequence.auxiliaryOut, layer),
                                  firstPositions[index], sequence.rows);
    for (const QwenTargetStateCapture &capture : sequence.stateCaptures)
      ops::BlockIndexer::addPending(step.graph, keys, pending, steps_.qsaPending(capture.auxiliary, layer),
                                    firstPositions[index], capture.row);
    if (!ops::BlockIndexer::sparse(firstPositions[index], sequence.rows, topBlocks)) continue;
    ops::BlockIndexer::addSelect(step.graph, u16Rows(h.qsaQueries, sequence, queryWidth), h.qsaScores, h.scoreRows,
                                 rowsOf<uint32_t>(h.qsaMask, sequence.rowBegin, sequence.rows, h.maskWords),
                                 qsaRows(index));
    blocks[index] = {h.qsaLists, h.qsaListCounts, h.listStride, lists};
    sparseRows.push_back({sequence.rowBegin, sequence.rows, firstPositions[index], lists});
    lists += step.attention[index].tiles;
  }
  if (!sparseRows.empty())
    ops::PagedAttention::addBlockLists(step.graph, h.qsaMask, h.maskWords, sparseRows,
                                       {h.qsaLists, h.qsaListCounts, h.listStride, 0});
  return blocks;
}

void Qwen4ExpTarget::prepareVerifyLane(uint32_t lane, uint64_t position, const QwenTargetRequest &request) {
  steps_.prepareVerifyLane(lane, position, request);
}

void Qwen4ExpTarget::addVerifyInput(metal::CommandGraph &graph, metal::MetalBuffer draftInput,
                                    metal::MetalBuffer proposals, metal::MetalBuffer verifyInput,
                                    uint32_t lanes) const {
  QwenTarget::addVerifyInput(graph, std::move(draftInput), std::move(proposals), verifyInput, lanes);
  steps_.addVerifyInput(graph, std::move(verifyInput), lanes);
}

ops::PreparedInput Qwen4ExpTarget::addVerifyMix(VerifyStep &step, const Qwen4HyperWeights &hyper,
                                                metal::MetalBuffer output, const ops::Projection *next) const {
  const Qwen4Scratch &h = step.scratch;
  const ops::LinearScratch scratch = step.buffers.linearScratch;
  const ops::HyperShape hs = hyperShape(geometry_);
  ops::HyperConnection::addNorm(step.graph, h.streams, hyper.norm, hyper.inject, h.normalized, h.weights, hs,
                                step.rows);
  addDecodeProjection(step.graph, h.normalized, hyper.down, h.low, step.lanes, scratch);
  ops::PreparedInput prepared =
      ops::HyperConnection::addLow(step.graph, h.low, hs, step.rows, scratch, decodeInput(hyper.up, step.lanes));
  addDecodeProjection(step.graph, h.low, hyper.up, h.gate, step.lanes, scratch, std::move(prepared));
  return ops::HyperConnection::addMix(step.graph, h.normalized, h.gate, std::move(output), hs, step.rows, scratch,
                                      next ? decodeInput(*next, step.lanes) : ops::LinearInput::Plain);
}

// Every lane's QSA in one dispatch of each kernel. Once a lane's rows attend
// sparsely, every lane's attends through its list: a lane's rows that keep
// every block select them all.
ops::AttentionBlocks Qwen4ExpTarget::addVerifyQsa(VerifyStep &step, const Qwen4IndexerWeights &indexer) const {
  const QwenTargetVerifyBuffers &b = step.buffers;
  const Qwen4Scratch &h = step.scratch;
  const ops::Linear &linear = operators_.linear();
  const uint32_t layer = step.attentionLayer, topBlocks = weights_.layout.indexerTopBlocks;
  const uint32_t keyWidth = geometry_.indexerHeadDimension, queryWidth = geometry_.indexerHeads * keyWidth;
  addIndexerProjections(step.graph, linear, b.normalized, indexer, h.indexerQuery, h.indexerKey, step.rows);
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> firstPositions{};
  bool sparse = false;
  for (uint32_t lane = 0; lane < step.lanes; ++lane) {
    firstPositions[lane] = step.chunks[lane].committed_tokens;
    sparse = sparse || ops::BlockIndexer::sparse(firstPositions[lane], kVerifyRows, topBlocks);
  }
  const ops::IndexerRows qsa{kVerifyRows, std::span(firstPositions).first(step.lanes),
                             {std::span(b.indexTables).first(step.lanes), step.indexOffsets[layer]}, topBlocks,
                             h.maskWords, h.scoreStride};
  const metal::MetalBuffer queries = rowsOf<uint16_t>(h.qsaQueries, 0, step.rows, queryWidth);
  // The layer's keys, which the commit reads after the last layer, and each
  // lane's pending keys before the step.
  const metal::MetalBuffer keys = rowsOf<uint16_t>(h.qsaKeys, layer * h.keyLayerRows, step.rows, keyWidth);
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> pending;
  for (uint32_t lane = 0; lane < step.lanes; ++lane) pending[lane] = steps_.qsaPending(b.auxiliary[lane], layer);
  ops::BlockIndexer::addProject(step.graph, rowsOf<float>(h.indexerQuery, 0, step.rows, queryWidth),
                                rowsOf<float>(h.indexerKey, 0, step.rows, keyWidth), indexer.queryNorm,
                                rowsOf<float>(b.ropeCos, 0, step.rows, geometry_.rotaryPairs),
                                rowsOf<float>(b.ropeSin, 0, step.rows, geometry_.rotaryPairs), queries, keys, qsa);
  ops::BlockIndexer::addPool(step.graph,
                             {step.blocks.records, std::span(step.blocks.counts).first(step.lanes),
                              step.blocks.laneRecords, keys, std::span(pending).first(step.lanes)},
                             indexer.keyNorm, b.ropeInverseFrequencies, geometry_.ropeScaling.attentionFactor(),
                             qsa);
  if (!sparse) return {h.qsaLists, h.qsaListCounts, 0, 0};
  const ops::AttentionBlocks blocks{h.qsaLists, h.qsaListCounts, h.listStride, 0};
  ops::BlockIndexer::addSelect(step.graph, queries, h.qsaScores, h.scoreRows,
                               rowsOf<uint32_t>(h.qsaMask, 0, step.rows, h.maskWords), qsa);
  std::array<ops::BlockListRows, ExecutionLimits::maximumBatchWidth> laneLists{};
  for (uint32_t lane = 0; lane < step.lanes; ++lane)
    laneLists[lane] = {lane * kVerifyRows, kVerifyRows, firstPositions[lane], lane};
  ops::PagedAttention::addBlockLists(step.graph, h.qsaMask, h.maskWords, std::span(laneLists).first(step.lanes),
                                     blocks);
  return blocks;
}

// The residual streams start as the embedding's rows, every block reads
// their mix and adds its branch back to them, and the head reads the final
// mix.
void Qwen4ExpTarget::addVerify(metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
                               const kv::PageStorage &kv, SpilledKvReads &spilled,
                               std::span<const kv::ChunkedPrefillParams> chunks, uint32_t lanes) const {
  VerifyStep step{verifyStep(graph, buffers, kv, spilled, chunks, lanes)};
  const std::vector<uint32_t> offsets = indexOffsets(kv, geometry_.kvLayout.attentionLayers);
  step.scratch = steps_.verifyScratch(decodeStorageLanes(lanes) * kVerifyRows);
  step.indexOffsets = offsets;
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> positions{};
  for (uint32_t lane = 0; lane < lanes; ++lane) positions[lane] = chunks[lane].committed_tokens;
  step.blocks = steps_.verifyBlocks(std::span(positions).first(lanes));
  const QwenTargetVerifyBuffers &b = step.buffers;
  const Qwen4Scratch &h = step.scratch;
  const ops::HyperShape hs = hyperShape(geometry_);
  const uint32_t rows = step.rows;
  ops::HyperConnection::addInit(step.graph, b.hidden[0], h.streams, hs, rows);
  for (uint32_t index = 0; index < geometry_.layers; ++index) {
    const Qwen4ExpLayerWeights &layer = weights_.layers[index];
    if (layer.ple) {
      const Qwen4PleWeights &ple = *layer.ple;
      const ops::NgramShape ps = ngramShape(geometry_);
      addPleGather(step.graph, h, rows * geometry_.pleHeads());
      addDecodeProjection(step.graph, h.pleRows, ple.key, h.gate, lanes, b.linearScratch);
      addDecodeProjection(step.graph, h.pleRows, ple.value, h.pleValue, lanes, b.linearScratch);
      ops::NgramEmbedding::addGate(step.graph, h.gate, h.pleValue, h.streams, ple.keyNorm, ple.queryNorm,
                                   ple.convolutionNorm, h.pleGated, h.pleConvolution, ps, rows);
      // Each lane's rows after its history; the commit writes the history
      // after its retained rows.
      for (uint32_t lane = 0; lane < lanes; ++lane)
        ops::NgramEmbedding::addConvolution(
            step.graph, rowsOf<uint16_t>(h.pleConvolution, lane * kVerifyRows, kVerifyRows, ps.width()),
            b.auxiliary[lane], {}, ple.convolution,
            rowsOf<float>(h.pleGated, lane * kVerifyRows, kVerifyRows, ps.width()),
            rowsOf<float>(h.streams, lane * kVerifyRows, kVerifyRows, ps.width()), ps, kVerifyRows);
    }
    std::visit(
        [&](const auto &mixer) {
          const ops::PreparedInput input = addVerifyMix(step, layer.mixerHyper, b.normalized, &mixer.inputProjection);
          if constexpr (std::is_same_v<std::remove_cvref_t<decltype(mixer)>, QwenGdnWeights>)
            addVerifyMixer(step, mixer, input, {h.branch, {}});
          else
            addVerifyMixer(step, mixer, input, {h.branch, {}}, addVerifyQsa(step, *layer.indexer));
        },
        layer.mixer);
    ops::HyperConnection::addCombine(step.graph, h.streams, h.branch, h.weights, hs, rows);
    addVerifyMix(step, layer.ffnHyper, b.normalized);
    ops::MoE::add(step.graph, {b.normalized, h.zeros, h.branch, b.moe, b.verifyRows}, layer.ffn, *step.moe);
    ops::HyperConnection::addCombine(step.graph, h.streams, h.branch, h.weights, hs, rows);
    if (const auto slot = geometry_.captureSlot(index))
      addStreamCapture(step.graph, geometry_, *slot, h.streams, b.capturedTargetHidden, 0, rows);
  }
  requireLayerPartition(step.gdnLayer, step.attentionLayer);
  addDecodeProjection(step.graph, b.finalHidden, weights_.logitsProjection, b.logits, lanes, b.linearScratch,
                      addVerifyMix(step, weights_.finalHyper, b.finalHidden, &weights_.logitsProjection));
}

void Qwen4ExpTarget::addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden, metal::MetalBuffer,
                                  metal::MetalBuffer logits, uint32_t lanes, ops::LinearScratch scratch) const {
  addDecodeProjection(graph, std::move(hidden), vocabularyProjection(), std::move(logits), lanes, scratch);
}

void Qwen4ExpTarget::addStateCommit(metal::CommandGraph &graph, QwenTargetCommitBuffers buffers,
                                    uint32_t lanes) const {
  QwenTarget::addStateCommit(graph, buffers, lanes);
  // Each lane's PLE history and QSA pending keys after its retained rows,
  // from the convolution inputs and the keys its verify wrote.
  const ops::NgramShape shape = ngramShape(geometry_);
  const Qwen4Scratch scratch = steps_.verifyScratch(lanes * kVerifyRows);
  const uint64_t laneBytes = uint64_t{kVerifyRows} * shape.width() * sizeof(uint16_t);
  for (uint32_t lane = 0; lane < lanes; ++lane)
    ops::NgramEmbedding::addCommit(graph, backend_.view(scratch.pleConvolution, lane * laneBytes, laneBytes),
                                   buffers.auxiliaryIn[lane], buffers.auxiliaryOut[lane],
                                   backend_.view(buffers.retainedCounts, lane * sizeof(uint32_t), sizeof(uint32_t)),
                                   shape);
  const uint32_t layers = geometry_.kvLayout.attentionLayers;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> in, out;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    in[lane] = steps_.qsaPending(buffers.auxiliaryIn[lane], 0, layers);
    out[lane] = steps_.qsaPending(buffers.auxiliaryOut[lane], 0, layers);
  }
  const auto positions = steps_.verifyPositions(lanes);
  ops::BlockIndexer::addCommit(graph, scratch.qsaKeys, layers, scratch.keyLayerRows, kVerifyRows,
                               std::span(positions).first(lanes), buffers.retainedCounts, std::span(in).first(lanes),
                               std::span(out).first(lanes));
}

void Qwen4ExpTarget::addPleGather(metal::CommandGraph &graph, const Qwen4Scratch &scratch, uint32_t count) const {
  if (!weights_.streamedTable) {
    ops::Embedding::add(graph, scratch.pleIndices, weights_.pleTable, scratch.pleRows, count);
    return;
  }
  // At most `count` distinct rows, each slot below it: the gather of the
  // table's rows with the staging as a table of `count` rows, so a row's
  // values are computed exactly as from the table.
  const ops::EmbeddingWeights staged{count, geometry_.pleHeadDimension,
                                     ops::NativeRows(scratch.pleStaging, weights_.pleFormat)};
  ops::Embedding::add(graph, scratch.pleSlots, staged, scratch.pleRows, count);
}

void includeFamilyProjections(QwenTargetGeometry &geometry, const Qwen4ExpWeights &weights) {
  const auto hyper = [&](const Qwen4HyperWeights &mix) {
    geometry.include(mix.down);
    geometry.include(mix.up);
  };
  for (const Qwen4ExpLayerWeights &layer : weights.layers) {
    hyper(layer.mixerHyper);
    hyper(layer.ffnHyper);
    if (layer.indexer) {
      geometry.include(layer.indexer->query);
      geometry.include(layer.indexer->key);
    }
    if (layer.ple) {
      geometry.include(layer.ple->key);
      geometry.include(layer.ple->value);
    }
  }
  hyper(weights.finalHyper);
}

std::unique_ptr<QwenTarget> makeQwenTarget(const Qwen4ExpWeights &weights, const QwenTargetGeometry &geometry,
                                           metal::MetalBackend &backend, const ops::ExecutionPlans &operators,
                                           uint32_t prefillChunks) {
  return std::make_unique<Qwen4ExpTarget>(weights, geometry, backend, operators, prefillChunks);
}

QwenTargetScratchBytes qwenTargetScratchBytes(const Qwen4ExpWeights &, const QwenTargetGeometry &geometry,
                                              uint32_t prefillChunks, uint64_t tableRowBytes) {
  return Qwen4Steps::scratchBytes(geometry, tableRowBytes, prefillChunks);
}

// A stage takes a prefill chunk's table rows, pleHeads() for each of its
// rows (Qwen4Steps::preparePrefill), and the read-ahead of the next prefill
// command, a group's chunks at most (Qwen4Steps::readAhead), may have them
// queued or in flight beside a chunk's more.
SparseRowStoreOptions streamedTableRows(const Qwen4ExpLayout &layout) {
  const uint32_t stageRows = ExecutionLimits::prefillTokenBudget * layout.pleHeads();
  return {.prefetchRows = (kPrefillGroupChunks + 1) * stageRows, .stageRows = stageRows};
}

} // namespace splash::model

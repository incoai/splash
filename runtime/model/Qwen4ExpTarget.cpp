#include "model/Qwen4ExpTarget.hpp"

#include "metal/abi/BlockIndexer.h"
#include "model/Qwen4Exp.hpp"
#include "model/RuntimeArenas.hpp"
#include "model/SparseRowStore.hpp"
#include "ops/AneFfn.hpp"
#include "ops/BlockIndexer.hpp"
#include "ops/Embedding.hpp"
#include "ops/HyperConnection.hpp"
#include "ops/NgramEmbedding.hpp"
#include "ops/PageStorage.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <variant>

namespace splash::model {
namespace {

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

// A prompt chunk's rows as its layout holds them
// (QwenTargetPrefillBuffers::laneRows): its sequences' `rows` rows, each at
// most the longest's.
ops::LayoutRows promptRows(std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows) {
  uint32_t longest = 0;
  for (const QwenTargetPrefillSequence &sequence : sequences) longest = std::max(longest, sequence.rows);
  return {static_cast<uint32_t>(sequences.size()), rows, longest};
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

std::vector<uint32_t> Qwen4ExpTarget::verifyTableRows(const ops::LaneRows &rows) const {
  return steps_.verifyPleRows(rows);
}

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
    step.indexOffsets = offsets;
    if (const QwenTargetDecodeBuffers *d = step.decode) {
      // A decode step's inputs are its lanes' (prepareVerifyLane,
      // addVerifyInput): the QSA blocks a lane's rows may complete from its
      // position.
      step.scratch = steps_.verifyScratch(decodeStorageRows(geometry_.lanes));
      std::array<uint32_t, ExecutionLimits::maximumBatchWidth> positions{};
      for (uint32_t lane = 0; lane < d->step.lanes; ++lane)
        positions[lane] = static_cast<uint32_t>(step.chunks[lane].committed_tokens);
      step.laneBlocks = steps_.verifyBlocks(std::span(positions).first(d->step.lanes));
    } else {
      step.scratch = steps_.prefillScratch(index);
      step.blocks = begins ? steps_.preparePrefill(index, chunks[index]) : steps_.prefillBlocks(index, chunks[index]);
    }
    step.ffn = {.input = step.buffers.normalized,
                .residual = step.scratch.zeros,
                .output = step.scratch.branch,
                .scratch = step.buffers.moe};
    if (const QwenTargetDecodeBuffers *d = step.decode) {
      step.ffn.lanes = d->step.lanes;
      step.ffn.laneRows = step.buffers.laneRows;
    }
  }
  if (begins && !chunks.front().decode) steps_.readAhead(chunks);
  return addPrefillLayers(std::span(steps), *this, layers);
}

ops::PreparedInput Qwen4ExpTarget::addMix(PrefillStep &step, const Qwen4HyperWeights &hyper,
                                          metal::MetalBuffer output, const ops::Projection *next) const {
  const Qwen4Scratch &h = step.scratch;
  const ops::LinearScratch scratch = step.buffers.linearScratch;
  const ops::Linear &linear = this->linear(step);
  const ops::HyperShape hs = hyperShape(geometry_);
  ops::HyperConnection::addNorm(step.graph, h.streams, hyper.norm, hyper.inject, h.normalized, h.weights, hs,
                                step.rows);
  linear.add(step.graph, {.input = h.normalized, .output = h.low, .scratch = scratch}, hyper.down,
             linear.plan(hyper.down, step.rows));
  const ops::LinearPlan upPlan = linear.plan(hyper.up, step.rows);
  ops::PreparedInput prepared = ops::HyperConnection::addLow(step.graph, h.low, hs, step.rows, scratch, upPlan.input());
  linear.add(step.graph, {.input = h.low, .output = h.gate, .scratch = scratch, .prepared = std::move(prepared)},
             hyper.up, upPlan);
  return ops::HyperConnection::addMix(step.graph, h.normalized, h.gate, std::move(output), hs, step.rows, scratch,
                                      next ? linear.plan(*next, step.rows).input() : ops::LinearInput::Plain);
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
    const ops::Linear &linear = this->linear(step);
    addPleGather(step.graph, h, rows * geometry_.pleHeads());
    linear.addProjection(step.graph, h.pleRows, ple.key, h.gate, rows, b.linearScratch);
    linear.addProjection(step.graph, h.pleRows, ple.value, h.pleValue, rows, b.linearScratch);
    ops::NgramEmbedding::addGate(step.graph, h.gate, h.pleValue, h.streams, ple.keyNorm, ple.queryNorm,
                                 ple.convolutionNorm, h.pleGated, h.pleConvolution, ps, rows);
    // Every sequence's rows after its history in one dispatch, by the step's
    // layout: a prompt sequence's in order, writing its history after them
    // (and a state captured inside the command the history after its row),
    // a decode lane's each along its path of the lane's tree, whose commit
    // writes the history after the path it accepts.
    std::vector<metal::MetalBuffer> historyIn, historyOut;
    ops::LayoutRows layoutRows;
    metal::MetalBuffer tree;
    for (const QwenTargetPrefillSequence &sequence : step.sequences) {
      historyIn.push_back(sequence.auxiliaryIn);
      if (!step.decode) historyOut.push_back(sequence.auxiliaryOut);
    }
    if (const QwenTargetDecodeBuffers *d = step.decode) {
      layoutRows = {d->step.lanes, d->step.rows, d->step.laneCap};
      tree = d->tree;
    } else {
      layoutRows = promptRows(step.sequences, rows);
    }
    ops::NgramEmbedding::addConvolution(step.graph, u16Rows(h.pleConvolution, 0, rows, ps.width()), historyIn,
                                        historyOut, ple.convolution, f32Rows(h.pleGated, 0, rows, ps.width()),
                                        f32Rows(h.streams, 0, rows, ps.width()), ps, b.laneRows, layoutRows, tree);
    for (const QwenTargetPrefillSequence &sequence : step.sequences)
      for (const QwenTargetStateCapture &capture : sequence.stateCaptures)
        addCaptureHistory(step.graph, h, sequence.rowBegin, sequence.auxiliaryIn, capture);
  }
  std::visit(
      [&](const auto &mixer) {
        const ops::PreparedInput input = addMix(step, layer.mixerHyper, b.normalized, &mixer.inputProjection);
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(mixer)>, QwenGdnWeights>)
          addPrefillMixer(step, mixer, input, {h.branch, {}});
        else
          addPrefillMixer(step, mixer, input, {h.branch, {}},
                          step.decode ? addDecodeQsa(step, *layer.indexer) : addPrefillQsa(step, *layer.indexer));
      },
      layer.mixer);
  ops::HyperConnection::addCombine(step.graph, h.streams, h.branch, h.weights, hyperShape(geometry_), rows);
  addMix(step, layer.ffnHyper, b.normalized);
  ops::MoE::addRouting(step.graph, step.ffn, layer.ffn, *step.moe);
}

const ops::BlockMoeWeights *Qwen4ExpTarget::moe(uint32_t index) const { return &weights_.layers[index].ffn; }

void Qwen4ExpTarget::addPrefillExperts(PrefillStep &step, uint32_t index) const {
  const Qwen4Scratch &h = step.scratch;
  ops::MoE::addRouted(step.graph, step.ffn, weights_.layers[index].ffn, *step.moe);
  ops::HyperConnection::addCombine(step.graph, h.streams, h.branch, h.weights, hyperShape(geometry_), step.rows);
  // The draft reads a decode step's every row of the layers it captures.
  if (step.decode) {
    if (const auto slot = geometry_.captureSlot(index))
      addStreamCapture(step.graph, geometry_, *slot, h.streams, step.decode->capturedTargetHidden, 0, step.rows);
    return;
  }
  forEachPrefillCapture(step, index, [&](uint32_t slot, uint32_t source, uint32_t destination, uint32_t rows) {
    addStreamCapture(step.graph, geometry_, slot, rowsOf<float>(h.streams, source, rows, geometry_.streamWidth()),
                     step.buffers.captured, destination, rows);
  });
}

// A decode step's head reads every row of the final mix, which takes the
// table the head's plan reads where it runs in one pass (addHead).
metal::MetalBuffer Qwen4ExpTarget::addPrefillEnd(PrefillStep &step) const {
  if (const QwenTargetDecodeBuffers *d = step.decode) {
    const ops::Projection *head = step.rows <= ops::kMaximumDecodeTileRows ? &vocabularyProjection() : nullptr;
    addHead(step.graph, linear(step), d->finalHidden, d->logits, step.rows, step.buffers.linearScratch,
            addMix(step, weights_.finalHyper, d->finalHidden, head));
    return d->finalHidden;
  }
  addMix(step, weights_.finalHyper, step.buffers.hidden[1]);
  return step.buffers.hidden[1];
}

// The indexer projects every sequence's rows in one dispatch, by the chunk's
// layout; then each sequence's QSA is a dispatch of one: it pools the blocks
// its rows complete and keeps the keys of the block they leave open, and
// those of any state captured inside them; a sequence whose rows attend
// sparsely then selects their blocks, its lists after those of the sequences
// before it.
std::vector<ops::AttentionBlocks> Qwen4ExpTarget::addPrefillQsa(PrefillStep &step,
                                                                const Qwen4IndexerWeights &indexer) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const Qwen4Scratch &h = step.scratch;
  const ops::Linear &linear = this->linear(step);
  const uint32_t layer = step.attentionLayer, topBlocks = weights_.layout.indexerTopBlocks;
  const uint32_t keyWidth = geometry_.indexerHeadDimension, queryWidth = geometry_.indexerHeads * keyWidth;
  const auto u16Rows = [&](const metal::MetalBuffer &buffer, const QwenTargetPrefillSequence &sequence,
                           uint32_t width) {
    return rowsOf<uint16_t>(buffer, sequence.rowBegin, sequence.rows, width);
  };
  // A sequence's rows are the whole of the views of them each dispatch takes.
  const auto qsaRows = [&](const QwenTargetPrefillSequence &sequence) {
    return ops::IndexerSequence{sequence.rows,
                                static_cast<uint32_t>(sequence.chunk.committed_tokens),
                                sequence.indexTable,
                                step.indexOffsets[layer],
                                topBlocks,
                                h.maskWords,
                                h.scoreStride};
  };
  addIndexerProjections(step.graph, linear, b.normalized, indexer, h.indexerQuery, h.indexerKey, step.rows);
  ops::BlockIndexer::addProject(step.graph, rowsOf<float>(h.indexerQuery, 0, step.rows, queryWidth),
                                rowsOf<float>(h.indexerKey, 0, step.rows, keyWidth), indexer.queryNorm,
                                rowsOf<float>(b.ropeCos, 0, step.rows, geometry_.rotaryPairs),
                                rowsOf<float>(b.ropeSin, 0, step.rows, geometry_.rotaryPairs),
                                rowsOf<uint16_t>(h.qsaQueries, 0, step.rows, queryWidth),
                                rowsOf<uint16_t>(h.qsaKeys, 0, step.rows, keyWidth), b.laneRows,
                                promptRows(step.sequences, step.rows));
  std::vector<ops::AttentionBlocks> blocks(step.sequences.size());
  std::vector<ops::BlockListRows> sparseRows;
  uint32_t lists = 0;
  for (size_t index = 0; index < step.sequences.size(); ++index) {
    const QwenTargetPrefillSequence &sequence = step.sequences[index];
    const metal::MetalBuffer keys = u16Rows(h.qsaKeys, sequence, keyWidth);
    const metal::MetalBuffer pending = steps_.qsaPending(sequence.auxiliaryIn, layer);
    const ops::IndexerSequence rows = qsaRows(sequence);
    ops::BlockIndexer::addPool(step.graph, {step.blocks[index].records, step.blocks[index].count, keys, pending},
                               indexer.keyNorm, b.ropeInverseFrequencies, geometry_.ropeScaling.attentionFactor(),
                               rows);
    ops::BlockIndexer::addPending(step.graph, keys, pending, steps_.qsaPending(sequence.auxiliaryOut, layer),
                                  rows.firstPosition, sequence.rows);
    for (const QwenTargetStateCapture &capture : sequence.stateCaptures)
      ops::BlockIndexer::addPending(step.graph, keys, pending, steps_.qsaPending(capture.auxiliary, layer),
                                    rows.firstPosition, capture.row);
    if (!ops::BlockIndexer::sparse(rows.firstPosition, sequence.rows, topBlocks)) continue;
    ops::BlockIndexer::addSelect(step.graph, u16Rows(h.qsaQueries, sequence, queryWidth), h.qsaScores, h.scoreRows,
                                 rowsOf<uint32_t>(h.qsaMask, sequence.rowBegin, sequence.rows, h.maskWords), rows);
    blocks[index] = {h.qsaLists, h.qsaListCounts, h.listStride, lists};
    sparseRows.push_back({static_cast<uint32_t>(index), sequence.rows, rows.firstPosition, lists});
    lists += step.attention->sequences()[index].tiles();
  }
  if (!sparseRows.empty())
    ops::PagedAttention::addBlockLists(step.graph, h.qsaMask, h.maskWords, sparseRows, {b.laneRows, step.rows},
                                       {h.qsaLists, h.qsaListCounts, h.listStride, 0});
  return blocks;
}

void Qwen4ExpTarget::addCaptureHistory(metal::CommandGraph &graph, const Qwen4Scratch &scratch, uint32_t rowBegin,
                                       metal::MetalBuffer auxiliaryIn, const QwenTargetStateCapture &capture) const {
  const ops::NgramShape ps = ngramShape(geometry_);
  ops::NgramEmbedding::addHistory(graph, rowsOf<uint16_t>(scratch.pleConvolution, rowBegin, capture.row, ps.width()),
                                  std::move(auxiliaryIn), capture.auxiliary, ps, capture.row);
}

void Qwen4ExpTarget::addInsideCapture(metal::CommandGraph &graph, metal::MetalBuffer auxiliaryIn,
                                      metal::MetalBuffer captured) const {
  addCaptureHistory(graph, steps_.prefillScratch(0), 0, std::move(auxiliaryIn), {.row = 1, .auxiliary = captured});
}

void Qwen4ExpTarget::prepareVerifyLane(uint32_t lane, uint64_t position, const QwenTargetRequest &request) {
  steps_.prepareVerifyLane(lane, position, request);
}

void Qwen4ExpTarget::addVerifyInput(metal::CommandGraph &graph, const QwenVerifyInput &input) const {
  steps_.addVerifyInput(graph, input);
}

// Every lane's QSA in one dispatch of each kernel, over its rows as the
// step lays them out on the device: a row that completes a block keeps the
// block, pooled from its path, as its block key, and a lane's rows select
// from the blocks complete before its anchor and their paths' block keys.
// Once a lane's rows attend sparsely, every lane's attends through its
// list: a lane's rows that keep every block select them all.
std::vector<ops::AttentionBlocks> Qwen4ExpTarget::addDecodeQsa(PrefillStep &step,
                                                               const Qwen4IndexerWeights &indexer) const {
  const QwenTargetPrefillBuffers &b = step.buffers;
  const QwenTargetDecodeBuffers &d = *step.decode;
  const Qwen4Scratch &h = step.scratch;
  const ops::Linear &linear = this->linear(step);
  const uint32_t layer = step.attentionLayer, topBlocks = weights_.layout.indexerTopBlocks;
  const uint32_t keyWidth = geometry_.indexerHeadDimension, queryWidth = geometry_.indexerHeads * keyWidth;
  const uint32_t lanes = d.step.lanes, laneCap = d.step.laneCap;
  addIndexerProjections(step.graph, linear, b.normalized, indexer, h.indexerQuery, h.indexerKey, step.rows);
  std::array<uint32_t, ExecutionLimits::maximumBatchWidth> firstPositions{};
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> indexTables;
  bool sparse = false;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    firstPositions[lane] = step.chunks[lane].committed_tokens;
    indexTables[lane] = step.sequences[lane].indexTable;
    sparse = sparse || ops::BlockIndexer::sparse(firstPositions[lane], laneCap, topBlocks);
  }
  const ops::IndexerTreeRows tree{b.laneRows,
                                  d.tree,
                                  step.rows,
                                  laneCap,
                                  std::span(firstPositions).first(lanes),
                                  {std::span(indexTables).first(lanes), step.indexOffsets[layer]},
                                  topBlocks,
                                  h.maskWords,
                                  h.scoreStride};
  const metal::MetalBuffer queries = rowsOf<uint16_t>(h.qsaQueries, 0, step.rows, queryWidth);
  // The layer's keys and block keys, which the commit reads after the last
  // layer, and each lane's pending keys before the step.
  const metal::MetalBuffer keys = rowsOf<uint16_t>(h.qsaKeys, layer * h.keyLayerRows, step.rows, keyWidth);
  const metal::MetalBuffer blockKeys = rowsOf<uint16_t>(h.qsaBlockKeys, layer * h.keyLayerRows, step.rows, keyWidth);
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> pending;
  for (uint32_t lane = 0; lane < lanes; ++lane) pending[lane] = steps_.qsaPending(step.sequences[lane].auxiliaryIn, layer);
  ops::BlockIndexer::addProject(step.graph, rowsOf<float>(h.indexerQuery, 0, step.rows, queryWidth),
                                rowsOf<float>(h.indexerKey, 0, step.rows, keyWidth), indexer.queryNorm,
                                rowsOf<float>(b.ropeCos, 0, step.rows, geometry_.rotaryPairs),
                                rowsOf<float>(b.ropeSin, 0, step.rows, geometry_.rotaryPairs), queries, keys,
                                b.laneRows, {lanes, step.rows, laneCap});
  ops::BlockIndexer::addTreePool(step.graph,
                                 {step.laneBlocks.records, std::span(step.laneBlocks.counts).first(lanes),
                                  step.laneBlocks.laneRecords, keys, std::span(pending).first(lanes)},
                                 blockKeys, indexer.keyNorm, b.ropeInverseFrequencies,
                                 geometry_.ropeScaling.attentionFactor(), tree);
  if (!sparse) return std::vector<ops::AttentionBlocks>(lanes, {h.qsaLists, h.qsaListCounts, 0, 0});
  const metal::MetalBuffer rowMasks = rowsOf<uint32_t>(h.qsaRowMasks, 0, step.rows, 1);
  const ops::AttentionBlocks lists{h.qsaLists, h.qsaListCounts, h.listStride, 0, {}, {}, rowMasks};
  ops::BlockIndexer::addTreeSelect(step.graph, queries, blockKeys, h.qsaScores,
                                   rowsOf<uint32_t>(h.qsaMask, 0, step.rows, h.maskWords), rowMasks, tree);
  // A lane's tiles at the lane cap take their lists after the lanes before
  // it, as a prompt chunk's sequences' do, a tile past its rows an empty one.
  std::array<ops::BlockListRows, ExecutionLimits::maximumBatchWidth> laneLists{};
  std::vector<ops::AttentionBlocks> blocks(lanes, lists);
  for (uint32_t lane = 0, list = 0; lane < lanes; ++lane) {
    laneLists[lane] = {lane, laneCap, firstPositions[lane], list};
    blocks[lane].list = list;
    list += step.attention->sequences()[lane].tiles();
  }
  ops::PagedAttention::addBlockLists(step.graph, h.qsaMask, h.maskWords, std::span(laneLists).first(lanes),
                                     {b.laneRows, step.rows}, lists);
  return blocks;
}

void Qwen4ExpTarget::addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden, metal::MetalBuffer,
                                  metal::MetalBuffer logits, uint32_t rows, ops::LinearScratch scratch) const {
  addHead(graph, operators_.linear(), std::move(hidden), std::move(logits), rows, scratch, {});
}

void Qwen4ExpTarget::addStateCommit(metal::CommandGraph &graph, QwenTargetCommitBuffers buffers,
                                    const ops::StepRows &step) const {
  QwenTarget::addStateCommit(graph, buffers, step);
  const uint32_t lanes = step.lanes;
  // Every lane's PLE history after its accepted path in one dispatch, from
  // the convolution inputs its verify wrote of its rows, by the step's
  // layout.
  const ops::NgramShape shape = ngramShape(geometry_);
  const ops::LayoutRows rows{lanes, step.rows, step.laneCap};
  const Qwen4Scratch scratch = steps_.verifyScratch(rows.rows);
  ops::NgramEmbedding::addCommit(graph, rowsOf<uint16_t>(scratch.pleConvolution, 0, rows.rows, shape.width()),
                                 std::span(buffers.auxiliaryIn).first(lanes),
                                 std::span(buffers.auxiliaryOut).first(lanes), buffers.retainedRows, shape,
                                 buffers.laneRows, rows);
  // Each lane's blocks its accepted path completes into the index, and its
  // QSA pending keys after the path.
  const uint32_t layers = geometry_.kvLayout.attentionLayers;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth> in, out;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    in[lane] = steps_.qsaPending(buffers.auxiliaryIn[lane], 0, layers);
    out[lane] = steps_.qsaPending(buffers.auxiliaryOut[lane], 0, layers);
  }
  const auto positions = steps_.verifyPositions(lanes);
  ops::BlockIndexer::addCommit(graph, {scratch.qsaKeys, scratch.qsaBlockKeys, layers, scratch.keyLayerRows,
                                       buffers.retainedRows, buffers.laneRows, std::span(positions).first(lanes),
                                       std::span(in).first(lanes), std::span(out).first(lanes),
                                       std::span(buffers.indexTables).first(lanes), buffers.indexLayerBytes});
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
                                     ops::StoredRows(scratch.pleStaging, weights_.pleFormat)};
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

QwenTargetScratchBytes qwenTargetScratchBytes(const Qwen4ExpLayout &, const QwenTargetGeometry &geometry,
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

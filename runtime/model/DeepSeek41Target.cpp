#include "model/DeepSeek41Target.hpp"

#include "metal/abi/LatentAttention.h"
#include "metal/abi/LatentIndex.h"
#include "model/EngramHash.hpp"
#include "model/RuntimeArenas.hpp"
#include "ops/Embedding.hpp"
#include "ops/Engram.hpp"
#include "ops/LatentAttention.hpp"
#include "ops/LatentIndex.hpp"
#include "ops/LatentRotation.hpp"
#include "ops/PageStorage.hpp"
#include "ops/RowCopy.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace splash::model {
namespace {

// The rows a decode step verifies of each of its lanes, and the lanes.
constexpr uint32_t kVerifyRows = ExecutionLimits::targetVerifyRows;
constexpr uint32_t kLanes = ExecutionLimits::maximumBatchWidth;
// The rows of a tile of a prefill chunk's layers' work: whole tiles of the
// prefill projections (Linear::ggufBaseline), so that a tile's projections
// read and write inside the chunk's buffers.
constexpr uint32_t kLayerTile = 256;
static_assert(kPrefillRows % kLayerTile == 0, "a prefill chunk is whole tiles");
// The rows of a prefill's indexer layer between host steps. Metal fails a
// wait still unmet 5 s after its queue reached it (MetalBackend.hpp), and an
// indexer's scores of every group grow with the context: at 60K tokens a
// layer's attention of a command's 8192 rows took 2.5 s on an M4 Pro, and
// past 5 s while it was hot. Two tiles take a sixteenth of that, and a
// command's 16 steps a layer keep its buffers within the queue's.
constexpr uint32_t kIndexerStepRows = 2 * kLayerTile;
static_assert(kPrefillRows % kIndexerStepRows == 0, "a prefill chunk is whole indexer steps");
constexpr uint32_t kLatent = LATENT_ATTENTION_DIM;

template <class T> T *contents(const metal::MetalBuffer &buffer) { return static_cast<T *>(buffer.contents()); }

// A float projection (BF16 or F32 as the GGUF stores it) into fp32 rows, on
// the float tile a router runs on.
void addFloatProjection(metal::CommandGraph &graph, const metal::MetalBuffer &input, const ops::Projection &projection,
                        const metal::MetalBuffer &output, uint32_t rows) {
  ops::addGgufFloat(graph, input, projection.blocks().segments.front(), output, rows, projection.outputSize, 0,
                    ops::FloatOutput::Float32, ops::FloatTile::Simdgroup);
}

} // namespace

// A step's rows of one sequence (a prefill's) or lane (a verify's): from row
// rowBegin of the step, positions committed on, its pages and its state
// before the step and, a prefill's, after its rows and at its captures.
struct DeepSeek41Target::Rows {
  uint32_t rowBegin = 0;
  uint32_t rows = 0;
  uint32_t committed = 0;
  uint32_t lane = 0;
  metal::MetalBuffer pageTable;
  metal::MetalBuffer stateIn;
  metal::MetalBuffer stateOut;
  std::span<const QwenTargetStateCapture> captures;
  // The positions of its groups from its first on (ratio * group), of a
  // paired source's and a single one's.
  std::array<metal::MetalBuffer, 2> groupPositions;
  // A drafted verify's rows each lane verifies (a uint32 per lane), past
  // which its rows attend nothing; none for a prefill's.
  metal::MetalBuffer verifyRows;
};

// A step's scratch: what lives across a chunk's layers, its own; a layer's
// work, which a prefill's chunks share; and a verify's latents of every
// layer, its paired sources' projections of each lane and its lanes' prior
// tokens, which live until its commit.
struct DeepSeek41Target::Scratch {
  uint32_t rows = 0;
  metal::MetalBuffer streams, attentionMixes, ffnMixes, selected, kept, positions, groupPositions, ffnOut;
  metal::MetalBuffer engramIds, engramSlots, engramStaging;
  metal::MetalBuffer mixChunks, branch, latentsRaw, latents, compressKv, compressGate, pooled, groupNormalized,
      compressed, indexKeysFloat, indexKeys;
  metal::MetalBuffer qa, qr, q, qRotated, indexQueries, indexRotated, indexWeights, indexScores, blockScores;
  metal::MetalBuffer keyIndices, keyCounts, scores, parts, out, rotatedBack, groupInput, groupOutput, joined;
  metal::MetalBuffer engramRows, engramKv;
  metal::MetalBuffer layerLatents, sourceKv, sourceGate, prior;
};

namespace {

// The buffers of a scratch of `rows` rows in tiles of `tile` (a verify's one
// tile), through `allocate(bytes, label)`: with `own`, what lives across a
// chunk's layers; with `work`, a layer's work; with `verify`, what lives
// until a verify's commit.
template <class Scratch, class Allocate>
void layoutScratch(const DeepSeek41Layout &layout, const ops::ManifoldHyperShape &mhc, uint64_t tableRowBytes,
                   Scratch &s, uint32_t rows, uint32_t tile, bool own, bool work, bool verify, Allocate allocate) {
  const DeepSeek41Dimensions &d = layout.deepseek;
  const uint64_t hidden = layout.hiddenSize, width = layout.attentionWidth, heads = layout.attentionQueryHeads;
  const uint64_t keyStride = layout.slidingWindow + d.indexerTopGroups;
  const uint64_t engramIds =
      uint64_t{static_cast<uint32_t>(std::popcount(d.engramLayers))} * rows * (d.engramNgram - 1) * d.engramHeads;
  s.rows = rows;
  if (own) {
    s.streams = allocate(rows * MHC_STREAMS * hidden * 4, "deepseek41/streams");
    s.attentionMixes = allocate(uint64_t{rows} * MHC_MIXES * 4, "deepseek41/mixes");
    s.ffnMixes = allocate(uint64_t{rows} * MHC_MIXES * 4, "deepseek41/mixes");
    s.selected = allocate(uint64_t{rows} * d.indexerTopGroups * 4, "deepseek41/selected");
    s.kept = allocate(uint64_t{rows} * d.candidateTopBlocks * 4, "deepseek41/kept");
    s.positions = allocate(uint64_t{rows} * 4, "deepseek41/positions");
    s.groupPositions = allocate(uint64_t{kLanes} * 2 * (rows + 1) * 4, "deepseek41/group-positions");
    s.ffnOut = allocate(rows * hidden * 2, "deepseek41/ffn-out");
    s.engramIds = allocate(engramIds * 4, "deepseek41/engram-ids");
    s.engramSlots = allocate(engramIds * 4, "deepseek41/engram-slots");
    s.engramStaging = allocate(engramIds * std::max<uint64_t>(tableRowBytes, 1), "deepseek41/engram-staging");
  }
  if (work) {
    s.mixChunks = allocate(ops::ManifoldHyperConnection::mixScratchBytes(mhc, rows), "deepseek41/mix");
    s.branch = allocate(rows * hidden * 2, "deepseek41/branch");
    s.latentsRaw = allocate(uint64_t{rows} * kLatent * 2, "deepseek41/kv");
    s.latents = allocate(uint64_t{rows} * kLatent * 2, "deepseek41/latents");
    s.compressKv = allocate(uint64_t{rows + 1} * kLatent * 4, "deepseek41/compress-kv");
    s.compressGate = allocate(uint64_t{rows + 1} * kLatent * 4, "deepseek41/compress-gate");
    s.pooled = allocate(uint64_t{rows + 1} * kLatent * 4, "deepseek41/pooled");
    s.groupNormalized = allocate(uint64_t{rows + 1} * kLatent * 2, "deepseek41/group-normalized");
    s.compressed = allocate(uint64_t{rows + 1} * kLatent * 2, "deepseek41/compressed");
    s.indexKeysFloat = allocate(uint64_t{rows + 1} * LATENT_INDEX_DIM * 4, "deepseek41/index-keys");
    s.indexKeys = allocate(uint64_t{rows + 1} * LATENT_INDEX_DIM * 2, "deepseek41/index-keys");
    s.qa = allocate(uint64_t{tile} * d.queryRank * 2, "deepseek41/qa");
    s.qr = allocate(uint64_t{tile} * d.queryRank * 2, "deepseek41/qr");
    s.q = allocate(tile * width * 2, "deepseek41/q");
    s.qRotated = allocate(tile * width * 2, "deepseek41/q-rotated");
    const uint64_t indexWidth = uint64_t{d.indexerHeads} * d.indexerHeadDimension;
    s.indexQueries = allocate(tile * indexWidth * 4, "deepseek41/index-q");
    s.indexRotated = allocate(tile * indexWidth * 2, "deepseek41/index-q-rotated");
    s.indexWeights = allocate(uint64_t{tile} * d.indexerHeads * 4, "deepseek41/index-weights");
    // An indexer's scores and their blocks of a piece of a tile's rows at
    // once (ops::LatentIndex::pieceRows).
    s.indexScores = allocate(ops::LatentIndex::kScoreBytes, "deepseek41/index-scores");
    s.blockScores = allocate(ops::LatentIndex::blockBytes(d.candidateBlockGroups, tile), "deepseek41/block-scores");
    s.keyIndices = allocate(tile * keyStride * 4, "deepseek41/keys");
    s.keyCounts = allocate(uint64_t{tile} * 4, "deepseek41/key-counts");
    s.scores = allocate(tile * heads * keyStride * 4, "deepseek41/scores");
    // A values pass over few rows splits their keys into at most
    // LATENT_ATTENTION_MAX_SPLITS parts of all of them (LatentAttention::splits).
    s.parts = allocate(uint64_t{LATENT_ATTENTION_MAX_SPLITS} * heads * kLatent * 4, "deepseek41/parts");
    s.out = allocate(tile * heads * kLatent * 4, "deepseek41/attention-out");
    s.rotatedBack = allocate(tile * width * 2, "deepseek41/rotated-back");
    s.groupInput = allocate(uint64_t{tile} * d.groupWidth(layout.attentionWidth) * 2, "deepseek41/group-input");
    s.groupOutput = allocate(uint64_t{tile} * d.outputRank * 2, "deepseek41/group-output");
    s.joined = allocate(uint64_t{tile} * d.outputGroups * d.outputRank * 2, "deepseek41/joined");
    s.engramRows = allocate(uint64_t{tile} * d.engramWidth() * 2, "deepseek41/engram-rows");
    s.engramKv = allocate(tile * (MHC_STREAMS + 1) * hidden * 2, "deepseek41/engram-kv");
  }
  if (verify) {
    const uint64_t paired = static_cast<uint64_t>(std::popcount(d.compressorLayers & d.pairedLayers));
    s.layerLatents = allocate(uint64_t{layout.layers} * rows * kLatent * 2, "deepseek41/layer-latents");
    s.sourceKv = allocate(paired * kLanes * (kVerifyRows + 1) * kLatent * 4, "deepseek41/source-kv");
    s.sourceGate = allocate(paired * kLanes * (kVerifyRows + 1) * kLatent * 4, "deepseek41/source-gate");
    s.prior = allocate(uint64_t{kLanes} * (d.engramNgram - 1) * 4, "deepseek41/prior");
  }
}

// The fixed buffers beside the scratch: the RoPE frequencies, the first
// mix's weights and the MoE's zeros of a chunk's rows.
uint64_t fixedBytes(const DeepSeek41Layout &layout) {
  return 2 * uint64_t{layout.rotaryPairs} * 4 + uint64_t{kPrefillRows} * MHC_MIXES * 4 +
         uint64_t{kPrefillRows} * layout.hiddenSize * 2;
}

} // namespace

QwenTargetScratchBytes DeepSeek41Target::scratchBytes(const DeepSeek41Layout &layout,
                                                      const ops::ManifoldHyperShape &mhc, uint32_t prefillChunks,
                                                      uint64_t tableRowBytes) {
  QwenTargetScratchBytes bytes{.prefill = fixedBytes(layout)};
  const auto counter = [](uint64_t &total) {
    return [&total](uint64_t size, const char *) {
      total += std::max<uint64_t>(size, 16);
      return metal::MetalBuffer{};
    };
  };
  for (uint32_t chunk = 0; chunk < std::max<uint32_t>(prefillChunks, 1); ++chunk) {
    Scratch s;
    layoutScratch(layout, mhc, tableRowBytes, s, kPrefillRows, kLayerTile, true, chunk == 0, false,
                  counter(bytes.prefill));
  }
  Scratch s;
  layoutScratch(layout, mhc, tableRowBytes, s, kLanes * kVerifyRows, kLanes * kVerifyRows, true, true, true,
                counter(bytes.decode));
  return bytes;
}

DeepSeek41Target::DeepSeek41Target(const DeepSeek41Weights &weights, const QwenTargetGeometry &geometry,
                                   metal::MetalBackend &backend, const ops::ExecutionPlans &operators,
                                   uint32_t prefillChunks)
    : QwenTarget(weights, geometry, backend, operators), weights_(weights), d_(weights.layout.deepseek),
      prefillChunks_(std::max<uint32_t>(prefillChunks, 1)) {
  const DeepSeek41Layout &layout = weights.layout;
  if (weights.layers.size() != layout.layers) throw std::invalid_argument("DeepSeek-V4.1 layers mismatch");
  // Its Engram tables only stream (their store's rows staged by each step),
  // which the memory plan always gives memory (StreamedPart::streamsOnly).
  if (!weights.streamedTable || weights.engramFirstRows.size() != static_cast<size_t>(std::popcount(d_.engramLayers)))
    throw std::invalid_argument("DeepSeek-V4.1 runs with its Engram tables streaming, which these weights do not");
  mhc_ = {layout.hiddenSize, d_.sinkhornIterations, d_.hcEpsilon, d_.normEpsilon};
  moeShape_ = geometry_.moeShape();
  for (uint32_t layer = 0; layer < layout.layers; ++layer)
    if (DeepSeek41Dimensions::has(d_.compressorLayers, layer)) {
      pairedOf_.push_back(DeepSeek41Dimensions::has(d_.pairedLayers, layer)
                              ? static_cast<uint32_t>(std::popcount(d_.compressorLayers & d_.pairedLayers &
                                                                    ((uint64_t{1} << layer) - 1)))
                              : UINT32_MAX);
      sources_.push_back(layer);
    }
  if (sources_.size() != geometry_.kvLayout.attentionLayers)
    throw std::invalid_argument("DeepSeek-V4.1 KV sources do not match its pages");
  const auto allocate = [&](uint64_t &total) {
    return [&](uint64_t bytes, const char *label) {
      total += std::max<uint64_t>(bytes, 16);
      return backend_.allocateBuffer(std::max<uint64_t>(bytes, 16), metal::BufferStorage::Shared, label);
    };
  };
  const auto upload = [&](const auto &values, const char *label) {
    using T = typename std::decay_t<decltype(values)>::value_type;
    metal::MetalBuffer buffer = allocate(allocatedPrefill_)(values.size() * sizeof(T), label);
    std::memcpy(buffer.contents(), values.data(), values.size() * sizeof(T));
    return buffer;
  };
  frequencies_ = {upload(deepSeek41RopeFrequencies(layout, false), "deepseek41/rope"),
                  upload(deepSeek41RopeFrequencies(layout, true), "deepseek41/rope-yarn")};
  std::vector<float> first(uint64_t{kPrefillRows} * MHC_MIXES, 0.0F);
  for (uint32_t row = 0; row < kPrefillRows; ++row) first[uint64_t{row} * MHC_MIXES] = 1.0F;
  firstMixes_ = upload(first, "deepseek41/first-mixes");
  zeros_ = upload(std::vector<uint16_t>(uint64_t{kPrefillRows} * layout.hiddenSize, 0), "deepseek41/zeros");
  for (const DeepSeek41LayerWeights &layer : weights.layers)
    if (layer.engram) engramGates_.push_back(layer.engram->gate);
  const uint64_t rowBytes = weights_.streamedTable->table().rowBytes;
  for (uint32_t chunk = 0; chunk < prefillChunks_; ++chunk) {
    Scratch &s = prefill_.emplace_back();
    if (chunk) s = prefill_.front();
    layoutScratch(layout, mhc_, rowBytes, s, kPrefillRows, kLayerTile, true, chunk == 0, false,
                  allocate(allocatedPrefill_));
  }
  verify_ = std::make_unique<Scratch>();
  layoutScratch(layout, mhc_, rowBytes, *verify_, kLanes * kVerifyRows, kLanes * kVerifyRows, true, true, true,
                allocate(allocatedDecode_));
}

DeepSeek41Target::~DeepSeek41Target() = default;

QwenTargetScratchBytes DeepSeek41Target::allocatedScratchBytes() const noexcept {
  return {allocatedPrefill_, allocatedDecode_};
}

uint32_t DeepSeek41Target::sourceOf(uint32_t index) const {
  uint32_t source = 0;
  for (uint32_t s = 0; s < sources_.size(); ++s)
    if (sources_[s] <= index) source = s;
  return source;
}

uint32_t DeepSeek41Target::ratioOf(uint32_t source) const {
  return DeepSeek41Dimensions::has(d_.pairedLayers, sources_[source]) ? 2 : 1;
}

metal::MetalBuffer DeepSeek41Target::ring(const metal::MetalBuffer &state, uint32_t layer) const {
  const uint64_t bytes = uint64_t{weights_.layout.slidingWindow} * kLatent * 2;
  return backend_.view(state, layer * bytes, bytes);
}

metal::MetalBuffer DeepSeek41Target::openGroup(const metal::MetalBuffer &state, uint32_t paired) const {
  const uint64_t bytes = uint64_t{2} * kLatent * sizeof(float);
  return backend_.view(state, weights_.layout.deepseekWindowBytes() + paired * bytes, bytes);
}

metal::MetalBuffer DeepSeek41Target::fromRow(const metal::MetalBuffer &buffer, uint32_t row, uint32_t width,
                                             uint32_t bytes) const {
  const uint64_t at = uint64_t{row} * width * bytes;
  return backend_.view(buffer, at, buffer.sizeBytes() - at);
}

void DeepSeek41Target::project(metal::CommandGraph &graph, metal::MetalBuffer input, const ops::Projection &projection,
                               metal::MetalBuffer output, uint32_t rows, bool decode, uint32_t lanes,
                               const ops::LinearScratch &linear) const {
  if (decode)
    addDecodeProjection(graph, std::move(input), projection, std::move(output), lanes, linear);
  else
    operators_.linear().addPrefill(graph, std::move(input), projection, std::move(output), rows, linear);
}

// Each tile of the step's rows: its Engram rows of the layer, gathered from
// the staging as a table of as many rows as the step staged
// (Qwen4ExpTarget::addPleGather), projected into a key per stream and a
// value, which the gate adds to the streams.
void DeepSeek41Target::addEngram(metal::CommandGraph &graph, const Scratch &s, const ops::LinearScratch &linear,
                                 bool decode, uint32_t lanes, uint32_t index, uint32_t rows, uint32_t staged) const {
  const DeepSeek41LayerWeights &layer = weights_.layers[index];
  const uint32_t table = static_cast<uint32_t>(std::popcount(d_.engramLayers & ((uint64_t{1} << index) - 1)));
  const uint32_t hidden = weights_.layout.hiddenSize, perToken = (d_.engramNgram - 1) * d_.engramHeads;
  const ops::EmbeddingWeights stagedRows{staged, d_.engramHeadDimension,
                                         ops::NativeRows(s.engramStaging, weights_.engramFormat)};
  const uint32_t tile = decode ? rows : kLayerTile;
  for (uint32_t begin = 0; begin < rows; begin += tile) {
    const uint32_t n = std::min(tile, rows - begin);
    ops::Embedding::add(graph, rowsOf<uint32_t>(s.engramSlots, table * rows + begin, n, perToken), stagedRows,
                        s.engramRows, n * perToken);
    project(graph, s.engramRows, layer.engram->kv, s.engramKv, n, decode, lanes, linear);
    ops::Engram::addGate(graph, rowsOf<float>(s.streams, begin, n, MHC_STREAMS * hidden), s.engramKv,
                         engramGates_[table], hidden, d_.normEpsilon, n);
  }
}

// A KV source's groups its rows complete: their latents and index keys into
// the sequence's pages, of the projections of its rows and, of a paired
// source, of its open group's first token before them; and the group left
// open after its rows and at each capture. A verify keeps a paired source's
// projections for its commit (addStateCommit), and writes no state.
void DeepSeek41Target::addSourceGroups(metal::CommandGraph &graph, const Scratch &s, uint32_t index,
                                       const Rows &sequence, const metal::MetalBuffer &normalized,
                                       std::span<const SplashKvLayer> kvLayers) const {
  const DeepSeek41CompressorWeights &compressor = *weights_.layers[index].compressor;
  const uint32_t source = sourceOf(index), ratio = ratioOf(source), pairs = weights_.layout.rotaryPairs;
  const uint32_t n = sequence.rows, committed = sequence.committed;
  const bool open = ratio > 1 && committed % ratio;
  const uint32_t first = committed / ratio, groups = (n + (open ? 1 : 0)) / ratio;
  const metal::MetalBuffer input = rowsOf<uint16_t>(normalized, sequence.rowBegin, n, weights_.layout.hiddenSize);
  const bool verify = &s == verify_.get();
  const uint32_t paired = pairedOf_[source];
  // The projections from row 1 on, the open group's first token's in row 0.
  metal::MetalBuffer kv = s.compressKv, gate = s.compressGate;
  if (verify && ratio > 1) {
    const uint64_t laneBytes = uint64_t{kVerifyRows + 1} * kLatent * sizeof(float);
    const uint64_t at = (uint64_t{paired} * kLanes + sequence.lane) * laneBytes;
    kv = backend_.view(s.sourceKv, at, laneBytes);
    gate = backend_.view(s.sourceGate, at, laneBytes);
  }
  addFloatProjection(graph, input, compressor.kv, rowsOf<float>(kv, 1, n, kLatent), n);
  if (ratio > 1) addFloatProjection(graph, input, *compressor.gate, rowsOf<float>(gate, 1, n, kLatent), n);
  // An fp32 row copies as a row of bf16 pairs.
  constexpr uint32_t row = 2 * kLatent;
  if (open) {
    const metal::MetalBuffer state = openGroup(sequence.stateIn, paired);
    ops::RowCopy::add(graph, state, {0, 2 * row, 0}, kv, {0, row, 0}, 1, row);
    ops::RowCopy::add(graph, state, {0, 2 * row, row}, gate, {0, row, 0}, 1, row);
  }
  const uint32_t start = open ? 0 : 1;
  if (groups) {
    const metal::MetalBuffer positions = sequence.groupPositions[ratio > 1 ? 0 : 1];
    metal::MetalBuffer pooled = s.pooled;
    if (ratio > 1)
      ops::LatentIndex::addCompress(graph, rowsOf<float>(kv, start, groups * ratio, kLatent),
                                    rowsOf<float>(gate, start, groups * ratio, kLatent), pooled, groups, kLatent, ratio,
                                    true);
    else
      pooled = rowsOf<float>(kv, start, groups, kLatent);
    const metal::MetalBuffer &freqs = frequencies_[1];
    ops::LatentRotation::add(graph, pooled, true, compressor.norm.buffer, freqs, positions, s.groupNormalized,
                             {1, kLatent, 0, false, ops::LatentRounding::None, true, d_.normEpsilon}, groups);
    ops::LatentRotation::add(graph, pooled, true, compressor.norm.buffer, freqs, positions, s.compressed,
                             {1, kLatent, pairs, false, ops::LatentRounding::Fp4E4m3Scale, true, d_.normEpsilon},
                             groups);
    addFloatProjection(graph, s.groupNormalized, compressor.indexKey, s.indexKeysFloat, groups);
    ops::LatentRotation::add(graph, s.indexKeysFloat, true, compressor.indexKeyNorm.buffer, freqs, positions,
                             s.indexKeys,
                             {1, d_.indexerHeadDimension, pairs, false, ops::LatentRounding::Fp4Pow2Scale, true,
                              d_.normEpsilon},
                             groups);
    ops::LatentIndex::addStore(graph, s.compressed, s.indexKeys, {sequence.pageTable, kvLayers[source]}, first,
                               groups, ratio);
  }
  if (ratio == 1 || verify) return;
  // The group open after `rows` of the rows: its first token's projections,
  // at an odd count with the open group's before them.
  const auto keepOpen = [&](uint32_t rows, const metal::MetalBuffer &state) {
    const uint32_t count = rows + (open ? 1 : 0);
    if (!(count % 2)) return;
    const metal::MetalBuffer target = openGroup(state, paired);
    const uint32_t last = start + count - 1;
    ops::RowCopy::add(graph, kv, {last, row, 0}, target, {0, 2 * row, 0}, 1, row);
    ops::RowCopy::add(graph, gate, {last, row, 0}, target, {0, 2 * row, row}, 1, row);
  };
  keepOpen(n, sequence.stateOut);
  for (const QwenTargetStateCapture &capture : sequence.captures) keepOpen(capture.row, capture.auxiliary);
}

// A sequence's rows [first, first + count) of the step (inside the tile from
// row `begin` on, whose query latents are in s.qr) through the indexer of
// layer `index`: each row's top groups of its KV source, among the
// candidate blocks after the candidate layer, into s.selected; in pieces of
// rows whose scores of every group fit the scratch.
void DeepSeek41Target::addIndexer(metal::CommandGraph &graph, const Scratch &s, uint32_t index, const Rows &sequence,
                                  uint32_t begin, uint32_t first, uint32_t count,
                                  const metal::MetalBuffer &normalized, std::span<const SplashKvLayer> kvLayers) const {
  const DeepSeek41IndexerWeights &indexer = *weights_.layers[index].indexer;
  const uint32_t source = sourceOf(index), ratio = ratioOf(source), pairs = weights_.layout.rotaryPairs;
  const uint32_t groups = (sequence.committed + sequence.rows) / ratio;
  const uint32_t heads = d_.indexerHeads, dim = d_.indexerHeadDimension, topK = d_.indexerTopGroups;
  const uint32_t block = d_.candidateBlockGroups, keptStride = d_.candidateTopBlocks;
  const uint32_t blocks = (groups + block - 1) / block;
  const uint32_t piece = ops::LatentIndex::pieceRows(groups, count);
  const float scale = 1.0F / std::sqrt(float(dim)) / std::sqrt(float(heads));
  for (uint32_t at = 0; at < count; at += piece) {
    const uint32_t m = std::min(piece, count - at), row = first + at;
    const metal::MetalBuffer positions = rowsOf<uint32_t>(s.positions, row, m, 1);
    addFloatProjection(graph, rowsOf<uint16_t>(s.qr, row - begin, m, d_.queryRank), indexer.query, s.indexQueries, m);
    ops::LatentRotation::add(graph, s.indexQueries, true, weights_.layers[index].queryNorm.buffer, frequencies_[1],
                             positions, s.indexRotated,
                             {heads, dim, pairs, false, ops::LatentRounding::Fp4Pow2Scale, false, 0.0F}, m);
    // The heads' weights, of fewer columns than a 64-row tile, on the float
    // tile alone, as a router's.
    ops::addGgufFloat(graph, rowsOf<uint16_t>(normalized, row, m, weights_.layout.hiddenSize),
                      indexer.weights.blocks().segments.front(), s.indexWeights, m, heads, 0,
                      ops::FloatOutput::Float32, ops::FloatTile::Simdgroup);
    ops::LatentIndex::addScores(graph, s.indexRotated, s.indexWeights, {sequence.pageTable, kvLayers[source]},
                                positions, s.indexScores, heads, groups, groups, ratio, scale, m);
    const metal::MetalBuffer kept = rowsOf<uint32_t>(s.kept, row, m, keptStride);
    if (index == d_.candidateLayer) {
      ops::LatentIndex::addBlocks(graph, s.indexScores, positions, s.blockScores, groups, groups, block, ratio, m);
      ops::LatentIndex::addSelect(graph, s.blockScores, kept, blocks, keptStride, blocks, keptStride, m);
    } else if (index > d_.candidateLayer &&
               (sequence.committed + sequence.rows) / ratioOf(sourceOf(d_.candidateLayer))) {
      // A later layer keeps to the blocks the candidate layer chose, which a
      // sequence too short for its groups left it none of.
      ops::LatentIndex::addMask(graph, s.indexScores, kept, groups, groups, block, keptStride, m);
    }
    ops::LatentIndex::addSelect(graph, s.indexScores, rowsOf<uint32_t>(s.selected, row, m, topK), groups, topK,
                                groups, topK, m);
  }
}

// A layer's attention of the step's rows: its mix and norm of the streams,
// every row's latent (the step's window latents), a KV source's groups into
// the pages; then by tiles of rows, the queries, and of each sequence's
// rows in the tile its indexer's selection, its key lists and its attention,
// the output's groups and projection into the branch; a prefill's rings
// after each sequence's rows and captures; and the branch into the streams.
void DeepSeek41Target::addAttention(metal::CommandGraph &graph, const Scratch &s, const ops::LinearScratch &linear,
                                    bool decode, uint32_t lanes, uint32_t index, std::span<const Rows> sequences,
                                    uint32_t rows, const metal::MetalBuffer &normalized,
                                    const metal::MetalBuffer &previousMixes,
                                    std::span<const SplashKvLayer> kvLayers) const {
  const DeepSeek41Layout &layout = weights_.layout;
  const DeepSeek41LayerWeights &layer = weights_.layers[index];
  const bool compressed = DeepSeek41Dimensions::has(d_.compressedLayers, index);
  const metal::MetalBuffer &freqs = frequencies_[compressed ? 1 : 0];
  const uint32_t hidden = layout.hiddenSize, heads = layout.attentionQueryHeads, width = layout.attentionWidth;
  const uint32_t pairs = layout.rotaryPairs, window = layout.slidingWindow, topK = d_.indexerTopGroups;
  const uint32_t keyStride = window + topK, source = sourceOf(index);
  const uint32_t groupWidth = d_.groupWidth(width), rank = d_.outputRank, joinedWidth = d_.outputGroups * rank;
  ops::ManifoldHyperConnection::addMix(graph, s.streams, layer.attentionMix.fn, layer.attentionMix.scale,
                                       layer.attentionMix.base, s.mixChunks, s.attentionMixes, mhc_, rows);
  ops::ManifoldHyperConnection::addPreNorm(graph, s.streams, previousMixes, layer.attentionNorm.buffer, normalized,
                                           mhc_, rows);
  // The rows' latents, normed, rotated at their positions and rounded to
  // FP8; a verify keeps every layer's for its commit.
  const metal::MetalBuffer latents =
      decode ? rowsOf<uint16_t>(s.layerLatents, index * s.rows, s.rows, kLatent) : s.latents;
  project(graph, normalized, layer.kv, s.latentsRaw, rows, decode, lanes, linear);
  ops::LatentRotation::add(graph, s.latentsRaw, false, layer.kvNorm.buffer, freqs, s.positions, latents,
                           {1, kLatent, pairs, false, ops::LatentRounding::Fp8, true, d_.normEpsilon}, rows);
  if (layer.compressor)
    for (const Rows &sequence : sequences) addSourceGroups(graph, s, index, sequence, normalized, kvLayers);
  const uint32_t tile = decode ? rows : kLayerTile;
  const float scale = 1.0F / std::sqrt(float(kLatent));
  for (uint32_t begin = 0; begin < rows; begin += tile) {
    const uint32_t n = std::min(tile, rows - begin);
    const metal::MetalBuffer positions = rowsOf<uint32_t>(s.positions, begin, n, 1);
    project(graph, decode ? normalized : fromRow(normalized, begin, hidden, 2), layer.queryA, s.qa, n, decode, lanes,
            linear);
    ops::LatentRotation::add(graph, s.qa, false, layer.queryNorm.buffer, freqs, positions, s.qr,
                             {1, d_.queryRank, 0, false, ops::LatentRounding::None, true, d_.normEpsilon}, n);
    project(graph, s.qr, layer.queryB, s.q, n, decode, lanes, linear);
    ops::LatentRotation::add(graph, s.q, false, layer.queryNorm.buffer, freqs, positions, s.qRotated,
                             {heads, kLatent, pairs, false, ops::LatentRounding::None, false, 0.0F}, n);
    for (const Rows &sequence : sequences) {
      // Without a draft a verify attends a lane's anchor alone, the one row
      // it retains: the rows after it repeat it, and no row it reads is
      // theirs (positions past its own).
      const uint32_t first = std::max(sequence.rowBegin, begin);
      const uint32_t end = decode && !geometry_.captureLayerCount
                               ? first + 1
                               : std::min(sequence.rowBegin + sequence.rows, begin + n);
      if (first >= end) continue;
      const uint32_t count = end - first, at = first - begin;
      // A row reads the groups of its KV source its last indexer selected,
      // none before its source's first group.
      const uint32_t groups = compressed ? (sequence.committed + sequence.rows) / ratioOf(source) : 0;
      if (layer.indexer && groups)
        addIndexer(graph, s, index, sequence, begin, first, count, normalized, kvLayers);
      const uint32_t selectedStride = groups ? topK : 0;
      const metal::MetalBuffer keyIndices = rowsOf<uint32_t>(s.keyIndices, at, count, keyStride);
      const metal::MetalBuffer keyCounts = rowsOf<uint32_t>(s.keyCounts, at, count, 1);
      const metal::MetalBuffer rowPositions = rowsOf<uint32_t>(s.positions, first, count, 1);
      ops::LatentAttention::addKeys(
          graph, rowPositions, groups ? rowsOf<uint32_t>(s.selected, first, count, topK) : rowPositions, keyIndices,
          keyCounts,
          {window, window, sequence.committed, groups ? ratioOf(source) : 0, sequence.verifyRows, sequence.lane},
          selectedStride, keyStride,
          count);
      ops::LatentAttention::add(
          graph, rowsOf<uint16_t>(s.qRotated, at, count, width),
          {rowsOf<uint16_t>(latents, sequence.rowBegin, sequence.rows, kLatent), ring(sequence.stateIn, index), window},
          {{}, sequence.pageTable, kvLayers[source]}, keyIndices, keyCounts, layer.sinks,
          rowsOf<float>(s.scores, at, count, heads * keyStride), s.parts,
          rowsOf<float>(s.out, at, count, heads * kLatent), heads, keyStride, scale, count);
    }
    ops::LatentRotation::add(graph, s.out, true, layer.queryNorm.buffer, freqs, positions, s.rotatedBack,
                             {heads, kLatent, pairs, true, ops::LatentRounding::Fp8, false, 0.0F}, n);
    for (uint32_t group = 0; group < d_.outputGroups; ++group) {
      ops::RowCopy::add(graph, s.rotatedBack, {0, width, group * groupWidth}, s.groupInput, {0, groupWidth, 0}, n,
                        groupWidth);
      project(graph, s.groupInput, layer.outputA[group], s.groupOutput, n, decode, lanes, linear);
      ops::RowCopy::add(graph, s.groupOutput, {0, rank, 0}, s.joined, {0, joinedWidth, group * rank}, n, rank);
    }
    project(graph, s.joined, layer.outputB, decode ? s.branch : fromRow(s.branch, begin, hidden, 2), n, decode, lanes,
            linear);
    // A prefill's indexer layer waits for the host after each
    // kIndexerStepRows of a chunk's rows: the work before a wait is a
    // sixteenth of a command's.
    if (!decode && layer.indexer && (begin + n) % kIndexerStepRows == 0) graph.host([] {});
  }
  if (!decode)
    for (const Rows &sequence : sequences) {
      const metal::MetalBuffer step = rowsOf<uint16_t>(latents, sequence.rowBegin, sequence.rows, kLatent);
      const metal::MetalBuffer before = ring(sequence.stateIn, index);
      ops::LatentAttention::addRingUpdate(graph, before, step, ring(sequence.stateOut, index), {}, sequence.committed,
                                          sequence.rows, window);
      for (const QwenTargetStateCapture &capture : sequence.captures)
        ops::LatentAttention::addRingUpdate(graph, before, step, ring(capture.auxiliary, index), {},
                                            sequence.committed, capture.row, window);
    }
  ops::ManifoldHyperConnection::addPost(graph, s.streams, s.branch, s.attentionMixes, mhc_, rows);
}

void DeepSeek41Target::addStreamCapture(metal::CommandGraph &graph, const metal::MetalBuffer &streams, uint32_t slot,
                                        const metal::MetalBuffer &captured, uint32_t destinationRow,
                                        uint32_t rows) const {
  const uint32_t width = geometry_.capturedHiddenSize();
  if (slot >= geometry_.captureLayerCount)
    throw std::logic_error("DeepSeek-V4.1 capture slot past the captured hidden rows");
  ops::ManifoldHyperConnection::addStreamMean(graph, streams, rowsOf<uint16_t>(captured, destinationRow, rows, width),
                                              width, slot * weights_.layout.hiddenSize, mhc_, rows);
}

std::vector<metal::MetalBuffer> DeepSeek41Target::addPrefill(metal::CommandGraph &graph,
                                                             std::span<const QwenTargetPrefillChunk> chunks,
                                                             const kv::PageStorage &kv, SpilledKvReads &,
                                                             QwenPrefillLayers layers, ops::AneFfn *) {
  const DeepSeek41Layout &layout = weights_.layout;
  if (chunks.empty() || chunks.size() > prefill_.size() || kv.layers().size() != sources_.size() ||
      layers.first >= layers.end || layers.end > layout.layers)
    throw std::invalid_argument("invalid DeepSeek-V4.1 prefill");
  const std::span<const SplashKvLayer> kvLayers = kv.layers();
  const uint32_t perToken = (d_.engramNgram - 1) * d_.engramHeads;
  const uint32_t engramLayers = static_cast<uint32_t>(std::popcount(d_.engramLayers));
  const bool begins = !layers.first;
  std::vector<std::vector<Rows>> sequences(chunks.size());
  std::vector<ops::MoePlan> plans;
  std::vector<ops::MoeBuffers> ffn(chunks.size());
  for (uint32_t c = 0; c < chunks.size(); ++c) {
    const QwenTargetPrefillChunk &chunk = chunks[c];
    const Scratch &s = prefill_[c];
    if (chunk.sequences.empty() || chunk.sequences.size() > kLanes || !chunk.rows || chunk.rows > s.rows)
      throw std::invalid_argument("invalid DeepSeek-V4.1 prefill chunk");
    plans.push_back(operators_.moePrefill(moeShape_, chunk.rows));
    ffn[c] = {chunk.buffers.normalized, zeros_, s.ffnOut, chunk.buffers.moe, {}};
    for (uint32_t index = 0; index < chunk.sequences.size(); ++index) {
      const QwenTargetPrefillSequence &sequence = chunk.sequences[index];
      if (!sequence.auxiliaryIn || !sequence.auxiliaryOut || sequence.lane >= kLanes ||
          sequence.rowBegin + sequence.rows > chunk.rows)
        throw std::invalid_argument("invalid DeepSeek-V4.1 prefill sequence");
      Rows rows{.rowBegin = sequence.rowBegin,
                .rows = sequence.rows,
                .committed = static_cast<uint32_t>(sequence.chunk.committed_tokens),
                .lane = sequence.lane,
                .pageTable = sequence.pageTable,
                .stateIn = sequence.auxiliaryIn,
                .stateOut = sequence.auxiliaryOut,
                .captures = sequence.stateCaptures};
      for (uint32_t ratio = 0; ratio < 2; ++ratio)
        rows.groupPositions[ratio] =
            rowsOf<uint32_t>(s.groupPositions, (index * 2 + ratio) * (s.rows + 1), s.rows + 1, 1);
      sequences[c].push_back(rows);
    }
    if (!begins) continue;
    // The rows' positions, each sequence's groups' positions, and its rows'
    // Engram rows, staged from the tables' file; then the streams of the
    // embedded rows.
    auto *positions = contents<uint32_t>(s.positions);
    auto *ids = contents<uint32_t>(s.engramIds);
    const EngramHashTables hash = weights_.engramHash();
    std::vector<uint32_t> rowIds(uint64_t{hash.layers} * hash.rowsPerLayer());
    for (uint32_t index = 0; index < chunk.sequences.size(); ++index) {
      const QwenTargetPrefillSequence &sequence = chunk.sequences[index];
      const Rows &rows = sequences[c][index];
      if (sequence.request.tokens.size() < uint64_t{rows.committed} + rows.rows)
        throw std::logic_error("a DeepSeek-V4.1 prefill sequence lacks its tokens");
      for (uint32_t row = 0; row < rows.rows; ++row) {
        const uint32_t position = rows.committed + row;
        positions[rows.rowBegin + row] = position;
        engramRows(hash, sequence.request.tokens, position, rowIds.data());
        for (uint32_t table = 0; table < engramLayers; ++table)
          std::memcpy(ids + (uint64_t{table} * chunk.rows + rows.rowBegin + row) * perToken,
                      rowIds.data() + uint64_t{table} * perToken, perToken * sizeof(uint32_t));
      }
      for (uint32_t ratio = 1; ratio <= 2; ++ratio) {
        auto *groups = contents<uint32_t>(rows.groupPositions[ratio == 2 ? 0 : 1]);
        const uint32_t firstGroup = rows.committed / ratio;
        for (uint32_t g = 0; g <= rows.rows; ++g) groups[g] = (firstGroup + g) * ratio;
      }
    }
    const uint64_t staged = uint64_t{engramLayers} * chunk.rows * perToken;
    for (uint32_t table = 0; table < engramLayers; ++table)
      for (uint64_t at = uint64_t{table} * chunk.rows * perToken; at < (table + 1ull) * chunk.rows * perToken; ++at)
        ids[at] += weights_.engramFirstRows[table];
    const uint64_t rowBytes = weights_.streamedTable->table().rowBytes;
    weights_.streamedTable->stage({ids, staged}, {contents<uint8_t>(s.engramStaging), staged * rowBytes},
                                  {contents<uint32_t>(s.engramSlots), staged}, SparseRowUse::Prefill, true);
    ops::ManifoldHyperConnection::addInit(graph, chunk.buffers.hidden[0], s.streams, mhc_, chunk.rows);
  }
  if (begins) readAhead(chunks);
  std::vector<ops::MoeLoadPart> parts;
  for (uint32_t c = 0; c < chunks.size(); ++c) parts.push_back({&ffn[c], &plans[c]});
  for (uint32_t index = layers.first; index < layers.end; ++index) {
    const DeepSeek41LayerWeights &layer = weights_.layers[index];
    for (uint32_t c = 0; c < chunks.size(); ++c) {
      const QwenTargetPrefillChunk &chunk = chunks[c];
      const Scratch &s = prefill_[c];
      if (layer.engram)
        addEngram(graph, s, chunk.buffers.linearScratch, false, 0, index, chunk.rows,
                  engramLayers * chunk.rows * perToken);
      if (const auto slot = geometry_.captureSlot(index))
        for (const QwenTargetPrefillSequence &sequence : chunk.sequences)
          for (uint32_t i = 0; i < sequence.captureCount; ++i) {
            const QwenTargetPrefillCapture &capture = sequence.captures[i];
            addStreamCapture(graph,
                             rowsOf<float>(s.streams, capture.sourceStart, capture.rows,
                                           MHC_STREAMS * layout.hiddenSize),
                             *slot, chunk.buffers.captured, capture.destinationStart, capture.rows);
          }
      addAttention(graph, s, chunk.buffers.linearScratch, false, 0, index, sequences[c], chunk.rows,
                   chunk.buffers.normalized, index ? s.ffnMixes : firstMixes_, kvLayers);
      ops::ManifoldHyperConnection::addMix(graph, s.streams, layer.ffnMix.fn, layer.ffnMix.scale, layer.ffnMix.base,
                                           s.mixChunks, s.ffnMixes, mhc_, chunk.rows);
      ops::ManifoldHyperConnection::addPreNorm(graph, s.streams, s.attentionMixes, layer.ffnNorm.buffer,
                                               chunk.buffers.normalized, mhc_, chunk.rows);
      ops::MoE::addRouting(graph, ffn[c], layer.ffn, plans[c]);
    }
    if (layer.ffn.stream) layer.ffn.stream->addLoad(graph, parts, layer.ffn.streamBlock);
    for (uint32_t c = 0; c < chunks.size(); ++c) {
      const Scratch &s = prefill_[c];
      ops::MoE::addRouted(graph, ffn[c], layer.ffn, plans[c]);
      ops::ManifoldHyperConnection::addPost(graph, s.streams, s.ffnOut, s.ffnMixes, mhc_, chunks[c].rows);
    }
  }
  std::vector<metal::MetalBuffer> outputs;
  if (layers.end != layout.layers) return outputs;
  for (uint32_t c = 0; c < chunks.size(); ++c) {
    ops::ManifoldHyperConnection::addPreNorm(graph, prefill_[c].streams, prefill_[c].ffnMixes,
                                             weights_.finalNorm.buffer, chunks[c].buffers.hidden[1], mhc_,
                                             chunks[c].rows);
    outputs.push_back(chunks[c].buffers.hidden[1]);
  }
  return outputs;
}

void DeepSeek41Target::readAhead(std::span<const QwenTargetPrefillChunk> chunks) const {
  std::array<const QwenTargetPrefillSequence *, kLanes> last{};
  for (const QwenTargetPrefillChunk &chunk : chunks)
    for (const QwenTargetPrefillSequence &sequence : chunk.sequences) last[sequence.lane] = &sequence;
  const EngramHashTables hash = weights_.engramHash();
  const uint32_t perToken = hash.rowsPerLayer();
  std::vector<uint32_t> ahead, rowIds(uint64_t{hash.layers} * perToken);
  uint64_t budget = uint64_t{prefillChunks_} * kPrefillRows;
  for (const QwenTargetPrefillSequence *sequence : last) {
    if (!sequence) continue;
    const std::span<const uint32_t> prompt = sequence->request.tokens;
    const uint64_t from = sequence->chunk.committed_tokens + sequence->rows;
    const uint64_t end = std::min<uint64_t>(prompt.size(), from + budget);
    if (from >= end) continue;
    for (uint64_t position = from; position < end; ++position) {
      engramRows(hash, prompt, position, rowIds.data());
      for (uint32_t table = 0; table < hash.layers; ++table)
        for (uint32_t i = 0; i < perToken; ++i)
          ahead.push_back(rowIds[uint64_t{table} * perToken + i] + weights_.engramFirstRows[table]);
    }
    budget -= end - from;
    if (!budget) break;
  }
  if (!ahead.empty()) weights_.streamedTable->prefetch(ahead);
}

void DeepSeek41Target::prepareVerifyLane(uint32_t lane, uint64_t position, const QwenTargetRequest &request) {
  if (lane >= kLanes) throw std::invalid_argument("invalid DeepSeek-V4.1 verify lane");
  if (request.tokens.size() < position) throw std::logic_error("DeepSeek-V4.1 verify history is shorter than its position");
  const uint32_t prior = d_.engramNgram - 1;
  auto *before = contents<uint32_t>(verify_->prior) + uint64_t{lane} * prior;
  for (uint32_t back = 1; back <= prior; ++back)
    before[back - 1] = position >= back ? request.tokens[position - back] : d_.engramPadToken;
  preparedPositions_[lane] = position;
}

void DeepSeek41Target::addVerifyInput(metal::CommandGraph &graph, metal::MetalBuffer draftInput,
                                      metal::MetalBuffer proposals, metal::MetalBuffer verifyInput,
                                      uint32_t lanes) const {
  QwenTarget::addVerifyInput(graph, std::move(draftInput), std::move(proposals), verifyInput, lanes);
  const Scratch &s = *verify_;
  const uint32_t engramLayers = static_cast<uint32_t>(std::popcount(d_.engramLayers));
  EngramHashParams params{.rows = lanes * kVerifyRows,
                          .lane_rows = kVerifyRows,
                          .layers = engramLayers,
                          .ngram = d_.engramNgram,
                          .heads = d_.engramHeads,
                          .pad = d_.engramPadToken,
                          .first_rows = {}};
  for (uint32_t table = 0; table < engramLayers; ++table) params.first_rows[table] = weights_.engramFirstRows[table];
  ops::Engram::addHash(graph, std::move(verifyInput), s.prior, weights_.engramTokenMap, weights_.engramMultipliers,
                       weights_.engramPrimes, weights_.engramOffsets, s.engramIds, params);
  // A verify step stages a few rows a lane, most of them cached, so its
  // host step is short and keeps the GPU awake (Qwen4Steps::addVerifyInput).
  const uint64_t count = uint64_t{engramLayers} * params.rows * (d_.engramNgram - 1) * d_.engramHeads;
  graph.host(
      [store = weights_.streamedTable, ids = s.engramIds, slots = s.engramSlots, staging = s.engramStaging, count] {
        const uint64_t rowBytes = store->table().rowBytes;
        store->stage({contents<const uint32_t>(ids), count}, {contents<uint8_t>(staging), count * rowBytes},
                     {contents<uint32_t>(slots), count}, SparseRowUse::Decode, false);
      },
      {}, metal::HostStepGpu::Awake);
}

void DeepSeek41Target::addVerify(metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
                                 const kv::PageStorage &kv, SpilledKvReads &,
                                 std::span<const kv::ChunkedPrefillParams> chunks, uint32_t lanes) const {
  const DeepSeek41Layout &layout = weights_.layout;
  if (!lanes || lanes > kLanes || chunks.size() != lanes || kv.layers().size() != sources_.size())
    throw std::invalid_argument("invalid DeepSeek-V4.1 verify batch");
  const std::span<const SplashKvLayer> kvLayers = kv.layers();
  const Scratch &s = *verify_;
  const uint32_t rows = lanes * kVerifyRows;
  const uint32_t engramLayers = static_cast<uint32_t>(std::popcount(d_.engramLayers));
  std::vector<Rows> sequences;
  auto *positions = contents<uint32_t>(s.positions);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const uint32_t committed = chunks[lane].committed_tokens;
    if (preparedPositions_[lane] != committed)
      throw std::logic_error("a DeepSeek-V4.1 verify lane's inputs were prepared for another position");
    if (!buffers.auxiliary[lane]) throw std::invalid_argument("DeepSeek-V4.1 verify lane without its state");
    Rows lanes{.rowBegin = lane * kVerifyRows,
               .rows = kVerifyRows,
               .committed = committed,
               .lane = lane,
               .pageTable = buffers.pageTables[lane],
               .stateIn = buffers.auxiliary[lane]};
    // With a draft, a lane's rows past those it verifies attend nothing.
    if (geometry_.captureLayerCount) lanes.verifyRows = buffers.verifyRows;
    for (uint32_t row = 0; row < kVerifyRows; ++row) positions[lane * kVerifyRows + row] = committed + row;
    for (uint32_t ratio = 1; ratio <= 2; ++ratio) {
      lanes.groupPositions[ratio == 2 ? 0 : 1] =
          rowsOf<uint32_t>(s.groupPositions, (lane * 2 + (ratio == 2 ? 0 : 1)) * (s.rows + 1), kVerifyRows + 1, 1);
      auto *groups = contents<uint32_t>(lanes.groupPositions[ratio == 2 ? 0 : 1]);
      for (uint32_t g = 0; g <= kVerifyRows; ++g) groups[g] = (committed / ratio + g) * ratio;
    }
    sequences.push_back(lanes);
  }
  const ops::MoePlan plan = operators_.moeDecode(moeShape_, lanes);
  ops::ManifoldHyperConnection::addInit(graph, buffers.hidden[0], s.streams, mhc_, rows);
  for (uint32_t index = 0; index < layout.layers; ++index) {
    const DeepSeek41LayerWeights &layer = weights_.layers[index];
    if (layer.engram)
      addEngram(graph, s, buffers.linearScratch, true, lanes, index, rows,
                engramLayers * rows * (d_.engramNgram - 1) * d_.engramHeads);
    if (const auto slot = geometry_.captureSlot(index))
      addStreamCapture(graph, s.streams, *slot, buffers.capturedTargetHidden, 0, rows);
    addAttention(graph, s, buffers.linearScratch, true, lanes, index, sequences, rows, buffers.normalized,
                 index ? s.ffnMixes : firstMixes_, kvLayers);
    ops::ManifoldHyperConnection::addMix(graph, s.streams, layer.ffnMix.fn, layer.ffnMix.scale, layer.ffnMix.base,
                                         s.mixChunks, s.ffnMixes, mhc_, rows);
    ops::ManifoldHyperConnection::addPreNorm(graph, s.streams, s.attentionMixes, layer.ffnNorm.buffer,
                                             buffers.normalized, mhc_, rows);
    ops::MoE::add(graph, {buffers.normalized, zeros_, s.ffnOut, buffers.moe, buffers.verifyRows}, layer.ffn, plan);
    ops::ManifoldHyperConnection::addPost(graph, s.streams, s.ffnOut, s.ffnMixes, mhc_, rows);
  }
  ops::ManifoldHyperConnection::addPreNorm(graph, s.streams, s.ffnMixes, weights_.finalNorm.buffer,
                                           buffers.finalHidden, mhc_, rows);
  addDecodeProjection(graph, buffers.finalHidden, weights_.logitsProjection, buffers.logits, lanes,
                      buffers.linearScratch);
}

void DeepSeek41Target::addHeadBatch(metal::CommandGraph &graph, metal::MetalBuffer hidden, metal::MetalBuffer,
                                    metal::MetalBuffer logits, uint32_t lanes, ops::LinearScratch scratch) const {
  addDecodeProjection(graph, std::move(hidden), vocabularyProjection(), std::move(logits), lanes, scratch);
}

void DeepSeek41Target::addStateCommit(metal::CommandGraph &graph, QwenTargetCommitBuffers buffers,
                                      uint32_t lanes) const {
  QwenTarget::addStateCommit(graph, buffers, lanes);
  const Scratch &s = *verify_;
  const uint32_t window = weights_.layout.slidingWindow;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const uint32_t committed = static_cast<uint32_t>(preparedPositions_[lane]);
    for (uint32_t layer = 0; layer < weights_.layout.layers; ++layer)
      ops::LatentAttention::addRingUpdate(
          graph, ring(buffers.auxiliaryIn[lane], layer),
          rowsOf<uint16_t>(s.layerLatents, layer * s.rows + lane * kVerifyRows, kVerifyRows, kLatent),
          ring(buffers.auxiliaryOut[lane], layer), buffers.retainedCounts, committed, kVerifyRows, window, lane);
    for (uint32_t source = 0; source < sources_.size(); ++source) {
      if (ratioOf(source) == 1) continue;
      const uint32_t paired = pairedOf_[source];
      const uint64_t laneBytes = uint64_t{kVerifyRows + 1} * kLatent * sizeof(float);
      const uint64_t at = (uint64_t{paired} * kLanes + lane) * laneBytes;
      ops::LatentAttention::addOpenCommit(graph, backend_.view(s.sourceKv, at, laneBytes),
                                          backend_.view(s.sourceGate, at, laneBytes),
                                          openGroup(buffers.auxiliaryOut[lane], paired), buffers.retainedCounts,
                                          committed, kLatent, lane);
    }
  }
}

QwenTargetGeometry qwenTargetGeometry(const DeepSeek41Weights &weights, std::span<const uint32_t> captureLayers) {
  QwenTargetGeometry geometry(weights.layout, captureLayers);
  geometry.kvLayout = weights.layout.kvLayout();
  geometry.stateLayout = weights.layout.gdnStateLayout();
  std::vector<const ops::BlockMoeWeights *> moeBlocks;
  for (const DeepSeek41LayerWeights &layer : weights.layers) {
    for (const ops::Projection *projection : {&layer.queryA, &layer.queryB, &layer.kv, &layer.outputB})
      geometry.include(*projection);
    for (const ops::Projection &group : layer.outputA) geometry.include(group);
    if (layer.engram) geometry.include(layer.engram->kv);
    moeBlocks.push_back(&layer.ffn);
  }
  geometry.finish(moeBlocks, weights.logitsProjection);
  return geometry;
}

std::unique_ptr<QwenTarget> makeQwenTarget(const DeepSeek41Weights &weights, const QwenTargetGeometry &geometry,
                                           metal::MetalBackend &backend, const ops::ExecutionPlans &operators,
                                           uint32_t prefillChunks) {
  return std::make_unique<DeepSeek41Target>(weights, geometry, backend, operators, prefillChunks);
}

QwenTargetScratchBytes qwenTargetScratchBytes(const DeepSeek41Weights &weights, const QwenTargetGeometry &,
                                              uint32_t prefillChunks, uint64_t tableRowBytes) {
  const DeepSeek41Dimensions &d = weights.layout.deepseek;
  return DeepSeek41Target::scratchBytes(
      weights.layout, {weights.layout.hiddenSize, d.sinkhornIterations, d.hcEpsilon, d.normEpsilon}, prefillChunks,
      tableRowBytes);
}

// A step stages the Engram rows of its rows, every Engram layer's at once
// (each Engram layer reads (n-gram - 1) * heads rows of a token), and the
// read-ahead of the next prefill command, a group's chunks at most, may have
// them queued or in flight beside a chunk's more.
SparseRowStoreOptions streamedTableRows(const DeepSeek41Layout &layout) {
  const DeepSeek41Dimensions &d = layout.deepseek;
  const uint32_t stageRows = ExecutionLimits::prefillTokenBudget * static_cast<uint32_t>(std::popcount(d.engramLayers)) *
                             (d.engramNgram - 1) * d.engramHeads;
  return {.prefetchRows = (kPrefillGroupChunks + 1) * stageRows, .stageRows = stageRows};
}

} // namespace splash::model

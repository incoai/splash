#include "model/DSparkDraft.hpp"

#include "metal/abi/DSpark.h"
#include "metal/abi/LatentAttention.h"
#include "ops/DSparkMarkov.hpp"
#include "ops/DraftAttention.hpp"
#include "ops/Embedding.hpp"
#include "ops/LatentAttention.hpp"
#include "ops/LatentRotation.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"
#include "ops/RowCopy.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace splash::model {
namespace {

constexpr uint32_t kLanes = ExecutionLimits::maximumBatchWidth;
// A lane's rows of a decode step's buffers: its block's, the rest unused.
constexpr uint32_t kLaneRows = ExecutionLimits::draftQueryRows;
// A commit's most rows: the target's step's, a verify block a lane at most.
constexpr uint32_t kCommitRows = kLanes * ExecutionLimits::targetVerifyRows;
constexpr uint32_t kLatent = LATENT_ATTENTION_DIM;

template <class T> T *contents(const metal::MetalBuffer &buffer) { return static_cast<T *>(buffer.contents()); }

// The shape of DSpark's MoE blocks, its layout's, and the formats its plans
// follow.
ops::MoeShape moeShapeOf(const DSparkWeights &weights) {
  const ops::MoeShape shape = weights.layout.layers.moeBlockShape();
  if (weights.layers.empty() || !shape.valid()) throw std::invalid_argument("invalid DSpark MoE shape");
  return shape;
}

uint32_t moeRouterFormatOf(const DSparkWeights &weights) {
  std::vector<const ops::BlockMoeWeights *> blocks;
  for (const DeepSeek41LayerWeights &layer : weights.layers) blocks.push_back(&layer.ffn);
  return ops::moeRouterFormat(blocks);
}

// Every projection of its decode steps: its layers', its main projection
// (a commit's) and the target's head.
template <class Use> void forEachDecodeProjection(const DSparkWeights &weights, const ops::Projection &head, Use use) {
  for (const DeepSeek41LayerWeights &layer : weights.layers) {
    for (const ops::Projection *p : {&layer.queryA, &layer.queryB, &layer.kv, &layer.outputB}) use(*p);
    for (const ops::Projection &p : layer.outputA) use(p);
  }
  use(weights.mainProjection);
  use(head);
}

} // namespace

struct DSparkDraft::Scratch {
  // Rows of a decode step's buffers (its lanes' rows and their storage) and
  // of a context's.
  uint32_t rows = 0;
  uint32_t contextRows = 0;
  uint32_t keyStride = 0;
  metal::MetalBuffer streams, attentionMixes, ffnMixes, firstMixes, mixChunks, x, qa, qr, q, qRotated, raw, block;
  metal::MetalBuffer keyIndices, keyCounts, scores, parts, out, rotatedBack, groupInput, groupOutput, joined;
  // Each lane's block of its rows (SplashLaneRows), which its MoE routes.
  metal::MetalBuffer branch, zeros, ffnOut, blocks, frequencies;
  ops::MoeScratch moe;
  // The Markov head's state, and each lane's proposals and confidences.
  metal::MetalBuffer markov, bias, previous, proposals, confidence;
  // A decode command's commit and a prefill command's context run apart,
  // each on buffers of its own; a decode step's rows' positions and its
  // commit's.
  metal::MetalBuffer commitRaw, commitLatents, positions, commitPositions;
  metal::MetalBuffer latentsRaw, latents, rebuildPositions;
  ops::LinearScratch decodeLinear{}, prefillLinear{};
};

namespace {

// The scratch of DSpark of `weights`, through `allocate(bytes, label)`.
template <class Scratch, class Allocate>
void layoutScratch(const DSparkWeights &weights, const ops::Projection &head, const ops::ExecutionPlans &operators,
                   const ops::ManifoldHyperShape &mhc, const ops::MoeShape &moeShape, uint32_t moeRouterFormat, Scratch &s, Allocate allocate) {
  const DeepSeek41Layout &layout = weights.layout.layers;
  const DeepSeek41Dimensions &d = layout.deepseek;
  const ops::Linear &linear = operators.linear();
  uint32_t rows = kLanes * kLaneRows;
  ops::LinearScratchSize decode{}, prefill{};
  forEachDecodeProjection(weights, head, [&](const ops::Projection &p) {
    rows = std::max(rows, linear.storageRows(kLanes * kLaneRows, p.shape()));
    decode.include(linear.scratchSize(p.shape(), kCommitRows));
  });
  prefill.include(linear.scratchSize(weights.mainProjection.shape(), ExecutionLimits::prefillTokenBudget));
  for (const DeepSeek41LayerWeights &layer : weights.layers)
    prefill.include(linear.scratchSize(layer.kv.shape(), ExecutionLimits::prefillTokenBudget));
  const uint64_t r = rows, hidden = layout.hiddenSize, width = layout.attentionWidth;
  const uint64_t heads = layout.attentionQueryHeads;
  s.rows = rows;
  s.contextRows = std::max<uint32_t>(rows, ExecutionLimits::prefillTokenBudget);
  s.keyStride = layout.slidingWindow + kLaneRows;
  s.streams = allocate(r * MHC_STREAMS * hidden * 4, "dspark/streams");
  s.attentionMixes = allocate(r * MHC_MIXES * 4, "dspark/mixes");
  s.ffnMixes = allocate(r * MHC_MIXES * 4, "dspark/mixes");
  s.firstMixes = allocate(r * MHC_MIXES * 4, "dspark/first-mixes");
  s.mixChunks = allocate(ops::ManifoldHyperConnection::mixScratchBytes(mhc, rows), "dspark/mix");
  s.x = allocate(r * hidden * 2, "dspark/x");
  s.qa = allocate(r * d.queryRank * 2, "dspark/qa");
  s.qr = allocate(r * d.queryRank * 2, "dspark/qr");
  s.q = allocate(r * width * 2, "dspark/q");
  s.qRotated = allocate(r * width * 2, "dspark/q-rotated");
  s.raw = allocate(r * kLatent * 2, "dspark/kv");
  s.block = allocate(r * kLatent * 2, "dspark/latents");
  s.keyIndices = allocate(r * s.keyStride * 4, "dspark/keys");
  s.keyCounts = allocate(r * 4, "dspark/key-counts");
  s.scores = allocate(r * heads * s.keyStride * 4, "dspark/scores");
  uint64_t parts = 0;
  for (uint32_t blockRows = 1; blockRows <= kLaneRows; ++blockRows)
    parts = std::max(parts, ops::LatentAttention::partBytes(layout.attentionQueryHeads, blockRows));
  s.parts = allocate(parts, "dspark/parts");
  s.out = allocate(r * heads * kLatent * 4, "dspark/attention-out");
  s.rotatedBack = allocate(r * width * 2, "dspark/rotated-back");
  s.groupInput = allocate(r * d.groupWidth(layout.attentionWidth) * 2, "dspark/group-input");
  s.groupOutput = allocate(r * d.outputRank * 2, "dspark/group-output");
  s.joined = allocate(r * d.outputGroups * d.outputRank * 2, "dspark/joined");
  s.branch = allocate(r * hidden * 2, "dspark/branch");
  s.zeros = allocate(r * hidden * 2, "dspark/zeros");
  s.ffnOut = allocate(r * hidden * 2, "dspark/ffn-out");
  s.blocks = allocate(uint64_t{kLanes} * sizeof(SplashLaneRows), "dspark/blocks");
  s.frequencies = allocate(uint64_t{layout.rotaryPairs} * 4, "dspark/rope");
  const ops::MoeWorkspace workspace = operators.moeWorkspace(moeShape, moeRouterFormat, kLanes * kLaneRows);
  for (const ops::MoeScratchField &field : ops::kMoeScratchFields)
    s.moe.*field.buffer = allocate(workspace.*field.bytes, field.name);
  s.markov = allocate(uint64_t{kLanes} * weights.layout.markovRank * 2, "dspark/markov");
  s.bias = allocate(uint64_t{kLanes} * layout.vocabularySize * 4, "dspark/markov-bias");
  s.previous = allocate(uint64_t{kLanes} * 4, "dspark/previous");
  s.proposals = allocate(uint64_t{kLanes} * SPLASH_DRAFT_PROPOSAL_TOKENS * 4, "dspark/proposals");
  s.confidence = allocate(uint64_t{kLanes} * SPLASH_DRAFT_PROPOSAL_TOKENS * 4, "dspark/confidence");
  s.commitRaw = allocate(uint64_t{kCommitRows} * kLatent * 2, "dspark/commit-kv");
  s.commitLatents = allocate(uint64_t{kCommitRows} * kLatent * 2, "dspark/commit-latents");
  s.positions = allocate(r * 4, "dspark/positions");
  s.commitPositions = allocate(uint64_t{kCommitRows} * 4, "dspark/commit-positions");
  s.latentsRaw = allocate(uint64_t{s.contextRows} * kLatent * 2, "dspark/context-kv");
  s.latents = allocate(uint64_t{s.contextRows} * kLatent * 2, "dspark/context-latents");
  s.rebuildPositions = allocate(uint64_t{kLanes} * ExecutionLimits::draftContextTokens * 4, "dspark/rebuild");
  s.decodeLinear = {allocate(decode.input, "dspark/linear"), allocate(decode.sums, "dspark/linear"),
                    allocate(decode.partials, "dspark/linear"), allocate(decode.counters, "dspark/linear"),
                    allocate(decode.rotated, "dspark/linear")};
  s.prefillLinear = {allocate(prefill.input, "dspark/linear"), allocate(prefill.sums, "dspark/linear"),
                     allocate(prefill.partials, "dspark/linear"), allocate(prefill.counters, "dspark/linear"),
                     allocate(prefill.rotated, "dspark/linear")};
}

ops::ManifoldHyperShape mhcOf(const DSparkWeights &weights) {
  const DeepSeek41Layout &layout = weights.layout.layers;
  const DeepSeek41Dimensions &d = layout.deepseek;
  return {layout.hiddenSize, d.sinkhornIterations, d.hcEpsilon, d.normEpsilon};
}

} // namespace

uint64_t DSparkDraft::scratchBytes(const DSparkWeights &weights, const ops::Projection &head,
                                   const ops::ExecutionPlans &operators) {
  uint64_t total = 0;
  Scratch s;
  layoutScratch(weights, head, operators, mhcOf(weights), moeShapeOf(weights), moeRouterFormatOf(weights), s, [&](uint64_t bytes, const char *) {
    total += std::max<uint64_t>(bytes, 16);
    return metal::MetalBuffer{};
  });
  return total;
}

DSparkDraft::DSparkDraft(const DSparkWeights &weights, const ops::Projection &head, metal::MetalBackend &backend,
                         const ops::ExecutionPlans &operators)
    : weights_(weights), head_(head), backend_(backend), operators_(operators), mhc_(mhcOf(weights)),
      moeShape_(moeShapeOf(weights)), moeRouterFormat_(moeRouterFormatOf(weights)), scratch_(std::make_unique<Scratch>()) {
  const DSparkLayout &dspark = weights.layout;
  const DeepSeek41Layout &layout = dspark.layers;
  if (weights.layers.size() != layout.layers || !dspark.blockSize || dspark.blockSize > kLaneRows - 1 ||
      layout.attentionHeadDimension != kLatent)
    throw std::invalid_argument("invalid DSpark weights");
  // Every buffer starts zeroed: a lane's rows past its block are never
  // written, and pass through the layers as zeros.
  layoutScratch(weights, head, operators, mhc_, moeShape_, moeRouterFormat_, *scratch_, [&](uint64_t bytes, const char *label) {
    bytes = std::max<uint64_t>(bytes, 16);
    allocated_ += bytes;
    metal::MetalBuffer buffer = backend_.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
    std::memset(buffer.contents(), 0, bytes);
    return buffer;
  });
  Scratch &s = *scratch_;
  const std::vector<float> frequencies = deepSeek41RopeFrequencies(layout, false);
  std::memcpy(s.frequencies.contents(), frequencies.data(), frequencies.size() * sizeof(float));
  // The first mix's weights of a step's rows: stream 0 alone.
  auto *first = contents<float>(s.firstMixes);
  for (uint32_t row = 0; row < s.rows; ++row) first[uint64_t{row} * MHC_MIXES] = 1.0F;
  // Each lane routes its block's rows.
  auto *blocks = contents<SplashLaneRows>(s.blocks);
  for (uint32_t lane = 0; lane < kLanes; ++lane) blocks[lane] = {lane * kLaneRows, dspark.blockSize};
}

DSparkDraft::~DSparkDraft() = default;

metal::MetalBuffer DSparkDraft::ring(const metal::MetalBuffer &keys) const {
  const uint64_t bytes = uint64_t{ringSlots()} * kLatent * 2;
  if (keys.sizeBytes() < bytes) throw std::invalid_argument("a DSpark ring is past its lane's ring keys");
  return backend_.view(keys, 0, bytes);
}

void DSparkDraft::project(metal::CommandGraph &graph, const metal::MetalBuffer &input, const ops::Projection &projection,
                          const metal::MetalBuffer &output, uint32_t rows) const {
  operators_.linear().addProjection(graph, input, projection, output, rows, scratch_->decodeLinear);
}

uint32_t DSparkDraft::decodeStorageLanes(uint32_t lanes) const {
  const ops::Linear &linear = operators_.linear();
  const uint32_t rows = lanes * kLaneRows;
  uint32_t storageRows = rows;
  forEachDecodeProjection(weights_, head_, [&](const ops::Projection &p) {
    storageRows = std::max(storageRows, linear.storageRows(rows, p.shape()));
  });
  return (storageRows + kLaneRows - 1) / kLaneRows;
}

void DSparkDraft::addContextLatents(metal::CommandGraph &graph, const metal::MetalBuffer &hidden,
                                    const metal::MetalBuffer &positions, uint32_t rows,
                                    std::span<const DFlashPrefillSpan> spans) const {
  const Scratch &s = *scratch_;
  const DeepSeek41Layout &layout = weights_.layout.layers;
  if (rows > s.contextRows) throw std::invalid_argument("a DSpark context is past its scratch");
  for (uint32_t index = 0; index < layout.layers; ++index) {
    const DeepSeek41LayerWeights &layer = weights_.layers[index];
    operators_.linear().addProjection(graph, hidden, layer.kv, s.latentsRaw, rows, s.prefillLinear);
    ops::LatentRotation::add(graph, s.latentsRaw, false, layer.kvNorm.buffer, s.frequencies, positions, s.latents,
                             {1, kLatent, layout.rotaryPairs, false, ops::LatentRounding::Fp8, true,
                              layout.deepseek.normEpsilon},
                             rows);
    for (const DFlashPrefillSpan &span : spans) {
      const metal::MetalBuffer lane = ring(span.ring[index].keys);
      ops::LatentAttention::addRingUpdate(
          graph, lane, backend_.view(s.latents, uint64_t{span.compactRow} * kLatent * 2, uint64_t{span.rows} * kLatent * 2),
          lane, {}, span.startPosition, span.rows, ringSlots());
    }
  }
}

void DSparkDraft::addContextPrefill(metal::CommandGraph &graph, const DFlashPrefillBuffers &buffers,
                                    const metal::MetalBuffer &positions, uint32_t rows,
                                    std::span<const DFlashPrefillSpan> spans,
                                    std::span<const DFlashWindowCapture> windowCaptures) const {
  const DeepSeek41Layout &layout = weights_.layout.layers;
  if (!rows || rows > ExecutionLimits::prefillTokenBudget || spans.empty())
    throw std::invalid_argument("invalid DSpark context prefill");
  for (const DFlashPrefillSpan &span : spans)
    if (span.ring.size() != layout.layers || span.compactRow + span.rows > rows)
      throw std::invalid_argument("invalid DSpark context span");
  operators_.linear().addProjection(graph, buffers.capturedTargetHidden, weights_.mainProjection, buffers.projected,
                                    rows, scratch_->prefillLinear);
  ops::Normalization::addRms(graph, buffers.projected, weights_.mainNorm, buffers.hidden, layout.hiddenSize, rows);
  const uint64_t rowBytes = uint64_t{layout.hiddenSize} * 2;
  const auto store = [&](uint32_t compactRow, uint32_t count, uint32_t startPosition, const metal::MetalBuffer &window) {
    ops::DraftAttention::addWindowStore(graph, backend_.view(buffers.hidden, compactRow * rowBytes, count * rowBytes),
                                        window, count, startPosition, layout.hiddenSize);
  };
  for (const DFlashPrefillSpan &span : spans) store(span.compactRow, span.rows, span.startPosition, span.window);
  // Only the window: a cached state keeps no rings.
  for (const DFlashWindowCapture &capture : windowCaptures)
    store(capture.compactRow, capture.rows, capture.startPosition, capture.window);
  addContextLatents(graph, buffers.hidden, positions, rows, spans);
}

void DSparkDraft::addContextRebuild(metal::CommandGraph &graph, const DFlashPrefillBuffers &buffers,
                                    const metal::MetalBuffer &window, std::span<const DFlashDraftRingLayer> ring,
                                    uint32_t rows, uint32_t startPosition, uint32_t lane) const {
  const DeepSeek41Layout &layout = weights_.layout.layers;
  if (!rows || rows > ExecutionLimits::draftContextTokens || ring.size() != layout.layers || lane >= kLanes)
    throw std::invalid_argument("invalid DSpark context rebuild");
  // The rings hold the last ring-slot positions alone.
  const uint32_t kept = std::min(rows, ringSlots()), first = startPosition + rows - kept;
  ops::DraftAttention::addWindowLoad(graph, window, buffers.hidden, kept, first, layout.hiddenSize);
  const metal::MetalBuffer positions =
      backend_.view(scratch_->rebuildPositions, uint64_t{lane} * ExecutionLimits::draftContextTokens * 4,
                    uint64_t{kept} * 4);
  auto *values = contents<uint32_t>(positions);
  for (uint32_t row = 0; row < kept; ++row) values[row] = first + row;
  const DFlashPrefillSpan span{0, kept, first, ring, window};
  addContextLatents(graph, buffers.hidden, positions, kept, {&span, 1});
}

void DSparkDraft::addContextCommit(metal::CommandGraph &graph, const DSparkCommitBuffers &buffers,
                                   std::span<const uint32_t> startPositions, const ops::StepRows &step) const {
  const Scratch &s = *scratch_;
  const DeepSeek41Layout &layout = weights_.layout.layers;
  const uint32_t lanes = static_cast<uint32_t>(startPositions.size());
  ops::requireStepRows(step);
  if (lanes != step.lanes || buffers.rings.size() != layout.layers)
    throw std::invalid_argument("invalid DSpark context commit");
  const uint32_t stride = deepSeek41LaneStride(step), rows = lanes * stride;
  project(graph, buffers.captured, weights_.mainProjection, buffers.projected, rows);
  ops::Normalization::addRms(graph, buffers.projected, weights_.mainNorm, buffers.hidden, layout.hiddenSize, rows);
  for (uint32_t index = 0; index < layout.layers; ++index) {
    const DeepSeek41LayerWeights &layer = weights_.layers[index];
    project(graph, buffers.hidden, layer.kv, s.commitRaw, rows);
    ops::LatentRotation::add(graph, s.commitRaw, false, layer.kvNorm.buffer, s.frequencies, s.commitPositions,
                             s.commitLatents,
                             {1, kLatent, layout.rotaryPairs, false, ops::LatentRounding::Fp8, true,
                              layout.deepseek.normEpsilon},
                             rows);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const metal::MetalBuffer laneRing = ring(buffers.rings[index][lane]);
      ops::LatentAttention::addRingUpdate(
          graph, laneRing,
          backend_.view(s.commitLatents, uint64_t{lane} * stride * kLatent * 2, uint64_t{stride} * kLatent * 2),
          laneRing, buffers.retainedRows, startPositions[lane], stride, ringSlots(), lane);
    }
  }
}

void DSparkDraft::addDecode(metal::CommandGraph &graph, const DSparkDecodeBuffers &buffers,
                            std::span<const uint32_t> cacheLengths, const ops::StepRows &step) const {
  const Scratch &s = *scratch_;
  const DSparkLayout &dspark = weights_.layout;
  const DeepSeek41Layout &layout = dspark.layers;
  const DeepSeek41Dimensions &d = layout.deepseek;
  const uint32_t lanes = static_cast<uint32_t>(cacheLengths.size());
  ops::requireStepRows(step);
  if (lanes != step.lanes || buffers.rings.size() != layout.layers)
    throw std::invalid_argument("invalid DSpark decode batch");
  const uint32_t rows = lanes * kLaneRows, block = dspark.blockSize, slots = ringSlots();
  // Its blocks' rows' positions, and those of the commit's rows of the
  // target's step, each lane's from its first position on.
  auto *positions = contents<uint32_t>(s.positions);
  auto *commitPositions = contents<uint32_t>(s.commitPositions);
  const uint32_t stride = deepSeek41LaneStride(step);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    for (uint32_t row = 0; row < kLaneRows; ++row) positions[lane * kLaneRows + row] = cacheLengths[lane] + row;
    for (uint32_t row = 0; row < stride; ++row) commitPositions[lane * stride + row] = cacheLengths[lane] + row;
  }
  const uint32_t heads = layout.attentionQueryHeads, width = layout.attentionWidth, pairs = layout.rotaryPairs;
  const uint32_t groupWidth = d.groupWidth(width), rank = d.outputRank, joinedWidth = d.outputGroups * rank;
  // Each block row's keys: its lane's ring's filled slots, then every row of
  // its block, which attend each other whatever their order.
  auto *indices = contents<uint32_t>(s.keyIndices);
  auto *counts = contents<uint32_t>(s.keyCounts);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const uint32_t filled = std::min(slots, cacheLengths[lane]);
    for (uint32_t row = lane * kLaneRows; row < lane * kLaneRows + block; ++row) {
      uint32_t *keys = indices + uint64_t{row} * s.keyStride;
      for (uint32_t slot = 0; slot < filled; ++slot) keys[slot] = slot | LATENT_ATTENTION_WINDOW_KEY;
      for (uint32_t other = 0; other < block; ++other)
        keys[filled + other] = (slots + other) | LATENT_ATTENTION_WINDOW_KEY;
      counts[row] = filled + block;
    }
  }
  const ops::MoePlan plan = operators_.moe(moeShape_, moeRouterFormat_, rows);
  const ops::MoeBuffers ffn{
      .input = s.x, .residual = s.zeros, .output = s.ffnOut, .scratch = s.moe, .lanes = lanes, .laneRows = s.blocks};
  const float scale = 1.0F / std::sqrt(float(kLatent));
  ops::ManifoldHyperConnection::addInit(graph, buffers.input, s.streams, mhc_, rows);
  metal::MetalBuffer previous = s.firstMixes;
  for (uint32_t index = 0; index < layout.layers; ++index) {
    const DeepSeek41LayerWeights &layer = weights_.layers[index];
    ops::ManifoldHyperConnection::addMix(graph, s.streams, layer.attentionMix.fn, layer.attentionMix.scale,
                                         layer.attentionMix.base, s.mixChunks, s.attentionMixes, mhc_, rows);
    ops::ManifoldHyperConnection::addPreNorm(graph, s.streams, previous, layer.attentionNorm.buffer, s.x, mhc_, rows);
    project(graph, s.x, layer.queryA, s.qa, rows);
    ops::LatentRotation::add(graph, s.qa, false, layer.queryNorm.buffer, s.frequencies, s.positions, s.qr,
                             {1, d.queryRank, 0, false, ops::LatentRounding::None, true, d.normEpsilon}, rows);
    project(graph, s.qr, layer.queryB, s.q, rows);
    ops::LatentRotation::add(graph, s.q, false, layer.queryNorm.buffer, s.frequencies, s.positions, s.qRotated,
                             {heads, kLatent, pairs, false, ops::LatentRounding::None, false, 0.0F}, rows);
    project(graph, s.x, layer.kv, s.raw, rows);
    ops::LatentRotation::add(graph, s.raw, false, layer.kvNorm.buffer, s.frequencies, s.positions, s.block,
                             {1, kLatent, pairs, false, ops::LatentRounding::Fp8, true, d.normEpsilon}, rows);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint32_t first = lane * kLaneRows;
      const auto laneRows = [&](const metal::MetalBuffer &buffer, uint64_t rowBytes) {
        return backend_.view(buffer, first * rowBytes, block * rowBytes);
      };
      // Its keys are the window's alone: the block's latents stand as the
      // pool of compressed ones, which no key names.
      const metal::MetalBuffer latents = laneRows(s.block, kLatent * 2);
      ops::LatentAttention::add(graph, laneRows(s.qRotated, uint64_t{width} * 2),
                                {latents, ring(buffers.rings[index][lane]), slots}, {latents, {}, {}},
                                laneRows(s.keyIndices, uint64_t{s.keyStride} * 4), laneRows(s.keyCounts, 4),
                                layer.sinks, laneRows(s.scores, uint64_t{heads} * s.keyStride * 4), s.parts,
                                laneRows(s.out, uint64_t{heads} * kLatent * 4), heads, s.keyStride, scale, block);
    }
    ops::LatentRotation::add(graph, s.out, true, layer.queryNorm.buffer, s.frequencies, s.positions,
                             s.rotatedBack, {heads, kLatent, pairs, true, ops::LatentRounding::Fp8, false, 0.0F},
                             rows);
    for (uint32_t group = 0; group < d.outputGroups; ++group) {
      ops::RowCopy::add(graph, s.rotatedBack, {0, width, group * groupWidth}, s.groupInput, {0, groupWidth, 0}, rows,
                        groupWidth);
      project(graph, s.groupInput, layer.outputA[group], s.groupOutput, rows);
      ops::RowCopy::add(graph, s.groupOutput, {0, rank, 0}, s.joined, {0, joinedWidth, group * rank}, rows, rank);
    }
    project(graph, s.joined, layer.outputB, s.branch, rows);
    ops::ManifoldHyperConnection::addPost(graph, s.streams, s.branch, s.attentionMixes, mhc_, rows);
    ops::ManifoldHyperConnection::addMix(graph, s.streams, layer.ffnMix.fn, layer.ffnMix.scale, layer.ffnMix.base,
                                         s.mixChunks, s.ffnMixes, mhc_, rows);
    ops::ManifoldHyperConnection::addPreNorm(graph, s.streams, s.attentionMixes, layer.ffnNorm.buffer, s.x, mhc_,
                                             rows);
    ops::MoE::addRouting(graph, ffn, layer.ffn, plan);
    ops::MoE::addRouted(graph, ffn, layer.ffn, plan);
    ops::ManifoldHyperConnection::addPost(graph, s.streams, s.ffnOut, s.ffnMixes, mhc_, rows);
    previous = s.ffnMixes;
  }
  // The head: the last mix's pre-sum through the output norm, the target's
  // logits of it.
  ops::ManifoldHyperConnection::addPreNorm(graph, s.streams, s.ffnMixes, weights_.outputNorm.buffer, s.x, mhc_, rows);
  project(graph, s.x, head_, buffers.logits, rows);
}

void DSparkDraft::addSelection(metal::CommandGraph &graph, const ops::DraftSelectorBuffers &tree,
                               std::span<const ops::DraftTreeLane> lanes) const {
  const Scratch &s = *scratch_;
  const DSparkLayout &dspark = weights_.layout;
  const DeepSeek41Layout &layout = dspark.layers;
  const auto count = static_cast<uint32_t>(lanes.size());
  if (!count || count > kLanes) throw std::invalid_argument("invalid DSpark selection batch");
  DSparkTreeParams trees{.lanes = count, .block = dspark.blockSize};
  auto *previous = contents<uint32_t>(s.previous);
  for (uint32_t lane = 0; lane < count; ++lane) {
    const ops::DraftTreeLane &tree = lanes[lane];
    previous[lane] = tree.anchor;
    trees.anchor[lane] = tree.anchor;
    trees.depth[lane] = tree.depth;
    trees.chain[lane] = tree.children ? 0 : tree.chain;
    trees.slope[lane] = tree.calibration.slope;
    std::copy(tree.calibration.intercepts.begin(), tree.calibration.intercepts.end(), trees.intercept[lane]);
  }
  DSparkStepParams params{.lanes = count,
                          .vocabulary = layout.vocabularySize,
                          .hidden = layout.hiddenSize,
                          .rank = dspark.markovRank,
                          .block = dspark.blockSize,
                          .position = 0,
                          .lane_rows = kLaneRows};
  const metal::MetalBuffer markov = backend_.view(s.markov, 0, uint64_t{count} * dspark.markovRank * 2);
  for (uint32_t position = 0; position < dspark.blockSize; ++position) {
    params.position = position;
    ops::Embedding::add(graph, s.previous, weights_.markovEmbedding, markov, count);
    ops::addGgufFloat(graph, markov, weights_.markovHead.blocks().segments.front(), s.bias, count,
                      layout.vocabularySize, 0, ops::FloatOutput::Float32, ops::FloatTile::Simdgroup);
    ops::DSparkMarkov::addStep(graph,
                               {tree.logits, s.bias, s.streams, s.ffnMixes, markov, weights_.confidence, s.proposals,
                                s.confidence, s.previous},
                               params);
  }
  ops::DSparkMarkov::addTrees(graph,
                              {s.proposals, s.confidence, tree.chainTokens, tree.chainRates, tree.treeTokens, tree.trees,
                               tree.treeScores, tree.treeSizes, tree.treeDraws, tree.treeCaps},
                              trees);
}

} // namespace splash::model

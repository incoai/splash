#include "model/Inkling2LoopSteps.hpp"

#include "Checked.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "ops/MessageGate.hpp"
#include "ops/RelativeAttention.hpp"

#include <cstring>
#include <stdexcept>

namespace splash::model {
namespace {

constexpr uint32_t kChunkRows = ExecutionLimits::prefillTokenBudget;
// The most queries a block mix takes: a full layer's q, k and v inputs.
constexpr uint32_t kMixerQueries = 3;
// The messages of a loop layer: its q, k, v and FFN inputs'.
constexpr uint32_t kMessages = 4;

template <class T> constexpr uint64_t bytesFor(uint64_t elements) noexcept { return elements * sizeof(T); }

// A scratch tensor (Inkling2LoopScratch): its bytes per row of a step and its
// fixed bytes, and whether a prefill group's chunks each hold their own: the
// block stack and the pass before's states, which outlive a command, and the
// residual, which a layer's FFN reads after the group's load of the layer's
// experts (QwenTarget::addPrefillLayers). The rest a chunk uses within its
// part of a layer, one chunk after another.
struct Tensor final {
  metal::MetalBuffer Inkling2LoopScratch::*field;
  uint64_t rowBytes;
  uint64_t fixedBytes;
  bool carried;
};

// A prefill chunk's or a decode step's tensors. The attention stages its
// relative states and row sinks in the queries' staged rows: a step's rows
// and, each sequence's rows taking whole KV blocks, the rest of a block per
// lane (RuntimeArenas.cpp targetTensorBytes).
std::vector<Tensor> tensors(const QwenTargetGeometry &g, const Inkling2LoopShape &shape, bool decode) {
  const Inkling2LoopSchedule schedule(shape);
  const uint64_t hidden = g.hiddenSize, kv = shape.kvWidth;
  // A full layer's dynamic taps, as their projection writes them: whole
  // tiles of GGUF_TILE_COLUMNS columns, those past its KV heads' taps
  // padding.
  const uint64_t taps =
      (uint64_t{g.attentionKvHeads} * shape.convolutionTaps + GGUF_TILE_COLUMNS - 1) / GGUF_TILE_COLUMNS *
      GGUF_TILE_COLUMNS;
  const uint64_t relative = uint64_t{g.attentionQueryHeads} * shape.relativeRank;
  const uint64_t stagedLaneRows = uint64_t{g.lanes} * (kv::kPageTokens - 1);
  std::vector<Tensor> result{
      {&Inkling2LoopScratch::stack, bytesFor<uint16_t>(schedule.stackPlanes() * hidden), 0, true},
      {&Inkling2LoopScratch::previous, bytesFor<uint16_t>(schedule.previousPlanes() * hidden), 0, true},
      {&Inkling2LoopScratch::residual, bytesFor<float>(hidden), 0, true},
      {&Inkling2LoopScratch::mixed, bytesFor<float>(kMixerQueries * hidden), 0, false},
      {&Inkling2LoopScratch::messages, bytesFor<float>(kMessages * hidden), 0, false},
      {&Inkling2LoopScratch::keyInput, bytesFor<uint16_t>(hidden), 0, false},
      {&Inkling2LoopScratch::valueInput, bytesFor<uint16_t>(hidden), 0, false},
      {&Inkling2LoopScratch::keys, bytesFor<uint16_t>(kv), 0, false},
      {&Inkling2LoopScratch::values, bytesFor<uint16_t>(kv), 0, false},
      {&Inkling2LoopScratch::keyTaps, bytesFor<uint16_t>(taps), 0, false},
      {&Inkling2LoopScratch::valueTaps, bytesFor<uint16_t>(taps), 0, false},
      {&Inkling2LoopScratch::convolvedKeys, bytesFor<uint16_t>(kv), 0, false},
      {&Inkling2LoopScratch::convolvedValues, bytesFor<uint16_t>(kv), 0, false},
      {&Inkling2LoopScratch::relativeStates, bytesFor<float>(relative), bytesFor<float>(stagedLaneRows * relative),
       false},
      {&Inkling2LoopScratch::rowSinks, bytesFor<float>(g.attentionQueryHeads),
       bytesFor<float>(stagedLaneRows * g.attentionQueryHeads), false},
      {&Inkling2LoopScratch::gateConcat, bytesFor<uint16_t>(2 * hidden), 0, false},
      {&Inkling2LoopScratch::gateLow, bytesFor<uint16_t>(shape.gateRank), 0, false},
      {&Inkling2LoopScratch::gateForget, bytesFor<float>(hidden), 0, false},
      {&Inkling2LoopScratch::gateGain, bytesFor<float>(hidden), 0, false},
      {&Inkling2LoopScratch::gateCounters, 0, ops::MessageGate::kCountersBytes, false},
      {&Inkling2LoopScratch::zeros, bytesFor<uint16_t>(hidden), 0, false},
  };
  // A decode step's convolution inputs of every application, which its
  // commit reads after the last (ops::ShortConvolution::addCommit).
  if (decode) result.push_back({&Inkling2LoopScratch::tape, bytesFor<uint16_t>(schedule.stateColumns()), 0, false});
  return result;
}

uint64_t bytesOf(const Tensor &tensor, uint64_t rows) noexcept { return tensor.rowBytes * rows + tensor.fixedBytes; }

uint32_t stepRows(const QwenTargetGeometry &g, bool decode) {
  return decode ? decodeStorageRows(g.lanes) : kChunkRows;
}

// The tensors placed one after another, a carried one's once per chunk.
std::vector<Inkling2LoopSteps::Placed> place(const QwenTargetGeometry &g, const Inkling2LoopShape &shape, bool decode,
                                             uint32_t chunks, uint64_t &bytes) {
  std::vector<Inkling2LoopSteps::Placed> placed;
  bytes = 0;
  for (const Tensor &tensor : tensors(g, shape, decode)) {
    placed.push_back({tensor.field, tensor.rowBytes, tensor.fixedBytes, tensor.carried, bytes});
    bytes += alignUp(bytesOf(tensor, stepRows(g, decode))) * (tensor.carried && !decode ? chunks : 1);
  }
  return placed;
}

// Throws unless `scratch` holds the relative attention's staging of the most
// staged rows a command of `rows` rows over the runtime's lanes takes: every
// lane but the last a row, each sequence staged in whole KV blocks
// (Runtime's chunkParams), so each lane but one stages a block for a row.
void requireStaging(const Inkling2LoopScratch &scratch, const QwenTargetGeometry &g, uint32_t rows) {
  std::vector<kv::ChunkedPrefillParams> sequences;
  for (uint32_t lane = 0; lane < g.lanes; ++lane) {
    const uint32_t count = lane + 1 < g.lanes ? 1 : rows - (g.lanes - 1);
    const uint32_t stride = (count + kv::kPageTokens - 1) / kv::kPageTokens * kv::kPageTokens;
    sequences.push_back(ops::PagedAttention::chunkParams(0, count, stride, stride / kv::kPageTokens));
  }
  const ops::RelativeAttentionStagingBytes needed = ops::RelativeAttention::stagingBytes(sequences, g.kvLayout);
  if (scratch.relativeStates.sizeBytes() < needed.states || scratch.rowSinks.sizeBytes() < needed.sinks)
    throw std::logic_error("Inkling-2-Mini's step scratch holds less than its relative attention stages");
}

} // namespace

Inkling2LoopSteps::Inkling2LoopSteps(const QwenTargetGeometry &geometry, const Inkling2LoopShape &shape,
                                     metal::MetalBackend &backend, uint32_t prefillChunks)
    : geometry_(geometry), shape_(shape), backend_(backend), prefillChunks_(prefillChunks) {
  // The layers a step runs are the loop's applications.
  const Inkling2LoopSchedule schedule(shape_);
  if (!prefillChunks_ || !schedule.valid() || schedule.applications() != geometry_.layers)
    throw std::invalid_argument("invalid Inkling-2-Mini step geometry");
  uint64_t prefillBytes = 0, decodeBytes = 0;
  prefillTensors_ = place(geometry_, shape_, false, prefillChunks_, prefillBytes);
  decodeTensors_ = place(geometry_, shape_, true, 1, decodeBytes);
  prefill_ = backend_.allocateBuffer(prefillBytes, metal::BufferStorage::Shared, "inkling2-prefill-scratch");
  decode_ = backend_.allocateBuffer(decodeBytes, metal::BufferStorage::Shared, "inkling2-decode-scratch");
  // The rows the MoE adds its weighted sum to, which nothing writes, and the
  // fused gate's counters, which each gate leaves zero.
  for (const Inkling2LoopScratch &scratch : {prefillScratch(0), verifyScratch()})
    for (const metal::MetalBuffer &zeros : {scratch.zeros, scratch.gateCounters})
      std::memset(zeros.contents(), 0, zeros.sizeBytes());
  requireStaging(prefillScratch(0), geometry_, kChunkRows);
  requireStaging(verifyScratch(), geometry_, ExecutionLimits::stepRows(geometry_.lanes));
}

QwenTargetScratchBytes Inkling2LoopSteps::scratchBytes(const QwenTargetGeometry &geometry,
                                                       const Inkling2LoopShape &shape, uint32_t prefillChunks) {
  QwenTargetScratchBytes bytes;
  static_cast<void>(place(geometry, shape, false, prefillChunks, bytes.prefill));
  static_cast<void>(place(geometry, shape, true, 1, bytes.decode));
  return bytes;
}

QwenTargetScratchBytes Inkling2LoopSteps::allocatedBytes() const noexcept {
  return {prefill_.sizeBytes(), decode_.sizeBytes()};
}

Inkling2LoopScratch Inkling2LoopSteps::prefillScratch(uint32_t index) const {
  if (index >= prefillChunks_) throw std::out_of_range("Inkling-2-Mini prefill chunk");
  Inkling2LoopScratch scratch{.planeRows = kChunkRows};
  for (const Placed &tensor : prefillTensors_) {
    const uint64_t bytes = tensor.rowBytes * kChunkRows + tensor.fixedBytes;
    scratch.*tensor.field =
        backend_.view(prefill_, tensor.offset + (tensor.carried ? index * alignUp(bytes) : 0), bytes);
  }
  return scratch;
}

Inkling2LoopScratch Inkling2LoopSteps::verifyScratch() const {
  const uint32_t rows = decodeStorageRows(geometry_.lanes);
  Inkling2LoopScratch scratch{.planeRows = rows};
  for (const Placed &tensor : decodeTensors_)
    scratch.*tensor.field = backend_.view(decode_, tensor.offset, tensor.rowBytes * rows + tensor.fixedBytes);
  return scratch;
}

} // namespace splash::model

#include "model/Inkling2LoopSteps.hpp"

#include "Checked.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "ops/MessageGate.hpp"
#include "ops/RelativeAttention.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace splash::model {
namespace {

constexpr uint32_t kChunkRows = ExecutionLimits::prefillTokenBudget;
// The most queries a block mix takes: a full layer's q, k and v inputs.
constexpr uint32_t kMixerQueries = 3;
// The messages of a loop layer: its q, k, v and FFN inputs'.
constexpr uint32_t kMessages = 4;
// A layer's mixer gates run as one call (ops::MessageGate): a decode step's
// fused gates take each its rank values, partial sums and gated rows, a step's
// rows each; a prompt chunk's of a decode step's rows at most take theirs from
// a chunk's single rows and kFusedRows rank rows, which never overlap the
// tiled gates' concatenation rows; a longer chunk's tiled gates keep each
// gate's gated rows in its own of the forget, gain and concatenation rows,
// fp32 rows of hiddenSize each.
static_assert(kMixerQueries <= ops::MessageGate::kMaximumGates, "a layer's mixer gates run as one call");
static_assert(decodeStorageRows(ExecutionLimits::maximumBatchWidth) <= ops::MessageGate::kFusedRows,
              "every decode step runs the fused message gates");
static_assert(kChunkRows >= kMixerQueries * ops::MessageGate::kFusedRows,
              "a prompt chunk's gated rows hold its fused gates' of their most rows");
static_assert(kChunkRows * sizeof(float) >=
                  ops::MessageGate::partialSumsBytes(kMixerQueries, 1, ops::MessageGate::kFusedRows),
              "a prompt chunk's gain rows hold its fused gates' partial sums");

template <class T> constexpr uint64_t bytesFor(uint64_t elements) noexcept { return elements * sizeof(T); }

uint32_t stepRows(const QwenTargetGeometry &g, bool decode) {
  return decode ? decodeStorageRows(g.lanes) : kChunkRows;
}

// Places a prefill chunk's or a decode step's scratch tensors. Persistent
// tensors keep disjoint bytes: carried tensors (stack, previous, residual)
// take one aligned slice per prefill chunk, read-only or reset-to-zero buffers
// (zeros, gateCounters, attentionCounters) and decode's tape stay untouched,
// messages live across attention into the FFN gate, and keyInput/valueInput
// bridge mixer inputs into attention projections. Two mutually exclusive
// transient phases — Gate (block mixes and message gates, with prefill's
// tiled gateConcat and fused gateLow overlaid) and Attention (keys/values or
// decode's paired keyValues, dynamic taps, convolved K/V and prefill's
// relative staging) — share one page-aligned region after them (saving 20.6
// MiB of prefill scratch on Inkling-2-Mini).
std::vector<Inkling2LoopSteps::Placed> place(const QwenTargetGeometry &g, const Inkling2LoopShape &shape, bool decode,
                                             uint32_t chunks, uint64_t &bytes) {
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
  const uint32_t rows = stepRows(g, decode);

  std::vector<Inkling2LoopSteps::Placed> placed;
  const auto add = [&](metal::MetalBuffer Inkling2LoopScratch::*field, uint64_t rowBytes, uint64_t fixedBytes,
                       bool carried, uint64_t offset) -> uint64_t {
    placed.push_back({field, rowBytes, fixedBytes, carried, offset});
    const uint64_t perSlice = alignUp(checkedAdd(checkedMultiply(rowBytes, rows, "Inkling-2-Mini scratch"), fixedBytes,
                                                 "Inkling-2-Mini scratch"));
    return checkedMultiply(perSlice, carried && !decode ? chunks : 1, "Inkling-2-Mini scratch");
  };
  const auto append = [&](uint64_t &cursor, metal::MetalBuffer Inkling2LoopScratch::*field, uint64_t rowBytes,
                          uint64_t fixedBytes, bool carried) {
    cursor = checkedAdd(cursor, add(field, rowBytes, fixedBytes, carried, cursor), "Inkling-2-Mini scratch");
  };

  uint64_t cursor = 0;
  append(cursor, &Inkling2LoopScratch::stack, bytesFor<uint16_t>(schedule.stackPlanes() * hidden), 0, true);
  append(cursor, &Inkling2LoopScratch::previous, bytesFor<uint16_t>(schedule.previousPlanes() * hidden), 0, true);
  append(cursor, &Inkling2LoopScratch::residual, bytesFor<float>(hidden), 0, true);
  append(cursor, &Inkling2LoopScratch::messages, bytesFor<float>(kMessages * hidden), 0, false);
  append(cursor, &Inkling2LoopScratch::keyInput, bytesFor<uint16_t>(hidden), 0, false);
  append(cursor, &Inkling2LoopScratch::valueInput, bytesFor<uint16_t>(hidden), 0, false);
  append(cursor, &Inkling2LoopScratch::gateCounters, 0, ops::MessageGate::kCountersBytes, false);
  append(cursor, &Inkling2LoopScratch::zeros, bytesFor<uint16_t>(hidden), 0, false);
  if (decode) {
    append(cursor, &Inkling2LoopScratch::tape, bytesFor<uint16_t>(schedule.stateColumns()), 0, false);
    append(cursor, &Inkling2LoopScratch::attentionCounters, 0, ops::RelativeAttention::kCountersBytes, false);
  }

  const uint64_t transient = cursor;
  uint64_t gate = transient;
  append(gate, &Inkling2LoopScratch::mixed, bytesFor<float>(kMixerQueries * hidden), 0, false);
  append(gate, &Inkling2LoopScratch::gateForget, bytesFor<float>((decode ? kMixerQueries : 1) * hidden), 0, false);
  if (decode) {
    append(gate, &Inkling2LoopScratch::gateGain, 0, ops::MessageGate::partialSumsBytes(kMixerQueries, hidden, rows),
           false);
    append(gate, &Inkling2LoopScratch::gateLow, bytesFor<uint16_t>(kMixerQueries * shape.gateRank), 0, false);
  } else {
    append(gate, &Inkling2LoopScratch::gateGain, bytesFor<float>(hidden), 0, false);
    // In prefill, gateConcat is read only past kFusedRows and gateLow only up
    // to kFusedRows (ops::MessageGate::add).
    const uint64_t concatBytes = add(&Inkling2LoopScratch::gateConcat, bytesFor<uint16_t>(2 * hidden), 0, false, gate);
    const uint64_t lowBytes =
        add(&Inkling2LoopScratch::gateLow, 0,
            bytesFor<uint16_t>(kMixerQueries * shape.gateRank) * ops::MessageGate::kFusedRows, false, gate);
    gate = checkedAdd(gate, std::max(concatBytes, lowBytes), "Inkling-2-Mini scratch");
  }

  uint64_t attention = transient;
  const uint64_t keysBytes = add(&Inkling2LoopScratch::keys, bytesFor<uint16_t>(kv), 0, false, attention);
  const uint64_t valuesBytes =
      add(&Inkling2LoopScratch::values, bytesFor<uint16_t>(kv), 0, false, checkedAdd(attention, keysBytes, "Inkling-2-Mini scratch"));
  uint64_t kvBytes = checkedAdd(keysBytes, valuesBytes, "Inkling-2-Mini scratch");
  // In decode, a layer projects either paired keyValues or separate keys and
  // values, never both (Inkling2LoopTarget::addAttention).
  if (decode) kvBytes = std::max(kvBytes, add(&Inkling2LoopScratch::keyValues, bytesFor<uint16_t>(2 * kv), 0, false, attention));
  attention = checkedAdd(attention, kvBytes, "Inkling-2-Mini scratch");
  append(attention, &Inkling2LoopScratch::keyTaps, bytesFor<uint16_t>(taps), 0, false);
  append(attention, &Inkling2LoopScratch::valueTaps, bytesFor<uint16_t>(taps), 0, false);
  append(attention, &Inkling2LoopScratch::convolvedKeys, bytesFor<uint16_t>(kv), 0, false);
  append(attention, &Inkling2LoopScratch::convolvedValues, bytesFor<uint16_t>(kv), 0, false);
  // A prompt chunk's attention stages its relative states and row sinks in the
  // queries' staged rows: a step's rows and, each sequence's rows taking whole
  // KV blocks, the rest of a block per lane (RuntimeArenas.cpp
  // targetTensorBytes); a decode step's prepares them in its single pass.
  if (!decode) {
    append(attention, &Inkling2LoopScratch::relativeStates, bytesFor<float>(relative),
           bytesFor<float>(stagedLaneRows * relative), false);
    append(attention, &Inkling2LoopScratch::rowSinks, bytesFor<float>(g.attentionQueryHeads),
           bytesFor<float>(stagedLaneRows * g.attentionQueryHeads), false);
  }

  bytes = std::max(gate, attention);
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
  // fused gate's counters and a decode step's attention's, which each gate
  // and each layer's attention leaves zero.
  for (const Inkling2LoopScratch &scratch : {prefillScratch(0), verifyScratch()})
    for (const metal::MetalBuffer &zeros : {scratch.zeros, scratch.gateCounters})
      std::memset(zeros.contents(), 0, zeros.sizeBytes());
  const metal::MetalBuffer attentionCounters = verifyScratch().attentionCounters;
  std::memset(attentionCounters.contents(), 0, attentionCounters.sizeBytes());
  requireStaging(prefillScratch(0), geometry_, kChunkRows);
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

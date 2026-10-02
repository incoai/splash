#pragma once

// Qwen3.8-Flash-Next's own operators (kernels/shared/qwen4.metal): the
// hyper-connection mixes around every block and the PLE n-gram embedding's
// gate and dilated convolution. The projections between them run through
// ops::Linear; model/QwenTarget.cpp composes them.

#include "metal/CommandGraph.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>

struct Qwen4PleHashParams;

namespace splash::ops {

struct HyperShape final {
  uint32_t hidden = 0;
  uint32_t streams = 0;
  uint32_t rank = 0;
  [[nodiscard]] constexpr uint32_t width() const noexcept { return hidden * streams; }
};

struct PleShape final {
  uint32_t hidden = 0;
  uint32_t streams = 0;
  uint32_t taps = 0;
  uint32_t dilation = 0;
  [[nodiscard]] constexpr uint32_t width() const noexcept { return hidden * streams; }
  [[nodiscard]] constexpr uint32_t history() const noexcept { return (taps - 1) * dilation; }
};

// QSA of one sequence's (or lane's) rows: `rows` query rows from logical
// position `firstPosition`, which attend sparsely past topBlocks complete
// blocks; bitmaps of maskWords words per row.
struct QsaRows final {
  uint32_t rows = 0;
  uint32_t firstPosition = 0;
  uint32_t topBlocks = 0;
  uint32_t maskWords = 0;
  uint32_t scoreStride = 0;
};

struct Qwen4 final {
  // Whether any row of `rows` from `firstPosition` attends sparsely: has
  // more than topBlocks complete blocks of four.
  [[nodiscard]] static bool qsaSparse(uint32_t firstPosition, uint32_t rows, uint32_t topBlocks) noexcept {
    return (uint64_t{firstPosition} + rows) / 4 > topBlocks;
  }
  // The indexer's query heads, normalized and rotated (fp32 rows of
  // QWEN4_QSA_HEADS heads), and the raw keys stored at the rows' slots.
  static void addQsaProject(metal::CommandGraph &graph, metal::MetalBuffer query, metal::MetalBuffer key,
                            const NormWeights &queryNorm, metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
                            metal::MetalBuffer pageTable, metal::MetalBuffer rawKeys, metal::MetalBuffer queries,
                            QsaRows rows);
  // The pooled keys of `blocks` blocks (Qwen4QsaBlock records).
  static void addQsaPool(metal::CommandGraph &graph, metal::MetalBuffer blocks, uint32_t count,
                         metal::MetalBuffer rawKeys, const NormWeights &keyNorm,
                         metal::MetalBuffer inverseFrequencies, metal::MetalBuffer pageTable,
                         metal::MetalBuffer pooledKeys);
  // Rows' bitmaps (mask rows rowOffset..) from their scores, scored in
  // batches of the score scratch's rows.
  static void addQsaSelect(metal::CommandGraph &graph, metal::MetalBuffer queries, metal::MetalBuffer pooledKeys,
                           metal::MetalBuffer pageTable, metal::MetalBuffer scores, uint32_t scoreRows,
                           metal::MetalBuffer mask, QsaRows rows);

  // streams[r] = rows of fp32 [streams][hidden], each a copy of embedding[r].
  static void addInit(metal::CommandGraph &graph, metal::MetalBuffer embedding, metal::MetalBuffer streams,
                      HyperShape shape, uint32_t rows);
  // The normalized streams (bf16) and, with inject, the injection weights
  // (fp32 [rows][streams]).
  static void addNorm(metal::CommandGraph &graph, metal::MetalBuffer streams, const NormWeights &norm,
                      metal::MetalBuffer inject, metal::MetalBuffer normalized, metal::MetalBuffer weights,
                      HyperShape shape, uint32_t rows);
  // silu(low / streams) in place.
  static void addLow(metal::CommandGraph &graph, metal::MetalBuffer low, HyperShape shape, uint32_t rows);
  static void addMix(metal::CommandGraph &graph, metal::MetalBuffer normalized, metal::MetalBuffer gate,
                     metal::MetalBuffer mixed, HyperShape shape, uint32_t rows);
  static void addCombine(metal::CommandGraph &graph, metal::MetalBuffer streams, metal::MetalBuffer branch,
                         metal::MetalBuffer weights, HyperShape shape, uint32_t rows);

  static void addPleGate(metal::CommandGraph &graph, metal::MetalBuffer key, metal::MetalBuffer value,
                         metal::MetalBuffer streams, const NormWeights &keyNorm, const NormWeights &queryNorm,
                         const NormWeights &convolutionNorm, metal::MetalBuffer gated,
                         metal::MetalBuffer convolutionInput, PleShape shape, uint32_t rows);
  // One sequence's rows; writes the history after them unless historyOut is empty.
  static void addPleConvolution(metal::CommandGraph &graph, metal::MetalBuffer convolutionInput,
                                metal::MetalBuffer historyIn, metal::MetalBuffer historyOut,
                                metal::MetalBuffer taps, metal::MetalBuffer gated, metal::MetalBuffer streams,
                                PleShape shape, uint32_t rows);
  // The MTP head (metal/abi/Qwen4.h Qwen4MtpParams). A draft step's tokens:
  // each lane's anchor and its first limit - 1 proposals, then repeats.
  static void addMtpTokens(metal::CommandGraph &graph, metal::MetalBuffer anchors, metal::MetalBuffer proposals,
                           metal::MetalBuffer tokens, uint32_t lanes, uint32_t laneRows, uint32_t proposalsPerLane,
                           uint32_t limit);
  // One lane's (or sequence's) `rows` eh-projection inputs, rows rowOffset..
  // of each stream's `totalRows` rows of `output`.
  static void addMtpInput(metal::CommandGraph &graph, metal::MetalBuffer embedding, metal::MetalBuffer carried,
                          metal::MetalBuffer history, const NormWeights &embedNorm, const NormWeights &hiddenNorm,
                          metal::MetalBuffer output, HyperShape shape, uint32_t rows, uint32_t limit,
                          uint32_t rowOffset, uint32_t totalRows);
  static void addMtpWiden(metal::CommandGraph &graph, metal::MetalBuffer projected, metal::MetalBuffer streams,
                          HyperShape shape, uint32_t rows);
  static void addMtpArgmax(metal::CommandGraph &graph, metal::MetalBuffer logits, metal::MetalBuffer proposals,
                           metal::MetalBuffer candidates, uint32_t lanes, uint32_t laneRows, uint32_t vocabulary,
                           uint32_t proposalsPerLane, uint32_t candidatesPerProposal, uint32_t row, bool last);
  // Row retained - 1 of `streams` (retained a GPU count) or, without
  // `retained`, row `rows` - 1.
  static void addMtpCarry(metal::CommandGraph &graph, metal::MetalBuffer streams, metal::MetalBuffer carried,
                          metal::MetalBuffer retained, HyperShape shape, uint32_t rows);
  static void addPleHash(metal::CommandGraph &graph, metal::MetalBuffer tokens, metal::MetalBuffer before,
                         metal::MetalBuffer indices, const Qwen4PleHashParams &params, uint32_t heads);

  // The history after a lane's `retained` (a GPU count) rows.
  static void addPleCommit(metal::CommandGraph &graph, metal::MetalBuffer convolutionInput,
                           metal::MetalBuffer historyIn, metal::MetalBuffer historyOut,
                           metal::MetalBuffer retained, PleShape shape);
};

} // namespace splash::ops

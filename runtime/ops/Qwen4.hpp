#pragma once

// Qwen3.8-Flash-Next's own operators: the hyper-connection mixes around every
// block and the PLE n-gram embedding's gate and dilated convolution
// (kernels/shared/qwen4.metal), and the QSA indexer that picks the blocks its
// full-attention layers attend (kernels/shared/qwen4_qsa.metal). The
// projections between them run through ops::Linear; model/QwenTarget.cpp
// composes them.

#include "metal/CommandGraph.hpp"
#include "metal/abi/Qwen4.h"
#include "ops/Normalization.hpp"
#include "ops/RowCopy.hpp"

#include <cstdint>
#include <span>

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

// QSA of a dispatch's lanes (a verify step's, or a prefill sequence as one):
// `rows` query rows of each lane, lane l's from logical position
// firstPositions[l] and its KV through pageTables[l], which attend sparsely
// past topBlocks complete blocks; bitmaps of maskWords words and scores of
// scoreStride per row, which cover each row's complete blocks and its tail.
struct QsaRows final {
  uint32_t rows = 0;
  std::span<const uint32_t> firstPositions;
  std::span<const metal::MetalBuffer> pageTables;
  uint32_t topBlocks = 0;
  uint32_t maskWords = 0;
  uint32_t scoreStride = 0;
};

// The QSA blocks a dispatch's lanes complete, which the pool pools: lane l's
// counts[l] Qwen4QsaBlock records from record l * laneRecords of `records`,
// its KV through pageTables[l].
struct QsaBlocks final {
  metal::MetalBuffer records;
  std::span<const uint32_t> counts;
  uint32_t laneRecords = 0;
  std::span<const metal::MetalBuffer> pageTables;
};

// Where a layer's QSA index tensors sit in every KV extent, in bytes
// (kv::PageStorage::indexPlacement). Page tables hold SplashKvPage entries.
struct QsaIndex final {
  uint32_t keysOffset = 0;
  uint32_t pooledOffset = 0;
};

struct Qwen4 final {
  // The complete blocks of QWEN4_QSA_BLOCK tokens before the last of `rows`
  // rows from `firstPosition`, its own included.
  [[nodiscard]] static constexpr uint64_t qsaCompleteBlocks(uint64_t firstPosition, uint32_t rows) noexcept {
    return (firstPosition + rows) / QWEN4_QSA_BLOCK;
  }
  // Whether any row of `rows` from `firstPosition` attends sparsely: has
  // more than topBlocks complete blocks.
  [[nodiscard]] static bool qsaSparse(uint32_t firstPosition, uint32_t rows, uint32_t topBlocks) noexcept {
    return qsaCompleteBlocks(firstPosition, rows) > topBlocks;
  }
  // The indexer's query heads, normalized and rotated (bf16 rows of
  // QWEN4_QSA_HEADS heads), and the raw keys stored at the rows' slots.
  static void addQsaProject(metal::CommandGraph &graph, metal::MetalBuffer query, metal::MetalBuffer key,
                            const NormWeights &queryNorm, metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
                            QsaIndex index, metal::MetalBuffer queries, const QsaRows &rows);
  // The pooled keys of the lanes' blocks.
  static void addQsaPool(metal::CommandGraph &graph, const QsaBlocks &blocks, const NormWeights &keyNorm,
                         metal::MetalBuffer inverseFrequencies, QsaIndex index);
  // The rows' scores of their complete blocks' pooled keys and, from them,
  // their bitmaps. A dispatch of one lane scores its rows in batches of the
  // score scratch's scoreRows rows; one of several lanes scores every lane's
  // rows at once, which the scratch must hold.
  static void addQsaSelect(metal::CommandGraph &graph, metal::MetalBuffer queries, QsaIndex index,
                           metal::MetalBuffer scores, uint32_t scoreRows, metal::MetalBuffer mask,
                           const QsaRows &rows);

  // streams[r] = rows of fp32 [streams][hidden], each a copy of embedding[r].
  static void addInit(metal::CommandGraph &graph, metal::MetalBuffer embedding, metal::MetalBuffer streams,
                      HyperShape shape, uint32_t rows);
  // The normalized streams (bf16) and, with inject, the injection weights
  // (fp32 [rows][streams]), of Flash-Next's shape, for which its kernels are
  // compiled (QWEN4_NORM_HIDDEN). With a table `layout` (decode rows), also
  // the table in `scratch` that the projection reading the output takes,
  // which then prepares none (ops::Linear PreparedInput).
  static PreparedInput addNorm(metal::CommandGraph &graph, metal::MetalBuffer streams, const NormWeights &norm,
                               metal::MetalBuffer inject, metal::MetalBuffer normalized, metal::MetalBuffer weights,
                               HyperShape shape, uint32_t rows, LinearScratch scratch = {},
                               LinearInput layout = LinearInput::Plain);
  // silu(low / streams) in place or, with a table layout, into the table
  // alone, leaving `low` as it was.
  static PreparedInput addLow(metal::CommandGraph &graph, metal::MetalBuffer low, HyperShape shape, uint32_t rows,
                              LinearScratch scratch = {}, LinearInput layout = LinearInput::Plain);
  static PreparedInput addMix(metal::CommandGraph &graph, metal::MetalBuffer normalized, metal::MetalBuffer gate,
                              metal::MetalBuffer mixed, HyperShape shape, uint32_t rows, LinearScratch scratch = {},
                              LinearInput layout = LinearInput::Plain);
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
  // The mean of each of `rows` rows' streams, as bf16 into `destination` at
  // `to`: the capture of a layer's output for the DFlash2 draft.
  static void addStreamMean(metal::CommandGraph &graph, metal::MetalBuffer streams, metal::MetalBuffer destination,
                            RowRegion to, HyperShape shape, uint32_t rows);
  static void addPleHash(metal::CommandGraph &graph, metal::MetalBuffer tokens, metal::MetalBuffer before,
                         metal::MetalBuffer indices, const Qwen4PleHashParams &params, uint32_t heads);

  // The history after the first `rows` rows of one sequence's prefill (a
  // state captured inside its command).
  static void addPleHistory(metal::CommandGraph &graph, metal::MetalBuffer convolutionInput,
                            metal::MetalBuffer historyIn, metal::MetalBuffer historyOut, PleShape shape,
                            uint32_t rows);
  // The history after a lane's `retained` (a GPU count) rows.
  static void addPleCommit(metal::CommandGraph &graph, metal::MetalBuffer convolutionInput,
                           metal::MetalBuffer historyIn, metal::MetalBuffer historyOut,
                           metal::MetalBuffer retained, PleShape shape);
};

} // namespace splash::ops

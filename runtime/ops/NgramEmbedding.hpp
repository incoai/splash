#pragma once

// The n-gram embedding (kernels/shared/ngram_embedding.metal),
// Qwen3.8-Flash-Next's PLE: the GPU's hash of a step's n-grams into table
// rows, and the gate and dilated convolution through which the rows'
// projections reach each stream of a hyper-connected residual
// (ops/HyperConnection.hpp). Its table's gather is ops::Embedding's and its
// projections run through ops::Linear; model/Qwen4ExpTarget.cpp composes
// them.

#include "metal/CommandGraph.hpp"
#include "metal/abi/NgramEmbedding.h"
#include "metal/abi/VerifyTree.h"
#include "ops/LaneRows.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <span>

namespace splash::ops {

// The gate and convolution over `streams` streams of `hidden` values: `taps`
// taps `dilation` rows apart.
struct NgramShape final {
  uint32_t hidden = 0;
  uint32_t streams = 0;
  uint32_t taps = 0;
  uint32_t dilation = 0;
  [[nodiscard]] constexpr uint32_t width() const noexcept { return hidden * streams; }
  [[nodiscard]] constexpr uint32_t history() const noexcept { return (taps - 1) * dilation; }
};

// The buffers of a verify step's n-gram hash: each of the step's rows' token
// and tree row (metal/abi/VerifyTree.h), each lane's rows of the step
// (SplashLaneRows), which the step's proposer writes, the ngram - 1 tokens
// before each lane's first row (NgramHashParams), and the table rows it
// writes for each of the step's rows.
struct NgramHashBuffers final {
  metal::MetalBuffer tokens;
  metal::MetalBuffer tree;
  metal::MetalBuffer laneRows;
  metal::MetalBuffer before;
  metal::MetalBuffer indices;
};

struct NgramEmbedding final {
  static void addGate(metal::CommandGraph &graph, metal::MetalBuffer key, metal::MetalBuffer value,
                      metal::MetalBuffer streams, const NormWeights &keyNorm, const NormWeights &queryNorm,
                      const NormWeights &convolutionNorm, metal::MetalBuffer gated,
                      metal::MetalBuffer convolutionInput, NgramShape shape, uint32_t rows);
  // The convolution of a command's sequences' rows (a prompt chunk's
  // sequences, or a verify step's lanes) as its layout holds them (`layout`,
  // a SplashLaneRows a sequence, of `rows`), in one dispatch whose grid spans
  // the sequences' bound: sequence s's taps before its first row read
  // historyIn[s], and with historyOut its history after its rows goes to
  // historyOut[s]. With `tree`, the rows' places (a SplashTreeRow a row) of
  // verify blocks of at most 32 rows, whose taps follow each row's path; the
  // commit then writes their history (addCommit).
  static void addConvolution(metal::CommandGraph &graph, metal::MetalBuffer convolutionInput,
                             std::span<const metal::MetalBuffer> historyIn,
                             std::span<const metal::MetalBuffer> historyOut, metal::MetalBuffer taps,
                             metal::MetalBuffer gated, metal::MetalBuffer streams, NgramShape shape,
                             metal::MetalBuffer layout, const LayoutRows &rows, metal::MetalBuffer tree = {});
  // The history after the first `rows` rows of one sequence's prefill (a
  // state captured inside its command).
  static void addHistory(metal::CommandGraph &graph, metal::MetalBuffer convolutionInput,
                         metal::MetalBuffer historyIn, metal::MetalBuffer historyOut, NgramShape shape,
                         uint32_t rows);
  // The history after each lane's accepted path (`paths`, a row mask the GPU
  // wrote a lane, metal/abi/VerifyTree.h) of its verify block's rows in
  // convolutionInput, as the step's layout holds them (`layout`, a
  // SplashLaneRows a lane, of `rows`, at most 32 rows a lane), in one
  // dispatch: lane l's history before it historyIn[l], after it
  // historyOut[l].
  static void addCommit(metal::CommandGraph &graph, metal::MetalBuffer convolutionInput,
                        std::span<const metal::MetalBuffer> historyIn, std::span<const metal::MetalBuffer> historyOut,
                        metal::MetalBuffer paths, NgramShape shape, metal::MetalBuffer layout,
                        const LayoutRows &rows);
  // The table rows of each lane's rows of a verify step (`step`) by the hash
  // `params` describes, whose lane count it sets.
  static void addHash(metal::CommandGraph &graph, NgramHashBuffers buffers, NgramHashParams params,
                      const StepRows &step, uint32_t heads);
};

} // namespace splash::ops

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
  // One sequence's rows; writes the history after them unless historyOut is empty.
  static void addConvolution(metal::CommandGraph &graph, metal::MetalBuffer convolutionInput,
                             metal::MetalBuffer historyIn, metal::MetalBuffer historyOut, metal::MetalBuffer taps,
                             metal::MetalBuffer gated, metal::MetalBuffer streams, NgramShape shape, uint32_t rows);
  // The history after the first `rows` rows of one sequence's prefill (a
  // state captured inside its command).
  static void addHistory(metal::CommandGraph &graph, metal::MetalBuffer convolutionInput,
                         metal::MetalBuffer historyIn, metal::MetalBuffer historyOut, NgramShape shape,
                         uint32_t rows);
  // The history after a lane's `retained` (a GPU count) rows of its verify
  // block's `rows` in convolutionInput.
  static void addCommit(metal::CommandGraph &graph, metal::MetalBuffer convolutionInput,
                        metal::MetalBuffer historyIn, metal::MetalBuffer historyOut, metal::MetalBuffer retained,
                        NgramShape shape, uint32_t rows);
  // The table rows of each lane's rows of a verify step (`step`) by the hash
  // `params` describes, whose lane count it sets.
  static void addHash(metal::CommandGraph &graph, NgramHashBuffers buffers, NgramHashParams params,
                      const StepRows &step, uint32_t heads);
};

} // namespace splash::ops

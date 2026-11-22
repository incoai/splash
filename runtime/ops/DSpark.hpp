#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/DSpark.h"
#include "ops/DraftSelector.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace splash::ops {

// One consumer of a DSpark draft's target features (metal/abi/DSpark.h): its
// softmax weights of the SPLASH_DSPARK_TAPS features and its F32 gamma.
struct DSparkMix final {
  std::array<float, SPLASH_DSPARK_TAPS> alpha{};
  metal::MetalBuffer gamma;
};

// The Markov head's weights: the joint projection (bf16 [SPLASH_DSPARK_JOINT]
// rows of [state; prev embedding; hidden; f_rnn]) and its F32 bias, the
// previous token's embedding table (bf16 [vocabulary][SPLASH_DSPARK_RANK]),
// and its output projection folded into the vocabulary head: W2 = head x
// W_out (bf16 [vocabulary][SPLASH_DSPARK_RANK]) and b2 = head x b_out (F32
// [vocabulary]), so that a step's correction of the base logits is W2 o + b2.
struct DSparkMarkovWeights final {
  metal::MetalBuffer joint;
  metal::MetalBuffer jointBias;
  metal::MetalBuffer embedding;
  metal::MetalBuffer foldWeights;
  metal::MetalBuffer foldBias;
};

// The vocabulary a Markov chain scores: the head's rows (the base logits'
// width), the first `proposed` of which it may propose, and the cap of its
// corrected logits; and the draft's hidden width.
struct DSparkMarkovShape final {
  uint32_t hidden = 0;
  uint32_t vocabulary = 0;
  uint32_t proposed = 0;
  float cap = 0.0F;
};

// What a chain reads: each lane's block of SPLASH_DRAFT_QUERY_ROWS draft rows,
// its normed hidden rows (bf16) and their base logits (fp32 rows of the
// vocabulary), and its f_rnn row (bf16 [lanes][hidden]); its workspace
// (DSpark::markovWorkspace); and what it writes, the selector's lattice:
// each lane's candidates (uint32) and their logits (fp32), [lane][position]
// [SPLASH_DRAFT_CANDIDATES], the candidate it chose first.
struct DSparkMarkovBuffers final {
  metal::MetalBuffer hidden;
  metal::MetalBuffer base;
  metal::MetalBuffer frnn;
  metal::MetalBuffer joint;
  metal::MetalBuffer state;
  metal::MetalBuffer previous;
  metal::MetalBuffer logits;
  metal::MetalBuffer partialValues;
  metal::MetalBuffer partialIds;
  metal::MetalBuffer candidates;
  metal::MetalBuffer unary;
};

struct DSparkMarkovWorkspace final {
  uint64_t jointBytes = 0;
  uint64_t stateBytes = 0;
  uint64_t previousBytes = 0;
  uint64_t logitsBytes = 0;
  uint64_t partialValuesBytes = 0;
  uint64_t partialIdsBytes = 0;
};

// A DSpark draft's own kernels (kernels/shared/dspark_mix.metal,
// kernels/decode/dspark.metal).
class DSpark final {
public:
  // Rows of a consumer's context from `rows` captured rows of the
  // SPLASH_DSPARK_TAPS features of `width` values each (bf16
  // [rows][taps][width]) into bf16 rows of `width` values.
  static void addMix(metal::CommandGraph &graph, metal::MetalBuffer features, const DSparkMix &mix,
                     metal::MetalBuffer output, uint32_t rows, uint32_t width);
  // Each of `lanes` lanes' last retained row (`retained`, acceptance's mask,
  // of its rows `laneRows`, SplashLaneRows) of a decode step's f_rnn rows
  // (bf16, `width` values) into its f_rnn row, frnn[lane] (a lane's binding
  // past `lanes` is unread).
  static void addFrnnCommit(metal::CommandGraph &graph, metal::MetalBuffer rows,
                            std::span<const metal::MetalBuffer> frnn, metal::MetalBuffer retained,
                            metal::MetalBuffer laneRows, uint32_t lanes, uint32_t width);
  // The workspace of a chain of `lanes` lanes over a vocabulary of
  // `vocabulary` rows.
  [[nodiscard]] static DSparkMarkovWorkspace markovWorkspace(uint32_t lanes, uint32_t vocabulary);
  // Every lane's chain of SPLASH_DRAFT_PROPOSAL_TOKENS steps from its anchor,
  // each over the whole vocabulary (`lanes`: their anchors, temperatures and
  // noise): a greedy lane's most likely token a step, a sampled lane's draw
  // from its step's candidates at its temperature.
  static void addMarkov(metal::CommandGraph &graph, const DSparkMarkovWeights &weights,
                        const DSparkMarkovBuffers &buffers, DSparkMarkovShape shape,
                        std::span<const DraftTreeLane> lanes);
  // Rows [first, first + rows) of the fold's rank + 2 rows, the vocabulary
  // head's projection of the output projection's columns and of its bias's
  // two bf16 parts (fp32 [rows][vocabulary], `projected`), into W2 and b2.
  static void addFold(metal::CommandGraph &graph, metal::MetalBuffer projected, const DSparkMarkovWeights &weights,
                      uint32_t first, uint32_t rows, uint32_t vocabulary);
};

} // namespace splash::ops

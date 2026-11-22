#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels: a DSpark
// draft's context mix and its Markov head (ops::DSpark,
// kernels/shared/dspark_mix.metal, kernels/decode/dspark.metal).
#include "metal/abi/ExecutionGeometry.h"
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// A DSpark draft reads four features of every row its target runs: the rows
// of three of the target's layers and its final normed row, its captured
// rows' four slots. Each consumer of them (a draft layer's context row, and
// the Markov head's context f_rnn) mixes the four by its own softmax weights
// and RMS-normalizes the mix by its own gamma.
#define SPLASH_DSPARK_TAPS 4u
// The Markov head: a recurrent state of SPLASH_DSPARK_RANK values that a step
// updates from the joint projection of [state; the previous token's
// embedding; the row's draft hidden; f_rnn] into its gate, candidate and
// output (SPLASH_DSPARK_JOINT values), the output's projection correcting the
// row's base logits over the whole vocabulary.
#define SPLASH_DSPARK_RANK 256u
#define SPLASH_DSPARK_JOINT (3u * SPLASH_DSPARK_RANK)
// The vocabulary pass of a Markov step runs in this many shards, each of
// which keeps its top SPLASH_DRAFT_CANDIDATES of every lane; a lane's merge
// takes the step's top candidates from them.
#define SPLASH_DSPARK_SHARDS 32u

// `rows` rows of SPLASH_DSPARK_TAPS features of `width` values each (bf16, a
// row's taps side by side), mixed by `alpha` and RMS-normalized by an F32
// gamma of `width` values into rows of `width` bf16 values.
struct DSparkMixParams {
  uint32_t rows;
  uint32_t width;
  float alpha[SPLASH_DSPARK_TAPS];
};

static_assert(sizeof(DSparkMixParams) == 24, "DSpark mix parameters are 24 bytes on both sides");

// A decode step's f_rnn rows of `width` bf16 values, of which each of `lanes`
// lanes keeps the last row it retained (the retained rows' mask among its
// rows, SplashLaneRows) as its context for its next draft.
struct DSparkCommitParams {
  uint32_t lanes;
  uint32_t width;
};

static_assert(sizeof(DSparkCommitParams) == 8, "DSpark commit parameters are 8 bytes on both sides");

// Markov step `step` (0-based: the draft's row step + 1, its proposal
// position) of `lanes` lanes, each lane's block of SPLASH_DRAFT_QUERY_ROWS
// rows: the draft's hidden rows (`hidden` values) and their base logits,
// rows of `vocabulary` values of which the first `proposed` are tokens the
// draft may propose. The corrected logits are capped as cap * tanh(x / cap).
// Step 0 starts every lane's state at zero and follows its anchor; a later
// step follows the token the step before chose. A greedy lane
// (temperature 0) chooses its most likely candidate; a sampled lane draws
// one from its candidates at its temperature with noise of its own (a
// 64-bit seed, low word first).
struct DSparkMarkovParams {
  uint32_t lanes;
  uint32_t step;
  uint32_t hidden;
  uint32_t vocabulary;
  uint32_t proposed;
  float cap;
  uint32_t anchor[SPLASH_MAXIMUM_BATCH_WIDTH];
  float temperature[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t noise[SPLASH_MAXIMUM_BATCH_WIDTH][2];
};

static_assert(sizeof(DSparkMarkovParams) == 24 + 16 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "DSpark Markov parameters are the same size on both sides");

// The fold of the Markov head's output projection into the vocabulary
// (dspark_fold_rows): rows [first, first + rows) of the `rank` + 2 rows the
// vocabulary head projected (the output projection's columns, then its bias
// as two bf16 parts), `vocabulary` fp32 values each, into the folded weights
// (bf16 [vocabulary][rank]) and bias (F32 [vocabulary]).
struct DSparkFoldParams {
  uint32_t first;
  uint32_t rows;
  uint32_t rank;
  uint32_t vocabulary;
};

static_assert(sizeof(DSparkFoldParams) == 16, "DSpark fold parameters are 16 bytes on both sides");

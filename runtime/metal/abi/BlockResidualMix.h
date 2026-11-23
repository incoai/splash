#pragma once

// Parameters of the block residual mix kernels (kernels/shared/block_residual_mix.metal,
// block_residual_mix_rows.metal, ops/BlockResidualMix.cpp): learned queries
// attend over a row's states, its block stack and its current rows, each
// RMS-normalized without weights, and each query's output is the
// softmax-weighted sum of the normalized states. fp32 on fp32 rows: a call of
// at most BLOCK_RESIDUAL_MIX_QUERY_ROWS rows (a decode step's) runs a
// threadgroup of BLOCK_RESIDUAL_MIX_THREADS threads a row and query, a call
// of more (a prompt chunk's) one of BLOCK_RESIDUAL_MIX_ROW_THREADS a row (the
// _rows kernels).
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// A decode step's mix is a few threadgroups, each a chain of dependent
// passes over its row: the more threads, the fewer columns each thread's
// pass takes. Inkling-2-Mini's 120 mixes of a 1-row decode step took 1.34 ms
// on an M5 Pro at 1024 threads (8 rows: 3.72 ms), against 3.10 ms (4.80 ms)
// with 256 threads a row and its queries' sums one at a time.
#define BLOCK_RESIDUAL_MIX_THREADS 1024u
// A prompt chunk's rows fill the GPU a row each: its mix runs a threadgroup
// of 256 threads a row, which computes the row's norms once for all its
// queries (block_residual_mix_rows.metal). A threadgroup a row and query
// computes them for every query, at four times the threads a row: a 2,048-row
// chunk's mixes took 33% longer that way (Inkling-2-Mini's 16K-token prompt
// on a 20-core M5 Pro: 588 ms against 440, its prefill 1.35% longer).
#define BLOCK_RESIDUAL_MIX_ROW_THREADS 256u
// The most rows a call runs a threadgroup a row and query for; more run the
// _rows kernels. On a 20-core M5 Pro one each of Inkling-2-Mini's six mix
// kinds (4, 3 and 1 queries over 2 to 5 states, bf16 and fp32 current rows)
// took, a row and query against a row: 32 rows 110 us against 143, 64 rows
// 200 against 222, 128 rows 375 against 353, 512 rows 1,407 against 1,199,
// 2,048 rows 5,511 against 4,633. A one-lane decode step (at most 16 rows)
// keeps a threadgroup a row and query.
#define BLOCK_RESIDUAL_MIX_QUERY_ROWS 64u
// The most states a mix reads, a stack of four blocks' inputs and the
// current rows, and the most queries it runs, four (the most outputs a mix
// of a model in the tree needs: four messages of the previous pass).
#define BLOCK_RESIDUAL_MIX_MAXIMUM_STATES 5u
#define BLOCK_RESIDUAL_MIX_MAXIMUM_QUERIES 4u
// The most outputs block_residual_mix_norm normalizes, each into its own
// rows with its own weights (an attention's q, k and v inputs).
#define BLOCK_RESIDUAL_MIX_MAXIMUM_NORMS 3u

// A mix of `rows` rows over `stack` bf16 stack states a row, planes of
// `rows`-row rows of `hidden` values `stack_plane` values apart (row r's
// state s at s * stack_plane + r * hidden), and the current rows, bf16 or
// fp32 (the _f32 instances) rows of `hidden` values, by `queries` queries of
// `hidden` values, the scores scaled by `scale` (hidden^-1/2):
// block_residual_mix writes each row's queries' outputs as fp32 rows
// [rows][queries][hidden], block_residual_mix_norm each query's through its
// own weighted RMS norm into its own bf16 rows.
struct BlockResidualMixParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t stack;
  uint32_t queries;
  uint32_t stack_plane;
  float scale;
};

static_assert(sizeof(BlockResidualMixParams) == 24, "block residual mix parameters are 24 bytes on both sides");

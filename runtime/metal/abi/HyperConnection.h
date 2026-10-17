#pragma once

// Parameters of the hyper-connection kernels (kernels/shared/hyper_connection.metal,
// ops/HyperConnection.cpp): a residual of `streams` parallel streams of
// `hidden` fp32 values per row, which a mix normalizes, gates by a low-rank
// projection of all of them and averages into a block's input, and to which
// the block's output is added back, scaled by each stream's injection weight.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// Threads of the per-row kernels: one threadgroup per row.
#define HYPER_CONNECTION_ROW_THREADS 256u
// The most streams a mix reads.
#define HYPER_CONNECTION_MAXIMUM_STREAMS 4u
// The mix norm's kernels, compiled for HYPER_CONNECTION_MAXIMUM_STREAMS
// streams of HYPER_CONNECTION_NORM_HIDDEN values (ops::HyperConnection::addNorm,
// Qwen3.8-Flash-Next's shape, which model/Qwen4Exp.hpp asserts):
// hyper_connection_norm_quarters, a threadgroup of
// HYPER_CONNECTION_NORM_QUARTER_THREADS threads per row, for an injecting mix
// of at most HYPER_CONNECTION_NORM_QUARTER_ROWS rows, and
// hyper_connection_norm_loaded, a threadgroup of HYPER_CONNECTION_ROW_THREADS
// per row, for the rest.
#define HYPER_CONNECTION_NORM_HIDDEN 2560u
#define HYPER_CONNECTION_NORM_QUARTER_THREADS (HYPER_CONNECTION_MAXIMUM_STREAMS * HYPER_CONNECTION_ROW_THREADS)
#define HYPER_CONNECTION_NORM_QUARTER_ROWS 64u

// A mix over `streams` streams of `hidden` values per row: its norm and
// injection (norm), its low-rank gate (low, `rank` values) and its average
// (mix); or the injection of a block's output (combine) and the streams'
// start (init).
struct HyperConnectionParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t streams;
  uint32_t rank;
  uint32_t inject; // norm: whether to compute injection weights
};

static_assert(sizeof(HyperConnectionParams) == 20,
              "hyper-connection parameters are 20 bytes on both sides");

// The mean of each of `rows` rows of `streams` streams of `hidden` fp32
// values (hyper_connection_stream_mean), written as bf16 at column `column`
// of row `row` + r of rows of `stride` values.
struct HyperStreamMeanParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t streams;
  uint32_t row;
  uint32_t stride;
  uint32_t column;
};

static_assert(sizeof(HyperStreamMeanParams) == 24,
              "hyper-connection stream mean parameters are 24 bytes on both sides");

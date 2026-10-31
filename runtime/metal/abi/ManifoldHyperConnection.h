#pragma once

// Parameters of the manifold-constrained hyper-connection kernels
// (kernels/shared/manifold_hyper_connection.metal,
// ops/ManifoldHyperConnection.cpp), DeepSeek-V4.1's residual: MHC_STREAMS
// parallel fp32 streams of `hidden` values per row. A sublayer's mix reads
// all of them and writes its MHC_MIXES weights: the pre weights that sum the
// streams into a sublayer's input, the post weights that scale its output
// into each stream, and the comb matrix, balanced by Sinkhorn iterations,
// that mixes the streams themselves.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#define MHC_STREAMS 4u
// A row's mix weights: MHC_STREAMS pre, MHC_STREAMS post, then the comb
// matrix, row i (the stream it reads) of column j (the stream it writes) at
// 2 * MHC_STREAMS + i * MHC_STREAMS + j.
#define MHC_MIXES (2u * MHC_STREAMS + MHC_STREAMS * MHC_STREAMS)
// Threads of every kernel's threadgroup.
#define MHC_THREADS 256u
// The rows a mix's threadgroup weighs together, reading the mix's
// projection once for all of them, and the columns of the projection it
// reads: a mix sums a row's dots over chunks of MHC_MIX_CHUNK columns on many
// cores (mhc_mix_partial), then each row's chunks (mhc_mix_finish), each
// chunk MHC_MIX_ROW_PARTIALS values, the row's dots with the projection's
// rows and its sum of squares.
#define MHC_MIX_ROWS 4u
#define MHC_MIX_CHUNK 512u
#define MHC_MIX_ROW_PARTIALS (MHC_MIXES + 1u)
// The most values a stream holds: a pre-norm thread keeps its share of a
// row's input in registers.
#define MHC_MAX_HIDDEN 8192u

struct ManifoldHyperParams {
  uint32_t rows;
  uint32_t hidden;
  // Sinkhorn iterations of the comb matrix, the first of them its softmax.
  uint32_t iterations;
  // The epsilon of the weights (hc_eps) and of the RMS norms.
  float hc_epsilon;
  float norm_epsilon;
};

static_assert(sizeof(ManifoldHyperParams) == 20,
              "manifold hyper-connection parameters are 20 bytes on both sides");

// Each row's streams averaged into bf16 columns from `offset` on of rows
// `stride` wide (mhc_stream_mean): a DSpark draft's view of the target's
// hidden rows at a layer's input.
struct MhcMeanParams {
  uint32_t rows;
  uint32_t hidden;
  uint32_t stride;
  uint32_t offset;
};

static_assert(sizeof(MhcMeanParams) == 16, "manifold hyper-connection mean parameters are 16 bytes on both sides");

#pragma once

// Manifold-constrained hyper-connections
// (kernels/shared/manifold_hyper_connection.metal), DeepSeek-V4.1's residual:
// MHC_STREAMS parallel fp32 streams per row. Each sublayer's mix weighs all
// of them (addMix); a sublayer's input is the streams summed by a mix's pre
// weights and normalized (addPreNorm), and its output enters each stream by
// the mix's post weights while the comb matrix mixes the streams
// (addPost). The streams start as copies of the token's embedding
// (HyperConnection::addInit), and the model's head reads them as a
// sublayer does (addPreNorm with the output norm).

#include "metal/CommandGraph.hpp"
#include "metal/abi/ManifoldHyperConnection.h"

#include <cstdint>

namespace splash::ops {

// MHC_STREAMS streams of `hidden` values per row, the Sinkhorn iterations of
// a mix's comb matrix, and the epsilons of its weights and of the RMS norms.
struct ManifoldHyperShape final {
  uint32_t hidden = 0;
  uint32_t iterations = 0;
  float hcEpsilon = 0.0f;
  float normEpsilon = 0.0f;
  [[nodiscard]] constexpr uint32_t width() const noexcept { return hidden * MHC_STREAMS; }
};

struct ManifoldHyperConnection final {
  // streams = fp32 [rows][MHC_STREAMS][hidden], each stream of a row a copy
  // of its embedding (bf16 [rows][hidden]).
  static void addInit(metal::CommandGraph &graph, metal::MetalBuffer embedding, metal::MetalBuffer streams,
                      ManifoldHyperShape shape, uint32_t rows);
  // mixes = rows of MHC_MIXES fp32 weights (metal/abi/ManifoldHyperConnection.h)
  // of `streams`, fp32 [rows][MHC_STREAMS][hidden], from the mix's
  // projection fn, fp32 [MHC_MIXES][MHC_STREAMS * hidden], its scale (3
  // values: pre, post, comb) and base (MHC_MIXES values). `chunks` holds
  // the partial sums between its passes (mixScratchBytes).
  static void addMix(metal::CommandGraph &graph, metal::MetalBuffer streams, metal::MetalBuffer fn,
                     metal::MetalBuffer scale, metal::MetalBuffer base, metal::MetalBuffer chunks,
                     metal::MetalBuffer mixes, ManifoldHyperShape shape, uint32_t rows);
  // The bytes of the partial sums a mix of `rows` rows takes.
  [[nodiscard]] static uint64_t mixScratchBytes(ManifoldHyperShape shape, uint32_t rows) noexcept;
  // output = bf16 [rows][hidden]: the streams summed by the pre weights of
  // `mixes`, RMS normalized and scaled by `norm` (fp32 [hidden]).
  static void addPreNorm(metal::CommandGraph &graph, metal::MetalBuffer streams, metal::MetalBuffer mixes,
                         metal::MetalBuffer norm, metal::MetalBuffer output, ManifoldHyperShape shape,
                         uint32_t rows);
  // out = bf16 [rows][stride]: each row's streams averaged into its columns
  // from `offset` on, `hidden` of them: a DSpark draft's view of the
  // target's hidden rows at a layer's input (model/DSpark.hpp).
  static void addStreamMean(metal::CommandGraph &graph, metal::MetalBuffer streams, metal::MetalBuffer out,
                            uint32_t stride, uint32_t offset, ManifoldHyperShape shape, uint32_t rows);
  // The streams, in place, after a sublayer whose output `branch` (bf16
  // [rows][hidden]) they take by the post weights and comb matrix of
  // `mixes`.
  static void addPost(metal::CommandGraph &graph, metal::MetalBuffer streams, metal::MetalBuffer branch,
                      metal::MetalBuffer mixes, ManifoldHyperShape shape, uint32_t rows);
};

} // namespace splash::ops

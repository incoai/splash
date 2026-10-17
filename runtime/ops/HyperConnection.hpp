#pragma once

// Hyper-connections (kernels/shared/hyper_connection.metal): a residual of
// several parallel fp32 streams per row, which a mix (norm, low, mix) joins
// into a block's input in place of a norm, and to which the block's output
// is added back, scaled by each stream's injection weight (combine).
// Qwen3.8-Flash-Next's (model/Qwen4ExpTarget.cpp) composes them with the
// mixes' projections, which run through ops::Linear.

#include "metal/CommandGraph.hpp"
#include "metal/abi/HyperConnection.h"
#include "ops/Normalization.hpp"
#include "ops/RowCopy.hpp"

#include <cstdint>

namespace splash::ops {

// `streams` streams of `hidden` values per row, and a mix's low rank.
struct HyperShape final {
  uint32_t hidden = 0;
  uint32_t streams = 0;
  uint32_t rank = 0;
  [[nodiscard]] constexpr uint32_t width() const noexcept { return hidden * streams; }
};

struct HyperConnection final {
  // streams[r] = rows of fp32 [streams][hidden], each a copy of embedding[r].
  static void addInit(metal::CommandGraph &graph, metal::MetalBuffer embedding, metal::MetalBuffer streams,
                      HyperShape shape, uint32_t rows);
  // The normalized streams (bf16) and, with inject, the injection weights
  // (fp32 [rows][streams]), of the shape its kernels are compiled for
  // (HYPER_CONNECTION_NORM_HIDDEN). With a table `layout` (decode rows), also
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
  // The block's input: the mean of the normalized streams gated by sigmoid of
  // the mix's up projection.
  static PreparedInput addMix(metal::CommandGraph &graph, metal::MetalBuffer normalized, metal::MetalBuffer gate,
                              metal::MetalBuffer mixed, HyperShape shape, uint32_t rows, LinearScratch scratch = {},
                              LinearInput layout = LinearInput::Plain);
  // Every stream plus the block's output scaled by the stream's injection
  // weight.
  static void addCombine(metal::CommandGraph &graph, metal::MetalBuffer streams, metal::MetalBuffer branch,
                         metal::MetalBuffer weights, HyperShape shape, uint32_t rows);
  // The mean of each of `rows` rows' streams, as bf16 into `destination` at
  // `to`: Qwen3.8-Flash-Next's capture of a layer's output for its DFlash2
  // draft.
  static void addStreamMean(metal::CommandGraph &graph, metal::MetalBuffer streams, metal::MetalBuffer destination,
                            RowRegion to, HyperShape shape, uint32_t rows);
};

} // namespace splash::ops

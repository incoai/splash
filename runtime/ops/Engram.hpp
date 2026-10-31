#pragma once

// DeepSeek-V4.1's Engram (kernels/shared/engram.metal): a row's hashed table
// rows, gathered and projected by ops::Embedding and ops::Linear into a key
// per stream and a value, gate the value into each stream of the manifold
// hyper-connected residual (ops/ManifoldHyperConnection.hpp). The rows'
// hashes come from the host, which reads the rows from the table's file.

#include "metal/CommandGraph.hpp"
#include "metal/abi/Engram.h"

#include <cstdint>

namespace splash::ops {

struct Engram final {
  // The streams (fp32 [rows][MHC_STREAMS][hidden]) in place, each plus the
  // value of `kv` (bf16 [rows][MHC_STREAMS + 1][hidden]: a key per stream,
  // then the value) gated by the stream's product with its key weighted by
  // `weights` (fp32 [MHC_STREAMS][hidden], engram_gate, the product of
  // DeepSeek's gate query and key), over both RMS norms and sqrt(hidden).
  static void addGate(metal::CommandGraph &graph, metal::MetalBuffer streams, metal::MetalBuffer kv,
                      metal::MetalBuffer weights, uint32_t hidden, float normEpsilon, uint32_t rows);
  // The table rows of a verify step's rows (EngramHashParams), into `rows`
  // (uint32 [layers][rows][(ngram - 1) * heads]): of their tokens (uint32
  // [rows]) and each lane's prior tokens (uint32 [lanes][ngram - 1], the one
  // just before the lane's first row first), by the hash tables (the token
  // map, int32 [vocabulary]; multipliers, int64 [layers][ngram]; primes and
  // offsets, int64 [layers][(ngram - 1) * heads]).
  static void addHash(metal::CommandGraph &graph, metal::MetalBuffer tokens, metal::MetalBuffer prior,
                      metal::MetalBuffer tokenMap, metal::MetalBuffer multipliers, metal::MetalBuffer primes,
                      metal::MetalBuffer offsets, metal::MetalBuffer rows, const EngramHashParams &params);
};

} // namespace splash::ops

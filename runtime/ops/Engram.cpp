#include "ops/Engram.hpp"

#include "ops/BufferExtent.hpp"

#include <stdexcept>
#include <utility>

namespace splash::ops {

void Engram::addHash(metal::CommandGraph &graph, metal::MetalBuffer tokens, metal::MetalBuffer prior,
                     metal::MetalBuffer tokenMap, metal::MetalBuffer multipliers, metal::MetalBuffer primes,
                     metal::MetalBuffer offsets, metal::MetalBuffer rows, const EngramHashParams &params) {
  if (!params.rows || !params.lane_rows || !params.layers || params.layers > ENGRAM_MAX_LAYERS || params.ngram < 2 ||
      params.ngram > ENGRAM_MAX_NGRAM || !params.heads)
    throw std::invalid_argument("invalid Engram hash");
  const uint64_t per = uint64_t{params.ngram - 1} * params.heads;
  const uint32_t lanes = (params.rows + params.lane_rows - 1) / params.lane_rows;
  requireBytes(tokens, uint64_t{params.rows} * sizeof(uint32_t), "Engram hash tokens");
  requireBytes(prior, uint64_t{lanes} * (params.ngram - 1) * sizeof(uint32_t), "Engram hash prior tokens");
  requireBytes(multipliers, uint64_t{params.layers} * params.ngram * sizeof(int64_t), "Engram hash multipliers");
  requireBytes(primes, uint64_t{params.layers} * per * sizeof(int64_t), "Engram hash primes");
  requireBytes(offsets, uint64_t{params.layers} * per * sizeof(int64_t), "Engram hash offsets");
  requireBytes(rows, uint64_t{params.rows} * params.layers * per * sizeof(uint32_t), "Engram hash rows");
  graph.add("engram_hash",
            {std::move(tokens), std::move(prior), std::move(tokenMap), std::move(multipliers), std::move(primes),
             std::move(offsets), std::move(rows)},
            params, {params.rows, params.layers, 1}, {1, 1, 1});
}

void Engram::addGate(metal::CommandGraph &graph, metal::MetalBuffer streams, metal::MetalBuffer kv,
                     metal::MetalBuffer weights, uint32_t hidden, float normEpsilon, uint32_t rows) {
  if (!rows || !hidden) throw std::invalid_argument("invalid Engram shape");
  requireBytes(streams, uint64_t{rows} * MHC_STREAMS * hidden * sizeof(float), "Engram streams");
  requireBytes(kv, uint64_t{rows} * (MHC_STREAMS + 1) * hidden * sizeof(uint16_t), "Engram keys and value");
  requireBytes(weights, uint64_t{MHC_STREAMS} * hidden * sizeof(float), "Engram weights");
  // engram.py's floor of |dot| under the square root.
  graph.add("engram_gate", {std::move(streams), std::move(kv), std::move(weights)},
            EngramParams{rows, hidden, normEpsilon, 1e-6f}, {rows, 1, 1}, {ENGRAM_THREADS, 1, 1});
}

} // namespace splash::ops

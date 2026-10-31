#include "ops/ManifoldHyperConnection.hpp"

#include "ops/BufferExtent.hpp"

#include <stdexcept>
#include <utility>

namespace splash::ops {
namespace {

constexpr uint32_t kThreads = MHC_THREADS;

void requireShape(ManifoldHyperShape shape, uint32_t rows) {
  if (!rows || !shape.hidden || shape.hidden > MHC_MAX_HIDDEN || !shape.iterations)
    throw std::invalid_argument("invalid manifold hyper-connection shape");
}

void requireStreams(const metal::MetalBuffer &streams, ManifoldHyperShape shape, uint32_t rows) {
  requireBytes(streams, uint64_t{rows} * shape.width() * sizeof(float), "manifold hyper-connection streams");
}

void requireMixes(const metal::MetalBuffer &mixes, uint32_t rows) {
  requireBytes(mixes, uint64_t{rows} * MHC_MIXES * sizeof(float), "manifold hyper-connection mixes");
}

ManifoldHyperParams params(ManifoldHyperShape shape, uint32_t rows) {
  return {rows, shape.hidden, shape.iterations, shape.hcEpsilon, shape.normEpsilon};
}

} // namespace

void ManifoldHyperConnection::addInit(metal::CommandGraph &graph, metal::MetalBuffer embedding,
                                      metal::MetalBuffer streams, ManifoldHyperShape shape, uint32_t rows) {
  requireShape(shape, rows);
  requireBytes(embedding, uint64_t{rows} * shape.hidden * sizeof(uint16_t), "manifold hyper-connection embedding");
  requireStreams(streams, shape, rows);
  graph.add("mhc_init", {std::move(embedding), std::move(streams)}, params(shape, rows),
            {rows, (shape.hidden + kThreads - 1) / kThreads, 1}, {kThreads, 1, 1});
}

void ManifoldHyperConnection::addStreamMean(metal::CommandGraph &graph, metal::MetalBuffer streams,
                                            metal::MetalBuffer out, uint32_t stride, uint32_t offset,
                                            ManifoldHyperShape shape, uint32_t rows) {
  requireShape(shape, rows);
  if (uint64_t{offset} + shape.hidden > stride) throw std::invalid_argument("invalid manifold hyper-connection mean");
  requireStreams(streams, shape, rows);
  requireBytes(out, (uint64_t{rows - 1} * stride + offset + shape.hidden) * sizeof(uint16_t),
               "manifold hyper-connection mean");
  graph.add("mhc_stream_mean", {std::move(streams), std::move(out)}, MhcMeanParams{rows, shape.hidden, stride, offset},
            {rows, (shape.hidden + kThreads - 1) / kThreads, 1}, {kThreads, 1, 1});
}

uint64_t ManifoldHyperConnection::mixScratchBytes(ManifoldHyperShape shape, uint32_t rows) noexcept {
  const uint64_t chunks = (uint64_t{shape.width()} + MHC_MIX_CHUNK - 1) / MHC_MIX_CHUNK;
  return uint64_t{rows} * chunks * MHC_MIX_ROW_PARTIALS * sizeof(float);
}

void ManifoldHyperConnection::addMix(metal::CommandGraph &graph, metal::MetalBuffer streams, metal::MetalBuffer fn,
                                     metal::MetalBuffer scale, metal::MetalBuffer base, metal::MetalBuffer chunks,
                                     metal::MetalBuffer mixes, ManifoldHyperShape shape, uint32_t rows) {
  requireShape(shape, rows);
  requireStreams(streams, shape, rows);
  requireBytes(fn, uint64_t{MHC_MIXES} * shape.width() * sizeof(float), "manifold hyper-connection projection");
  requireBytes(scale, 3 * sizeof(float), "manifold hyper-connection scale");
  requireBytes(base, MHC_MIXES * sizeof(float), "manifold hyper-connection base");
  requireBytes(chunks, mixScratchBytes(shape, rows), "manifold hyper-connection partial sums");
  requireMixes(mixes, rows);
  const uint32_t count = (shape.width() + MHC_MIX_CHUNK - 1) / MHC_MIX_CHUNK;
  graph.add("mhc_mix_partial", {std::move(streams), std::move(fn), chunks}, params(shape, rows),
            {(rows + MHC_MIX_ROWS - 1) / MHC_MIX_ROWS, count, 1}, {kThreads, 1, 1});
  graph.add("mhc_mix_finish", {std::move(chunks), std::move(scale), std::move(base), std::move(mixes)},
            params(shape, rows), {rows, 1, 1}, {32, 1, 1});
}

void ManifoldHyperConnection::addPreNorm(metal::CommandGraph &graph, metal::MetalBuffer streams,
                                         metal::MetalBuffer mixes, metal::MetalBuffer norm,
                                         metal::MetalBuffer output, ManifoldHyperShape shape, uint32_t rows) {
  requireShape(shape, rows);
  requireStreams(streams, shape, rows);
  requireMixes(mixes, rows);
  requireBytes(norm, uint64_t{shape.hidden} * sizeof(float), "manifold hyper-connection norm");
  requireBytes(output, uint64_t{rows} * shape.hidden * sizeof(uint16_t), "manifold hyper-connection input");
  graph.add("mhc_pre_norm", {std::move(streams), std::move(mixes), std::move(norm), std::move(output)},
            params(shape, rows), {rows, 1, 1}, {kThreads, 1, 1});
}

void ManifoldHyperConnection::addPost(metal::CommandGraph &graph, metal::MetalBuffer streams,
                                      metal::MetalBuffer branch, metal::MetalBuffer mixes, ManifoldHyperShape shape,
                                      uint32_t rows) {
  requireShape(shape, rows);
  requireStreams(streams, shape, rows);
  requireBytes(branch, uint64_t{rows} * shape.hidden * sizeof(uint16_t), "manifold hyper-connection output");
  requireMixes(mixes, rows);
  graph.add("mhc_post", {std::move(streams), std::move(branch), std::move(mixes)}, params(shape, rows),
            {rows, (shape.hidden + kThreads - 1) / kThreads, 1}, {kThreads, 1, 1});
}

} // namespace splash::ops

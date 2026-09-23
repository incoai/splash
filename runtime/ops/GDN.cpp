#include "ops/GDN.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/GDN.h"
#include "ops/LaneBindings.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace splash::ops {
namespace {

static_assert(offsetof(GDNDecodeBatchParams, conv_layer_bytes) == 16);
static_assert(offsetof(GDNBatchCommitParams, conv_layer_bytes) == 32);

enum class KernelLayout : uint8_t { Value48, Value32 };

[[nodiscard]] KernelLayout kernelShape(const GdnShape &shape) {
  if (!shape.valid())
    throw std::invalid_argument("invalid GDN shape");
  if (shape == GdnShape{16, 48, 128, 10240, 16640})
    return KernelLayout::Value48;
  if (shape == GdnShape{16, 32, 128, 8192, 12544})
    return KernelLayout::Value32;
  throw std::invalid_argument("unsupported compiled GDN shape");
}

[[nodiscard]] const char *kernelName(KernelLayout shape,
                                     const char *value48,
                                     const char *value32) noexcept {
  return shape == KernelLayout::Value48 ? value48 : value32;
}

} // namespace

void GDN::addPrefill(metal::CommandGraph &graph, GdnPrefillBuffers buffers,
                     GdnShape shape, uint32_t tokens, GdnHeadOrder order) {
  if (!tokens)
    throw std::invalid_argument("invalid GDN prefill geometry");
  const KernelLayout kernel = kernelShape(shape);
  const GDNPreparePrefillParams prepare{tokens, shape.packedWidth};
  graph.add(kernelName(kernel, "prefill_gdn_prepare",
                       "prefill_gdn_prepare_vh32"),
            {buffers.packed, buffers.convolutionWeights,
             buffers.convolutionIn, buffers.convolutionOut, buffers.queries,
             buffers.keys, buffers.values, buffers.decayWeights,
             buffers.timeBias, buffers.decay, buffers.beta},
            prepare, {uint64_t{tokens} * shape.keyHeads, 1, 1},
            {shape.headDimension, 1, 1});
  graph.add(kernelName(kernel, "prefill_gdn_scan",
                       "prefill_gdn_scan_vh32"),
            {buffers.queries, buffers.keys, buffers.values, buffers.decay,
             buffers.beta, buffers.recurrentIn, buffers.recurrentOut,
             buffers.recurrentRows},
            GDNPrefillParams{tokens},
            {uint64_t{shape.valueHeads} * shape.headDimension /
                 SPLASH_GDN_SCAN_STATE_ROWS,
             1, 1},
            {SPLASH_GDN_SCAN_THREADS, 1, 1});
  graph.add(kernelName(kernel, "prefill_gdn_gate",
                       "prefill_gdn_gate_vh32"),
            {buffers.recurrentRows, buffers.packed, buffers.mixerNorm,
             buffers.hidden},
            GDNGatePrefillParams{tokens, shape.packedWidth,
                                 order == GdnHeadOrder::Tiled},
            {uint64_t{tokens} * shape.valueHeads, 1, 1}, {128, 1, 1});
}

void GDN::addDecode(metal::CommandGraph &graph, GdnDecodeBuffers buffers,
                    GdnShape shape, uint32_t lanes, uint32_t layer,
                    GdnStateStrides state, GdnHeadOrder order) {
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH || !state.valid())
    throw std::invalid_argument("invalid GDN decode geometry");
  const KernelLayout kernel = kernelShape(shape);
  std::vector<metal::MetalBuffer> bindings{buffers.packed,
                                           buffers.convolutionWeights};
  const bool prepare = bool(buffers.linearScratch.input);
  const uint64_t outputWidth = uint64_t{shape.valueHeads} * shape.headDimension;
  if (prepare && (buffers.linearScratch.input.sizeBytes() < outputWidth * 16 * lanes ||
                  buffers.linearScratch.sums.sizeBytes() < outputWidth / 2 * lanes))
    throw std::invalid_argument("Q4 GDN preparation scratch is below requirement");
  bindings.reserve(prepare ? 22 : 20);
  appendLaneBindings(bindings, buffers.currentStates, buffers.nextStates);
  bindings.insert(bindings.end(),
                  {buffers.mixed, buffers.decayWeights, buffers.timeBias,
                   buffers.decay, buffers.beta, buffers.recurrent,
                   buffers.mixerNorm, buffers.hidden, buffers.arrived,
                   buffers.generation});
  if (prepare)
    bindings.insert(bindings.end(), {buffers.linearScratch.input, buffers.linearScratch.sums});
  const GDNDecodeBatchParams params{order == GdnHeadOrder::Tiled,
                                    shape.packedWidth,
                                    lanes,
                                    layer,
                                    state.convolutionLayerBytes,
                                    state.recurrentLayerBytes,
                                    state.convolutionStateBytes};
  graph.add(prepare ? kernelName(kernel, "verify_gdn_fused_q4", "verify_gdn_fused_q4_vh32")
                    : kernelName(kernel, "verify_gdn_fused", "verify_gdn_fused_vh32"),
            std::move(bindings), params, {shape.valueHeads, lanes, 1});
}

void GDN::addCommit(metal::CommandGraph &graph, GdnCommitBuffers buffers,
                    GdnShape shape, uint32_t layers, uint32_t lanes,
                    GdnStateStrides state) {
  if (!layers || !lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH ||
      !state.valid())
    throw std::invalid_argument("invalid GDN commit geometry");
  const KernelLayout kernel = kernelShape(shape);
  std::vector<metal::MetalBuffer> bindings{
      buffers.packed, buffers.mixed, buffers.decay, buffers.beta};
  bindings.reserve(13);
  appendLaneBindings(bindings, buffers.currentStates, buffers.nextStates);
  bindings.push_back(buffers.retainedCounts);
  constexpr uint32_t rows = SPLASH_TARGET_VERIFY_ROWS;
  const GDNBatchCommitParams params{shape.valueHeads,
                                    shape.packedWidth,
                                    lanes,
                                    rows * shape.packedWidth,
                                    rows * shape.convolutionDimension,
                                    rows * shape.valueHeads,
                                    rows * shape.valueHeads,
                                    0,
                                    state.convolutionLayerBytes,
                                    state.recurrentLayerBytes,
                                    state.convolutionStateBytes};
  graph.add(kernelName(kernel, "verify_gdn_commit",
                       "verify_gdn_commit_vh32"),
            std::move(bindings), params,
            {shape.valueHeads, layers, lanes});
}

} // namespace splash::ops

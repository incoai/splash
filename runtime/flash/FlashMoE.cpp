#include "FlashMoE.hpp"

#include "metal/abi/FlashMoE.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::flash {
namespace {

constexpr uint32_t kRouteThreads = 256;
constexpr uint32_t kPointwiseThreads = 256;

uint64_t multiply(uint64_t left, uint64_t right) {
  if (right && left > std::numeric_limits<uint64_t>::max() / right)
    throw std::invalid_argument("Flash MoE logical byte extent overflows");
  return left * right;
}

void requireBuffer(const metal::MetalBuffer &buffer, uint64_t bytes,
                   const char *name) {
  if (!bytes || !buffer || buffer.sizeBytes() < bytes)
    throw std::invalid_argument(std::string("Flash MoE insufficient ") + name);
}

void requireDistinct(const metal::MetalBuffer &output,
                     const metal::MetalBuffer &input) {
  if (output.sameView(input))
    throw std::invalid_argument("Flash MoE input/output must be disjoint");
}

void requireGeometry(uint32_t rows, uint32_t width, uint32_t selections) {
  if (!rows || rows > kFlashMoEMaxRows || !width ||
      width > kFlashMoEMaxWidth || !selections ||
      selections > kFlashMoEMaxSelections)
    throw std::invalid_argument("Flash MoE invalid pointwise geometry");
}

void requireRouting(uint32_t rows, uint32_t experts, uint32_t selections) {
  if (!rows || rows > kFlashMoEMaxRows || !experts ||
      experts > kFlashMoEMaxExperts || !selections ||
      selections > experts || selections > kFlashMoEMaxSelections)
    throw std::invalid_argument("Flash MoE invalid routing geometry");
}

uint64_t bf16Bytes(uint32_t rows, uint32_t width, uint32_t selections = 1) {
  return multiply(multiply(multiply(rows, width), selections), 2);
}

metal::DispatchSize pointwiseGroups(uint32_t rows, uint32_t width,
                                    uint32_t selections) {
  return {(uint64_t{width} + kPointwiseThreads - 1) / kPointwiseThreads,
          rows, selections};
}

} // namespace

void addRoute(metal::CommandGraph &graph, metal::MetalBuffer logits,
              metal::MetalBuffer expertIDs, metal::MetalBuffer routeWeights,
              metal::MetalBuffer diagnostics, uint32_t rows,
              uint32_t experts, uint32_t selections, bool normalizeTopK) {
  requireRouting(rows, experts, selections);
  requireBuffer(logits, bf16Bytes(rows, experts), "router logits");
  requireBuffer(expertIDs, multiply(multiply(rows, selections), sizeof(int64_t)),
                "route IDs");
  requireBuffer(routeWeights, bf16Bytes(rows, selections), "route weights");
  requireBuffer(diagnostics, sizeof(uint32_t), "diagnostics");
  requireDistinct(expertIDs, logits);
  requireDistinct(routeWeights, logits);
  requireDistinct(expertIDs, routeWeights);
  requireDistinct(diagnostics, logits);
  requireDistinct(diagnostics, expertIDs);
  requireDistinct(diagnostics, routeWeights);
  graph.add("flash_moe_route",
            {std::move(logits), std::move(expertIDs), std::move(routeWeights),
             std::move(diagnostics)},
            FlashMoERouteParams{rows, experts, selections, normalizeTopK ? 1u : 0u},
            {rows, 1, 1}, {kRouteThreads, 1, 1});
}

void addSiLUMultiply(metal::CommandGraph &graph, metal::MetalBuffer gate,
                     metal::MetalBuffer up, metal::MetalBuffer intermediate,
                     metal::MetalBuffer diagnostics, uint32_t rows,
                     uint32_t width, uint32_t selections) {
  requireGeometry(rows, width, selections);
  const uint64_t bytes = bf16Bytes(rows, width, selections);
  requireBuffer(gate, bytes, "expert gate");
  requireBuffer(up, bytes, "expert up");
  requireBuffer(intermediate, bytes, "expert activation");
  requireBuffer(diagnostics, sizeof(uint32_t), "diagnostics");
  requireDistinct(intermediate, gate);
  requireDistinct(intermediate, up);
  requireDistinct(intermediate, diagnostics);
  requireDistinct(diagnostics, gate);
  requireDistinct(diagnostics, up);
  graph.add("flash_moe_silu_multiply",
            {std::move(gate), std::move(up), std::move(intermediate),
             std::move(diagnostics)},
            FlashMoEPointwiseParams{rows, width, selections, 0},
            pointwiseGroups(rows, width, selections), {kPointwiseThreads, 1, 1});
}

void addCombine(metal::CommandGraph &graph, metal::MetalBuffer expertDown,
                metal::MetalBuffer expertIDs, metal::MetalBuffer routeWeights,
                metal::MetalBuffer sharedDown, metal::MetalBuffer sharedGate,
                metal::MetalBuffer output, metal::MetalBuffer diagnostics,
                uint32_t rows, uint32_t width, uint32_t experts,
                uint32_t selections) {
  requireGeometry(rows, width, selections);
  requireRouting(rows, experts, selections);
  requireBuffer(expertDown, bf16Bytes(rows, width, selections), "expert down");
  requireBuffer(expertIDs, multiply(multiply(rows, selections), sizeof(int64_t)),
                "route IDs");
  requireBuffer(routeWeights, bf16Bytes(rows, selections), "route weights");
  requireBuffer(sharedDown, bf16Bytes(rows, width), "shared expert down");
  requireBuffer(sharedGate, bf16Bytes(rows, 1), "shared expert gate");
  requireBuffer(output, bf16Bytes(rows, width), "combined output");
  requireBuffer(diagnostics, sizeof(uint32_t), "diagnostics");
  for (const auto &input : {expertDown, expertIDs, routeWeights, sharedDown,
                            sharedGate, diagnostics})
    requireDistinct(output, input);
  for (const auto &input : {expertDown, expertIDs, routeWeights, sharedDown,
                            sharedGate})
    requireDistinct(diagnostics, input);
  graph.add("flash_moe_combine",
            {std::move(expertDown), std::move(expertIDs), std::move(routeWeights),
             std::move(sharedDown), std::move(sharedGate), std::move(output),
             std::move(diagnostics)},
            FlashMoEPointwiseParams{rows, width, selections, experts},
            pointwiseGroups(rows, width, 1), {kPointwiseThreads, 1, 1});
}

} // namespace splash::flash

#include "FlashMoEFused.hpp"

#include "metal/abi/FlashMoEFused.h"

#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::flash {
namespace {

uint64_t multiply(uint64_t left, uint64_t right) {
  if (right && left > std::numeric_limits<uint64_t>::max() / right)
    throw std::invalid_argument("Flash fused MoE logical byte extent overflows");
  return left * right;
}

uint64_t add(uint64_t left, uint64_t right) {
  if (left > std::numeric_limits<uint64_t>::max() - right)
    throw std::invalid_argument("Flash fused MoE logical byte extent overflows");
  return left + right;
}

void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes,
                  const char *name) {
  if (!buffer || !bytes || buffer.sizeBytes() < bytes)
    throw std::invalid_argument(std::string("Flash fused MoE insufficient ") + name);
}

uint64_t extent(uint32_t experts, uint32_t outputs, uint64_t expertStride,
                uint64_t rowStride, uint64_t rowBytes) {
  const uint64_t matrixBytes = add(multiply(outputs - 1, rowStride), rowBytes);
  if (rowStride < rowBytes || expertStride < matrixBytes)
    throw std::invalid_argument("Flash fused MoE overlapping source strides");
  return add(multiply(experts - 1, expertStride), matrixBytes);
}

void requireTensor(const FlashTensor *tensor, FlashDType type, uint64_t bytes,
                   const char *name) {
  if (!tensor || tensor->dtype != type || tensor->logicalBytes < bytes)
    throw std::invalid_argument(std::string("Flash fused MoE invalid ") + name);
  requireBytes(tensor->buffer, bytes, name);
}

void requireProjection(const FlashAffineProjection &p, bool packedWordLoads,
                       bool isDown = false) {
  const uint32_t outputs = isDown ? 2560 : 640;
  const uint32_t inputs = isDown ? 640 : 2560;
  if (p.experts != 512 || p.outputSize != outputs || p.inputSize != inputs ||
      p.bits != 4 || p.groupSize != 64 || p.parameterRowStrideBytes % 2 ||
      p.parameterExpertStrideBytes % 2 ||
      (packedWordLoads && (p.weightRowStrideBytes % 4 ||
                            p.weightExpertStrideBytes % 4)))
    throw std::invalid_argument("Flash fused MoE unsupported projection geometry");
  requireTensor(p.weights, FlashDType::U32,
                extent(p.experts, p.outputSize, p.weightExpertStrideBytes,
                       p.weightRowStrideBytes, inputs / 2), "Q4 weights");
  const uint64_t coefficients = extent(p.experts, p.outputSize,
                                      p.parameterExpertStrideBytes,
                                      p.parameterRowStrideBytes, inputs / 32);
  requireTensor(p.scales, FlashDType::BF16, coefficients, "BF16 scales");
  requireTensor(p.biases, FlashDType::BF16, coefficients, "BF16 biases");
  if (packedWordLoads) {
    const void *address = p.weights->buffer.contents();
    if (!address || reinterpret_cast<uintptr_t>(address) % 4)
      throw std::invalid_argument("Flash fused MoE packed word load requires aligned mapped weights");
  }
}

void requireDownGeometry(uint32_t rows, uint32_t selections, uint32_t simds) {
  if (!rows || rows > 2048 || !selections || selections > 10 ||
      (simds != 2 && simds != 4 && simds != 8))
    throw std::invalid_argument("Flash fused down invalid geometry");
}

void requireOutputDisjoint(const metal::MetalBuffer &output,
                           const metal::MetalBuffer &diagnostics,
                           std::initializer_list<metal::MetalBuffer> inputs) {
  if (output.sameView(diagnostics))
    throw std::invalid_argument("Flash fused down output aliases diagnostics");
  for (const auto &buffer : inputs)
    if (output.sameView(buffer) || diagnostics.sameView(buffer))
      throw std::invalid_argument("Flash fused down output/diagnostics alias input");
}

FlashMoEDownFusedParams downParams(const FlashAffineProjection &p,
                                  uint32_t rows, uint32_t selections) {
  return {rows, selections, 640, 2560, 512, 0, 0, 0,
          p.weightRowStrideBytes, p.weightExpertStrideBytes,
          p.parameterRowStrideBytes, p.parameterExpertStrideBytes};
}

std::string downPipeline(const char *stem, uint32_t simds) {
  return std::string(stem) + (simds == 8 ? "" : "_s" + std::to_string(simds));
}

} // namespace

void addFusedExpertGateUp(metal::CommandGraph &graph, metal::MetalBuffer input,
                          const FlashAffineProjection &gate,
                          const FlashAffineProjection &up,
                          metal::MetalBuffer expertIDs,
                          metal::MetalBuffer output,
                          metal::MetalBuffer diagnostics, uint32_t rows,
                          uint32_t selections, uint32_t columnsPerSimd,
                          bool packedWordLoads,
                          uint32_t simdGroupsPerThreadgroup) {
  if (!rows || rows > 2048 || !selections || selections > 10 ||
      (columnsPerSimd != 1 && columnsPerSimd != 2 && columnsPerSimd != 4 &&
       columnsPerSimd != 8) ||
      (simdGroupsPerThreadgroup != 2 && simdGroupsPerThreadgroup != 4 &&
       simdGroupsPerThreadgroup != 8) ||
      (simdGroupsPerThreadgroup != 8 && columnsPerSimd != 1))
    throw std::invalid_argument("Flash fused MoE invalid row/selection/column count");
  requireProjection(gate, packedWordLoads);
  requireProjection(up, packedWordLoads);
  requireBytes(input, multiply(multiply(rows, 2560), 2), "input");
  requireBytes(expertIDs, multiply(multiply(rows, selections), sizeof(int64_t)), "IDs");
  requireBytes(output, multiply(multiply(multiply(rows, selections), 640), 2), "output");
  requireBytes(diagnostics, sizeof(uint32_t), "diagnostics");
  for (const auto &buffer : {input, expertIDs, diagnostics, gate.weights->buffer,
                            gate.scales->buffer, gate.biases->buffer,
                            up.weights->buffer, up.scales->buffer,
                            up.biases->buffer})
    if (output.sameView(buffer))
      throw std::invalid_argument("Flash fused MoE output must be disjoint from every input");
  for (const auto &buffer : {input, expertIDs, gate.weights->buffer,
                            gate.scales->buffer, gate.biases->buffer,
                            up.weights->buffer, up.scales->buffer,
                            up.biases->buffer})
    if (diagnostics.sameView(buffer))
      throw std::invalid_argument("Flash fused MoE diagnostics must be disjoint from inputs");
  const FlashMoEFusedParams params{
      rows, selections, 2560, 640, 512, 0, 0, 0,
      gate.weightRowStrideBytes, gate.weightExpertStrideBytes,
      gate.parameterRowStrideBytes, gate.parameterExpertStrideBytes,
      up.weightRowStrideBytes, up.weightExpertStrideBytes,
      up.parameterRowStrideBytes, up.parameterExpertStrideBytes};
  std::string pipeline = std::string("flash_moe_fused_q4_gate_up_") +
      (packedWordLoads ? "u32_" : "") + "c" + std::to_string(columnsPerSimd);
  if (simdGroupsPerThreadgroup != 8)
    pipeline += "_s" + std::to_string(simdGroupsPerThreadgroup);
  graph.add(pipeline,
            {std::move(input), gate.weights->buffer, gate.scales->buffer,
             gate.biases->buffer, up.weights->buffer, up.scales->buffer,
             up.biases->buffer, std::move(expertIDs), std::move(output),
             std::move(diagnostics)}, params,
            {(640u + simdGroupsPerThreadgroup * columnsPerSimd - 1) /
                 (simdGroupsPerThreadgroup * columnsPerSimd), rows,
             selections}, {32u * simdGroupsPerThreadgroup, 1, 1});
}

void addFusedExpertDownTerms(metal::CommandGraph &graph,
                             metal::MetalBuffer input,
                             const FlashAffineProjection &down,
                             metal::MetalBuffer expertIDs,
                             metal::MetalBuffer routeWeights,
                             metal::MetalBuffer output,
                             metal::MetalBuffer diagnostics, uint32_t rows,
                             uint32_t selections, uint32_t simds) {
  requireDownGeometry(rows, selections, simds);
  requireProjection(down, true, true);
  const uint64_t routes = multiply(rows, selections);
  requireBytes(input, multiply(multiply(routes, 640), 2), "down input");
  requireBytes(expertIDs, multiply(routes, 8), "down IDs");
  requireBytes(routeWeights, multiply(routes, 2), "down route weights");
  requireBytes(output, multiply(multiply(routes, 2560), 2), "down weighted terms");
  requireBytes(diagnostics, 4, "down diagnostics");
  requireOutputDisjoint(output, diagnostics,
                       {input, expertIDs, routeWeights, down.weights->buffer,
                        down.scales->buffer, down.biases->buffer});
  graph.add(downPipeline("flash_moe_fused_down_terms", simds),
            {input, down.weights->buffer, down.scales->buffer, down.biases->buffer,
             expertIDs, routeWeights, output, diagnostics},
            downParams(down, rows, selections), {(2560u + simds - 1) / simds, rows, selections},
            {32u * simds, 1, 1});
}

void addFusedExpertDownCombine(metal::CommandGraph &graph,
                               metal::MetalBuffer input,
                               const FlashAffineProjection &down,
                               metal::MetalBuffer expertIDs,
                               metal::MetalBuffer routeWeights,
                               metal::MetalBuffer sharedDown,
                               metal::MetalBuffer sharedGate,
                               metal::MetalBuffer output,
                               metal::MetalBuffer diagnostics, uint32_t rows,
                               uint32_t selections, uint32_t simds) {
  requireDownGeometry(rows, selections, simds);
  requireProjection(down, true, true);
  const uint64_t routes = multiply(rows, selections);
  requireBytes(input, multiply(multiply(routes, 640), 2), "down input");
  requireBytes(expertIDs, multiply(routes, 8), "down IDs");
  requireBytes(routeWeights, multiply(routes, 2), "down route weights");
  requireBytes(sharedDown, multiply(multiply(rows, 2560), 2), "shared down");
  requireBytes(sharedGate, multiply(rows, 2), "shared gate");
  requireBytes(output, multiply(multiply(rows, 2560), 2), "combined down output");
  requireBytes(diagnostics, 4, "down diagnostics");
  requireOutputDisjoint(output, diagnostics,
                       {input, expertIDs, routeWeights, sharedDown, sharedGate,
                        down.weights->buffer, down.scales->buffer, down.biases->buffer});
  graph.add(downPipeline("flash_moe_fused_down_combine", simds),
            {input, down.weights->buffer, down.scales->buffer, down.biases->buffer,
             expertIDs, routeWeights, sharedDown, sharedGate, output, diagnostics},
            downParams(down, rows, selections), {(2560u + simds - 1) / simds, rows, 1},
            {32u * simds, 1, 1});
}

void addWeightedExpertCombine(metal::CommandGraph &graph,
                              metal::MetalBuffer terms,
                              metal::MetalBuffer expertIDs,
                              metal::MetalBuffer sharedDown,
                              metal::MetalBuffer sharedGate,
                              metal::MetalBuffer output,
                              metal::MetalBuffer diagnostics, uint32_t rows,
                              uint32_t selections) {
  requireDownGeometry(rows, selections, 8);
  const uint64_t routes = multiply(rows, selections);
  requireBytes(terms, multiply(multiply(routes, 2560), 2), "weighted terms");
  requireBytes(expertIDs, multiply(routes, 8), "down IDs");
  requireBytes(sharedDown, multiply(multiply(rows, 2560), 2), "shared down");
  requireBytes(sharedGate, multiply(rows, 2), "shared gate");
  requireBytes(output, multiply(multiply(rows, 2560), 2), "combined output");
  requireBytes(diagnostics, 4, "down diagnostics");
  requireOutputDisjoint(output, diagnostics, {terms, expertIDs, sharedDown, sharedGate});
  graph.add("flash_moe_fused_weighted_combine",
            {terms, expertIDs, sharedDown, sharedGate, output, diagnostics},
            FlashMoEDownFusedParams{rows, selections, 640, 2560, 512, 0, 0, 0, 0, 0, 0, 0},
            {10, rows, 1}, {256, 1, 1});
}

} // namespace splash::flash

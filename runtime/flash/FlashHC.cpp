#include "flash/FlashHC.hpp"

#include "metal/abi/FlashHC.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::flash {
namespace {

constexpr uint32_t kPointwiseThreads = 64;

uint32_t normThreads(uint32_t width) noexcept {
  // MLX's non-looped RMS kernel covers four contiguous elements per thread.
  // The looped path retains the 1024-thread/32-SIMDgroup reduction capacity.
  if (width > 4096)
    return 1024;
  return ((width + 3) / 4 + 31) / 32 * 32;
}

uint64_t multiply(uint64_t left, uint64_t right) {
  if (right && left > std::numeric_limits<uint64_t>::max() / right)
    throw std::invalid_argument("Flash HC geometry overflows byte count");
  return left * right;
}

FlashHCParams parameters(FlashHCGeometry geometry) {
  if (!geometry.rows || !geometry.width || !geometry.streams ||
      geometry.streams > 8 || !std::isfinite(geometry.epsilon) ||
      geometry.epsilon <= 0.0f) {
    throw std::invalid_argument("invalid Flash HC geometry");
  }
  return {geometry.rows, geometry.width, geometry.streams,
          FlashHCOnePlusWeight, geometry.epsilon, 0, 0, 0};
}

uint64_t rowBytes(FlashHCGeometry geometry, uint32_t streams = 1) {
  return multiply(multiply(multiply(geometry.rows, geometry.width), streams),
                  sizeof(uint16_t));
}

uint64_t injectionBytes(FlashHCGeometry geometry) {
  return multiply(multiply(geometry.rows, geometry.streams), sizeof(uint16_t));
}

void requireBuffer(const metal::MetalBuffer &buffer, uint64_t bytes,
                   const char *name) {
  if (!buffer || buffer.sizeBytes() < bytes)
    throw std::invalid_argument(std::string("Flash HC ") + name +
                                " buffer is smaller than its logical shape");
}

void requireDistinct(const metal::MetalBuffer &output,
                     const metal::MetalBuffer &input) {
  if (output.sameView(input))
    throw std::invalid_argument("unsupported Flash HC output alias");
}

metal::DispatchSize pointwiseGroups(FlashHCGeometry geometry) {
  return {(uint64_t{geometry.width} + kPointwiseThreads - 1) /
              kPointwiseThreads,
          geometry.rows, 1};
}

void requireMixInputs(const metal::MetalBuffer &normalized,
                      const metal::MetalBuffer &rawUp,
                      const metal::MetalBuffer &mixed,
                      FlashHCGeometry geometry) {
  const uint64_t hyperBytes = rowBytes(geometry, geometry.streams);
  requireBuffer(normalized, hyperBytes, "normalized state");
  requireBuffer(rawUp, hyperBytes, "up projection");
  requireBuffer(mixed, rowBytes(geometry), "mixed state");
  requireDistinct(mixed, normalized);
  requireDistinct(mixed, rawUp);
}

} // namespace

void addHCExpand(metal::CommandGraph &graph, metal::MetalBuffer input,
                 metal::MetalBuffer output, FlashHCGeometry geometry) {
  const auto params = parameters(geometry);
  requireBuffer(input, rowBytes(geometry), "input");
  requireBuffer(output, rowBytes(geometry, geometry.streams), "expanded state");
  requireDistinct(output, input);
  graph.add("flash_hc_expand", {std::move(input), std::move(output)}, params,
            pointwiseGroups(geometry), {kPointwiseThreads, 1, 1});
}

void addHCGroupedNorm(metal::CommandGraph &graph, metal::MetalBuffer input,
                      const FlashTensor &normWeight,
                      metal::MetalBuffer output, FlashHCGeometry geometry,
                      NormConvention convention) {
  auto params = parameters(geometry);
  switch (convention) {
  case NormConvention::OnePlusWeight:
    params.norm_convention = FlashHCOnePlusWeight;
    break;
  case NormConvention::DirectGamma:
    params.norm_convention = FlashHCDirectGamma;
    break;
  default:
    throw std::invalid_argument("invalid Flash HC norm convention");
  }
  const uint64_t width = multiply(geometry.width, geometry.streams);
  if (normWeight.shape.size() != 1 || normWeight.shape[0] != width ||
      (normWeight.dtype != FlashDType::BF16 &&
       normWeight.dtype != FlashDType::F32)) {
    throw std::invalid_argument("Flash HC norm weight has incompatible shape or dtype");
  }
  const uint64_t weightBytes =
      multiply(width, normWeight.dtype == FlashDType::F32 ? sizeof(float)
                                                        : sizeof(uint16_t));
  if (normWeight.logicalBytes != weightBytes)
    throw std::invalid_argument("Flash HC norm weight has inconsistent logical bytes");
  requireBuffer(normWeight.buffer, weightBytes, "norm weight");
  const uint64_t hyperBytes = rowBytes(geometry, geometry.streams);
  requireBuffer(input, hyperBytes, "norm input");
  requireBuffer(output, hyperBytes, "norm output");
  requireDistinct(output, normWeight.buffer);
  graph.add(normWeight.dtype == FlashDType::F32 ? "flash_hc_norm_f32_weight"
                                               : "flash_hc_norm_bf16_weight",
            {std::move(input), normWeight.buffer, std::move(output)}, params,
            {geometry.rows, geometry.streams, 1},
            {normThreads(geometry.width), 1, 1});
}

void addHCMix(metal::CommandGraph &graph, metal::MetalBuffer normalized,
              metal::MetalBuffer rawUp, metal::MetalBuffer mixed,
              FlashHCGeometry geometry) {
  const auto params = parameters(geometry);
  requireMixInputs(normalized, rawUp, mixed, geometry);
  graph.add("flash_hc_mix",
            {std::move(normalized), std::move(rawUp), std::move(mixed)}, params,
            pointwiseGroups(geometry), {kPointwiseThreads, 1, 1});
}

void addHCMixWithInjection(metal::CommandGraph &graph,
                           metal::MetalBuffer normalized,
                           metal::MetalBuffer rawUp,
                           metal::MetalBuffer rawInjection,
                           metal::MetalBuffer mixed,
                           metal::MetalBuffer injectionWeights,
                           FlashHCGeometry geometry) {
  const auto params = parameters(geometry);
  requireMixInputs(normalized, rawUp, mixed, geometry);
  requireBuffer(rawInjection, injectionBytes(geometry), "injection projection");
  requireBuffer(injectionWeights, injectionBytes(geometry), "injection weights");
  requireDistinct(mixed, rawInjection);
  requireDistinct(injectionWeights, normalized);
  requireDistinct(injectionWeights, rawUp);
  requireDistinct(injectionWeights, rawInjection);
  requireDistinct(injectionWeights, mixed);
  graph.add("flash_hc_mix_with_injection",
            {std::move(normalized), std::move(rawUp), std::move(rawInjection),
             std::move(mixed), std::move(injectionWeights)},
            params, pointwiseGroups(geometry), {kPointwiseThreads, 1, 1});
}

void addHCInject(metal::CommandGraph &graph, metal::MetalBuffer hyperInput,
                 metal::MetalBuffer branch,
                 metal::MetalBuffer injectionWeights,
                 metal::MetalBuffer output, FlashHCGeometry geometry) {
  const auto params = parameters(geometry);
  const uint64_t hyperBytes = rowBytes(geometry, geometry.streams);
  requireBuffer(hyperInput, hyperBytes, "residual state");
  requireBuffer(branch, rowBytes(geometry), "branch");
  requireBuffer(injectionWeights, injectionBytes(geometry), "injection weights");
  requireBuffer(output, hyperBytes, "injected state");
  requireDistinct(output, branch);
  requireDistinct(output, injectionWeights);
  graph.add("flash_hc_inject",
            {std::move(hyperInput), std::move(branch),
             std::move(injectionWeights), std::move(output)},
            params, pointwiseGroups(geometry), {kPointwiseThreads, 1, 1});
}

} // namespace splash::flash

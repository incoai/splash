#include "flash/FlashPLEPostFused.hpp"

#include "metal/abi/FlashPLEPostFused.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace splash::flash {
namespace {

uint64_t multiply(uint64_t a, uint64_t b) {
  if (b && a > UINT64_MAX / b)
    throw std::invalid_argument("Flash PLE fused post extent overflow");
  return a * b;
}

void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes,
                  const char *name) {
  if (!bytes || !buffer || buffer.sizeBytes() < bytes)
    throw std::invalid_argument(std::string("Flash PLE fused post insufficient ") +
                                name);
}

uint32_t normThreads(uint32_t width) {
  return width > 4096 ? 1024 : ((width + 3) / 4 + 31) / 32 * 32;
}

uint32_t gateThreads(uint32_t width) {
  if (width <= 64) return 256;
  if (width <= 512) return 32;
  if (width <= 1024) return 128;
  return static_cast<uint32_t>(std::min<uint64_t>(
      1024, ((uint64_t(width) + 3) / 4 + 31) / 32 * 32));
}

void requireNorm(const FlashTensor &tensor, uint64_t width) {
  if ((tensor.dtype != FlashDType::BF16 && tensor.dtype != FlashDType::F32) ||
      tensor.shape.size() != 1 || tensor.shape[0] != width)
    throw std::invalid_argument("Flash PLE fused post invalid norm tensor");
  const uint64_t bytes = multiply(width, tensor.dtype == FlashDType::F32 ? 4 : 2);
  if (tensor.logicalBytes != bytes)
    throw std::invalid_argument("Flash PLE fused post invalid norm byte extent");
  requireBytes(tensor.buffer, bytes, "norm weights");
}

} // namespace

void addPLEPostProjectFused(
    metal::CommandGraph &graph, const FlashPLEWeights &weights,
    metal::MetalBuffer hyperInput, metal::MetalBuffer keys,
    metal::MetalBuffer values, const FlashPLEPostScratch &scratch,
    metal::MetalBuffer state, metal::MetalBuffer pleOutput,
    metal::MetalBuffer injectedOutput, metal::MetalBuffer diagnostics,
    FlashPLEGeometry g, metal::MetalBuffer mask) {
  const uint64_t rows = multiply(g.lanes, g.rows);
  const uint64_t width = multiply(g.width, g.streams);
  if (!g.lanes || !g.rows || !g.width || !g.streams || g.streams > 8 ||
      !g.vocabularySize || g.eosToken >= g.vocabularySize ||
      !std::isfinite(g.epsilon) || g.epsilon <= 0 ||
      rows > UINT32_MAX || width > UINT32_MAX)
    throw std::invalid_argument("Flash PLE fused post invalid geometry");
  const uint64_t bytes = multiply(multiply(rows, width), 2);
  requireBytes(hyperInput, bytes, "hyper input");
  requireBytes(keys, bytes, "projected keys");
  requireBytes(values, multiply(multiply(rows, g.width), 2), "projected values");
  requireBytes(state, multiply(multiply(multiply(g.lanes, 9), width), 2),
               "convolution state");
  requireBytes(pleOutput, bytes, "PLE output");
  requireBytes(injectedOutput, bytes, "injected output");
  requireBytes(diagnostics, 4, "diagnostics");
  if (mask) requireBytes(mask, multiply(rows, 4), "mask");

  const std::array<metal::MetalBuffer, 4> scratchViews{
      scratch.normalizedKeys, scratch.normalizedQueries, scratch.gatedValues,
      scratch.normalizedConvolution};
  const std::array<metal::MetalBuffer, 8> externalViews{
      hyperInput, keys, values, state, pleOutput, injectedOutput, diagnostics,
      mask};
  for (size_t i = 0; i < scratchViews.size(); ++i) {
    requireBytes(scratchViews[i], bytes, "post scratch");
    for (const auto &view : externalViews)
      if (view && scratchViews[i].sameView(view))
        throw std::invalid_argument("Flash PLE fused post scratch alias");
    for (size_t j = 0; j < i; ++j)
      if (scratchViews[i].sameView(scratchViews[j]))
        throw std::invalid_argument("Flash PLE fused post duplicate scratch");
  }
  for (const auto &output : {pleOutput, injectedOutput}) {
    for (const auto &input : {keys, values, state, diagnostics, mask})
      if (input && output.sameView(input))
        throw std::invalid_argument("Flash PLE fused post output alias");
  }
  if (pleOutput.sameView(hyperInput) || pleOutput.sameView(injectedOutput))
    throw std::invalid_argument("Flash PLE fused post PLE output alias");
  for (const auto &input : {hyperInput, keys, values, diagnostics, mask})
    if (input && state.sameView(input))
      throw std::invalid_argument("Flash PLE fused post state alias");

  for (const auto *norm : {&weights.normKey, &weights.normQuery,
                           &weights.normConvolution}) {
    requireNorm(*norm, width);
    for (const auto &scratchView : scratchViews)
      if (scratchView.sameView(norm->buffer))
        throw std::invalid_argument("Flash PLE fused post norm scratch alias");
    if (state.sameView(norm->buffer) || pleOutput.sameView(norm->buffer) ||
        injectedOutput.sameView(norm->buffer))
      throw std::invalid_argument("Flash PLE fused post mutable norm alias");
  }
  if (weights.normConvention != NormConvention::OnePlusWeight &&
      weights.normConvention != NormConvention::DirectGamma)
    throw std::invalid_argument("Flash PLE fused post invalid norm convention");
  const auto &convolution = weights.convolution;
  if (convolution.dtype != FlashDType::BF16 ||
      convolution.shape.size() != 3 || convolution.shape[0] != width ||
      convolution.shape[1] != 4 || convolution.shape[2] != 1 ||
      convolution.logicalBytes < multiply(multiply(width, 4), 2))
    throw std::invalid_argument("Flash PLE fused post invalid convolution");
  requireBytes(convolution.buffer, multiply(multiply(width, 4), 2),
               "convolution weights");
  for (const auto &view : scratchViews)
    if (view.sameView(convolution.buffer))
      throw std::invalid_argument("Flash PLE fused post convolution scratch alias");
  if (state.sameView(convolution.buffer) || pleOutput.sameView(convolution.buffer) ||
      injectedOutput.sameView(convolution.buffer))
    throw std::invalid_argument("Flash PLE fused post convolution output alias");

  const FlashPLEPostParams geometry{g.lanes, g.rows, g.width, g.streams,
                                    mask ? 1u : 0u, 9, 4, 3};
  const uint32_t normFlags =
      (weights.normKey.dtype == FlashDType::F32 ? FlashPLEPostFusedKeyF32 : 0u) |
      (weights.normQuery.dtype == FlashDType::F32 ? FlashPLEPostFusedQueryF32 : 0u) |
      (weights.normConvolution.dtype == FlashDType::F32 ?
           FlashPLEPostFusedConvolutionF32 : 0u);
  const FlashPLEPostFusedParams params{
      geometry, g.epsilon,
      weights.normConvention == NormConvention::OnePlusWeight ?
          FlashHCOnePlusWeight : FlashHCDirectGamma,
      normFlags, normThreads(g.width), gateThreads(g.width), 0, 0, 0};
  // All validation precedes the first mutation of the caller's graph.
  graph.add("flash_ple_post_norm_gate_fused",
            {keys, hyperInput, values, weights.normKey.buffer,
             weights.normQuery.buffer, weights.normConvolution.buffer,
             mask ? mask : diagnostics, scratch.normalizedKeys,
             scratch.normalizedQueries, scratch.gatedValues,
             scratch.normalizedConvolution, diagnostics}, params,
            {rows, g.streams, 1},
            {std::max(params.norm_threads, params.gate_threads), 1, 1});
  graph.add("flash_ple_post_convolution_inject_fused",
            {scratch.normalizedConvolution, scratch.gatedValues, state,
             convolution.buffer, hyperInput, pleOutput, injectedOutput,
             diagnostics}, geometry,
            {(width - 1) / 256 + 1, rows, 1}, {256, 1, 1});
  graph.add("flash_ple_update_convolution_state",
            {scratch.normalizedConvolution, state}, geometry,
            {(width - 1) / 256 + 1, g.lanes, 1}, {256, 1, 1});
}

} // namespace splash::flash

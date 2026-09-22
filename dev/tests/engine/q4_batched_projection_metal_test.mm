#include "../../../runtime/metal/MetalBackend.hpp"
#include "metal/abi/Linear.h"
#include "tuning/LinearNumerics.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::ComputeDispatch;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;

constexpr uint32_t kRows = 8;
constexpr uint32_t kMaximumBatch = 4;
constexpr uint32_t kInput = 5120;
constexpr uint32_t kOutput = 16640;
constexpr uint32_t kGroups = 60;
constexpr uint32_t kQuantGroup = 64;
constexpr std::array<const char *, kMaximumBatch> kN64AffinePipelines{
    "decode_linear_q4_n64", "decode_linear_q4_n64_m16",
    "decode_linear_q4_n64_m24", "decode_linear_q4_n64_m32"};
constexpr std::array<const char *, kMaximumBatch> kN64ResidualPipelines{
    "decode_linear_q4_n64_residual", "decode_linear_q4_n64_residual_m16",
    "decode_linear_q4_n64_residual_m24", "decode_linear_q4_n64_residual_m32"};

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "FAIL: " << message << '\n';
  std::exit(1);
}

MetalBuffer shared(MetalBackend &backend, uint64_t bytes, const char *label) {
  return backend.allocateBuffer(bytes, BufferStorage::Shared, label);
}

ComputeDispatch affine(std::string pipeline, MetalBuffer input,
                       MetalBuffer weights, MetalBuffer scales,
                       MetalBuffer biases, MetalBuffer output,
                       const Q4Params &params) {
  ComputeDispatch result;
  result.pipelineName = std::move(pipeline);
  result.buffers = {{0, std::move(input)},
                    {1, std::move(weights)},
                    {2, std::move(scales)},
                    {3, std::move(biases)},
                    {4, std::move(output)}};
  result.bytes = {{5, &params, sizeof(params)}};
  result.threadgroups = {params.persistent_groups, 1, 1};
  result.threadsPerThreadgroup = {256, 1, 1};
  return result;
}

ComputeDispatch gateUp(std::string pipeline, MetalBuffer input,
                       MetalBuffer weights, MetalBuffer scales,
                       MetalBuffer biases, MetalBuffer output,
                       const Q4Params &params) {
  ComputeDispatch result;
  result.pipelineName = std::move(pipeline);
  result.buffers = {{0, std::move(input)},
                    {1, weights},
                    {2, scales},
                    {3, biases},
                    {4, std::move(output)},
                    {5, std::move(weights)},
                    {6, std::move(scales)},
                    {7, std::move(biases)}};
  result.bytes = {{8, &params, sizeof(params)}};
  result.threadgroups = {params.persistent_groups, 1, 1};
  result.threadsPerThreadgroup = {256, 1, 1};
  return result;
}

ComputeDispatch residualDispatch(std::string pipeline, MetalBuffer input,
                                 MetalBuffer weights, MetalBuffer scales,
                                 MetalBuffer biases, MetalBuffer residual,
                                 MetalBuffer output, const Q4Params &params) {
  ComputeDispatch result;
  result.pipelineName = std::move(pipeline);
  result.buffers = {{0, std::move(input)},
                    {1, std::move(weights)},
                    {2, std::move(scales)},
                    {3, std::move(biases)},
                    {4, std::move(residual)},
                    {5, std::move(output)}};
  result.bytes = {{6, &params, sizeof(params)}};
  result.threadgroups = {params.persistent_groups, 1, 1};
  result.threadsPerThreadgroup = {256, 1, 1};
  return result;
}

ComputeDispatch withThreads(ComputeDispatch dispatch, uint32_t threads) {
  dispatch.threadsPerThreadgroup = {threads, 1, 1};
  return dispatch;
}

// Split-K outputs against the sequential kernel's: every element within the
// derived bound of tuning/LinearNumerics.hpp (one bf16 ulp of the projection,
// the epilogue's propagation of that step, fp32 reassociation slack).
void requireSplitTolerance(const char *what, splash::ops::LinearEpilogue epilogue,
                           const MetalBuffer &exact, const MetalBuffer &split,
                           const MetalBuffer *residual, const MetalBuffer *gateUpValue,
                           uint64_t elements, uint32_t inputSize) {
  using namespace splash::ops::tuning;
  const auto *exactValues = static_cast<const uint16_t *>(exact.contents());
  const auto *splitValues = static_cast<const uint16_t *>(split.contents());
  const auto *residualValues =
      residual ? static_cast<const uint16_t *>(residual->contents()) : nullptr;
  const auto *gateValues =
      gateUpValue ? static_cast<const uint16_t *>(gateUpValue->contents()) : nullptr;
  float maxAbs = 0;
  for (uint64_t i = 0; i < elements; ++i)
    maxAbs = std::max(maxAbs, std::fabs(bf16ToFloat(exactValues[i])));
  const float slack = reassociationSlack(inputSize, maxAbs);
  float maxDiff = 0;
  for (uint64_t i = 0; i < elements; ++i) {
    SplitReference reference{bf16ToFloat(exactValues[i])};
    if (residualValues) reference.residual = bf16ToFloat(residualValues[i]);
    // Gate and up streams read the same weights here, so one plain
    // projection is both the exact gate and the exact up value.
    if (gateValues) reference.gate = reference.up = bf16ToFloat(gateValues[i]);
    const float actual = bf16ToFloat(splitValues[i]);
    maxDiff = std::max(maxDiff, std::fabs(actual - reference.value));
    if (!withinSplitTolerance(actual, epilogue, reference, slack))
      fail(std::string(what) + " element " + std::to_string(i) + " actual=" +
           std::to_string(actual) + " reference=" + std::to_string(reference.value) +
           " bound=" + std::to_string(splitTolerance(epilogue, reference, slack)));
  }
  std::cout << "PASS " << what << " within_bound=true max_abs_diff=" << maxDiff
            << " max_abs_ref=" << maxAbs << " slack=" << slack << '\n';
}

ComputeDispatch upSilu(std::string pipeline, MetalBuffer input,
                       MetalBuffer weights, MetalBuffer scales,
                       MetalBuffer biases, MetalBuffer gate,
                       MetalBuffer output, const Q4Params &params) {
  ComputeDispatch result;
  result.pipelineName = std::move(pipeline);
  result.buffers = {{0, std::move(input)},
                    {1, std::move(weights)},
                    {2, std::move(scales)},
                    {3, std::move(biases)},
                    {4, std::move(gate)},
                    {5, std::move(output)}};
  result.bytes = {{6, &params, sizeof(params)}};
  result.threadgroups = {params.persistent_groups, 1, 1};
  result.threadsPerThreadgroup = {256, 1, 1};
  return result;
}

ComputeDispatch residual(std::string pipeline, MetalBuffer input,
                         MetalBuffer weights, MetalBuffer scales,
                         MetalBuffer biases, MetalBuffer residualInput,
                         MetalBuffer output, const Q4Params &params) {
  ComputeDispatch result;
  result.pipelineName = std::move(pipeline);
  result.buffers = {{0, std::move(input)},
                    {1, std::move(weights)},
                    {2, std::move(scales)},
                    {3, std::move(biases)},
                    {4, std::move(residualInput)},
                    {5, std::move(output)}};
  result.bytes = {{6, &params, sizeof(params)}};
  result.threadgroups = {params.persistent_groups, 1, 1};
  result.threadsPerThreadgroup = {256, 1, 1};
  return result;
}

void checkN64Persistent(MetalBackend &backend, MetalBuffer input,
                        MetalBuffer weights, MetalBuffer scales,
                        MetalBuffer biases, uint32_t inputSize,
                        std::mt19937 &random) {
  constexpr uint32_t outputSize = 768;
  const uint64_t laneInputBytes = uint64_t{kRows} * inputSize * sizeof(__bf16);
  const uint64_t laneOutputBytes = uint64_t{kRows} * outputSize * sizeof(__bf16);
  const uint64_t maximumOutputBytes = kMaximumBatch * laneOutputBytes;
  MetalBuffer residualInput = shared(backend, maximumOutputBytes,
                                      "q4-n64-residual-input");
  std::uniform_real_distribution<float> residualValues(-1.0f, 1.0f);
  auto *residualValuesPtr = static_cast<__bf16 *>(residualInput.contents());
  for (uint64_t index = 0; index < maximumOutputBytes / sizeof(__bf16); ++index)
    residualValuesPtr[index] = __bf16(residualValues(random));

  const Q4Params referenceParams{outputSize, inputSize, outputSize / 128};
  const Q4Params singleGroupParams{outputSize, inputSize, 1};
  const Q4Params fullGridParams{outputSize, inputSize, outputSize / 64};
  for (const bool addResidual : {false, true}) {
    MetalBuffer reference = shared(backend, maximumOutputBytes,
                                    "q4-n64-m8-reference");
    std::memset(reference.contents(), 0, maximumOutputBytes);
    std::vector<ComputeDispatch> singles;
    for (uint32_t lane = 0; lane < kMaximumBatch; ++lane) {
      auto laneInput = backend.view(input, lane * laneInputBytes, laneInputBytes);
      auto laneOutput = backend.view(reference, lane * laneOutputBytes,
                                      laneOutputBytes);
      if (addResidual) {
        singles.push_back(residual(
            "decode_linear_q4_n128_residual", std::move(laneInput),
            weights, scales, biases,
            backend.view(residualInput, lane * laneOutputBytes, laneOutputBytes),
            std::move(laneOutput), referenceParams));
      } else {
        singles.push_back(affine("decode_linear_q4_n128", std::move(laneInput),
                                 weights, scales, biases, std::move(laneOutput),
                                 referenceParams));
      }
    }
    (void)backend.submitCommand(singles);
    for (uint32_t width = 1; width <= kMaximumBatch; ++width) {
      const uint64_t comparedBytes = width * laneOutputBytes;
      MetalBuffer candidate = shared(backend, comparedBytes, "q4-n64-output");
      for (const bool persistent : {false, true}) {
        // Sentinel catches missing output rows or columns in a fragment.
        std::memset(candidate.contents(), 0xA5, comparedBytes);
        const Q4Params &params = persistent ? singleGroupParams : fullGridParams;
        auto batchInput = backend.view(input, 0, width * laneInputBytes);
        ComputeDispatch dispatch;
        if (addResidual) {
          dispatch = residual(kN64ResidualPipelines[width - 1],
                              std::move(batchInput), weights, scales, biases,
                              backend.view(residualInput, 0, comparedBytes),
                              candidate, params);
        } else {
          dispatch = affine(kN64AffinePipelines[width - 1], std::move(batchInput),
                            weights, scales, biases, candidate, params);
        }
        (void)backend.submitCommand({&dispatch, 1});
        if (std::memcmp(reference.contents(), candidate.contents(), comparedBytes))
          fail("N64 M" + std::to_string(width * kRows) +
               (addResidual ? " residual" : " affine") +
               " K=" + std::to_string(inputSize) +
               " groups=" + std::to_string(params.persistent_groups) +
               " differs from its N128 M8 references");
        std::cout << "PASS q4 N64 M" << width * kRows
                  << (addResidual ? " residual" : " affine")
                  << " K=" << inputSize
                  << " groups=" << params.persistent_groups
                  << " exact=true\n";
      }
    }
  }
}

void run(const std::string &metallibPath) {
  MetalBackend backend(metallibPath);
  const uint64_t inputElements = uint64_t{kRows} * kInput;
  const uint64_t outputElements = uint64_t{kRows} * kOutput;
  const uint64_t weightElements = uint64_t{kInput} * kOutput;
  const uint64_t parameterElements = weightElements / kQuantGroup;

  MetalBuffer input = shared(
      backend, kMaximumBatch * inputElements * sizeof(__bf16), "q4-input");
  MetalBuffer weights = shared(backend, weightElements / 2, "q4-weights");
  MetalBuffer scales =
      shared(backend, parameterElements * sizeof(__bf16), "q4-scales");
  MetalBuffer biases =
      shared(backend, parameterElements * sizeof(__bf16), "q4-biases");
  MetalBuffer reference =
      shared(backend, kMaximumBatch * outputElements * sizeof(__bf16),
             "q4-reference");

  std::mt19937 random(7319);
  std::uniform_real_distribution<float> inputValues(-1.0f, 1.0f);
  std::uniform_real_distribution<float> parameters(-0.02f, 0.02f);
  auto *inputValuesPtr = static_cast<__bf16 *>(input.contents());
  for (uint64_t index = 0; index < kMaximumBatch * inputElements; ++index)
    inputValuesPtr[index] = __bf16(inputValues(random));
  auto *weight = static_cast<uint8_t *>(weights.contents());
  for (uint64_t index = 0; index < weightElements / 2; ++index)
    weight[index] = static_cast<uint8_t>(random());
  auto *scale = static_cast<__bf16 *>(scales.contents());
  auto *bias = static_cast<__bf16 *>(biases.contents());
  for (uint64_t index = 0; index < parameterElements; ++index) {
    scale[index] = __bf16(parameters(random));
    bias[index] = __bf16(parameters(random));
  }
  std::memset(reference.contents(), 0, reference.sizeBytes());

  // Every Q4 projection has one StorageN=256 representation. These compute
  // kernels consume it with TileN=128 for the four fixed DFlash batch widths.
  const Q4Params params{kOutput, kInput, kGroups};
  std::vector<ComputeDispatch> singles;
  std::memset(reference.contents(), 0, reference.sizeBytes());
  singles.clear();
  for (uint32_t lane = 0; lane < kMaximumBatch; ++lane) {
    singles.push_back(affine(
        "decode_linear_q4_n128",
        backend.view(input, uint64_t{lane} * inputElements * sizeof(__bf16),
                     inputElements * sizeof(__bf16)),
        weights, scales, biases,
        backend.view(reference,
                     uint64_t{lane} * outputElements * sizeof(__bf16),
                     outputElements * sizeof(__bf16)),
        params));
  }
  (void)backend.submitCommand(singles);

  // The pipelined narrow-projection kernel issues two quant groups before
  // either epilogue; its outputs must be byte-identical to the sequential M8.
  MetalBuffer paired = shared(backend, kMaximumBatch * outputElements *
                                           sizeof(__bf16),
                              "q4-paired-output");
  std::memset(paired.contents(), 0, paired.sizeBytes());
  std::vector<ComputeDispatch> pairedSingles;
  for (uint32_t lane = 0; lane < kMaximumBatch; ++lane) {
    pairedSingles.push_back(affine(
        "decode_linear_q4_n128_paired",
        backend.view(input, uint64_t{lane} * inputElements * sizeof(__bf16),
                     inputElements * sizeof(__bf16)),
        weights, scales, biases,
        backend.view(paired, uint64_t{lane} * outputElements * sizeof(__bf16),
                     outputElements * sizeof(__bf16)),
        params));
  }
  (void)backend.submitCommand(pairedSingles);
  if (std::memcmp(reference.contents(), paired.contents(), paired.sizeBytes()))
    fail("paired M8 projection differs from the sequential M8 projection");
  std::cout << "PASS q4 paired M8 exact=true\n";
  constexpr std::array<const char *, 3> genericPipelines{
      "decode_linear_q4_n128_m16", "decode_linear_q4_n128_m24",
      "decode_linear_q4_n128_m32"};
  for (uint32_t width = 2; width <= kMaximumBatch; ++width) {
    MetalBuffer candidate =
        shared(backend, uint64_t{width} * outputElements * sizeof(__bf16),
               "q4-generic-batch-output");
    std::memset(candidate.contents(), 0, candidate.sizeBytes());
    ComputeDispatch batch = affine(
        genericPipelines[width - 2],
        backend.view(input, 0,
                     uint64_t{width} * inputElements * sizeof(__bf16)),
        weights, scales, biases, candidate, params);
    const auto timing = backend.submitCommand({&batch, 1});
    const uint64_t comparedBytes =
        uint64_t{width} * outputElements * sizeof(__bf16);
    if (std::memcmp(reference.contents(), candidate.contents(), comparedBytes))
      fail("generic M" + std::to_string(width * kRows) +
           " projection differs from its M8 references");
    std::cout << "PASS q4 generic M" << width * kRows
              << " exact=true wall_seconds=" << timing.wallSeconds << '\n';
  }

  // Reuse the production-sized fixture without multiplying its long serial
  // workloads: persistent scratch is exercised on the small cases below.
  const Q4Params n64FullGrid{kOutput, kInput, kOutput / 64};
  for (uint32_t width = 1; width <= kMaximumBatch; ++width) {
    const uint64_t comparedBytes = uint64_t{width} * outputElements * sizeof(__bf16);
    MetalBuffer candidate = shared(backend, comparedBytes, "q4-n64-wide-output");
    std::memset(candidate.contents(), 0xA5, comparedBytes);
    ComputeDispatch dispatch = affine(
        kN64AffinePipelines[width - 1],
        backend.view(input, 0, uint64_t{width} * inputElements * sizeof(__bf16)),
        weights, scales, biases, candidate, n64FullGrid);
    (void)backend.submitCommand({&dispatch, 1});
    if (std::memcmp(reference.contents(), candidate.contents(), comparedBytes))
      fail("wide N64 M" + std::to_string(width * kRows) +
           " projection differs from its N128 M8 references");
    std::cout << "PASS q4 wide N64 M" << width * kRows
              << " exact=true\n";
  }

  const uint64_t m24Bytes = uint64_t{3} * outputElements * sizeof(__bf16);
  MetalBuffer gateUpReference =
      shared(backend, m24Bytes, "q4-m8-gate-up-reference");
  MetalBuffer gateScratch = shared(backend, m24Bytes, "q4-m24-gate");
  MetalBuffer combined = shared(backend, m24Bytes, "q4-m24-up-silu");
  std::array<ComputeDispatch, 3> gateUpSingles;
  for (uint32_t lane = 0; lane < gateUpSingles.size(); ++lane) {
    gateUpSingles[lane] = gateUp(
        "decode_linear_q4_n256_gate_up",
        backend.view(input, uint64_t{lane} * inputElements * sizeof(__bf16),
                     inputElements * sizeof(__bf16)),
        weights, scales, biases,
        backend.view(gateUpReference,
                     uint64_t{lane} * outputElements * sizeof(__bf16),
                     outputElements * sizeof(__bf16)),
        params);
  }
  (void)backend.submitCommand(gateUpSingles);
  std::array<ComputeDispatch, 2> splitDispatches{
      affine("decode_linear_q4_n256_m24",
             backend.view(input, 0, uint64_t{3} * inputElements * sizeof(__bf16)),
             weights, scales, biases, gateScratch, params),
      upSilu("decode_linear_q4_n256_up_silu_m24",
             backend.view(input, 0, uint64_t{3} * inputElements * sizeof(__bf16)),
             weights, scales, biases, gateScratch, combined, params)};
  const auto timing = backend.submitCommand(splitDispatches);
  if (std::memcmp(gateUpReference.contents(), combined.contents(), m24Bytes))
    fail("M24 split gate/up differs from its M8 references");
  std::cout << "PASS q4 M24 split-gate exact=true wall_seconds="
            << timing.wallSeconds << '\n';

  // A persistent threadgroup runs its tiles back-to-back on one input-sum
  // scratch: the next tile's prologue rewrites region 0, which the last
  // quant-group block still reads when K % 512 == 256. One threadgroup
  // striding over every tile must match one tile per threadgroup.
  constexpr uint32_t kPersistentOutput = 768;
  constexpr uint32_t kPersistentLanes = 3;
  for (const uint32_t persistentInput : {768u, 1280u}) {
    const uint64_t laneInputBytes =
        uint64_t{kRows} * persistentInput * sizeof(__bf16);
    const uint64_t laneOutputBytes =
        uint64_t{kRows} * kPersistentOutput * sizeof(__bf16);
    const uint64_t outputBytes = kPersistentLanes * laneOutputBytes;
    MetalBuffer singleTile = shared(backend, outputBytes, "q4-single-tile");
    MetalBuffer persistent = shared(backend, outputBytes, "q4-persistent");
    std::memset(singleTile.contents(), 0, outputBytes);
    std::memset(persistent.contents(), 0, outputBytes);
    const Q4Params oneTileEach{kPersistentOutput, persistentInput,
                               kPersistentOutput / 128};
    std::vector<ComputeDispatch> dispatches;
    for (uint32_t lane = 0; lane < kPersistentLanes; ++lane) {
      dispatches.push_back(affine(
          "decode_linear_q4_n128",
          backend.view(input, lane * laneInputBytes, laneInputBytes), weights,
          scales, biases,
          backend.view(singleTile, lane * laneOutputBytes, laneOutputBytes),
          oneTileEach));
    }
    const Q4Params oneGroup{kPersistentOutput, persistentInput, 1};
    dispatches.push_back(affine(
        "decode_linear_q4_n128_m24",
        backend.view(input, 0, kPersistentLanes * laneInputBytes), weights,
        scales, biases, persistent, oneGroup));
    (void)backend.submitCommand(dispatches);
    if (std::memcmp(singleTile.contents(), persistent.contents(), outputBytes))
      fail("persistent M24 projection at K=" + std::to_string(persistentInput) +
           " differs from its single-tile M8 references");
    std::cout << "PASS q4 persistent M24 K=" << persistentInput
              << " exact=true\n";
    checkN64Persistent(backend, input, weights, scales, biases, persistentInput,
                       random);
  }

  // One-lane tiles against the sequential N128 reference (lane 0). The
  // four-simdgroup N256 tile changes only the cooperative scope and must be
  // bitwise identical. The split tiles reduce four fp32 range sums before the
  // single bf16 rounding: within the derived bound of the sequential result,
  // and bitwise identical to each other since they share that reduction.
  using splash::ops::LinearEpilogue;
  const uint64_t laneBytes = outputElements * sizeof(__bf16);
  const MetalBuffer lane0Input = backend.view(input, 0, inputElements * sizeof(__bf16));
  const MetalBuffer lane0Reference = backend.view(reference, 0, laneBytes);
  MetalBuffer wide = shared(backend, laneBytes, "q4-n256-sg4-output");
  for (const uint32_t groups : {20u, kOutput / 256}) {
    std::memset(wide.contents(), 0, laneBytes);
    const Q4Params wideParams{kOutput, kInput, groups};
    (void)backend.submit(withThreads(affine("decode_linear_q4_n256_paired_sg4", lane0Input,
                                            weights, scales, biases, wide, wideParams), 128));
    if (std::memcmp(lane0Reference.contents(), wide.contents(), laneBytes))
      fail("four-simdgroup N256 M8 projection differs from the sequential N128 projection");
  }
  std::cout << "PASS q4 paired N256 sg4 M8 exact=true\n";

  MetalBuffer split32 = shared(backend, laneBytes, "q4-split32-output");
  MetalBuffer split64 = shared(backend, laneBytes, "q4-split64-output");
  const Q4Params split32Params{kOutput, kInput, kOutput / 32};
  const Q4Params split64Params{kOutput, kInput, kOutput / 64};
  std::memset(split32.contents(), 0, laneBytes);
  std::memset(split64.contents(), 0, laneBytes);
  (void)backend.submit(withThreads(affine("decode_linear_q4_n32_split4", lane0Input, weights,
                                          scales, biases, split32, split32Params), 128));
  (void)backend.submit(withThreads(affine("decode_linear_q4_n64_split4", lane0Input, weights,
                                          scales, biases, split64, split64Params), 256));
  requireSplitTolerance("q4 n32_split4 M8 vs N128", LinearEpilogue::None, lane0Reference,
                        split32, nullptr, nullptr, outputElements, kInput);
  requireSplitTolerance("q4 n64_split4 M8 vs N128", LinearEpilogue::None, lane0Reference,
                        split64, nullptr, nullptr, outputElements, kInput);
  if (std::memcmp(split32.contents(), split64.contents(), laneBytes))
    fail("Split32 and Split64 disagree although they share the four-partial reduction");
  std::cout << "PASS q4 split32/split64 M8 exact=true\n";

  MetalBuffer residual = shared(backend, laneBytes, "q4-residual");
  auto *residualValues = static_cast<__bf16 *>(residual.contents());
  for (uint64_t index = 0; index < outputElements; ++index)
    residualValues[index] = __bf16(inputValues(random));
  MetalBuffer residualReference = shared(backend, laneBytes, "q4-residual-reference");
  const Q4Params residualParams{kOutput, kInput, kGroups};
  (void)backend.submit(residualDispatch("decode_linear_q4_n128_residual", lane0Input, weights,
                                        scales, biases, residual, residualReference,
                                        residualParams));
  std::memset(split32.contents(), 0, laneBytes);
  std::memset(split64.contents(), 0, laneBytes);
  (void)backend.submit(withThreads(residualDispatch("decode_linear_q4_n32_split4_residual",
                                                    lane0Input, weights, scales, biases,
                                                    residual, split32, split32Params), 128));
  (void)backend.submit(withThreads(residualDispatch("decode_linear_q4_n64_split4_residual",
                                                    lane0Input, weights, scales, biases,
                                                    residual, split64, split64Params), 256));
  requireSplitTolerance("q4 n32_split4_residual M8 vs N128", LinearEpilogue::Residual,
                        residualReference, split32, &residual, nullptr, outputElements, kInput);
  requireSplitTolerance("q4 n64_split4_residual M8 vs N128", LinearEpilogue::Residual,
                        residualReference, split64, &residual, nullptr, outputElements, kInput);
  if (std::memcmp(split32.contents(), split64.contents(), laneBytes))
    fail("Split32 and Split64 residual outputs disagree");

  // Gate/up: both streams read the same weights, so lane 0 of the sequential
  // plain projection is the exact gate and up value of every element.
  MetalBuffer splitGateUp = shared(backend, laneBytes, "q4-split32-gate-up");
  std::memset(splitGateUp.contents(), 0, laneBytes);
  (void)backend.submit(withThreads(gateUp("decode_linear_q4_n32_split4_gate_up", lane0Input,
                                          weights, scales, biases, splitGateUp, split32Params),
                                   128));
  requireSplitTolerance("q4 n32_split4_gate_up M8 vs N256 gate/up", LinearEpilogue::GateUp,
                        backend.view(gateUpReference, 0, laneBytes), splitGateUp, nullptr,
                        &lane0Reference, outputElements, kInput);
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: q4_batched_projection_metal_test <metallib>\n";
      return 2;
    }
    try {
      run(argv[1]);
    } catch (const std::exception &error) {
      std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
      return 1;
    }
  }
  return 0;
}

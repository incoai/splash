#include "../../../runtime/metal/MetalBackend.hpp"
#include "metal/abi/Linear.h"

#import <Foundation/Foundation.h>

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

constexpr uint32_t kQuantGroup = 64;
constexpr uint32_t kTileRows = 32;

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "FAIL: " << message << '\n';
  std::exit(1);
}

MetalBuffer shared(MetalBackend &backend, uint64_t bytes, const char *label) {
  return backend.allocateBuffer(bytes, BufferStorage::Shared, label);
}

// The prefill policy switches tiles and cooperative scope by device and
// workload width (N256/N128, eight/four simdgroups). Every variant claims one
// shared per-element accumulation order; this test pins that claim down:
// on real projection shapes, all prefill tile variants of one epilogue must
// produce byte-identical outputs (and byte-identical fused output sums).
struct ProjectionShape {
  uint32_t rows;
  uint32_t inputSize;
  uint32_t outputSize;
};

void runShape(MetalBackend &backend, const ProjectionShape &shape,
              std::mt19937 &random) {
  const uint32_t rowTiles = (shape.rows + kTileRows - 1) / kTileRows;
  const uint64_t inputElements =
      uint64_t{rowTiles * kTileRows} * shape.inputSize;
  const uint64_t outputElements =
      uint64_t{rowTiles * kTileRows} * shape.outputSize;
  const uint64_t weightElements = uint64_t{shape.inputSize} * shape.outputSize;
  const uint64_t parameterElements = weightElements / kQuantGroup;
  const uint64_t sumElements =
      uint64_t{rowTiles * kTileRows} * (shape.inputSize / kQuantGroup);
  const uint64_t outputSumElements =
      uint64_t{rowTiles * kTileRows} * (shape.outputSize / kQuantGroup);

  MetalBuffer input =
      shared(backend, inputElements * sizeof(__bf16), "q4-prefill-input");
  MetalBuffer weights =
      shared(backend, weightElements / 2, "q4-prefill-weights");
  MetalBuffer scales =
      shared(backend, parameterElements * sizeof(__bf16), "q4-prefill-scales");
  MetalBuffer biases =
      shared(backend, parameterElements * sizeof(__bf16), "q4-prefill-biases");
  MetalBuffer residual =
      shared(backend, outputElements * sizeof(__bf16), "q4-prefill-residual");
  MetalBuffer gate =
      shared(backend, outputElements * sizeof(__bf16), "q4-prefill-gate");
  MetalBuffer sums = shared(backend, sumElements * sizeof(float), "q4-sums");

  std::uniform_real_distribution<float> inputValues(-1.0f, 1.0f);
  std::uniform_real_distribution<float> parameters(-0.02f, 0.02f);
  auto *inputPtr = static_cast<__bf16 *>(input.contents());
  for (uint64_t index = 0; index < inputElements; ++index)
    inputPtr[index] = __bf16(inputValues(random));
  auto *weightPtr = static_cast<uint8_t *>(weights.contents());
  for (uint64_t index = 0; index < weightElements / 2; ++index)
    weightPtr[index] = static_cast<uint8_t>(random());
  auto *scalePtr = static_cast<__bf16 *>(scales.contents());
  auto *biasPtr = static_cast<__bf16 *>(biases.contents());
  for (uint64_t index = 0; index < parameterElements; ++index) {
    scalePtr[index] = __bf16(parameters(random));
    biasPtr[index] = __bf16(parameters(random));
  }
  auto *residualPtr = static_cast<__bf16 *>(residual.contents());
  auto *gatePtr = static_cast<__bf16 *>(gate.contents());
  for (uint64_t index = 0; index < outputElements; ++index) {
    residualPtr[index] = __bf16(inputValues(random));
    gatePtr[index] = __bf16(inputValues(random));
  }

  const Q4PrefillParams params{shape.outputSize, shape.inputSize, shape.inputSize};
  {
    ComputeDispatch sum;
    sum.pipelineName = "prefill_linear_q4_sums32";
    sum.buffers = {{0, input}, {1, sums}};
    sum.bytes = {{2, &params, sizeof(params)}};
    sum.threadgroups = {rowTiles, 1, 1};
    sum.threadsPerThreadgroup = {256, 1, 1};
    (void)backend.submit(sum);
  }

  const uint64_t outputBytes = outputElements * sizeof(__bf16);
  const std::string label = "K" + std::to_string(shape.inputSize) + "N" +
      std::to_string(shape.outputSize) + "R" + std::to_string(shape.rows);
  const auto freshOutput = [&](const char *name) {
    MetalBuffer output = shared(backend, outputBytes, name);
    std::memset(output.contents(), 0xA5, outputBytes);
    return output;
  };
  auto runPlain = [&](const char *pipeline, uint32_t tileColumns,
                      uint32_t threads, MetalBuffer output) {
    ComputeDispatch dispatch;
    dispatch.pipelineName = pipeline;
    dispatch.buffers = {{0, input}, {1, weights}, {2, scales}, {3, biases},
                        {4, std::move(output)}, {5, sums}};
    dispatch.bytes = {{6, &params, sizeof(params)}};
    dispatch.threadgroups = {rowTiles, shape.outputSize / tileColumns, 1};
    dispatch.threadsPerThreadgroup = {threads, 1, 1};
    (void)backend.submit(dispatch);
  };
  auto runResidual = [&](const char *pipeline, uint32_t tileColumns,
                         uint32_t threads, MetalBuffer output) {
    ComputeDispatch dispatch;
    dispatch.pipelineName = pipeline;
    dispatch.buffers = {{0, input}, {1, weights}, {2, scales}, {3, biases},
                        {4, residual}, {5, std::move(output)}, {6, sums}};
    dispatch.bytes = {{7, &params, sizeof(params)}};
    dispatch.threadgroups = {rowTiles, shape.outputSize / tileColumns, 1};
    dispatch.threadsPerThreadgroup = {threads, 1, 1};
    (void)backend.submit(dispatch);
  };
  auto runUpSilu = [&](const char *pipeline, uint32_t tileColumns,
                       uint32_t threads, MetalBuffer output,
                       MetalBuffer outputSumsSink) {
    ComputeDispatch dispatch;
    dispatch.pipelineName = pipeline;
    dispatch.buffers = {{0, input}, {1, weights}, {2, scales}, {3, biases},
                        {4, gate}, {5, std::move(output)}, {6, sums},
                        {7, std::move(outputSumsSink)}};
    dispatch.bytes = {{8, &params, sizeof(params)}};
    dispatch.threadgroups = {rowTiles, shape.outputSize / tileColumns, 1};
    dispatch.threadsPerThreadgroup = {threads, 1, 1};
    (void)backend.submit(dispatch);
  };

  // Plain projection: N256/8 is the reference; N128/8 and N128/4 must match.
  {
    struct Variant {
      const char *pipeline;
      uint32_t tileColumns;
      uint32_t threads;
    };
    const std::vector<Variant> variants{
        {"prefill_linear_q4_n256", 256, 256},
        {"prefill_linear_q4_n128", 128, 256},
        {"prefill_linear_q4_n128_sg4", 128, 128}};
    MetalBuffer reference = freshOutput(variants.front().pipeline);
    runPlain(variants.front().pipeline, variants.front().tileColumns,
             variants.front().threads, reference);
    for (size_t index = 1; index < variants.size(); ++index) {
      MetalBuffer candidate = freshOutput(variants[index].pipeline);
      runPlain(variants[index].pipeline, variants[index].tileColumns,
               variants[index].threads, candidate);
      if (std::memcmp(reference.contents(), candidate.contents(), outputBytes))
        fail(std::string(variants[index].pipeline) +
             " differs from the N256 prefill reference at " + label);
      std::cout << "PASS q4 prefill " << variants[index].pipeline << " "
                << label << " exact=true\n";
    }
  }

  // Residual projection: same contract across its three tiles.
  {
    struct Variant {
      const char *pipeline;
      uint32_t tileColumns;
      uint32_t threads;
    };
    const std::vector<Variant> variants{
        {"prefill_linear_q4_n256_residual", 256, 256},
        {"prefill_linear_q4_n128_residual", 128, 256},
        {"prefill_linear_q4_n128_residual_sg4", 128, 128}};
    MetalBuffer reference = freshOutput(variants.front().pipeline);
    runResidual(variants.front().pipeline, variants.front().tileColumns,
                variants.front().threads, reference);
    for (size_t index = 1; index < variants.size(); ++index) {
      MetalBuffer candidate = freshOutput(variants[index].pipeline);
      runResidual(variants[index].pipeline, variants[index].tileColumns,
                  variants[index].threads, candidate);
      if (std::memcmp(reference.contents(), candidate.contents(), outputBytes))
        fail(std::string(variants[index].pipeline) +
             " differs from the N256 residual reference at " + label);
      std::cout << "PASS q4 prefill " << variants[index].pipeline << " "
                << label << " exact=true\n";
    }
  }

  // Fused up projection: activations and the fused output sums must both
  // match across the N256/8 and N128/4 kernels.
  {
    MetalBuffer reference = freshOutput("prefill_linear_q4_n256_up_silu_sums");
    MetalBuffer candidate =
        freshOutput("prefill_linear_q4_n128_up_silu_sums_sg4");
    MetalBuffer referenceSums = shared(
        backend, outputSumElements * sizeof(float), "up-sums-reference");
    MetalBuffer candidateSums =
        shared(backend, outputSumElements * sizeof(float), "up-sums-candidate");
    std::memset(referenceSums.contents(), 0, referenceSums.sizeBytes());
    std::memset(candidateSums.contents(), 0, candidateSums.sizeBytes());
    runUpSilu("prefill_linear_q4_n256_up_silu_sums", 256, 256, reference,
              referenceSums);
    runUpSilu("prefill_linear_q4_n128_up_silu_sums_sg4", 128, 128, candidate,
              candidateSums);
    if (std::memcmp(reference.contents(), candidate.contents(), outputBytes))
      fail("prefill_linear_q4_n128_up_silu_sums_sg4 differs from the N256 "
           "up/silu reference at " + label);
    if (std::memcmp(referenceSums.contents(), candidateSums.contents(),
                    referenceSums.sizeBytes()))
      fail("prefill_linear_q4_n128_up_silu_sums_sg4 output sums differ from "
           "the N256 reference at " + label);
    std::cout << "PASS q4 prefill prefill_linear_q4_n128_up_silu_sums_sg4 "
              << label << " exact=true\n";
  }
}

// A view of the leading inputs of wider weight rows (plane_input_size), as the
// ANE FFN split runs down's GPU share, equals bit for bit every prefill tile of
// those inputs copied out of each 256-column weight tile on their own.
void runLeadingInputs(MetalBackend &backend, std::mt19937 &random) {
  constexpr uint32_t rows = 176, outputs = 2048, wide = 2048, inputs = 1024, columnTile = 256;
  const uint32_t rowTiles = (rows + kTileRows - 1) / kTileRows, storage = rowTiles * kTileRows;
  const uint32_t tiles = outputs / columnTile, wideGroups = wide / kQuantGroup, groups = inputs / kQuantGroup;
  std::uniform_real_distribution<float> values(-1.0f, 1.0f), parameters(-0.02f, 0.02f);
  const auto filled = [&](uint64_t count, auto make, const char *label) {
    MetalBuffer buffer = shared(backend, count * sizeof(make()), label);
    auto *data = static_cast<decltype(make()) *>(buffer.contents());
    for (uint64_t index = 0; index < count; ++index) data[index] = make();
    return buffer;
  };
  const auto bf16 = [&] { return __bf16(values(random)); };
  MetalBuffer input = filled(uint64_t{storage} * inputs, bf16, "leading-input");
  MetalBuffer residual = filled(uint64_t{storage} * outputs, bf16, "leading-residual");
  MetalBuffer gate = filled(uint64_t{storage} * outputs, bf16, "leading-gate");
  MetalBuffer weights = filled(uint64_t{outputs} * wide / 2, [&] { return uint8_t(random()); }, "leading-weights");
  const uint64_t wideParameters = uint64_t{outputs} * wideGroups;
  MetalBuffer scales = filled(wideParameters, [&] { return __bf16(parameters(random)); }, "leading-scales");
  MetalBuffer biases = filled(wideParameters, [&] { return __bf16(parameters(random)); }, "leading-biases");
  // Each 256-column tile holds its groups in order: copy the first `groups`.
  const auto leading = [&](const MetalBuffer &source, uint64_t groupBytes, const char *label) {
    MetalBuffer copy = shared(backend, uint64_t{tiles} * groups * groupBytes, label);
    for (uint32_t tile = 0; tile < tiles; ++tile)
      std::memcpy(static_cast<uint8_t *>(copy.contents()) + uint64_t{tile} * groups * groupBytes,
                  static_cast<const uint8_t *>(source.contents()) + uint64_t{tile} * wideGroups * groupBytes,
                  uint64_t{groups} * groupBytes);
    return copy;
  };
  const MetalBuffer compactWeights = leading(weights, columnTile * kQuantGroup / 2, "leading-compact-weights");
  const MetalBuffer compactScales = leading(scales, columnTile * sizeof(__bf16), "leading-compact-scales");
  const MetalBuffer compactBiases = leading(biases, columnTile * sizeof(__bf16), "leading-compact-biases");
  const Q4PrefillParams compact{outputs, inputs, inputs}, view{outputs, inputs, wide};
  MetalBuffer sums = shared(backend, uint64_t{storage} * groups * sizeof(float), "leading-sums");
  {
    ComputeDispatch sum;
    sum.pipelineName = "prefill_linear_q4_sums32";
    sum.buffers = {{0, input}, {1, sums}};
    sum.bytes = {{2, &compact, sizeof(compact)}};
    sum.threadgroups = {rowTiles, 1, 1};
    sum.threadsPerThreadgroup = {256, 1, 1};
    (void)backend.submit(sum);
  }
  const uint64_t outputBytes = uint64_t{storage} * outputs * sizeof(__bf16);
  struct Kernel {
    const char *pipeline;
    uint32_t tileColumns, threads;
    enum { Plain, Residual, UpSilu } epilogue;
  };
  for (const Kernel kernel : std::vector<Kernel>{{"prefill_linear_q4_n256", 256, 256, Kernel::Plain},
                                                 {"prefill_linear_q4_n128", 128, 256, Kernel::Plain},
                                                 {"prefill_linear_q4_n128_sg4", 128, 128, Kernel::Plain},
                                                 {"prefill_linear_q4_n256_residual", 256, 256, Kernel::Residual},
                                                 {"prefill_linear_q4_n128_residual", 128, 256, Kernel::Residual},
                                                 {"prefill_linear_q4_n128_residual_sg4", 128, 128, Kernel::Residual},
                                                 {"prefill_linear_q4_n256_up_silu_sums", 256, 256, Kernel::UpSilu},
                                                 {"prefill_linear_q4_n128_up_silu_sums_sg4", 128, 128,
                                                  Kernel::UpSilu}}) {
    const auto run = [&](const MetalBuffer &w, const MetalBuffer &s, const MetalBuffer &b,
                         const Q4PrefillParams &params) {
      MetalBuffer output = shared(backend, outputBytes, kernel.pipeline);
      std::memset(output.contents(), 0xA5, outputBytes);
      ComputeDispatch dispatch;
      dispatch.pipelineName = kernel.pipeline;
      dispatch.buffers = {{0, input}, {1, w}, {2, s}, {3, b}};
      uint32_t next = 4;
      if (kernel.epilogue != Kernel::Plain) dispatch.buffers.push_back({next++, kernel.epilogue == Kernel::Residual ? residual : gate});
      dispatch.buffers.push_back({next++, output});
      dispatch.buffers.push_back({next++, sums});
      if (kernel.epilogue == Kernel::UpSilu)
        dispatch.buffers.push_back({next++, shared(backend, uint64_t{storage} * (outputs / kQuantGroup) * 4, "sink")});
      dispatch.bytes = {{next, &params, sizeof(params)}};
      dispatch.threadgroups = {rowTiles, outputs / kernel.tileColumns, 1};
      dispatch.threadsPerThreadgroup = {kernel.threads, 1, 1};
      (void)backend.submit(dispatch);
      return output;
    };
    const MetalBuffer expected = run(compactWeights, compactScales, compactBiases, compact);
    const MetalBuffer got = run(weights, scales, biases, view);
    if (std::memcmp(expected.contents(), got.contents(), outputBytes))
      fail(std::string(kernel.pipeline) + " over a view of leading inputs differs from them copied");
    std::cout << "PASS q4 prefill " << kernel.pipeline << " leading " << inputs << " of " << wide
              << " inputs exact=true\n";
  }
}

// A 32-row prefill chunk and the decode M32 kernels cover the same rows with
// different grid strategies (matrix grid vs persistent groups). Pin the shared
// accumulation-order claim across the phase boundary: both routes must
// produce identical bytes, so any future execution placement that moves small
// prefill chunks onto decode kernels stays numerics-neutral by construction.
void runDecodeCrossCheck(MetalBackend &backend, std::mt19937 &random) {
  const ProjectionShape shape{32, 6144, 5120};
  const uint64_t inputElements = uint64_t{shape.rows} * shape.inputSize;
  const uint64_t outputElements = uint64_t{shape.rows} * shape.outputSize;
  const uint64_t weightElements = uint64_t{shape.inputSize} * shape.outputSize;
  const uint64_t parameterElements = weightElements / kQuantGroup;
  const uint64_t sumElements =
      uint64_t{shape.rows} * (shape.inputSize / kQuantGroup);

  MetalBuffer input =
      shared(backend, inputElements * sizeof(__bf16), "q4-cross-input");
  MetalBuffer weights =
      shared(backend, weightElements / 2, "q4-cross-weights");
  MetalBuffer scales =
      shared(backend, parameterElements * sizeof(__bf16), "q4-cross-scales");
  MetalBuffer biases =
      shared(backend, parameterElements * sizeof(__bf16), "q4-cross-biases");
  MetalBuffer residual =
      shared(backend, outputElements * sizeof(__bf16), "q4-cross-residual");
  MetalBuffer sums = shared(backend, sumElements * sizeof(float), "q4-cross-sums");

  std::uniform_real_distribution<float> inputValues(-1.0f, 1.0f);
  std::uniform_real_distribution<float> parameters(-0.02f, 0.02f);
  auto *inputPtr = static_cast<__bf16 *>(input.contents());
  for (uint64_t index = 0; index < inputElements; ++index)
    inputPtr[index] = __bf16(inputValues(random));
  auto *weightPtr = static_cast<uint8_t *>(weights.contents());
  for (uint64_t index = 0; index < weightElements / 2; ++index)
    weightPtr[index] = static_cast<uint8_t>(random());
  auto *scalePtr = static_cast<__bf16 *>(scales.contents());
  auto *biasPtr = static_cast<__bf16 *>(biases.contents());
  for (uint64_t index = 0; index < parameterElements; ++index) {
    scalePtr[index] = __bf16(parameters(random));
    biasPtr[index] = __bf16(parameters(random));
  }
  auto *residualPtr = static_cast<__bf16 *>(residual.contents());
  for (uint64_t index = 0; index < outputElements; ++index)
    residualPtr[index] = __bf16(inputValues(random));

  const Q4PrefillParams prefillParams{shape.outputSize, shape.inputSize, shape.inputSize};
  {
    ComputeDispatch sum;
    sum.pipelineName = "prefill_linear_q4_sums32";
    sum.buffers = {{0, input}, {1, sums}};
    sum.bytes = {{2, &prefillParams, sizeof(prefillParams)}};
    sum.threadgroups = {1, 1, 1};
    sum.threadsPerThreadgroup = {256, 1, 1};
    (void)backend.submit(sum);
  }

  const uint64_t outputBytes = outputElements * sizeof(__bf16);
  const auto freshOutput = [&](const char *name) {
    MetalBuffer output = shared(backend, outputBytes, name);
    std::memset(output.contents(), 0xA5, outputBytes);
    return output;
  };

  const auto runPrefillPlain = [&](const char *pipeline, uint32_t tileColumns,
                                   MetalBuffer output) {
    ComputeDispatch prefill;
    prefill.pipelineName = pipeline;
    prefill.buffers = {{0, input}, {1, weights}, {2, scales}, {3, biases},
                       {4, std::move(output)}, {5, sums}};
    prefill.bytes = {{6, &prefillParams, sizeof(prefillParams)}};
    prefill.threadgroups = {1, shape.outputSize / tileColumns, 1};
    prefill.threadsPerThreadgroup = {256, 1, 1};
    (void)backend.submit(prefill);
  };
  const auto runDecodePlain = [&](const char *pipeline, uint32_t groups,
                                  MetalBuffer output) {
    const Q4Params decodeParams{shape.outputSize, shape.inputSize, groups};
    ComputeDispatch decode;
    decode.pipelineName = pipeline;
    decode.buffers = {{0, input}, {1, weights}, {2, scales}, {3, biases},
                      {4, std::move(output)}};
    decode.bytes = {{5, &decodeParams, sizeof(decodeParams)}};
    decode.threadgroups = {groups, 1, 1};
    decode.threadsPerThreadgroup = {256, 1, 1};
    (void)backend.submit(decode);
  };

  // Plain projection: prefill N256 (one row tile) vs both decode M32 tiles,
  // at the persistent group counts the decode policy actually dispatches.
  for (const auto &decodeCase :
       std::vector<std::pair<const char *, uint32_t>>{
           {"decode_linear_q4_n256_m32", shape.outputSize / 256},
           {"decode_linear_q4_n128_m32", shape.outputSize / 128}}) {
    MetalBuffer prefillOut = freshOutput("q4-cross-prefill");
    runPrefillPlain("prefill_linear_q4_n256", 256, prefillOut);
    MetalBuffer decodeOut = freshOutput("q4-cross-decode");
    runDecodePlain(decodeCase.first, decodeCase.second, decodeOut);
    const bool identical =
        std::memcmp(prefillOut.contents(), decodeOut.contents(), outputBytes) == 0;
    std::cout << "INFO q4 " << decodeCase.first
              << "-vs-prefill-n256 K" << shape.inputSize << "N"
              << shape.outputSize << "R32 exact="
              << (identical ? "true" : "false") << '\n';
    if (!identical)
      fail(std::string(decodeCase.first) +
           " differs from the 32-row prefill reference");
  }

  // Gate/up chain: prefill plain-gate + fused up/silu vs decode M32 plain
  // gate + M32 up/silu. The tail route must reproduce the chain byte-for-byte
  // (the fused output sums the prefill kernel writes are consumed only by
  // prefill kernels, which the tail route does not dispatch).
  {
    MetalBuffer gateScratchA = freshOutput("q4-cross-gate-a");
    MetalBuffer gateScratchB = freshOutput("q4-cross-gate-b");
    runPrefillPlain("prefill_linear_q4_n256", 256, gateScratchA);
    runDecodePlain("decode_linear_q4_n256_m32", shape.outputSize / 256,
                   gateScratchB);

    MetalBuffer prefillOut = freshOutput("q4-cross-up-prefill");
    MetalBuffer outputSumsA = shared(
        backend,
        uint64_t{shape.rows} * (shape.outputSize / kQuantGroup) * sizeof(float),
        "q4-cross-up-sums");
    ComputeDispatch up;
    up.pipelineName = "prefill_linear_q4_n256_up_silu_sums";
    up.buffers = {{0, input}, {1, weights}, {2, scales}, {3, biases},
                  {4, gateScratchA}, {5, prefillOut}, {6, sums},
                  {7, outputSumsA}};
    up.bytes = {{8, &prefillParams, sizeof(prefillParams)}};
    up.threadgroups = {1, shape.outputSize / 256, 1};
    up.threadsPerThreadgroup = {256, 1, 1};
    (void)backend.submit(up);

    MetalBuffer decodeOut = freshOutput("q4-cross-up-decode");
    const Q4Params decodeParams{shape.outputSize, shape.inputSize,
                                shape.outputSize / 256};
    ComputeDispatch upDecode;
    upDecode.pipelineName = "decode_linear_q4_n256_up_silu_m32";
    upDecode.buffers = {{0, input}, {1, weights}, {2, scales}, {3, biases},
                        {4, gateScratchB}, {5, decodeOut}};
    upDecode.bytes = {{6, &decodeParams, sizeof(decodeParams)}};
    upDecode.threadgroups = {decodeParams.persistent_groups, 1, 1};
    upDecode.threadsPerThreadgroup = {256, 1, 1};
    (void)backend.submit(upDecode);

    const bool identical =
        std::memcmp(prefillOut.contents(), decodeOut.contents(), outputBytes) == 0;
    std::cout << "INFO q4 decode-up-silu-m32-vs-prefill-chain K"
              << shape.inputSize << "N" << shape.outputSize << "R32 exact="
              << (identical ? "true" : "false") << '\n';
    if (!identical)
      fail("decode M32 up/silu differs from the prefill gate/up chain");
  }

  // Residual projection: prefill N256 residual vs decode M32 N128 residual.
  {
    MetalBuffer prefillOut = freshOutput("q4-cross-prefill-residual");
    ComputeDispatch prefill;
    prefill.pipelineName = "prefill_linear_q4_n256_residual";
    prefill.buffers = {{0, input}, {1, weights}, {2, scales}, {3, biases},
                       {4, residual}, {5, prefillOut}, {6, sums}};
    prefill.bytes = {{7, &prefillParams, sizeof(prefillParams)}};
    prefill.threadgroups = {1, shape.outputSize / 256, 1};
    prefill.threadsPerThreadgroup = {256, 1, 1};
    (void)backend.submit(prefill);

    MetalBuffer decodeOut = freshOutput("q4-cross-decode-residual");
    const Q4Params decodeParams{shape.outputSize, shape.inputSize,
                                shape.outputSize / 128};
    ComputeDispatch decode;
    decode.pipelineName = "decode_linear_q4_n128_residual_m32";
    decode.buffers = {{0, input}, {1, weights}, {2, scales}, {3, biases},
                      {4, residual}, {5, decodeOut}};
    decode.bytes = {{6, &decodeParams, sizeof(decodeParams)}};
    decode.threadgroups = {decodeParams.persistent_groups, 1, 1};
    decode.threadsPerThreadgroup = {256, 1, 1};
    (void)backend.submit(decode);

    const bool identical =
        std::memcmp(prefillOut.contents(), decodeOut.contents(), outputBytes) == 0;
    std::cout << "INFO q4 decode-m32-residual-vs-prefill-n256 K"
              << shape.inputSize << "N" << shape.outputSize << "R32 exact="
              << (identical ? "true" : "false") << '\n';
    if (!identical)
      fail("decode M32 residual differs from the 32-row prefill reference");
  }
}

void run(const std::string &metallibPath) {
  MetalBackend backend(metallibPath);
  std::mt19937 random(20260920);
  // The Qwen3.8-27B projection shapes: GDN input (5120->16640), attention
  // input (5120->14336), mixer output (6144->5120) and FFN down
  // (17408->5120). Rows cover a full chunk, a contended slice and a tail.
  for (const ProjectionShape shape : std::vector<ProjectionShape>{
           {1, 2048, 2048}, {31, 2048, 2048}, {33, 2048, 2048}, {176, 2048, 2048},
           {64, 6144, 5120}, {256, 5120, 14336}, {2048, 17408, 5120}})
    runShape(backend, shape, random);
  runLeadingInputs(backend, random);
  runDecodeCrossCheck(backend, random);
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: q4_prefill_projection_metal_test <metallib>\n";
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

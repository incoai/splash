#include "AffineQ4Fixture.hpp"
#include "../../../runtime/metal/MetalBackend.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/Linear.h"
#include "ops/Linear.hpp"
#include "tuning/LinearNumerics.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <utility>
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
                       const Q4PersistentParams &params) {
  ComputeDispatch result;
  result.pipelineName = std::move(pipeline);
  result.buffers = {{0, std::move(input)},
                    {1, std::move(weights)},
                    {2, std::move(scales)},
                    {3, std::move(biases)},
                    {4, std::move(output)}};
  result.bytes = {{5, &params, sizeof(params)}};
  result.threadgroups = {params.groups, 1, 1};
  result.threadsPerThreadgroup = {256, 1, 1};
  return result;
}

ComputeDispatch gateUp(std::string pipeline, MetalBuffer input,
                       MetalBuffer weights, MetalBuffer scales,
                       MetalBuffer biases, MetalBuffer output,
                       const Q4PersistentParams &params) {
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
  result.threadgroups = {params.groups, 1, 1};
  result.threadsPerThreadgroup = {256, 1, 1};
  return result;
}

ComputeDispatch withThreads(ComputeDispatch dispatch, uint32_t threads) {
  dispatch.threadsPerThreadgroup = {threads, 1, 1};
  return dispatch;
}

ComputeDispatch upSilu(std::string pipeline, MetalBuffer input,
                       MetalBuffer weights, MetalBuffer scales,
                       MetalBuffer biases, MetalBuffer gate,
                       MetalBuffer output, const Q4PersistentParams &params) {
  ComputeDispatch result;
  result.pipelineName = std::move(pipeline);
  result.buffers = {{0, std::move(input)},
                    {1, std::move(weights)},
                    {2, std::move(scales)},
                    {3, std::move(biases)},
                    {4, std::move(gate)},
                    {5, std::move(output)}};
  result.bytes = {{6, &params, sizeof(params)}};
  result.threadgroups = {params.groups, 1, 1};
  result.threadsPerThreadgroup = {256, 1, 1};
  return result;
}

// The Split128 tile (ops::LinearTile::Split128) through the production
// encoder. Every buffer has exactly the bytes its plan asks for
// (LinearPlan::scratchSize and gateScratchBytes, as the decode arena sizes
// them) followed by a guard band, and the split partials are set to NaN
// before every dispatch, so a partition that reads a partial before its
// writer published it poisons the output. The poison word is an fp32 NaN
// made of two bf16 NaNs, so it poisons bf16 buffers too.
constexpr uint64_t kGuardBytes = 4096;
constexpr uint32_t kPoisonNaN = 0x7FC07FC0u;
constexpr uint16_t kPoisonBf16 = 0x7FC0u;
constexpr uint32_t kSplitMaximumRows = kRows * kMaximumBatch;

struct Guarded final {
  MetalBuffer backing, view;
  uint64_t bytes = 0;
  Guarded(MetalBackend &backend, uint64_t size) : bytes(size) {
    backing = shared(backend, bytes + kGuardBytes, "q4-split-guarded");
    std::memset(backing.contents(), 0, bytes);
    std::memset(static_cast<uint8_t *>(backing.contents()) + bytes, 0x5a, kGuardBytes);
    view = backend.view(backing, 0, bytes);
  }
  [[nodiscard]] bool intact() const {
    const auto *guard = static_cast<const uint8_t *>(backing.contents()) + bytes;
    return std::all_of(guard, guard + kGuardBytes, [](uint8_t byte) { return byte == 0x5a; });
  }
};

struct SplitCase final {
  splash::ops::LinearMatrix matrix;
  splash::ops::LinearEpilogue epilogue;
  splash::ops::FloatOutput destination;
  uint32_t splits;
};

std::string describe(const SplitCase &c) {
  using splash::ops::LinearEpilogue;
  return "Split128 " + std::to_string(c.matrix.outputSize) + "x" + std::to_string(c.matrix.inputSize) + " S" +
         std::to_string(c.splits) +
         (c.epilogue == LinearEpilogue::Residual ? " residual" : c.epilogue == LinearEpilogue::GateUp ? " gate/up"
                                                                                                        : " plain") +
         (c.destination == splash::ops::FloatOutput::Float32 ? " fp32" : "");
}

// Runs `plan` over `rows` rows of the operands (`input`, and `residual` for a
// residual epilogue) twice back to back in one command buffer, the output
// set to NaN in between, and returns the second run's output bytes: every
// element is stored again through the counters the first run left. Each
// operand is bound from a buffer of the widest step's rows whose rows past
// the plan's, and its guard band, hold bf16 NaN, so a read past the plan's
// rows or inputs that reaches a stored element poisons it. Fails on a write
// past any buffer or a counter left nonzero.
std::vector<uint8_t> runSplitPlan(MetalBackend &backend, const splash::ops::Linear &linear,
                                  const splash::ops::LinearPlan &plan, const SplitCase &c,
                                  const splash::ops::Projection &up, const splash::ops::Projection &gate,
                                  const uint16_t *input, const uint16_t *residual, const MetalBuffer &poison) {
  using namespace splash::ops;
  const auto [n, k] = c.matrix;
  const uint32_t rows = plan.storageRows();
  const LinearScratchSize scratch = plan.scratchSize();
  Guarded output(backend, uint64_t{rows} * n * elementBytes(c.destination));
  Guarded partials(backend, scratch.partials), counters(backend, scratch.counters);
  const uint64_t gateBytes = plan.gateScratchBytes();
  std::optional<Guarded> gateScratch;
  if (gateBytes) gateScratch.emplace(backend, gateBytes);
  const auto operand = [&](const uint16_t *values, uint32_t width) {
    const uint64_t used = uint64_t{rows} * width, capacity = uint64_t{kSplitMaximumRows} * width + kGuardBytes / 2;
    MetalBuffer buffer = shared(backend, capacity * 2, "q4-split-operand");
    auto *data = static_cast<uint16_t *>(buffer.contents());
    std::copy_n(values, used, data);
    std::fill(data + used, data + capacity, kPoisonBf16);
    return backend.view(buffer, 0, used * 2);
  };
  const bool residualEpilogue = c.epilogue == LinearEpilogue::Residual;
  LinearBuffers buffers{operand(input, k), output.view, {}, residualEpilogue ? operand(residual, n) : MetalBuffer{},
                        gateScratch ? gateScratch->view : MetalBuffer{}, {},
                        LinearScratch{{}, {}, partials.view, counters.view}};
  splash::metal::CommandGraph projection;
  linear.add(projection, buffers, up, plan, c.epilogue == LinearEpilogue::GateUp ? &gate : nullptr);
  // The poison before each of the plan's dispatches (gate/up runs two), and
  // the output's before the second run.
  splash::metal::CommandGraph poisoning;
  poisoning.add("test_copy_u32", {poison, partials.view}, uint32_t(scratch.partials / 4),
                {uint32_t((scratch.partials / 4 + 255) / 256), 1, 1}, {256, 1, 1});
  poisoning.add("test_copy_u32", {poison, output.view}, uint32_t(output.bytes / 4),
                {uint32_t((output.bytes / 4 + 255) / 256), 1, 1}, {256, 1, 1});
  std::vector<ComputeDispatch> dispatches;
  for (uint32_t run = 0; run < 2; ++run) {
    if (run) dispatches.push_back(poisoning.dispatches()[1]);
    for (const ComputeDispatch &dispatch : projection.dispatches()) {
      dispatches.push_back(poisoning.dispatches()[0]);
      dispatches.push_back(dispatch);
    }
  }
  (void)backend.submitCommandAsync(dispatches).wait();
  const auto *count = static_cast<const uint32_t *>(counters.view.contents());
  if (!output.intact() || !partials.intact() || !counters.intact() || (gateScratch && !gateScratch->intact()) ||
      !std::all_of(count, count + scratch.counters / 4, [](uint32_t value) { return value == 0; }))
    fail(describe(c) + " M" + std::to_string(rows) + ": a write past a buffer or a counter left nonzero");
  const auto *bytes = static_cast<const uint8_t *>(output.view.contents());
  return {bytes, bytes + output.bytes};
}

// One Split128 configuration at every batch width: each lane's rows at 16,
// 24 and 32 rows are the bytes of that lane alone at 8 rows, and the 8-row
// outputs are within the derived bf16 bound of the sequential N128 tile
// (N256 for fused gate/up, with the exact gate and up projections of the
// plain N128 tile), tuning/LinearNumerics.hpp.
void splitCase(MetalBackend &backend, const SplitCase &c) {
  using namespace splash::ops;
  using namespace splash::ops::tuning;
  const auto [n, k] = c.matrix;
  const Linear linear(backend.capabilities());
  const Projection up = splash::test::deterministicQ4Projection(backend, c.matrix, 31);
  const Projection gate = splash::test::deterministicQ4Projection(backend, c.matrix, 157);
  MetalBuffer input = shared(backend, uint64_t{kSplitMaximumRows} * k * 2, "q4-split-input");
  MetalBuffer residual = shared(backend, uint64_t{kSplitMaximumRows} * n * 2, "q4-split-residual");
  std::mt19937 random(n * 31 + k);
  std::uniform_real_distribution<float> values(-1.0f, 1.0f);
  for (auto [buffer, count] : {std::pair{&input, uint64_t{kSplitMaximumRows} * k},
                               std::pair{&residual, uint64_t{kSplitMaximumRows} * n}}) {
    auto *data = static_cast<uint16_t *>(buffer->contents());
    for (uint64_t i = 0; i < count; ++i) data[i] = floatToBf16(values(random));
  }
  const LinearConfig config{LinearTile::Split128, 0, LinearSimdgroups::Eight, c.splits};
  const auto plan = [&](uint32_t lanes) {
    return Linear::plan({c.matrix, lanes * kRows, LinearPhase::Decode, c.epilogue}, config, c.destination);
  };
  // The poison covers the widest step's partials, which hold its output too.
  const LinearScratchSize widest = plan(kMaximumBatch).scratchSize();
  MetalBuffer poison = shared(backend, widest.partials, "q4-split-poison");
  std::fill_n(static_cast<uint32_t *>(poison.contents()), widest.partials / 4, kPoisonNaN);
  const uint64_t element = elementBytes(c.destination), laneBytes = uint64_t{kRows} * n * element;
  const auto *inputValues = static_cast<const uint16_t *>(input.contents());
  const auto *residualValues = static_cast<const uint16_t *>(residual.contents());
  // Each lane alone, at 8 rows.
  std::vector<uint8_t> lanes;
  for (uint32_t lane = 0; lane < kMaximumBatch; ++lane) {
    const auto bytes = runSplitPlan(backend, linear, plan(1), c, up, gate, inputValues + uint64_t{lane} * kRows * k,
                                    residualValues + uint64_t{lane} * kRows * n, poison);
    lanes.insert(lanes.end(), bytes.begin(), bytes.end());
  }
  for (uint32_t width = 2; width <= kMaximumBatch; ++width)
    if (runSplitPlan(backend, linear, plan(width), c, up, gate, inputValues, residualValues, poison) !=
        std::vector<uint8_t>(lanes.begin(), lanes.begin() + width * laneBytes))
      fail(describe(c) + " M" + std::to_string(width * kRows) + " differs from its lanes at 8 rows");
  // The sequential tile of lane 0 and, for gate/up, its exact gate and up,
  // into bf16: both destinations compare as bf16, since an fp32 output rounds
  // to its bf16 plan's.
  const auto sequential = [&](LinearEpilogue epilogue, const Projection &weights, LinearTile tile) {
    const LinearPlan reference =
        Linear::plan({c.matrix, kRows, LinearPhase::Decode, epilogue},
                     {tile, n / (tile == LinearTile::N256 ? 256 : 128)}, FloatOutput::BFloat16);
    MetalBuffer output = shared(backend, uint64_t{kRows} * n * 2, "q4-split-reference");
    MetalBuffer gateScratch = shared(backend, std::max<uint64_t>(reference.gateScratchBytes(), 2), "q4-split-gate");
    splash::metal::CommandGraph graph;
    linear.add(graph, {input, output, {}, epilogue == LinearEpilogue::Residual ? residual : MetalBuffer{},
                       reference.gateScratchBytes() ? gateScratch : MetalBuffer{}, {}},
               weights, reference, epilogue == LinearEpilogue::GateUp ? &gate : nullptr);
    (void)backend.submitCommandAsync(graph.dispatches()).wait();
    return output;
  };
  const bool gateUp = c.epilogue == LinearEpilogue::GateUp;
  const MetalBuffer exact = sequential(c.epilogue, up, gateUp ? LinearTile::N256 : LinearTile::N128);
  const MetalBuffer exactGate = gateUp ? sequential(LinearEpilogue::None, gate, LinearTile::N128) : MetalBuffer{};
  const MetalBuffer exactUp = gateUp ? sequential(LinearEpilogue::None, up, LinearTile::N128) : MetalBuffer{};
  const uint64_t elements = uint64_t{kRows} * n;
  const auto bf16At = [](const MetalBuffer &buffer, uint64_t i) {
    return bf16ToFloat(static_cast<const uint16_t *>(buffer.contents())[i]);
  };
  const auto actualAt = [&](uint64_t i) {
    return element == 4 ? bf16ToFloat(floatToBf16(reinterpret_cast<const float *>(lanes.data())[i]))
                        : bf16ToFloat(reinterpret_cast<const uint16_t *>(lanes.data())[i]);
  };
  float maxAbs = 0, maxDiff = 0;
  for (uint64_t i = 0; i < elements; ++i) maxAbs = std::max(maxAbs, std::fabs(bf16At(exact, i)));
  const float slack = reassociationSlack(k, maxAbs);
  for (uint64_t i = 0; i < elements; ++i) {
    SplitReference reference{bf16At(exact, i)};
    if (c.epilogue == LinearEpilogue::Residual) reference.residual = bf16ToFloat(residualValues[i]);
    if (gateUp) {
      reference.gate = bf16At(exactGate, i);
      reference.up = bf16At(exactUp, i);
    }
    const float actual = actualAt(i);
    maxDiff = std::max(maxDiff, std::fabs(actual - reference.value));
    if (!withinSplitTolerance(actual, c.epilogue, reference, slack))
      fail(describe(c) + " element " + std::to_string(i) + " actual=" + std::to_string(actual) +
           " sequential=" + std::to_string(reference.value) +
           " bound=" + std::to_string(splitTolerance(c.epilogue, reference, slack)));
  }
  std::cout << "PASS q4 " << describe(c) << " M8-M32 lanes exact=true within_bound=true max_abs_diff=" << maxDiff
            << '\n';
}

void splitTiles(MetalBackend &backend) {
  using splash::ops::FloatOutput;
  using splash::ops::LinearEpilogue;
  // The 35B mixer output; 17408 inputs in 8 or 9 blocks per partition; 25600
  // inputs, the largest production K (the 27B draft's context projection), in
  // 25 blocks per partition at four splits and 12 or 13 at eight; 5 and 9
  // blocks, partitions of one to three; the 248320-column vocabulary head, the
  // largest N, over two single-block partitions.
  struct Shape {
    splash::ops::LinearMatrix matrix;
    std::vector<uint32_t> splits;
  };
  for (const Shape &shape : {Shape{{2048, 4096}, {2, 4, 8}}, Shape{{512, 17408}, {8}}, Shape{{256, 25600}, {4, 8}},
                             Shape{{768, 1280}, {2, 4}}, Shape{{5120, 2304}, {2, 4, 8}}, Shape{{248320, 512}, {2}}})
    for (const uint32_t splits : shape.splits) {
      for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp})
        splitCase(backend, {shape.matrix, epilogue, FloatOutput::BFloat16, splits});
      splitCase(backend, {shape.matrix, LinearEpilogue::None, FloatOutput::Float32, splits});
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

  // Every Q4 projection has one packed 256-column storage representation.
  // These compute kernels consume it with TileN=128 for the four fixed DFlash
  // batch widths.
  const Q4PersistentParams params{kOutput, kInput, kGroups};
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
  (void)backend.submitCommandAsync(singles).wait();

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
  (void)backend.submitCommandAsync(pairedSingles).wait();
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
    const auto timing = backend.submitCommandAsync({&batch, 1}).wait();
    const uint64_t comparedBytes =
        uint64_t{width} * outputElements * sizeof(__bf16);
    if (std::memcmp(reference.contents(), candidate.contents(), comparedBytes))
      fail("generic M" + std::to_string(width * kRows) +
           " projection differs from its M8 references");
    std::cout << "PASS q4 generic M" << width * kRows
              << " exact=true wall_seconds=" << timing.wallSeconds << '\n';
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
  (void)backend.submitCommandAsync(gateUpSingles).wait();
  std::array<ComputeDispatch, 2> splitDispatches{
      affine("decode_linear_q4_n256_m24",
             backend.view(input, 0, uint64_t{3} * inputElements * sizeof(__bf16)),
             weights, scales, biases, gateScratch, params),
      upSilu("decode_linear_q4_n256_up_silu_m24",
             backend.view(input, 0, uint64_t{3} * inputElements * sizeof(__bf16)),
             weights, scales, biases, gateScratch, combined, params)};
  const auto timing = backend.submitCommandAsync(splitDispatches).wait();
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
    const Q4PersistentParams oneTileEach{kPersistentOutput, persistentInput,
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
    const Q4PersistentParams oneGroup{kPersistentOutput, persistentInput, 1};
    dispatches.push_back(affine(
        "decode_linear_q4_n128_m24",
        backend.view(input, 0, kPersistentLanes * laneInputBytes), weights,
        scales, biases, persistent, oneGroup));
    (void)backend.submitCommandAsync(dispatches).wait();
    if (std::memcmp(singleTile.contents(), persistent.contents(), outputBytes))
      fail("persistent M24 projection at K=" + std::to_string(persistentInput) +
           " differs from its single-tile M8 references");
    std::cout << "PASS q4 persistent M24 K=" << persistentInput
              << " exact=true\n";
  }

  // The one-lane four-simdgroup N256 tile against the sequential N128
  // reference (lane 0): it changes only the cooperative scope and must be
  // bitwise identical.
  const uint64_t laneBytes = outputElements * sizeof(__bf16);
  const MetalBuffer lane0Input = backend.view(input, 0, inputElements * sizeof(__bf16));
  const MetalBuffer lane0Reference = backend.view(reference, 0, laneBytes);
  MetalBuffer wide = shared(backend, laneBytes, "q4-n256-sg4-output");
  for (const uint32_t groups : {20u, kOutput / 256}) {
    std::memset(wide.contents(), 0, laneBytes);
    const Q4PersistentParams wideParams{kOutput, kInput, groups};
    (void)backend.submit(withThreads(affine("decode_linear_q4_n256_paired_sg4", lane0Input,
                                            weights, scales, biases, wide, wideParams), 128));
    if (std::memcmp(lane0Reference.contents(), wide.contents(), laneBytes))
      fail("four-simdgroup N256 M8 projection differs from the sequential N128 projection");
  }
  std::cout << "PASS q4 paired N256 sg4 M8 exact=true\n";

  splitTiles(backend);
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

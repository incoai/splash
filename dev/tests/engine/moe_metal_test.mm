// Sparse MoE against a CPU reference with real Q4 expert weights: routing,
// expert grouping (including partially filled tiles), the grouped gate/up and
// down tiles, and the combine, for every operator-owned M8/M32 plan and the
// Apple9 four-simdgroup decode tiles (bitwise against the shipped tile). Routing
// fixtures cover dispersed, concentrated and skewed expert utilization with
// hidden width 1024 and intermediate width 512.
#include "../../../runtime/metal/CommandGraph.hpp"
#include "../../../runtime/metal/MetalBackend.hpp"
#include "../../../runtime/model/WeightStore.hpp"
#include "../../../runtime/ops/MoE.hpp"
#include "metal/abi/MoE.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::model::q4PackedBytes;
using splash::ops::ExpertQ4Projection;
using splash::ops::kMoeRouteWideRows;
using splash::ops::MoE;
using splash::ops::MoeBuffers;
using splash::ops::MoeConfig;
using splash::ops::MoeExpertSimdgroups;
using splash::ops::MoeExpertTile;
using splash::ops::MoePlan;
using splash::ops::MoeShape;
using splash::ops::MoeWeights;
using splash::ops::Q8Projection;

// 16 quant groups on the hidden side, so the 32-row expert tiles refill their
// staged scales and biases several times, as the production shapes do.
constexpr uint32_t kHidden = 1024;
constexpr uint32_t kIntermediate = 512;
constexpr uint32_t kExperts = 8;
constexpr uint32_t kTopK = 2;
constexpr uint32_t kRoutesPerRow = kTopK + 1;
// Covers both router tiles: 8-row tiles below kMoeRouteWideRows and 32-row
// tiles with a ragged 8-row tail above it.
constexpr uint32_t kMaximumRows = 520;
constexpr uint32_t kStorageN = 256;

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "FAIL: " << message << '\n';
  std::exit(1);
}

void require(bool condition, const std::string &message) {
  if (!condition)
    fail(message);
}

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
  uint32_t next() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(state_ >> 33);
  }
  float unit() { return static_cast<float>(next() & 0xFFFFFF) / 8388608.0F - 1.0F; }

private:
  uint64_t state_;
};

MetalBuffer shared(MetalBackend &backend, uint64_t bytes, const char *label) {
  MetalBuffer buffer = backend.allocateBuffer(bytes, BufferStorage::Shared, label);
  std::memset(buffer.contents(), 0, buffer.sizeBytes());
  return buffer;
}

float bf16(float value) { return float(__bf16(value)); }

// One expert's Q4 slab in StorageN=256 order: [weights][scales][biases], the
// parameter of output n and quant group g at (tile(n) * groups + g) * 256 +
// n % 256 and its 64 nibbles right after the previous column's.
struct ExpertSlab final {
  std::vector<uint8_t> nibbles; // [n][k] dequantized as q * scale + bias
  std::vector<float> scales;    // [parameter]
  std::vector<float> biases;    // [parameter]
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;

  [[nodiscard]] uint32_t parameter(uint32_t n, uint32_t g) const {
    const uint32_t groups = inputSize / 64;
    return ((n / kStorageN) * groups + g) * kStorageN + n % kStorageN;
  }
  [[nodiscard]] float weight(uint32_t n, uint32_t k) const {
    return nibbles[uint64_t{n} * inputSize + k];
  }
};

ExpertSlab randomSlab(Random &random, uint32_t outputSize, uint32_t inputSize) {
  ExpertSlab slab;
  slab.outputSize = outputSize;
  slab.inputSize = inputSize;
  slab.nibbles.resize(uint64_t{outputSize} * inputSize);
  for (uint8_t &value : slab.nibbles)
    value = static_cast<uint8_t>(random.next() & 15);
  const uint64_t parameters = uint64_t{outputSize} * inputSize / 64;
  slab.scales.resize(parameters);
  slab.biases.resize(parameters);
  for (uint64_t index = 0; index < parameters; ++index) {
    slab.scales[index] = bf16(0.01F + 0.01F * (random.unit() + 1.0F));
    slab.biases[index] = bf16(-0.2F + 0.05F * random.unit());
  }
  return slab;
}

void packSlab(const ExpertSlab &slab, uint8_t *destination) {
  const uint64_t elements = uint64_t{slab.outputSize} * slab.inputSize;
  const uint32_t groups = slab.inputSize / 64;
  auto *scales = reinterpret_cast<__bf16 *>(destination + elements / 2);
  auto *biases = reinterpret_cast<__bf16 *>(destination + elements / 2 + elements / 32);
  for (uint32_t n = 0; n < slab.outputSize; ++n) {
    for (uint32_t k = 0; k < slab.inputSize; ++k) {
      const uint64_t nibble = uint64_t{slab.parameter(n, k / 64)} * 64 + k % 64;
      uint8_t &byte = destination[nibble / 2];
      const uint8_t value = slab.weight(n, k);
      byte = static_cast<uint8_t>((nibble & 1) ? (byte & 0x0F) | (value << 4)
                                               : (byte & 0xF0) | value);
    }
    for (uint32_t g = 0; g < groups; ++g) {
      scales[slab.parameter(n, g)] = __bf16(slab.scales[slab.parameter(n, g)]);
      biases[slab.parameter(n, g)] = __bf16(slab.biases[slab.parameter(n, g)]);
    }
  }
}

// y = W x with the kernels' rounding points: fp32 group sums, bf16 result.
std::vector<float> project(const ExpertSlab &slab, const std::vector<float> &x) {
  std::vector<float> result(slab.outputSize);
  const uint32_t groups = slab.inputSize / 64;
  for (uint32_t n = 0; n < slab.outputSize; ++n) {
    float total = 0.0F;
    for (uint32_t g = 0; g < groups; ++g) {
      float dot = 0.0F;
      float sum = 0.0F;
      for (uint32_t k = g * 64; k < g * 64 + 64; ++k) {
        dot += x[k] * slab.weight(n, k);
        sum += x[k];
      }
      total += dot * slab.scales[slab.parameter(n, g)] +
               sum * slab.biases[slab.parameter(n, g)];
    }
    result[n] = bf16(total);
  }
  return result;
}

float silu(float value) { return value / (1.0F + std::exp(-value)); }

// Expert e reads input column e, allowing each fixture to produce its desired
// routing distribution through the production router without CPU routing
// injection. The shared gate remains a scalar bias-only Q8 projection.
Q8Projection fixtureRouter(MetalBackend &backend, bool sharedGate) {
  const uint64_t elements = uint64_t{kStorageN} * kHidden;
  Q8Projection projection{shared(backend, elements, "moe-router-weights"),
                          shared(backend, elements / 32, "moe-router-scales"),
                          shared(backend, elements / 32, "moe-router-biases"),
                          kStorageN, kHidden};
  if (!sharedGate) {
    auto *weights = static_cast<uint8_t *>(projection.weights.contents());
    auto *scales = static_cast<__bf16 *>(projection.scales.contents());
    for (uint32_t expert = 0; expert < kExperts; ++expert) {
      weights[expert * 64 + expert] = 1;
      scales[expert] = __bf16(4.0F);
    }
    return projection;
  }
  auto *values = static_cast<__bf16 *>(projection.biases.contents());
  for (uint32_t group = 0; group < kHidden / 64; ++group) {
    for (uint32_t output = 0; output < kStorageN; ++output) {
      const float value = output ? 0.0F : 0.003F;
      values[group * kStorageN + output] = __bf16(value);
    }
  }
  return projection;
}

struct Experts final {
  std::vector<ExpertSlab> slabs;
  ExpertQ4Projection projection;
};

Experts randomExperts(MetalBackend &backend, Random &random,
                      uint32_t experts, uint32_t outputSize,
                      uint32_t inputSize, const char *label) {
  Experts result;
  const uint64_t stride = q4PackedBytes(outputSize, inputSize);
  result.projection = {shared(backend, uint64_t{experts} * stride, label),
                       experts, outputSize, inputSize, stride};
  auto *bytes = static_cast<uint8_t *>(result.projection.packed.contents());
  for (uint32_t expert = 0; expert < experts; ++expert) {
    result.slabs.push_back(randomSlab(random, outputSize, inputSize));
    packSlab(result.slabs.back(), bytes + uint64_t{expert} * stride);
  }
  return result;
}

struct Fixture final {
  MoeShape shape{kHidden, kExperts, kTopK, kIntermediate};
  MoeWeights weights;
  Experts gate;
  Experts up;
  Experts down;
  Experts sharedGate;
  Experts sharedUp;
  Experts sharedDown;
  MoeBuffers buffers;
  std::vector<std::vector<float>> input;
  std::vector<std::vector<float>> residual;
};

Fixture makeFixture(MetalBackend &backend) {
  Random random(0x0e5);
  Fixture fixture;
  fixture.weights.router = fixtureRouter(backend, false);
  fixture.weights.sharedExpertGate = fixtureRouter(backend, true);
  fixture.gate =
      randomExperts(backend, random, kExperts, kIntermediate, kHidden, "gate");
  fixture.up =
      randomExperts(backend, random, kExperts, kIntermediate, kHidden, "up");
  fixture.down =
      randomExperts(backend, random, kExperts, kHidden, kIntermediate, "down");
  fixture.sharedGate =
      randomExperts(backend, random, 1, kIntermediate, kHidden, "shared-gate");
  fixture.sharedUp =
      randomExperts(backend, random, 1, kIntermediate, kHidden, "shared-up");
  fixture.sharedDown =
      randomExperts(backend, random, 1, kHidden, kIntermediate, "shared-down");
  fixture.weights.expertGate = fixture.gate.projection;
  fixture.weights.expertUp = fixture.up.projection;
  fixture.weights.expertDown = fixture.down.projection;
  fixture.weights.sharedGate = fixture.sharedGate.projection;
  fixture.weights.sharedUp = fixture.sharedUp.projection;
  fixture.weights.sharedDown = fixture.sharedDown.projection;

  const uint64_t rowElements = uint64_t{kMaximumRows} * kHidden;
  MoeBuffers &b = fixture.buffers;
  b.input = shared(backend, rowElements * 2, "input");
  b.residual = shared(backend, rowElements * 2, "residual");
  b.output = shared(backend, rowElements * 2, "output");

  auto *inputValues = static_cast<__bf16 *>(b.input.contents());
  auto *residualValues = static_cast<__bf16 *>(b.residual.contents());
  fixture.input.resize(kMaximumRows, std::vector<float>(kHidden));
  fixture.residual.resize(kMaximumRows, std::vector<float>(kHidden));
  for (uint32_t row = 0; row < kMaximumRows; ++row) {
    // Exercise both signs through the affine experts and shared scalar gate.
    const float sign = (row % 3 == 0) ? -1.0F : 1.0F;
    for (uint32_t column = 0; column < kHidden; ++column) {
      const float value = bf16(sign * (0.55F + 0.45F * random.unit()));
      const float residual = bf16(0.1F * random.unit());
      fixture.input[row][column] = value;
      fixture.residual[row][column] = residual;
      inputValues[uint64_t{row} * kHidden + column] = __bf16(value);
      residualValues[uint64_t{row} * kHidden + column] = __bf16(residual);
    }
  }
  return fixture;
}

void allocateScratch(MetalBackend &backend, Fixture &fixture,
                     const MoePlan &plan) {
  const auto &w = plan.workspace();
  auto &b = fixture.buffers;
  b.selectedExperts = shared(backend, w.selectedExpertsBytes, "selected");
  b.routingWeights = shared(backend, w.routingWeightsBytes, "routing");
  b.tileDescriptors = shared(backend, w.tileDescriptorsBytes, "tiles");
  b.tileCount = shared(backend, w.tileCountBytes, "tile-count");
  b.groupedRoutes = shared(backend, w.groupedRoutesBytes, "grouped-routes");
  b.routeRows = shared(backend, w.routeRowsBytes, "route-rows");
  b.groupedInput = shared(backend, w.groupedInputBytes, "grouped-input");
  b.expertIntermediate =
      shared(backend, w.expertIntermediateBytes, "intermediate");
  b.expertOutput = shared(backend, w.expertOutputBytes, "expert-output");
}

enum class Routing { Dispersed, Concentrated, Skewed };

const char *configureRouting(Fixture &fixture, Routing distribution) {
  auto *values = static_cast<__bf16 *>(fixture.buffers.input.contents());
  for (uint32_t row = 0; row < kMaximumRows; ++row) {
    const uint32_t first =
        distribution == Routing::Concentrated ||
                (distribution == Routing::Skewed && row % 8 < 6)
            ? 0 : row % kExperts;
    const uint32_t second = (first + 1) % kExperts;
    for (uint32_t expert = 0; expert < kExperts; ++expert) {
      const float value = bf16(expert == first ? 1.0F :
                               expert == second ? 0.5F : 0.01F * expert);
      fixture.input[row][expert] = value;
      values[uint64_t{row} * kHidden + expert] = __bf16(value);
    }
  }
  switch (distribution) {
  case Routing::Dispersed: return "dispersed";
  case Routing::Concentrated: return "concentrated";
  case Routing::Skewed: return "skewed";
  }
  return "invalid";
}

void check(const Fixture &fixture, uint32_t rows, uint32_t tileRows,
           const std::string &label) {
  const auto *selected =
      static_cast<const uint32_t *>(fixture.buffers.selectedExperts.contents());
  const auto *routing =
      static_cast<const float *>(fixture.buffers.routingWeights.contents());
  const auto *actual = static_cast<const __bf16 *>(fixture.buffers.output.contents());
  const auto *tileCount =
      static_cast<const uint32_t *>(fixture.buffers.tileCount.contents());
  require(*tileCount >= kTopK + 1 &&
              *tileCount <= rows * kTopK + (rows + tileRows - 1) / tileRows,
          label + ": grouped tile count is out of range");

  std::array<uint32_t, kExperts + 1> expertCounts{};
  const auto *routeRows =
      static_cast<const uint32_t *>(fixture.buffers.routeRows.contents());
  const auto *groupedRoutes =
      static_cast<const uint32_t *>(fixture.buffers.groupedRoutes.contents());
  const auto *tiles =
      static_cast<const uint32_t *>(fixture.buffers.tileDescriptors.contents());
  for (uint32_t route = 0; route < rows * kRoutesPerRow; ++route) {
    require(selected[route] <= kExperts, label + ": invalid selected expert");
    ++expertCounts[selected[route]];
    const uint32_t grouped = routeRows[route];
    require(grouped < *tileCount * tileRows && groupedRoutes[grouped] == route &&
                tiles[(grouped / tileRows) * 2] == selected[route],
            label + ": grouped route mapping does not round-trip");
  }
  uint32_t expectedTiles = 0;
  for (const uint32_t count : expertCounts)
    expectedTiles += (count + tileRows - 1) / tileRows;
  require(*tileCount == expectedTiles,
          label + ": tile count differs from per-expert occupancy");
  for (uint32_t tile = 0; tile < *tileCount; ++tile) {
    const uint32_t liveRows = tiles[tile * 2 + 1];
    require(liveRows > 0 && liveRows <= tileRows,
            label + ": invalid tile live row count");
    for (uint32_t row = liveRows; row < tileRows; ++row)
      require(groupedRoutes[tile * tileRows + row] == UINT32_MAX,
              label + ": partial tile padding is not marked empty");
  }

  for (uint32_t row = 0; row < rows; ++row) {
    const std::vector<float> &x = fixture.input[row];
    float sum = 0.0F;
    for (float value : x)
      sum += value;
    std::array<uint32_t, kExperts> ordered;
    std::iota(ordered.begin(), ordered.end(), 0);
    std::sort(ordered.begin(), ordered.end(), [&](uint32_t left, uint32_t right) {
      const float a = 4.0F * x[left];
      const float b = 4.0F * x[right];
      return a != b ? a > b : left < right;
    });
    const uint32_t first = selected[row * kRoutesPerRow];
    const uint32_t second = selected[row * kRoutesPerRow + 1];
    require(first == ordered[0] && second == ordered[1],
            label + ": routing does not follow the router scores at row " +
                std::to_string(row));
    const float route0 = routing[row * kRoutesPerRow];
    const float route1 = routing[row * kRoutesPerRow + 1];
    const float expectedRoute0 = 1.0F / (1.0F + std::exp(
        4.0F * x[ordered[1]] - 4.0F * x[ordered[0]]));
    const float expectedRoute1 = 1.0F / (1.0F + std::exp(
        4.0F * x[ordered[0]] - 4.0F * x[ordered[1]]));
    require(std::abs(route0 + route1 - 1.0F) < 1e-5F &&
                std::abs(route0 - expectedRoute0) < 1e-5F &&
                std::abs(route1 - expectedRoute1) < 1e-5F,
            label + ": routing weights differ from CPU softmax");
    // The shared expert is the last route: its id is the expert count and
    // its weight the sigmoid of the scalar gate, here bias-only 0.003 * sum.
    const uint32_t sharedRoute = row * kRoutesPerRow + kTopK;
    const float sharedWeight = routing[sharedRoute];
    const float expectedSharedWeight =
        1.0F / (1.0F + std::exp(-sum * bf16(0.003F)));
    require(selected[sharedRoute] == kExperts &&
                std::abs(sharedWeight - expectedSharedWeight) < 0.01F,
            label + ": shared expert route is wrong at row " +
                std::to_string(row));

    std::vector<float> expected = fixture.residual[row];
    std::vector<float> magnitude(kHidden);
    for (uint32_t column = 0; column < kHidden; ++column)
      magnitude[column] = std::abs(expected[column]);
    for (uint32_t slot = 0; slot < kRoutesPerRow; ++slot) {
      const uint32_t expert = slot < kTopK ? ordered[slot] : kExperts;
      const bool sharedExpert = expert == kExperts;
      const ExpertSlab &gateSlab = sharedExpert ? fixture.sharedGate.slabs[0]
                                                : fixture.gate.slabs[expert];
      const ExpertSlab &upSlab =
          sharedExpert ? fixture.sharedUp.slabs[0] : fixture.up.slabs[expert];
      const ExpertSlab &downSlab = sharedExpert ? fixture.sharedDown.slabs[0]
                                                : fixture.down.slabs[expert];
      const std::vector<float> gate = project(gateSlab, x);
      const std::vector<float> up = project(upSlab, x);
      std::vector<float> intermediate(kIntermediate);
      std::vector<float> gpuIntermediateValues(kIntermediate);
      for (uint32_t n = 0; n < kIntermediate; ++n)
        intermediate[n] = bf16(silu(gate[n]) * up[n]);
      // The down pass is checked from the intermediate the GPU actually
      // produced: its bf16 rounding differs from the CPU's in a few elements,
      // and over a wide hidden size those propagate past the tolerance
      // through no fault of the down kernel.
      const uint32_t grouped = routeRows[row * kRoutesPerRow + slot];
      const auto *gpuIntermediate = static_cast<const __bf16 *>(
          fixture.buffers.expertIntermediate.contents()) +
          uint64_t{grouped} * kIntermediate;
      const auto *gpuDown = static_cast<const __bf16 *>(
          fixture.buffers.expertOutput.contents()) + uint64_t{grouped} * kHidden;
      for (uint32_t n = 0; n < kIntermediate; ++n) {
        require(std::isfinite(float(gpuIntermediate[n])) &&
                    std::abs(float(gpuIntermediate[n]) - intermediate[n]) <=
                        0.02F + 0.01F * std::abs(intermediate[n]),
                label + ": grouped gate/up differs from CPU reference");
        gpuIntermediateValues[n] = float(gpuIntermediate[n]);
      }
      const std::vector<float> down = project(downSlab, gpuIntermediateValues);
      for (uint32_t n = 0; n < kHidden; ++n)
        require(std::isfinite(float(gpuDown[n])) &&
                    std::abs(float(gpuDown[n]) - down[n]) <=
                        0.02F + 0.01F * std::abs(down[n]),
                label + ": grouped down differs from CPU reference");
      const float weight = slot == 0 ? expectedRoute0 :
                           slot == 1 ? expectedRoute1 : expectedSharedWeight;
      for (uint32_t column = 0; column < kHidden; ++column) {
        expected[column] += weight * down[column];
        magnitude[column] += std::abs(weight * down[column]);
      }
    }
    for (uint32_t column = 0; column < kHidden; ++column) {
      const float value = float(actual[uint64_t{row} * kHidden + column]);
      const float reference = expected[column];
      // Signed expert outputs can nearly cancel. Bound accumulated bf16
      // error against the contributing magnitudes; the individual expert
      // projections above still use their own absolute/relative bound.
      if (!std::isfinite(value) ||
          std::abs(value - reference) > 0.02F + 0.01F * magnitude[column]) {
        fail(label + ": output differs from the CPU reference at row " +
             std::to_string(row) + " column " + std::to_string(column) + ": " +
             std::to_string(value) + " vs " + std::to_string(reference) +
             " (routing " + std::to_string(route0) + "/" +
             std::to_string(expectedRoute0) + ", " +
             std::to_string(route1) + "/" + std::to_string(expectedRoute1) +
             ", shared " + std::to_string(sharedWeight) + "/" +
             std::to_string(expectedSharedWeight) +
             ", contribution magnitude " + std::to_string(magnitude[column]) + ")");
      }
    }
  }
}

template <class Function> void rejects(Function function, const char *label) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  fail(std::string(label) + ": invalid plan or buffers were accepted");
}

bool sameWorkspace(const splash::ops::MoeWorkspace &a,
                   const splash::ops::MoeWorkspace &b) {
  return a.selectedExpertsBytes == b.selectedExpertsBytes &&
         a.routingWeightsBytes == b.routingWeightsBytes &&
         a.tileDescriptorsBytes == b.tileDescriptorsBytes &&
         a.tileCountBytes == b.tileCountBytes &&
         a.groupedRoutesBytes == b.groupedRoutesBytes &&
         a.routeRowsBytes == b.routeRowsBytes &&
         a.groupedInputBytes == b.groupedInputBytes &&
         a.expertIntermediateBytes == b.expertIntermediateBytes &&
         a.expertOutputBytes == b.expertOutputBytes;
}

void checkPlan(const MoePlan &plan) {
  const auto shape = plan.shape();
  const uint64_t rows = plan.rows();
  const uint64_t tileRows = plan.tileRows();
  const uint64_t routed = rows * shape.expertsPerToken;
  const uint64_t tiles = std::min(routed / tileRows + shape.experts, routed) +
                         (rows + tileRows - 1) / tileRows;
  const uint64_t grouped = tiles * tileRows;
  const uint64_t routes = rows * (shape.expertsPerToken + 1);
  const uint64_t outputWidth =
      plan.splitExperts()
          ? std::max(shape.hiddenSize, shape.expertIntermediateSize)
          : shape.hiddenSize;
  const auto &w = plan.workspace();
  require(plan.maximumTiles() == tiles &&
              w.selectedExpertsBytes == routes * 4 &&
              w.routingWeightsBytes == routes * 4 &&
              w.tileDescriptorsBytes == tiles * 8 && w.tileCountBytes == 4 &&
              w.groupedRoutesBytes == grouped * 4 && w.routeRowsBytes == routes * 4 &&
              w.groupedInputBytes == std::max<uint64_t>(
                  grouped * shape.hiddenSize * 2, rows * 256 * 4) &&
              w.expertIntermediateBytes == grouped * shape.expertIntermediateSize * 2 &&
              w.expertOutputBytes == grouped * outputWidth * 2,
          "candidate workspace disagrees with independent geometry bound");
}

void planBounds() {
  // The wide-tile threshold scales with core count; unknown counts use the
  // measured 512-row fallback.
  require(splash::ops::moeRouteWideRows(20) == 520 &&
              splash::ops::moeRouteWideRows(40) == 1040 &&
              splash::ops::moeRouteWideRows(10) == 260 &&
              splash::ops::moeRouteWideRows(0) == 512,
          "router wide-tile threshold does not scale with the core count");
  require(splash::ops::moeRouteTile(519, 520).rows == 8 &&
              splash::ops::moeRouteTile(520, 520).rows == 32 &&
              splash::ops::moeRouteTile(512).rows == 32,
          "router tile selection ignores the configured threshold");
  for (const MoeShape shape : {MoeShape{256, 8, 2, 512},
                              MoeShape{2048, 256, 8, 512}}) {
    for (uint32_t rows = 1; rows <= 2048; ++rows) {
      const auto plans = MoE::prefillCandidates(shape, rows, kMoeRouteWideRows);
      require(plans[0].config() == MoeConfig{MoeExpertTile::M32} &&
                  plans[1].config() == MoeConfig{MoeExpertTile::M8},
              "prefill candidates must preserve the shipped baseline first");
      require(plans[0].splitExperts() && !plans[1].splitExperts(),
              "only the M32 prefill plan runs the split expert passes");
      for (const auto &plan : plans) {
        require(plan.rows() == rows, "prefill candidate changed actual rows");
        checkPlan(plan);
      }
    }
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      const auto plans = MoE::decodeCandidates(shape, lanes, kMoeRouteWideRows);
      require(plans[0].config() == MoeConfig{MoeExpertTile::M8} &&
                  plans[1].config() == MoeConfig{MoeExpertTile::M32},
              "decode candidates must preserve the shipped baseline first");
      for (const auto &plan : plans) {
        require(plan.rows() == lanes * 8, "decode candidate changed DFlash rows");
        require(!plan.splitExperts(), "decode plans keep the fused expert tile");
        checkPlan(plan);
      }
      // The Apple9 four-simdgroup tiles change only the down pass's column
      // grid: same rows, tiles and scratch as the shipped candidates.
      const auto narrow = MoE::decodeCandidates(shape, lanes, kMoeRouteWideRows,
                                                MoeExpertSimdgroups::Four);
      require(narrow[0].config() == MoeConfig{MoeExpertTile::M8, kMoeRouteWideRows,
                                              MoeExpertSimdgroups::Four} &&
                  narrow[1].config() == MoeConfig{MoeExpertTile::M32, kMoeRouteWideRows,
                                                  MoeExpertSimdgroups::Four},
              "four-simdgroup decode candidates must carry the device tile policy");
      for (size_t index = 0; index < narrow.size(); ++index) {
        require(narrow[index].rows() == plans[index].rows() &&
                    narrow[index].tileRows() == plans[index].tileRows() &&
                    narrow[index].maximumTiles() == plans[index].maximumTiles() &&
                    !narrow[index].splitExperts() &&
                    sameWorkspace(narrow[index].workspace(), plans[index].workspace()),
                "four-simdgroup tiles changed the plan geometry or workspace");
        checkPlan(narrow[index]);
      }
    }
    rejects([&] { (void)MoE::prefillCandidates(shape, 0, kMoeRouteWideRows); }, "zero prefill");
    rejects([&] { (void)MoE::prefillCandidates(shape, 2049, kMoeRouteWideRows); }, "large prefill");
    rejects([&] { (void)MoE::decodeCandidates(shape, 0, kMoeRouteWideRows); }, "zero batch");
    rejects([&] { (void)MoE::decodeCandidates(shape, 5, kMoeRouteWideRows); }, "large batch");
    rejects([&] { (void)MoE::prefillPlan(shape, 1, {static_cast<MoeExpertTile>(16)}); },
            "uncompiled expert tile");
    rejects([&] { (void)MoE::decodePlan(shape, 1, {MoeExpertTile::M8, kMoeRouteWideRows,
                                                  static_cast<MoeExpertSimdgroups>(6)}); },
            "uncompiled expert simdgroups");
  }
  rejects([] { (void)MoE::prefillPlan({}, 1); }, "invalid shape");
  // Only family 9 runs the four-simdgroup decode tiles; an unknown family
  // and families 10 and later keep the shipped tile.
  require(splash::ops::moeDecodeSimdgroups(9) == MoeExpertSimdgroups::Four &&
              splash::ops::moeDecodeSimdgroups(10) == MoeExpertSimdgroups::Eight &&
              splash::ops::moeDecodeSimdgroups(11) == MoeExpertSimdgroups::Eight &&
              splash::ops::moeDecodeSimdgroups(0) == MoeExpertSimdgroups::Eight,
          "decode expert simdgroups are not gated on GPU family 9");
}

void checkEncoding(const CommandGraph &graph, const MoePlan &plan) {
  const auto dispatches = graph.dispatches();
  const bool split = plan.splitExperts();
  const size_t expertPasses = split ? 3 : 2;
  // Two router dispatches, grouping, gather, the expert passes and combine.
  require(dispatches.size() == 5 + expertPasses,
          "MoE plan must encode the entire operator");
  const bool m8 = plan.config().expertTile == MoeExpertTile::M8;
  const auto route =
      splash::ops::moeRouteTile(plan.rows(), plan.config().routeWideRows);
  const std::string scores = route.rows == 8 ? "moe_route_scores_q8_m8"
                                             : "moe_route_scores_q8_m32";
  const size_t experts = 4;
  const size_t combine = experts + expertPasses;
  require(dispatches[0].pipelineName == scores &&
              dispatches[1].pipelineName == "moe_route_select_q8" &&
              dispatches[2].pipelineName == "moe_group_routes" &&
              dispatches[3].pipelineName == "moe_gather_rows" &&
              dispatches[combine].pipelineName == "moe_combine",
          "MoE plan chose inconsistent pipelines");
  if (split) {
    require(dispatches[experts].pipelineName ==
                    "prefill_moe_expert_q4_n256_m32" &&
                dispatches[experts + 1].pipelineName ==
                    "prefill_moe_expert_q4_n256_up_silu_m32" &&
                dispatches[experts + 2].pipelineName ==
                    "prefill_moe_expert_q4_n256_m32",
            "split MoE plan chose inconsistent expert pipelines");
    // The gate parks in expertOutput and the up pass reads it back.
    const auto &gate = dispatches[experts].buffers[5].buffer;
    require(gate.sameView(dispatches[experts + 1].buffers[5].buffer) &&
                gate.sameView(dispatches[experts + 2].buffers[5].buffer) &&
                dispatches[experts].threadgroups.x == kIntermediate / 256 &&
                dispatches[experts + 1].threadgroups.x == kIntermediate / 256 &&
                dispatches[experts + 2].threadgroups.x == kHidden / 256,
            "split MoE plan chose inconsistent gate scratch or column tiles");
  } else {
    // Four-simdgroup 8-row tiles launch 128 threads and widen the down tile
    // to N256; every other fused pass keeps N128 at 256 threads.
    const bool four = m8 && plan.config().m8Simdgroups == MoeExpertSimdgroups::Four;
    const std::string gateUp = four ? "moe_expert_gate_up_q4_m8_n128_sg4"
                               : m8 ? "moe_expert_gate_up_q4_m8"
                                    : "moe_expert_gate_up_q4_m32";
    const std::string down = four ? "moe_expert_down_q4_m8_n256_sg4"
                             : m8 ? "moe_expert_down_q4_m8"
                                  : "moe_expert_down_q4_m32";
    require(dispatches[experts].pipelineName == gateUp &&
                dispatches[experts + 1].pipelineName == down,
            "fused MoE plan chose inconsistent expert pipelines");
    const uint32_t threads = four ? 128 : 256;
    require(dispatches[experts].threadgroups.x == kIntermediate / 128 &&
                dispatches[experts + 1].threadgroups.x == kHidden / (four ? 256 : 128) &&
                dispatches[experts].threadsPerThreadgroup.x == threads &&
                dispatches[experts + 1].threadsPerThreadgroup.x == threads,
            "fused MoE plan chose inconsistent column tiles or threadgroup width");
  }
  bool tileGrids = true;
  for (size_t pass = experts; pass < combine; ++pass)
    tileGrids &= dispatches[pass].threadgroups.y == plan.maximumTiles();
  for (size_t pass = 0; pass <= combine; ++pass) {
    const bool expertPass = pass >= experts && pass < combine;
    require((expertPass || dispatches[pass].threadsPerThreadgroup.x == 256) &&
                dispatches[pass].threadsPerThreadgroup.y == 1 &&
                dispatches[pass].threadsPerThreadgroup.z == 1,
            "MoE plan changed a non-expert threadgroup width");
  }
  require(dispatches[0].threadgroups.x ==
                  (plan.rows() + route.rows - 1) / route.rows &&
              dispatches[0].threadgroups.y == kStorageN / route.experts &&
              dispatches[1].threadgroups.x == plan.rows() &&
              dispatches[3].threadgroups.x == plan.maximumTiles() && tileGrids &&
              dispatches[combine].threadgroups.x == plan.rows(),
          "MoE plan chose inconsistent dispatch bounds");
}

void bufferBounds(MetalBackend &backend, Fixture &fixture) {
  const uint64_t submissions = backend.submissionCount();
  for (const auto &plan : MoE::prefillCandidates(fixture.shape, 33, kMoeRouteWideRows)) {
    allocateScratch(backend, fixture, plan);
    const auto rejectWeights = [&](const MoeWeights &weights, const char *label) {
      CommandGraph graph;
      rejects([&] { MoE::add(graph, fixture.buffers, weights, plan); }, label);
      require(graph.empty(), "invalid weights partially encoded MoE");
    };
    for (auto projection : {&MoeWeights::router, &MoeWeights::sharedExpertGate}) {
      for (auto field : {&Q8Projection::weights, &Q8Projection::scales,
                         &Q8Projection::biases}) {
        MoeWeights shortWeights = fixture.weights;
        auto &buffer = (shortWeights.*projection).*field;
        buffer = backend.view(buffer, 0, buffer.sizeBytes() - 1);
        rejectWeights(shortWeights, "undersized Q8 weight view");
      }
    }
    for (auto member : {&MoeWeights::expertGate, &MoeWeights::expertUp,
                        &MoeWeights::expertDown, &MoeWeights::sharedGate,
                        &MoeWeights::sharedUp, &MoeWeights::sharedDown}) {
      const auto &source = fixture.weights.*member;
      const uint64_t payload = q4PackedBytes(source.outputSize, source.inputSize);
      MoeWeights changed = fixture.weights;
      auto &projection = changed.*member;
      projection.packed = backend.view(source.packed, 0, payload - 1);
      rejectWeights(changed, "undersized expert payload");
      projection.packed = source.packed;
      for (uint64_t stride : {uint64_t{0}, payload - 2, payload + 1}) {
        projection.expertStrideBytes = stride;
        rejectWeights(changed, "invalid expert stride");
      }
      if (source.experts > 1) {
        projection.expertStrideBytes = std::numeric_limits<uint64_t>::max() - 1;
        rejectWeights(changed, "overflowing final expert boundary");
      }
      // A padded stride is legal. Only the last expert's payload must fit;
      // padding after that payload is never addressed by the shader.
      projection.expertStrideBytes = payload + 2;
      const uint64_t bytes = uint64_t{source.experts - 1} *
                                 projection.expertStrideBytes + payload;
      projection.packed = shared(backend, bytes, "moe-padded-expert-bound");
      CommandGraph graph;
      MoE::add(graph, fixture.buffers, changed, plan);
      checkEncoding(graph, plan);
      projection.packed = backend.view(projection.packed, 0, bytes - 1);
      rejectWeights(changed, "undersized final expert boundary");
    }
    for (auto member : {&MoeBuffers::selectedExperts, &MoeBuffers::routingWeights,
                        &MoeBuffers::tileDescriptors, &MoeBuffers::tileCount,
                        &MoeBuffers::groupedRoutes, &MoeBuffers::routeRows,
                        &MoeBuffers::groupedInput, &MoeBuffers::expertIntermediate,
                        &MoeBuffers::expertOutput}) {
      MoeBuffers shortBuffers = fixture.buffers;
      const auto &buffer = fixture.buffers.*member;
      shortBuffers.*member = backend.view(buffer, 0, buffer.sizeBytes() - 1);
      CommandGraph graph;
      rejects([&] { MoE::add(graph, shortBuffers, fixture.weights, plan); },
              "undersized scratch");
      require(graph.empty(), "invalid scratch partially encoded MoE");
    }
    for (auto member : {&MoeBuffers::input, &MoeBuffers::residual, &MoeBuffers::output}) {
      MoeBuffers shortBuffers = fixture.buffers;
      shortBuffers.*member = backend.view(fixture.buffers.*member, 0,
          uint64_t{plan.rows()} * fixture.shape.hiddenSize * 2 - 1);
      CommandGraph graph;
      rejects([&] { MoE::add(graph, shortBuffers, fixture.weights, plan); },
              "undersized row buffer");
      require(graph.empty(), "invalid row buffer partially encoded MoE");
    }
  }
  const auto smallTiles = MoE::prefillPlan(fixture.shape, 33, {MoeExpertTile::M8});
  const auto largeTiles = MoE::prefillPlan(fixture.shape, 33, {MoeExpertTile::M32});
  allocateScratch(backend, fixture, smallTiles);
  CommandGraph graph;
  rejects([&] { MoE::add(graph, fixture.buffers, fixture.weights, largeTiles); },
          "scratch from incompatible plan");
  require(graph.empty(), "incompatible plan partially encoded MoE");
  require(backend.submissionCount() == submissions,
          "MoE buffer validation submitted a GPU command");
}

// A dense random Q8 router: the routing fixture's one-weight experts cannot
// expose accumulation-order differences between the scores tiles.
Q8Projection randomRouter(MetalBackend &backend, Random &random) {
  const uint64_t elements = uint64_t{kStorageN} * kHidden;
  Q8Projection projection{shared(backend, elements, "dense-router-weights"),
                          shared(backend, elements / 32, "dense-router-scales"),
                          shared(backend, elements / 32, "dense-router-biases"),
                          kStorageN, kHidden};
  auto *weights = static_cast<uint8_t *>(projection.weights.contents());
  auto *scales = static_cast<__bf16 *>(projection.scales.contents());
  auto *biases = static_cast<__bf16 *>(projection.biases.contents());
  for (uint64_t index = 0; index < elements; ++index)
    weights[index] = static_cast<uint8_t>(random.next());
  for (uint64_t index = 0; index < elements / 64; ++index) {
    scales[index] = __bf16(0.002F + 0.001F * random.unit());
    biases[index] = __bf16(-0.25F + 0.05F * random.unit());
  }
  return projection;
}

// Both scores tiles must write bitwise-identical scores for the same rows, so
// routing never depends on the tile the row count selects, and the fp32
// scores must match an fp64 reference of the affine Q8 projection to fp32
// accumulation: K u (u = 2^-24) times the magnitudes its terms sum.
void routerTiles(MetalBackend &backend, const Fixture &fixture) {
  Random random(0x7a11);
  const Q8Projection router = randomRouter(backend, random);
  const auto *weights = static_cast<const uint8_t *>(router.weights.contents());
  const auto *scales = static_cast<const __bf16 *>(router.scales.contents());
  const auto *biases = static_cast<const __bf16 *>(router.biases.contents());
  const uint64_t scoreBytes = uint64_t{kMaximumRows} * kStorageN * 4;
  MetalBuffer narrow = shared(backend, scoreBytes, "scores-m8");
  MetalBuffer wide = shared(backend, scoreBytes, "scores-m32");
  for (const uint32_t rows : {8U, 33U, kMaximumRows}) {
    const MoeRouteParams params{rows, kHidden, kExperts, kTopK};
    CommandGraph graph;
    graph.add("moe_route_scores_q8_m8",
              {fixture.buffers.input, router.weights, router.scales,
               router.biases, narrow},
              params, {(rows + 7) / 8, kStorageN / 32, 1});
    graph.add("moe_route_scores_q8_m32",
              {fixture.buffers.input, router.weights, router.scales,
               router.biases, wide},
              params, {(rows + 31) / 32, kStorageN / 128, 1});
    (void)backend.submitCommand(graph.dispatches());
    const std::string label = "router tiles rows=" + std::to_string(rows);
    require(std::memcmp(narrow.contents(), wide.contents(),
                        uint64_t{rows} * kStorageN * 4) == 0,
            label + ": 8-row and 32-row tiles disagree");
    const auto *values = static_cast<const float *>(narrow.contents());
    for (uint32_t row = 0; row < rows; ++row) {
      for (uint32_t expert = 0; expert < kStorageN; ++expert) {
        double reference = 0;
        double magnitude = 0;
        for (uint32_t g = 0; g < kHidden / 64; ++g) {
          const double scale = float(scales[g * kStorageN + expert]);
          const double bias = float(biases[g * kStorageN + expert]);
          for (uint32_t k = g * 64; k < g * 64 + 64; ++k) {
            const double x = fixture.input[row][k];
            const double q =
                weights[(uint64_t{g} * kStorageN + expert) * 64 + k % 64];
            reference += x * (q * scale + bias);
            // The kernel sums the two affine terms apart.
            magnitude += std::abs(x) * (q * std::abs(scale) + std::abs(bias));
          }
        }
        const float value = values[uint64_t{row} * kStorageN + expert];
        require(std::isfinite(value) &&
                    std::abs(value - reference) <=
                        std::ldexp(magnitude * kHidden, -24),
                label + ": scores differ from the fp64 reference");
      }
    }
  }
}

void run(const std::string &metallibPath) {
  planBounds();
  MetalBackend backend(metallibPath);
  Fixture fixture = makeFixture(backend);
  bufferBounds(backend, fixture);
  routerTiles(backend, fixture);
  uint32_t cases = 0;
  double wallSeconds = 0;
  for (const auto distribution : {Routing::Dispersed, Routing::Concentrated,
                                  Routing::Skewed}) {
    const std::string routingLabel = configureRouting(fixture, distribution);
    // Rows every plan shares must route identically whatever the plan's row
    // count: another lane count or chunk size changes a row's tile position
    // and neighbours, never its routes (the B1..B4 invariant).
    std::vector<uint32_t> sharedSelected;
    std::vector<float> sharedRouting;
    auto execute = [&](const MoePlan &plan, const std::string &label) {
      allocateScratch(backend, fixture, plan);
      CommandGraph graph;
      MoE::add(graph, fixture.buffers, fixture.weights, plan);
      checkEncoding(graph, plan);
      std::memset(fixture.buffers.output.contents(), 0,
                  fixture.buffers.output.sizeBytes());
      wallSeconds += backend.submitCommand(graph.dispatches()).wallSeconds;
      check(fixture, plan.rows(), plan.tileRows(), label + " " + routingLabel +
            " M" + std::to_string(plan.tileRows()));
      ++cases;
      const auto *selected = static_cast<const uint32_t *>(
          fixture.buffers.selectedExperts.contents());
      const auto *routing = static_cast<const float *>(
          fixture.buffers.routingWeights.contents());
      const size_t routes = size_t{plan.rows()} * kRoutesPerRow;
      const size_t common = std::min(routes, sharedSelected.size());
      require(std::equal(selected, selected + common, sharedSelected.begin()) &&
                  std::equal(routing, routing + common, sharedRouting.begin()),
              label + " " + routingLabel + ": routes depend on the row count");
      if (routes > sharedSelected.size()) {
        sharedSelected.assign(selected, selected + routes);
        sharedRouting.assign(routing, routing + routes);
      }
      const auto *output = static_cast<const __bf16 *>(fixture.buffers.output.contents());
      std::vector<float> result(uint64_t{plan.rows()} * kHidden);
      for (uint32_t index = 0; index < result.size(); ++index)
        result[index] = float(output[index]);
      return result;
    };
    const auto requireEqual = [&](const std::vector<float> &candidate,
                                   const std::vector<float> &baseline,
                                   const std::string &label) {
      require(candidate.size() == baseline.size(), label + ": output size differs");
      const auto mismatch = std::mismatch(candidate.begin(), candidate.end(), baseline.begin());
      if (mismatch.first != candidate.end()) {
        const size_t index = static_cast<size_t>(mismatch.first - candidate.begin());
        std::cerr << label << ' ' << routingLabel
                  << " first_difference row=" << index / kHidden
                  << " column=" << index % kHidden
                  << " baseline=" << *mismatch.second
                  << " candidate=" << *mismatch.first
                  << " baseline_fp32_bits=" << std::bit_cast<uint32_t>(*mismatch.second)
                  << " candidate_fp32_bits=" << std::bit_cast<uint32_t>(*mismatch.first)
                  << '\n';
        fail(label + " " + routingLabel + ": outputs differ from the shipped tile");
      }
    };
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      const auto candidates = MoE::decodeCandidates(fixture.shape, lanes, kMoeRouteWideRows);
      const std::string label = "decode B" + std::to_string(lanes);
      const auto baseline = execute(candidates[0], label);
      requireEqual(execute(candidates[1], label), baseline, label + " M32");
      // The Apple9 four-simdgroup tiles (gate/up N128, down N256, 128
      // threads) must reproduce the shipped N128 x 8 tile bit for bit: each
      // output element sums its quant groups in the same order, so no
      // tolerance is granted. The M32 candidate carries the device policy
      // but keeps its eight-simdgroup tiles.
      const auto narrow = MoE::decodeCandidates(fixture.shape, lanes, kMoeRouteWideRows,
                                                MoeExpertSimdgroups::Four);
      requireEqual(execute(narrow[0], label + " sg4"), baseline,
                   label + " four-simdgroup M8");
      requireEqual(execute(narrow[1], label + " sg4"), baseline,
                   label + " four-simdgroup M32");
    }
    // 12 and 48 rows leave 16-row ragged tiles in every routing fixture; 9,
    // 33 and 263 leave 8-row ones next to full tiles; 511/512 straddle the
    // wide router tile threshold.
    for (uint32_t rows : {1U, 3U, 7U, 8U, 9U, 12U, 31U, 32U, 33U, 48U, 100U,
                          255U, 256U, 263U, 511U, 512U, kMaximumRows}) {
      const auto candidates = MoE::prefillCandidates(fixture.shape, rows, kMoeRouteWideRows);
      const std::string label = "prefill rows=" + std::to_string(rows);
      const auto baseline = execute(candidates[0], label);
      requireEqual(execute(candidates[1], label), baseline, label);
    }
  }
  std::cout << "moe_metal_test: PASS cases=" << cases
            << " wall_seconds=" << wallSeconds << '\n';
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: moe_metal_test <metallib>\n";
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

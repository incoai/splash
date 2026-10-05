// Sparse MoE against a CPU reference with real Q4 expert weights: routing,
// expert grouping (including partially filled tiles), the grouped gate/up and
// down tiles, and the combine, for the M8 decode and M32 prefill plans and the
// Apple9 four-simdgroup decode tiles (bitwise against the shipped tile). Routing
// fixtures cover dispersed, concentrated and skewed expert utilization with
// hidden width 1024 and intermediate width 512.
#include "AffineQ4Fixture.hpp"
#include "MoeExtents.hpp"
#include "TestChecks.hpp"
#include "../../../runtime/metal/BackendInstrumentation.hpp"
#include "../../../runtime/metal/CommandGraph.hpp"
#include "../../../runtime/metal/MetalBackend.hpp"
#include "../../../runtime/model/WeightStore.hpp"
#include "../../../runtime/ops/ExecutionPlans.hpp"
#include "../../../runtime/ops/MoE.hpp"
#include "metal/abi/MoE.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
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

using splash::metal::BackendInstrumentation;
using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::model::kBFloat16Bytes;
using splash::model::q4PackedBytes;
using splash::ops::AffineMoeWeights;
using splash::ops::ExecutionPlans;
using splash::ops::ExpertProjection;
using splash::ops::kAssumedGpuCores;
using splash::ops::kMoeScratchFields;
using splash::ops::MoE;
using splash::ops::MoeBuffers;
using splash::ops::MoeConfig;
using splash::ops::MoeScratchField;
using splash::ops::MoeExpertSimdgroups;
using splash::ops::MoeExpertTile;
using splash::ops::MoePhase;
using splash::ops::moeRouteWideRows;
using splash::ops::MoePlan;
using splash::ops::MoeShape;
using splash::ops::MoeWeights;
using splash::ops::Q8Projection;
using splash::test::rejects;

// 16 quant groups on the hidden side, so the 32-row expert tiles refill their
// staged scales and biases several times, as the production shapes do.
constexpr uint32_t kHidden = 1024;
constexpr uint32_t kIntermediate = 512;
constexpr uint32_t kExperts = 8;
constexpr uint32_t kTopK = 2;
constexpr uint32_t kRoutesPerRow = kTopK + 1;
// The planned GPU: 20 cores, whose router takes the 32-row scores tile from
// moeRouteWideRows(20) = 520 rows.
constexpr uint32_t kGpuCores = 20;
// Covers both router tiles: 8-row tiles below 520 rows and 32-row tiles with
// a ragged 8-row tail from there.
constexpr uint32_t kMaximumRows = 520;
constexpr uint32_t kStorageColumns = 256;

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

// The production plans of a kGpuCores-core GPU of `family`: family 9 decodes
// with the four-simdgroup 8-row tiles, other families with the shipped eight.
ExecutionPlans plans(uint32_t family) {
  splash::DeviceCapabilities device;
  device.appleGpuFamily = family;
  device.gpuCoreCount = kGpuCores;
  return ExecutionPlans(device);
}

MetalBuffer shared(MetalBackend &backend, uint64_t bytes, const char *label) {
  MetalBuffer buffer = backend.allocateBuffer(bytes, BufferStorage::Shared, label);
  std::memset(buffer.contents(), 0, buffer.sizeBytes());
  return buffer;
}

float bf16(float value) { return float(__bf16(value)); }

// One expert's Q4 slab in the packed 256-column storage order:
// [weights][scales][biases], the parameter of output n and quant group g at
// (tile(n) * groups + g) * 256 + n % 256 and its 64 nibbles right after the
// previous column's.
struct ExpertSlab final {
  std::vector<uint8_t> nibbles; // [n][k] dequantized as q * scale + bias
  std::vector<float> scales;    // [parameter]
  std::vector<float> biases;    // [parameter]
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;

  [[nodiscard]] uint32_t parameter(uint32_t n, uint32_t g) const {
    const uint32_t groups = inputSize / 64;
    return ((n / kStorageColumns) * groups + g) * kStorageColumns + n % kStorageColumns;
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

void packSlab(const ExpertSlab &slab, const splash::test::AffineQ4Planes &planes) {
  const uint32_t groups = slab.inputSize / 64;
  for (uint32_t n = 0; n < slab.outputSize; ++n) {
    for (uint32_t k = 0; k < slab.inputSize; ++k) {
      const uint64_t nibble = uint64_t{slab.parameter(n, k / 64)} * 64 + k % 64;
      uint8_t &byte = planes.weights[nibble / 2];
      const uint8_t value = slab.weight(n, k);
      byte = static_cast<uint8_t>((nibble & 1) ? (byte & 0x0F) | (value << 4)
                                               : (byte & 0xF0) | value);
    }
    for (uint32_t g = 0; g < groups; ++g) {
      planes.scales[slab.parameter(n, g)] = std::bit_cast<uint16_t>(__bf16(slab.scales[slab.parameter(n, g)]));
      planes.biases[slab.parameter(n, g)] = std::bit_cast<uint16_t>(__bf16(slab.biases[slab.parameter(n, g)]));
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
  const uint64_t elements = uint64_t{kStorageColumns} * kHidden;
  Q8Projection projection{{shared(backend, elements, "moe-router-weights"),
                           shared(backend, elements / 32, "moe-router-scales"),
                           shared(backend, elements / 32, "moe-router-biases")},
                          kStorageColumns, kHidden};
  if (!sharedGate) {
    auto *weights = static_cast<uint8_t *>(projection.planes.weights.contents());
    auto *scales = static_cast<__bf16 *>(projection.planes.scales.contents());
    for (uint32_t expert = 0; expert < kExperts; ++expert) {
      weights[expert * 64 + expert] = 1;
      scales[expert] = __bf16(4.0F);
    }
    return projection;
  }
  auto *values = static_cast<__bf16 *>(projection.planes.biases.contents());
  for (uint32_t group = 0; group < kHidden / 64; ++group) {
    for (uint32_t output = 0; output < kStorageColumns; ++output) {
      const float value = output ? 0.0F : 0.003F;
      values[group * kStorageColumns + output] = __bf16(value);
    }
  }
  return projection;
}

struct Experts final {
  std::vector<ExpertSlab> slabs;
  ExpertProjection projection;
};

Experts randomExperts(MetalBackend &backend, Random &random,
                      uint32_t experts, uint32_t outputSize,
                      uint32_t inputSize, const char *label) {
  Experts result;
  result.projection = splash::test::expertSlabs(
      backend, experts, outputSize, inputSize, label, [&](uint32_t, const splash::test::AffineQ4Planes &planes) {
        result.slabs.push_back(randomSlab(random, outputSize, inputSize));
        packSlab(result.slabs.back(), planes);
      });
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
  AffineMoeWeights weights;
  weights.router = fixtureRouter(backend, false);
  weights.sharedScalarGate = fixtureRouter(backend, true);
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
  weights.expertGate = fixture.gate.projection;
  weights.expertUp = fixture.up.projection;
  weights.expertDown = fixture.down.projection;
  weights.sharedGate = fixture.sharedGate.projection;
  weights.sharedUp = fixture.sharedUp.projection;
  weights.sharedDown = fixture.sharedDown.projection;
  fixture.weights = weights;

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
  for (const MoeScratchField &field : kMoeScratchFields)
    if (const uint64_t bytes = w.*field.bytes)
      fixture.buffers.scratch.*field.buffer = shared(backend, bytes, "moe-scratch");
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

// The rows a tile's matmul reads (moe_matmul_rows,
// kernels/common/moe_expert_slab.h): the smallest of 8, 16 and the tile's
// rows that holds its live rows.
uint32_t matmulRows(uint32_t liveRows, uint32_t tileRows) {
  return std::min(tileRows, liveRows <= 8 ? 8u : liveRows <= 16 ? 16u : tileRows);
}

// The byte every grouped route and grouped input holds before a run.
constexpr uint8_t kCanary = 0xa5;

bool untouched(const void *data, uint64_t bytes) {
  const auto *begin = static_cast<const uint8_t *>(data);
  return std::all_of(begin, begin + bytes, [](uint8_t byte) { return byte == kCanary; });
}

void check(const Fixture &fixture, uint32_t rows, uint32_t tileRows,
           const std::string &label) {
  const auto *selected =
      static_cast<const uint32_t *>(fixture.buffers.scratch.selectedExperts.contents());
  const auto *routing =
      static_cast<const float *>(fixture.buffers.scratch.routingWeights.contents());
  const auto *actual = static_cast<const __bf16 *>(fixture.buffers.output.contents());
  const auto *tileCount =
      static_cast<const uint32_t *>(fixture.buffers.scratch.tileCount.contents());
  require(*tileCount >= kTopK + 1 &&
              *tileCount <= rows * kTopK + (rows + tileRows - 1) / tileRows,
          label + ": grouped tile count is out of range");

  std::array<uint32_t, kExperts + 1> expertCounts{};
  const auto *routeRows =
      static_cast<const uint32_t *>(fixture.buffers.scratch.routeRows.contents());
  const auto *groupedRoutes =
      static_cast<const uint32_t *>(fixture.buffers.scratch.groupedRoutes.contents());
  const auto *tiles =
      static_cast<const uint32_t *>(fixture.buffers.scratch.tileDescriptors.contents());
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
  const auto *groupedInput = static_cast<const uint8_t *>(
      fixture.buffers.scratch.groupedInput.contents());
  const uint64_t inputRowBytes = uint64_t{kHidden} * kBFloat16Bytes;
  const uint64_t scoresBytes = uint64_t{rows} * kStorageColumns * sizeof(float);
  for (uint32_t tile = 0; tile < *tileCount; ++tile) {
    const uint32_t liveRows = tiles[tile * 2 + 1];
    require(liveRows > 0 && liveRows <= tileRows,
            label + ": invalid tile live row count");
    const uint32_t matmul = matmulRows(liveRows, tileRows);
    for (uint32_t row = liveRows; row < matmul; ++row)
      require(groupedRoutes[tile * tileRows + row] == UINT32_MAX,
              label + ": padding a tile's matmul reads is not marked empty");
    // Neither the grouping nor the gather writes the rows past the matmul's,
    // so they keep the canary, except where the router's scores, which come
    // first in the grouped input, overwrote it.
    for (uint32_t row = matmul; row < tileRows; ++row) {
      const uint64_t grouped = uint64_t{tile} * tileRows + row;
      require(untouched(groupedRoutes + grouped, sizeof(uint32_t)) &&
                  (grouped * inputRowBytes < scoresBytes ||
                   untouched(groupedInput + grouped * inputRowBytes, inputRowBytes)),
              label + ": grouping or gather wrote past a tile's matmul rows");
    }
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
          fixture.buffers.scratch.expertIntermediate.contents()) +
          uint64_t{grouped} * kIntermediate;
      const auto *gpuDown = static_cast<const __bf16 *>(
          fixture.buffers.scratch.expertOutput.contents()) + uint64_t{grouped} * kHidden;
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

// Expert ids, route rows and grouped routes are uint32, routing weights fp32
// and activations bf16. The grouped input first holds the router's fp32
// scores, one kStorageColumns row per token.
void checkPlan(const MoePlan &plan) {
  const auto shape = plan.shape();
  const uint64_t rows = plan.rows();
  const uint64_t tiles = splash::ops::moeMaximumTiles(plan.rows(), shape, plan.tileRows());
  const uint64_t grouped = tiles * plan.tileRows();
  const uint64_t routes = rows * shape.routesPerToken();
  const uint64_t outputWidth =
      plan.splitExperts()
          ? std::max(shape.hiddenSize, shape.expertIntermediateSize)
          : shape.hiddenSize;
  const auto &w = plan.workspace();
  require(plan.maximumTiles() == tiles &&
              w.selectedExpertsBytes == routes * sizeof(uint32_t) &&
              w.routingWeightsBytes == routes * sizeof(float) &&
              w.tileDescriptorsBytes == tiles * sizeof(MoeTileDescriptor) &&
              w.tileCountBytes == sizeof(uint32_t) &&
              w.groupedRoutesBytes == grouped * sizeof(uint32_t) &&
              w.routeRowsBytes == routes * sizeof(uint32_t) &&
              w.groupedInputBytes == std::max<uint64_t>(grouped * shape.hiddenSize * kBFloat16Bytes,
                                                        rows * kStorageColumns * sizeof(float)) &&
              w.expertIntermediateBytes == grouped * shape.expertIntermediateSize * kBFloat16Bytes &&
              w.expertOutputBytes == grouped * outputWidth * kBFloat16Bytes && w.groupedSumsBytes == 0,
          "plan workspace disagrees with independent geometry bound");
}

void planBounds() {
  // The wide-tile threshold scales with the planned core count.
  require(moeRouteWideRows(20) == 520 && moeRouteWideRows(40) == 1040 && moeRouteWideRows(10) == 260,
          "router wide-tile threshold does not scale with the core count");
  require(splash::ops::moeRouteTile(519, 520).rows == 8 &&
              splash::ops::moeRouteTile(520, 520).rows == 32 &&
              splash::ops::moeRouteTile(832, moeRouteWideRows(kAssumedGpuCores)).rows == 32,
          "router tile selection ignores the configured threshold");
  for (const MoeShape shape : {MoeShape{256, 8, 2, 512},
                              MoeShape{2048, 256, 8, 512}}) {
    const ExecutionPlans shipped = plans(10);
    for (uint32_t rows = 1; rows <= 2048; ++rows) {
      const MoePlan plan = shipped.moePrefill(shape, rows);
      require(plan.configuration() == MoeConfig{MoeExpertTile::M32, moeRouteWideRows(kGpuCores)} &&
                  plan.splitExperts() && plan.rows() == rows,
              "prefill plans run the split 32-row expert passes over their actual rows");
      checkPlan(plan);
    }
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      const MoePlan plan = shipped.moeDecode(shape, lanes);
      require(plan.configuration() == MoeConfig{MoeExpertTile::M8, moeRouteWideRows(kGpuCores)} &&
                  !plan.splitExperts() && plan.rows() == lanes * 8,
              "decode plans run the fused 8-row expert tile over the DFlash rows");
      checkPlan(plan);
      // The Apple9 four-simdgroup tiles change only the down pass's column
      // grid: same rows, tiles and scratch as the shipped plan.
      const MoePlan narrow = plans(9).moeDecode(shape, lanes);
      require(narrow.configuration() == MoeConfig{MoeExpertTile::M8, moeRouteWideRows(kGpuCores),
                                                  MoeExpertSimdgroups::Four} &&
                  narrow.rows() == plan.rows() && narrow.tileRows() == plan.tileRows() &&
                  narrow.maximumTiles() == plan.maximumTiles() && !narrow.splitExperts() &&
                  narrow.workspace() == plan.workspace(),
              "four-simdgroup tiles changed the plan geometry or workspace");
    }
    rejects([&] { (void)shipped.moePrefill(shape, 0); }, "invalid MoE prefill rows",
            "a prefill of no rows was accepted");
    rejects([&] { (void)shipped.moePrefill(shape, 2049); }, "invalid MoE prefill rows",
            "a prefill past the token budget was accepted");
    rejects([&] { (void)shipped.moeDecode(shape, 0); }, "invalid MoE decode width",
            "a decode of no lanes was accepted");
    rejects([&] { (void)shipped.moeDecode(shape, 5); }, "invalid MoE decode width",
            "a decode wider than a batch was accepted");
    rejects([&] { (void)MoE::prefillPlan(shape, 1, {MoeExpertTile::M8}); },
            "invalid MoE expert tile configuration", "an affine prefill plan took 8-row tiles");
    rejects([&] { (void)MoE::decodePlan(shape, 1, {MoeExpertTile::M32}); },
            "invalid MoE expert tile configuration", "an affine decode plan took 32-row tiles");
  }
  rejects([] { (void)MoE::prefillPlan({}, 1, {MoeExpertTile::M32}); }, "invalid MoE workspace shape",
          "a plan of an empty shape was accepted");
  // Only family 9 runs the four-simdgroup decode tiles; families 10 and
  // later run eight.
  using splash::ops::gpuFamilyClass;
  require(splash::ops::moeDecodeSimdgroups(gpuFamilyClass(9)) == MoeExpertSimdgroups::Four &&
              splash::ops::moeDecodeSimdgroups(gpuFamilyClass(10)) == MoeExpertSimdgroups::Eight &&
              splash::ops::moeDecodeSimdgroups(gpuFamilyClass(11)) == MoeExpertSimdgroups::Eight,
          "decode expert simdgroups are not gated on GPU family 9");
}

void checkEncoding(const CommandGraph &graph, const MoePlan &plan) {
  const auto dispatches = graph.dispatches();
  const bool split = plan.splitExperts();
  const size_t expertPasses = split ? 3 : 2;
  // Two router dispatches, grouping, gather, the expert passes and combine.
  require(dispatches.size() == 5 + expertPasses,
          "MoE plan must encode the entire operator");
  const auto route =
      splash::ops::moeRouteTile(plan.rows(), plan.configuration().routeWideRows);
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
    const bool four = plan.configuration().m8Simdgroups == MoeExpertSimdgroups::Four;
    const std::string gateUp = four ? "moe_expert_gate_up_q4_m8_n128_sg4"
                                    : "moe_expert_gate_up_q4_m8";
    const std::string down = four ? "moe_expert_down_q4_m8_n256_sg4"
                                  : "moe_expert_down_q4_m8";
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
              dispatches[0].threadgroups.y == kStorageColumns / route.experts &&
              dispatches[1].threadgroups.x == plan.rows() &&
              dispatches[3].threadgroups.x == plan.maximumTiles() && tileGrids &&
              dispatches[combine].threadgroups.x == plan.rows(),
          "MoE plan chose inconsistent dispatch bounds");
}

void bufferBounds(MetalBackend &backend, Fixture &fixture) {
  const uint64_t submissions = BackendInstrumentation::submittedCommands(backend);
  const ExecutionPlans shipped = plans(10);
  // The split prefill passes and the fused decode tile.
  for (const MoePlan &plan : {shipped.moePrefill(fixture.shape, 33),
                              shipped.moeDecode(fixture.shape, 4)}) {
    allocateScratch(backend, fixture, plan);
    const auto rejectWeights = [&](const MoeWeights &weights, const char *label) {
      CommandGraph graph;
      rejects([&] { MoE::add(graph, fixture.buffers, weights, plan); },
              "MoE weights do not match execution shape", std::string(label) + " was accepted");
      require(graph.empty(), "invalid weights partially encoded MoE");
    };
    for (auto projection : {&splash::ops::AffineMoeWeights::router, &splash::ops::AffineMoeWeights::sharedScalarGate}) {
      for (auto field : {&splash::ops::AffineWeights::weights, &splash::ops::AffineWeights::scales,
                         &splash::ops::AffineWeights::biases}) {
        AffineMoeWeights shortWeights = fixture.weights.affine();
        auto &buffer = (shortWeights.*projection).planes.*field;
        buffer = backend.view(buffer, 0, buffer.sizeBytes() - 1);
        rejectWeights(shortWeights, "undersized Q8 weight view");
      }
    }
    for (auto member : {&splash::ops::AffineMoeWeights::expertGate, &splash::ops::AffineMoeWeights::expertUp,
                        &splash::ops::AffineMoeWeights::expertDown, &splash::ops::AffineMoeWeights::sharedGate,
                        &splash::ops::AffineMoeWeights::sharedUp, &splash::ops::AffineMoeWeights::sharedDown}) {
      const auto &source = fixture.weights.affine().*member;
      const uint64_t payload = q4PackedBytes(source.outputSize, source.inputSize);
      AffineMoeWeights changed = fixture.weights.affine();
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
    splash::test::requireMoeExtents(backend, fixture.buffers, fixture.weights, plan);
  }
  const auto smallTiles = shipped.moeDecode(fixture.shape, 4);
  const auto largeTiles = shipped.moePrefill(fixture.shape, 33);
  allocateScratch(backend, fixture, smallTiles);
  CommandGraph graph;
  // The decode plan's 32 rows select fewer experts than the prefill plan's 33.
  rejects([&] { MoE::add(graph, fixture.buffers, fixture.weights, largeTiles); },
          "MoE selected experts buffer holds", "scratch from an incompatible plan was accepted");
  require(graph.empty(), "incompatible plan partially encoded MoE");
  require(BackendInstrumentation::submittedCommands(backend) == submissions,
          "MoE buffer validation submitted a GPU command");
}

// A dense random Q8 router: the routing fixture's one-weight experts cannot
// expose accumulation-order differences between the scores tiles.
Q8Projection randomRouter(MetalBackend &backend, Random &random) {
  const uint64_t elements = uint64_t{kStorageColumns} * kHidden;
  Q8Projection projection{{shared(backend, elements, "dense-router-weights"),
                           shared(backend, elements / 32, "dense-router-scales"),
                           shared(backend, elements / 32, "dense-router-biases")},
                          kStorageColumns, kHidden};
  auto *weights = static_cast<uint8_t *>(projection.planes.weights.contents());
  auto *scales = static_cast<__bf16 *>(projection.planes.scales.contents());
  auto *biases = static_cast<__bf16 *>(projection.planes.biases.contents());
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
  const auto *weights = static_cast<const uint8_t *>(router.planes.weights.contents());
  const auto *scales = static_cast<const __bf16 *>(router.planes.scales.contents());
  const auto *biases = static_cast<const __bf16 *>(router.planes.biases.contents());
  const uint64_t scoreBytes = uint64_t{kMaximumRows} * kStorageColumns * 4;
  MetalBuffer narrow = shared(backend, scoreBytes, "scores-m8");
  MetalBuffer wide = shared(backend, scoreBytes, "scores-m32");
  for (const uint32_t rows : {8U, 33U, kMaximumRows}) {
    const MoeRouteParams params{rows, kHidden, kExperts, kTopK};
    CommandGraph graph;
    graph.add("moe_route_scores_q8_m8",
              {fixture.buffers.input, router.planes.weights, router.planes.scales,
               router.planes.biases, narrow},
              params, {(rows + 7) / 8, kStorageColumns / 32, 1});
    graph.add("moe_route_scores_q8_m32",
              {fixture.buffers.input, router.planes.weights, router.planes.scales,
               router.planes.biases, wide},
              params, {(rows + 31) / 32, kStorageColumns / 128, 1});
    (void)backend.submitCommandAsync(graph.dispatches()).wait();
    const std::string label = "router tiles rows=" + std::to_string(rows);
    require(std::memcmp(narrow.contents(), wide.contents(),
                        uint64_t{rows} * kStorageColumns * 4) == 0,
            label + ": 8-row and 32-row tiles disagree");
    const auto *values = static_cast<const float *>(narrow.contents());
    for (uint32_t row = 0; row < rows; ++row) {
      for (uint32_t expert = 0; expert < kStorageColumns; ++expert) {
        double reference = 0;
        double magnitude = 0;
        for (uint32_t g = 0; g < kHidden / 64; ++g) {
          const double scale = float(scales[g * kStorageColumns + expert]);
          const double bias = float(biases[g * kStorageColumns + expert]);
          for (uint32_t k = g * 64; k < g * 64 + 64; ++k) {
            const double x = fixture.input[row][k];
            const double q =
                weights[(uint64_t{g} * kStorageColumns + expert) * 64 + k % 64];
            reference += x * (q * scale + bias);
            // The kernel sums the two affine terms apart.
            magnitude += std::abs(x) * (q * std::abs(scale) + std::abs(bias));
          }
        }
        const float value = values[uint64_t{row} * kStorageColumns + expert];
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
      for (const MetalBuffer *buffer : {&fixture.buffers.scratch.groupedRoutes,
                                        &fixture.buffers.scratch.groupedInput})
        std::memset(buffer->contents(), kCanary, buffer->sizeBytes());
      wallSeconds += backend.submitCommandAsync(graph.dispatches()).wait().wallSeconds;
      check(fixture, plan.rows(), plan.tileRows(), label + " " + routingLabel +
            " M" + std::to_string(plan.tileRows()));
      ++cases;
      const auto *selected = static_cast<const uint32_t *>(
          fixture.buffers.scratch.selectedExperts.contents());
      const auto *routing = static_cast<const float *>(
          fixture.buffers.scratch.routingWeights.contents());
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
    const ExecutionPlans shipped = plans(10), narrow = plans(9);
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      const std::string label = "decode B" + std::to_string(lanes);
      const auto baseline = execute(shipped.moeDecode(fixture.shape, lanes), label);
      // The Apple9 four-simdgroup tiles (gate/up N128, down N256, 128
      // threads) must reproduce the shipped N128 x 8 tile bit for bit: each
      // output element sums its quant groups in the same order, so no
      // tolerance is granted.
      requireEqual(execute(narrow.moeDecode(fixture.shape, lanes), label + " sg4"), baseline,
                   label + " four-simdgroup");
    }
    // 12 and 48 rows leave 16-row ragged tiles in every routing fixture; 9,
    // 33 and 263 leave 8-row ones next to full tiles; 519/520 straddle the
    // wide router tile threshold.
    for (uint32_t rows : {1U, 3U, 7U, 8U, 9U, 12U, 31U, 32U, 33U, 48U, 100U,
                          255U, 256U, 263U, 519U, kMaximumRows})
      (void)execute(shipped.moePrefill(fixture.shape, rows), "prefill rows=" + std::to_string(rows));
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

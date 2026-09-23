#include "ops/Linear.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "tuning/LinearNumerics.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace splash;
using namespace splash::ops;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class Function> void rejects(Function function) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("invalid Linear plan or buffer was accepted");
}

uint16_t bf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fff + ((bits >> 16) & 1);
  return uint16_t(bits >> 16);
}

float fp32(uint16_t value) {
  return std::bit_cast<float>(uint32_t{value} << 16);
}

// Expected policy across GPU families, core counts and workload tile counts.
// expectedGroups restates the group distribution independently; expectedOneLane
// mirrors the one-lane rule. The literal anchors below independently guard
// selected policy boundaries and production shapes.
struct ExpectedConfig final {
  LinearTile tile;
  uint32_t groups;
  LinearSimdgroups simdgroups = LinearSimdgroups::Eight;
  uint32_t splits = 1;
};

// Tiles on the busiest core when `groups` threadgroups are placed round-robin
// on `cores` and group g streams tiles g, g + groups, ...; the operator's
// closed form is restated tile by tile.
uint32_t busiestCoreTiles(uint32_t tiles, uint32_t groups, uint32_t cores) {
  std::vector<uint32_t> load(cores);
  for (uint32_t tile = 0; tile < tiles; ++tile) ++load[tile % groups % cores];
  return *std::max_element(load.begin(), load.end());
}

// Groups per core up to which the one-tile grid wins, resident groups per
// core (one wave) and tiles per core from which the many-wave grid wins.
struct GroupRule final { uint32_t grid, wave, manyWaves; };

// The Apple10 rule as properties: the grid up to one wave per core and from
// many waves per core; between them the smallest count of at most two-tile
// groups, never above one wave, that leaves every core at ceil(tiles / cores)
// tiles while at least three quarters of the full-grid limit stays resident,
// and one full wave of longer chains when no such count exists.
uint32_t expectedGroups(uint32_t tiles, uint32_t cores, GroupRule rule) {
  if (tiles <= rule.grid * cores || tiles >= rule.manyWaves * cores) return tiles;
  for (uint32_t groups = (tiles + 1) / 2; groups <= rule.wave * cores; ++groups)
    if (groups >= rule.grid * cores * 3 / 4 &&
        busiestCoreTiles(tiles, groups, cores) == (tiles + cores - 1) / cores)
      return groups;
  return rule.wave * cores;
}

// Apple10 one-lane MPP rules: paired N256 from eight tiles per core.
// Split-K is available for offline experiments but never selected by default.
std::optional<ExpectedConfig> expectedOneLane(uint32_t cores,
                                              LinearMatrix matrix, LinearEpilogue epilogue) {
  const uint32_t n = matrix.outputSize;
  const uint32_t tiles256 = n / 256;
  if (epilogue == LinearEpilogue::None && tiles256 >= 8 * cores)
    return ExpectedConfig{LinearTile::Paired256,
                          std::min(tiles256, 4 * cores),
                          LinearSimdgroups::Four};
  return std::nullopt;
}

ExpectedConfig expectedDecode(uint32_t family, uint32_t cores, LinearMatrix matrix,
                              uint32_t lanes, LinearEpilogue epilogue) {
  if (family == 9 && !(lanes >= 3 && epilogue == LinearEpilogue::None &&
                      matrix.outputSize / 256 >= 2 * cores)) {
    const uint32_t columns = epilogue == LinearEpilogue::GateUp ? 32 : 64;
    const uint32_t grid = matrix.outputSize / columns;
    uint32_t selected = 1;
    for (uint32_t split : {1U, 2U, 4U, 8U}) {
      if (split > 1 && (matrix.inputSize % (64 * split) || matrix.inputSize / (64 * split) < 12)) break;
      selected = split;
      if (uint64_t(grid) * split >= uint64_t(cores) * 16) break;
    }
    return {LinearTile::Simdgroup, grid, LinearSimdgroups::Four, selected};
  }
  constexpr GroupRule n128{4, 4, 12}, m16{5, 4, 12}, n256{3, 3, 8}, gateUp{3, 3, 8},
      fourSimdgroups{8, 8, 24};
  const uint32_t tiles128 = matrix.outputSize / 128;
  const uint32_t tiles256 = matrix.outputSize / 256;
  // Apple9 is unmeasured under the balanced rule and keeps its one-tile grids
  // and the fused gate/up clamp of 2.25 resident groups per core.
  const auto groups = [&](uint32_t tiles, GroupRule rule) {
    return family >= 10 ? expectedGroups(tiles, cores, rule) : tiles;
  };
  if (family >= 10 && lanes == 1)
    if (const auto oneLane = expectedOneLane(cores, matrix, epilogue)) return *oneLane;
  if (epilogue == LinearEpilogue::GateUp) {
    if (family >= 10) return {LinearTile::N256, expectedGroups(tiles256, cores, gateUp)};
    return {LinearTile::N256, std::min(tiles256, uint32_t(std::lround(2.25 * cores)))};
  }
  if (lanes == 1) return {LinearTile::Paired128, groups(tiles128, n128)};
  if (family >= 10 && lanes == 3 && tiles128 <= cores && matrix.inputSize >= 4096)
    return {LinearTile::N128, tiles128, LinearSimdgroups::Eight};
  if (lanes == 3 && (family >= 10 ||
      (family == 9 && epilogue == LinearEpilogue::None)))
    return {LinearTile::N128, groups(tiles128, fourSimdgroups), LinearSimdgroups::Four};
  if (lanes >= 3 && epilogue == LinearEpilogue::None && tiles256 >= 2 * cores)
    return {LinearTile::N256, groups(tiles256, n256)};
  return {LinearTile::N128, groups(tiles128, lanes == 2 ? m16 : n128)};
}

std::string expectedPipeline(const ExpectedConfig &expected, uint32_t lanes,
                             LinearEpilogue epilogue) {
  if (expected.tile == LinearTile::Simdgroup)
    return epilogue == LinearEpilogue::GateUp ? "decode_linear_q4_sg_gate_up" :
        epilogue == LinearEpilogue::Residual ? "decode_linear_q4_sg_residual" : "decode_linear_q4_sg";
  if (expected.tile == LinearTile::Split32 || expected.tile == LinearTile::Split64) {
    std::string name = expected.tile == LinearTile::Split32 ? "decode_linear_q4_n32_split4"
                                                            : "decode_linear_q4_n64_split4";
    if (epilogue == LinearEpilogue::GateUp) name += "_gate_up";
    if (epilogue == LinearEpilogue::Residual) name += "_residual";
    return name;
  }
  if (expected.tile == LinearTile::Paired256) return "decode_linear_q4_n256_paired_sg4";
  if (epilogue == LinearEpilogue::GateUp)
    return lanes == 1 ? "decode_linear_q4_n256_gate_up" : lanes == 2 ? "decode_linear_q4_n256_gate_up_m16"
        : lanes == 3 ? "decode_linear_q4_n256_m24" : "decode_linear_q4_n256_m32";
  std::string name = expected.tile == LinearTile::N256 ? "decode_linear_q4_n256" : "decode_linear_q4_n128";
  if (epilogue == LinearEpilogue::Residual) name += "_residual";
  if (expected.tile == LinearTile::Paired128) return name + "_paired";
  if (lanes > 1) name += "_m" + std::to_string(lanes * 8);
  if (expected.simdgroups == LinearSimdgroups::Four) name += "_sg4";
  return name;
}

struct ProductionShape final { LinearMatrix matrix; LinearEpilogue epilogue; };
// Every decode projection of the Qwen3.8-27B and Qwen3.6-35B-A3B targets and
// their DFlash drafts (N x K, epilogue), plus K % 1024 != 0 controls.
constexpr std::array kProductionShapes{
    ProductionShape{{16640, 5120}, LinearEpilogue::None},
    ProductionShape{{14336, 5120}, LinearEpilogue::None},
    ProductionShape{{5120, 6144}, LinearEpilogue::Residual},
    ProductionShape{{17408, 5120}, LinearEpilogue::GateUp},
    ProductionShape{{5120, 17408}, LinearEpilogue::Residual},
    ProductionShape{{248320, 5120}, LinearEpilogue::None},
    ProductionShape{{1280, 5120}, LinearEpilogue::None},
    ProductionShape{{6144, 5120}, LinearEpilogue::None},
    ProductionShape{{5120, 4096}, LinearEpilogue::None},
    ProductionShape{{5120, 17408}, LinearEpilogue::None},
    ProductionShape{{5120, 25600}, LinearEpilogue::None},
    ProductionShape{{256, 5120}, LinearEpilogue::None},
    ProductionShape{{12544, 2048}, LinearEpilogue::None},
    ProductionShape{{9216, 2048}, LinearEpilogue::None},
    ProductionShape{{2048, 4096}, LinearEpilogue::Residual},
    ProductionShape{{248320, 2048}, LinearEpilogue::None},
    ProductionShape{{512, 2048}, LinearEpilogue::None},
    ProductionShape{{6144, 2048}, LinearEpilogue::None},
    ProductionShape{{2048, 4096}, LinearEpilogue::None},
    ProductionShape{{6144, 2048}, LinearEpilogue::GateUp},
    ProductionShape{{2048, 6144}, LinearEpilogue::None},
    ProductionShape{{2048, 16384}, LinearEpilogue::None},
    ProductionShape{{256, 2048}, LinearEpilogue::None},
    ProductionShape{{5120, 4352}, LinearEpilogue::None},
    ProductionShape{{5120, 4352}, LinearEpilogue::Residual},
    ProductionShape{{6144, 4352}, LinearEpilogue::GateUp},
    ProductionShape{{2048, 768}, LinearEpilogue::None}};

void narrowM24BoundaryPlans() {
  for (uint32_t cores : {16U, 20U}) {
    DeviceCapabilities device;
    device.appleGpuFamily = 10;
    device.gpuCoreCount = cores;
    Q4Linear linear(device);
    for (auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual}) {
      // Explicit values on both sides of the tile and K boundaries.
      for (auto [matrix, threads] : std::array<std::pair<LinearMatrix, uint32_t>, 4>{{
               {{cores * 128, 4096}, 256}, {{cores * 128 + 256, 4096}, 128},
               {{cores * 128, 3840}, 128}, {{256, 16384}, 256}}}) {
        const auto plan = linear.plan({matrix, 24, LinearPhase::Decode, epilogue});
        require(plan.threadsPerThreadgroup() == threads &&
                    plan.configuration().groups == matrix.outputSize / 128,
                "narrow M24 occupancy boundary changed");
      }
    }
  }
}

void baselinePlans() {
  for (const uint32_t family : {9U, 10U, 11U}) {
    for (const uint32_t reportedCores : {0U, 8U, 10U, 16U, 18U, 20U, 31U, 32U, 33U, 40U, 80U}) {
      const uint32_t cores = reportedCores ? reportedCores : 32U;
      DeviceCapabilities device;
      device.appleGpuFamily = family;
      device.gpuCoreCount = reportedCores;
      Q4Linear linear(device);
      for (const auto &shape : kProductionShapes) {
        for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
          const LinearWorkload workload{shape.matrix, lanes * 8, LinearPhase::Decode, shape.epilogue};
          const auto plan = linear.plan(workload);
          const auto expected = expectedDecode(family, cores, shape.matrix, lanes, shape.epilogue);
          require(plan.configuration() ==
                      LinearConfig{expected.tile, expected.groups, expected.simdgroups, expected.splits},
                  "Linear one-lane policy differs from its stated rules");
          require(plan.threadsPerThreadgroup() ==
                      (expected.simdgroups == LinearSimdgroups::Four ? 128U : 256U),
                  "Linear one-lane scope differs from its configuration");
          require(plan.pipeline() == expectedPipeline(expected, lanes, shape.epilogue),
                  "Linear one-lane pipeline differs from its configuration");
          const bool split = expected.tile == LinearTile::Split32 ||
              expected.tile == LinearTile::Split64;
          require(plan.partialSums() == (expected.tile == LinearTile::Simdgroup ? expected.splits : split ? 4U : 1U) &&
                      plan.tileColumns() == (expected.tile == LinearTile::Simdgroup ? (shape.epilogue == LinearEpilogue::GateUp ? 32U : 64U) : expected.tile == LinearTile::Split32 ? 32U
                          : expected.tile == LinearTile::Split64 ? 64U
                          : expected.tile == LinearTile::N256 ||
                            expected.tile == LinearTile::Paired256 ? 256U : 128U) &&
                      (!split || plan.configuration().groups ==
                           shape.matrix.outputSize / plan.tileColumns()),
                  "Linear one-lane tile geometry differs from its configuration");
        }
      }
      for (const uint32_t hidden : {5120U, 2048U}) {
        const bool large = hidden == 5120;
        const uint32_t intermediate = large ? 17408U : 6144U;
        struct Call final { LinearMatrix matrix; LinearEpilogue epilogue; };
        const std::array calls{
            Call{{large ? 16640U : 12544U, hidden}, LinearEpilogue::None},
            Call{{large ? 14336U : 9216U, hidden}, LinearEpilogue::None},
            Call{{hidden, large ? 6144U : 4096U}, LinearEpilogue::Residual},
            Call{{intermediate, hidden}, LinearEpilogue::GateUp},
            Call{{hidden, intermediate}, LinearEpilogue::Residual},
            Call{{hidden, intermediate}, LinearEpilogue::None},
            Call{{hidden, large ? 25600U : 16384U}, LinearEpilogue::None},
            Call{{248320, hidden}, LinearEpilogue::None},
            Call{{hidden / 4, hidden}, LinearEpilogue::None},
            Call{{6144, hidden}, LinearEpilogue::None},
            Call{{256, hidden}, LinearEpilogue::None}};
        for (const auto &call : calls) {
          for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
            const auto plan = linear.plan({call.matrix, lanes * 8, LinearPhase::Decode, call.epilogue});
            const auto expected = expectedDecode(family, cores, call.matrix, lanes, call.epilogue);
            require(plan.configuration() ==
                        LinearConfig{expected.tile, expected.groups, expected.simdgroups, expected.splits},
                    "Linear decode policy differs from its stated rules");
            require(plan.threadsPerThreadgroup() ==
                        (expected.simdgroups == LinearSimdgroups::Four ? 128U : 256U),
                    "Linear decode scope differs from its configuration");
            require(plan.pipeline() == expectedPipeline(expected, lanes, call.epilogue),
                    "Linear decode pipeline differs from its configuration");
            require(plan.secondPipeline().empty() ==
                        !(call.epilogue == LinearEpilogue::GateUp && lanes >= 3 && !plan.usesSimdgroup()),
                    "gate/up dispatch decomposition changed");
          }
        }
        for (const LinearMatrix matrix :
             {LinearMatrix{6144, hidden}, LinearMatrix{large ? 16640U : 12544U, hidden},
              LinearMatrix{hidden, large ? 6144U : 4096U}, LinearMatrix{intermediate, hidden}}) {
          for (const uint32_t rows : {1U, 7U, 31U, 32U, 33U, 127U, 2048U}) {
            for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                                       LinearEpilogue::UpWithGate}) {
              const auto plan = linear.plan({matrix, rows, LinearPhase::Prefill, epilogue});
              const double rowTiles = (rows + 31) / 32;
              const bool wide = rowTiles * (matrix.outputSize / 256) >= 8.0 * cores;
              // Apple10 everywhere, and Apple9 up to the measured 32-core
              // device, prefill with the four-simdgroup N128 tile; larger
              // Apple9 GPUs keep the wide-tile rule.
              const LinearConfig expected = family >= 10 || cores <= 32
                  ? LinearConfig{LinearTile::N128, 0, LinearSimdgroups::Four}
                  : LinearConfig{epilogue == LinearEpilogue::UpWithGate || wide
                                     ? LinearTile::N256 : LinearTile::N128, 0};
              require(plan.configuration() == expected,
                      "prefill policy differs from its stated rule");
              require(plan.threadsPerThreadgroup() ==
                          (family >= 10 || cores <= 32 ? 128U : 256U),
                      "prefill cooperative execution scope changed");
            }
          }
        }
      }
    }
  }
  // Anchors from the measured machines: 16- and 20-core Apple10 GPUs (M5 Pro)
  // and a 40-core Apple9 GPU (M3 Max). Changing a rule must change these
  // knowingly.
  const auto configured = [](uint32_t family, uint32_t cores, LinearWorkload workload) {
    DeviceCapabilities device;
    device.appleGpuFamily = family;
    device.gpuCoreCount = cores;
    return Q4Linear(device).plan(workload).configuration();
  };
  const LinearWorkload gateUp{{17408, 5120}, 8, LinearPhase::Decode, LinearEpilogue::GateUp};
  require(configured(10, 16, gateUp) == LinearConfig{LinearTile::N256, 36} &&
              configured(10, 20, gateUp) == LinearConfig{LinearTile::N256, 48} &&
              configured(9, 40, gateUp) == LinearConfig{LinearTile::Simdgroup, 544, LinearSimdgroups::Four, 2} &&
              // Unknown counts use the same intermediate estimate on both families.
              configured(10, 0, gateUp) == configured(10, 32, gateUp) &&
              configured(9, 0, gateUp) == configured(9, 32, gateUp),
          "fused gate/up grid does not follow the balanced two-tile rule");
  // Apple9 matrix K splits cover all decode widths; broad plain projections
  // retain their old multi-lane grids.
  require(configured(9, 16, gateUp) == LinearConfig{LinearTile::Simdgroup, 544, LinearSimdgroups::Four, 1} &&
              configured(9, 20, gateUp) == LinearConfig{LinearTile::Simdgroup, 544, LinearSimdgroups::Four, 1} &&
              configured(9, 20, {{16640, 5120}, 8}) == LinearConfig{LinearTile::Simdgroup, 260, LinearSimdgroups::Four, 2} &&
              configured(9, 16, {{16640, 5120}, 16}) == LinearConfig{LinearTile::Simdgroup, 260, LinearSimdgroups::Four, 1} &&
              configured(9, 20, {{16640, 5120}, 32}) == LinearConfig{LinearTile::N256, 65},
          "Apple9 decode grids changed without a measurement");
  // Former split-K defaults return to sequential tiles. Apple9 simdgroup and
  // wide paired N256 anchors remain unchanged.
  const LinearWorkload mixer27{{5120, 6144}, 8, LinearPhase::Decode, LinearEpilogue::Residual};
  const LinearWorkload mixer35{{2048, 4096}, 8, LinearPhase::Decode, LinearEpilogue::Residual};
  const LinearWorkload draftGateUp{{6144, 2048}, 8, LinearPhase::Decode, LinearEpilogue::GateUp};
  require(configured(10, 20, mixer27) == LinearConfig{LinearTile::Paired128, 40} &&
              configured(10, 16, mixer27) == LinearConfig{LinearTile::Paired128, 40} &&
              configured(10, 20, mixer35) == LinearConfig{LinearTile::Paired128, 16} &&
              configured(10, 16, mixer35) == LinearConfig{LinearTile::Paired128, 16} &&
              configured(10, 10, mixer35) == LinearConfig{LinearTile::Paired128, 16} &&
              configured(10, 10, {{5120, 17408}, 8}) == LinearConfig{LinearTile::Paired128, 40} &&
              configured(10, 20, {{1280, 5120}, 8}) == LinearConfig{LinearTile::Paired128, 10} &&
              configured(10, 16, {{256, 5120}, 8}) == LinearConfig{LinearTile::Paired128, 2} &&
              configured(10, 20, {{6144, 5120}, 8}) == LinearConfig{LinearTile::Paired128, 48} &&
              configured(10, 20, draftGateUp) == LinearConfig{LinearTile::N256, 24} &&
              configured(10, 16, draftGateUp) == LinearConfig{LinearTile::N256, 24},
          "Apple10 sequential defaults were not restored");
  require(configured(9, 40, mixer27) == LinearConfig{LinearTile::Simdgroup, 80, LinearSimdgroups::Four, 8} &&
              configured(9, 40, {{16640, 5120}, 8}) == LinearConfig{LinearTile::Simdgroup, 260, LinearSimdgroups::Four, 4} &&
              configured(9, 40, {{5120, 17408}, 8}) == LinearConfig{LinearTile::Simdgroup, 80, LinearSimdgroups::Four, 8} &&
              configured(9, 40, {{256, 5120}, 8}) == LinearConfig{LinearTile::Simdgroup, 4, LinearSimdgroups::Four, 4} &&
              configured(9, 18, mixer27) == LinearConfig{LinearTile::Simdgroup, 80, LinearSimdgroups::Four, 4},
          "Apple9 simdgroup policy anchors changed");
  const LinearConfig paired256Apple10_20{LinearTile::Paired256, 80, LinearSimdgroups::Four};
  require(configured(10, 20, {{248320, 5120}, 8}) == paired256Apple10_20 &&
              configured(10, 20, {{248320, 2048}, 8}) == paired256Apple10_20 &&
              configured(10, 16, {{248320, 5120}, 8}) ==
                  LinearConfig{LinearTile::Paired256, 64, LinearSimdgroups::Four} &&
              configured(10, 10, {{248320, 2048}, 8}) ==
                  LinearConfig{LinearTile::Paired256, 40, LinearSimdgroups::Four} &&
              configured(9, 40, {{248320, 5120}, 8}) ==
                  LinearConfig{LinearTile::Simdgroup, 3880, LinearSimdgroups::Four, 1} &&
              configured(9, 80, {{248320, 2048}, 8}) ==
                  LinearConfig{LinearTile::Simdgroup, 3880, LinearSimdgroups::Four, 1} &&
              configured(10, 20, {{40960, 5120}, 8}) ==
                  LinearConfig{LinearTile::Paired256, 80, LinearSimdgroups::Four} &&
              configured(10, 20, {{40704, 5120}, 8}) == LinearConfig{LinearTile::Paired128, 318},
          "one-lane paired N256 anchors changed");
  // K % 1024 != 0 is legal for matrix tiles; Apple10 keeps its shipped rules.
  require(configured(10, 20, {{5120, 4352}, 8}) == LinearConfig{LinearTile::Paired128, 40} &&
              configured(9, 40, {{5120, 4352}, 8, LinearPhase::Decode, LinearEpilogue::Residual}) ==
                  LinearConfig{LinearTile::Simdgroup, 80, LinearSimdgroups::Four, 4} &&
              configured(9, 40, {{6144, 4352}, 8, LinearPhase::Decode, LinearEpilogue::GateUp}) ==
                  LinearConfig{LinearTile::Simdgroup, 192, LinearSimdgroups::Four, 4} &&
              configured(10, 20, {{2048, 768}, 8}) == LinearConfig{LinearTile::Paired128, 16} &&
              configured(10, 20, {{2048, 4096}, 16}) == LinearConfig{LinearTile::N128, 16} &&
              configured(9, 40, {{5120, 6144}, 24, LinearPhase::Decode, LinearEpilogue::Residual}) ==
                  LinearConfig{LinearTile::Simdgroup, 80, LinearSimdgroups::Four, 8} &&
              configured(10, 20, {{6144, 2048}, 32, LinearPhase::Decode, LinearEpilogue::GateUp}) ==
                  LinearConfig{LinearTile::N256, 24},
          "one-lane fallbacks or multi-lane rules changed");
  // Balanced two-tile groups above one wave: 130 paired tiles keep three
  // groups per core on 20 cores and one full wave of longer chains on 16; 98
  // tiles land on 60 and 50 groups; the M16 grid holds to five per core.
  require(configured(10, 20, {{16640, 5120}, 8}) == LinearConfig{LinearTile::Paired128, 70} &&
              configured(10, 16, {{16640, 5120}, 8}) == LinearConfig{LinearTile::Paired128, 64} &&
              configured(10, 20, {{12544, 2048}, 8}) == LinearConfig{LinearTile::Paired128, 60} &&
              configured(10, 16, {{12544, 2048}, 8}) == LinearConfig{LinearTile::Paired128, 50} &&
              configured(10, 20, {{16640, 5120}, 16}) == LinearConfig{LinearTile::N128, 75} &&
              configured(10, 16, {{16640, 5120}, 16}) == LinearConfig{LinearTile::N128, 64} &&
              configured(10, 20, {{12544, 2048}, 16}) == LinearConfig{LinearTile::N128, 98} &&
              configured(10, 16, {{16640, 5120}, 24}) ==
                  LinearConfig{LinearTile::N128, 96, LinearSimdgroups::Four} &&
              configured(10, 20, {{16640, 5120}, 24}) ==
                  LinearConfig{LinearTile::N128, 130, LinearSimdgroups::Four} &&
              configured(10, 20, {{16640, 5120}, 32}) == LinearConfig{LinearTile::N256, 45} &&
              configured(10, 20, {{248320, 5120}, 16}) == LinearConfig{LinearTile::N128, 1940} &&
              configured(9, 20, {{16640, 5120}, 8}) == LinearConfig{LinearTile::Simdgroup, 260, LinearSimdgroups::Four, 2} &&
              configured(10, 0, {{16640, 5120}, 8}) == configured(10, 32, {{16640, 5120}, 8}),
          "persistent decode groups changed for the measured shapes");
  require(configured(10, 16, {{14336, 5120}, 24}) ==
              LinearConfig{LinearTile::N128, 112, LinearSimdgroups::Four} &&
          configured(10, 16, {{14336, 5120}, 32}) == LinearConfig{LinearTile::N256, 40} &&
          configured(10, 40, {{14336, 5120}, 32}) == LinearConfig{LinearTile::N128, 112} &&
          configured(9, 40, {{14336, 5120}, 24}) ==
              LinearConfig{LinearTile::Simdgroup, 224, LinearSimdgroups::Four, 4} &&
          configured(9, 40, {{248320, 5120}, 24}) ==
              LinearConfig{LinearTile::N128, 1940, LinearSimdgroups::Four} &&
          configured(9, 40, {{5120, 17408}, 24, LinearPhase::Decode, LinearEpilogue::Residual}) ==
              LinearConfig{LinearTile::Simdgroup, 80, LinearSimdgroups::Four, 8},
          "decode tile rules changed for the measured shapes");
  require(configured(10, 16, {{5120, 17408}, 8}) == LinearConfig{LinearTile::Paired128, 40} &&
          configured(10, 16, {{5120, 17408}, 16}) == LinearConfig{LinearTile::N128, 40} &&
          configured(10, 16, {{6144, 5120}, 2048, LinearPhase::Prefill}) ==
              LinearConfig{LinearTile::N128, 0, LinearSimdgroups::Four} &&
          configured(10, 16, {{17408, 5120}, 2048, LinearPhase::Prefill,
                              LinearEpilogue::UpWithGate}) ==
              LinearConfig{LinearTile::N128, 0, LinearSimdgroups::Four} &&
          configured(9, 40, {{6144, 5120}, 2048, LinearPhase::Prefill}) ==
              LinearConfig{LinearTile::N256, 0} &&
          configured(9, 40, {{6144, 5120}, 32, LinearPhase::Prefill}) ==
              LinearConfig{LinearTile::N128, 0} &&
          configured(9, 40, {{17408, 5120}, 32, LinearPhase::Prefill,
                              LinearEpilogue::UpWithGate}) ==
              LinearConfig{LinearTile::N256, 0},
          "one-lane pipelining or prefill tile rule changed for the measured shapes");
}

// `widestCandidates` accumulates the largest candidate set seen, so main() can
// check that the bound below is reached and not merely respected.
void planContracts(uint32_t family, uint32_t cores, size_t &widestCandidates) {
  DeviceCapabilities device;
  device.appleGpuFamily = family;
  device.gpuCoreCount = cores;
  Q4Linear linear(device);
  // Include a wide Apple9 projection with four distinct persistent grids,
  // both paired N256 grids and all four K splits to reach the candidate bound.
  for (const LinearMatrix matrix : {LinearMatrix{512, 256}, LinearMatrix{768, 768},
                                    LinearMatrix{16640, 5120}, LinearMatrix{12544, 2048},
                                    LinearMatrix{23040, 2048}, LinearMatrix{131072, 4096}}) {
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                                 LinearEpilogue::GateUp}) {
        const LinearWorkload workload{matrix, lanes * 8, LinearPhase::Decode, epilogue};
        const auto candidates = linear.candidates(workload);
        require(linear.plan(workload).configuration().tile != LinearTile::Split32 &&
                    linear.plan(workload).configuration().tile != LinearTile::Split64,
                "offline split-K candidate became a serving default");
        require(!candidates.empty() && candidates.size() <= Q4Linear::kMaximumCandidates,
                "Linear candidates exceed the bounded set");
        widestCandidates = std::max(widestCandidates, candidates.size());
        require(candidates.front().configuration() == linear.plan(workload).configuration(),
                "Linear baseline is not first candidate");
        uint32_t fourScopeCandidates = 0, splitCandidates = 0, simdgroupCandidates = 0;
        for (size_t index = 0; index < candidates.size(); ++index) {
          const auto &plan = candidates[index];
          const bool four = plan.configuration().simdgroups == LinearSimdgroups::Four;
          const auto tile = plan.configuration().tile;
          const bool split = tile == LinearTile::Split32 || tile == LinearTile::Split64;
          if (split) {
            ++splitCandidates;
            require(lanes == 1 && matrix.inputSize % 1024 == 0 && plan.partialSums() == 4 &&
                        plan.configuration().groups == matrix.outputSize / plan.tileColumns() &&
                        (epilogue != LinearEpilogue::GateUp || tile == LinearTile::Split32) &&
                        plan.secondPipeline().empty(),
                    "split-K candidate escaped its one-lane full-grid contract");
          } else if (plan.usesSimdgroup()) {
            ++simdgroupCandidates;
            require(family == 9 && four &&
                        plan.partialSums() == plan.configuration().splits &&
                        plan.configuration().groups == matrix.outputSize / plan.tileColumns(),
                    "simdgroup candidate escaped its full-grid contract");
          } else {
            require(plan.partialSums() == 1, "sequential candidate reports partial sums");
          }
          if (four) {
            ++fourScopeCandidates;
            const bool oneLane = lanes == 1 &&
                (tile == LinearTile::Split32 ||
                 (tile == LinearTile::Paired256 && epilogue == LinearEpilogue::None));
            require(((lanes == 3 && tile == LinearTile::N128 && epilogue != LinearEpilogue::GateUp) ||
                     oneLane || plan.usesSimdgroup()) && plan.secondPipeline().empty(),
                    "four-SIMDgroup candidate escaped its precompiled workload set");
            if (lanes == 3 && !plan.usesSimdgroup())
              require(plan.pipeline() == (epilogue == LinearEpilogue::Residual
                          ? "decode_linear_q4_n128_residual_m24_sg4" : "decode_linear_q4_n128_m24_sg4"),
                      "four-SIMDgroup plan chose the wrong pipeline");
          }
          require(plan.threadsPerThreadgroup() == (four ? 128 : 256),
                  "Linear plan scope/thread count disagree");
          require(plan.storageRows() == lanes * 8 && !plan.sumsBytes() && !plan.downSumsBytes(),
                  "decode storage/sums contract changed");
          require(plan.gateScratchBytes() == (epilogue == LinearEpilogue::GateUp && lanes >= 3 && !plan.usesSimdgroup()
                      ? uint64_t{lanes} * 8 * matrix.outputSize * 2 : 0),
                  "decode gate scratch disagrees with decomposition");
          for (size_t prior = 0; prior < index; ++prior)
            require(candidates[prior].configuration() != plan.configuration(),
                    "duplicate Linear candidates");
        }
        // One lane always lists the paired N256 tile for plain projections and
        // the split tiles when K allows them: Split32 for every decode
        // epilogue, Split64 for the single-stream ones.
        require((fourScopeCandidates != 0) ==
                    (family == 9 || (lanes == 3 && epilogue != LinearEpilogue::GateUp) ||
                     (lanes == 1 && (family == 9 || epilogue == LinearEpilogue::None || matrix.inputSize % 1024 == 0))),
                "Linear candidate set omitted or added four-SIMDgroup plans");
        uint32_t legalSplits = 0;
        if (family == 9)
          for (uint32_t split : {1U, 2U, 4U, 8U})
            legalSplits += matrix.inputSize % (64 * split) == 0;
        require(simdgroupCandidates == legalSplits,
                "Linear candidates omit a legal Apple9 K split");
        require(splitCandidates == (lanes == 1 && matrix.inputSize % 1024 == 0
                                        ? (epilogue == LinearEpilogue::GateUp ? 1U : 2U) : 0U),
                "Linear candidate set omitted or added split-K plans");
      }
    }
    for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                               LinearEpilogue::UpWithGate}) {
      // Every prefill epilogue has the eight-simdgroup N256 tile and the
      // four-simdgroup N128 tile; the plain and residual ones also N128/8.
      const auto candidates = linear.candidates({matrix, 2048, LinearPhase::Prefill, epilogue});
      const bool up = epilogue == LinearEpilogue::UpWithGate;
      require(candidates.size() == (up ? 2U : 3U) &&
                  candidates.front().configuration() ==
                      linear.plan({matrix, 2048, LinearPhase::Prefill, epilogue}).configuration(),
              "prefill candidate set changed");
      for (const auto &plan : candidates) {
        const bool four = plan.configuration().simdgroups == LinearSimdgroups::Four;
        require(plan.configuration().groups == 0 && plan.threadsPerThreadgroup() == (four ? 128 : 256) &&
                    (!four || plan.configuration().tile == LinearTile::N128) &&
                    (!up || four || plan.configuration().tile == LinearTile::N256) &&
                    (plan.pipeline().ends_with("_sg4") == four),
                "prefill candidate scope, tile or pipeline name disagree");
      }
      for (uint32_t rows = 1; rows <= 2048; ++rows) {
        const auto plan = linear.plan({matrix, rows, LinearPhase::Prefill, epilogue});
        const uint64_t storageRows = (rows + 31) / 32 * 32;
        require(plan.storageRows() == storageRows &&
                    plan.sumsBytes() == storageRows * (matrix.inputSize / 64) * 4,
                "prefill padding or sums bound incorrect");
        require(plan.gateScratchBytes() == (up ? storageRows * matrix.outputSize * 2 : 0) &&
                    plan.downSumsBytes() == (up ? storageRows * (matrix.outputSize / 64) * 4 : 0),
                "prefill gate/output sums bound incorrect");
      }
    }
  }
  const LinearWorkload valid{{512, 256}, 8, LinearPhase::Decode, LinearEpilogue::None};
  for (const LinearWorkload invalid : {
           LinearWorkload{{0, 256}, 8}, LinearWorkload{{128, 256}, 8},
           LinearWorkload{{384, 256}, 8}, LinearWorkload{{512, 64}, 8},
           LinearWorkload{{512, 320}, 8}, LinearWorkload{{512, 0}, 8},
           LinearWorkload{{512, 256}, 0}, LinearWorkload{{512, 256}, 7},
           LinearWorkload{{512, 256}, 40},
           LinearWorkload{{512, 256}, 8, static_cast<LinearPhase>(255)},
           LinearWorkload{{512, 256}, 8, LinearPhase::Decode, static_cast<LinearEpilogue>(255)},
           LinearWorkload{{512, 256}, 8, LinearPhase::Decode, LinearEpilogue::UpWithGate},
           LinearWorkload{{512, 256}, 8, LinearPhase::Prefill, LinearEpilogue::GateUp},
           LinearWorkload{{512, 256}, 2049, LinearPhase::Prefill}})
    rejects([&] { (void)linear.plan(invalid); });
  for (const LinearConfig invalid : {
           LinearConfig{LinearTile::N128, 0}, LinearConfig{LinearTile::N128, 5},
           LinearConfig{static_cast<LinearTile>(255), 1}})
    rejects([&] { (void)Q4Linear::plan(valid, invalid); });
  rejects([&] { (void)Q4Linear::plan({{512, 256}, 16}, {LinearTile::Paired128, 1}); });
  rejects([&] { (void)Q4Linear::plan({{512, 256}, 32, LinearPhase::Prefill}, {LinearTile::N128, 1}); });
  rejects([&] { (void)Q4Linear::plan({{512, 256}, 32, LinearPhase::Prefill}, {LinearTile::Paired128, 0}); });
  rejects([&] { (void)Q4Linear::plan({{512, 256}, 8, LinearPhase::Decode, LinearEpilogue::Residual}, {LinearTile::N256, 1}); });
  rejects([&] { (void)Q4Linear::plan({{512, 256}, 8, LinearPhase::Decode, LinearEpilogue::GateUp}, {LinearTile::N128, 1}); });
  rejects([&] { (void)Q4Linear::plan({{512, 256}, 8, LinearPhase::Prefill, LinearEpilogue::UpWithGate}, {LinearTile::N128, 0}); });
  // Split tiles: one lane, K % 1024 == 0, the full grid and the kernel's own
  // simdgroup count; Split64 has no gate/up form. Paired256: one lane, plain
  // epilogue, four simdgroups. Neither exists in prefill.
  const LinearWorkload splitWorkload{{512, 1024}, 8, LinearPhase::Decode, LinearEpilogue::None};
  const LinearWorkload splitResidual{{512, 1024}, 8, LinearPhase::Decode, LinearEpilogue::Residual};
  const LinearWorkload splitGateUp{{512, 1024}, 8, LinearPhase::Decode, LinearEpilogue::GateUp};
  const LinearConfig matrixTile{LinearTile::Simdgroup, 8, LinearSimdgroups::Four, 4};
  require(Q4Linear::plan(splitWorkload, matrixTile).scratchSize().partials == 4ULL * 64 * 512,
          "simdgroup partial workspace does not cover both fragment streams");
  for (const uint32_t splits : {0U, 3U, 16U}) {
    auto invalid = matrixTile;
    invalid.splits = splits;
    rejects([&] { (void)Q4Linear::plan(splitWorkload, invalid); });
  }
  rejects([&] { (void)Q4Linear::plan({{512, 768}, 8},
      {LinearTile::Simdgroup, 8, LinearSimdgroups::Four, 8}); });
  for (uint32_t rows : {8U,16U,24U,32U}) {
    const auto size = Q4Linear::plan({{512,1024},rows}, matrixTile).scratchSize();
    require(size.input == uint64_t(rows)*1024*2 && size.sums == uint64_t(rows)*16*4 &&
                size.partials == uint64_t(rows)*512*2*4*4 && size.counters == uint64_t(rows/8)*8*4,
            "matrix row tiles must own disjoint input, sums, partials and counters");
  }
  rejects([&] { (void)Q4Linear::plan(splitWorkload,
      {LinearTile::Simdgroup, 4, LinearSimdgroups::Four, 4}); });
  rejects([&] { (void)Q4Linear::plan(splitWorkload,
      {LinearTile::Simdgroup, 8, LinearSimdgroups::Eight, 4}); });
  const LinearConfig split32{LinearTile::Split32, 16, LinearSimdgroups::Four};
  const LinearConfig split64{LinearTile::Split64, 8, LinearSimdgroups::Eight};
  const LinearConfig paired256{LinearTile::Paired256, 2, LinearSimdgroups::Four};
  {
    const auto plan = Q4Linear::plan(splitWorkload, split64);
    require(plan.pipeline() == "decode_linear_q4_n64_split4" && plan.threadsPerThreadgroup() == 256 &&
                plan.tileColumns() == 64 && plan.partialSums() == 4 && plan.secondPipeline().empty(),
            "Split64 plan geometry or pipeline is wrong");
    const auto residual = Q4Linear::plan(splitResidual, split32);
    require(residual.pipeline() == "decode_linear_q4_n32_split4_residual" &&
                residual.threadsPerThreadgroup() == 128 && residual.tileColumns() == 32 &&
                residual.partialSums() == 4,
            "Split32 residual plan geometry or pipeline is wrong");
    require(Q4Linear::plan(splitResidual, split64).pipeline() == "decode_linear_q4_n64_split4_residual" &&
                Q4Linear::plan(splitWorkload, split32).pipeline() == "decode_linear_q4_n32_split4",
            "split plan pipeline names are wrong");
    const auto gateUpPlan = Q4Linear::plan(splitGateUp, split32);
    require(gateUpPlan.pipeline() == "decode_linear_q4_n32_split4_gate_up" &&
                gateUpPlan.threadsPerThreadgroup() == 128 && gateUpPlan.secondPipeline().empty() &&
                gateUpPlan.gateScratchBytes() == 0,
            "Split32 gate/up plan geometry or pipeline is wrong");
    const auto wide = Q4Linear::plan(splitWorkload, paired256);
    require(wide.pipeline() == "decode_linear_q4_n256_paired_sg4" && wide.threadsPerThreadgroup() == 128 &&
                wide.tileColumns() == 256 && wide.partialSums() == 1,
            "Paired256 plan geometry or pipeline is wrong");
  }
  rejects([&] { (void)Q4Linear::plan(splitWorkload, {LinearTile::Split32, 16, LinearSimdgroups::Eight}); });
  rejects([&] { (void)Q4Linear::plan(splitWorkload, {LinearTile::Split64, 8, LinearSimdgroups::Four}); });
  rejects([&] { (void)Q4Linear::plan(splitWorkload, {LinearTile::Paired256, 2, LinearSimdgroups::Eight}); });
  rejects([&] { (void)Q4Linear::plan(splitWorkload, {LinearTile::Split32, 8, LinearSimdgroups::Four}); });
  rejects([&] { (void)Q4Linear::plan(splitWorkload, {LinearTile::Split64, 4, LinearSimdgroups::Eight}); });
  rejects([&] { (void)Q4Linear::plan(splitWorkload, {LinearTile::Split64, 16, LinearSimdgroups::Eight}); });
  rejects([&] { (void)Q4Linear::plan(splitGateUp, split64); });
  rejects([&] { (void)Q4Linear::plan(splitResidual, paired256); });
  rejects([&] { (void)Q4Linear::plan(splitGateUp, paired256); });
  for (const LinearConfig config : {split32, split64})
    rejects([&] { (void)Q4Linear::plan({{512, 768}, 8}, config); });
  for (uint32_t rows : {16U, 24U, 32U})
    for (const LinearConfig config : {split32, split64, paired256})
      rejects([&] { (void)Q4Linear::plan({{512, 1024}, rows}, config); });
  for (const auto tile : {LinearTile::Split32, LinearTile::Split64, LinearTile::Paired256})
    rejects([&] { (void)Q4Linear::plan({{512, 1024}, 32, LinearPhase::Prefill},
        {tile, 0, tile == LinearTile::Split64 ? LinearSimdgroups::Eight : LinearSimdgroups::Four}); });
  const LinearWorkload fourWorkload{{512, 256}, 24, LinearPhase::Decode, LinearEpilogue::None};
  const LinearConfig fourConfig{LinearTile::N128, 1, LinearSimdgroups::Four};
  for (uint32_t scope : {0U, 1U, 2U, 3U, 16U, 255U})
    rejects([&] { (void)Q4Linear::plan(fourWorkload,
        {LinearTile::N128, 1, static_cast<LinearSimdgroups>(scope)}); });
  for (uint32_t rows : {8U, 16U, 32U})
    rejects([&] { (void)Q4Linear::plan({{512, 256}, rows}, fourConfig); });
  for (const auto tile : {LinearTile::N256, LinearTile::Paired128})
    rejects([&] { (void)Q4Linear::plan(fourWorkload,
        {tile, 1, LinearSimdgroups::Four}); });
  for (const auto epilogue : {LinearEpilogue::GateUp, LinearEpilogue::UpWithGate})
    rejects([&] { (void)Q4Linear::plan({{512, 256}, 24, LinearPhase::Decode, epilogue},
                                      fourConfig); });
  for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                             LinearEpilogue::UpWithGate}) {
    const LinearWorkload prefill{{512, 256}, 24, LinearPhase::Prefill, epilogue};
    require(Q4Linear::plan(prefill, {LinearTile::N128, 0, LinearSimdgroups::Four})
                    .threadsPerThreadgroup() == 128,
            "four-SIMDgroup prefill plan was rejected");
    rejects([&] { (void)Q4Linear::plan(prefill, {LinearTile::N256, 0, LinearSimdgroups::Four}); });
    rejects([&] { (void)Q4Linear::plan(prefill, {LinearTile::N128, 1, LinearSimdgroups::Four}); });
  }

  const LinearWorkload other{{768, 768}, 16, LinearPhase::Decode, LinearEpilogue::None};
  const auto original = linear.plan(other).configuration();
  const LinearConfig selected{LinearTile::N256, 1};
  const std::array choices{LinearChoice{other, selected}, LinearChoice{valid, selected}};
  linear.setChoices(choices);
  require(linear.plan(valid).configuration() == selected &&
              linear.plan(other).configuration() == selected,
          "unsorted profile choices did not take effect");
  const std::array duplicate{choices[0], choices[0]};
  rejects([&] { linear.setChoices(duplicate); });
  const std::array invalid{LinearChoice{valid, {LinearTile::N128, 0}}};
  rejects([&] { linear.setChoices(invalid); });
  require(linear.plan(other).configuration() == selected,
          "invalid profile update changed installed choices");
  linear.setChoices({});
  require(linear.plan(other).configuration() == original,
          "clearing profile did not restore the shipped baseline");
  const auto baselineFourWorkload = linear.plan(fourWorkload);
  const std::array fourChoices{LinearChoice{fourWorkload, fourConfig}};
  linear.setChoices(fourChoices);
  require(linear.plan(fourWorkload).configuration() == fourConfig &&
              linear.plan(fourWorkload).threadsPerThreadgroup() == 128,
          "installed four-SIMDgroup choice did not reach the selected plan");
  const std::array invalidScope{LinearChoice{valid, fourConfig}};
  rejects([&] { linear.setChoices(invalidScope); });
  require(linear.plan(fourWorkload).configuration() == fourConfig,
          "invalid scope update changed installed choices");
  linear.setChoices({});
  require(linear.plan(fourWorkload).configuration() == baselineFourWorkload.configuration() &&
              linear.plan(fourWorkload).threadsPerThreadgroup() ==
                  baselineFourWorkload.threadsPerThreadgroup(),
          "clearing choices did not restore the shipped execution scope");
}

// Exercise continuous core counts, not just measured SKU anchors. These
// contracts check legal grids, bounded candidates and override workspace;
// they do not claim performance on simulated hardware.
// Every affine plan (configuration, pipelines, input layout and scratch) for
// families 9-11, core counts 0-128 and a grid of shapes covering the rule
// boundaries, hashed. The expected value was recorded from the policy before
// GGUF projections joined LinearPlan; affine planning must not move. Run with
// LINEAR_POLICY_HASH=print to see the current value.
void affinePolicyIdentity() {
  constexpr uint64_t kExpected = 0xcb12687a116bc240ULL;  // 893970 plans
  uint64_t hash = 1469598103934665603ULL;
  const auto mix = [&](const void *data, size_t bytes) {
    const auto *p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < bytes; ++i) hash = (hash ^ p[i]) * 1099511628211ULL;
  };
  const auto mixValue = [&](uint64_t value) { mix(&value, sizeof value); };
  const auto mixText = [&](std::string_view text) { mixValue(text.size()); mix(text.data(), text.size()); };
  uint64_t plans = 0;
  for (const uint32_t family : {9U, 10U, 11U})
    for (uint32_t cores = 0; cores <= 128; ++cores) {
      DeviceCapabilities device;
      device.appleGpuFamily = family;
      device.gpuCoreCount = cores;
      const Q4Linear linear(device);
      for (const uint32_t n : {256U, 512U, 1024U, 2048U, 4096U, 5120U, 6144U, 9216U, 10240U,
                               12544U, 14336U, 16640U, 17408U, 248320U})
        for (const uint32_t k : {2048U, 4096U, 5120U, 6144U, 17408U}) {
          const auto record = [&](LinearWorkload w) {
            const LinearPlan plan = linear.plan(w);
            const LinearConfig c = plan.configuration();
            const LinearScratchSize scratch = plan.scratchSize();
            for (const uint64_t v : {uint64_t{family}, uint64_t{cores}, uint64_t{n}, uint64_t{k},
                                     uint64_t{w.rows}, uint64_t(w.phase), uint64_t(w.epilogue),
                                     uint64_t(c.tile), uint64_t{c.groups}, uint64_t(c.simdgroups),
                                     uint64_t{c.splits}, uint64_t(plan.input()), scratch.input,
                                     scratch.sums, scratch.partials, scratch.counters})
              mixValue(v);
            mixText(plan.pipeline());
            mixText(plan.secondPipeline());
            ++plans;
          };
          for (uint32_t lanes = 1; lanes <= 4; ++lanes)
            for (const auto e : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp})
              record({{n, k}, lanes * 8, LinearPhase::Decode, e});
          for (const uint32_t rows : {1U, 8U, 32U, 33U, 128U, 512U, 2048U})
            for (const auto e : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate})
              record({{n, k}, rows, LinearPhase::Prefill, e});
        }
    }
  if (const char *mode = std::getenv("LINEAR_POLICY_HASH"); mode && std::string_view(mode) == "print")
    std::cout << "affine policy hash " << plans << " plans 0x" << std::hex << hash << std::dec << '\n';
  require(hash == kExpected, "affine Linear policy changed (LINEAR_POLICY_HASH=print shows the new hash)");
}

// GGUF projections plan with their segments: the staged split policy for
// single tensors, no K splits for fused tensors and gate/up, exact split
// scratch, and prefill tiles of 32 or 128 rows.
void ggufPlans() {
  DeviceCapabilities device;
  device.appleGpuFamily = 10;
  device.gpuCoreCount = 16;
  const Q4Linear linear(device);
  const auto projection = [](uint32_t n, uint32_t k, uint32_t segments) {
    Q4Projection p;
    p.outputSize = n;
    p.inputSize = k;
    for (uint32_t i = 0; i < segments; ++i) {
      GgufSegment s;
      s.outputSize = n / segments;
      s.inputSize = k;
      s.columnOffset = i * (n / segments);
      p.gguf.push_back(s);
    }
    return p;
  };
  const LinearWorkload down{{5120, 17408}, 8, LinearPhase::Decode, LinearEpilogue::Residual};
  const LinearPlan single = linear.plan(down, projection(5120, 17408, 1));
  require(single.workload().quant == QuantFamily::Gguf &&
              single.configuration() == LinearConfig{LinearTile::GgufStaged, 80, LinearSimdgroups::Two, 8} &&
              single.input() == LinearInput::Plain && single.partialSums() == 8 &&
              single.scratchSize().partials == uint64_t{8} * 8 * 5120 * 4 &&
              single.scratchSize().counters == 80 * 4 && single.scratchSize().input == 0,
          "GGUF single-tensor decode plan");
  require(linear.plan({{5120, 6144}, 32, LinearPhase::Decode, LinearEpilogue::Residual},
                      projection(5120, 6144, 1)).configuration().splits == 4 &&
              linear.plan({{10240, 5120}, 24, LinearPhase::Decode, LinearEpilogue::None},
                          projection(10240, 5120, 1)).configuration().splits == 2,
          "GGUF staged split policy");
  const LinearPlan fused = linear.plan({{10240, 5120}, 8, LinearPhase::Decode, LinearEpilogue::None},
                                       projection(10240, 5120, 2));
  const LinearPlan gateUp = linear.plan({{17408, 5120}, 16, LinearPhase::Decode, LinearEpilogue::GateUp},
                                        projection(17408, 5120, 1));
  require(fused.configuration().splits == 1 && fused.scratchSize().bytes() == 0 &&
              gateUp.configuration().splits == 1 && gateUp.gateScratchBytes() == 0,
          "GGUF fused and gate/up plans take no K splits");
  for (const auto [rows, storage] : {std::pair{1U, 32U}, {20U, 32U}, {32U, 32U}, {33U, 128U},
                                     {100U, 128U}, {2048U, 2048U}}) {
    const LinearPlan prefill = linear.plan({{5120, 17408}, rows, LinearPhase::Prefill, LinearEpilogue::UpWithGate},
                                           projection(5120, 17408, 1));
    require(prefill.storageRows() == storage && prefill.sumsBytes() == 0 && prefill.downSumsBytes() == 0 &&
                prefill.gateScratchBytes() == uint64_t{storage} * 5120 * 2 &&
                prefill.scratchSize().bytes() == 0,
            "GGUF prefill tile rows");
  }
  // Affine and GGUF plans do not mix.
  rejects([&] { (void)Q4Linear::plan(down, {LinearTile::GgufStaged, 80, LinearSimdgroups::Two, 8}); });
  LinearWorkload gguf = down;
  gguf.quant = QuantFamily::Gguf;
  rejects([&] { (void)Q4Linear::plan(gguf, {LinearTile::N128, 40}); });
  rejects([&] { (void)Q4Linear::plan(gguf, {LinearTile::GgufStaged, 40, LinearSimdgroups::Two, 8}); });
  rejects([&] { (void)Q4Linear::plan(gguf, {LinearTile::GgufStaged, 80, LinearSimdgroups::Two, 3}); });
  // The arena bound is the single-tensor plan.
  require(linear.decodeScratchSize(gguf).partials == single.scratchSize().partials,
          "GGUF decode scratch bound");

  // Apple9 decodes every GGUF width with the exact register tile, all lanes
  // in one threadgroup, and K splits from the core count; prefill stages.
  const auto registerSplits = [&](uint32_t cores, uint32_t n, uint32_t k, uint32_t rows,
                                  LinearEpilogue epilogue, uint32_t segments) {
    DeviceCapabilities apple9;
    apple9.appleGpuFamily = 9;
    apple9.gpuCoreCount = cores;
    const LinearPlan plan = Q4Linear(apple9).plan({{n, k}, rows, LinearPhase::Decode, epilogue},
                                                  projection(n, k, segments));
    require(plan.configuration().tile == LinearTile::GgufSimdgroup &&
                plan.configuration().groups == n / 64 &&
                plan.configuration().simdgroups == LinearSimdgroups::Four &&
                plan.input() == LinearInput::Table16,
            "Apple9 GGUF register plan");
    return plan.configuration().splits;
  };
  // Sixteen column/K threadgroups per core, at least two 256-input units per
  // partition: 27B down, out_proj, gdn_in, attn_in, gate/up and the
  // vocabulary head on 40 and 10 cores, and narrow tensors.
  require(registerSplits(40, 5120, 17408, 8, LinearEpilogue::Residual, 1) == 8 &&
              registerSplits(40, 5120, 6144, 32, LinearEpilogue::Residual, 1) == 8 &&
              registerSplits(40, 16640, 5120, 16, LinearEpilogue::None, 3) == 4 &&
              registerSplits(40, 14336, 5120, 8, LinearEpilogue::None, 3) == 4 &&
              registerSplits(40, 17408, 5120, 24, LinearEpilogue::GateUp, 1) == 4 &&
              registerSplits(40, 248320, 5120, 8, LinearEpilogue::None, 1) == 1 &&
              registerSplits(10, 5120, 17408, 8, LinearEpilogue::Residual, 1) == 2 &&
              registerSplits(10, 16640, 5120, 8, LinearEpilogue::None, 3) == 1 &&
              registerSplits(40, 1024, 5120, 8, LinearEpilogue::None, 1) == 8 &&
              registerSplits(40, 1024, 1024, 8, LinearEpilogue::None, 1) == 2,
          "Apple9 GGUF register split policy");
  DeviceCapabilities apple9;
  apple9.appleGpuFamily = 9;
  apple9.gpuCoreCount = 40;
  const Q4Linear m3(apple9);
  for (const uint32_t rows : {8U, 32U}) {
    const LinearPlan plan = m3.plan({{5120, 17408}, rows, LinearPhase::Decode, LinearEpilogue::Residual},
                                    projection(5120, 17408, 1));
    const LinearScratchSize size = plan.scratchSize();
    require(plan.partialSums() == 8 && size.input == uint64_t{rows} * 17408 * 2 &&
                size.sums == uint64_t{rows} / 8 * (17408 * 3 / 4) * 4 &&
                size.partials == uint64_t{8} * rows * 5120 * 4 && size.counters == 80 * 4 &&
                plan.gateScratchBytes() == 0,
            "Apple9 GGUF register scratch");
  }
  const LinearPlan head = m3.plan({{248320, 5120}, 8, LinearPhase::Decode, LinearEpilogue::None},
                                  projection(248320, 5120, 1));
  require(head.scratchSize().partials == 4 && head.scratchSize().counters == 4,
          "Apple9 GGUF register scratch without splits binds placeholders");
  const LinearPlan registerGateUp = m3.plan({{17408, 5120}, 16, LinearPhase::Decode, LinearEpilogue::GateUp},
                                            projection(17408, 5120, 1));
  require(registerGateUp.gateScratchBytes() == uint64_t{16} * 17408 * 2,
          "Apple9 GGUF gate/up runs a gate pass into the gate scratch");
  require(m3.plan({{5120, 17408}, 100, LinearPhase::Prefill, LinearEpilogue::Residual},
                  projection(5120, 17408, 1)).configuration().tile == LinearTile::GgufStaged,
          "Apple9 GGUF prefill stages");
  LinearWorkload registerDown = down;
  registerDown.quant = QuantFamily::Gguf;
  require(m3.decodeScratchSize(registerDown).partials == uint64_t{8} * 8 * 5120 * 4,
          "Apple9 GGUF decode scratch bound");
  for (const LinearConfig config : {LinearConfig{LinearTile::GgufSimdgroup, 80, LinearSimdgroups::Four, 3},
                                    LinearConfig{LinearTile::GgufSimdgroup, 80, LinearSimdgroups::Four, 16},
                                    LinearConfig{LinearTile::GgufSimdgroup, 40, LinearSimdgroups::Four, 8},
                                    LinearConfig{LinearTile::GgufSimdgroup, 80, LinearSimdgroups::Two, 8}})
    rejects([&] { (void)Q4Linear::plan(registerDown, config); });
  // Split boundaries fall on 256-input units, and prefill has no register tile.
  rejects([&] {
    (void)Q4Linear::plan({{5120, 512}, 8, LinearPhase::Decode, LinearEpilogue::None, QuantFamily::Gguf},
                         {LinearTile::GgufSimdgroup, 80, LinearSimdgroups::Four, 4});
  });
  rejects([&] {
    (void)Q4Linear::plan({{5120, 17408}, 128, LinearPhase::Prefill, LinearEpilogue::None, QuantFamily::Gguf},
                         {LinearTile::GgufSimdgroup, 0, LinearSimdgroups::Four, 1});
  });
}

void scalingContracts() {
  for (uint32_t family : {9U, 10U, 11U}) {
    for (uint32_t index = 0; index <= 129; ++index) {
      const uint32_t reported = index == 129 ? 4096 : index;
      const uint32_t cores = reported ? reported : 32U;
      DeviceCapabilities device;
      device.appleGpuFamily = family;
      device.gpuCoreCount = reported;
      Q4Linear linear(device);
      for (uint32_t n : {256U, 5120U, 131072U}) {
        for (uint32_t k : {256U, 768U, 1024U, 4096U, 5120U, 17408U}) {
          for (uint32_t rows : {8U, 16U, 24U, 32U}) {
            for (auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                                 LinearEpilogue::GateUp}) {
              const LinearWorkload w{{n, k}, rows, LinearPhase::Decode, epilogue};
              const auto baseline = linear.plan(w);
              const auto candidates = linear.candidates(w);
              require(candidates.front().configuration() == baseline.configuration() &&
                          candidates.size() <= Q4Linear::kMaximumCandidates,
                      "core scaling lost or displaced the baseline");
              for (size_t i = 0; i < candidates.size(); ++i) {
                const auto &plan = candidates[i];
                require(plan.configuration().groups > 0 &&
                            plan.configuration().groups <= n / plan.tileColumns(),
                        "core scaling produced an invalid grid");
                for (size_t j = 0; j < i; ++j)
                  require(plan.configuration() != candidates[j].configuration(),
                          "core scaling produced duplicate candidates");
                const std::array choice{LinearChoice{w, plan.configuration()}};
                linear.setChoices(choice);
                const auto scratch = linear.decodeScratchSize(w);
                for (const auto required : {baseline.scratchSize(), plan.scratchSize()})
                  require(scratch.input >= required.input && scratch.sums >= required.sums &&
                              scratch.partials >= required.partials && scratch.counters >= required.counters,
                          "installed candidate exceeds admitted Q4 workspace");
              }
              linear.setChoices({});
              // N128 is available for every non-gated decode workload. Its
              // candidate waves must follow the device, including tiny GPUs.
              if (epilogue != LinearEpilogue::GateUp) {
                for (uint32_t wave : {2U, 3U, 4U}) {
                  const LinearConfig wanted{LinearTile::N128, std::min(n / 128, cores * wave)};
                  require(std::any_of(candidates.begin(), candidates.end(), [&](const auto &p) {
                    return p.configuration() == wanted;
                  }), "candidate waves do not scale with GPU core count");
                }
              }
            }
          }
        }
      }
    }
  }
}

metal::MetalBuffer allocate(metal::MetalBackend &backend, uint64_t bytes) {
  if (!bytes) return {};
  auto buffer = backend.allocateBuffer(bytes);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

uint32_t mix(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352d;
  value ^= value >> 15;
  value *= 0x846ca68b;
  return value ^ (value >> 16);
}

std::array<uint64_t, 3> projectionFingerprint(const Q4Projection &projection) {
  std::array<uint64_t, 3> result{};
  const std::array buffers{projection.weights, projection.scales, projection.biases};
  for (size_t slot = 0; slot < buffers.size(); ++slot) {
    const auto *bytes = static_cast<const uint8_t *>(buffers[slot].contents());
    uint64_t hash = 14695981039346656037ULL;
    for (uint64_t i = 0; i < buffers[slot].sizeBytes(); ++i)
      hash = (hash ^ bytes[i]) * 1099511628211ULL;
    result[slot] = hash;
  }
  return result;
}

Q4Projection projection(metal::MetalBackend &backend, LinearMatrix matrix, uint32_t seed) {
  const uint64_t parameters = uint64_t{matrix.outputSize} * (matrix.inputSize / 64);
  Q4Projection p{allocate(backend, parameters * 32), allocate(backend, parameters * 2),
                  allocate(backend, parameters * 2), matrix.outputSize, matrix.inputSize};
  auto *weights = static_cast<uint8_t *>(p.weights.contents());
  auto *scales = static_cast<uint16_t *>(p.scales.contents());
  auto *biases = static_cast<uint16_t *>(p.biases.contents());
  for (uint64_t i = 0; i < parameters * 32; ++i) weights[i] = mix(uint32_t(i) + seed);
  for (uint64_t i = 0; i < parameters; ++i) {
    const float scale = 0.004f + float(mix(uint32_t(i) + seed) % 17) * 0.0001f;
    scales[i] = bf16(scale);
    biases[i] = bf16(-7.5f * scale);
  }
  return p;
}

// Scalar reference uses the actual StorageN=256 bytes and FP32 per-group
// affine accumulation, including the BF16 boundary before each epilogue.
float affineReference(const Q4Projection &p, const uint16_t *input,
                        uint32_t row, uint32_t column) {
  const auto *weights = static_cast<const uint8_t *>(p.weights.contents());
  const auto *scales = static_cast<const uint16_t *>(p.scales.contents());
  const auto *biases = static_cast<const uint16_t *>(p.biases.contents());
  const uint32_t groups = p.inputSize / 64;
  float result = 0;
  for (uint32_t group = 0; group < groups; ++group) {
    const uint64_t parameter = (uint64_t{column / 256} * groups + group) * 256 + column % 256;
    float partial = 0, sum = 0;
    for (uint32_t k = 0; k < 64; ++k) {
      const uint8_t byte = weights[parameter * 32 + k / 2];
      const uint32_t quantized = (byte >> ((k & 1) * 4)) & 15;
      const float x = fp32(input[uint64_t{row} * p.inputSize + group * 64 + k]);
      sum += x;
      partial += x * quantized;
    }
    result += partial * fp32(scales[parameter]) + sum * fp32(biases[parameter]);
  }
  return fp32(bf16(result));
}

void checkReference(const Q4Projection &p, const Q4Projection &gate,
                      LinearWorkload workload, const LinearBuffers &buffers,
                      const uint16_t *savedResidual = nullptr, bool split = false, float slack = 0) {
  const auto *input = static_cast<const uint16_t *>(buffers.input.contents());
  const auto *output = static_cast<const uint16_t *>(buffers.output.contents());
  const auto *residual = savedResidual ? savedResidual
      : static_cast<const uint16_t *>(buffers.residual.contents());
  const auto *gateValues = static_cast<const uint16_t *>(buffers.gateScratch.contents());
  for (const uint32_t row : {0U, workload.rows / 2, workload.rows - 1}) {
    for (const uint32_t column : {0U, 127U, 128U, 255U,
                                  p.outputSize / 2, p.outputSize - 1}) {
      const float projection = affineReference(p, input, row, column);
      float expected = projection;
      float residualValue = 0, gateValue = 0;
      const uint64_t index = uint64_t{row} * p.outputSize + column;
      if (workload.epilogue == LinearEpilogue::Residual) {
        residualValue = fp32(residual[index]);
        expected += residualValue;
      }
      if (workload.epilogue == LinearEpilogue::GateUp ||
          workload.epilogue == LinearEpilogue::UpWithGate) {
        gateValue = workload.epilogue == LinearEpilogue::GateUp
            ? affineReference(gate, input, row, column) : fp32(gateValues[index]);
        expected *= gateValue / (1 + std::exp(-gateValue));
      }
      expected = fp32(bf16(expected));
      // The oracle accumulates in the sequential kernel's order. A split tile
      // reassociates that sum, so it is held to the derived bf16 bound
      // (tuning/LinearNumerics.hpp) on top of the oracle's own margin.
      float tolerance = 0.004f;
      if (split)
        tolerance += tuning::splitTolerance(workload.epilogue,
            {expected, residualValue, gateValue, projection}, slack);
      const float actual = fp32(output[index]);
      if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::cerr << "reference row=" << row << " col=" << column
                  << " actual=" << actual << " expected=" << expected << '\n';
        throw std::runtime_error("Linear failed independent packed-Q4 oracle");
      }
    }
  }
  if (workload.epilogue == LinearEpilogue::UpWithGate) {
    const auto *sums = static_cast<const float *>(buffers.downSums.contents());
    const uint32_t quantGroups = p.outputSize / 64;
    for (uint32_t row = 0; row < workload.rows; ++row) {
      for (uint32_t group = 0; group < quantGroups; ++group) {
        double expected = 0;
        for (uint32_t k = 0; k < 64; ++k)
          expected += fp32(output[uint64_t{row} * p.outputSize + group * 64 + k]);
        const uint64_t index = uint64_t{row / 32} * 32 * quantGroups + group * 32 + row % 32;
        require(std::abs(sums[index] - expected) <= 1e-6 * std::max(1.0, std::abs(expected)),
                "fused prefill output sums have wrong layout/value");
      }
    }
  }
}

void bufferContracts(metal::MetalBackend &backend, Q4Linear &linear,
                        LinearBuffers buffers, const Q4Projection &p,
                        const Q4Projection &gate, const LinearPlan &plan) {
  const bool gateUp = plan.workload().epilogue == LinearEpilogue::GateUp;
  const auto add = [&](metal::CommandGraph &graph, LinearBuffers b,
                        const Q4Projection &projection, const Q4Projection *g) {
    linear.add(graph, b, projection, plan, g);
  };
  const std::array members{&LinearBuffers::input, &LinearBuffers::output,
                           &LinearBuffers::sums, &LinearBuffers::residual,
                           &LinearBuffers::gateScratch, &LinearBuffers::downSums};
  for (auto member : members) {
    if (!(buffers.*member)) continue;
    auto shortBuffers = buffers;
    shortBuffers.*member = backend.view(buffers.*member, 0, (buffers.*member).sizeBytes() - 1);
    metal::CommandGraph graph;
    rejects([&] { add(graph, shortBuffers, p, gateUp ? &gate : nullptr); });
    require(graph.empty(), "invalid Linear buffers partially encoded a graph");
  }
  if (plan.usesSimdgroup()) {
    for (auto member : {&LinearScratch::input, &LinearScratch::sums,
                        &LinearScratch::partials, &LinearScratch::counters}) {
      auto shortBuffers = buffers;
      shortBuffers.scratch.*member = backend.view(buffers.scratch.*member, 0,
                                                  (buffers.scratch.*member).sizeBytes()-1);
      metal::CommandGraph graph;
      rejects([&] { add(graph, shortBuffers, p, gateUp ? &gate : nullptr); });
      require(graph.empty(), "invalid simdgroup workspace partially encoded a graph");
    }
  }
  for (auto member : {&Q4Projection::weights, &Q4Projection::scales, &Q4Projection::biases}) {
    auto shortProjection = p;
    shortProjection.*member = backend.view(p.*member, 0, (p.*member).sizeBytes() - 1);
    metal::CommandGraph graph;
    rejects([&] { add(graph, buffers, shortProjection, gateUp ? &gate : nullptr); });
    require(graph.empty(), "invalid projection partially encoded a graph");
  }
  auto mismatch = p;
  mismatch.inputSize += 256;
  metal::CommandGraph graph;
  rejects([&] { add(graph, buffers, mismatch, gateUp ? &gate : nullptr); });
  rejects([&] { add(graph, buffers, p, gateUp ? nullptr : &gate); });
  require(graph.empty(), "invalid Linear gate/projection partially encoded graph");
  if (plan.workload().phase == LinearPhase::Prefill) {
    const auto workload = plan.workload();
    rejects([&] {
      linear.addPrefillSums(graph,
          backend.view(buffers.input, 0, buffers.input.sizeBytes() - 1), buffers.sums,
          workload.matrix, workload.rows);
    });
    rejects([&] {
      linear.addPrefillSums(graph, buffers.input,
          backend.view(buffers.sums, 0, buffers.sums.sizeBytes() - 1),
          workload.matrix, workload.rows);
    });
    for (const LinearMatrix matrix : {LinearMatrix{0, workload.matrix.inputSize},
           LinearMatrix{128, workload.matrix.inputSize},
           LinearMatrix{workload.matrix.outputSize, 0},
           LinearMatrix{workload.matrix.outputSize, 63}})
      rejects([&] { linear.addPrefillSums(graph, buffers.input, buffers.sums,
                                         matrix, workload.rows); });
    for (uint32_t rows : {0U, SPLASH_PREFILL_TOKEN_BUDGET + 1U})
      rejects([&] { linear.addPrefillSums(graph, buffers.input, buffers.sums,
                                         workload.matrix, rows); });
    require(graph.empty(), "invalid prefill sums input partially encoded graph");
  }
}

void numericalCase(metal::MetalBackend &backend, Q4Linear &linear,
                      const Q4Projection &p, const Q4Projection &gate,
                      LinearWorkload workload, bool inPlaceResidual = false) {
  require(!inPlaceResidual || workload.epilogue == LinearEpilogue::Residual,
          "in-place residual fixture requires residual epilogue");
  const auto candidates = linear.candidates(workload);
  const uint32_t storageRows = candidates[0].storageRows();
  auto input = allocate(backend, uint64_t{storageRows} * p.inputSize * 2);
  auto *inputValues = static_cast<uint16_t *>(input.contents());
  for (uint64_t i = 0; i < uint64_t{workload.rows} * p.inputSize; ++i)
    inputValues[i] = bf16(float(int(mix(uint32_t(i) + 1949) % 257) - 128) / 257.0f);
  // Sequential candidates share their output bytes; split-K candidates are
  // held to the derived bound against them once every candidate has run.
  std::vector<uint16_t> baseline;
  struct SplitOutput final {
    std::vector<uint16_t> output, residual;
    std::string_view pipeline;
    bool simdgroup = false;
  };
  std::vector<SplitOutput> splitOutputs;
  for (const auto &plan : candidates) {
    const uint64_t outputBytes = uint64_t{storageRows} * p.outputSize * 2;
    const uint64_t guardBytes = uint64_t{8} * p.outputSize * 2;
    auto outputBacking = allocate(backend, outputBytes + guardBytes);
    std::memset(static_cast<uint8_t *>(outputBacking.contents()) + outputBytes, 0x5a, guardBytes);
    const uint64_t gateBytes = plan.gateScratchBytes();
    auto gateBacking = allocate(backend, gateBytes ? gateBytes + guardBytes : 0);
    if (gateBytes)
      std::memset(static_cast<uint8_t *>(gateBacking.contents()) + gateBytes, 0x5a, guardBytes);
    LinearBuffers b{input, backend.view(outputBacking, 0, outputBytes),
                     allocate(backend, plan.sumsBytes()), {},
                     gateBytes ? backend.view(gateBacking, 0, gateBytes) : metal::MetalBuffer{},
                     allocate(backend, plan.downSumsBytes())};
    const auto scratch = plan.scratchSize();
    b.scratch = {allocate(backend, scratch.input), allocate(backend, scratch.sums),
                 allocate(backend, scratch.partials), allocate(backend, scratch.counters)};
    if (workload.epilogue == LinearEpilogue::Residual) {
      b.residual = inPlaceResidual ? b.output : allocate(backend, b.output.sizeBytes());
      auto *residual = static_cast<uint16_t *>(b.residual.contents());
      for (uint64_t i = 0; i < uint64_t{workload.rows} * p.outputSize; ++i)
        residual[i] = bf16(float(int(mix(uint32_t(i) + 7919) % 257) - 128) / 257.0f);
    }
    bufferContracts(backend, linear, b, p, gate, plan);
    metal::CommandGraph graph;
    if (workload.phase == LinearPhase::Prefill)
      linear.addPrefillSums(graph, input, b.sums, workload.matrix, workload.rows);
    if (workload.epilogue == LinearEpilogue::UpWithGate) {
      const auto gatePlan = linear.plan({workload.matrix, workload.rows, LinearPhase::Prefill,
                                         LinearEpilogue::None});
      linear.add(graph, {input, b.gateScratch, b.sums, {}, {}, {}}, gate, gatePlan);
    }
    const size_t prepasses = graph.dispatches().size();
    Q4DispatchStats stats;
    linear.add(graph, b, p, plan, workload.epilogue == LinearEpilogue::GateUp ? &gate : nullptr, &stats);
    const auto &last = graph.dispatches().back();
    require(last.pipelineName == (plan.secondPipeline().empty() ? plan.pipeline() : plan.secondPipeline()),
            "production dispatch differs from Linear plan");
    for (const auto &dispatch : graph.dispatches().subspan(prepasses))
      require(dispatch.threadsPerThreadgroup.x == plan.threadsPerThreadgroup() &&
                  dispatch.threadsPerThreadgroup.y == 1 && dispatch.threadsPerThreadgroup.z == 1,
              "production dispatch threads differ from Linear plan scope");
    if (workload.phase == LinearPhase::Decode) {
      const uint32_t lanes = workload.rows / 8;
      const uint32_t dispatches = plan.usesSimdgroup() ? 2 : plan.secondPipeline().empty() ? 1 : 2;
      require(last.threadgroups.x == plan.configuration().groups &&
                  graph.dispatches().size() == dispatches,
              "Linear decode plan/graph geometry mismatch");
      // These counters describe projection fusion, excluding input preparation.
      const uint32_t projections = plan.secondPipeline().empty() ? 1 : 2;
      require(stats.fusedSourceOperations == (lanes == 1 ? 0 : lanes * projections) &&
                  stats.m16Dispatches == (lanes == 2 ? projections : 0) &&
                  stats.m24Dispatches == (lanes == 3 ? projections : 0) &&
                  stats.m32Dispatches == (lanes == 4 ? projections : 0),
              "Linear dispatch statistics changed");
    } else {
      require(last.threadgroups.x == storageRows / 32 &&
                  last.threadgroups.y == p.outputSize / plan.tileColumns() &&
                  graph.dispatches().size() == prepasses + 1,
              "Linear prefill plan/graph geometry mismatch");
    }
    const auto snapshot = [](const metal::MetalBuffer &buffer) {
      if (!buffer) return std::vector<uint8_t>{};
      const auto *begin = static_cast<const uint8_t *>(buffer.contents());
      return std::vector<uint8_t>{begin, begin + buffer.sizeBytes()};
    };
    const auto immutableInput = snapshot(b.input);
    std::vector<uint16_t> immutableResidual;
    if (b.residual) {
      const auto *values = static_cast<const uint16_t *>(b.residual.contents());
      immutableResidual.assign(values, values + b.residual.sizeBytes() / sizeof(uint16_t));
    }
    (void)backend.submitCommand(graph.dispatches());
    if (workload.phase == LinearPhase::Decode && workload.rows == 24) {
      const auto firstOutput = snapshot(b.output);
      if (inPlaceResidual)
        std::memcpy(b.residual.contents(), immutableResidual.data(),
                    immutableResidual.size() * sizeof(uint16_t));
      (void)backend.submitCommand(graph.dispatches());
      require(std::memcmp(firstOutput.data(), b.output.contents(), firstOutput.size()) == 0,
              "repeated M24 dispatch changed its output bytes");
    }
    const auto checkGuard = [&](const metal::MetalBuffer &backing, uint64_t payload,
                                 const char *role) {
      const auto *guard = static_cast<const uint8_t *>(backing.contents()) + payload;
      for (uint64_t i = 0; i < guardBytes; ++i) {
        if (guard[i] != 0x5a) {
          std::cerr << role << " guard overwritten matrix=" << p.outputSize << 'x' << p.inputSize
                    << " rows=" << workload.rows << " epilogue=" << uint32_t(workload.epilogue)
                    << " pipeline=" << plan.pipeline() << " first_extra_byte=" << i << '\n';
          throw std::runtime_error("Linear wrote beyond its exact workspace/output view");
        }
      }
    };
    checkGuard(outputBacking, outputBytes, "output");
    if (gateBytes) checkGuard(gateBacking, gateBytes, "gate scratch");
    require(std::memcmp(immutableInput.data(), b.input.contents(), immutableInput.size()) == 0,
            "Linear modified its immutable input");
    if (!inPlaceResidual && !immutableResidual.empty())
      require(std::memcmp(immutableResidual.data(), b.residual.contents(),
                          immutableResidual.size() * sizeof(uint16_t)) == 0,
              "Linear modified its immutable residual");
    try {
      checkReference(p, gate, workload, b,
                     inPlaceResidual ? immutableResidual.data() : nullptr,
                     plan.partialSums() > 1 || plan.usesSimdgroup(),
                     plan.usesSimdgroup() ? std::max(tuning::simdgroupSlack(workload, b.input, p),
                         tuning::simdgroupSlack(workload, b.input, gate)) : 0);
    } catch (const std::exception &) {
      std::cerr << "matrix=" << p.outputSize << 'x' << p.inputSize
                << " rows=" << workload.rows << " phase=" << uint32_t(workload.phase)
                << " epilogue=" << uint32_t(workload.epilogue)
                << " tile=" << uint32_t(plan.configuration().tile)
                << " groups=" << plan.configuration().groups
                << " simdgroups=" << uint32_t(plan.configuration().simdgroups)
                << " in_place_residual=" << inPlaceResidual
                << " pipeline=" << plan.pipeline() << '\n';
      throw;
    }
    const auto *output = static_cast<const uint16_t *>(b.output.contents());
    const uint64_t elements = uint64_t{storageRows} * p.outputSize;
    if (plan.partialSums() > 1 || plan.usesSimdgroup()) {
      splitOutputs.push_back({{output, output + elements}, immutableResidual, plan.pipeline(), plan.usesSimdgroup()});
    } else {
      if (baseline.empty()) baseline.assign(output, output + elements);
      require(std::memcmp(baseline.data(), output, elements * 2) == 0,
              "Linear candidate differs from baseline output bytes");
    }
    for (uint64_t i = uint64_t{workload.rows} * p.outputSize; i < elements; ++i)
      require(output[i] == 0, "padded prefill rows were not zero");
  }
  if (splitOutputs.empty()) return;
  require(!baseline.empty(), "split-K candidates have no sequential reference");
  // The gate/up bound needs the exact gate and up projections: the sequential
  // N128 plain plan on both weight sets.
  std::vector<uint16_t> gateReference, upReference;
  if (workload.epilogue == LinearEpilogue::GateUp) {
    const auto plain = Q4Linear::plan({workload.matrix, workload.rows, LinearPhase::Decode,
                                       LinearEpilogue::None},
                                      {LinearTile::N128, p.outputSize / 128});
    auto gateOutput = allocate(backend, uint64_t{storageRows} * p.outputSize * 2);
    auto upOutput = allocate(backend, uint64_t{storageRows} * p.outputSize * 2);
    metal::CommandGraph graph;
    linear.add(graph, {input, gateOutput, {}, {}, {}, {}}, gate, plain);
    linear.add(graph, {input, upOutput, {}, {}, {}, {}}, p, plain);
    (void)backend.submitCommand(graph.dispatches());
    const auto *g = static_cast<const uint16_t *>(gateOutput.contents());
    const auto *u = static_cast<const uint16_t *>(upOutput.contents());
    gateReference.assign(g, g + baseline.size());
    upReference.assign(u, u + baseline.size());
  }
  float maxAbs = 0;
  for (const uint16_t value : baseline) maxAbs = std::max(maxAbs, std::fabs(fp32(value)));
  const float slack = tuning::reassociationSlack(p.inputSize, maxAbs);
  const float operandSlack = std::max(tuning::simdgroupSlack(workload, input, p),
                                      tuning::simdgroupSlack(workload, input, gate));
  for (const auto &split : splitOutputs) {
    const float toleranceSlack = slack + (split.simdgroup ? operandSlack : 0);
    for (uint64_t i = 0; i < uint64_t{workload.rows} * p.outputSize; ++i) {
      tuning::SplitReference reference{fp32(baseline[i])};
      if (workload.epilogue == LinearEpilogue::Residual) reference.residual = fp32(split.residual[i]);
      if (workload.epilogue == LinearEpilogue::GateUp) {
        reference.gate = fp32(gateReference[i]);
        reference.up = fp32(upReference[i]);
      }
      if (!tuning::withinSplitTolerance(fp32(split.output[i]), workload.epilogue, reference, toleranceSlack)) {
        std::cerr << "split element=" << i << " actual=" << fp32(split.output[i])
                  << " reference=" << reference.value << " residual=" << reference.residual
                  << " gate=" << reference.gate << " up=" << reference.up
                  << " bound=" << tuning::splitTolerance(workload.epilogue, reference, toleranceSlack)
                  << " matrix=" << p.outputSize << 'x' << p.inputSize
                  << " epilogue=" << uint32_t(workload.epilogue)
                  << " pipeline=" << split.pipeline << '\n';
        throw std::runtime_error("split-K candidate exceeds its bf16 bound against the sequential tiles");
      }
    }
  }
}

void pipelineCapabilities(const char *metallib, const DeviceCapabilities &capabilities) {
  Q4Linear linear(capabilities);
  std::map<std::string, uint32_t> names{{"prefill_linear_q4_sums32", 256}};
  const auto collect = [&](LinearWorkload workload) {
    for (const auto &plan : linear.candidates(workload)) {
      for (auto name : {plan.pipeline(), plan.secondPipeline()}) {
        if (name.empty()) continue;
        const auto [found, inserted] = names.emplace(name, plan.threadsPerThreadgroup());
        require(inserted || found->second == plan.threadsPerThreadgroup(),
                "one Linear pipeline was assigned incompatible execution scopes");
      }
    }
  };
  for (uint32_t lanes = 1; lanes <= 4; ++lanes)
    for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                               LinearEpilogue::GateUp})
      collect({{16640, 5120}, lanes * 8, LinearPhase::Decode, epilogue});
  for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                             LinearEpilogue::UpWithGate})
    collect({{16640, 5120}, 33, LinearPhase::Prefill, epilogue});
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithURL:
      [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib]] error:&error];
  require(library != nil, "could not load Linear library for resource inspection");
  uint64_t largestStaticMemory = 0;
  uint64_t smallestThreadLimit = std::numeric_limits<uint64_t>::max();
  for (const auto &[name, threads] : names) {
    id<MTLFunction> function = [library newFunctionWithName:
        [NSString stringWithUTF8String:name.c_str()]];
    require(function != nil, "missing precompiled Linear candidate");
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    require(pipeline != nil, "Linear candidate pipeline could not be created");
    largestStaticMemory = std::max(largestStaticMemory, uint64_t(pipeline.staticThreadgroupMemoryLength));
    smallestThreadLimit = std::min(smallestThreadLimit, uint64_t(pipeline.maxTotalThreadsPerThreadgroup));
    require(pipeline.staticThreadgroupMemoryLength <= capabilities.maxThreadgroupMemoryBytes &&
                pipeline.maxTotalThreadsPerThreadgroup >= threads && pipeline.threadExecutionWidth == 32,
            "Linear candidate exceeds current device resources");
  }
  std::cout << "Linear pipelines=" << names.size() << " maximum_static_tg_bytes="
            << largestStaticMemory << " minimum_thread_limit=" << smallestThreadLimit
            << " apple_family=" << capabilities.appleGpuFamily << " PASS\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: linear-plan <production.metallib|--cpu>");
    baselinePlans();
    affinePolicyIdentity();
    ggufPlans();
    narrowM24BoundaryPlans();
    scalingContracts();
    // Apple9 at the assumed core count reaches the expanded split set;
    // Apple10 exercises the MPP-only candidate bound.
    size_t widestCandidates = 0;
    planContracts(9, 0, widestCandidates);
    planContracts(10, 16, widestCandidates);
    require(widestCandidates == Q4Linear::kMaximumCandidates,
            "no covered workload reaches the Linear candidate bound");
    if (std::string_view(argv[1]) == "--cpu") {
      std::cout << "Linear CPU plans: PASS\n";
      return 0;
    }
    metal::MetalBackend backend(argv[1]);
    Q4Linear linear(backend.capabilities());
    // Instrumented shader builds can report additional validation storage;
    // inspect production requirements in the ordinary/API-validation run.
    if (!std::getenv("MTL_SHADER_VALIDATION"))
      pipelineCapabilities(argv[1], backend.capabilities());
    for (const LinearMatrix matrix : {LinearMatrix{512, 256}, LinearMatrix{768, 768},
                                      LinearMatrix{16640, 5120}, LinearMatrix{12544, 2048},
                                      LinearMatrix{5120, 17408},
                                      LinearMatrix{256, 64}, LinearMatrix{512, 320}}) {
      const auto p = projection(backend, matrix, 31);
      const auto gate = projection(backend, matrix, 157);
      const auto immutableProjection = projectionFingerprint(p);
      const auto immutableGateProjection = projectionFingerprint(gate);
      if (matrix.inputSize % 256 == 0)
        for (uint32_t lanes = 1; lanes <= 4; ++lanes)
          for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                                     LinearEpilogue::GateUp}) {
            numericalCase(backend, linear, p, gate,
                          {matrix, lanes * 8, LinearPhase::Decode, epilogue});
            if (epilogue == LinearEpilogue::Residual)
              numericalCase(backend, linear, p, gate,
                            {matrix, lanes * 8, LinearPhase::Decode, epilogue}, true);
          }
      for (const uint32_t rows : {1U, 33U, 2048U})
        for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                                   LinearEpilogue::UpWithGate})
          numericalCase(backend, linear, p, gate,
                        {matrix, rows, LinearPhase::Prefill, epilogue});
      require(projectionFingerprint(p) == immutableProjection &&
                  projectionFingerprint(gate) == immutableGateProjection,
              "Linear changed immutable Q4 weights or quantization metadata");
      std::cout << "Linear N=" << matrix.outputSize << " K=" << matrix.inputSize
                << " all epilogues/candidates/row cases PASS\n";
    }
    std::cout << "Linear plans and candidates: PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "Linear plans: FAIL: " << error.what() << '\n';
    return 1;
  }
}

#include "AffineQ4Fixture.hpp"
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "ops/GDN.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"
#include "ops/PagedAttention.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "metal/abi/Linear.h"
#include "metal/abi/QuantFormat.h"
#include "tuning/LinearNumerics.hpp"
#include "tuning/LinearTuning.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

using namespace splash;
using namespace splash::ops;
using test::mix;

using splash::test::rejects;
using splash::test::require;
using splash::test::requireExtent;

DeviceCapabilities simulatedDevice(uint32_t family, uint32_t cores) {
  DeviceCapabilities device;
  device.appleGpuFamily = family;
  device.gpuCoreCount = cores;
  return device;
}
Linear gpu(uint32_t family, uint32_t cores) { return Linear(simulatedDevice(family, cores)); }

// Throws naming the rule a plan broke and the plan.
[[noreturn]] void broke(const char *rule, uint32_t family, uint32_t cores, LinearWorkload w) {
  throw std::runtime_error(std::string(rule) + ": family " + std::to_string(family) + ", " +
                           std::to_string(cores) + " cores, N " + std::to_string(w.matrix.outputSize) +
                           ", K " + std::to_string(w.matrix.inputSize) + ", " + std::to_string(w.rows) +
                           (w.phase == LinearPhase::Decode ? " decode" : " prefill") + " rows, epilogue " +
                           std::to_string(unsigned(w.epilogue)));
}

// The affine policy across GPU families, core counts and workload tile counts,
// stated independently of the operator: expectedGroups restates the group
// distribution, expectedOneLane the one-lane rule, expectedDecode and
// expectedPrefill the tile rules. The literal anchors below independently
// guard selected policy boundaries and production shapes.

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

// Apple10's K splits of the Split128 tile, the same at every lane count: the
// largest power of two up to eight whose grid of 128-column threadgroups fits
// four per core, every partition keeping one 256-input block.
uint32_t expectedApple10Splits(uint32_t cores, LinearMatrix matrix) {
  uint32_t selected = 1;
  for (const uint32_t split : {2U, 4U, 8U})
    if (uint64_t{matrix.outputSize / 128} * split <= 4ULL * cores && matrix.inputSize / 256 >= split)
      selected = split;
  return selected;
}

// Apple10 one-lane MPP rule: paired N256 from two tiles per core while the
// grid fits one wave of four groups per core, and from eight tiles per core.
std::optional<LinearConfig> expectedOneLane(uint32_t cores,
                                            LinearMatrix matrix, LinearEpilogue epilogue) {
  const uint32_t n = matrix.outputSize;
  const uint32_t tiles256 = n / 256;
  const bool oneWave = tiles256 >= 2 * cores && tiles256 <= 4 * cores;
  if (epilogue == LinearEpilogue::None && (oneWave || tiles256 >= 8 * cores))
    return LinearConfig{LinearTile::Paired256,
                        std::min(tiles256, 4 * cores),
                        LinearSimdgroups::Four};
  return std::nullopt;
}

LinearConfig expectedDecode(uint32_t family, uint32_t cores, LinearMatrix matrix,
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
    return {LinearTile::Q4Register, 0, LinearSimdgroups::Four, selected};
  }
  if (family >= 10)
    if (const uint32_t splits = expectedApple10Splits(cores, matrix); splits > 1)
      return {LinearTile::Split128, 0, LinearSimdgroups::Eight, splits};
  constexpr GroupRule n128{4, 4, 12}, m16{5, 4, 12}, n256{3, 3, 8}, gateUp{3, 3, 8},
      fourSimdgroups{8, 8, 24};
  const uint32_t tiles128 = matrix.outputSize / 128;
  const uint32_t tiles256 = matrix.outputSize / 256;
  // Apple9 reaches here only for wide plain projections of three or four
  // lanes, which keep their one-tile grids.
  const auto groups = [&](uint32_t tiles, GroupRule rule) {
    return family >= 10 ? expectedGroups(tiles, cores, rule) : tiles;
  };
  if (family >= 10 && lanes == 1)
    if (const auto oneLane = expectedOneLane(cores, matrix, epilogue)) return *oneLane;
  if (epilogue == LinearEpilogue::GateUp) return {LinearTile::N256, groups(tiles256, gateUp)};
  if (lanes == 1) return {LinearTile::Paired128, groups(tiles128, n128)};
  if (lanes == 3)
    return {LinearTile::N128, groups(tiles128, fourSimdgroups), LinearSimdgroups::Four};
  if (lanes >= 3 && epilogue == LinearEpilogue::None && tiles256 >= 2 * cores)
    return {LinearTile::N256, groups(tiles256, n256)};
  return {LinearTile::N128, groups(tiles128, lanes == 2 ? m16 : n128)};
}

// Apple10 and later, and Apple9 up to the measured 32-core device, prefill
// with the four-simdgroup N128 tile; larger Apple9 GPUs keep the wide-tile
// rule: N256 for the fused up projection and once the N256 grid holds eight
// threadgroups per core.
LinearConfig expectedPrefill(uint32_t family, uint32_t cores, LinearWorkload w) {
  if (family >= 10 || cores <= 32) return {LinearTile::N128, 0, LinearSimdgroups::Four};
  const uint64_t grid = uint64_t{(w.rows + 31) / 32} * (w.matrix.outputSize / 256);
  return {w.epilogue == LinearEpilogue::UpWithGate || grid >= 8ULL * cores ? LinearTile::N256
                                                                           : LinearTile::N128, 0};
}

// The rows suffix of a batched decode kernel.
std::string rowsSuffix(uint32_t lanes) { return lanes > 1 ? "_m" + std::to_string(lanes * 8) : ""; }

std::string expectedPipeline(LinearConfig expected, uint32_t lanes, LinearEpilogue epilogue) {
  if (expected.tile == LinearTile::Q4Register)
    return epilogue == LinearEpilogue::GateUp ? "decode_linear_q4_sg_gate_up" :
        epilogue == LinearEpilogue::Residual ? "decode_linear_q4_sg_residual" : "decode_linear_q4_sg";
  // Gate/up runs the plain split kernel as its gate pass.
  if (expected.tile == LinearTile::Split128)
    return std::string("decode_linear_q4_n128_split") +
        (epilogue == LinearEpilogue::Residual ? "_residual" : "") + rowsSuffix(lanes);
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

// Split128 at every lane count and the N256 gate/up tile at three and four
// lanes run gate/up as the gate projection and then the up projection with the
// SiLU product.
std::string expectedSecondPipeline(LinearConfig expected, uint32_t lanes, LinearEpilogue epilogue) {
  if (epilogue != LinearEpilogue::GateUp || expected.tile == LinearTile::Q4Register) return {};
  if (expected.tile == LinearTile::Split128) return "decode_linear_q4_n128_split_up_silu" + rowsSuffix(lanes);
  if (lanes < 3) return {};
  return "decode_linear_q4_n256_up_silu_m" + std::to_string(lanes * 8);
}

std::string expectedPrefillPipeline(LinearConfig expected, LinearEpilogue epilogue) {
  std::string name = expected.tile == LinearTile::N256 ? "prefill_linear_q4_n256" : "prefill_linear_q4_n128";
  if (epilogue == LinearEpilogue::UpWithGate) name += "_up_silu_sums";
  if (epilogue == LinearEpilogue::Residual) name += "_residual";
  if (expected.simdgroups == LinearSimdgroups::Four) name += "_sg4";
  return name;
}

// The Q4 register tile reads the Table64 activation table its producer writes
// (tableBytes and tableSumsBytes) and, split over K, reduces two fp32 fragment
// streams per partition, row and column with one completion counter per lane
// and column tile; one partition needs neither. Split128
// reduces one fp32 partial per partition, row and column with one counter per
// column tile, which covers every lane. Every other affine tile reads the
// plain rows and binds no scratch.
constexpr uint64_t kFragmentStreams = 2;
LinearScratchSize expectedScratch(LinearConfig expected, LinearWorkload w, uint32_t tileColumns) {
  const auto [n, k] = w.matrix;
  if (expected.tile == LinearTile::Split128)
    return {0, 0, uint64_t{expected.splits} * w.rows * n * sizeof(float), uint64_t{n / tileColumns} * sizeof(uint32_t)};
  if (expected.tile != LinearTile::Q4Register) return {};
  const uint64_t lanes = w.rows / 8;
  const bool split = expected.splits > 1;
  return {tableBytes(k, w.rows), tableSumsBytes(LinearInput::Table64, k, w.rows),
          split ? expected.splits * kFragmentStreams * w.rows * n * sizeof(float) : 0,
          split ? lanes * (n / tileColumns) * sizeof(uint32_t) : 0};
}

bool sameScratch(LinearScratchSize a, LinearScratchSize b) {
  return a.input == b.input && a.sums == b.sums && a.partials == b.partials && a.counters == b.counters;
}

// Every stated decode rule for the plan `linear` makes of `w`, on a GPU that
// reports `reportedCores` (zero: unknown, planned as 32).
void checkAffineDecode(const Linear &linear, uint32_t family, uint32_t reportedCores, LinearWorkload w) {
  const uint32_t lanes = w.rows / 8;
  const LinearPlan plan = linear.plan(w);
  const LinearConfig expected = expectedDecode(family, reportedCores ? reportedCores : 32U, w.matrix, lanes, w.epilogue);
  const auto rule = [&](bool holds, const char *name) { if (!holds) broke(name, family, reportedCores, w); };
  rule(plan.configuration() == expected, "affine decode configuration differs from its stated rules");
  rule(plan.threadsPerThreadgroup() == static_cast<uint32_t>(expected.simdgroups) * 32,
       "affine decode scope differs from its configuration");
  rule(plan.pipeline() == expectedPipeline(expected, lanes, w.epilogue),
       "affine decode pipeline differs from its configuration");
  rule(plan.secondPipeline() == expectedSecondPipeline(expected, lanes, w.epilogue),
       "gate/up dispatch decomposition changed");
  const bool q4Register = expected.tile == LinearTile::Q4Register;
  const uint32_t columns = q4Register ? (w.epilogue == LinearEpilogue::GateUp ? 32U : 64U)
      : expected.tile == LinearTile::N256 || expected.tile == LinearTile::Paired256 ? 256U : 128U;
  rule(plan.tileColumns() == columns &&
           plan.groups() == (expected.groups ? expected.groups : w.matrix.outputSize / columns),
       "affine decode tile geometry differs from its configuration");
  rule(plan.input() == (q4Register ? LinearInput::Table64 : LinearInput::Plain),
       "affine decode input layout differs from its tile");
  rule(sameScratch(plan.scratchSize(), expectedScratch(expected, w, columns)),
       "affine decode scratch differs from its tile");
}

void checkAffinePrefill(const Linear &linear, uint32_t family, uint32_t reportedCores, LinearWorkload w) {
  const LinearPlan plan = linear.plan(w);
  const LinearConfig expected = expectedPrefill(family, reportedCores ? reportedCores : 32U, w);
  const auto rule = [&](bool holds, const char *name) { if (!holds) broke(name, family, reportedCores, w); };
  rule(plan.configuration() == expected, "affine prefill configuration differs from its stated rule");
  rule(plan.threadsPerThreadgroup() == static_cast<uint32_t>(expected.simdgroups) * 32,
       "affine prefill cooperative execution scope changed");
  rule(plan.pipeline() == expectedPrefillPipeline(expected, w.epilogue) && plan.secondPipeline().empty(),
       "affine prefill pipeline differs from its configuration");
  rule(plan.input() == LinearInput::Plain && !plan.scratchSize().bytes(),
       "affine prefill reads more than its plain rows");
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
    ProductionShape{{2048, 6144}, LinearEpilogue::Residual},
    ProductionShape{{2048, 16384}, LinearEpilogue::None},
    ProductionShape{{256, 2048}, LinearEpilogue::None},
    ProductionShape{{5120, 4352}, LinearEpilogue::None},
    ProductionShape{{5120, 4352}, LinearEpilogue::Residual},
    ProductionShape{{6144, 4352}, LinearEpilogue::GateUp},
    ProductionShape{{2048, 768}, LinearEpilogue::None}};

void baselinePlans() {
  for (const uint32_t family : {9U, 10U, 11U}) {
    for (const uint32_t reportedCores : {0U, 8U, 10U, 16U, 18U, 20U, 31U, 32U, 33U, 40U, 80U}) {
      const Linear linear = gpu(family, reportedCores);
      for (const auto &shape : kProductionShapes)
        for (uint32_t lanes = 1; lanes <= 4; ++lanes)
          checkAffineDecode(linear, family, reportedCores,
                            {shape.matrix, lanes * 8, LinearPhase::Decode, shape.epilogue});
      for (const uint32_t hidden : {5120U, 2048U}) {
        const bool large = hidden == 5120;
        const uint32_t intermediate = large ? 17408U : 6144U;
        for (const LinearMatrix matrix :
             {LinearMatrix{6144, hidden}, LinearMatrix{large ? 16640U : 12544U, hidden},
              LinearMatrix{hidden, large ? 6144U : 4096U}, LinearMatrix{intermediate, hidden}})
          for (const uint32_t rows : {1U, 7U, 31U, 32U, 33U, 127U, 2048U})
            for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                                       LinearEpilogue::UpWithGate})
              checkAffinePrefill(linear, family, reportedCores, {matrix, rows, LinearPhase::Prefill, epilogue});
      }
    }
  }
  // Anchors from the measured machines: 16- and 20-core Apple10 GPUs (M5 Pro)
  // and a 40-core Apple9 GPU (M3 Max). Changing a rule must change these
  // knowingly.
  const auto configured = [](uint32_t family, uint32_t cores, LinearWorkload workload) {
    return gpu(family, cores).plan(workload).configuration();
  };
  const LinearWorkload gateUp{{17408, 5120}, 8, LinearPhase::Decode, LinearEpilogue::GateUp};
  require(configured(10, 16, gateUp) == LinearConfig{LinearTile::N256, 36} &&
              configured(10, 20, gateUp) == LinearConfig{LinearTile::N256, 48} &&
              configured(9, 40, gateUp) == LinearConfig{LinearTile::Q4Register, 0, LinearSimdgroups::Four, 2} &&
              // Unknown counts use the same intermediate estimate on both families.
              configured(10, 0, gateUp) == configured(10, 32, gateUp) &&
              configured(9, 0, gateUp) == configured(9, 32, gateUp),
          "fused gate/up grid does not follow the balanced two-tile rule");
  // Apple9 matrix K splits cover all decode widths; broad plain projections
  // retain their old multi-lane grids.
  require(configured(9, 16, gateUp) == LinearConfig{LinearTile::Q4Register, 0, LinearSimdgroups::Four, 1} &&
              configured(9, 20, gateUp) == LinearConfig{LinearTile::Q4Register, 0, LinearSimdgroups::Four, 1} &&
              configured(9, 20, {{16640, 5120}, 8}) == LinearConfig{LinearTile::Q4Register, 0, LinearSimdgroups::Four, 2} &&
              configured(9, 16, {{16640, 5120}, 16}) == LinearConfig{LinearTile::Q4Register, 0, LinearSimdgroups::Four, 1} &&
              configured(9, 20, {{16640, 5120}, 32}) == LinearConfig{LinearTile::N256, 65},
          "Apple9 decode grids changed without a measurement");
  // Apple10 splits K where the Split128 grid fits four threadgroups per core
  // and keeps the sequential tiles elsewhere; Apple9 Q4 register tile and
  // wide paired N256 anchors are unchanged.
  const LinearWorkload mixer27{{5120, 6144}, 8, LinearPhase::Decode, LinearEpilogue::Residual};
  const LinearWorkload mixer35{{2048, 4096}, 8, LinearPhase::Decode, LinearEpilogue::Residual};
  const LinearWorkload draftGateUp{{6144, 2048}, 8, LinearPhase::Decode, LinearEpilogue::GateUp};
  const auto split = [](uint32_t splits) {
    return LinearConfig{LinearTile::Split128, 0, LinearSimdgroups::Eight, splits};
  };
  require(configured(10, 20, mixer27) == split(2) &&
              configured(10, 16, mixer27) == LinearConfig{LinearTile::Paired128, 40} &&
              configured(10, 40, mixer27) == split(4) &&
              configured(10, 20, mixer35) == split(4) &&
              configured(10, 16, mixer35) == split(4) &&
              configured(10, 10, mixer35) == split(2) &&
              configured(10, 40, mixer35) == split(8) &&
              configured(10, 10, {{5120, 17408}, 8}) ==
                  LinearConfig{LinearTile::Paired256, 20, LinearSimdgroups::Four} &&
              configured(10, 20, {{1280, 5120}, 8}) == split(8) &&
              configured(10, 16, {{1280, 5120}, 8}) == split(4) &&
              configured(10, 16, {{256, 5120}, 8}) == split(8) &&
              configured(10, 20, {{6144, 5120}, 8}) == LinearConfig{LinearTile::Paired128, 48} &&
              configured(10, 40, {{6144, 5120}, 8}) == split(2) &&
              configured(10, 20, draftGateUp) == LinearConfig{LinearTile::N256, 24} &&
              configured(10, 16, draftGateUp) == LinearConfig{LinearTile::N256, 24} &&
              configured(10, 40, draftGateUp) == split(2),
          "Apple10 K split anchors changed");
  require(configured(9, 40, mixer27) == LinearConfig{LinearTile::Q4Register, 0, LinearSimdgroups::Four, 8} &&
              configured(9, 40, {{16640, 5120}, 8}) == LinearConfig{LinearTile::Q4Register, 0, LinearSimdgroups::Four, 4} &&
              configured(9, 40, {{5120, 17408}, 8}) == LinearConfig{LinearTile::Q4Register, 0, LinearSimdgroups::Four, 8} &&
              configured(9, 40, {{256, 5120}, 8}) == LinearConfig{LinearTile::Q4Register, 0, LinearSimdgroups::Four, 4} &&
              configured(9, 18, mixer27) == LinearConfig{LinearTile::Q4Register, 0, LinearSimdgroups::Four, 4},
          "Apple9 Q4 register tile anchors changed");
  // One-lane paired N256 grids of one wave, 2 to 4 tiles per core, and of 8
  // tiles per core and more.
  const LinearConfig paired256Apple10_20{LinearTile::Paired256, 80, LinearSimdgroups::Four};
  require(configured(10, 20, {{248320, 5120}, 8}) == paired256Apple10_20 &&
              configured(10, 20, {{248320, 2048}, 8}) == paired256Apple10_20 &&
              configured(10, 16, {{248320, 5120}, 8}) ==
                  LinearConfig{LinearTile::Paired256, 64, LinearSimdgroups::Four} &&
              configured(10, 10, {{248320, 2048}, 8}) ==
                  LinearConfig{LinearTile::Paired256, 40, LinearSimdgroups::Four} &&
              configured(9, 40, {{248320, 5120}, 8}) ==
                  LinearConfig{LinearTile::Q4Register, 0, LinearSimdgroups::Four, 1} &&
              configured(9, 80, {{248320, 2048}, 8}) ==
                  LinearConfig{LinearTile::Q4Register, 0, LinearSimdgroups::Four, 1} &&
              configured(10, 20, {{40960, 5120}, 8}) ==
                  LinearConfig{LinearTile::Paired256, 80, LinearSimdgroups::Four} &&
              configured(10, 20, {{40704, 5120}, 8}) == LinearConfig{LinearTile::Paired128, 318} &&
              configured(10, 20, {{20480, 5120}, 8}) == paired256Apple10_20 &&
              configured(10, 20, {{20736, 5120}, 8}) == LinearConfig{LinearTile::Paired128, 80} &&
              configured(10, 20, {{10240, 5120}, 8}) ==
                  LinearConfig{LinearTile::Paired256, 40, LinearSimdgroups::Four} &&
              configured(10, 20, {{9984, 5120}, 8}) == LinearConfig{LinearTile::Paired128, 78} &&
              configured(11, 12, {{6144, 5120}, 8}) ==
                  LinearConfig{LinearTile::Paired256, 24, LinearSimdgroups::Four} &&
              configured(10, 40, {{16640, 5120}, 8}) == LinearConfig{LinearTile::Paired128, 130},
          "one-lane paired N256 anchors changed");
  // K % 1024 != 0 is legal for the Q4 register tile, and Split128 partitions
  // differ by at most one 256-input block (17 into 8 and 9, 3 into 1 and 2).
  require(configured(10, 20, {{5120, 4352}, 8}) == split(2) &&
              configured(9, 40, {{5120, 4352}, 8, LinearPhase::Decode, LinearEpilogue::Residual}) ==
                  LinearConfig{LinearTile::Q4Register, 0, LinearSimdgroups::Four, 4} &&
              configured(9, 40, {{6144, 4352}, 8, LinearPhase::Decode, LinearEpilogue::GateUp}) ==
                  LinearConfig{LinearTile::Q4Register, 0, LinearSimdgroups::Four, 4} &&
              configured(10, 20, {{2048, 768}, 8}) == split(2) &&
              configured(10, 20, {{2048, 4096}, 16}) == split(4) &&
              configured(9, 40, {{5120, 6144}, 24, LinearPhase::Decode, LinearEpilogue::Residual}) ==
                  LinearConfig{LinearTile::Q4Register, 0, LinearSimdgroups::Four, 8} &&
              configured(10, 20, {{6144, 2048}, 32, LinearPhase::Decode, LinearEpilogue::GateUp}) ==
                  LinearConfig{LinearTile::N256, 24},
          "one-lane fallbacks or multi-lane rules changed");
  // Balanced two-tile groups above one wave: 130 paired tiles run one full
  // wave of longer chains on 16 cores, where their 65 paired N256 tiles would
  // need a second wave; on 20 cores those 65, and the 49 of the 98-tile
  // projection on 16 and 20, run as one paired N256 wave. The M16 grid holds
  // to five per core.
  require(configured(10, 20, {{16640, 5120}, 8}) ==
                  LinearConfig{LinearTile::Paired256, 65, LinearSimdgroups::Four} &&
              configured(10, 16, {{16640, 5120}, 8}) == LinearConfig{LinearTile::Paired128, 64} &&
              configured(10, 20, {{12544, 2048}, 8}) ==
                  LinearConfig{LinearTile::Paired256, 49, LinearSimdgroups::Four} &&
              configured(10, 16, {{12544, 2048}, 8}) ==
                  LinearConfig{LinearTile::Paired256, 49, LinearSimdgroups::Four} &&
              configured(10, 20, {{16640, 5120}, 16}) == LinearConfig{LinearTile::N128, 75} &&
              configured(10, 16, {{16640, 5120}, 16}) == LinearConfig{LinearTile::N128, 64} &&
              configured(10, 20, {{12544, 2048}, 16}) == LinearConfig{LinearTile::N128, 98} &&
              configured(10, 16, {{16640, 5120}, 24}) ==
                  LinearConfig{LinearTile::N128, 96, LinearSimdgroups::Four} &&
              configured(10, 20, {{16640, 5120}, 24}) ==
                  LinearConfig{LinearTile::N128, 130, LinearSimdgroups::Four} &&
              configured(10, 20, {{16640, 5120}, 32}) == LinearConfig{LinearTile::N256, 45} &&
              configured(10, 20, {{248320, 5120}, 16}) == LinearConfig{LinearTile::N128, 1940} &&
              configured(9, 20, {{16640, 5120}, 8}) == LinearConfig{LinearTile::Q4Register, 0, LinearSimdgroups::Four, 2} &&
              configured(10, 0, {{16640, 5120}, 8}) == configured(10, 32, {{16640, 5120}, 8}),
          "persistent decode groups changed for the measured shapes");
  require(configured(10, 16, {{14336, 5120}, 24}) ==
              LinearConfig{LinearTile::N128, 112, LinearSimdgroups::Four} &&
          configured(10, 16, {{14336, 5120}, 32}) == LinearConfig{LinearTile::N256, 40} &&
          configured(10, 40, {{14336, 5120}, 32}) == LinearConfig{LinearTile::N128, 112} &&
          configured(9, 40, {{14336, 5120}, 24}) ==
              LinearConfig{LinearTile::Q4Register, 0, LinearSimdgroups::Four, 4} &&
          configured(9, 40, {{248320, 5120}, 24}) ==
              LinearConfig{LinearTile::N128, 1940, LinearSimdgroups::Four} &&
          configured(9, 40, {{5120, 17408}, 24, LinearPhase::Decode, LinearEpilogue::Residual}) ==
              LinearConfig{LinearTile::Q4Register, 0, LinearSimdgroups::Four, 8},
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

// The tuner's candidates (tuning::linearCandidates): the policy's plan first,
// then each configuration that won a measured key wherever the workload
// admits it, once.
void planContracts(uint32_t family, uint32_t cores) {
  const DeviceCapabilities device = simulatedDevice(family, cores);
  const Linear linear(device);
  for (const LinearMatrix matrix : {LinearMatrix{512, 256}, LinearMatrix{768, 768},
                                    LinearMatrix{16640, 5120}, LinearMatrix{12544, 2048},
                                    LinearMatrix{23040, 2048}, LinearMatrix{131072, 4096}}) {
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                                 LinearEpilogue::GateUp}) {
        const LinearWorkload workload{matrix, lanes * 8, LinearPhase::Decode, epilogue};
        const auto candidates = tuning::linearCandidates(device, workload);
        // N128 and N256 on their full grids where the epilogue has the tile,
        // Split128 at each K split of whole 256-input blocks from Apple10,
        // and the one-lane plain paired N256 tile on its full grid.
        std::vector<LinearConfig> expected{linear.plan(workload).configuration()};
        const auto list = [&](LinearConfig config) {
          if (std::find(expected.begin(), expected.end(), config) == expected.end())
            expected.push_back(config);
        };
        const uint32_t n = matrix.outputSize;
        if (epilogue != LinearEpilogue::GateUp) list({LinearTile::N128, n / 128});
        if (epilogue != LinearEpilogue::Residual) list({LinearTile::N256, n / 256});
        if (family >= 10)
          for (const uint32_t splits : {2U, 4U, 8U})
            if (matrix.inputSize / 256 >= splits)
              list({LinearTile::Split128, 0, LinearSimdgroups::Eight, splits});
        if (lanes == 1 && epilogue == LinearEpilogue::None)
          list({LinearTile::Paired256, n / 256, LinearSimdgroups::Four});
        require(candidates.size() == expected.size(), "Linear candidate set omitted or added a plan");
        for (size_t index = 0; index < candidates.size(); ++index) {
          const auto &plan = candidates[index];
          require(plan.configuration() == expected[index], "Linear candidates are out of their order");
          const bool four = plan.configuration().simdgroups == LinearSimdgroups::Four;
          const auto tile = plan.configuration().tile;
          const bool split128 = tile == LinearTile::Split128;
          if (split128)
            require(plan.secondPipeline().empty() == (epilogue != LinearEpilogue::GateUp),
                    "Split128 gate/up runs a gate pass and an up pass");
          else if (plan.usesQ4Register())
            require(family == 9 && four && plan.configuration().groups == 0,
                    "Q4 register candidate escaped its full-grid contract");
          if (four) {
            const bool oneLane = lanes == 1 && tile == LinearTile::Paired256 &&
                epilogue == LinearEpilogue::None;
            require(((lanes == 3 && tile == LinearTile::N128 && epilogue != LinearEpilogue::GateUp) ||
                     oneLane || plan.usesQ4Register()) && plan.secondPipeline().empty(),
                    "four-SIMDgroup candidate escaped its precompiled workload set");
            if (lanes == 3 && !plan.usesQ4Register())
              require(plan.pipeline() == (epilogue == LinearEpilogue::Residual
                          ? "decode_linear_q4_n128_residual_m24_sg4" : "decode_linear_q4_n128_m24_sg4"),
                      "four-SIMDgroup plan chose the wrong pipeline");
          }
          require(plan.threadsPerThreadgroup() == (four ? 128 : 256),
                  "Linear plan scope/thread count disagree");
          require(plan.storageRows() == lanes * 8 && !plan.sumsBytes() && !plan.downSumsBytes(),
                  "decode storage/sums contract changed");
          require(plan.gateScratchBytes() ==
                      (epilogue == LinearEpilogue::GateUp && (split128 || (lanes >= 3 && !plan.usesQ4Register()))
                           ? uint64_t{lanes} * 8 * matrix.outputSize * 2 : 0),
                  "decode gate scratch disagrees with decomposition");
        }
      }
    }
    for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                               LinearEpilogue::UpWithGate}) {
      // Every prefill epilogue has the eight-simdgroup N256 tile and the
      // four-simdgroup N128 tile; the plain and residual ones also N128/8.
      const auto candidates =
          tuning::linearCandidates(device, {matrix, 2048, LinearPhase::Prefill, epilogue});
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
  for (const LinearMatrix matrix :
       {LinearMatrix{0, 256}, LinearMatrix{128, 256}, LinearMatrix{384, 256}, LinearMatrix{512, 0}})
    rejects([&] { (void)linear.plan({matrix, 8}); }, "invalid linear matrix",
            "a matrix of no or partial column tiles or quant groups was planned");
  for (const LinearWorkload invalid : {
           LinearWorkload{{512, 64}, 8}, LinearWorkload{{512, 320}, 8},
           LinearWorkload{{512, 256}, 0}, LinearWorkload{{512, 256}, 7},
           LinearWorkload{{512, 256}, 40},
           LinearWorkload{{512, 256}, 8, LinearPhase::Decode, LinearEpilogue::UpWithGate}})
    rejects([&] { (void)linear.plan(invalid); }, "invalid linear decode workload",
            "an invalid decode workload was planned");
  for (const LinearWorkload invalid : {
           LinearWorkload{{512, 256}, 8, LinearPhase::Prefill, LinearEpilogue::GateUp},
           LinearWorkload{{512, 256}, 2049, LinearPhase::Prefill}})
    rejects([&] { (void)linear.plan(invalid); }, "invalid linear prefill workload",
            "an invalid prefill workload was planned");
  for (const LinearConfig invalid : {LinearConfig{LinearTile::N128, 0}, LinearConfig{LinearTile::N128, 5}})
    rejects([&] { (void)Linear::plan(valid, invalid, FloatOutput::BFloat16); },
            "a persistent decode tile takes 1 to its column tiles in groups",
            "a persistent tile took no groups or more than its column tiles");
  rejects([&] { (void)Linear::plan({{512, 256}, 16}, {LinearTile::Paired128, 1}, FloatOutput::BFloat16); },
          "paired Q4 tile requires one lane", "a paired tile took two lanes");
  rejects(
      [&] {
        (void)Linear::plan({{512, 256}, 32, LinearPhase::Prefill}, {LinearTile::N128, 1}, FloatOutput::BFloat16);
      },
      "a persistent decode tile takes 1 to its column tiles in groups", "a prefill plan took groups");
  rejects(
      [&] {
        (void)Linear::plan({{512, 256}, 32, LinearPhase::Prefill}, {LinearTile::Paired128, 0},
                           FloatOutput::BFloat16);
      },
      "invalid Q4 prefill configuration", "a prefill plan took the paired tile");
  rejects(
      [&] {
        (void)Linear::plan({{512, 256}, 8, LinearPhase::Decode, LinearEpilogue::Residual}, {LinearTile::N256, 1},
                           FloatOutput::BFloat16);
      },
      "Q4 decode residual requires N128", "a decode residual took the N256 tile");
  rejects(
      [&] {
        (void)Linear::plan({{512, 256}, 8, LinearPhase::Decode, LinearEpilogue::GateUp}, {LinearTile::N128, 1},
                           FloatOutput::BFloat16);
      },
      "Q4 gate/up requires N256", "a decode gate/up took the N128 tile");
  rejects(
      [&] {
        (void)Linear::plan({{512, 256}, 8, LinearPhase::Prefill, LinearEpilogue::UpWithGate},
                           {LinearTile::N128, 0}, FloatOutput::BFloat16);
      },
      "Q4 fused prefill up requires N256 or four simdgroups",
      "an eight-simdgroup prefill up took the N128 tile");
  // Split128: eight simdgroups and two to eight K partitions of at least one
  // 256-input block, at every lane count and epilogue. Paired256: one lane,
  // plain epilogue, four simdgroups. Neither exists in prefill.
  const LinearWorkload splitWorkload{{512, 1024}, 8, LinearPhase::Decode, LinearEpilogue::None};
  const LinearWorkload splitResidual{{512, 1024}, 8, LinearPhase::Decode, LinearEpilogue::Residual};
  const LinearWorkload splitGateUp{{512, 1024}, 8, LinearPhase::Decode, LinearEpilogue::GateUp};
  const LinearConfig q4Register{LinearTile::Q4Register, 0, LinearSimdgroups::Four, 4};
  for (const uint32_t splits : {0U, 3U, 16U}) {
    auto invalid = q4Register;
    invalid.splits = splits;
    rejects([&] { (void)Linear::plan(splitWorkload, invalid, FloatOutput::BFloat16); },
            "the Q4 register tile requires whole power-of-two K partitions",
            "a Q4 register plan took a K split that is not a power of two up to eight");
  }
  rejects([&] { (void)Linear::plan({{512, 768}, 8},
      {LinearTile::Q4Register, 0, LinearSimdgroups::Four, 8}, FloatOutput::BFloat16); },
          "the Q4 register tile requires whole power-of-two K partitions",
          "a Q4 register plan split twelve quant groups eight ways");
  for (uint32_t rows : {8U, 16U, 24U, 32U}) {
    const LinearWorkload workload{{512, 1024}, rows};
    const LinearPlan plan = Linear::plan(workload, q4Register, FloatOutput::BFloat16);
    require(sameScratch(plan.scratchSize(), expectedScratch(q4Register, workload, plan.tileColumns())),
            "Q4 register row tiles must own disjoint input, sums, both fragment streams' partials and counters");
  }
  rejects([&] { (void)Linear::plan(splitWorkload,
      {LinearTile::Q4Register, 0, LinearSimdgroups::Eight, 4}, FloatOutput::BFloat16); },
          "Q4 tile requires its kernel's simdgroup count", "a Q4 register plan took eight simdgroups");
  const LinearConfig split128{LinearTile::Split128, 0, LinearSimdgroups::Eight, 4};
  const LinearConfig paired256{LinearTile::Paired256, 2, LinearSimdgroups::Four};
  // A tile whose grid covers the matrix takes no group count, not even its
  // grid's: the plan derives it.
  for (LinearConfig fullGrid : {q4Register, split128}) {
    fullGrid.groups = 512 / Linear::plan(splitWorkload, fullGrid, FloatOutput::BFloat16).tileColumns();
    rejects([&] { (void)Linear::plan(splitWorkload, fullGrid, FloatOutput::BFloat16); },
            "a persistent decode tile takes 1 to its column tiles in groups",
            "a tile whose grid covers the matrix took a group count");
  }
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    const std::string rows = rowsSuffix(lanes);
    const LinearWorkload plain{{512, 1024}, lanes * 8, LinearPhase::Decode, LinearEpilogue::None};
    const auto plan = Linear::plan(plain, split128, FloatOutput::BFloat16);
    require(plan.pipeline() == "decode_linear_q4_n128_split" + rows && plan.threadsPerThreadgroup() == 256 &&
                plan.tileColumns() == 128 && plan.secondPipeline().empty() &&
                plan.storageRows() == lanes * 8 && plan.input() == LinearInput::Plain &&
                sameScratch(plan.scratchSize(), expectedScratch(split128, plain, 128)),
            "Split128 plan geometry, pipeline or scratch is wrong");
    require(Linear::plan({plain.matrix, lanes * 8, LinearPhase::Decode, LinearEpilogue::Residual}, split128,
                         FloatOutput::BFloat16)
                    .pipeline() == "decode_linear_q4_n128_split_residual" + rows,
            "Split128 residual pipeline is wrong");
    const auto gateUpPlan = Linear::plan({plain.matrix, lanes * 8, LinearPhase::Decode, LinearEpilogue::GateUp},
                                         split128, FloatOutput::BFloat16);
    require(gateUpPlan.pipeline() == "decode_linear_q4_n128_split" + rows &&
                gateUpPlan.secondPipeline() == "decode_linear_q4_n128_split_up_silu" + rows &&
                gateUpPlan.gateScratchBytes() == uint64_t{lanes} * 8 * 512 * 2,
            "Split128 gate/up runs a gate pass into the gate scratch and an up pass");
  }
  const auto wide = Linear::plan(splitWorkload, paired256, FloatOutput::BFloat16);
  require(wide.pipeline() == "decode_linear_q4_n256_paired_sg4" && wide.threadsPerThreadgroup() == 128 &&
              wide.tileColumns() == 256,
          "Paired256 plan geometry or pipeline is wrong");
  // Split128 takes eight simdgroups and 2, 4 or 8 partitions of at least one
  // block: 1024 inputs hold four.
  rejects(
      [&] {
        (void)Linear::plan(splitWorkload, {LinearTile::Split128, 0, LinearSimdgroups::Four, 4},
                           FloatOutput::BFloat16);
      },
      "invalid Q4 cooperative execution scope", "Split128 took four simdgroups");
  for (const uint32_t splits : {0U, 1U, 3U, 8U, 16U})
    rejects(
        [&] {
          (void)Linear::plan(splitWorkload, {LinearTile::Split128, 0, LinearSimdgroups::Eight, splits},
                             FloatOutput::BFloat16);
        },
        "Split128 requires 2, 4 or 8 K partitions of 256-input blocks",
        "Split128 split four 256-input blocks other than two or four ways");
  rejects(
      [&] {
        (void)Linear::plan(splitWorkload, {LinearTile::Paired256, 2, LinearSimdgroups::Eight},
                           FloatOutput::BFloat16);
      },
      "Q4 tile requires its kernel's simdgroup count", "Paired256 took eight simdgroups");
  rejects([&] { (void)Linear::plan(splitResidual, paired256, FloatOutput::BFloat16); },
          "invalid Q4 cooperative execution scope", "Paired256 took a residual");
  rejects([&] { (void)Linear::plan(splitGateUp, paired256, FloatOutput::BFloat16); },
          "invalid Q4 cooperative execution scope", "Paired256 took gate/up");
  for (uint32_t rows : {16U, 24U, 32U})
    rejects([&] { (void)Linear::plan({{512, 1024}, rows}, paired256, FloatOutput::BFloat16); },
            "invalid Q4 cooperative execution scope", "Paired256 took more than one lane");
  rejects([&] { (void)Linear::plan({{512, 1024}, 32, LinearPhase::Prefill},
                                   {LinearTile::Paired256, 0, LinearSimdgroups::Four}, FloatOutput::BFloat16); },
          "invalid Q4 cooperative execution scope", "a prefill plan took Paired256");
  rejects([&] { (void)Linear::plan({{512, 1024}, 32, LinearPhase::Prefill},
                                   {LinearTile::Split128, 0, LinearSimdgroups::Eight, 4}, FloatOutput::BFloat16); },
          "invalid Q4 prefill configuration", "a prefill plan took Split128");
  const LinearWorkload fourWorkload{{512, 256}, 24, LinearPhase::Decode, LinearEpilogue::None};
  const LinearConfig fourConfig{LinearTile::N128, 1, LinearSimdgroups::Four};
  for (uint32_t rows : {8U, 16U, 32U})
    rejects([&] { (void)Linear::plan({{512, 256}, rows}, fourConfig, FloatOutput::BFloat16); },
            "invalid Q4 cooperative execution scope", "a four-simdgroup decode took other than three lanes");
  for (const auto tile : {LinearTile::N256, LinearTile::Paired128})
    rejects([&] { (void)Linear::plan(fourWorkload,
        {tile, 1, LinearSimdgroups::Four}, FloatOutput::BFloat16); },
            "invalid Q4 cooperative execution scope", "a four-simdgroup decode took a tile other than N128");
  // Four simdgroups decode no gate/up, and no decode takes the up-with-gate
  // epilogue.
  rejects([&] { (void)Linear::plan({{512, 256}, 24, LinearPhase::Decode, LinearEpilogue::GateUp},
                                    fourConfig, FloatOutput::BFloat16); },
          "invalid Q4 cooperative execution scope", "a four-simdgroup decode took gate/up");
  rejects([&] { (void)Linear::plan({{512, 256}, 24, LinearPhase::Decode, LinearEpilogue::UpWithGate},
                                    fourConfig, FloatOutput::BFloat16); },
          "invalid linear decode workload", "a decode took the up-with-gate epilogue");
  for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                             LinearEpilogue::UpWithGate}) {
    const LinearWorkload prefill{{512, 256}, 24, LinearPhase::Prefill, epilogue};
    require(Linear::plan(prefill, {LinearTile::N128, 0, LinearSimdgroups::Four}, FloatOutput::BFloat16)
                    .threadsPerThreadgroup() == 128,
            "four-SIMDgroup prefill plan was rejected");
    rejects([&] { (void)Linear::plan(prefill, {LinearTile::N256, 0, LinearSimdgroups::Four}, FloatOutput::BFloat16); },
            "invalid Q4 cooperative execution scope", "a four-simdgroup prefill took the N256 tile");
    rejects([&] { (void)Linear::plan(prefill, {LinearTile::N128, 1, LinearSimdgroups::Four}, FloatOutput::BFloat16); },
            "a persistent decode tile takes 1 to its column tiles in groups", "a prefill plan took groups");
  }
}

// Every affine plan for families 9-11, reported core counts 0-128 and a grid
// of shapes covering the rule boundaries follows the stated rules: its
// configuration, execution scope, pipelines, tile geometry, input layout and
// scratch. A policy change fails here with the rule and the plan it changed.
void affinePolicyLaws() {
  for (const uint32_t family : {9U, 10U, 11U})
    for (uint32_t cores = 0; cores <= 128; ++cores) {
      const Linear linear = gpu(family, cores);
      for (const uint32_t n : {256U, 512U, 1024U, 2048U, 4096U, 5120U, 6144U, 9216U, 10240U,
                               12544U, 14336U, 16640U, 17408U, 248320U})
        for (const uint32_t k : {2048U, 4096U, 5120U, 6144U, 17408U}) {
          for (uint32_t lanes = 1; lanes <= 4; ++lanes)
            for (const auto e : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp})
              checkAffineDecode(linear, family, cores, {{n, k}, lanes * 8, LinearPhase::Decode, e});
          for (const uint32_t rows : {1U, 8U, 32U, 33U, 128U, 512U, 2048U})
            for (const auto e : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate})
              checkAffinePrefill(linear, family, cores, {{n, k}, rows, LinearPhase::Prefill, e});
        }
    }
}

// A block projection of equal segments tiling its columns, one per format;
// planning reads only their geometry and formats.
Projection blockProjection(uint32_t n, uint32_t k, std::span<const uint32_t> formats) {
  BlockWeights weights;
  const uint32_t width = n / uint32_t(formats.size());
  for (const uint32_t format : formats) {
    QuantizedSegment s = QuantizedSegment::planes(format, width, k, {}, {}, {});
    s.columnOffset = uint32_t(weights.segments.size()) * width;
    weights.segments.push_back(s);
  }
  return Projection(n, k, std::move(weights));
}
// ... of `segments` segments in `format`, Q4_K unless it says otherwise.
Projection blockProjection(uint32_t n, uint32_t k, uint32_t segments, uint32_t format = GGUF_FMT_Q4K) {
  return blockProjection(n, k, std::vector<uint32_t>(segments, format));
}

// fp32 destinations (the logits): the plan of a projection with an fp32
// destination keeps the configuration, tile kernel, input table and scratch of
// its bf16 plan on every family and core count, and only plain decode
// workloads take one. Returns the kernels of those plans, whose fp32
// instances floatInstances looks up.
std::set<std::string_view> floatOutputPlans() {
  const auto fp32 = [](Projection p) {
    p.destination = FloatOutput::Float32;
    return p;
  };
  std::set<std::string_view> kernels;
  for (const uint32_t family : {9U, 10U, 11U})
    for (uint32_t cores = 0; cores <= 128; ++cores) {
      const Linear linear = gpu(family, cores);
      for (const uint32_t n : {256U, 5120U, 16640U, 248320U})
        for (const uint32_t k : {2048U, 5120U})
          for (const Projection &p : {Projection(n, k, AffineWeights{}), blockProjection(n, k, 1)})
            for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
              const LinearWorkload w{{n, k}, lanes * 8, LinearPhase::Decode, LinearEpilogue::None, p.layout()};
              const LinearPlan bf16 = linear.plan(w, p), head = linear.plan(w, fp32(p));
              if (head.configuration() != bf16.configuration() || head.destination() != FloatOutput::Float32 ||
                  head.pipeline() != bf16.pipeline() || head.input() != bf16.input() ||
                  head.storageRows() != bf16.storageRows() || !sameScratch(head.scratchSize(), bf16.scratchSize()))
                broke("an fp32 plan differs from its bf16 plan", family, cores, w);
              if (!head.pipeline().empty()) kernels.insert(head.pipeline());
            }
    }
  std::vector<LinearWorkload> others{{{5120, 2048}, 8, LinearPhase::Decode, LinearEpilogue::Residual},
                                     {{5120, 2048}, 8, LinearPhase::Decode, LinearEpilogue::GateUp}};
  for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate})
    for (const uint32_t rows : {8U, 2048U}) others.push_back({{5120, 2048}, rows, LinearPhase::Prefill, epilogue});
  const Linear linear = gpu(10, 20);
  for (const Projection &p : {Projection(5120, 2048, AffineWeights{}), blockProjection(5120, 2048, 1)})
    for (const LinearWorkload &w : others) {
      (void)linear.plan(w, p);
      rejects([&] { (void)linear.plan(w, fp32(p)); }, "an fp32 destination takes a plain decode projection",
              "an fp32 plan of other than a plain decode was accepted");
    }
  return kernels;
}

// A GGUF decode projection whose split count is pinned on one core count.
struct SplitAnchor final {
  const char *projection;
  uint32_t cores, n, k, rows;
  LinearEpilogue epilogue;
  uint32_t segments, splits;
};

LinearPlan anchorPlan(uint32_t family, const SplitAnchor &anchor) {
  return gpu(family, anchor.cores).plan({{anchor.n, anchor.k}, anchor.rows, LinearPhase::Decode, anchor.epilogue},
                                        blockProjection(anchor.n, anchor.k, anchor.segments));
}

void requireAnchorSplits(const LinearPlan &plan, const SplitAnchor &anchor, const char *policy) {
  const uint32_t splits = plan.configuration().splits;
  if (splits != anchor.splits)
    throw std::runtime_error(std::string(policy) + ": " + anchor.projection + " on " + std::to_string(anchor.cores) +
                             " cores plans " + std::to_string(splits) + " K splits, not " +
                             std::to_string(anchor.splits));
}

constexpr auto kNone = LinearEpilogue::None, kResidual = LinearEpilogue::Residual, kGateUp = LinearEpilogue::GateUp;

// Six threadgroups per core, at least 512 inputs per partition, for every
// projection kind.
constexpr std::array kStagedSplitAnchors{
    SplitAnchor{"27B down", 16, 5120, 17408, 8, kResidual, 1, 2},
    SplitAnchor{"27B down", 20, 5120, 17408, 8, kResidual, 1, 2},
    SplitAnchor{"27B down", 10, 5120, 17408, 8, kResidual, 1, 1},
    SplitAnchor{"27B down", 40, 5120, 17408, 8, kResidual, 1, 4},
    SplitAnchor{"35B output projection", 16, 2048, 4096, 8, kResidual, 1, 4},
    SplitAnchor{"35B shared-expert gate/up", 16, 512, 2048, 8, kGateUp, 1, 4},
    SplitAnchor{"27B gate/up", 16, 17408, 5120, 8, kGateUp, 1, 1},
    SplitAnchor{"two-segment fused projection", 16, 4096, 5120, 8, kNone, 2, 2},
    SplitAnchor{"27B three-segment GDN input", 40, 16640, 5120, 8, kNone, 3, 1},
    SplitAnchor{"27B vocabulary head", 16, 248320, 5120, 8, kNone, 1, 1},
    SplitAnchor{"narrow 1024 x 256 tensor", 16, 1024, 256, 8, kNone, 1, 1},
    SplitAnchor{"narrow 1024 x 3072 tensor", 16, 1024, 3072, 8, kNone, 1, 4}};

// One wave (four threadgroups per core) down to one 256-input unit per
// partition, eight waves while partitions keep 1024 inputs, at most eight.
constexpr std::array kRegisterSplitAnchors{
    SplitAnchor{"27B down", 40, 5120, 17408, 8, kResidual, 1, 8},
    SplitAnchor{"27B out_proj, four lanes", 40, 5120, 6144, 32, kResidual, 1, 4},
    SplitAnchor{"27B three-segment GDN input, two lanes", 40, 16640, 5120, 16, kNone, 3, 4},
    SplitAnchor{"27B three-segment attention input", 40, 14336, 5120, 8, kNone, 3, 4},
    SplitAnchor{"27B gate/up, three lanes", 40, 17408, 5120, 24, kGateUp, 1, 4},
    SplitAnchor{"27B vocabulary head", 40, 248320, 5120, 8, kNone, 1, 1},
    SplitAnchor{"27B down", 10, 5120, 17408, 8, kResidual, 1, 4},
    SplitAnchor{"27B three-segment GDN input", 10, 16640, 5120, 8, kNone, 3, 2},
    SplitAnchor{"35B two-segment GDN input", 40, 12288, 2048, 8, kNone, 2, 2},
    SplitAnchor{"35B two-segment GDN input", 80, 12288, 2048, 8, kNone, 2, 2},
    SplitAnchor{"35B shared-expert gate/up", 40, 512, 2048, 8, kGateUp, 1, 8},
    SplitAnchor{"35B shared-expert down", 40, 2048, 512, 8, kResidual, 1, 2},
    SplitAnchor{"35B output projection", 40, 2048, 4096, 8, kResidual, 1, 8},
    SplitAnchor{"35B vocabulary head", 40, 248320, 2048, 8, kNone, 1, 1},
    SplitAnchor{"narrow 1024 x 5120 tensor", 40, 1024, 5120, 8, kNone, 1, 8},
    SplitAnchor{"narrow 1024 x 1024 tensor", 40, 1024, 1024, 8, kNone, 1, 4}};

// fp32 K-split partials over the plan's rows and columns and one completion
// counter per 64-column tile (LinearPlan::scratchSize).
uint64_t splitPartialsBytes(const LinearPlan &plan) {
  return uint64_t{plan.configuration().splits} * plan.storageRows() * plan.workload().matrix.outputSize *
         sizeof(float);
}
uint64_t splitCountersBytes(const LinearPlan &plan) {
  return uint64_t{plan.groups()} * sizeof(uint32_t);
}
// The bf16 gate projection a GGUF gate/up plan writes before its up pass.
uint64_t gateBytes(const LinearPlan &plan) {
  return uint64_t{plan.storageRows()} * plan.workload().matrix.outputSize * sizeof(uint16_t);
}

// GGUF projections plan with their segments: the staged split policy for
// every projection kind, exact split scratch over the tile's rows, and
// prefill tiles of 8, 16, 32 or 128 rows.
void ggufPlans() {
  const Linear linear = gpu(10, 16);
  // A block projection without segments would reach the dispatch paths with
  // nothing to index or encode.
  rejects([] { (void)Projection(5120, 17408, BlockWeights{}); }, "block projection has no segments",
          "a block projection without segments was accepted");
  // Its segments tile the leading columns in order: a gap, an overlap, another
  // input width or a segment past the end is not a projection; padding past
  // the last segment is.
  const auto segment = [](uint32_t n, uint32_t k, uint32_t offset) {
    QuantizedSegment s = QuantizedSegment::planes(GGUF_FMT_Q4K, n, k, {}, {}, {});
    s.columnOffset = offset;
    return s;
  };
  rejects([&] { (void)Projection(768, 256, BlockWeights{{segment(256, 256, 0), segment(256, 256, 320)}}); },
          "block segments do not tile the projection", "segments with a gap between them were accepted");
  rejects([&] { (void)Projection(768, 256, BlockWeights{{segment(256, 256, 0), segment(256, 256, 128)}}); },
          "block segments do not tile the projection", "overlapping segments were accepted");
  rejects([&] { (void)Projection(768, 256, BlockWeights{{segment(256, 512, 0)}}); },
          "block segments do not tile the projection", "a segment of another input width was accepted");
  rejects([&] { (void)Projection(768, 256, BlockWeights{{segment(256, 256, 0), segment(768, 256, 256)}}); },
          "block segments do not tile the projection", "a segment past the projection's end was accepted");
  require(Projection(768, 256, BlockWeights{{segment(256, 256, 0), segment(256, 256, 256)}}).blocks().segments.size() == 2,
          "a projection with padding past its segments was rejected");
  const LinearWorkload down{{5120, 17408}, 8, LinearPhase::Decode, LinearEpilogue::Residual};
  const LinearPlan single = linear.plan(down, blockProjection(5120, 17408, 1));
  require(single.workload().weightLayout == WeightLayout::Block32 &&
              single.configuration() == LinearConfig{.tile = LinearTile::GgufStaged, .splits = 2} &&
              single.groups() == 80 &&
              single.input() == LinearInput::Plain &&
              single.scratchSize().partials == splitPartialsBytes(single) &&
              single.scratchSize().counters == splitCountersBytes(single) && single.scratchSize().input == 0,
          "GGUF single-tensor decode plan");
  for (const SplitAnchor &anchor : kStagedSplitAnchors)
    requireAnchorSplits(anchorPlan(10, anchor), anchor, "GGUF staged split policy");
  const LinearPlan gateUpPlan = linear.plan({{512, 2048}, 16, LinearPhase::Decode, LinearEpilogue::GateUp},
                                            blockProjection(512, 2048, 1));
  require(gateUpPlan.configuration().splits == 4 && gateUpPlan.gateScratchBytes() == gateBytes(gateUpPlan) &&
              gateUpPlan.scratchSize().partials == splitPartialsBytes(gateUpPlan),
          "GGUF staged gate/up runs a gate pass into the gate scratch");
  // Decode tiles hold 8, 16 or 32 rows: three lanes run the 32-row tile.
  const LinearPlan three = linear.plan({{5120, 17408}, 24, LinearPhase::Decode, LinearEpilogue::Residual},
                                       blockProjection(5120, 17408, 1));
  require(three.storageRows() == 32 && three.configuration() == single.configuration() &&
              three.scratchSize().partials == splitPartialsBytes(three),
          "GGUF staged three-lane plans run the 32-row tile");
  for (const auto [rows, storage] : {std::pair{1U, 8U}, {8U, 8U}, {9U, 16U}, {17U, 32U}, {25U, 32U}, {32U, 32U},
                                     {33U, 128U}, {100U, 128U}, {129U, 256U}, {2048U, 2048U}}) {
    const LinearPlan prefill = linear.plan({{5120, 17408}, rows, LinearPhase::Prefill, LinearEpilogue::UpWithGate},
                                           blockProjection(5120, 17408, 1));
    // Chunks of up to 32 rows take the staged tile and its split rule (two
    // partitions of the 80-tile grid on 16 cores); the 128-row prefill tile
    // takes none.
    const uint32_t splits = rows <= 32 ? 2 : 1;
    require(prefill.storageRows() == storage && prefill.sumsBytes() == 0 && prefill.downSumsBytes() == 0 &&
                prefill.gateScratchBytes() == gateBytes(prefill) &&
                prefill.configuration().tile == (rows <= 32 ? LinearTile::GgufStaged : LinearTile::GgufPrefill) &&
                prefill.configuration().splits == splits &&
                prefill.scratchSize().partials == (splits > 1 ? splitPartialsBytes(prefill) : 0),
            "GGUF prefill tile rows and splits");
  }
  // The decode tiles hold at most a decode batch.
  LinearWorkload longPrefill{{5120, 17408}, 33, LinearPhase::Prefill, LinearEpilogue::None, WeightLayout::Block32};
  rejects([&] { (void)Linear::plan(longPrefill, {.tile = LinearTile::GgufStaged}, FloatOutput::BFloat16); },
          "the staged block tile runs prefill chunks of up to 32 rows",
          "the staged tile took a prefill chunk past a decode batch");
  // Affine and GGUF plans do not mix.
  rejects([&] { (void)Linear::plan(down, {.tile = LinearTile::GgufStaged, .splits = 8}, FloatOutput::BFloat16); },
          "block projections run the GGUF tiles, affine ones the Q4 tiles", "an affine plan took a GGUF tile");
  LinearWorkload gguf = down;
  gguf.weightLayout = WeightLayout::Block32;
  rejects([&] { (void)Linear::plan(gguf, {LinearTile::N128, 40}, FloatOutput::BFloat16); },
          "block projections run the GGUF tiles, affine ones the Q4 tiles", "a GGUF plan took a Q4 tile");
  rejects(
      [&] {
        (void)Linear::plan(gguf, {.tile = LinearTile::GgufStaged, .groups = 80, .splits = 8},
                           FloatOutput::BFloat16);
      },
      "a persistent decode tile takes 1 to its column tiles in groups", "the staged tile took a group count");
  rejects([&] { (void)Linear::plan(gguf, {.tile = LinearTile::GgufStaged, .splits = 3}, FloatOutput::BFloat16); },
          "staged block splits take whole 32-input groups", "the staged tile split K three ways");
  // The arena bound is the single-tensor plan of the 32-row tile, which fused
  // and gate/up plans share.
  const ProjectionShape downShape{5120, 17408, WeightLayout::Block32};
  require(linear.decodeScratchSize(downShape).partials == three.scratchSize().partials,
          "GGUF decode scratch bound");
  // Float projections take the neural accelerator tile from three of its
  // 64 x 32 tiles per two cores: on 16 cores the 35B router (N 256) from 129
  // rows, alpha/beta (N 64) from 705; never on Apple9 or below 16 rows.
  const Linear oneCore = gpu(10, 1);
  require(linear.ggufFloatTile(32, 256) == FloatTile::Simdgroup && linear.ggufFloatTile(128, 256) == FloatTile::Simdgroup &&
              linear.ggufFloatTile(129, 256) == FloatTile::NeuralAccelerator &&
              linear.ggufFloatTile(2048, 256) == FloatTile::NeuralAccelerator &&
              linear.ggufFloatTile(704, 64) == FloatTile::Simdgroup &&
              linear.ggufFloatTile(705, 64) == FloatTile::NeuralAccelerator &&
              oneCore.ggufFloatTile(15, 256) == FloatTile::Simdgroup &&
              oneCore.ggufFloatTile(16, 256) == FloatTile::NeuralAccelerator,
          "GGUF float tile rule");

  // Apple9 decodes every GGUF width with the exact register tile, all lanes
  // in one threadgroup, and K splits from the core count; prefill stages.
  for (const SplitAnchor &anchor : kRegisterSplitAnchors) {
    const LinearPlan plan = anchorPlan(9, anchor);
    require(plan.configuration().tile == LinearTile::GgufRegister &&
                plan.configuration().groups == 0 && plan.groups() == anchor.n / 64 &&
                plan.input() == LinearInput::Table16,
            "Apple9 GGUF register plan");
    requireAnchorSplits(plan, anchor, "Apple9 GGUF register split policy");
  }
  const Linear m3 = gpu(9, 40);
  for (const uint32_t rows : {8U, 32U}) {
    const LinearPlan plan = m3.plan({{5120, 17408}, rows, LinearPhase::Decode, LinearEpilogue::Residual},
                                    blockProjection(5120, 17408, 1));
    const LinearScratchSize size = plan.scratchSize();
    require(plan.configuration().splits == 8 && size.input == tableBytes(17408, rows) &&
                size.sums == tableSumsBytes(LinearInput::Table16, 17408, rows) &&
                size.partials == splitPartialsBytes(plan) && size.counters == splitCountersBytes(plan) &&
                plan.gateScratchBytes() == 0,
            "Apple9 GGUF register scratch");
  }
  const LinearPlan head = m3.plan({{248320, 5120}, 8, LinearPhase::Decode, LinearEpilogue::None},
                                  blockProjection(248320, 5120, 1));
  require(head.scratchSize().partials == 0 && head.scratchSize().counters == 0,
          "Apple9 GGUF register scratch without splits holds partials or counters");
  const LinearPlan registerGateUp = m3.plan({{17408, 5120}, 16, LinearPhase::Decode, LinearEpilogue::GateUp},
                                            blockProjection(17408, 5120, 1));
  require(registerGateUp.gateScratchBytes() == gateBytes(registerGateUp),
          "Apple9 GGUF gate/up runs a gate pass into the gate scratch");
  require(m3.plan({{5120, 17408}, 100, LinearPhase::Prefill, LinearEpilogue::Residual},
                  blockProjection(5120, 17408, 1)).configuration().tile == LinearTile::GgufPrefill,
          "Apple9 GGUF prefill stages");
  require(m3.ggufFloatTile(2048, 256) == FloatTile::Simdgroup, "Apple9 float projections take the simdgroup tile");
  LinearWorkload registerDown = down;
  registerDown.weightLayout = WeightLayout::Block32;
  const LinearScratchSize bound = m3.decodeScratchSize(downShape);
  require(bound.partials ==
              m3.plan({down.matrix, 32, LinearPhase::Decode, LinearEpilogue::Residual}, blockProjection(5120, 17408, 1))
                  .scratchSize().partials,
          "Apple9 GGUF decode scratch bound");
  // Apple9 stages the IQ2, IQ3_XXS and IQ1 formats wherever the staged tile
  // holds the lanes' rows unpadded, and Q2_K from two lanes. A projection
  // stages only when every quantized segment's format does; its plan binds
  // the register tile's rows, and the decode scratch bound covers both tiles.
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    const auto plan = [&](std::initializer_list<uint32_t> formats) {
      const uint32_t n = uint32_t(formats.size()) * 1024;
      return m3.plan({{n, 5120}, lanes * 8, LinearPhase::Decode, LinearEpilogue::None},
                     blockProjection(n, 5120, formats));
    };
    const LinearTile iq = lanes == 3 ? LinearTile::GgufRegister : LinearTile::GgufStaged;
    const LinearTile q2k = lanes == 2 || lanes == 4 ? LinearTile::GgufStaged : LinearTile::GgufRegister;
    bool staged = true;
    for (const uint32_t format :
         {GGUF_FMT_IQ3XXS, GGUF_FMT_IQ2XXS, GGUF_FMT_IQ2XS, GGUF_FMT_IQ2S, GGUF_FMT_IQ1S, GGUF_FMT_IQ1M})
      staged = staged && plan({format}).configuration().tile == iq;
    require(staged && plan({GGUF_FMT_IQ3XXS, GGUF_FMT_IQ2S, GGUF_FMT_IQ1M}).configuration().tile == iq,
            "Apple9 stages the IQ2, IQ3_XXS and IQ1 formats at unpadded lanes");
    require(plan({GGUF_FMT_Q2K}).configuration().tile == q2k, "Apple9 stages Q2_K from two unpadded lanes");
    for (const uint32_t format : {GGUF_FMT_Q4K, GGUF_FMT_Q6K, GGUF_FMT_IQ4XS, GGUF_FMT_IQ3S, GGUF_FMT_Q80,
                                  GGUF_FMT_Q40, GGUF_FMT_MXFP4})
      require(plan({format}).configuration().tile == LinearTile::GgufRegister, "Apple9 keeps other formats' registers");
    require(plan({GGUF_FMT_IQ3XXS, GGUF_FMT_Q4K}).configuration().tile == LinearTile::GgufRegister &&
                plan({GGUF_FMT_IQ3XXS}).storageRows() == plan({GGUF_FMT_Q4K}).storageRows(),
            "Apple9 keeps a mixed projection's registers, and a staged plan binds the register rows");
    // A gate/up plan runs its gate on the same tile: it stages only when the gate's formats do too.
    const LinearWorkload gateUp{{1024, 5120}, lanes * 8, LinearPhase::Decode, LinearEpilogue::GateUp};
    const Projection up = blockProjection(1024, 5120, 1, GGUF_FMT_IQ2XXS),
                     iqGate = blockProjection(1024, 5120, 1, GGUF_FMT_IQ2XS), q4kGate = blockProjection(1024, 5120, 1);
    require(m3.plan(gateUp, up, &iqGate).configuration().tile == iq &&
                m3.plan(gateUp, up, &q4kGate).configuration().tile == LinearTile::GgufRegister,
            "Apple9 staged a gate/up plan whose gate keeps the register tile");
    const LinearPlan stagedDown = m3.plan({down.matrix, lanes * 8, LinearPhase::Decode, LinearEpilogue::Residual},
                                          blockProjection(5120, 17408, 1, GGUF_FMT_IQ2XS));
    const LinearPlan registerPlan = m3.plan({down.matrix, lanes * 8, LinearPhase::Decode, LinearEpilogue::Residual},
                                            blockProjection(5120, 17408, 1));
    for (const LinearScratchSize size : {stagedDown.scratchSize(), registerPlan.scratchSize()})
      require(bound.input >= size.input && bound.sums >= size.sums && bound.partials >= size.partials &&
                  bound.counters >= size.counters,
              "Apple9 GGUF decode scratch bound covers both tiles");
  }
  require(linear.plan({{5120, 17408}, 24, LinearPhase::Decode, LinearEpilogue::Residual},
                      blockProjection(5120, 17408, 1, GGUF_FMT_IQ2XXS)).configuration().tile == LinearTile::GgufStaged,
          "Apple10 stages every format");
  // Apple9's staged tile, in decode and in prefill chunks, splits K by the
  // register tile's tiers: on 40 cores 17408 x 5120 in four, 5120 x 17408 in
  // eight.
  for (const auto [n, k, splits] : {std::tuple{17408U, 5120U, 4U}, {5120U, 17408U, 8U}}) {
    const LinearPlan decode = m3.plan({{n, k}, 8, LinearPhase::Decode, LinearEpilogue::None},
                                      blockProjection(n, k, 1, GGUF_FMT_IQ2XXS));
    const LinearPlan chunk = m3.plan({{n, k}, 8, LinearPhase::Prefill, LinearEpilogue::None}, blockProjection(n, k, 1));
    require(decode.configuration() == LinearConfig{.tile = LinearTile::GgufStaged, .splits = splits} &&
                chunk.configuration().splits == splits,
            "Apple9 staged split tiers");
  }
  for (const LinearConfig config : {LinearConfig{.tile = LinearTile::GgufRegister, .splits = 3},
                                    LinearConfig{.tile = LinearTile::GgufRegister, .splits = 16}})
    rejects([&] { (void)Linear::plan(registerDown, config, FloatOutput::BFloat16); },
            "the register block decode tile takes a K unit per split",
            "the register tile took a K split that is not a power of two up to eight");
  rejects(
      [&] {
        (void)Linear::plan(registerDown, {.tile = LinearTile::GgufRegister, .groups = 80, .splits = 8},
                           FloatOutput::BFloat16);
      },
      "a persistent decode tile takes 1 to its column tiles in groups", "the register tile took a group count");
  // Split boundaries fall on 256-input units, and prefill has no register tile.
  const LinearWorkload chunk{{5120, 17408}, 128, LinearPhase::Prefill, LinearEpilogue::None, WeightLayout::Block32};
  rejects(
      [&] {
        (void)Linear::plan({{5120, 512}, 8, LinearPhase::Decode, LinearEpilogue::None, WeightLayout::Block32},
                           {.tile = LinearTile::GgufRegister, .splits = 4}, FloatOutput::BFloat16);
      },
      "the register block decode tile takes a K unit per split", "the register tile split two K units four ways");
  rejects([&] { (void)Linear::plan(chunk, {.tile = LinearTile::GgufRegister}, FloatOutput::BFloat16); },
          "the register block decode tile takes a K unit per split", "a prefill plan took the register tile");
  // The GGUF kernels fix their threadgroups (GGUF_*_THREADS): every GGUF tile
  // takes the default simdgroups.
  const auto fixedThreadgroup = [](LinearWorkload workload, LinearConfig config, uint32_t threads) {
    require(Linear::plan(workload, config, FloatOutput::BFloat16).threadsPerThreadgroup() == threads,
            "a GGUF plan's threadgroup is not its kernel's");
    config.simdgroups = LinearSimdgroups::Four;
    rejects([&] { (void)Linear::plan(workload, config, FloatOutput::BFloat16); },
            "the GGUF tiles fix their threadgroups and take the default simdgroups",
            "a GGUF tile took four simdgroups");
  };
  fixedThreadgroup(registerDown, {.tile = LinearTile::GgufRegister, .splits = 8}, GGUF_REGISTER_THREADS);
  fixedThreadgroup(registerDown, {.tile = LinearTile::GgufStaged, .splits = 2}, GGUF_STAGED_THREADS);
  fixedThreadgroup(chunk, {.tile = LinearTile::GgufPrefill}, GGUF_PREFILL_THREADS);
  // The prefill tile runs prefill chunks on their matrix grid, unsplit.
  rejects([&] { (void)Linear::plan(registerDown, {.tile = LinearTile::GgufPrefill}, FloatOutput::BFloat16); },
          "invalid block prefill configuration", "a decode plan took the prefill tile");
  rejects([&] { (void)Linear::plan(chunk, {.tile = LinearTile::GgufPrefill, .groups = 80}, FloatOutput::BFloat16); },
          "a persistent decode tile takes 1 to its column tiles in groups", "the prefill tile took a group count");
  rejects([&] { (void)Linear::plan(chunk, {.tile = LinearTile::GgufPrefill, .splits = 2}, FloatOutput::BFloat16); },
          "invalid block prefill configuration", "the prefill tile split K");
}

// A rotated projection's scratch holds the bf16 rotated rows of a full decode
// batch (32 rows) in decode and of the token budget (2048) in prefill; an
// unrotated one's holds none.
void scratchBoundsRotated() {
  const Linear linear = gpu(10, 16);
  for (const bool rotated : {false, true}) {
    const ProjectionShape shape{5120, 17408, WeightLayout::Block32, rotated};
    require(linear.decodeScratchSize(shape).rotated == (rotated ? uint64_t{17408} * 32 * 2 : 0) &&
                linear.prefillScratchSize(shape).rotated == (rotated ? uint64_t{17408} * 2048 * 2 : 0),
            "rotated scratch bounds");
  }
}

// The GGUF decode split rules are per-core laws, checked at every core count
// (zero is the fallback), both families and a grid of widths, inputs, epilogues
// and segment counts rather than at the measured machines:
// - a request's sums do not depend on the requests it is batched with: the whole
//   plan is the same at every batch width;
// - a split count is a power of two up to eight whose partitions keep the kernel's
//   floor (register: one 256-input unit, staged: 512 inputs in whole 32-input
//   groups);
// - it depends on the grid per core only: doubling the width and the core count
//   keeps it, more cores never lower it and a wider grid never raises it;
// - the arena bound (the single-tensor plan) covers every segment count.
void ggufCoreLaws() {
  constexpr std::array<uint32_t, 16> widths{256, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120,
                                            6144, 8192, 12288, 16384, 24576, 65536, 248320};
  constexpr std::array<uint32_t, 11> inputs{256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 8192, 12288, 17408};
  for (const uint32_t family : {9U, 10U, 11U})
    for (uint32_t cores = 0; cores <= 128; ++cores) {
      const Linear linear = gpu(family, cores), more = gpu(family, cores + 1), twice = gpu(family, 2 * cores);
      for (const uint32_t n : widths)
        for (const uint32_t k : inputs)
          for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp})
            for (uint32_t segments = 1; segments <= (epilogue == LinearEpilogue::None ? 3U : 1U); ++segments) {
              const Projection p = blockProjection(n, k, segments);
              const auto plan = [&](const Linear &l, uint32_t width, uint32_t rows) {
                return l.plan({{width, k}, rows, LinearPhase::Decode, epilogue}, blockProjection(width, k, segments));
              };
              const LinearPlan one = plan(linear, n, 8);
              const LinearConfig c = one.configuration();
              for (const uint32_t rows : {16U, 24U, 32U}) {
                const LinearPlan wider = plan(linear, n, rows);
                require(wider.configuration() == c && wider.input() == one.input(),
                        "GGUF decode plan depends on the batch width");
              }
              const uint32_t s = c.splits;
              const bool staged = c.tile == LinearTile::GgufStaged;
              require(c.tile == (family == 9 ? LinearTile::GgufRegister : LinearTile::GgufStaged) &&
                          c.groups == 0 && one.groups() == n / 64 && s >= 1 && s <= 8 &&
                          (s & (s - 1)) == 0,
                      "GGUF decode plan tile or split count");
              require(s == 1 || (staged ? k / s >= 512 && (k / 32) % s == 0 : k / 256 / s >= 1),
                      "GGUF decode partition below the kernel floor");
              require(one.storageRows() == 8 && plan(linear, n, 24).storageRows() == (staged ? 32U : 24U),
                      "GGUF decode tile rows");
              if (cores) {
                require(plan(twice, 2 * n, 8).configuration().splits == s,
                        "GGUF split count depends on more than the grid per core");
                require(plan(more, n, 8).configuration().splits >= s,
                        "GGUF split count falls with more cores");
                require(plan(linear, 2 * n, 8).configuration().splits <= s,
                        "GGUF split count rises with the width");
              }
              const LinearScratchSize bound = linear.decodeScratchSize({n, k, WeightLayout::Block32}),
                                      need = linear.plan({{n, k}, 32, LinearPhase::Decode, epilogue}, p).scratchSize();
              require(bound.input >= need.input && bound.sums >= need.sums && bound.partials >= need.partials &&
                          bound.counters >= need.counters,
                      "GGUF decode arena bound below a plan");
            }
    }
  // The float tile follows the grid per core: once a chunk takes the neural
  // accelerator, longer chunks and fewer cores keep it; never below 16 rows or
  // on Apple9.
  for (uint32_t cores = 1; cores <= 128; ++cores)
    for (const uint32_t n : {16U, 64U, 256U, 1024U}) {
      for (const uint32_t family : {10U, 11U}) {
        const Linear linear = gpu(family, cores), more = gpu(family, cores + 1);
        bool accelerator = false;
        for (uint32_t rows = 1; rows <= 2048; ++rows) {
          const bool now = linear.ggufFloatTile(rows, n) == FloatTile::NeuralAccelerator;
          require((!accelerator || now) && (!now || rows >= 16) &&
                      (more.ggufFloatTile(rows, n) == FloatTile::Simdgroup || now),
                  "GGUF float tile is not monotone in rows and cores");
          accelerator = now;
        }
      }
      require(gpu(9, cores).ggufFloatTile(2048, n) == FloatTile::Simdgroup, "Apple9 GGUF float tile");
    }
}

// The Apple10 affine split rule as per-core laws, checked at every core count
// (zero is the fallback), families 10 and 11 and a grid of widths, inputs and
// epilogues, as ggufCoreLaws checks the GGUF rule:
// - a request's sums do not depend on the requests it is batched with: whether
//   a projection splits, and its whole plan when it does, is the same at every
//   batch width;
// - a split count is a power of two up to eight whose partitions each keep one
//   256-input block;
// - it depends on the grid per core only: doubling the width and the core count
//   keeps it, more cores never lower it and a wider grid never raises it.
// The decode arena's bound over every lane count is checked on the production
// geometries (model_execution_plan_test.cpp).
void apple10AffineCoreLaws() {
  constexpr std::array<uint32_t, 14> widths{256, 512, 768, 1024, 1280, 2048, 2560, 4096,
                                            5120, 6144, 9216, 12544, 17408, 248320};
  constexpr std::array<uint32_t, 10> inputs{256, 512, 768, 1024, 2048, 4096, 5120, 6144, 17408, 25600};
  for (const uint32_t family : {10U, 11U})
    for (uint32_t cores = 0; cores <= 128; ++cores) {
      const Linear linear = gpu(family, cores), more = gpu(family, cores + 1), twice = gpu(family, 2 * cores);
      for (const uint32_t n : widths)
        for (const uint32_t k : inputs)
          for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp}) {
            const auto plan = [&](const Linear &l, uint32_t width, uint32_t rows) {
              return l.plan({{width, k}, rows, LinearPhase::Decode, epilogue});
            };
            const LinearPlan one = plan(linear, n, 8);
            const LinearConfig c = one.configuration();
            const bool split = c.tile == LinearTile::Split128;
            for (const uint32_t rows : {16U, 24U, 32U}) {
              const LinearConfig wider = plan(linear, n, rows).configuration();
              require((wider.tile == LinearTile::Split128) == split && (!split || wider == c),
                      "Apple10 split plan depends on the batch width");
            }
            const uint32_t s = c.splits;
            require(split == (s > 1) && s <= 8 && (s & (s - 1)) == 0 &&
                        (!split || (c.groups == 0 && c.simdgroups == LinearSimdgroups::Eight)),
                    "Apple10 split plan tile or split count");
            require(k / 256 >= s, "Apple10 split partition below one 256-input block");
            if (cores) {
              require(plan(twice, 2 * n, 8).configuration().splits == s,
                      "Apple10 split count depends on more than the grid per core");
              require(plan(more, n, 8).configuration().splits >= s, "Apple10 split count falls with more cores");
              require(plan(linear, 2 * n, 8).configuration().splits <= s,
                      "Apple10 split count rises with the width");
            }
          }
    }
}

// Exercise continuous core counts, not just measured SKU anchors. These
// contracts check the tuner's candidates keep the policy's plan first, legal
// grids and no duplicates; they do not claim performance on simulated
// hardware.
void scalingContracts() {
  for (uint32_t family : {9U, 10U, 11U}) {
    for (uint32_t index = 0; index <= 129; ++index) {
      const uint32_t reported = index == 129 ? 4096 : index;
      const DeviceCapabilities device = simulatedDevice(family, reported);
      const Linear linear(device);
      for (uint32_t n : {256U, 5120U, 131072U}) {
        for (uint32_t k : {256U, 768U, 1024U, 4096U, 5120U, 17408U}) {
          for (uint32_t rows : {8U, 16U, 24U, 32U}) {
            for (auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                                 LinearEpilogue::GateUp}) {
              const LinearWorkload w{{n, k}, rows, LinearPhase::Decode, epilogue};
              const auto baseline = linear.plan(w);
              const auto candidates = tuning::linearCandidates(device, w);
              require(candidates.front().configuration() == baseline.configuration(),
                      "core scaling lost or displaced the baseline");
              for (size_t i = 0; i < candidates.size(); ++i) {
                const auto &plan = candidates[i];
                require(plan.groups() > 0 && plan.groups() <= n / plan.tileColumns(),
                        "core scaling produced an invalid grid");
                for (size_t j = 0; j < i; ++j)
                  require(plan.configuration() != candidates[j].configuration(),
                          "core scaling produced duplicate candidates");
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
  auto buffer = test::sharedBuffer(backend, bytes);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

// An n x k segment in `format` at column `offset`, whose planes hold whole
// tiles of its rows, unwritten: encoding reads only their sizes.
QuantizedSegment segmentPlanes(metal::MetalBackend &backend, uint32_t format, uint32_t n, uint32_t k,
                               uint32_t offset = 0) {
  const QuantFormat &f = kQuantFormats[format];
  const uint64_t units = uint64_t{(n + QUANT_TILE_ROWS - 1) / QUANT_TILE_ROWS * QUANT_TILE_ROWS} * (k / 32);
  QuantizedSegment s =
      QuantizedSegment::planes(format, n, k, allocate(backend, units * f.plane0_bytes),
                               allocate(backend, units * f.plane1_bytes),
                               allocate(backend, units / f.meta_groups * f.meta_bytes));
  s.columnOffset = offset;
  return s;
}

// A block projection dispatches only as the plan's matrix, and each segment
// fills whole 64-column tiles, with every buffer the plan needs: the matching
// projection encodes its dispatch, so only the projection can reject.
void ggufProjectionMatrix(metal::MetalBackend &backend) {
  const Linear linear = gpu(10, 16);
  const auto segment = [&](uint32_t n, uint32_t offset) {
    return segmentPlanes(backend, GGUF_FMT_Q4K, n, 17408, offset);
  };
  const Projection matching(5120, 17408, BlockWeights{{segment(5120, 0)}});
  const LinearPlan plan = linear.plan({{5120, 17408}, 8}, matching);
  const uint64_t rows = plan.storageRows();
  const LinearScratchSize scratch = plan.scratchSize();
  const LinearBuffers buffers{
      .input = allocate(backend, rows * 17408 * 2),
      .output = allocate(backend, rows * 5120 * 2),
      .sums = allocate(backend, plan.sumsBytes()),
      .gateScratch = allocate(backend, plan.gateScratchBytes()),
      .downSums = allocate(backend, plan.downSumsBytes()),
      .scratch = {allocate(backend, scratch.input), allocate(backend, scratch.sums),
                  allocate(backend, scratch.partials), allocate(backend, scratch.counters)}};
  {
    metal::CommandGraph graph;
    (void)linear.add(graph, buffers, matching, plan);
    require(graph.dispatches().size() == 1, "the matching block projection did not encode its dispatch");
  }
  // The padding past a projection's segments is part of it, not room for a
  // narrower plan; a segment of 32 columns leaves a partial tile.
  for (const auto &[projection, refusal] :
       {std::pair{Projection(5376, 17408, BlockWeights{{segment(5120, 0)}}), "block projection does not match plan"},
        std::pair{Projection(5120, 17408, BlockWeights{{segment(5088, 0), segment(32, 5088)}}),
                  "block segments do not fill whole column tiles"}}) {
    metal::CommandGraph graph;
    rejects([&] { (void)linear.add(graph, buffers, projection, plan); }, refusal,
            "a block projection that is not the plan's tiles was accepted");
    require(graph.empty(), "a block projection that is not the plan's tiles encoded a dispatch");
  }
  // A projection with an fp32 destination writes fp32 rows through the
  // kernel's fp32 instance; the fused kernels have none.
  Projection head = matching, fused(5120, 17408, BlockWeights{{segment(2560, 0), segment(2560, 2560)}});
  head.destination = fused.destination = FloatOutput::Float32;
  const LinearPlan fp32 = linear.plan({{5120, 17408}, 8}, head);
  LinearBuffers fp32Buffers = buffers;
  fp32Buffers.output = allocate(backend, rows * 5120 * sizeof(float));
  metal::CommandGraph graph;
  rejects([&] { (void)linear.add(graph, buffers, head, fp32); }, "projection output buffer holds",
          "fp32 rows were written into a bf16 output");
  rejects([&] { (void)linear.add(graph, fp32Buffers, fused, linear.plan({{5120, 17408}, 8}, fused)); },
          "an fp32 destination takes a single-tensor block projection", "a fused fp32 block projection was accepted");
  require(graph.empty(), "an invalid fp32 block projection encoded a dispatch");
  (void)linear.add(graph, fp32Buffers, head, fp32);
  require(graph.dispatches().size() == 1 && graph.dispatches()[0].pipelineName == "gguf_decode_q4k_m8_a_f32",
          "the fp32 block projection did not encode its fp32 kernel");
}

// A producer writes the table its consumer's plan reads into scratch that
// holds it: a table layout without that scratch is refused before anything
// is encoded, and Plain runs the plain kernel. Every other buffer holds what
// its kernel reaches.
void producerTableContract(metal::MetalBackend &backend) {
  constexpr uint32_t kLanes = 1, kRows = 8, kHidden = 5120;
  const NormWeights norm{allocate(backend, kHidden * 2)};
  const auto plainKernel = [](const metal::CommandGraph &graph, std::string_view kernel) {
    return graph.dispatches().size() == 1 && graph.dispatches()[0].pipelineName == kernel;
  };
  {
    const metal::MetalBuffer rows = allocate(backend, kRows * kHidden * 2);
    metal::CommandGraph graph;
    rejects([&] { (void)Normalization::addRms(graph, rows, norm, rows, kHidden, kRows, {}, LinearInput::Table64); },
            "linear table buffer holds", "the norm wrote a table without scratch");
    require(graph.empty() &&
                Normalization::addRms(graph, rows, norm, rows, kHidden, kRows).layout == LinearInput::Plain &&
                plainKernel(graph, "norm_rms"),
            "the norm's table contract");
  }
  {
    const GdnShape shape{16, 32, 128, 8192, 12544};
    const uint64_t carried = 3 * uint64_t{shape.convolutionDimension} * 2,
                   recurrent = uint64_t{shape.valueHeads} * shape.headDimension * shape.headDimension * 4;
    const GdnStateStrides strides{carried, recurrent, carried};
    const metal::MetalBuffer state = allocate(backend, carried + recurrent);
    const std::array<metal::MetalBuffer, SPLASH_MAXIMUM_BATCH_WIDTH> states{state, state, state, state};
    GdnDecodeBuffers buffers;
    buffers.packed = allocate(backend, kRows * shape.packedWidth * 2);
    buffers.convolutionWeights = allocate(backend, shape.convolutionDimension * 4 * 2);
    buffers.currentStates = states;
    buffers.nextStates = states;
    buffers.mixed = allocate(backend, kRows * shape.convolutionDimension * 2);
    buffers.decayWeights = allocate(backend, shape.valueHeads * 4);
    buffers.timeBias = allocate(backend, shape.valueHeads * 2);
    buffers.decay = allocate(backend, kRows * shape.valueHeads * 4);
    buffers.beta = allocate(backend, kRows * shape.valueHeads * 2);
    buffers.mixerNorm = {allocate(backend, shape.headDimension * 2)};
    buffers.hidden = allocate(backend, kRows * shape.valueHeads * shape.headDimension * 2);
    metal::CommandGraph graph;
    rejects(
        [&] {
          (void)GDN::addDecode(graph, buffers, shape, kLanes, 0, strides, GdnHeadOrder::Grouped,
                               LinearInput::Table16);
        },
        "linear table buffer holds", "the GDN decode wrote a table without scratch");
    require(graph.empty() &&
                GDN::addDecode(graph, buffers, shape, kLanes, 0, strides, GdnHeadOrder::Grouped, LinearInput::Plain)
                        .layout == LinearInput::Plain &&
                plainKernel(graph, "verify_gdn_fused_vh32"),
            "the GDN decode's table contract");
  }
  {
    // The 27B's attention: 24 query heads over 4 KV heads of 256, and the
    // verify staging of one lane's 8 rows in 32 per KV head.
    const kv::Layout layout{1, 4, 256};
    const metal::MetalBuffer packed = allocate(backend, kRows * (24 * 2 + 4 * 2) * 256 * 2);
    const metal::MetalBuffer attention = allocate(backend, 4 * 32 * 6 * 256 * 2);
    const metal::MetalBuffer hidden = allocate(backend, kRows * 24 * 256 * 2);
    metal::CommandGraph graph;
    rejects(
        [&] {
          (void)PagedAttention::addVerifyGate(graph, packed, attention, hidden, 24, layout, kLanes, {},
                                              LinearInput::Table64);
        },
        "linear table buffer holds", "the attention gate wrote a table without scratch");
    require(graph.empty() &&
                PagedAttention::addVerifyGate(graph, packed, attention, hidden, 24, layout, kLanes, {},
                                              LinearInput::Plain)
                        .layout == LinearInput::Plain &&
                plainKernel(graph, "verify_attention_gate"),
            "the attention gate's table contract");
  }
}

// A rotated projection prepares its register tile's table from the rotated
// rows (LinearGguf.cpp), so its plan asks its producer for plain rows: on
// Apple9 a rotated PQ2_0 projection keeps the register tile and its input
// norm runs the plain kernel, where an unrotated one reads Table16.
void rotatedRegisterInput(metal::MetalBackend &backend) {
  const Linear m3 = gpu(9, 40);
  Projection rotated = blockProjection(5120, 17408, 1, GGUF_FMT_PQ20);
  const Projection plain = rotated;
  rotated.rotation.signs = allocate(backend, 17408);
  const LinearScratch scratch{.input = allocate(backend, tableBytes(17408, 32)),
                              .sums = allocate(backend, tableSumsBytes(LinearInput::Table16, 17408, 32))};
  const NormWeights norm{allocate(backend, 17408 * 2)};
  const metal::MetalBuffer rows = allocate(backend, uint64_t{32} * 17408 * 2);
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    const LinearPlan plan = m3.decodePlan(rotated, lanes);
    require(plan.configuration().tile == LinearTile::GgufRegister && plan.input() == LinearInput::Plain &&
                m3.decodePlan(plain, lanes).input() == LinearInput::Table16,
            "a rotated register plan's producer writes a table it discards");
    metal::CommandGraph graph;
    (void)Normalization::addRms(graph, rows, norm, rows, 17408, lanes * 8, scratch, plan.input());
    require(graph.dispatches().back().pipelineName.find("_table") == std::string::npos,
            "a rotated register plan's input norm wrote a table");
  }
}

// Each buffer only a block projection's dispatches reach, at its extent and
// one element short: a rotated projection's int8 sign of every input and its
// rotated rows of the plan's storage; a float projection's fp32 weights, its
// input rows and its output rows of 512 columns up to its last column, 192;
// and the planes of a quantized segment.
void blockExtents(metal::MetalBackend &backend) {
  const Linear linear = gpu(10, 16);
  constexpr uint32_t n = 256, k = 2048;
  Projection rotated(n, k, BlockWeights{{segmentPlanes(backend, GGUF_FMT_PQ20, n, k)}});
  rotated.rotation.signs = allocate(backend, k);
  const LinearPlan plan = linear.decodePlan(rotated, 1);
  const uint64_t rows = plan.storageRows();
  const LinearScratchSize scratch = plan.scratchSize();
  const LinearBuffers buffers{.input = allocate(backend, rows * k * 2),
                              .output = allocate(backend, rows * n * 2),
                              .scratch = {allocate(backend, scratch.input), allocate(backend, scratch.sums),
                                          allocate(backend, scratch.partials), allocate(backend, scratch.counters),
                                          allocate(backend, rows * k * 2)}};
  requireExtent(backend, rotated.rotation.signs, k, 1, "rotation sign",
                [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                  Projection changed = rotated;
                  changed.rotation.signs = view;
                  (void)linear.add(graph, buffers, changed, plan);
                });
  requireExtent(backend, buffers.scratch.rotated, rows * k * 2, 2, "projection rotated input",
                [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                  LinearBuffers changed = buffers;
                  changed.scratch.rotated = view;
                  (void)linear.add(graph, changed, rotated, plan);
                });

  constexpr uint32_t floatRows = 37, floatColumns = 64, outStride = 512, outOffset = 128;
  const QuantizedSegment weights =
      QuantizedSegment::floats(floatColumns, k, allocate(backend, uint64_t{floatColumns} * k * 4));
  const metal::MetalBuffer input = allocate(backend, uint64_t{floatRows} * k * 2),
                           output = allocate(backend, uint64_t{floatRows} * outStride * 4);
  const auto project = [&](metal::CommandGraph &graph, const metal::MetalBuffer &rowsIn,
                           const QuantizedSegment &segment, const metal::MetalBuffer &rowsOut) {
    addGgufFloat(graph, rowsIn, segment, rowsOut, floatRows, outStride, outOffset, FloatOutput::Float32,
                 FloatTile::Simdgroup);
  };
  requireExtent(backend, weights.plane0, uint64_t{floatColumns} * k * 4, 4, "float projection weight",
                [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                  project(graph, input, QuantizedSegment::floats(floatColumns, k, view), output);
                });
  requireExtent(backend, input, uint64_t{floatRows} * k * 2, 2, "float projection input",
                [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                  project(graph, view, weights, output);
                });
  requireExtent(backend, output, (uint64_t{floatRows - 1} * outStride + outOffset + floatColumns) * 4, 4,
                "float projection output", [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                  project(graph, input, weights, view);
                });

  // A fused projection of a 320-row Q5_K segment and a 192-row Q4_K one.
  // Each plane is tiles of 256 rows by groups of 32 inputs, or meta units,
  // and ends at the last row's unit of the last group: row 63 of the Q5_K
  // segment's second tile, row 191 of the Q4_K segment's first.
  const std::array segments{segmentPlanes(backend, GGUF_FMT_Q5K, 320, k),
                            segmentPlanes(backend, GGUF_FMT_Q4K, 192, k, 320)};
  const Projection fused(512, k, BlockWeights{{segments[0], segments[1]}});
  const LinearPlan fusedPlan = linear.decodePlan(fused, 1);
  const LinearScratchSize fusedScratch = fusedPlan.scratchSize();
  const LinearBuffers fusedBuffers{.input = allocate(backend, fusedPlan.storageRows() * k * 2),
                                   .output = allocate(backend, fusedPlan.storageRows() * 512 * 2),
                                   .scratch = {allocate(backend, fusedScratch.input),
                                               allocate(backend, fusedScratch.sums),
                                               allocate(backend, fusedScratch.partials),
                                               allocate(backend, fusedScratch.counters)}};
  for (size_t index = 0; index < segments.size(); ++index) {
    const QuantFormat &format = kQuantFormats[segments[index].formatId];
    const uint64_t groups = k / 32, units = groups / format.meta_groups;
    const auto lastUnit = [&](uint64_t blocks) {
      return index == 0 ? (2 * blocks - 1) * 256 + 64 : (blocks - 1) * 256 + 192;
    };
    for (const auto &[member, bytes, element, name] :
         std::initializer_list<std::tuple<metal::MetalBuffer QuantizedSegment::*, uint64_t, uint64_t, const char *>>{
             {&QuantizedSegment::plane0, lastUnit(groups) * format.plane0_bytes, format.plane0_bytes,
              "projection plane0"},
             {&QuantizedSegment::plane1, lastUnit(groups) * format.plane1_bytes, format.plane1_bytes,
              "projection plane1"},
             {&QuantizedSegment::meta, lastUnit(units) * format.meta_bytes, format.meta_bytes, "projection meta"}}) {
      if (!bytes) continue;
      requireExtent(backend, segments[index].*member, bytes, element, name,
                    [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                      std::array changed = segments;
                      changed[index].*member = view;
                      (void)linear.add(graph, fusedBuffers, Projection(512, k, BlockWeights{{changed[0], changed[1]}}),
                                       fusedPlan);
                    });
    }
  }
}

std::array<uint64_t, 3> projectionFingerprint(const Projection &projection) {
  std::array<uint64_t, 3> result{};
  const std::array buffers{projection.affine().weights, projection.affine().scales, projection.affine().biases};
  for (size_t slot = 0; slot < buffers.size(); ++slot) {
    const auto *bytes = static_cast<const uint8_t *>(buffers[slot].contents());
    uint64_t hash = 14695981039346656037ULL;
    for (uint64_t i = 0; i < buffers[slot].sizeBytes(); ++i)
      hash = (hash ^ bytes[i]) * 1099511628211ULL;
    result[slot] = hash;
  }
  return result;
}

// Scalar reference uses the actual packed storage tiles' bytes and FP32
// per-group affine accumulation, including the BF16 boundary before each
// epilogue.
float affineReference(const Projection &p, const uint16_t *input,
                        uint32_t row, uint32_t column) {
  const auto *weights = static_cast<const uint8_t *>(p.affine().weights.contents());
  const auto *scales = static_cast<const uint16_t *>(p.affine().scales.contents());
  const auto *biases = static_cast<const uint16_t *>(p.affine().biases.contents());
  const uint32_t groups = p.inputSize / 64;
  constexpr uint32_t tile = SPLASH_AFFINE_TILE_ROWS;
  float result = 0;
  for (uint32_t group = 0; group < groups; ++group) {
    const uint64_t parameter = (uint64_t{column / tile} * groups + group) * tile + column % tile;
    float partial = 0, sum = 0;
    for (uint32_t k = 0; k < 64; ++k) {
      const uint8_t byte = weights[parameter * 32 + k / 2];
      const uint32_t quantized = (byte >> ((k & 1) * 4)) & 15;
      const float x = tuning::bf16ToFloat(input[uint64_t{row} * p.inputSize + group * 64 + k]);
      sum += x;
      partial += x * quantized;
    }
    result += partial * tuning::bf16ToFloat(scales[parameter]) + sum * tuning::bf16ToFloat(biases[parameter]);
  }
  return tuning::bf16ToFloat(tuning::floatToBf16(result));
}

void checkReference(const Projection &p, const Projection &gate,
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
        residualValue = tuning::bf16ToFloat(residual[index]);
        expected += residualValue;
      }
      if (workload.epilogue == LinearEpilogue::GateUp ||
          workload.epilogue == LinearEpilogue::UpWithGate) {
        gateValue = workload.epilogue == LinearEpilogue::GateUp
            ? affineReference(gate, input, row, column) : tuning::bf16ToFloat(gateValues[index]);
        expected *= gateValue / (1 + std::exp(-gateValue));
      }
      expected = tuning::bf16ToFloat(tuning::floatToBf16(expected));
      // The oracle accumulates in the sequential kernel's order. A split tile
      // reassociates that sum, so it is held to the derived bf16 bound
      // (tuning/LinearNumerics.hpp) on top of the oracle's own margin.
      float tolerance = 0.004f;
      if (split)
        tolerance += tuning::splitTolerance(workload.epilogue,
            {expected, residualValue, gateValue, projection}, slack);
      const float actual = tuning::bf16ToFloat(output[index]);
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
          expected += tuning::bf16ToFloat(output[uint64_t{row} * p.outputSize + group * 64 + k]);
        const uint64_t index = uint64_t{row / 32} * 32 * quantGroups + group * 32 + row % 32;
        require(std::abs(sums[index] - expected) <= 1e-6 * std::max(1.0, std::abs(expected)),
                "fused prefill output sums have wrong layout/value");
      }
    }
  }
}

// requireExtent for each scratch field of the `scratch` bytes a plan uses,
// add(graph, buffers) encoding the plan with that field cut.
template <class Add>
void scratchExtents(metal::MetalBackend &backend, const LinearBuffers &buffers, LinearScratchSize scratch,
                    const Add &add) {
  for (const auto &[member, bytes, element, name] :
       std::initializer_list<std::tuple<metal::MetalBuffer LinearScratch::*, uint64_t, uint64_t, const char *>>{
           {&LinearScratch::input, scratch.input, 2, "projection scratch table"},
           {&LinearScratch::sums, scratch.sums, 4, "projection scratch sums"},
           {&LinearScratch::partials, scratch.partials, 4, "projection partials"},
           {&LinearScratch::counters, scratch.counters, 4, "projection counters"}}) {
    if (!bytes) continue;
    requireExtent(backend, buffers.scratch.*member, bytes, element, name,
                  [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                    LinearBuffers changed = buffers;
                    changed.scratch.*member = view;
                    add(graph, changed);
                  });
  }
}

// Each buffer and weight plane a plan's dispatches reach, at its extent and
// one element short: the input, output and residual rows of the plan's
// storage, the sums, gate scratch and down sums its epilogue reads or writes,
// every scratch field its tile uses, and the Q4 weights of the projection and
// of a gate/up plan's gate, with a scale and a bias per 64 values.
void bufferContracts(metal::MetalBackend &backend, Linear &linear,
                        LinearBuffers buffers, const Projection &p,
                        const Projection &gate, const LinearPlan &plan) {
  const bool gateUp = plan.workload().epilogue == LinearEpilogue::GateUp;
  const auto [n, k] = plan.workload().matrix;
  const uint64_t rows = plan.storageRows(), outputElement = elementBytes(plan.destination());
  const auto add = [&](metal::CommandGraph &graph, LinearBuffers b,
                        const Projection &projection, const Projection *g) {
    linear.add(graph, b, projection, plan, g);
  };
  const LinearScratchSize scratch = plan.scratchSize();
  for (const auto &[member, bytes, element, name] :
       std::initializer_list<std::tuple<metal::MetalBuffer LinearBuffers::*, uint64_t, uint64_t, const char *>>{
           {&LinearBuffers::input, rows * k * 2, 2, "projection input"},
           {&LinearBuffers::output, rows * n * outputElement, outputElement, "projection output"},
           {&LinearBuffers::residual,
            plan.workload().epilogue == LinearEpilogue::Residual ? rows * n * 2 : 0, 2, "projection residual"},
           {&LinearBuffers::sums, plan.sumsBytes(), 4, "projection sums"},
           {&LinearBuffers::gateScratch, plan.gateScratchBytes(), 2, "projection gate scratch"},
           {&LinearBuffers::downSums, plan.downSumsBytes(), 4, "projection down sums"}}) {
    if (!bytes) continue;
    requireExtent(backend, buffers.*member, bytes, element, name,
                  [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                    LinearBuffers changed = buffers;
                    changed.*member = view;
                    add(graph, changed, p, gateUp ? &gate : nullptr);
                  });
  }
  // Every scratch field the plan uses (the Q4 register tile's table, sums,
  // partials and counters; the Split128 partials and counters).
  scratchExtents(backend, buffers, scratch, [&](metal::CommandGraph &graph, const LinearBuffers &changed) {
    add(graph, changed, p, gateUp ? &gate : nullptr);
  });
  const uint64_t parameters = uint64_t{n} * (k / 64) * 2;
  for (const bool ofGate : {false, true}) {
    if (ofGate && !gateUp) continue;
    const Projection &weights = ofGate ? gate : p;
    for (const auto &[member, bytes, element, name] :
         std::initializer_list<std::tuple<metal::MetalBuffer AffineWeights::*, uint64_t, uint64_t, const char *>>{
             {&AffineWeights::weights, uint64_t{n} * k / 2, 1, "projection weight"},
             {&AffineWeights::scales, parameters, 2, "projection scale"},
             {&AffineWeights::biases, parameters, 2, "projection bias"}})
      requireExtent(backend, weights.affine().*member, bytes, element, name,
                    [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                      AffineWeights planes = weights.affine();
                      planes.*member = view;
                      const Projection changed(weights.outputSize, weights.inputSize, planes);
                      add(graph, buffers, ofGate ? p : changed, gateUp ? (ofGate ? &changed : &gate) : nullptr);
                    });
  }
  auto mismatch = p;
  mismatch.inputSize += 256;
  metal::CommandGraph graph;
  rejects([&] { add(graph, buffers, mismatch, gateUp ? &gate : nullptr); },
          "affine projection does not match plan", "a projection of another matrix was accepted");
  rejects([&] { add(graph, buffers, p, gateUp ? nullptr : &gate); },
          "a gate/up plan takes a gate projection and no other plan does",
          "a gate projection was given to the wrong plan or withheld from gate/up");
  require(graph.empty(), "invalid Linear gate/projection partially encoded graph");
  if (plan.workload().phase == LinearPhase::Prefill) {
    const auto workload = plan.workload();
    // The input sums of the plan's 32-row tiles.
    requireExtent(backend, buffers.input, rows * k * 2, 2, "projection input",
                  [&](metal::CommandGraph &sums, const metal::MetalBuffer &view) {
                    linear.addPrefillSums(sums, view, buffers.sums, p, workload.rows);
                  });
    requireExtent(backend, buffers.sums, rows * (k / 64) * 4, 4, "projection sums",
                  [&](metal::CommandGraph &sums, const metal::MetalBuffer &view) {
                    linear.addPrefillSums(sums, buffers.input, view, p, workload.rows);
                  });
    for (const LinearMatrix matrix : {LinearMatrix{0, workload.matrix.inputSize},
           LinearMatrix{128, workload.matrix.inputSize},
           LinearMatrix{workload.matrix.outputSize, 0},
           LinearMatrix{workload.matrix.outputSize, 63}})
      rejects([&] { linear.addPrefillSums(graph, buffers.input, buffers.sums,
                                         Projection(matrix.outputSize, matrix.inputSize, p.affine()),
                                         workload.rows); },
              "invalid linear matrix", "prefill sums of a matrix of no or partial tiles or quant groups were accepted");
    for (uint32_t rows : {0U, SPLASH_PREFILL_TOKEN_BUDGET + 1U})
      rejects([&] { linear.addPrefillSums(graph, buffers.input, buffers.sums, p, rows); },
              "invalid linear prefill workload", "prefill sums of no rows or past the budget were accepted");
    require(graph.empty(), "invalid prefill sums input partially encoded graph");
  }
}

// Every buffer of the Q4 register tile's plans of each epilogue and the scratch
// of a GGUF register plan, at their extents and one element short: the
// Apple9 decode tiles that read an activation table, on plans of a 40-core
// Apple9 GPU that split K. requireExtent only encodes, so this runs on any
// device.
void registerTileExtents(metal::MetalBackend &backend) {
  Linear m3 = gpu(9, 40);
  constexpr uint32_t n = 512, k = 2048, lanes = 2;
  constexpr uint64_t rows = lanes * 8;
  const auto tableAndSplits = [](const LinearScratchSize &s) {
    return s.input && s.sums && s.partials && s.counters;
  };
  const Projection p = test::deterministicQ4Projection(backend, {n, k}, 31),
                   gate = test::deterministicQ4Projection(backend, {n, k}, 157);
  for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp}) {
    const LinearPlan plan = m3.plan({{n, k}, rows, LinearPhase::Decode, epilogue});
    const LinearScratchSize scratch = plan.scratchSize();
    require(tableAndSplits(scratch), "the affine Apple9 plan reads no table or does not split K");
    bufferContracts(backend, m3,
                    {.input = allocate(backend, rows * k * 2),
                     .output = allocate(backend, rows * n * 2),
                     .residual = allocate(backend, epilogue == LinearEpilogue::Residual ? rows * n * 2 : 0),
                     .scratch = {allocate(backend, scratch.input), allocate(backend, scratch.sums),
                                 allocate(backend, scratch.partials), allocate(backend, scratch.counters)}},
                    p, gate, plan);
  }
  const Projection block(n, k, BlockWeights{{segmentPlanes(backend, GGUF_FMT_Q4K, n, k)}});
  const LinearPlan plan = m3.decodePlan(block, lanes);
  const LinearScratchSize scratch = plan.scratchSize();
  require(plan.configuration().tile == LinearTile::GgufRegister && tableAndSplits(scratch),
          "the GGUF Apple9 plan is not a register plan that splits K");
  scratchExtents(backend,
                 {.input = allocate(backend, rows * k * 2),
                  .output = allocate(backend, rows * n * 2),
                  .scratch = {allocate(backend, scratch.input), allocate(backend, scratch.sums),
                              allocate(backend, scratch.partials), allocate(backend, scratch.counters)}},
                 scratch, [&](metal::CommandGraph &graph, const LinearBuffers &buffers) {
                   (void)m3.add(graph, buffers, block, plan);
                 });
}

// Every tuning candidate of the workload against the CPU reference and the
// others; a plain decode kernel of `floatKernels` also into fp32.
void numericalCase(metal::MetalBackend &backend, Linear &linear,
                      const std::set<std::string_view> &floatKernels,
                      const Projection &p, const Projection &gate,
                      LinearWorkload workload, bool inPlaceResidual = false) {
  require(!inPlaceResidual || workload.epilogue == LinearEpilogue::Residual,
          "in-place residual fixture requires residual epilogue");
  const auto candidates = tuning::linearCandidates(backend.capabilities(), workload);
  const uint32_t storageRows = candidates[0].storageRows();
  auto input = allocate(backend, uint64_t{storageRows} * p.inputSize * 2);
  auto *inputValues = static_cast<uint16_t *>(input.contents());
  for (uint64_t i = 0; i < uint64_t{workload.rows} * p.inputSize; ++i)
    inputValues[i] = tuning::floatToBf16(float(int(mix(uint32_t(i) + 1949) % 257) - 128) / 257.0f);
  // Sequential candidates share their output bytes; split-K candidates are
  // held to the derived bound against them once every candidate has run.
  std::vector<uint16_t> baseline;
  struct SplitOutput final {
    std::vector<uint16_t> output, residual;
    std::string_view pipeline;
    bool q4Register = false;
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
        residual[i] = tuning::floatToBf16(float(int(mix(uint32_t(i) + 7919) % 257) - 128) / 257.0f);
    }
    bufferContracts(backend, linear, b, p, gate, plan);
    metal::CommandGraph graph;
    if (workload.phase == LinearPhase::Prefill)
      linear.addPrefillSums(graph, input, b.sums, p, workload.rows);
    if (workload.epilogue == LinearEpilogue::UpWithGate) {
      const auto gatePlan = linear.plan({workload.matrix, workload.rows, LinearPhase::Prefill,
                                         LinearEpilogue::None});
      linear.add(graph, {input, b.gateScratch, b.sums, {}, {}, {}}, gate, gatePlan);
    }
    const size_t prepasses = graph.dispatches().size();
    linear.add(graph, b, p, plan, workload.epilogue == LinearEpilogue::GateUp ? &gate : nullptr);
    const auto &last = graph.dispatches().back();
    require(last.pipelineName == (plan.secondPipeline().empty() ? plan.pipeline() : plan.secondPipeline()),
            "production dispatch differs from Linear plan");
    for (const auto &dispatch : graph.dispatches().subspan(prepasses))
      require(dispatch.threadsPerThreadgroup.x == plan.threadsPerThreadgroup() &&
                  dispatch.threadsPerThreadgroup.y == 1 && dispatch.threadsPerThreadgroup.z == 1,
              "production dispatch threads differ from Linear plan scope");
    if (workload.phase == LinearPhase::Decode) {
      const uint32_t dispatches = plan.usesQ4Register() ? 2 : plan.secondPipeline().empty() ? 1 : 2;
      require(last.threadgroups.x == plan.groups() &&
                  graph.dispatches().size() == dispatches,
              "Linear decode plan/graph geometry mismatch");
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
    (void)backend.submitCommandAsync(graph.dispatches()).wait();
    if (workload.phase == LinearPhase::Decode && workload.rows == 24) {
      const auto firstOutput = snapshot(b.output);
      if (inPlaceResidual)
        std::memcpy(b.residual.contents(), immutableResidual.data(),
                    immutableResidual.size() * sizeof(uint16_t));
      (void)backend.submitCommandAsync(graph.dispatches()).wait();
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
                     plan.configuration().splits > 1 || plan.usesQ4Register(),
                     plan.usesQ4Register() ? std::max(tuning::q4RegisterSlack(workload, b.input, p),
                         tuning::q4RegisterSlack(workload, b.input, gate)) : 0);
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
    if (workload.phase == LinearPhase::Decode && workload.epilogue == LinearEpilogue::None &&
        floatKernels.contains(plan.pipeline())) {
      // The fp32 instance (the logits) holds the values the bf16 one
      // rounds, bit for bit, and writes nothing past its rows.
      const LinearPlan fp32 = Linear::plan(workload, plan.configuration(), FloatOutput::Float32);
      const uint64_t fp32Bytes = elements * sizeof(float);
      auto fp32Backing = allocate(backend, fp32Bytes + guardBytes);
      std::memset(static_cast<uint8_t *>(fp32Backing.contents()) + fp32Bytes, 0x5a, guardBytes);
      LinearBuffers fp32Buffers = b;
      fp32Buffers.output = backend.view(fp32Backing, 0, fp32Bytes);
      metal::CommandGraph fp32Graph;
      linear.add(fp32Graph, fp32Buffers, p, fp32);
      require(fp32Graph.dispatches().back().pipelineName == kernelInstance(plan.pipeline(), FloatOutput::Float32),
              "an fp32 plan did not dispatch its kernel's fp32 instance");
      (void)backend.submitCommandAsync(fp32Graph.dispatches()).wait();
      checkGuard(fp32Backing, fp32Bytes, "fp32 output");
      const auto *values = static_cast<const float *>(fp32Buffers.output.contents());
      for (uint64_t i = 0; i < elements; ++i)
        if (tuning::floatToBf16(values[i]) != output[i]) {
          std::cerr << "fp32 element=" << i << " value=" << values[i] << " bf16=" << tuning::bf16ToFloat(output[i])
                    << " pipeline=" << fp32.pipeline() << '\n';
          throw std::runtime_error("an fp32 output does not round to its bf16 plan's output");
        }
    }
    if (plan.configuration().splits > 1 || plan.usesQ4Register()) {
      splitOutputs.push_back({{output, output + elements}, immutableResidual, plan.pipeline(), plan.usesQ4Register()});
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
    const auto plain = Linear::plan({workload.matrix, workload.rows, LinearPhase::Decode,
                                       LinearEpilogue::None},
                                      {LinearTile::N128, p.outputSize / 128}, FloatOutput::BFloat16);
    auto gateOutput = allocate(backend, uint64_t{storageRows} * p.outputSize * 2);
    auto upOutput = allocate(backend, uint64_t{storageRows} * p.outputSize * 2);
    metal::CommandGraph graph;
    linear.add(graph, {input, gateOutput, {}, {}, {}, {}}, gate, plain);
    linear.add(graph, {input, upOutput, {}, {}, {}, {}}, p, plain);
    (void)backend.submitCommandAsync(graph.dispatches()).wait();
    const auto *g = static_cast<const uint16_t *>(gateOutput.contents());
    const auto *u = static_cast<const uint16_t *>(upOutput.contents());
    gateReference.assign(g, g + baseline.size());
    upReference.assign(u, u + baseline.size());
  }
  float maxAbs = 0;
  for (const uint16_t value : baseline) maxAbs = std::max(maxAbs, std::fabs(tuning::bf16ToFloat(value)));
  const float slack = tuning::reassociationSlack(p.inputSize, maxAbs);
  const float operandSlack = std::max(tuning::q4RegisterSlack(workload, input, p),
                                      tuning::q4RegisterSlack(workload, input, gate));
  for (const auto &split : splitOutputs) {
    const float toleranceSlack = slack + (split.q4Register ? operandSlack : 0);
    for (uint64_t i = 0; i < uint64_t{workload.rows} * p.outputSize; ++i) {
      tuning::SplitReference reference{tuning::bf16ToFloat(baseline[i])};
      if (workload.epilogue == LinearEpilogue::Residual) reference.residual = tuning::bf16ToFloat(split.residual[i]);
      if (workload.epilogue == LinearEpilogue::GateUp) {
        reference.gate = tuning::bf16ToFloat(gateReference[i]);
        reference.up = tuning::bf16ToFloat(upReference[i]);
      }
      if (!tuning::withinSplitTolerance(tuning::bf16ToFloat(split.output[i]), workload.epilogue, reference,
                                        toleranceSlack)) {
        std::cerr << "split element=" << i << " actual=" << tuning::bf16ToFloat(split.output[i])
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

// Every plain decode kernel of floatOutputPlans has the fp32 instance its
// fp32 plans run.
void floatInstances(const char *metallib, const std::set<std::string_view> &kernels) {
  id<MTLLibrary> library = [MTLCreateSystemDefaultDevice() newLibraryWithURL:
      [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib]] error:nil];
  require(library != nil, "could not load the Linear library");
  for (const std::string_view kernel : kernels)
    require([library newFunctionWithName:[NSString stringWithUTF8String:
                kernelInstance(kernel, FloatOutput::Float32).c_str()]] != nil,
            "a plain decode kernel has no fp32 instance");
}

// Every Linear pipeline a tuning candidate of any family and core count
// launches, the policy's plain decode plans also into fp32, with its threads
// per threadgroup: one pipeline never takes two execution scopes.
// Device-free.
std::map<std::string, uint32_t> pipelineScopes() {
  std::map<std::string, uint32_t> names{{"prefill_linear_q4_sums32", 256}};
  const auto collect = [&](const DeviceCapabilities &device, LinearWorkload workload) {
    std::vector<LinearPlan> plans = tuning::linearCandidates(device, workload);
    if (workload.phase == LinearPhase::Decode && workload.epilogue == LinearEpilogue::None)
      plans.push_back(Linear::plan(workload, plans.front().configuration(), FloatOutput::Float32));
    for (const auto &plan : plans)
      for (auto name : {plan.pipeline(), plan.secondPipeline()}) {
        if (name.empty()) continue;
        const auto [found, inserted] = names.emplace(kernelInstance(name, plan.destination()),
                                                     plan.threadsPerThreadgroup());
        require(inserted || found->second == plan.threadsPerThreadgroup(),
                "one Linear pipeline was assigned incompatible execution scopes");
      }
  };
  for (const uint32_t family : {9U, 10U, 11U})
    for (const uint32_t cores : {0U, 10U, 16U, 20U, 40U, 80U}) {
      const DeviceCapabilities device = simulatedDevice(family, cores);
      for (uint32_t lanes = 1; lanes <= 4; ++lanes)
        for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                                   LinearEpilogue::GateUp})
          collect(device, {{16640, 5120}, lanes * 8, LinearPhase::Decode, epilogue});
      for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                                 LinearEpilogue::UpWithGate})
        collect(device, {{16640, 5120}, 33, LinearPhase::Prefill, epilogue});
    }
  return names;
}

// This device's resources for every pipeline of pipelineScopes: its static
// threadgroup memory fits the device, its thread limit covers the plans'
// threads, and SIMD groups are 32 wide. Shader validation instruments the
// pipelines and inflates these numbers, so this runs without it.
void pipelineCapabilities(const char *metallib, const DeviceCapabilities &capabilities,
                          const std::map<std::string, uint32_t> &names) {
  require(!std::getenv("MTL_SHADER_VALIDATION"),
          "the pipeline resource check inspects production pipelines: run it without MTL_SHADER_VALIDATION");
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
    const bool capabilities = argc == 3 && std::string_view(argv[1]) == "--capabilities";
    require(argc == 2 || capabilities,
            "usage: linear-plan <production.metallib|--cpu|--capabilities production.metallib>");
    if (capabilities) {
      metal::MetalBackend backend(argv[2]);
      pipelineCapabilities(argv[2], backend.capabilities(), pipelineScopes());
      return 0;
    }
    baselinePlans();
    affinePolicyLaws();
    ggufPlans();
    scratchBoundsRotated();
    ggufCoreLaws();
    const std::set<std::string_view> plainKernels = floatOutputPlans();
    apple10AffineCoreLaws();
    scalingContracts();
    // Apple9 at the assumed core count, and Apple10, which lists Split128.
    planContracts(9, 0);
    planContracts(10, 16);
    // The resources behind these scopes are --capabilities' check.
    static_cast<void>(pipelineScopes());
    if (std::string_view(argv[1]) == "--cpu") {
      std::cout << "Linear CPU plans: PASS\n";
      return 0;
    }
    metal::MetalBackend backend(argv[1]);
    floatInstances(argv[1], plainKernels);
    ggufProjectionMatrix(backend);
    producerTableContract(backend);
    rotatedRegisterInput(backend);
    blockExtents(backend);
    registerTileExtents(backend);
    Linear linear(backend.capabilities());
    for (const LinearMatrix matrix : {LinearMatrix{512, 256}, LinearMatrix{768, 768},
                                      LinearMatrix{16640, 5120}, LinearMatrix{12544, 2048},
                                      LinearMatrix{5120, 17408},
                                      LinearMatrix{256, 64}, LinearMatrix{512, 320}}) {
      const auto p = test::deterministicQ4Projection(backend, matrix, 31);
      const auto gate = test::deterministicQ4Projection(backend, matrix, 157);
      const auto immutableProjection = projectionFingerprint(p);
      const auto immutableGateProjection = projectionFingerprint(gate);
      if (matrix.inputSize % 256 == 0)
        for (uint32_t lanes = 1; lanes <= 4; ++lanes)
          for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                                     LinearEpilogue::GateUp}) {
            numericalCase(backend, linear, plainKernels, p, gate,
                          {matrix, lanes * 8, LinearPhase::Decode, epilogue});
            if (epilogue == LinearEpilogue::Residual)
              numericalCase(backend, linear, plainKernels, p, gate,
                            {matrix, lanes * 8, LinearPhase::Decode, epilogue}, true);
          }
      for (const uint32_t rows : {1U, 33U, 2048U})
        for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                                   LinearEpilogue::UpWithGate})
          numericalCase(backend, linear, plainKernels, p, gate,
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

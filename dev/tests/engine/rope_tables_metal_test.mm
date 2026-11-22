// The target's RoPE rows of a decode step (ops::RoPE::addTreeTables): each
// lane's rows, as its selector lays them out on the device, at its anchor's
// position plus their depths in its tree, against rope_build_tables' rows at
// the same positions, bit for bit, for every family's rotary layout.
//
//   rope-tables METALLIB

#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "model/ModelDescriptor.hpp"
#include "model/RopeScaling.hpp"
#include "ops/LaneRows.hpp"
#include "ops/RoPE.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace splash;
using test::rejects;
using test::require;

// A family's rotary layout: its pairs and base, and one scaling it serves
// (none, or YaRN's to its most context).
struct RotaryLayout final {
  std::string name;
  uint32_t pairs = 0;
  float theta = 0.0F;
  model::RopeScaling scaling{};
};

std::vector<RotaryLayout> rotaryLayouts() {
  std::vector<RotaryLayout> layouts;
  model::forEachFamily([&](const auto &family) {
    // A family none of whose layers rotates has no tables.
    if (!family.rotaryPairs) return;
    layouts.push_back({std::string(family.family), family.rotaryPairs, family.rotaryTheta, {}});
    if (family.yarnContextTokens > family.nativeContextTokens)
      layouts.push_back({std::string(family.family) + " YaRN", family.rotaryPairs, family.rotaryTheta,
                         {.factor = static_cast<double>(family.yarnContextTokens) / family.nativeContextTokens,
                          .originalContextTokens = family.nativeContextTokens}});
  });
  return layouts;
}

// A decode step's lane: its anchor's RoPE position, and its rows' parents in
// its tree, in preorder (row 0 the anchor, each other row after its parent).
struct Lane final {
  uint32_t anchor = 0;
  std::vector<uint16_t> parents;
};

// A step's lanes packed in lane order: its layout (SplashLaneRows a lane),
// tree rows (SplashTreeRow a packed row) and each row's position, the
// lane's anchor plus the row's depth.
struct Step final {
  ops::StepRows rows{};
  std::vector<ops::LaneRows> layout;
  std::vector<SplashTreeRow> tree;
  std::vector<uint32_t> anchors;
  std::vector<uint32_t> positions;
};

Step stepOf(const std::vector<Lane> &lanes, uint32_t laneCap) {
  Step step;
  for (const Lane &lane : lanes) {
    const auto first = static_cast<uint32_t>(step.tree.size());
    step.layout.push_back({first, static_cast<uint32_t>(lane.parents.size())});
    step.anchors.push_back(lane.anchor);
    for (uint32_t row = 0; row < lane.parents.size(); ++row) {
      const uint16_t parent = lane.parents[row];
      const SplashTreeRow up = row ? step.tree[first + parent] : SplashTreeRow{0, 0, 0};
      const auto depth = static_cast<uint16_t>(row ? up.depth + 1 : 0);
      step.tree.push_back({(row ? up.ancestors : 0U) | 1U << row, row ? parent : uint16_t{SPLASH_TREE_ROOT}, depth});
      step.positions.push_back(lane.anchor + depth);
    }
  }
  step.rows = {static_cast<uint32_t>(lanes.size()), static_cast<uint32_t>(step.tree.size()), laneCap};
  return step;
}

// A chain of `rows` rows, and a tree of `rows` rows whose row r follows row
// (r - 1) / 2 (two children a row, depths of about log2 rows).
std::vector<uint16_t> chain(uint32_t rows) {
  std::vector<uint16_t> parents(rows);
  for (uint32_t row = 1; row < rows; ++row) parents[row] = static_cast<uint16_t>(row - 1);
  return parents;
}
std::vector<uint16_t> branching(uint32_t rows) {
  std::vector<uint16_t> parents(rows);
  for (uint32_t row = 1; row < rows; ++row) parents[row] = static_cast<uint16_t>((row - 1) / 2);
  return parents;
}

template <class T> metal::MetalBuffer upload(metal::MetalBackend &backend, std::span<const T> values) {
  metal::MetalBuffer buffer = test::sharedBuffer(backend, std::max<uint64_t>(values.size_bytes(), 4));
  std::memcpy(buffer.contents(), values.data(), values.size_bytes());
  return buffer;
}

constexpr uint32_t kCanary = 0xa5a5a5a5U;

// Steps of every shape, their lanes' anchors over the sampled positions:
// cycles from position 0 on, random ones up to 131071, and chains whose last
// rows are positions 131071 and 262143. Their tree tables, each into rows of
// its own past which the lanes' bound holds canaries, equal one
// rope_build_tables over every row's position, bit for bit, and leave the
// canaries.
void checkTreeTables(metal::MetalBackend &backend, const RotaryLayout &rotary) {
  constexpr uint32_t kCap = SPLASH_TARGET_VERIFY_ROWS, kSteps = 160;
  std::mt19937 random(rotary.pairs * 977 + static_cast<uint32_t>(rotary.scaling.factor * 10));
  std::uniform_int_distribution<uint32_t> anchors(0, 131071 - kCap);
  std::vector<Step> steps;
  for (uint32_t index = 0; index < kSteps; ++index) {
    const auto anchor = [&](uint32_t lane) { return index < 8 ? (index * 4 + lane) * kCap : anchors(random); };
    switch (index % 4) {
    case 0:
      steps.push_back(stepOf({{anchor(0), branching(16)}, {anchor(1), chain(3)}, {anchor(2), branching(9)},
                              {anchor(3), chain(1)}},
                             kCap));
      break;
    case 1:
      steps.push_back(stepOf({{anchor(0), chain(8)}, {anchor(1), branching(8)}}, 8));
      break;
    case 2:
      steps.push_back(stepOf({{anchor(0), chain(16)}, {anchor(1), branching(16)}, {anchor(2), chain(16)},
                              {anchor(3), branching(13)}},
                             kCap));
      break;
    default:
      steps.push_back(stepOf({{anchor(0), branching(5)}}, 8));
    }
  }
  steps.push_back(stepOf({{131071 - 15, chain(16)}, {262143 - 15, chain(16)}, {262143 - 3, branching(7)}}, kCap));

  std::vector<float> frequencies(rotary.pairs);
  model::rotaryInverseFrequencies(rotary.theta, rotary.pairs, rotary.scaling, frequencies);
  const float scale = rotary.scaling.attentionFactor();
  const metal::MetalBuffer inverse = upload<float>(backend, frequencies);
  // Each step's tables hold its lanes' bound of rows.
  const uint64_t rowBytes = uint64_t{rotary.pairs} * sizeof(float);
  std::vector<uint32_t> positions;
  uint64_t boundRows = 0;
  for (const Step &step : steps) {
    positions.insert(positions.end(), step.positions.begin(), step.positions.end());
    boundRows += uint64_t{step.rows.lanes} * step.rows.laneCap;
  }
  const metal::MetalBuffer cosine = test::sharedBuffer(backend, boundRows * rowBytes),
                           sine = test::sharedBuffer(backend, boundRows * rowBytes);
  std::memset(cosine.contents(), 0xa5, cosine.sizeBytes());
  std::memset(sine.contents(), 0xa5, sine.sizeBytes());
  metal::CommandGraph graph;
  std::vector<metal::MetalBuffer> held;
  uint64_t row = 0;
  for (const Step &step : steps) {
    const uint64_t bound = uint64_t{step.rows.lanes} * step.rows.laneCap;
    held.push_back(upload<ops::LaneRows>(backend, step.layout));
    held.push_back(upload<SplashTreeRow>(backend, step.tree));
    ops::RoPE::addTreeTables(graph, held[held.size() - 2], held.back(), step.anchors, inverse,
                             backend.view(cosine, row * rowBytes, bound * rowBytes),
                             backend.view(sine, row * rowBytes, bound * rowBytes), step.rows, scale, rotary.pairs);
    row += bound;
  }
  // The reference: rope_build_tables over every row's position, its three
  // axes alike, as a text row's.
  const auto rows = static_cast<uint32_t>(positions.size());
  std::vector<uint32_t> axes;
  for (const uint32_t position : positions) axes.insert(axes.end(), {position, position, position});
  const metal::MetalBuffer targetPositions = upload<uint32_t>(backend, axes);
  const metal::MetalBuffer referenceCosine = test::sharedBuffer(backend, uint64_t{rows} * rowBytes),
                           referenceSine = test::sharedBuffer(backend, uint64_t{rows} * rowBytes);
  const metal::MetalBuffer draft = test::sharedBuffer(backend, 4);
  ops::RoPE::addTables(graph, targetPositions, draft, inverse, draft, referenceCosine, referenceSine, draft, draft,
                       {.target_rows = rows, .draft_rows = 0, .target_scale = scale}, rows, rotary.pairs);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());

  const auto *treeCosine = static_cast<const uint32_t *>(cosine.contents());
  const auto *treeSine = static_cast<const uint32_t *>(sine.contents());
  const auto *expectedCosine = static_cast<const uint32_t *>(referenceCosine.contents());
  const auto *expectedSine = static_cast<const uint32_t *>(referenceSine.contents());
  uint64_t at = 0, reference = 0, compared = 0;
  bool reached131071 = false, reached262143 = false;
  for (const Step &step : steps) {
    const uint64_t bound = uint64_t{step.rows.lanes} * step.rows.laneCap;
    for (uint64_t packed = 0; packed < bound; ++packed)
      for (uint32_t pair = 0; pair < rotary.pairs; ++pair) {
        const uint64_t element = (at + packed) * rotary.pairs + pair;
        if (packed >= step.rows.rows) {
          require(treeCosine[element] == kCanary && treeSine[element] == kCanary,
                  rotary.name + ": a tree table wrote a row past its step's rows");
          continue;
        }
        const uint64_t expected = (reference + packed) * rotary.pairs + pair;
        require(treeCosine[element] == expectedCosine[expected] && treeSine[element] == expectedSine[expected],
                rotary.name + ": a tree row's RoPE differs from rope_build_tables' at its position " +
                    std::to_string(step.positions[packed]));
        ++compared;
      }
    for (const uint32_t position : step.positions) {
      reached131071 = reached131071 || position == 131071;
      reached262143 = reached262143 || position == 262143;
    }
    at += bound;
    reference += step.rows.rows;
  }
  require(reached131071 && reached262143, rotary.name + ": the steps left out positions 131071 or 262143");
  std::cout << "RoPE tree tables " << rotary.name << " (" << rotary.pairs << " pairs, scale " << scale
            << "): " << rows << " rows, " << compared << " pairs bit for bit PASS\n";
}

// A lane whose layout runs past the step's rows writes nothing, and the
// other lanes' rows are their positions' as ever.
void checkLayoutPastRows(metal::MetalBackend &backend) {
  constexpr uint32_t kPairs = SPLASH_TARGET_ROPE_PAIRS;
  Step step = stepOf({{100, chain(4)}, {200, branching(6)}}, 8);
  step.layout[1].rows = 7;
  std::vector<float> frequencies(kPairs);
  model::rotaryInverseFrequencies(1.0e7F, kPairs, {}, frequencies);
  const metal::MetalBuffer inverse = upload<float>(backend, frequencies);
  const uint64_t tableBytes = uint64_t{16} * kPairs * sizeof(float);
  const metal::MetalBuffer cosine = test::sharedBuffer(backend, tableBytes), sine = test::sharedBuffer(backend, tableBytes);
  std::memset(cosine.contents(), 0xa5, tableBytes);
  std::memset(sine.contents(), 0xa5, tableBytes);
  metal::CommandGraph graph;
  ops::RoPE::addTreeTables(graph, upload<ops::LaneRows>(backend, step.layout), upload<SplashTreeRow>(backend, step.tree),
                           step.anchors, inverse, cosine, sine, step.rows, 1.0F);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  const auto *values = static_cast<const uint32_t *>(cosine.contents());
  for (uint32_t row = 0; row < 16; ++row)
    for (uint32_t pair = 0; pair < kPairs; ++pair) {
      const bool canary = values[row * kPairs + pair] == kCanary;
      require(row < 4 ? !canary : canary, "a lane past the step's rows wrote, or the lane before it did not");
    }
  std::cout << "RoPE tree tables: a lane past the step's rows writes nothing PASS\n";
}

// The draft's rows alone, as a decode step's cycle start builds them: no
// target buffer, the draft's rows as a build of both kinds holds them.
void checkDraftOnly(metal::MetalBackend &backend) {
  constexpr uint32_t kRows = 64;
  std::vector<float> draftFrequencies(SPLASH_DRAFT_ROPE_PAIRS), targetFrequencies(SPLASH_TARGET_ROPE_PAIRS);
  for (uint32_t pair = 0; pair < SPLASH_DRAFT_ROPE_PAIRS; ++pair)
    draftFrequencies[pair] = std::pow(1.0e6F, -float(pair) / SPLASH_DRAFT_ROPE_PAIRS);
  model::rotaryInverseFrequencies(1.0e7F, SPLASH_TARGET_ROPE_PAIRS, {}, targetFrequencies);
  std::vector<uint32_t> draftPositions(kRows), targetPositions(kRows * 3);
  for (uint32_t row = 0; row < kRows; ++row) {
    draftPositions[row] = 70000 + row * 3;
    std::fill_n(targetPositions.begin() + row * 3, 3, row);
  }
  const uint64_t draftBytes = uint64_t{kRows} * SPLASH_DRAFT_ROPE_PAIRS * sizeof(float);
  const uint64_t targetBytes = uint64_t{kRows} * SPLASH_TARGET_ROPE_PAIRS * sizeof(float);
  const metal::MetalBuffer draft = upload<uint32_t>(backend, draftPositions),
                           target = upload<uint32_t>(backend, targetPositions),
                           draftInverse = upload<float>(backend, draftFrequencies),
                           targetInverse = upload<float>(backend, targetFrequencies);
  std::array<metal::MetalBuffer, 4> both, alone;
  for (auto *tables : {&both, &alone})
    for (metal::MetalBuffer &table : *tables) table = test::sharedBuffer(backend, std::max(draftBytes, targetBytes));
  metal::CommandGraph graph;
  ops::RoPE::addTables(graph, target, draft, targetInverse, draftInverse, both[0], both[1], both[2], both[3],
                       {.target_rows = kRows, .draft_rows = kRows, .target_scale = 1.0F}, kRows);
  ops::RoPE::addTables(graph, {}, draft, {}, draftInverse, {}, {}, alone[2], alone[3],
                       {.target_rows = 0, .draft_rows = kRows, .target_scale = 1.0F}, kRows);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  for (uint32_t table : {2U, 3U})
    require(std::memcmp(both[table].contents(), alone[table].contents(), draftBytes) == 0,
            "the draft's RoPE rows alone differ from a build of both kinds");
  metal::CommandGraph refused;
  rejects([&] {
    ops::RoPE::addTables(refused, {}, draft, {}, draftInverse, {}, {}, alone[2], alone[3],
                         {.target_rows = 0, .draft_rows = 0, .target_scale = 1.0F}, kRows);
  }, "invalid RoPE table row count", "a RoPE table build of no rows was accepted");
  require(refused.empty(), "a RoPE table build of no rows encoded a dispatch");
  std::cout << "RoPE draft rows alone PASS\n";
}

// Each buffer a step's tree tables reach, at its extent and one element
// short, and the step's lanes, anchors, scale and pairs it refuses.
void checkTreeTableBounds(metal::MetalBackend &backend) {
  constexpr ops::StepRows kStep{3, 29, 16};
  constexpr std::array<uint32_t, 3> kAnchors{0, 5, 9};
  for (const uint32_t pairs : {SPLASH_TARGET_ROPE_PAIRS, 64U}) {
    const uint64_t table = uint64_t{kStep.rows} * pairs * 4;
    test::requireExtents(backend,
                         std::initializer_list<test::BufferExtent>{
                             {0, kStep.lanes * sizeof(SplashLaneRows), sizeof(SplashLaneRows), "RoPE lane rows"},
                             {1, kStep.rows * sizeof(SplashTreeRow), sizeof(SplashTreeRow), "RoPE tree rows"},
                             {2, uint64_t{pairs} * 4, 4, "target inverse frequency"},
                             {3, table, 4, "target RoPE cosine"},
                             {4, table, 4, "target RoPE sine"}},
                         [&](metal::CommandGraph &graph, const std::vector<metal::MetalBuffer> &b) {
                           ops::RoPE::addTreeTables(graph, b[0], b[1], kAnchors, b[2], b[3], b[4], kStep, 1.0F,
                                                    pairs);
                         });
  }
  const metal::MetalBuffer any = test::sharedBuffer(backend, 1 << 16);
  metal::CommandGraph refused;
  const auto add = [&](std::span<const uint32_t> anchors, ops::StepRows step, float scale, uint32_t pairs) {
    ops::RoPE::addTreeTables(refused, any, any, anchors, any, any, any, step, scale, pairs);
  };
  rejects([&] { add(std::span(kAnchors).first(2), kStep, 1.0F, 32); }, "an anchor a lane",
          "a step's tree tables took fewer anchors than lanes");
  rejects([&] { add(kAnchors, {3, 49, 16}, 1.0F, 32); }, "invalid decode step rows",
          "a step of more rows than its lanes hold was accepted");
  rejects([&] { add(kAnchors, kStep, 0.0F, 32); }, "invalid target RoPE scale", "a RoPE scale of zero was accepted");
  rejects([&] { add(kAnchors, kStep, 1.0F, 16); }, "no RoPE table kernel", "an uncompiled pair count was accepted");
  require(refused.empty(), "a refused tree table encoded a dispatch");
  std::cout << "RoPE tree table bounds PASS\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: rope-tables METALLIB");
    metal::MetalBackend backend(argv[1]);
    for (const RotaryLayout &rotary : rotaryLayouts()) checkTreeTables(backend, rotary);
    checkLayoutPastRows(backend);
    checkDraftOnly(backend);
    checkTreeTableBounds(backend);
    std::cout << "RoPE tables: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "rope-tables: " << error.what() << '\n';
    return 1;
  }
}

// What one prefill command of a runtime may hold (model::commandLimits) and
// what a command or a pass of it computes (model::prefillCommandWork), at the
// layouts of the families the runtime serves: Qwen3.8-27B, Qwen3.6-35B-A3B,
// Qwen3.5-122B-A10B, Flint and Qwen3.8-Flash-Next. Every family's bound of
// one attention layer's pairs is a full chunk's at the end of its native
// window of 262,144 tokens, as Bootstrap set it, not that times its
// attention layers; a streamed 35B's group of 8192 rows fits it up to
// position 61,183, half the rows up to 128,511; a 122B's group past it
// exceeds it in every pass, one that runs no attention layer too, as passes
// never lower a layer's pairs; Flint past its native window exceeds it as a
// dense target does, its full-history layers being the heaviest; Flash-Next
// keeps its group within 3.2 % of it at any depth. A
// command holds its chunks' rows, passes over the target's layers at a pass
// a layer (one with the Neural Engine's FFN split, whose least chunk is the
// fewest rows of a command at its rate), one lane past a chunk, and one
// lane's streamed history where K/V spills. A pass counts the pairs of its
// own layers, and the heaviest layer's pairs of the target whatever the
// pass.
#include "TestChecks.hpp"
#include "model/Flint.hpp"
#include "model/Qwen3_6Moe.hpp"
#include "model/Qwen3_8.hpp"
#include "model/Qwen4Exp.hpp"
#include "model/RuntimeArenas.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <span>
#include <string>

namespace {

using namespace splash;
using splash::test::require;

constexpr uint32_t kChunk = model::ExecutionLimits::prefillTokenBudget;
constexpr uint64_t kLayerPairs = 534'774'784;

// A runtime's geometry of `layout`: its commands a prefill group's chunks
// (`groups`) or one, its K/V spilling (`spills`, a block-sparse target's) or
// not.
model::RuntimeGeometry geometryOf(const model::QwenTargetDimensions &layout, kv::Layout kvLayout, bool groups,
                                  bool spills) {
  constexpr std::array<uint32_t, 1> captured{1};
  model::RuntimeGeometry geometry;
  geometry.target = model::QwenTargetGeometry(layout, captured);
  geometry.target.kvLayout = kvLayout;
  if (spills)
    geometry.spilledKv = model::SparseKvShape{.blockTokens = layout.indexerBlockTokens,
                                              .selectedBlocks = layout.indexerTopBlocks,
                                              .verifyRows = model::ExecutionLimits::targetVerifyRows,
                                              .denseHistoryTokens = layout.indexerTokens()};
  geometry.prefillChunks = groups ? model::kPrefillGroupChunks : 1;
  return geometry;
}

template <class Layout> model::RuntimeGeometry geometryOf(bool groups = false, bool spills = false) {
  const Layout layout;
  return geometryOf(layout, layout.kvLayout(), groups, spills);
}

// The pairs the heaviest attention layer of the target computes for a lane
// of `rows` rows from `position`, in a pass from layer `first` with `passes` passes left.
uint64_t layerPairs(const model::RuntimeGeometry &geometry, uint32_t position, uint32_t rows, uint32_t first = 0,
                    uint32_t passes = 1) {
  const std::array<model::PrefillLane, 1> lane{{{.position = position, .rows = rows}}};
  return model::prefillCommandWork(geometry.target, lane, first, passes).layerAttentionPairs;
}

void testPairCapIsTodaysPerLayerCap() {
  // Today's bound: a chunk's rows each attend to the keys before the chunk and
  // to the chunk's up to their own, rows * position + rows * (rows + 1) / 2.
  require(uint64_t{kChunk} * (262'144 - kChunk) + uint64_t{kChunk} * (kChunk + 1) / 2 == kLayerPairs,
          "the per-layer bound is not a full chunk's at the end of the native window");
  for (const model::RuntimeGeometry &geometry :
       {geometryOf<model::Qwen3_8Layout>(), geometryOf<model::Qwen3_6MoeLayout>(true),
        geometryOf<model::FlintLayout>(true), geometryOf<model::Qwen4ExpLayout>(true, true)})
    require(model::commandLimits(geometry, 0).attentionPairs == kLayerPairs &&
                geometry.target.attentionLayerCount() > 1,
            "a family's bound of one attention layer's pairs is not today's per-layer bound");
}

void testStreamedGroupHalvesPast61K() {
  const model::RuntimeGeometry geometry = geometryOf<model::Qwen3_6MoeLayout>(true);
  require(layerPairs(geometry, 61'183, 4 * kChunk) <= kLayerPairs &&
              layerPairs(geometry, 61'184, 4 * kChunk) > kLayerPairs,
          "a streamed 35B's group of 8192 rows does not fit the bound up to position 61,183 alone");
  require(layerPairs(geometry, 128'511, 2 * kChunk) <= kLayerPairs &&
              layerPairs(geometry, 128'512, 2 * kChunk) > kLayerPairs,
          "half a streamed 35B's group does not fit the bound up to position 128,511 alone");
}

// Qwen3.5-122B-A10B's 48 layers, every fourth attention (layers 3, 7, ...,
// 47): a group of 8192 rows at position 70,000 computes in each attention
// layer more pairs than one layer may, and every pass of 19 counts them,
// those that run no attention layer too (pass 0 runs layers 0-1, pass 2
// layers 4-5), as a pass does not split a layer.
void testLayerPairsIgnorePasses() {
  const model::RuntimeGeometry geometry =
      geometryOf(model::kQwen3_5_122BLayout, model::kQwen3_5_122BLayout.kvLayout(), true, false);
  const uint64_t whole = layerPairs(geometry, 70'000, 4 * kChunk);
  require(geometry.target.layers == 48 && geometry.target.attentionLayerCount() == 12 && whole > kLayerPairs,
          "a 122B group of 8192 rows at position 70,000 kept within the bound of one layer's pairs");
  for (uint32_t pass = 0, first = 0; pass < 19; ++pass) {
    require(layerPairs(geometry, 70'000, 4 * kChunk, first, 19 - pass) == whole,
            "pass " + std::to_string(pass) + " of 19 did not count the heaviest attention layer's pairs of the target");
    first = model::passLayers(first, 19 - pass, geometry.target.layers).end;
  }
}

void testFlintHalvesPastNativeLikeDense() {
  const model::RuntimeGeometry geometry = geometryOf<model::FlintLayout>(true);
  require(layerPairs(geometry, 260'096, kChunk) == kLayerPairs && layerPairs(geometry, 260'097, kChunk) > kLayerPairs,
          "a Flint chunk past its native window kept within the bound: its full-history layers did not count");
  // A pass of sliding layers alone counts the full-history layers' pairs.
  for (uint32_t pass = 0; pass < geometry.target.layers; ++pass)
    require(layerPairs(geometry, 260'097, kChunk, pass, geometry.target.layers - pass) > kLayerPairs,
            "a Flint pass of one layer lowered the heaviest layer's pairs");
}

void testFlashNextKeepsItsGroupAtAnyDepth() {
  const model::RuntimeGeometry geometry = geometryOf<model::Qwen4ExpLayout>(true, true);
  for (const uint32_t position : {0u, 131'072u, 1'048'576u})
    require(layerPairs(geometry, position, 4 * kChunk) * 1000 <= kLayerPairs * 32,
            "a Flash-Next group at position " + std::to_string(position) + " took more than 3.2 % of the bound");
}

void testLimitsAreTheRuntimes() {
  const struct {
    model::RuntimeGeometry geometry;
    uint32_t rows, streams;
  } runtimes[] = {{geometryOf<model::Qwen3_8Layout>(), kChunk, 0},
                  {geometryOf<model::Qwen3_6MoeLayout>(), kChunk, 0},
                  {geometryOf<model::Qwen3_6MoeLayout>(true), 4 * kChunk, 0},
                  {geometryOf<model::FlintLayout>(true), 4 * kChunk, 0},
                  {geometryOf<model::Qwen4ExpLayout>(true), 4 * kChunk, 0},
                  {geometryOf<model::Qwen4ExpLayout>(true, true), 4 * kChunk, 1}};
  for (const auto &runtime : runtimes) {
    const model::CommandLimits limits = model::commandLimits(runtime.geometry, 0);
    require(limits == model::CommandLimits{.commandRows = runtime.rows,
                                           .layers = runtime.geometry.target.layers,
                                           .maximumPasses = runtime.geometry.target.layers,
                                           .lanesPastChunk = 1,
                                           .indexPageLanes = runtime.streams,
                                           .attentionPairs = kLayerPairs},
            "a runtime's limits are not its chunks' rows in a pass a layer, one lane past a chunk and its "
            "streamed history");
    // The split's least chunk (ops::AneFfn::minimumRows: kMinimumRows to the
    // chunk's rows) is the fewest rows of a command at its rate.
    for (const uint32_t least : {512u, 640u, kChunk}) {
      const model::CommandLimits split = model::commandLimits(runtime.geometry, least);
      require(split.maximumPasses == 1 && split.layers == runtime.geometry.target.layers,
              "a runtime with the Neural Engine's FFN split ran passes, or its commands lost layers");
      require(split.minimumUnitRows == least, "a runtime with the Neural Engine's FFN split from " +
                                                  std::to_string(least) + " rows planned units from " +
                                                  std::to_string(split.minimumUnitRows));
    }
  }
}

void testPassesCountTheirOwnLayers() {
  for (const model::RuntimeGeometry &geometry :
       {geometryOf<model::Qwen3_6MoeLayout>(true), geometryOf<model::FlintLayout>(true),
        geometryOf<model::Qwen4ExpLayout>(true, true)}) {
    const std::array<model::PrefillLane, 2> lanes{
        {{.position = 300, .rows = 1500}, {.position = 140'000, .rows = 3000}}};
    const model::CommandWork whole = model::prefillCommandWork(geometry.target, lanes, 0, 1);
    for (const uint32_t passes : {3u, geometry.target.layers}) {
      uint64_t summed = 0;
      for (uint32_t pass = 0, first = 0; pass < passes; ++pass) {
        const model::CommandWork work = model::prefillCommandWork(geometry.target, lanes, first, passes - pass);
        require(work.layerAttentionPairs == whole.layerAttentionPairs,
                "a pass's heaviest layer's pairs are not the command's");
        summed += work.attentionPairs;
        first = model::passLayers(first, passes - pass, geometry.target.layers).end;
      }
      require(summed == whole.attentionPairs, "a command's passes did not count its layers' pairs once each");
    }
  }
}

} // namespace

int main() {
  try {
    testPairCapIsTodaysPerLayerCap();
    testStreamedGroupHalvesPast61K();
    testLayerPairsIgnorePasses();
    testFlintHalvesPastNativeLikeDense();
    testFlashNextKeepsItsGroupAtAnyDepth();
    testLimitsAreTheRuntimes();
    testPassesCountTheirOwnLayers();
    std::cout << "command limits tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "command limits tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

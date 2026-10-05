#include "TestChecks.hpp"
#include "metal/abi/MoE.h"
#include "model/WeightLayout.hpp"
#include "ops/DraftSelector.hpp"
#include "ops/MoE.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/Sampling.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

using splash::test::rejects;
using splash::test::require;

void attention() {
  using namespace splash;
  for (const uint32_t queryHeads : {16U, 24U}) {
    const kv::Layout layout{1, queryHeads == 16 ? 2U : 4U, 256};
    uint32_t maximumSlots = 0;
    for (uint32_t rows = 1; rows <= 2048; ++rows) {
      // Split counts decrease as the M8 tile count grows, so a shorter
      // request can need more scratch than the exact requested row count.
      const uint32_t tiles = (rows + 7) / 8;
      const uint32_t splits = std::clamp(32U / tiles, 1U, 32U);
      maximumSlots = std::max(maximumSlots, tiles * splits);
      const auto workspace = ops::PagedAttention::prefillWorkspace(rows, queryHeads, layout);
      require(workspace.partialsBytes == uint64_t{maximumSlots} * 8 * queryHeads * 256 * 4,
              "prefill partial workspace differs from the maximum over actual row counts");
      require(workspace.statisticsBytes == uint64_t{maximumSlots} * 8 * queryHeads * 2 * 4,
              "prefill statistics workspace differs from the maximum over actual row counts");
      const auto plan = ops::PagedAttention::prefillPlan(rows, queryHeads, layout);
      require(workspace.partialsBytes >= plan.workspace.partialsBytes &&
                  workspace.statisticsBytes >= plan.workspace.statisticsBytes,
              "prefill workspace omitted the plan of its own rows");
    }
    const auto maximum = ops::PagedAttention::prefillWorkspace(2048, queryHeads, layout);
    require(maximum.partialsBytes == (queryHeads == 24 ? 48U : 32U) * uint64_t{1024 * 1024},
            "2048-row prefill partial workspace changed");
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      const auto workspace =
          ops::PagedAttention::verifyWorkspace(lanes, queryHeads, layout);
      const uint64_t values =
          uint64_t{lanes} * 8 * kv::kVerifyMaximumSplits * queryHeads;
      require(workspace.partialsBytes == values * 256 * 4,
              "verify partial workspace is not sized for the maximum split count");
      require(workspace.statisticsBytes == values * 2 * 4,
              "verify statistics workspace is not sized for the maximum split count");
    }
    for (const uint32_t rows : {0U, 2049U})
      rejects([&] { (void)ops::PagedAttention::prefillWorkspace(rows, queryHeads, layout); },
              "invalid attention workspace rows",
              "a prefill workspace of no rows or past the budget was sized");
    for (const uint32_t lanes : {0U, 5U})
      rejects([&] { (void)ops::PagedAttention::verifyWorkspace(lanes, queryHeads, layout); },
              "invalid attention workspace batch width",
              "a verify workspace of no lanes or more than a batch was sized");
    rejects([&] { (void)ops::PagedAttention::verifyWorkspace(1, 0, layout); },
            "no paged-attention kernel for layout",
            "a verify workspace of no query heads was sized");
  }
}

// Expert ids, route rows and grouped routes are uint32, routing weights fp32
// and activations bf16. The grouped input first holds the router's fp32
// scores, one row of 256 per token. The split prefill plan parks its gate in
// expertOutput, so that field spans the wider of the hidden and intermediate
// widths.
void checkMoe(splash::ops::MoeWorkspace workspace,
              splash::ops::MoeShape shape, uint32_t rows, uint32_t tileRows,
              uint32_t outputWidth) {
  using splash::model::kBFloat16Bytes;
  constexpr uint64_t kRouterScores = 256;
  const uint64_t routes = uint64_t{rows} * shape.routesPerToken();
  const uint64_t tiles = splash::ops::moeMaximumTiles(rows, shape, tileRows);
  const uint64_t grouped = tiles * tileRows;
  require(workspace.selectedExpertsBytes == routes * sizeof(uint32_t) &&
              workspace.routingWeightsBytes == routes * sizeof(float) &&
              workspace.tileDescriptorsBytes == tiles * sizeof(MoeTileDescriptor) &&
              workspace.tileCountBytes == sizeof(uint32_t) &&
              workspace.groupedRoutesBytes == grouped * sizeof(uint32_t) &&
              workspace.routeRowsBytes == routes * sizeof(uint32_t) &&
              workspace.groupedInputBytes == std::max(grouped * shape.hiddenSize * kBFloat16Bytes,
                                                      uint64_t{rows} * kRouterScores * sizeof(float)) &&
              workspace.expertIntermediateBytes ==
                  grouped * shape.expertIntermediateSize * kBFloat16Bytes &&
              workspace.expertOutputBytes == grouped * outputWidth * kBFloat16Bytes &&
              workspace.groupedSumsBytes == 0,
          "MoE workspace changed from baseline");
}

void moe() {
  using namespace splash::ops;
  for (const MoeShape shape : {MoeShape{256, 8, 2, 512},
                               MoeShape{2048, 256, 8, 512}}) {
    const uint32_t splitWidth =
        std::max(shape.hiddenSize, shape.expertIntermediateSize);
    for (uint32_t rows = 1; rows <= 2048; ++rows)
      checkMoe(MoE::prefillPlan(shape, rows, {MoeExpertTile::M32}).workspace(), shape, rows, 32, splitWidth);
    const auto single = MoE::decodePlan(shape, 1, {MoeExpertTile::M8}).workspace();
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      const auto workspace = MoE::decodePlan(shape, lanes, {MoeExpertTile::M8}).workspace();
      checkMoe(workspace, shape, lanes * 8, 8, shape.hiddenSize);
      require(workspace.groupedInputBytes <= lanes * single.groupedInputBytes &&
                  workspace.tileDescriptorsBytes <=
                      lanes * single.tileDescriptorsBytes,
              "packed decode workspace exceeds per-lane allocation bound");
    }
    for (const uint32_t rows : {0U, 2049U})
      rejects([&] { (void)MoE::prefillPlan(shape, rows, {MoeExpertTile::M32}); },
              "invalid MoE prefill rows",
              "a MoE prefill of no rows or past the budget was planned");
    for (const uint32_t lanes : {0U, 5U})
      rejects([&] { (void)MoE::decodePlan(shape, lanes, {MoeExpertTile::M8}); },
              "invalid MoE decode batch width",
              "a MoE decode of no lanes or more than a batch was planned");
  }
  rejects([] { (void)MoE::prefillPlan({}, 1, {MoeExpertTile::M32}); },
          "invalid MoE workspace shape", "a MoE plan of an empty shape was sized");
}

void sampling() {
  using splash::ops::DraftSelector;
  using splash::ops::Sampling;
  // Independent ABI formulas, including one lane and B1-B4 packed
  // extents. No backend, allocation or GPU graph is needed to size buffers.
  for (uint32_t rows : {1U, 8U, 16U, 24U, 32U,
                        std::numeric_limits<uint32_t>::max()}) {
    const auto workspace = Sampling::workspace(rows);
    const uint64_t count = rows;
    require(workspace.argmaxValuesBytes == count * 16 * 4 &&
                workspace.argmaxIndicesBytes == count * 16 * 4 &&
                workspace.partialMassesBytes == count * 16 * 12 &&
                workspace.vocabularyRowsBytes == count * 16 &&
                workspace.vocabularyRangesBytes == count * 256 * 8 &&
                workspace.vocabularyArrivalsBytes == count * 4,
            "target sampling workspace changed from the shipped 16-shard ABI");
  }
  for (uint32_t positions : {1U, 7U, 14U, 21U, 28U,
                             std::numeric_limits<uint32_t>::max()}) {
    const auto workspace = DraftSelector::workspace(positions);
    const uint64_t count = positions;
    // Partial values carry the 16 x 16 edge table of every position behind
    // the eight shard partials.
    require(workspace.partialIdsBytes == count * 8 * 16 * 4 &&
                workspace.partialValuesBytes == count * (8 + 16) * 16 * 4 &&
                workspace.candidatesBytes == count * 16 * 4 &&
                workspace.unaryBytes == count * 16 * 4 &&
                workspace.proposalProbabilitiesBytes == count * 16 * 4,
            "draft sampling workspace changed from the shipped 8-shard ABI");
  }
  rejects([] { (void)Sampling::workspace(0); }, "invalid sampling workspace row count",
          "a sampling workspace of no rows was sized");
  rejects([] { (void)DraftSelector::workspace(0); },
          "invalid draft selector workspace position count",
          "a draft selector workspace of no positions was sized");
}

} // namespace

int main() {
  try {
    attention();
    moe();
    sampling();
    std::cout << "operator workspace: PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "operator workspace: FAIL: " << error.what() << '\n';
    return 1;
  }
}

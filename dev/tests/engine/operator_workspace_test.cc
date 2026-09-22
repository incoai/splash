#include "ops/MoE.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/Sampling.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class Function> void rejects(Function function) {
  try {
    (void)function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("invalid workspace request was accepted");
}

void attention() {
  using namespace splash;
  for (const uint32_t queryHeads : {16U, 24U}) {
    const kv::Layout layout{1, queryHeads == 16 ? 2U : 4U, 256};
    for (const auto config : ops::PagedAttention::prefillCandidates()) {
      uint32_t maximumSlots = 0;
      for (uint32_t rows = 1; rows <= 2048; ++rows) {
        // Split counts decrease as the M8 tile count grows, so a shorter
        // request can need more scratch than the exact requested row count.
        const uint32_t tiles = (rows + 7) / 8;
        const uint32_t splits = std::min(
            32U, static_cast<uint32_t>(config.splitMultiplier) *
                     std::clamp(32U / tiles, 1U, 32U));
        maximumSlots = std::max(maximumSlots, tiles * splits);
        const auto workspace = ops::PagedAttention::prefillWorkspace(rows, queryHeads, layout, config);
        require(workspace.partialsBytes == uint64_t{maximumSlots} * 8 * queryHeads * 256 * 4,
                "prefill partial workspace differs from the maximum over actual row counts");
        require(workspace.statisticsBytes == uint64_t{maximumSlots} * 8 * queryHeads * 2 * 4,
                "prefill statistics workspace differs from the maximum over actual row counts");
        for (uint32_t history : {0U, 4095U, 4096U, 131072U, kv::kMaximumPhysicalTokens - rows}) {
          const auto plan = ops::PagedAttention::prefillPlan(rows, queryHeads, layout, history, config);
          require(workspace.partialsBytes >= plan.workspace.partialsBytes &&
                      workspace.statisticsBytes >= plan.workspace.statisticsBytes,
                  "prefill workspace omitted actual rows at a valid context boundary");
        }
      }
      const auto maximum = ops::PagedAttention::prefillWorkspace(2048, queryHeads, layout, config);
      const uint64_t expectedMiB = (queryHeads == 24 ? 48U : 32U) *
                                  static_cast<uint32_t>(config.splitMultiplier);
      require(maximum.partialsBytes == expectedMiB * 1024 * 1024,
              "2048-row prefill partial workspace changed");
    }
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      const auto workspace =
          ops::PagedAttention::verifyWorkspace(lanes, queryHeads, layout);
      const uint64_t values =
          uint64_t{lanes} * 8 * kv::kQ8VerifyMaximumSplits * queryHeads;
      require(workspace.partialsBytes == values * 256 * 4,
              "verify partial workspace is not sized for the maximum split count");
      require(workspace.statisticsBytes == values * 2 * 4,
              "verify statistics workspace is not sized for the maximum split count");
    }
    rejects([&] { return ops::PagedAttention::prefillWorkspace(0, queryHeads, layout); });
    rejects([&] { return ops::PagedAttention::prefillWorkspace(2049, queryHeads, layout); });
    rejects([&] { return ops::PagedAttention::verifyWorkspace(0, queryHeads, layout); });
    rejects([&] { return ops::PagedAttention::verifyWorkspace(5, queryHeads, layout); });
    rejects([&] { return ops::PagedAttention::verifyWorkspace(1, 0, layout); });
  }
}

// The split prefill plan parks its gate in expertOutput, so that field spans
// the wider of the hidden and intermediate widths.
void checkMoe(splash::ops::MoeWorkspace workspace,
              splash::ops::MoeShape shape, uint32_t rows, uint32_t tileRows,
              uint32_t outputWidth) {
  const uint64_t routed = uint64_t{rows} * shape.expertsPerToken;
  const uint64_t routes = uint64_t{rows} * (shape.expertsPerToken + 1);
  const uint64_t tiles = std::min(routed / tileRows + shape.experts, routed) +
                         (rows + tileRows - 1) / tileRows;
  const uint64_t grouped = tiles * tileRows;
  require(workspace.selectedExpertsBytes == routes * 4 &&
              workspace.routingWeightsBytes == routes * 2 &&
              workspace.tileDescriptorsBytes == tiles * 8 &&
              workspace.tileCountBytes == 4 &&
              workspace.groupedRoutesBytes == grouped * 4 &&
              workspace.routeRowsBytes == routes * 4 &&
              workspace.groupedInputBytes == grouped * shape.hiddenSize * 2 &&
              workspace.expertIntermediateBytes ==
                  grouped * shape.expertIntermediateSize * 2 &&
              workspace.expertOutputBytes == grouped * outputWidth * 2,
          "MoE workspace changed from baseline");
}

void moe() {
  using namespace splash::ops;
  for (const MoeShape shape : {MoeShape{256, 8, 2, 512},
                               MoeShape{2048, 256, 8, 512}}) {
    const uint32_t splitWidth =
        std::max(shape.hiddenSize, shape.expertIntermediateSize);
    for (uint32_t rows = 1; rows <= 2048; ++rows)
      checkMoe(MoE::prefillPlan(shape, rows).workspace(), shape, rows, 32, splitWidth);
    const auto single = MoE::decodePlan(shape, 1).workspace();
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      const auto workspace = MoE::decodePlan(shape, lanes).workspace();
      checkMoe(workspace, shape, lanes * 8, 8, shape.hiddenSize);
      require(workspace.groupedInputBytes <= lanes * single.groupedInputBytes &&
                  workspace.tileDescriptorsBytes <=
                      lanes * single.tileDescriptorsBytes,
              "packed decode workspace exceeds per-lane allocation bound");
    }
    rejects([&] { return MoE::prefillPlan(shape, 0); });
    rejects([&] { return MoE::prefillPlan(shape, 2049); });
    rejects([&] { return MoE::decodePlan(shape, 0); });
    rejects([&] { return MoE::decodePlan(shape, 5); });
  }
  rejects([] { return MoE::prefillPlan({}, 1); });
}

void sampling() {
  using splash::ops::Sampling;
  // Independent ABI formulas, including one lane and B1-B4 packed
  // extents. No backend, allocation or GPU graph is needed to size buffers.
  for (uint32_t rows : {1U, 8U, 16U, 24U, 32U,
                        std::numeric_limits<uint32_t>::max()}) {
    const auto workspace = Sampling::workspace(rows);
    const uint64_t count = rows;
    require(workspace.argmaxValuesBytes == count * 16 * 4 &&
                workspace.argmaxIndicesBytes == count * 16 * 4 &&
                workspace.partialIdsBytes == count * 16 * 32 * 4 &&
                workspace.partialValuesBytes == count * 16 * 32 * 4 &&
                workspace.topIdsBytes == count * 32 * 4 &&
                workspace.topProbabilitiesBytes == count * 32 * 4,
            "target sampling workspace changed from the shipped 16-shard ABI");
  }
  for (uint32_t positions : {1U, 7U, 14U, 21U, 28U,
                             std::numeric_limits<uint32_t>::max()}) {
    const auto workspace = Sampling::draftWorkspace(positions);
    const uint64_t count = positions;
    // Partial values carry the 16 x 16 edge table of every position behind
    // the eight shard partials.
    require(workspace.partialIdsBytes == count * 8 * 16 * 4 &&
                workspace.partialValuesBytes == count * (8 + 16) * 16 * 4 &&
                workspace.candidatesBytes == count * 16 * 4 &&
                workspace.unaryBytes == count * 16 * 2 &&
                workspace.proposalProbabilitiesBytes == count * 16 * 4,
            "draft sampling workspace changed from the shipped 8-shard ABI");
  }
  rejects([] { return Sampling::workspace(0); });
  rejects([] { return Sampling::draftWorkspace(0); });
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

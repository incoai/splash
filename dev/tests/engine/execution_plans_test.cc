#include "ops/ExecutionPlans.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace splash;
using namespace splash::ops;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
template <typename Function> void rejects(Function function) {
  bool rejected = false;
  try { function(); }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "invalid operator choice or lookup was accepted");
}

constexpr std::array attentionShapes{
    AttentionShape{24, 4, 256}, AttentionShape{16, 2, 256}};
constexpr std::array draftShapes{
    DraftAttentionShape{5120, 1280, 6144, 4096, 32, 8, 128},
    DraftAttentionShape{2048, 512, 6144, 4096, 32, 8, 128}};
constexpr MoeShape routedShape{2048, 256, 8, 512};
constexpr std::array moeShapes{
    routedShape, MoeShape{768, 7, 3, 256}, MoeShape{256, 1, 1, 256}};
constexpr std::array matrices{
    LinearMatrix{17408, 5120}, LinearMatrix{6144, 5120},
    LinearMatrix{6144, 2048}, LinearMatrix{512, 2048},
    LinearMatrix{768, 768}};
constexpr std::array attentionFields{
    &AttentionWorkspace::partialsBytes, &AttentionWorkspace::statisticsBytes};
constexpr std::array draftFields{
    &DraftAttentionWorkspace::convolutionBytes,
    &DraftAttentionWorkspace::qkvBytes,
    &DraftAttentionWorkspace::groupedQueriesBytes,
    &DraftAttentionWorkspace::queryKeysBytes,
    &DraftAttentionWorkspace::queryValuesBytes};
constexpr std::array moeFields{
    &MoeWorkspace::selectedExpertsBytes, &MoeWorkspace::routingWeightsBytes,
    &MoeWorkspace::tileDescriptorsBytes, &MoeWorkspace::tileCountBytes,
    &MoeWorkspace::groupedRoutesBytes, &MoeWorkspace::routeRowsBytes,
    &MoeWorkspace::groupedInputBytes, &MoeWorkspace::expertIntermediateBytes,
    &MoeWorkspace::expertOutputBytes};

kv::Layout layout(AttentionShape shape, uint32_t layers = 1) {
  return {layers, shape.kvHeads, shape.headDimension, shape.format};
}
DeviceCapabilities device(uint32_t family = 10) {
  DeviceCapabilities value;
  value.appleGpuFamily = family;
  return value;
}
template <typename Workspace, size_t N>
void equalWorkspace(const Workspace &actual, const Workspace &expected,
                    const std::array<uint64_t Workspace::*, N> &fields) {
  for (auto field : fields)
    require(actual.*field == expected.*field, "workspace bound changed");
}
template <typename Workspace, size_t N>
void covers(const Workspace &stride, const Workspace &needed, uint32_t lanes,
            const std::array<uint64_t Workspace::*, N> &fields) {
  for (auto field : fields)
    require((stride.*field) * lanes >= needed.*field,
            "workspace does not cover an installed plan");
}

void baselinePlans() {
  for (uint32_t family : {9U, 10U, 11U}) {
    const ExecutionPlans plans(device(family));
    const Q4Linear baseline(device(family));
    for (auto matrix : matrices) {
      uint64_t gateBound = 0;
      for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
        for (auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                              LinearEpilogue::GateUp}) {
          const LinearWorkload w{matrix, lanes * 8, LinearPhase::Decode, epilogue};
          require(plans.linear().plan(w).configuration() ==
                      baseline.plan(w).configuration(),
                  "empty linear choices changed the device baseline");
          if (epilogue == LinearEpilogue::GateUp)
            gateBound = std::max(gateBound, baseline.plan(w).gateScratchBytes());
        }
      }
      require(plans.gateUpWorkspace(matrix) == gateBound &&
                  gateBound == (family == 9 ? 0 : uint64_t{32} * matrix.outputSize * 2),
              "gate/up workspace disagrees with fused or decomposed baseline");
      for (uint32_t rows : {1U, 17U, 2048U})
        for (auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                              LinearEpilogue::UpWithGate}) {
          const LinearWorkload w{matrix, rows, LinearPhase::Prefill, epilogue};
          require(plans.linear().plan(w).configuration() ==
                      baseline.plan(w).configuration(),
                  "empty choices changed prefill baseline");
        }
    }
    for (auto shape : attentionShapes) {
      const auto memory = plans.prefillAttentionWorkspace(2048, shape.queryHeads,
                                                         layout(shape));
      equalWorkspace(memory, PagedAttention::prefillWorkspace(
                                 2048, shape.queryHeads, layout(shape)),
                     attentionFields);
      for (uint32_t rows = 1; rows <= 2048; ++rows) {
        const auto selected = plans.prefillAttention(rows, shape.queryHeads,
                                                     layout(shape), 2049);
        const auto expected = PagedAttention::prefillPlan(rows, shape.queryHeads,
                                                         layout(shape), 2049);
        require(selected.configuration == expected.configuration &&
                    selected.sameExecutionAs(expected),
                "empty choices changed attention baseline");
        covers(memory, selected.workspace, 1, attentionFields);
      }
      const auto stride = plans.verifyAttentionWorkspacePerLane(shape.queryHeads,
                                                                layout(shape));
      equalWorkspace(stride,
                     PagedAttention::verifyWorkspace(1, shape.queryHeads, layout(shape)),
                     attentionFields);
      for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
        const std::array<uint32_t, 4> histories{0, 31, 2048, 8192};
        const auto selected = plans.verifyAttention(lanes, shape.queryHeads,
                                                    layout(shape), histories);
        require(selected.splits == 32, "verify baseline changed");
        require(selected.configuration == VerifyAttentionConfig{},
                "verify baseline changed with GPU family");
        covers(stride, selected.workspace, lanes, attentionFields);
      }
      {
        const std::array<uint32_t, 4> deep{131072, 0, 0, 0};
        const auto scaled = plans.verifyAttention(1, shape.queryHeads,
                                                  layout(shape), deep);
        require(scaled.splits == kv::kQ8VerifyMaximumSplits &&
                    scaled.laneSplits[0] == scaled.splits,
                "verify splits did not scale with history");
        covers(stride, scaled.workspace, 1, attentionFields);
      }
    }
    for (auto shape : draftShapes) {
      const auto stride = plans.draftAttentionWorkspacePerLane(shape);
      equalWorkspace(stride, DraftAttention::plan(shape, 1).workspace(),
                     draftFields);
      for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
        const auto selected = plans.draftAttention(shape, lanes);
        require(selected.configuration() == DraftAttentionConfiguration{}, "draft baseline changed");
        covers(stride, selected.workspace(), lanes, draftFields);
      }
    }
    for (auto shape : moeShapes) {
      const auto stride = plans.moeDecodeWorkspacePerLane(shape);
      equalWorkspace(stride, MoE::decodePlan(shape, 1).workspace(), moeFields);
      const auto prefill = plans.moePrefillWorkspace(shape, 2048);
      for (uint32_t rows = 1; rows <= 2048; ++rows)
        covers(prefill, plans.moePrefill(shape, rows).workspace(), 1, moeFields);
      for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
        const auto selected = plans.moeDecode(shape, lanes);
        require(selected.tileRows() == 8 &&
                    selected.config().m8Simdgroups == moeDecodeSimdgroups(family),
                "MoE decode baseline changed");
        covers(stride, selected.workspace(), lanes, moeFields);
      }
    }
  }
}

// Apple9 decode plans run the four-simdgroup 8-row expert tiles; every other
// family, and prefill on every family, keeps the shipped N128 x 8 tile. The
// choice is the device's: candidates carry it and installed tables cannot
// override it.
void moeDeviceTiles() {
  for (uint32_t family : {0U, 9U, 10U, 11U}) {
    const auto expected = family == 9 ? MoeExpertSimdgroups::Four
                                      : MoeExpertSimdgroups::Eight;
    require(moeDecodeSimdgroups(family) == expected,
            "decode expert simdgroups are not gated on GPU family 9");
    ExecutionPlans plans(device(family));
    for (auto shape : moeShapes) {
      for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
        const MoeWorkload workload{shape, lanes * 8, MoePhase::Decode};
        require(plans.moeDecode(shape, lanes).config().m8Simdgroups == expected,
                "MoE decode plan departed from the device tile policy");
        for (const auto &candidate : plans.moeCandidates(workload)) {
          require(candidate.config().m8Simdgroups == expected,
                  "MoE decode candidate departed from the device tile policy");
          OperatorChoices choices;
          choices.moe.push_back({workload, candidate.config()});
          choices.moe.back().configuration.m8Simdgroups =
              expected == MoeExpertSimdgroups::Four ? MoeExpertSimdgroups::Eight
                                                    : MoeExpertSimdgroups::Four;
          plans.install(choices);
          require(plans.moeDecode(shape, lanes).config() == candidate.config(),
                  "installed MoE choice overrode the device tile policy");
        }
      }
      for (uint32_t rows : {1U, 8U, 17U, 2048U}) {
        require(plans.moePrefill(shape, rows).config().m8Simdgroups ==
                    MoeExpertSimdgroups::Eight,
                "MoE prefill plan left the shipped expert tile");
        for (const auto &candidate : plans.moeCandidates({shape, rows, MoePhase::Prefill}))
          require(candidate.config().m8Simdgroups == MoeExpertSimdgroups::Eight,
                  "MoE prefill candidate left the shipped expert tile");
      }
    }
  }
}

void allCandidates() {
  ExecutionPlans plans(device());
  const ExecutionPlans shipped(device());
  const Q4Linear baseline(device());
  for (auto matrix : matrices) {
    for (uint32_t rows : {1U, 17U, 2048U, 8U, 16U, 24U, 32U}) {
      const auto phase = rows == 1 || rows == 17 || rows == 2048
                             ? LinearPhase::Prefill : LinearPhase::Decode;
      for (auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                            phase == LinearPhase::Prefill
                                ? LinearEpilogue::UpWithGate
                                : LinearEpilogue::GateUp}) {
        const LinearWorkload w{matrix, rows, phase, epilogue};
        for (const auto &candidate : baseline.candidates(w)) {
          OperatorChoices choices;
          choices.linear.push_back({w, candidate.configuration()});
          plans.install(choices);
          const auto selected = plans.linear().plan(w);
          require(selected.configuration() == candidate.configuration() &&
                      selected.pipeline() == candidate.pipeline() &&
                      selected.threadsPerThreadgroup() == candidate.threadsPerThreadgroup(),
                  "linear candidate configuration/pipeline/scope not selected together");
          if (candidate.configuration().simdgroups == LinearSimdgroups::Four) {
            const auto original = shipped.linear().plan(w);
            require(selected.storageRows() == original.storageRows() &&
                        selected.sumsBytes() == original.sumsBytes() &&
                        selected.gateScratchBytes() == original.gateScratchBytes() &&
                        selected.downSumsBytes() == original.downSumsBytes() &&
                        plans.gateUpWorkspace(matrix) == shipped.gateUpWorkspace(matrix),
                    "four-SIMDgroup choice changed an external workspace requirement");
          }
          require(plans.gateUpWorkspace(matrix) >= candidate.gateScratchBytes() ||
                      phase == LinearPhase::Prefill,
                  "gate scratch omitted a selected decode width");
        }
      }
    }
  }
  for (auto shape : attentionShapes) {
    for (uint32_t rows : {1U, 9U, 17U, 2048U}) {
      for (auto config : PagedAttention::prefillCandidates()) {
        OperatorChoices choices;
        choices.prefillAttention.push_back({{shape, rows}, config});
        plans.install(choices);
        const auto selected = plans.prefillAttention(rows, shape.queryHeads,
                                                     layout(shape), 2049);
        require(selected.configuration == config, "prefill candidate not selected");
        covers(plans.prefillAttentionWorkspace(2048, shape.queryHeads, layout(shape)),
               selected.workspace, 1, attentionFields);
      }
    }
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      std::array<uint32_t, 4> histories{};
      for (uint32_t lane = 0; lane < lanes; ++lane) histories[lane] = 31 + lane;
      for (auto config : PagedAttention::verifyCandidates()) {
        OperatorChoices choices;
        choices.verifyAttention.push_back({{shape, lanes}, config});
        plans.install(choices);
        const auto selected = plans.verifyAttention(lanes, shape.queryHeads,
                                                    layout(shape), histories);
        require(selected.configuration == config, "verify candidate not selected");
        covers(plans.verifyAttentionWorkspacePerLane(shape.queryHeads, layout(shape)),
               selected.workspace, lanes, attentionFields);
      }
    }
  }
  for (auto shape : draftShapes)
    for (uint32_t lanes = 1; lanes <= 4; ++lanes)
      for (auto config : DraftAttention::candidates(shape)) {
        OperatorChoices choices;
        choices.draftAttention.push_back({{shape, lanes}, config});
        plans.install(choices);
        const auto selected = plans.draftAttention(shape, lanes);
        require(selected.configuration() == config, "draft candidate not selected");
        covers(plans.draftAttentionWorkspacePerLane(shape), selected.workspace(),
               lanes, draftFields);
      }
  for (auto shape : moeShapes) {
    for (uint32_t rows : {1U, 17U, 2048U})
      for (const auto &candidate : plans.moeCandidates({shape, rows, MoePhase::Prefill})) {
        OperatorChoices choices;
        choices.moe.push_back({{shape, rows, MoePhase::Prefill}, candidate.config()});
        plans.install(choices);
        require(plans.moePrefill(shape, rows).config() == candidate.config(),
                "MoE prefill candidate not selected");
        covers(plans.moePrefillWorkspace(shape, 2048), candidate.workspace(), 1,
               moeFields);
      }
    for (uint32_t lanes = 1; lanes <= 4; ++lanes)
      for (const auto &candidate : plans.moeCandidates({shape, lanes * 8, MoePhase::Decode})) {
        OperatorChoices choices;
        choices.moe.push_back({{shape, lanes * 8, MoePhase::Decode}, candidate.config()});
        plans.install(choices);
        require(plans.moeDecode(shape, lanes).config() == candidate.config(),
                "MoE decode candidate not selected");
        covers(plans.moeDecodeWorkspacePerLane(shape), candidate.workspace(), lanes,
               moeFields);
      }
  }
}

OperatorChoices mixedChoices() {
  OperatorChoices choices;
  choices.linear.push_back({{matrices[0], 8, LinearPhase::Decode,
                              LinearEpilogue::GateUp}, {LinearTile::N256, 60}});
  choices.prefillAttention.push_back({{attentionShapes[0]},
                                      {PrefillSplitMultiplier::Two}});
  choices.verifyAttention.push_back({{attentionShapes[0], 3},
                                     {VerifySplitCount::One}});
  choices.draftAttention.push_back({{draftShapes[0], 3}, {32}});
  choices.moe.push_back({{routedShape, 24, MoePhase::Decode}, {MoeExpertTile::M32}});
  choices.moe.push_back({{routedShape, 9, MoePhase::Prefill}, {MoeExpertTile::M8}});
  return choices;
}

void requireMixed(const ExecutionPlans &plans) {
  require(plans.linear().plan(mixedChoices().linear[0].workload).configuration() ==
              LinearConfig{LinearTile::N256, 60}, "linear table was partially replaced");
  require(plans.prefillAttention(2048, 24, layout(attentionShapes[0]), 2049).configuration.splitMultiplier ==
              PrefillSplitMultiplier::Two,
          "prefill table was partially replaced");
  const std::array<uint32_t, 3> histories{31, 32, 2049};
  const auto verify = plans.verifyAttention(3, 24, layout(attentionShapes[0]), histories);
  require(verify.configuration.splitCount == VerifySplitCount::One &&
              verify.laneSplits[0] == 1 &&
              verify.laneSplits[2] == kv::q8VerifyAttentionSplits(1, 2049, 8),
          "verify table was partially replaced");
  require(plans.draftAttention(draftShapes[0], 3).configuration().groups == 32,
          "draft table was partially replaced");
  require(plans.moeDecode(routedShape, 3).tileRows() == 32 &&
              plans.moePrefill(routedShape, 9).tileRows() == 8,
          "MoE table was partially replaced");
}

void policyKeysAndBounds() {
  ExecutionPlans plans(device());
  auto choices = mixedChoices();
  // Deliberately unsorted: each shape has an exact 2048-row selection.
  choices.prefillAttention.push_back({{attentionShapes[1]},
                                      {PrefillSplitMultiplier::Two}});
  plans.install(choices);
  requireMixed(plans);
  const kv::Layout bf16{1, 4, 256, kv::Format::BFloat16};
  const std::array<uint32_t, 3> bf16Histories{31, 32, 2049};
  require(plans.prefillAttention(2048, 24, bf16, 2049).configuration ==
              PrefillAttentionConfig{} &&
              plans.verifyAttention(3, 24, bf16, bf16Histories).configuration ==
              VerifyAttentionConfig{},
          "INT8 calibration leaked into the BF16 policy");
  require(!plans.prefillAttention(2048, 24, bf16, 2049).sameExecutionAs(
              plans.prefillAttention(2048, 24, layout(attentionShapes[0]), 2049)),
          "different cache formats aliased the same execution plan");
  require(plans.prefillAttention(2048, 24, layout(attentionShapes[0], 64), 0).sameExecutionAs(
              plans.prefillAttention(2048, 24, layout(attentionShapes[0]), 0)),
          "layer count leaked into one-layer plan identity");
  for (uint32_t history : {1U, 31U, 32U, 33U, 2047U, 2048U, 2050U, 131079U})
    require(plans.prefillAttention(2048, 24, layout(attentionShapes[0]), history).configuration.splitMultiplier ==
                PrefillSplitMultiplier::Two,
            "prefill policy was restricted to sampled exact histories");
  equalWorkspace(plans.prefillAttentionWorkspace(17, 24, layout(attentionShapes[0])),
                 PagedAttention::prefillWorkspace(17, 24, layout(attentionShapes[0])),
                 attentionFields);
  for (const auto shape : attentionShapes)
    for (uint32_t rows = 1; rows <= 2048; ++rows) {
      const auto selected = plans.prefillAttention(rows, shape.queryHeads, layout(shape), 131079);
      const auto bound = plans.prefillAttentionWorkspace(rows, shape.queryHeads, layout(shape));
      covers(bound, selected.workspace, 1, attentionFields);
      const auto baseline = PagedAttention::prefillPlan(rows, shape.queryHeads, layout(shape), 131079);
      if (rows < 2048) {
        require(selected.configuration == PrefillAttentionConfig{} &&
                    selected.sameExecutionAs(baseline),
                "fixed-chunk attention selection changed shorter or ragged rows");
        equalWorkspace(bound, PagedAttention::prefillWorkspace(rows, shape.queryHeads, layout(shape)),
                       attentionFields);
      } else {
        require(selected.configuration.splitMultiplier == PrefillSplitMultiplier::Two &&
                    !selected.sameExecutionAs(baseline),
                "fixed-chunk attention selection did not reach its exact row key");
        equalWorkspace(bound, PagedAttention::prefillWorkspace(
                                  rows, shape.queryHeads, layout(shape),
                                  {PrefillSplitMultiplier::Two}),
                       attentionFields);
        require(bound.partialsBytes >= selected.workspace.partialsBytes &&
                    bound.statisticsBytes >= selected.workspace.statisticsBytes,
                "fixed-chunk arena omitted selected scratch or its baseline fallback");
      }
    }
  std::array<uint32_t, 4> histories{31, 32, 2049, std::numeric_limits<uint32_t>::max()};
  const auto padded = plans.verifyAttention(3, 24, layout(attentionShapes[0]), histories);
  require(padded.configuration.splitCount == VerifySplitCount::One &&
              padded.laneSplits[3] == 0 &&
              padded.splits == kv::q8VerifyAttentionSplits(1, 2049, 8),
          "padded inactive lookup history was not ignored");
  std::swap(histories[0], histories[1]);
  const auto swapped = plans.verifyAttention(3, 24, layout(attentionShapes[0]), histories);
  require(swapped.configuration.splitCount == VerifySplitCount::One &&
              swapped.splits == padded.splits,
          "verify policy did not apply to a mixed lane order");
  require(plans.verifyAttention(2, 24, layout(attentionShapes[0]), histories).splits == 32,
          "choice extrapolated to another packed width");
  require(plans.verifyAttention(3, 16, layout(attentionShapes[1]), histories).splits == 32,
          "choice extrapolated to another GQA shape");
  const auto verify = plans.verifyAttentionWorkspacePerLane(24, layout(attentionShapes[0]));
  require(verify.partialsBytes ==
                  uint64_t{8} * kv::kQ8VerifyMaximumSplits * 24 * 256 * 4 &&
              verify.statisticsBytes ==
                  uint64_t{8} * kv::kQ8VerifyMaximumSplits * 24 * 2 * 4,
          "verify workspace does not cover the maximum split count");
  const auto moe = plans.moeDecodeWorkspacePerLane(routedShape);
  require(moe.groupedInputBytes == 8432299 && moe.expertOutputBytes == 8432299 &&
              moe.expertIntermediateBytes == 2108075 && moe.groupedRoutesBytes == 8235 &&
              moe.tileDescriptorsBytes == 520 && moe.tileCountBytes == 4,
          "B3 MoE workspace must use componentwise ceiling, including baseline");
  require(plans.gateUpWorkspace(matrices[0]) == 1114112 &&
              plans.gateUpWorkspace(matrices[1]) == 393216,
          "B1 gate choice hid the B3/B4 baseline requirement");
  const auto draft = plans.draftAttentionWorkspacePerLane(draftShapes[0]);
  // Grouped queries per lane plus eight heads x four splits of 32 x 130 fp32
  // attention partials behind them.
  require(draft.convolutionBytes == 81920 && draft.qkvBytes == 98304 &&
              draft.groupedQueriesBytes == 65536 + 8 * 4 * 16640 &&
              draft.queryKeysBytes == 16384 && draft.queryValuesBytes == 16384,
          "draft workspace ABI changed");
  require(plans.draftAttention(draftShapes[0], 2).configuration() == DraftAttentionConfiguration{} &&
              plans.draftAttention(draftShapes[1], 3).configuration() == DraftAttentionConfiguration{},
          "draft choice leaked across width or shape");
  plans.install({});
  require(plans.moeDecode(routedShape, 3).tileRows() == 8 &&
              plans.draftAttention(draftShapes[0], 3).configuration() == DraftAttentionConfiguration{} &&
              plans.prefillAttention(2048, 24, layout(attentionShapes[0]), 0).configuration == PrefillAttentionConfig{},
          "empty install did not reset all tables");
}

void atomicInvalidChoices() {
  ExecutionPlans plans(device());
  plans.install(mixedChoices());
  const auto invalid = [&](auto change) {
    auto pending = mixedChoices();
    pending.linear[0].configuration.groups = 32;
    pending.draftAttention[0].configuration.groups = 80;
    change(pending);
    rejects([&] { plans.install(pending); });
    requireMixed(plans);
  };
  invalid([](auto &c) { c.linear[0].configuration.groups = 0; });
  invalid([](auto &c) { c.linear[0].configuration.simdgroups = LinearSimdgroups::Four; });
  invalid([](auto &c) { c.linear[0].configuration.simdgroups = LinearSimdgroups(6); });
  invalid([](auto &c) { c.linear[0].workload.rows = 9; });
  invalid([](auto &c) { c.prefillAttention[0].configuration.splitMultiplier = PrefillSplitMultiplier(0); });
  invalid([](auto &c) { c.prefillAttention[0].configuration.splitMultiplier = PrefillSplitMultiplier(3); });
  invalid([](auto &c) { c.prefillAttention[0].configuration.scalePlacement = AttentionScalePlacement(2); });
  invalid([](auto &c) { c.prefillAttention[0].workload.shape.queryHeads = 32; });
  invalid([](auto &c) { c.prefillAttention[0].workload.rows = 0; });
  invalid([](auto &c) { c.prefillAttention[0].workload.rows = 2049; });
  invalid([](auto &c) { c.verifyAttention[0].configuration.splitCount = VerifySplitCount(0); });
  invalid([](auto &c) { c.verifyAttention[0].configuration.scalePlacement = AttentionScalePlacement(2); });
  invalid([](auto &c) { c.verifyAttention[0].workload.lanes = 5; });
  invalid([](auto &c) { c.verifyAttention[0].workload.lanes = 0; });
  invalid([](auto &c) { c.draftAttention[0].configuration.groups = 1; });
  invalid([](auto &c) { c.draftAttention[0].workload.shape.dynamicSize = 256; });
  invalid([](auto &c) { c.draftAttention[0].workload.lanes = 0; });
  invalid([](auto &c) { c.moe[0].configuration.expertTile = MoeExpertTile(16); });
  invalid([](auto &c) { c.moe[0].configuration.m8Simdgroups = MoeExpertSimdgroups(6); });
  invalid([](auto &c) { c.moe[0].workload.rows = 9; });
  invalid([](auto &c) { c.moe[0].workload.rows = 40; });
  invalid([](auto &c) { c.moe[0].workload.phase = MoePhase(255); });
  invalid([](auto &c) { c.moe[0].workload.shape.expertsPerToken = 257; });
  invalid([](auto &c) { c.linear.push_back(c.linear[0]); });
  invalid([](auto &c) { c.prefillAttention.push_back(c.prefillAttention[0]); });
  invalid([](auto &c) { c.verifyAttention.push_back(c.verifyAttention[0]); });
  invalid([](auto &c) { c.draftAttention.push_back(c.draftAttention[0]); });
  invalid([](auto &c) { c.moe.push_back(c.moe[0]); });
}

void invalidLookupsAndContextEdges() {
  ExecutionPlans plans(device());
  const auto kvLayout = layout(attentionShapes[0]);
  const std::array<uint32_t, 4> histories{0, 1, 2, 3};
  rejects([&] { (void)plans.verifyAttention(0, 24, kvLayout, histories); });
  rejects([&] { (void)plans.verifyAttention(UINT32_MAX, 24, kvLayout, histories); });
  rejects([&] { (void)plans.verifyAttention(3, 24, kvLayout, std::span(histories).first(2)); });
  rejects([&] { (void)plans.verifyAttention(1, 24, {}, histories); });
  rejects([&] { (void)plans.prefillAttention(1, 24, kvLayout, UINT32_MAX); });
  rejects([&] { (void)plans.prefillAttention(1, 24, {}, 0); });
  rejects([&] { (void)plans.prefillAttentionWorkspace(0, 24, kvLayout); });
  rejects([&] { (void)plans.prefillAttentionWorkspace(UINT32_MAX, 24, kvLayout); });
  rejects([&] { (void)plans.moeDecode(routedShape, UINT32_MAX); });
  rejects([&] { (void)plans.moeDecode(routedShape, 0); });
  rejects([&] { (void)plans.moePrefillWorkspace(routedShape, 0); });
  rejects([&] { (void)plans.gateUpWorkspace({256, 64}); });
  rejects([&] { (void)plans.draftAttentionWorkspacePerLane({}); });
  OperatorChoices choices;
  choices.prefillAttention.push_back({{attentionShapes[0]},
                                      {PrefillSplitMultiplier::Two}});
  choices.verifyAttention.push_back({{attentionShapes[0], 1},
                                     {VerifySplitCount::One}});
  plans.install(choices);
  const auto finalPrefill = plans.prefillAttention(
      1, 24, kvLayout, kv::kMaximumPhysicalTokens - 1);
  require(finalPrefill.rows == 1 &&
              finalPrefill.historyTokens == kv::kMaximumPhysicalTokens - 1 &&
              finalPrefill.splits == 32,
          "valid final physical token was rejected");
  std::array<uint32_t, 1> edge{kv::kMaximumPhysicalTokens - 8};
  const auto finalVerify = plans.verifyAttention(1, 24, kvLayout, edge);
  require(finalVerify.configuration.splitCount == VerifySplitCount::One &&
              finalVerify.splits == kv::kQ8VerifyMaximumSplits,
          "valid final physical verify rows were rejected");
  ++edge[0];
  rejects([&] { (void)plans.verifyAttention(1, 24, kvLayout, edge); });
}
} // namespace

int main() {
  try {
    baselinePlans();
    moeDeviceTiles();
    allCandidates();
    policyKeysAndBounds();
    atomicInvalidChoices();
    invalidLookupsAndContextEdges();
    std::cout << "PASS execution plans: typed policies, device MoE tiles, atomic "
                 "install, all candidates, B1-B4 and prefill workspace bounds "
                 "(CPU only)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL execution plans: " << error.what() << '\n';
    return 1;
  }
}

#include "TestChecks.hpp"
#include "ops/ExecutionPlans.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace splash;
using namespace splash::ops;

using splash::test::rejects;
using splash::test::require;

// The target attention shapes: query heads over a KV layout.
struct AttentionShape final {
  uint32_t queryHeads;
  kv::Layout layout;
};
constexpr std::array attentionShapes{
    AttentionShape{24, {1, 4, 256}}, AttentionShape{16, {1, 2, 256}}};
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
// The gate/up projection of `matrix`, which gateUpWorkspace sizes.
constexpr ProjectionShape gateUp(LinearMatrix matrix) { return {matrix.outputSize, matrix.inputSize}; }
constexpr std::array attentionFields{
    &AttentionWorkspace::partialsBytes, &AttentionWorkspace::statisticsBytes};
constexpr std::array draftFields{
    &DraftAttentionWorkspace::convolutionBytes,
    &DraftAttentionWorkspace::qkvBytes,
    &DraftAttentionWorkspace::groupedQueriesBytes,
    &DraftAttentionWorkspace::queryKeysBytes,
    &DraftAttentionWorkspace::queryValuesBytes};

DeviceCapabilities device(uint32_t family = 10) {
  DeviceCapabilities value;
  value.appleGpuFamily = family;
  return value;
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
    const Linear baseline(device(family));
    for (auto matrix : matrices) {
      uint64_t gateBound = 0;
      for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
        for (auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                              LinearEpilogue::GateUp}) {
          const LinearWorkload w{matrix, lanes * 8, LinearPhase::Decode, epilogue};
          require(plans.linear().plan(w).configuration() ==
                      baseline.plan(w).configuration(),
                  "the plans' decode Linear departed from the device policy");
          if (epilogue == LinearEpilogue::GateUp)
            gateBound = std::max(gateBound, baseline.plan(w).gateScratchBytes());
        }
      }
      require(plans.gateUpWorkspace(gateUp(matrix)) == gateBound &&
                  gateBound == uint64_t{32} * matrix.outputSize * 2,
              "gate/up workspace disagrees with the baseline's gate pass");
      for (uint32_t rows : {1U, 17U, 2048U})
        for (auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                              LinearEpilogue::UpWithGate}) {
          const LinearWorkload w{matrix, rows, LinearPhase::Prefill, epilogue};
          require(plans.linear().plan(w).configuration() ==
                      baseline.plan(w).configuration(),
                  "the plans' prefill Linear departed from the device policy");
        }
    }
    for (const auto &[queryHeads, kvLayout] : attentionShapes) {
      const auto memory = plans.prefillAttentionWorkspace(2048, queryHeads, kvLayout);
      for (uint32_t rows = 1; rows <= 2048; ++rows)
        covers(memory, plans.prefillAttention(rows, queryHeads, kvLayout).workspace, 1,
               attentionFields);
      const auto stride = plans.verifyAttentionWorkspacePerLane(queryHeads, kvLayout);
      for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
        const std::array<uint32_t, 4> histories{0, 31, 2048, 8192};
        const auto selected = plans.verifyAttention(lanes, queryHeads, kvLayout,
                                                    std::span(histories).first(lanes));
        require(selected.splits == 32, "verify baseline changed");
        covers(stride, selected.workspace, lanes, attentionFields);
      }
      {
        const std::array<uint32_t, 1> deep{131072};
        const auto scaled = plans.verifyAttention(1, queryHeads, kvLayout, deep);
        require(scaled.splits == kv::kVerifyMaximumSplits &&
                    scaled.laneSplits[0] == scaled.splits,
                "verify splits did not scale with history");
        covers(stride, scaled.workspace, 1, attentionFields);
      }
    }
    for (auto shape : draftShapes) {
      const auto stride = plans.draftAttentionWorkspacePerLane(shape);
      for (uint32_t lanes = 1; lanes <= 4; ++lanes)
        covers(stride, plans.draftAttention(shape, lanes).workspace(), lanes, draftFields);
    }
    for (auto shape : moeShapes) {
      const auto stride = plans.moeDecodeWorkspacePerLane(shape);
      require(stride == plans.moeDecode(shape, 1).workspace(), "workspace bound changed");
      const auto prefill = plans.moePrefillWorkspace(shape, 2048);
      for (uint32_t rows = 1; rows <= 2048; ++rows)
        covers(prefill, plans.moePrefill(shape, rows).workspace(), 1, kMoeWorkspaceFields);
      for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
        const auto selected = plans.moeDecode(shape, lanes);
        require(selected.tileRows() == 8, "MoE decode baseline changed");
        covers(stride, selected.workspace(), lanes, kMoeWorkspaceFields);
      }
    }
  }
}

// MoE plans run the three expert passes: the exact register tile on Apple9,
// with its Table16 row sums in the workspace bounds, staged tiles everywhere
// else (32-row tiles for prefill chunks past one route per expert).
void ggufMoePlans() {
  const MoeShape shape = routedShape;
  for (uint32_t family : {9U, 10U, 11U}) {
    ExecutionPlans plans(device(family));
    const MoeGgufTile expected = family == 9 ? MoeGgufTile::Register : MoeGgufTile::Staged;
    require(moeGgufTile(gpuFamilyClass(family), shape) == expected, "GGUF expert tile is not gated on GPU family 9");
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      const MoePlan plan = plans.moeDecode(shape, lanes);
      require(plan.configuration().ggufTile == expected && plan.tileRows() == 8 &&
                  plan.configuration().ggufRouterTile == FloatTile::Simdgroup,
              "GGUF MoE decode plan left its device tile");
      // Sums of the widest input (hidden, 3 K / 4 fp32) per 8-row tile.
      require(plan.workspace().groupedSumsBytes ==
                  (expected == MoeGgufTile::Register ? uint64_t{plan.maximumTiles()} * 2048 * 3 : 0),
              "GGUF register plan sums its Table16 tiles");
      covers(plans.moeDecodeWorkspacePerLane(shape), plan.workspace(), lanes, kMoeWorkspaceFields);
    }
    // Apple9 stages experts mostly in a format it stages (IQ2_XS: UD-Q2_K_XL)
    // and keeps the register tile for the others (Q4_K: UD-Q4_K_M).
    MoeShape staged = shape, q4k = shape;
    staged.expertFormat = GGUF_FMT_IQ2XS;
    q4k.expertFormat = GGUF_FMT_Q4K;
    require(moeGgufTile(gpuFamilyClass(family), staged) == MoeGgufTile::Staged &&
                moeGgufTile(gpuFamilyClass(family), q4k) == expected,
            "GGUF expert tile does not follow the experts' format on GPU family 9");
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      const MoePlan plan = plans.moeDecode(staged, lanes);
      require(plan.configuration().ggufTile == MoeGgufTile::Staged && plan.workspace().groupedSumsBytes == 0,
              "GGUF MoE plan of staged experts took the register tile");
      covers(plans.moeDecodeWorkspacePerLane(staged), plan.workspace(), lanes, kMoeWorkspaceFields);
    }
    // Prefill: the register tile's 8 rows, or staged 8-row tiles while the
    // routes average at most one row per expert (32 rows of 8 of 256
    // experts). The router's float tile follows Linear::ggufFloatTile (32
    // assumed cores: the neural accelerator from 321 rows, never on Apple9).
    for (uint32_t rows : {1U, 8U, 17U, 32U, 33U, 100U, 256U, 257U, 320U, 321U, 2048U}) {
      const MoePlan plan = plans.moePrefill(shape, rows);
      const uint32_t tileRows = expected == MoeGgufTile::Register || rows <= 32 ? 8 : 32;
      const FloatTile router = family != 9 && rows > 320 ? FloatTile::NeuralAccelerator : FloatTile::Simdgroup;
      require(plan.configuration().ggufTile == expected && plan.tileRows() == tileRows &&
                  (plan.workspace().groupedSumsBytes > 0) == (expected == MoeGgufTile::Register) &&
                  plan.configuration().ggufRouterTile == router,
              "GGUF MoE prefill plan left the device's tile");
      covers(plans.moePrefillWorkspace(shape, 2048), plan.workspace(), 1, kMoeWorkspaceFields);
    }
    // The prefill bound holds the device's plans and nothing else: on Apple9
    // the register tile's 8-row tiles (20480 grouped rows at 2048 rows), not
    // the staged 32-row tiles it never runs (26624).
    MoeWorkspace devicePlans;
    for (uint32_t rows = 1; rows <= 2048; ++rows) {
      const MoeWorkspace workspace = plans.moePrefill(shape, rows).workspace();
      for (const auto field : kMoeWorkspaceFields) devicePlans.*field = std::max(devicePlans.*field, workspace.*field);
    }
    require(plans.moePrefillWorkspace(shape, 2048) == devicePlans,
            "GGUF MoE prefill bound is not the bound of the device's plans");
  }
  // The register tile reads 8-row tiles only, and the kernels exist for
  // 8-row tiles and 32-row prefill tiles only.
  rejects(
      [&] {
        (void)MoE::prefillPlan(shape, 1, {.expertTile = MoeExpertTile::M32, .ggufTile = MoeGgufTile::Register});
      },
      "the register expert tile takes 8-row tiles", "the register tile took 32-row tiles");
  rejects([&] { (void)MoE::decodePlan(shape, 1, {MoeExpertTile::M32}); },
          "invalid MoE expert tile configuration", "a decode plan took 32-row tiles");
}

// A device that reports no core count gets the plans of kAssumedGpuCores
// cores, Linear and MoE alike.
void unknownCoreCount() {
  for (uint32_t family : {9U, 10U}) {
    DeviceCapabilities assumed = device(family);
    assumed.gpuCoreCount = kAssumedGpuCores;
    const ExecutionPlans unknown(device(family)), planned(assumed);
    for (auto matrix : matrices)
      for (uint32_t lanes = 1; lanes <= 4; ++lanes)
        for (auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp}) {
          const LinearWorkload w{matrix, lanes * 8, LinearPhase::Decode, epilogue};
          require(unknown.linear().plan(w).configuration() == planned.linear().plan(w).configuration(),
                  "an unknown core count planned a decode projection for other than the assumed cores");
        }
    for (uint32_t lanes = 1; lanes <= 4; ++lanes)
      require(unknown.moeDecode(routedShape, lanes).configuration() ==
                  planned.moeDecode(routedShape, lanes).configuration(),
              "an unknown core count planned a MoE decode step for other than the assumed cores");
  }
}

void workspaceBounds() {
  const ExecutionPlans plans(device());
  const std::array<uint32_t, 4> histories{31, 32, 2049, std::numeric_limits<uint32_t>::max()};
  rejects([&] { (void)plans.verifyAttention(3, 24, attentionShapes[0].layout, histories); },
          "invalid verify attention history vector", "a verify plan took more histories than lanes");
  const auto exact =
      plans.verifyAttention(3, 24, attentionShapes[0].layout, std::span(histories).first(3));
  require(exact.laneSplits[3] == 0 && exact.splits == kv::verifyAttentionSplits(2049),
          "verify policy did not resolve one history per lane");
  const auto verify = plans.verifyAttentionWorkspacePerLane(24, attentionShapes[0].layout);
  require(verify.partialsBytes ==
                  uint64_t{8} * kv::kVerifyMaximumSplits * 24 * 256 * 4 &&
              verify.statisticsBytes ==
                  uint64_t{8} * kv::kVerifyMaximumSplits * 24 * 2 * 4,
          "verify workspace does not cover the maximum split count");
  // 65 tiles of 8 grouped rows per lane at every width.
  const auto moe = plans.moeDecodeWorkspacePerLane(routedShape);
  require(moe.groupedInputBytes == 2129920 && moe.expertOutputBytes == 2129920 &&
              moe.expertIntermediateBytes == 532480 && moe.groupedRoutesBytes == 2080 &&
              moe.tileDescriptorsBytes == 520 && moe.tileCountBytes == 28,
          "MoE decode workspace per lane changed");
  require(plans.gateUpWorkspace(gateUp(matrices[0])) == 1114112 &&
              plans.gateUpWorkspace(gateUp(matrices[1])) == 393216,
          "gate/up workspace omitted the B3/B4 gate pass");
  const auto draft = plans.draftAttentionWorkspacePerLane(draftShapes[0]);
  // Grouped queries per lane plus eight heads x four splits of 32 x 130 fp32
  // attention partials behind them.
  require(draft.convolutionBytes == 81920 && draft.qkvBytes == 98304 &&
              draft.groupedQueriesBytes == 65536 + 8 * 4 * 16640 &&
              draft.queryKeysBytes == 16384 && draft.queryValuesBytes == 16384,
          "draft workspace ABI changed");
}

void invalidLookupsAndContextEdges() {
  const ExecutionPlans plans(device());
  const auto kvLayout = attentionShapes[0].layout;
  const std::array<uint32_t, 4> histories{0, 1, 2, 3};
  for (const uint32_t lanes : {0U, UINT32_MAX})
    rejects([&] { (void)plans.verifyAttention(lanes, 24, kvLayout, histories); },
            "invalid attention workspace batch width", "a verify plan of no lanes or past a batch was accepted");
  rejects([&] { (void)plans.verifyAttention(3, 24, kvLayout, std::span(histories).first(2)); },
          "invalid verify attention history vector", "a verify plan without a history per lane was accepted");
  rejects([&] { (void)plans.verifyAttention(1, 24, {}, std::span(histories).first(1)); },
          "no KV store kernel for layout", "a verify plan of an empty KV layout was accepted");
  rejects([&] { (void)plans.prefillAttention(1, 24, {}); }, "no KV store kernel for layout",
          "a prefill plan of an empty KV layout was accepted");
  for (const uint32_t rows : {0U, UINT32_MAX})
    rejects([&] { (void)plans.prefillAttentionWorkspace(rows, 24, kvLayout); },
            "invalid attention workspace rows", "a prefill workspace of no rows or past the budget was accepted");
  for (const uint32_t lanes : {UINT32_MAX, 0U})
    rejects([&] { (void)plans.moeDecode(routedShape, lanes); }, "invalid MoE decode width",
            "a MoE decode of no lanes or past a batch was accepted");
  rejects([&] { (void)plans.moePrefillWorkspace(routedShape, 0); }, "invalid MoE prefill rows",
          "a MoE prefill workspace of no rows was accepted");
  rejects([&] { (void)plans.gateUpWorkspace({256, 32}); }, "invalid linear matrix",
          "a gate/up workspace of a partial 64-input span was accepted");
  rejects([&] { (void)plans.draftAttentionWorkspacePerLane({}); },
          "unsupported compiled draft attention shape", "a draft workspace of an empty shape was accepted");
  std::array<uint32_t, 1> edge{kv::kMaximumPhysicalTokens - 8};
  const auto finalVerify = plans.verifyAttention(1, 24, kvLayout, edge);
  require(finalVerify.splits == kv::kVerifyMaximumSplits,
          "valid final physical verify rows were rejected");
  ++edge[0];
  rejects([&] { (void)plans.verifyAttention(1, 24, kvLayout, edge); },
          "verify attention history exceeds physical context",
          "verify rows past the physical context were accepted");
}
} // namespace

int main() {
  try {
    baselinePlans();
    ggufMoePlans();
    unknownCoreCount();
    workspaceBounds();
    invalidLookupsAndContextEdges();
    std::cout << "PASS execution plans: device policies, device MoE tiles, B1-B4 "
                 "and prefill workspace bounds (CPU only)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL execution plans: " << error.what() << '\n';
    return 1;
  }
}

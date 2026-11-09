#include "TestChecks.hpp"
#include "ops/ExecutionPlans.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

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
// A MoE target of F32 routers.
constexpr uint32_t kRouterFormat = QuantizedSegment::kFloat32;
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
    const ExecutionPlans plans(device(family), Int8Activations::On);
    const Linear baseline(device(family), Int8Activations::On);
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
        for (const uint32_t history : {0U, 32768U, 1'000'000U})
          covers(memory, plans.prefillAttention(rows, history, queryHeads, kvLayout).workspace, 1,
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
      const auto stride = plans.moeDecodeWorkspacePerLane(shape, kRouterFormat);
      require(stride == plans.moeDecode(shape, kRouterFormat, 1).workspace(), "workspace bound changed");
      const auto prefill = plans.moePrefillWorkspace(shape, kRouterFormat, 2048);
      for (uint32_t rows = 1; rows <= 2048; ++rows)
        covers(prefill, plans.moePrefill(shape, kRouterFormat, rows).workspace(), 1, kMoeWorkspaceFields);
      for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
        const auto selected = plans.moeDecode(shape, kRouterFormat, lanes);
        require(selected.tileRows() == 8, "MoE decode baseline changed");
        covers(stride, selected.workspace(), lanes, kMoeWorkspaceFields);
      }
    }
  }
}

// The int8-activation setting reaches every selection that reads it and
// leaves the device's row as measured: on, Apple10 and Apple11 run an af4g64
// projection's 2048-row chunk on the int8 prefill tile and score INT8 queries
// in a 64-row chunk after 32K tokens (33 pages a split) and in a verify after
// 128K; off, no family does. kernel-selection checks every plan.
void int8ActivationsSetting() {
  const Projection af4(5120, 17408,
                       BlockWeights{{QuantizedSegment::planes(GGUF_FMT_AF4G64, 5120, 17408, {}, {}, {})}});
  const kv::Layout layout{1, 2, 256};
  const std::array<uint32_t, 1> history{131072};
  const auto int8Queries = [](const std::string &pipeline) { return pipeline.find("q8q") != std::string::npos; };
  for (uint32_t family : {9U, 10U, 11U}) {
    const ExecutionPlans on(device(family), Int8Activations::On), off(device(family), Int8Activations::Off);
    const ProjectionCalibration &row = on.calibration().projection, &offRow = off.calibration().projection;
    require(on.linear().int8Activations() == Int8Activations::On &&
                off.linear().int8Activations() == Int8Activations::Off && row.prefillInt8 == (family >= 10) &&
                offRow.prefillInt8 == row.prefillInt8 && offRow.decodeInt8FromRows == row.decodeInt8FromRows &&
                off.calibration().attention.int8QueriesFromPages == on.calibration().attention.int8QueriesFromPages,
            "the int8-activation setting did not reach the plans' Linear, or moved the device's row");
    const bool int8 = family >= 10;
    const auto tile = [&](const ExecutionPlans &plans) {
      return plans.linear().prefillPlan(af4, 2048, LinearEpilogue::Residual).configuration().tile;
    };
    require(tile(on) == (int8 ? LinearTile::GgufPrefillInt8 : LinearTile::GgufPrefill) &&
                tile(off) == LinearTile::GgufPrefill,
            "the int8 prefill tile does not follow the int8-activation setting");
    require(int8Queries(on.prefillAttention(64, 32768, 16, layout).splitPipeline) == int8 &&
                !int8Queries(off.prefillAttention(64, 32768, 16, layout).splitPipeline) &&
                int8Queries(on.verifyAttention(1, 16, layout, history).splitPipeline) == int8 &&
                !int8Queries(off.verifyAttention(1, 16, layout, history).splitPipeline),
            "INT8 attention queries do not follow the int8-activation setting");
  }
}

// A DFlash2 draft's plans (draftLinear) keep bf16 activations, int8
// activations on or off: the 27B draft's context projection, prepared af4g64,
// prefills a full chunk on the bf16 prefill tile where the target's plans
// would take the int8 one, its shorter chunks and its decode steps on the
// target's plans.
void draftPlans() {
  const Projection context(5120, 25600,
                           BlockWeights{{QuantizedSegment::planes(GGUF_FMT_AF4G64, 5120, 25600, {}, {}, {})}});
  for (uint32_t family : {9U, 10U, 11U})
    for (const Int8Activations setting : {Int8Activations::On, Int8Activations::Off}) {
      const ExecutionPlans plans(device(family), setting);
      const Linear &draft = plans.draftLinear(), &target = plans.linear();
      require(draft.int8Activations() == Int8Activations::Off, "a draft's plans took int8 activations");
      for (uint32_t rows : {1U, 64U, 2048U}) {
        const LinearConfig own = draft.prefillPlan(context, rows, LinearEpilogue::None).configuration(),
                           targets = target.prefillPlan(context, rows, LinearEpilogue::None).configuration();
        const bool int8 = family >= 10 && setting == Int8Activations::On && rows == 2048;
        require((targets.tile == LinearTile::GgufPrefillInt8) == int8 &&
                    own == (int8 ? LinearConfig{.tile = LinearTile::GgufPrefill} : targets),
                "a draft's prefill chunk took the int8 prefill tile or left the target's bf16 plan");
      }
      for (uint32_t lanes = 1; lanes <= 4; ++lanes)
        require(draft.decodePlan(context, lanes).configuration() == target.decodePlan(context, lanes).configuration(),
                "a draft's decode plan departed from the target's");
    }
}

// MoE plans run the three expert passes on staged tiles on every GPU: 8-row
// tiles in decode and while a prefill chunk's routes average at most one row
// per expert, 32-row tiles past that.
void ggufMoePlans() {
  const MoeShape shape = routedShape;
  // A GPU family and the cores its device reports (none: kAssumedGpuCores).
  for (const auto [family, reportedCores] : {std::pair{9U, 0U}, {9U, 11U}, {9U, 10U}, {10U, 0U}, {11U, 0U}}) {
    DeviceCapabilities capabilities = device(family);
    capabilities.gpuCoreCount = reportedCores;
    ExecutionPlans plans(capabilities, Int8Activations::On);
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      const MoePlan plan = plans.moeDecode(shape, kRouterFormat, lanes);
      require(plan.tileRows() == 8 && plan.configuration().ggufRouterTile == FloatTile::Simdgroup,
              "GGUF MoE decode plan left its 8-row tiles");
      covers(plans.moeDecodeWorkspacePerLane(shape, kRouterFormat), plan.workspace(), lanes, kMoeWorkspaceFields);
    }
    // Prefill: 8-row tiles while the routes average at most one row per
    // expert (32 rows of 8 of 256 experts). The router's float tile follows
    // selectFloatTile and the router's format (32 assumed cores: the neural
    // accelerator from 321 rows of BF16 values, 449 of F32 values and 705 of
    // a quantized format's rows, never on Apple9).
    for (const auto [routerFormat, first] :
         {std::pair{uint32_t{GGUF_FMT_BF16}, 321U}, std::pair{QuantizedSegment::kFloat32, 449U},
          std::pair{uint32_t{GGUF_FMT_AF8G64}, 705U}}) {
      const MoeWorkspace bound = plans.moePrefillWorkspace(shape, routerFormat, 2048);
      for (uint32_t rows : {1U, 8U, 17U, 32U, 33U, 100U, 256U, 257U, 320U, 321U, 448U, 449U, 704U, 705U, 2048U}) {
        const MoePlan plan = plans.moePrefill(shape, routerFormat, rows);
        const FloatTile router = family != 9 && rows >= first ? FloatTile::NeuralAccelerator : FloatTile::Simdgroup;
        require(plan.tileRows() == (rows <= 32 ? 8U : 32U) && plan.configuration().ggufRouterTile == router,
                "GGUF MoE prefill plan left its tiles");
        covers(bound, plan.workspace(), 1, kMoeWorkspaceFields);
      }
    }
    // The prefill bound is the bound of the device's plans: from 33 rows
    // their 32-row tiles (26624 grouped rows at 2048 rows).
    MoeWorkspace devicePlans;
    for (uint32_t rows = 1; rows <= 2048; ++rows) {
      const MoeWorkspace workspace = plans.moePrefill(shape, kRouterFormat, rows).workspace();
      for (const auto field : kMoeWorkspaceFields) devicePlans.*field = std::max(devicePlans.*field, workspace.*field);
    }
    require(plans.moePrefillWorkspace(shape, kRouterFormat, 2048) == devicePlans,
            "GGUF MoE prefill bound is not the bound of the device's plans");
  }
  // The kernels exist for 8-row tiles and 32-row prefill tiles only.
  rejects([&] { (void)MoE::decodePlan(shape, 1, {MoeExpertTile::M32}); },
          "invalid MoE expert tile configuration", "a decode plan took 32-row tiles");
}

// A dense prefill tile's history splits follow the device's pages per split
// (KernelCalibration.hpp): one per 64 pages on Apple9 GPUs of 10 cores or
// fewer, one per 32 elsewhere. Qwen3.6-35B-A3B's attention: a 512-row chunk
// after 4K tokens (144 pages) and a 64-row chunk after 32K (1026 pages).
void prefillSplitsFollowTheDeviceRow() {
  const kv::Layout layout{1, 2, 256};
  struct Expected final {
    uint32_t rows, history, small, other;
  };
  for (const auto [rows, history, small, other] : {Expected{512, 4096, 2, 4}, Expected{64, 32768, 16, 32}})
    for (const auto [family, cores] : {std::pair{9U, 10U}, {9U, 1U}, {9U, 11U}, {9U, 40U}, {10U, 12U}, {10U, 32U}}) {
      DeviceCapabilities capabilities = device(family);
      capabilities.gpuCoreCount = cores;
      const uint32_t splits =
          ExecutionPlans(capabilities, Int8Activations::On).prefillAttention(rows, history, 16, layout).splits;
      require(splits == (family == 9 && cores <= 10 ? small : other),
              "a prefill attention plan of family " + std::to_string(family) + ", " + std::to_string(cores) +
                  " cores took " + std::to_string(splits) + " splits");
    }
}

// A device that reports no core count gets the plans of kAssumedGpuCores
// cores, Linear and MoE alike.
void unknownCoreCount() {
  for (uint32_t family : {9U, 10U}) {
    DeviceCapabilities assumed = device(family);
    assumed.gpuCoreCount = DeviceCapabilities::kAssumedGpuCores;
    const ExecutionPlans unknown(device(family), Int8Activations::On), planned(assumed, Int8Activations::On);
    for (auto matrix : matrices)
      for (uint32_t lanes = 1; lanes <= 4; ++lanes)
        for (auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp}) {
          const LinearWorkload w{matrix, lanes * 8, LinearPhase::Decode, epilogue};
          require(unknown.linear().plan(w).configuration() == planned.linear().plan(w).configuration(),
                  "an unknown core count planned a decode projection for other than the assumed cores");
        }
    for (uint32_t lanes = 1; lanes <= 4; ++lanes)
      require(unknown.moeDecode(routedShape, kRouterFormat, lanes).configuration() ==
                  planned.moeDecode(routedShape, kRouterFormat, lanes).configuration(),
              "an unknown core count planned a MoE decode step for other than the assumed cores");
  }
}

void workspaceBounds() {
  const ExecutionPlans plans(device(), Int8Activations::On);
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
  const auto moe = plans.moeDecodeWorkspacePerLane(routedShape, kRouterFormat);
  require(moe.groupedInputBytes == 2129920 && moe.expertOutputBytes == 2129920 &&
              moe.expertIntermediateBytes == 532480 && moe.groupedRoutesBytes == 2080 &&
              moe.tileDescriptorsBytes == 520 && moe.tileCountBytes == 80 && moe.expertSlotsBytes == 1024,
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
  const ExecutionPlans plans(device(), Int8Activations::On);
  const auto kvLayout = attentionShapes[0].layout;
  const std::array<uint32_t, 4> histories{0, 1, 2, 3};
  for (const uint32_t lanes : {0U, UINT32_MAX})
    rejects([&] { (void)plans.verifyAttention(lanes, 24, kvLayout, histories); },
            "invalid attention workspace batch width", "a verify plan of no lanes or past a batch was accepted");
  rejects([&] { (void)plans.verifyAttention(3, 24, kvLayout, std::span(histories).first(2)); },
          "invalid verify attention history vector", "a verify plan without a history per lane was accepted");
  rejects([&] { (void)plans.verifyAttention(1, 24, {}, std::span(histories).first(1)); },
          "no KV store kernel for layout", "a verify plan of an empty KV layout was accepted");
  rejects([&] { (void)plans.prefillAttention(1, 0, 24, {}); }, "no KV store kernel for layout",
          "a prefill plan of an empty KV layout was accepted");
  for (const uint32_t rows : {0U, UINT32_MAX})
    rejects([&] { (void)plans.prefillAttentionWorkspace(rows, 24, kvLayout); },
            "invalid attention workspace rows", "a prefill workspace of no rows or past the budget was accepted");
  for (const uint32_t lanes : {UINT32_MAX, 0U})
    rejects([&] { (void)plans.moeDecode(routedShape, kRouterFormat, lanes); }, "invalid MoE decode width",
            "a MoE decode of no lanes or past a batch was accepted");
  rejects([&] { (void)plans.moePrefillWorkspace(routedShape, kRouterFormat, 0); }, "invalid MoE prefill rows",
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
    int8ActivationsSetting();
    draftPlans();
    ggufMoePlans();
    unknownCoreCount();
    prefillSplitsFollowTheDeviceRow();
    workspaceBounds();
    invalidLookupsAndContextEdges();
    std::cout << "PASS execution plans: device policies, int8 activations on and off, the draft's bf16 prefill, "
                 "device MoE tiles, B1-B4 and prefill workspace bounds (CPU only)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL execution plans: " << error.what() << '\n';
    return 1;
  }
}

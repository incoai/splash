#include "TestChecks.hpp"
#include "ops/ExecutionPlans.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

// A sequence of `rows` rows after `history` tokens, staged in whole KV
// blocks, and decode lanes of 8 rows after each of `histories`.
kv::ChunkedPrefillParams sequenceOf(uint32_t history, uint32_t rows) {
  return PagedAttention::chunkParams(history, rows, (rows + 31) / 32 * 32, (history + rows + 31) / 32);
}
std::vector<kv::ChunkedPrefillParams> lanesAfter(std::span<const uint32_t> histories) {
  std::vector<kv::ChunkedPrefillParams> lanes;
  for (const uint32_t history : histories) lanes.push_back(sequenceOf(history, 8));
  return lanes;
}
// The workspace of `lanes` lanes of `stride`'s each.
AttentionWorkspace workspaceOf(AttentionWorkspace stride, uint32_t lanes) {
  return {stride.partialsBytes * lanes, stride.statisticsBytes * lanes};
}

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
          const LinearWorkload w{matrix, lanes * 8, epilogue};
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
          const LinearWorkload w{matrix, rows, epilogue};
          require(plans.linear().plan(w).configuration() ==
                      baseline.plan(w).configuration(),
                  "the plans' prefill Linear departed from the device policy");
        }
    }
    for (const auto &[queryHeads, kvLayout] : attentionShapes) {
      const auto chunk = plans.sequenceAttentionWorkspace(queryHeads, kvLayout);
      for (uint32_t rows = 1; rows <= 2048; ++rows)
        for (const uint32_t history : {0U, 32768U, 1'000'000U}) {
          const std::array sequence{sequenceOf(history, rows)};
          covers(chunk, plans.attention(sequence, queryHeads, kvLayout, chunk).workspace(), 1, attentionFields);
        }
      const auto stride = plans.laneAttentionWorkspace(queryHeads, kvLayout);
      for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
        const std::array<uint32_t, 4> histories{0, 31, 2048, 8192};
        const auto chunks = lanesAfter(std::span(histories).first(lanes));
        const auto selected = plans.attention(chunks, queryHeads, kvLayout, workspaceOf(stride, lanes));
        require(std::ranges::all_of(selected.sequences(), [](const auto &lane) { return lane.splits == 32; }) &&
                    selected.dispatches().size() == 1,
                "a decode step's lanes no longer attend in one dispatch of 32 splits");
        covers(stride, selected.workspace(), lanes, attentionFields);
      }
      {
        const std::array<uint32_t, 1> deep{131072};
        const auto scaled = plans.attention(lanesAfter(deep), queryHeads, kvLayout, stride);
        require(scaled.sequences()[0].splits == kv::kAttentionMaximumSplits,
                "a decode lane's splits did not scale with history");
        covers(stride, scaled.workspace(), 1, attentionFields);
      }
    }
    for (auto shape : draftShapes) {
      const auto stride = plans.draftAttentionWorkspacePerLane(shape);
      for (uint32_t lanes = 1; lanes <= 4; ++lanes)
        covers(stride, plans.draftAttention(shape, lanes).workspace(), lanes, draftFields);
    }
    for (auto shape : moeShapes) {
      const auto stride = plans.moeDecodeWorkspacePerLane(shape, kRouterFormat);
      require(stride == plans.moe(shape, kRouterFormat, 8).workspace(), "workspace bound changed");
      const auto prefill = plans.moePrefillWorkspace(shape, kRouterFormat, 2048);
      for (uint32_t rows = 1; rows <= 2048; ++rows)
        covers(prefill, plans.moe(shape, kRouterFormat, rows).workspace(), 1, kMoeWorkspaceFields);
      // A decode step's plan is its rows': 8-row tiles while they route at
      // most once per expert, as the routed shape's 32 rows do and the tiny
      // shapes' 8 do not.
      for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
        const auto selected = plans.moe(shape, kRouterFormat, lanes * 8);
        require(selected.tileRows() == (lanes * 8 * shape.expertsPerToken > shape.experts ? 32U : 8U),
                "MoE decode baseline changed");
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
      return plans.linear().plan(af4, 2048, LinearEpilogue::Residual).configuration().tile;
    };
    require(tile(on) == (int8 ? LinearTile::GgufPrefillInt8 : LinearTile::GgufPrefill) &&
                tile(off) == LinearTile::GgufPrefill,
            "the int8 prefill tile does not follow the int8-activation setting");
    const auto queries = [&](const ExecutionPlans &plans, kv::ChunkedPrefillParams sequence) {
      const std::array sequences{sequence};
      const auto plan = plans.attention(sequences, 16, layout,
                                        plans.sequenceAttentionWorkspace(16, layout));
      return int8Queries(plan.splitPipeline());
    };
    require(queries(on, sequenceOf(32768, 64)) == int8 && !queries(off, sequenceOf(32768, 64)) &&
                queries(on, sequenceOf(history[0], 8)) == int8 && !queries(off, sequenceOf(history[0], 8)),
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
        const LinearConfig own = draft.plan(context, rows, LinearEpilogue::None).configuration(),
                           targets = target.plan(context, rows, LinearEpilogue::None).configuration();
        const bool int8 = family >= 10 && setting == Int8Activations::On && rows == 2048;
        require((targets.tile == LinearTile::GgufPrefillInt8) == int8 &&
                    own == (int8 ? LinearConfig{.tile = LinearTile::GgufPrefill} : targets),
                "a draft's prefill chunk took the int8 prefill tile or left the target's bf16 plan");
      }
      for (uint32_t lanes = 1; lanes <= 4; ++lanes)
        require(draft.plan(context, lanes * SPLASH_TARGET_VERIFY_ROWS).configuration() == target.plan(context, lanes * SPLASH_TARGET_VERIFY_ROWS).configuration(),
                "a draft's decode plan departed from the target's");
    }
}

// MoE plans run the three expert passes on staged tiles on every GPU: 8-row
// tiles while a step's routes average at most one row per expert, as a
// decode step's of up to four lanes do, 32-row tiles past that.
void ggufMoePlans() {
  const MoeShape shape = routedShape;
  // A GPU family and the cores its device reports (none: kAssumedGpuCores).
  for (const auto [family, reportedCores] : {std::pair{9U, 0U}, {9U, 11U}, {9U, 10U}, {10U, 0U}, {11U, 0U}}) {
    DeviceCapabilities capabilities = device(family);
    capabilities.gpuCoreCount = reportedCores;
    ExecutionPlans plans(capabilities, Int8Activations::On);
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      const MoePlan plan = plans.moe(shape, kRouterFormat, lanes * 8);
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
        const MoePlan plan = plans.moe(shape, routerFormat, rows);
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
      const MoeWorkspace workspace = plans.moe(shape, kRouterFormat, rows).workspace();
      for (const auto field : kMoeWorkspaceFields) devicePlans.*field = std::max(devicePlans.*field, workspace.*field);
    }
    require(plans.moePrefillWorkspace(shape, kRouterFormat, 2048) == devicePlans,
            "GGUF MoE prefill bound is not the bound of the device's plans");
  }
}

// A dense tile of a sequence of more than one tile takes a history split per
// the device's pages per split (KernelCalibration.hpp): one per 64 pages on
// Apple9 GPUs of 10 cores or fewer, one per 32 elsewhere. Qwen3.6-35B-A3B's
// attention: a 512-row chunk after 4K tokens (144 pages), at most 4 splits a
// tile of its 64, and a 64-row chunk after 32K (1026 pages), at most 32.
void prefillSplitsFollowTheDeviceRow() {
  const kv::Layout layout{1, 2, 256};
  struct Expected final {
    uint32_t rows, history, small, other;
  };
  for (const auto [rows, history, small, other] : {Expected{512, 4096, 2, 4}, Expected{64, 32768, 16, 32}})
    for (const auto [family, cores] : {std::pair{9U, 10U}, {9U, 1U}, {9U, 11U}, {9U, 40U}, {10U, 12U}, {10U, 32U}}) {
      DeviceCapabilities capabilities = device(family);
      capabilities.gpuCoreCount = cores;
      const ExecutionPlans plans(capabilities, Int8Activations::On);
      const std::array sequence{sequenceOf(history, rows)};
      const uint32_t splits =
          plans.attention(sequence, 16, layout, plans.sequenceAttentionWorkspace(16, layout))
              .sequences()[0]
              .splits;
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
          const LinearWorkload w{matrix, lanes * 8, epilogue};
          require(unknown.linear().plan(w).configuration() == planned.linear().plan(w).configuration(),
                  "an unknown core count planned a decode projection for other than the assumed cores");
        }
    for (uint32_t lanes = 1; lanes <= 4; ++lanes)
      require(unknown.moe(routedShape, kRouterFormat, lanes * 8).configuration() ==
                  planned.moe(routedShape, kRouterFormat, lanes * 8).configuration(),
              "an unknown core count planned a MoE decode step for other than the assumed cores");
  }
}

void workspaceBounds() {
  const ExecutionPlans plans(device(), Int8Activations::On);
  const std::array<uint32_t, 3> histories{31, 32, 32769};
  const auto verify = plans.laneAttentionWorkspace(24, attentionShapes[0].layout);
  const auto exact = plans.attention(lanesAfter(histories), 24, attentionShapes[0].layout, workspaceOf(verify, 3));
  require(exact.sequences().size() == 3 && exact.sequences()[1].splits == 32 && exact.sequences()[2].splits == 65,
          "a decode step's plan did not resolve one history per lane");
  require(verify.partialsBytes ==
                  uint64_t{8} * kv::kAttentionMaximumSplits * 24 * 256 * 4 &&
              verify.statisticsBytes ==
                  uint64_t{8} * kv::kAttentionMaximumSplits * 24 * 2 * 4,
          "a decode lane's workspace does not cover the maximum split count");
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
  const auto workspace = workspaceOf(plans.laneAttentionWorkspace(24, kvLayout), 4);
  const std::array<uint32_t, 5> histories{0, 1, 2, 3, 4};
  for (const size_t lanes : {size_t{0}, histories.size()})
    rejects([&] { (void)plans.attention(lanesAfter(std::span(histories).first(lanes)), 24, kvLayout, workspace); },
            "invalid attention sequences", "an attention of no sequences or past a batch was accepted");
  rejects([&] { (void)plans.attention(lanesAfter(std::span(histories).first(1)), 24, {}, workspace); },
          "no KV store kernel for layout", "an attention plan of an empty KV layout was accepted");
  rejects([&] { (void)plans.attention(lanesAfter(std::span(histories).first(1)), 24, kvLayout, {}); },
          "the attention workspace holds no sequence's splits", "an attention plan without a workspace was accepted");
  rejects([&] { (void)PagedAttention::workspace(0, 24, kvLayout); }, "invalid attention workspace slots",
          "a workspace of no tile splits was accepted");
  for (const uint32_t rows : {UINT32_MAX, 0U})
    rejects([&] { (void)plans.moe(routedShape, kRouterFormat, rows); }, "invalid MoE rows",
            "a MoE step of no rows or past the budget was accepted");
  rejects([&] { (void)plans.moePrefillWorkspace(routedShape, kRouterFormat, 0); }, "invalid MoE rows",
          "a MoE prefill workspace of no rows was accepted");
  rejects([&] { (void)plans.gateUpWorkspace({256, 32}); }, "invalid linear matrix",
          "a gate/up workspace of a partial 64-input span was accepted");
  rejects([&] { (void)plans.draftAttentionWorkspacePerLane({}); },
          "unsupported compiled draft attention shape", "a draft workspace of an empty shape was accepted");
  std::array<uint32_t, 1> edge{kv::kMaximumPhysicalTokens - 8};
  const auto finalVerify = plans.attention(lanesAfter(edge), 24, kvLayout, workspace);
  require(finalVerify.sequences()[0].splits == kv::kAttentionMaximumSplits,
          "valid final physical decode rows were rejected");
  ++edge[0];
  rejects([&] { (void)lanesAfter(edge); }, "context_out_of_range",
          "decode rows past the physical context were accepted");
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

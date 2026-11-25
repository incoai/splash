#include "TestChecks.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/LaneRows.hpp"

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
// A decode step's most rows: the most lanes of a verify block's most.
constexpr uint32_t kStepRows = SPLASH_MAXIMUM_BATCH_WIDTH * SPLASH_TARGET_VERIFY_ROWS;
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
  for (uint32_t family : {8U, 9U, 10U, 11U}) {
    const ExecutionPlans plans(device(family), Int8Activations::On);
    const Linear baseline(device(family), Int8Activations::On, StepKind::Prefill);
    for (auto matrix : matrices) {
      uint64_t gateBound = 0;
      for (uint32_t rows = 8; rows <= kMaximumDecodeTileRows; rows += 8) {
        for (auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                              LinearEpilogue::GateUp}) {
          const LinearWorkload w{matrix, rows, epilogue};
          require(plans.linear().plan(w).configuration() ==
                      baseline.plan(w).configuration(),
                  "the plans' decode Linear departed from the device policy");
          if (epilogue == LinearEpilogue::GateUp)
            gateBound = std::max(gateBound, baseline.plan(w).gateScratchBytes());
        }
      }
      require(plans.gateUpWorkspace(gateUp(matrix), kMaximumDecodeTileRows) == gateBound &&
                  gateBound == uint64_t{kMaximumDecodeTileRows} * matrix.outputSize * 2,
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
      // A bound holds the plan of every step up to its rows: a decode step's
      // and a prompt chunk's.
      for (const uint32_t most : {kStepRows, 2048U}) {
        const auto bound = plans.moeWorkspace(shape, kRouterFormat, most, StepKind::Prefill);
        for (uint32_t rows = 1; rows <= most; ++rows)
          covers(bound, plans.moe(shape, kRouterFormat, rows, StepKind::Prefill).workspace(), 1, kMoeWorkspaceFields);
      }
      // A decode step's plan is its rows': 8-row tiles while they route at
      // most once per expert, as the routed shape's 32 rows do and the tiny
      // shapes' 8 do not.
      for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
        const auto selected = plans.moe(shape, kRouterFormat, lanes * 8, StepKind::Decode);
        require(selected.tileRows() == (lanes * 8 * shape.expertsPerToken > shape.experts ? 32U : 8U),
                "MoE decode baseline changed");
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
  for (uint32_t family : {8U, 9U, 10U, 11U}) {
    const ExecutionPlans on(device(family), Int8Activations::On), off(device(family), Int8Activations::Off);
    const ProjectionCalibration &row = on.calibration().projection, &offRow = off.calibration().projection;
    require(on.linear().int8Activations() == Int8Activations::On &&
                off.linear().int8Activations() == Int8Activations::Off && row.prefillInt8 == (family >= 10) &&
                offRow.prefillInt8 == row.prefillInt8 && offRow.decodeInt8FromRows == row.decodeInt8FromRows &&
                off.calibration().attention.int8QueriesFromPages == on.calibration().attention.int8QueriesFromPages,
            "the int8-activation setting did not reach the plans' Linear, or moved the device's row");
    const bool int8 = family >= 10;
    // Apple8 and Apple10/11 score INT8 attention queries from 8 pages a
    // split; Apple9 scores BF16 ones at every history. The projections' INT8
    // prefill tile still stops at Apple10.
    const bool int8QueriesRow = family == 8 || int8;
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
    require(queries(on, sequenceOf(32768, 64)) == int8QueriesRow &&
                !queries(off, sequenceOf(32768, 64)) &&
                queries(on, sequenceOf(history[0], 8)) == int8QueriesRow &&
                !queries(off, sequenceOf(history[0], 8)),
            "INT8 attention queries do not follow the int8-activation setting");
    if (family == 9 || family == 10) {
      // Apple9 scores no INT8 queries at any history; Apple10 takes them
      // from eight pages a split, which a decode lane after 16,384 tokens
      // reads.
      const std::array sequence{sequenceOf(16384, 8)};
      const auto shortQueries = [&](const ExecutionPlans &plans) {
        return int8Queries(plans.attention(sequence, 16, layout, plans.laneAttentionWorkspace(16, layout))
                               .splitPipeline());
      };
      require(shortQueries(on) == (family == 10) && !shortQueries(off),
              "an Apple9 plan did not keep its BF16 queries");
    }
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
  for (uint32_t family : {8U, 9U, 10U, 11U})
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
        require(draft.plan(context, lanes * SPLASH_DRAFT_QUERY_ROWS).configuration() == target.plan(context, lanes * SPLASH_DRAFT_QUERY_ROWS).configuration(),
                "a draft's decode plan departed from the target's");
    }
}

// A decode step's plans take no int8 prefill path whatever its rows
// (StepKind): at 65, 96 and 128 rows, five to eight lanes, a 27B-shaped
// af4g64 projection runs the staged tile's row groups in bf16 activations in
// each epilogue where a prefill chunk of those rows runs the int8 prefill tile
// on Apple10 and Apple11, and a
// block of top 8 of 128 experts, six routes an expert from 96 rows, the
// staged passes where a chunk runs the int8 passes. A rotated PQ2_0 unit's
// decode plans take the int8 decode tiles as a chunk's of their rows do.
void decodeStepsRoundNoActivations() {
  const Projection af4(5120, 17408,
                       BlockWeights{{QuantizedSegment::planes(GGUF_FMT_AF4G64, 5120, 17408, {}, {}, {})}});
  MoeShape shape{2048, 128, 8, 768};
  shape.int8Experts = true;
  const UnitFormats rotated{.registerTakes = false, .weightBits16 = 34, .decodeInt8Takes = true};
  for (uint32_t family : {10U, 11U}) {
    const ExecutionPlans plans(device(family), Int8Activations::On);
    for (const uint32_t rows : {65U, 96U, 128U}) {
      for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate})
        require(plans.linear().plan(af4, rows, epilogue).configuration().tile == LinearTile::GgufPrefillInt8 &&
                    plans.decodeLinear().plan(af4, rows, epilogue).configuration().tile == LinearTile::GgufStaged,
                "a decode step's projection took the int8 prefill tile, or a chunk's did not");
      require(plans.moe(shape, kRouterFormat, rows, StepKind::Prefill).configuration().int8Passes == (rows >= 96) &&
                  !plans.moe(shape, kRouterFormat, rows, StepKind::Decode).configuration().int8Passes,
              "a decode step's MoE plan took the int8 passes, or a chunk's did not");
    }
    bool decodeInt8 = false;
    for (const uint32_t rows : {16U, 32U}) {
      const LinearWorkload w{{6144, 5120}, rows};
      const LinearConfig step =
          selectProjection(plans.calibration(), Int8Activations::On, StepKind::Decode, w, rotated);
      require(step == selectProjection(plans.calibration(), Int8Activations::On, StepKind::Prefill, w, rotated),
              "a rotated PQ2_0 unit's decode plan departed from a chunk's of its rows");
      decodeInt8 |= step.decodeInt8;
    }
    require(decodeInt8, "a rotated PQ2_0 unit's decode plans took no int8 decode tile");
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
      const MoePlan plan = plans.moe(shape, kRouterFormat, lanes * 8, StepKind::Decode);
      require(plan.tileRows() == 8 && plan.configuration().ggufRouterTile == FloatTile::Simdgroup,
              "GGUF MoE decode plan left its 8-row tiles");
      covers(plans.moeWorkspace(shape, kRouterFormat, kStepRows, StepKind::Decode), plan.workspace(), 1, kMoeWorkspaceFields);
    }
    // Prefill: 8-row tiles while the routes average at most one row per
    // expert (32 rows of 8 of 256 experts). The router's float tile follows
    // selectFloatTile and the router's format (32 assumed cores: the neural
    // accelerator from 321 rows of BF16 values, 449 of F32 values and 705 of
    // a quantized format's rows, never on Apple9).
    for (const auto [routerFormat, first] :
         {std::pair{uint32_t{GGUF_FMT_BF16}, 321U}, std::pair{QuantizedSegment::kFloat32, 449U},
          std::pair{uint32_t{GGUF_FMT_AF8G64}, 705U}}) {
      const MoeWorkspace bound = plans.moeWorkspace(shape, routerFormat, 2048, StepKind::Prefill);
      for (uint32_t rows : {1U, 8U, 17U, 32U, 33U, 100U, 256U, 257U, 320U, 321U, 448U, 449U, 704U, 705U, 2048U}) {
        const MoePlan plan = plans.moe(shape, routerFormat, rows, StepKind::Prefill);
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
      const MoeWorkspace workspace = plans.moe(shape, kRouterFormat, rows, StepKind::Prefill).workspace();
      for (const auto field : kMoeWorkspaceFields) devicePlans.*field = std::max(devicePlans.*field, workspace.*field);
    }
    require(plans.moeWorkspace(shape, kRouterFormat, 2048, StepKind::Prefill) == devicePlans,
            "GGUF MoE prefill bound is not the bound of the device's plans");
  }
  // The M5 Pro (Apple10, 16 to 31 cores) runs a prefill chunk whose routes
  // average MoeCalibration::wideTileRoutes or more an expert on 64-row tiles:
  // from 20 cores 64 (2048 rows of 8 of 256 experts, 1024 of 8 of 128), on the
  // 16-core 128 (2048 rows of 8 of 128 alone). Its decode steps and shorter
  // chunks run as every GPU's do; the 10-core M5, the M5 Max, the M6 and
  // Apple9 keep 32-row tiles. The prefill bound holds the 64-row plans.
  MoeShape eightOf128 = shape;
  eightOf128.experts = 128;
  // Flint's routing (a sigmoid router's top 6 of 256 experts and two ungated
  // shared slices), whose chunks average at most 48 routes an expert, takes
  // 64-row tiles on no device, and a plan of it on them throws: its passes
  // have no 64-row kernels.
  MoeShape flint = shape;
  flint.expertsPerToken = 6;
  flint.router = MoeRouter::SigmoidNormalized;
  flint.sharedSlices = 2;
  flint.sharedGate = false;
  for (const auto [family, cores] : {std::pair{10U, 16U}, {10U, 19U}, {10U, 20U}, {10U, 31U}, {10U, 10U}, {10U, 32U},
                                     {10U, 40U}, {11U, 20U}, {9U, 20U}}) {
    DeviceCapabilities capabilities = device(family);
    capabilities.gpuCoreCount = cores;
    const ExecutionPlans plans(capabilities, Int8Activations::On);
    const bool pro = family == 10 && cores >= 16 && cores < 32, sixtyFour = pro && cores >= 20;
    for (const uint32_t rows : {33U, 1024U, 2047U, 2048U})
      for (const auto &[block, wide] : {std::pair{shape, sixtyFour && rows == 2048},
                                        {eightOf128, (sixtyFour && rows >= 1024) || (pro && rows == 2048)}})
        require(plans.moe(block, kRouterFormat, rows, StepKind::Prefill).tileRows() == (wide ? 64U : 32U),
                "GGUF MoE prefill plan of " + std::to_string(block.experts) + " experts on Apple" +
                    std::to_string(family) + ", " + std::to_string(cores) + " cores, at " + std::to_string(rows) +
                    " rows left its tiles");
    for (uint32_t rows = 1; rows <= SPLASH_PREFILL_TOKEN_BUDGET; ++rows)
      require(plans.moe(flint, kRouterFormat, rows, StepKind::Prefill).tileRows() <= 32,
              "Flint's MoE prefill plan on Apple" + std::to_string(family) + ", " + std::to_string(cores) +
                  " cores, at " + std::to_string(rows) + " rows runs 64-row tiles");
    for (const MoeShape &block : {shape, eightOf128}) {
      for (uint32_t lanes = 1; lanes <= 4; ++lanes)
        require(plans.moe(block, kRouterFormat, lanes * 16, StepKind::Decode).tileRows() <= 32,
                "a GGUF MoE decode plan runs 64-row tiles");
      const MoePlan widest = plans.moe(block, kRouterFormat, 2048, StepKind::Prefill);
      covers(plans.moeWorkspace(block, kRouterFormat, 2048, StepKind::Prefill), widest.workspace(), 1,
             kMoeWorkspaceFields);
    }
  }
  rejects([&] { (void)MoE::plan(flint, 2048, MoeConfig{MoeExpertTile::M64}); },
          "no 64-row passes for two shared slices", "a MoE plan of two shared slices on 64-row tiles was accepted");
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

// Apple8 (M2) multiplies on the FP32 pipe as Apple9 does, so its linear plans
// are the Apple9 row's at every core count, small or large. Its attention is
// its own row (kApple8Attention): the same page partition as Apple9's, but
// INT8 query codes from eight pages a split where Apple9 scores BF16 ones.
void apple8TakesTheApple9Row() {
  const kv::Layout layout{1, 2, 256};
  const std::array prefill{sequenceOf(32768, 64)};
  const std::array<uint32_t, 1> decodeHistory{131072};
  const auto decode = lanesAfter(decodeHistory);
  for (const uint32_t cores : {10U, 32U, 38U}) {
    DeviceCapabilities m2 = device(8), m3 = device(9);
    m2.gpuCoreCount = m3.gpuCoreCount = cores;
    require(m2.gpuClass() == GpuClass::Apple8 && m3.gpuClass() == GpuClass::Apple9,
            "the GPU family 8 or 9 did not map to its class");
    const ExecutionPlans apple8(m2, Int8Activations::On), apple9(m3, Int8Activations::On);
    for (auto matrix : matrices) {
      for (uint32_t rows : {64U, 256U, 2048U})
        for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate})
          require(apple8.linear().plan({matrix, rows, epilogue}).configuration() ==
                      apple9.linear().plan({matrix, rows, epilogue}).configuration(),
                  "an Apple8 prefill plan took another row than Apple9's");
      for (const uint32_t rows : {1U, 8U, 16U, 32U, 64U})
        for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp})
          require(apple8.linear().plan({matrix, rows, epilogue}).configuration() ==
                      apple9.linear().plan({matrix, rows, epilogue}).configuration(),
                  "an Apple8 decode plan took another row than Apple9's");
    }
    if (cores > kApple9SmallGpuCores) {
      const auto apple8Prefill =
          apple8.attention(prefill, 16, layout, apple8.sequenceAttentionWorkspace(16, layout));
      const auto apple9Prefill =
          apple9.attention(prefill, 16, layout, apple9.sequenceAttentionWorkspace(16, layout));
      const auto apple8Decode = apple8.attention(decode, 16, layout, apple8.laneAttentionWorkspace(16, layout));
      const auto apple9Decode = apple9.attention(decode, 16, layout, apple9.laneAttentionWorkspace(16, layout));
      require(apple8Prefill.sequences()[0].splits == apple9Prefill.sequences()[0].splits &&
                  apple8Decode.sequences()[0].splits == apple9Decode.sequences()[0].splits,
              "an Apple8 attention plan took another split count than Apple9's");
      const auto scoresInt8 = [](const AttentionPlan &plan) {
        return plan.splitPipeline().find("q8q") != std::string::npos;
      };
      require(scoresInt8(apple8Prefill) && scoresInt8(apple8Decode) && !scoresInt8(apple9Prefill) &&
                  !scoresInt8(apple9Decode),
              "an Apple8 plan did not score the INT8 queries Apple9's scored BF16");
    }
  }
  // Apple8 keeps the 32-page split at every core count; Apple9's small row
  // (10 cores or fewer) takes 64, so the same prompt splits in two.
  DeviceCapabilities small{device(8)}, smallM3{device(9)};
  small.gpuCoreCount = smallM3.gpuCoreCount = kApple9SmallGpuCores;
  const ExecutionPlans smallApple8(small, Int8Activations::On), smallApple9(smallM3, Int8Activations::On);
  const auto small8 = smallApple8.attention(prefill, 16, layout, smallApple8.sequenceAttentionWorkspace(16, layout));
  const auto small9 = smallApple9.attention(prefill, 16, layout, smallApple9.sequenceAttentionWorkspace(16, layout));
  require(small8.sequences()[0].splits > small9.sequences()[0].splits,
          "an Apple8 prefill plan on a small Apple9 row did not keep its 32-page splits");
  require(small8.splitPipeline().find("q8q") != std::string::npos &&
              small9.splitPipeline().find("q8q") == std::string::npos,
          "a small Apple9 row did not keep the BF16 queries the M4 measured");
}

// A device that reports no core count gets the plans of kAssumedGpuCores
// cores, Linear and MoE alike.
void unknownCoreCount() {
  for (uint32_t family : {8U, 9U, 10U}) {
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
      require(unknown.moe(routedShape, kRouterFormat, lanes * 8, StepKind::Decode).configuration() ==
                  planned.moe(routedShape, kRouterFormat, lanes * 8, StepKind::Decode).configuration(),
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
  // A four-lane decode step's, of up to 64 rows: from 33 rows its plans take 32-row
  // tiles, 274 of them at 64 rows (16 full tiles' routes, a partial tile for
  // each of 256 experts, and two of the shared expert's), whose grouped rows
  // are the 576 routes' and, where the passes' matmuls read, the 31 rows a
  // last tile's may read past its live rows.
  const auto moe = plans.moeWorkspace(routedShape, kRouterFormat, 4 * SPLASH_TARGET_VERIFY_ROWS, StepKind::Decode);
  require(moe.groupedInputBytes == 607 * 2048 * 2 && moe.expertOutputBytes == 576 * 2048 * 2 &&
              moe.expertIntermediateBytes == 607 * 512 * 2 && moe.groupedRoutesBytes == 576 * 4 &&
              moe.tileDescriptorsBytes == 274 * 12 && moe.tileCountBytes == 80 && moe.expertSlotsBytes == 1024,
          "MoE decode workspace changed");
  // The gate pass of the decode tiles' most rows, 64 a gate/up pair runs in two passes.
  require(plans.gateUpWorkspace(gateUp(matrices[0]), kMaximumDecodeTileRows) == 2228224 &&
              plans.gateUpWorkspace(gateUp(matrices[1]), kMaximumDecodeTileRows) == 786432,
          "gate/up workspace omitted the gate pass of the decode tiles' 64 rows");
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
  const auto workspace = workspaceOf(plans.laneAttentionWorkspace(24, kvLayout), SPLASH_MAXIMUM_BATCH_WIDTH);
  std::array<uint32_t, SPLASH_MAXIMUM_BATCH_WIDTH + 1> histories{};
  for (uint32_t lane = 0; lane < histories.size(); ++lane) histories[lane] = lane;
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
    rejects([&] { (void)plans.moe(routedShape, kRouterFormat, rows, StepKind::Prefill); }, "invalid MoE rows",
            "a MoE step of no rows or past the budget was accepted");
  rejects([&] { (void)plans.moeWorkspace(routedShape, kRouterFormat, 0, StepKind::Prefill); }, "invalid MoE rows",
          "a MoE workspace of no rows was accepted");
  rejects([&] { (void)plans.gateUpWorkspace({256, 32}, kMaximumDecodeTileRows); }, "invalid linear matrix",
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

// A draft's gate and up, af4g64 planes (DraftCheckpoint), run as one pair
// over its block's rows, 8 a lane, on the tile their format takes, which the
// shape gateUpWorkspace sizes by does not name. Where the pair runs two
// passes, below ProjectionCalibration::onePassFromCores or by the wave law,
// its gate pass writes the rows its tile stores: 16 at one lane's 8 on the
// 16-row tile, which Inkling-2-Mini's draft takes on an M5 of 10 GPU cores
// (a start at one lane threw "projection gate scratch buffer holds 114688
// bytes, needs 221184") and Ornith's on an M5 Pro of 16.
void draftGateUpScratch() {
  // Inkling-2-Mini's, Qwen3.6-35B-A3B's and Flint's, Qwen3.8-27B's,
  // Qwen3.8-Flash-Next's and Ornith's.
  constexpr std::array pairs{LinearMatrix{6912, 2304}, LinearMatrix{6144, 2048}, LinearMatrix{17408, 5120},
                             LinearMatrix{7680, 2560}, LinearMatrix{9216, 3072}};
  const auto planes = [](LinearMatrix matrix) {
    return Projection(matrix.outputSize, matrix.inputSize,
                      BlockWeights{{QuantizedSegment::planes(GGUF_FMT_AF4G64, matrix.outputSize, matrix.inputSize, {},
                                                             {}, {})}});
  };
  bool tallerTile = false;
  for (uint32_t family : {9U, 10U, 11U})
    for (uint32_t cores : {8U, 10U, 12U, 14U, 16U, 18U, 20U, 30U, 32U, 40U, 60U, 80U}) {
      DeviceCapabilities value = device(family);
      value.gpuCoreCount = cores;
      const ExecutionPlans plans(value, Int8Activations::On);
      for (const auto matrix : pairs) {
        const Projection up = planes(matrix);
        for (uint32_t lanes = 1; lanes <= SPLASH_MAXIMUM_BATCH_WIDTH; ++lanes) {
          const uint64_t bound = plans.gateUpWorkspace(gateUp(matrix), lanes * SPLASH_DRAFT_QUERY_ROWS);
          // No more than the staged tile of the lanes' rows, 16 at the fewest.
          require(bound == uint64_t{stagedStorageRows(lanes * SPLASH_DRAFT_QUERY_ROWS, true)} * matrix.outputSize * 2,
                  "the gate/up workspace is not the staged tile's rows of a draft's lanes");
          for (uint32_t step = 1; step <= lanes; ++step) {
            const uint32_t rows = step * SPLASH_DRAFT_QUERY_ROWS;
            const LinearPlan pair = plans.draftLinear().plan(up, rows, LinearEpilogue::GateUp, &up);
            require(bound >= pair.gateScratchBytes(),
                    "the gate/up workspace is below a draft's pair on its formats' tile");
            tallerTile |= pair.gateScratchBytes() && pair.storageRows() > rows;
          }
        }
      }
    }
  require(tallerTile, "no device ran a draft's two-pass pair on a tile of more rows than its own");
  DeviceCapabilities m5 = device(10);
  m5.gpuCoreCount = 10;
  const ExecutionPlans plans(m5, Int8Activations::On);
  const Projection up = planes(pairs[0]);
  const LinearPlan pair = plans.draftLinear().plan(up, SPLASH_DRAFT_QUERY_ROWS, LinearEpilogue::GateUp, &up);
  require(pair.configuration().sixteenRows && !pair.configuration().oneGateUpPass &&
              pair.gateScratchBytes() == 221184 &&
              plans.gateUpWorkspace(gateUp(pairs[0]), SPLASH_DRAFT_QUERY_ROWS) == 221184,
          "Inkling-2-Mini's draft at one lane on an M5 of 10 cores left its gate scratch's 16 rows");
}
} // namespace

int main() {
  try {
    baselinePlans();
    apple8TakesTheApple9Row();
    int8ActivationsSetting();
    draftPlans();
    ggufMoePlans();
    decodeStepsRoundNoActivations();
    unknownCoreCount();
    prefillSplitsFollowTheDeviceRow();
    workspaceBounds();
    draftGateUpScratch();
    invalidLookupsAndContextEdges();
    std::cout << "PASS execution plans: device policies, int8 activations on and off, the draft's bf16 prefill, "
                 "device MoE tiles, B1-B4 and prefill workspace bounds (CPU only)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL execution plans: " << error.what() << '\n';
    return 1;
  }
}

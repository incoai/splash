#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "ops/GDN.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"
#include "ops/PagedAttention.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "metal/abi/QuantFormat.h"

#include <array>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using namespace splash;
using namespace splash::ops;

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

bool sameScratch(LinearScratchSize a, LinearScratchSize b) {
  return a.input == b.input && a.sums == b.sums && a.partials == b.partials && a.counters == b.counters;
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
// destination keeps the configuration, input table and scratch of its bf16
// plan on every family and core count, and only plain decode workloads take
// one. ggufProjectionMatrix checks the fp32 kernel it encodes.
void floatOutputPlans() {
  const auto fp32 = [](Projection p) {
    p.destination = FloatOutput::Float32;
    return p;
  };
  for (const uint32_t family : {9U, 10U, 11U})
    for (uint32_t cores = 0; cores <= 128; ++cores) {
      const Linear linear = gpu(family, cores);
      for (const uint32_t n : {256U, 5120U, 16640U, 248320U})
        for (const uint32_t k : {2048U, 5120U}) {
          const Projection p = blockProjection(n, k, 1);
          for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
            const LinearWorkload w{{n, k}, lanes * 8, LinearPhase::Decode, LinearEpilogue::None};
            const LinearPlan bf16 = linear.plan(w, p), head = linear.plan(w, fp32(p));
            if (head.configuration() != bf16.configuration() || head.destination() != FloatOutput::Float32 ||
                head.input() != bf16.input() || head.storageRows() != bf16.storageRows() ||
                !sameScratch(head.scratchSize(), bf16.scratchSize()))
              broke("an fp32 plan differs from its bf16 plan", family, cores, w);
          }
        }
    }
  std::vector<LinearWorkload> others{{{5120, 2048}, 8, LinearPhase::Decode, LinearEpilogue::Residual},
                                     {{5120, 2048}, 8, LinearPhase::Decode, LinearEpilogue::GateUp}};
  for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate})
    for (const uint32_t rows : {8U, 2048U}) others.push_back({{5120, 2048}, rows, LinearPhase::Prefill, epilogue});
  const Linear linear = gpu(10, 20);
  const Projection p = blockProjection(5120, 2048, 1);
  for (const LinearWorkload &w : others) {
    (void)linear.plan(w, p);
    rejects([&] { (void)linear.plan(w, fp32(p)); }, "an fp32 destination takes a plain decode projection",
            "an fp32 plan of other than a plain decode was accepted");
  }
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
// projection kind; the narrow grids at 32 rows, where Apple10 stages them
// (its register tile takes the steps whose grid of column tiles and lanes
// holds fewer than 1.5 threadgroups per core).
constexpr std::array kStagedSplitAnchors{
    SplitAnchor{"27B down", 16, 5120, 17408, 8, kResidual, 1, 2},
    SplitAnchor{"27B down", 20, 5120, 17408, 8, kResidual, 1, 2},
    SplitAnchor{"27B down", 10, 5120, 17408, 8, kResidual, 1, 1},
    SplitAnchor{"27B down", 40, 5120, 17408, 8, kResidual, 1, 4},
    SplitAnchor{"35B output projection", 16, 2048, 4096, 8, kResidual, 1, 4},
    SplitAnchor{"35B shared-expert gate/up", 16, 512, 2048, 32, kGateUp, 1, 4},
    SplitAnchor{"27B gate/up", 16, 17408, 5120, 8, kGateUp, 1, 1},
    SplitAnchor{"two-segment fused projection", 16, 4096, 5120, 8, kNone, 2, 2},
    SplitAnchor{"27B three-segment GDN input", 40, 16640, 5120, 8, kNone, 3, 1},
    SplitAnchor{"27B vocabulary head", 16, 248320, 5120, 8, kNone, 1, 1},
    SplitAnchor{"narrow 1024 x 256 tensor", 16, 1024, 256, 32, kNone, 1, 1},
    SplitAnchor{"narrow 1024 x 3072 tensor", 16, 1024, 3072, 32, kNone, 1, 4}};

// One wave (four threadgroups per core) down to one 256-input unit per
// partition, eight waves while partitions keep 1024 inputs, at most eight;
// sixteen where eight leave fewer than three threadgroups per core and
// partitions keep 512 inputs, which 512 x 5120's sixteen would not.
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
    SplitAnchor{"narrow 512 x 5120 tensor", 40, 512, 5120, 8, kNone, 1, 8},
    SplitAnchor{"narrow 1024 x 1024 tensor", 40, 1024, 1024, 8, kNone, 1, 4},
    SplitAnchor{"Flash-Next hyper-connection down", 12, 320, 10240, 8, kNone, 1, 8},
    SplitAnchor{"Flash-Next hyper-connection down", 14, 320, 10240, 8, kNone, 1, 16},
    SplitAnchor{"Flash-Next hyper-connection down, four lanes", 40, 320, 10240, 32, kNone, 1, 16}};

// fp32 K-split partials over the plan's rows and columns and one completion
// counter per 64-column tile (LinearPlan::scratchSize).
uint64_t splitPartialsBytes(const LinearPlan &plan) {
  return uint64_t{plan.configuration().splits} * plan.storageRows() * plan.workload().matrix.outputSize *
         sizeof(float);
}
uint64_t splitCountersBytes(const LinearPlan &plan) {
  return uint64_t{plan.workload().matrix.outputSize / GGUF_TILE_COLUMNS} * sizeof(uint32_t);
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
  require(single.configuration() == LinearConfig{.tile = LinearTile::GgufStaged, .splits = 2, .spread = true} &&
              single.input() == LinearInput::Plain &&
              single.scratchSize().partials == splitPartialsBytes(single) &&
              single.scratchSize().counters == splitCountersBytes(single) && single.scratchSize().input == 0,
          "GGUF single-tensor decode plan");
  for (const SplitAnchor &anchor : kStagedSplitAnchors)
    requireAnchorSplits(anchorPlan(10, anchor), anchor, "GGUF staged split policy");
  const LinearPlan gateUpPlan = linear.plan({{512, 2048}, 32, LinearPhase::Decode, LinearEpilogue::GateUp},
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
    require(prefill.storageRows() == storage && prefill.gateScratchBytes() == gateBytes(prefill) &&
                prefill.configuration().tile == (rows <= 32 ? LinearTile::GgufStaged : LinearTile::GgufPrefill) &&
                prefill.configuration().splits == splits &&
                prefill.scratchSize().partials == (splits > 1 ? splitPartialsBytes(prefill) : 0),
            "GGUF prefill tile rows and splits");
  }
  // The decode tiles hold at most a decode batch.
  const LinearWorkload longPrefill{{5120, 17408}, 33, LinearPhase::Prefill, LinearEpilogue::None};
  rejects([&] { (void)Linear::plan(longPrefill, {.tile = LinearTile::GgufStaged}, FloatOutput::BFloat16); },
          "the staged block tile runs prefill chunks of up to 32 rows",
          "the staged tile took a prefill chunk past a decode batch");
  rejects([&] { (void)Linear::plan(down, {.tile = LinearTile::GgufStaged, .splits = 3}, FloatOutput::BFloat16); },
          "staged block splits take whole 32-input groups", "the staged tile split K three ways");
  // The arena bound is the single-tensor plan of the 32-row tile, which fused
  // and gate/up plans share.
  const ProjectionShape downShape{5120, 17408};
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
    require(plan.configuration().tile == LinearTile::GgufRegister && plan.input() == LinearInput::Table16,
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
                                  GGUF_FMT_Q40, GGUF_FMT_Q50, GGUF_FMT_Q51, GGUF_FMT_MXFP4, GGUF_FMT_BF16})
      require(plan({format}).configuration().tile == LinearTile::GgufRegister, "Apple9 keeps other formats' registers");
    // F16, which the register tile does not take (quant_register_format), stages at every lane count, alone or
    // beside a register format.
    require(plan({GGUF_FMT_F16}).configuration().tile == LinearTile::GgufStaged &&
                plan({GGUF_FMT_Q4K, GGUF_FMT_F16}).configuration().tile == LinearTile::GgufStaged,
            "Apple9 stages a projection with an F16 segment");
    // Its decode storage holds the staged tile's rows: three lanes' 24 rows pad to 32.
    const ProjectionShape f16Shape = blockProjection(1024, 5120, std::array<uint32_t, 1>{GGUF_FMT_F16}).shape();
    require(!f16Shape.registerFormats &&
                m3.decodeStorageRows(lanes * 8, f16Shape) == plan({GGUF_FMT_F16}).storageRows() &&
                m3.decodeStorageRows(lanes * 8, ProjectionShape{1024, 5120}) == lanes * 8,
            "Apple9 decode storage holds an F16 projection's staged rows");
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
  // Apple9 registers projections of whole 64-input spans in a format of one group per meta unit, of fewer inputs
  // than 256 or no multiple of it (Qwen3.8-Flash-Next's Q8_0 hyper-connection projections take 320), and its decode
  // scratch bound covers them. A format of several groups per meta unit (Q2_0) stages such a projection, and its bound
  // holds no table.
  for (const uint32_t k : {64U, 192U, 320U, 640U}) {
    const Projection q80 = blockProjection(1024, k, 1, GGUF_FMT_Q80), q20 = blockProjection(1024, k, 1, GGUF_FMT_Q20);
    const LinearScratchSize narrow = m3.decodeScratchSize(q80.shape());
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      const LinearWorkload w{{1024, k}, lanes * 8, LinearPhase::Decode, kNone};
      const LinearPlan plan = m3.plan(w, q80);
      const LinearScratchSize need = plan.scratchSize();
      require(plan.configuration().tile == LinearTile::GgufRegister && narrow.input >= need.input &&
                  narrow.sums >= need.sums && narrow.partials >= need.partials && narrow.counters >= need.counters,
              "Apple9 registers Q8_0 projections of " + std::to_string(k) + " inputs within the scratch bound");
      require(m3.plan(w, q20).configuration().tile == LinearTile::GgufStaged &&
                  m3.decodeScratchSize(q20.shape()).input == 0,
              "Apple9 registers a Q2_0 projection of " + std::to_string(k) + " inputs");
    }
  }
  require(linear.plan({{5120, 17408}, 24, LinearPhase::Decode, LinearEpilogue::Residual},
                      blockProjection(5120, 17408, 1, GGUF_FMT_IQ2XXS)).configuration().tile == LinearTile::GgufStaged,
          "Apple10 stages every format");
  // Apple10 registers decode steps on grids of fewer than 1.5 threadgroups
  // per core (64 columns and lane each), half that in formats of fewer than
  // four bits, a one-pass gate/up pair on both tensors' columns. Its decode
  // storage follows the shape's formats, and the decode scratch bound covers
  // every plan.
  {
    struct Anchor {
      const char *projection;
      uint32_t cores, n, k, rows, format;
      LinearTile tile;
    };
    constexpr Anchor kAnchors[] = {
        {"Q4_K 2560 x 6144", 32, 2560, 6144, 8, GGUF_FMT_Q4K, LinearTile::GgufRegister},
        {"Q4_K 2560 x 6144, 20 cores", 20, 2560, 6144, 8, GGUF_FMT_Q4K, LinearTile::GgufStaged},
        {"IQ2_XXS 2560 x 6144", 32, 2560, 6144, 8, GGUF_FMT_IQ2XXS, LinearTile::GgufStaged},
        {"IQ2_XXS 1024 x 5120", 32, 1024, 5120, 8, GGUF_FMT_IQ2XXS, LinearTile::GgufRegister},
        {"IQ2_XXS 1024 x 5120, 16 rows", 32, 1024, 5120, 16, GGUF_FMT_IQ2XXS, LinearTile::GgufStaged},
        {"Q2_0 512 x 2560, 20 cores", 20, 512, 2560, 8, GGUF_FMT_Q20, LinearTile::GgufRegister},
        {"BF16 320 x 10240", 32, 320, 10240, 8, GGUF_FMT_BF16, LinearTile::GgufRegister},
    };
    for (const Anchor &a : kAnchors)
      require(gpu(10, a.cores).plan({{a.n, a.k}, a.rows, LinearPhase::Decode, kNone},
                                    blockProjection(a.n, a.k, 1, a.format)).configuration().tile == a.tile,
              std::string("Apple10 register policy: ") + a.projection);
    const auto pair = [](uint32_t cores, uint32_t n, uint32_t k) {
      const Projection up = blockProjection(n, k, 1), gate = blockProjection(n, k, 1);
      return gpu(10, cores).plan({{n, k}, 8, LinearPhase::Decode, kGateUp}, up, &gate).configuration();
    };
    require(pair(20, 1024, 5120).oneGateUpPass && pair(32, 1024, 5120).tile == LinearTile::GgufRegister &&
                pair(20, 512, 2560).tile == LinearTile::GgufRegister,
            "Apple10 registers a one-pass gate/up pair on both tensors' columns");
    // Flash-Next's hyper-connection down projection splits K sixteen ways on the register tile at every lane count
    // on 20 and 32 cores; its prefill chunks keep the staged tile's eight.
    const Projection hyperDown = blockProjection(320, 10240, 1, GGUF_FMT_Q80);
    for (const uint32_t cores : {20U, 32U}) {
      for (uint32_t rows = 8; rows <= 32; rows += 8)
        require(gpu(10, cores).plan({{320, 10240}, rows, LinearPhase::Decode, kNone}, hyperDown).configuration() ==
                    LinearConfig{.tile = LinearTile::GgufRegister, .splits = 16},
                "Apple10 splits the hyper-connection down projection sixteen ways");
      require(gpu(10, cores).plan({{320, 10240}, 32, LinearPhase::Prefill, kNone}, hyperDown).configuration() ==
                  LinearConfig{.tile = LinearTile::GgufStaged, .splits = 8},
              "Apple10 stages a hyper-connection down chunk in eight K partitions");
    }
    const Linear m5 = gpu(10, 32);
    for (const uint32_t format : {GGUF_FMT_Q4K, GGUF_FMT_IQ2XXS, GGUF_FMT_Q20})
      for (const auto [n, k] : {std::pair{1024U, 5120U}, {512U, 2560U}, {2560U, 6144U}, {320U, 10240U}}) {
        const Projection p = blockProjection(n, k, 1, format);
        const LinearScratchSize bound = m5.decodeScratchSize(p.shape());
        for (uint32_t rows = 8; rows <= 32; rows += 8) {
          const LinearPlan plan = m5.plan({{n, k}, rows, LinearPhase::Decode, kNone}, p);
          const LinearScratchSize need = plan.scratchSize();
          require(plan.storageRows() == m5.decodeStorageRows(rows, p.shape()) && bound.input >= need.input &&
                      bound.sums >= need.sums && bound.partials >= need.partials && bound.counters >= need.counters,
                  "Apple10 decode storage and scratch bound follow the formats");
        }
      }
  }
  // Apple10 decodes one lane of a projection whose quantized segments store
  // fewer than four and a half bits per weight together, float segments
  // aside, on the 16-row tile where its staged plan runs unsplit, and a
  // gate/up pair where both its tensors' own plans do. The plan's other
  // fields are the two-lane plan's, so the lane's sums stay those of every
  // width, and the decode storage of its shapes holds its rows. Apple9 never
  // does.
  {
    // A projection of `k` inputs in segments of (format, columns), F32 for
    // QuantizedSegment::kFloat32.
    using Parts = std::vector<std::pair<uint32_t, uint32_t>>;
    const auto segments = [](uint32_t k, const Parts &parts) {
      BlockWeights weights;
      uint32_t n = 0;
      for (const auto &[format, columns] : parts) {
        weights.segments.push_back(format == QuantizedSegment::kFloat32
                                       ? QuantizedSegment::floats(columns, k, {})
                                       : QuantizedSegment::planes(format, columns, k, {}, {}, {}));
        weights.segments.back().columnOffset = n;
        n += columns;
      }
      return Projection(n, k, std::move(weights));
    };
    constexpr uint32_t kF32 = QuantizedSegment::kFloat32;
    struct Weighed {
      const char *projection;
      Parts parts;
      bool under;
    };
    for (const Weighed &w : {Weighed{"IQ4_XS", {{GGUF_FMT_IQ4XS, 1024}}, true},
                             Weighed{"MXFP4", {{GGUF_FMT_MXFP4, 1024}}, true},
                             Weighed{"MLX 4-bit in groups of 128", {{GGUF_FMT_AF4G128, 1024}}, true},
                             Weighed{"Q4_K", {{GGUF_FMT_Q4K, 1024}}, false},
                             Weighed{"Q4_0", {{GGUF_FMT_Q40, 1024}}, false},
                             Weighed{"IQ4_NL", {{GGUF_FMT_IQ4NL, 1024}}, false},
                             Weighed{"MLX 4-bit in groups of 64", {{GGUF_FMT_AF4G64, 1024}}, false},
                             Weighed{"the 27B UD-IQ3_XXS attention input, Q2_K q beside Q4_K k and v",
                                     {{GGUF_FMT_Q2K, 12288}, {GGUF_FMT_Q4K, 1024}, {GGUF_FMT_Q4K, 1024}}, true},
                             Weighed{"IQ2_XXS beside as much Q8_0",
                                     {{GGUF_FMT_IQ2XXS, 512}, {GGUF_FMT_Q80, 512}}, false},
                             Weighed{"IQ3_XXS beside F32", {{GGUF_FMT_IQ3XXS, 1024}, {kF32, 96}}, true},
                             Weighed{"F32 alone", {{kF32, 1024}}, false}})
      require(segments(5120, w.parts).shape().underFourAndHalfBits == w.under,
              std::string("the bits per weight of ") + w.projection);
    // The 27B's 17408 x 5120 IQ3_XXS runs it on a 12-core M6 and a 32-core M5
    // Max, its 5120 x 17408 IQ3_S on the M6 but not the M5 Max, which splits
    // its K in four, and Q4_K on neither.
    struct Pick {
      const char *projection;
      uint32_t family, cores, n, k, format;
      bool sixteenRows;
    };
    constexpr Pick kPicks[] = {
        {"IQ3_XXS 17408 x 5120, 12 cores", 11, 12, 17408, 5120, GGUF_FMT_IQ3XXS, true},
        {"IQ3_XXS 17408 x 5120, 32 cores", 10, 32, 17408, 5120, GGUF_FMT_IQ3XXS, true},
        {"IQ3_S 5120 x 17408, 12 cores", 11, 12, 5120, 17408, GGUF_FMT_IQ3S, true},
        {"IQ3_S 5120 x 17408, 32 cores", 10, 32, 5120, 17408, GGUF_FMT_IQ3S, false},
        {"Q4_K 17408 x 5120, 12 cores", 11, 12, 17408, 5120, GGUF_FMT_Q4K, false},
        {"Q4_K 17408 x 5120, 32 cores", 10, 32, 17408, 5120, GGUF_FMT_Q4K, false},
    };
    for (const Pick &p : kPicks)
      require(gpu(p.family, p.cores)
                      .plan({{p.n, p.k}, 8, LinearPhase::Decode, kNone}, blockProjection(p.n, p.k, 1, p.format))
                      .configuration()
                      .sixteenRows == p.sixteenRows,
              std::string("one lane on the 16-row tile: ") + p.projection);
    // The 27B's gate/up pair runs one lane on the 16-row tile in two passes on
    // 12 cores and in one on 32, but beside a Q4_K gate; a pair of 8192
    // columns, unsplit in one pass on 32 cores where each tensor splits, does
    // not.
    for (const uint32_t cores : {12U, 32U}) {
      const Linear device = gpu(11, cores);
      const auto pair = [&](uint32_t n, uint32_t gateFormat) {
        const Projection gate = blockProjection(n, 5120, 1, gateFormat);
        return device.plan({{n, 5120}, 8, LinearPhase::Decode, kGateUp}, blockProjection(n, 5120, 1, GGUF_FMT_IQ3XXS),
                           &gate).configuration();
      };
      const LinearConfig iq = pair(17408, GGUF_FMT_IQ3XXS), wide = pair(8192, GGUF_FMT_IQ3XXS);
      require(iq.sixteenRows && iq.oneGateUpPass == (cores == 32) && !pair(17408, GGUF_FMT_Q4K).sixteenRows &&
                  (cores == 12 || (wide.oneGateUpPass && wide.splits == 1 && !wide.sixteenRows)),
              "the 16-row tile of a gate/up pair at " + std::to_string(cores) + " cores");
    }
    // The projection of `formats`, with a float segment of 64 columns last.
    const auto withFloat = [](uint32_t n, uint32_t k, std::span<const uint32_t> formats) {
      BlockWeights weights = blockProjection(n - 64, k, formats).blocks();
      weights.segments.push_back(QuantizedSegment::floats(64, k, {}));
      weights.segments.back().columnOffset = n - 64;
      return Projection(n, k, std::move(weights));
    };
    using Formats = std::vector<uint32_t>;
    const std::array<Formats, 6> kFormats{Formats{GGUF_FMT_IQ3XXS}, Formats{GGUF_FMT_IQ4XS}, Formats{GGUF_FMT_Q4K},
                                          Formats{GGUF_FMT_AF4G64}, Formats{GGUF_FMT_IQ3XXS, GGUF_FMT_Q4K},
                                          Formats{GGUF_FMT_IQ2XXS, GGUF_FMT_Q80}};
    for (const uint32_t family : {9U, 10U, 11U})
      for (uint32_t cores = 0; cores <= 128; ++cores) {
        const Linear device = gpu(family, cores);
        for (const auto [n, k] : {std::pair{17408U, 5120U}, {5120U, 17408U}, {16384U, 5120U}, {248320U, 5120U},
                                  {1024U, 5120U}})
          for (const Formats &formats : kFormats) {
            const Projection plain = blockProjection(n, k, formats), mixed = withFloat(n, k, formats),
                             q4kGate = blockProjection(n, k, 1);
            const auto alone = [&](const Projection &p) {
              return device.plan({{n, k}, 8, LinearPhase::Decode, kNone}, p).configuration().sixteenRows;
            };
            for (const auto &[p, gate, epilogue] :
                 {std::tuple<const Projection *, const Projection *, LinearEpilogue>{&plain, nullptr, kNone},
                  {&plain, nullptr, kResidual}, {&mixed, nullptr, kNone}, {&plain, &plain, kGateUp},
                  {&plain, &q4kGate, kGateUp}}) {
              if (gate && formats.size() > 1) continue;
              const LinearPlan one = device.plan({{n, k}, 8, LinearPhase::Decode, epilogue}, *p, gate),
                               two = device.plan({{n, k}, 16, LinearPhase::Decode, epilogue}, *p, gate);
              LinearConfig config = one.configuration();
              const bool sixteen = gate ? alone(*p) && alone(*gate)
                                        : family >= 10 && config.tile == LinearTile::GgufStaged &&
                                              config.splits == 1 && p->shape().underFourAndHalfBits;
              uint32_t storage = device.decodeStorageRows(8, p->shape());
              if (gate) storage = std::max(storage, device.decodeStorageRows(8, gate->shape()));
              require(config.sixteenRows == sixteen && one.storageRows() == (sixteen ? 16U : 8U) &&
                          (gate ? one.storageRows() <= storage : one.storageRows() == storage),
                      "one lane on the 16-row tile: family " + std::to_string(family) + ", " +
                          std::to_string(cores) + " cores, " + std::to_string(n) + " x " + std::to_string(k));
              config.sixteenRows = false;
              require(!sixteen || config == two.configuration(), "one lane on the 16-row tile took another plan");
            }
          }
        const Projection floats = segments(5120, {{kF32, 17408}});
        require(!device.plan({{17408, 5120}, 8, LinearPhase::Decode, kNone}, floats).configuration().sixteenRows,
                "a projection of float segments alone took the 16-row tile");
      }
  }
  // Apple9's staged tile, in decode and in prefill chunks, splits K by the
  // register tile's tiers of eight partitions: on 40 cores 17408 x 5120 in
  // four, 5120 x 17408 and 320 x 10240 (sixteen on the register tile) in
  // eight.
  for (const auto [n, k, splits] : {std::tuple{17408U, 5120U, 4U}, {5120U, 17408U, 8U}, {320U, 10240U, 8U}}) {
    const LinearPlan decode = m3.plan({{n, k}, 8, LinearPhase::Decode, LinearEpilogue::None},
                                      blockProjection(n, k, 1, GGUF_FMT_IQ2XXS));
    const LinearPlan chunk = m3.plan({{n, k}, 8, LinearPhase::Prefill, LinearEpilogue::None}, blockProjection(n, k, 1));
    require(decode.configuration() == LinearConfig{.tile = LinearTile::GgufStaged, .splits = splits} &&
                chunk.configuration().splits == splits,
            "Apple9 staged split tiers");
  }
  for (const LinearConfig config :
       {LinearConfig{.tile = LinearTile::GgufRegister, .splits = 3},
        LinearConfig{.tile = LinearTile::GgufRegister, .splits = 2 * LinearConfig::kMaximumSplits}})
    rejects([&] { (void)Linear::plan(down, config, FloatOutput::BFloat16); },
            "the register block decode tile takes a K unit per split",
            "the register tile took a K split that is not a power of two up to the maximum");
  // Split boundaries fall on coefficient units, of 64 inputs at the fewest, and prefill has no register tile.
  const LinearWorkload chunk{{5120, 17408}, 128, LinearPhase::Prefill, LinearEpilogue::None};
  rejects(
      [&] {
        (void)Linear::plan({{5120, 192}, 8, LinearPhase::Decode, LinearEpilogue::None},
                           {.tile = LinearTile::GgufRegister, .splits = 4}, FloatOutput::BFloat16);
      },
      "the register block decode tile takes a K unit per split", "the register tile split three K units four ways");
  rejects([&] { (void)Linear::plan(chunk, {.tile = LinearTile::GgufRegister}, FloatOutput::BFloat16); },
          "the register block decode tile takes a K unit per split", "a prefill plan took the register tile");
  // The prefill tile runs prefill chunks on their matrix grid, unsplit.
  rejects([&] { (void)Linear::plan(down, {.tile = LinearTile::GgufPrefill}, FloatOutput::BFloat16); },
          "invalid block prefill configuration", "a decode plan took the prefill tile");
  rejects([&] { (void)Linear::plan(chunk, {.tile = LinearTile::GgufPrefill, .splits = 2}, FloatOutput::BFloat16); },
          "invalid block prefill configuration", "the prefill tile split K");
  // The staged decode tiles spread their walks on Apple10 and later from 16
  // cores; Apple9's, those of fewer cores and every prefill plan, the staged
  // chunks of up to 32 rows among them, walk in lockstep. No other tile
  // spreads.
  const Projection iq = blockProjection(5120, 17408, 1, GGUF_FMT_IQ2XXS);
  for (const uint32_t family : {9U, 10U, 11U})
    for (const uint32_t cores : {8U, 12U, 15U, 16U, 20U, 40U}) {
      const Linear device = gpu(family, cores);
      const LinearConfig decode =
          device.plan({down.matrix, 8, LinearPhase::Decode, LinearEpilogue::Residual}, iq).configuration();
      require(decode.tile == LinearTile::GgufStaged && decode.spread == (family >= 10 && cores >= 16),
              "GGUF staged decode walk default");
      for (const uint32_t rows : {8U, 32U, 128U})
        require(!device.plan({down.matrix, rows, LinearPhase::Prefill, LinearEpilogue::Residual}, iq)
                     .configuration().spread,
                "a GGUF prefill plan spreads its walks");
      // Where the walks spread, a gate/up pair of one format runs in one pass, its K split as one projection of the
      // pair's columns; a pair of two formats runs two passes, as every pair does elsewhere. The decode scratch bound
      // covers the one pass's partials of both tensors and counters per 32 columns.
      for (const uint32_t n : {1024U, 17408U}) {
        const LinearMatrix matrix{n, 5120};
        const Projection up = blockProjection(n, 5120, 1, GGUF_FMT_IQ2XXS),
                         same = blockProjection(n, 5120, 1, GGUF_FMT_IQ2XXS),
                         other = blockProjection(n, 5120, 1, GGUF_FMT_IQ2XS);
        const bool spreads = family >= 10 && cores >= 16;
        for (const uint32_t rows : {8U, 32U}) {
          const LinearWorkload gateUp{matrix, rows, LinearPhase::Decode, LinearEpilogue::GateUp};
          const LinearPlan one = device.plan(gateUp, up, &same), two = device.plan(gateUp, up, &other);
          const uint32_t pairSplits =
              device.plan({{2 * n, 5120}, rows, LinearPhase::Decode, LinearEpilogue::None},
                          blockProjection(2 * n, 5120, 1, GGUF_FMT_IQ2XXS)).configuration().splits;
          require(one.configuration().oneGateUpPass == spreads && !two.configuration().oneGateUpPass &&
                      (!spreads || (one.configuration().splits == pairSplits && one.configuration().spread &&
                                    one.gateScratchBytes() == 0)),
                  "GGUF gate/up pass default");
          const LinearScratchSize bound = device.decodeScratchSize({n, 5120}), need = one.scratchSize();
          const uint32_t splits = one.configuration().splits;
          require(bound.partials >= need.partials && bound.counters >= need.counters &&
                      (!spreads || (need.partials == (splits > 1 ? uint64_t{splits} * one.storageRows() * 2 * n * 4 : 0) &&
                                    need.counters == (splits > 1 ? n / 32 * 4 : 0))),
                  "GGUF one-pass gate/up scratch");
        }
      }
    }
  rejects(
      [&] {
        (void)Linear::plan(down, {.tile = LinearTile::GgufRegister, .splits = 8, .spread = true},
                           FloatOutput::BFloat16);
      },
      "only the staged block tile spreads its walks", "the register tile spread its walks");
  rejects(
      [&] {
        (void)Linear::plan(down, {.tile = LinearTile::GgufStaged, .oneGateUpPass = true}, FloatOutput::BFloat16);
      },
      "only a staged block gate/up decode runs gate and up in one pass", "a residual plan ran gate/up in one pass");
  // The 16-row tile holds one lane's decode rows on the staged tile only.
  for (const auto &[w, config] :
       {std::pair{down, LinearConfig{.tile = LinearTile::GgufRegister, .sixteenRows = true}},
        {LinearWorkload{down.matrix, 16, LinearPhase::Decode, LinearEpilogue::Residual},
         LinearConfig{.tile = LinearTile::GgufStaged, .sixteenRows = true}},
        {LinearWorkload{down.matrix, 8, LinearPhase::Prefill, LinearEpilogue::Residual},
         LinearConfig{.tile = LinearTile::GgufStaged, .sixteenRows = true}}})
    rejects([&] { (void)Linear::plan(w, config, FloatOutput::BFloat16); },
            "only a staged one-lane block decode runs the 16-row tile",
            "the 16-row tile took other rows than one decode lane's or another tile");
  rejects([&] { (void)Linear::plan(chunk, {.tile = LinearTile::GgufPrefill, .spread = true}, FloatOutput::BFloat16); },
          "only the staged block tile spreads its walks", "the prefill tile spread its walks");
}

// A rotated projection's scratch holds the bf16 rotated rows of a full decode
// batch (32 rows) in decode and of the token budget (2048) in prefill; an
// unrotated one's holds none.
void scratchBoundsRotated() {
  const Linear linear = gpu(10, 16);
  for (const bool rotated : {false, true}) {
    const ProjectionShape shape{5120, 17408, rotated};
    require(linear.decodeScratchSize(shape).rotated == (rotated ? uint64_t{17408} * 32 * 2 : 0) &&
                linear.prefillScratchSize(shape).rotated == (rotated ? uint64_t{17408} * 2048 * 2 : 0),
            "rotated scratch bounds");
  }
}

// The GGUF decode split rules are per-core laws, checked at every core count
// (zero is the fallback), both families and a grid of widths, inputs, epilogues
// and segment counts rather than at the measured machines:
// - a request's sums do not depend on the requests it is batched with on one
//   tile: every batch width runs the register tile's plan (Apple9; Apple10
//   where the grid of column tiles and lanes holds fewer than 1.5 per core,
//   Q4_K's) or the staged tile's, the same at every width, over its tile's
//   rows;
// - a split count is a power of two up to the maximum whose partitions keep
//   the kernel's floor (register: one 256-input unit, staged: 512 inputs in
//   whole 32-input groups);
// - the staged tile spreads its walks from 16 planned cores, the register tile
//   never;
// - it depends on the grid per core only: doubling the width and the core count
//   keeps it, more cores never lower it and a wider grid never raises it, on
//   one tile;
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
              const uint32_t planned = plannedGpuCores(simulatedDevice(family, cores));
              const auto registers = [&](uint32_t lanes) {
                return family == 9 || 2 * uint64_t{n / 64} * lanes < uint64_t{3} * planned;
              };
              // Every batch width takes its tile's configuration and binds its
              // tile's rows.
              std::optional<LinearConfig> registerConfig, stagedConfig;
              for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
                const uint32_t rows = lanes * 8;
                const LinearPlan other = plan(linear, n, rows);
                const LinearConfig c = other.configuration();
                const bool staged = !registers(lanes);
                std::optional<LinearConfig> &first = staged ? stagedConfig : registerConfig;
                if (!first) first = c;
                require(c.tile == (staged ? LinearTile::GgufStaged : LinearTile::GgufRegister) && c == *first &&
                            other.input() == (staged ? LinearInput::Plain : LinearInput::Table16) &&
                            other.storageRows() == (staged && rows == 24 ? 32U : rows),
                        "GGUF decode plan depends on the batch width");
                const uint32_t s = c.splits;
                require(s >= 1 && s <= LinearConfig::kMaximumSplits && (s & (s - 1)) == 0 &&
                            c.spread == (staged && planned >= 16),
                        "GGUF decode plan split count or walk");
                require(s == 1 || (staged ? k / s >= 512 && (k / 32) % s == 0 : k / 256 / s >= 1),
                        "GGUF decode partition below the kernel floor");
              }
              if (cores) {
                for (const uint32_t rows : {8U, 32U}) {
                  const LinearConfig at = plan(linear, n, rows).configuration();
                  // The tile and its split; the staged walk follows the core count (spreadsStagedDecode).
                  const LinearConfig doubled = plan(twice, 2 * n, rows).configuration();
                  require(doubled.tile == at.tile && doubled.splits == at.splits,
                          "GGUF decode plan depends on more than the grid per core");
                  const LinearConfig withMore = plan(more, n, rows).configuration(),
                                     wider = plan(linear, 2 * n, rows).configuration();
                  require(withMore.tile != at.tile || withMore.splits >= at.splits,
                          "GGUF split count falls with more cores");
                  require(wider.tile != at.tile || wider.splits <= at.splits, "GGUF split count rises with the width");
                }
              }
              const LinearScratchSize bound = linear.decodeScratchSize({n, k});
              for (const uint32_t rows : {8U, 16U, 24U, 32U}) {
                const LinearScratchSize need =
                    linear.plan({{n, k}, rows, LinearPhase::Decode, epilogue}, p).scratchSize();
                require(bound.input >= need.input && bound.sums >= need.sums && bound.partials >= need.partials &&
                            bound.counters >= need.counters,
                        "GGUF decode arena bound below a plan");
              }
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
      .gateScratch = allocate(backend, plan.gateScratchBytes()),
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
    rejects([&] { (void)Normalization::addRms(graph, rows, norm, rows, kHidden, kRows, {}, LinearInput::Table16); },
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
    const GdnStateStrides strides{carried, recurrent, carried, GdnStateFormat::Float32};
    const metal::MetalBuffer state = allocate(backend, carried + recurrent);
    const std::array<metal::MetalBuffer, SPLASH_MAXIMUM_BATCH_WIDTH> states{state, state, state, state};
    GdnDecodeBuffers buffers;
    buffers.packed = allocate(backend, kRows * shape.packedWidth * 2);
    buffers.convolutionWeights = allocate(backend, shape.convolutionDimension * 4 * 2);
    buffers.currentStates = states;
    buffers.nextStates = states;
    const std::array<GdnTapeLane, kLanes> tapes{GdnTapeLane{0, gdnTapeLayerBytes(shape), 0}};
    buffers.tape = allocate(backend, 2 * gdnTapeLayerBytes(shape));
    buffers.tapeLanes = tapes;
    buffers.decayWeights = allocate(backend, shape.valueHeads * 4);
    buffers.timeBias = allocate(backend, shape.valueHeads * 2);
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
                                              LinearInput::Table16);
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
// input rows and its output rows of 512 columns up to its last column, 192,
// and those of a second one in its dispatch; and the planes of a quantized
// segment.
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
  // The second projection of a dispatch of two, as the QSA indexer's key beside its query: its weights and its own
  // rows of floatColumns columns.
  const QuantizedSegment second =
      QuantizedSegment::floats(floatColumns, k, allocate(backend, uint64_t{floatColumns} * k * 4));
  const metal::MetalBuffer secondRows = allocate(backend, uint64_t{floatRows} * floatColumns * 4);
  const auto pair = [&](metal::CommandGraph &graph, const QuantizedSegment &segment,
                        const metal::MetalBuffer &rowsOut) {
    const std::array<FloatProjection, 2> projections{FloatProjection{&weights, output, outStride, outOffset},
                                                     FloatProjection{&segment, rowsOut, floatColumns, 0}};
    addGgufFloat(graph, input, projections, floatRows, FloatOutput::Float32);
  };
  requireExtent(backend, second.plane0, uint64_t{floatColumns} * k * 4, 4, "float projection weight",
                [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                  pair(graph, QuantizedSegment::floats(floatColumns, k, view), secondRows);
                });
  requireExtent(backend, secondRows, uint64_t{floatRows} * floatColumns * 4, 4, "float projection output",
                [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) { pair(graph, second, view); });

  // A fused projection of a 320-row Q5_K segment and a 192-row Q4_K one.
  // Each plane is tiles of QUANT_TILE_ROWS rows by groups of 32 inputs, or
  // meta units, and ends at the last row's unit of the last group: row 319 of
  // the Q5_K segment and row 191 of the Q4_K one, in the tile that holds it.
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
      const uint64_t last = (index == 0 ? 320 : 192) - 1;
      return (last / QUANT_TILE_ROWS * blocks + blocks - 1) * QUANT_TILE_ROWS + last % QUANT_TILE_ROWS + 1;
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

// A view of the leading 512 inputs of rows of 1024 (Projection::leadingInputs)
// over 512 outputs, in tiles of QUANT_TILE_ROWS rows: the prefill residual
// tile encodes its leading-input instance with the view's parameters, and each
// plane is read up to the last group the view reads of its last tile, after
// every group of the tiles before. A plane at that extent encodes, one element
// shorter or holding only a matrix of the view's own inputs is refused. Every
// other plan and a rotated view are refused before anything is encoded. The
// views' factories refuse inputs beyond the projection's or of part of a quant
// group or meta unit, rows of part of a tile, and float or rotated weights or a
// view; a view of leading rows holds its planes' first tiles.
void leadingInputViews(metal::MetalBackend &backend) {
  const Linear linear = gpu(10, 16);
  constexpr uint32_t n = 512, k = 512, wide = 1024, rows = 168;
  const auto buffersOf = [&](const LinearPlan &plan) {
    const uint64_t storage = plan.storageRows();
    const LinearScratchSize scratch = plan.scratchSize();
    return LinearBuffers{.input = allocate(backend, storage * k * 2),
                         .output = allocate(backend, storage * n * 2),
                         .residual = allocate(backend, storage * n * 2),
                         .gateScratch = allocate(backend, plan.gateScratchBytes()),
                         .scratch = {allocate(backend, scratch.input), allocate(backend, scratch.sums),
                                     allocate(backend, scratch.partials), allocate(backend, scratch.counters)}};
  };
  const auto residualPrefill = [&](LinearConfig config, LinearEpilogue epilogue) {
    return Linear::plan({{n, k}, rows, LinearPhase::Prefill, epilogue}, config, FloatOutput::BFloat16);
  };
  // The extent of a plane the view reads, in units of `unitBytes`: every tile
  // but the last, `rowGroups` groups (or meta units) of each of its
  // QUANT_TILE_ROWS rows, then the first `readGroups` of the last tile.
  const auto reach = [](uint64_t rowGroups, uint64_t readGroups, uint64_t unitBytes) {
    return ((n / QUANT_TILE_ROWS - 1) * rowGroups + readGroups) * QUANT_TILE_ROWS * unitBytes;
  };

  const QuantFormat &q5k = kQuantFormats[GGUF_FMT_Q5K];
  const QuantizedSegment wideSegment = segmentPlanes(backend, GGUF_FMT_Q5K, n, wide);
  const auto ggufView = [&](const QuantizedSegment &planes) {
    return Projection(n, wide, BlockWeights{{planes}}).leadingInputs(k);
  };
  const Projection gguf = ggufView(wideSegment);
  const LinearPlan ggufPlan = residualPrefill({.tile = LinearTile::GgufPrefill}, LinearEpilogue::Residual);
  {
    metal::CommandGraph graph;
    (void)linear.add(graph, buffersOf(ggufPlan), gguf, ggufPlan);
    const std::string kernel = leadingInputsInstance("gguf_prefill_q5k_r");
    const GgufPrefillLeadingParams params{{k, rows, n, 0}, wide};
    require(graph.dispatches().size() == 1, "a view of leading inputs did not encode one dispatch");
    const metal::ComputeDispatch &dispatch = graph.dispatches()[0];
    require(dispatch.pipelineName == kernel && dispatch.bytes.size() == 1 &&
                dispatch.bytes[0].sizeBytes == sizeof(params) &&
                std::memcmp(dispatch.bytes[0].data, &params, sizeof(params)) == 0,
            "a view of leading inputs did not encode " + kernel + " with its parameters");
  }
  const LinearBuffers ggufBuffers = buffersOf(ggufPlan);
  const QuantizedSegment narrow = segmentPlanes(backend, GGUF_FMT_Q5K, n, k);
  for (const auto &[member, rowGroups, readGroups, unitBytes, name] :
       std::initializer_list<
           std::tuple<metal::MetalBuffer QuantizedSegment::*, uint64_t, uint64_t, uint64_t, const char *>>{
           {&QuantizedSegment::plane0, wide / 32, k / 32, q5k.plane0_bytes, "projection plane0"},
           {&QuantizedSegment::plane1, wide / 32, k / 32, q5k.plane1_bytes, "projection plane1"},
           {&QuantizedSegment::meta, wide / 32 / q5k.meta_groups, k / 32 / q5k.meta_groups, q5k.meta_bytes,
            "projection meta"}}) {
    const auto add = [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
      QuantizedSegment planes = wideSegment;
      planes.*member = view;
      (void)linear.add(graph, ggufBuffers, ggufView(planes), ggufPlan);
    };
    requireExtent(backend, wideSegment.*member, reach(rowGroups, readGroups, unitBytes), unitBytes, name, add);
    metal::CommandGraph graph;
    rejects([&] { add(graph, narrow.*member); }, std::string(name) + " buffer holds",
            "a view of leading inputs over planes of its own inputs was accepted");
    require(graph.empty(), "a view of leading inputs over planes of its own inputs encoded a dispatch");
  }

  Projection rotated = gguf;
  rotated.rotation.signs = allocate(backend, k);
  const auto refuses = [&](const Projection &view, const LinearPlan &plan) {
    metal::CommandGraph graph;
    rejects([&] { (void)linear.add(graph, buffersOf(plan), view, plan); },
            "a view of leading inputs runs only the quantized prefill residual tiles",
            "a view of leading inputs ran on a plan without its instance");
    require(graph.empty(), "a refused view of leading inputs encoded a dispatch");
  };
  refuses(gguf, linear.decodePlan(gguf, 1, LinearEpilogue::Residual));
  for (const LinearEpilogue epilogue : {LinearEpilogue::None, LinearEpilogue::UpWithGate})
    refuses(gguf, residualPrefill({.tile = LinearTile::GgufPrefill}, epilogue));
  refuses(gguf, Linear::plan({{n, k}, 32, LinearPhase::Prefill, LinearEpilogue::Residual},
                             {.tile = LinearTile::GgufStaged}, FloatOutput::BFloat16));
  refuses(rotated, ggufPlan);

  // The views' factories: inputs beyond the projection's or of part of a quant group or meta unit, and a projection
  // of float or rotated weights or of a view, are refused.
  const Projection wideGguf(n, wide, BlockWeights{{wideSegment}});
  constexpr std::string_view kInputs = "a view of leading inputs takes whole quant groups and meta units";
  constexpr std::string_view kSource = "views of a projection's planes take one unrotated quantized tensor";
  rejects([&] { (void)wideGguf.leadingInputs(wide + 256); }, kInputs, "a view of more inputs than rows hold");
  rejects([&] { (void)wideGguf.leadingInputs(k + 16); }, kInputs, "a view of part of a quant group");
  rejects([&] { (void)wideGguf.leadingInputs(k + 64); }, kInputs, "a view of part of a Q5_K meta unit");
  const Projection floats(n, wide, BlockWeights{{QuantizedSegment::floats(n, wide, allocate(backend, n * wide * 4))}});
  Projection rotatedSource = wideGguf;
  rotatedSource.rotation.signs = allocate(backend, wide);
  rejects([&] { (void)floats.leadingInputs(k); }, kSource, "a view of float weights");
  rejects([&] { (void)rotatedSource.leadingInputs(k); }, kSource, "a view of rotated weights");
  rejects([&] { (void)gguf.leadingInputs(k / 2); }, kSource, "a view of a view");
  rejects([&] { (void)floats.leadingRows(backend, 256); }, kSource, "a view of the rows of float weights");

  // A view of leading rows, whole plane tiles, takes each plane's first tiles.
  const Projection ggufRows = wideGguf.leadingRows(backend, 256);
  const QuantizedSegment &segment = ggufRows.blocks().segments.front();
  require(ggufRows.outputSize == 256 && !ggufRows.planeInputs() && segment.outputSize == 256 &&
              segment.inputSize == wide &&
              segment.plane0.sizeBytes() == uint64_t{256} * (wide / 32) * q5k.plane0_bytes &&
              segment.plane1.sizeBytes() == uint64_t{256} * (wide / 32) * q5k.plane1_bytes &&
              segment.meta.sizeBytes() == uint64_t{256} * (wide / 32 / q5k.meta_groups) * q5k.meta_bytes,
          "a view of 256 Q5_K rows does not hold their tile");
  for (const uint32_t count : {0u, QUANT_TILE_ROWS / 2, n + QUANT_TILE_ROWS})
    rejects([&] { (void)wideGguf.leadingRows(backend, count); }, "whole plane tiles",
            "a view of " + std::to_string(count) + " leading rows");
}

// Each scratch field of a GGUF register plan at its extent and one element
// short: the Apple9 decode tile that reads an activation table, on a plan of
// a 40-core Apple9 GPU that splits K. requireExtent only encodes, so this
// runs on any device.
void registerTileExtents(metal::MetalBackend &backend) {
  const Linear m3 = gpu(9, 40);
  constexpr uint32_t n = 512, k = 2048, lanes = 2;
  constexpr uint64_t rows = lanes * 8;
  const Projection block(n, k, BlockWeights{{segmentPlanes(backend, GGUF_FMT_Q4K, n, k)}});
  const LinearPlan plan = m3.decodePlan(block, lanes);
  const LinearScratchSize scratch = plan.scratchSize();
  require(plan.configuration().tile == LinearTile::GgufRegister && scratch.input && scratch.sums &&
              scratch.partials && scratch.counters,
          "the GGUF Apple9 plan is not a register plan that splits K");
  const LinearBuffers buffers{.input = allocate(backend, rows * k * 2),
                              .output = allocate(backend, rows * n * 2),
                              .scratch = {allocate(backend, scratch.input), allocate(backend, scratch.sums),
                                          allocate(backend, scratch.partials), allocate(backend, scratch.counters)}};
  for (const auto &[member, bytes, element, name] :
       std::initializer_list<std::tuple<metal::MetalBuffer LinearScratch::*, uint64_t, uint64_t, const char *>>{
           {&LinearScratch::input, scratch.input, 2, "projection scratch table"},
           {&LinearScratch::sums, scratch.sums, 4, "projection scratch sums"},
           {&LinearScratch::partials, scratch.partials, 4, "projection partials"},
           {&LinearScratch::counters, scratch.counters, 4, "projection counters"}})
    requireExtent(backend, buffers.scratch.*member, bytes, element, name,
                  [&](metal::CommandGraph &graph, const metal::MetalBuffer &view) {
                    LinearBuffers changed = buffers;
                    changed.scratch.*member = view;
                    (void)m3.add(graph, changed, block, plan);
                  });
}

// The register tile splits K into whole coefficient units of each segment's format (quant_register_unit), where a
// plan, which knows no format, bounds the splits by the fewest inputs a unit holds, 64: 512 Q4_K inputs, two units,
// take two splits but not four, 512 Q8_0 inputs four, 192 two, and Q2_K's 256 inputs, two 128-input units, two but
// not four. A refused projection encodes nothing.
void registerSplitUnits(metal::MetalBackend &backend) {
  const Linear m3 = gpu(9, 40);
  constexpr uint32_t n = 512, rows = 8;
  for (const auto [format, k, splits, takes] :
       {std::tuple{GGUF_FMT_Q4K, 512U, 2U, true}, {GGUF_FMT_Q4K, 512U, 4U, false}, {GGUF_FMT_Q80, 512U, 4U, true},
        {GGUF_FMT_Q80, 192U, 2U, true}, {GGUF_FMT_Q2K, 256U, 2U, true}, {GGUF_FMT_Q2K, 256U, 4U, false}}) {
    const Projection block(n, k, BlockWeights{{segmentPlanes(backend, format, n, k)}});
    const LinearPlan plan = Linear::plan({{n, k}, rows, LinearPhase::Decode, kNone},
                                         {.tile = LinearTile::GgufRegister, .splits = splits}, FloatOutput::BFloat16);
    const LinearScratchSize scratch = plan.scratchSize();
    const LinearBuffers buffers{.input = allocate(backend, uint64_t{rows} * k * 2),
                                .output = allocate(backend, uint64_t{rows} * n * 2),
                                .scratch = {allocate(backend, scratch.input), allocate(backend, scratch.sums),
                                            allocate(backend, scratch.partials), allocate(backend, scratch.counters)}};
    const std::string what = std::string(kQuantFormats[format].name) + " weights of " + std::to_string(k) +
                             " inputs in " + std::to_string(splits) + " splits";
    metal::CommandGraph graph;
    if (takes) {
      (void)m3.add(graph, buffers, block, plan);
      require(!graph.empty(), "the register tile encoded nothing for " + what);
    } else {
      rejects([&] { (void)m3.add(graph, buffers, block, plan); }, "the register tile takes a coefficient unit per split",
              "the register tile took " + what);
      require(graph.empty(), "the register tile encoded refused " + what);
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: linear-plan <production.metallib|--cpu>");
    ggufPlans();
    scratchBoundsRotated();
    ggufCoreLaws();
    floatOutputPlans();
    if (std::string_view(argv[1]) == "--cpu") {
      std::cout << "Linear CPU plans: PASS\n";
      return 0;
    }
    metal::MetalBackend backend(argv[1]);
    ggufProjectionMatrix(backend);
    producerTableContract(backend);
    rotatedRegisterInput(backend);
    blockExtents(backend);
    leadingInputViews(backend);
    registerTileExtents(backend);
    registerSplitUnits(backend);
    std::cout << "Linear plans: PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "Linear plans: FAIL: " << error.what() << '\n';
    return 1;
  }
}

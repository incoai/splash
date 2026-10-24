// GPU time of one sparse MoE layer in GGUF formats through ops::MoE, the numbers behind the MoE plans of ops/MoE.cpp,
// at the 35B's shape (hidden 2048, 256 experts, top 8, intermediate 512), Qwen3.8-Flash-Next's (hidden 2560, 512
// experts, top 10, intermediate 640) or Flint's (hidden 2048, 256 experts, top 6 by a sigmoid router, intermediate
// 512, an ungated shared expert of two 512-wide slices), by default in the 35B UD-Q4_K_M's formats (Q4_K gate/up and
// Q5_K down experts, a Q8_0 shared expert) with an F32 router:
//   gguf-moe-benchmark <metallib> [35b|flash-next|flint] [rounds] [gate/up format] [down format] [shared format]
// - decode steps of one to four lanes of 8 rows on the device's plan and on the other GGUF tile, each row routed to
//   top-k experts drawn at random from all of them (random) or, per lane of 8 rows, from a pool of three times top-k
//   (pooled: the rows of a verify sharing their experts), with the share of the plan's expert tiles that run the
//   column tile (an expert's only tile of MOE_GGUF_COLUMN_ROWS live rows or fewer);
// - prefill chunks of 64 to 2048 rows routed at random, on both tiles;
// - every dispatch of the 2048-row chunk and of the 8- and 32-row steps replayed as its own command.
// Each command runs layers of their own routes and buffers over one set of weights larger than the system cache, so a
// layer streams its experts from DRAM: kDecodeLayers per decode step, kPrefillLayers per prefill chunk. Printed are
// medians of GPU ms per layer over `rounds` (20) commands.
#include "../tests/engine/MoeFixture.hpp"
#include "DispatchReplay.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/MoE.h"
#include "model/ExpertStore.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/MoE.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::ops::BlockMoeWeights;
using splash::ops::MoE;
using splash::ops::MoeBuffers;
using splash::ops::MoeConfig;
using splash::ops::MoeScratchField;
using splash::ops::kMoeScratchFields;
using splash::ops::MoeGgufTile;
using splash::ops::MoePlan;
using splash::ops::MoeShape;
using splash::ops::QuantizedSegment;
using namespace gguf_reference;
using moe_fixture::bf16;
using moe_fixture::bfloatBuffer;
using moe_fixture::floatSegment;
using moe_fixture::planeSegment;
using moe_fixture::upload;
using moe_fixture::zeros;

constexpr uint32_t kDecodeLayers = 16, kPrefillLayers = 2, kLaneRows = 8, kPoolTopK = 3;
constexpr uint32_t kPrefillRows[] = {64, 128, 256, 512, 2048};

void allocate(MetalBackend &backend, MoeBuffers &m, const MoePlan &plan) {
  const auto &w = plan.workspace();
  for (const MoeScratchField &field : kMoeScratchFields)
    m.scratch.*field.buffer = zeros(backend, w.*field.bytes, "moe-scratch");
}

enum class Routing { Random, Pooled };

int timing(MetalBackend &backend, const MoeShape &shape, uint32_t rounds, Fmt gateUpFormat, Fmt downFormat,
           Fmt sharedFormat) {
  const uint32_t H = shape.hiddenSize, I = shape.expertIntermediateSize, E = shape.experts,
                 topK = shape.expertsPerToken;
  std::mt19937 local(9);
  // The routing fixture's F32 router, and weights at a model's magnitudes in
  // every format as the tests draw them (scaleRange; an MLX affine format
  // takes bf16 scales and a bias).
  const auto planes = [&](Fmt f, uint32_t rows, uint32_t k) {
    const std::vector<uint8_t> native = makeNative(f, rows, k, local);
    return planeSegment(backend, f, repack(f, native, rows, k, nullptr), rows, k);
  };
  std::vector<float> router(uint64_t{E} * H, 0.0f), sharedGate(H, 0.0f);
  for (uint32_t e = 0; e < E; ++e) router[uint64_t{e} * H + e] = 4.0f;
  const QuantizedSegment routerSegment = floatSegment(backend, router, E, H);
  // An ungated block's weights carry no scalar gate.
  const QuantizedSegment sharedGateSegment =
      shape.sharedGate ? floatSegment(backend, sharedGate, 1, H) : QuantizedSegment{};
  const BlockMoeWeights gguf{routerSegment, sharedGateSegment,
                             {planes(gateUpFormat, E * I, H), planes(sharedFormat, shape.sharedSlices * I, H)},
                             {planes(gateUpFormat, E * I, H), planes(sharedFormat, shape.sharedSlices * I, H)},
                             {planes(downFormat, E * H, I), planes(sharedFormat, shape.sharedSlices * H, I)},
                             splash::model::residentExpertSlots(backend, E), E};
  // Rows whose expert e scores 4 x[e]: the first E inputs rank the experts, the row's routes at 1 - 0.05 rank and the
  // others below 0.05.
  const auto input = [&](uint32_t rows, Routing routing) {
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::vector<float> x(uint64_t{rows} * H);
    std::vector<uint32_t> experts(E);
    std::iota(experts.begin(), experts.end(), 0u);
    const uint32_t pool = routing == Routing::Pooled ? kPoolTopK * topK : E;
    for (uint32_t r = 0; r < rows; ++r) {
      if (routing == Routing::Random || r % kLaneRows == 0) std::shuffle(experts.begin(), experts.end(), local);
      std::vector<uint32_t> candidates(experts.begin(), experts.begin() + pool);
      std::shuffle(candidates.begin(), candidates.end(), local);
      for (uint32_t k = 0; k < H; ++k) x[uint64_t{r} * H + k] = bf16(k < E ? 0.05f * unit(local) : unit(local));
      for (uint32_t rank = 0; rank < topK; ++rank) x[uint64_t{r} * H + candidates[rank]] = bf16(1.0f - 0.05f * rank);
    }
    return x;
  };
  // Layers of one plan in one command, each of its own routes; the share of expert tiles on the column tile.
  struct Layers {
    std::vector<MoeBuffers> buffers;
    CommandGraph graph;
  };
  const auto layers = [&](const MoePlan &plan, Routing routing) {
    Layers l;
    l.buffers.resize(plan.phase() == splash::ops::MoePhase::Decode ? kDecodeLayers : kPrefillLayers);
    for (MoeBuffers &b : l.buffers) {
      b.input = bfloatBuffer(backend, input(plan.rows(), routing), "input");
      b.residual = zeros(backend, uint64_t{plan.rows()} * H * 2, "residual");
      b.output = zeros(backend, uint64_t{plan.rows()} * H * 2, "output");
      allocate(backend, b, plan);
      MoE::add(l.graph, b, gguf, plan);
    }
    return l;
  };
  const auto time = [&](const Layers &l) {
    std::vector<double> samples;
    for (uint32_t i = 0; i < rounds + 1; ++i) {
      const double seconds = backend.submitCommandAsync(l.graph.dispatches()).wait().gpuSeconds;
      if (i) samples.push_back(seconds * 1e3 / l.buffers.size());   // the first round warms the pipelines
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
  };
  const auto columnShare = [&](const Layers &l, const MoePlan &plan) {
    if (plan.configuration().ggufTile == MoeGgufTile::Register) return 0.0;
    uint64_t tiles = 0, column = 0;
    for (const MoeBuffers &b : l.buffers) {
      const uint32_t count = *static_cast<const uint32_t *>(b.scratch.tileCount.contents());
      const auto *descriptors = static_cast<const MoeTileDescriptor *>(b.scratch.tileDescriptors.contents());
      tiles += count;
      for (uint32_t t = 0; t < count; ++t)
        column += moe_column_tile(descriptors, t);
    }
    return 100.0 * column / tiles;
  };
  const splash::ops::ExecutionPlans plans(backend.capabilities());
  const auto otherTile = [](MoeGgufTile tile) {
    return tile == MoeGgufTile::Register ? MoeGgufTile::Staged : MoeGgufTile::Register;
  };
  const auto tileName = [](MoeGgufTile tile) { return tile == MoeGgufTile::Register ? "register" : "staged"; };
  printf("%s, GPU family %u, %u cores: MoE %ux%u, %u experts, top %u, %s router, %u shared slice(s)%s, %s/%s/%s, "
         "median GPU ms per layer of %u rounds\n",
         backend.capabilities().deviceName.c_str(), backend.capabilities().appleGpuFamily,
         backend.capabilities().gpuCoreCount, H, I, E, topK,
         shape.router == splash::ops::MoeRouter::Softmax ? "softmax" : "sigmoid", shape.sharedSlices,
         shape.sharedGate ? " gated" : "", fmtName(gateUpFormat), fmtName(downFormat), fmtName(sharedFormat), rounds);
  // Prefill chunks first: they also bring the GPU clocks up for the short decode layers.
  for (const uint32_t rows : kPrefillRows) {
    const MoePlan plan = plans.moePrefill(shape, rows);
    const MoeGgufTile device = plan.configuration().ggufTile, other = otherTile(device);
    printf("  prefill chunk of %4u rows: %s %.4f", rows, tileName(device), time(layers(plan, Routing::Random)));
    // The register tile takes the weights of quant_register_tensor only, on 8-row tiles.
    if (other == MoeGgufTile::Staged || shape.registerFormats) {
      MoeConfig config = plan.configuration();
      config.ggufTile = other;
      config.expertTile = splash::ops::moeGgufPrefillTile(shape, rows, other);
      printf(", %s %.4f", tileName(other), time(layers(MoE::prefillPlan(shape, rows, config), Routing::Random)));
    }
    printf("\n");
  }
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    printf("  decode %2u rows:", lanes * kLaneRows);
    for (const Routing routing : {Routing::Random, Routing::Pooled}) {
      const MoePlan plan = plans.moeDecode(shape, lanes);
      const MoeGgufTile device = plan.configuration().ggufTile, other = otherTile(device);
      MoeConfig config = plan.configuration();
      config.ggufTile = other;
      const Layers mine = layers(plan, routing);
      const double ms = time(mine);
      printf(" | %s %s %.4f (%2.0f%% column)", routing == Routing::Random ? "random" : "pooled", tileName(device), ms,
             columnShare(mine, plan));
      // The register tile takes the weights of quant_register_tensor only.
      if (other == MoeGgufTile::Register && !shape.registerFormats) continue;
      const MoePlan otherPlan = MoE::decodePlan(shape, lanes, config);
      const Layers theirs = layers(otherPlan, routing);
      const double otherMs = time(theirs);
      printf(", %s %.4f (%2.0f%% column)", tileName(other), otherMs, columnShare(theirs, otherPlan));
    }
    printf("\n");
  }
  // Where the time goes: every dispatch replayed as its own command.
  for (const auto &[label, plan] : {std::pair{"prefill 2048", plans.moePrefill(shape, 2048)},
                                    std::pair{"decode 8", plans.moeDecode(shape, 1)},
                                    std::pair{"decode 32", plans.moeDecode(shape, 4)}}) {
    const Layers l = layers(plan, Routing::Random);
    std::map<std::string, double> spent;
    for (uint32_t i = 0; i < rounds; ++i)
      for (const auto &[name, seconds] : splash::benchmark::replayDispatches(backend, l.graph.dispatches()))
        spent[name] += seconds * 1e3 / (rounds * l.buffers.size());
    printf("  %s:", label);
    for (const auto &[name, ms] : spent) printf(" %s %.4f", name.c_str(), ms);
    printf("\n");
  }
  return 0;
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    const std::string shapeName = argc > 2 ? argv[2] : "35b";
    const Fmt gateUp = argc > 4 ? fmtNamed(argv[4]) : Q4K, down = argc > 5 ? fmtNamed(argv[5]) : Q5K,
              shared = argc > 6 ? fmtNamed(argv[6]) : Q80;
    if (argc < 2 || argc > 7 || (shapeName != "35b" && shapeName != "flash-next" && shapeName != "flint") ||
        gateUp == FMT_COUNT || down == FMT_COUNT || shared == FMT_COUNT) {
      std::cerr << "usage: gguf-moe-benchmark <metallib> [35b|flash-next|flint] [rounds] [gate/up format] "
                   "[down format] [shared format]\n";
      return 2;
    }
    const bool flashNext = shapeName == "flash-next", flint = shapeName == "flint";
    const uint32_t H = flashNext ? 2560 : 2048, I = flashNext ? 640 : 512;
    // gate and up hold most of the routed expert weights.
    MoeShape shape{H, flashNext ? 512u : 256u, flashNext ? 10u : flint ? 6u : 8u, I, uint32_t(gateUp),
                   quant_register_tensor(gateUp, H) && quant_register_tensor(down, I) &&
                       quant_register_tensor(shared, H) && quant_register_tensor(shared, I)};
    if (flint) {
      shape.router = splash::ops::MoeRouter::SigmoidNormalized;
      shape.sharedSlices = 2;
      shape.sharedGate = false;
    }
    try {
      MetalBackend backend(argv[1]);
      return timing(backend, shape, argc > 3 ? std::stoul(argv[3]) : 20, gateUp, down, shared);
    } catch (const std::exception &error) {
      std::cerr << "gguf-moe-benchmark: " << error.what() << '\n';
      return 1;
    }
  }
}

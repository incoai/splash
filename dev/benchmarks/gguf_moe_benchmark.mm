// GPU time of one sparse MoE layer at the 35B shape (hidden 2048, 256 experts, top 8, intermediate 512) in GGUF
// (by default the 35B UD-Q4_K_M's Q4_K gate/up and Q5_K down experts, a Q8_0 shared expert, F32 router), on the
// device's plans and the other GGUF tile: a 2048-row prefill chunk, decode B1-B4 and prefill chunks of their rows,
// prefill chunks of 64 to 512 rows, then every dispatch of the long chunk, B1 and B4 replayed as its own command. The
// numbers behind the MoE plans of ops/MoE.cpp:
//   gguf-moe-benchmark <metallib> [rounds] [gate/up format] [down format] [decode pool]
// Printed are medians of GPU ms per layer over `rounds` (20) commands. Every row routes to 8 experts of a pool of 24
// per request lane (decode, short chunks) or of all 256 (chunks of 64 rows or more): expert e scores 4 x[e], so the
// first 256 inputs pick the routes. The weights exceed the system cache, so each layer streams its experts from DRAM.
#include "../tests/engine/GgufFormatReference.hpp"
#include "DispatchReplay.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/MoE.hpp"

#include <algorithm>
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

float bf16(double value) { return float(__bf16(float(value))); }

MetalBuffer upload(MetalBackend &backend, const void *data, uint64_t bytes, const char *label) {
  MetalBuffer buffer = backend.allocateBuffer(bytes, BufferStorage::Shared, label);
  std::memcpy(buffer.contents(), data, bytes);
  return buffer;
}

MetalBuffer zeros(MetalBackend &backend, uint64_t bytes, const char *label) {
  if (!bytes) return {};
  MetalBuffer buffer = backend.allocateBuffer(bytes, BufferStorage::Shared, label);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

MetalBuffer bfloatBuffer(MetalBackend &backend, const std::vector<float> &values, const char *label) {
  std::vector<__bf16> bits(values.begin(), values.end());
  return upload(backend, bits.data(), bits.size() * 2, label);
}

// The segments of a repacked [rows, K] tensor in format f and of [rows, K] floats.
QuantizedSegment planeSegment(MetalBackend &backend, Fmt f, const Packed &planes, uint32_t rows, uint32_t K) {
  return QuantizedSegment::planes(
      f, rows, K, upload(backend, planes.w0.data(), planes.w0.size(), "gguf-plane0"),
      kQuantFormats[f].plane1_bytes ? upload(backend, planes.w1.data(), planes.w1.size(), "gguf-plane1")
                                    : MetalBuffer{},
      upload(backend, planes.meta.data(), planes.meta.size(), "gguf-meta"));
}
QuantizedSegment floatSegment(MetalBackend &backend, const std::vector<float> &values, uint32_t rows,
                              uint32_t K) {
  return QuantizedSegment::floats(rows, K, upload(backend, values.data(), values.size() * sizeof(float),
                                                  "gguf-floats"));
}

void allocate(MetalBackend &backend, MoeBuffers &m, const MoePlan &plan) {
  const auto &w = plan.workspace();
  for (const MoeScratchField &field : kMoeScratchFields)
    m.scratch.*field.buffer = zeros(backend, w.*field.bytes, "moe-scratch");
}

int timing(MetalBackend &backend, uint32_t rounds, Fmt gateUpFormat, Fmt downFormat, uint32_t decodePool) {
  constexpr uint32_t H = 2048, I = 512, E = 256, kRowsMax = 2048;
  // gate and up hold most of the routed expert weights.
  const MoeShape ggufShape{H, E, 8, I, uint32_t(gateUpFormat)};
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
  const QuantizedSegment sharedGateSegment = floatSegment(backend, sharedGate, 1, H);
  const BlockMoeWeights gguf{routerSegment, sharedGateSegment,
                             {planes(gateUpFormat, E * I, H), planes(Q80, I, H)},
                             {planes(gateUpFormat, E * I, H), planes(Q80, I, H)},
                             {planes(downFormat, E * H, I), planes(Q80, H, I)}};
  // Rows route to 8 of a pool of decodePool experts per lane of 8 rows (decode) or of all 256 (prefill).
  const auto input = [&](uint32_t rows, uint32_t pool) {
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::vector<float> x(uint64_t{rows} * H);
    std::vector<uint32_t> experts(E);
    std::iota(experts.begin(), experts.end(), 0u);
    for (uint32_t r = 0; r < rows; ++r) {
      if (r % 8 == 0) std::shuffle(experts.begin(), experts.end(), local);
      std::vector<uint32_t> candidates(experts.begin(), experts.begin() + pool);
      std::shuffle(candidates.begin(), candidates.end(), local);
      for (uint32_t k = 0; k < H; ++k) x[uint64_t{r} * H + k] = bf16(k < E ? 0.05f * unit(local) : unit(local));
      for (uint32_t rank = 0; rank < 8; ++rank) x[uint64_t{r} * H + candidates[rank]] = bf16(1.0f - 0.05f * rank);
    }
    return x;
  };
  MoeBuffers b;
  b.residual = zeros(backend, uint64_t{kRowsMax} * H * 2, "residual");
  b.output = zeros(backend, uint64_t{kRowsMax} * H * 2, "output");
  const splash::ops::ExecutionPlans plans(backend.capabilities());
  const auto time = [&](const BlockMoeWeights &weights, const MoePlan &plan) {
    allocate(backend, b, plan);
    CommandGraph graph;
    MoE::add(graph, b, weights, plan);
    std::vector<double> samples;
    for (uint32_t i = 0; i < rounds + 1; ++i) {
      const double seconds = backend.submitCommandAsync(graph.dispatches()).wait().gpuSeconds;
      if (i) samples.push_back(seconds * 1e3);   // the first round warms the pipelines
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
  };
  const MoeGgufTile device =
      splash::ops::moeGgufTile(splash::ops::gpuFamilyClass(backend.capabilities().appleGpuFamily), ggufShape);
  const MoeGgufTile other = device == MoeGgufTile::Register ? MoeGgufTile::Staged : MoeGgufTile::Register;
  printf("%s, GPU family %u, %u cores: median GPU ms per MoE layer of %u rounds\n",
         backend.capabilities().deviceName.c_str(), backend.capabilities().appleGpuFamily,
         backend.capabilities().gpuCoreCount, rounds);
  // Prefill chunks first: they also bring the GPU clocks up for the short decode layers.
  b.input = bfloatBuffer(backend, input(kRowsMax, E), "input");
  printf("  prefill %u rows: %.3f\n", kRowsMax, time(gguf, plans.moePrefill(ggufShape, kRowsMax)));
  // Decode on the device's plan against the prefill plan of the same rows (on Apple9 the register tile's MMA, where
  // decode runs the live-row passes), alternating eight times, each a median of `rounds` commands: both medians.
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    b.input = bfloatBuffer(backend, input(lanes * 8, decodePool), "input");
    std::vector<double> decodeTimes, prefillTimes;
    for (int i = 0; i < 8; ++i) {
      decodeTimes.push_back(time(gguf, plans.moeDecode(ggufShape, lanes)));
      prefillTimes.push_back(time(gguf, plans.moePrefill(ggufShape, lanes * 8)));
    }
    std::sort(decodeTimes.begin(), decodeTimes.end());
    std::sort(prefillTimes.begin(), prefillTimes.end());
    printf("  A/B B%u: decode plan %.3f  prefill plan of its rows %.3f  (%+.1f%%)\n", lanes, decodeTimes[4],
           prefillTimes[4], 100.0 * (decodeTimes[4] / prefillTimes[4] - 1.0));
  }
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    b.input = bfloatBuffer(backend, input(lanes * 8, decodePool), "input");
    MoeConfig config = plans.moeDecode(ggufShape, lanes).configuration();
    config.ggufTile = other;
    const double g = time(gguf, plans.moeDecode(ggufShape, lanes));
    const double o = time(gguf, MoE::decodePlan(ggufShape, lanes, config));
    const double gp = time(gguf, plans.moePrefill(ggufShape, lanes * 8));
    printf("  decode B%u: %s %.3f  %s %.3f  | prefill chunk of %u rows: %.3f\n", lanes,
           device == MoeGgufTile::Register ? "register" : "staged", g,
           other == MoeGgufTile::Register ? "register" : "staged", o, lanes * 8, gp);
  }
  for (const uint32_t rows : {64u, 128u, 256u, 512u}) {
    b.input = bfloatBuffer(backend, input(rows, E), "input");
    printf("  prefill chunk of %u rows: %.3f\n", rows, time(gguf, plans.moePrefill(ggufShape, rows)));
  }
  // Where the time goes: every dispatch replayed as its own command.
  for (const auto &[label, plan] : {std::pair{"prefill", plans.moePrefill(ggufShape, kRowsMax)},
                                    std::pair{"B1", plans.moeDecode(ggufShape, 1)},
                                    std::pair{"B4", plans.moeDecode(ggufShape, 4)}}) {
    b.input = bfloatBuffer(backend, input(plan.rows(), plan.rows() == kRowsMax ? E : decodePool), "input");
    allocate(backend, b, plan);
    CommandGraph graph;
    MoE::add(graph, b, gguf, plan);
    std::map<std::string, double> spent;
    for (uint32_t i = 0; i < rounds; ++i)
      for (const auto &[name, seconds] : splash::benchmark::replayDispatches(backend, graph.dispatches()))
        spent[name] += seconds * 1e3 / rounds;
    printf("  %s:", label);
    for (const auto &[name, ms] : spent) printf(" %s %.3f", name.c_str(), ms);
    printf("\n");
  }
  return 0;
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    const Fmt gateUp = argc > 3 ? fmtNamed(argv[3]) : Q4K, down = argc > 4 ? fmtNamed(argv[4]) : Q5K;
    if (argc < 2 || argc > 6 || gateUp == FMT_COUNT || down == FMT_COUNT) {
      std::cerr << "usage: gguf-moe-benchmark <metallib> [rounds] [gate/up format] [down format] [decode pool]\n";
      return 2;
    }
    try {
      MetalBackend backend(argv[1]);
      // argv[5]: the experts a decode lane's routes draw from (24); 256 routes every row at random.
      return timing(backend, argc > 2 ? std::stoul(argv[2]) : 20, gateUp, down, argc > 5 ? std::stoul(argv[5]) : 24);
    } catch (const std::exception &error) {
      std::cerr << "gguf-moe-benchmark: " << error.what() << '\n';
      return 1;
    }
  }
}

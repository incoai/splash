// GPU time of one sparse MoE layer at the 35B shape (hidden 2048, 256 experts, top 8, intermediate 512) in GGUF
// (by default the 35B UD-Q4_K_M's Q4_K gate/up and Q5_K down experts, a Q8_0 shared expert, F32 router) against the
// affine Q4 layer, on the device's plans and the other GGUF tile: a 2048-row prefill chunk, decode B1-B4 and prefill
// chunks of their rows, then every dispatch of the long chunk, B1 and B4 replayed as its own command. The numbers
// behind the MoE plans of ops/MoE.cpp:
//   gguf-moe-benchmark <metallib> [rounds] [gate/up format] [down format]
// Printed are medians of GPU ms per layer over `rounds` (20) commands. Every row routes to 8 experts of a pool of 24
// per request lane (decode, short chunks) or of all 256 (the long chunk), identically for both formats: expert e
// scores 4 x[e], so the first 256 inputs pick the routes. The weights exceed the system cache, so each layer streams
// its experts from DRAM.
#include "../tests/engine/AffineQ4Fixture.hpp"
#include "../tests/engine/GgufFormatReference.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/MoE.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
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
using splash::ops::MoeWeights;
using splash::ops::QuantizedSegment;
using splash::ops::WeightLayout;
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

int timing(MetalBackend &backend, uint32_t rounds, Fmt gateUpFormat, Fmt downFormat) {
  constexpr uint32_t H = 2048, I = 512, E = 256, kRowsMax = 2048;
  // gate and up hold most of the routed expert weights.
  const MoeShape affineShape{H, E, 8, I}, ggufShape{H, E, 8, I, WeightLayout::Block32, uint32_t(gateUpFormat)};
  std::mt19937 local(9);
  // Affine: random Q4 slabs with finite scales, the router and shared gate of the routing fixture.
  const auto affineExperts = [&](uint32_t experts, uint32_t n, uint32_t k) {
    const uint64_t parameters = uint64_t{n} * k / 64;
    const uint16_t scale = std::bit_cast<uint16_t>(__bf16(0.01f)), bias = std::bit_cast<uint16_t>(__bf16(-0.05f));
    return splash::test::expertSlabs(
        backend, experts, n, k, "affine-experts", [&](uint32_t, const splash::test::AffineQ4Planes &planes) {
          for (uint64_t i = 0; i < parameters * 32; ++i) planes.weights[i] = uint8_t(local());
          std::fill_n(planes.scales, parameters, scale);
          std::fill_n(planes.biases, parameters, bias);
        });
  };
  const auto affineRouter = [&](bool routes) {
    const uint64_t elements = uint64_t{256} * H;
    MetalBuffer weights = zeros(backend, elements, "router-weights"),
                scales = zeros(backend, elements / 32, "router-scales"),
                biases = zeros(backend, elements / 32, "router-biases");
    auto *w = static_cast<uint8_t *>(weights.contents());
    auto *sc = static_cast<__bf16 *>(scales.contents());
    for (uint32_t e = 0; e < 256; ++e) {
      if (!routes) break;
      w[(uint64_t(e / 64) * 256 + e) * 64 + e % 64] = 1;
      sc[(e / 64) * 256 + e] = __bf16(4.0f);
    }
    return splash::ops::Q8Projection{{weights, scales, biases}, 256, H};
  };
  MoeWeights affine = splash::ops::AffineMoeWeights{
      .router = affineRouter(true),
      .expertGate = affineExperts(E, I, H),
      .expertUp = affineExperts(E, I, H),
      .expertDown = affineExperts(E, H, I),
      .sharedGate = affineExperts(1, I, H),
      .sharedUp = affineExperts(1, I, H),
      .sharedDown = affineExperts(1, H, I),
      .sharedScalarGate = affineRouter(false),
  };
  // GGUF: the same routing in an F32 router.
  const auto planes = [&](Fmt f, uint32_t rows, uint32_t k) {
    std::uniform_real_distribution<float> d(0.0005f, 0.004f);
    const std::vector<uint8_t> native = makeNative(f, rows, k, local, [&] { return f2h(d(local)); });
    return planeSegment(backend, f, repack(f, native, rows, k, nullptr), rows, k);
  };
  std::vector<float> router(uint64_t{E} * H, 0.0f), sharedGate(H, 0.0f);
  for (uint32_t e = 0; e < E; ++e) router[uint64_t{e} * H + e] = 4.0f;
  const QuantizedSegment routerSegment = floatSegment(backend, router, E, H);
  const QuantizedSegment sharedGateSegment = floatSegment(backend, sharedGate, 1, H);
  MoeWeights gguf;
  gguf = BlockMoeWeights{routerSegment, sharedGateSegment,
                             {planes(gateUpFormat, E * I, H), planes(Q80, I, H)},
                             {planes(gateUpFormat, E * I, H), planes(Q80, I, H)},
                             {planes(downFormat, E * H, I), planes(Q80, H, I)}};
  // Rows route to 8 of a pool of 24 experts per lane of 8 rows (decode) or of all 256 (prefill).
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
  const auto time = [&](const MoeWeights &weights, const MoePlan &plan) {
    allocate(backend, b, plan);
    CommandGraph graph;
    MoE::add(graph, b, weights, plan);
    std::vector<double> samples;
    for (uint32_t i = 0; i < rounds + 1; ++i) {
      const double seconds = backend.submitCommand(graph.dispatches()).gpuSeconds;
      if (i) samples.push_back(seconds * 1e3);   // the first round warms the pipelines
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
  };
  const MoeGgufTile device = splash::ops::moeGgufTile(backend.capabilities().appleGpuFamily, ggufShape);
  const MoeGgufTile other = device == MoeGgufTile::Register ? MoeGgufTile::Staged : MoeGgufTile::Register;
  printf("%s, GPU family %u, %u cores: median GPU ms per MoE layer of %u rounds\n",
         backend.capabilities().deviceName.c_str(), backend.capabilities().appleGpuFamily,
         backend.capabilities().gpuCoreCount, rounds);
  // Prefill chunks first: they also bring the GPU clocks up for the short decode layers.
  b.input = bfloatBuffer(backend, input(kRowsMax, E), "input");
  const double affinePrefill = time(affine, plans.moePrefill(affineShape, kRowsMax));
  printf("  prefill %u rows: affine %.3f  gguf %.3f\n", kRowsMax, affinePrefill,
         time(gguf, plans.moePrefill(ggufShape, kRowsMax)));
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    b.input = bfloatBuffer(backend, input(lanes * 8, 24), "input");
    MoeConfig config = plans.moeDecode(ggufShape, lanes).configuration();
    config.ggufTile = other;
    const double a = time(affine, plans.moeDecode(affineShape, lanes));
    const double g = time(gguf, plans.moeDecode(ggufShape, lanes));
    const double o = time(gguf, MoE::decodePlan(ggufShape, lanes, config));
    const double ap = time(affine, plans.moePrefill(affineShape, lanes * 8));
    const double gp = time(gguf, plans.moePrefill(ggufShape, lanes * 8));
    printf("  decode B%u: affine %.3f  gguf %s %.3f  gguf %s %.3f  | prefill chunk of %u rows: affine %.3f  "
           "gguf %.3f\n",
           lanes, a, device == MoeGgufTile::Register ? "register" : "staged", g,
           other == MoeGgufTile::Register ? "register" : "staged", o, lanes * 8, ap, gp);
  }
  // Where the time goes: every dispatch replayed as its own command.
  for (const auto &[label, weights, plan] :
       {std::tuple{"affine prefill", &affine, plans.moePrefill(affineShape, kRowsMax)},
        std::tuple{"gguf prefill", &gguf, plans.moePrefill(ggufShape, kRowsMax)},
        std::tuple{"affine B1", &affine, plans.moeDecode(affineShape, 1)},
        std::tuple{"gguf B1", &gguf, plans.moeDecode(ggufShape, 1)},
        std::tuple{"affine B4", &affine, plans.moeDecode(affineShape, 4)},
        std::tuple{"gguf B4", &gguf, plans.moeDecode(ggufShape, 4)}}) {
    b.input = bfloatBuffer(backend, input(plan.rows(), plan.rows() == kRowsMax ? E : 24), "input");
    allocate(backend, b, plan);
    CommandGraph graph;
    MoE::add(graph, b, *weights, plan);
    backend.setDispatchProfiling(true);
    for (uint32_t i = 0; i < rounds; ++i) static_cast<void>(backend.submitCommand(graph.dispatches()));
    backend.setDispatchProfiling(false);
    std::map<std::string, double> spent;
    for (const auto &t : backend.takeDispatchProfile()) spent[t.pipelineName] += t.gpuSeconds * 1e3 / rounds;
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
    if (argc < 2 || argc > 5 || gateUp == FMT_COUNT || down == FMT_COUNT) {
      std::cerr << "usage: gguf-moe-benchmark <metallib> [rounds] [gate/up format] [down format]\n";
      return 2;
    }
    try {
      MetalBackend backend(argv[1]);
      return timing(backend, argc > 2 ? std::stoul(argv[2]) : 20, gateUp, down);
    } catch (const std::exception &error) {
      std::cerr << "gguf-moe-benchmark: " << error.what() << '\n';
      return 1;
    }
  }
}

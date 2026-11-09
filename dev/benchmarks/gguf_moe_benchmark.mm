// GPU time of one sparse MoE layer in GGUF formats through ops::MoE, the numbers behind the MoE plans of ops/MoE.cpp,
// at the 35B's shape (hidden 2048, 256 experts, top 8, intermediate 512), Qwen3.8-Flash-Next's (hidden 2560, 512
// experts, top 10, intermediate 640) or Flint's (hidden 2048, 256 experts, top 6 by a sigmoid router, intermediate
// 512, an ungated shared expert of two 512-wide slices), by default in the 35B UD-Q4_K_M's formats (Q4_K gate/up and
// Q5_K down experts, a Q8_0 shared expert) with an F32 router:
//   gguf-moe-benchmark <metallib> [35b|flash-next|flint] [rounds] [gate/up format] [down format] [shared format]
// - decode steps of one to four lanes of 8 rows on the device's plan, each row routed to top-k experts drawn at random
//   from all of them (random) or, per lane of 8 rows, from a pool of three times top-k (pooled: the rows of a verify
//   sharing their experts), with the share of the plan's expert tiles that run the column tile (an expert's only tile
//   of MOE_GGUF_COLUMN_ROWS live rows or fewer);
// - prefill chunks of 64 to 2048 rows routed at random;
// - every dispatch of the 2048-row chunk and of the 8- and 32-row steps replayed as its own command.
// Each command runs layers of their own routes and buffers over one set of weights larger than the system cache, so a
// layer streams its experts from DRAM: kDecodeLayers per decode step, kPrefillLayers per prefill chunk. Printed are
// medians of GPU ms per layer over `rounds` (20) commands.
//   gguf-moe-benchmark <metallib> repack [rounds]
// - the repack of streamed experts into their slots (moe_expert_load) at 1, 2, 4, 8 and 32 jobs, and of a tensor of
//   32 experts' rows at model preparation (gguf_repack), in each of Q4_K, Q5_K, Q6_K, Q8_0, Q2_0, IQ2_XXS and af4g64
//   at the 35B's expert shape, the 35B's Q4_K/Q5_K pair and Flash-Next's Q2_0 shape: median GPU ms per dispatch over
//   `rounds` commands of dispatches of their own staging and slots (over 50 MB), native GB/s, and every plane
//   compared bytewise with the reference repack.
#include "../tests/engine/MoeFixture.hpp"
#include "DispatchReplay.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/GgufRepack.h"
#include "metal/abi/MoE.h"
#include "model/ExpertStore.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/MoE.hpp"

#include <algorithm>
#include <array>
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
using splash::ops::MoeScratchField;
using splash::ops::kMoeScratchFields;
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
  BlockMoeWeights gguf{routerSegment, sharedGateSegment,
                       {planes(gateUpFormat, E * I, H), planes(sharedFormat, shape.sharedSlices * I, H)},
                       {planes(gateUpFormat, E * I, H), planes(sharedFormat, shape.sharedSlices * I, H)},
                       {planes(downFormat, E * H, I), planes(sharedFormat, shape.sharedSlices * H, I)},
                       splash::model::residentExpertSlots(backend, E), E};
  gguf.chunks = splash::ops::residentExpertChunks(gguf);
  const std::array<const BlockMoeWeights *, 1> blocks{&gguf};
  const uint32_t routerFormat = splash::ops::moeRouterFormat(blocks);
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
  const auto columnShare = [&](const Layers &l) {
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
  const splash::ops::ExecutionPlans plans(backend.capabilities(), splash::ops::Int8Activations::On);
  printf("%s, GPU family %u, %u cores: MoE %ux%u, %u experts, top %u, %s router, %u shared slice(s)%s, %s/%s/%s, "
         "median GPU ms per layer of %u rounds\n",
         backend.capabilities().deviceName.c_str(), backend.capabilities().appleGpuFamily,
         backend.capabilities().gpuCoreCount, H, I, E, topK,
         shape.router == splash::ops::MoeRouter::Softmax ? "softmax" : "sigmoid", shape.sharedSlices,
         shape.sharedGate ? " gated" : "", fmtName(gateUpFormat), fmtName(downFormat), fmtName(sharedFormat), rounds);
  // Prefill chunks first: they also bring the GPU clocks up for the short decode layers.
  for (const uint32_t rows : kPrefillRows)
    printf("  prefill chunk of %4u rows: %.4f\n", rows,
           time(layers(plans.moePrefill(shape, routerFormat, rows), Routing::Random)));
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    printf("  decode %2u rows:", lanes * kLaneRows);
    for (const Routing routing : {Routing::Random, Routing::Pooled}) {
      const Layers l = layers(plans.moeDecode(shape, routerFormat, lanes), routing);
      const double ms = time(l);
      printf(" | %s %.4f (%2.0f%% column)", routing == Routing::Random ? "random" : "pooled", ms, columnShare(l));
    }
    printf("\n");
  }
  // Where the time goes: every dispatch replayed as its own command.
  for (const auto &[label, plan] : {std::pair{"prefill 2048", plans.moePrefill(shape, routerFormat, 2048)},
                                    std::pair{"decode 8", plans.moeDecode(shape, routerFormat, 1)},
                                    std::pair{"decode 32", plans.moeDecode(shape, routerFormat, 4)}}) {
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

// The repack of native expert rows into slot planes (moe_expert_load) and into an image (gguf_repack).
int repackTiming(MetalBackend &backend, uint32_t rounds) {
  struct Case {
    const char *shape;
    Fmt gateUp, down;
    uint32_t hidden, intermediate;
  };
  std::vector<Case> cases;
  for (const char *name : {"q4k", "q5k", "q6k", "q80", "q20", "iq2xxs", "af4g64"})
    cases.push_back({"35b", fmtNamed(name), fmtNamed(name), 2048, 512});
  cases.push_back({"35b", Q4K, Q5K, 2048, 512});
  cases.push_back({"flash-next", Q20, Q20, 2560, 640});
  constexpr uint32_t kNativeExperts = 8, kJobs[] = {1, 2, 4, 8, 32};
  constexpr uint64_t kPlace = 16384;
  const auto pad = [&](uint64_t bytes) { return (bytes + 2 * kPlace - 1) / kPlace * kPlace; };
  printf("%s, GPU family %u, %u cores: expert repack, median GPU ms per dispatch of %u rounds, native GB/s\n",
         backend.capabilities().deviceName.c_str(), backend.capabilities().appleGpuFamily,
         backend.capabilities().gpuCoreCount, rounds);
  std::mt19937 local(11);
  bool same = true;
  for (const Case &c : cases) {
    const std::array<Fmt, 3> formats{c.gateUp, c.gateUp, c.down};
    const std::array<uint32_t, 3> rows{c.intermediate, c.intermediate, c.hidden},
        columns{c.hidden, c.hidden, c.intermediate};
    std::array<std::vector<std::vector<uint8_t>>, 3> native;
    std::array<std::vector<Packed>, 3> planes;
    MoeExpertLoadParams params{};
    uint64_t expertBytes = 0;
    for (uint32_t z = 0; z < 3; ++z) {
      const QuantFormat &f = kQuantFormats[formats[z]];
      const uint32_t G = columns[z] / 32;
      for (uint32_t e = 0; e < kNativeExperts; ++e) {
        native[z].push_back(makeNative(formats[z], rows[z], columns[z], local));
        planes[z].push_back(repack(formats[z], native[z].back(), rows[z], columns[z], nullptr));
      }
      params.projections[z] = {formats[z], rows[z], columns[z], rowBytes(formats[z], columns[z]),
                               {uint64_t{rows[z]} * G * f.plane0_bytes, uint64_t{rows[z]} * G * f.plane1_bytes,
                                uint64_t{rows[z]} * (G / f.meta_groups) * f.meta_bytes}};
      expertBytes += native[z][0].size();
    }
    printf("%s %s/%s/%s, %.2f MB an expert\n", c.shape, fmtName(formats[0]), fmtName(formats[1]),
           fmtName(formats[2]), expertBytes / 1e6);
    for (const uint32_t jobs : kJobs) {
      const uint32_t layers = std::max(2u, 32 / jobs), slots = layers * jobs;
      // Staging places of a page past their rows, and the slots of one chunk.
      std::array<uint64_t, 3> place{};
      uint64_t stagingBytes = 0;
      for (uint32_t z = 0; z < 3; ++z) stagingBytes += uint64_t{slots} * (place[z] = pad(native[z][0].size()));
      MetalBuffer staging = backend.allocateBuffer(stagingBytes, BufferStorage::Shared, "repack-staging");
      auto chunk = std::make_shared<std::vector<MetalBuffer>>();
      std::array<MetalBuffer, 3> w0, w1, meta;
      for (uint32_t z = 0; z < 3; ++z) {
        w0[z] = zeros(backend, slots * params.projections[z].slot_bytes[0], "repack-plane0");
        if (params.projections[z].slot_bytes[1])
          w1[z] = zeros(backend, slots * params.projections[z].slot_bytes[1], "repack-plane1");
        meta[z] = zeros(backend, slots * params.projections[z].slot_bytes[2], "repack-meta");
        chunk->insert(chunk->end(), {w0[z], w1[z] ? w1[z] : meta[z], meta[z]});
      }
      MoeExpertLoadParams load = params;
      load.floor_slots = load.chunk_slots = slots;
      // Layer l's job j takes slot l * jobs + j and native expert (l * jobs + j) % kNativeExperts.
      uint64_t at = 0;
      std::vector<uint64_t> sources(uint64_t{slots} * 3);
      for (uint32_t z = 0; z < 3; ++z)
        for (uint32_t s = 0; s < slots; ++s, at += place[z]) {
          const std::vector<uint8_t> &rowsOf = native[z][s % kNativeExperts];
          std::memcpy(static_cast<uint8_t *>(staging.contents()) + at, rowsOf.data(), rowsOf.size());
          sources[uint64_t{s} * 3 + z] = at;
        }
      CommandGraph graph;
      for (uint32_t l = 0; l < layers; ++l) {
        std::vector<uint32_t> ids(jobs);
        std::iota(ids.begin(), ids.end(), l * jobs);
        graph.add("moe_expert_load",
                  {staging, upload(backend, ids.data(), ids.size() * 4, "repack-slots"),
                   upload(backend, sources.data() + uint64_t{l} * jobs * 3, jobs * 3 * 8, "repack-sources")},
                  load, {MoE::expertLoadGroups(params), jobs, 3}, {MOE_EXPERT_LOAD_THREADS, 1, 1});
        graph.addresses(chunk, true);
      }
      // Every slot's planes against the reference repack, after filling them with 0xA5.
      for (uint32_t z = 0; z < 3; ++z)
        for (const MetalBuffer *buffer : {&w0[z], &w1[z], &meta[z]})
          if (*buffer) std::memset(buffer->contents(), 0xA5, buffer->sizeBytes());
      static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
      const bool ok = [&] {
        for (uint32_t z = 0; z < 3; ++z)
          for (uint32_t s = 0; s < slots; ++s) {
            const Packed &want = planes[z][s % kNativeExperts];
            const auto &bytes = params.projections[z].slot_bytes;
            if (std::memcmp(static_cast<const uint8_t *>(w0[z].contents()) + s * bytes[0], want.w0.data(),
                            bytes[0]) ||
                (bytes[1] && std::memcmp(static_cast<const uint8_t *>(w1[z].contents()) + s * bytes[1],
                                         want.w1.data(), bytes[1])) ||
                std::memcmp(static_cast<const uint8_t *>(meta[z].contents()) + s * bytes[2], want.meta.data(),
                            bytes[2]))
              return false;
          }
        return true;
      }();
      same = same && ok;
      std::vector<double> samples;
      for (uint32_t i = 0; i < rounds + 1; ++i) {
        const double seconds = backend.submitCommandAsync(graph.dispatches()).wait().gpuSeconds;
        if (i) samples.push_back(seconds / layers);
      }
      std::sort(samples.begin(), samples.end());
      const double ms = samples[samples.size() / 2] * 1e3;
      printf("  moe_expert_load %2u jobs: %.4f ms %.1f GB/s%s\n", jobs, ms, jobs * expertBytes / (ms * 1e6),
             ok ? "" : " PLANES DIFFER");
    }
    // Model preparation: 32 experts' gate rows, then 8 experts' down rows, each a tensor of its own.
    for (const uint32_t z : {0u, 2u}) {
      const uint32_t experts = z ? 8 : 32, n = experts * rows[z], K = columns[z];
      const QuantFormat &f = kQuantFormats[formats[z]];
      std::vector<uint8_t> rowsOf;
      for (uint32_t e = 0; e < experts; ++e)
        rowsOf.insert(rowsOf.end(), native[z][e % kNativeExperts].begin(), native[z][e % kNativeExperts].end());
      const Packed want = repack(formats[z], rowsOf, n, K, nullptr);
      const uint64_t bytes0 = uint64_t{n} * (K / 32) * f.plane0_bytes, bytes1 = uint64_t{n} * (K / 32) * f.plane1_bytes,
                     bytesMeta = want.meta.size();
      MetalBuffer src = upload(backend, rowsOf.data(), rowsOf.size(), "repack-rows");
      MetalBuffer dst = zeros(backend, bytes0 + bytes1 + bytesMeta, "repack-image");
      GgufRepackParams p{};
      p.rows = n;
      p.input_size = K;
      p.fmt = formats[z];
      p.src_row_bytes = rowBytes(formats[z], K);
      p.dst_plane1 = bytes1 ? static_cast<uint32_t>(bytes0) : 0;
      p.dst_meta = static_cast<uint32_t>(bytes0 + bytes1);
      CommandGraph graph;
      graph.add("gguf_repack", {src, dst}, p, {(uint64_t{n} * (K / 32) + 255) / 256, 1, 1}, {256, 1, 1});
      std::memset(dst.contents(), 0xA5, dst.sizeBytes());
      static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
      const uint8_t *image = static_cast<const uint8_t *>(dst.contents());
      const bool ok = !std::memcmp(image, want.w0.data(), bytes0) &&
                      (!bytes1 || !std::memcmp(image + bytes0, want.w1.data(), bytes1)) &&
                      !std::memcmp(image + bytes0 + bytes1, want.meta.data(), bytesMeta);
      same = same && ok;
      std::vector<double> samples;
      for (uint32_t i = 0; i < rounds + 1; ++i) {
        const double seconds = backend.submitCommandAsync(graph.dispatches()).wait().gpuSeconds;
        if (i) samples.push_back(seconds);
      }
      std::sort(samples.begin(), samples.end());
      const double ms = samples[samples.size() / 2] * 1e3;
      printf("  gguf_repack %5u x %4u: %.4f ms %.1f GB/s%s\n", n, K, ms, rowsOf.size() / (ms * 1e6),
             ok ? "" : " PLANES DIFFER");
    }
  }
  printf("planes %s\n", same ? "bytewise the reference repack's" : "DIFFER");
  return same ? 0 : 1;
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc > 2 && std::string(argv[2]) == "repack") {
      if (argc > 4) {
        std::cerr << "usage: gguf-moe-benchmark <metallib> repack [rounds]\n";
        return 2;
      }
      try {
        MetalBackend backend(argv[1]);
        return repackTiming(backend, argc > 3 ? std::stoul(argv[3]) : 20);
      } catch (const std::exception &error) {
        std::cerr << "gguf-moe-benchmark: " << error.what() << '\n';
        return 1;
      }
    }
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
    MoeShape shape{H, flashNext ? 512u : 256u, flashNext ? 10u : flint ? 6u : 8u, I};
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

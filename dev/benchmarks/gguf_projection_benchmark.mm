// GPU time of one GGUF projection through ops::Linear on both decode tiles (the Apple9 register tile and the staged
// tile, on the device policy's walk) at one to four lanes and every K split, the measurements behind the split tiers
// of runtime/ops/LinearGguf.cpp:
//   gguf-projection-benchmark <metallib> <fmt[+fmt+fmt]> <N[+N+N]> <K> [none|residual|gateup] [rounds]
//                             [prefill=R[,R...]]
// The projection has one segment per format and width (up to three, fused, no epilogue), each a multiple of 256
// columns. With prefill=, it times the 128-row prefill tile instead, one case per chunk of R rows (more than 32: the
// decode tiles run the smaller chunks) of a single segment; gateup is then the up projection's gated epilogue. The
// weights stay DRAM-cold: a command streams every copy of a ring of at least 384 MiB once, each projection with the
// next copy (decode gate/up: the gate and up weights of two copies). The cases run in ABBA order after a warm-up;
// printed are the medians of `rounds` (20) commands in ms per projection, the device policy's choice marked with *.
// Widths emulate core counts: the policy's tiers count threadgroups per core.
#include "../tests/engine/GgufFormatReference.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/Linear.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::ops;
using namespace gguf_reference;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;

namespace {

constexpr uint64_t kRingBytes = 384ull << 20;
constexpr uint32_t kMinimumCopies = 3, kMaximumCopies = 64, kMaximumLanes = 4, kLaneRows = 8;
constexpr double kWarmupSeconds = 0.1;

std::vector<std::string> split(const std::string &text, char separator) {
  std::vector<std::string> parts;
  std::stringstream stream(text);
  for (std::string part; std::getline(stream, part, separator);) parts.push_back(part);
  return parts;
}

MetalBuffer upload(MetalBackend &backend, const std::vector<uint8_t> &bytes) {
  MetalBuffer buffer = backend.allocateBuffer(bytes.size(), metal::BufferStorage::Shared, "projection operand");
  std::memcpy(buffer.contents(), bytes.data(), bytes.size());
  return buffer;
}

// One plan, printed on its line (L<lanes> or R<rows>) under its label.
struct Case {
  std::string line, label;
  LinearPlan plan;
  bool chosen;
  std::vector<double> ms;
};

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    const std::string prefillOption = "prefill=";
    const bool prefill = argc == 8;
    if (argc < 5 || argc > 8 || (prefill && std::string(argv[7]).rfind(prefillOption, 0))) {
      std::cerr << "usage: gguf-projection-benchmark <metallib> <fmt[+fmt+fmt]> <N[+N+N]> <K> "
                   "[none|residual|gateup] [rounds] [prefill=R[,R...]]\n";
      return 2;
    }
    const std::vector<std::string> formats = split(argv[2], '+'), widths = split(argv[3], '+');
    const uint32_t K = uint32_t(std::stoul(argv[4])), rounds = argc > 6 ? uint32_t(std::stoul(argv[6])) : 20;
    const std::string epilogueName = argc > 5 ? argv[5] : "none";
    std::vector<uint32_t> prefillRows;
    if (prefill)
      for (const std::string &rows : split(argv[7] + prefillOption.size(), ','))
        prefillRows.push_back(uint32_t(std::stoul(rows)));
    const LinearEpilogue epilogue = epilogueName == "residual" ? LinearEpilogue::Residual
                                  : epilogueName != "gateup"   ? LinearEpilogue::None
                                  : prefill                    ? LinearEpilogue::UpWithGate
                                                               : LinearEpilogue::GateUp;
    if (formats.size() != widths.size() || formats.size() > 3 ||
        (formats.size() > 1 && (epilogue != LinearEpilogue::None || prefill)) ||
        (epilogue == LinearEpilogue::None && epilogueName != "none") || !rounds ||
        (prefill && (prefillRows.empty() ||
                     std::any_of(prefillRows.begin(), prefillRows.end(), [](uint32_t r) { return r <= 32; })))) {
      std::cerr << "one format per width, at most three, fused projections take no epilogue; prefill takes one "
                   "format and chunks of more than 32 rows\n";
      return 2;
    }
    try {
      MetalBackend backend(argv[1]);
      std::mt19937 rng(9);
      // One image of the projection's segments, uploaded once per copy of the ring.
      struct Image { Fmt format; uint32_t N, offset; Packed planes; };
      std::vector<Image> images;
      uint32_t N = 0;
      uint64_t bytes = 0;
      for (size_t i = 0; i < formats.size(); ++i) {
        const Fmt f = fmtNamed(formats[i]);
        const uint32_t n = uint32_t(std::stoul(widths[i]));
        if (f == FMT_COUNT || !n || n % 256 || !K || K % 256) throw std::invalid_argument("bad segment " + formats[i]);
        images.push_back({f, n, N, repack(f, makeNative(f, n, K, rng), n, K, nullptr)});
        N += n;
        for (const std::vector<uint8_t> *plane : {&images.back().planes.w0, &images.back().planes.w1,
                                                  &images.back().planes.meta})
          bytes += plane->size();
      }
      const uint32_t copies = uint32_t(std::clamp<uint64_t>((kRingBytes + bytes - 1) / bytes, kMinimumCopies,
                                                            kMaximumCopies));
      std::vector<Projection> ring;
      for (uint32_t c = 0; c < copies; ++c) {
        BlockWeights weights;
        for (const Image &image : images) {
          const Packed &p = image.planes;
          weights.segments.push_back(
              QuantizedSegment::planes(image.format, image.N, K, upload(backend, p.w0),
                                       kQuantFormats[image.format].plane1_bytes ? upload(backend, p.w1) : MetalBuffer{},
                                       upload(backend, p.meta)));
          weights.segments.back().columnOffset = image.offset;
        }
        ring.emplace_back(N, K, std::move(weights));
      }
      const Linear linear(backend.capabilities());
      std::vector<Case> cases;
      for (const uint32_t rows : prefillRows) {
        const LinearWorkload w{{N, K}, rows, LinearPhase::Prefill, epilogue};
        const LinearConfig config{.tile = LinearTile::GgufPrefill};
        cases.push_back({"R" + std::to_string(rows), "prefill128", Linear::plan(w, config, FloatOutput::BFloat16),
                         config == linear.plan(w, ring.front()).configuration(), {}});
      }
      for (uint32_t lanes = 1; lanes <= kMaximumLanes && !prefill; ++lanes) {
        const LinearWorkload w{{N, K}, lanes * kLaneRows, LinearPhase::Decode, epilogue};
        const LinearConfig policy = linear.plan(w, ring.front()).configuration();
        for (const LinearTile tile : {LinearTile::GgufRegister, LinearTile::GgufStaged})
          for (uint32_t splits = 1; splits <= LinearConfig::kMaximumSplits; splits *= 2) {
            const bool registerTile = tile == LinearTile::GgufRegister;
            if (registerTile ? K / 256 < splits : (K / 32) % splits) continue;
            const LinearConfig config{.tile = tile, .splits = splits, .spread = !registerTile && policy.spread};
            cases.push_back({"L" + std::to_string(lanes),
                             std::string(registerTile ? "register" : "staged") + " S" + std::to_string(splits),
                             Linear::plan(w, config, FloatOutput::BFloat16), config == policy, {}});
          }
      }
      // Buffers of the widest plan; the scratch of the plan that needs the most.
      LinearScratchSize size;
      uint64_t rows = 0;
      for (const Case &c : cases) {
        size.include(c.plan.scratchSize());
        rows = std::max<uint64_t>(rows, c.plan.storageRows());
      }
      const auto zeros = [&](uint64_t n) {
        if (!n) return MetalBuffer{};
        MetalBuffer b = backend.allocateBuffer(n, metal::BufferStorage::Shared, "projection scratch");
        std::memset(b.contents(), 0, n);
        return b;
      };
      MetalBuffer input = zeros(rows * K * 2), output = zeros(rows * N * 2), residual = zeros(rows * N * 2),
                  gate = zeros(rows * N * 2);
      {
        std::uniform_real_distribution<float> unit(-1.f, 1.f);
        auto *x = static_cast<__bf16 *>(input.contents());
        for (uint64_t i = 0; i < rows * K; ++i) x[i] = __bf16(unit(rng));
      }
      const LinearScratch scratch{zeros(size.input), zeros(size.sums), zeros(size.partials), zeros(size.counters)};
      uint32_t next = 0;
      const auto time = [&](Case &c) {
        CommandGraph graph;
        const LinearBuffers b{.input = input, .output = output,
                              .residual = epilogue == LinearEpilogue::Residual ? residual : MetalBuffer{},
                              .gateScratch = gate, .scratch = scratch};
        const uint32_t step = epilogue == LinearEpilogue::GateUp ? 2 : 1;
        for (uint32_t i = 0; i < copies / step; ++i, next += step)
          static_cast<void>(linear.add(graph, b, ring[next % copies], c.plan,
                                       step > 1 ? &ring[(next + 1) % copies] : nullptr));
        return backend.submitCommandAsync(graph.dispatches()).wait().gpuSeconds * 1e3 / (copies / step);
      };
      for (Case &c : cases)
        for (double spent = 0; spent < kWarmupSeconds * 1e3;) spent += time(c) * copies;
      for (uint32_t r = 0; r < rounds; ++r)
        for (size_t i = 0; i < cases.size(); ++i) {
          Case &c = cases[r % 2 ? cases.size() - 1 - i : i];
          c.ms.push_back(time(c));
        }
      const DeviceCapabilities &device = backend.capabilities();
      std::printf("%s (GPU family %u, %u cores): %s %sx%u %s, %u weight copies of %.1f MB, median ms of %u rounds:\n",
                  device.deviceName.c_str(), device.appleGpuFamily, device.gpuCoreCount, argv[2], argv[3], K,
                  epilogueName.c_str(), copies, bytes / 1e6, rounds);
      for (size_t i = 0; i < cases.size(); ++i) {
        Case &c = cases[i];
        if (!i || c.line != cases[i - 1].line) std::printf("%s  %s", i ? "\n" : "", c.line.c_str());
        std::sort(c.ms.begin(), c.ms.end());
        std::printf(" | %s%s %.4f", c.label.c_str(), c.chosen ? "*" : "", c.ms[c.ms.size() / 2]);
      }
      std::printf("\n");
    } catch (const std::exception &e) {
      std::cerr << "gguf-projection-benchmark: " << e.what() << '\n';
      return 1;
    }
    return 0;
  }
}

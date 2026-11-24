// Times the prefill FFN's Neural Engine split (ops/AneFfn) at the Omni
// decoders' shape (hidden 2048, intermediate 6144, affine Q4 g64) against
// the GPU alone: for each ANE share (units of 512 of the 6144 intermediate
// channels) and chunk of rows, a command of LAYERS FFN layers (residual +
// SwiGLU over normalized rows), each layer its own weights, as a prefill
// command runs them. Reports the median over REPEAT commands of the GPU
// time (first to last Metal command buffer, so the split's waits for the
// ANE count) per layer, and the split's gain. The FFN is about three
// quarters of a prefill layer's GPU time; the rest (norms, attention, QKV
// and output projections) runs on the GPU alone either way.
//   omni-ane-ffn-bench METALLIB [--units U,U,...] [--rows R,R,...] [--layers L] [--repeat N]
// The weights are random: the GPU's and the ANE's times do not depend on
// their values.

#include "ane/Program.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/AneFfn.hpp"
#include "ops/Linear.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace splash;

namespace {

constexpr uint32_t kHidden = 2048, kIntermediate = 6144, kMaxRows = ops::AneFfn::kMaximumRows;
std::mt19937 rng(11);

metal::MetalBuffer buffer(metal::MetalBackend &backend, uint64_t bytes, const char *label) {
  metal::MetalBuffer result = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
  std::memset(result.contents(), 0, bytes);
  return result;
}

metal::MetalBuffer randomBytes(metal::MetalBackend &backend, uint64_t bytes, const char *label) {
  metal::MetalBuffer result = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, label);
  auto *bits = static_cast<uint8_t *>(result.contents());
  for (uint64_t i = 0; i < bytes; ++i) bits[i] = static_cast<uint8_t>(rng());
  return result;
}

metal::MetalBuffer bf16(metal::MetalBackend &backend, uint64_t count, float low, float high, const char *label) {
  metal::MetalBuffer result = backend.allocateBuffer(count * 2, metal::BufferStorage::Shared, label);
  auto *bits = static_cast<uint16_t *>(result.contents());
  std::uniform_real_distribution<float> value(low, high);
  for (uint64_t i = 0; i < count; ++i) {
    const float f = value(rng);
    uint32_t u;
    std::memcpy(&u, &f, 4);
    bits[i] = static_cast<uint16_t>(u >> 16);
  }
  return result;
}

ops::Projection projection(metal::MetalBackend &backend, uint32_t n, uint32_t k) {
  const uint64_t groups = uint64_t{n} * k / 64;
  return ops::Projection(n, k,
                         ops::AffineWeights{randomBytes(backend, uint64_t{n} * k / 2, "codes"),
                                            bf16(backend, groups, 0.001F, 0.004F, "scales"),
                                            bf16(backend, groups, -0.02F, 0.0F, "biases")});
}

uint64_t sumsBytes(uint32_t rows, uint32_t width) { return uint64_t{(rows + 31) / 32} * (width / 64) * 32 * 4; }

std::vector<uint32_t> list(const std::string &text) {
  std::vector<uint32_t> values;
  std::stringstream stream(text);
  for (std::string item; std::getline(stream, item, ',');) values.push_back(std::stoul(item));
  return values;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: omni-ane-ffn-bench METALLIB [--units U,U,...] [--rows R,R,...] [--layers L] [--repeat N]\n");
    return 2;
  }
  std::vector<uint32_t> unitsList{3, 4, 5, 6, 7}, rowsList{128, 256, 384, 512, 768, 1024, 1536, 2048};
  uint32_t layerCount = 28, repeat = 5;
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string flag = argv[i], value = argv[i + 1];
    if (flag == "--units") unitsList = list(value);
    else if (flag == "--rows") rowsList = list(value);
    else if (flag == "--layers") layerCount = std::stoul(value);
    else if (flag == "--repeat") repeat = std::stoul(value);
  }
  try {
    if (const auto reason = ane::unavailable()) {
      std::printf("Neural Engine unavailable: %s\n", reason->c_str());
      return 1;
    }
    metal::MetalBackend backend(argv[1]);
    const ops::Linear linear(backend.capabilities());
    std::vector<ops::Projection> projections;
    projections.reserve(3 * layerCount);
    std::vector<ops::SwiGluProjections> layers;
    for (uint32_t layer = 0; layer < layerCount; ++layer) {
      projections.push_back(projection(backend, kIntermediate, kHidden));
      projections.push_back(projection(backend, kIntermediate, kHidden));
      projections.push_back(projection(backend, kHidden, kIntermediate));
    }
    for (uint32_t layer = 0; layer < layerCount; ++layer)
      layers.push_back({&projections[3 * layer], &projections[3 * layer + 1], &projections[3 * layer + 2]});
    ops::LinearScratchSize scratch;
    for (const ops::Projection *p : {layers[0].gate, layers[0].down})
      scratch.include(linear.prefillScratchSize(p->shape()));
    const ops::LinearPlan upPlan = linear.prefillPlan(*layers[0].up, kMaxRows, ops::LinearEpilogue::UpWithGate);
    const ops::PrefillFfnBuffers ffn{
        bf16(backend, uint64_t{kMaxRows} * kHidden, -2.0F, 2.0F, "normalized"),
        buffer(backend, sumsBytes(kMaxRows, kHidden), "sums"),
        buffer(backend, std::max(upPlan.gateScratchBytes(), uint64_t{kMaxRows} * kIntermediate * 2), "gate"),
        buffer(backend, uint64_t{kMaxRows} * kIntermediate * 2, "intermediate"),
        buffer(backend, std::max(upPlan.downSumsBytes(), sumsBytes(kMaxRows, kIntermediate)), "down sums"),
        {buffer(backend, std::max<uint64_t>(scratch.input, 16), "scratch input"),
         buffer(backend, std::max<uint64_t>(scratch.sums, 16), "scratch sums"),
         buffer(backend, std::max<uint64_t>(scratch.partials, 16), "scratch partials"),
         buffer(backend, std::max<uint64_t>(scratch.counters, 16), "scratch counters")}};
    const std::array<metal::MetalBuffer, 2> hidden{bf16(backend, uint64_t{kMaxRows} * kHidden, -1, 1, "hidden a"),
                                                    bf16(backend, uint64_t{kMaxRows} * kHidden, -1, 1, "hidden b")};
    std::vector<uint32_t> functionRows;
    for (uint32_t rows : rowsList) functionRows.push_back((rows + 31) / 32 * 32);
    std::sort(functionRows.begin(), functionRows.end());
    functionRows.erase(std::unique(functionRows.begin(), functionRows.end()), functionRows.end());

    const auto gpuAlone = [&](uint32_t rows) {
      metal::CommandGraph graph;
      linear.addPrefillSums(graph, ffn.normalized, ffn.sums, *layers[0].gate, rows);
      for (uint32_t layer = 0; layer < layerCount; ++layer)
        linear.addPrefillSwiGlu(graph, layers[layer], ffn, hidden[layer & 1], hidden[(layer & 1) ^ 1], rows);
      return backend.submitCommandAsync(graph.command()).wait().gpuSeconds;
    };
    std::vector<double> gpuMs(rowsList.size());
    std::printf("GPU alone, %u layers (ms per layer):", layerCount);
    for (size_t r = 0; r < rowsList.size(); ++r) {
      static_cast<void>(gpuAlone(rowsList[r]));
      std::vector<double> times;
      for (uint32_t i = 0; i < repeat; ++i) times.push_back(gpuAlone(rowsList[r]) * 1e3 / layerCount);
      gpuMs[r] = median(times);
      std::printf(" %u rows %.3f", rowsList[r], gpuMs[r]);
    }
    std::printf("\n");
    for (uint32_t units : unitsList) {
      const auto t0 = AwakeClock::now();
      ops::AneFfn split(backend, layers, units, functionRows);
      const double built = millisecondsSince(t0);
      const double error = split.verify(layers, ffn, hidden);
      std::printf("share %u/12 (%u ANE channels): built in %.0f ms, %.2f MiB, verify error %.2f%%\n", units,
                  split.aneChannels(), built, split.allocatedBytes() / 1048576.0, error * 100);
      for (size_t r = 0; r < rowsList.size(); ++r) {
        const uint32_t rows = rowsList[r];
        const auto once = [&] {
          if (!split.begin()) throw std::runtime_error("split stopped: " + split.reason());
          metal::CommandGraph graph;
          linear.addPrefillSums(graph, ffn.normalized, ffn.sums, *layers[0].gate, rows);
          for (uint32_t layer = 0; layer < layerCount; ++layer)
            split.add(graph, layer, ffn, hidden[layer & 1], hidden[(layer & 1) ^ 1], rows);
          const metal::CommandTiming timing = split.commit(graph).wait();
          if (!split.finish()) throw std::runtime_error("split failed: " + split.reason());
          return std::pair{timing.gpuSeconds * 1e3 / layerCount, split.lastAneMilliseconds() / layerCount};
        };
        static_cast<void>(once());
        std::vector<double> times, aneTimes;
        for (uint32_t i = 0; i < repeat; ++i) {
          const auto [gpu, ane] = once();
          times.push_back(gpu);
          aneTimes.push_back(ane);
        }
        const double ms = median(times);
        std::printf("  %4u rows: split %.3f ms/layer (ANE busy %.3f), GPU alone %.3f, gain %+.1f%%\n", rows, ms,
                    median(aneTimes), gpuMs[r], 100.0 * (gpuMs[r] - ms) / gpuMs[r]);
      }
    }
  } catch (const std::exception &error) {
    std::fprintf(stderr, "omni ane ffn bench: %s\n", error.what());
    return 1;
  }
  return 0;
}

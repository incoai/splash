// GPU time of the draft reduction and layout transfer, without model weights.
// The baseline library must predate the row-major reduction's output binding.
#include "metal/CommandGraph.hpp"
#include "metal/abi/DraftAttention.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

using namespace splash::metal;
constexpr uint32_t kRepeat = 256, kWarmup = 4, kRounds = 20;
constexpr uint32_t kRows = SPLASH_DRAFT_QUERY_ROWS;
constexpr uint32_t kHeads = SPLASH_DRAFT_KV_HEADS;
constexpr uint32_t kWidth = SPLASH_DRAFT_ATTENTION_WIDTH;
constexpr uint32_t kGroupRows = SPLASH_DRAFT_ATTENTION_ROWS;
constexpr uint32_t kDimension = SPLASH_DRAFT_HEAD_DIMENSION;
constexpr uint64_t kPartialFloats = uint64_t{kGroupRows} * (kDimension + 2);

class Fixture final {
public:
  Fixture(const char *library, uint32_t lanes, bool fused) : backend_(library) {
    const uint64_t queryWords = uint64_t{lanes} * kRows * kWidth;
    const uint64_t partialFloats = uint64_t{lanes} * kHeads * SPLASH_DRAFT_ATTENTION_SPLITS * kPartialFloats;
    queries_ = backend_.allocateBuffer(queryWords * 2 + partialFloats * 4,
                                       BufferStorage::Shared, "draft stage queries");
    output_ = backend_.allocateBuffer(queryWords * 2, BufferStorage::Shared, "draft stage output");
    std::memset(queries_.contents(), 0, queries_.sizeBytes());
    auto *partials = reinterpret_cast<float *>(static_cast<uint16_t *>(queries_.contents()) + queryWords);
    for (uint64_t part = 0; part < partialFloats / kPartialFloats; ++part) {
      float *data = partials + part * kPartialFloats;
      for (uint32_t index = 0; index < kGroupRows * kDimension; ++index)
        data[index] = float(int((index * 13 + part * 31) % 257) - 128) / 512;
      for (uint32_t row = 0; row < kGroupRows; ++row) {
        data[kGroupRows * kDimension + row] = float(part % 5) * 0.25F;
        data[kGroupRows * (kDimension + 1) + row] = 2.0F + float(row % 7);
      }
    }
    const DraftAttentionBatchParams params{SPLASH_DRAFT_SLIDING_WINDOW, lanes, {}};
    for (uint32_t repeat = 0; repeat < kRepeat; ++repeat) {
      if (fused) {
        graph_.add("draft_attention_bf16_reduce", {queries_, output_}, params, {kHeads, lanes, 1});
      } else {
        graph_.add("draft_attention_bf16_reduce", {queries_}, params, {kHeads, lanes, 1});
        graph_.add("draft_attention_reorder", {queries_, output_},
                   {(uint64_t{kRows} * kWidth + 255) / 256, lanes, 1});
      }
    }
  }

  double run() {
    const double microseconds = backend_.submitCommandAsync(graph_.dispatches()).wait().gpuSeconds * 1e6 / kRepeat;
    if (!std::isfinite(microseconds) || microseconds <= 0)
      throw std::runtime_error("invalid GPU timing");
    return microseconds;
  }

  bool sameOutput(const Fixture &other) const {
    return output_.sizeBytes() == other.output_.sizeBytes() &&
           std::memcmp(output_.contents(), other.output_.contents(), output_.sizeBytes()) == 0;
  }

private:
  MetalBackend backend_;
  MetalBuffer queries_, output_;
  CommandGraph graph_;
};

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return (values[(values.size() - 1) / 2] + values[values.size() / 2]) / 2;
}

struct Round {
  double a1, b1, b2, a2;
};

void measure(const char *baselineLibrary, const char *currentLibrary, uint32_t lanes) {
  Fixture baseline(baselineLibrary, lanes, false), current(currentLibrary, lanes, true);
  std::vector<Round> rounds;
  std::vector<double> oldTimes, newTimes, speedups;
  for (uint32_t round = 0; round < kWarmup + kRounds; ++round) {
    const Round values{baseline.run(), current.run(), current.run(), baseline.run()};
    if (!baseline.sameOutput(current)) throw std::runtime_error("draft stage output bytes differ");
    if (round < kWarmup) continue;
    rounds.push_back(values);
    const double oldTime = (values.a1 + values.a2) / 2, newTime = (values.b1 + values.b2) / 2;
    oldTimes.push_back(oldTime);
    newTimes.push_back(newTime);
    speedups.push_back(oldTime / newTime);
  }
  std::printf("{\"lanes\":%u,\"baseline_us\":%.6f,\"current_us\":%.6f,\"paired_speedup\":%.6f,"
              "\"exact_output_bytes\":true,\"rounds_abba_us\":[", lanes, median(oldTimes), median(newTimes), median(speedups));
  for (size_t index = 0; index < rounds.size(); ++index) {
    const auto &r = rounds[index];
    std::printf("%s[%.6f,%.6f,%.6f,%.6f]", index ? "," : "", r.a1, r.b1, r.b2, r.a2);
  }
  std::printf("]}");
  std::fflush(stdout);
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc != 3)
        throw std::invalid_argument("usage: draft-attention-benchmark BASELINE_METALLIB CURRENT_METALLIB");
      std::printf("{\"repeats_per_command\":%u,\"warmup_rounds\":%u,\"measured_rounds\":%u,\"results\":[",
                  kRepeat, kWarmup, kRounds);
      for (uint32_t lanes = 1; lanes <= SPLASH_MAXIMUM_BATCH_WIDTH; ++lanes) {
        if (lanes != 1) std::printf(",");
        measure(argv[1], argv[2], lanes);
      }
      std::printf("]}\n");
    } catch (const std::exception &error) {
      std::fprintf(stderr, "draft-attention-benchmark: %s\n", error.what());
      return EXIT_FAILURE;
    }
  }
}

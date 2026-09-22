// Qualification of exact GPU-only MoE preprocessing against a scalar oracle.
// No checkpoint is loaded. Root runs this executable serially with GPU work.
#include "flash/FlashMoEBuckets.hpp"
#include "metal/abi/FlashMoEBuckets.h"
#include "../flash/FlashMoEBucketsReference.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace ref = splash::flash::bucket_reference;
using namespace splash::flash;
using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Function> void rejects(Function function, const char *name) {
  try { function(); }
  catch (const std::invalid_argument &) { return; }
  throw std::runtime_error(std::string("invalid bucket graph accepted: ") + name);
}

template <class T>
MetalBuffer copyBuffer(MetalBackend &backend, std::span<const T> data,
                        const char *label) {
  auto buffer = backend.allocateBuffer(data.size_bytes(), BufferStorage::Shared, label);
  std::memcpy(buffer.contents(), data.data(), data.size_bytes());
  return buffer;
}

template <class T>
MetalBuffer copyBuffer(MetalBackend &backend, const std::vector<T> &data,
                        const char *label) {
  return copyBuffer(backend, std::span<const T>(data), label);
}

template <class T>
void compare(const MetalBuffer &buffer, std::span<const T> expected,
              const char *label, uint64_t &elements) {
  require(buffer.sizeBytes() >= expected.size_bytes(),
          std::string(label) + " comparison exceeds buffer");
  const auto *got = static_cast<const T *>(buffer.contents());
  if (std::memcmp(got, expected.data(), expected.size_bytes()) == 0) {
    elements += expected.size();
    return;
  }
  for (uint64_t index = 0; index < expected.size(); ++index) {
    if (got[index] != expected[index])
      throw std::runtime_error(std::string(label) + " differs at " + std::to_string(index) +
                                " got=" + std::to_string(got[index]) +
                                " expected=" + std::to_string(expected[index]));
    ++elements;
  }
}

std::array<MetalBuffer, 8> buffers(const FlashMoEBucketScratch &scratch) {
  return {scratch.counts, scratch.offsets, scratch.routeMap, scratch.canonicalToPacked, scratch.packedInputs,
          scratch.jobOffsets, scratch.jobCount, scratch.tileJobs};
}

void poison(const FlashMoEBucketScratch &scratch) {
  for (const auto &buffer : buffers(scratch))
    std::memset(buffer.contents(), 0xa5, buffer.sizeBytes());
}

std::vector<uint16_t> inputBits(uint32_t rows) {
  std::vector<uint16_t> values(uint64_t{rows} * ref::kWidth);
  for (uint64_t index = 0; index < values.size(); ++index) {
    auto bits = static_cast<uint16_t>((index * 4051 + index / ref::kWidth * 79) & 0xffffU);
    if ((bits & 0x7f80U) == 0x7f80U) bits ^= 0x0080U;
    values[index] = bits;
  }
  values[0] = 0x8000; // Signed zero must survive raw bit copying.
  values[1] = 0x0001;
  values[2] = 0x807f;
  values[3] = 0x7f7f;
  values[4] = 0xff7f;
  return values;
}

enum class Pattern { Spread, Concentrated, MixedInvalid, AllInvalid, Fragmented };

const char *patternName(Pattern pattern) {
  switch (pattern) {
  case Pattern::Spread: return "spread";
  case Pattern::Concentrated: return "concentrated";
  case Pattern::MixedInvalid: return "mixed-invalid";
  case Pattern::AllInvalid: return "all-invalid";
  case Pattern::Fragmented: return "fragmented";
  }
  return "unknown";
}

std::vector<int64_t> makeIDs(uint32_t rows, uint32_t selections, Pattern pattern) {
  const std::array<int64_t, 6> invalid{
      -1, 512, std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max(),
      int64_t{1} << 32, (int64_t{1} << 32) + 511};
  std::vector<int64_t> ids(uint64_t{rows} * selections);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t slot = 0; slot < selections; ++slot) {
      const uint32_t flat = row * selections + slot;
      int64_t value = (row * 73 + slot * 53) % 512;
      switch (pattern) {
      case Pattern::Spread: break;
      case Pattern::Concentrated: value = 511; break;
      case Pattern::MixedInvalid:
        if (flat % 3 == 0) value = invalid[(flat / 3) % invalid.size()];
        break;
      case Pattern::AllInvalid: value = invalid[flat % invalid.size()]; break;
      case Pattern::Fragmented: value = flat < 511 ? flat : 511; break;
      }
      ids[flat] = value;
    }
  }
  return ids;
}

void runFixture(MetalBackend &backend, FlashMoEBucketScratch &scratch,
                 uint32_t rows, uint32_t selections, Pattern pattern,
                 uint32_t tile, uint64_t &elements, bool nonfinite = false,
                 uint32_t stickySeed = 0x80) {
  auto input = inputBits(rows);
  const auto ids = makeIDs(rows, selections, pattern);
  if (nonfinite) {
    // Last row is unrouted in AllInvalid; both NaN payloads and infinities
    // are copied unchanged when it is routed in Spread.
    input.back() = 0x7f81;
    input[input.size() - 2] = 0xffc1;
    input[input.size() - 3] = 0x7f80;
    input[input.size() - 4] = 0xff80;
  }
  const auto expected = ref::pack(input, ids, rows, selections, stickySeed);
  const auto jobs = ref::makeJobs(expected, tile);
  require(moEBucketJobCapacity(rows, selections, tile) == jobs.entries.size(),
          "native job capacity differs from independent reference");
  auto hidden = copyBuffer(backend, input, "bucket fixture hidden");
  auto expertIDs = copyBuffer(backend, ids, "bucket fixture IDs");
  auto diagnostics = copyBuffer(backend, std::vector<uint32_t>{stickySeed},
                                 "bucket fixture sticky diagnostics");
  CommandGraph graph;
  addMoEBucketPack(graph, hidden, expertIDs, scratch, diagnostics, rows, selections);
  addMoEBucketJobs(graph, scratch, diagnostics, rows, tile, selections);
  require(graph.dispatches().size() == 6,
          "pack+jobs is not one six-dispatch graph");
  const auto timing = backend.submitCommand(graph.dispatches());

  compare(scratch.counts, std::span<const uint32_t>(expected.counts), "counts", elements);
  compare(scratch.offsets, std::span<const uint32_t>(expected.offsets), "offsets", elements);
  compare(scratch.routeMap, std::span<const uint32_t>(expected.routeMap), "stable map", elements);
  compare(scratch.canonicalToPacked, std::span<const uint32_t>(expected.canonicalToPacked), "inverse map", elements);
  compare(scratch.packedInputs, std::span<const uint16_t>(expected.inputs), "raw BF16 inputs", elements);
  compare(scratch.jobOffsets, std::span<const uint32_t>(jobs.offsets), "job offsets", elements);
  const auto *gotJobs = static_cast<const FlashMoEBucketJob *>(scratch.tileJobs.contents());
  for (uint32_t index = 0; index < jobs.entries.size(); ++index) {
    if (gotJobs[index].expert != jobs.entries[index].expert ||
        gotJobs[index].row_begin != jobs.entries[index].rowBegin)
      throw std::runtime_error("tile job differs at " + std::to_string(index));
    elements += 2;
  }
  require(*static_cast<const uint32_t *>(scratch.jobCount.contents()) == jobs.count,
          "GPU-generated jobCount differs");
  require(*static_cast<const uint32_t *>(diagnostics.contents()) == expected.diagnostic,
          "sticky diagnostics differ");
  compare(hidden, std::span<const uint16_t>(input), "immutable hidden", elements);
  compare(expertIDs, std::span<const int64_t>(ids), "immutable IDs", elements);
  std::cout << "fixture rows=" << rows << " selections=" << selections
            << " M=" << tile << " pattern=" << patternName(pattern)
            << " nonfinite=" << nonfinite << " routes=" << expected.offsets[512]
            << " jobs=" << jobs.count << " diag=" << expected.diagnostic
            << " gpu-ms=" << timing.gpuSeconds * 1000.0 << '\n';
}

void benchmark(MetalBackend &backend) {
  // Complete six-dispatch preprocessing commands, with normal encoder behavior.
  // This is setup cost only, not an end-to-end model performance measurement.
  for (uint32_t rows : {32U, 128U}) {
    constexpr uint32_t selections = 10;
    auto scratch = allocateMoEBucketScratch(backend, rows, selections);
    auto input = copyBuffer(backend, inputBits(rows), "bucket benchmark hidden");
    auto ids = copyBuffer(backend, makeIDs(rows, selections, Pattern::Spread), "bucket benchmark IDs");
    auto diagnostics = copyBuffer(backend, std::vector<uint32_t>{0}, "bucket benchmark diagnostics");
    for (uint32_t tile : {8U, 16U, 32U}) {
      CommandGraph graph;
      addMoEBucketPack(graph, input, ids, scratch, diagnostics, rows, selections);
      addMoEBucketJobs(graph, scratch, diagnostics, rows, tile, selections);
      for (uint32_t warmup = 0; warmup < 3; ++warmup)
        (void)backend.submitCommand(graph.dispatches());
      std::vector<double> timings;
      for (uint32_t sample = 0; sample < 11; ++sample)
        timings.push_back(backend.submitCommand(graph.dispatches()).gpuSeconds * 1e6);
      std::sort(timings.begin(), timings.end());
      require(*static_cast<const uint32_t *>(diagnostics.contents()) == 0,
              "valid bucket benchmark set diagnostics");
      std::cout << "benchmark full-pack-jobs rows=" << rows << " selections=" << selections
                << " M=" << tile << " dispatches=" << graph.dispatches().size()
                << " samples=" << timings.size() << " median-gpu-us=" << timings[timings.size() / 2]
                << " min-gpu-us=" << timings.front() << " max-gpu-us=" << timings.back() << '\n';
    }
  }
}

void hostRejections(MetalBackend &backend) {
  auto scratch = allocateMoEBucketScratch(backend, 2, 10);
  auto input = copyBuffer(backend, inputBits(2), "host bounds hidden");
  auto ids = copyBuffer(backend, makeIDs(2, 10, Pattern::Spread), "host bounds IDs");
  auto diagnostic = copyBuffer(backend, std::vector<uint32_t>{0x80}, "host bounds diagnostic");
  const auto pack = [&](const FlashMoEBucketScratch &s, const MetalBuffer &h,
                         const MetalBuffer &i, const MetalBuffer &d,
                         uint32_t rows = 2, uint32_t selections = 10) {
    CommandGraph graph;
    addMoEBucketPack(graph, h, i, s, d, rows, selections);
  };
  for (uint32_t rows : {0U, 3U, 8193U})
    rejects([&] { pack(scratch, input, ids, diagnostic, rows); }, "row bound");
  for (uint32_t selections : {0U, 11U})
    rejects([&] { pack(scratch, input, ids, diagnostic, 2, selections); }, "selection bound");
  rejects([&] { (void)allocateMoEBucketScratch(backend, 0, 10); }, "allocator zero rows");
  rejects([&] { (void)allocateMoEBucketScratch(backend, 8193, 10); }, "allocator oversized rows");
  rejects([&] { (void)allocateMoEBucketScratch(backend, 1, 11); }, "allocator oversized selections");
  for (uint32_t tile : {0U, 1U, 4U, 7U, 64U, std::numeric_limits<uint32_t>::max()})
    rejects([&] {
      CommandGraph graph;
      addMoEBucketJobs(graph, scratch, diagnostic, 2, tile, 10);
    }, "unsupported tile");
  rejects([&] { pack(scratch, MetalBuffer{}, ids, diagnostic); }, "null hidden");
  rejects([&] { pack(scratch, input, MetalBuffer{}, diagnostic); }, "null IDs");
  rejects([&] { pack(scratch, input, ids, MetalBuffer{}); }, "null diagnostic");
  rejects([&] { pack(scratch, backend.view(input, 0, input.sizeBytes() - 2), ids, diagnostic); },
          "short hidden");
  rejects([&] { pack(scratch, input, backend.view(ids, 0, ids.sizeBytes() - 8), diagnostic); },
          "short IDs");
  rejects([&] { pack(scratch, input, ids, backend.view(diagnostic, 0, 2)); },
          "short diagnostic");
  rejects([&] { pack(scratch, input, ids, scratch.counts); }, "diagnostic aliases scratch");
  rejects([&] { pack(scratch, input, ids, ids); }, "diagnostic aliases IDs");
  rejects([&] { pack(scratch, scratch.packedInputs, ids, diagnostic); }, "hidden aliases packed output");
  auto alias = scratch;
  alias.counts = scratch.offsets;
  rejects([&] { pack(alias, input, ids, diagnostic); }, "scratch exact alias");
  alias = scratch;
  alias.canonicalToPacked = scratch.routeMap;
  rejects([&] { pack(alias, input, ids, diagnostic); }, "forward/inverse map alias");
  auto malformed = scratch;
  --malformed.routeCapacity;
  rejects([&] { pack(malformed, input, ids, diagnostic); }, "scratch inconsistent routes");
  malformed = scratch;
  --malformed.jobCapacity;
  rejects([&] { pack(malformed, input, ids, diagnostic); }, "scratch insufficient job capacity");
  const std::array<MetalBuffer FlashMoEBucketScratch::*, 8> members{
      &FlashMoEBucketScratch::counts, &FlashMoEBucketScratch::offsets,
      &FlashMoEBucketScratch::routeMap, &FlashMoEBucketScratch::canonicalToPacked,
      &FlashMoEBucketScratch::packedInputs,
      &FlashMoEBucketScratch::jobOffsets, &FlashMoEBucketScratch::jobCount,
      &FlashMoEBucketScratch::tileJobs};
  for (auto member : members) {
    malformed = scratch;
    malformed.*member = backend.view(scratch.*member, 0, (scratch.*member).sizeBytes() - 1);
    rejects([&] { pack(malformed, input, ids, diagnostic); }, "undersized scratch buffer");
  }
  // Differently-sized views sharing storage are aliases too. Equal-start and
  // partial overlaps must fail before any dispatch reaches the driver.
  auto base = backend.allocateBuffer(input.sizeBytes() + scratch.packedInputs.sizeBytes(),
                                     BufferStorage::Shared, "host bounds overlap base");
  const auto overlappingHidden = backend.view(base, 0, input.sizeBytes());
  alias = scratch;
  alias.packedInputs = backend.view(base, 0, scratch.packedInputs.sizeBytes());
  rejects([&] { pack(alias, overlappingHidden, ids, diagnostic); }, "same-start unequal-size overlap");
  alias.packedInputs = backend.view(base, 2, scratch.packedInputs.sizeBytes());
  rejects([&] { pack(alias, overlappingHidden, ids, diagnostic); }, "partial shared-buffer overlap");
}

void malformedShaderFixtures(MetalBackend &backend) {
  auto scratch = allocateMoEBucketScratch(backend, 1, 1);
  auto hidden = copyBuffer(backend, inputBits(1), "malformed hidden");
  auto ids = copyBuffer(backend, std::vector<int64_t>{0}, "malformed IDs");
  auto diagnostic = copyBuffer(backend, std::vector<uint32_t>{0x80}, "malformed diagnostics");
  const auto seed = [&] {
    poison(scratch);
    *static_cast<uint32_t *>(diagnostic.contents()) = 0x80;
  };
  const auto check = [&] {
    require(*static_cast<const uint32_t *>(diagnostic.contents()) == 0x82,
            "malformed shader did not OR parameter diagnostic");
  };
  const FlashMoEBucketParams valid{1, 1, 2560, 512, 1, 0, 0, 0};
  const std::array<std::string, 6> names{
      "flash_moe_bucket_histogram", "flash_moe_bucket_prefix",
      "flash_moe_bucket_stable_map", "flash_moe_bucket_pack",
      "flash_moe_bucket_job_prefix", "flash_moe_bucket_jobs"};
  const std::array<std::vector<MetalBuffer>, 6> bindings{
      std::vector<MetalBuffer>{ids, scratch.counts, scratch.canonicalToPacked, diagnostic},
      std::vector<MetalBuffer>{scratch.counts, scratch.offsets, diagnostic},
      std::vector<MetalBuffer>{ids, scratch.counts, scratch.offsets, scratch.routeMap,
                               scratch.canonicalToPacked, diagnostic},
      std::vector<MetalBuffer>{hidden, scratch.offsets, scratch.routeMap,
                               scratch.packedInputs, diagnostic},
      std::vector<MetalBuffer>{scratch.counts, scratch.offsets, scratch.jobOffsets,
                               scratch.jobCount, diagnostic},
      std::vector<MetalBuffer>{scratch.offsets, scratch.jobOffsets, scratch.jobCount,
                               scratch.tileJobs, diagnostic}};
  // Every shader independently validates base geometry and group width.
  for (uint32_t kernel = 0; kernel < names.size(); ++kernel) {
    for (uint32_t mode = 0; mode < 2; ++mode) {
      seed();
      auto parameters = valid;
      if (kernel >= 4) {
        parameters.tile_rows = 8;
        parameters.job_capacity = moEBucketJobCapacity(1, 1, 8);
      }
      if (mode == 0) parameters.experts = 513;
      CommandGraph malformed;
      malformed.add(names[kernel], bindings[kernel], parameters, {1, 1, 1},
                      {mode == 1 ? 128U : 256U, 1, 1});
      (void)backend.submitCommand(malformed.dispatches());
      check();
    }
  }
  auto invalid = valid;
  invalid.width = 2559;
  seed();
  CommandGraph histogram;
  histogram.add("flash_moe_bucket_histogram", {ids, scratch.counts, scratch.canonicalToPacked, diagnostic},
                 invalid, {512, 1, 1}, {256, 1, 1});
  (void)backend.submitCommand(histogram.dispatches());
  check();
  invalid = valid;
  invalid.routes = 2;
  seed();
  CommandGraph gather;
  gather.add("flash_moe_bucket_pack", {hidden, scratch.offsets, scratch.routeMap,
                                        scratch.packedInputs, diagnostic},
              invalid, {1, 1, 1}, {256, 1, 1});
  (void)backend.submitCommand(gather.dispatches());
  check();

  // Bypass host checks to supply corrupted counts. The prefix must empty every
  // expert range; jobs subsequently become invalid sentinels with count zero.
  seed();
  std::fill_n(static_cast<uint32_t *>(scratch.counts.contents()), 512, uint32_t{0});
  static_cast<uint32_t *>(scratch.counts.contents())[511] = std::numeric_limits<uint32_t>::max();
  CommandGraph corruptPrefix;
  corruptPrefix.add("flash_moe_bucket_prefix", {scratch.counts, scratch.offsets, diagnostic},
                     valid, {1, 1, 1}, {256, 1, 1});
  (void)backend.submitCommand(corruptPrefix.dispatches());
  check();
  require(std::all_of(static_cast<const uint32_t *>(scratch.offsets.contents()),
                      static_cast<const uint32_t *>(scratch.offsets.contents()) + 513,
                      [](uint32_t value) { return value == 0; }),
          "corrupt count prefix left live expert ranges");
  CommandGraph corruptJobs;
  addMoEBucketJobs(corruptJobs, scratch, diagnostic, 1, 8, 1);
  (void)backend.submitCommand(corruptJobs.dispatches());
  check();
  require(*static_cast<const uint32_t *>(scratch.jobCount.contents()) == 0,
          "corrupt count buffer generated jobs");
  const auto *jobs = static_cast<const FlashMoEBucketJob *>(scratch.tileJobs.contents());
  for (uint32_t index = 0; index < moEBucketJobCapacity(1, 1, 8); ++index)
    require(jobs[index].expert == ref::kInvalid && jobs[index].row_begin == 0,
            "corrupt count fixture did not sentinel-fill jobs");

  // A count/offset disagreement in a middle expert must also fail closed.
  seed();
  std::fill_n(static_cast<uint32_t *>(scratch.counts.contents()), 512, uint32_t{0});
  std::fill_n(static_cast<uint32_t *>(scratch.offsets.contents()), 513, uint32_t{0});
  static_cast<uint32_t *>(scratch.offsets.contents())[255] = 1;
  CommandGraph disagree;
  addMoEBucketJobs(disagree, scratch, diagnostic, 1, 16, 1);
  (void)backend.submitCommand(disagree.dispatches());
  check();
  require(*static_cast<const uint32_t *>(scratch.jobCount.contents()) == 0,
          "count/offset disagreement generated jobs");
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      require(argc == 2 || (argc == 3 && std::string(argv[2]) == "--benchmark"),
              "usage: flash-moe-buckets-metal-test /absolute/private/buckets.metallib [--benchmark]");
      MetalBackend backend(argv[1]);
      uint64_t elements = 0;
      hostRejections(backend);
      for (uint32_t rows : {1U, 2U, 32U, 128U, 2048U}) {
        for (uint32_t selections : {1U, 10U}) {
          auto scratch = allocateMoEBucketScratch(backend, rows, selections);
          if (rows == 2048) {
            poison(scratch);
            runFixture(backend, scratch, rows, selections, Pattern::Spread, 8, elements);
            runFixture(backend, scratch, rows, selections, Pattern::Concentrated, 16, elements);
            runFixture(backend, scratch, rows, selections, Pattern::MixedInvalid, 32, elements);
            runFixture(backend, scratch, rows, selections, Pattern::AllInvalid, 8, elements);
            continue;
          }
          for (uint32_t tile : {8U, 16U, 32U}) {
            poison(scratch);
            runFixture(backend, scratch, rows, selections, Pattern::Spread, tile, elements);
            // Dense -> sparse/all-invalid reuses existing output without clearing.
            runFixture(backend, scratch, rows, selections, Pattern::Concentrated, tile, elements);
            runFixture(backend, scratch, rows, selections, Pattern::MixedInvalid, tile, elements);
            runFixture(backend, scratch, rows, selections, Pattern::AllInvalid, tile, elements);
          }
        }
      }
      auto scratch = allocateMoEBucketScratch(backend, 2048, 10);
      for (uint32_t tile : {8U, 16U, 32U}) {
        poison(scratch);
        runFixture(backend, scratch, 2048, 10, Pattern::Fragmented, tile, elements);
        // Smaller logical geometry reuses the large allocation.
        runFixture(backend, scratch, 2, 10, Pattern::AllInvalid, tile, elements, true);
        runFixture(backend, scratch, 2, 10, Pattern::Spread, tile, elements, true, 0x81);
        runFixture(backend, scratch, 1, 1, Pattern::Spread, tile, elements, false, 0x85);
      }
      malformedShaderFixtures(backend);
      if (argc == 3) benchmark(backend);
      std::cout << "PASS exact Flash MoE bucket GPU oracle elements=" << elements
                << " stable maps, BF16 bit copies, sticky diagnostics, bounded jobs\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "FAIL " << error.what() << '\n';
      return 1;
    }
  }
}

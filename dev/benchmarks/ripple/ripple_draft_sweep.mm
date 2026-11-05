// Step-1 prototype sweep for the Ripple draft kernels: drives the variant
// instantiations of dev/benchmarks/ripple/ripple_draft_variant.metal beside
// the production draft attention and convolution, checking each against a
// direct CPU reference (the draft_attention_metal_test's, generalized to a
// group's row count) and timing each dispatch as its own command. The
// shipped-geometry instantiation must time and check like the production
// kernels it mirrors; its output digest matching the production kernel's
// bit for bit is the harness's own validation.
//
// usage: ripple-draft-sweep VARIANT_METALLIB [--baseline METALLIB]
//                            [--lanes 1,4] [--lengths 512,1023,...]
//                            [--repeat N] [--warmup-seconds S]
//                            [--no-correctness]
#include "../DispatchReplay.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/DraftAttention.h"
#include "metal/abi/ExecutionGeometry.h"
#include "tests/engine/LinearNumerics.hpp"
#include "tests/engine/TestChecks.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::metal::CommandGraph;
using splash::test::bf16ToFloat;
using splash::test::floatToBf16;

constexpr uint32_t kRows = 8;
constexpr uint32_t kKvHeads = 8;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kMaxLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr float kScale = 0.08838834765F;

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
  float unit() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>((state_ >> 40) & 0xFFFFFF) / 8388608.0F - 1.0F;
  }

private:
  uint64_t state_;
};

MetalBuffer randomBfloat(MetalBackend &backend, uint64_t count, Random &random,
                         const char *label) {
  MetalBuffer buffer =
      backend.allocateBuffer(count * sizeof(uint16_t), BufferStorage::Shared,
                             label);
  auto *values = static_cast<uint16_t *>(buffer.contents());
  for (uint64_t index = 0; index < count; ++index)
    values[index] = floatToBf16(random.unit());
  return buffer;
}

struct AttentionCase final {
  std::string name;
  std::string split;
  std::string reduce;
  uint32_t queryHeads;
  uint32_t headsPerGroup;
  uint32_t splits;
  uint32_t window;
  // Non-empty when the case runs the production metallib's kernels.
  std::string productionSplit;
  std::string productionReduce;
};

struct ConvCase final {
  std::string name;
  std::string kernel;
  uint32_t taps;
  uint32_t hidden;
};

struct AttentionBuffers final {
  MetalBuffer queries;
  std::vector<MetalBuffer> keys;
  std::vector<MetalBuffer> values;
  MetalBuffer queryKeys;
  MetalBuffer queryValues;
};

uint64_t groupRows(const AttentionCase &c) {
  return uint64_t{c.headsPerGroup} * kRows;
}

// The kernel writes its attention rows into the grouped-queries tensor at
// [lane][head][head_group][rows][dim], with the split partials of every
// (lane, head, head group) behind every lane's rows as floats.
uint64_t attentionRowsBytes(const AttentionCase &c, uint32_t lanes) {
  return uint64_t{lanes} * kRows * c.queryHeads * kHeadDim * 2;
}

uint64_t partialBytes(const AttentionCase &c, uint32_t lanes) {
  const uint64_t perGroup = (groupRows(c) * (kHeadDim + 2)) * sizeof(float);
  return uint64_t{lanes} * kKvHeads *
         (c.queryHeads / kKvHeads / c.headsPerGroup) * c.splits * perGroup;
}

uint64_t groupedBytes(const AttentionCase &c, uint32_t lanes) {
  return attentionRowsBytes(c, lanes) + partialBytes(c, lanes);
}

AttentionBuffers makeAttentionBuffers(MetalBackend &backend,
                                      const AttentionCase &c, uint32_t lanes,
                                      uint64_t seed) {
  Random random(seed);
  AttentionBuffers buffers;
  buffers.queries = randomBfloat(backend, groupedBytes(c, lanes) / 2, random,
                                 "grouped queries");
  const uint64_t ringElements = uint64_t{kKvHeads} * c.window * kHeadDim;
  const uint64_t tailElements = uint64_t{128} * kHeadDim;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    buffers.keys.push_back(
        randomBfloat(backend, ringElements + tailElements, random, "draft keys"));
    buffers.values.push_back(randomBfloat(backend, ringElements + tailElements,
                                          random, "draft values"));
    for (const MetalBuffer &tensor :
         {buffers.keys.back(), buffers.values.back()})
      std::fill_n(static_cast<uint16_t *>(tensor.contents()) + ringElements,
                  tailElements, uint16_t{0x7FC0});
  }
  // The current-rows tile over-reads one head block of query keys (its
  // sixteen padded columns span the next block), which stays finite here.
  buffers.queryKeys = randomBfloat(
      backend,
      uint64_t{lanes} * kKvHeads * kRows * kHeadDim + kRows * kHeadDim,
      random, "draft query keys");
  buffers.queryValues = randomBfloat(
      backend, uint64_t{lanes} * kKvHeads * kHeadDim * kRows, random,
      "draft query values");
  return buffers;
}

CommandGraph attentionGraph(const AttentionCase &c,
                            const AttentionBuffers &buffers, uint32_t lanes,
                            std::span<const uint32_t> cacheLengths) {
  DraftAttentionBatchParams params{c.window, lanes, {}};
  std::copy(cacheLengths.begin(), cacheLengths.end(),
            std::begin(params.cache_length));
  CommandGraph graph;
  std::vector<MetalBuffer> bindings{buffers.queries};
  // The kernel binds keys0..keys3 then values0..values3 (buffers 1..8).
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane)
    bindings.push_back(buffers.keys[lane % lanes]);
  for (uint32_t lane = 0; lane < kMaxLanes; ++lane)
    bindings.push_back(buffers.values[lane % lanes]);
  bindings.push_back(buffers.queryKeys);
  bindings.push_back(buffers.queryValues);
  const uint32_t headGroups = c.queryHeads / kKvHeads / c.headsPerGroup;
  const std::string &split = c.productionSplit.empty() ? c.split : c.productionSplit;
  const std::string &reduce =
      c.productionReduce.empty() ? c.reduce : c.productionReduce;
  graph.add(split, bindings, params, {kKvHeads * headGroups, lanes, c.splits});
  graph.add(reduce, {buffers.queries}, params, {kKvHeads * headGroups, lanes, 1});
  return graph;
}

// The test's reference, generalized: `headsPerGroup` query heads' rows of one
// KV head, whose ring the caller's pointers address.
void referenceRows(const uint16_t *queries, const uint16_t *keys,
                   const uint16_t *values, const uint16_t *queryKeys,
                   const uint16_t *queryValues, uint32_t cacheLength,
                   uint32_t window, uint32_t headsPerGroup,
                   std::vector<float> &output) {
  const uint32_t rows = headsPerGroup * kRows;
  const uint32_t commonStart =
      cacheLength >= window - 1 ? cacheLength - (window - 1) : 0;
  const uint32_t oldCount = cacheLength - commonStart;
  std::vector<double> scores(oldCount + kRows);
  std::vector<double> accumulated(kHeadDim);
  for (uint32_t row = 0; row < rows; ++row) {
    const uint32_t proposal = row % kRows;
    const uint32_t queryPosition = cacheLength + proposal;
    const uint32_t rowStart =
        queryPosition >= window - 1 ? queryPosition - (window - 1) : 0;
    const uint32_t hiddenPrefix = rowStart - commonStart;
    const uint16_t *query = queries + uint64_t{row} * kHeadDim;
    double best = -INFINITY;
    for (uint32_t key = 0; key < oldCount + kRows; ++key) {
      double score = -INFINITY;
      if (key >= oldCount) {
        const uint32_t current = key - oldCount;
        double dot = 0.0;
        for (uint32_t d = 0; d < kHeadDim; ++d) {
          dot += double(bf16ToFloat(query[d])) *
                 bf16ToFloat(queryKeys[current * kHeadDim + d]);
        }
        score = dot * kScale;
      } else if (key >= hiddenPrefix) {
        const uint32_t slot = (commonStart + key) % window;
        double dot = 0.0;
        for (uint32_t d = 0; d < kHeadDim; ++d) {
          dot += double(bf16ToFloat(query[d])) *
                 bf16ToFloat(keys[uint64_t{slot} * kHeadDim + d]);
        }
        score = dot * kScale;
      }
      scores[key] = score;
      best = std::max(best, score);
    }
    double sum = 0.0;
    std::fill(accumulated.begin(), accumulated.end(), 0.0);
    for (uint32_t key = 0; key < oldCount + kRows; ++key) {
      if (scores[key] == -INFINITY)
        continue;
      const double probability = std::exp(scores[key] - best);
      sum += probability;
      for (uint32_t d = 0; d < kHeadDim; ++d) {
        const float value =
            key >= oldCount
                ? bf16ToFloat(queryValues[uint64_t{d} * kRows + key - oldCount])
                : bf16ToFloat(
                      values[uint64_t{d} * window +
                             (commonStart + key) % window]);
        accumulated[d] += probability * value;
      }
    }
    for (uint32_t d = 0; d < kHeadDim; ++d)
      output[uint64_t{row} * kHeadDim + d] =
          static_cast<float>(accumulated[d] / sum);
  }
}

void checkCase(const AttentionCase &c, const AttentionBuffers &buffers,
               const std::vector<uint16_t> &queryInput, uint32_t lanes,
               std::span<const uint32_t> cacheLengths) {
  const uint32_t headGroups = c.queryHeads / kKvHeads / c.headsPerGroup;
  const auto *output = static_cast<const uint16_t *>(buffers.queries.contents());
  std::vector<float> reference(groupRows(c) * kHeadDim);
  const bool dump = getenv("SPLASH_DEBUG_DUMP") != nullptr;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    for (const uint32_t head : {0U, kKvHeads - 1}) {
      for (uint32_t headGroup = 0; headGroup < headGroups; ++headGroup) {
        const uint64_t rowsBase =
            (uint64_t{lane} * kRows * c.queryHeads +
             uint64_t{head} * (c.queryHeads / kKvHeads) * kRows +
             uint64_t{headGroup} * groupRows(c)) *
            kHeadDim;
        // The kernel's run overwrote the rows with the attention output, so
        // the reference's queries come from the pre-run snapshot.
        referenceRows(
            queryInput.data() + rowsBase,
            static_cast<const uint16_t *>(buffers.keys[lane].contents()) +
                uint64_t{head} * c.window * kHeadDim,
            static_cast<const uint16_t *>(buffers.values[lane].contents()) +
                uint64_t{head} * c.window * kHeadDim,
            static_cast<const uint16_t *>(buffers.queryKeys.contents()) +
                (uint64_t{lane} * kKvHeads + head) * kRows * kHeadDim,
            static_cast<const uint16_t *>(buffers.queryValues.contents()) +
                (uint64_t{lane} * kKvHeads + head) * kHeadDim * kRows,
            cacheLengths[lane], c.window, c.headsPerGroup, reference);
        if (dump && lane == 0 && head == 0 && headGroup == 0) {
          const auto *keys = static_cast<const uint16_t *>(
              buffers.keys[0].contents());
          const auto *values = static_cast<const uint16_t *>(
              buffers.values[0].contents());
          const auto *queryKeys = static_cast<const uint16_t *>(
              buffers.queryKeys.contents());
          const auto *queryValues = static_cast<const uint16_t *>(
              buffers.queryValues.contents());
          std::cerr << "dump cache_length=" << cacheLengths[0] << "\n";
          for (uint32_t row = 0; row < 4; ++row) {
            std::cerr << "q[" << row << "][0..3] =";
            for (uint32_t d = 0; d < 4; ++d)
              std::cerr << " " << bf16ToFloat(queryInput[rowsBase +
                                                          uint64_t{row} *
                                                              kHeadDim +
                                                          d]);
            std::cerr << "  out[" << row << "][0..1] ="
                      << bf16ToFloat(output[rowsBase + uint64_t{row} * kHeadDim])
                      << " "
                      << bf16ToFloat(output[rowsBase + uint64_t{row} * kHeadDim +
                                           1])
                      << "  ref[" << row << "][0..1] = "
                      << reference[uint64_t{row} * kHeadDim] << " "
                      << reference[uint64_t{row} * kHeadDim + 1] << "\n";
          }
          std::cerr << "row map (out[r][0] vs ref[r][0]):";
          for (uint32_t row = 0; row < groupRows(c); ++row)
            std::cerr << " [" << row << "]"
                      << bf16ToFloat(output[rowsBase + uint64_t{row} * kHeadDim])
                      << "/" << reference[uint64_t{row} * kHeadDim];
          std::cerr << "\n";
          std::cerr << "qk[0..7][0] =";
          for (uint32_t row = 0; row < 8; ++row)
            std::cerr << " " << bf16ToFloat(queryKeys[row * kHeadDim]);
          std::cerr << "\nqv[0][0..7] =";
          for (uint32_t row = 0; row < 8; ++row)
            std::cerr << " " << bf16ToFloat(queryValues[row]);
          std::cerr << "\nring k[0][0..2] =";
          for (uint32_t d = 0; d < 3; ++d)
            std::cerr << " " << bf16ToFloat(keys[d]);
          std::cerr << "\nring v[0][0..2] =";
          for (uint32_t d = 0; d < 3; ++d)
            std::cerr << " " << bf16ToFloat(values[d * c.window]);
          std::cerr << "\n";
        }
        for (uint64_t index = 0; index < reference.size(); ++index) {
          const float actual = bf16ToFloat(output[rowsBase + index]);
          const float expected = reference[index];
          if (!std::isfinite(actual) ||
              std::fabs(actual - expected) >
                  0.02F + 0.02F * std::fabs(expected)) {
            std::cerr << c.name << " lane " << lane << " head " << head
                      << " head_group " << headGroup << " length "
                      << cacheLengths[lane] << " row " << index / kHeadDim
                      << " dim " << index % kHeadDim << ": " << actual
                      << " vs " << expected << '\n';
            throw std::runtime_error(c.name + " diverged from reference");
          }
        }
      }
    }
  }
}

uint64_t digest(const void *data, uint64_t bytes) {
  auto *values = static_cast<const uint16_t *>(data);
  uint64_t hash = 0xcbf29ce484222325ULL;
  for (uint64_t index = 0; index < bytes / 2; ++index) {
    hash ^= values[index];
    hash *= 0x100000001b3ULL;
  }
  return hash;
}

double median(const std::vector<double> &values) {
  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  return sorted[sorted.size() / 2];
}

void runAttention(MetalBackend &variant, MetalBackend *production,
                  const AttentionCase &c, const std::vector<uint32_t> &lanes,
                  const std::vector<uint32_t> &lengths, uint32_t repeat,
                  double warmupSeconds, bool check) {
  MetalBackend &backend = c.productionSplit.empty() ? variant : *production;
  for (uint32_t laneCount : lanes) {
    std::vector<uint32_t> cacheLengths(laneCount);
    for (uint32_t length : lengths) {
      std::fill(cacheLengths.begin(), cacheLengths.end(), length);
      AttentionBuffers buffers =
          makeAttentionBuffers(backend, c, laneCount, 0x5eed0000ULL + length);
      std::vector<uint16_t> queryInput(
          static_cast<const uint16_t *>(buffers.queries.contents()),
          static_cast<const uint16_t *>(buffers.queries.contents()) +
              attentionRowsBytes(c, laneCount) / 2);
      uint64_t outputDigest = 0;
      if (check) {
        CommandGraph graph =
            attentionGraph(c, buffers, laneCount, cacheLengths);
        static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
        checkCase(c, buffers, queryInput, laneCount, cacheLengths);
        outputDigest =
            digest(buffers.queries.contents(), attentionRowsBytes(c, laneCount));
      }
      CommandGraph graph = attentionGraph(c, buffers, laneCount, cacheLengths);
      const auto dispatches = graph.dispatches();
      const auto warmupUntil =
          std::chrono::steady_clock::now() +
          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
              std::chrono::duration<double>(warmupSeconds));
      while (std::chrono::steady_clock::now() < warmupUntil)
        static_cast<void>(
            splash::benchmark::replayDispatches(backend, dispatches));
      std::map<std::string, std::vector<double>> samples;
      for (uint32_t iteration = 0; iteration < repeat; ++iteration)
        for (auto &[pipeline, value] :
             splash::benchmark::replayDispatches(backend, dispatches))
          samples[pipeline].push_back(value);
      const std::string &split =
          c.productionSplit.empty() ? c.split : c.productionSplit;
      const std::string &reduce =
          c.productionReduce.empty() ? c.reduce : c.productionReduce;
      const double splitUs = median(samples[split]) * 1e6 / laneCount;
      const double reduceUs = median(samples[reduce]) * 1e6 / laneCount;
      const double splitFloorUs =
          *std::min_element(samples[split].begin(), samples[split].end()) *
          1e6 / laneCount;
      std::cout << "{\"case\":\"" << c.name << "\",\"lanes\":" << laneCount
                << ",\"length\":" << length << ",\"heads\":" << c.queryHeads
                << ",\"splits\":" << c.splits << ",\"window\":" << c.window
                << ",\"split_us\":" << splitUs
                << ",\"split_floor_us\":" << splitFloorUs
                << ",\"reduce_us\":" << reduceUs;
      if (check)
        std::cout << ",\"digest\":\"" << std::hex << std::setw(16)
                  << std::setfill('0') << outputDigest << std::dec << "\"";
      std::cout << "}\n";
      std::cerr << c.name << " lanes=" << laneCount << " length=" << length
                << " split=" << splitUs << " us/lane reduce=" << reduceUs
                << " us/lane";
      if (check)
        std::cerr << " digest=" << std::hex << std::setw(16)
                  << std::setfill('0') << outputDigest << std::dec;
      std::cerr << "\n";
    }
  }
}

void runConv(MetalBackend &backend, const ConvCase &c, uint32_t lanes,
             uint32_t repeat, double warmupSeconds, bool check) {
  const uint64_t dynamicWidth =
      2ULL * c.taps * (c.hidden / SPLASH_DRAFT_CONVOLUTION_GROUP);
  const uint64_t blockElements = uint64_t{kRows} * c.hidden;
  Random random(0x5eedc0deULL);
  MetalBuffer input =
      randomBfloat(backend, lanes * blockElements, random, "conv input");
  MetalBuffer residual =
      randomBfloat(backend, lanes * blockElements, random, "conv residual");
  MetalBuffer dynamic = randomBfloat(backend, lanes * kRows * dynamicWidth,
                                     random, "conv dynamic");
  MetalBuffer weights =
      randomBfloat(backend, 2ULL * c.taps * c.hidden, random, "conv weights");
  MetalBuffer prepared = backend.allocateBuffer(
      lanes * blockElements * 2, BufferStorage::Shared, "conv prepared");
  MetalBuffer output = backend.allocateBuffer(
      lanes * blockElements * 2, BufferStorage::Shared, "conv output");
  DraftConvBatchParams prepare{0, c.hidden};
  DraftConvBatchParams finish{1, c.hidden};
  const uint32_t groups = static_cast<uint32_t>((blockElements + 255) / 256);
  CommandGraph graph;
  graph.add(c.kernel, {input, dynamic, weights, residual, prepared}, prepare,
            {groups, lanes, 1});
  graph.add(c.kernel, {prepared, dynamic, weights, residual, output}, finish,
            {groups, lanes, 1});
  if (check) {
    static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
    // The chain the two stages compute: prepare convolves `input`, finish
    // convolves the prepared rows as the kernel rounded them to bf16 and adds
    // the residual.
    const std::vector<uint16_t> preparedSnapshot(
        static_cast<const uint16_t *>(prepared.contents()),
        static_cast<const uint16_t *>(prepared.contents()) +
            lanes * blockElements);
    const auto *inputValues = static_cast<const uint16_t *>(input.contents());
    const auto *dynamicValues =
        static_cast<const uint16_t *>(dynamic.contents());
    const auto *weightValues =
        static_cast<const uint16_t *>(weights.contents());
    const uint32_t convGroups = c.hidden / SPLASH_DRAFT_CONVOLUTION_GROUP;
    for (uint32_t stage = 0; stage < 2; ++stage) {
      const uint16_t *stageInput =
          stage == 0 ? inputValues : preparedSnapshot.data();
      const uint16_t *stageOutput =
          stage == 0
              ? static_cast<const uint16_t *>(prepared.contents())
              : static_cast<const uint16_t *>(output.contents());
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        for (uint32_t row = 0; row < kRows; ++row) {
          for (uint32_t channel = 0; channel < c.hidden; ++channel) {
            const uint64_t element =
                (uint64_t{lane} * kRows + row) * c.hidden + channel;
            const uint32_t convGroup =
                channel / SPLASH_DRAFT_CONVOLUTION_GROUP;
            double value = 0.0;
            for (uint32_t tap = 0; tap < c.taps; ++tap) {
              if (row < tap)
                continue;
              const uint64_t source =
                  (uint64_t{lane} * kRows + row - tap) * c.hidden + channel;
              value += bf16ToFloat(stageInput[source]) *
                       (bf16ToFloat(weightValues[(stage * c.taps + tap) *
                                                     c.hidden +
                                                 channel]) +
                        bf16ToFloat(dynamicValues[(uint64_t{lane} * kRows +
                                                   row) *
                                                      dynamicWidth +
                                                  (stage * c.taps + tap) *
                                                      convGroups +
                                                  convGroup]));
            }
            if (stage == 1)
              value += bf16ToFloat(
                  static_cast<const uint16_t *>(residual.contents())[element]);
            const float actual = bf16ToFloat(stageOutput[element]);
            if (!std::isfinite(actual) ||
                std::fabs(actual - static_cast<float>(value)) >
                    0.02F + 0.02F * std::fabs(static_cast<float>(value))) {
              std::cerr << c.name << " stage " << stage << " lane " << lane
                        << " row " << row << " channel " << channel << ": "
                        << actual << " vs " << value << '\n';
              throw std::runtime_error(c.name + " diverged from reference");
            }
          }
        }
      }
    }
  }
  const auto dispatches = graph.dispatches();
  const auto warmupUntil =
      std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(warmupSeconds));
  while (std::chrono::steady_clock::now() < warmupUntil)
    static_cast<void>(splash::benchmark::replayDispatches(backend, dispatches));
  std::map<std::string, std::vector<double>> samples;
  for (uint32_t iteration = 0; iteration < repeat; ++iteration)
    for (auto &[pipeline, value] :
         splash::benchmark::replayDispatches(backend, dispatches))
      samples[pipeline].push_back(value);
  // Both stages share the pipeline; one iteration runs them back to back, so
  // a stage's sample is half of each consecutive pair's sum.
  std::vector<double> stageSamples;
  for (size_t index = 0; index + 1 < samples[c.kernel].size(); index += 2)
    stageSamples.push_back(
        (samples[c.kernel][index] + samples[c.kernel][index + 1]) / 2);
  const double stageUs = median(stageSamples) * 1e6 / lanes;
  const double stageFloorUs =
      *std::min_element(stageSamples.begin(), stageSamples.end()) * 1e6 /
      lanes;
  std::cout << "{\"case\":\"" << c.name << "\",\"taps\":" << c.taps
            << ",\"hidden\":" << c.hidden << ",\"lanes\":" << lanes
            << ",\"stage_us\":" << stageUs
            << ",\"stage_floor_us\":" << stageFloorUs << "}\n";
  std::cerr << c.name << " lanes=" << lanes << " stage=" << stageUs
            << " (floor " << stageFloorUs << ") us/lane/stage\n";
}

std::vector<uint32_t> parseList(const std::string &value) {
  std::vector<uint32_t> result;
  size_t start = 0;
  while (true) {
    const size_t comma = value.find(',', start);
    const std::string item =
        value.substr(start,
                     comma == std::string::npos ? std::string::npos
                                                : comma - start);
    if (item.empty())
      throw std::invalid_argument("list has an empty item");
    result.push_back(static_cast<uint32_t>(std::stoul(item)));
    if (comma == std::string::npos)
      break;
    start = comma + 1;
  }
  return result;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2)
      throw std::invalid_argument(
          "usage: ripple-draft-sweep VARIANT_METALLIB [--baseline METALLIB] "
          "[--lanes 1,4] [--lengths 512,1023,...] [--repeat N] "
          "[--warmup-seconds S] [--no-correctness]");
    std::string baselineLibrary;
    std::vector<uint32_t> lanes{1, 4};
    std::vector<uint32_t> lengths{0, 1, 512, 1023, 2047, 2048, 4096};
    uint32_t repeat = 30;
    double warmupSeconds = 0.5;
    bool check = true;
    std::vector<std::string> caseFilter;
    for (int index = 2; index < argc; ++index) {
      const std::string option = argv[index];
      const std::string value = index + 1 < argc ? argv[index + 1] : "";
      if (option == "--baseline") {
        baselineLibrary = value;
        ++index;
      } else if (option == "--lanes") {
        lanes = parseList(value);
        ++index;
      } else if (option == "--lengths") {
        lengths = parseList(value);
        ++index;
      } else if (option == "--repeat") {
        repeat = static_cast<uint32_t>(std::stoul(value));
        ++index;
      } else if (option == "--warmup-seconds") {
        warmupSeconds = std::stod(value);
        ++index;
      } else if (option == "--cases") {
        for (const auto &item : std::views::split(value, ','))
          caseFilter.emplace_back(item.begin(), item.end());
        ++index;
      } else if (option == "--no-correctness") {
        check = false;
      } else
        throw std::invalid_argument("unknown option " + option);
    }
    for (uint32_t laneCount : lanes)
      if (!laneCount || laneCount > kMaxLanes)
        throw std::invalid_argument("--lanes takes 1 to " +
                                    std::to_string(kMaxLanes));
    MetalBackend variant(argv[1]);
    std::unique_ptr<MetalBackend> production;
    if (!baselineLibrary.empty())
      production = std::make_unique<MetalBackend>(baselineLibrary);
    std::cerr << "device " << variant.capabilities().deviceName << "\n";
    std::cout << "{\"device\":\"" << variant.capabilities().deviceName
              << "\",\"cases\":[";
    const AttentionCase cases[]{
        AttentionCase{"baseline-variant", "ripple_split_q32_h4_s4_w2048",
                      "ripple_reduce_q32_h4_s4", 32, 4, 4, 2048, "", ""},
        AttentionCase{"baseline-production", "", "", 32, 4, 4, 2048,
                      "draft_attention_bf16_split",
                      "draft_attention_bf16_reduce"},
        AttentionCase{"ripple-h5-s4-w2048", "ripple_split_q80_h5_s4_w2048",
                      "ripple_reduce_q80_h5_s4", 80, 5, 4, 2048, "", ""},
        AttentionCase{"ripple-h5-s8-w2048", "ripple_split_q80_h5_s8_w2048",
                      "ripple_reduce_q80_h5_s8", 80, 5, 8, 2048, "", ""},
        AttentionCase{"ripple-h5-s4-w4096", "ripple_split_q80_h5_s4_w4096",
                      "ripple_reduce_q80_h5_s4_w4096", 80, 5, 4, 4096, "", ""},
        AttentionCase{"ripple-h5-s8-w4096", "ripple_split_q80_h5_s8_w4096",
                      "ripple_reduce_q80_h5_s8_w4096", 80, 5, 8, 4096, "", ""},
        AttentionCase{"ripple-h10-s4-w2048", "ripple_split_q80_h10_s4_w2048",
                      "ripple_reduce_q80_h10_s4", 80, 10, 4, 2048, "", ""},
    };
    bool first = true;
    for (const AttentionCase &c : cases) {
      if (!c.productionSplit.empty() && !production)
        continue;
      if (!caseFilter.empty() &&
          std::ranges::find(caseFilter, c.name) == caseFilter.end())
        continue;
      std::cout << (first ? "" : ",");
      first = false;
      try {
        runAttention(variant, production.get(), c, lanes, lengths, repeat,
                     warmupSeconds, check);
      } catch (const std::exception &error) {
        std::cout << "{\"case\":\"" << c.name << "\",\"failed\":true}\n";
        std::cerr << c.name << " failed: " << error.what() << '\n';
      }
    }
    const ConvCase convCases[]{
        ConvCase{"conv-t2", "ripple_conv_t2", 2, 5120},
        ConvCase{"conv-t4", "ripple_conv_t4", 4, 5120},
    };
    for (const ConvCase &c : convCases) {
      std::cout << (first ? "" : ",");
      first = false;
      runConv(variant, c, lanes.back(), repeat, warmupSeconds, check);
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ripple-draft-sweep: " << error.what() << '\n';
    return 70;
  }
}

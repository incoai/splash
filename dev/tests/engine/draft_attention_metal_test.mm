#include "metal/abi/DraftAttention.h"
// Draft sliding-window attention against a direct CPU reference. The kernel
// streams the 2048-slot ring in ring-aligned page tiles over a fixed
// number of splits and combines their partials in split order; these cases
// cover an unaligned wrap, a wrap that starts exactly one slot before the ring
// end, the short prefix before any wrap, aligned wraps, windows that fill
// exactly one or several splits, and the last slot of the ring, so that tile
// bounds, the window mask, empty splits and the empty-row softmax guard are
// all exercised.
#include "metal/MetalBackend.hpp"
#include "ops/DraftAttention.hpp"
#include "tuning/LinearNumerics.hpp"

#import <Foundation/Foundation.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::metal::CommandGraph;
using namespace splash::ops;

constexpr uint32_t kRows = 8;
constexpr uint32_t kKvHeads = 8;
constexpr uint32_t kQueryHeadsPerKv = 4;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kWindow = 2048;
constexpr uint32_t kLanes = 4;
// The shipped split count and the fp32 partial one split leaves per (lane,
// head) behind the grouped queries: 32 rows x (128 + max + sum).
constexpr uint32_t kSplits = 4;
constexpr uint64_t kPartialBytes = uint64_t{32} * 130 * sizeof(float);
constexpr uint32_t kAttention = kKvHeads * kQueryHeadsPerKv * kHeadDim;
constexpr uint32_t kGroupRows = kQueryHeadsPerKv * kRows;
constexpr float kScale = 0.08838834765F;

constexpr std::array kShapes{
    DraftAttentionShape{5120, 1280, 6144, 4096, 32, 8, 128},
    DraftAttentionShape{2048, 512, 6144, 4096, 32, 8, 128}};

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class Function> void rejects(Function function) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("invalid draft attention request was accepted");
}

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
    values[index] = tuning::floatToBf16(random.unit());
  return buffer;
}

// Reference for one lane and kv head: every row attends to the ring window it
// can see plus all eight current rows, exactly as the kernel's mask defines.
void referenceRows(const uint16_t *queries, const uint16_t *keys,
                   const uint16_t *values, const uint16_t *queryKeys,
                   const uint16_t *queryValues, uint32_t cacheLength,
                   std::vector<float> &output) {
  const uint32_t commonStart =
      cacheLength >= kWindow - 1 ? cacheLength - (kWindow - 1) : 0;
  const uint32_t oldCount = cacheLength - commonStart;
  std::vector<double> scores(oldCount + kRows);
  std::vector<double> accumulated(kHeadDim);
  for (uint32_t row = 0; row < kGroupRows; ++row) {
    const uint32_t proposal = row % kRows;
    const uint32_t queryPosition = cacheLength + proposal;
    const uint32_t rowStart =
        queryPosition >= kWindow - 1 ? queryPosition - (kWindow - 1) : 0;
    const uint32_t hiddenPrefix = rowStart - commonStart;
    const uint16_t *query = queries + uint64_t{row} * kHeadDim;
    double best = -INFINITY;
    for (uint32_t key = 0; key < oldCount + kRows; ++key) {
      double score = -INFINITY;
      if (key >= oldCount) {
        const uint32_t current = key - oldCount;
        double dot = 0.0;
        for (uint32_t d = 0; d < kHeadDim; ++d) {
          dot += double(tuning::bf16ToFloat(query[d])) *
                 tuning::bf16ToFloat(queryKeys[current * kHeadDim + d]);
        }
        score = dot * kScale;
      } else if (key >= hiddenPrefix) {
        const uint32_t slot = (commonStart + key) % kWindow;
        double dot = 0.0;
        for (uint32_t d = 0; d < kHeadDim; ++d) {
          dot += double(tuning::bf16ToFloat(query[d])) *
                 tuning::bf16ToFloat(keys[uint64_t{slot} * kHeadDim + d]);
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
                ? tuning::bf16ToFloat(queryValues[uint64_t{d} * kRows + key - oldCount])
                : tuning::bf16ToFloat(values[uint64_t{d} * kWindow +
                                    (commonStart + key) % kWindow]);
        accumulated[d] += probability * value;
      }
    }
    for (uint32_t d = 0; d < kHeadDim; ++d)
      output[uint64_t{row} * kHeadDim + d] =
          static_cast<float>(accumulated[d] / sum);
  }
}

std::array<splash::ops::DraftKvBuffers, kLanes>
denseContext(std::span<const MetalBuffer> keys,
             std::span<const MetalBuffer> values) {
  std::array<splash::ops::DraftKvBuffers, kLanes> result;
  for (size_t lane = 0; lane < kLanes; ++lane)
    result[lane] = {keys[lane], values[lane]};
  return result;
}

std::array<splash::ops::DraftKvBuffers, kLanes>
pagedContext(MetalBackend &backend, std::span<const MetalBuffer> keys,
             std::span<const MetalBuffer> values) {
  std::array<splash::ops::DraftKvBuffers, kLanes> result;
  constexpr uint32_t pageTokens = SPLASH_DRAFT_PAGE_TOKENS;
  constexpr uint64_t elements = uint64_t{kKvHeads} * pageTokens * kHeadDim;
  for (size_t lane = 0; lane < kLanes; ++lane)
    for (uint32_t page = 0; page < kWindow / pageTokens; ++page) {
      auto allocation =
          backend.allocateBuffer((2 * elements + pageTokens * kHeadDim) * 2,
                                 BufferStorage::Shared, "paged draft oracle");
      auto key = backend.view(allocation, 0, elements * 2);
      auto value = backend.view(allocation, elements * 2, elements * 2);
      auto *k = static_cast<uint16_t *>(key.contents());
      auto *v = static_cast<uint16_t *>(value.contents());
      for (uint32_t head = 0; head < kKvHeads; ++head)
        std::memcpy(k + head * pageTokens * kHeadDim,
                    static_cast<uint16_t *>(keys[lane].contents()) +
                        (head * kWindow + page * pageTokens) * kHeadDim,
                    pageTokens * kHeadDim * 2);
      for (uint32_t channel = 0; channel < kKvHeads * kHeadDim; ++channel)
        std::memcpy(v + channel * pageTokens,
                    static_cast<uint16_t *>(values[lane].contents()) +
                        channel * kWindow + page * pageTokens,
                    pageTokens * 2);
      std::fill_n(v + elements, pageTokens * kHeadDim, uint16_t{0x7fc0});
      result[lane].keyPages.push_back(backend.view(key, 0, elements * 2));
      result[lane].valuePages.push_back(backend.view(value, 0, elements * 2));
    }
  return result;
}

void samePagedContext(
    const std::array<splash::ops::DraftKvBuffers, kLanes> &left,
    const std::array<splash::ops::DraftKvBuffers, kLanes> &right) {
  for (size_t lane = 0; lane < kLanes; ++lane)
    for (size_t page = 0; page < left[lane].keyPages.size(); ++page) {
      for (const auto &pair :
           {std::pair{left[lane].keyPages[page], right[lane].keyPages[page]},
            std::pair{left[lane].valuePages[page],
                      right[lane].valuePages[page]}})
        require(pair.first.sizeBytes() == pair.second.sizeBytes() &&
                    std::memcmp(pair.first.contents(), pair.second.contents(),
                                pair.first.sizeBytes()) == 0,
                "paged context writer differs from dense oracle");
    }
}

void runCase(MetalBackend &backend, uint32_t lanes, DraftAttentionShape shape,
             const std::array<uint32_t, kLanes> &cacheLengths) {
  Random random(0x5eed0000ULL + cacheLengths[0]);
  const uint64_t ringElements = uint64_t{kKvHeads} * kWindow * kHeadDim;
  // The grouped-queries tensor is sized by the plan so the split partials
  // behind the query rows end exactly at the allocation under validation.
  MetalBuffer queries = randomBfloat(
      backend, DraftAttention::plan(shape, lanes).workspace().groupedQueriesBytes / 2,
      random, "draft queries");
  std::vector<MetalBuffer> keys;
  std::vector<MetalBuffer> values;
  // Each tensor follows the last head's ring with one 128-token tile of bf16
  // NaN. Shader validation does not see the MPP loads of a tile that reads
  // past the ring, but a masked NaN value still reaches its row as
  // 0 x NaN, which the finiteness check below rejects.
  const uint64_t tailElements = uint64_t{128} * kHeadDim;
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    keys.push_back(randomBfloat(backend, ringElements + tailElements, random,
                                "draft keys"));
    values.push_back(randomBfloat(backend, ringElements + tailElements, random,
                                  "draft values"));
    for (const MetalBuffer &tensor : {keys.back(), values.back()})
      std::fill_n(static_cast<uint16_t *>(tensor.contents()) + ringElements,
                  tailElements, uint16_t{0x7FC0});
  }
  MetalBuffer queryKeys = randomBfloat(
      backend, uint64_t{lanes} * kKvHeads * kRows * kHeadDim, random,
      "draft query keys");
  MetalBuffer queryValues = randomBfloat(
      backend, uint64_t{lanes} * kKvHeads * kHeadDim * kRows, random,
      "draft query values");
  std::vector<uint16_t> input(
      static_cast<const uint16_t *>(queries.contents()),
      static_cast<const uint16_t *>(queries.contents()) +
          uint64_t{lanes} * kRows * kAttention);

  const auto dense = denseContext(keys, values);
  const auto paged = pagedContext(backend, keys, values);
  std::vector<uint16_t> baseline;
  for (const auto &context : {dense, paged})
    for (const auto configuration : DraftAttention::candidates(shape)) {
      std::memcpy(queries.contents(), input.data(), input.size() * 2);
      CommandGraph graph;
      DraftAttention::addDecode(
          graph, {queries, context, queryKeys, queryValues}, cacheLengths,
          kWindow, DraftAttention::plan(shape, lanes, configuration));
      const auto dispatches = graph.dispatches();
      require(dispatches.size() == 2 &&
                  dispatches[0].threadgroups.x == kKvHeads &&
                  dispatches[0].threadgroups.y == lanes &&
                  dispatches[0].threadgroups.z == kSplits &&
                  dispatches[1].threadgroups.x == kKvHeads &&
                  dispatches[1].threadgroups.y == lanes &&
                  dispatches[1].threadgroups.z == 1,
              "draft attention core dispatch changed with configuration");
      static_cast<void>(backend.submitCommand(dispatches));
      const auto *output = static_cast<const uint16_t *>(queries.contents());
      if (baseline.empty())
        baseline.assign(output, output + input.size());
      else
        require(std::equal(baseline.begin(), baseline.end(), output),
                "draft attention candidate changed core output");
    }

  const auto *output = static_cast<const uint16_t *>(queries.contents());
  std::vector<float> reference(uint64_t{kGroupRows} * kHeadDim);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    // Reference the first and final head, whose ring the NaN tile follows;
    // all eight execute above.
    for (const uint32_t head : {0U, kKvHeads - 1}) {
      const uint64_t queryOffset =
          uint64_t{lane} * kRows * kAttention + uint64_t{head} * kGroupRows *
                                                    kHeadDim;
      referenceRows(
          input.data() + queryOffset,
          static_cast<const uint16_t *>(keys[lane].contents()) +
              uint64_t{head} * kWindow * kHeadDim,
          static_cast<const uint16_t *>(values[lane].contents()) +
              uint64_t{head} * kWindow * kHeadDim,
          static_cast<const uint16_t *>(queryKeys.contents()) +
              (uint64_t{lane} * kKvHeads + head) * kRows * kHeadDim,
          static_cast<const uint16_t *>(queryValues.contents()) +
              (uint64_t{lane} * kKvHeads + head) * kHeadDim * kRows,
          cacheLengths[lane], reference);
      for (uint64_t index = 0; index < reference.size(); ++index) {
        const float actual = tuning::bf16ToFloat(output[queryOffset + index]);
        const float expected = reference[index];
        if (!std::isfinite(actual) ||
            std::fabs(actual - expected) >
                0.02F + 0.02F * std::fabs(expected)) {
          std::cerr << "lane " << lane << " head " << head << " length "
                    << cacheLengths[lane] << " row " << index / kHeadDim
                    << " dim " << index % kHeadDim << ": " << actual
                    << " vs " << expected << '\n';
          throw std::runtime_error("draft attention diverged from reference");
        }
      }
    }
  }
}

void planGeometry() {
  for (const auto shape : kShapes) {
    const auto candidates = DraftAttention::candidates(shape);
    require(candidates.size() == 4 && candidates.front() == DraftAttentionConfiguration{},
            "draft candidate list lost the full-grid baseline");
    for (uint32_t lanes = 1; lanes <= kLanes; ++lanes) {
      const uint64_t rows = uint64_t{lanes} * kRows;
      for (const auto configuration : candidates) {
        const auto plan = DraftAttention::plan(shape, lanes, configuration);
        const auto workspace = plan.workspace();
        require(plan.lanes() == lanes && plan.shape() == shape &&
                    plan.configuration() == configuration &&
                    workspace.convolutionBytes == rows * shape.hiddenSize * 2 &&
                    workspace.qkvBytes == rows * 6144 * 2 &&
                    workspace.groupedQueriesBytes ==
                        rows * 4096 * 2 +
                            uint64_t{lanes} * kKvHeads * kSplits * kPartialBytes &&
                    workspace.queryKeysBytes == rows * 8 * 128 * 2 &&
                    workspace.queryValuesBytes == rows * 8 * 128 * 2,
                "draft plan padded lanes or changed tensor storage");
      }
    }
    rejects([&] { (void)DraftAttention::plan(shape, 0); });
    rejects([&] { (void)DraftAttention::plan(shape, 5); });
    rejects([&] { (void)DraftAttention::plan(shape, 1, {1}); });
    rejects([&] { (void)DraftAttention::plan(shape, 1, {61}); });
  }
  auto unsupported = kShapes[0];
  unsupported.queryHeads = 16;
  rejects([&] { (void)DraftAttention::candidates(unsupported); });
  rejects([&] { (void)DraftAttention::plan({}, 1); });
}

void fillDyadic(const MetalBuffer &buffer, uint32_t multiplier, uint32_t modulus) {
  auto *values = static_cast<uint16_t *>(buffer.contents());
  for (uint64_t i = 0; i < buffer.sizeBytes() / 2; ++i)
    values[i] = tuning::floatToBf16((int((i * multiplier) % modulus) - int(modulus / 2)) /
                        8.0F);
}

void surroundingPhases(MetalBackend &backend, DraftAttentionShape shape,
                       uint32_t lanes) {
  const auto baselinePlan = DraftAttention::plan(shape, lanes);
  const auto workspace = baselinePlan.workspace();
  const uint64_t rows = uint64_t{lanes} * kRows;
  auto allocate = [&](uint64_t bytes) {
    return backend.allocateBuffer(bytes, BufferStorage::Shared, "draft phases");
  };
  const auto input = allocate(workspace.convolutionBytes);
  const auto dynamic = allocate(rows * shape.dynamicSize * 2);
  const auto weights = allocate(uint64_t{4} * shape.hiddenSize * 2);
  const auto residual = allocate(workspace.convolutionBytes);
  const auto output = allocate(workspace.convolutionBytes);
  fillDyadic(input, 7, 23);
  fillDyadic(dynamic, 5, 17);
  fillDyadic(weights, 3, 13);
  fillDyadic(residual, 11, 19);
  const auto *in = static_cast<const uint16_t *>(input.contents());
  const auto *dyn = static_cast<const uint16_t *>(dynamic.contents());
  const auto *base = static_cast<const uint16_t *>(weights.contents());
  const auto *res = static_cast<const uint16_t *>(residual.contents());
  const uint32_t channelsPerGroup = 16;
  const uint32_t convolutionGroups = shape.hiddenSize / channelsPerGroup;

  Random random(0x5eed1234 + shape.hiddenSize + lanes);
  const auto qkv = randomBfloat(backend, workspace.qkvBytes / 2, random, "QKV");
  const auto queries = allocate(workspace.groupedQueriesBytes);
  const auto queryKeys = allocate(workspace.queryKeysBytes);
  const auto queryValues = allocate(workspace.queryValuesBytes);
  const auto packed = allocate(workspace.groupedQueriesBytes);
  const auto queryNorm = allocate(kHeadDim * 2);
  const auto keyNorm = allocate(kHeadDim * 2);
  std::fill_n(static_cast<uint16_t *>(queryNorm.contents()), kHeadDim,
              tuning::floatToBf16(1));
  std::fill_n(static_cast<uint16_t *>(keyNorm.contents()), kHeadDim,
              tuning::floatToBf16(1));
  const auto ropeCos = allocate(rows * kHeadDim / 2 * sizeof(float));
  const auto ropeSin = allocate(rows * kHeadDim / 2 * sizeof(float));
  for (uint64_t i = 0; i < rows * kHeadDim / 2; ++i) {
    static_cast<float *>(ropeCos.contents())[i] = std::cos(float(i) * 0.01F);
    static_cast<float *>(ropeSin.contents())[i] = std::sin(float(i) * 0.01F);
  }
  std::vector<uint16_t> originalQkv(
      static_cast<const uint16_t *>(qkv.contents()),
      static_cast<const uint16_t *>(qkv.contents()) + workspace.qkvBytes / 2);
  const std::array preparedBuffers{qkv, queries, queryKeys, queryValues};
  std::array<std::vector<uint16_t>, 4> preparedBaseline;
  for (const auto configuration : DraftAttention::candidates(shape)) {
    const auto plan = DraftAttention::plan(shape, lanes, configuration);
    for (const auto stage : {DraftConvolutionStage::Prepare,
                            DraftConvolutionStage::Residual}) {
      std::memset(output.contents(), 0xFF, output.sizeBytes());
      CommandGraph graph;
      DraftAttention::addConvolution(graph,
          {input, dynamic, weights, residual, output}, plan, stage);
      std::array<uint32_t, 3> params{};
      std::memcpy(params.data(), graph.dispatches()[0].bytes[0].data,
                  sizeof(params));
      const uint32_t expectedGroups = configuration.groups
          ? configuration.groups
          : (kRows * shape.hiddenSize + 255) / 256;
      require(graph.dispatches()[0].threadgroups.x == params[0] &&
                  params[0] == expectedGroups && params[2] == lanes,
              "convolution dispatch and loop stride permit overlapping writes");
      static_cast<void>(backend.submitCommand(graph.dispatches()));
      const auto *actual = static_cast<const uint16_t *>(output.contents());
      const bool finish = stage == DraftConvolutionStage::Residual;
      const uint32_t kind = finish ? 1 : 0;
      for (uint64_t row = 0; row < rows; ++row) {
        for (uint32_t channel = 0; channel < shape.hiddenSize; ++channel) {
          const uint64_t index = row * shape.hiddenSize + channel;
          const uint32_t group = channel / channelsPerGroup;
          float value = tuning::bf16ToFloat(in[index]) *
              (tuning::bf16ToFloat(base[(kind * 2) * shape.hiddenSize + channel]) +
               tuning::bf16ToFloat(dyn[row * shape.dynamicSize +
                              (kind * 2) * convolutionGroups + group]));
          if (row % kRows != 0)
            value += tuning::bf16ToFloat(in[index - shape.hiddenSize]) *
                (tuning::bf16ToFloat(base[(kind * 2 + 1) * shape.hiddenSize + channel]) +
                 tuning::bf16ToFloat(dyn[row * shape.dynamicSize +
                                (kind * 2 + 1) * convolutionGroups + group]));
          if (finish)
            value += tuning::bf16ToFloat(res[index]);
          require(actual[index] == tuning::floatToBf16(value),
                  "draft convolution differed from exact CPU arithmetic");
        }
      }
    }

    std::memcpy(qkv.contents(), originalQkv.data(), qkv.sizeBytes());
    CommandGraph graph;
    DraftAttention::addPrepare(graph,
        {qkv, queries, queryNorm, keyNorm, ropeCos, ropeSin, queryKeys,
         queryValues}, plan);
    DraftAttention::addReorder(graph, queries, packed, plan);
    static_cast<void>(backend.submitCommand(graph.dispatches()));
    for (size_t i = 0; i < preparedBuffers.size(); ++i) {
      const auto *values =
          static_cast<const uint16_t *>(preparedBuffers[i].contents());
      if (configuration == DraftAttentionConfiguration{})
        preparedBaseline[i].assign(values,
                                   values + preparedBuffers[i].sizeBytes() / 2);
      else
        require(std::equal(preparedBaseline[i].begin(), preparedBaseline[i].end(),
                           values),
                "draft prepare candidate changed normalization/RoPE/storage");
    }
    const auto *grouped = static_cast<const uint16_t *>(queries.contents());
    const auto *reordered = static_cast<const uint16_t *>(packed.contents());
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint64_t laneOffset = uint64_t{lane} * kRows * kAttention;
      for (uint32_t row = 0; row < kRows; ++row) {
        for (uint32_t head = 0; head < shape.queryHeads; ++head) {
          for (uint32_t dim = 0; dim < kHeadDim; ++dim)
            require(reordered[laneOffset + row * kAttention + head * kHeadDim +
                               dim] ==
                        grouped[laneOffset + (head * kRows + row) * kHeadDim +
                                dim],
                    "draft reorder differed from CPU layout reference");
        }
      }
    }
  }

  CommandGraph invalid;
  rejects([&] {
    DraftAttention::addConvolution(invalid,
        {input, dynamic, weights, residual, output}, baselinePlan,
        static_cast<DraftConvolutionStage>(2));
  });
  const auto shortBuffer = backend.view(output, 0, output.sizeBytes() - 2);
  rejects([&] {
    DraftAttention::addConvolution(invalid,
        {input, dynamic, weights, residual, shortBuffer}, baselinePlan,
        DraftConvolutionStage::Prepare);
  });
  rejects([&] {
    DraftAttention::addPrepare(invalid,
        {qkv, queries, queryNorm, keyNorm, ropeCos, ropeSin, {}, queryValues},
        baselinePlan);
  });
  rejects([&] {
    DraftAttention::addReorder(invalid, queries, {}, baselinePlan);
  });
  const std::array<splash::ops::DraftKvBuffers, kLanes> emptyRings{};
  const std::array<uint32_t, kLanes> lengths{};
  rejects([&] {
    DraftAttention::addDecode(invalid,
                              {queries, emptyRings, queryKeys, queryValues},
                              lengths, kWindow, baselinePlan);
  });
  rejects([&] {
    DraftAttention::addDecode(invalid,
                              {queries, emptyRings, queryKeys, queryValues},
                              lengths, kWindow - 1, baselinePlan);
  });
  require(invalid.empty(), "invalid draft request partially encoded a graph");
}

// The context writers against a CPU ring: a row's key is RMS-normalized,
// scaled by the key norm and rotated, its value copied, into slot
// position % 2048 of each KV head's ring (keys [head][slot][dim], values
// [head][dim][slot]); every other slot keeps its bits. The prefill writes
// its rows from a start position, the commit each lane's retained verify
// rows (at most eight). The buffers hold exactly what the writers read, and
// the host rejects another ring stride or a buffer below its rows.
void contextWriters(MetalBackend &backend, DraftAttentionShape shape) {
  constexpr uint32_t kPacked = 6144, kKeyColumn = 4096, kValueColumn = 5120;
  constexpr uint16_t kUntouched = 0xC2C2;
  const uint64_t ringElements = uint64_t{kKvHeads} * kWindow * kHeadDim;
  Random random(0xc0de0000ULL + shape.hiddenSize);
  const MetalBuffer keyNorm =
      randomBfloat(backend, kHeadDim, random, "draft key norm");
  const auto *norm = static_cast<const uint16_t *>(keyNorm.contents());
  const auto ring = [&] {
    MetalBuffer buffer = backend.allocateBuffer(
        ringElements * 2, BufferStorage::Shared, "draft ring");
    std::fill_n(static_cast<uint16_t *>(buffer.contents()), ringElements,
                kUntouched);
    return buffer;
  };
  const auto table = [&](uint64_t rows, bool sine) {
    MetalBuffer buffer =
        backend.allocateBuffer(rows * kHeadDim / 2 * sizeof(float),
                               BufferStorage::Shared, "draft rope");
    for (uint64_t i = 0; i < rows * kHeadDim / 2; ++i)
      static_cast<float *>(buffer.contents())[i] =
          sine ? std::sin(float(i) * 0.37F) : std::cos(float(i) * 0.37F);
    return buffer;
  };
  // Compares one lane's rings with `rows` rows of `qkv` and the RoPE tables
  // written from `start` on.
  const auto check = [&](const MetalBuffer &keys, const MetalBuffer &values,
                         const uint16_t *qkv, const float *cosines,
                         const float *sines, uint32_t rows, uint32_t start) {
    std::vector<uint16_t> wantValues(ringElements, kUntouched);
    std::vector<float> wantKeys(ringElements, NAN);
    for (uint32_t row = 0; row < rows; ++row) {
      const uint32_t slot = (start + row) % kWindow;
      for (uint32_t head = 0; head < kKvHeads; ++head) {
        const uint16_t *key =
            qkv + uint64_t{row} * kPacked + kKeyColumn + head * kHeadDim;
        const uint16_t *value =
            qkv + uint64_t{row} * kPacked + kValueColumn + head * kHeadDim;
        float square = 0.0F;
        for (uint32_t d = 0; d < kHeadDim; ++d)
          square += tuning::bf16ToFloat(key[d]) * tuning::bf16ToFloat(key[d]);
        const float inverse = 1.0F / std::sqrt(square / kHeadDim + 1e-6F);
        std::array<float, kHeadDim> normalized;
        for (uint32_t d = 0; d < kHeadDim; ++d)
          normalized[d] = tuning::bf16ToFloat(
              tuning::floatToBf16(tuning::bf16ToFloat(key[d]) * inverse *
                                  tuning::bf16ToFloat(norm[d])));
        float *out =
            wantKeys.data() + (uint64_t{head} * kWindow + slot) * kHeadDim;
        for (uint32_t d = 0; d < kHeadDim / 2; ++d) {
          const float c = cosines[uint64_t{row} * kHeadDim / 2 + d],
                      s = sines[uint64_t{row} * kHeadDim / 2 + d];
          out[d] = normalized[d] * c - normalized[d + kHeadDim / 2] * s;
          out[d + kHeadDim / 2] =
              normalized[d + kHeadDim / 2] * c + normalized[d] * s;
        }
        for (uint32_t d = 0; d < kHeadDim; ++d)
          wantValues[(uint64_t{head} * kHeadDim + d) * kWindow + slot] =
              value[d];
      }
    }
    const auto *gotKeys = static_cast<const uint16_t *>(keys.contents());
    const auto *gotValues = static_cast<const uint16_t *>(values.contents());
    for (uint64_t i = 0; i < ringElements; ++i) {
      const float expected = wantKeys[i],
                  actual = tuning::bf16ToFloat(gotKeys[i]);
      require(gotValues[i] == wantValues[i] &&
                  (std::isnan(expected)
                       ? gotKeys[i] == kUntouched
                       : std::fabs(actual - expected) <=
                             0.02F + 0.02F * std::fabs(expected)),
              "draft context writer diverged from the CPU ring");
    }
  };

  // Prefill: 37 rows from position 6130 (slot 2034), across the ring's end.
  constexpr uint32_t kTokens = 37, kStart = 6130;
  const MetalBuffer qkv = randomBfloat(backend, uint64_t{kTokens} * kPacked,
                                       random, "draft context qkv");
  const MetalBuffer ropeCos = table(kTokens, false),
                    ropeSin = table(kTokens, true);
  const MetalBuffer keys = ring(), values = ring();
  std::array<MetalBuffer, kLanes> initialKeys, initialValues;
  initialKeys.fill(keys);
  initialValues.fill(values);
  auto pagedPrefill = pagedContext(backend, initialKeys, initialValues);
  CommandGraph prefill;
  DraftAttention::addContextPrefill(prefill, qkv, keyNorm, ropeCos, ropeSin,
                                    {keys, values}, kTokens, kWindow, kStart,
                                    shape);
  static_cast<void>(backend.submitCommand(prefill.dispatches()));
  check(keys, values, static_cast<const uint16_t *>(qkv.contents()),
        static_cast<const float *>(ropeCos.contents()),
        static_cast<const float *>(ropeSin.contents()), kTokens, kStart);

  CommandGraph pagedWrite;
  for (auto &lane : pagedPrefill)
    DraftAttention::addContextPrefill(pagedWrite, qkv, keyNorm, ropeCos,
                                      ropeSin, lane, kTokens, kWindow, kStart,
                                      shape);
  static_cast<void>(backend.submitCommand(pagedWrite.dispatches()));
  samePagedContext(pagedPrefill,
                   pagedContext(backend, initialKeys, initialValues));

  // Commit: three lanes retaining 8, 3 and (clamped) 8 of their verify rows.
  constexpr uint32_t kCommitLanes = 3;
  const std::array<uint32_t, kLanes> starts{2044, 0, 4101, 0};
  const uint32_t retained[kCommitLanes] = {8, 3, 12};
  const MetalBuffer laneQkv =
      randomBfloat(backend, uint64_t{kCommitLanes} * kRows * kPacked, random,
                   "draft lane qkv");
  const MetalBuffer laneCos = table(kCommitLanes * kRows, false),
                    laneSin = table(kCommitLanes * kRows, true);
  MetalBuffer retainedCounts = backend.allocateBuffer(
      sizeof(retained), BufferStorage::Shared, "draft retained counts");
  std::memcpy(retainedCounts.contents(), retained, sizeof(retained));
  std::array<MetalBuffer, kLanes> laneKeys, laneValues;
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    laneKeys[lane] = lane < kCommitLanes ? ring() : laneKeys[0];
    laneValues[lane] = lane < kCommitLanes ? ring() : laneValues[0];
  }
  auto pagedCommit = pagedContext(backend, laneKeys, laneValues);
  CommandGraph commit;
  DraftAttention::addContextCommit(commit, laneQkv, keyNorm, laneCos, laneSin,
                                   denseContext(laneKeys, laneValues),
                                   retainedCounts, starts, kWindow, shape,
                                   kCommitLanes);
  static_cast<void>(backend.submitCommand(commit.dispatches()));
  for (uint32_t lane = 0; lane < kCommitLanes; ++lane)
    check(laneKeys[lane], laneValues[lane],
          static_cast<const uint16_t *>(laneQkv.contents()) +
              uint64_t{lane} * kRows * kPacked,
          static_cast<const float *>(laneCos.contents()) +
              uint64_t{lane} * kRows * kHeadDim / 2,
          static_cast<const float *>(laneSin.contents()) +
              uint64_t{lane} * kRows * kHeadDim / 2,
          std::min(retained[lane], kRows), starts[lane]);

  CommandGraph pagedCommitGraph;
  DraftAttention::addContextCommit(pagedCommitGraph, laneQkv, keyNorm, laneCos,
                                   laneSin, pagedCommit, retainedCounts, starts,
                                   kWindow, shape, kCommitLanes);
  static_cast<void>(backend.submitCommand(pagedCommitGraph.dispatches()));
  // The padded fourth lane aliases dense lane zero but its separate paged
  // allocation is not executed. Compare only the active lanes.
  auto expectedPages = pagedContext(backend, laneKeys, laneValues);
  expectedPages[3] = pagedCommit[3];
  samePagedContext(pagedCommit, expectedPages);

  const auto shorter = [&](const MetalBuffer &buffer) {
    return backend.view(buffer, 0, buffer.sizeBytes() - 2);
  };
  CommandGraph invalid;
  auto separated = pagedPrefill[0];
  separated.valuePages[0] = separated.valuePages[1];
  rejects([&] {
    DraftAttention::addContextPrefill(invalid, qkv, keyNorm, ropeCos, ropeSin,
                                      separated, kTokens, kWindow, kStart,
                                      shape);
  });
  const auto prefillWith = [&](const MetalBuffer &q, const MetalBuffer &sines,
                               const MetalBuffer &k, uint32_t stride) {
    DraftAttention::addContextPrefill(invalid, q, keyNorm, ropeCos, sines,
                                      {k, values}, kTokens, stride, kStart,
                                      shape);
  };
  // A ring of another stride: head h's upper slots would land in head h + 1
  // and the last head's past the ring.
  const MetalBuffer halfRing = backend.view(keys, 0, keys.sizeBytes() / 2);
  rejects([&] { prefillWith(qkv, ropeSin, halfRing, kWindow / 2); });
  rejects([&] { prefillWith(qkv, ropeSin, shorter(keys), kWindow); });
  rejects([&] { prefillWith(shorter(qkv), ropeSin, keys, kWindow); });
  rejects([&] { prefillWith(qkv, shorter(ropeSin), keys, kWindow); });
  // The last lane's values ring.
  const MetalBuffer lastRing = laneValues[kCommitLanes - 1];
  const auto commitWith = [&](const MetalBuffer &q, const MetalBuffer &counts,
                              const MetalBuffer &last, uint32_t stride) {
    std::array<MetalBuffer, kLanes> rings = laneValues;
    rings[kCommitLanes - 1] = last;
    DraftAttention::addContextCommit(invalid, q, keyNorm, laneCos, laneSin,
                                     denseContext(laneKeys, rings), counts,
                                     starts, stride, shape, kCommitLanes);
  };
  rejects([&] { commitWith(laneQkv, retainedCounts, lastRing, kWindow / 2); });
  rejects([&] { commitWith(laneQkv, retainedCounts, shorter(lastRing), kWindow); });
  rejects([&] { commitWith(shorter(laneQkv), retainedCounts, lastRing, kWindow); });
  rejects([&] { commitWith(laneQkv, shorter(retainedCounts), lastRing, kWindow); });
  require(invalid.empty(),
          "invalid draft context write partially encoded a graph");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: draft-attention METALLIB");
    planGeometry();
    MetalBackend backend(argv[1]);
    for (const auto shape : kShapes) {
      for (uint32_t lanes = 1; lanes <= kLanes; ++lanes)
        surroundingPhases(backend, shape, lanes);
      contextWriters(backend, shape);
      runCase(backend, 1, shape, {0, 0, 0, 0});
      runCase(backend, 1, shape, {512, 0, 0, 0});
      runCase(backend, 1, shape, {2048, 0, 0, 0});
      runCase(backend, 1, shape, {8191, 0, 0, 0});
      runCase(backend, 2, shape, {2048, 2047, 0, 0});
      runCase(backend, 3, shape, {2100, 4094, 6143, 0});
      runCase(backend, 4, shape, {262137, 4094, 500, 6143});
      runCase(backend, 4, shape, {512, 513, 1024, 1536});
      runCase(backend, 4, shape, {2046, 2049, 4095, 4096});
      runCase(backend, 2, shape, {8191, 262144, 0, 0});
    }
    std::cout << "draft_attention_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "draft_attention_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}

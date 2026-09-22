// DFlash draft selector against a direct CPU reference: the sharded top-16
// scan and its reduce must return exactly the sixteen largest logits of every
// proposal row in (value desc, id asc) order, and the codebook walk must pick
// the same tokens as a double-precision evaluation of the same scores. The
// vocabularies cover the production size, an odd size that misaligns the
// 16-byte vectors and leaves shards with only a few tokens, and one wider
// than a single register chunk per thread.
#include "metal/MetalBackend.hpp"
#include "ops/Sampling.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using namespace splash::ops;

constexpr uint32_t kRows = SPLASH_DRAFT_QUERY_ROWS;
constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
constexpr uint32_t kCandidates = 16;
constexpr uint32_t kRank = 256;
constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;

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
  throw std::runtime_error("invalid draft selector request was accepted");
}

uint16_t toBfloat(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7FFFU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>(bits >> 16);
}

float fromBfloat(uint16_t value) {
  const uint32_t bits = uint32_t{value} << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
  float unit() {
    return static_cast<float>(next() & 0xFFFFFF) / 8388608.0F - 1.0F;
  }
  uint32_t next() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(state_ >> 33);
  }

private:
  uint64_t state_;
};

MetalBuffer allocate(MetalBackend &backend, uint64_t bytes) {
  MetalBuffer buffer =
      backend.allocateBuffer(bytes, BufferStorage::Shared, "draft selector");
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

MetalBuffer randomBfloat(MetalBackend &backend, uint64_t count, Random &random,
                         float scale) {
  MetalBuffer buffer = allocate(backend, count * sizeof(uint16_t));
  auto *values = static_cast<uint16_t *>(buffer.contents());
  for (uint64_t index = 0; index < count; ++index)
    values[index] = toBfloat(random.unit() * scale);
  return buffer;
}

// Row patterns: peaked logits with a few spikes (the production shape),
// uniform noise, heavy ties from a seven-value alphabet with -inf runs, and
// a row with ten finite tokens so -inf tokens fill the tail by id.
enum class Pattern : uint8_t { Peaked, Uniform, Ties, Sparse };

void fillRow(uint16_t *row, uint32_t vocabulary, Pattern pattern,
             Random &random) {
  for (uint32_t token = 0; token < vocabulary; ++token) {
    float value = 0.0F;
    switch (pattern) {
    case Pattern::Peaked:
      value = -8.0F + 3.0F * random.unit() * random.unit();
      break;
    case Pattern::Uniform:
      value = 4.0F * random.unit();
      break;
    case Pattern::Ties:
      value = random.next() % 11 == 0
                  ? -INFINITY
                  : static_cast<float>(int(random.next() % 7)) - 3.0F;
      break;
    case Pattern::Sparse:
      value = -INFINITY;
      break;
    }
    row[token] = toBfloat(value);
  }
  if (pattern == Pattern::Peaked) {
    for (uint32_t spike = 0; spike < 40; ++spike)
      row[random.next() % vocabulary] = toBfloat(4.0F + 6.0F * random.unit());
  }
  if (pattern == Pattern::Sparse) {
    for (uint32_t finite = 0; finite < 10; ++finite)
      row[random.next() % vocabulary] = toBfloat(random.unit());
  }
}

// The row's sixteen largest tokens: value descending, id ascending on ties.
std::vector<uint32_t> referenceTop16(const uint16_t *row, uint32_t vocabulary) {
  std::vector<uint32_t> order(vocabulary);
  std::iota(order.begin(), order.end(), 0U);
  const auto beats = [&](uint32_t a, uint32_t b) {
    const float va = fromBfloat(row[a]), vb = fromBfloat(row[b]);
    return va > vb || (va == vb && a < b);
  };
  const size_t keep = std::min<size_t>(kCandidates, vocabulary);
  std::partial_sort(order.begin(), order.begin() + keep, order.end(), beats);
  order.resize(keep);
  return order;
}

struct Case final {
  uint32_t vocabulary;
  uint32_t lanes;
  bool sampling;
};

void runCase(MetalBackend &backend, const Case &c) {
  Random random(0x5e1ec7 + uint64_t{c.vocabulary} * 8 + c.lanes * 2 + c.sampling);
  const uint32_t rows = c.lanes * kRows;
  const uint32_t positions = c.lanes * kPositions;
  const auto workspace = Sampling::draftWorkspace(positions);
  Sampling sampling(backend, c.vocabulary, kRows);

  MetalBuffer logits = allocate(backend, uint64_t{rows} * c.vocabulary * 2);
  auto *logitRows = static_cast<uint16_t *>(logits.contents());
  const std::array patterns{Pattern::Peaked, Pattern::Uniform, Pattern::Ties,
                            Pattern::Sparse};
  for (uint32_t row = 0; row < rows; ++row)
    fillRow(logitRows + uint64_t{row} * c.vocabulary, c.vocabulary,
            patterns[(row / kRows + row % kRows) % patterns.size()], random);
  DraftSelectorBuffers buffers{
      logits,
      allocate(backend, workspace.partialIdsBytes),
      allocate(backend, workspace.partialValuesBytes),
      allocate(backend, workspace.candidatesBytes),
      allocate(backend, workspace.unaryBytes),
      randomBfloat(backend, uint64_t{rows} * kRank, random, 0.1F),
      randomBfloat(backend, uint64_t{c.vocabulary} * kRank, random, 0.1F),
      randomBfloat(backend, uint64_t{c.vocabulary} * kRank, random, 0.1F),
      allocate(backend, uint64_t{c.lanes} * 2 * kRows * sizeof(float)),
      allocate(backend, uint64_t{positions} * sizeof(uint32_t)),
      allocate(backend, workspace.proposalProbabilitiesBytes)};
  auto *uniforms = static_cast<float *>(buffers.uniforms.contents());
  for (uint32_t index = 0; index < c.lanes * 2 * kRows; ++index)
    uniforms[index] = (random.unit() + 1.0F) * 0.5F;
  std::vector<uint32_t> anchors(c.lanes);
  std::vector<SamplingPolicy> policies(c.lanes);
  for (uint32_t lane = 0; lane < c.lanes; ++lane) {
    anchors[lane] = random.next() % c.vocabulary;
    policies[lane] = SamplingPolicy{16, c.sampling ? 0.8F : 0.0F, 1.0F, false};
  }

  CommandGraph graph;
  sampling.addDraftSelector(graph, buffers, anchors, policies, kPositions);
  require(graph.dispatches().size() == 3,
          "draft selector dispatch count changed");
  static_cast<void>(backend.submitCommand(graph.dispatches()));

  const auto *candidates =
      static_cast<const uint32_t *>(buffers.candidates.contents());
  const auto *unary = static_cast<const uint16_t *>(buffers.unary.contents());
  const auto *tokens =
      static_cast<const uint32_t *>(buffers.proposedTokens.contents());
  const auto *probabilities =
      static_cast<const float *>(buffers.proposalProbabilities.contents());
  const auto *hidden =
      static_cast<const uint16_t *>(buffers.selectorHidden.contents());
  const auto *predecessors =
      static_cast<const uint16_t *>(buffers.predecessorCodebook.contents());
  const auto *successors =
      static_cast<const uint16_t *>(buffers.successorCodebook.contents());

  for (uint32_t lane = 0; lane < c.lanes; ++lane) {
    uint32_t predecessor = anchors[lane];
    for (uint32_t position = 0; position < kPositions; ++position) {
      const uint32_t global = lane * kPositions + position;
      const uint16_t *row =
          logitRows + (uint64_t{lane} * kRows + position + 1) * c.vocabulary;
      const auto expected = referenceTop16(row, c.vocabulary);
      for (uint32_t rank = 0; rank < kCandidates; ++rank) {
        const uint32_t id = candidates[global * kCandidates + rank];
        const uint16_t value = unary[global * kCandidates + rank];
        if (rank < expected.size()) {
          require(id == expected[rank] && value == row[expected[rank]],
                  "draft top-16 candidates differ from the exact sorted order");
        } else {
          require(id == 0xFFFFFFFFU && value == toBfloat(-INFINITY),
                  "draft top-16 padding lost the empty sentinel");
        }
      }

      // Scores as the kernel defines them, in double.
      std::array<double, kCandidates> scores{};
      for (uint32_t rank = 0; rank < kCandidates; ++rank) {
        const uint32_t candidate =
            std::min(candidates[global * kCandidates + rank], c.vocabulary - 1);
        double edge = 0.0;
        for (uint32_t dim = 0; dim < kRank; ++dim) {
          edge += double(fromBfloat(predecessors[uint64_t{predecessor} * kRank + dim])) *
                  fromBfloat(hidden[(uint64_t{lane} * kRows + position + 1) * kRank + dim]) *
                  fromBfloat(successors[uint64_t{candidate} * kRank + dim]);
        }
        scores[rank] = double(fromBfloat(unary[global * kCandidates + rank])) + edge;
      }
      const uint32_t token = tokens[global];
      uint32_t selected = kCandidates;
      for (uint32_t rank = 0; rank < kCandidates; ++rank)
        if (candidates[global * kCandidates + rank] == token)
          selected = selected == kCandidates ? rank : selected;
      require(selected < kCandidates,
              "draft selector proposed a token outside its candidates");
      if (c.sampling) {
        const double maximum = *std::max_element(scores.begin(), scores.end());
        double sum = 0.0;
        std::array<double, kCandidates> reference{};
        for (uint32_t rank = 0; rank < kCandidates; ++rank) {
          reference[rank] = std::exp((scores[rank] - maximum) / 0.8);
          sum += reference[rank];
        }
        const float uniform = uniforms[lane * 2 * kRows + position + 1];
        std::array<double, kCandidates> cumulative{};
        uint32_t expectedSelection = kCandidates - 1;
        for (uint32_t rank = 0; rank < kCandidates; ++rank) {
          const float probability = probabilities[global * kCandidates + rank];
          require(std::fabs(probability - reference[rank] / sum) < 1e-4,
                  "draft selector probabilities diverged from the softmax");
          cumulative[rank] = (rank ? cumulative[rank - 1] : 0.0) + reference[rank] / sum;
          if (expectedSelection == kCandidates - 1 && cumulative[rank] > uniform)
            expectedSelection = rank;
        }
        // A draw within fp32 rounding of the crossed boundary may go either way.
        const double boundary =
            std::fabs(cumulative[std::min(selected, expectedSelection)] - uniform);
        require(selected == expectedSelection || boundary < 1e-5,
                "draft selector drew a different candidate than the reference");
      } else {
        uint32_t expectedSelection = 0;
        for (uint32_t rank = 1; rank < kCandidates; ++rank)
          if (scores[rank] > scores[expectedSelection])
            expectedSelection = rank;
        // fp32 accumulation over 256 products of magnitude 1e-3 stays far
        // below this margin; only an exact tie could legitimately differ.
        require(selected == expectedSelection ||
                    std::fabs(scores[selected] - scores[expectedSelection]) < 1e-5,
                "draft selector picked a different greedy candidate");
      }
      predecessor = token;
    }
  }
}

// Compare every mixed policy mask against isolated lane execution. Poison
// outputs so a missing greedy argmax cannot pass by reading old token data.
void mixedVerify(MetalBackend &backend, uint32_t lanes, uint32_t samplingMask) {
  constexpr uint32_t vocabulary = 1003;
  Sampling sampling(backend, vocabulary, kRows);
  auto buffers = [&](uint32_t width) {
    const uint32_t rows = width * kRows;
    const auto space = Sampling::workspace(rows);
    return SamplingBuffers{
        allocate(backend, uint64_t{rows} * vocabulary * 2),
        allocate(backend, space.partialIdsBytes),
        allocate(backend, space.partialValuesBytes),
        allocate(backend, space.topIdsBytes),
        allocate(backend, space.topProbabilitiesBytes),
        allocate(backend, uint64_t{width} * 2 * kRows * sizeof(float)),
        allocate(backend, uint64_t{rows} * ((vocabulary + 31) / 32) * 4),
        allocate(backend, uint64_t{rows} * sizeof(uint32_t)),
        allocate(backend, space.argmaxValuesBytes),
        allocate(backend, space.argmaxIndicesBytes)};
  };
  auto batch = buffers(lanes);
  std::memset(batch.outputTokens.contents(), 0xFF, batch.outputTokens.sizeBytes());
  Random random(9831 + lanes);
  std::vector<SamplingPolicy> policies;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    policies.push_back(
        {8 + lane, (samplingMask & (1U << lane)) ? 0.7F : 0.0F, 0.9F, false});
    for (uint32_t row = 0; row < kRows; ++row)
      fillRow(static_cast<uint16_t *>(batch.logits.contents()) +
                  (lane * kRows + row) * vocabulary,
              vocabulary, row % 2 ? Pattern::Ties : Pattern::Peaked, random);
  }
  CommandGraph graph;
  sampling.addVerify(graph, policies, batch);
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    auto single = buffers(1);
    std::memcpy(single.logits.contents(),
                static_cast<uint16_t *>(batch.logits.contents()) +
                    lane * kRows * vocabulary,
                single.logits.sizeBytes());
    CommandGraph reference;
    sampling.addVerify(reference, std::span(policies).subspan(lane, 1), single);
    static_cast<void>(backend.submitCommand(reference.dispatches()));
    if (policies[lane].samples()) {
      const uint64_t offset = uint64_t{lane} * kRows * kTargetSamplingCandidates;
      require(std::memcmp(static_cast<uint32_t *>(batch.topIds.contents()) + offset,
                          single.topIds.contents(), single.topIds.sizeBytes()) == 0,
              "mixed verification changed sampling candidates");
      const auto *actual = static_cast<float *>(batch.topProbabilities.contents());
      const auto *expected =
          static_cast<float *>(single.topProbabilities.contents());
      for (uint32_t i = 0; i < kRows * kTargetSamplingCandidates; ++i)
        require(std::abs(actual[offset + i] - expected[i]) < 1e-6F,
                "mixed verification changed sampling probabilities");
    } else {
      require(std::memcmp(static_cast<uint32_t *>(batch.outputTokens.contents()) +
                              lane * kRows,
                          single.outputTokens.contents(),
                          single.outputTokens.sizeBytes()) == 0,
              "mixed verification changed greedy tokens");
    }
  }
}

// Top-k=1 and constrained greedy must agree with a full-vocabulary CPU
// argmax, including ties, row offsets and masks. Poison every scratch/output
// buffer: the compact path must not consume stale top-32 entries.
void targetTop1(MetalBackend &backend, uint32_t vocabulary, uint32_t lanes) {
  const uint32_t rows = lanes * kRows;
  const uint32_t words = (vocabulary + 31) / 32;
  const auto space = Sampling::workspace(rows);
  Sampling sampling(backend, vocabulary, kRows);
  SamplingBuffers b{
      allocate(backend, uint64_t{rows} * vocabulary * 2),
      allocate(backend, space.partialIdsBytes),
      allocate(backend, space.partialValuesBytes),
      allocate(backend, space.topIdsBytes),
      allocate(backend, space.topProbabilitiesBytes),
      allocate(backend, uint64_t{lanes} * 2 * kRows * sizeof(float)),
      allocate(backend, uint64_t{lanes} * (kRows + 1) * words * 4),
      allocate(backend, uint64_t{rows} * sizeof(uint32_t)),
      allocate(backend, space.argmaxValuesBytes),
      allocate(backend, space.argmaxIndicesBytes)};
  auto *logits = static_cast<uint16_t *>(b.logits.contents());
  auto *masks = static_cast<uint32_t *>(b.constraintMasks.contents());
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t token = 0; token < vocabulary; ++token)
      logits[uint64_t{row} * vocabulary + token] =
          toBfloat(float(int((token * 7 + row * 13) % 23) - 11));
  for (uint32_t row = 0; row < lanes * (kRows + 1); ++row)
    for (uint32_t token = 0; token < vocabulary; ++token)
      if ((token + row) % 17 == 0)
        masks[uint64_t{row} * words + token / 32] |= 1U << (token % 32);
  auto expected = [&](uint32_t row, uint32_t maskRow) {
    float best = -INFINITY;
    uint32_t id = ~0U;
    for (uint32_t token = 0; token < vocabulary; ++token) {
      if (!(masks[uint64_t{maskRow} * words + token / 32] & (1U << (token % 32))))
        continue;
      const float value = fromBfloat(logits[uint64_t{row} * vocabulary + token]);
      if (value > best) { best = value; id = token; }
    }
    return id;
  };
  auto poison = [&] {
    for (const auto &buffer : {b.partialIds, b.partialValues, b.topIds,
                              b.topProbabilities, b.outputTokens})
      std::memset(buffer.contents(), 0xA5, buffer.sizeBytes());
  };
  auto check = [&](uint32_t row, uint32_t id) {
    const auto *ids = static_cast<uint32_t *>(b.topIds.contents());
    const auto *probabilities = static_cast<float *>(b.topProbabilities.contents());
    double mass = 0;
    for (uint32_t rank = 0; rank < 32; ++rank) {
      const uint32_t index = row * 32 + rank;
      require(std::isfinite(probabilities[index]), "nonfinite target probability");
      require(probabilities[index] == (ids[index] == id ? 1.0F : 0.0F),
              "top-1 target differs from masked CPU argmax");
      mass += probabilities[index];
    }
    require(mass == 1.0, "top-1 target is not normalized");
  };
  for (const float temperature : {0.0F, 0.8F}) {
    poison();
    CommandGraph initial;
    sampling.addInitial(initial, {1, temperature, 0.5F, true}, b, kRows - 1);
    static_cast<void>(backend.submitCommand(initial.dispatches()));
    const uint32_t id = expected(kRows - 1, 0);
    require(static_cast<uint32_t *>(b.outputTokens.contents())[0] == id,
            "initial target differs from masked CPU argmax");
    check(0, id);
  }
  poison();
  std::vector<SamplingPolicy> policies(lanes);
  for (uint32_t lane = 0; lane < lanes; ++lane)
    policies[lane] = {1, lane % 2 ? 0.8F : 0.0F, 0.5F, true};
  CommandGraph verify;
  sampling.addVerify(verify, policies, b);
  static_cast<void>(backend.submitCommand(verify.dispatches()));
  for (uint32_t row = 0; row < rows; ++row) {
    const uint32_t maskRow = row / kRows * (kRows + 1) + row % kRows + 1;
    const uint32_t id = expected(row, maskRow);
    check(row, id);
    require(static_cast<uint32_t *>(b.outputTokens.contents())[row] == id,
            "batched target differs from masked CPU argmax");
  }
}

void invalidRequests(MetalBackend &backend) {
  Sampling sampling(backend, 1024, kRows);
  const auto workspace = Sampling::draftWorkspace(kPositions);
  DraftSelectorBuffers buffers{
      allocate(backend, uint64_t{kRows} * 1024 * 2),
      allocate(backend, workspace.partialIdsBytes),
      allocate(backend, workspace.partialValuesBytes),
      allocate(backend, workspace.candidatesBytes),
      allocate(backend, workspace.unaryBytes),
      allocate(backend, uint64_t{kRows} * kRank * 2),
      allocate(backend, uint64_t{1024} * kRank * 2),
      allocate(backend, uint64_t{1024} * kRank * 2),
      allocate(backend, 2 * kRows * sizeof(float)),
      allocate(backend, kPositions * sizeof(uint32_t)),
      allocate(backend, workspace.proposalProbabilitiesBytes)};
  const std::array<uint32_t, 2> anchors{1, 2};
  const std::array<SamplingPolicy, 1> policies{SamplingPolicy{}};
  CommandGraph graph;
  rejects([&] {
    sampling.addDraftSelector(graph, buffers, anchors, policies, kPositions);
  });
  rejects([&] {
    sampling.addDraftSelector(graph, buffers, std::span(anchors).first(1),
                              policies, 0);
  });
  require(graph.empty(), "invalid draft selector request encoded a graph");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: draft-selector METALLIB");
    MetalBackend backend(argv[1]);
    invalidRequests(backend);
    for (const uint32_t vocabulary : {1003U, 248320U})
      for (uint32_t lanes = 1; lanes <= kLanes; ++lanes)
        targetTop1(backend, vocabulary, lanes);
    for (uint32_t lanes = 1; lanes <= kLanes; ++lanes)
      for (uint32_t mask = 0; mask < (1U << lanes); ++mask)
        mixedVerify(backend, lanes, mask);
    for (const uint32_t vocabulary : {248320U, 1003U, 270005U}) {
      for (uint32_t lanes = 1; lanes <= kLanes; ++lanes) {
        runCase(backend, {vocabulary, lanes, false});
        runCase(backend, {vocabulary, lanes, true});
      }
    }
    std::cout << "draft_selector_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "draft_selector_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}

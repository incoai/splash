// DFlash draft selector against a direct CPU reference: the sharded top-16
// scan and its reduce must return exactly the sixteen largest logits of every
// proposal row in (value desc, id asc) order, and the codebook walk must pick
// the same tokens as a double-precision evaluation of the same scores. The
// vocabularies cover the production size, an odd size that misaligns the
// 16-byte vectors and leaves shards with only a few tokens, and one wider
// than a single register chunk per thread. The logits are fp32, and their
// order is decided below the bf16 spacing. Look-ahead selection must reach
// the optimum of a double-precision dynamic program over the same tables,
// leave sampling lanes untouched, and differ from the greedy walk exactly
// where a crafted table makes the greedy choice a dead end.
#include "metal/MetalBackend.hpp"
#include "metal/abi/Sampling.h"
#include "ops/Sampling.hpp"
#include "tuning/LinearNumerics.hpp"
#include "tests/engine/draft_selector_reference.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
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

// Target sampling takes the model's stop tokens: here one in the first shard
// and one in the last, so a lane that excludes them skips them in both.
std::array<uint32_t, 2> stopTokens(uint32_t vocabulary) {
  return {1, vocabulary - 2};
}

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
    values[index] = tuning::floatToBf16(random.unit() * scale);
  return buffer;
}

// Row patterns: peaked logits with a few spikes (the production shape),
// uniform noise, heavy ties from a seven-value alphabet with -inf runs, and
// a row with ten finite tokens so -inf tokens fill the tail by id.
enum class Pattern : uint8_t { Peaked, Uniform, Ties, Sparse };

void fillRow(float *row, uint32_t vocabulary, Pattern pattern,
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
    row[token] = value;
  }
  if (pattern == Pattern::Peaked) {
    for (uint32_t spike = 0; spike < 40; ++spike)
      row[random.next() % vocabulary] = 4.0F + 6.0F * random.unit();
  }
  if (pattern == Pattern::Sparse) {
    for (uint32_t finite = 0; finite < 10; ++finite)
      row[random.next() % vocabulary] = random.unit();
  }
}

// The row's sixteen largest tokens: value descending, id ascending on ties.
std::vector<uint32_t> referenceTop16(const float *row, uint32_t vocabulary) {
  std::vector<uint32_t> order(vocabulary);
  std::iota(order.begin(), order.end(), 0U);
  const auto beats = [&](uint32_t a, uint32_t b) {
    return row[a] > row[b] || (row[a] == row[b] && a < b);
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

  MetalBuffer logits = allocate(backend, uint64_t{rows} * c.vocabulary * sizeof(float));
  auto *logitRows = static_cast<float *>(logits.contents());
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
  const auto *unary = static_cast<const float *>(buffers.unary.contents());
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
      const float *row =
          logitRows + (uint64_t{lane} * kRows + position + 1) * c.vocabulary;
      const auto expected = referenceTop16(row, c.vocabulary);
      for (uint32_t rank = 0; rank < kCandidates; ++rank) {
        const uint32_t id = candidates[global * kCandidates + rank];
        const float value = unary[global * kCandidates + rank];
        if (rank < expected.size()) {
          require(id == expected[rank] && value == row[expected[rank]],
                  "draft top-16 candidates differ from the exact sorted order");
        } else {
          require(id == 0xFFFFFFFFU && value == -INFINITY,
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
          edge += double(tuning::bf16ToFloat(predecessors[uint64_t{predecessor} * kRank + dim])) *
                  tuning::bf16ToFloat(hidden[(uint64_t{lane} * kRows + position + 1) * kRank + dim]) *
                  tuning::bf16ToFloat(successors[uint64_t{candidate} * kRank + dim]);
        }
        scores[rank] = double(unary[global * kCandidates + rank]) + edge;
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
        allocate(backend, uint64_t{rows} * vocabulary * sizeof(float)),
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
  auto *logits = static_cast<float *>(batch.logits.contents());
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    policies.push_back(
        {8 + lane, (samplingMask & (1U << lane)) ? 0.7F : 0.0F, 0.9F, false});
    for (uint32_t row = 0; row < kRows; ++row)
      fillRow(logits + (lane * kRows + row) * vocabulary, vocabulary,
              row % 2 ? Pattern::Ties : Pattern::Peaked, random);
  }
  const auto stops = stopTokens(vocabulary);
  CommandGraph graph;
  sampling.addVerify(graph, policies, batch, stops[0], stops[1]);
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    auto single = buffers(1);
    std::memcpy(single.logits.contents(), logits + lane * kRows * vocabulary,
                single.logits.sizeBytes());
    CommandGraph reference;
    sampling.addVerify(reference, std::span(policies).subspan(lane, 1), single,
                       stops[0], stops[1]);
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

// Top-k=1, constrained greedy and unconstrained greedy (the argmax kernels)
// must agree with a full-vocabulary CPU argmax, including ties, row offsets
// and masks. Poison every scratch/output buffer: the compact path must not
// consume stale top-32 entries. The fp32 logits carry offsets below the bf16
// spacing of their values, which decide the argmax among equal integer parts:
// reading them rounded would pick the lowest id instead.
void targetTop1(MetalBackend &backend, uint32_t vocabulary, uint32_t lanes) {
  const uint32_t rows = lanes * kRows;
  const uint32_t words = (vocabulary + 31) / 32;
  const auto space = Sampling::workspace(rows);
  Sampling sampling(backend, vocabulary, kRows);
  SamplingBuffers b{
      allocate(backend, uint64_t{rows} * vocabulary * sizeof(float)),
      allocate(backend, space.partialIdsBytes),
      allocate(backend, space.partialValuesBytes),
      allocate(backend, space.topIdsBytes),
      allocate(backend, space.topProbabilitiesBytes),
      allocate(backend, uint64_t{lanes} * 2 * kRows * sizeof(float)),
      allocate(backend, uint64_t{lanes} * (kRows + 1) * words * 4),
      allocate(backend, uint64_t{rows} * sizeof(uint32_t)),
      allocate(backend, space.argmaxValuesBytes),
      allocate(backend, space.argmaxIndicesBytes)};
  auto *logits = static_cast<float *>(b.logits.contents());
  auto *masks = static_cast<uint32_t *>(b.constraintMasks.contents());
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t token = 0; token < vocabulary; ++token)
      logits[uint64_t{row} * vocabulary + token] =
          float(int((token * 7 + row * 13) % 23) - 11) + float(token % 3) * 0x1p-12F;
  for (uint32_t row = 0; row < lanes * (kRows + 1); ++row)
    for (uint32_t token = 0; token < vocabulary; ++token)
      if ((token + row) % 17 == 0)
        masks[uint64_t{row} * words + token / 32] |= 1U << (token % 32);
  constexpr uint32_t kUnmasked = ~0U;
  auto expected = [&](uint32_t row, uint32_t maskRow) {
    float best = -INFINITY;
    uint32_t id = ~0U;
    for (uint32_t token = 0; token < vocabulary; ++token) {
      if (maskRow != kUnmasked &&
          !(masks[uint64_t{maskRow} * words + token / 32] & (1U << (token % 32))))
        continue;
      const float value = logits[uint64_t{row} * vocabulary + token];
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
  const auto stops = stopTokens(vocabulary);
  for (const float temperature : {0.0F, 0.8F}) {
    poison();
    CommandGraph initial;
    sampling.addInitial(initial, {1, temperature, 0.5F, true}, b, kRows - 1,
                        stops[0], stops[1]);
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
  sampling.addVerify(verify, policies, b, stops[0], stops[1]);
  static_cast<void>(backend.submitCommand(verify.dispatches()));
  for (uint32_t row = 0; row < rows; ++row) {
    const uint32_t maskRow = row / kRows * (kRows + 1) + row % kRows + 1;
    const uint32_t id = expected(row, maskRow);
    check(row, id);
    require(static_cast<uint32_t *>(b.outputTokens.contents())[row] == id,
            "batched target differs from masked CPU argmax");
  }
  const SamplingPolicy greedy{1, 0.0F, 1.0F, false};
  poison();
  CommandGraph initialArgmax;
  sampling.addInitial(initialArgmax, greedy, b, kRows - 1, stops[0], stops[1]);
  static_cast<void>(backend.submitCommand(initialArgmax.dispatches()));
  require(static_cast<uint32_t *>(b.outputTokens.contents())[0] ==
              expected(kRows - 1, kUnmasked),
          "initial argmax differs from CPU argmax");
  poison();
  CommandGraph verifyArgmax;
  sampling.addVerify(verifyArgmax, std::vector<SamplingPolicy>(lanes, greedy), b,
                     stops[0], stops[1]);
  static_cast<void>(backend.submitCommand(verifyArgmax.dispatches()));
  for (uint32_t row = 0; row < rows; ++row)
    require(static_cast<uint32_t *>(b.outputTokens.contents())[row] ==
                expected(row, kUnmasked),
            "batched argmax differs from CPU argmax");
}

// A lane that ignores end-of-sequence never selects a stop token: not in the
// sample after a prefill chunk, not in any verify row, greedy or sampled,
// alone or batched beside lanes that keep them, and a stop token its draft
// proposes is rejected. The stop tokens lead every row, so a lane that keeps
// them selects one; the other lanes take the row's best remaining token.
void excludedStopTokens(MetalBackend &backend, uint32_t vocabulary) {
  const uint32_t rows = kLanes * kRows;
  const auto space = Sampling::workspace(rows);
  const auto stops = stopTokens(vocabulary);
  Sampling sampling(backend, vocabulary, kRows);
  SamplingBuffers b{
      allocate(backend, uint64_t{rows} * vocabulary * sizeof(float)),
      allocate(backend, space.partialIdsBytes),
      allocate(backend, space.partialValuesBytes),
      allocate(backend, space.topIdsBytes),
      allocate(backend, space.topProbabilitiesBytes),
      allocate(backend, uint64_t{kLanes} * 2 * kRows * sizeof(float)),
      allocate(backend, uint64_t{kLanes} * (kRows + 1) * ((vocabulary + 31) / 32) * 4),
      allocate(backend, uint64_t{rows} * sizeof(uint32_t)),
      allocate(backend, space.argmaxValuesBytes),
      allocate(backend, space.argmaxIndicesBytes)};
  const auto best = [&](uint32_t row) { return 5 + row * 7919 % (vocabulary - 8); };
  auto *logits = static_cast<float *>(b.logits.contents());
  for (uint32_t row = 0; row < rows; ++row) {
    float *values = logits + uint64_t{row} * vocabulary;
    for (uint32_t token = 0; token < vocabulary; ++token)
      values[token] = -4.0F + float(token % 7) * 0.01F;
    values[best(row)] = 8.0F;
    values[stops[0]] = 12.0F;
    values[stops[1]] = 11.0F;
  }
  auto *uniforms = static_cast<float *>(b.uniforms.contents());
  const auto *tokens = static_cast<const uint32_t *>(b.outputTokens.contents());
  const auto *ids = static_cast<const uint32_t *>(b.topIds.contents());
  const auto *probabilities =
      static_cast<const float *>(b.topProbabilities.contents());
  const auto stop = [&](uint32_t token) {
    return token == stops[0] || token == stops[1];
  };
  // Whether a row's sparse target distribution gives a stop token any mass.
  const auto stopMass = [&](uint32_t row) {
    for (uint32_t rank = 0; rank < kTargetSamplingCandidates; ++rank) {
      const uint64_t index = uint64_t{row} * kTargetSamplingCandidates + rank;
      if (stop(ids[index]) && probabilities[index] > 0.0F)
        return true;
    }
    return false;
  };
  auto poison = [&] {
    for (const auto &buffer : {b.partialIds, b.partialValues, b.topIds,
                              b.topProbabilities, b.outputTokens})
      std::memset(buffer.contents(), 0xA5, buffer.sizeBytes());
  };

  for (const bool excludes : {false, true}) {
    for (const float temperature : {0.0F, 0.8F}) {
      for (const float uniform : {0.0F, 0.5F, 0.999F}) {
        poison();
        uniforms[0] = uniform;
        CommandGraph initial;
        sampling.addInitial(initial, {32, temperature, 1.0F, false, excludes}, b,
                            kRows - 1, stops[0], stops[1]);
        static_cast<void>(backend.submitCommand(initial.dispatches()));
        if (excludes)
          require(tokens[0] == best(kRows - 1) ||
                      (temperature > 0.0F && tokens[0] < vocabulary &&
                       !stop(tokens[0])),
                  "an excluding lane selected a stop token after prefill");
        else
          require(stop(tokens[0]), "the initial sample skipped the stop tokens");
      }
    }
  }

  // Drafts propose the first stop token, with all of their mass on it.
  const auto proposalSpace = Sampling::draftWorkspace(kLanes * kPositions);
  AcceptanceBuffers acceptance{
      allocate(backend, uint64_t{kLanes} * kPositions * sizeof(uint32_t)),
      allocate(backend, proposalSpace.candidatesBytes),
      allocate(backend, proposalSpace.proposalProbabilitiesBytes),
      b.topIds,
      b.topProbabilities,
      b.uniforms,
      b.outputTokens,
      allocate(backend, kLanes * sizeof(uint32_t)),
      allocate(backend, kLanes * sizeof(uint32_t)),
      allocate(backend, kLanes * sizeof(uint32_t))};
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    static_cast<uint32_t *>(acceptance.proposedTokens.contents())[lane * kPositions] =
        stops[0];
    static_cast<uint32_t *>(acceptance.candidates.contents())
        [uint64_t{lane} * kPositions * kCandidates] = stops[0];
    static_cast<float *>(acceptance.proposalProbabilities.contents())
        [uint64_t{lane} * kPositions * kCandidates] = 1.0F;
  }
  std::fill(uniforms, uniforms + kLanes * 2 * kRows, 0.5F);
  const std::array<uint32_t, kLanes> maximumRetained{kRows, kRows, kRows, kRows};
  for (uint32_t lanes = 1; lanes <= kLanes; ++lanes) {
    for (uint32_t excludeMask = 0; excludeMask < (1U << lanes); ++excludeMask) {
      for (const uint32_t samplingMask : {0U, 0b0101U, 0b1111U}) {
        std::vector<SamplingPolicy> policies;
        for (uint32_t lane = 0; lane < lanes; ++lane)
          policies.push_back({32, (samplingMask >> lane & 1U) ? 0.8F : 0.0F,
                              1.0F, false, (excludeMask >> lane & 1U) != 0});
        poison();
        CommandGraph verify;
        sampling.addVerify(verify, policies, b, stops[0], stops[1]);
        static_cast<void>(backend.submitCommand(verify.dispatches()));
        for (uint32_t row = 0; row < lanes * kRows; ++row) {
          const SamplingPolicy &policy = policies[row / kRows];
          if (policy.samples())
            require(stopMass(row) != policy.excludesStopTokens,
                    "a verify distribution mishandled the stop tokens");
          else
            require(tokens[row] ==
                        (policy.excludesStopTokens ? best(row) : stops[0]),
                    "a greedy verify row mishandled the stop tokens");
        }
        CommandGraph accept;
        sampling.addAcceptance(accept, acceptance,
                               std::span(maximumRetained).first(lanes), policies,
                               stops[0], stops[1]);
        static_cast<void>(backend.submitCommand(accept.dispatches()));
        const auto *retained =
            static_cast<const uint32_t *>(acceptance.retainedCounts.contents());
        const auto *anchors =
            static_cast<const uint32_t *>(acceptance.nextAnchors.contents());
        for (uint32_t lane = 0; lane < lanes; ++lane) {
          const uint32_t *output = tokens + lane * kRows;
          if (!policies[lane].excludesStopTokens) {
            require(retained[lane] == 1 && output[0] == stops[0] &&
                        anchors[lane] == stops[0],
                    "the target did not accept the drafted stop token");
            continue;
          }
          require(retained[lane] >= 1 && retained[lane] <= kRows &&
                      !stop(anchors[lane]),
                  "an excluding lane accepted a stop token");
          for (uint32_t index = 0; index < retained[lane]; ++index)
            require(output[index] < vocabulary && !stop(output[index]),
                    "an excluding lane accepted a stop token");
        }
      }
    }
  }
}

using splash::test::draft_selector::conditional;
using splash::test::draft_selector::expectedLength;
using splash::test::draft_selector::referenceLookahead;
using Score = splash::test::draft_selector::Score<double>;

// Direct randomized fp32 tables isolate the DP from the top-k and edge
// kernels. Poison the unused position-0 rows; they must never affect a path.
// Exact ties and sparse/all-excluded rows keep the lowest candidate index.
void lookaheadTables(MetalBackend &backend, uint32_t lanes) {
  const uint32_t positions = lanes * kPositions;
  const auto workspace = Sampling::draftWorkspace(positions);
  MetalBuffer candidates = allocate(backend, workspace.candidatesBytes);
  MetalBuffer unary = allocate(backend, workspace.unaryBytes);
  MetalBuffer partials = allocate(backend, workspace.partialValuesBytes);
  MetalBuffer uniforms = allocate(backend, lanes * 2 * kRows * sizeof(float));
  MetalBuffer tokens = allocate(backend, positions * sizeof(uint32_t));
  MetalBuffer probabilities = allocate(backend, workspace.proposalProbabilitiesBytes);
  auto *ids = static_cast<uint32_t *>(candidates.contents());
  auto *scores = static_cast<float *>(unary.contents());
  float *allTables = static_cast<float *>(partials.contents()) +
                    positions * SPLASH_DRAFT_SAMPLING_SHARDS * kCandidates;
  for (uint32_t trial = 0; trial < 32; ++trial) {
    Random random(0xabc + trial * kLanes + lanes);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint32_t first = lane * kPositions;
      float *tables = allTables + first * kCandidates * kCandidates;
      for (uint32_t p = 0; p < kPositions; ++p) {
        for (uint32_t c = 0; c < kCandidates; ++c) {
          ids[(first + p) * kCandidates + c] =
              1000 + (first + p) * kCandidates + c;
          scores[(first + p) * kCandidates + c] =
              trial == 0 ? 0.0F : trial == 1 && c > 2 ? -INFINITY
                                                        : random.unit() * 3.0F;
        }
        for (uint32_t s = 0; s < kCandidates; ++s)
          for (uint32_t c = 0; c < kCandidates; ++c)
            tables[(p * kCandidates + s) * kCandidates + c] =
                p == 0 && s > 0 ? NAN
                : trial == 0 ? 0.0F
                : trial == 2 && p == kPositions - 1 ? -INFINITY
                                                  : random.unit() * 3.0F;
      }
    }
    SelectorBatchParams params{};
    params.lanes = lanes;
    params.vocabulary = 1 << 20;
    std::memset(probabilities.contents(), 0xA5, probabilities.sizeBytes());
    CommandGraph graph;
    graph.add("draft_select_dflash_lookahead",
              {candidates, unary, partials, uniforms, tokens, probabilities},
              params, {lanes, 1, 1}, {1, 1, 1});
    static_cast<void>(backend.submitCommand(graph.dispatches()));
    const auto *proposed = static_cast<const uint32_t *>(tokens.contents());
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const uint32_t first = lane * kPositions;
      const float *tables = allTables + first * kCandidates * kCandidates;
      const splash::test::draft_selector::Score<float> score =
          [&](uint32_t p, uint32_t s, uint32_t c) {
            require(p != 0 || s == 0, "reference read an unused anchor row");
            return scores[(first + p) * kCandidates + c] +
                   tables[(p * kCandidates + s) * kCandidates + c];
          };
      const auto [best, path] = referenceLookahead(score, kPositions, kCandidates);
      static_cast<void>(best);
      for (uint32_t p = 0; p < kPositions; ++p)
        require(proposed[first + p] == ids[(first + p) * kCandidates + path[p]],
                "look-ahead tokens differ from the fp32 reference DP");
      if (trial == 0)
        require(std::all_of(path.begin(), path.end(), [](uint32_t c) { return c == 0; }),
                "look-ahead tie did not keep the lowest candidate");
    }
    const auto *bytes = static_cast<const unsigned char *>(probabilities.contents());
    require(std::all_of(bytes, bytes + probabilities.sizeBytes(),
                        [](unsigned char byte) { return byte == 0xA5; }),
            "look-ahead greedy lanes wrote sampling probabilities");
  }
}

// One thread per lane: the selection kernel alone over a crafted table where
// the best first candidate leads nowhere and the second starts a chain the
// selector is sure of. Greedy follows the first; look-ahead takes the chain.
void lookaheadCraftedTable(MetalBackend &backend) {
  constexpr uint32_t shards = SPLASH_DRAFT_SAMPLING_SHARDS;
  const auto workspace = Sampling::draftWorkspace(kPositions);
  MetalBuffer candidates = allocate(backend, workspace.candidatesBytes);
  MetalBuffer unary = allocate(backend, workspace.unaryBytes);
  MetalBuffer partials = allocate(backend, workspace.partialValuesBytes);
  MetalBuffer uniforms = allocate(backend, 2 * kRows * sizeof(float));
  MetalBuffer tokens = allocate(backend, kPositions * sizeof(uint32_t));
  MetalBuffer probabilities = allocate(backend, workspace.proposalProbabilitiesBytes);
  auto *ids = static_cast<uint32_t *>(candidates.contents());
  auto *scores = static_cast<float *>(unary.contents());
  float *tables = static_cast<float *>(partials.contents()) +
                  kPositions * shards * kCandidates;
  for (uint32_t position = 0; position < kPositions; ++position) {
    for (uint32_t c = 0; c < kCandidates; ++c) {
      ids[position * kCandidates + c] = 1000 + position * kCandidates + c;
      scores[position * kCandidates + c] = 0.0F;
    }
    // From predecessor 1, candidate 1 is all but certain; every other
    // predecessor spreads uniformly.
    if (position > 0)
      tables[(position * kCandidates + 1) * kCandidates + 1] = 10.0F;
  }
  scores[0] = 1.0F;
  scores[1] = 0.5F;
  for (uint32_t c = 2; c < kCandidates; ++c)
    scores[c] = -10.0F;

  SelectorBatchParams params{};
  params.lanes = 1;
  params.vocabulary = 1 << 20;
  for (const bool lookahead : {false, true}) {
    std::memset(tokens.contents(), 0xFF, tokens.sizeBytes());
    CommandGraph graph;
    graph.add(lookahead ? "draft_select_dflash_lookahead" : "draft_select_dflash",
              {candidates, unary, partials, uniforms, tokens, probabilities},
              params, {1, 1, 1}, {1, 1, 1});
    static_cast<void>(backend.submitCommand(graph.dispatches()));
    const auto *proposed = static_cast<const uint32_t *>(tokens.contents());
    for (uint32_t position = 0; position < kPositions; ++position)
      require(proposed[position] ==
                  1000 + position * kCandidates + (lookahead ? 1 : 0),
              lookahead ? "look-ahead selection missed the confident chain"
                        : "greedy selection changed on the crafted table");
  }
}

// The production pipeline in look-ahead mode. Greedy lanes must reach the
// reference optimum over the tables the edge kernel actually wrote; sampling
// lanes must match greedy mode bit for bit, tokens and probabilities.
// Returns how many greedy lanes look-ahead moved off the greedy path.
uint32_t lookaheadPipeline(MetalBackend &backend, uint32_t vocabulary,
                           uint32_t lanes, uint32_t samplingMask) {
  uint32_t changed = 0;
  Random random(0x10c4 + vocabulary + lanes * 16 + samplingMask);
  const uint32_t rows = lanes * kRows;
  const uint32_t positions = lanes * kPositions;
  const auto workspace = Sampling::draftWorkspace(positions);
  Sampling sampling(backend, vocabulary, kRows);
  require(sampling.draftSelection() == DraftSelection::Greedy,
          "draft selection does not default to greedy");

  MetalBuffer logits = allocate(backend, uint64_t{rows} * vocabulary * sizeof(float));
  auto *logitRows = static_cast<float *>(logits.contents());
  const std::array patterns{Pattern::Peaked, Pattern::Uniform, Pattern::Ties,
                            Pattern::Sparse};
  for (uint32_t row = 0; row < rows; ++row)
    fillRow(logitRows + uint64_t{row} * vocabulary, vocabulary,
            patterns[(row / kRows + row % kRows) % patterns.size()], random);
  DraftSelectorBuffers buffers{
      logits,
      allocate(backend, workspace.partialIdsBytes),
      allocate(backend, workspace.partialValuesBytes),
      allocate(backend, workspace.candidatesBytes),
      allocate(backend, workspace.unaryBytes),
      // Larger codebook values than runCase: edges that move the choice.
      randomBfloat(backend, uint64_t{rows} * kRank, random, 0.5F),
      randomBfloat(backend, uint64_t{vocabulary} * kRank, random, 0.5F),
      randomBfloat(backend, uint64_t{vocabulary} * kRank, random, 0.5F),
      allocate(backend, uint64_t{lanes} * 2 * kRows * sizeof(float)),
      allocate(backend, uint64_t{positions} * sizeof(uint32_t)),
      allocate(backend, workspace.proposalProbabilitiesBytes)};
  auto *uniforms = static_cast<float *>(buffers.uniforms.contents());
  for (uint32_t index = 0; index < lanes * 2 * kRows; ++index)
    uniforms[index] = (random.unit() + 1.0F) * 0.5F;
  std::vector<uint32_t> anchors(lanes);
  std::vector<SamplingPolicy> policies(lanes);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    anchors[lane] = random.next() % vocabulary;
    policies[lane] =
        SamplingPolicy{16, (samplingMask >> lane & 1U) ? 0.8F : 0.0F, 1.0F, false};
  }

  auto run = [&](DraftSelection selection) {
    sampling.setDraftSelection(selection);
    std::memset(buffers.proposedTokens.contents(), 0xFF,
                buffers.proposedTokens.sizeBytes());
    std::memset(buffers.proposalProbabilities.contents(), 0xFF,
                buffers.proposalProbabilities.sizeBytes());
    CommandGraph graph;
    sampling.addDraftSelector(graph, buffers, anchors, policies, kPositions);
    require(graph.dispatches().size() == 3,
            "draft selector dispatch count changed");
    require(graph.dispatches()[2].pipelineName ==
                (selection == DraftSelection::Lookahead
                     ? "draft_select_dflash_lookahead"
                     : "draft_select_dflash"),
            "draft selection chose the wrong pipeline");
    static_cast<void>(backend.submitCommand(graph.dispatches()));
    const auto *tokens =
        static_cast<const uint32_t *>(buffers.proposedTokens.contents());
    const auto *probabilities =
        static_cast<const float *>(buffers.proposalProbabilities.contents());
    return std::pair{
        std::vector<uint32_t>(tokens, tokens + positions),
        std::vector<float>(probabilities,
                           probabilities + uint64_t{positions} * kCandidates)};
  };
  const auto greedy = run(DraftSelection::Greedy);
  const auto lookahead = run(DraftSelection::Lookahead);
  const auto restored = run(DraftSelection::Greedy);
  require(restored.first == greedy.first &&
              std::memcmp(restored.second.data(), greedy.second.data(),
                          greedy.second.size() * sizeof(float)) == 0,
          "restoring greedy mode changed tokens or probabilities");

  constexpr uint32_t shards = SPLASH_DRAFT_SAMPLING_SHARDS;
  const auto *candidates =
      static_cast<const uint32_t *>(buffers.candidates.contents());
  const auto *unary = static_cast<const float *>(buffers.unary.contents());
  const float *allTables =
      static_cast<const float *>(buffers.partialValues.contents()) +
      uint64_t{lanes} * kPositions * shards * kCandidates;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const uint64_t first = uint64_t{lane} * kPositions;
    if (policies[lane].samples()) {
      require(std::equal(greedy.first.begin() + first,
                         greedy.first.begin() + first + kPositions,
                         lookahead.first.begin() + first),
              "look-ahead mode changed a sampling lane's draw");
      require(std::memcmp(greedy.second.data() + first * kCandidates,
                          lookahead.second.data() + first * kCandidates,
                          kPositions * kCandidates * sizeof(float)) == 0,
              "look-ahead mode changed a sampling lane's probabilities");
      continue;
    }
    // The kernel adds fp32 unary and edge scores; evaluate the same sums.
    const float *tables = allTables + uint64_t{lane} * kPositions * kCandidates * kCandidates;
    const Score score = [&](uint32_t p, uint32_t s, uint32_t c) {
      return double(unary[(first + p) * kCandidates + c] +
                    tables[(p * kCandidates + s) * kCandidates + c]);
    };
    std::vector<uint32_t> path(kPositions);
    for (uint32_t position = 0; position < kPositions; ++position) {
      const uint32_t token = lookahead.first[first + position];
      uint32_t selected = kCandidates;
      for (uint32_t c = 0; c < kCandidates && selected == kCandidates; ++c)
        if (candidates[(first + position) * kCandidates + c] == token)
          selected = c;
      require(selected < kCandidates,
              "look-ahead proposed a token outside its candidates");
      path[position] = selected;
    }
    const auto [best, reference] = referenceLookahead(score, kPositions, kCandidates);
    const double actual = expectedLength(score, path, kCandidates);
    // fp32 softmax and products differ from double only near exact ties.
    require(actual >= best - 1e-4 * (1.0 + best),
            "look-ahead path is worse than the reference optimum");
    // Greedy lanes in greedy mode keep the greedy walk.
    std::vector<uint32_t> greedyPath(kPositions);
    uint32_t predecessor = 0;
    for (uint32_t position = 0; position < kPositions; ++position) {
      uint32_t selected = 0;
      for (uint32_t c = 1; c < kCandidates; ++c)
        if (score(position, predecessor, c) > score(position, predecessor, selected))
          selected = c;
      greedyPath[position] = selected;
      predecessor = selected;
      require(greedy.first[first + position] ==
                  candidates[(first + position) * kCandidates + selected],
              "greedy mode no longer walks greedily");
    }
    require(actual >= expectedLength(score, greedyPath, kCandidates) - 1e-4,
            "look-ahead path is worse than the greedy walk");
    changed += path != greedyPath ? 1U : 0U;
  }
  return changed;
}

void invalidRequests(MetalBackend &backend) {
  Sampling sampling(backend, 1024, kRows);
  const auto workspace = Sampling::draftWorkspace(kPositions);
  DraftSelectorBuffers buffers{
      allocate(backend, uint64_t{kRows} * 1024 * sizeof(float)),
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
  // The kernels compile the proposal count and a lane's rows in.
  for (const uint32_t proposals : {0U, kPositions - 1, kPositions + 1})
    rejects([&] {
      sampling.addDraftSelector(graph, buffers, std::span(anchors).first(1),
                                policies, proposals);
    });
  for (const uint32_t rows : {0U, kRows - 1, kRows + 1})
    rejects([&] { (void)Sampling(backend, 1024, rows); });
  require(graph.empty(), "invalid draft selector request encoded a graph");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: draft-selector METALLIB");
    MetalBackend backend(argv[1]);
    invalidRequests(backend);
    for (const uint32_t vocabulary : {1003U, 248320U}) {
      for (uint32_t lanes = 1; lanes <= kLanes; ++lanes)
        targetTop1(backend, vocabulary, lanes);
      excludedStopTokens(backend, vocabulary);
    }
    for (uint32_t lanes = 1; lanes <= kLanes; ++lanes)
      for (uint32_t mask = 0; mask < (1U << lanes); ++mask)
        mixedVerify(backend, lanes, mask);
    for (const uint32_t vocabulary : {248320U, 1003U, 270005U}) {
      for (uint32_t lanes = 1; lanes <= kLanes; ++lanes) {
        runCase(backend, {vocabulary, lanes, false});
        runCase(backend, {vocabulary, lanes, true});
      }
    }
    lookaheadCraftedTable(backend);
    for (uint32_t lanes = 1; lanes <= kLanes; ++lanes)
      lookaheadTables(backend, lanes);
    uint32_t changed = 0;
    for (const uint32_t vocabulary : {248320U, 1003U})
      for (uint32_t lanes = 1; lanes <= kLanes; ++lanes)
        for (uint32_t mask = 0; mask < (1U << lanes); ++mask)
          changed += lookaheadPipeline(backend, vocabulary, lanes, mask);
    // The random tables are not built to favour look-ahead; it must still
    // leave the greedy path somewhere, or the pipeline cases test nothing.
    require(changed > 0, "look-ahead never left the greedy path");
    std::cout << "draft_selector_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "draft_selector_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}

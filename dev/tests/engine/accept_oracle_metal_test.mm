// Enumerate the speculative output marginal through the production verify
// and acceptance kernels. For each possible draft token, integrate the
// accept/reject intervals and the correction distribution, then require the
// weighted position-0 output to equal the normalized target distribution.
// The fixtures' residual probabilities are multiples of 1/correctionBins,
// so midpoint enumeration integrates every correction interval exactly,
// regardless of the kernel's vocabulary/candidate traversal order.
#include "TestBuffers.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Sampling.h"
#include "ops/Sampling.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::test::require;
using namespace splash::ops;

constexpr uint32_t kRows = SPLASH_TARGET_VERIFY_ROWS;
constexpr uint32_t kProposals = SPLASH_DRAFT_PROPOSAL_TOKENS;
constexpr uint32_t kCandidates = SPLASH_DRAFT_CANDIDATES;
constexpr uint32_t kUniforms = SPLASH_SAMPLING_UNIFORMS;
constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kVocabulary = 4099; // Odd size, crossing vocabulary shards.
constexpr uint32_t kSentinel = 0xFFFFFFFFU;
constexpr double kTolerance = 2e-6;
using Support = std::vector<std::pair<uint32_t, double>>;

struct OracleConfig final {
  const char *name;
  Support draft;
  Support target;
  uint32_t correctionBins;
};

double probability(const Support &support, uint32_t token) {
  for (const auto &[id, mass] : support)
    if (id == token)
      return mass;
  return 0.0;
}

void validateSupport(const Support &support) {
  double total = 0.0;
  std::vector<uint32_t> ids;
  for (const auto &[id, mass] : support) {
    require(id < kVocabulary && mass > 0.0 && mass <= 1.0,
            "invalid oracle support");
    require(std::find(ids.begin(), ids.end(), id) == ids.end(),
            "duplicate oracle token");
    ids.push_back(id);
    total += mass;
  }
  require(std::fabs(total - 1.0) < 1e-12, "unnormalized oracle support");
}

struct Trial final {
  uint32_t proposal;
  float acceptance;
  float correction;
  double weight;
};

// Partition acceptance at min(1,p/q), with one weighted midpoint on either
// side. Correction bins have equal mass; the fixture check below ensures
// no bin straddles a residual boundary. No random trials or statistical
// tolerances are involved.
std::vector<Trial> enumerate(const OracleConfig &config) {
  validateSupport(config.draft);
  validateSupport(config.target);
  require(config.draft.size() <= kCandidates && config.correctionBins,
          "invalid oracle geometry");
  std::vector<double> residual;
  for (const auto &[token, p] : config.target)
    residual.push_back(std::max(0.0, p - probability(config.draft, token)));
  double total = std::accumulate(residual.begin(), residual.end(), 0.0);
  if (total == 0.0) {
    for (size_t index = 0; index < residual.size(); ++index)
      residual[index] = config.target[index].second;
    total = 1.0;
  }
  for (double mass : residual) {
    const double bins = mass / total * config.correctionBins;
    require(std::fabs(bins - std::round(bins)) < 1e-10,
            "correction grid does not integrate this fixture exactly");
  }
  std::vector<Trial> trials;
  for (const auto &[token, q] : config.draft) {
    const double cut = std::min(1.0, probability(config.target, token) / q);
    for (const auto &[low, high] : {std::pair{0.0, cut}, std::pair{cut, 1.0}}) {
      if (low == high)
        continue;
      for (uint32_t bin = 0; bin < config.correctionBins; ++bin)
        trials.push_back({token, static_cast<float>((low + high) / 2.0),
                          (float(bin) + 0.5F) / config.correctionBins,
                          q * (high - low) / config.correctionBins});
    }
  }
  return trials;
}

class Fixture final {
public:
  explicit Fixture(MetalBackend &backend) : sampling_(kVocabulary) {
    const uint32_t rows = kLanes * kRows;
    const auto workspace = Sampling::workspace(rows);
    const auto allocate = [&](uint64_t bytes) {
      MetalBuffer buffer = splash::test::sharedBuffer(backend, bytes);
      std::memset(buffer.contents(), 0, bytes);
      return buffer;
    };
    buffers_ = {
        allocate(uint64_t{rows} * kVocabulary * sizeof(float)),
        allocate(workspace.partialMassesBytes),
        allocate(workspace.vocabularyRowsBytes),
        allocate(uint64_t{kLanes} * kUniforms * sizeof(float)),
        allocate(uint64_t{kLanes} * (kRows + 1) * ((kVocabulary + 31) / 32) *
                 sizeof(uint32_t)), // Bound but unread without constraints.
        allocate(uint64_t{rows} * sizeof(uint32_t)),
        allocate(workspace.argmaxValuesBytes),
        allocate(workspace.argmaxIndicesBytes),
        allocate(uint64_t{rows} * sizeof(uint32_t)),
        allocate(uint64_t{kLanes} * kProposals * kCandidates *
                 sizeof(uint32_t)),
        allocate(uint64_t{kLanes} * kProposals * kCandidates * sizeof(float)),
        allocate(workspace.vocabularyRangesBytes),
        allocate(workspace.vocabularyArrivalsBytes)};
    acceptance_ = {allocate(uint64_t{kLanes} * kProposals * sizeof(uint32_t)),
                   buffers_.draftCandidates,
                   buffers_.draftProbabilities,
                   buffers_.vocabularyRows,
                   buffers_.uniforms,
                   buffers_.outputTokens,
                   allocate(uint64_t{kLanes} * sizeof(uint32_t)),
                   allocate(uint64_t{kLanes} * sizeof(uint32_t))};
  }

  void configure(const OracleConfig &config) {
    target_ = config.target;
    auto *logits = static_cast<float *>(buffers_.logits.contents());
    auto *ids = static_cast<uint32_t *>(buffers_.draftCandidates.contents());
    auto *q = static_cast<float *>(buffers_.draftProbabilities.contents());
    for (uint32_t row = 0; row < kLanes * kRows; ++row) {
      float *values = logits + uint64_t{row} * kVocabulary;
      std::fill_n(values, kVocabulary, -INFINITY);
      for (const auto &[token, mass] : config.target)
        values[token] = static_cast<float>(std::log(mass));
    }
    for (uint32_t position = 0; position < kLanes * kProposals; ++position) {
      for (uint32_t slot = 0; slot < kCandidates; ++slot) {
        const uint32_t index = position * kCandidates + slot;
        ids[index] =
            slot < config.draft.size() ? config.draft[slot].first : kSentinel;
        q[index] = slot < config.draft.size()
                       ? static_cast<float>(config.draft[slot].second)
                       : 0.0F;
      }
    }
  }

  // Return verify draws too: p=q acceptance always succeeds and would hide
  // a broken zero-residual fallback if only the final marginal were checked.
  std::vector<uint32_t> run(MetalBackend &backend,
                            const std::vector<Trial> &trials,
                            std::vector<uint32_t> *corrections = nullptr) {
    const uint32_t lanes = static_cast<uint32_t>(trials.size());
    require(lanes && lanes <= kLanes, "invalid trial batch");
    auto *input = static_cast<uint32_t *>(buffers_.inputTokens.contents());
    auto *proposed =
        static_cast<uint32_t *>(acceptance_.proposedTokens.contents());
    auto *uniforms = static_cast<float *>(buffers_.uniforms.contents());
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      input[lane * kRows] = 0;
      for (uint32_t position = 0; position < kProposals; ++position) {
        input[lane * kRows + position + 1] = trials[lane].proposal;
        proposed[lane * kProposals + position] = trials[lane].proposal;
      }
      std::fill_n(uniforms + lane * kUniforms, kUniforms, 0.9375F);
      uniforms[lane * kUniforms + SPLASH_UNIFORM_ACCEPTANCE] =
          trials[lane].acceptance;
      uniforms[lane * kUniforms + SPLASH_UNIFORM_CORRECTION] =
          trials[lane].correction;
    }
    // Poison selection results; arrival counts start and finish at zero.
    for (const auto &buffer :
         {buffers_.partialMasses, buffers_.vocabularyRows,
          buffers_.vocabularyRanges, buffers_.outputTokens,
          acceptance_.retainedCounts, acceptance_.acceptedCounts})
      std::memset(buffer.contents(), 0xFF, buffer.sizeBytes());
    const std::vector<SamplingPolicy> policies(lanes, {0, 1.0F, 1.0F, false});
    const std::vector<uint32_t> maximumRetained(lanes, kRows);
    CommandGraph verify;
    sampling_.addVerify(verify, policies, buffers_, 1, 2, {});
    static_cast<void>(backend.submitCommand(verify.dispatches()));
    const auto *tokens =
        static_cast<const uint32_t *>(buffers_.outputTokens.contents());
    const auto *records = static_cast<const TargetVocabularyRow *>(
        buffers_.vocabularyRows.contents());
    if (corrections)
      for (uint32_t lane = 0; lane < lanes; ++lane)
        corrections->push_back(tokens[lane * kRows]);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      const float actual = records[lane * kRows].draft_probability;
      const double expected = probability(target_, trials[lane].proposal);
      require(std::isfinite(actual) &&
                  std::fabs(actual - expected) < kTolerance,
              "verify wrote the wrong proposal probability");
    }
    CommandGraph accept;
    sampling_.addAcceptance(accept, acceptance_, maximumRetained, policies, 1,
                            2);
    static_cast<void>(backend.submitCommand(accept.dispatches()));
    const auto *counts =
        static_cast<const uint32_t *>(acceptance_.retainedCounts.contents());
    std::vector<uint32_t> outputs;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      require(counts[lane] > 0 && counts[lane] <= kRows,
              "acceptance did not retain an output");
      outputs.push_back(tokens[lane * kRows]);
    }
    return outputs;
  }

  uint32_t accepted(uint32_t lane) const {
    return static_cast<const uint32_t *>(
        acceptance_.acceptedCounts.contents())[lane];
  }

private:
  Sampling sampling_;
  SamplingBuffers buffers_;
  AcceptanceBuffers acceptance_;
  Support target_;
};

void checkMarginal(MetalBackend &backend, const OracleConfig &config,
                   bool checkFallback = false) {
  const auto trials = enumerate(config);
  Fixture fixture(backend);
  fixture.configure(config);
  std::vector<double> marginal(kVocabulary, 0.0);
  std::vector<double> fallback(kVocabulary, 0.0);
  for (size_t begin = 0; begin < trials.size(); begin += kLanes) {
    const size_t end = std::min(begin + kLanes, trials.size());
    const std::vector<Trial> batch(trials.begin() + begin,
                                   trials.begin() + end);
    std::vector<uint32_t> corrections;
    const auto outputs = fixture.run(backend, batch, &corrections);
    for (size_t lane = 0; lane < batch.size(); ++lane) {
      require(outputs[lane] < kVocabulary && corrections[lane] < kVocabulary,
              std::string(config.name) + ": output outside the vocabulary");
      marginal[outputs[lane]] += batch[lane].weight;
      fallback[corrections[lane]] += batch[lane].weight;
    }
  }
  for (uint32_t token = 0; token < kVocabulary; ++token) {
    const double expected = probability(config.target, token);
    const std::string message =
        std::string(config.name) + " token " + std::to_string(token);
    require(std::fabs(marginal[token] - expected) < kTolerance,
            message + ": enumerated marginal " +
                std::to_string(marginal[token]) + ", expected " +
                std::to_string(expected));
    if (checkFallback)
      require(std::fabs(fallback[token] - expected) < kTolerance,
              message + ": zero-residual draw did not recover target");
  }
}

void checkZeroDraftProbability(MetalBackend &backend) {
  const OracleConfig config{"q-zero",
                            {{17, 0.5}, {43, 0.5}},
                            {{17, 0.25}, {43, 0.25}, {4096, 0.5}},
                            2};
  Fixture fixture(backend);
  fixture.configure(config);
  // This inconsistent proposal cannot be drawn from q; it is a defensive
  // branch check, not a target-marginal fixture.
  std::vector<Trial> trials;
  for (uint32_t lane = 0; lane < kLanes; ++lane)
    trials.push_back({4096, (float(lane) + 0.5F) / kLanes,
                      (float(kLanes - lane) - 0.5F) / kLanes, 0.0});
  for (uint32_t output : fixture.run(backend, trials))
    require(output == 4096, "q=0, p>0 proposal was not accepted");
  // Its residual also draws 4096, so the token alone cannot prove acceptance.
  for (uint32_t lane = 0; lane < kLanes; ++lane)
    require(fixture.accepted(lane) == kProposals,
            "q=0, p>0 proposal was corrected instead of accepted");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: accept-oracle METALLIB");
    MetalBackend backend(argv[1]);
    // Accept sometimes, always and never; target-only support crosses shards,
    // while the candidate order differs from token order.
    checkMarginal(backend, {"interior",
                            {{1000, 0.25}, {17, 0.5}, {43, 0.25}},
                            {{17, 0.25}, {43, 0.375}, {4096, 0.375}},
                            4});
    // Uniform p=q makes the residual zero without rounding ambiguity.
    checkMarginal(backend,
                  {"zero-residual",
                   {{4096, 0.25}, {17, 0.25}, {1000, 0.25}, {43, 0.25}},
                   {{17, 0.25}, {43, 0.25}, {1000, 0.25}, {4096, 0.25}},
                   4},
                  true);
    // A 64-token target guards against resurrecting the former top-32 cap.
    Support wide;
    for (uint32_t token = 0; token < 64; ++token)
      wide.emplace_back(3 + token * 64, 1.0 / 64.0);
    checkMarginal(
        backend,
        {"wide", {{3, 0.25}, {67, 0.25}, {131, 0.25}, {195, 0.25}}, wide, 60});
    checkZeroDraftProbability(backend);
    std::cout << "accept_oracle_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "accept_oracle_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}

// DFlash acceptance oracle: decode_accept_dflash must reproduce the
// speculative-sampling marginal. For a fixed draft/target pair, the output
// distribution over draft draws and accept uniforms must equal the target
// distribution (Leviathan/Chen exactness), including the q=0 acceptance path
// and the zero-residual fallback. Three fixed configurations cover the
// interior accept/reject rule, deterministic q=0 acceptance, and the
// residual-total-zero branch. Histograms over thousands of trials must match
// the known marginals within a wide statistical tolerance (~4 sigma).
#include "metal/MetalBackend.hpp"
#include "ops/Sampling.hpp"

#import <Foundation/Foundation.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using namespace splash::ops;

constexpr uint32_t kRows = SPLASH_TARGET_VERIFY_ROWS;
constexpr uint32_t kProposals = SPLASH_DRAFT_PROPOSAL_TOKENS;
constexpr uint32_t kDraftWidth = 16;
constexpr uint32_t kTargetWidth = 32;
constexpr uint32_t kUniforms = 2 * SPLASH_TARGET_VERIFY_ROWS;
constexpr uint32_t kLanes = 4;
constexpr uint32_t kTrials = 2048;
constexpr double kTolerance = 0.04;
constexpr uint32_t kSentinel = 0xFFFFFFFFU;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
  // Uniform in [0, 1): the low 24 bits of the generator output.
  float unit() {
    return static_cast<float>(next() & 0xFFFFFF) / 16777216.0F;
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
      backend.allocateBuffer(bytes, BufferStorage::Shared, "accept oracle");
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

// One fixed draft/target pair plus the expected position-0 output marginal.
// Draft tokens are drawn from proposalTokens/proposalProbabilities on the
// host each trial; the kernel sees the same draw.
struct OracleConfig final {
  // (token, probability) support of the draft table; remaining slots carry
  // the sentinel id with zero probability.
  std::vector<std::pair<uint32_t, double>> draft;
  // (token, probability) support of the target table; same padding rule.
  std::vector<std::pair<uint32_t, double>> target;
  // How the position-0 draft token is chosen: drawn from the draft support,
  // or a deterministic token (which may lie outside the draft support to
  // exercise the q=0 path).
  uint32_t fixedToken = 0;
  bool useFixedToken = false;
  // Expected P(output token) at position 0.
  std::vector<std::pair<uint32_t, double>> expected;
};

uint32_t drawFromSupport(const std::vector<std::pair<uint32_t, double>> &support,
                         double uniform) {
  double cumulative = 0.0;
  for (const auto &[token, probability] : support) {
    cumulative += probability;
    if (uniform < cumulative)
      return token;
  }
  return support.back().first;
}

void fillSupport(uint32_t *ids, float *probabilities, uint32_t width,
                 const std::vector<std::pair<uint32_t, double>> &support) {
  for (uint32_t slot = 0; slot < width; ++slot) {
    if (slot < support.size()) {
      ids[slot] = support[slot].first;
      probabilities[slot] = static_cast<float>(support[slot].second);
    } else {
      ids[slot] = kSentinel;
      probabilities[slot] = 0.0F;
    }
  }
}

// Runs kTrials accept dispatches (kLanes trials per dispatch) and histograms
// the position-0 output token.
std::vector<std::pair<uint32_t, uint32_t>>
runOracle(MetalBackend &backend, const OracleConfig &config, uint64_t seed) {
  constexpr uint32_t vocabulary = 16;
  Sampling sampling(backend, vocabulary, kRows);
  const auto allocateLane = [&](uint64_t elements, uint64_t elementBytes) {
    return allocate(backend, kLanes * elements * elementBytes);
  };
  AcceptanceBuffers buffers{
      allocateLane(kProposals, sizeof(uint32_t)),
      allocateLane(kProposals * kDraftWidth, sizeof(uint32_t)),
      allocateLane(kProposals * kDraftWidth, sizeof(float)),
      allocateLane(kRows * kTargetWidth, sizeof(uint32_t)),
      allocateLane(kRows * kTargetWidth, sizeof(float)),
      allocateLane(kUniforms, sizeof(float)),
      allocateLane(kRows, sizeof(uint32_t)),
      allocateLane(1, sizeof(uint32_t)),
      allocateLane(1, sizeof(uint32_t)),
      allocateLane(1, sizeof(uint32_t)),
  };
  auto *draftTokens = static_cast<uint32_t *>(buffers.proposedTokens.contents());
  auto *draftIds = static_cast<uint32_t *>(buffers.candidates.contents());
  auto *draftProbs = static_cast<float *>(buffers.proposalProbabilities.contents());
  auto *targetIds = static_cast<uint32_t *>(buffers.targetTopIds.contents());
  auto *targetProbs = static_cast<float *>(buffers.targetTopProbabilities.contents());
  auto *uniforms = static_cast<float *>(buffers.uniforms.contents());
  auto *outputs = static_cast<uint32_t *>(buffers.outputTokens.contents());
  // The draft/target tables are identical at every position and row; only
  // position 0 is asserted, so later positions reuse the same tables with
  // independently drawn tokens and uniforms.
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    for (uint32_t position = 0; position < kProposals; ++position) {
      fillSupport(draftIds + (lane * kProposals + position) * kDraftWidth,
                  draftProbs + (lane * kProposals + position) * kDraftWidth,
                  kDraftWidth, config.draft);
    }
    for (uint32_t row = 0; row < kRows; ++row) {
      fillSupport(targetIds + (lane * kRows + row) * kTargetWidth,
                  targetProbs + (lane * kRows + row) * kTargetWidth,
                  kTargetWidth, config.target);
    }
  }
  const std::vector<SamplingPolicy> policies(
      kLanes, SamplingPolicy{4, 0.7F, 0.9F, false});
  const std::vector<uint32_t> maximumRetained(kLanes, kRows);
  std::vector<std::pair<uint32_t, uint32_t>> histogram;
  for (const auto &[token, ignored] : config.expected) {
    (void)ignored;
    histogram.emplace_back(token, 0);
  }
  Random random(seed);
  for (uint32_t trial = 0; trial < kTrials; trial += kLanes) {
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      for (uint32_t position = 0; position < kProposals; ++position) {
        const uint32_t token =
            config.useFixedToken && position == 0
                ? config.fixedToken
                : drawFromSupport(config.draft, random.unit());
        draftTokens[(lane * kProposals + position)] = token;
      }
      for (uint32_t slot = 0; slot < kUniforms; ++slot)
        uniforms[lane * kUniforms + slot] = random.unit();
      std::memset(outputs + lane * kRows, 0xFF, kRows * sizeof(uint32_t));
    }
    CommandGraph graph;
    sampling.addAcceptance(graph, buffers, maximumRetained, policies,
                           1000 /* stopToken0 */, 1001 /* stopToken1 */);
    static_cast<void>(backend.submitCommand(graph.dispatches()));
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      const uint32_t output = outputs[lane * kRows];
      bool counted = false;
      for (auto &[token, count] : histogram) {
        if (token == output) {
          ++count;
          counted = true;
          break;
        }
      }
      require(counted, "oracle output token outside the expected support");
    }
  }
  return histogram;
}

void checkMarginal(MetalBackend &backend, const OracleConfig &config,
                    uint64_t seed, const char *name) {
  const auto histogram = runOracle(backend, config, seed);
  for (const auto &[token, count] : histogram) {
    double expected = 0.0;
    for (const auto &[want, probability] : config.expected) {
      if (want == token) {
        expected = probability;
        break;
      }
    }
    const double actual = static_cast<double>(count) / kTrials;
    char message[160];
    std::snprintf(message, sizeof(message),
                  "%s: token %u marginal %.4f, expected %.4f", name, token,
                  actual, expected);
    require(std::fabs(actual - expected) < kTolerance, message);
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: accept-oracle METALLIB");
    MetalBackend backend(argv[1]);
    // Interior accept/reject: q={3:.5,4:.3,5:.2}, p={3:.2,4:.5,5:.3}.
    // Accept probs .4/1/1 give total accept .7; the residual over
    // {4:.2,5:.1}/.3 restores the target, so the marginal must equal p.
    checkMarginal(backend,
                  {{{3, 0.5}, {4, 0.3}, {5, 0.2}},
                   {{3, 0.2}, {4, 0.5}, {5, 0.3}},
                   0, false,
                   {{3, 0.2}, {4, 0.5}, {5, 0.3}}},
                  7129, "interior");
    // q=0 path: token 5 is proposed deterministically but absent from the
    // draft table while p(5) > 0, so u*0 < p accepts on every draw.
    checkMarginal(backend,
                  {{{3, 0.6}, {4, 0.4}},
                   {{3, 0.3}, {4, 0.3}, {5, 0.4}},
                   5, true,
                   {{5, 1.0}}},
                  5417, "q-zero");
    // Zero-residual fallback: q={3:.5,4:.5}, p={3:.3,4:.3}; every rejection
    // finds sum(max(p-q,0)) == 0, so the kernel must fall back to sampling
    // the target itself: accept .6 each, fallback 50/50.
    checkMarginal(backend,
                  {{{3, 0.5}, {4, 0.5}},
                   {{3, 0.3}, {4, 0.3}},
                   0, false,
                   {{3, 0.5}, {4, 0.5}}},
                  9371, "zero-residual");
    std::cout << "accept_oracle_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "accept_oracle_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}

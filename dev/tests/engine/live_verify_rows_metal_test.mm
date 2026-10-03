#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/LiveRows.h"
#include "metal/abi/MoE.h"
#include "metal/abi/Sampling.h"
#include "ops/DraftSelector.hpp"
#include "ops/Sampling.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace {
using namespace splash::metal;
using namespace splash::ops;
using splash::test::require;
constexpr uint32_t Positions = SPLASH_DRAFT_PROPOSAL_TOKENS;
constexpr uint32_t Candidates = SPLASH_DRAFT_CANDIDATES;
constexpr uint32_t Rows = SPLASH_TARGET_VERIFY_ROWS;

MetalBuffer allocate(MetalBackend &backend, uint64_t bytes) {
  auto buffer = backend.allocateBuffer(bytes, BufferStorage::Shared, "live verify test");
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

template <class F> void rejects(F function) {
  bool rejected = false;
  try { function(); } catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "invalid live-row configuration was accepted");
}

void confidence(MetalBackend &backend) {
  const DraftSelector selector(1024);
  const std::array<float, Positions> q{0.95F, 0.90F, 0.85F, 0.60F, 0.40F, 0.30F, 0.02F};
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    const auto workspace = DraftSelector::workspace(lanes * Positions);
    DraftSelectorBuffers b{};
    b.candidates = allocate(backend, workspace.candidatesBytes);
    b.unary = allocate(backend, workspace.unaryBytes);
    b.partialValues = allocate(backend, workspace.partialValuesBytes);
    b.proposedTokens = allocate(backend, lanes * Positions * sizeof(uint32_t));
    b.proposalProbabilities = allocate(backend, workspace.proposalProbabilitiesBytes);
    std::memset(b.proposalProbabilities.contents(), 0xa5, b.proposalProbabilities.sizeBytes());
    const auto budget = allocate(backend, DraftSelector::liveRowsBytes(lanes));
    auto *ids = static_cast<uint32_t *>(b.candidates.contents());
    auto *tokens = static_cast<uint32_t *>(b.proposedTokens.contents());
    auto *unary = static_cast<float *>(b.unary.contents());
    auto *partial = static_cast<float *>(b.partialValues.contents());
    const uint32_t tables = lanes * Positions * SPLASH_DRAFT_SAMPLING_SHARDS * Candidates;
    std::fill(partial, partial + workspace.partialValuesBytes / sizeof(float), NAN);
    std::vector<SamplingPolicy> policies(lanes);
    for (auto &policy : policies) policy.temperature = 0.0F;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      uint32_t predecessor = 0;
      for (uint32_t p = 0; p < Positions; ++p) {
        const uint32_t selected = (lane + p + 3) % Candidates;
        tokens[lane * Positions + p] = p * Candidates + selected;
        for (uint32_t c = 0; c < Candidates; ++c) {
          const uint32_t index = (lane * Positions + p) * Candidates + c;
          ids[index] = p * Candidates + c;
          unary[index] = c == selected ? std::log(q[p] * (Candidates - 1) / (1.0F - q[p])) : 0.0F;
          // Only the chosen predecessor's edge row is valid. In particular,
          // all other rows at position zero are poisoned.
          partial[tables + ((lane * Positions + p) * Candidates + predecessor) * Candidates + c] = 0.0F;
        }
        predecessor = selected;
      }
    }
    for (float threshold : {0.0F, 0.001F, 0.01F, 0.1F, 0.2F, 0.5F, 0.8F, 0.94F, 1.0F}) {
      for (uint32_t mixed = 0; mixed < 3; ++mixed) {
        policies.back().temperature = mixed == 1 ? 0.8F : 0.0F;
        policies.back().constrained = mixed == 2;
        CommandGraph graph;
        selector.addLiveRows(graph, b, budget, policies, threshold);
        static_cast<void>(backend.submitCommand(graph.dispatches()));
        const auto *actual = static_cast<const VerifyLiveRows *>(budget.contents());
        for (uint32_t lane = 0; lane < lanes; ++lane) {
          uint32_t expected = 1;
          float prefix = 1.0F;
          for (uint32_t p = 0; p < Positions; ++p) {
            prefix *= q[p];
            if (expected == p + 1 && (threshold == 0.0F || prefix >= threshold)) ++expected;
            if (!(lane == lanes - 1 && mixed))
              require(std::abs(actual[lane].prefix_confidence[p] - prefix) < 1e-5F,
                      "chosen-path confidence differs from the reference");
          }
          if (lane == lanes - 1 && mixed) expected = Rows;
          require(actual[lane].count == expected, "live prefix differs from the reference");
        }
      }
    }
    const auto *bytes = static_cast<const uint8_t *>(b.proposalProbabilities.contents());
    for (uint64_t i = 0; i < b.proposalProbabilities.sizeBytes(); ++i)
      require(bytes[i] == 0xa5, "confidence overwrote sampled proposal probabilities");
    policies.back().temperature = 0.0F;
    policies.back().constrained = false;
    unary[0] = NAN;
    CommandGraph invalid;
    selector.addLiveRows(invalid, b, budget, policies, 0.9F);
    static_cast<void>(backend.submitCommand(invalid.dispatches()));
    require(static_cast<const VerifyLiveRows *>(budget.contents())[0].count == Rows,
            "nonfinite confidence must retain full verification");
    for (float threshold : {-0.1F, 1.1F, NAN, INFINITY})
      rejects([&] { CommandGraph graph; selector.addLiveRows(graph, b, budget, policies, threshold); });
    rejects([&] { CommandGraph graph; selector.addLiveRows(graph, b, {}, policies, 0.5F); });
  }
  rejects([] { static_cast<void>(DraftSelector::liveRowsBytes(0)); });
  rejects([] { static_cast<void>(DraftSelector::liveRowsBytes(5)); });
}

// Production routing geometry (256 experts, top-8, 288 routes at B4),
// including multiple routes per thread and poisoned inactive expert IDs.
void grouping(MetalBackend &backend) {
  constexpr uint32_t Experts = 256, TopK = 8, Routes = TopK + 1, Tile = 8;
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    const uint32_t rows = lanes * Rows;
    const uint32_t maximumTiles = rows * TopK + lanes;
    const auto selected = allocate(backend, rows * Routes * sizeof(uint32_t));
    const auto tiles = allocate(backend, maximumTiles * sizeof(MoeTileDescriptor));
    const auto count = allocate(backend, sizeof(uint32_t));
    const auto grouped = allocate(backend, maximumTiles * Tile * sizeof(uint32_t));
    const auto mapping = allocate(backend, rows * Routes * sizeof(uint32_t));
    const auto budget = allocate(backend, DraftSelector::liveRowsBytes(lanes));
    auto *limits = static_cast<VerifyLiveRows *>(budget.contents());
    auto *ids = static_cast<uint32_t *>(selected.contents());
    for (uint32_t cutoff = 1; cutoff <= 8; ++cutoff) {
      for (uint32_t distribution = 0; distribution < 2; ++distribution) {
        std::array<uint32_t, Experts + 1> expected{};
        for (uint32_t lane = 0; lane < lanes; ++lane)
          limits[lane].count = (cutoff + lane - 1) % Rows + 1;
        for (uint32_t row = 0; row < rows; ++row) {
          for (uint32_t slot = 0; slot < Routes; ++slot) {
            const uint32_t expert = slot == TopK ? Experts
                : distribution ? slot : (row * 7 + slot * 13) % Experts;
            const bool live = row % Rows < limits[row / Rows].count;
            ids[row * Routes + slot] = live ? expert : 0xffffffffu;
            if (live) ++expected[expert];
          }
        }
        CommandGraph graph;
        graph.add("moe_group_routes_live_rows",
                  {selected, tiles, count, grouped, mapping, budget},
                  MoeGroupParams{rows, TopK, Tile, Experts}, {1, 1, 1});
        static_cast<void>(backend.submitCommand(graph.dispatches()));
        const uint32_t tileCount = *static_cast<const uint32_t *>(count.contents());
        const auto *descriptors = static_cast<const MoeTileDescriptor *>(tiles.contents());
        const auto *routes = static_cast<const uint32_t *>(grouped.contents());
        const auto *routeRows = static_cast<const uint32_t *>(mapping.contents());
        std::array<uint32_t, Experts + 1> actual{};
        uint32_t expectedTiles = 0;
        for (uint32_t n : expected) expectedTiles += (n + Tile - 1) / Tile;
        require(tileCount == expectedTiles, "production live tile count is incorrect");
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
          require(descriptors[tile].expert <= Experts, "invalid grouped expert");
          actual[descriptors[tile].expert] += descriptors[tile].rows;
        }
        require(actual == expected, "production live route counts differ");
        for (uint32_t row = 0; row < rows; ++row) {
          for (uint32_t slot = 0; slot < Routes; ++slot) {
            const uint32_t route = row * Routes + slot;
            if (row % Rows >= limits[row / Rows].count) {
              require(routeRows[route] == 0xffffffffu, "inactive production route survived");
            } else {
              require(routeRows[route] < tileCount * Tile &&
                          routes[routeRows[route]] == route &&
                          descriptors[routeRows[route] / Tile].expert == ids[route],
                      "production live route mapping does not round-trip");
            }
          }
        }
      }
    }
  }
}

void acceptance(MetalBackend &backend) {
  Sampling sampling(1024);
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    AcceptanceBuffers b{
        allocate(backend, lanes * Positions * sizeof(uint32_t)),
        allocate(backend, lanes * Positions * Candidates * sizeof(uint32_t)),
        allocate(backend, lanes * Positions * Candidates * sizeof(float)),
        allocate(backend, lanes * Rows * sizeof(TargetVocabularyRow)),
        allocate(backend, lanes * SPLASH_SAMPLING_UNIFORMS * sizeof(float)),
        allocate(backend, lanes * Rows * sizeof(uint32_t)),
        allocate(backend, lanes * sizeof(uint32_t)),
        allocate(backend, lanes * sizeof(uint32_t)),
        allocate(backend, DraftSelector::liveRowsBytes(lanes))};
    auto *draft = static_cast<uint32_t *>(b.proposedTokens.contents());
    auto *ids = static_cast<uint32_t *>(b.candidates.contents());
    auto *q = static_cast<float *>(b.proposalProbabilities.contents());
    auto *target = static_cast<TargetVocabularyRow *>(b.targetVocabularyRows.contents());
    auto *tokens = static_cast<uint32_t *>(b.outputTokens.contents());
    auto *limits = static_cast<VerifyLiveRows *>(b.liveRows.contents());
    std::vector<SamplingPolicy> policies(lanes);
    std::vector<uint32_t> remaining(lanes);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      policies[lane].temperature = lane % 2 ? 0.8F : 0.0F;
      for (uint32_t p = 0; p < Positions; ++p) {
        draft[lane * Positions + p] = 10 + p;
        ids[(lane * Positions + p) * Candidates] = 10 + p;
        q[(lane * Positions + p) * Candidates] = 1.0F;
        target[lane * Rows + p].draft_probability = 1.0F;
      }
    }
    for (uint32_t live = 1; live <= Rows; ++live) {
      for (uint32_t reject = 0; reject <= Positions; ++reject) {
        for (uint32_t cap : {1U, 3U, Rows}) {
          for (uint32_t lane = 0; lane < lanes; ++lane) {
            limits[lane].count = live;
            remaining[lane] = cap;
            const uint32_t limit = policies[lane].samples() ? Rows : live;
            for (uint32_t r = 0; r < Rows; ++r) {
              tokens[lane * Rows + r] = r < limit ? 10 + r : 0xffffffffu;
              target[lane * Rows + r].draft_probability = r < reject ? 1.0F : 0.0F;
            }
            if (reject < limit - 1) tokens[lane * Rows + reject] = 777;
          }
          CommandGraph graph;
          sampling.addAcceptance(graph, b, remaining, policies, 999, 998);
          static_cast<void>(backend.submitCommand(graph.dispatches()));
          const auto *retained = static_cast<const uint32_t *>(b.retainedCounts.contents());
          const auto *accepted = static_cast<const uint32_t *>(b.acceptedCounts.contents());
          for (uint32_t lane = 0; lane < lanes; ++lane) {
            const uint32_t proposals = policies[lane].samples() ? Positions : live - 1;
            const uint32_t expected = std::min(reject, proposals);
            require(accepted[lane] == expected, "acceptance crossed the live-row limit");
            require(retained[lane] == std::min(expected + 1, cap), "retention crossed the live-row limit");
            for (uint32_t r = 0; r < retained[lane]; ++r)
              require(tokens[lane * Rows + r] < 1024, "inactive token escaped acceptance");
          }
        }
      }
      // A stop token in the accepted prefix still cuts retention immediately.
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        limits[lane].count = live;
        remaining[lane] = Rows;
        for (uint32_t r = 0; r < Rows; ++r) {
          tokens[lane * Rows + r] = 10 + r;
          target[lane * Rows + r].draft_probability = 1.0F;
        }
      }
      CommandGraph stop;
      sampling.addAcceptance(stop, b, remaining, policies, 10, 11);
      static_cast<void>(backend.submitCommand(stop.dispatches()));
      for (uint32_t lane = 0; lane < lanes; ++lane)
        require(static_cast<const uint32_t *>(b.retainedCounts.contents())[lane] == 1,
                "live acceptance ignored a stop token");
    }
  }
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: live-verify-rows METALLIB");
    MetalBackend backend(argv[1]);
    confidence(backend);
    grouping(backend);
    acceptance(backend);
    std::cout << "live_verify_rows_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "live_verify_rows_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}

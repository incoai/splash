// Independent CPU equations for adjacent feature injection and conditional
// final-write proposals. Covers greedy and sampled chains in a mixed batch.
#include "LinearNumerics.hpp"
#include "TestChecks.hpp"
#include "ops/DSpine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>

namespace {

#ifndef SPLASH_TEST_VERIFY_ROWS
#define SPLASH_TEST_VERIFY_ROWS 8
#endif
constexpr splash::ops::DecodeGeometry kGeometry(SPLASH_TEST_VERIFY_ROWS);
using namespace splash;
using namespace splash::ops;
using test::require;
using test::floatToBf16;
using test::bf16ToFloat;
using metal::MetalBuffer;
constexpr uint32_t V = 128;
constexpr uint32_t Rows = kGeometry.rows(), Positions = kGeometry.proposals();
constexpr uint32_t K = SPLASH_DRAFT_CANDIDATES;

float bf(float x) { return bf16ToFloat(floatToBf16(x)); }
MetalBuffer allocate(metal::MetalBackend &backend, uint64_t bytes) {
  auto b = backend.allocateBuffer(bytes, metal::BufferStorage::Shared, "DSpine test");
  std::memset(b.contents(), 0, bytes);
  return b;
}
MetalBuffer random(metal::MetalBackend &backend, uint64_t elements, float scale, uint32_t seed) {
  auto b = allocate(backend, elements * 2);
  auto *p = static_cast<uint16_t *>(b.contents());
  uint32_t state = seed;
  for (uint64_t i = 0; i < elements; ++i) {
    state = state * 1664525u + 1013904223u;
    p[i] = floatToBf16((float(state >> 8) / 8388608.0f - 1) * scale);
  }
  return b;
}
float value(const MetalBuffer &b, uint64_t i) { return bf16ToFloat(static_cast<const uint16_t *>(b.contents())[i]); }
Projection projection(const MetalBuffer &b, uint32_t output, uint32_t input) {
  return {output, input, BlockWeights{{QuantizedSegment::rows(GGUF_FMT_BF16, output, input, b)}}};
}
void close(float actual, double expected, double tolerance, const char *what) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance * (1 + std::abs(expected))) {
    std::cerr << what << ": " << actual << " != " << expected << '\n';
    throw std::runtime_error(what);
  }
}

template <uint32_t H = 256, uint32_t C = 64, uint32_t R = 32>
void run(metal::MetalBackend &backend, uint32_t lanes, bool sampleAll = false) {
  const uint32_t rows = lanes * Rows;
  DSpineParams shape{H, C, R, rows};
  DSpineBuffers b{allocate(backend, uint64_t{rows} * H * 2), allocate(backend, rows * 4),
      random(backend, uint64_t{rows} * C, 0.3f, 19), random(backend, uint64_t{rows} * 3 * R, 0.3f, 23),
      allocate(backend, uint64_t{rows} * R * 2), allocate(backend, uint64_t{lanes} * Positions * K * H * 2),
      allocate(backend, uint64_t{lanes} * R * 4), allocate(backend, uint64_t{lanes} * H * 4)};
  DSpineWeights weights;
  weights.codebook = random(backend, uint64_t{V} * C, 0.15f, 29);
  weights.gateBias = random(backend, R, 0.1f, 31);
  weights.predecessorGate = random(backend, R * C, 0.05f, 37);
  weights.previous = random(backend, R * C, 0.05f, 41);
  auto write = random(backend, H * R, 0.05f, 43);
  weights.write.push_back(projection(write, H, R));
  auto input = random(backend, uint64_t{rows} * H, 1.0f, 47);
  auto normalized = allocate(backend, uint64_t{rows} * H * 2);
  auto delta = random(backend, uint64_t{rows} * H, 0.1f, 53);
  auto output = allocate(backend, uint64_t{rows} * H * 2);
  auto tokens = allocate(backend, rows * 4);
  for (uint32_t lane = 0; lane < lanes; ++lane) static_cast<uint32_t *>(tokens.contents())[lane * Rows] = 3 + lane;
  std::vector<float> originalCodes(rows * C);
  for (uint32_t i = 0; i < rows * C; ++i) originalCodes[i] = value(b.codes, i);
  metal::CommandGraph graph;
  DSpine::addReadInput(graph, input, normalized, b, shape, kGeometry);
  DSpine::addCodes(graph, tokens, weights, b, shape, kGeometry);
  DSpine::addMessage(graph, weights, b, shape, kGeometry);
  DSpine::addWrite(graph, delta, output, b, shape, kGeometry);
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
  for (uint32_t row = 0; row < rows; ++row) {
    double square = 0, codeSquare = 0;
    for (uint32_t h = 0; h < H; ++h) square += double(value(input, row * H + h)) * value(input, row * H + h);
    const double rms = std::sqrt(square / H);
    close(static_cast<const float *>(b.rms.contents())[row], rms, 1e-6, "bridge RMS");
    for (uint32_t h = 0; h < H; ++h) {
      close(value(normalized, row * H + h), bf(value(input, row * H + h) / (rms + 1e-6)), 0.005, "bridge read");
      const float expected = row % Rows == 0 ? value(input, row * H + h)
          : bf(value(input, row * H + h) + rms * value(delta, row * H + h));
      close(value(output, row * H + h), expected, 0.005, "bridge write");
    }
    for (uint32_t c = 0; c < C; ++c) codeSquare += double(originalCodes[row * C + c]) * originalCodes[row * C + c];
    for (uint32_t c = 0; c < C; ++c) {
      const double expected = row % Rows == 0 ? std::sqrt(C) * value(weights.codebook, (3 + row / Rows) * C + c)
          : std::sqrt(C) * originalCodes[row * C + c] / (std::sqrt(codeSquare) + 1e-6);
      close(value(b.codes, row * C + c), bf(expected), 0.005, "bridge codes");
    }
    for (uint32_t r = 0; r < R; ++r) {
      double expected = 0;
      if (row % Rows) {
        double gate = value(b.packed, row * 3 * R + r) + value(b.packed, (row - 1) * 3 * R + R + r) + value(weights.gateBias, r);
        expected = value(b.packed, (row - 1) * 3 * R + 2 * R + r) / (1 + std::exp(-gate));
      }
      close(value(b.message, row * R + r), bf(expected), 0.005, "bridge message");
    }
  }

  auto head = random(backend, V * H, 0.2f, 59);
  auto norm = random(backend, H, 0.1f, 61);
  for (uint32_t h = 0; h < H; ++h) static_cast<uint16_t *>(norm.contents())[h] = floatToBf16(1 + value(norm, h));
  const auto ws = DraftSelector::workspace(lanes * Positions);
  DraftSelectorBuffers s;
  s.logits = allocate(backend, uint64_t{rows} * V * 4);
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t token = 0; token < V; ++token)
      static_cast<float *>(s.logits.contents())[row * V + token] = std::sin(float(row * 137 + token) * 0.17f);
  s.partialIds = allocate(backend, ws.partialIdsBytes);
  s.partialValues = allocate(backend, ws.partialValuesBytes);
  s.candidates = allocate(backend, ws.candidatesBytes);
  s.proposedTokens = allocate(backend, lanes * Positions * 4);
  s.proposalProbabilities = allocate(backend, ws.proposalProbabilitiesBytes);
  s.confidence = allocate(backend, lanes * Positions * 4);
  s.uniforms = allocate(backend, lanes * kGeometry.uniformCount() * 4);
  for (uint32_t i = 0; i < lanes * kGeometry.uniformCount(); ++i)
    static_cast<float *>(s.uniforms.contents())[i] = (i % 13 + 0.5f) / 13;
  std::vector<uint32_t> anchors(lanes);
  std::vector<SamplingPolicy> policies(lanes);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    anchors[lane] = 3 + lane;
    policies[lane].temperature = !sampleAll && lane % 3 == 0 ? 0 : 0.5f + 0.4f * lane;
  }
  metal::CommandGraph refine;
  DSpine::addRefinement(refine, weights, {norm, false}, projection(head, V, H), b, s, shape, anchors, policies, kGeometry);
  static_cast<void>(backend.submitCommandAsync(refine.dispatches()).wait());
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    uint32_t predecessor = anchors[lane];
    for (uint32_t pos = 0; pos < Positions; ++pos) {
      const uint32_t row = lane * Rows + pos + 1, at = lane * Positions + pos;
      std::vector<uint32_t> ids(V);
      std::iota(ids.begin(), ids.end(), 0);
      const auto *logits = static_cast<const float *>(s.logits.contents()) + row * V;
      std::ranges::sort(ids, [&](uint32_t a, uint32_t z) { return logits[a] > logits[z] || (logits[a] == logits[z] && a < z); });
      ids.resize(K);
      std::ranges::sort(ids);
      for (uint32_t k = 0; k < K; ++k)
        require(static_cast<const uint32_t *>(s.candidates.contents())[at * K + k] == ids[k], "DSpine candidate support mismatch");
      std::vector<double> message(R), state(H), score(K);
      for (uint32_t r = 0; r < R; ++r) {
        double gate = 0, v = 0;
        for (uint32_t c = 0; c < C; ++c) {
          const double code = std::sqrt(C) * value(weights.codebook, predecessor * C + c);
          gate += code * value(weights.predecessorGate, r * C + c);
          v += code * value(weights.previous, r * C + c);
        }
        message[r] = v / (1 + std::exp(-(value(b.packed, row * 3 * R + r) + gate + value(weights.gateBias, r))));
      }
      double squares = 0;
      for (uint32_t h = 0; h < H; ++h) {
        double delta = 0;
        for (uint32_t r = 0; r < R; ++r) delta += message[r] * value(write, h * R + r);
        state[h] = value(b.raw, row * H + h) + static_cast<const float *>(b.rms.contents())[row] * delta;
        squares += state[h] * state[h];
      }
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t h = 0; h < H; ++h)
          score[k] += state[h] / std::sqrt(squares / H + 1e-6) * value(norm, h) * value(head, ids[k] * H + h);
      const auto selected = std::max_element(score.begin(), score.end());
      const double best = *selected;
      double mass = 0;
      for (double x : score) mass += std::exp(x - best);
      close(static_cast<const float *>(s.confidence.contents())[at], 1 / mass, 2e-5, "DSpine confidence");
      uint32_t choice = static_cast<uint32_t>(selected - score.begin());
      if (policies[lane].samples()) {
        double total = 0;
        for (double &x : score) { x = std::exp((x - best) / policies[lane].temperature); total += x; }
        double cumulative = 0;
        choice = K - 1;
        bool found = false;
        const float uniform = static_cast<const float *>(s.uniforms.contents())[lane * kGeometry.uniformCount() + SPLASH_UNIFORM_PROPOSALS + pos];
        for (uint32_t k = 0; k < K; ++k) {
          close(static_cast<const float *>(s.proposalProbabilities.contents())[at * K + k], score[k] / total, 2e-5, "conditional DSpine q");
          cumulative += score[k] / total;
          if (!found && cumulative > uniform) { choice = k; found = true; }
        }
      }
      predecessor = ids[choice];
      require(static_cast<const uint32_t *>(s.proposedTokens.contents())[at] == predecessor, "DSpine serial chain mismatch");
    }
  }
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: dspine METALLIB");
    metal::MetalBackend backend(argv[1]);
    for (uint32_t lanes = 1; lanes <= SPLASH_MAXIMUM_BATCH_WIDTH; ++lanes) run(backend, lanes);
    run<257, 65, 35>(backend, 2);
    run<5120, 1024, 512>(backend, 1, true);
    run<5120, 1024, 512>(backend, SPLASH_MAXIMUM_BATCH_WIDTH);
    std::cout << "DSpine bridge and refinement: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "DSpine: " << error.what() << '\n';
    return 1;
  }
}

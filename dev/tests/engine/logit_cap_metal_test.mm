// The head's logit cap (ops::addLogitCap, head_logit_cap in
// kernels/decode/sampling.metal) against fp64, and the target policy over the
// logits it leaves (ops::Sampling), at Inkling's head: rows of 201,024 logits
// (3,141 tiles of 64 rows of the head), of which the first 200,058 are its
// vocabulary's, and a softcap of 30 (model::QwenTargetDimensions::logitCap).
// - The cap: every valid logit within 2^-20 of 30 tanh(logit / 30), relative,
//   from tiny values to saturated ones, every padded one -inf, at the rows a
//   decode step's head and a prompt's first tokens' head take, the row after
//   them untouched. A cap of the vocabulary alone leaves the valid logits
//   bitwise; a softcap alone caps every token; a cap that does nothing adds no
//   dispatch; a negative or non-finite softcap, a valid vocabulary past the
//   row and logits one float short are refused.
// - The policy over capped rows whose padded tokens carry the largest raw
//   logits: a greedy first token and greedy verify rows take the argmax of
//   the valid tokens; a sampled first token (temperature 1, every token kept,
//   Inkling's defaults) draws, at the middle of a token's interval of the
//   capped distribution in token order (fp64), that token, and at the end of
//   the distribution its last valid token; and a verify chain whose draft
//   proposes a padded token rejects it at once, where the same chain over the
//   raw logits accepts every proposal.
#include "ChainRows.hpp"
#include "TestChecks.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/LaneRows.hpp"
#include "ops/Sampling.hpp"

#import <Foundation/Foundation.h>

#include "metal/abi/Sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using namespace splash::ops;
using splash::test::chainRow;
using splash::test::rejects;
using splash::test::require;

constexpr uint32_t kVocabulary = 201'024;
constexpr uint32_t kValid = 200'058;
constexpr float kSoftcap = 30.0F;
constexpr LogitCap kInklingCap{kSoftcap, kValid};
// A padded token the tests' drafts propose and their raw rows rank first.
constexpr uint32_t kPadded = 200'500;
// A lane's rows of a verify chain: a draft's block.
constexpr uint32_t kRows = kNarrowLaneRows;
constexpr float kInfinity = std::numeric_limits<float>::infinity();

MetalBuffer allocate(MetalBackend &backend, uint64_t bytes, uint8_t fill = 0) {
  MetalBuffer buffer = backend.allocateBuffer(bytes, BufferStorage::Shared, "logit cap");
  std::memset(buffer.contents(), fill, bytes);
  return buffer;
}

float *floats(const MetalBuffer &buffer) { return static_cast<float *>(buffer.contents()); }

void run(MetalBackend &backend, const CommandGraph &graph) {
  static_cast<void>(backend.submitCommandAsync(graph.dispatches()).wait());
}

// `rows` rows of raw logits: normal ones of scale 20, and in every row tiny,
// zero, large and huge values of both signs, which the softcap passes through
// nearly as they are or saturates.
std::vector<float> rawRows(uint32_t rows, std::mt19937 &random) {
  std::normal_distribution<float> normal(0.0F, 20.0F);
  std::vector<float> values(uint64_t{rows} * kVocabulary);
  for (float &value : values) value = normal(random);
  constexpr float kSpecials[] = {0.0F, 1e-30F, -1e-30F, 1e-8F, -3e-6F, 0.25F, -1.5F, 29.0F, -45.0F, 300.0F, -1e30F, 1e30F};
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t i = 0; i < std::size(kSpecials); ++i)
      values[uint64_t{row} * kVocabulary + (row * 977 + i * 16'411) % kValid] = kSpecials[i];
  return values;
}

// A cap of `rows` rows of `raw` into a buffer of one row more, whose last row
// holds a sentinel the cap must not touch; returns the buffer after the cap.
MetalBuffer capped(MetalBackend &backend, const std::vector<float> &raw, uint32_t rows, LogitCap cap) {
  MetalBuffer logits = allocate(backend, uint64_t{rows + 1} * kVocabulary * sizeof(float), 0x7F);
  std::memcpy(logits.contents(), raw.data(), uint64_t{rows} * kVocabulary * sizeof(float));
  CommandGraph graph;
  addLogitCap(graph, logits, rows, kVocabulary, cap);
  require(graph.dispatches().size() == 1, "an active logit cap is not one dispatch");
  run(backend, graph);
  const auto *sentinel = static_cast<const uint8_t *>(logits.contents()) + uint64_t{rows} * kVocabulary * 4;
  require(std::all_of(sentinel, sentinel + uint64_t{kVocabulary} * 4, [](uint8_t byte) { return byte == 0x7F; }),
          "the logit cap wrote past its rows");
  return logits;
}

// Each softcapped logit within 2^-20 of the fp64 softcap, relative.
bool nearSoftcap(float got, float raw) {
  const double want = kSoftcap * std::tanh(double(raw) / kSoftcap);
  return std::fabs(got - want) <= std::ldexp(std::fabs(want), -20);
}

void cap(MetalBackend &backend) {
  std::mt19937 random(0x1C0FFEE);
  for (const uint32_t rows : {1U, 5U, 16U, 64U}) {
    const std::vector<float> raw = rawRows(rows, random);
    const MetalBuffer inkling = capped(backend, raw, rows, kInklingCap);
    const MetalBuffer masked = capped(backend, raw, rows, {0.0F, kValid});
    const MetalBuffer softcapped = capped(backend, raw, rows, {kSoftcap, 0});
    uint64_t far = 0, unmasked = 0, moved = 0, uncapped = 0;
    for (uint64_t at = 0; at < uint64_t{rows} * kVocabulary; ++at) {
      const bool valid = at % kVocabulary < kValid;
      far += valid && !nearSoftcap(floats(inkling)[at], raw[at]);
      unmasked += !valid && floats(inkling)[at] != -kInfinity;
      moved += valid ? std::memcmp(&floats(masked)[at], &raw[at], 4) != 0 : floats(masked)[at] != -kInfinity;
      uncapped += !nearSoftcap(floats(softcapped)[at], raw[at]);
    }
    require(!far && !unmasked, std::to_string(rows) + " rows: " + std::to_string(far) +
                                   " valid logits off the softcap, " + std::to_string(unmasked) +
                                   " padded ones not -inf");
    require(!moved, std::to_string(rows) + " rows: a cap of the vocabulary alone moved " + std::to_string(moved) +
                        " logits");
    require(!uncapped, std::to_string(rows) + " rows: a softcap alone left " + std::to_string(uncapped) +
                           " logits off the softcap");
  }
  // A cap that does nothing adds no dispatch; invalid caps and logits that
  // do not hold the rows are refused before anything is encoded.
  const MetalBuffer logits = allocate(backend, uint64_t{2} * kVocabulary * sizeof(float));
  for (const LogitCap none : {LogitCap{}, LogitCap{0.0F, kVocabulary}}) {
    CommandGraph graph;
    addLogitCap(graph, logits, 2, kVocabulary, none);
    require(graph.empty(), "a logit cap that does nothing added a dispatch");
  }
  const auto refused = [&](const MetalBuffer &buffer, LogitCap invalid, std::string_view expected,
                           const std::string &what) {
    CommandGraph graph;
    rejects([&] { addLogitCap(graph, buffer, 2, kVocabulary, invalid); }, expected, what + " was accepted");
    require(graph.empty(), what + " added a dispatch");
  };
  refused(logits, {-30.0F, kValid}, "softcap is negative", "a negative softcap");
  refused(logits, {std::numeric_limits<float>::quiet_NaN(), kValid}, "softcap is negative", "a NaN softcap");
  refused(logits, {kInfinity, kValid}, "softcap is negative", "an infinite softcap");
  refused(logits, {kSoftcap, kVocabulary + 1}, "past the vocabulary", "a valid vocabulary past the row");
  refused(backend.view(logits, 0, logits.sizeBytes() - 4), kInklingCap, "capped logits buffer holds",
          "logits one float short");
  printf("cap: rows 1/5/16/64 of %u logits, %u valid: softcap 30 within 2^-20, padded -inf, mask alone bitwise, "
         "inactive and invalid caps: ok\n",
         kVocabulary, kValid);
}

// One lane's sampling buffers over `rows` rows of logits, its tree a chain.
struct Batch final {
  SamplingBuffers buffers;
  AcceptanceBuffers acceptance;
};

Batch makeBatch(MetalBackend &backend, const MetalBuffer &logits, uint32_t rows) {
  const SamplingWorkspace space = Sampling::workspace(rows);
  const uint64_t words = (kVocabulary + 31) / 32;
  Batch batch;
  batch.buffers = {logits,
                   allocate(backend, space.partialMassesBytes),
                   allocate(backend, space.vocabularyRowsBytes),
                   allocate(backend, SPLASH_SAMPLING_UNIFORMS * sizeof(float)),
                   allocate(backend, (SPLASH_TARGET_VERIFY_ROWS + 1) * words * 4),
                   allocate(backend, uint64_t{rows} * sizeof(uint32_t)),
                   allocate(backend, space.argmaxValuesBytes),
                   allocate(backend, space.argmaxIndicesBytes),
                   allocate(backend, uint64_t{rows} * sizeof(uint32_t)),
                   allocate(backend, uint64_t{rows} * sizeof(TreeRow)),
                   allocate(backend, space.vocabularyRangesBytes),
                   allocate(backend, space.vocabularyArrivalsBytes),
                   allocate(backend, sizeof(LaneRows)),
                   // No row tests children.
                   allocate(backend, uint64_t{rows} * sizeof(uint32_t), 0xFF),
                   allocate(backend, uint64_t{SPLASH_DRAFT_PROPOSAL_TOKENS} * SPLASH_DRAFT_CANDIDATES * 4),
                   allocate(backend, uint64_t{SPLASH_TARGET_VERIFY_ROWS} * SPLASH_TREE_CHILDREN, 0xFF),
                   allocate(backend, uint64_t{SPLASH_TARGET_VERIFY_ROWS} * SPLASH_DRAFT_CANDIDATES * sizeof(float))};
  auto *tree = static_cast<TreeRow *>(batch.buffers.treeRows.contents());
  for (uint32_t row = 0; row < rows; ++row) tree[row] = chainRow(row);
  batch.acceptance = {batch.buffers.inputTokens, batch.buffers.treeRows,     batch.buffers.outputTokens,
                      allocate(backend, 4),      allocate(backend, 4),       allocate(backend, 4),
                      batch.buffers.laneRows};
  return batch;
}

const uint32_t *tokens(const Batch &batch) { return static_cast<const uint32_t *>(batch.buffers.outputTokens.contents()); }

// The argmax of a row's valid tokens, the lowest id among equal logits.
uint32_t validArgmax(const float *row) {
  return uint32_t(std::max_element(row, row + kValid, [](float a, float b) { return a < b; }) - row);
}

// The fp64 weights of a capped row's tokens at temperature 1 (exp(logit -
// max), the padded ones 0), and the uniform at the middle of `token`'s
// interval of them in token order.
std::vector<double> weightsOf(const float *row) {
  const double maximum = *std::max_element(row, row + kVocabulary);
  std::vector<double> weights(kVocabulary);
  for (uint32_t token = 0; token < kVocabulary; ++token) weights[token] = std::exp(double(row[token]) - maximum);
  return weights;
}
float intervalMiddle(const std::vector<double> &weights, uint32_t token) {
  const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
  const double before = std::accumulate(weights.begin(), weights.begin() + token, 0.0);
  return float((before + 0.5 * weights[token]) / total);
}

// Rows of noise around 0 with spikes at tokens 7, 100,000 and the last valid
// token, which hold most of a row's mass after the softcap, and the padded
// tokens far above them all, the first padded proposal highest.
std::vector<float> spikedRows(uint32_t rows, std::mt19937 &random) {
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<float> values(uint64_t{rows} * kVocabulary);
  for (uint32_t row = 0; row < rows; ++row) {
    float *logits = values.data() + uint64_t{row} * kVocabulary;
    for (uint32_t token = 0; token < kValid; ++token) logits[token] = normal(random);
    logits[7] = 20.0F + row;
    logits[100'000] = 19.0F;
    logits[kValid - 1] = 18.5F;
    for (uint32_t token = kValid; token < kVocabulary; ++token) logits[token] = 1e4F;
    logits[kPadded] = 2e4F;
  }
  return values;
}

void policies(MetalBackend &backend) {
  std::mt19937 random(0x5A3D1E);
  const std::vector<float> raw = spikedRows(kRows, random);
  const MetalBuffer logits = capped(backend, raw, kRows, kInklingCap);
  const Sampling sampling(kVocabulary);
  const StopTokens stops{200'006};
  Batch batch = makeBatch(backend, logits, kRows);
  float *uniforms = floats(batch.buffers.uniforms);
  const float *row0 = floats(logits);

  // The first token after a prompt, greedy and sampled.
  const SamplingPolicy greedy{1, 0.0F, 1.0F, false};
  const SamplingPolicy sampled{0, 1.0F, 1.0F, false};
  const auto first = [&](const SamplingPolicy &policy, float uniform) {
    uniforms[0] = uniform;
    CommandGraph graph;
    sampling.addInitial(graph, {&policy, 1}, batch.buffers, std::vector<uint32_t>{0}, stops, {});
    run(backend, graph);
    return tokens(batch)[0];
  };
  require(first(greedy, 0.0F) == validArgmax(row0) && validArgmax(row0) == 7,
          "a greedy first token is not the valid argmax");
  const std::vector<double> weights = weightsOf(row0);
  require(std::all_of(weights.begin() + kValid, weights.end(), [](double w) { return w == 0.0; }),
          "a padded token kept a weight");
  for (const uint32_t token : {7U, 100'000U, kValid - 1})
    require(first(sampled, intervalMiddle(weights, token)) == token,
            "a sampled first token missed token " + std::to_string(token) + " at its interval's middle");
  require(first(sampled, std::nextafter(1.0F, 0.0F)) == kValid - 1,
          "a sampled first token at the distribution's end is not its last valid token");

  // A greedy verify chain whose draft proposes the padded token at every
  // row: row 0 rejects it for its own token, the only one the lane keeps.
  const auto verify = [&](const MetalBuffer &rows) {
    Batch chain = makeBatch(backend, rows, kRows);
    *static_cast<LaneRows *>(chain.buffers.laneRows.contents()) = {0, kRows};
    auto *inputs = static_cast<uint32_t *>(chain.buffers.inputTokens.contents());
    inputs[0] = 5;
    for (uint32_t row = 1; row < kRows; ++row) inputs[row] = kPadded;
    const uint32_t remaining = kRows;
    CommandGraph selection, acceptance;
    sampling.addVerify(selection, {&greedy, 1}, chain.buffers, StepRows{1, kRows, kRows}, stops, {});
    sampling.addAcceptance(acceptance, chain.acceptance, StepRows{1, kRows, kRows}, {&remaining, 1}, stops);
    // The selections, before acceptance rewrites them.
    run(backend, selection);
    std::vector<uint32_t> selected(tokens(chain), tokens(chain) + kRows);
    run(backend, acceptance);
    return std::pair{selected, *static_cast<const uint32_t *>(chain.acceptance.retainedCounts.contents())};
  };
  const auto [capSelections, capRetained] = verify(logits);
  for (uint32_t row = 0; row < kRows; ++row)
    require(capSelections[row] == validArgmax(floats(logits) + uint64_t{row} * kVocabulary),
            "greedy verify row " + std::to_string(row) + " is not its valid argmax");
  require(capRetained == 1, "a verify chain of padded proposals kept " + std::to_string(capRetained) +
                                " tokens, not row 0's own token alone");
  // The raw rows select the padded token, so the same chain accepts every
  // proposal.
  MetalBuffer rawLogits = allocate(backend, uint64_t{kRows} * kVocabulary * sizeof(float));
  std::memcpy(rawLogits.contents(), raw.data(), rawLogits.sizeBytes());
  const auto [rawSelections, rawRetained] = verify(rawLogits);
  require(rawSelections[0] == kPadded && rawRetained == kRows,
          "the raw rows' chain does not accept the padded proposals, so the cap's rejection tests nothing");
  printf("policy: greedy and sampled first tokens of valid tokens only, draws at intervals of the capped "
         "distribution, padded proposals rejected (accepted over the raw rows): ok\n");
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    std::string stage = "setup";
    try {
      if (argc != 2) throw std::invalid_argument("usage: logit-cap METALLIB");
      MetalBackend backend(argv[1]);
      stage = "cap";
      cap(backend);
      stage = "policies";
      policies(backend);
      std::cout << "logit_cap_metal_test: PASS\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << "logit_cap_metal_test: FAIL (" << stage << "): " << error.what() << '\n';
      return 1;
    }
  }
}

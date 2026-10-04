#include "ops/PromptLookup.hpp"
#include "tests/engine/TestChecks.hpp"

#include <chrono>
#include <iostream>
#include <vector>

namespace {

using splash::test::require;
using splash::ops::PromptLookup;
using splash::ops::rfc186::PromptSpan;

void testEmptyAndShortPrompts() {
  PromptLookup lookup;
  require(lookup.empty(), "lookup must be empty on initialization");
  require(lookup.size() == 0, "size must be 0 on initialization");

  std::vector<uint32_t> empty;
  lookup.indexPrompt(empty);
  require(lookup.empty(), "lookup must be empty after indexing empty prompt");

  std::array<uint32_t, 5> outDrafts{};
  std::array<uint32_t, 3> query = {1, 2, 3};
  require(lookup.propose(query, outDrafts, 5) == 0,
          "propose on empty lookup must return 0");
  require(!lookup.findMatch(query).has_value(),
          "findMatch on empty lookup must return nullopt");

  const std::vector<uint32_t> shortTokens = {10, 20};
  lookup.indexPrompt(shortTokens);
  require(lookup.size() == 2, "size must be 2 for short prompt");
  require(lookup.propose(query, outDrafts, 5) == 0,
          "propose on prompt shorter than 3 tokens must return 0");
}

void testExactMatchAndContinuation() {
  PromptLookup lookup;
  // Prompt with repeating boilerplate: [100, 200, 300, 400, 500, 600, 700]
  std::vector<uint32_t> prompt = {
      1, 2, 3, 100, 200, 300, 400, 500, 600, 700, 999
  };
  lookup.indexPrompt(prompt);
  require(lookup.size() == prompt.size(), "size mismatch after indexing");

  std::array<uint32_t, 4> outDrafts{};
  const std::array<uint32_t, 3> query = {100, 200, 300};
  const uint32_t proposed = lookup.propose(query, outDrafts, 4);

  require(proposed == 4, "expected 4 proposed tokens");
  require(outDrafts[0] == 400, "outDrafts[0] mismatch");
  require(outDrafts[1] == 500, "outDrafts[1] mismatch");
  require(outDrafts[2] == 600, "outDrafts[2] mismatch");
  require(outDrafts[3] == 700, "outDrafts[3] mismatch");
}

void testReverseChronologicalRecency() {
  PromptLookup lookup;
  // Sequence where [10, 20, 30] appears twice:
  // First occurrence continues with 40, 50.
  // Second (more recent) occurrence continues with 80, 90.
  std::vector<uint32_t> prompt = {
      1, 10, 20, 30, 40, 50, 99, 10, 20, 30, 80, 90, 999
  };
  lookup.indexPrompt(prompt);

  std::array<uint32_t, 2> outDrafts{};
  const std::array<uint32_t, 3> query = {10, 20, 30};
  const uint32_t proposed = lookup.propose(query, outDrafts, 2);

  require(proposed == 2, "expected 2 proposed tokens");
  require(outDrafts[0] == 80, "must prefer most recent occurrence (80)");
  require(outDrafts[1] == 90, "must prefer most recent occurrence (90)");
}

void testLongerMatchPreference() {
  PromptLookup lookup;
  // Prefix [5, 10, 20, 30] vs [99, 10, 20, 30]
  std::vector<uint32_t> prompt = {
      1, 5, 10, 20, 30, 77, 88, 2, 99, 10, 20, 30, 44, 55, 3
  };
  lookup.indexPrompt(prompt);

  // Query has 4 tokens matching the earlier sequence [5, 10, 20, 30]
  std::array<uint32_t, 2> outDrafts{};
  const std::array<uint32_t, 4> query = {5, 10, 20, 30};
  const uint32_t proposed = lookup.propose(query, outDrafts, 2);

  require(proposed == 2, "expected 2 proposed tokens");
  require(outDrafts[0] == 77, "longer match must take precedence over recent short match");
  require(outDrafts[1] == 88, "longer match continuation token mismatch");
}

void testIncrementalAppendToken() {
  PromptLookup lookup;
  std::vector<uint32_t> prompt = {1, 2, 3, 4, 5};
  lookup.indexPrompt(prompt);

  // Append new tokens as generation proceeds
  lookup.appendToken(101);
  lookup.appendToken(102);
  lookup.appendToken(103);
  lookup.appendToken(104);
  lookup.appendToken(105);

  require(lookup.size() == 10, "size must reflect appended tokens");

  // Query matching the newly appended stream
  std::array<uint32_t, 2> outDrafts{};
  const std::array<uint32_t, 3> query = {101, 102, 103};
  const uint32_t proposed = lookup.propose(query, outDrafts, 2);

  require(proposed == 2, "expected 2 proposed tokens from appended stream");
  require(outDrafts[0] == 104, "appended draft[0] mismatch");
  require(outDrafts[1] == 105, "appended draft[1] mismatch");
}

void testAmbiguityResolution() {
  PromptLookup lookup;
  // [10, 20, 30] followed by 40, and another [10, 20, 30] followed by 50
  std::vector<uint32_t> prompt = {
      1, 10, 20, 30, 40, 2, 10, 20, 30, 50, 3
  };
  lookup.indexPrompt(prompt);

  std::array<uint32_t, 2> outDrafts{};
  const std::array<uint32_t, 3> query = {10, 20, 30};

  // With unambiguous = false, returns most recent
  uint32_t proposed = lookup.propose(query, outDrafts, 1, 3, false);
  require(proposed == 1 && outDrafts[0] == 50, "default should return most recent");

  // With unambiguous = true, detects conflicting continuations and refuses
  proposed = lookup.propose(query, outDrafts, 1, 3, true);
  require(proposed == 0, "requireUnambiguous must return 0 on conflicting continuations");
}

void testRfc186Contract() {
  PromptLookup lookup;
  std::vector<uint32_t> prompt = {1, 2, 3};
  // Add a span of 20 identical predictable tokens
  for (uint32_t i = 0; i < 25; ++i) {
    prompt.push_back(1000 + i);
  }
  prompt.push_back(9999);
  lookup.indexPrompt(prompt);

  const std::array<uint32_t, 3> query = {1000, 1001, 1002};
  auto match = lookup.findMatch(query, 3, 16);
  require(match.has_value(), "findMatch must succeed for >= 16 tokens");
  require(match->length >= 16, "matched continuation span must be >= 16");
  require(splash::ops::rfc186::meetsMinSpan(match->length),
          "matched span must satisfy meetsMinSpan");
  require(lookup.tokens()[match->start] == 1003,
          "continuation must start immediately after matched query");

  // Now test span gate rejection when continuation is shorter than minSpan
  auto shortMatch = lookup.findMatch(query, 3, 30);
  require(!shortMatch.has_value(), "findMatch must return nullopt when available span < minSpan");
}

void testEnvGate() {
  // Unset environment variable defaults to false
  ::unsetenv(splash::ops::rfc186::kEnvVar);
  require(!PromptLookup::isEnvEnabled(), "isEnvEnabled must default to false");

  ::setenv(splash::ops::rfc186::kEnvVar, "1", 1);
  require(PromptLookup::isEnvEnabled(), "isEnvEnabled must be true when set to 1");

  ::setenv(splash::ops::rfc186::kEnvVar, "true", 1);
  require(PromptLookup::isEnvEnabled(), "isEnvEnabled must be true when set to true");

  ::setenv(splash::ops::rfc186::kEnvVar, "0", 1);
  require(!PromptLookup::isEnvEnabled(), "isEnvEnabled must be false when set to 0");

  ::unsetenv(splash::ops::rfc186::kEnvVar);
}

void testQueryLatency() {
  PromptLookup lookup;
  std::vector<uint32_t> prompt;
  for (uint32_t i = 0; i < 2000; ++i) {
    prompt.push_back(1000 + (i % 250));
  }
  lookup.indexPrompt(prompt);

  std::array<uint32_t, 8> outDrafts{};
  const std::array<uint32_t, 3> query = {1010, 1011, 1012};

  constexpr int kWarmup = 1000;
  for (int i = 0; i < kWarmup; ++i) {
    static_cast<void>(lookup.propose(query, outDrafts, 8));
  }

  constexpr int kIterations = 50000;
  const auto start = std::chrono::steady_clock::now();
  uint32_t totalTokens = 0;
  for (int i = 0; i < kIterations; ++i) {
    totalTokens += lookup.propose(query, outDrafts, 8);
  }
  const auto elapsed = std::chrono::steady_clock::now() - start;
  const double nsPerQuery =
      std::chrono::duration<double, std::nano>(elapsed).count() / kIterations;

  require(totalTokens > 0, "propose must have returned tokens in benchmark loop");
  require(nsPerQuery < 500.0, "query latency must be sub-microsecond (< 500 ns)");
}

} // namespace

int main() {
  try {
    testEmptyAndShortPrompts();
    testExactMatchAndContinuation();
    testReverseChronologicalRecency();
    testLongerMatchPreference();
    testIncrementalAppendToken();
    testAmbiguityResolution();
    testRfc186Contract();
    testEnvGate();
    testQueryLatency();
    std::cout << "Prompt lookup tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Prompt lookup test error: " << error.what() << '\n';
    return 1;
  }
}

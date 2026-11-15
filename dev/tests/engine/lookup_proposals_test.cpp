// The lookup proposals' index (model::LookupIndex). After every token
// appended, its match is what a direct search of the tokens finds: the
// earlier occurrence of the longest suffix, its length counted up to
// kLongestMatch, the most recent of the longest, none shorter than
// kMinimumMatch; on runs of one token, short periods and random tokens
// alike. Tokens appended at once match as those appended one at a time;
// matches and proposals reach across the prompt's end; and on two traced
// requests (the fixture), the proposals at every cycle are the offline
// replay's lookup chains.
#include "TestChecks.hpp"
#include "model/LookupProposals.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using splash::model::LookupIndex;
using splash::test::require;

namespace {

using Match = LookupIndex::Match;
// The replay's chains: a 32-row lane less its anchor.
constexpr uint32_t kChain = 31;

std::string describe(const Match &match) {
  return "(" + std::to_string(match.length) + ", " + std::to_string(match.end) + ")";
}

// The match of tokens' last token by direct search: each earlier end, most
// recent first, compared backwards up to kLongestMatch tokens.
Match searched(std::span<const uint32_t> tokens) {
  Match best;
  const size_t last = tokens.size() - 1;
  for (size_t end = last; end-- > 0;) {
    uint32_t length = 0;
    while (length < LookupIndex::kLongestMatch && length <= end && tokens[end - length] == tokens[last - length])
      ++length;
    if (length >= LookupIndex::kMinimumMatch && length > best.length)
      best = {length, static_cast<uint32_t>(end)};
  }
  return best;
}

LookupIndex indexOf(const std::vector<uint32_t> &tokens) {
  LookupIndex index;
  index.append(tokens);
  return index;
}

std::vector<uint32_t> proposalsOf(const LookupIndex &index, uint32_t count) {
  const std::span<const uint32_t> proposals = index.proposals(count);
  return {proposals.begin(), proposals.end()};
}

// Appending one token at a time, the match after each equals the search's.
void matchesTheSearch(const std::vector<uint32_t> &tokens, const std::string &name) {
  LookupIndex index;
  for (size_t size = 1; size <= tokens.size(); ++size) {
    index.append(std::span(tokens).subspan(size - 1, 1));
    const Match expected = searched(std::span(tokens).first(size));
    if (index.match() != expected)
      throw std::runtime_error(name + ": after " + std::to_string(size) + " tokens the match was " +
                               describe(index.match()) + ", not " + describe(expected));
  }
  require(indexOf(tokens).match() == index.match(), name + ": tokens appended at once matched otherwise");
  require(std::ranges::equal(index.tokens(), tokens), name + ": the index lost its tokens");
}

void searchedSequences() {
  std::mt19937 random(1114);
  matchesTheSearch(std::vector<uint32_t>(300, 7), "a run of one token");
  // After another token, the run's 64 tokens first share a class with that
  // token and then move to a class of their own as the run goes on.
  std::vector<uint32_t> afterAnother(300, 7);
  afterAnother[0] = 5;
  matchesTheSearch(afterAnother, "a run after another token");
  // Runs of random lengths, each of a token from three.
  std::vector<uint32_t> runs;
  while (runs.size() < 3000)
    runs.insert(runs.end(), 1 + random() % 150, static_cast<uint32_t>(random() % 3));
  matchesTheSearch(runs, "runs of random lengths");
  for (const uint32_t period : {2U, 3U, 7U, 63U, 64U, 65U}) {
    std::vector<uint32_t> tokens;
    for (uint32_t i = 0; i < 400; ++i)
      tokens.push_back(i % period);
    matchesTheSearch(tokens, "period " + std::to_string(period));
    // The same with a token changed every 50, so matches break and resume.
    for (size_t i = 25; i < tokens.size(); i += 50)
      tokens[i] = 1000;
    matchesTheSearch(tokens, "period " + std::to_string(period) + " with breaks");
  }
  for (const uint32_t alphabet : {2U, 3U, 4U, 8U, 50U, 150000U}) {
    for (int round = 0; round < 3; ++round) {
      std::uniform_int_distribution<uint32_t> token(0, alphabet - 1);
      std::vector<uint32_t> tokens(1500);
      for (uint32_t &value : tokens)
        value = token(random);
      // Copies of earlier stretches, as a file rewrite makes: long matches.
      for (size_t at = 500; at + 200 < tokens.size(); at += 400) {
        const size_t from = random() % (at - 200);
        std::copy_n(tokens.begin() + static_cast<std::ptrdiff_t>(from), 150 + random() % 50,
                    tokens.begin() + static_cast<std::ptrdiff_t>(at));
      }
      matchesTheSearch(tokens, std::to_string(alphabet) + " tokens, round " + std::to_string(round));
    }
  }
}

void longestThenMostRecent() {
  // "2 3 4 5" ended at 4; "3 4 5" ended at 8 too: the longer wins.
  const LookupIndex longest = indexOf({1, 2, 3, 4, 5, 9, 3, 4, 5, 2, 3, 4, 5});
  require(longest.match() == Match{4, 4}, "the longest match did not win over a more recent shorter one");
  require(proposalsOf(longest, kChain) == std::vector<uint32_t>{9, 3, 4, 5, 2, 3, 4, 5},
          "the proposals did not run from the match to the latest token");
  require(proposalsOf(longest, 3) == std::vector<uint32_t>{9, 3, 4}, "the proposals ran past their count");
  // "1 2 3" ended at 3 and at 8, after different tokens: the latest wins.
  const LookupIndex recent = indexOf({5, 1, 2, 3, 7, 6, 1, 2, 3, 8, 4, 1, 2, 3});
  require(recent.match() == Match{3, 8}, "the most recent of equal matches did not win");
  require(proposalsOf(recent, kChain) == std::vector<uint32_t>{8, 4, 1, 2, 3}, "the recent match's proposals");
  // Two tokens are too few to match.
  const LookupIndex two = indexOf({1, 2, 9, 1, 2});
  require(two.match() == Match{} && two.proposals(kChain).empty(), "a 2-token match proposed");
  require(indexOf({1, 2, 3, 9, 1, 2, 3}).match() == Match{3, 2}, "a 3-token match did not count");
  require(indexOf({4}).match() == Match{} && LookupIndex().proposals(kChain).empty(),
          "a sequence too short to repeat matched");
}

void countedUpToTheLongest() {
  // A 100-token stretch after 5 6 (at 2-101), again after 7 (103-202).
  const auto stretch = [](std::vector<uint32_t> &tokens) {
    for (uint32_t i = 0; i < 100; ++i)
      tokens.push_back(10 + i);
  };
  std::vector<uint32_t> tokens{5, 6};
  stretch(tokens);
  tokens.push_back(7);
  stretch(tokens);
  require(indexOf(tokens).match() == Match{LookupIndex::kLongestMatch, 101},
          "a 100-token match did not count as 64 tokens");
  // Again after 5 6 (205-304): the first copy matches 102 tokens, the second
  // 100; both count as 64, so the latest wins.
  tokens.push_back(5);
  tokens.push_back(6);
  stretch(tokens);
  require(indexOf(tokens).match() == Match{LookupIndex::kLongestMatch, 202},
          "of two matches past 64 tokens, the latest did not win");
  // 42 tokens into the third copy, the first copy matches all of them (5 6
  // included) and the second 40.
  tokens.resize(245);
  require(indexOf(tokens).match() == Match{42, 41}, "a 42-token match was not exact");
}

void acrossThePromptsEnd() {
  // The output's first tokens repeat the prompt's middle: its continuation
  // runs from the prompt into the output.
  LookupIndex index = indexOf({10, 11, 12, 13, 14, 15});
  index.append(std::vector<uint32_t>{12, 13});
  require(index.match() == Match{}, "two output tokens matched");
  index.append(std::vector<uint32_t>{14});
  require(index.match() == Match{3, 4}, "the output did not match the prompt");
  require(proposalsOf(index, kChain) == std::vector<uint32_t>{15, 12, 13, 14},
          "the proposals did not run on into the output");
  // An occurrence that spans the prompt's end: its last token is the
  // output's first.
  LookupIndex spanning = indexOf({20, 21, 22, 23});
  spanning.append(std::vector<uint32_t>{24, 25, 9, 22, 23});
  require(spanning.match() == Match{}, "a 2-token match counted");
  spanning.append(std::vector<uint32_t>{24});
  require(spanning.match() == Match{3, 4} && proposalsOf(spanning, 2) == std::vector<uint32_t>{25, 9},
          "a match ending at the output's first token was not found");
}

void appendsAtOnceOrApart() {
  std::mt19937 random(7);
  std::uniform_int_distribution<uint32_t> token(0, 5);
  std::vector<uint32_t> tokens(2000);
  for (uint32_t &value : tokens)
    value = token(random);
  LookupIndex apart;
  size_t at = 0;
  // Chunks of 1 to 40 tokens, as prefill chunks and decode cycles commit.
  while (at < tokens.size()) {
    const size_t count = std::min<size_t>(1 + random() % 40, tokens.size() - at);
    apart.append(std::span(tokens).subspan(at, count));
    at += count;
    require(apart.match() == searched(std::span(tokens).first(at)), "a chunked append matched otherwise");
  }
  const LookupIndex once = indexOf(tokens);
  require(once.match() == apart.match() && proposalsOf(once, kChain) == proposalsOf(apart, kChain),
          "appending at once and in chunks disagreed");
}

// The fixture: requests of a decode trace with the offline replay's match
// and lookup chain at each cycle (its header says how they were made).
void replaysTheTrace(const std::string &path) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("cannot open " + path);
  std::stringstream words;
  for (std::string line; std::getline(file, line);)
    if (!line.starts_with("#"))
      words << line << '\n';
  std::string word;
  uint32_t requests = 0, cycles = 0, matched = 0;
  LookupIndex index;
  std::vector<uint32_t> tokens;
  size_t appended = 0;
  while (words >> word) {
    if (word == "request") {
      size_t promptLength = 0, size = 0;
      words >> promptLength >> size;
      tokens.assign(size, 0);
      for (uint32_t &token : tokens)
        words >> token;
      index = LookupIndex();
      index.append(std::span(tokens).first(promptLength));
      appended = promptLength;
      ++requests;
      continue;
    }
    if (word != "cycle")
      throw std::runtime_error("unexpected fixture word " + word);
    size_t position = 0;
    uint32_t length = 0, end = 0, count = 0;
    words >> position >> length >> end >> count;
    std::vector<uint32_t> chain(count);
    for (uint32_t &token : chain)
      words >> token;
    if (!words || position >= tokens.size() || position < appended)
      throw std::runtime_error("malformed fixture cycle at " + std::to_string(position));
    index.append(std::span(tokens).subspan(appended, position + 1 - appended));
    appended = position + 1;
    const Match expected = length ? Match{std::min(length, LookupIndex::kLongestMatch), end} : Match{};
    if (index.match() != expected || proposalsOf(index, kChain) != chain)
      throw std::runtime_error("request " + std::to_string(requests) + ", cycle at " + std::to_string(position) +
                               ": the match was " + describe(index.match()) + ", the replay's " +
                               describe(expected));
    ++cycles;
    matched += length != 0;
  }
  require(requests == 2 && cycles == 239 && matched == 147, "the fixture did not hold the traced cycles");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("usage: lookup-proposals FIXTURE");
    searchedSequences();
    longestThenMostRecent();
    countedUpToTheLongest();
    acrossThePromptsEnd();
    appendsAtOnceOrApart();
    replaysTheTrace(argv[1]);
  } catch (const std::exception &error) {
    std::cerr << "lookup_proposals_test: FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "lookup_proposals_test: PASS\n";
  return 0;
}

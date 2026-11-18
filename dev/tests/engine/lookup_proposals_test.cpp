// The lookup proposals' index and acceptance estimate (model::LookupIndex,
// model::LookupAcceptance). After every token appended, the index's match is
// what a direct search of the tokens finds: the earlier occurrence of the
// longest suffix, its length counted up to kLongestMatch, the most recent of
// the longest, none shorter than kMinimumMatch; on runs of one token, short
// periods and random tokens alike. Tokens appended at once match as those
// appended one at a time; matches and proposals reach across the prompt's
// end; and on two traced requests (the fixture), the proposals at every
// cycle are the offline replay's lookup chains. The estimate is the fitted
// table until it observes; each accepted token and the rejected one are
// labels, those past the rejection none; a known agreement's label teaches
// Unknown too; counts weigh half as much kHalfLife labels later; and greedy
// lanes and sampled ones count apart. A cycle's outcome follows the chain's
// rows down the lane's tree, its tokens' agreements with the draft's
// candidates, to its first rejection, the first token past the verified
// rows, or past the tokens the lane retained.
#include "TestChecks.hpp"
#include "model/LookupProposals.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using splash::model::LookupAcceptance;
using splash::model::LookupAgreement;
using splash::model::LookupIndex;
using splash::test::rejects;
using splash::test::require;

namespace {

using Match = LookupIndex::Match;
// The replay's chains: a 32-row lane less its anchor.
constexpr uint32_t kChain = 31;
// The acceptance counts of greedy lanes and of lanes that sample.
constexpr bool kGreedy = false, kSampled = true;

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

void priorTable() {
  const LookupAcceptance acceptance;
  using Buckets = std::vector<std::pair<uint32_t, uint32_t>>;
  for (const auto &[length, bucket] : Buckets{{3, 0}, {4, 1}, {7, 1}, {8, 2}, {15, 2}, {16, 3}, {31, 3}, {32, 4},
                                              {63, 4}, {64, 5}, {1000000, 5}})
    require(LookupAcceptance::lengthBucket(length) == bucket, "length " + std::to_string(length) + "'s bucket");
  for (const auto &[depth, bucket] :
       Buckets{{0, 0}, {1, 1}, {2, 2}, {3, 2}, {4, 3}, {7, 3}, {8, 4}, {15, 4}, {16, 5}, {1000, 5}})
    require(LookupAcceptance::depthBucket(depth) == bucket, "depth " + std::to_string(depth) + "'s bucket");
  const auto &table = LookupAcceptance::kPrior;
  for (const uint32_t length : {3U, 5U, 12U, 20U, 40U, 64U, 5000U})
    for (const uint32_t depth : {0U, 1U, 3U, 6U, 9U, 30U})
      for (uint32_t a = 0; a < LookupAcceptance::kAgreements; ++a)
        require(acceptance.rate(kGreedy, length, depth, static_cast<LookupAgreement>(a)) ==
                    table[LookupAcceptance::lengthBucket(length)][LookupAcceptance::depthBucket(depth)][a],
                "an estimate before any observation was not the table's");
  // The fit's findings: the draft's top candidate is accepted 96% of the
  // time or more, more than its other candidates and tokens it lacks; and
  // without the draft's view, a longer match or a deeper token is no less
  // likely.
  const auto top = static_cast<uint32_t>(LookupAgreement::Top);
  const auto candidate = static_cast<uint32_t>(LookupAgreement::Candidate);
  const auto absent = static_cast<uint32_t>(LookupAgreement::Absent);
  const auto unknown = static_cast<uint32_t>(LookupAgreement::Unknown);
  for (uint32_t l = 0; l < LookupAcceptance::kLengthBuckets; ++l)
    for (uint32_t d = 0; d < LookupAcceptance::kDepthBuckets; ++d) {
      const auto &cell = table[l][d];
      require(cell[top] >= 0.96F && cell[top] > cell[candidate] && cell[top] > cell[absent],
              "the draft's top candidate was not the likeliest");
      require((l == 0 || cell[unknown] >= table[l - 1][d][unknown]) &&
                  (d == 0 || cell[unknown] >= table[l][d - 1][unknown]),
              "an unknown agreement's estimate fell with length or depth");
    }
}

void learnsFromOutcomes() {
  const auto &table = LookupAcceptance::kPrior;
  const double w = LookupAcceptance::kPriorLabels;
  const auto near = [](double a, double b) { return std::fabs(a - b) < 1e-6; };
  const auto cell = [&](uint32_t length, uint32_t depth, LookupAgreement agreement) {
    return double{table[LookupAcceptance::lengthBucket(length)][LookupAcceptance::depthBucket(depth)]
                       [static_cast<uint32_t>(agreement)]};
  };
  LookupAcceptance acceptance;
  // Five tokens verified, two accepted: depths 0 and 1 accepted, depth 2
  // rejected, depths 3 and 4 no label.
  const std::array verified{LookupAgreement::Top, LookupAgreement::Absent, LookupAgreement::Candidate,
                            LookupAgreement::Absent, LookupAgreement::Unknown};
  acceptance.observe(kGreedy, 10, verified, 2);
  const auto expect = [&](uint32_t depth, LookupAgreement agreement, double accepted, double labels,
                          const std::string &what) {
    const double expected = (accepted + w * cell(10, depth, agreement)) / (labels + w);
    require(near(acceptance.rate(kGreedy, 10, depth, agreement), expected), what);
  };
  expect(0, LookupAgreement::Top, 1, 1, "an accepted token was not a label");
  expect(1, LookupAgreement::Absent, 1, 1, "an accepted absent token was not a label");
  expect(2, LookupAgreement::Candidate, 0, 1, "the rejected token was not a label");
  expect(3, LookupAgreement::Absent, 0, 0, "a token past the rejection was a label");
  expect(4, LookupAgreement::Unknown, 0, 0, "a token past the rejection taught Unknown");
  // Depths 2 and 3 share a bucket: one label, the rejection.
  expect(2, LookupAgreement::Unknown, 0, 1, "a known agreement's rejection did not teach Unknown");
  expect(0, LookupAgreement::Unknown, 1, 1, "a known agreement's acceptance did not teach Unknown");
  require(near(acceptance.rate(kGreedy, 20, 0, LookupAgreement::Top), cell(20, 0, LookupAgreement::Top)),
          "a label taught another length's estimate");

  // Every token accepted: no rejection to label.
  LookupAcceptance all;
  all.observe(kGreedy, 3, std::array{LookupAgreement::Unknown, LookupAgreement::Unknown}, 2);
  require(near(all.rate(kGreedy, 3, 1, LookupAgreement::Unknown),
               (1 + w * cell(3, 1, LookupAgreement::Unknown)) / (1 + w)),
          "a chain accepted whole lost a label");
  // Nothing verified: nothing learned.
  all.observe(kGreedy, 3, std::span<const LookupAgreement>{}, 0);
  require(near(all.rate(kGreedy, 3, 1, LookupAgreement::Unknown),
               (1 + w * cell(3, 1, LookupAgreement::Unknown)) / (1 + w)),
          "an empty outcome changed the counts");

  // kHalfLife labels later, one weighs half: at depth 16+ (bucket 5) of
  // length 64, the earlier depth 0 label of length 10 halves.
  LookupAcceptance decaying;
  decaying.observe(kGreedy, 10, std::array{LookupAgreement::Top}, 1);
  std::vector<LookupAgreement> elsewhere(64, LookupAgreement::Unknown);
  const auto rounds = static_cast<uint32_t>(LookupAcceptance::kHalfLife) / 64;
  for (uint32_t round = 0; round < rounds; ++round)
    decaying.observe(kGreedy, 64, elsewhere, 64);
  require(near(decaying.rate(kGreedy, 10, 0, LookupAgreement::Top),
               (0.5 + w * cell(10, 0, LookupAgreement::Top)) / (0.5 + w)),
          "a label did not weigh half kHalfLife labels later");
  // Many rejections pull an estimate toward none.
  LookupAcceptance rejected;
  for (int round = 0; round < 1000; ++round)
    rejected.observe(kGreedy, 5, std::array{LookupAgreement::Top}, 0);
  require(rejected.rate(kGreedy, 5, 0, LookupAgreement::Top) < 0.01F, "rejections did not lower the estimate");

  rejects([&] { acceptance.observe(kGreedy, 2, verified, 0); }, "invalid lookup match length",
          "a 2-token match was observed");
  rejects([&] { acceptance.observe(kGreedy, 10, std::span(verified).first(2), 3); }, "invalid lookup outcome",
          "more tokens accepted than verified");
  rejects([&] { static_cast<void>(acceptance.rate(kGreedy, 2, 0, LookupAgreement::Top)); },
          "invalid lookup match length",
          "a 2-token match had an estimate");
}

// Greedy lanes and lanes that sample count apart: the outcomes of one leave
// the other's estimates at the table, and its labels decay its own counts
// alone.
void greedyAndSampledLanesCountApart() {
  using enum LookupAgreement;
  const auto prior = [](uint32_t length, uint32_t depth, LookupAgreement agreement) {
    return LookupAcceptance::kPrior[LookupAcceptance::lengthBucket(length)][LookupAcceptance::depthBucket(depth)]
                                   [static_cast<uint32_t>(agreement)];
  };
  LookupAcceptance acceptance;
  for (int round = 0; round < 100; ++round)
    acceptance.observe(kSampled, 10, std::array{Top}, 0);
  require(acceptance.rate(kSampled, 10, 0, Top) < 0.1F, "a sampled lane's rejections did not lower its estimate");
  require(acceptance.rate(kGreedy, 10, 0, Top) == prior(10, 0, Top) &&
              acceptance.rate(kGreedy, 10, 0, Unknown) == prior(10, 0, Unknown),
          "a sampled lane's outcomes moved a greedy lane's estimates");
  const float sampled = acceptance.rate(kSampled, 10, 0, Top);
  for (int round = 0; round < 100; ++round)
    acceptance.observe(kGreedy, 10, std::array{Absent}, 1);
  require(acceptance.rate(kGreedy, 10, 0, Absent) > 0.9F, "a greedy lane's acceptances did not raise its estimate");
  require(acceptance.rate(kSampled, 10, 0, Top) == sampled &&
              acceptance.rate(kSampled, 10, 0, Absent) == prior(10, 0, Absent),
          "a greedy lane's labels moved or decayed a sampled lane's counts");
}

} // namespace

// A tree whose draft chain is 10, 11, 12 and whose lookup chain, 10, 21, 22,
// 23, 24, joins its first row and branches after it: rows 0 (the anchor), 1
// (10), 2 (11 after 1), 3 (21 after 1), 4 (12 after 2), 5 (22 after 3), 6
// (23 after 5), in the order the selector took them. The draft's candidates:
// 10 its likeliest at position 0, 21 another of its candidates at 1, 22
// none of them at 2, 23 another at 3.
void outcomesOfCycles() {
  using splash::model::lookupAgreement;
  using splash::model::lookupOutcome;
  using splash::ops::TreeRow;
  constexpr uint32_t kCandidates = SPLASH_DRAFT_CANDIDATES;
  const std::vector<uint32_t> tokens{1, 10, 11, 21, 12, 22, 23};
  const std::vector<uint16_t> parents{SPLASH_TREE_ROOT, 0, 1, 1, 2, 3, 5};
  std::vector<TreeRow> tree(tokens.size());
  for (uint32_t row = 0; row < tree.size(); ++row)
    tree[row] = {row ? tree[parents[row]].ancestors | (1U << row) : 1U, parents[row],
                 static_cast<uint16_t>(row ? tree[parents[row]].depth + 1 : 0)};
  std::vector<uint32_t> candidates(SPLASH_DRAFT_PROPOSAL_TOKENS * kCandidates);
  for (uint32_t position = 0; position < SPLASH_DRAFT_PROPOSAL_TOKENS; ++position)
    for (uint32_t i = 0; i < kCandidates; ++i)
      candidates[position * kCandidates + i] = 1000 + position * kCandidates + i;
  candidates[0] = 10;
  candidates[1 * kCandidates] = 11;
  candidates[1 * kCandidates + 5] = 21;
  candidates[2 * kCandidates] = 12;
  candidates[3 * kCandidates + 15] = 23;
  require(lookupAgreement(10, 0, candidates) == LookupAgreement::Top &&
              lookupAgreement(21, 1, candidates) == LookupAgreement::Candidate &&
              lookupAgreement(22, 2, candidates) == LookupAgreement::Absent &&
              lookupAgreement(23, 3, candidates) == LookupAgreement::Candidate &&
              lookupAgreement(1000 + 7 * kCandidates, SPLASH_DRAFT_PROPOSAL_TOKENS, candidates) ==
                  LookupAgreement::Unknown,
          "a lookup token's agreement is not its draft candidates'");
  const std::vector<uint32_t> chain{10, 21, 22, 23, 24};
  const auto outcome = [&](uint32_t verifyRows, const std::vector<uint32_t> &retained) {
    return lookupOutcome(chain, tokens, tree, verifyRows, retained, candidates);
  };
  using enum LookupAgreement;
  const auto is = [](const splash::model::LookupOutcome &o, std::vector<LookupAgreement> verified,
                     uint32_t accepted) { return o.verified == verified && o.accepted == accepted; };
  // The target follows the chain three tokens, then takes another.
  require(is(outcome(7, {10, 21, 22, 99}), {Top, Candidate, Absent, Candidate}, 3),
          "a chain the target left at its fourth token");
  // Rows 5 and 6, the chain's third and fourth tokens, were not verified.
  require(is(outcome(5, {10, 21, 22}), {Top, Candidate}, 2), "a chain past the verified rows");
  // A limit cut the lane's tokens after two.
  require(is(outcome(7, {10, 21}), {Top, Candidate}, 2), "a chain past the retained tokens");
  // The target took the draft's second token, then its own.
  require(is(outcome(7, {10, 11, 12}), {Top, Candidate}, 1), "a chain the target left for the draft's");
  require(is(outcome(7, {11}), {Top}, 0), "a chain rejected at its first token");
  require(lookupOutcome(std::vector<uint32_t>{30, 31}, tokens, tree, 7, std::vector<uint32_t>{30, 31}, candidates)
              .verified.empty(),
          "a chain the tree does not hold was verified");
  rejects([&] { static_cast<void>(outcome(8, {10})); }, "invalid lookup outcome tree",
          "verify rows past the tree were accepted");
}

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
    priorTable();
    learnsFromOutcomes();
    greedyAndSampledLanesCountApart();
    outcomesOfCycles();
  } catch (const std::exception &error) {
    std::cerr << "lookup_proposals_test: FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "lookup_proposals_test: PASS\n";
  return 0;
}

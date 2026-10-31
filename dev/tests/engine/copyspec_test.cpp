// The prompt-lookup index and its copy acceptance (model::CopySpec.hpp), against
// the algorithm PLAN.md ports from dflash_mlx's engine/copyspec.py. The
// index buffers the prompt and every committed output token and drafts the
// continuation of the query that is the last five committed tokens plus the
// uncommitted one: every contiguous six-token window is bucketed by a 64-bit
// FNV-1a hash of its ids, but the hash is only a bucket key and every
// candidate is confirmed token by token against the query. Of the confirmed
// matches the draft comes from the one with the most tokens left after it
// (the earliest end), strictly, so the first chronological wins a tie; the
// copied span must be the full length asked for or the lookup misses, and a
// copied token in the forbidden (suppress/stop) ids voids the whole draft.
// A prompt of six tokens or fewer enables nothing, and appendCommitted grows
// the buffer one newest window at a time so generated text becomes copy
// source. A match that ends at the buffer's end has nothing to copy and
// misses.
//
// The conservative mode's one-strike rule (PLAN.md "Cycle integration"): a
// verified copy block that accepted nothing disables the request's index for
// the rest of the request; LOOKUP MISSES DO NOT COUNT, and only conservative
// one-strikes. The rule is engine-side; the small driver below pins it over
// CopySpecState.
//
// The copy acceptance is the per-position rate a request's copied proposals
// earn: each verified copy block counts its accepted proposals as reached
// acceptances and its first refused proposal, where the verify reached it, as
// a reached rejection, over a Beta(3, 1) prior.
#include "TestChecks.hpp"
#include "model/CopySpec.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using splash::model::CopySpecAcceptance;
using splash::model::CopySpecIndex;
using splash::model::CopySpecMode;
using splash::model::CopySpecState;
using splash::model::copySpecMode;
using splash::test::require;

namespace {

constexpr uint32_t kWindow = 6;

// The plan's FNV-1a over token ids: offset 14695981039346656037, prime
// 1099511628211, modulo 2^64, each id masked to 64 bits before the xor.
// This suite's collision pair is constructed against it.
uint64_t fnv1a(std::span<const uint32_t> tokens) {
  uint64_t hash = 14695981039346656037ULL;
  for (const uint32_t token : tokens) {
    hash ^= static_cast<uint64_t>(token);
    hash *= 1099511628211ULL;
  }
  return hash;
}

// Two distinct windows with the same FNV-1a hash, found by solving the hash
// for a difference of 2533359615 between the fifth ids' xor operands (whose
// product with the prime stays inside one 32-bit block of the state) and a
// sixth id that levels the low halves: both hash to kCollidingHash.
constexpr std::array<uint32_t, kWindow> kCollidingA{1, 2, 3, 4, 3348793250u,
                                                    2499805113u};
constexpr std::array<uint32_t, kWindow> kCollidingB{1, 2, 3, 4, 1365604445u,
                                                    500};
constexpr uint64_t kCollidingHash = 0xC04C9A0F0003519CULL;

// The index's draft for `staged` after `index`, at most `maxTokens` tokens.
std::vector<uint32_t> draft(const CopySpecIndex &index, uint32_t staged,
                            uint32_t maxTokens,
                            std::span<const uint32_t> forbidden = {}) {
  std::vector<uint32_t> out(maxTokens);
  const uint32_t count =
      index.draftAfter(staged, maxTokens, forbidden, out);
  require(count <= maxTokens, "the lookup wrote past its output span");
  out.resize(count);
  return out;
}

std::vector<uint32_t> expected(std::initializer_list<uint32_t> tokens) {
  return {tokens};
}

void modeStrings() {
  require(copySpecMode("off") == CopySpecMode::Off, "\"off\" is not the off mode");
  require(copySpecMode("conservative") == CopySpecMode::Conservative,
          "\"conservative\" is not the conservative mode");
  // auto is gone (its gate reset the verify budget every request shares).
  bool refused = false;
  try {
    static_cast<void>(copySpecMode("auto"));
  } catch (const std::invalid_argument &) {
    refused = true;
  }
  require(refused, "\"auto\" is still a copyspec mode");
}

// enabled is len(prompt) > 6 and is computed once at construction: a prompt
// of six tokens or fewer indexes nothing and drafts nothing however much is
// later committed, and one token more enables the whole machinery.
void enableCondition() {
  std::vector<uint32_t> prompt{1, 2, 3, 4, 5, 6};
  CopySpecIndex disabled(prompt);
  require(!disabled.enabled(), "six prompt tokens enabled the index");
  require(draft(disabled, 5, 1).empty(), "a disabled index drafted");
  // Growth never re-evaluates the condition: with the tokens appended this
  // buffer's tail would otherwise hit its own first window.
  const std::array<uint32_t, 6> growth{7, 1, 2, 3, 4, 5};
  disabled.appendCommitted(growth);
  require(!disabled.enabled(), "growth enabled a disabled index");
  require(draft(disabled, 6, 3).empty(), "a disabled index consulted its growth");
  prompt.push_back(7);
  CopySpecIndex enabled(prompt);
  require(enabled.enabled(), "seven prompt tokens did not enable the index");
  CopySpecIndex control(prompt);
  const std::array<uint32_t, 5> controlGrowth{1, 2, 3, 4, 5};
  control.appendCommitted(controlGrowth);
  require(draft(control, 6, 3) == expected({7, 1, 2}),
          "the enabled control of the disabled case did not draft");
}

// The window a query matches is the one ending at its sixth token; the copy
// is the continuation after that end, and it is all of what was asked for or
// nothing at all.
void windowIndexingAndFullLength() {
  const std::vector<uint32_t> prompt{1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 1, 2, 3, 4,
                                     5};
  CopySpecIndex index(prompt);
  // The query [1,2,3,4,5,6] is the tail [1,2,3,4,5] plus the staged 6; its
  // window ends at 6 and its continuation is the rest of the buffer.
  require(draft(index, 6, 3) == expected({7, 7, 7}),
          "a three-token draft is not the window's continuation");
  require(draft(index, 6, 9) ==
              expected({7, 7, 7, 7, 1, 2, 3, 4, 5}),
          "a full-buffer draft is not the window's continuation");
  require(draft(index, 6, 10).empty(),
          "a draft one token longer than the continuation was returned");
  require(draft(index, 99, 3).empty(), "a query that never occurred drafted");
  require(draft(index, 6, 0).empty(), "a zero-length lookup drafted");
  // The query is the last five tokens and the staged one; a single token off
  // is a different window and matches nothing.
  require(draft(index, 5, 3).empty(), "a query one token off drafted");
  // A match may end at the buffer's end, where there is nothing after it to
  // copy: the lookup must miss rather than return a short draft.
  const std::vector<uint32_t> ended{7, 8, 5, 5, 5, 5, 5, 5};
  CopySpecIndex atEnd(ended);
  require(draft(atEnd, 5, 1).empty(),
          "a match with no continuation after it drafted");
}

// Of the matches of the query the copy comes from the one with the most
// tokens left after its end, which is the earliest end; the later matches
// are never chosen, not even when the earliest cannot fill the request.
void maxAvailableMatchChoice() {
  const std::vector<uint32_t> prompt{
      1, 2, 3, 4, 5, 6, 8, 8, 8, 8, 8, 8, 9,
      1, 2, 3, 4, 5, 6, 9, 9, 1, 2, 3, 4, 5};
  CopySpecIndex index(prompt);
  // The query [1,2,3,4,5,6] matches at ends 6 (twenty tokens after it) and
  // 19 (seven): the copy must come from end 6.
  require(draft(index, 6, 4) == expected({8, 8, 8, 8}),
          "the draft came from the later match");
  require(draft(index, 6, 8) ==
              expected({8, 8, 8, 8, 8, 8, 9, 1}),
          "the eight-token draft came from the later match");
  // Only the earlier match can fill nine tokens; the later one cannot, so a
  // draft from it would either differ or miss entirely.
  require(draft(index, 6, 9) ==
              expected({8, 8, 8, 8, 8, 8, 9, 1, 2}),
          "the nine-token draft did not come from the earliest match");
}

// The hash is only a bucket key: two distinct windows with one FNV-1a hash
// must not draft each other's continuations.
void exactCompareCollisionGuard() {
  require(kCollidingA != kCollidingB, "the collision pair is not two windows");
  require(fnv1a(kCollidingA) == kCollidingHash &&
              fnv1a(kCollidingB) == kCollidingHash,
          "the collision pair does not collide under the plan's FNV-1a");
  // The buffer holds window A at end 6 (seventeen tokens after it) and
  // window B at end 16 (seven), then the tail [b0..b4] the query comes from.
  std::vector<uint32_t> prompt(kCollidingA.begin(), kCollidingA.end());
  prompt.insert(prompt.end(), {90, 91, 92, 93});
  prompt.insert(prompt.end(), kCollidingB.begin(), kCollidingB.end());
  prompt.insert(prompt.end(), {70, 71});
  prompt.insert(prompt.end(), kCollidingB.begin(), kCollidingB.begin() + 5);
  CopySpecIndex index(prompt);
  // The query is window B itself: only B may confirm, so the copy is what
  // follows B. A's continuation, which has more available and would be
  // chosen were the candidates not compared token by token, is wrong.
  require(draft(index, kCollidingB[5], 3) == expected({70, 71, kCollidingB[0]}),
          "a hash collision drafted the wrong window's continuation");
}

// A copied token the request forbids (its suppress/stop ids) voids the whole
// draft, and only the copied span is checked.
void forbiddenTokenVeto() {
  const std::vector<uint32_t> prompt{1, 2, 3, 4, 5, 6, 40, 77, 41, 42, 43,
                                     1, 2, 3, 4, 5};
  CopySpecIndex index(prompt);
  // The continuation after end 6 is [40, 77, 41, 42, 43, ...].
  const std::array<uint32_t, 1> forbidden77{77};
  require(draft(index, 6, 3, forbidden77).empty(),
          "a forbidden token inside the copied span was drafted");
  require(draft(index, 6, 1, forbidden77) == expected({40}),
          "a forbidden token past the copied span vetoed the draft");
  const std::array<uint32_t, 1> forbidden43{43};
  require(draft(index, 6, 4, forbidden43) == expected({40, 77, 41, 42}),
          "a forbidden token past the copied span vetoed the draft");
  require(draft(index, 6, 5, forbidden43).empty(),
          "a forbidden token at the end of the copied span was drafted");
  require(draft(index, 6, 3) == expected({40, 77, 41}),
          "an empty forbidden set vetoed the draft");
}

// appendCommitted indexes the newest window of every id it appends, whatever
// the batching: committed output becomes copy source, including windows that
// end inside one append's span and continuations drawn from it.
void appendCommittedGrowth() {
  const std::vector<uint32_t> prompt{1, 2, 3, 4, 5, 6, 7};
  CopySpecIndex index(prompt);
  CopySpecIndex oneByOne(prompt);
  // Nothing here yet: the query below is not in the prompt.
  require(draft(index, 15, 2).empty(), "the prompt held the query");
  const std::vector<uint32_t> committed{10, 11, 12, 13, 14, 15, 90,
                                        10, 11, 12, 13, 14};
  index.appendCommitted(committed);
  for (const uint32_t token : committed) {
    const std::array<uint32_t, 1> one{token};
    oneByOne.appendCommitted(one);
  }
  // The tail [10,11,12,13,14] and the staged 15 are the window that ends at
  // 13, inside the appended span: its continuation [90, 10, 11, 12, 13, 14]
  // is appended output too.
  require(draft(index, 15, 2) == expected({90, 10}),
          "committed output did not become copy source");
  require(draft(index, 15, 6) ==
              expected({90, 10, 11, 12, 13, 14}),
          "the whole appended continuation was not copied");
  require(draft(index, 15, 7).empty(),
          "a draft past the appended continuation was returned");
  require(draft(oneByOne, 15, 6) == draft(index, 15, 6),
          "one-by-one appends indexed differently than one span");
}

// PLAN.md's conservative one-strike rule over the per-request state, as
// Runtime.cpp will apply it. The rule itself is engine-side (it is not part
// of model/CopySpec.hpp): this driver pins the documented rule so a change
// to either side is a change here.
bool consults(const CopySpecState &state, CopySpecMode mode) {
  return mode != CopySpecMode::Off && !state.disabled;
}

void noteVerifiedBlock(CopySpecState &state, CopySpecMode mode,
                       bool usedCopyBlock, uint32_t acceptanceLen) {
  if (mode == CopySpecMode::Conservative && usedCopyBlock && acceptanceLen == 0)
    state.disabled = true;
}

void conservativeOneStrike() {
  const std::vector<uint32_t> prompt{1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4,
                                     5};
  CopySpecState state{CopySpecIndex(prompt), false, CopySpecAcceptance{}};
  require(consults(state, CopySpecMode::Conservative),
          "a fresh request does not consult its index");
  require(!consults(state, CopySpecMode::Off), "off consulted the index");
  // The query hits, so the rule's inputs are those of a real copy cycle.
  require(!draft(state.index, 6, 3).empty(),
          "the one-strike scenario lost its lookup hit");
  // A verified copy block that accepted nothing disables the request.
  noteVerifiedBlock(state, CopySpecMode::Conservative, true, 0);
  require(state.disabled, "a verified zero-acceptance copy block did not disable");
  require(!consults(state, CopySpecMode::Conservative),
          "a disabled request still consults its index");
  // The index itself is untouched: the state suppresses consulting, so a
  // later query would still hit were the rule to allow it.
  require(!draft(state.index, 6, 3).empty(),
          "the one-strike rule damaged the index");
  // LOOKUP MISSES DO NOT COUNT: a fallback cycle that accepted nothing
  // leaves the request enabled.
  CopySpecState misses{CopySpecIndex(prompt), false, CopySpecAcceptance{}};
  for (uint32_t cycle = 0; cycle < 4; ++cycle)
    noteVerifiedBlock(misses, CopySpecMode::Conservative, false, 0);
  require(!misses.disabled, "a lookup miss disabled the request");
  // Nor does a copy block that accepted something.
  CopySpecState accepted{CopySpecIndex(prompt), false, CopySpecAcceptance{}};
  noteVerifiedBlock(accepted, CopySpecMode::Conservative, true, 3);
  require(!accepted.disabled, "an accepted copy block disabled the request");
  // Only conservative one-strikes; off never drafts at all.
  CopySpecState off{CopySpecIndex(prompt), false, CopySpecAcceptance{}};
  noteVerifiedBlock(off, CopySpecMode::Off, true, 0);
  require(!off.disabled, "a mode other than conservative one-striked");
}

bool near(double value, double expected) {
  return value > expected - 1.0e-12 && value < expected + 1.0e-12;
}

void copyAcceptance() {
  // The prior alone: three accepted to one refused.
  CopySpecAcceptance fresh;
  require(near(fresh.rate(), 0.75), "the prior is not Beta(3, 1)");
  // A fully accepted block of seven: seven reached acceptances, no
  // rejection, since the verify reached no refused proposal.
  CopySpecAcceptance full;
  full.observe(7, 7);
  require(near(full.rate(), 10.0 / 11.0), "a fully accepted block counted a rejection");
  // Two of seven accepted: two acceptances and the third proposal's refusal.
  CopySpecAcceptance partial;
  partial.observe(7, 2);
  require(near(partial.rate(), 5.0 / 7.0), "a partly accepted block miscounted");
  // A copy that keeps verifying one token of seven falls toward one half,
  // where a block claims fewer rows.
  CopySpecAcceptance poor;
  for (uint32_t cycle = 0; cycle < 50; ++cycle)
    poor.observe(7, 1);
  require(near(poor.rate(), 53.0 / 104.0), "repeated one-token copies miscounted");
  // Rows the verify did not reach say nothing: three verified proposals all
  // accepted are three acceptances, not a rejection at the fourth.
  CopySpecAcceptance capped;
  capped.observe(3, 3);
  require(near(capped.rate(), 6.0 / 7.0), "an unverified proposal counted as refused");
  // An acceptance count past the verified proposals is clamped to them.
  CopySpecAcceptance clamped;
  clamped.observe(2, 5);
  require(near(clamped.rate(), 5.0 / 6.0), "acceptance past the verified rows counted");
}

} // namespace

int main() {
  try {
    modeStrings();
    enableCondition();
    windowIndexingAndFullLength();
    maxAvailableMatchChoice();
    exactCompareCollisionGuard();
    forbiddenTokenVeto();
    appendCommittedGrowth();
    conservativeOneStrike();
    copyAcceptance();
  } catch (const std::exception &error) {
    std::cerr << "copyspec_test: FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "copyspec_test: PASS\n";
  return 0;
}

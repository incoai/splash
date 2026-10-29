// The prompt-lookup index and its auto gate (model::CopySpec.hpp), against
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
// one-strikes (auto is the gate's job). The rule is engine-side; the small
// driver below pins it over CopySpecState.
//
// The auto gate is the four-phase windowed A/B latch of copyspec.py, driven
// only through recordCycle/engageCopy/takeReset: measure_off for 24 cycles
// (suppressed) sets the baseline rate; measure_on probes until six
// copy-sourced cycles or 48 total; the probe engages exactly when a copy was
// seen and the probe's rate (commits per cycle cost second) is at least the
// baseline's (margin 1.0); an engage latches for 200 cycles and then hands
// out a one-shot reset; a loss sends the gate dormant for
// 200 * 2^min(consecutiveLosses, 4) cycles (doublings capped at 4, the count
// reset by any engage) and also hands out the reset, and the measure_off
// window that follows the dormancy is counted exactly.
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

using splash::model::CopySpecAutoGate;
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
  require(copySpecMode("auto") == CopySpecMode::Auto, "\"auto\" is not the auto mode");
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
  CopySpecState state{CopySpecIndex(prompt), false, CopySpecAutoGate{}};
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
  CopySpecState misses{CopySpecIndex(prompt), false, CopySpecAutoGate{}};
  for (uint32_t cycle = 0; cycle < 4; ++cycle)
    noteVerifiedBlock(misses, CopySpecMode::Conservative, false, 0);
  require(!misses.disabled, "a lookup miss disabled the request");
  // Nor does a copy block that accepted something.
  CopySpecState accepted{CopySpecIndex(prompt), false, CopySpecAutoGate{}};
  noteVerifiedBlock(accepted, CopySpecMode::Conservative, true, 3);
  require(!accepted.disabled, "an accepted copy block disabled the request");
  // Only conservative one-strikes; auto is the gate's job and off never
  // drafts at all.
  for (const CopySpecMode mode : {CopySpecMode::Auto, CopySpecMode::Off}) {
    CopySpecState other{CopySpecIndex(prompt), false, CopySpecAutoGate{}};
    noteVerifiedBlock(other, mode, true, 0);
    require(!other.disabled, "a mode other than conservative one-striked");
  }
}

void autoGateConstants() {
  require(CopySpecAutoGate::kProbeOffCycles == 24,
          "the measure_off window is not 24 cycles");
  require(CopySpecAutoGate::kProbeOnMinCopy == 6,
          "the probe does not end at six copy-sourced cycles");
  require(CopySpecAutoGate::kProbeOnMaxCycles == 48,
          "the probe does not end at 48 cycles");
  require(CopySpecAutoGate::kLatchCycles == 200,
          "the latch is not 200 cycles");
  require(CopySpecAutoGate::kBackoffCap == 4,
          "the backoff exponent is not capped at 4");
}

// The measure_off window: 24 suppressed cycles set the baseline rate, two
// commits per 1000 ns here, and the probe starts at the 24th.
void measureOff(CopySpecAutoGate &gate) {
  for (uint32_t cycle = 1; cycle <= CopySpecAutoGate::kProbeOffCycles; ++cycle) {
    require(!gate.engageCopy(),
            "measure_off consulted the index at cycle " + std::to_string(cycle));
    gate.recordCycle(false, 2, 1000);
  }
  require(gate.engageCopy(), "the probe did not start after 24 cycles");
}

// Six copy-sourced cycles at one commit per 1000 ns lose the probe (half the
// baseline rate): the gate goes dormant and hands out its reset once.
void losingProbe(CopySpecAutoGate &gate) {
  for (uint32_t copy = 0; copy < CopySpecAutoGate::kProbeOnMinCopy; ++copy)
    gate.recordCycle(true, 1, 1000);
  require(!gate.engageCopy(), "a losing probe engaged");
  require(gate.takeReset(), "a losing probe asked for no reset");
  require(!gate.takeReset(), "the reset flag is not one-shot");
}

// The dormancy of `dormant` cycles and the measure_off window after it: the
// gate may draft again only after dormant + 24 recordCycle calls, exactly.
void dormantThenProbe(CopySpecAutoGate &gate, uint32_t dormant) {
  const uint32_t offAgain = dormant + CopySpecAutoGate::kProbeOffCycles;
  for (uint32_t cycle = 1; cycle <= offAgain; ++cycle) {
    gate.recordCycle(false, 2, 1000);
    if (cycle < offAgain)
      require(!gate.engageCopy(),
              "the gate left its backoff early at cycle " + std::to_string(cycle));
  }
  require(gate.engageCopy(), "the gate did not probe after its backoff");
}

void autoGatePhases() {
  // The latch: an equal-rate probe engages (on_rate >= off_rate, margin
  // 1.0), holds 200 cycles, then hands out one reset and measures again.
  {
    CopySpecAutoGate gate;
    measureOff(gate);
    for (uint32_t copy = 0; copy < CopySpecAutoGate::kProbeOnMinCopy; ++copy) {
      require(gate.engageCopy(), "the probe suppressed a cycle before its sixth copy");
      gate.recordCycle(true, 2, 1000);
    }
    require(gate.engageCopy(), "an equal-rate probe did not engage");
    for (uint32_t cycle = 1; cycle <= CopySpecAutoGate::kLatchCycles; ++cycle) {
      require(gate.engageCopy(), "the latch ended early at cycle " + std::to_string(cycle));
      gate.recordCycle(false, 2, 1000);
    }
    require(!gate.engageCopy(), "the latch outlived 200 cycles");
    require(gate.takeReset(), "the latch end asked for no reset");
    require(!gate.takeReset(), "the reset flag is not one-shot");
  }
  // The probe's cycle cap: three copy cycles among 48 decide at the cap,
  // with the copy cycles carrying the whole window's commits. The latch that
  // follows ends exactly 200 cycles after the probe's 48th: a probe that
  // resolved early would end it early too.
  {
    CopySpecAutoGate gate;
    measureOff(gate);
    for (uint32_t cycle = 1; cycle <= CopySpecAutoGate::kProbeOnMaxCycles;
         ++cycle) {
      if (cycle < CopySpecAutoGate::kProbeOnMaxCycles)
        require(gate.engageCopy(),
                "the probe ended before its 48th cycle at " + std::to_string(cycle));
      const bool copy = cycle == 5 || cycle == 20 || cycle == 40;
      gate.recordCycle(copy, copy ? 16 : 2, 1000);
    }
    require(gate.engageCopy(), "a winning probe at the cycle cap did not engage");
    for (uint32_t cycle = 1; cycle <= CopySpecAutoGate::kLatchCycles; ++cycle) {
      require(gate.engageCopy(), "the latch ended early at cycle " + std::to_string(cycle));
      gate.recordCycle(false, 2, 1000);
    }
    require(!gate.engageCopy(), "the latch outlived 200 cycles");
  }
  // Engage needs a copy seen: a probe whose cycles all tie the baseline but
  // never used a copy block goes dormant at the cap, for the first backoff.
  {
    CopySpecAutoGate gate;
    measureOff(gate);
    for (uint32_t cycle = 1; cycle <= CopySpecAutoGate::kProbeOnMaxCycles;
         ++cycle) {
      if (cycle < CopySpecAutoGate::kProbeOnMaxCycles)
        require(gate.engageCopy(),
                "the probe ended before its 48th cycle at " + std::to_string(cycle));
      gate.recordCycle(false, 2, 1000);
    }
    require(!gate.engageCopy(), "a probe that saw no copy engaged");
    require(gate.takeReset(), "a losing probe asked for no reset");
    require(!gate.takeReset(), "the reset flag is not one-shot");
    dormantThenProbe(gate, CopySpecAutoGate::kLatchCycles);
  }
  // The backoff doubles with consecutive losses and caps at 2^4: the dormant
  // periods are 200, 400, 800, 1600, 3200 and 3200 cycles, each followed by
  // its own measure_off window of 24.
  {
    CopySpecAutoGate gate;
    measureOff(gate);
    for (uint32_t loss = 0; loss < 6; ++loss) {
      losingProbe(gate);
      const uint32_t dormant =
          CopySpecAutoGate::kLatchCycles *
          (1u << std::min(loss, CopySpecAutoGate::kBackoffCap));
      dormantThenProbe(gate, dormant);
    }
  }
  // An engage resets the losses: after a loss and a winning latch the next
  // loss starts the backoff over at 200, not at 400.
  {
    CopySpecAutoGate gate;
    measureOff(gate);
    losingProbe(gate);
    dormantThenProbe(gate, CopySpecAutoGate::kLatchCycles);
    for (uint32_t copy = 0; copy < CopySpecAutoGate::kProbeOnMinCopy; ++copy)
      gate.recordCycle(true, 8, 1000);
    require(gate.engageCopy(), "a winning probe did not engage");
    for (uint32_t cycle = 1; cycle <= CopySpecAutoGate::kLatchCycles; ++cycle) {
      require(gate.engageCopy(), "the latch ended early at cycle " + std::to_string(cycle));
      gate.recordCycle(false, 2, 1000);
    }
    require(!gate.engageCopy(), "the latch outlived 200 cycles");
    require(gate.takeReset(), "the latch end asked for no reset");
    measureOff(gate);
    losingProbe(gate);
    dormantThenProbe(gate, CopySpecAutoGate::kLatchCycles);
  }
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
    autoGateConstants();
    autoGatePhases();
  } catch (const std::exception &error) {
    std::cerr << "copyspec_test: FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "copyspec_test: PASS\n";
  return 0;
}

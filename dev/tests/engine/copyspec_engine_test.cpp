// The prompt-lookup (copyspec) cycle integration (PLAN.md "Cycle
// integration") around the request's real per-request components
// (model/CopySpec.hpp), on a scripted greedy target: the whole output stream
// is decided in advance, so any token the integration commits wrongly shows
// up as a transcript mismatch. The scenario is the one the feature targets:
// a prompt containing a repeated n-gram, greedy decoding, --verify-rows all
// (eight rows a block), baseline (copyspec off) against copyspec enabled.
// The outputs must be byte-identical whatever the blocks were, and on a copy
// cycle whose target agrees the drafted tokens equal the accepted ones and
// the committed tokens are exactly the copied block.
//
// The loop follows the documented semantics exactly: blockLen =
// clamp(1, 8, remaining); the index is consulted only for a mode that is on
// and a request the one-strike rule has not disabled; on a hit the copied
// span replaces the draft block; the
// commit is the anchor plus the accepted proposals (1 + acceptanceLen) and
// the target's token at the first mismatch, or the bonus token, is the next
// anchor; every committed id is appended to the index, so generated text
// becomes copy source; a verified copy block that accepted nothing is
// conservative mode's one strike and a lookup miss is not.
//
// What this test cannot reach: the production seams the core workstream
// owns in Runtime.cpp — the host step that overwrites a hit lane's
// ProposedTokens/VerifyConfidence before the grammar simulation reads them,
// the VerifyBudget observe exclusion for copy lanes, the draft-state commit
// on lookup cycles, and the clamp of proposals at min(7, maximumRetained -
// 1). The loop below stands in for those seams and pins the behavior they
// must deliver; when they land, these scenarios belong behind the real
// decode path. The draft selector's verify-row choice under a lookup lane's
// confidence is covered by copyspec_acceptance_test.mm.
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

using splash::model::CopySpecIndex;
using splash::model::CopySpecMode;
using splash::model::CopySpecState;
using splash::test::require;

namespace {

constexpr uint32_t kBlockRows = 8;
constexpr uint32_t kProposals = kBlockRows - 1;
// A fallback draft's wrong token: outside every scenario's vocabulary.
constexpr uint32_t kFiller = 0;

// One decode cycle's outcome, as the engine's speculation counters see it.
struct Cycle final {
  uint32_t drafted = 0;
  uint32_t accepted = 0;
  bool usedCopyBlock = false;
  std::vector<uint32_t> copied;
  std::vector<uint32_t> committed;
};

struct Run final {
  std::vector<uint32_t> transcript;
  std::vector<Cycle> cycles;
  uint32_t copyspecHits = 0;
  uint32_t copyspecTokens = 0;
  bool disabled = false;
  // What the raw index would still draft at the end, whatever the state
  // allows: the one-strike rule suppresses consulting, not the index.
  uint32_t rawFinalLookup = 0;
};

// The scripted target's greedy stream: `script[position]` is the token at
// output position `position`, so a commit that deviates from the script
// breaks the transcript's agreement with it.
struct Request final {
  std::vector<uint32_t> prompt;
  std::vector<uint32_t> script;
  uint32_t maxNewTokens = 0;
};

// PLAN.md's conservative one-strike rule over the per-request state, as
// Runtime.cpp will apply it (the rule is engine-side, not part of
// model/CopySpec.hpp; copyspec_test.cpp pins it on its own).
bool consults(const CopySpecState &state, CopySpecMode mode) {
  return mode != CopySpecMode::Off && !state.disabled;
}

Run run(const Request &request, CopySpecMode mode) {
  require(request.script.size() >= request.maxNewTokens,
          "the scripted target ends before its budget");
  CopySpecState state{CopySpecIndex(request.prompt), false, {}};
  Run result;
  uint32_t generated = 0;
  uint32_t staged = request.script[0];
  while (generated < request.maxNewTokens) {
    const uint32_t remaining = request.maxNewTokens - generated;
    const uint32_t blockLen = std::clamp(remaining, 1u, kBlockRows);
    Cycle cycle;
    std::array<uint32_t, kProposals> proposals{};
    if (blockLen > 1) {
      const uint32_t maxTokens = blockLen - 1;
      if (consults(state, mode)) {
        uint32_t count = state.index.draftAfter(
            staged, maxTokens, {}, std::span<uint32_t>(proposals).first(maxTokens));
        require(count == 0 || count == maxTokens,
                "the lookup returned a partial block");
        if (count) {
          cycle.usedCopyBlock = true;
          cycle.drafted = count;
          cycle.copied.assign(proposals.begin(), proposals.begin() + count);
        }
      }
      if (!cycle.usedCopyBlock) {
        // The DFlash fallback: a shifted guess of the target's stream, wrong
        // at its first proposal, so fallback cycles exercise the correction
        // and never draft the scripted continuation by accident.
        cycle.drafted = maxTokens;
        for (uint32_t position = 0; position < maxTokens; ++position) {
          const uint32_t at = generated + 2 + position;
          proposals[position] = at < request.script.size() ? request.script[at] : kFiller;
        }
      }
    }
    // Verify: the proposals are compared with the target's tokens after the
    // anchor, up to the first mismatch.
    while (cycle.accepted < cycle.drafted &&
           proposals[cycle.accepted] == request.script[generated + 1 + cycle.accepted])
      ++cycle.accepted;
    // Commit 1 + acceptanceLen; the target's token at the first mismatch, or
    // the bonus token, is the next anchor.
    cycle.committed.push_back(staged);
    cycle.committed.insert(cycle.committed.end(), proposals.begin(),
                           proposals.begin() + cycle.accepted);
    if (cycle.committed.size() > remaining)
      cycle.committed.resize(remaining);
    result.transcript.insert(result.transcript.end(), cycle.committed.begin(),
                             cycle.committed.end());
    state.index.appendCommitted(cycle.committed);
    generated += static_cast<uint32_t>(cycle.committed.size());
    if (cycle.usedCopyBlock) {
      ++result.copyspecHits;
      result.copyspecTokens += cycle.drafted;
    }
    // Conservative's one strike: a verified copy block that accepted
    // nothing. LOOKUP MISSES DO NOT COUNT, and only conservative.
    if (mode == CopySpecMode::Conservative && cycle.usedCopyBlock &&
        cycle.accepted == 0)
      state.disabled = true;
    result.cycles.push_back(std::move(cycle));
    // The next anchor is the target's token at the first mismatch, or the
    // bonus token; the loop may end before it is committed.
    if (generated < request.script.size())
      staged = request.script[generated];
  }
  result.disabled = state.disabled;
  std::array<uint32_t, kProposals> out{};
  result.rawFinalLookup = state.index.draftAfter(staged, kProposals, {}, out);
  return result;
}

// The repeated n-gram of the scenario's prompt and target: a twelve-token
// pattern, twice in the prompt, and the target keeps cycling it.
std::vector<uint32_t> pattern(size_t positions) {
  std::vector<uint32_t> tokens;
  for (size_t at = 0; at < positions; ++at)
    tokens.push_back(10 + static_cast<uint32_t>(at % 12));
  return tokens;
}

// Every copy cycle where the target agreed with the block drafted exactly
// what it accepted and committed exactly what it copied.
void requireAgreeingCycles(const Run &run, const std::string &label) {
  for (size_t at = 0; at < run.cycles.size(); ++at) {
    const Cycle &cycle = run.cycles[at];
    if (!cycle.usedCopyBlock)
      continue;
    const std::string cycleLabel =
        label + " copy cycle " + std::to_string(at + 1);
    if (cycle.accepted == cycle.drafted) {
      require(cycle.accepted > 0, cycleLabel + " accepted nothing");
      require(cycle.committed.size() == cycle.accepted + 1,
              cycleLabel + " did not commit 1 + acceptanceLen");
      require(std::equal(cycle.copied.begin(), cycle.copied.end(),
                         cycle.committed.begin() + 1),
              cycleLabel + " did not commit its copied block");
    }
  }
}

// A prompt containing a repeated n-gram, --verify-rows all, greedy: the
// copyspec output is the baseline output token for token, and every copy
// cycle the target agrees with drafts exactly what it accepts.
void repeatedNgramOutputsIdentical() {
  const Request request{pattern(24), pattern(72), 60};
  const Run baseline = run(request, CopySpecMode::Off);
  const Run copied = run(request, CopySpecMode::Conservative);
  require(baseline.transcript == copied.transcript,
          "copyspec changed the greedy output");
  require(copied.transcript ==
              std::vector<uint32_t>(request.script.begin(),
                                    request.script.begin() + request.maxNewTokens),
          "the copyspec transcript is not the scripted target");
  require(baseline.copyspecHits == 0, "the baseline used copy blocks");
  require(copied.copyspecHits == 8,
          "the copyspec run used " + std::to_string(copied.copyspecHits) +
              " copy cycles, not 8");
  require(!copied.disabled, "an all-agreeing run was one-striked");
  uint32_t drafted = 0;
  for (const Cycle &cycle : copied.cycles) {
    require(cycle.usedCopyBlock, "a cycle of the copyspec run did not draft");
    require(cycle.accepted == cycle.drafted,
            "a copy cycle the target agreed with drafted " +
                std::to_string(cycle.drafted) + " and accepted " +
                std::to_string(cycle.accepted));
    drafted += cycle.drafted;
  }
  require(copied.copyspecTokens == drafted,
          "copyspecTokens does not count the copied proposals");
  requireAgreeingCycles(copied, "repeated n-gram");
}

// Conservative's one strike: where the target refuses the first copied
// token the request stops consulting its index for good, while its output
// stays on the scripted stream.
void oneStrikeOnVerifiedZeroAcceptance() {
  std::vector<uint32_t> script = pattern(72);
  // The target turns off the pattern right after the second cycle's anchor,
  // so that cycle's copied block is refused at its first proposal.
  script[9] = 900009;
  const Request request{pattern(24), script, 60};
  const Run run0 = run(request, CopySpecMode::Conservative);
  require(run0.transcript ==
              std::vector<uint32_t>(script.begin(),
                                    script.begin() + request.maxNewTokens),
          "the one-strike run left the scripted target");
  require(run0.disabled,
          "a verified zero-acceptance copy block did not disable");
  require(run0.copyspecHits == 2,
          "the one-strike run used " + std::to_string(run0.copyspecHits) +
              " copy cycles, not 2");
  uint32_t refused = 0;
  for (size_t at = 0; at < run0.cycles.size(); ++at) {
    const Cycle &cycle = run0.cycles[at];
    if (cycle.usedCopyBlock && cycle.accepted == 0) {
      ++refused;
      require(at + 1 == run0.copyspecHits,
              "a copy block was used after the zero-acceptance one");
    }
  }
  require(refused == 1, "the run refused " + std::to_string(refused) +
                            " copy blocks, not 1");
  // The index still hits at the end; the one-strike rule suppresses
  // consulting it, not its ability to draft.
  require(run0.rawFinalLookup > 0,
          "the scenario lost its lookup hit: it no longer tests the rule");
}

// LOOKUP MISSES DO NOT COUNT: a stretch of target text nothing can match
// falls back to DFlash with zero acceptance, yet copying resumes where the
// text repeats again.
void lookupMissesDoNotDisable() {
  std::vector<uint32_t> script = pattern(24);
  for (uint32_t token = 0; token < 12; ++token)
    script.push_back(500 + token); // nothing here matches anything
  const std::vector<uint32_t> tail = pattern(48);
  script.insert(script.end(), tail.begin(), tail.end());
  const Request request{pattern(24), script, 72};
  const Run run0 = run(request, CopySpecMode::Conservative);
  require(run0.transcript ==
              std::vector<uint32_t>(script.begin(),
                                    script.begin() + request.maxNewTokens),
          "the miss run left the scripted target");
  require(!run0.disabled, "a lookup miss disabled the request");
  require(run0.copyspecHits == 7,
          "the miss run used " + std::to_string(run0.copyspecHits) +
              " copy cycles, not 7");
  size_t firstHit = run0.cycles.size(), lastHit = 0, misses = 0;
  for (size_t at = 0; at < run0.cycles.size(); ++at) {
    const Cycle &cycle = run0.cycles[at];
    if (cycle.usedCopyBlock) {
      firstHit = std::min(firstHit, at);
      lastHit = std::max(lastHit, at);
    } else if (cycle.accepted == 0) {
      ++misses;
    }
  }
  require(misses > 0, "the run had no fallback cycle");
  for (size_t at = firstHit; at <= lastHit; ++at)
    if (!run0.cycles[at].usedCopyBlock)
      require(run0.cycles[at].accepted == 0,
              "a fallback cycle drafted the scripted continuation");
  require(firstHit < lastHit, "copying did not resume after the misses");
  requireAgreeingCycles(run0, "lookup misses");
}

// The copy source is the whole buffer: a query whose only match lies in
// generated output copies from it, which needs no repeat in the prompt at
// all. The prompt here holds nothing that repeats.
void generatedTextIsCopySource() {
  std::vector<uint32_t> prompt;
  for (uint32_t token = 0; token < 11; ++token)
    prompt.push_back(900 + token);
  std::vector<uint32_t> script;
  for (uint32_t repeat = 0; repeat < 3; ++repeat) {
    for (uint32_t token = 0; token < 12; ++token)
      script.push_back(21 + token);
    for (uint32_t token = 0; token < 5; ++token)
      script.push_back(41 + token);
  }
  const Request request{prompt, script, 51};
  const Run copied = run(request, CopySpecMode::Conservative);
  require(copied.transcript == script,
          "the generated-text run left the scripted target");
  require(copied.copyspecHits == 4,
          "the generated-text run used " + std::to_string(copied.copyspecHits) +
              " copy cycles, not 4");
  require(!copied.disabled, "the generated-text run was one-striked");
  const Run baseline = run(request, CopySpecMode::Off);
  require(baseline.transcript == copied.transcript,
          "copyspec changed the generated-text output");
  requireAgreeingCycles(copied, "generated text");
}

} // namespace

int main() {
  try {
    repeatedNgramOutputsIdentical();
    oneStrikeOnVerifiedZeroAcceptance();
    lookupMissesDoNotDisable();
    generatedTextIsCopySource();
  } catch (const std::exception &error) {
    std::cerr << "copyspec_engine_test: FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "copyspec_engine_test: PASS\n";
  return 0;
}

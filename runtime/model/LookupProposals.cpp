#include "model/LookupProposals.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>

namespace splash::model {
namespace {

// The transition table's first size in slots, 4 KB a request, which holds
// the transitions of a prompt of a few hundred tokens before it first
// doubles; a 100K-token prompt doubles it 9 times, which took 2 of the 6 ms
// its append took.
constexpr uint32_t kInitialSlots = 1024;

uint64_t slotHash(uint32_t from, uint32_t token) noexcept {
  uint64_t key = (uint64_t{from} << 32) | token;
  key = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9ULL;
  key = (key ^ (key >> 27)) * 0x94d049bb133111ebULL;
  return key ^ (key >> 31);
}

} // namespace

LookupIndex::LookupIndex() : states_(1), slots_(kInitialSlots, 0) {}

void LookupIndex::append(std::span<const uint32_t> tokens) {
  for (const uint32_t token : tokens)
    extend(token);
}

std::span<const uint32_t> LookupIndex::proposals(uint32_t count) const noexcept {
  if (!match_.length)
    return {};
  const size_t begin = size_t{match_.end} + 1;
  return std::span(tokens_).subspan(begin, std::min<size_t>(count, tokens_.size() - begin));
}

uint32_t LookupIndex::find(uint32_t state, uint32_t token) const noexcept {
  const uint64_t mask = slots_.size() - 1;
  for (uint64_t slot = slotHash(state, token) & mask;; slot = (slot + 1) & mask) {
    const uint32_t entry = slots_[slot];
    if (!entry)
      return kNone;
    const Edge &edge = edges_[entry - 1];
    if (edge.from == state && edge.token == token)
      return entry - 1;
  }
}

void LookupIndex::place(uint32_t edge) noexcept {
  const uint64_t mask = slots_.size() - 1;
  uint64_t slot = slotHash(edges_[edge].from, edges_[edge].token) & mask;
  while (slots_[slot])
    slot = (slot + 1) & mask;
  slots_[slot] = edge + 1;
}

void LookupIndex::connect(uint32_t from, uint32_t token, uint32_t to) {
  const auto edge = static_cast<uint32_t>(edges_.size());
  edges_.push_back({from, token, to, states_[from].edges});
  states_[from].edges = edge;
  if (2 * edges_.size() > slots_.size()) {
    slots_.assign(2 * slots_.size(), 0);
    for (uint32_t index = 0; index < edges_.size(); ++index)
      place(index);
  } else {
    place(edge);
  }
}

// The suffix automaton's online construction, then the match of the new
// token and the end of every class it ends that a match reads.
void LookupIndex::extend(uint32_t token) {
  const auto position = static_cast<uint32_t>(tokens_.size());
  tokens_.push_back(token);
  const auto current = static_cast<uint32_t>(states_.size());
  states_.push_back({states_[last_].length + 1, 0, kNone, position});
  uint32_t state = last_;
  for (; state != kNone && find(state, token) == kNone; state = states_[state].link)
    connect(state, token, current);
  if (state != kNone) {
    const uint32_t next = edges_[find(state, token)].to;
    if (states_[state].length + 1 == states_[next].length) {
      states_[current].link = next;
    } else {
      // The strings of `next` up to state's length plus one also end here
      // now: they move to a class of their own, which ends where `next` did.
      const auto clone = static_cast<uint32_t>(states_.size());
      states_.push_back({states_[state].length + 1, states_[next].link, kNone, states_[next].recent});
      for (uint32_t edge = states_[next].edges; edge != kNone; edge = edges_[edge].next)
        connect(clone, edges_[edge].token, edges_[edge].to);
      for (; state != kNone; state = states_[state].link) {
        Edge &edge = edges_[find(state, token)];
        if (edge.to != next)
          break;
        edge.to = clone;
      }
      states_[next].link = clone;
      states_[current].link = clone;
    }
  }
  last_ = current;

  // The class of the last min(size, kLongestMatch) tokens: past the first
  // kLongestMatch, the previous one's last kLongestMatch - 1 tokens (in it or
  // its link, once the clone above took its shorter strings) and this token.
  if (position < kLongestMatch) {
    capped_ = current;
  } else {
    uint32_t held = capped_;
    while (states_[states_[held].link].length >= kLongestMatch)
      held = states_[held].link;
    if (states_[states_[held].link].length == kLongestMatch - 1)
      held = states_[held].link;
    capped_ = edges_[find(held, token)].to;
  }
  // The longest suffix that also ended before is the longest string of
  // current's link; one longer than kLongestMatch counts as its last
  // kLongestMatch tokens, capped_'s. Neither class's recent has this token
  // yet: it is where their strings ended before.
  const uint32_t repeated = states_[current].link;
  const uint32_t length = states_[repeated].length;
  match_ = length < kMinimumMatch
               ? Match{}
               : Match{std::min(length, kLongestMatch), states_[length > kLongestMatch ? capped_ : repeated].recent};
  // Every class that holds a suffix of kMinimumMatch to kLongestMatch tokens
  // now ends here.
  for (state = capped_; states_[state].length >= kMinimumMatch; state = states_[state].link)
    states_[state].recent = position;
}

uint32_t LookupAcceptance::lengthBucket(uint32_t length) {
  if (length < LookupIndex::kMinimumMatch)
    throw std::invalid_argument("invalid lookup match length");
  return std::min(static_cast<uint32_t>(std::bit_width(length)) - 2, kLengthBuckets - 1);
}

uint32_t LookupAcceptance::depthBucket(uint32_t depth) noexcept {
  return std::min(static_cast<uint32_t>(std::bit_width(depth)), kDepthBuckets - 1);
}

float LookupAcceptance::rate(bool samples, uint32_t length, uint32_t depth, LookupAgreement agreement) const {
  const uint32_t l = lengthBucket(length), d = depthBucket(depth);
  const auto a = static_cast<uint32_t>(agreement);
  const Count &count = counts_[samples][l][d][a];
  return static_cast<float>((count.accepted + kPriorLabels * kPrior[l][d][a]) / (count.labels + kPriorLabels));
}

void LookupAcceptance::observe(bool samples, uint32_t length, std::span<const LookupAgreement> verified,
                               uint32_t accepted) {
  const uint32_t l = lengthBucket(length);
  if (accepted > verified.size())
    throw std::invalid_argument("invalid lookup outcome");
  const size_t labels = std::min<size_t>(accepted + 1, verified.size());
  const double decay = std::exp2(-static_cast<double>(labels) / kHalfLife);
  Counts &counts = counts_[samples];
  for (auto &depths : counts)
    for (auto &agreements : depths)
      for (Count &count : agreements) {
        count.accepted *= decay;
        count.labels *= decay;
      }
  for (uint32_t depth = 0; depth < labels; ++depth) {
    auto &cells = counts[l][depthBucket(depth)];
    const double label = depth < accepted ? 1.0 : 0.0;
    const auto add = [&](LookupAgreement agreement) {
      Count &count = cells[static_cast<uint32_t>(agreement)];
      count.accepted += label;
      count.labels += 1.0;
    };
    add(verified[depth]);
    if (verified[depth] != LookupAgreement::Unknown)
      add(LookupAgreement::Unknown);
  }
}

LookupAgreement lookupAgreement(uint32_t token, uint32_t depth, std::span<const uint32_t> candidates) {
  constexpr uint32_t kCandidates = SPLASH_DRAFT_CANDIDATES;
  if ((uint64_t{depth} + 1) * kCandidates > candidates.size())
    return LookupAgreement::Unknown;
  const std::span<const uint32_t> at = candidates.subspan(uint64_t{depth} * kCandidates, kCandidates);
  if (at.front() == token)
    return LookupAgreement::Top;
  return std::ranges::find(at, token) != at.end() ? LookupAgreement::Candidate : LookupAgreement::Absent;
}

LookupOutcome lookupOutcome(std::span<const uint32_t> chain, std::span<const uint32_t> treeTokens,
                            std::span<const ops::TreeRow> tree, uint32_t verifyRows, std::span<const uint32_t> retained,
                            std::span<const uint32_t> candidates) {
  if (verifyRows > treeTokens.size() || verifyRows > tree.size())
    throw std::invalid_argument("invalid lookup outcome tree");
  LookupOutcome outcome;
  uint32_t row = 0;
  for (uint32_t depth = 0; depth < chain.size() && depth < retained.size(); ++depth) {
    uint32_t child = 0;
    for (uint32_t at = row + 1; at < verifyRows && !child; ++at)
      child = tree[at].parent == row && treeTokens[at] == chain[depth] ? at : 0;
    if (!child)
      break;
    outcome.verified.push_back(lookupAgreement(chain[depth], depth, candidates));
    if (retained[depth] != chain[depth])
      break;
    ++outcome.accepted;
    row = child;
  }
  return outcome;
}

} // namespace splash::model

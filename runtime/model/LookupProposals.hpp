#pragma once

// Proposals a request's own tokens make (prompt lookup): where its latest
// tokens occurred before, the tokens that followed them there are a guess at
// what follows now, which a decode cycle can verify beside the draft's.
// LookupIndex finds that earlier occurrence; LookupAcceptance estimates how
// likely the target accepts each token copied from it.
#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace splash::model {

// A request's tokens, its prompt and then every token its decode selects (an
// anchor included), and for the latest of them the match: the earlier
// occurrence of the longest suffix of the tokens, its length counted up to
// kLongestMatch, the most recent among those of that length, and none shorter
// than kMinimumMatch. A suffix automaton over the tokens finds it as each
// token is appended, in amortized constant time and a walk of at most
// kLongestMatch states, so reading the match costs nothing. It keeps two
// states and three transitions a token at most: a traced 100K-token agent
// prompt took 5-6 ms to append on an M5 Max, and 89 bytes a token.
class LookupIndex final {
public:
  // In decode traces, the target chose a 2-token match's first token 34%
  // (27B) and 39% (35B) of the time, a 3-token match's 55% and 61%; with the
  // draft's own proposals beside it, minimums of 2, 3 and 4 differed by under
  // 1% in the tokens a cycle gained.
  static constexpr uint32_t kMinimumMatch = 3;
  // Counting up to 64 tokens chose the same occurrence as counting up to
  // 4096 at all 17,419 traced decode cycles; up to 32 chose another at 25 of
  // them and lost 0.2% of the lookup tokens the target accepted. It bounds
  // the walk each append makes, which a run of one repeated token would
  // otherwise make one state longer each token: 100K of one token append in
  // under 10 ms.
  static constexpr uint32_t kLongestMatch = 64;

  struct Match final {
    // From kMinimumMatch to kLongestMatch tokens; 0 when the latest tokens
    // occurred nowhere before.
    uint32_t length = 0;
    // The index of the earlier occurrence's last token.
    uint32_t end = 0;

    bool operator==(const Match &) const = default;
  };

  LookupIndex();

  void append(std::span<const uint32_t> tokens);
  [[nodiscard]] std::span<const uint32_t> tokens() const noexcept { return tokens_; }
  [[nodiscard]] Match match() const noexcept { return match_; }
  // The lookup proposals after the latest token: up to `count` tokens that
  // followed the match's occurrence, at most as far as the latest token.
  // Empty without a match.
  [[nodiscard]] std::span<const uint32_t> proposals(uint32_t count) const noexcept;

private:
  static constexpr uint32_t kNone = UINT32_MAX;

  // A class of substrings that end at the same positions: those of lengths
  // (link's length, length], each a suffix of the longest.
  struct State final {
    uint32_t length = 0;
    // The class of the longest suffix of its strings that ends elsewhere
    // too; kNone for the empty string's.
    uint32_t link = kNone;
    // Its first transition (Edge::next chains the rest).
    uint32_t edges = kNone;
    // Where its strings last ended. Exact while it holds a string of at most
    // kLongestMatch tokens, the only states a match reads it from.
    uint32_t recent = 0;
  };
  struct Edge final {
    uint32_t from = 0;
    uint32_t token = 0;
    uint32_t to = 0;
    uint32_t next = kNone;
  };

  void extend(uint32_t token);
  // The transition from `state` on `token`; kNone without one.
  [[nodiscard]] uint32_t find(uint32_t state, uint32_t token) const noexcept;
  void connect(uint32_t from, uint32_t token, uint32_t to);
  void place(uint32_t edge) noexcept;

  std::vector<uint32_t> tokens_;
  std::vector<State> states_;
  std::vector<Edge> edges_;
  // An open-addressed table of the transitions by (from, token): an edge's
  // index plus one, 0 where empty; at most half full.
  std::vector<uint32_t> slots_;
  // The class of all the tokens, and that of their last min(size,
  // kLongestMatch) tokens.
  uint32_t last_ = 0;
  uint32_t capped_ = 0;
  Match match_;
};

// How a lookup token compares with the draft's candidates at its depth (the
// selector's most likely tokens at that proposal position).
enum class LookupAgreement : uint8_t {
  // The draft's most likely token there.
  Top,
  // Another of its candidates.
  Candidate,
  // None of them.
  Absent,
  // Not known: the draft proposes no token at that depth, or has not run.
  Unknown,
};

// The probability that the target accepts a lookup token given that it
// accepted the token's parent, from the match's length, the token's depth
// (0 for the first proposal after the anchor) and its agreement with the
// draft: a table fitted on decode traces, and on top of it decayed counts of
// what the engine's own cycles accepted. One serves every request: replaying
// the traced sessions, counts over a session predicted better than each
// request's own (0.074 against 0.079 bits a label on the 27B).
class LookupAcceptance final {
public:
  // Lengths 3, 4-7, 8-15, 16-31, 32-63 and 64 tokens; depths 0, 1, 2-3, 4-7,
  // 8-15 and 16 or more.
  static constexpr uint32_t kLengthBuckets = 6;
  static constexpr uint32_t kDepthBuckets = 6;
  static constexpr uint32_t kAgreements = 4;
  using Table = std::array<std::array<std::array<float, kAgreements>, kDepthBuckets>, kLengthBuckets>;

  // [length][depth][agreement], fitted on 161,360 labels: each token of a
  // lookup chain of up to 31 tokens, to its first rejection, at the 8,670 of
  // 17,419 decode cycles traced at temperature 0 whose anchor had a match, on
  // Qwen3.8-27B (67,268 labels) and Qwen3.6-35B-A3B (94,092) with their
  // DFlash2 drafts, over file rewrites, short and 8K-token benchmark
  // prompts, agent and edit sessions of the serving benchmark and three
  // coding agents. A logistic regression on the one-hot length, depth and
  // agreement and the length-agreement and depth-agreement products (L2
  // penalty 10) gives every cell; Unknown's fit every label. Cross-validated
  // by workload (each predicted by a fit on the others), in bits a label on
  // the 27B / 35B: one rate per agreement 0.214 / 0.222; length and depth
  // alone 0.124 / 0.133; this table 0.090 / 0.118, or 0.091 / 0.146 fitted
  // on the other model alone; finer length or depth buckets moved it by
  // 0.0003 at most. Cells of a known agreement past depth 6 hold no label,
  // as the draft proposes 7 tokens: the fit extends its trends there.
  static constexpr Table kPrior{{
      {{{0.966F, 0.214F, 0.125F, 0.648F},
        {0.973F, 0.307F, 0.232F, 0.709F},
        {0.980F, 0.516F, 0.559F, 0.787F},
        {0.979F, 0.652F, 0.732F, 0.830F},
        {0.973F, 0.569F, 0.551F, 0.856F},
        {0.979F, 0.634F, 0.617F, 0.911F}}},
      {{{0.975F, 0.249F, 0.153F, 0.749F},
        {0.980F, 0.349F, 0.276F, 0.798F},
        {0.985F, 0.564F, 0.614F, 0.857F},
        {0.984F, 0.694F, 0.775F, 0.888F},
        {0.980F, 0.616F, 0.607F, 0.906F},
        {0.984F, 0.678F, 0.670F, 0.943F}}},
      {{{0.981F, 0.307F, 0.163F, 0.812F},
        {0.985F, 0.419F, 0.292F, 0.851F},
        {0.989F, 0.634F, 0.633F, 0.897F},
        {0.988F, 0.753F, 0.789F, 0.920F},
        {0.985F, 0.682F, 0.626F, 0.933F},
        {0.989F, 0.738F, 0.688F, 0.960F}}},
      {{{0.987F, 0.327F, 0.220F, 0.867F},
        {0.990F, 0.441F, 0.374F, 0.896F},
        {0.992F, 0.655F, 0.714F, 0.929F},
        {0.992F, 0.770F, 0.844F, 0.946F},
        {0.990F, 0.702F, 0.708F, 0.955F},
        {0.992F, 0.756F, 0.761F, 0.973F}}},
      {{{0.995F, 0.648F, 0.439F, 0.935F},
        {0.996F, 0.750F, 0.624F, 0.950F},
        {0.997F, 0.878F, 0.874F, 0.967F},
        {0.997F, 0.927F, 0.937F, 0.975F},
        {0.996F, 0.899F, 0.871F, 0.979F},
        {0.997F, 0.921F, 0.898F, 0.988F}}},
      {{{0.999F, 0.955F, 0.859F, 0.993F},
        {1.000F, 0.972F, 0.928F, 0.995F},
        {1.000F, 0.988F, 0.982F, 0.997F},
        {1.000F, 0.993F, 0.992F, 0.997F},
        {1.000F, 0.990F, 0.981F, 0.998F},
        {1.000F, 0.993F, 0.986F, 0.999F}}},
  }};
  // The table weighs as kPriorLabels labels of its cell, and a label weighs
  // half as much kHalfLife labels later. Replaying each traced workload in
  // order with the table fitted without it, they cut the loss from 0.090 /
  // 0.118 to 0.074 / 0.101 bits, and with the other model's table (a model
  // pair it never saw) from 0.091 / 0.146 to 0.075 / 0.106; the table
  // weighing 16 or 64 labels did worse (0.076 and 0.080 on the 27B). Running
  // every workload of a model in one session, with the other model's table,
  // the half-life beat none (0.071 against 0.076 bits on the 27B) and 1024
  // labels (0.073).
  static constexpr double kPriorLabels = 4.0;
  static constexpr double kHalfLife = 4096.0;

  [[nodiscard]] static uint32_t lengthBucket(uint32_t length);
  [[nodiscard]] static uint32_t depthBucket(uint32_t depth) noexcept;

  // The estimate for a token `depth` past the anchor, copied from a match of
  // `length` tokens.
  [[nodiscard]] float rate(uint32_t length, uint32_t depth, LookupAgreement agreement) const;
  // A cycle's outcome for one lookup chain: the agreements of the tokens it
  // verified, in depth order, and how many of them the target accepted
  // before its first rejection. Each accepted token and the rejected one is
  // a label (those past the rejection are none); a label of a known
  // agreement teaches Unknown too.
  void observe(uint32_t length, std::span<const LookupAgreement> verified, uint32_t accepted);

private:
  struct Count final {
    double accepted = 0.0;
    double labels = 0.0;
  };
  std::array<std::array<std::array<Count, kAgreements>, kDepthBuckets>, kLengthBuckets> counts_{};
};

} // namespace splash::model

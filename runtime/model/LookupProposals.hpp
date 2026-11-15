#pragma once

// Proposals a request's own tokens make (prompt lookup): where its latest
// tokens occurred before, the tokens that followed them there are a guess at
// what follows now, which a decode cycle can verify beside the draft's.
// LookupIndex finds that earlier occurrence.
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

} // namespace splash::model

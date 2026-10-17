#pragma once

// A request's draft context beside the MTP head's draft vocabulary
// (Qwen4DraftVocabulary.hpp): the tokens outside the vocabulary its own
// context holds, whose rows of the vocabulary projection its state lane
// keeps, and the count of its emitted tokens the head could not have drafted,
// by which the request falls back to the whole head. The context and the
// count survive a suspension with the request; the rows a lane holds do not
// (Qwen4DraftContextCache).

#include "model/Qwen4DraftVocabulary.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace splash::model {

struct Qwen4DraftContext final {
  // The context tokens, in the order they first appear, a bitmap of them,
  // and how far the request's tokens were scanned.
  std::vector<uint32_t> tokens;
  std::vector<uint64_t> bits;
  size_t scanned = 0;
  uint32_t draftTokens = kQwen4DraftVocabularyTokens;
  // Decayed counts of the tokens the request's decode emitted and of those
  // outside the head's draft vocabulary and context, which it never
  // proposes: about the last 50 tokens (kDecay per token).
  float misses = 0.0F;
  float observations = 0.0F;
  // A prior of kPrior tokens observed inside the vocabulary before any
  // real observation (a warm start), which decays as they do: one early
  // token outside does not decide a request's head for its first cycles.
  float coveredPrior = kPrior;
  static constexpr float kDecay = 0.98F;
  static constexpr float kPrior = 50.0F;

  // Whether the head drafts `token` for this request.
  [[nodiscard]] bool contains(uint32_t token) const noexcept {
    return qwen4DraftContains(token, draftTokens) ||
           (token / 64 < bits.size() && ((bits[token / 64] >> (token % 64)) & 1));
  }
  // Adds the context tokens of `history` since the last scan, up to
  // `capacity`, of the `selected`-token draft vocabulary over `vocabulary`
  // tokens.
  void scan(std::span<const uint32_t> history, uint32_t vocabulary, uint32_t capacity = kQwen4DraftContextTokens,
            uint32_t selected = kQwen4DraftVocabularyTokens) {
    draftTokens = selected;
    if (bits.empty()) bits.assign((vocabulary + 63) / 64, 0);
    for (; scanned < history.size(); ++scanned) {
      if (tokens.size() >= capacity) {
        scanned = history.size();
        break;
      }
      const uint32_t token = history[scanned];
      if (token >= vocabulary || contains(token)) continue;
      bits[token / 64] |= uint64_t{1} << (token % 64);
      tokens.push_back(token);
    }
  }
  // Counts the tokens a cycle selected.
  void record(std::span<const uint32_t> selected) noexcept {
    for (const uint32_t token : selected) {
      misses = kDecay * misses + (contains(token) ? 0.0F : 1.0F);
      observations = kDecay * observations + 1.0F;
      coveredPrior *= kDecay;
    }
  }
  // Whether the request drafts from the vocabulary and its context: while
  // at most `threshold` of its recent tokens (with the prior's when
  // `warmStart`) fell outside them.
  [[nodiscard]] bool useVocabulary(bool warmStart, float threshold) const noexcept {
    return misses <= threshold * (observations + (warmStart ? coveredPrior : 0.0F));
  }
};

// Which request's context rows each state lane holds, and how many of them
// were copied in. A lane's rows are valid only during one activation: the
// runtime releases them on completion, cancellation and suspension, even if
// the next request has the same id. bind() returns the first row that still
// needs a copy.
template <size_t Lanes> class Qwen4DraftContextCache final {
public:
  void release(uint32_t lane) noexcept {
    owners_[lane].reset();
    synced_[lane] = 0;
  }
  [[nodiscard]] uint32_t bind(uint32_t lane, uint64_t request, uint32_t count) {
    if (owners_[lane] != request) {
      owners_[lane] = request;
      synced_[lane] = 0;
    }
    const uint32_t synced = synced_[lane];
    if (synced > count) throw std::logic_error("draft context shrank within a lane activation");
    synced_[lane] = count;
    return synced;
  }

private:
  std::array<std::optional<uint64_t>, Lanes> owners_{};
  std::array<uint32_t, Lanes> synced_{};
};

} // namespace splash::model

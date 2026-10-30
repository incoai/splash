#pragma once

// Prompt-lookup ("copyspec") drafting: a request drafts by copying its own
// earlier tokens wherever they repeat what it is about to say, instead of
// running the DFlash draft. The lookup finds where the request's history
// first held the window its pending anchor ends, and copies what followed
// there; the verify is the target's unchanged, so a wrong copy costs only
// the verify rows its block claims, which follow the request's measured
// acceptance of its copies (CopySpecAcceptance). Repetition-heavy traffic
// (code, agent loops) gains the acceptance of near-oracle drafts; fresh text
// misses and falls back to DFlash for free.
//
// The index (CopySpecIndex) hashes every window of W = 6 token ids with
// 64-bit FNV-1a into a map from hash to the ends of the windows that hash
// there (an id until a hash first collides, then a list), and confirms a
// candidate with an exact token-by-token compare: the hash is only a bucket
// key. The lookup drafts FULL-LENGTH-OR-NOTHING: a partial draft is no
// draft, so a hit always proposes exactly the block the cycle verifies.
// A draft holding any of the request's forbidden (stop) tokens is dropped
// whole as well.
//
// Whether a request copies is per request and per mode (CopySpecMode): off
// never consults its index, and conservative stops copying for the rest of
// the request once a copied block verifies nothing (one strike; lookup
// misses are free and never strike).

#include <cstdint>
#include <span>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace splash::model {

enum class CopySpecMode : uint8_t { Off, Conservative };
CopySpecMode copySpecMode(std::string_view);

struct CopySpecIndex {
  CopySpecIndex(std::span<const uint32_t> promptTokens,
                uint32_t windowSize = 6);
  bool enabled() const;
  void appendCommitted(std::span<const uint32_t> ids);
  // Returns copied-token count written to out (0 = miss). maxTokens <= 0 -> 0.
  uint32_t draftAfter(uint32_t stagedFirst, uint32_t maxTokens,
                      std::span<const uint32_t> forbidden,
                      std::span<uint32_t> out) const;
  // Replaces the buffer with a request's recomputed history (Runtime::resume
  // reassigns its tokens wholesale), indexing its windows anew; `enabled`
  // stays what construction computed and is never re-evaluated.
  void rebuild(std::span<const uint32_t> history);

private:
  // The buffer: the prompt and every committed output token, which grows
  // only. A hash maps to one window's end, or to the ends that hash there in
  // chronological order once a hash first collides.
  using Ends = std::variant<uint32_t, std::vector<uint32_t>>;
  std::vector<uint32_t> tokens_;
  std::unordered_map<uint64_t, Ends> index_;
  uint32_t window_ = 6;
  bool enabled_ = false;

  void indexWindow(uint32_t start);
};

// A request's measured acceptance of its copied proposals: the chance that a
// copied proposal the verify reaches (every proposal before it accepted) is
// accepted too, the per-position rate the verify rows' acceptance model
// speaks in (ops::VerifyRowsModel). It starts from a Beta(3, 1) prior, so a
// request's first copy claims a block about as long as a confident draft's,
// and each verified copy moves it toward what the request's copies earn.
struct CopySpecAcceptance {
  static constexpr double kPriorAccepted = 3.0;
  static constexpr double kPriorRejected = 1.0;
  // A copied block's verification: `accepted` of its first `verified`
  // proposals matched the target. Each accepted proposal is a reached
  // acceptance, and the first refused one, if the verify reached it, a
  // reached rejection.
  void observe(uint32_t verified, uint32_t accepted);
  double rate() const;

private:
  uint64_t reached_ = 0;
  uint64_t accepted_ = 0;
};

struct CopySpecState {
  CopySpecIndex index;
  bool disabled = false;
  CopySpecAcceptance acceptance;
};

} // namespace splash::model

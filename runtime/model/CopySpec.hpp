#pragma once

// Prompt-lookup ("copyspec") drafting: a request drafts by copying its own
// earlier tokens wherever they repeat what it is about to say, instead of
// running the DFlash draft. The lookup finds where the request's history
// last held the window its pending anchor ends, and copies what followed
// there; the verify is the target's unchanged, so a wrong copy costs only
// the cycle's draft work. Repetition-heavy traffic (code, agent loops)
// gains the acceptance of near-oracle drafts; fresh text misses and falls
// back to DFlash for free.
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
// never consults its index, conservative stops copying for the rest of the
// request once a copied block verifies nothing (one strike; lookup misses
// are free and never strike), and auto leaves it to a windowed A/B latch
// (CopySpecAutoGate) that measures the throughput copying gives.

#include <cstdint>
#include <span>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace splash::model {

enum class CopySpecMode : uint8_t { Off, Conservative, Auto };
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

struct CopySpecAutoGate {
  static constexpr uint32_t kProbeOffCycles = 24;
  static constexpr uint32_t kProbeOnMinCopy = 6;
  static constexpr uint32_t kProbeOnMaxCycles = 48;
  static constexpr uint32_t kLatchCycles = 200;
  static constexpr uint32_t kBackoffCap = 4;
  bool engageCopy() const;                    // phase is measure_on|engaged
  void recordCycle(bool sourceCopy, uint32_t commitCount, uint64_t costNs);
  bool takeReset();                           // one-shot pending-reset flag

private:
  // The four phases: measure_off gathers a baseline with copying suppressed,
  // measure_on probes with it allowed, engaged keeps it on for
  // kLatchCycles cycles once the probe saw it win, and dormant keeps it off
  // for a backoff while it loses.
  enum class Phase : uint8_t { MeasureOff, MeasureOn, Engaged, Dormant };
  Phase phase_ = Phase::MeasureOff;
  // The current window's realized throughput.
  uint64_t windowCommits_ = 0;
  uint64_t windowCostNs_ = 0;
  uint32_t windowCycles_ = 0;
  uint32_t windowCopies_ = 0;
  double offRate_ = 0.0;
  bool hasOffRate_ = false;
  uint32_t latchRemaining_ = 0;
  // Consecutive disengagements lengthen the dormant backoff; any engage
  // resets them, so a workload that turns copy-heavy stays responsive.
  uint32_t consecutiveLosses_ = 0;
  bool pendingReset_ = false;
};

struct CopySpecState {
  CopySpecIndex index;
  bool disabled = false;
  CopySpecAutoGate gate;
};

} // namespace splash::model

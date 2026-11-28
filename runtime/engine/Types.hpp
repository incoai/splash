#pragma once

#include "model/Model.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace splash::engine {

// Values are the native request frame's priority byte.
enum class RequestPriority : uint8_t {
  Foreground = 0,
  Normal = 1,
  Background = 2,
};

// A score request's full-vocabulary readout at its final position, beside
// its option logits: log-sum-exp over the vocabulary and the highest logits.
struct ScoreReadout final {
  double logNormalizer = 0.0;
  std::vector<uint32_t> topIds;
  std::vector<float> topLogits;
};

struct StepResult final {
  uint64_t requestId = 0;
  uint32_t consumedPromptTokens = 0;
  bool finished = false;
  DecodeStage nextDecodeStage = DecodeStage::Regular;
  // The output tokens the command committed for the lane
  // (ModelStepResult::outputTokens).
  uint32_t committedTokens = 0;
};

// Where a request's committed KV lives, chosen at admission and kept across
// suspension: in memory (Resident), or, on a target that can spill, in the
// SSD cache tier for the pages a claim released once their copies landed, or
// under --kv-spill always as they land, memory keeping each page's index
// (Spill).
enum class KvMode : uint8_t { Resident, Spill };

// Values are the Done event's reason byte.
enum class EngineFinishReason : uint8_t {
  Stop = 0,
  Length = 1,
  Cancelled = 2,
};

// A request's speculative decoding: the tokens its draft proposed, those
// verification accepted, and the decode steps that verified a draft.
struct SpeculativeCounts final {
  uint32_t draftedTokens = 0;
  uint32_t acceptedDraftTokens = 0;
  uint32_t cycles = 0;

  bool operator==(const SpeculativeCounts &) const = default;
};

// A request's cache salt: its API's cache_salt, which the server hashes to
// 128 bits. Requests share cached KV and states only with requests of the
// same salt; zero for a request without one.
struct CacheSalt final {
  uint64_t lo = 0;
  uint64_t hi = 0;

  bool operator==(const CacheSalt &) const = default;
};

struct EngineRequest final {
  uint64_t id = 0;
  RequestPriority priority = RequestPriority::Normal;
  std::vector<uint32_t> prompt;
  // Trailing prompt tokens a later request may not share (a chat template's
  // generation prompt, which the next turn may render differently), so
  // reusable state is kept before them. Zero when unknown; it must leave at
  // least one prompt token.
  uint32_t generationPromptTokens = 0;
  // Where the prompt's head ends, its system prompt and tools before any other message, which other conversations may
  // share: a state is kept there on first use (Engine::configureDraftStatePlan). Zero when there is none.
  uint32_t headTokens = 0;
  std::vector<ImageSpan> images;
  std::vector<uint8_t> imagePixels;
  uint32_t maxNewTokens = 0;
  SamplingParameters sampling;
  ConstraintMode constraint = ConstraintMode::None;
  double deadlineMilliseconds = 0.0;
  bool returnProgress = false;
  // Nonempty selects score-only mode: prefill runs to completion, no token is
  // generated, and the raw final-position logits at these ids are returned in
  // the completion callback. maxNewTokens must be zero.
  std::vector<uint32_t> scoreTokens{};
  // RequestFlag bits.
  uint32_t flags = 0;
  // Set only for a request that generates inside a thinking block.
  std::optional<ThinkingBudget> thinkingBudget;
  CacheSalt cacheSalt;

  [[nodiscard]] ModelRequest modelView() const noexcept {
    return {id,           prompt,   images,     imagePixels,
            maxNewTokens, sampling, constraint, scoreTokens,
            flags,        thinkingBudget ? &*thinkingBudget : nullptr};
  }
};

// How a request ends other than by its own output: decided by the engine,
// reported by the protocol adapter from this one table. Cancelled and Stopped
// finish the request and are never sent as an error; the rest fail it.
enum class LaneOutcome : uint8_t {
  // Completes with EngineFinishReason::Cancelled: the client abandoned it.
  Cancelled,
  // Completes with EngineFinishReason::Stop: the client matched a stop
  // condition in its output, so it counts as completed.
  Stopped,
  DeadlineExceeded,
  ResourceTimeout,
  // A lone lane that cannot fit even after every cached prefix went:
  // retrying the same request fails the same way.
  CapacityExhausted,
  ModelResultInvalid,
  InvalidMask,
  // The server left a token-mask request unanswered (Engine.cpp's limit).
  MaskTimeout,
  // The SSD cache could not read back KV of an earlier part of the request
  // (ModelStepResult::unreadableKvBlocks); its copies are given up, so a
  // retry recomputes it.
  KvUnreadable,
  // Its KV does not fit in memory and would go to the SSD cache, whose
  // daily write budget (--max-cache-disk-writes) is spent until it refills.
  KvWriteBudget,
  // Its KV does not fit in memory, and the SSD cache's quota of active KV
  // (--max-cache-disk, and the volume's free space) cannot hold even its
  // prompt's, without any other request's: retrying the same request fails
  // the same way.
  KvDiskQuota,
};

struct LaneOutcomeWire final {
  std::string_view code;
  bool retryable;
};

[[nodiscard]] constexpr LaneOutcomeWire
laneOutcomeWire(LaneOutcome outcome) noexcept {
  switch (outcome) {
  case LaneOutcome::Cancelled:
    return {"cancelled", false};
  case LaneOutcome::Stopped:
    return {"stopped", false};
  case LaneOutcome::DeadlineExceeded:
    return {"deadline_exceeded", false};
  case LaneOutcome::ResourceTimeout:
    return {"resource_timeout", true};
  case LaneOutcome::CapacityExhausted:
    return {"capacity_exhausted", false};
  case LaneOutcome::ModelResultInvalid:
    return {"model_result_invalid", false};
  case LaneOutcome::InvalidMask:
    return {"invalid_mask_response", false};
  case LaneOutcome::MaskTimeout:
    return {"mask_timeout", true};
  case LaneOutcome::KvUnreadable:
    return {"kv_unreadable", true};
  case LaneOutcome::KvWriteBudget:
    return {"kv_write_budget", true};
  case LaneOutcome::KvDiskQuota:
    return {"kv_disk_quota", false};
  }
}

// What a request that cannot fit even alone (CapacityExhausted, KvDiskQuota)
// lacked: memory, memory beside its images, an SSD cache that takes no
// writes, or the SSD cache's quota of active KV. The values are on the wire
// (Error), by which the server answers it as a prompt over the context.
enum class CapacityLimit : uint8_t { None = 0, Memory = 1, ImageMemory = 2, ClosedDisk = 3, DiskQuota = 4 };

// A capacity failure's limit, and the most prompt tokens that limit holds
// this request alone now, with its first output token: 0 where the engine
// cannot say.
struct CapacityShortfall {
  CapacityLimit limit = CapacityLimit::None;
  uint32_t fitsTokens = 0;

  bool operator==(const CapacityShortfall &) const = default;
};

inline constexpr std::string_view kDeadlineExceededMessage =
    "request deadline exceeded";

// What held a request before its first token, in
// milliseconds, each interval counted once by precedence: a command in
// flight that carries its prompt's rows, its memory claim waiting, a decode
// command, another request's prefill command, the next prefill unit waiting
// for a decoder whose memory lands (Scheduler::waitsUntil), and the rest
// (the host, admission, idle). The engine measures the first five between
// its ticks; the rest is the time to the first token less them.
enum class TtftPart : uint8_t { OwnPrefill, ClaimWait, Decode, OtherPrefill, UnitWait, Rest };
inline constexpr size_t kTtftParts = 6;
using TtftParts = std::array<double, kTtftParts>;

// Where the state a request resumes from was: none for a cold start, a RAM
// copy, or the SSD cache's copy alone. The values are on the wire (Start).
enum class StateSource : uint8_t { None = 0, Memory = 1, Disk = 2 };

// Why a start's reuse stopped where it did, from its final lookup: no block
// of its prompt matched; the prompt parts from the cache at the matched KV's
// end, every matched token reused; KV matched past the state it resumed from,
// read again; or a read from the SSD failed on an earlier attempt, which then
// matched less. The values are on the wire (Start).
enum class ReuseEnd : uint8_t { NothingMatched = 0, PrefixEnd = 1, KvWithoutState = 2, DiskReadFailed = 3 };
inline constexpr size_t kReuseEnds = 4;

// What a request's start found in the cache: the tokens of the state it
// resumes from, zero for a cold start; the prompt's tokens whose KV the cache
// matched, the state's included; where that state was; whether a matched
// block past the state once held a state that is gone from both tiers; and
// why its reuse stopped, with that end's tokens: the matched KV's end at a
// prefix end, the KV past the state read again, the tokens the failed read
// would have restored, none where nothing matched.
struct StartFound {
  uint32_t matchedPromptTokens = 0;
  uint32_t prefixTokens = 0;
  StateSource stateSource = StateSource::None;
  bool lostState = false;
  ReuseEnd reuseEnd = ReuseEnd::NothingMatched;
  uint32_t reuseEndTokens = 0;
};

class EngineEventSink {
public:
  virtual ~EngineEventSink() = default;
  // Kind, width, input, output, drafted and accepted tokens, then the
  // command's wall time and the engine's cycle for it: from the previous
  // command's retirement, or from its plan when the engine was idle, to its
  // own retirement.
  virtual void batchCompleted(WorkKind, uint32_t, uint32_t, uint32_t,
                              uint32_t, uint32_t, double, double) = 0;
  virtual void started(uint64_t requestId, const StartFound &found,
                       uint32_t lane) = 0;
  virtual void promptProgress(uint64_t, uint32_t) {}
  // Before a request's first tokens: what held it until then, the rest not
  // counted (TtftPart::Rest is 0).
  virtual void firstTokenParts(uint64_t, const TtftParts &) {}
  virtual void tokens(uint64_t requestId, std::span<const uint32_t> tokens) = 0;
  virtual void maskRequested(const ModelMaskRequest &request) = 0;
  // bounded: a Length end short of the request's maxNewTokens, at a bound
  // of the engine's own (the SSD cache's quota, or the memory a lone lane
  // could not grow past), which ends the response as its context would.
  // deduplicatedTokens: tokens whose KV the request read from cached blocks
  // another request computed, past the cached prefix it started from: blocks
  // it computed and then found cached, and blocks a resume after a
  // suspension restored past those it had published.
  virtual void completed(uint64_t requestId, EngineFinishReason reason, bool bounded,
                         uint32_t promptTokens, uint32_t completionTokens,
                         uint32_t deduplicatedTokens,
                         std::span<const float> optionLogits,
                         const SpeculativeCounts &speculation,
                         const ScoreReadout &score) = 0;
  // retryAfterSeconds: for a retryable outcome, when a retry may succeed;
  // 0 for at once. capacity: for a request that cannot fit even alone, what
  // it lacked and what fits.
  virtual void failed(uint64_t requestId, LaneOutcome outcome,
                      std::string message, uint32_t retryAfterSeconds,
                      CapacityShortfall capacity) = 0;
};

} // namespace splash::engine

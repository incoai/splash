#pragma once

#include "engine/Types.hpp"
#include "model/Model.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

// The engine decodes client frames and encodes engine events;
// server/protocol.py does the reverse. dev/tests/engine/protocol_golden.txt
// pins the bytes both sides agree on.
namespace splash::protocol {

inline constexpr uint16_t kProtocolVersion = 13;
// A frame is its header, the magic "SPLH", the u16 protocol version, the
// u16 FrameType and the u64 payload length, then the payload.
inline constexpr size_t kFrameHeaderBytes = 16;
inline constexpr uint32_t kStatusSchemaVersion = 8;
// Image pixels travel inside the request frame; a multi-image agent turn can
// carry well over 64 MiB of resized RGB bytes.
inline constexpr uint64_t kAbsoluteMaxFramePayloadBytes =
    256ULL * 1024 * 1024;

// Every integer, including binary prompt/token words, is little-endian on the
// wire.
enum class FrameType : uint16_t {
  Request = 0x0001,
  Cancel = 0x0002,
  MaskResponse = 0x0003,
  StatusRequest = 0x0004,
  ReleaseRequest = 0x0005,
  Stop = 0x0006,

  Ready = 0x0100,
  Start = 0x0101,
  Tokens = 0x0102,
  MaskRequest = 0x0103,
  Done = 0x0104,
  Error = 0x0105,
  StatusJson = 0x0107,
  PromptProgress = 0x0108,
  StartFailed = 0x0109,
  Warming = 0x010A,
};

// Request errors reject one request while preserving the stream.  An
// engine-unhealthy error asks the supervisor to replace the engine.  A
// protocol-fatal error means framing is no longer trusted, or a client frame
// failed without naming its request, so the stream must close.
enum class FailureClass : uint8_t {
  RequestError = 1,
  EngineUnhealthy = 2,
  ProtocolFatal = 3,
};

[[nodiscard]] bool connectionMustClose(FailureClass failureClass);

enum class IssueCode : uint16_t {
  None = 0,
  BadMagic,
  UnsupportedVersion,
  UnknownFrameType,
  FrameTooLarge,
  InvalidPayloadLength,
  TruncatedFrame,
  InvalidRequestId,
  InvalidEnumValue,
  InvalidDeadline,
  InvalidSampling,
  InvalidCount,
  InvalidConstraint,
  InvalidErrorClassification,
  LimitExceeded,
  IntegerOverflow,
  AllocationFailure,
};

[[nodiscard]] std::string_view issueCodeName(IssueCode code);

struct ProtocolIssue {
  FailureClass failureClass = FailureClass::ProtocolFatal;
  IssueCode code = IssueCode::None;
  uint64_t requestId = 0;
  std::string message;

  [[nodiscard]] std::string describe() const;
  bool operator==(const ProtocolIssue &) const = default;
};

template <typename T> struct ProtocolResult {
  std::optional<T> value;
  std::optional<ProtocolIssue> issue;

  [[nodiscard]] explicit operator bool() const noexcept {
    return value.has_value() && !issue.has_value();
  }
};

// Valid as validateLimits checks them: NativeRuntime checks its limits once,
// and the codec and the frame parser rely on that.
struct ProtocolLimits {
  uint64_t maxFramePayloadBytes = kAbsoluteMaxFramePayloadBytes;
  uint64_t maxStatusJsonBytes = 32ULL * 1024 * 1024;
  uint64_t maxErrorStringBytes = 1ULL * 1024 * 1024;
  uint32_t maxPromptTokens = 1U << 20;
  uint32_t maxLogicalOutputTokens = 1U << 20;
  uint32_t maxTokenBatch = 4096;
  uint32_t maxSimulationTokens = 32;
  uint32_t maxMaskWords = 1U << 20;
  uint32_t maxImageSpans = 64;
};

[[nodiscard]] std::optional<ProtocolIssue>
validateLimits(const ProtocolLimits &limits);

// A request payload starts with these fields, at these byte offsets:
//    0 u64 requestId
//    8 u8  priority
//    9 u8  constraint
//   10 u64 absoluteDeadlineUnixMicros
//   18 u64 remainingDeadlineMicros
//   26 u32 logicalMaxOutputTokens
//   30 u32 prompt token count
//   34 u32 image span count
//   38 sampling: f32 temperature, f32 topP, u32 topK, f32 presencePenalty,
//      f32 frequencyPenalty, f32 repetitionPenalty, f32 minP, u64 seed
//   74 u32 score token count
//   78 u32 generationPromptTokens
//   82 u32 flags
//   86 u64 cacheSalt.lo, u64 cacheSalt.hi
//  102 u32 thinkingBudget.tokens, kNoThinkingBudget for none
//  106 u32 thinking end token count, u32 thinkingBudget.alsoEnds, zero
//      without a budget
// then the prompt tokens, the 32-byte image spans, the image pixels, the
// score tokens and the thinking end tokens.
inline constexpr uint64_t kRequestFixedBytes = 114;
inline constexpr uint32_t kNoThinkingBudget = 0xFFFFFFFFU;
// A thinking budget's end holds 1 to this many tokens.
inline constexpr uint32_t kMaximumThinkingEndTokens = 64;

// The request flags word holds the RequestFlag bits and this one, which asks
// for PromptProgressEvents while the prompt prefills.
inline constexpr uint32_t kReturnProgressFlag = 1U << 1;
static_assert(!(kReturnProgressFlag & kRequestFlagBits));

struct RequestFrame {
  uint64_t requestId = 0;
  engine::RequestPriority priority = engine::RequestPriority::Normal;

  // absoluteDeadlineUnixMicros is wall-clock UTC. remainingDeadlineMicros
  // is the sender's remaining budget at serialization time.  Admission
  // should honor the earlier of the two after accounting for transit time.
  uint64_t absoluteDeadlineUnixMicros = 0;
  uint64_t remainingDeadlineMicros = 0;

  uint32_t logicalMaxOutputTokens = 0;
  std::vector<uint32_t> promptTokens;
  // Sorted, non-overlapping image spans and their resized uint8 RGB pixels,
  // concatenated in span order (gridHeight*16 x gridWidth*16 x 3 each).
  // Both are empty for text-only requests.
  std::vector<ImageSpan> imageSpans;
  std::vector<uint8_t> imagePixels;
  // The defaults are greedy selection with nothing changing the logits,
  // which score requests require (seed aside).
  SamplingParameters sampling;
  ConstraintMode constraint = ConstraintMode::None;
  // Empty selects ordinary generation. Nonempty selects score-only mode:
  // 2..255 distinct token ids, logicalMaxOutputTokens must be zero, and the
  // request must be text-only, unconstrained, and greedy.
  std::vector<uint32_t> scoreTokens{};
  // Trailing prompt tokens of the chat template's generation prompt; zero
  // when unknown. It must leave at least one prompt token.
  uint32_t generationPromptTokens = 0;
  // RequestFlag bits and kReturnProgressFlag.
  uint32_t flags = 0;
  // The request's cache_salt as the server hashed it; zero without one.
  engine::CacheSalt cacheSalt;
  // Only a request that generates inside a thinking block has one; a score
  // request never does.
  std::optional<ThinkingBudget> thinkingBudget;

  bool operator==(const RequestFrame &) const = default;
};

// The other payloads, in wire order; counts precede what they count, and
// the variable parts come last:
//   Cancel         u64 requestId
//   MaskResponse   u64 requestId, u64 maskRequestId, u32 count, u32 words
//   StatusRequest  u64 correlationId
//   ReleaseRequest u64 correlationId
//   Stop           u64 requestId
//   Ready          u32 maxConcurrentRequests, u32 maxContextTokens, u8 vision,
//                  u32 vocabularySize, u32 maskRows, u32 stop token count,
//                  u32 stop tokens
//   Start          u64 requestId, u32 lane, u32 matchedPromptTokens
//   PromptProgress u64 requestId, u32 processedTokens, u64 elapsedMicros
//   Tokens         u64 requestId, u32 sequenceOffset, u32 count, u32 tokens
//   MaskRequest    u64 requestId, u64 maskRequestId, u32 wordsPerMask,
//                  u32 count, u32 simulation tokens
//   Done           u64 requestId, u8 reason, u8 bounded, u32 promptTokens,
//                  u32 completionTokens, u64 prefillMicros, u64 decodeMicros,
//                  u64 wallMicros, u32 deduplicatedTokens, u32 draftedTokens,
//                  u32 acceptedDraftTokens, u32 speculative cycles, u32 count,
//                  f32 option logits
//   Error          u8 failureClass, u8 retryable, u32 retryAfterSeconds,
//                  u64 requestId, u32 code bytes, u32 message bytes, the code,
//                  the message
//   StatusJson     u64 correlationId, the JSON
//   StartFailed    u8 kind, u8 bound, u64 requiredBytes, u64 budgetBytes,
//                  u32 planContextTokens, u32 message bytes, the message
//   Warming        u32 stepsDone, u32 steps
struct CancelFrame {
  uint64_t requestId = 0;

  bool operator==(const CancelFrame &) const = default;
};

struct MaskResponseFrame {
  uint64_t requestId = 0;
  uint64_t maskRequestId = 0;
  std::vector<uint32_t> maskWords;

  bool operator==(const MaskResponseFrame &) const = default;
};

struct StatusRequestFrame {
  uint64_t correlationId = 0;

  bool operator==(const StatusRequestFrame &) const = default;
};

// Asks the engine to give its weights back now, as the idle release does, when
// it holds no request and restores none, with the idle release on or off. The
// answer is a StatusJson with the same correlationId, whose `weights` section
// says whether the weights are released: `released` and not `restoring`.
struct ReleaseRequestFrame {
  uint64_t correlationId = 0;

  bool operator==(const ReleaseRequestFrame &) const = default;
};

// The client matched a stop condition: the engine finishes the request as
// Stop, a completion, or cancels it before its first token. The Done event
// still ends the request.
struct StopFrame {
  uint64_t requestId = 0;

  bool operator==(const StopFrame &) const = default;
};

// Sent before Ready once the engine holds its weights, as its warm-up begins and
// as each step ends (ServingRuntime::requireWarmup): on the first start after an
// update the warm-up also prepares the GPU's programs, and the 27B's start then
// took 56 s on a 48 GB M5 Pro.
struct WarmingEvent {
  // The steps done, those memory kept from running among them, of all.
  uint32_t stepsDone = 0;
  uint32_t steps = 0;

  bool operator==(const WarmingEvent &) const = default;
};

struct ReadyEvent {
  uint32_t maxConcurrentRequests = 0;
  // The context this start serves: the one asked for, or what its plan holds;
  // a relaunch may serve less than the first engine did.
  uint32_t maxContextTokens = 0;
  // Requests may carry image spans. A model serving without vision rejects
  // each image request with a request error.
  bool vision = false;
  // The target's logit rows a token mask covers, and the tokens that end
  // generation (its layout's vocabularySize and stopTokens).
  uint32_t vocabularySize = 0;
  std::vector<uint32_t> stopTokens{};
  // The rows of one mask request: before the anchor and after each draft
  // query row (ProtocolLimits::maxSimulationTokens + 1).
  uint32_t maskRows = 0;

  bool operator==(const ReadyEvent &) const = default;
};

struct StartEvent {
  uint64_t requestId = 0;
  uint32_t lane = 0;
  // The cached prefix the request starts from; zero for a cold start.
  uint32_t matchedPromptTokens = 0;

  bool operator==(const StartEvent &) const = default;
};

struct PromptProgressEvent {
  uint64_t requestId = 0;
  uint32_t processedTokens = 0;
  uint64_t elapsedMicros = 0;

  bool operator==(const PromptProgressEvent &) const = default;
};

struct TokensEvent {
  uint64_t requestId = 0;
  uint32_t sequenceOffset = 0;
  std::vector<uint32_t> tokens;

  bool operator==(const TokensEvent &) const = default;
};

struct MaskRequestEvent {
  uint64_t requestId = 0;
  uint64_t maskRequestId = 0;
  uint32_t wordsPerMask = 0;
  std::vector<uint32_t> simulationTokens;

  // Rows describe the constraint state before and after each simulated
  // token. Empty input is the initial-anchor request (one row); verification
  // passes [pending_anchor, draft...] and therefore requires len + 1 rows.
  bool operator==(const MaskRequestEvent &) const = default;
};

struct DoneEvent {
  uint64_t requestId = 0;
  engine::EngineFinishReason reason = engine::EngineFinishReason::Length;
  uint32_t promptTokens = 0;
  uint32_t completionTokens = 0;
  uint64_t prefillMicros = 0;
  uint64_t decodeMicros = 0;
  uint64_t wallMicros = 0;
  // Raw final-prompt-position logits at the request's scoreTokens, in
  // requested order. Empty for generation and for cancelled/failed scoring.
  std::vector<float> optionLogits{};
  // Tokens whose KV the request read from cached blocks another request
  // computed, past the cached prefix it started from
  // (engine::EngineEventSink::completed).
  uint32_t deduplicatedTokens = 0;
  // Its speculative decoding; zero without a draft and for scoring.
  engine::SpeculativeCounts speculation{};
  // A Length end short of the request's logical maximum, at a bound of the
  // engine's own: the SSD cache's quota, or memory a lone lane could not
  // grow past (engine::EngineEventSink::completed).
  bool bounded = false;

  bool operator==(const DoneEvent &) const = default;
};

struct ErrorEvent {
  FailureClass failureClass = FailureClass::RequestError;
  uint64_t requestId = 0;
  bool retryable = false;
  std::string code;
  std::string message;
  // For a retryable request error, the seconds after which a retry may
  // succeed, as when the SSD cache's write budget refills; 0 for at once.
  uint32_t retryAfterSeconds = 0;

  bool operator==(const ErrorEvent &) const = default;
};

// JSON is deliberately opaque to the transport: the document carries its
// own schema_version, and the frame length carries the exact JSON byte count
// (including whitespace) without line or C-string assumptions.
struct StatusJsonEvent {
  uint64_t correlationId = 0;
  std::string json;

  bool operator==(const StatusJsonEvent &) const = default;
};

// Why a start failed, for the supervisor's restart decision and the app's
// words (server/protocol.py StartFailureKind).
enum class StartFailureKind : uint8_t {
  // A defect, a start command that failed but not for want of GPU memory,
  // or a failure the engine does not class.
  Internal = 0,
  // The host had too little free memory (RuntimeResourceFailure::HostCapacity).
  HostMemory = 1,
  // Metal refused an allocation the budget allowed.
  DriverAllocation = 2,
  // The plan's floors exceed the Metal memory budget.
  MemoryBudget = 3,
  // --max-context exceeds the context the plan holds.
  ContextBudget = 4,
  // A setting the engine refuses.
  Configuration = 5,
  // The model's files, which the start cannot serve: a model root without
  // them, or files the loader refuses.
  Model = 6,
  // This Mac cannot run Splash.
  Device = 7,
  // A start command failed for want of GPU memory
  // (MTLCommandBufferErrorOutOfMemory): another process holds what the
  // model's commands need; a retry would take it from that process.
  CommandMemory = 8,
};

// What bounds a shortfall: the Metal budget this Mac gives, the one
// --max-memory set, or the SSD cache's room for spilled KV.
enum class ShortfallBound : uint8_t {
  DeviceMemory = 0,
  MaxMemory = 1,
  Disk = 2,
};

// What a start refused for want of memory or context needed against what it
// had: MemoryBudget sets the bytes, ContextBudget the context its plan holds
// (at least one token: a plan that holds none fails before the check).
struct StartShortfall {
  ShortfallBound bound = ShortfallBound::DeviceMemory;
  uint64_t requiredBytes = 0;
  uint64_t budgetBytes = 0;
  uint32_t planContextTokens = 0;

  bool operator==(const StartShortfall &) const = default;
};

// Sent in place of Ready by a start that fails (main.mm); the process then
// exits.
struct StartFailedEvent {
  StartFailureKind kind = StartFailureKind::Internal;
  // All zero but for MemoryBudget and ContextBudget.
  StartShortfall shortfall;
  std::string message;

  bool operator==(const StartFailedEvent &) const = default;
};

using ClientMessage = std::variant<RequestFrame, CancelFrame, MaskResponseFrame,
                                   StatusRequestFrame, ReleaseRequestFrame,
                                   StopFrame>;
using EngineEvent =
    std::variant<ReadyEvent, StartEvent, PromptProgressEvent, TokensEvent,
                 MaskRequestEvent, DoneEvent, ErrorEvent, StatusJsonEvent,
                 StartFailedEvent, WarmingEvent>;

struct Frame {
  FrameType type = FrameType::Request;
  std::vector<uint8_t> payload;

  bool operator==(const Frame &) const = default;
};

// Takes the frame: a request's image pixels stay in its payload's buffer.
[[nodiscard]] ProtocolResult<ClientMessage>
decodeFrame(Frame &&frame, const ProtocolLimits &limits);
// The whole frame, header included. An event that breaks its rules is the
// engine's defect: the issue is EngineUnhealthy.
[[nodiscard]] ProtocolResult<std::vector<uint8_t>>
serializeEvent(const EngineEvent &event, const ProtocolLimits &limits);

struct ParseStep {
  size_t consumedBytes = 0;
  std::optional<Frame> frame;
  std::optional<ProtocolIssue> issue;
};

// Incremental one-frame-at-a-time parser of client frames.  consume() stops
// as soon as it yields one complete frame, so callers can process arbitrarily
// long streams without retaining a batch of frames.  An event type is an
// unknown frame type.  finish() must be called at EOF to turn a partial
// header or payload into a protocol-fatal truncation.
class FrameParser {
public:
  explicit FrameParser(ProtocolLimits limits);

  [[nodiscard]] ParseStep consume(std::span<const uint8_t> bytes);
  [[nodiscard]] std::optional<ProtocolIssue> finish();

private:
  [[nodiscard]] std::optional<ProtocolIssue> parseHeader();
  [[nodiscard]] ParseStep fail(size_t consumed, ProtocolIssue issue);
  void resetCurrentFrame();

  ProtocolLimits limits_;
  std::array<uint8_t, kFrameHeaderBytes> header_{};
  size_t headerBytes_ = 0;
  bool readingPayload_ = false;
  FrameType currentType_ = FrameType::Request;
  uint64_t expectedPayloadBytes_ = 0;
  std::vector<uint8_t> payload_;
  std::optional<ProtocolIssue> terminalIssue_;
};

} // namespace splash::protocol

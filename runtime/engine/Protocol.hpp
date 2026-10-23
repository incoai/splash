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
inline constexpr uint32_t kStatusSchemaVersion = 6;
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
  VideoRequest = 0x0006,

  Ready = 0x0100,
  Start = 0x0101,
  Tokens = 0x0102,
  MaskRequest = 0x0103,
  Done = 0x0104,
  Error = 0x0105,
  StatusJson = 0x0107,
  PromptProgress = 0x0108,
  VideoProgress = 0x0109,
  VideoOutput = 0x010A,
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
//   Ready          u32 maxConcurrentRequests, u32 maxContextTokens,
//                  u8 features (ReadyFeature)
//   Start          u64 requestId, u32 lane, u32 matchedPromptTokens
//   PromptProgress u64 requestId, u32 processedTokens, u64 elapsedMicros
//   Tokens         u64 requestId, u32 sequenceOffset, u32 count, u32 tokens
//   MaskRequest    u64 requestId, u64 maskRequestId, u32 wordsPerMask,
//                  u32 count, u32 simulation tokens
//   Done           u64 requestId, u8 reason, u32 promptTokens,
//                  u32 completionTokens, u64 prefillMicros, u64 decodeMicros,
//                  u64 wallMicros, u32 deduplicatedTokens, u32 draftedTokens,
//                  u32 acceptedDraftTokens, u32 speculative cycles, u32 count,
//                  f32 option logits
//   Error          u8 failureClass, u8 retryable, u64 requestId, u32 code
//                  bytes, u32 message bytes, the code, the message
//   StatusJson     u64 correlationId, the JSON
//   VideoRequest   u64 requestId, u8 variant, u8 outputs (VideoOutputFlag),
//                  u32 frames, u32 width, u32 height, u32 steps, u64 seed,
//                  u8 keyframes (VideoKeyframeFlag), u32 prompt token
//                  count, the prompt tokens, then the first frame's and the
//                  last frame's pixels, each height x width x 3 bytes
//   VideoProgress  u64 requestId, u8 phase, u32 step, u32 steps,
//                  u64 elapsedMicros, u64 peakFootprintBytes
//   VideoOutput    u64 requestId, u8 output, u32 frames, u64 wallMicros,
//                  the mp4
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

// The bits of a ReadyEvent's features byte.
enum ReadyFeature : uint8_t {
  FeatureVision = 1U << 0,
  FeatureVideo = 1U << 1,
};

struct ReadyEvent {
  uint32_t maxConcurrentRequests = 0;
  uint32_t maxContextTokens = 0;
  // Requests may carry image spans. A model serving without vision rejects
  // each image request with a request error.
  bool vision = false;
  // The engine serves a video model (runtime/video/VideoRuntime.hpp): it
  // takes VideoRequests and Cancels, and maxContextTokens bounds a video
  // prompt's tokens.
  bool video = false;

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

// One clip of a video model, of a prompt of the text encoder's token ids.
// Answered by VideoProgress events, then the clip's VideoOutput, or an Error.
enum class VideoVariant : uint8_t { Turbo = 0, Default = 1 };
enum VideoOutputFlag : uint8_t {
  VideoOutputPreview = 1U << 0,
  VideoOutputAudio = 1U << 1,
};
// The keyframes a video request carries.
enum VideoKeyframeFlag : uint8_t {
  VideoKeyframeFirst = 1U << 0,
  VideoKeyframeLast = 1U << 1,
};

struct VideoRequestFrame {
  uint64_t requestId = 0;
  VideoVariant variant = VideoVariant::Turbo;
  // VideoOutputFlag bits: also the preview; the clip with its audio.
  uint8_t outputs = VideoOutputAudio;
  uint32_t frames = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  // Denoising steps; zero for the variant's.
  uint32_t steps = 0;
  uint64_t seed = 0;
  std::vector<uint32_t> promptTokens;
  // Keyframes (first-frame and first-and-last-frame conditioning), each
  // empty or uint8 RGB [height][width][3], already on the clip's canvas
  // (server/images.py keyframe): the first frame anchors the clip's first
  // latent frame, the last frame its last.
  std::vector<uint8_t> firstFrame;
  std::vector<uint8_t> lastFrame;

  bool operator==(const VideoRequestFrame &) const = default;
};

// The phases of a video request, in their order.
enum class VideoPhase : uint8_t {
  Encode = 1,
  Prepare,
  Denoise,
  Preview,
  Decode,
  Audio,
  Mux,
};

// A phase has done `step` of its `steps` (one but the denoising's), the
// phase's time so far and the process's peak footprint; step 0 starts it
// and `steps` completes it.
struct VideoProgressEvent {
  uint64_t requestId = 0;
  VideoPhase phase = VideoPhase::Encode;
  uint32_t step = 0;
  uint32_t steps = 1;
  uint64_t elapsedMicros = 0;
  uint64_t peakFootprintBytes = 0;

  bool operator==(const VideoProgressEvent &) const = default;
};

// An mp4 of the request: the preview, once its phase completes, or the clip,
// which ends the request.
enum class VideoOutputKind : uint8_t { Video = 0, Preview = 1 };

struct VideoOutputEvent {
  uint64_t requestId = 0;
  VideoOutputKind output = VideoOutputKind::Video;
  uint32_t frames = 0;
  // From the request's arrival.
  uint64_t wallMicros = 0;
  std::vector<uint8_t> mp4;

  bool operator==(const VideoOutputEvent &) const = default;
};

// JSON is deliberately opaque to the transport: the document carries its
// own schema_version, and the frame length carries the exact JSON byte count
// (including whitespace) without line or C-string assumptions.
struct StatusJsonEvent {
  uint64_t correlationId = 0;
  std::string json;

  bool operator==(const StatusJsonEvent &) const = default;
};

using ClientMessage =
    std::variant<RequestFrame, CancelFrame, MaskResponseFrame,
                 StatusRequestFrame, ReleaseRequestFrame, VideoRequestFrame>;
using EngineEvent =
    std::variant<ReadyEvent, StartEvent, PromptProgressEvent, TokensEvent,
                 MaskRequestEvent, DoneEvent, ErrorEvent, StatusJsonEvent,
                 VideoProgressEvent, VideoOutputEvent>;

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

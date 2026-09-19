#pragma once

#include "ops/Vision.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace splash::protocol {

inline constexpr uint16_t kProtocolVersion = 6;
inline constexpr size_t kFrameHeaderBytes = 24;
inline constexpr uint32_t kStatusSchemaVersion = 5;
// Image pixels travel inside the request frame; a multi-image agent turn can
// carry well over 64 MiB of resized RGB bytes.
inline constexpr uint64_t kAbsoluteMaxFramePayloadBytes =
    256ULL * 1024 * 1024;

// Every integer, including binary prompt/token words, is little-endian on the
// wire.  Header flags and reserved bytes must be zero in native protocol.
enum class FrameType : uint16_t {
  Request = 0x0001,
  Cancel = 0x0002,
  MaskResponse = 0x0003,
  StatusRequest = 0x0004,

  Ready = 0x0100,
  Start = 0x0101,
  Tokens = 0x0102,
  MaskRequest = 0x0103,
  Done = 0x0104,
  Error = 0x0105,
  CapacityExhausted = 0x0106,
  StatusJson = 0x0107,
  PromptProgress = 0x0108,
};

// Request errors reject one request while preserving the stream.  An
// engine-unhealthy error asks the supervisor to replace the engine.  A
// protocol-fatal error means framing is no longer trusted, so the stream must
// close without attempting another framing mode.
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
  InvalidHeaderSize,
  UnknownFrameType,
  NonZeroHeaderFlags,
  NonZeroReservedField,
  FrameTooLarge,
  InvalidPayloadLength,
  TruncatedFrame,
  ParserAlreadyFailed,
  InvalidRequestId,
  InvalidEnumValue,
  InvalidDeadline,
  InvalidSampling,
  InvalidCount,
  InvalidCohortConstraint,
  InvalidErrorClassification,
  InvalidStatusSchema,
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
  // Patches per image; the engine sizes its vision scratch from the same
  // value, so a frame limit violation is never a late allocation failure.
  uint32_t maxImagePatches = ops::kMaximumImagePatches;
};
// Direct finite-option scoring (SemIf/Jev System One): a request carrying
// scoreTokens runs prefill only and returns the raw final-position logits at
// those token ids in DoneEvent.optionLogits. The option count is bounded by
// the wire contract; the engine additionally requires vocabulary bounds.
inline constexpr uint32_t kMinimumScoreOptions = 2;
inline constexpr uint32_t kMaximumScoreOptions = 255;

enum class RequestPriority : uint8_t {
  Foreground = 0,
  Normal = 1,
  Background = 2,
};

enum class Cohort : uint8_t {
  Greedy = 0,
  Sampling = 1,
  Constrained = 2,
};

enum class ConstraintMode : uint8_t {
  None = 0,
  TokenMask = 1,
};

struct SamplingParameters {
  float temperature = 0.0f;
  float topP = 1.0f;
  uint32_t topK = 0;

  bool operator==(const SamplingParameters &) const = default;
};

// One image in the prompt: the run of placeholder tokens it occupies (one per
// merged 2x2 patch group, row-major over the merged grid), the patch grid of
// the frontend's resized pixels, and a 128-bit digest of that content.
// Placeholder token ids are identical for every image, so cache identity keys
// on the digest as well as the tokens.
struct ImageSpanFrame {
  uint32_t offset = 0;
  uint32_t tokens = 0;
  uint32_t gridHeight = 0;
  uint32_t gridWidth = 0;
  uint64_t digestLo = 0;
  uint64_t digestHi = 0;

  [[nodiscard]] uint64_t pixelBytes() const noexcept {
    return ops::imagePixelBytes(gridHeight, gridWidth);
  }
  bool operator==(const ImageSpanFrame &) const = default;
};

struct RequestFrame {
  uint64_t requestId = 0;
  RequestPriority priority = RequestPriority::Normal;

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
  std::vector<ImageSpanFrame> imageSpans;
  std::vector<uint8_t> imagePixels;
  SamplingParameters sampling;
  uint64_t seed = 0;
  Cohort cohort = Cohort::Greedy;
  ConstraintMode constraint = ConstraintMode::None;
  bool returnProgress = false;
  // Empty selects ordinary generation. Nonempty selects score-only mode:
  // 2..255 distinct token ids, logicalMaxOutputTokens must be zero, and the
  // request must be text-only, unconstrained, and greedy.
  std::vector<uint32_t> scoreTokens{};

  bool operator==(const RequestFrame &) const = default;
};

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

enum ReadyFeature : uint64_t {
  FeatureCancellation = 1ULL << 0,
  FeatureTokenMasks = 1ULL << 1,
  FeatureStatusJson = 1ULL << 2,
  FeatureMultiplexing = 1ULL << 3,
};

// The native runtime implements every feature; ReadyEvent announces them all.
inline constexpr uint64_t kNativeFeatureBits =
    FeatureCancellation | FeatureTokenMasks | FeatureStatusJson |
    FeatureMultiplexing;

struct ReadyEvent {
  uint64_t engineInstanceId = 0;
  uint32_t maxConcurrentRequests = 0;
  uint32_t maxContextTokens = 0;
  uint64_t featureBits = 0;

  bool operator==(const ReadyEvent &) const = default;
};

enum class CacheDisposition : uint8_t {
  Miss = 0,
  PrefixHit = 1,
};

struct StartEvent {
  uint64_t requestId = 0;
  CacheDisposition cacheDisposition = CacheDisposition::Miss;
  int32_t slotIndex = -1;
  uint32_t matchedPromptTokens = 0;
  uint32_t capacityTokens = 0;

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

enum class FinishReason : uint8_t {
  Stop = 0,
  Length = 1,
  Cancelled = 2,
};

struct DoneEvent {
  uint64_t requestId = 0;
  FinishReason reason = FinishReason::Length;
  uint32_t promptTokens = 0;
  uint32_t completionTokens = 0;
  uint64_t prefillMicros = 0;
  uint64_t decodeMicros = 0;
  uint64_t wallMicros = 0;
  // Raw final-prompt-position logits at the request's scoreTokens, in
  // requested order. Empty for generation and for cancelled/failed scoring.
  std::vector<float> optionLogits{};

  bool operator==(const DoneEvent &) const = default;
};

struct ErrorEvent {
  FailureClass failureClass = FailureClass::RequestError;
  uint64_t requestId = 0;
  bool retryable = false;
  std::string code;
  std::string message;

  bool operator==(const ErrorEvent &) const = default;
};

struct CapacityExhaustedEvent {
  uint64_t requestId = 0;
  uint32_t requiredKvPages = 0;
  uint32_t availableKvPages = 0;
  uint64_t retryAfterMicros = 0;

  bool operator==(const CapacityExhaustedEvent &) const = default;
};

// JSON is deliberately opaque to the transport.  Its independent schema
// number is always present, and the frame length carries the exact JSON byte
// count (including whitespace) without line or C-string assumptions.
struct StatusJsonEvent {
  uint64_t correlationId = 0;
  uint32_t schemaVersion = kStatusSchemaVersion;
  std::string json;

  bool operator==(const StatusJsonEvent &) const = default;
};

using Message =
    std::variant<RequestFrame, CancelFrame, MaskResponseFrame,
                 StatusRequestFrame, ReadyEvent, StartEvent,
                 PromptProgressEvent, TokensEvent, MaskRequestEvent, DoneEvent,
                 ErrorEvent, CapacityExhaustedEvent, StatusJsonEvent>;

struct Frame {
  FrameType type = FrameType::Request;
  std::vector<uint8_t> payload;

  bool operator==(const Frame &) const = default;
};

[[nodiscard]] ProtocolResult<Frame>
encodeMessage(const Message &message, const ProtocolLimits &limits = {});
[[nodiscard]] ProtocolResult<Message>
decodeFrame(const Frame &frame, const ProtocolLimits &limits = {});
[[nodiscard]] ProtocolResult<std::vector<uint8_t>>
serializeMessage(const Message &message, const ProtocolLimits &limits = {});

struct ParseStep {
  size_t consumedBytes = 0;
  std::optional<Frame> frame;
  std::optional<ProtocolIssue> issue;
};

// Incremental one-frame-at-a-time parser.  consume() stops as soon as it
// yields one complete frame, so callers can process arbitrarily long streams
// without retaining a batch of frames.  finish() must be called at EOF to
// turn a partial header or payload into a protocol-fatal truncation.
class FrameParser {
public:
  explicit FrameParser(ProtocolLimits limits = {});

  [[nodiscard]] ParseStep consume(std::span<const uint8_t> bytes);
  [[nodiscard]] std::optional<ProtocolIssue> finish();
  [[nodiscard]] bool failed() const noexcept {
    return terminalIssue_.has_value();
  }

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
  size_t payloadBytes_ = 0;
  std::vector<uint8_t> payload_;
  std::optional<ProtocolIssue> terminalIssue_;
};

} // namespace splash::protocol

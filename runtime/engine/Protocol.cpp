#include "engine/Protocol.hpp"
#include "Checked.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace splash::protocol {
namespace {

using engine::EngineFinishReason;
using engine::RequestPriority;
using model::ExecutionLimits;

std::string_view frameTypeName(FrameType type);
std::string_view failureClassName(FailureClass failureClass);

constexpr std::array<uint8_t, 4> kMagic{'S', 'P', 'L', 'H'};
constexpr uint64_t kCancelFixedBytes = 8;
constexpr uint64_t kMaskResponseFixedBytes = 20;
constexpr uint64_t kStatusRequestFixedBytes = 8;
constexpr uint64_t kReadyFixedBytes = 9;
constexpr uint64_t kStartFixedBytes = 16;
constexpr uint64_t kPromptProgressFixedBytes = 20;
constexpr uint64_t kTokensFixedBytes = 16;
constexpr uint64_t kMaskRequestFixedBytes = 24;
constexpr uint64_t kDoneFixedBytes = 45;
constexpr uint64_t kErrorFixedBytes = 18;
constexpr uint64_t kStatusJsonFixedBytes = 8;

struct PayloadBounds {
  uint64_t minimum = 0;
  uint64_t maximum = 0;
};

ProtocolIssue makeIssue(FailureClass failureClass, IssueCode code,
                        uint64_t requestId, std::string message) {
  return {failureClass, code, requestId, std::move(message)};
}

// A client frame's failure rejects the request it names. One that names none
// cannot be answered as a request's error: the stream closes.
ProtocolIssue requestIssue(IssueCode code, uint64_t requestId,
                           std::string message) {
  return makeIssue(requestId ? FailureClass::RequestError
                             : FailureClass::ProtocolFatal,
                   code, requestId, std::move(message));
}

template <typename T> ProtocolResult<T> success(T value) {
  return {std::move(value), std::nullopt};
}

template <typename T> ProtocolResult<T> failure(ProtocolIssue issue) {
  return {std::nullopt, std::move(issue)};
}

std::optional<PayloadBounds> payloadBounds(FrameType type,
                                           const ProtocolLimits &limits) {
  uint64_t variable = 0;
  uint64_t maximum = 0;
  auto bounded =
      [&](uint64_t minimum,
          uint64_t requestedMaximum) -> std::optional<PayloadBounds> {
    return PayloadBounds{
        minimum, std::min(requestedMaximum, limits.maxFramePayloadBytes)};
  };
  switch (type) {
  case FrameType::Request:
    // Image pixels dominate prompt tokens; the frame limit is the bound.
    return bounded(kRequestFixedBytes, limits.maxFramePayloadBytes);
  case FrameType::Cancel:
    return bounded(kCancelFixedBytes, kCancelFixedBytes);
  case FrameType::MaskResponse:
    if (!checkedMultiply(limits.maxMaskWords, sizeof(uint32_t), variable) ||
        !checkedAdd(kMaskResponseFixedBytes, variable, maximum)) {
      return std::nullopt;
    }
    return bounded(kMaskResponseFixedBytes, maximum);
  case FrameType::StatusRequest:
    return bounded(kStatusRequestFixedBytes, kStatusRequestFixedBytes);
  default:
    // The engine never receives an event.
    return std::nullopt;
  }
}

std::optional<ProtocolIssue>
validatePayloadLength(FrameType type, uint64_t payloadBytes,
                      const ProtocolLimits &limits) {
  auto bounds = payloadBounds(type, limits);
  if (!bounds) {
    return makeIssue(
        FailureClass::ProtocolFatal, IssueCode::UnknownFrameType, 0,
        "unknown frame type or overflow while deriving its bounds");
  }
  if (payloadBytes > bounds->maximum) {
    std::ostringstream message;
    message << frameTypeName(type) << " payload length " << payloadBytes
            << " exceeds its safe limit " << bounds->maximum;
    return makeIssue(FailureClass::ProtocolFatal, IssueCode::FrameTooLarge, 0,
                     message.str());
  }
  if (payloadBytes < bounds->minimum) {
    std::ostringstream message;
    message << frameTypeName(type) << " payload length " << payloadBytes
            << " is below its required minimum " << bounds->minimum;
    return makeIssue(FailureClass::ProtocolFatal,
                     IssueCode::InvalidPayloadLength, 0, message.str());
  }
  return std::nullopt;
}

bool clientFrameType(uint16_t raw, FrameType &type) {
  switch (static_cast<FrameType>(raw)) {
  case FrameType::Request:
  case FrameType::Cancel:
  case FrameType::MaskResponse:
  case FrameType::StatusRequest:
    type = static_cast<FrameType>(raw);
    return true;
  default:
    return false;
  }
}

uint16_t loadU16(const uint8_t *bytes) {
  return static_cast<uint16_t>(bytes[0]) |
         (static_cast<uint16_t>(bytes[1]) << 8);
}

uint32_t loadU32(const uint8_t *bytes) {
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) |
         (static_cast<uint32_t>(bytes[3]) << 24);
}

uint64_t loadU64(const uint8_t *bytes) {
  uint64_t result = 0;
  for (uint32_t index = 0; index < 8; ++index) {
    result |= static_cast<uint64_t>(bytes[index]) << (index * 8);
  }
  return result;
}

class Writer {
public:
  explicit Writer(size_t reserveBytes) { bytes_.reserve(reserveBytes); }

  void u8(uint8_t value) { bytes_.push_back(value); }

  void u16(uint16_t value) {
    for (uint32_t index = 0; index < 2; ++index) {
      u8(static_cast<uint8_t>(value >> (index * 8)));
    }
  }

  void u32(uint32_t value) {
    for (uint32_t index = 0; index < 4; ++index) {
      u8(static_cast<uint8_t>(value >> (index * 8)));
    }
  }

  void u64(uint64_t value) {
    for (uint32_t index = 0; index < 8; ++index) {
      u8(static_cast<uint8_t>(value >> (index * 8)));
    }
  }

  void f32(float value) { u32(std::bit_cast<uint32_t>(value)); }

  void raw(std::span<const uint8_t> bytes) {
    bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
  }

  void text(std::string_view value) {
    raw(std::span<const uint8_t>(
        reinterpret_cast<const uint8_t *>(value.data()), value.size()));
  }

  [[nodiscard]] std::vector<uint8_t> take() { return std::move(bytes_); }

private:
  std::vector<uint8_t> bytes_;
};

// A frame's header, in a buffer sized for the payload the caller writes next.
Writer frameWriter(FrameType type, uint64_t payloadBytes) {
  Writer writer(kFrameHeaderBytes + payloadBytes);
  writer.raw(kMagic);
  writer.u16(kProtocolVersion);
  writer.u16(static_cast<uint16_t>(type));
  writer.u64(payloadBytes);
  return writer;
}

class Reader {
public:
  explicit Reader(std::span<const uint8_t> bytes) : bytes_(bytes) {}

  bool u8(uint8_t &value) {
    if (remaining() < 1)
      return false;
    value = bytes_[offset_++];
    return true;
  }

  bool u32(uint32_t &value) {
    if (remaining() < 4)
      return false;
    value = loadU32(bytes_.data() + offset_);
    offset_ += 4;
    return true;
  }

  bool u64(uint64_t &value) {
    if (remaining() < 8)
      return false;
    value = loadU64(bytes_.data() + offset_);
    offset_ += 8;
    return true;
  }

  bool f32(float &value) {
    uint32_t bits = 0;
    if (!u32(bits))
      return false;
    value = std::bit_cast<float>(bits);
    return true;
  }

  bool skip(uint64_t count) {
    if (count > remaining())
      return false;
    offset_ += static_cast<size_t>(count);
    return true;
  }

  bool words(uint32_t count, std::vector<uint32_t> &values) {
    uint64_t bytes = 0;
    if (!checkedMultiply(count, sizeof(uint32_t), bytes) ||
        bytes > remaining()) {
      return false;
    }
    values.resize(count);
    for (uint32_t &value : values) {
      if (!u32(value))
        return false;
    }
    return true;
  }

  [[nodiscard]] size_t remaining() const { return bytes_.size() - offset_; }

private:
  std::span<const uint8_t> bytes_;
  size_t offset_ = 0;
};

template <typename Enum>
bool validEnum(uint8_t raw, std::initializer_list<Enum> values) {
  return std::any_of(values.begin(), values.end(), [raw](Enum candidate) {
    return raw == static_cast<uint8_t>(candidate);
  });
}

std::optional<ProtocolIssue> validateRequest(const RequestFrame &request,
                                             const ProtocolLimits &limits) {
  auto invalid = [&](IssueCode code, std::string message) {
    return std::optional<ProtocolIssue>(
        requestIssue(code, request.requestId, std::move(message)));
  };
  if (!request.requestId) {
    return invalid(IssueCode::InvalidRequestId, "request id must be non-zero");
  }
  if (!validEnum(static_cast<uint8_t>(request.priority),
                 {RequestPriority::Foreground, RequestPriority::Normal,
                  RequestPriority::Background})) {
    return invalid(IssueCode::InvalidEnumValue,
                   "request priority is not defined by native protocol");
  }
  if (!validEnum(static_cast<uint8_t>(request.constraint),
                 {ConstraintMode::None, ConstraintMode::TokenMask})) {
    return invalid(IssueCode::InvalidEnumValue,
                   "constraint mode is not defined by native protocol");
  }
  if (request.flags & ~(kRequestFlagBits | kReturnProgressFlag)) {
    return invalid(IssueCode::InvalidEnumValue,
                   "request flags are not defined by native protocol");
  }
  if (!request.absoluteDeadlineUnixMicros || !request.remainingDeadlineMicros) {
    return invalid(IssueCode::InvalidDeadline,
                   "absolute and remaining deadlines must be non-zero");
  }
  const bool scoring = !request.scoreTokens.empty();
  if (scoring) {
    if (request.logicalMaxOutputTokens != 0) {
      return invalid(IssueCode::InvalidCount,
                     "score requests must not produce output tokens");
    }
  } else if (!request.logicalMaxOutputTokens ||
             request.logicalMaxOutputTokens > limits.maxLogicalOutputTokens) {
    return invalid(IssueCode::LimitExceeded,
                   "logical max output token count exceeds its limit");
  }
  if (request.promptTokens.empty() ||
      request.promptTokens.size() > limits.maxPromptTokens) {
    return invalid(IssueCode::LimitExceeded,
                   "prompt token count exceeds its limit");
  }
  if (request.imageSpans.size() > limits.maxImageSpans) {
    return invalid(IssueCode::LimitExceeded,
                   "image span count exceeds its limit");
  }
  if (auto error = imageSpansValidationError(request.imageSpans,
                                             request.promptTokens.size(),
                                             request.imagePixels.size())) {
    return invalid(IssueCode::InvalidCount, std::string(*error));
  }
  if (request.generationPromptTokens >= request.promptTokens.size()) {
    return invalid(IssueCode::InvalidCount,
                   "generation prompt must leave a prompt token");
  }
  if (scoring) {
    if (!request.imageSpans.empty()) {
      return invalid(IssueCode::InvalidCount,
                     "score requests are text-only");
    }
    if (request.scoreTokens.size() < ExecutionLimits::minimumScoreOptions ||
        request.scoreTokens.size() > ExecutionLimits::maximumScoreOptions) {
      return invalid(IssueCode::InvalidCount,
                     "score option count must be in [2, 255]");
    }
    std::vector<uint32_t> distinct(request.scoreTokens.begin(),
                                   request.scoreTokens.end());
    std::sort(distinct.begin(), distinct.end());
    if (std::adjacent_find(distinct.begin(), distinct.end()) !=
        distinct.end()) {
      return invalid(IssueCode::InvalidCount,
                     "score option token ids must be distinct");
    }
  }
  if (auto error = request.sampling.validationError()) {
    return invalid(IssueCode::InvalidSampling, std::string(*error));
  }
  if (scoring) {
    if (request.constraint != ConstraintMode::None) {
      return invalid(IssueCode::InvalidConstraint,
                     "score requests cannot carry a constraint");
    }
    if (!request.sampling.isNeutral()) {
      return invalid(IssueCode::InvalidSampling,
                     "score requests require greedy default sampling");
    }
  }
  // A grammar decides where constrained output ends.
  if ((request.flags & RequestIgnoreEndOfSequence) &&
      (scoring || request.constraint != ConstraintMode::None)) {
    return invalid(IssueCode::InvalidConstraint,
                   "only unconstrained generation can ignore end-of-sequence");
  }
  return std::nullopt;
}

std::optional<ProtocolIssue> validateCancel(const CancelFrame &cancel) {
  if (cancel.requestId)
    return std::nullopt;
  return requestIssue(IssueCode::InvalidRequestId, 0,
                      "cancel request id must be non-zero");
}

std::optional<ProtocolIssue>
validateMaskResponse(const MaskResponseFrame &response,
                     const ProtocolLimits &limits) {
  if (!response.requestId || !response.maskRequestId) {
    return requestIssue(IssueCode::InvalidRequestId, response.requestId,
                        "mask response request ids must be non-zero");
  }
  if (response.maskWords.empty() ||
      response.maskWords.size() > limits.maxMaskWords) {
    return requestIssue(IssueCode::LimitExceeded, response.requestId,
                        "mask response word count exceeds its limit");
  }
  return std::nullopt;
}

ProtocolIssue invalidEvent(IssueCode code, uint64_t requestId,
                           std::string message) {
  return makeIssue(FailureClass::EngineUnhealthy, code, requestId,
                   std::move(message));
}

std::optional<ProtocolIssue> validateReady(const ReadyEvent &event) {
  if (event.maxConcurrentRequests && event.maxContextTokens)
    return std::nullopt;
  return invalidEvent(IssueCode::InvalidCount, 0,
                      "ready event capacities must be non-zero");
}

std::optional<ProtocolIssue> validateStart(const StartEvent &event) {
  if (!event.requestId) {
    return invalidEvent(IssueCode::InvalidRequestId, 0,
                        "start event request id must be non-zero");
  }
  if (event.lane >= ExecutionLimits::maximumBatchWidth) {
    return invalidEvent(IssueCode::InvalidCount, event.requestId,
                        "start event lane is invalid");
  }
  return std::nullopt;
}

std::optional<ProtocolIssue>
validatePromptProgress(const PromptProgressEvent &event,
                       const ProtocolLimits &limits) {
  if (!event.requestId || event.processedTokens > limits.maxPromptTokens)
    return invalidEvent(IssueCode::InvalidCount, event.requestId,
                        "prompt progress id or token count is invalid");
  return std::nullopt;
}

std::optional<ProtocolIssue> validateTokens(const TokensEvent &event,
                                            const ProtocolLimits &limits) {
  if (!event.requestId) {
    return invalidEvent(IssueCode::InvalidRequestId, 0,
                        "tokens event request id must be non-zero");
  }
  if (event.tokens.empty() || event.tokens.size() > limits.maxTokenBatch) {
    return invalidEvent(IssueCode::LimitExceeded, event.requestId,
                        "tokens event batch size exceeds its limit");
  }
  uint64_t end = 0;
  if (!checkedAdd(event.sequenceOffset, event.tokens.size(), end) ||
      end > std::numeric_limits<uint32_t>::max()) {
    return invalidEvent(IssueCode::IntegerOverflow, event.requestId,
                        "tokens event sequence range overflows uint32");
  }
  return std::nullopt;
}

std::optional<ProtocolIssue> validateMaskRequest(const MaskRequestEvent &event,
                                                 const ProtocolLimits &limits) {
  if (!event.requestId || !event.maskRequestId) {
    return invalidEvent(IssueCode::InvalidRequestId, event.requestId,
                        "mask request ids must be non-zero");
  }
  // An empty simulation sequence is the initial-anchor request and
  // deliberately produces one mask row. During speculative verification
  // the sequence is [pending_anchor, draft...], so len+1 rows represent
  // before-anchor and after each simulated token.
  if (!event.wordsPerMask ||
      event.simulationTokens.size() > limits.maxSimulationTokens) {
    return invalidEvent(IssueCode::InvalidCount, event.requestId,
                        "mask request dimensions are invalid");
  }
  uint64_t maskRows = 0;
  uint64_t totalWords = 0;
  if (!checkedAdd(event.simulationTokens.size(), 1, maskRows) ||
      !checkedMultiply(event.wordsPerMask, maskRows, totalWords) ||
      totalWords > limits.maxMaskWords) {
    return invalidEvent(IssueCode::LimitExceeded, event.requestId,
                        "mask request output would exceed the mask limit");
  }
  return std::nullopt;
}

std::optional<ProtocolIssue> validateDone(const DoneEvent &event) {
  if (!event.requestId) {
    return invalidEvent(IssueCode::InvalidRequestId, 0,
                        "done event request id must be non-zero");
  }
  if (!validEnum(static_cast<uint8_t>(event.reason),
                 {EngineFinishReason::Stop, EngineFinishReason::Length,
                  EngineFinishReason::Cancelled})) {
    return invalidEvent(IssueCode::InvalidEnumValue, event.requestId,
                        "done finish reason is invalid");
  }
  if (!event.optionLogits.empty()) {
    if (event.optionLogits.size() < ExecutionLimits::minimumScoreOptions ||
        event.optionLogits.size() > ExecutionLimits::maximumScoreOptions) {
      return invalidEvent(IssueCode::InvalidCount, event.requestId,
                          "done option logit count must be in [2, 255]");
    }
    if (event.reason != EngineFinishReason::Stop ||
        event.completionTokens != 0 || event.decodeMicros != 0) {
      return invalidEvent(IssueCode::InvalidCount, event.requestId,
                          "scored done events must stop and carry no "
                          "completion or decode activity");
    }
    for (float logit : event.optionLogits) {
      if (!std::isfinite(logit)) {
        return invalidEvent(IssueCode::InvalidCount, event.requestId,
                            "done option logits must be finite");
      }
    }
  }
  return std::nullopt;
}

std::optional<ProtocolIssue> validateError(const ErrorEvent &event,
                                           const ProtocolLimits &limits) {
  if (!validEnum(static_cast<uint8_t>(event.failureClass),
                 {FailureClass::RequestError, FailureClass::EngineUnhealthy,
                  FailureClass::ProtocolFatal})) {
    return invalidEvent(IssueCode::InvalidErrorClassification, event.requestId,
                        "error event classification is invalid");
  }
  if ((event.failureClass == FailureClass::RequestError && !event.requestId) ||
      (event.failureClass != FailureClass::RequestError && event.requestId)) {
    return invalidEvent(IssueCode::InvalidErrorClassification, event.requestId,
                        "only request errors may carry a non-zero request id");
  }
  if (event.code.empty() || event.code.size() > limits.maxErrorStringBytes ||
      event.message.size() > limits.maxErrorStringBytes) {
    return invalidEvent(IssueCode::LimitExceeded, event.requestId,
                        "error code or message exceeds its safe limit");
  }
  return std::nullopt;
}

std::optional<ProtocolIssue> validateStatusJson(const StatusJsonEvent &event,
                                                const ProtocolLimits &limits) {
  if (event.json.empty() || event.json.size() > limits.maxStatusJsonBytes) {
    return invalidEvent(IssueCode::LimitExceeded, 0,
                        "status JSON byte count exceeds its safe limit");
  }
  return std::nullopt;
}

using EncodedEvent = ProtocolResult<std::vector<uint8_t>>;

EncodedEvent encodeReady(const ReadyEvent &event) {
  if (auto issue = validateReady(event))
    return failure<std::vector<uint8_t>>(std::move(*issue));
  Writer writer = frameWriter(FrameType::Ready, kReadyFixedBytes);
  writer.u32(event.maxConcurrentRequests);
  writer.u32(event.maxContextTokens);
  writer.u8(event.vision);
  return success(writer.take());
}

EncodedEvent encodeStart(const StartEvent &event) {
  if (auto issue = validateStart(event))
    return failure<std::vector<uint8_t>>(std::move(*issue));
  Writer writer = frameWriter(FrameType::Start, kStartFixedBytes);
  writer.u64(event.requestId);
  writer.u32(event.lane);
  writer.u32(event.matchedPromptTokens);
  return success(writer.take());
}

EncodedEvent encodePromptProgress(const PromptProgressEvent &event,
                                  const ProtocolLimits &limits) {
  if (auto issue = validatePromptProgress(event, limits))
    return failure<std::vector<uint8_t>>(std::move(*issue));
  Writer writer =
      frameWriter(FrameType::PromptProgress, kPromptProgressFixedBytes);
  writer.u64(event.requestId);
  writer.u32(event.processedTokens);
  writer.u64(event.elapsedMicros);
  return success(writer.take());
}

EncodedEvent encodeTokens(const TokensEvent &event,
                          const ProtocolLimits &limits) {
  if (auto issue = validateTokens(event, limits))
    return failure<std::vector<uint8_t>>(std::move(*issue));
  Writer writer =
      frameWriter(FrameType::Tokens,
                  kTokensFixedBytes + event.tokens.size() * sizeof(uint32_t));
  writer.u64(event.requestId);
  writer.u32(event.sequenceOffset);
  writer.u32(static_cast<uint32_t>(event.tokens.size()));
  for (uint32_t token : event.tokens)
    writer.u32(token);
  return success(writer.take());
}

EncodedEvent encodeMaskRequest(const MaskRequestEvent &event,
                               const ProtocolLimits &limits) {
  if (auto issue = validateMaskRequest(event, limits))
    return failure<std::vector<uint8_t>>(std::move(*issue));
  Writer writer =
      frameWriter(FrameType::MaskRequest,
                  kMaskRequestFixedBytes +
                      event.simulationTokens.size() * sizeof(uint32_t));
  writer.u64(event.requestId);
  writer.u64(event.maskRequestId);
  writer.u32(event.wordsPerMask);
  writer.u32(static_cast<uint32_t>(event.simulationTokens.size()));
  for (uint32_t token : event.simulationTokens)
    writer.u32(token);
  return success(writer.take());
}

EncodedEvent encodeDone(const DoneEvent &event) {
  if (auto issue = validateDone(event))
    return failure<std::vector<uint8_t>>(std::move(*issue));
  Writer writer =
      frameWriter(FrameType::Done,
                  kDoneFixedBytes + event.optionLogits.size() * sizeof(float));
  writer.u64(event.requestId);
  writer.u8(static_cast<uint8_t>(event.reason));
  writer.u32(event.promptTokens);
  writer.u32(event.completionTokens);
  writer.u64(event.prefillMicros);
  writer.u64(event.decodeMicros);
  writer.u64(event.wallMicros);
  writer.u32(static_cast<uint32_t>(event.optionLogits.size()));
  for (float logit : event.optionLogits)
    writer.f32(logit);
  return success(writer.take());
}

EncodedEvent encodeError(const ErrorEvent &event,
                         const ProtocolLimits &limits) {
  if (auto issue = validateError(event, limits))
    return failure<std::vector<uint8_t>>(std::move(*issue));
  Writer writer =
      frameWriter(FrameType::Error,
                  kErrorFixedBytes + event.code.size() + event.message.size());
  writer.u8(static_cast<uint8_t>(event.failureClass));
  writer.u8(event.retryable ? 1 : 0);
  writer.u64(event.requestId);
  writer.u32(static_cast<uint32_t>(event.code.size()));
  writer.u32(static_cast<uint32_t>(event.message.size()));
  writer.text(event.code);
  writer.text(event.message);
  return success(writer.take());
}

EncodedEvent encodeStatusJson(const StatusJsonEvent &event,
                              const ProtocolLimits &limits) {
  if (auto issue = validateStatusJson(event, limits))
    return failure<std::vector<uint8_t>>(std::move(*issue));
  Writer writer = frameWriter(FrameType::StatusJson,
                              kStatusJsonFixedBytes + event.json.size());
  writer.u64(event.correlationId);
  writer.text(event.json);
  return success(writer.take());
}

ProtocolResult<ClientMessage> decodeRequest(std::vector<uint8_t> &&payload,
                                            const ProtocolLimits &limits) {
  Reader reader(payload);
  RequestFrame request;
  uint8_t priority = 0;
  uint8_t constraint = 0;
  uint32_t promptCount = 0;
  uint32_t imageSpanCount = 0;
  uint32_t scoreCount = 0;
  if (!reader.u64(request.requestId) || !reader.u8(priority) ||
      !reader.u8(constraint) ||
      !reader.u64(request.absoluteDeadlineUnixMicros) ||
      !reader.u64(request.remainingDeadlineMicros) ||
      !reader.u32(request.logicalMaxOutputTokens) || !reader.u32(promptCount) ||
      !reader.u32(imageSpanCount) ||
      !reader.f32(request.sampling.temperature) ||
      !reader.f32(request.sampling.topP) ||
      !reader.u32(request.sampling.topK) ||
      !reader.f32(request.sampling.presencePenalty) ||
      !reader.f32(request.sampling.frequencyPenalty) ||
      !reader.f32(request.sampling.repetitionPenalty) ||
      !reader.f32(request.sampling.minP) ||
      !reader.u64(request.sampling.seed) || !reader.u32(scoreCount) ||
      !reader.u32(request.generationPromptTokens) ||
      !reader.u32(request.flags)) {
    return failure<ClientMessage>(
        makeIssue(FailureClass::ProtocolFatal, IssueCode::InvalidPayloadLength,
                  0, "request fixed payload is truncated"));
  }
  request.priority = static_cast<RequestPriority>(priority);
  request.constraint = static_cast<ConstraintMode>(constraint);
  if (scoreCount > ExecutionLimits::maximumScoreOptions) {
    return failure<ClientMessage>(
        requestIssue(IssueCode::InvalidCount, request.requestId,
                     "score option count exceeds its limit"));
  }
  if (promptCount > limits.maxPromptTokens) {
    return failure<ClientMessage>(
        requestIssue(IssueCode::LimitExceeded, request.requestId,
                     "prompt token count exceeds its limit"));
  }
  if (imageSpanCount > limits.maxImageSpans) {
    return failure<ClientMessage>(
        requestIssue(IssueCode::LimitExceeded, request.requestId,
                     "image span count exceeds its limit"));
  }
  if (!reader.words(promptCount, request.promptTokens)) {
    return failure<ClientMessage>(
        requestIssue(IssueCode::InvalidPayloadLength, request.requestId,
                     "prompt count does not match the binary token payload"));
  }
  request.imageSpans.resize(imageSpanCount);
  uint64_t pixelBytes = 0;
  for (ImageSpan &span : request.imageSpans) {
    if (!reader.u32(span.offset) || !reader.u32(span.tokens) ||
        !reader.u32(span.gridHeight) || !reader.u32(span.gridWidth) ||
        !reader.u64(span.digestLo) || !reader.u64(span.digestHi) ||
        !checkedAdd(pixelBytes, span.pixelBytes(), pixelBytes)) {
      return failure<ClientMessage>(
          requestIssue(IssueCode::InvalidPayloadLength, request.requestId,
                       "image span payload is malformed"));
    }
  }
  const size_t pixelsOffset = payload.size() - reader.remaining();
  uint64_t scoreBytes = 0;
  if (!checkedMultiply(scoreCount, sizeof(uint32_t), scoreBytes) ||
      reader.remaining() != pixelBytes + scoreBytes ||
      !reader.skip(pixelBytes) ||
      !reader.words(scoreCount, request.scoreTokens)) {
    return failure<ClientMessage>(
        requestIssue(IssueCode::InvalidPayloadLength, request.requestId,
                     "image pixel or score payload does not match its counts"));
  }
  // The pixels are most of a large frame, so they stay in its buffer: the
  // score words behind them are read, and moving the pixels to its front
  // allocates nothing.
  if (pixelBytes) {
    payload.resize(pixelsOffset + pixelBytes);
    payload.erase(payload.begin(), payload.begin() + pixelsOffset);
    request.imagePixels = std::move(payload);
  }
  if (auto issue = validateRequest(request, limits)) {
    return failure<ClientMessage>(std::move(*issue));
  }
  return success(ClientMessage{std::move(request)});
}

ProtocolResult<ClientMessage> decodeCancel(const Frame &frame) {
  Reader reader(frame.payload);
  CancelFrame cancel;
  if (!reader.u64(cancel.requestId) || reader.remaining()) {
    return failure<ClientMessage>(
        makeIssue(FailureClass::ProtocolFatal, IssueCode::InvalidPayloadLength,
                  0, "cancel payload has an invalid length"));
  }
  if (auto issue = validateCancel(cancel)) {
    return failure<ClientMessage>(std::move(*issue));
  }
  return success(ClientMessage{cancel});
}

ProtocolResult<ClientMessage> decodeMaskResponse(const Frame &frame,
                                                 const ProtocolLimits &limits) {
  Reader reader(frame.payload);
  MaskResponseFrame response;
  uint32_t wordCount = 0;
  if (!reader.u64(response.requestId) || !reader.u64(response.maskRequestId) ||
      !reader.u32(wordCount)) {
    return failure<ClientMessage>(
        makeIssue(FailureClass::ProtocolFatal, IssueCode::InvalidPayloadLength,
                  0, "mask response fixed payload is truncated"));
  }
  if (wordCount > limits.maxMaskWords) {
    return failure<ClientMessage>(
        requestIssue(IssueCode::LimitExceeded, response.requestId,
                     "mask response word count exceeds its limit"));
  }
  uint64_t expectedBytes = 0;
  if (!checkedMultiply(wordCount, sizeof(uint32_t), expectedBytes) ||
      reader.remaining() != expectedBytes ||
      !reader.words(wordCount, response.maskWords)) {
    return failure<ClientMessage>(
        requestIssue(IssueCode::InvalidPayloadLength, response.requestId,
                     "mask word count does not match the binary payload"));
  }
  if (auto issue = validateMaskResponse(response, limits)) {
    return failure<ClientMessage>(std::move(*issue));
  }
  return success(ClientMessage{std::move(response)});
}

ProtocolResult<ClientMessage> decodeStatusRequest(const Frame &frame) {
  Reader reader(frame.payload);
  StatusRequestFrame request;
  if (!reader.u64(request.correlationId) || reader.remaining()) {
    return failure<ClientMessage>(
        makeIssue(FailureClass::ProtocolFatal, IssueCode::InvalidPayloadLength,
                  0, "status request payload has an invalid length"));
  }
  return success(ClientMessage{request});
}

std::string_view frameTypeName(FrameType type) {
  switch (type) {
  case FrameType::Request:
    return "request";
  case FrameType::Cancel:
    return "cancel";
  case FrameType::MaskResponse:
    return "mask_response";
  case FrameType::StatusRequest:
    return "status_request";
  case FrameType::Ready:
    return "ready";
  case FrameType::PromptProgress:
    return "prompt_progress";
  case FrameType::Start:
    return "start";
  case FrameType::Tokens:
    return "tokens";
  case FrameType::MaskRequest:
    return "mask_request";
  case FrameType::Done:
    return "done";
  case FrameType::Error:
    return "error";
  case FrameType::StatusJson:
    return "status_json";
  }
  return "unknown";
}

std::string_view failureClassName(FailureClass failureClass) {
  switch (failureClass) {
  case FailureClass::RequestError:
    return "request_error";
  case FailureClass::EngineUnhealthy:
    return "engine_unhealthy";
  case FailureClass::ProtocolFatal:
    return "protocol_fatal";
  }
  return "unknown";
}

} // namespace

std::optional<ProtocolIssue> validateLimits(const ProtocolLimits &limits) {
  if (limits.maxFramePayloadBytes < kRequestFixedBytes ||
      limits.maxFramePayloadBytes > kAbsoluteMaxFramePayloadBytes) {
    return makeIssue(FailureClass::ProtocolFatal, IssueCode::LimitExceeded, 0,
                     "maxFramePayloadBytes must be in [" +
                         std::to_string(kRequestFixedBytes) + ", 256 MiB]");
  }
  if (limits.maxStatusJsonBytes >
      limits.maxFramePayloadBytes - kStatusJsonFixedBytes) {
    return makeIssue(
        FailureClass::ProtocolFatal, IssueCode::LimitExceeded, 0,
        "status JSON limit does not fit the configured frame limit");
  }
  if (limits.maxErrorStringBytes > limits.maxFramePayloadBytes) {
    return makeIssue(FailureClass::ProtocolFatal, IssueCode::LimitExceeded, 0,
                     "error string limit exceeds the configured frame limit");
  }
  if (!limits.maxPromptTokens || !limits.maxLogicalOutputTokens ||
      !limits.maxTokenBatch || !limits.maxSimulationTokens ||
      !limits.maxMaskWords || !limits.maxImageSpans) {
    return makeIssue(FailureClass::ProtocolFatal, IssueCode::LimitExceeded, 0,
                     "all configured token, mask, and image limits must be "
                     "non-zero");
  }
  return std::nullopt;
}

bool connectionMustClose(FailureClass failureClass) {
  return failureClass != FailureClass::RequestError;
}

std::string_view issueCodeName(IssueCode code) {
  switch (code) {
  case IssueCode::None:
    return "none";
  case IssueCode::BadMagic:
    return "bad_magic";
  case IssueCode::UnsupportedVersion:
    return "unsupported_version";
  case IssueCode::UnknownFrameType:
    return "unknown_frame_type";
  case IssueCode::FrameTooLarge:
    return "frame_too_large";
  case IssueCode::InvalidPayloadLength:
    return "invalid_payload_length";
  case IssueCode::TruncatedFrame:
    return "truncated_frame";
  case IssueCode::InvalidRequestId:
    return "invalid_request_id";
  case IssueCode::InvalidEnumValue:
    return "invalid_enum_value";
  case IssueCode::InvalidDeadline:
    return "invalid_deadline";
  case IssueCode::InvalidSampling:
    return "invalid_sampling";
  case IssueCode::InvalidCount:
    return "invalid_count";
  case IssueCode::InvalidConstraint:
    return "invalid_constraint";
  case IssueCode::InvalidErrorClassification:
    return "invalid_error_classification";
  case IssueCode::LimitExceeded:
    return "limit_exceeded";
  case IssueCode::IntegerOverflow:
    return "integer_overflow";
  case IssueCode::AllocationFailure:
    return "allocation_failure";
  }
  return "unknown";
}

std::string ProtocolIssue::describe() const {
  std::ostringstream out;
  out << failureClassName(failureClass) << ':' << issueCodeName(code);
  if (requestId)
    out << " request=" << requestId;
  if (!message.empty())
    out << ": " << message;
  return out.str();
}

ProtocolResult<ClientMessage> decodeFrame(Frame &&frame,
                                          const ProtocolLimits &limits) {
  if (auto issue =
          validatePayloadLength(frame.type, frame.payload.size(), limits)) {
    return failure<ClientMessage>(std::move(*issue));
  }
  try {
    switch (frame.type) {
    case FrameType::Request:
      return decodeRequest(std::move(frame.payload), limits);
    case FrameType::Cancel:
      return decodeCancel(frame);
    case FrameType::MaskResponse:
      return decodeMaskResponse(frame, limits);
    case FrameType::StatusRequest:
      return decodeStatusRequest(frame);
    default:
      break;
    }
  } catch (const std::bad_alloc &) {
    return failure<ClientMessage>(
        makeIssue(FailureClass::EngineUnhealthy, IssueCode::AllocationFailure,
                  0, "allocation failed while decoding protocol message"));
  } catch (const std::length_error &) {
    return failure<ClientMessage>(makeIssue(
        FailureClass::EngineUnhealthy, IssueCode::AllocationFailure, 0,
        "container length failed while decoding protocol message"));
  }
  return failure<ClientMessage>(
      makeIssue(FailureClass::ProtocolFatal, IssueCode::UnknownFrameType, 0,
                "frame type is not a native protocol client frame"));
}

ProtocolResult<std::vector<uint8_t>>
serializeEvent(const EngineEvent &event, const ProtocolLimits &limits) {
  try {
    return std::visit(
        [&](const auto &value) -> EncodedEvent {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, ReadyEvent>) {
            return encodeReady(value);
          } else if constexpr (std::is_same_v<T, StartEvent>) {
            return encodeStart(value);
          } else if constexpr (std::is_same_v<T, PromptProgressEvent>) {
            return encodePromptProgress(value, limits);
          } else if constexpr (std::is_same_v<T, TokensEvent>) {
            return encodeTokens(value, limits);
          } else if constexpr (std::is_same_v<T, MaskRequestEvent>) {
            return encodeMaskRequest(value, limits);
          } else if constexpr (std::is_same_v<T, DoneEvent>) {
            return encodeDone(value);
          } else if constexpr (std::is_same_v<T, ErrorEvent>) {
            return encodeError(value, limits);
          } else {
            return encodeStatusJson(value, limits);
          }
        },
        event);
  } catch (const std::bad_alloc &) {
    return failure<std::vector<uint8_t>>(
        makeIssue(FailureClass::EngineUnhealthy, IssueCode::AllocationFailure,
                  0, "allocation failed while serializing protocol frame"));
  } catch (const std::length_error &) {
    return failure<std::vector<uint8_t>>(makeIssue(
        FailureClass::EngineUnhealthy, IssueCode::AllocationFailure, 0,
        "container length failed while serializing protocol frame"));
  }
}

FrameParser::FrameParser(ProtocolLimits limits) : limits_(limits) {}

std::optional<ProtocolIssue> FrameParser::parseHeader() {
  if (!std::equal(kMagic.begin(), kMagic.end(), header_.begin())) {
    return makeIssue(FailureClass::ProtocolFatal, IssueCode::BadMagic, 0,
                     "frame magic is not SPLH");
  }
  uint16_t version = loadU16(header_.data() + 4);
  if (version != kProtocolVersion) {
    return makeIssue(FailureClass::ProtocolFatal, IssueCode::UnsupportedVersion,
                     0, "unsupported native protocol version");
  }
  uint16_t rawType = loadU16(header_.data() + 6);
  if (!clientFrameType(rawType, currentType_)) {
    return makeIssue(FailureClass::ProtocolFatal, IssueCode::UnknownFrameType,
                     0, "frame type is not a native protocol client frame");
  }
  expectedPayloadBytes_ = loadU64(header_.data() + 8);
  if (auto issue =
          validatePayloadLength(currentType_, expectedPayloadBytes_, limits_)) {
    return issue;
  }
  payload_.clear();
  readingPayload_ = true;
  return std::nullopt;
}

ParseStep FrameParser::fail(size_t consumed, ProtocolIssue issue) {
  terminalIssue_ = std::move(issue);
  return {consumed, std::nullopt, terminalIssue_};
}

void FrameParser::resetCurrentFrame() {
  header_.fill(0);
  headerBytes_ = 0;
  readingPayload_ = false;
  expectedPayloadBytes_ = 0;
  payload_.clear();
}

ParseStep FrameParser::consume(std::span<const uint8_t> bytes) {
  if (terminalIssue_)
    throw std::logic_error("native frame parser used after it failed");

  size_t consumed = 0;
  while (consumed < bytes.size()) {
    if (!readingPayload_) {
      size_t count =
          std::min(kFrameHeaderBytes - headerBytes_, bytes.size() - consumed);
      std::memcpy(header_.data() + headerBytes_, bytes.data() + consumed,
                  count);
      headerBytes_ += count;
      consumed += count;
      if (headerBytes_ < kFrameHeaderBytes) {
        return {consumed, std::nullopt, std::nullopt};
      }
      if (auto issue = parseHeader()) {
        return fail(consumed, std::move(*issue));
      }
    }

    size_t needed =
        static_cast<size_t>(expectedPayloadBytes_) - payload_.size();
    size_t count = std::min(needed, bytes.size() - consumed);
    try {
      // The validated frame length is known. Reserve it once to avoid
      // geometric growth copies; each received byte is copied in once.
      if (payload_.capacity() < expectedPayloadBytes_)
        payload_.reserve(static_cast<size_t>(expectedPayloadBytes_));
      const auto received = bytes.subspan(consumed, count);
      payload_.insert(payload_.end(), received.begin(), received.end());
    } catch (const std::bad_alloc &) {
      return fail(consumed,
                  makeIssue(FailureClass::EngineUnhealthy,
                            IssueCode::AllocationFailure, 0,
                            "allocation failed while receiving frame payload"));
    } catch (const std::length_error &) {
      return fail(consumed,
                  makeIssue(FailureClass::EngineUnhealthy,
                            IssueCode::AllocationFailure, 0,
                            "frame payload length is not allocatable"));
    }
    consumed += count;
    if (payload_.size() == expectedPayloadBytes_) {
      Frame frame{currentType_, std::move(payload_)};
      resetCurrentFrame();
      return {consumed, std::move(frame), std::nullopt};
    }
  }
  return {consumed, std::nullopt, std::nullopt};
}

std::optional<ProtocolIssue> FrameParser::finish() {
  if (terminalIssue_)
    throw std::logic_error("native frame parser used after it failed");
  if (!headerBytes_ && !readingPayload_)
    return std::nullopt;

  std::ostringstream message;
  if (!readingPayload_) {
    message << "stream ended after " << headerBytes_ << " of "
            << kFrameHeaderBytes << " frame-header bytes";
  } else {
    message << "stream ended after " << payload_.size() << " of "
            << expectedPayloadBytes_ << ' ' << frameTypeName(currentType_)
            << " payload bytes";
  }
  terminalIssue_ = makeIssue(FailureClass::ProtocolFatal,
                             IssueCode::TruncatedFrame, 0, message.str());
  return terminalIssue_;
}

} // namespace splash::protocol

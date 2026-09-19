#include "engine/Protocol.hpp"
#include "engine/Checked.hpp"

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

std::string_view frameTypeName(FrameType type);
std::string_view failureClassName(FailureClass failureClass);

constexpr std::array<uint8_t, 4> kMagic{'S', 'P', 'L', 'H'};
constexpr uint64_t kRequestFixedBytes = 64;
constexpr uint64_t kImageSpanBytes = 32;
constexpr uint64_t kCancelFixedBytes = 8;
constexpr uint64_t kMaskResponseFixedBytes = 20;
constexpr uint64_t kStatusRequestFixedBytes = 8;
constexpr uint64_t kReadyFixedBytes = 24;
constexpr uint64_t kStartFixedBytes = 21;
constexpr uint64_t kPromptProgressFixedBytes = 20;
constexpr uint64_t kTokensFixedBytes = 16;
constexpr uint64_t kMaskRequestFixedBytes = 24;
constexpr uint64_t kDoneFixedBytes = 45;
constexpr uint64_t kErrorFixedBytes = 18;
constexpr uint64_t kCapacityExhaustedFixedBytes = 24;
constexpr uint64_t kStatusJsonFixedBytes = 12;

struct PayloadBounds {
  uint64_t minimum = 0;
  uint64_t maximum = 0;
};

ProtocolIssue makeIssue(FailureClass failureClass, IssueCode code,
                        uint64_t requestId, std::string message) {
  return {failureClass, code, requestId, std::move(message)};
}

template <typename T> ProtocolResult<T> success(T value) {
  return {std::move(value), std::nullopt};
}

template <typename T> ProtocolResult<T> failure(ProtocolIssue issue) {
  return {std::nullopt, std::move(issue)};
}

std::optional<ProtocolIssue> validateLimits(const ProtocolLimits &limits) {
  if (limits.maxFramePayloadBytes < kRequestFixedBytes ||
      limits.maxFramePayloadBytes > kAbsoluteMaxFramePayloadBytes) {
    return makeIssue(FailureClass::ProtocolFatal, IssueCode::LimitExceeded, 0,
                     "maxFramePayloadBytes must be in [64, 256 MiB]");
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
      !limits.maxMaskWords || !limits.maxImageSpans ||
      !limits.maxImagePatches) {
    return makeIssue(FailureClass::ProtocolFatal, IssueCode::LimitExceeded, 0,
                     "all configured token, mask, and image limits must be "
                     "non-zero");
  }
  return std::nullopt;
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
  case FrameType::Ready:
    return bounded(kReadyFixedBytes, kReadyFixedBytes);
  case FrameType::PromptProgress:
    return bounded(kPromptProgressFixedBytes, kPromptProgressFixedBytes);
  case FrameType::Start:
    return bounded(kStartFixedBytes, kStartFixedBytes);
  case FrameType::Tokens:
    if (!checkedMultiply(limits.maxTokenBatch, sizeof(uint32_t), variable) ||
        !checkedAdd(kTokensFixedBytes, variable, maximum)) {
      return std::nullopt;
    }
    return bounded(kTokensFixedBytes, maximum);
  case FrameType::MaskRequest:
    if (!checkedMultiply(limits.maxSimulationTokens, sizeof(uint32_t),
                         variable) ||
        !checkedAdd(kMaskRequestFixedBytes, variable, maximum)) {
      return std::nullopt;
    }
    return bounded(kMaskRequestFixedBytes, maximum);
  case FrameType::Done:
    if (!checkedMultiply(kMaximumScoreOptions, sizeof(float), variable) ||
        !checkedAdd(kDoneFixedBytes, variable, maximum)) {
      return std::nullopt;
    }
    return bounded(kDoneFixedBytes, maximum);
  case FrameType::Error:
    if (!checkedMultiply(limits.maxErrorStringBytes, 2, variable) ||
        !checkedAdd(kErrorFixedBytes, variable, maximum)) {
      return std::nullopt;
    }
    return bounded(kErrorFixedBytes, maximum);
  case FrameType::CapacityExhausted:
    return bounded(kCapacityExhaustedFixedBytes, kCapacityExhaustedFixedBytes);
  case FrameType::StatusJson:
    if (!checkedAdd(kStatusJsonFixedBytes, limits.maxStatusJsonBytes,
                    maximum)) {
      return std::nullopt;
    }
    return bounded(kStatusJsonFixedBytes, maximum);
  }
  return std::nullopt;
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

bool validFrameType(uint16_t raw, FrameType &type) {
  switch (static_cast<FrameType>(raw)) {
  case FrameType::Request:
  case FrameType::Cancel:
  case FrameType::MaskResponse:
  case FrameType::StatusRequest:
  case FrameType::Ready:
  case FrameType::Start:
  case FrameType::PromptProgress:
  case FrameType::Tokens:
  case FrameType::MaskRequest:
  case FrameType::Done:
  case FrameType::Error:
  case FrameType::CapacityExhausted:
  case FrameType::StatusJson:
    type = static_cast<FrameType>(raw);
    return true;
  }
  return false;
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
  explicit Writer(size_t reserveBytes = 0) { bytes_.reserve(reserveBytes); }

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

  bool bytes(uint64_t count, std::vector<uint8_t> &values) {
    if (count > remaining())
      return false;
    values.assign(bytes_.begin() + offset_, bytes_.begin() + offset_ + count);
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

  bool text(uint32_t count, std::string &value) {
    if (count > remaining())
      return false;
    value.assign(reinterpret_cast<const char *>(bytes_.data() + offset_),
                 count);
    offset_ += count;
    return true;
  }

  bool remainingText(std::string &value) {
    if (remaining() > std::numeric_limits<uint32_t>::max())
      return false;
    return text(static_cast<uint32_t>(remaining()), value);
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
    return std::optional<ProtocolIssue>(makeIssue(FailureClass::RequestError,
                                                  code, request.requestId,
                                                  std::move(message)));
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
  if (!validEnum(static_cast<uint8_t>(request.cohort),
                 {Cohort::Greedy, Cohort::Sampling, Cohort::Constrained})) {
    return invalid(IssueCode::InvalidEnumValue,
                   "cohort is not defined by native protocol");
  }
  if (!validEnum(static_cast<uint8_t>(request.constraint),
                 {ConstraintMode::None, ConstraintMode::TokenMask})) {
    return invalid(IssueCode::InvalidEnumValue,
                   "constraint mode is not defined by native protocol");
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
  uint64_t previousSpanEnd = 0;
  uint64_t pixelBytes = 0;
  for (const ImageSpanFrame &span : request.imageSpans) {
    const uint64_t patches = uint64_t{span.gridHeight} * span.gridWidth;
    if (span.gridHeight < 2 || span.gridWidth < 2 || span.gridHeight % 2 ||
        span.gridWidth % 2 || patches > limits.maxImagePatches) {
      return invalid(IssueCode::InvalidCount,
                     "image grid must be even-sided and within the patch "
                     "limit");
    }
    if (span.tokens != (span.gridHeight / 2) * (span.gridWidth / 2)) {
      return invalid(IssueCode::InvalidCount,
                     "image span tokens must equal the merged grid size");
    }
    const uint64_t end = uint64_t{span.offset} + span.tokens;
    if (span.offset < previousSpanEnd || end > request.promptTokens.size()) {
      return invalid(IssueCode::InvalidCount,
                     "image spans must be sorted, non-overlapping runs inside "
                     "the prompt");
    }
    previousSpanEnd = end;
    pixelBytes += span.pixelBytes();
  }
  if (request.imagePixels.size() != pixelBytes) {
    return invalid(IssueCode::InvalidCount,
                   "image pixels do not match the image grids");
  }
  if (scoring) {
    if (!request.imageSpans.empty()) {
      return invalid(IssueCode::InvalidCount,
                     "score requests are text-only");
    }
    if (request.scoreTokens.size() < kMinimumScoreOptions ||
        request.scoreTokens.size() > kMaximumScoreOptions) {
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
  const SamplingParameters &sampling = request.sampling;
  if (!std::isfinite(sampling.temperature) || sampling.temperature < 0.0f ||
      !std::isfinite(sampling.topP) || sampling.topP <= 0.0f ||
      sampling.topP > 1.0f || sampling.topK > 32 ||
      (sampling.temperature > 0.0f && !sampling.topK)) {
    return invalid(IssueCode::InvalidSampling,
                   "sampling requires temperature>=0, top_p in (0,1], and "
                   "top_k in [1,32] when sampling is enabled");
  }
  Cohort expected = Cohort::Constrained;
  if (request.constraint == ConstraintMode::None) {
    expected =
        request.sampling.temperature > 0.0f ? Cohort::Sampling : Cohort::Greedy;
  }
  if (request.cohort != expected) {
    return invalid(IssueCode::InvalidCohortConstraint,
                   "cohort does not match sampling and constraint semantics");
  }
  if (scoring) {
    if (request.constraint != ConstraintMode::None) {
      return invalid(IssueCode::InvalidCohortConstraint,
                     "score requests cannot carry a constraint");
    }
    if (sampling.temperature != 0.0f || sampling.topP != 1.0f ||
        sampling.topK != 0) {
      return invalid(IssueCode::InvalidSampling,
                     "score requests require greedy default sampling");
    }
  }
  return std::nullopt;
}

std::optional<ProtocolIssue> validateCancel(const CancelFrame &cancel) {
  if (cancel.requestId)
    return std::nullopt;
  return makeIssue(FailureClass::RequestError, IssueCode::InvalidRequestId, 0,
                   "cancel request id must be non-zero");
}

std::optional<ProtocolIssue>
validateMaskResponse(const MaskResponseFrame &response,
                     const ProtocolLimits &limits) {
  if (!response.requestId || !response.maskRequestId) {
    return makeIssue(FailureClass::RequestError, IssueCode::InvalidRequestId,
                     response.requestId,
                     "mask response request ids must be non-zero");
  }
  if (response.maskWords.empty() ||
      response.maskWords.size() > limits.maxMaskWords) {
    return makeIssue(FailureClass::RequestError, IssueCode::LimitExceeded,
                     response.requestId,
                     "mask response word count exceeds its limit");
  }
  return std::nullopt;
}

std::optional<ProtocolIssue> validateReady(const ReadyEvent &event,
                                           FailureClass failureClass) {
  if (event.engineInstanceId && event.maxConcurrentRequests &&
      event.maxContextTokens) {
    return std::nullopt;
  }
  return makeIssue(failureClass, IssueCode::InvalidCount, 0,
                   "ready event capacities and instance id must be non-zero");
}

std::optional<ProtocolIssue> validateStart(const StartEvent &event,
                                           FailureClass failureClass) {
  if (!event.requestId) {
    return makeIssue(failureClass, IssueCode::InvalidRequestId, 0,
                     "start event request id must be non-zero");
  }
  if (!validEnum(static_cast<uint8_t>(event.cacheDisposition),
                 {CacheDisposition::Miss, CacheDisposition::PrefixHit})) {
    return makeIssue(failureClass, IssueCode::InvalidEnumValue, event.requestId,
                     "start cache disposition is invalid");
  }
  if (event.slotIndex < -1 || !event.capacityTokens ||
      event.matchedPromptTokens > event.capacityTokens) {
    return makeIssue(failureClass, IssueCode::InvalidCount, event.requestId,
                     "start event capacity or slot is invalid");
  }
  return std::nullopt;
}

std::optional<ProtocolIssue> validateTokens(const TokensEvent &event,
                                            const ProtocolLimits &limits,
                                            FailureClass failureClass) {
  if (!event.requestId) {
    return makeIssue(failureClass, IssueCode::InvalidRequestId, 0,
                     "tokens event request id must be non-zero");
  }
  if (event.tokens.empty() || event.tokens.size() > limits.maxTokenBatch) {
    return makeIssue(failureClass, IssueCode::LimitExceeded, event.requestId,
                     "tokens event batch size exceeds its limit");
  }
  uint64_t end = 0;
  if (!checkedAdd(event.sequenceOffset, event.tokens.size(), end) ||
      end > std::numeric_limits<uint32_t>::max()) {
    return makeIssue(failureClass, IssueCode::IntegerOverflow, event.requestId,
                     "tokens event sequence range overflows uint32");
  }
  return std::nullopt;
}

std::optional<ProtocolIssue> validateMaskRequest(const MaskRequestEvent &event,
                                                 const ProtocolLimits &limits,
                                                 FailureClass failureClass) {
  if (!event.requestId || !event.maskRequestId) {
    return makeIssue(failureClass, IssueCode::InvalidRequestId, event.requestId,
                     "mask request ids must be non-zero");
  }
  // An empty simulation sequence is the initial-anchor request and
  // deliberately produces one mask row. During speculative verification
  // the sequence is [pending_anchor, draft...], so len+1 rows represent
  // before-anchor and after each simulated token.
  if (!event.wordsPerMask ||
      event.simulationTokens.size() > limits.maxSimulationTokens) {
    return makeIssue(failureClass, IssueCode::InvalidCount, event.requestId,
                     "mask request dimensions are invalid");
  }
  uint64_t maskRows = 0;
  uint64_t totalWords = 0;
  if (!checkedAdd(event.simulationTokens.size(), 1, maskRows) ||
      !checkedMultiply(event.wordsPerMask, maskRows, totalWords) ||
      totalWords > limits.maxMaskWords) {
    return makeIssue(failureClass, IssueCode::LimitExceeded, event.requestId,
                     "mask request output would exceed the mask limit");
  }
  return std::nullopt;
}

std::optional<ProtocolIssue> validateDone(const DoneEvent &event,
                                          FailureClass failureClass) {
  if (!event.requestId) {
    return makeIssue(failureClass, IssueCode::InvalidRequestId, 0,
                     "done event request id must be non-zero");
  }
  if (!validEnum(static_cast<uint8_t>(event.reason),
                 {FinishReason::Stop, FinishReason::Length,
                  FinishReason::Cancelled})) {
    return makeIssue(failureClass, IssueCode::InvalidEnumValue, event.requestId,
                     "done finish reason is invalid");
  }
  if (!event.optionLogits.empty()) {
    if (event.optionLogits.size() < kMinimumScoreOptions ||
        event.optionLogits.size() > kMaximumScoreOptions) {
      return makeIssue(failureClass, IssueCode::InvalidCount, event.requestId,
                       "done option logit count must be in [2, 255]");
    }
    if (event.reason != FinishReason::Stop || event.completionTokens != 0 ||
        event.decodeMicros != 0) {
      return makeIssue(failureClass, IssueCode::InvalidCount, event.requestId,
                       "scored done events must stop and carry no completion "
                       "or decode activity");
    }
    for (float logit : event.optionLogits) {
      if (!std::isfinite(logit)) {
        return makeIssue(failureClass, IssueCode::InvalidCount, event.requestId,
                         "done option logits must be finite");
      }
    }
  }
  return std::nullopt;
}

std::optional<ProtocolIssue> validateError(const ErrorEvent &event,
                                           const ProtocolLimits &limits,
                                           FailureClass invalidEventClass) {
  if (!validEnum(static_cast<uint8_t>(event.failureClass),
                 {FailureClass::RequestError, FailureClass::EngineUnhealthy,
                  FailureClass::ProtocolFatal})) {
    return makeIssue(invalidEventClass, IssueCode::InvalidErrorClassification,
                     event.requestId, "error event classification is invalid");
  }
  if ((event.failureClass == FailureClass::RequestError && !event.requestId) ||
      (event.failureClass != FailureClass::RequestError && event.requestId)) {
    return makeIssue(invalidEventClass, IssueCode::InvalidErrorClassification,
                     event.requestId,
                     "only request errors may carry a non-zero request id");
  }
  if (event.code.empty() || event.code.size() > limits.maxErrorStringBytes ||
      event.message.size() > limits.maxErrorStringBytes) {
    return makeIssue(invalidEventClass, IssueCode::LimitExceeded,
                     event.requestId,
                     "error code or message exceeds its safe limit");
  }
  return std::nullopt;
}

std::optional<ProtocolIssue>
validateCapacityExhausted(const CapacityExhaustedEvent &event,
                          FailureClass failureClass) {
  if (!event.requestId) {
    return makeIssue(failureClass, IssueCode::InvalidRequestId, 0,
                     "capacity event request id must be non-zero");
  }
  if (!event.requiredKvPages) {
    return makeIssue(failureClass, IssueCode::InvalidCount, event.requestId,
                     "capacity event required page count must be non-zero");
  }
  return std::nullopt;
}

std::optional<ProtocolIssue> validateStatusJson(const StatusJsonEvent &event,
                                                const ProtocolLimits &limits,
                                                FailureClass failureClass) {
  if (event.schemaVersion != kStatusSchemaVersion) {
    return makeIssue(
        failureClass, IssueCode::InvalidStatusSchema, 0,
        "status JSON schema version must match the native protocol");
  }
  if (event.json.empty() || event.json.size() > limits.maxStatusJsonBytes) {
    return makeIssue(failureClass, IssueCode::LimitExceeded, 0,
                     "status JSON byte count exceeds its safe limit");
  }
  return std::nullopt;
}

ProtocolResult<Frame> encodeRequest(const RequestFrame &request,
                                    const ProtocolLimits &limits) {
  if (auto issue = validateRequest(request, limits)) {
    return failure<Frame>(std::move(*issue));
  }
  uint64_t payloadBytes = 0;
  uint64_t tokenBytes = 0;
  uint64_t spanBytes = 0;
  uint64_t scoreBytes = 0;
  if (!checkedMultiply(request.promptTokens.size(), sizeof(uint32_t),
                       tokenBytes) ||
      !checkedMultiply(request.imageSpans.size(), kImageSpanBytes, spanBytes) ||
      !checkedMultiply(request.scoreTokens.size(), sizeof(uint32_t),
                       scoreBytes) ||
      !checkedAdd(kRequestFixedBytes, tokenBytes, payloadBytes) ||
      !checkedAdd(payloadBytes, spanBytes, payloadBytes) ||
      !checkedAdd(payloadBytes, request.imagePixels.size(), payloadBytes) ||
      !checkedAdd(payloadBytes, scoreBytes, payloadBytes) ||
      payloadBytes > std::numeric_limits<size_t>::max()) {
    return failure<Frame>(
        makeIssue(FailureClass::RequestError, IssueCode::IntegerOverflow,
                  request.requestId, "request payload size overflows size_t"));
  }
  Writer writer(static_cast<size_t>(payloadBytes));
  writer.u64(request.requestId);
  writer.u8(static_cast<uint8_t>(request.priority));
  writer.u8(static_cast<uint8_t>(request.cohort));
  writer.u8(static_cast<uint8_t>(request.constraint));
  writer.u64(request.absoluteDeadlineUnixMicros);
  writer.u64(request.remainingDeadlineMicros);
  writer.u32(request.logicalMaxOutputTokens);
  writer.u32(static_cast<uint32_t>(request.promptTokens.size()));
  writer.u32(static_cast<uint32_t>(request.imageSpans.size()));
  writer.f32(request.sampling.temperature);
  writer.f32(request.sampling.topP);
  writer.u32(request.sampling.topK);
  writer.u64(request.seed);
  writer.u8(request.returnProgress);
  writer.u32(static_cast<uint32_t>(request.scoreTokens.size()));
  for (uint32_t token : request.promptTokens)
    writer.u32(token);
  for (const ImageSpanFrame &span : request.imageSpans) {
    writer.u32(span.offset);
    writer.u32(span.tokens);
    writer.u32(span.gridHeight);
    writer.u32(span.gridWidth);
    writer.u64(span.digestLo);
    writer.u64(span.digestHi);
  }
  writer.raw(request.imagePixels);
  for (uint32_t token : request.scoreTokens)
    writer.u32(token);
  return success(Frame{FrameType::Request, writer.take()});
}

ProtocolResult<Frame> encodeCancel(const CancelFrame &cancel) {
  if (auto issue = validateCancel(cancel)) {
    return failure<Frame>(std::move(*issue));
  }
  Writer writer(kCancelFixedBytes);
  writer.u64(cancel.requestId);
  return success(Frame{FrameType::Cancel, writer.take()});
}

ProtocolResult<Frame> encodeMaskResponse(const MaskResponseFrame &response,
                                         const ProtocolLimits &limits) {
  if (auto issue = validateMaskResponse(response, limits)) {
    return failure<Frame>(std::move(*issue));
  }
  Writer writer(kMaskResponseFixedBytes +
                response.maskWords.size() * sizeof(uint32_t));
  writer.u64(response.requestId);
  writer.u64(response.maskRequestId);
  writer.u32(static_cast<uint32_t>(response.maskWords.size()));
  for (uint32_t word : response.maskWords)
    writer.u32(word);
  return success(Frame{FrameType::MaskResponse, writer.take()});
}

ProtocolResult<Frame> encodeStatusRequest(const StatusRequestFrame &request) {
  Writer writer(kStatusRequestFixedBytes);
  writer.u64(request.correlationId);
  return success(Frame{FrameType::StatusRequest, writer.take()});
}

ProtocolResult<Frame> encodeReady(const ReadyEvent &event) {
  if (auto issue = validateReady(event, FailureClass::EngineUnhealthy)) {
    return failure<Frame>(std::move(*issue));
  }
  Writer writer(kReadyFixedBytes);
  writer.u64(event.engineInstanceId);
  writer.u32(event.maxConcurrentRequests);
  writer.u32(event.maxContextTokens);
  writer.u64(event.featureBits);
  return success(Frame{FrameType::Ready, writer.take()});
}

ProtocolResult<Frame> encodeStart(const StartEvent &event) {
  if (auto issue = validateStart(event, FailureClass::EngineUnhealthy)) {
    return failure<Frame>(std::move(*issue));
  }
  Writer writer(kStartFixedBytes);
  writer.u64(event.requestId);
  writer.u8(static_cast<uint8_t>(event.cacheDisposition));
  writer.u32(static_cast<uint32_t>(event.slotIndex));
  writer.u32(event.matchedPromptTokens);
  writer.u32(event.capacityTokens);
  return success(Frame{FrameType::Start, writer.take()});
}

std::optional<ProtocolIssue>
validatePromptProgress(const PromptProgressEvent &event,
                       const ProtocolLimits &limits,
                       FailureClass failureClass) {
  if (!event.requestId || event.processedTokens > limits.maxPromptTokens)
    return makeIssue(failureClass, IssueCode::InvalidCount, event.requestId,
                     "prompt progress id or token count is invalid");
  return std::nullopt;
}

ProtocolResult<Frame> encodePromptProgress(const PromptProgressEvent &event,
                                           const ProtocolLimits &limits) {
  if (auto issue =
          validatePromptProgress(event, limits, FailureClass::EngineUnhealthy))
    return failure<Frame>(std::move(*issue));
  Writer writer(kPromptProgressFixedBytes);
  writer.u64(event.requestId);
  writer.u32(event.processedTokens);
  writer.u64(event.elapsedMicros);
  return success(Frame{FrameType::PromptProgress, writer.take()});
}

ProtocolResult<Frame> encodeTokens(const TokensEvent &event,
                                   const ProtocolLimits &limits) {
  if (auto issue =
          validateTokens(event, limits, FailureClass::EngineUnhealthy)) {
    return failure<Frame>(std::move(*issue));
  }
  Writer writer(kTokensFixedBytes + event.tokens.size() * sizeof(uint32_t));
  writer.u64(event.requestId);
  writer.u32(event.sequenceOffset);
  writer.u32(static_cast<uint32_t>(event.tokens.size()));
  for (uint32_t token : event.tokens)
    writer.u32(token);
  return success(Frame{FrameType::Tokens, writer.take()});
}

ProtocolResult<Frame> encodeMaskRequest(const MaskRequestEvent &event,
                                        const ProtocolLimits &limits) {
  if (auto issue =
          validateMaskRequest(event, limits, FailureClass::EngineUnhealthy)) {
    return failure<Frame>(std::move(*issue));
  }
  Writer writer(kMaskRequestFixedBytes +
                event.simulationTokens.size() * sizeof(uint32_t));
  writer.u64(event.requestId);
  writer.u64(event.maskRequestId);
  writer.u32(event.wordsPerMask);
  writer.u32(static_cast<uint32_t>(event.simulationTokens.size()));
  for (uint32_t token : event.simulationTokens)
    writer.u32(token);
  return success(Frame{FrameType::MaskRequest, writer.take()});
}

ProtocolResult<Frame> encodeDone(const DoneEvent &event) {
  if (auto issue = validateDone(event, FailureClass::EngineUnhealthy)) {
    return failure<Frame>(std::move(*issue));
  }
  Writer writer(kDoneFixedBytes +
                event.optionLogits.size() * sizeof(float));
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
  return success(Frame{FrameType::Done, writer.take()});
}

ProtocolResult<Frame> encodeError(const ErrorEvent &event,
                                  const ProtocolLimits &limits) {
  if (auto issue =
          validateError(event, limits, FailureClass::EngineUnhealthy)) {
    return failure<Frame>(std::move(*issue));
  }
  Writer writer(kErrorFixedBytes + event.code.size() + event.message.size());
  writer.u8(static_cast<uint8_t>(event.failureClass));
  writer.u8(event.retryable ? 1 : 0);
  writer.u64(event.requestId);
  writer.u32(static_cast<uint32_t>(event.code.size()));
  writer.u32(static_cast<uint32_t>(event.message.size()));
  writer.text(event.code);
  writer.text(event.message);
  return success(Frame{FrameType::Error, writer.take()});
}

ProtocolResult<Frame>
encodeCapacityExhausted(const CapacityExhaustedEvent &event) {
  if (auto issue =
          validateCapacityExhausted(event, FailureClass::EngineUnhealthy)) {
    return failure<Frame>(std::move(*issue));
  }
  Writer writer(kCapacityExhaustedFixedBytes);
  writer.u64(event.requestId);
  writer.u32(event.requiredKvPages);
  writer.u32(event.availableKvPages);
  writer.u64(event.retryAfterMicros);
  return success(Frame{FrameType::CapacityExhausted, writer.take()});
}

ProtocolResult<Frame> encodeStatusJson(const StatusJsonEvent &event,
                                       const ProtocolLimits &limits) {
  if (auto issue =
          validateStatusJson(event, limits, FailureClass::EngineUnhealthy)) {
    return failure<Frame>(std::move(*issue));
  }
  Writer writer(kStatusJsonFixedBytes + event.json.size());
  writer.u64(event.correlationId);
  writer.u32(event.schemaVersion);
  writer.text(event.json);
  return success(Frame{FrameType::StatusJson, writer.take()});
}

ProtocolResult<Message> decodeRequest(const Frame &frame,
                                      const ProtocolLimits &limits) {
  Reader reader(frame.payload);
  RequestFrame request;
  uint8_t priority = 0;
  uint8_t cohort = 0;
  uint8_t constraint = 0;
  uint8_t returnProgress = 0;
  uint32_t promptCount = 0;
  uint32_t imageSpanCount = 0;
  uint32_t scoreCount = 0;
  if (!reader.u64(request.requestId) || !reader.u8(priority) ||
      !reader.u8(cohort) || !reader.u8(constraint) ||
      !reader.u64(request.absoluteDeadlineUnixMicros) ||
      !reader.u64(request.remainingDeadlineMicros) ||
      !reader.u32(request.logicalMaxOutputTokens) || !reader.u32(promptCount) ||
      !reader.u32(imageSpanCount) ||
      !reader.f32(request.sampling.temperature) ||
      !reader.f32(request.sampling.topP) ||
      !reader.u32(request.sampling.topK) || !reader.u64(request.seed) ||
      !reader.u8(returnProgress) || !reader.u32(scoreCount)) {
    return failure<Message>(makeIssue(FailureClass::ProtocolFatal,
                                      IssueCode::InvalidPayloadLength, 0,
                                      "request fixed payload is truncated"));
  }
  if (returnProgress > 1) {
    return failure<Message>(
        makeIssue(FailureClass::RequestError, IssueCode::InvalidEnumValue,
                  request.requestId, "returnProgress must be a boolean"));
  }
  request.returnProgress = returnProgress;
  request.priority = static_cast<RequestPriority>(priority);
  request.cohort = static_cast<Cohort>(cohort);
  request.constraint = static_cast<ConstraintMode>(constraint);
  if (scoreCount > kMaximumScoreOptions) {
    return failure<Message>(
        makeIssue(FailureClass::RequestError, IssueCode::InvalidCount,
                  request.requestId, "score option count exceeds its limit"));
  }
  if (promptCount > limits.maxPromptTokens) {
    return failure<Message>(
        makeIssue(FailureClass::RequestError, IssueCode::LimitExceeded,
                  request.requestId, "prompt token count exceeds its limit"));
  }
  if (imageSpanCount > limits.maxImageSpans) {
    return failure<Message>(
        makeIssue(FailureClass::RequestError, IssueCode::LimitExceeded,
                  request.requestId, "image span count exceeds its limit"));
  }
  if (!reader.words(promptCount, request.promptTokens)) {
    return failure<Message>(
        makeIssue(FailureClass::RequestError, IssueCode::InvalidPayloadLength,
                  request.requestId,
                  "prompt count does not match the binary token payload"));
  }
  request.imageSpans.resize(imageSpanCount);
  uint64_t pixelBytes = 0;
  for (ImageSpanFrame &span : request.imageSpans) {
    if (!reader.u32(span.offset) || !reader.u32(span.tokens) ||
        !reader.u32(span.gridHeight) || !reader.u32(span.gridWidth) ||
        !reader.u64(span.digestLo) || !reader.u64(span.digestHi) ||
        !checkedAdd(pixelBytes, span.pixelBytes(), pixelBytes)) {
      return failure<Message>(
          makeIssue(FailureClass::RequestError, IssueCode::InvalidPayloadLength,
                    request.requestId, "image span payload is malformed"));
    }
  }
  uint64_t scoreBytes = 0;
  if (!checkedMultiply(scoreCount, sizeof(uint32_t), scoreBytes) ||
      reader.remaining() != pixelBytes + scoreBytes ||
      !reader.bytes(pixelBytes, request.imagePixels) ||
      !reader.words(scoreCount, request.scoreTokens)) {
    return failure<Message>(
        makeIssue(FailureClass::RequestError, IssueCode::InvalidPayloadLength,
                  request.requestId,
                  "image pixel or score payload does not match its counts"));
  }
  if (auto issue = validateRequest(request, limits)) {
    return failure<Message>(std::move(*issue));
  }
  return success(Message{std::move(request)});
}

ProtocolResult<Message> decodeCancel(const Frame &frame) {
  Reader reader(frame.payload);
  CancelFrame cancel;
  if (!reader.u64(cancel.requestId) || reader.remaining()) {
    return failure<Message>(makeIssue(FailureClass::ProtocolFatal,
                                      IssueCode::InvalidPayloadLength, 0,
                                      "cancel payload has an invalid length"));
  }
  if (auto issue = validateCancel(cancel)) {
    return failure<Message>(std::move(*issue));
  }
  return success(Message{cancel});
}

ProtocolResult<Message> decodeMaskResponse(const Frame &frame,
                                           const ProtocolLimits &limits) {
  Reader reader(frame.payload);
  MaskResponseFrame response;
  uint32_t wordCount = 0;
  if (!reader.u64(response.requestId) || !reader.u64(response.maskRequestId) ||
      !reader.u32(wordCount)) {
    return failure<Message>(
        makeIssue(FailureClass::ProtocolFatal, IssueCode::InvalidPayloadLength,
                  0, "mask response fixed payload is truncated"));
  }
  if (wordCount > limits.maxMaskWords) {
    return failure<Message>(makeIssue(
        FailureClass::RequestError, IssueCode::LimitExceeded,
        response.requestId, "mask response word count exceeds its limit"));
  }
  uint64_t expectedBytes = 0;
  if (!checkedMultiply(wordCount, sizeof(uint32_t), expectedBytes) ||
      reader.remaining() != expectedBytes ||
      !reader.words(wordCount, response.maskWords)) {
    return failure<Message>(
        makeIssue(FailureClass::RequestError, IssueCode::InvalidPayloadLength,
                  response.requestId,
                  "mask word count does not match the binary payload"));
  }
  if (auto issue = validateMaskResponse(response, limits)) {
    return failure<Message>(std::move(*issue));
  }
  return success(Message{std::move(response)});
}

ProtocolResult<Message> decodeStatusRequest(const Frame &frame) {
  Reader reader(frame.payload);
  StatusRequestFrame request;
  if (!reader.u64(request.correlationId) || reader.remaining()) {
    return failure<Message>(
        makeIssue(FailureClass::ProtocolFatal, IssueCode::InvalidPayloadLength,
                  0, "status request payload has an invalid length"));
  }
  return success(Message{request});
}

ProtocolResult<Message> decodeReady(const Frame &frame) {
  Reader reader(frame.payload);
  ReadyEvent event;
  if (!reader.u64(event.engineInstanceId) ||
      !reader.u32(event.maxConcurrentRequests) ||
      !reader.u32(event.maxContextTokens) || !reader.u64(event.featureBits) ||
      reader.remaining()) {
    return failure<Message>(makeIssue(FailureClass::ProtocolFatal,
                                      IssueCode::InvalidPayloadLength, 0,
                                      "ready payload has an invalid length"));
  }
  if (auto issue = validateReady(event, FailureClass::ProtocolFatal)) {
    return failure<Message>(std::move(*issue));
  }
  return success(Message{event});
}

ProtocolResult<Message> decodeStart(const Frame &frame) {
  Reader reader(frame.payload);
  StartEvent event;
  uint8_t disposition = 0;
  uint32_t slot = 0;
  if (!reader.u64(event.requestId) || !reader.u8(disposition) ||
      !reader.u32(slot) || !reader.u32(event.matchedPromptTokens) ||
      !reader.u32(event.capacityTokens) || reader.remaining()) {
    return failure<Message>(makeIssue(FailureClass::ProtocolFatal,
                                      IssueCode::InvalidPayloadLength, 0,
                                      "start payload has an invalid length"));
  }
  event.cacheDisposition = static_cast<CacheDisposition>(disposition);
  event.slotIndex = std::bit_cast<int32_t>(slot);
  if (auto issue = validateStart(event, FailureClass::ProtocolFatal)) {
    return failure<Message>(std::move(*issue));
  }
  return success(Message{event});
}

ProtocolResult<Message> decodePromptProgress(const Frame &frame,
                                             const ProtocolLimits &limits) {
  Reader reader(frame.payload);
  PromptProgressEvent event;
  if (!reader.u64(event.requestId) || !reader.u32(event.processedTokens) ||
      !reader.u64(event.elapsedMicros) || reader.remaining())
    return failure<Message>(
        makeIssue(FailureClass::ProtocolFatal, IssueCode::InvalidPayloadLength,
                  0, "prompt progress payload has an invalid length"));
  if (auto issue =
          validatePromptProgress(event, limits, FailureClass::ProtocolFatal))
    return failure<Message>(std::move(*issue));
  return success(Message{event});
}

ProtocolResult<Message> decodeTokens(const Frame &frame,
                                     const ProtocolLimits &limits) {
  Reader reader(frame.payload);
  TokensEvent event;
  uint32_t count = 0;
  if (!reader.u64(event.requestId) || !reader.u32(event.sequenceOffset) ||
      !reader.u32(count)) {
    return failure<Message>(makeIssue(FailureClass::ProtocolFatal,
                                      IssueCode::InvalidPayloadLength, 0,
                                      "tokens fixed payload is truncated"));
  }
  if (count > limits.maxTokenBatch) {
    return failure<Message>(makeIssue(
        FailureClass::ProtocolFatal, IssueCode::LimitExceeded, event.requestId,
        "tokens event batch size exceeds its limit"));
  }
  uint64_t expectedBytes = 0;
  if (!checkedMultiply(count, sizeof(uint32_t), expectedBytes) ||
      reader.remaining() != expectedBytes ||
      !reader.words(count, event.tokens)) {
    return failure<Message>(
        makeIssue(FailureClass::ProtocolFatal, IssueCode::InvalidPayloadLength,
                  event.requestId,
                  "token count does not match the binary event payload"));
  }
  if (auto issue = validateTokens(event, limits, FailureClass::ProtocolFatal)) {
    return failure<Message>(std::move(*issue));
  }
  return success(Message{std::move(event)});
}

ProtocolResult<Message> decodeMaskRequest(const Frame &frame,
                                          const ProtocolLimits &limits) {
  Reader reader(frame.payload);
  MaskRequestEvent event;
  uint32_t count = 0;
  if (!reader.u64(event.requestId) || !reader.u64(event.maskRequestId) ||
      !reader.u32(event.wordsPerMask) || !reader.u32(count)) {
    return failure<Message>(
        makeIssue(FailureClass::ProtocolFatal, IssueCode::InvalidPayloadLength,
                  0, "mask request fixed payload is truncated"));
  }
  if (count > limits.maxSimulationTokens) {
    return failure<Message>(makeIssue(
        FailureClass::ProtocolFatal, IssueCode::LimitExceeded, event.requestId,
        "mask request simulation token count exceeds its limit"));
  }
  uint64_t expectedBytes = 0;
  if (!checkedMultiply(count, sizeof(uint32_t), expectedBytes) ||
      reader.remaining() != expectedBytes ||
      !reader.words(count, event.simulationTokens)) {
    return failure<Message>(makeIssue(
        FailureClass::ProtocolFatal, IssueCode::InvalidPayloadLength,
        event.requestId,
        "simulation token count does not match the binary event payload"));
  }
  if (auto issue =
          validateMaskRequest(event, limits, FailureClass::ProtocolFatal)) {
    return failure<Message>(std::move(*issue));
  }
  return success(Message{std::move(event)});
}

ProtocolResult<Message> decodeDone(const Frame &frame) {
  Reader reader(frame.payload);
  DoneEvent event;
  uint8_t reason = 0;
  uint32_t logitCount = 0;
  if (!reader.u64(event.requestId) || !reader.u8(reason) ||
      !reader.u32(event.promptTokens) || !reader.u32(event.completionTokens) ||
      !reader.u64(event.prefillMicros) || !reader.u64(event.decodeMicros) ||
      !reader.u64(event.wallMicros) || !reader.u32(logitCount)) {
    return failure<Message>(makeIssue(FailureClass::ProtocolFatal,
                                      IssueCode::InvalidPayloadLength, 0,
                                      "done payload has an invalid length"));
  }
  if (logitCount > kMaximumScoreOptions) {
    return failure<Message>(
        makeIssue(FailureClass::ProtocolFatal, IssueCode::LimitExceeded,
                  event.requestId,
                  "done option logit count exceeds its limit"));
  }
  uint64_t logitBytes = 0;
  if (!checkedMultiply(logitCount, sizeof(float), logitBytes) ||
      reader.remaining() != logitBytes) {
    return failure<Message>(makeIssue(FailureClass::ProtocolFatal,
                                      IssueCode::InvalidPayloadLength, 0,
                                      "done option logits do not match the "
                                      "payload length"));
  }
  event.optionLogits.resize(logitCount);
  for (float &logit : event.optionLogits) {
    if (!reader.f32(logit)) {
      return failure<Message>(makeIssue(FailureClass::ProtocolFatal,
                                        IssueCode::InvalidPayloadLength, 0,
                                        "done option logits are truncated"));
    }
  }
  event.reason = static_cast<FinishReason>(reason);
  if (auto issue = validateDone(event, FailureClass::ProtocolFatal)) {
    return failure<Message>(std::move(*issue));
  }
  return success(Message{std::move(event)});
}

ProtocolResult<Message> decodeError(const Frame &frame,
                                    const ProtocolLimits &limits) {
  Reader reader(frame.payload);
  ErrorEvent event;
  uint8_t failureClass = 0;
  uint8_t retryable = 0;
  uint32_t codeBytes = 0;
  uint32_t messageBytes = 0;
  if (!reader.u8(failureClass) || !reader.u8(retryable) ||
      !reader.u64(event.requestId) || !reader.u32(codeBytes) ||
      !reader.u32(messageBytes)) {
    return failure<Message>(makeIssue(FailureClass::ProtocolFatal,
                                      IssueCode::InvalidPayloadLength, 0,
                                      "error fixed payload is truncated"));
  }
  event.failureClass = static_cast<FailureClass>(failureClass);
  event.retryable = retryable != 0;
  uint64_t stringsBytes = 0;
  if (retryable > 1 || !checkedAdd(codeBytes, messageBytes, stringsBytes) ||
      stringsBytes != reader.remaining() ||
      !reader.text(codeBytes, event.code) ||
      !reader.text(messageBytes, event.message)) {
    return failure<Message>(
        makeIssue(FailureClass::ProtocolFatal, IssueCode::InvalidPayloadLength,
                  0, "error string lengths do not match the frame payload"));
  }
  if (auto issue = validateError(event, limits, FailureClass::ProtocolFatal)) {
    return failure<Message>(std::move(*issue));
  }
  return success(Message{std::move(event)});
}

ProtocolResult<Message> decodeCapacityExhausted(const Frame &frame) {
  Reader reader(frame.payload);
  CapacityExhaustedEvent event;
  if (!reader.u64(event.requestId) || !reader.u32(event.requiredKvPages) ||
      !reader.u32(event.availableKvPages) ||
      !reader.u64(event.retryAfterMicros) || reader.remaining()) {
    return failure<Message>(
        makeIssue(FailureClass::ProtocolFatal, IssueCode::InvalidPayloadLength,
                  0, "capacity exhausted payload has an invalid length"));
  }
  if (auto issue =
          validateCapacityExhausted(event, FailureClass::ProtocolFatal)) {
    return failure<Message>(std::move(*issue));
  }
  return success(Message{event});
}

ProtocolResult<Message> decodeStatusJson(const Frame &frame,
                                         const ProtocolLimits &limits) {
  Reader reader(frame.payload);
  StatusJsonEvent event;
  if (!reader.u64(event.correlationId) || !reader.u32(event.schemaVersion) ||
      !reader.remainingText(event.json)) {
    return failure<Message>(
        makeIssue(FailureClass::ProtocolFatal, IssueCode::InvalidPayloadLength,
                  0, "status JSON payload has an invalid length"));
  }
  if (auto issue =
          validateStatusJson(event, limits, FailureClass::ProtocolFatal)) {
    return failure<Message>(std::move(*issue));
  }
  return success(Message{std::move(event)});
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
  case FrameType::CapacityExhausted:
    return "capacity_exhausted";
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
  case IssueCode::InvalidHeaderSize:
    return "invalid_header_size";
  case IssueCode::UnknownFrameType:
    return "unknown_frame_type";
  case IssueCode::NonZeroHeaderFlags:
    return "non_zero_header_flags";
  case IssueCode::NonZeroReservedField:
    return "non_zero_reserved_field";
  case IssueCode::FrameTooLarge:
    return "frame_too_large";
  case IssueCode::InvalidPayloadLength:
    return "invalid_payload_length";
  case IssueCode::TruncatedFrame:
    return "truncated_frame";
  case IssueCode::ParserAlreadyFailed:
    return "parser_already_failed";
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
  case IssueCode::InvalidCohortConstraint:
    return "invalid_cohort_constraint";
  case IssueCode::InvalidErrorClassification:
    return "invalid_error_classification";
  case IssueCode::InvalidStatusSchema:
    return "invalid_status_schema";
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

ProtocolResult<Frame> encodeMessage(const Message &message,
                                    const ProtocolLimits &limits) {
  if (auto issue = validateLimits(limits)) {
    return failure<Frame>(std::move(*issue));
  }
  try {
    ProtocolResult<Frame> encoded = std::visit(
        [&](const auto &value) -> ProtocolResult<Frame> {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, RequestFrame>) {
            return encodeRequest(value, limits);
          } else if constexpr (std::is_same_v<T, CancelFrame>) {
            return encodeCancel(value);
          } else if constexpr (std::is_same_v<T, MaskResponseFrame>) {
            return encodeMaskResponse(value, limits);
          } else if constexpr (std::is_same_v<T, StatusRequestFrame>) {
            return encodeStatusRequest(value);
          } else if constexpr (std::is_same_v<T, ReadyEvent>) {
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
          } else if constexpr (std::is_same_v<T, CapacityExhaustedEvent>) {
            return encodeCapacityExhausted(value);
          } else {
            return encodeStatusJson(value, limits);
          }
        },
        message);
    if (!encoded)
      return encoded;
    if (auto issue = validatePayloadLength(
            encoded.value->type, encoded.value->payload.size(), limits)) {
      const bool clientMessage =
          encoded.value->type == FrameType::Request ||
          encoded.value->type == FrameType::Cancel ||
          encoded.value->type == FrameType::MaskResponse ||
          encoded.value->type == FrameType::StatusRequest;
      issue->failureClass = clientMessage ? FailureClass::RequestError
                                          : FailureClass::EngineUnhealthy;
      issue->requestId = std::visit(
          [](const auto &value) -> uint64_t {
            if constexpr (requires { value.requestId; }) {
              return value.requestId;
            }
            return 0;
          },
          message);
      return failure<Frame>(std::move(*issue));
    }
    return encoded;
  } catch (const std::bad_alloc &) {
    return failure<Frame>(
        makeIssue(FailureClass::EngineUnhealthy, IssueCode::AllocationFailure,
                  0, "allocation failed while encoding protocol message"));
  } catch (const std::length_error &) {
    return failure<Frame>(makeIssue(
        FailureClass::EngineUnhealthy, IssueCode::AllocationFailure, 0,
        "container length failed while encoding protocol message"));
  }
}

ProtocolResult<Message> decodeFrame(const Frame &frame,
                                    const ProtocolLimits &limits) {
  if (auto issue = validateLimits(limits)) {
    return failure<Message>(std::move(*issue));
  }
  if (auto issue =
          validatePayloadLength(frame.type, frame.payload.size(), limits)) {
    return failure<Message>(std::move(*issue));
  }
  try {
    switch (frame.type) {
    case FrameType::Request:
      return decodeRequest(frame, limits);
    case FrameType::Cancel:
      return decodeCancel(frame);
    case FrameType::MaskResponse:
      return decodeMaskResponse(frame, limits);
    case FrameType::StatusRequest:
      return decodeStatusRequest(frame);
    case FrameType::Ready:
      return decodeReady(frame);
    case FrameType::Start:
      return decodeStart(frame);
    case FrameType::PromptProgress:
      return decodePromptProgress(frame, limits);
    case FrameType::Tokens:
      return decodeTokens(frame, limits);
    case FrameType::MaskRequest:
      return decodeMaskRequest(frame, limits);
    case FrameType::Done:
      return decodeDone(frame);
    case FrameType::Error:
      return decodeError(frame, limits);
    case FrameType::CapacityExhausted:
      return decodeCapacityExhausted(frame);
    case FrameType::StatusJson:
      return decodeStatusJson(frame, limits);
    }
  } catch (const std::bad_alloc &) {
    return failure<Message>(
        makeIssue(FailureClass::EngineUnhealthy, IssueCode::AllocationFailure,
                  0, "allocation failed while decoding protocol message"));
  } catch (const std::length_error &) {
    return failure<Message>(makeIssue(
        FailureClass::EngineUnhealthy, IssueCode::AllocationFailure, 0,
        "container length failed while decoding protocol message"));
  }
  return failure<Message>(
      makeIssue(FailureClass::ProtocolFatal, IssueCode::UnknownFrameType, 0,
                "frame type is not defined by native protocol"));
}

ProtocolResult<std::vector<uint8_t>>
serializeMessage(const Message &message, const ProtocolLimits &limits) {
  auto frame = encodeMessage(message, limits);
  if (!frame) {
    return failure<std::vector<uint8_t>>(std::move(*frame.issue));
  }
  try {
    Writer writer(kFrameHeaderBytes + frame.value->payload.size());
    writer.raw(kMagic);
    writer.u16(kProtocolVersion);
    writer.u16(static_cast<uint16_t>(kFrameHeaderBytes));
    writer.u16(static_cast<uint16_t>(frame.value->type));
    writer.u16(0); // flags
    writer.u64(frame.value->payload.size());
    writer.u32(0); // reserved
    writer.raw(frame.value->payload);
    return success(writer.take());
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

FrameParser::FrameParser(ProtocolLimits limits) : limits_(limits) {
  if (auto issue = validateLimits(limits_)) {
    terminalIssue_ = std::move(*issue);
  }
}

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
  uint16_t headerBytes = loadU16(header_.data() + 6);
  if (headerBytes != kFrameHeaderBytes) {
    return makeIssue(FailureClass::ProtocolFatal, IssueCode::InvalidHeaderSize,
                     0,
                     "native protocol frame header must be exactly 24 bytes");
  }
  uint16_t rawType = loadU16(header_.data() + 8);
  if (!validFrameType(rawType, currentType_)) {
    return makeIssue(FailureClass::ProtocolFatal, IssueCode::UnknownFrameType,
                     0, "frame type is not defined by native protocol");
  }
  if (loadU16(header_.data() + 10)) {
    return makeIssue(FailureClass::ProtocolFatal, IssueCode::NonZeroHeaderFlags,
                     0, "native protocol frame flags must be zero");
  }
  if (loadU32(header_.data() + 20)) {
    return makeIssue(FailureClass::ProtocolFatal,
                     IssueCode::NonZeroReservedField, 0,
                     "native protocol reserved header field must be zero");
  }
  expectedPayloadBytes_ = loadU64(header_.data() + 12);
  if (auto issue =
          validatePayloadLength(currentType_, expectedPayloadBytes_, limits_)) {
    return issue;
  }
  payload_.clear();
  readingPayload_ = true;
  payloadBytes_ = 0;
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
  payloadBytes_ = 0;
  payload_.clear();
}

ParseStep FrameParser::consume(std::span<const uint8_t> bytes) {
  if (terminalIssue_) {
    return {0, std::nullopt,
            makeIssue(FailureClass::ProtocolFatal,
                      IssueCode::ParserAlreadyFailed, 0,
                      terminalIssue_->describe())};
  }

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
      if (!expectedPayloadBytes_) {
        Frame frame{currentType_, {}};
        resetCurrentFrame();
        return {consumed, std::move(frame), std::nullopt};
      }
    }

    size_t needed = static_cast<size_t>(expectedPayloadBytes_) - payloadBytes_;
    size_t count = std::min(needed, bytes.size() - consumed);
    if (!count)
      return {consumed, std::nullopt, std::nullopt};
    try {
      // The validated frame length is known. Reserve it once to avoid
      // geometric growth copies; only received bytes are initialized.
      if (payload_.capacity() < expectedPayloadBytes_)
        payload_.reserve(static_cast<size_t>(expectedPayloadBytes_));
      payload_.resize(payloadBytes_ + count);
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
    std::memcpy(payload_.data() + payloadBytes_, bytes.data() + consumed,
                count);
    payloadBytes_ += count;
    consumed += count;
    if (payloadBytes_ == expectedPayloadBytes_) {
      Frame frame{currentType_, std::move(payload_)};
      resetCurrentFrame();
      return {consumed, std::move(frame), std::nullopt};
    }
  }
  return {consumed, std::nullopt, std::nullopt};
}

std::optional<ProtocolIssue> FrameParser::finish() {
  if (terminalIssue_)
    return terminalIssue_;
  if (!headerBytes_ && !readingPayload_)
    return std::nullopt;

  std::ostringstream message;
  if (!readingPayload_) {
    message << "stream ended after " << headerBytes_
            << " of 24 frame-header bytes";
  } else {
    message << "stream ended after " << payloadBytes_ << " of "
            << expectedPayloadBytes_ << ' ' << frameTypeName(currentType_)
            << " payload bytes";
  }
  terminalIssue_ = makeIssue(FailureClass::ProtocolFatal,
                             IssueCode::TruncatedFrame, 0, message.str());
  return terminalIssue_;
}

} // namespace splash::protocol

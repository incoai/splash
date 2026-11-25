#include "video/protocol/VideoProtocol.hpp"
#include "Checked.hpp"

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <new>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace splash::video::wire {
namespace {

using protocol::FailureClass;
using protocol::IssueCode;
using protocol::ProtocolIssue;
using protocol::ProtocolResult;

// Cancel: one u64 request id, as the engine protocol's
// (engine/Protocol.cpp kRequestEndFixedBytes).
constexpr uint64_t kRequestEndFixedBytes = 8;
constexpr uint64_t kVideoRequestFixedBytes = 43;
// A reference's header: u8 kind, u32 frames, u32 width, u32 height, u64
// soundtrack samples.
constexpr uint64_t kVideoReferenceHeaderBytes = 21;
constexpr uint64_t kVideoReadyFixedBytes = 8;
constexpr uint64_t kVideoProgressFixedBytes = 33;
constexpr uint64_t kVideoOutputFixedBytes = 21;

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

ProtocolIssue invalidEvent(IssueCode code, uint64_t requestId,
                           std::string message) {
  return makeIssue(FailureClass::EngineUnhealthy, code, requestId,
                   std::move(message));
}

template <typename T> ProtocolResult<T> success(T value) {
  return {std::move(value), std::nullopt};
}

template <typename T> ProtocolResult<T> failure(ProtocolIssue issue) {
  return {std::nullopt, std::move(issue)};
}

std::string_view frameTypeName(VideoFrameType type) {
  switch (type) {
  case VideoFrameType::Cancel:
    return "cancel";
  case VideoFrameType::VideoRequest:
    return "video_request";
  case VideoFrameType::Error:
    return "error";
  case VideoFrameType::StartFailed:
    return "start_failed";
  case VideoFrameType::VideoProgress:
    return "video_progress";
  case VideoFrameType::VideoOutput:
    return "video_output";
  case VideoFrameType::VideoReady:
    return "video_ready";
  }
  return "unknown";
}

std::optional<PayloadBounds> payloadBounds(VideoFrameType type,
                                           const VideoLimits &limits) {
  switch (type) {
  case VideoFrameType::VideoRequest:
    // Its references' frames dominate; its prompt count is checked as a
    // request's own.
    return PayloadBounds{kVideoRequestFixedBytes,
                         limits.maxVideoRequestPayloadBytes};
  case VideoFrameType::Cancel:
    return PayloadBounds{kRequestEndFixedBytes, kRequestEndFixedBytes};
  default:
    // The engine never receives an event.
    return std::nullopt;
  }
}

std::optional<ProtocolIssue> validatePayloadLength(VideoFrameType type,
                                                   uint64_t payloadBytes,
                                                   const VideoLimits &limits) {
  auto bounds = payloadBounds(type, limits);
  if (!bounds) {
    return makeIssue(FailureClass::ProtocolFatal, IssueCode::UnknownFrameType,
                     0, "frame type is not a video protocol client frame");
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

bool clientFrameType(uint16_t raw, VideoFrameType &type) {
  switch (static_cast<VideoFrameType>(raw)) {
  case VideoFrameType::VideoRequest:
  case VideoFrameType::Cancel:
    type = static_cast<VideoFrameType>(raw);
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

// The engine protocol's byte writer and reader (engine/Protocol.cpp), which
// it keeps to itself.
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

  void raw(std::span<const uint8_t> bytes) {
    bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
  }

  [[nodiscard]] std::vector<uint8_t> take() { return std::move(bytes_); }

private:
  std::vector<uint8_t> bytes_;
};

// A frame's header, in a buffer sized for the payload the caller writes next.
Writer frameWriter(VideoFrameType type, uint64_t payloadBytes) {
  Writer writer(protocol::kFrameHeaderBytes + payloadBytes);
  writer.raw(kVideoMagic);
  writer.u16(kVideoProtocolVersion);
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
  [[nodiscard]] size_t offset() const { return offset_; }

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

// The bytes of a keyframe of the request's canvas, height x width x 3; zero
// when they overflow.
uint64_t keyframeBytes(const VideoRequestFrame &request) {
  uint64_t pixels = 0, bytes = 0;
  if (!checkedMultiply(request.width, request.height, pixels) ||
      !checkedMultiply(pixels, 3, bytes))
    return 0;
  return bytes;
}

// A reference's pixel bytes, frames x height x width x 3; zero when they
// overflow.
uint64_t referencePixelBytes(const VideoReference &reference) {
  uint64_t pixels = 0, frames = 0, bytes = 0;
  if (!checkedMultiply(reference.width, reference.height, pixels) ||
      !checkedMultiply(pixels, reference.frames, frames) ||
      !checkedMultiply(frames, 3, bytes))
    return 0;
  return bytes;
}

std::optional<ProtocolIssue> validateVideoRequest(const VideoRequestFrame &request,
                                                  const VideoLimits &limits) {
  if (!request.requestId)
    return requestIssue(IssueCode::InvalidRequestId, 0,
                        "video request id must be non-zero");
  if (!validDitName(request.dit) ||
      (request.outputs & ~(VideoOutputPreview | VideoOutputAudio)))
    return requestIssue(IssueCode::InvalidEnumValue, request.requestId,
                        "video DiT name or outputs are not defined");
  if (!request.frames || !request.width || !request.height)
    return requestIssue(IssueCode::InvalidCount, request.requestId,
                        "a video request needs frames, a width and a height");
  if (request.promptTokens.empty() ||
      request.promptTokens.size() > limits.shared.maxPromptTokens)
    return requestIssue(IssueCode::LimitExceeded, request.requestId,
                        "a video prompt holds 1 token or more, within the prompt limit");
  for (const VideoReference &reference : request.references) {
    if (!validEnum(static_cast<uint8_t>(reference.kind),
                   {VideoReferenceKind::Image, VideoReferenceKind::Video,
                    VideoReferenceKind::Audio}))
      return requestIssue(IssueCode::InvalidEnumValue, request.requestId,
                          "video reference kind is not defined");
    const bool image = reference.kind == VideoReferenceKind::Image;
    if (reference.kind == VideoReferenceKind::Audio) {
      if (reference.frames || reference.width || reference.height ||
          reference.soundtrack.empty())
        return requestIssue(IssueCode::InvalidCount, request.requestId,
                            "an audio reference is a soundtrack without frames");
    } else if (!reference.frames || !reference.width || !reference.height ||
               (image && (reference.frames != 1 || !reference.soundtrack.empty())))
      return requestIssue(IssueCode::InvalidCount, request.requestId,
                          "a video reference needs frames, a width and a height; "
                          "an image is one frame without a soundtrack");
  }
  return std::nullopt;
}

std::optional<ProtocolIssue> validateVideoReady(const VideoReadyEvent &event) {
  if (!event.maxConcurrentRequests || !event.maxPromptTokens)
    return invalidEvent(IssueCode::InvalidCount, 0,
                        "video ready capacities must be non-zero");
  return std::nullopt;
}

std::optional<ProtocolIssue> validateVideoProgress(const VideoProgressEvent &event) {
  if (!event.requestId)
    return invalidEvent(IssueCode::InvalidRequestId, 0,
                        "video progress request id must be non-zero");
  if (!validEnum(static_cast<uint8_t>(event.phase),
                 {VideoPhase::Encode, VideoPhase::Prepare, VideoPhase::Denoise,
                  VideoPhase::Preview, VideoPhase::Decode, VideoPhase::Audio,
                  VideoPhase::Mux}))
    return invalidEvent(IssueCode::InvalidEnumValue, event.requestId,
                        "video phase is not defined");
  if (!event.steps || event.step > event.steps)
    return invalidEvent(IssueCode::InvalidCount, event.requestId,
                        "video progress steps must be 1 or more, its step at most them");
  return std::nullopt;
}

std::optional<ProtocolIssue> validateVideoOutput(const VideoOutputEvent &event,
                                                 const VideoLimits &limits) {
  if (!event.requestId)
    return invalidEvent(IssueCode::InvalidRequestId, 0,
                        "video output request id must be non-zero");
  if (!validEnum(static_cast<uint8_t>(event.output),
                 {VideoOutputKind::Video, VideoOutputKind::Preview}))
    return invalidEvent(IssueCode::InvalidEnumValue, event.requestId,
                        "video output is not defined");
  if (!event.frames || event.mp4.empty() ||
      event.mp4.size() > limits.shared.maxFramePayloadBytes - kVideoOutputFixedBytes)
    return invalidEvent(IssueCode::LimitExceeded, event.requestId,
                        "video output must hold frames and an mp4 within the frame limit");
  return std::nullopt;
}

using EncodedEvent = ProtocolResult<std::vector<uint8_t>>;

EncodedEvent encodeVideoReady(const VideoReadyEvent &event) {
  if (auto issue = validateVideoReady(event))
    return failure<std::vector<uint8_t>>(std::move(*issue));
  Writer writer = frameWriter(VideoFrameType::VideoReady, kVideoReadyFixedBytes);
  writer.u32(event.maxConcurrentRequests);
  writer.u32(event.maxPromptTokens);
  return success(writer.take());
}

EncodedEvent encodeVideoProgress(const VideoProgressEvent &event) {
  if (auto issue = validateVideoProgress(event))
    return failure<std::vector<uint8_t>>(std::move(*issue));
  Writer writer = frameWriter(VideoFrameType::VideoProgress, kVideoProgressFixedBytes);
  writer.u64(event.requestId);
  writer.u8(static_cast<uint8_t>(event.phase));
  writer.u32(event.step);
  writer.u32(event.steps);
  writer.u64(event.elapsedMicros);
  writer.u64(event.peakFootprintBytes);
  return success(writer.take());
}

EncodedEvent encodeVideoOutput(const VideoOutputEvent &event,
                               const VideoLimits &limits) {
  if (auto issue = validateVideoOutput(event, limits))
    return failure<std::vector<uint8_t>>(std::move(*issue));
  Writer writer =
      frameWriter(VideoFrameType::VideoOutput, kVideoOutputFixedBytes + event.mp4.size());
  writer.u64(event.requestId);
  writer.u8(static_cast<uint8_t>(event.output));
  writer.u32(event.frames);
  writer.u64(event.wallMicros);
  writer.raw(event.mp4);
  return success(writer.take());
}

// An Error or StartFailed frame as the engine protocol encodes it, its
// payload and type id kept, with the video stream's magic and version.
EncodedEvent videoHeaded(EncodedEvent encoded) {
  if (encoded) {
    std::vector<uint8_t> &frame = *encoded.value;
    std::copy(kVideoMagic.begin(), kVideoMagic.end(), frame.begin());
    frame[4] = static_cast<uint8_t>(kVideoProtocolVersion);
    frame[5] = static_cast<uint8_t>(kVideoProtocolVersion >> 8);
  }
  return encoded;
}

ProtocolResult<VideoClientMessage> decodeVideoRequest(const VideoFrame &frame,
                                                      const VideoLimits &limits) {
  Reader reader(frame.payload);
  VideoRequestFrame request;
  uint8_t nameBytes = 0, keyframes = 0;
  uint32_t promptCount = 0, referenceCount = 0;
  if (!reader.u64(request.requestId) || !reader.u8(nameBytes) ||
      !reader.u8(request.outputs) || !reader.u32(request.frames) ||
      !reader.u32(request.width) || !reader.u32(request.height) ||
      !reader.u32(request.steps) || !reader.u64(request.seed) ||
      !reader.u8(keyframes) || !reader.u32(promptCount) ||
      !reader.u32(referenceCount)) {
    return failure<VideoClientMessage>(
        makeIssue(FailureClass::ProtocolFatal, IssueCode::InvalidPayloadLength,
                  0, "video request fixed payload is truncated"));
  }
  if (keyframes & ~(VideoKeyframeFirst | VideoKeyframeLast)) {
    return failure<VideoClientMessage>(
        requestIssue(IssueCode::InvalidEnumValue, request.requestId,
                     "video keyframes are not defined"));
  }
  if (referenceCount > kMaxVideoReferences) {
    return failure<VideoClientMessage>(
        requestIssue(IssueCode::LimitExceeded, request.requestId,
                     "a video request holds at most 12 references"));
  }
  const uint64_t canvas = keyframeBytes(request);
  const uint64_t keyframePixels = canvas * (((keyframes & VideoKeyframeFirst) != 0) +
                                            ((keyframes & VideoKeyframeLast) != 0));
  auto mismatch = [&] {
    return failure<VideoClientMessage>(requestIssue(
        IssueCode::InvalidPayloadLength, request.requestId,
        "video DiT name, prompt count, keyframes or references do not match "
        "the payload"));
  };
  const size_t nameAt = reader.offset();
  if (!reader.skip(nameBytes))
    return mismatch();
  request.dit.assign(
      reinterpret_cast<const char *>(frame.payload.data()) + nameAt, nameBytes);
  if (!reader.words(promptCount, request.promptTokens) ||
      (keyframes && (!canvas || canvas > frame.payload.size())) ||
      reader.remaining() < referenceCount * kVideoReferenceHeaderBytes)
    return mismatch();
  // The references' headers, and the bytes their pixels and soundtracks take.
  std::vector<uint64_t> samples(referenceCount);
  uint64_t referenceBytes = 0;
  request.references.resize(referenceCount);
  for (uint32_t index = 0; index < referenceCount; ++index) {
    VideoReference &reference = request.references[index];
    uint8_t kind = 0;
    if (!reader.u8(kind) || !reader.u32(reference.frames) ||
        !reader.u32(reference.width) || !reader.u32(reference.height) ||
        !reader.u64(samples[index]))
      return mismatch();
    reference.kind = static_cast<VideoReferenceKind>(kind);
    uint64_t sound = 0, bytes = 0;
    const uint64_t pixels = referencePixelBytes(reference);
    // An audio reference has no pixels; anything else has some.
    if ((!pixels && reference.kind != VideoReferenceKind::Audio) ||
        !checkedMultiply(samples[index], 8, sound) ||
        !checkedAdd(pixels, sound, bytes) ||
        !checkedAdd(referenceBytes, bytes, referenceBytes))
      return mismatch();
  }
  uint64_t variable = 0;
  if (!checkedAdd(keyframePixels, referenceBytes, variable) ||
      reader.remaining() != variable)
    return mismatch();
  const uint8_t *cursor = frame.payload.data() + reader.offset();
  if (keyframes & VideoKeyframeFirst) {
    request.firstFrame.assign(cursor, cursor + canvas);
    cursor += canvas;
  }
  if (keyframes & VideoKeyframeLast) {
    request.lastFrame.assign(cursor, cursor + canvas);
    cursor += canvas;
  }
  for (uint32_t index = 0; index < referenceCount; ++index) {
    VideoReference &reference = request.references[index];
    const size_t pixels = static_cast<size_t>(referencePixelBytes(reference));
    reference.pixels.assign(cursor, cursor + pixels);
    cursor += pixels;
    reference.soundtrack.resize(static_cast<size_t>(2 * samples[index]));
    if (!reference.soundtrack.empty())
      std::memcpy(reference.soundtrack.data(), cursor, reference.soundtrack.size() * sizeof(float));
    cursor += reference.soundtrack.size() * sizeof(float);
  }
  if (auto issue = validateVideoRequest(request, limits)) {
    return failure<VideoClientMessage>(std::move(*issue));
  }
  return success(VideoClientMessage{std::move(request)});
}

// The engine protocol's Cancel, decoded by its codec.
ProtocolResult<VideoClientMessage> decodeCancel(VideoFrame &&frame,
                                                const VideoLimits &limits) {
  auto decoded = protocol::decodeFrame(
      protocol::Frame{protocol::FrameType::Cancel, std::move(frame.payload)},
      limits.shared);
  if (!decoded)
    return failure<VideoClientMessage>(std::move(*decoded.issue));
  return success(
      VideoClientMessage{std::get<protocol::CancelFrame>(*decoded.value)});
}

} // namespace

bool validDitName(std::string_view name) {
  return !name.empty() && name.size() <= kMaxDitNameBytes &&
         name.find("..") == std::string_view::npos &&
         std::all_of(name.begin(), name.end(), [](char c) {
           return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                  c == '.' || c == '_' || c == '-';
         });
}

ProtocolResult<VideoClientMessage> decodeVideoFrame(VideoFrame &&frame,
                                                    const VideoLimits &limits) {
  if (auto issue =
          validatePayloadLength(frame.type, frame.payload.size(), limits)) {
    return failure<VideoClientMessage>(std::move(*issue));
  }
  try {
    switch (frame.type) {
    case VideoFrameType::VideoRequest:
      return decodeVideoRequest(frame, limits);
    case VideoFrameType::Cancel:
      return decodeCancel(std::move(frame), limits);
    default:
      break;
    }
  } catch (const std::bad_alloc &) {
    return failure<VideoClientMessage>(
        makeIssue(FailureClass::EngineUnhealthy, IssueCode::AllocationFailure,
                  0, "allocation failed while decoding protocol message"));
  } catch (const std::length_error &) {
    return failure<VideoClientMessage>(makeIssue(
        FailureClass::EngineUnhealthy, IssueCode::AllocationFailure, 0,
        "container length failed while decoding protocol message"));
  }
  return failure<VideoClientMessage>(
      makeIssue(FailureClass::ProtocolFatal, IssueCode::UnknownFrameType, 0,
                "frame type is not a video protocol client frame"));
}

ProtocolResult<std::vector<uint8_t>>
serializeVideoEvent(const VideoEvent &event, const VideoLimits &limits) {
  try {
    return std::visit(
        [&](const auto &value) -> EncodedEvent {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, VideoReadyEvent>) {
            return encodeVideoReady(value);
          } else if constexpr (std::is_same_v<T, VideoProgressEvent>) {
            return encodeVideoProgress(value);
          } else if constexpr (std::is_same_v<T, VideoOutputEvent>) {
            return encodeVideoOutput(value, limits);
          } else {
            // Error and StartFailed.
            return videoHeaded(protocol::serializeEvent(value, limits.shared));
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

VideoFrameParser::VideoFrameParser(VideoLimits limits) : limits_(limits) {}

std::optional<ProtocolIssue> VideoFrameParser::parseHeader() {
  if (!std::equal(kVideoMagic.begin(), kVideoMagic.end(), header_.begin())) {
    return makeIssue(FailureClass::ProtocolFatal, IssueCode::BadMagic, 0,
                     "frame magic is not SPLV");
  }
  uint16_t version = loadU16(header_.data() + 4);
  if (version != kVideoProtocolVersion) {
    return makeIssue(FailureClass::ProtocolFatal, IssueCode::UnsupportedVersion,
                     0, "unsupported video protocol version");
  }
  uint16_t rawType = loadU16(header_.data() + 6);
  if (!clientFrameType(rawType, currentType_)) {
    return makeIssue(FailureClass::ProtocolFatal, IssueCode::UnknownFrameType,
                     0, "frame type is not a video protocol client frame");
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

VideoParseStep VideoFrameParser::fail(size_t consumed, ProtocolIssue issue) {
  terminalIssue_ = std::move(issue);
  return {consumed, std::nullopt, terminalIssue_};
}

void VideoFrameParser::resetCurrentFrame() {
  header_.fill(0);
  headerBytes_ = 0;
  readingPayload_ = false;
  expectedPayloadBytes_ = 0;
  payload_.clear();
}

VideoParseStep VideoFrameParser::consume(std::span<const uint8_t> bytes) {
  if (terminalIssue_)
    throw std::logic_error("video frame parser used after it failed");

  size_t consumed = 0;
  while (consumed < bytes.size()) {
    if (!readingPayload_) {
      size_t count = std::min(protocol::kFrameHeaderBytes - headerBytes_,
                              bytes.size() - consumed);
      std::memcpy(header_.data() + headerBytes_, bytes.data() + consumed,
                  count);
      headerBytes_ += count;
      consumed += count;
      if (headerBytes_ < protocol::kFrameHeaderBytes) {
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
      VideoFrame frame{currentType_, std::move(payload_)};
      resetCurrentFrame();
      return {consumed, std::move(frame), std::nullopt};
    }
  }
  return {consumed, std::nullopt, std::nullopt};
}

std::optional<ProtocolIssue> VideoFrameParser::finish() {
  if (terminalIssue_)
    throw std::logic_error("video frame parser used after it failed");
  if (!headerBytes_ && !readingPayload_)
    return std::nullopt;

  std::ostringstream message;
  if (!readingPayload_) {
    message << "stream ended after " << headerBytes_ << " of "
            << protocol::kFrameHeaderBytes << " frame-header bytes";
  } else {
    message << "stream ended after " << payload_.size() << " of "
            << expectedPayloadBytes_ << ' ' << frameTypeName(currentType_)
            << " payload bytes";
  }
  terminalIssue_ = makeIssue(FailureClass::ProtocolFatal,
                             IssueCode::TruncatedFrame, 0, message.str());
  return terminalIssue_;
}

} // namespace splash::video::wire

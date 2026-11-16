#include "engine/Protocol.hpp"
#include "video/protocol/VideoProtocol.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

// The video engine's stream (video/protocol/VideoProtocol.hpp): the golden
// bytes server/video_protocol.py writes and reads, each frame's rules, the
// parser's failures, and the engine protocol's payloads it carries as the
// engine protocol encodes and decodes them.
namespace {

using namespace splash::video::wire;
namespace protocol = splash::protocol;
using protocol::CancelFrame;
using protocol::ErrorEvent;
using protocol::FailureClass;
using protocol::IssueCode;
using protocol::kFrameHeaderBytes;
using protocol::ProtocolIssue;
using protocol::ProtocolResult;
using protocol::StartFailedEvent;
using protocol::StartFailureKind;

int failures = 0;

void check(bool condition, std::string_view expression, std::string_view test,
           int line) {
  if (condition)
    return;
  ++failures;
  std::cerr << "FAIL " << test << ':' << line << ": " << expression << '\n';
}

#define CHECK(testName, expression)                                            \
  check(static_cast<bool>(expression), #expression, testName, __LINE__)

const VideoLimits kLimits{};

uint16_t loadU16(const std::vector<uint8_t> &bytes, size_t offset) {
  return static_cast<uint16_t>(bytes.at(offset) | (bytes.at(offset + 1) << 8));
}

uint64_t loadU64(const std::vector<uint8_t> &bytes, size_t offset) {
  uint64_t result = 0;
  for (uint32_t index = 0; index < 8; ++index) {
    result |= static_cast<uint64_t>(bytes.at(offset + index)) << (index * 8);
  }
  return result;
}

void storeU16(std::vector<uint8_t> &bytes, size_t offset, uint16_t value) {
  for (uint32_t index = 0; index < 2; ++index) {
    bytes.at(offset + index) = static_cast<uint8_t>(value >> (index * 8));
  }
}

void storeU64(std::vector<uint8_t> &bytes, size_t offset, uint64_t value) {
  for (uint32_t index = 0; index < 8; ++index) {
    bytes.at(offset + index) = static_cast<uint8_t>(value >> (index * 8));
  }
}

// The server's side of the stream, written apart from the engine's codec and
// from server/video_protocol.py.
class Writer {
public:
  void u8(uint8_t value) { bytes_.push_back(value); }
  void u16(uint16_t value) {
    for (uint32_t shift = 0; shift < 16; shift += 8)
      u8(static_cast<uint8_t>(value >> shift));
  }
  void u32(uint32_t value) {
    for (uint32_t shift = 0; shift < 32; shift += 8)
      u8(static_cast<uint8_t>(value >> shift));
  }
  void u64(uint64_t value) {
    for (uint32_t shift = 0; shift < 64; shift += 8)
      u8(static_cast<uint8_t>(value >> shift));
  }
  void f32(float value) { u32(std::bit_cast<uint32_t>(value)); }
  void raw(std::span<const uint8_t> bytes) {
    bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
  }
  void words(const std::vector<uint32_t> &values) {
    for (uint32_t value : values)
      u32(value);
  }
  [[nodiscard]] std::vector<uint8_t> take() { return std::move(bytes_); }

private:
  std::vector<uint8_t> bytes_;
};

std::vector<uint8_t> frameBytes(VideoFrameType type,
                                const std::vector<uint8_t> &payload) {
  Writer out;
  out.raw(kVideoMagic);
  out.u16(kVideoProtocolVersion);
  out.u16(static_cast<uint16_t>(type));
  out.u64(payload.size());
  out.raw(payload);
  return out.take();
}

std::vector<uint8_t> clientFrame(const VideoRequestFrame &request) {
  Writer out;
  out.u64(request.requestId);
  out.u8(static_cast<uint8_t>(request.dit.size()));
  out.u8(request.outputs);
  out.u32(request.frames);
  out.u32(request.width);
  out.u32(request.height);
  out.u32(request.steps);
  out.u64(request.seed);
  out.u8(static_cast<uint8_t>((request.firstFrame.empty() ? 0 : VideoKeyframeFirst) |
                              (request.lastFrame.empty() ? 0 : VideoKeyframeLast)));
  out.u32(static_cast<uint32_t>(request.promptTokens.size()));
  out.u32(static_cast<uint32_t>(request.references.size()));
  out.raw(std::span(reinterpret_cast<const uint8_t *>(request.dit.data()), request.dit.size()));
  out.words(request.promptTokens);
  for (const VideoReference &reference : request.references) {
    out.u8(static_cast<uint8_t>(reference.kind));
    out.u32(reference.frames);
    out.u32(reference.width);
    out.u32(reference.height);
    out.u64(reference.soundtrack.size() / 2);
  }
  out.raw(request.firstFrame);
  out.raw(request.lastFrame);
  for (const VideoReference &reference : request.references) {
    out.raw(reference.pixels);
    for (float sample : reference.soundtrack)
      out.f32(sample);
  }
  return frameBytes(VideoFrameType::VideoRequest, out.take());
}

std::vector<uint8_t> clientFrame(const CancelFrame &cancel) {
  Writer out;
  out.u64(cancel.requestId);
  return frameBytes(VideoFrameType::Cancel, out.take());
}

std::vector<uint8_t> clientFrame(const VideoClientMessage &message) {
  return std::visit([](const auto &each) { return clientFrame(each); },
                    message);
}

// The server's reading of the video events of one frame.
class Reader {
public:
  explicit Reader(std::span<const uint8_t> bytes) : bytes_(bytes) {}

  uint64_t take(size_t count) {
    if (bytes_.size() - offset_ < count)
      throw std::runtime_error("the event is truncated");
    uint64_t value = 0;
    for (size_t index = 0; index < count; ++index)
      value |= static_cast<uint64_t>(bytes_[offset_ + index]) << (index * 8);
    offset_ += count;
    return value;
  }
  uint8_t u8() { return static_cast<uint8_t>(take(1)); }
  uint32_t u32() { return static_cast<uint32_t>(take(4)); }
  uint64_t u64() { return take(8); }
  std::vector<uint8_t> rest() {
    std::vector<uint8_t> result(bytes_.begin() + static_cast<std::ptrdiff_t>(offset_), bytes_.end());
    offset_ = bytes_.size();
    return result;
  }
  [[nodiscard]] bool done() const { return offset_ == bytes_.size(); }

private:
  std::span<const uint8_t> bytes_;
  size_t offset_ = 0;
};

VideoEvent decodeEvent(const std::vector<uint8_t> &frame) {
  if (frame.size() < kFrameHeaderBytes ||
      !std::equal(kVideoMagic.begin(), kVideoMagic.end(), frame.begin()) ||
      loadU16(frame, 4) != kVideoProtocolVersion ||
      loadU64(frame, 8) != frame.size() - kFrameHeaderBytes)
    throw std::runtime_error("the event's header is not the video stream's");
  Reader in(std::span<const uint8_t>(frame).subspan(kFrameHeaderBytes));
  VideoEvent event;
  switch (static_cast<VideoFrameType>(loadU16(frame, 6))) {
  case VideoFrameType::VideoReady: {
    VideoReadyEvent ready;
    ready.maxConcurrentRequests = in.u32();
    ready.maxPromptTokens = in.u32();
    event = ready;
    break;
  }
  case VideoFrameType::VideoProgress: {
    VideoProgressEvent progress;
    progress.requestId = in.u64();
    progress.phase = static_cast<VideoPhase>(in.u8());
    progress.step = in.u32();
    progress.steps = in.u32();
    progress.elapsedMicros = in.u64();
    progress.peakFootprintBytes = in.u64();
    event = progress;
    break;
  }
  case VideoFrameType::VideoOutput: {
    VideoOutputEvent output;
    output.requestId = in.u64();
    output.output = static_cast<VideoOutputKind>(in.u8());
    output.frames = in.u32();
    output.wallMicros = in.u64();
    output.mp4 = in.rest();
    event = std::move(output);
    break;
  }
  default:
    throw std::runtime_error("the frame is not a video event this test reads");
  }
  if (!in.done())
    throw std::runtime_error("the event has bytes past its fields");
  return event;
}

const VideoRequestFrame kVideoRequest{92, "turbo",
                                      VideoOutputPreview | VideoOutputAudio,
                                      73, 1344, 768, 0, 42, {5, 6, 7}, {}, {}, {}};

VideoRequestFrame exampleKeyframeRequest() {
  VideoRequestFrame request{92, "turbo", VideoOutputAudio, 73, 4, 2, 0, 42, {5}, {}, {}, {}};
  for (uint8_t index = 0; index < 24; ++index) {
    request.firstFrame.push_back(index);
    request.lastFrame.push_back(static_cast<uint8_t>(255 - index));
  }
  return request;
}

// A Ref2VA clip of a 4 x 2 canvas from a 2 x 2 video of two frames with a
// soundtrack of three samples, a 2 x 1 image and an audio reference of one
// sample.
VideoRequestFrame exampleReferenceRequest() {
  VideoRequestFrame request{92, "pika-ref2va-6", VideoOutputAudio, 73, 4, 2, 0, 7, {5, 6}, {}, {}, {}};
  VideoReference video{VideoReferenceKind::Video, 2, 2, 2, {}, {0.5f, -0.25f, 1.0f, 0.0f, -1.0f, 0.125f}};
  for (uint8_t index = 0; index < 24; ++index) video.pixels.push_back(static_cast<uint8_t>(3 * index));
  VideoReference image{VideoReferenceKind::Image, 1, 2, 1, {9, 8, 7, 6, 5, 4}, {}};
  VideoReference audio{VideoReferenceKind::Audio, 0, 0, 0, {}, {0.75f, -0.5f}};
  request.references = {video, image, audio};
  return request;
}

const std::vector<std::pair<std::string, VideoClientMessage>> &clientMessages() {
  static const std::vector<std::pair<std::string, VideoClientMessage>> messages{
      {"video_request", kVideoRequest},
      // The default DiT's clip without audio or preview.
      {"video_request_default",
       VideoRequestFrame{92, "default", 0, 73, 1344, 768, 50, 42,
                         {0xffffffffU}, {}, {}, {}}},
      // A clip of a 4 x 2 canvas from its first and last frames.
      {"video_request_keyframes", exampleKeyframeRequest()},
      // A Ref2VA clip from a video with its soundtrack, an image and a sound.
      {"video_request_references", exampleReferenceRequest()},
      {"cancel", CancelFrame{92}},
  };
  return messages;
}

const std::vector<std::pair<std::string, VideoEvent>> &videoEvents() {
  static const std::vector<std::pair<std::string, VideoEvent>> events{
      {"video_ready", VideoReadyEvent{1, 2048}},
      {"video_progress",
       VideoProgressEvent{92, VideoPhase::Denoise, 3, 8, 312'000'000, 15'200'000'000}},
      {"video_output",
       VideoOutputEvent{92, VideoOutputKind::Video, 73, 345'000'000,
                        {0, 0, 0, 24, 'f', 't', 'y', 'p'}}},
      {"error_request",
       ErrorEvent{FailureClass::RequestError, 92, false, "cancelled",
                  "the video request was cancelled"}},
      {"error_engine",
       ErrorEvent{FailureClass::EngineUnhealthy, 0, false,
                  "video_engine_failed", "the video VAE failed"}},
      {"start_failed",
       StartFailedEvent{StartFailureKind::Device, {},
                        "the video engine runs on an Apple10 GPU (M5) or later"}},
  };
  return events;
}

std::map<std::string, std::vector<uint8_t>> readGolden(const char *path) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error(std::string("cannot read ") + path);
  std::map<std::string, std::vector<uint8_t>> golden;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line.front() == '#')
      continue;
    std::istringstream fields(line);
    std::string name;
    std::string hex;
    fields >> name >> hex;
    std::vector<uint8_t> bytes;
    for (size_t index = 0; index + 1 < hex.size(); index += 2)
      bytes.push_back(
          static_cast<uint8_t>(std::stoul(hex.substr(index, 2), nullptr, 16)));
    golden.emplace(std::move(name), std::move(bytes));
  }
  return golden;
}

std::vector<VideoFrame> parseAll(const std::vector<uint8_t> &bytes,
                                 const VideoLimits &limits) {
  VideoFrameParser parser(limits);
  std::vector<VideoFrame> frames;
  size_t offset = 0;
  while (offset < bytes.size()) {
    VideoParseStep step =
        parser.consume(std::span<const uint8_t>(bytes).subspan(offset));
    if (step.issue)
      throw std::runtime_error(step.issue->describe());
    if (!step.consumedBytes)
      throw std::runtime_error("parser made no progress");
    offset += step.consumedBytes;
    if (step.frame)
      frames.push_back(std::move(*step.frame));
  }
  if (auto issue = parser.finish())
    throw std::runtime_error(issue->describe());
  return frames;
}

// The engine's view of one frame the server wrote.
ProtocolResult<VideoClientMessage> decodeClient(const std::vector<uint8_t> &bytes,
                                                const VideoLimits &limits) {
  std::vector<VideoFrame> frames = parseAll(bytes, limits);
  if (frames.size() != 1)
    throw std::runtime_error("expected one frame");
  return decodeVideoFrame(std::move(frames.front()), limits);
}

ProtocolIssue parserIssue(const std::vector<uint8_t> &bytes,
                          const VideoLimits &limits) {
  VideoFrameParser parser(limits);
  size_t offset = 0;
  while (offset < bytes.size()) {
    VideoParseStep step =
        parser.consume(std::span<const uint8_t>(bytes).subspan(offset));
    offset += step.consumedBytes;
    if (step.issue)
      return *step.issue;
    if (!step.consumedBytes)
      break;
  }
  if (auto issue = parser.finish())
    return *issue;
  throw std::runtime_error("expected a parser issue");
}

// Each client frame the server writes decodes to its message and is what this
// test's writer writes, and each event serializes to exactly the bytes the
// server decodes.
void testGoldenVectors(const char *path) {
  constexpr std::string_view test = "golden vectors";
  const auto golden = readGolden(path);
  CHECK(test, golden.size() == clientMessages().size() + videoEvents().size());
  for (const auto &[name, expected] : clientMessages()) {
    const auto found = golden.find(name);
    const auto decoded = found == golden.end()
                             ? ProtocolResult<VideoClientMessage>{}
                             : decodeClient(found->second, kLimits);
    check(decoded && *decoded.value == expected, name, test, __LINE__);
    check(found != golden.end() && clientFrame(expected) == found->second,
          name, test, __LINE__);
  }
  for (const auto &[name, expected] : videoEvents()) {
    const auto found = golden.find(name);
    const auto serialized = serializeVideoEvent(expected, kLimits);
    check(found != golden.end() && serialized &&
              *serialized.value == found->second,
          name, test, __LINE__);
  }
}

// Client frames arrive one after another on one stream; the video events
// read back as they were sent.
void testStreamsAndRoundTrips() {
  constexpr std::string_view test = "streams and round trips";
  std::vector<uint8_t> clients;
  for (const auto &[name, message] : clientMessages()) {
    const auto bytes = clientFrame(message);
    clients.insert(clients.end(), bytes.begin(), bytes.end());
  }
  std::vector<VideoFrame> frames = parseAll(clients, kLimits);
  CHECK(test, frames.size() == clientMessages().size());
  for (size_t index = 0;
       index < std::min(frames.size(), clientMessages().size()); ++index) {
    auto decoded = decodeVideoFrame(std::move(frames[index]), kLimits);
    CHECK(test, decoded && *decoded.value == clientMessages()[index].second);
  }
  for (const VideoEvent &event :
       {VideoEvent{VideoReadyEvent{1, 4096}},
        VideoEvent{VideoProgressEvent{7, VideoPhase::Mux, 1, 1, 9, 10}},
        VideoEvent{VideoOutputEvent{7, VideoOutputKind::Preview, 25, 11, {1, 2, 3}}}}) {
    const auto serialized = serializeVideoEvent(event, kLimits);
    CHECK(test, serialized && decodeEvent(*serialized.value) == event);
  }
}

// One byte at a time, the parser yields the frame once, at its last byte.
void testOneByteIncrementalParsing() {
  constexpr std::string_view test = "one-byte incremental parser";
  const VideoRequestFrame request = exampleReferenceRequest();
  VideoFrameParser parser(kLimits);
  std::optional<VideoFrame> frame;
  for (uint8_t byte : clientFrame(request)) {
    std::array<uint8_t, 1> input{byte};
    VideoParseStep step = parser.consume(input);
    CHECK(test, !step.issue);
    CHECK(test, step.consumedBytes == 1);
    if (step.frame) {
      CHECK(test, !frame);
      frame = std::move(step.frame);
    }
  }
  CHECK(test, frame.has_value());
  CHECK(test, !parser.finish());
  if (!frame)
    return;
  auto decoded = decodeVideoFrame(std::move(*frame), kLimits);
  CHECK(test, decoded && std::get<VideoRequestFrame>(*decoded.value) == request);
}

// Error and StartFailed are the engine protocol's frames under the video
// stream's magic and version, and Cancel decodes as the engine protocol
// decodes it.
template <typename Event> void checkCarried(std::string_view test, const Event &event) {
  const auto video = serializeVideoEvent(event, kLimits);
  const auto shared = protocol::serializeEvent(event, kLimits.shared);
  CHECK(test, video && shared && video.value->size() == shared.value->size());
  if (!video || !shared || video.value->size() != shared.value->size())
    return;
  CHECK(test, std::equal(kVideoMagic.begin(), kVideoMagic.end(), video.value->begin()));
  CHECK(test, loadU16(*video.value, 4) == kVideoProtocolVersion);
  CHECK(test, std::equal(video.value->begin() + 6, video.value->end(),
                         shared.value->begin() + 6));
}

void testEngineProtocolPayloads() {
  constexpr std::string_view test = "engine protocol payloads";
  for (const auto &[name, event] : videoEvents()) {
    if (const auto *error = std::get_if<ErrorEvent>(&event))
      checkCarried(test, *error);
    if (const auto *failure = std::get_if<StartFailedEvent>(&event))
      checkCarried(test, *failure);
  }
  // The engine protocol's rules hold for them in the video stream too.
  for (const VideoEvent &invalid :
       {VideoEvent{ErrorEvent{FailureClass::RequestError, 0, false, "bad", ""}},
        VideoEvent{ErrorEvent{FailureClass::RequestError, 5, false, "bad", "", 60}}}) {
    const auto serialized = serializeVideoEvent(invalid, kLimits);
    CHECK(test, !serialized && serialized.issue &&
                    serialized.issue->code == IssueCode::InvalidErrorClassification &&
                    serialized.issue->failureClass == FailureClass::EngineUnhealthy);
  }
  StartFailedEvent unknownKind;
  unknownKind.kind = static_cast<StartFailureKind>(9);
  const auto refused = serializeVideoEvent(unknownKind, kLimits);
  CHECK(test, !refused && refused.issue &&
                  refused.issue->code == IssueCode::InvalidEnumValue);
  // A Cancel names its request: without one the stream closes.
  const auto unnamed = decodeClient(clientFrame(CancelFrame{0}), kLimits);
  CHECK(test, !unnamed && unnamed.issue &&
                  unnamed.issue->failureClass == FailureClass::ProtocolFatal &&
                  unnamed.issue->code == IssueCode::InvalidRequestId);
}

// A request's error carries its id, so the stream survives it.
void expectRequestIssue(std::string_view test, const VideoRequestFrame &invalid,
                        IssueCode code) {
  auto decoded = decodeClient(clientFrame(invalid), kLimits);
  CHECK(test, !decoded);
  if (decoded.issue) {
    CHECK(test, decoded.issue->failureClass == FailureClass::RequestError);
    CHECK(test, decoded.issue->code == code);
    CHECK(test, decoded.issue->requestId == invalid.requestId);
  }
}

// An event that breaks its rules is the engine's defect.
void expectEventIssue(std::string_view test, const VideoEvent &invalid,
                      IssueCode code) {
  auto serialized = serializeVideoEvent(invalid, kLimits);
  CHECK(test, !serialized);
  if (serialized.issue) {
    CHECK(test, serialized.issue->failureClass == FailureClass::EngineUnhealthy);
    CHECK(test, serialized.issue->code == code);
  }
}

// A video request needs a prompt, frames and a DiT name; its progress
// a step within its steps, its output an mp4; Ready its capacities.
void testVideoFrameRejections() {
  constexpr std::string_view test = "video frame rejections";
  VideoRequestFrame empty = kVideoRequest, noFrames = empty, unnamed = empty, parent = empty, upper = empty;
  empty.promptTokens.clear();
  noFrames.frames = 0;
  unnamed.dit.clear();
  parent.dit = "..";
  upper.dit = "Turbo";
  expectRequestIssue(test, empty, IssueCode::LimitExceeded);
  expectRequestIssue(test, noFrames, IssueCode::InvalidCount);
  for (const VideoRequestFrame &name : {unnamed, parent, upper})
    expectRequestIssue(test, name, IssueCode::InvalidEnumValue);
  // A keyframe holds the canvas's height x width x 3 bytes.
  VideoRequestFrame keyframe = exampleKeyframeRequest();
  keyframe.lastFrame.pop_back();
  expectRequestIssue(test, keyframe, IssueCode::InvalidPayloadLength);
  // A reference holds frames x height x width x 3 bytes; an image is one
  // frame without a soundtrack; a kind is an image, a video or a sound.
  VideoRequestFrame pixels = exampleReferenceRequest(), imageFrames = pixels,
                    imageSound = pixels, kind = pixels, tooMany = pixels;
  pixels.references[0].pixels.pop_back();
  expectRequestIssue(test, pixels, IssueCode::InvalidPayloadLength);
  imageFrames.references[1].frames = 2;
  imageFrames.references[1].pixels.resize(12);
  expectRequestIssue(test, imageFrames, IssueCode::InvalidCount);
  imageSound.references[1].soundtrack = {0.0f, 0.0f};
  expectRequestIssue(test, imageSound, IssueCode::InvalidCount);
  kind.references[1].kind = static_cast<VideoReferenceKind>(3);
  expectRequestIssue(test, kind, IssueCode::InvalidEnumValue);
  tooMany.references.resize(kMaxVideoReferences + 1, tooMany.references[1]);
  expectRequestIssue(test, tooMany, IssueCode::LimitExceeded);
  // An audio reference is a soundtrack alone.
  VideoRequestFrame silent = exampleReferenceRequest(), framed = silent;
  silent.references[2].soundtrack.clear();
  expectRequestIssue(test, silent, IssueCode::InvalidCount);
  framed.references[2].frames = 1;
  expectRequestIssue(test, framed, IssueCode::InvalidCount);
  // Without its id, a request's failure closes the stream.
  VideoRequestFrame anonymous = kVideoRequest;
  anonymous.requestId = 0;
  const auto closed = decodeClient(clientFrame(anonymous), kLimits);
  CHECK(test, !closed && closed.issue &&
                  closed.issue->failureClass == FailureClass::ProtocolFatal &&
                  closed.issue->code == IssueCode::InvalidRequestId);

  expectEventIssue(test, VideoProgressEvent{92, VideoPhase::Denoise, 9, 8},
                   IssueCode::InvalidCount);
  expectEventIssue(test, VideoProgressEvent{0, VideoPhase::Denoise, 1, 8},
                   IssueCode::InvalidRequestId);
  expectEventIssue(test, VideoProgressEvent{92, static_cast<VideoPhase>(8), 1, 8},
                   IssueCode::InvalidEnumValue);
  expectEventIssue(test, VideoOutputEvent{92, VideoOutputKind::Video, 73, 0, {}},
                   IssueCode::LimitExceeded);
  expectEventIssue(test, VideoOutputEvent{92, static_cast<VideoOutputKind>(2), 73, 0, {1}},
                   IssueCode::InvalidEnumValue);
  expectEventIssue(test, VideoReadyEvent{0, 2048}, IssueCode::InvalidCount);
  expectEventIssue(test, VideoReadyEvent{1, 0}, IssueCode::InvalidCount);
}

// A request's DiT name and keyframes, as their bytes say them: the name's
// length (payload byte 8) past the bytes that follow, a byte outside
// validDitName's (the name at byte 43), or a keyframe bit the protocol does
// not define (byte 34).
void testMalformedPayloadClassification() {
  constexpr std::string_view test = "malformed payload classification";
  auto expect = [&](const std::vector<uint8_t> &bytes, IssueCode code) {
    auto decoded = decodeClient(bytes, kLimits);
    CHECK(test, !decoded && decoded.issue);
    if (decoded.issue) {
      CHECK(test, decoded.issue->failureClass == FailureClass::RequestError);
      CHECK(test, decoded.issue->code == code);
    }
  };
  auto longName = clientFrame(kVideoRequest);
  longName[kFrameHeaderBytes + 8] = 0xff;
  expect(longName, IssueCode::InvalidPayloadLength);
  auto highName = clientFrame(kVideoRequest);
  highName[kFrameHeaderBytes + 43] = 0x80;
  expect(highName, IssueCode::InvalidEnumValue);
  auto keyframes = clientFrame(kVideoRequest);
  keyframes[kFrameHeaderBytes + 34] = 4;
  expect(keyframes, IssueCode::InvalidEnumValue);
}

// The issue of the engine protocol's parser at a frame of the video stream.
ProtocolIssue engineParserIssue(const std::vector<uint8_t> &bytes) {
  protocol::FrameParser parser(protocol::ProtocolLimits{});
  protocol::ParseStep step = parser.consume(bytes);
  if (step.issue)
    return *step.issue;
  if (auto issue = parser.finish())
    return *issue;
  throw std::runtime_error("expected a parser issue");
}

void testHeaderFailures() {
  constexpr std::string_view test = "fatal header validation";
  const std::vector<uint8_t> valid = clientFrame(kVideoRequest);

  auto expect = [&](std::vector<uint8_t> bytes, IssueCode code,
                    const VideoLimits &limits = kLimits) {
    ProtocolIssue issue = parserIssue(bytes, limits);
    CHECK(test, issue.failureClass == FailureClass::ProtocolFatal);
    CHECK(test, issue.code == code);
  };

  auto badMagic = valid;
  badMagic[0] = 'X';
  expect(std::move(badMagic), IssueCode::BadMagic);

  auto version = valid;
  storeU16(version, 4, 2);
  expect(std::move(version), IssueCode::UnsupportedVersion);

  auto unknownType = valid;
  storeU16(unknownType, 6, 0x7777);
  expect(std::move(unknownType), IssueCode::UnknownFrameType);

  // The engine protocol's other client frames are not the video stream's.
  auto request = valid;
  storeU16(request, 6, 0x0001);
  expect(std::move(request), IssueCode::UnknownFrameType);

  // The engine never receives the frames it sends.
  for (const auto &[name, event] : videoEvents()) {
    auto bytes = serializeVideoEvent(event, kLimits);
    CHECK(test, bytes);
    if (bytes)
      expect(std::move(*bytes.value), IssueCode::UnknownFrameType);
  }

  auto enormous = valid;
  storeU64(enormous, 8, std::numeric_limits<uint64_t>::max());
  expect(std::move(enormous), IssueCode::FrameTooLarge);

  auto tooShort = valid;
  storeU64(tooShort, 8, 42);
  expect(std::move(tooShort), IssueCode::InvalidPayloadLength);

  // A video request's bound is the video stream's limit, 4 GiB by default.
  VideoLimits small;
  small.maxVideoRequestPayloadBytes = valid.size() - kFrameHeaderBytes - 1;
  expect(valid, IssueCode::FrameTooLarge, small);
  auto large = valid;
  storeU64(large, 8, kAbsoluteMaxVideoRequestPayloadBytes + 1);
  expect(std::move(large), IssueCode::FrameTooLarge);

  auto longCancel = clientFrame(CancelFrame{92});
  longCancel.push_back(0);
  storeU64(longCancel, 8, 9);
  expect(std::move(longCancel), IssueCode::FrameTooLarge);
  auto shortCancel = clientFrame(CancelFrame{92});
  shortCancel.pop_back();
  storeU64(shortCancel, 8, 7);
  expect(std::move(shortCancel), IssueCode::InvalidPayloadLength);

  // A frame sent to the other engine fails at its magic, either way.
  const auto native = protocol::serializeEvent(
      ErrorEvent{FailureClass::RequestError, 92, false, "cancelled", ""},
      kLimits.shared);
  CHECK(test, native);
  if (native)
    expect(*native.value, IssueCode::BadMagic);
  const ProtocolIssue misrouted = engineParserIssue(clientFrame(CancelFrame{92}));
  CHECK(test, misrouted.failureClass == FailureClass::ProtocolFatal);
  CHECK(test, misrouted.code == IssueCode::BadMagic);

  std::vector<uint8_t> invalidMagic(kFrameHeaderBytes, 'r');
  expect(std::move(invalidMagic), IssueCode::BadMagic);

  // A caller that keeps feeding a failed parser is a bug.
  VideoFrameParser sticky(kLimits);
  auto corrupted = valid;
  corrupted[0] = 0;
  VideoParseStep first = sticky.consume(corrupted);
  CHECK(test, first.issue);
  bool refused = false;
  try {
    static_cast<void>(sticky.consume(valid));
  } catch (const std::logic_error &) {
    refused = true;
  }
  CHECK(test, refused);
}

void testTruncationAtEveryBoundary() {
  constexpr std::string_view test = "truncation boundaries";
  const std::vector<uint8_t> bytes = clientFrame(exampleReferenceRequest());
  for (size_t cut = 1; cut < bytes.size(); ++cut) {
    VideoFrameParser parser(kLimits);
    size_t offset = 0;
    while (offset < cut) {
      VideoParseStep step = parser.consume(
          std::span<const uint8_t>(bytes.data() + offset, cut - offset));
      CHECK(test, !step.issue);
      CHECK(test, step.consumedBytes > 0);
      if (step.issue || !step.consumedBytes)
        break;
      offset += step.consumedBytes;
    }
    auto issue = parser.finish();
    CHECK(test, issue);
    if (issue) {
      CHECK(test, issue->failureClass == FailureClass::ProtocolFatal);
      CHECK(test, issue->code == IssueCode::TruncatedFrame);
    }
  }
  auto claimsOneMore = clientFrame(kVideoRequest);
  storeU64(claimsOneMore, 8, loadU64(claimsOneMore, 8) + 1);
  CHECK(test, parserIssue(claimsOneMore, kLimits).code == IssueCode::TruncatedFrame);

  VideoFrameParser empty(kLimits);
  CHECK(test, !empty.finish());
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " video_protocol_golden.txt\n";
    return 2;
  }
  try {
    testGoldenVectors(argv[1]);
    testStreamsAndRoundTrips();
    testOneByteIncrementalParsing();
    testEngineProtocolPayloads();
    testVideoFrameRejections();
    testMalformedPayloadClassification();
    testHeaderFailures();
    testTruncationAtEveryBoundary();
  } catch (const std::exception &error) {
    ++failures;
    std::cerr << "UNCAUGHT EXCEPTION: " << error.what() << '\n';
  }

  if (failures) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "video protocol tests passed\n";
  return 0;
}

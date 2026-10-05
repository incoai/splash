#include "ProtocolPeer.hpp"
#include "TestChecks.hpp"
#include "engine/Protocol.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace splash::protocol;
using splash::ConstraintMode;
using splash::ImageSpan;
using splash::RequestIgnoreEndOfSequence;
using splash::SamplingParameters;
using splash::engine::EngineFinishReason;
using splash::engine::RequestPriority;
using splash::model::ExecutionLimits;
using splash::test::rejects;

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

const ProtocolLimits kLimits{};

// Payload offsets of the request's fixed fields, each the previous field's
// offset plus that field's size.
namespace request_offset {
constexpr size_t priority = 8;
constexpr size_t constraint = priority + 1;
constexpr size_t absoluteDeadline = constraint + 1;
constexpr size_t remainingDeadline = absoluteDeadline + 8;
constexpr size_t logicalMaxOutput = remainingDeadline + 8;
constexpr size_t promptCount = logicalMaxOutput + 4;
constexpr size_t imageSpanCount = promptCount + 4;
constexpr size_t temperature = imageSpanCount + 4;
constexpr size_t topP = temperature + 4;
constexpr size_t topK = topP + 4;
constexpr size_t presencePenalty = topK + 4;
constexpr size_t frequencyPenalty = presencePenalty + 4;
constexpr size_t repetitionPenalty = frequencyPenalty + 4;
constexpr size_t minP = repetitionPenalty + 4;
constexpr size_t seed = minP + 4;
constexpr size_t scoreCount = seed + 8;
constexpr size_t generationPrompt = scoreCount + 4;
constexpr size_t flags = generationPrompt + 4;
static_assert(flags + 4 == kRequestFixedBytes);
} // namespace request_offset

uint32_t loadU32(const std::vector<uint8_t> &bytes, size_t offset) {
  uint32_t result = 0;
  for (uint32_t index = 0; index < 4; ++index) {
    result |= static_cast<uint32_t>(bytes.at(offset + index)) << (index * 8);
  }
  return result;
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

void storeU32(std::vector<uint8_t> &bytes, size_t offset, uint32_t value) {
  for (uint32_t index = 0; index < 4; ++index) {
    bytes.at(offset + index) = static_cast<uint8_t>(value >> (index * 8));
  }
}

void storeU64(std::vector<uint8_t> &bytes, size_t offset, uint64_t value) {
  for (uint32_t index = 0; index < 8; ++index) {
    bytes.at(offset + index) = static_cast<uint8_t>(value >> (index * 8));
  }
}

std::vector<Frame> parseAll(const std::vector<uint8_t> &bytes,
                            const ProtocolLimits &limits) {
  FrameParser parser(limits);
  std::vector<Frame> frames;
  size_t offset = 0;
  while (offset < bytes.size()) {
    ParseStep step =
        parser.consume(std::span<const uint8_t>(bytes).subspan(offset));
    if (step.issue)
      throw std::runtime_error(step.issue->describe());
    if (!step.consumedBytes) {
      throw std::runtime_error("parser made no progress");
    }
    offset += step.consumedBytes;
    if (step.frame)
      frames.push_back(std::move(*step.frame));
  }
  if (auto issue = parser.finish()) {
    throw std::runtime_error(issue->describe());
  }
  return frames;
}

Frame singleFrame(const std::vector<uint8_t> &wire,
                  const ProtocolLimits &limits) {
  std::vector<Frame> frames = parseAll(wire, limits);
  if (frames.size() != 1) {
    throw std::runtime_error("expected one frame");
  }
  return std::move(frames.front());
}

// The engine's view of one frame the server wrote.
ProtocolResult<ClientMessage> decodeClient(const std::vector<uint8_t> &wire,
                                           const ProtocolLimits &limits) {
  return decodeFrame(singleFrame(wire, limits), limits);
}

template <typename T> T roundTrip(const T &message) {
  auto decoded = decodeClient(peer::serialize(message), kLimits);
  if (!decoded)
    throw std::runtime_error(decoded.issue->describe());
  return std::get<T>(std::move(*decoded.value));
}

template <typename T> T roundTripEvent(const T &event) {
  auto bytes = serializeEvent(event, kLimits);
  if (!bytes)
    throw std::runtime_error(bytes.issue->describe());
  std::vector<EngineEvent> events = peer::decodeEvents(*bytes.value);
  if (events.size() != 1)
    throw std::runtime_error("round trip did not produce one event");
  return std::get<T>(std::move(events.front()));
}

RequestFrame exampleRequest() {
  RequestFrame request;
  request.requestId = 0x0123456789abcdefULL;
  request.priority = RequestPriority::Foreground;
  request.absoluteDeadlineUnixMicros = 1'800'000'000'000'000ULL;
  request.remainingDeadlineMicros = 45'000'000;
  request.logicalMaxOutputTokens = 32'768;
  request.promptTokens = {0, 1, 42, 0x80000000U, 0xffffffffU};
  request.sampling = {0.8f, 0.95f, 32, 1.5f, -0.25f, 1.1f, 0.05f};
  request.sampling.seed = 0xfedcba9876543210ULL;
  request.constraint = ConstraintMode::TokenMask;
  request.generationPromptTokens = 2;
  return request;
}

RequestFrame exampleImageRequest() {
  RequestFrame request = exampleRequest();
  request.promptTokens = {7, 3, 9};
  request.imageSpans = {
      {1, 1, 2, 2, 0x1111222233334444ULL, 0x5555666677778888ULL}};
  request.imagePixels.resize(request.imageSpans[0].pixelBytes());
  for (size_t index = 0; index < request.imagePixels.size(); ++index) {
    request.imagePixels[index] = static_cast<uint8_t>(index * 7 + 1);
  }
  return request;
}

RequestFrame exampleScoreRequest() {
  RequestFrame request = exampleRequest();
  request.logicalMaxOutputTokens = 0;
  request.promptTokens = {5, 6, 7};
  request.sampling = {.seed = request.sampling.seed};
  request.constraint = ConstraintMode::None;
  request.scoreTokens = {101, 202, 303};
  return request;
}

RequestFrame exampleIgnoreEosRequest() {
  RequestFrame request = exampleRequest();
  request.constraint = ConstraintMode::None;
  request.flags = RequestIgnoreEndOfSequence;
  return request;
}

RequestFrame exampleProgressRequest() {
  RequestFrame request = exampleRequest();
  request.flags = kReturnProgressFlag;
  return request;
}

const std::vector<std::pair<std::string, ClientMessage>> &clientMessages() {
  static const std::vector<std::pair<std::string, ClientMessage>> messages{
      {"request", exampleRequest()},
      {"request_image", exampleImageRequest()},
      {"request_score", exampleScoreRequest()},
      {"request_ignore_eos", exampleIgnoreEosRequest()},
      {"request_progress", exampleProgressRequest()},
      {"cancel", CancelFrame{91}},
      {"mask_response",
       MaskResponseFrame{91, 7, {0xffffffffU, 0, 0xa5a5a5a5U}}},
      {"status_request", StatusRequestFrame{808}},
  };
  return messages;
}

const std::vector<std::pair<std::string, EngineEvent>> &engineEvents() {
  static const std::vector<std::pair<std::string, EngineEvent>> events{
      {"ready", ReadyEvent{4, 524'288, false}},
      {"ready_vision", ReadyEvent{4, 524'288, true}},
      {"start", StartEvent{91, 2, 4096}},
      {"prompt_progress", PromptProgressEvent{91, 2048, 123456}},
      {"tokens", TokensEvent{91, 17, {10, 11, 12}}},
      {"mask_request_initial", MaskRequestEvent{91, 6, 4, {}}},
      {"mask_request_verify", MaskRequestEvent{91, 7, 4, {101, 102, 103}}},
      {"done", DoneEvent{91, EngineFinishReason::Stop, 4096, 512, 1000, 2000,
                         3500, {}}},
      {"done_scored", DoneEvent{91, EngineFinishReason::Stop, 4096, 0, 1000, 0,
                                3500, {1.5f, -2.25f, 0.5f}}},
      {"error_request",
       ErrorEvent{FailureClass::RequestError, 91, true, "deadline_exceeded",
                  "request deadline expired"}},
      {"error_engine", ErrorEvent{FailureClass::EngineUnhealthy, 0, false,
                                  "gpu_fault", "Metal command buffer failed"}},
      {"error_protocol",
       ErrorEvent{FailureClass::ProtocolFatal, 0, false, "bad_frame",
                  "stream framing cannot be trusted"}},
      {"status_json",
       StatusJsonEvent{808, "{\n  \"schema_version\": 6, \"ready\": true\n}"}},
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

// Each client frame the server writes decodes to its message, and each event
// serializes to exactly the bytes the server decodes.
void testGoldenVectors(const char *path) {
  constexpr std::string_view test = "golden vectors";
  const auto golden = readGolden(path);
  CHECK(test, golden.size() == clientMessages().size() + engineEvents().size());
  for (const auto &[name, expected] : clientMessages()) {
    const auto found = golden.find(name);
    const auto decoded = found == golden.end()
                             ? ProtocolResult<ClientMessage>{}
                             : decodeClient(found->second, kLimits);
    check(decoded && *decoded.value == expected, name, test, __LINE__);
  }
  for (const auto &[name, expected] : engineEvents()) {
    const auto found = golden.find(name);
    const auto serialized = serializeEvent(expected, kLimits);
    check(found != golden.end() && serialized &&
              *serialized.value == found->second,
          name, test, __LINE__);
  }
}

void testRequestRoundTrip() {
  constexpr std::string_view test = "request round trip";
  RequestFrame request = exampleRequest();
  CHECK(test, roundTrip(request) == request);
  RequestFrame withImage = exampleImageRequest();
  CHECK(test, roundTrip(withImage) == withImage);
  RequestFrame progress = exampleProgressRequest();
  CHECK(test, roundTrip(progress) == progress);
}

// A request's error carries its id, so the stream survives it.
void expectRequestIssue(std::string_view test, const RequestFrame &invalid,
                        IssueCode code,
                        const ProtocolLimits &limits = kLimits) {
  auto decoded = decodeClient(peer::serialize(invalid), limits);
  CHECK(test, !decoded);
  if (decoded.issue) {
    CHECK(test, decoded.issue->failureClass == FailureClass::RequestError);
    CHECK(test, decoded.issue->code == code);
    CHECK(test, decoded.issue->requestId == invalid.requestId);
  }
}

// An event that breaks its rules is the engine's defect.
void expectEventIssue(std::string_view test, const EngineEvent &invalid,
                      IssueCode code, const ProtocolLimits &limits = kLimits) {
  auto serialized = serializeEvent(invalid, limits);
  CHECK(test, !serialized);
  if (serialized.issue) {
    CHECK(test,
          serialized.issue->failureClass == FailureClass::EngineUnhealthy);
    CHECK(test, serialized.issue->code == code);
  }
}

void testScoreRequestAndDoneLogits() {
  constexpr std::string_view test = "score request and done logits";
  RequestFrame request = exampleScoreRequest();
  CHECK(test, roundTrip(request) == request);

  RequestFrame emptyScores = request;
  emptyScores.scoreTokens.clear();
  emptyScores.logicalMaxOutputTokens = 16;
  CHECK(test, roundTrip(emptyScores) == emptyScores);

  RequestFrame tooFew = request;
  tooFew.scoreTokens = {32};
  expectRequestIssue(test, tooFew, IssueCode::InvalidCount);

  RequestFrame tooMany = request;
  tooMany.scoreTokens.resize(ExecutionLimits::maximumScoreOptions + 1);
  for (uint32_t index = 0; index < tooMany.scoreTokens.size(); ++index)
    tooMany.scoreTokens[index] = index;
  expectRequestIssue(test, tooMany, IssueCode::InvalidCount);

  RequestFrame duplicates = request;
  duplicates.scoreTokens = {32, 65, 32};
  expectRequestIssue(test, duplicates, IssueCode::InvalidCount);

  RequestFrame withOutput = request;
  withOutput.logicalMaxOutputTokens = 8;
  expectRequestIssue(test, withOutput, IssueCode::InvalidCount);

  RequestFrame withImage = request;
  withImage.imageSpans = {
      {0, 1, 2, 2, 0x1111222233334444ULL, 0x5555666677778888ULL}};
  withImage.imagePixels.resize(withImage.imageSpans[0].pixelBytes());
  expectRequestIssue(test, withImage, IssueCode::InvalidCount);

  RequestFrame constrained = request;
  constrained.constraint = ConstraintMode::TokenMask;
  expectRequestIssue(test, constrained, IssueCode::InvalidConstraint);

  RequestFrame sampling = request;
  sampling.sampling = {0.8f, 0.95f, 32};
  expectRequestIssue(test, sampling, IssueCode::InvalidSampling);

  // A score request reads raw logits: it may carry any seed, and no other
  // sampling option.
  RequestFrame seeded = request;
  seeded.sampling.seed = 99;
  CHECK(test, roundTrip(seeded) == seeded);
  RequestFrame topK = request;
  topK.sampling.topK = 5;
  expectRequestIssue(test, topK, IssueCode::InvalidSampling);
  RequestFrame topP = request;
  topP.sampling.topP = 0.9f;
  expectRequestIssue(test, topP, IssueCode::InvalidSampling);

  DoneEvent scored{91, EngineFinishReason::Stop, 4096, 0, 1000, 0, 3500,
                   {1.5f, -2.0f, 0.25f}};
  CHECK(test, roundTripEvent(scored) == scored);
  auto encodedDone = serializeEvent(scored, kLimits);
  CHECK(test, encodedDone);
  if (encodedDone) {
    CHECK(test, encodedDone.value->size() == kFrameHeaderBytes + 45 + 12);
    CHECK(test, loadU32(*encodedDone.value, kFrameHeaderBytes + 41) == 3);
  }

  DoneEvent generation{91, EngineFinishReason::Length, 10, 4, 1, 2, 3, {}};
  CHECK(test, roundTripEvent(generation) == generation);

  DoneEvent cancelled{91, EngineFinishReason::Cancelled, 10, 0, 1, 0, 3, {}};
  CHECK(test, roundTripEvent(cancelled) == cancelled);

  DoneEvent oneLogit = scored;
  oneLogit.optionLogits = {1.0f};
  expectEventIssue(test, oneLogit, IssueCode::InvalidCount);

  DoneEvent nanLogit = scored;
  nanLogit.optionLogits = {1.0f, std::numeric_limits<float>::quiet_NaN()};
  expectEventIssue(test, nanLogit, IssueCode::InvalidCount);

  DoneEvent infLogit = scored;
  infLogit.optionLogits = {1.0f, std::numeric_limits<float>::infinity()};
  expectEventIssue(test, infLogit, IssueCode::InvalidCount);

  DoneEvent withCompletion = scored;
  withCompletion.completionTokens = 1;
  expectEventIssue(test, withCompletion, IssueCode::InvalidCount);

  DoneEvent withDecode = scored;
  withDecode.decodeMicros = 5;
  expectEventIssue(test, withDecode, IssueCode::InvalidCount);

  DoneEvent cancelledScored = scored;
  cancelledScored.reason = EngineFinishReason::Cancelled;
  expectEventIssue(test, cancelledScored, IssueCode::InvalidCount);

  DoneEvent lengthScored = scored;
  lengthScored.reason = EngineFinishReason::Length;
  expectEventIssue(test, lengthScored, IssueCode::InvalidCount);
}

// Client frames arrive one after another on one stream; the engine's events
// leave the same way.
void testClientAndEventStreams() {
  constexpr std::string_view test = "client and event streams";
  std::vector<uint8_t> clients;
  for (const auto &[name, message] : clientMessages()) {
    const auto wire = peer::serialize(message);
    clients.insert(clients.end(), wire.begin(), wire.end());
  }
  std::vector<Frame> frames = parseAll(clients, kLimits);
  CHECK(test, frames.size() == clientMessages().size());
  for (size_t index = 0;
       index < std::min(frames.size(), clientMessages().size()); ++index) {
    auto decoded = decodeFrame(std::move(frames[index]), kLimits);
    CHECK(test, decoded && *decoded.value == clientMessages()[index].second);
  }

  std::vector<uint8_t> events;
  for (const auto &[name, event] : engineEvents()) {
    auto wire = serializeEvent(event, kLimits);
    CHECK(test, wire);
    if (wire)
      events.insert(events.end(), wire.value->begin(), wire.value->end());
  }
  std::vector<EngineEvent> decoded = peer::decodeEvents(events);
  CHECK(test, decoded.size() == engineEvents().size());
  for (size_t index = 0;
       index < std::min(decoded.size(), engineEvents().size()); ++index)
    CHECK(test, decoded[index] == engineEvents()[index].second);
}

void testOneByteIncrementalParsing() {
  constexpr std::string_view test = "one-byte incremental parser";
  RequestFrame request = exampleRequest();
  FrameParser parser(kLimits);
  std::optional<Frame> frame;
  for (uint8_t byte : peer::serialize(request)) {
    std::array<uint8_t, 1> input{byte};
    ParseStep step = parser.consume(input);
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
  auto decoded = decodeFrame(std::move(*frame), kLimits);
  CHECK(test, decoded);
  if (decoded)
    CHECK(test, std::get<RequestFrame>(*decoded.value) == request);
}

ProtocolIssue parserIssue(const std::vector<uint8_t> &wire,
                          const ProtocolLimits &limits) {
  FrameParser parser(limits);
  size_t offset = 0;
  while (offset < wire.size()) {
    ParseStep step =
        parser.consume(std::span<const uint8_t>(wire).subspan(offset));
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

void testLargeIncrementalFrameHasNoGeometricCapacitySlack() {
  constexpr std::string_view test = "large incremental frame storage";
  RequestFrame expected = exampleRequest();
  expected.promptTokens.assign(256 * 1024 + 7, 11);
  const std::vector<uint8_t> wire = peer::serialize(expected);
  FrameParser parser(kLimits);
  size_t offset = 0;
  std::optional<Frame> frame;
  while (offset < wire.size()) {
    const size_t count = std::min<size_t>(64 * 1024, wire.size() - offset);
    auto step =
        parser.consume(std::span<const uint8_t>(wire).subspan(offset, count));
    CHECK(test, !step.issue);
    CHECK(test, step.consumedBytes == count);
    if (!step.consumedBytes)
      return;
    offset += step.consumedBytes;
    if (step.frame)
      frame = std::move(step.frame);
  }
  CHECK(test, frame);
  CHECK(test, !parser.finish());
  if (!frame)
    return;
  CHECK(test, frame->payload.capacity() == frame->payload.size());
  auto decoded = decodeFrame(std::move(*frame), kLimits);
  CHECK(test, decoded);
  if (decoded)
    CHECK(test, std::get<RequestFrame>(*decoded.value) == expected);
  auto step = parser.consume(peer::serialize(StatusRequestFrame{2}));
  CHECK(test, !step.issue && step.frame);
  if (step.frame)
    CHECK(test, step.frame->payload.size() == 8);
}

// A request's image pixels are most of its frame: decoding keeps them in the
// frame's buffer instead of copying them out.
void testImagePixelsMoveOutOfThePayload() {
  constexpr std::string_view test = "image pixels move out of the payload";
  RequestFrame expected = exampleImageRequest();
  expected.imageSpans[0].gridHeight = 64;
  expected.imageSpans[0].gridWidth = 32;
  expected.imageSpans[0].tokens = 32 * 16;
  expected.promptTokens.assign(expected.imageSpans[0].tokens + 2, 7);
  expected.imagePixels.resize(expected.imageSpans[0].pixelBytes());
  for (size_t index = 0; index < expected.imagePixels.size(); ++index)
    expected.imagePixels[index] = static_cast<uint8_t>(index * 13 + 5);
  CHECK(test, expected.imagePixels.size() >= 1024 * 1024);
  Frame frame = singleFrame(peer::serialize(expected), kLimits);
  const uint8_t *buffer = frame.payload.data();
  auto decoded = decodeFrame(std::move(frame), kLimits);
  CHECK(test, decoded);
  if (!decoded)
    return;
  const auto &request = std::get<RequestFrame>(*decoded.value);
  CHECK(test, request.imagePixels.data() == buffer);
  CHECK(test, request == expected);
}

void testHeaderFailures() {
  constexpr std::string_view test = "fatal header validation";
  const std::vector<uint8_t> valid = peer::serialize(exampleRequest());

  auto expect = [&](std::vector<uint8_t> wire, IssueCode code) {
    ProtocolIssue issue = parserIssue(wire, kLimits);
    CHECK(test, issue.failureClass == FailureClass::ProtocolFatal);
    CHECK(test, issue.code == code);
  };

  auto badMagic = valid;
  badMagic[0] = 'X';
  expect(std::move(badMagic), IssueCode::BadMagic);

  auto version = valid;
  storeU16(version, 4, 1);
  expect(std::move(version), IssueCode::UnsupportedVersion);

  auto unknownType = valid;
  storeU16(unknownType, 6, 0x7777);
  expect(std::move(unknownType), IssueCode::UnknownFrameType);

  // The engine never receives the frames it sends.
  for (const auto &[name, event] : engineEvents()) {
    auto wire = serializeEvent(event, kLimits);
    CHECK(test, wire);
    if (wire)
      expect(std::move(*wire.value), IssueCode::UnknownFrameType);
  }

  auto enormous = valid;
  storeU64(enormous, 8, std::numeric_limits<uint64_t>::max());
  expect(std::move(enormous), IssueCode::FrameTooLarge);

  auto tooShort = valid;
  storeU64(tooShort, 8, 54);
  expect(std::move(tooShort), IssueCode::InvalidPayloadLength);

  std::vector<uint8_t> invalidMagic(kFrameHeaderBytes, 'r');
  expect(std::move(invalidMagic), IssueCode::BadMagic);

  // A caller that keeps feeding a failed parser is a bug.
  FrameParser sticky(kLimits);
  auto corrupted = valid;
  corrupted[0] = 0;
  ParseStep first = sticky.consume(corrupted);
  CHECK(test, first.issue);
  rejects([&] { static_cast<void>(sticky.consume(valid)); },
          "native frame parser used after it failed", "a failed parser consumed more bytes");
}

void testTruncationAtEveryBoundary() {
  constexpr std::string_view test = "truncation boundaries";
  const std::vector<uint8_t> wire = peer::serialize(exampleRequest());
  for (size_t cut = 1; cut < wire.size(); ++cut) {
    FrameParser parser(kLimits);
    size_t offset = 0;
    while (offset < cut) {
      ParseStep step = parser.consume(
          std::span<const uint8_t>(wire.data() + offset, cut - offset));
      CHECK(test, !step.issue);
      CHECK(test, step.consumedBytes > 0);
      offset += step.consumedBytes;
    }
    auto issue = parser.finish();
    CHECK(test, issue);
    if (issue) {
      CHECK(test, issue->failureClass == FailureClass::ProtocolFatal);
      CHECK(test, issue->code == IssueCode::TruncatedFrame);
    }
  }

  FrameParser empty(kLimits);
  CHECK(test, !empty.finish());
}

void testMalformedPayloadClassification() {
  constexpr std::string_view test = "malformed payload classification";
  const std::vector<uint8_t> valid = peer::serialize(exampleRequest());
  auto expect = [&](const std::vector<uint8_t> &wire, IssueCode code) {
    auto decoded = decodeClient(wire, kLimits);
    CHECK(test, !decoded);
    if (decoded.issue) {
      CHECK(test, decoded.issue->failureClass == FailureClass::RequestError);
      CHECK(test, decoded.issue->code == code);
    }
  };

  auto invalidPriority = valid;
  invalidPriority[kFrameHeaderBytes + request_offset::priority] = 0xff;
  expect(invalidPriority, IssueCode::InvalidEnumValue);

  // A request error leaves the stream aligned on the next frame.
  std::vector<uint8_t> recoverableStream = invalidPriority;
  recoverableStream.insert(recoverableStream.end(), valid.begin(), valid.end());
  std::vector<Frame> recoverableFrames = parseAll(recoverableStream, kLimits);
  CHECK(test, recoverableFrames.size() == 2);
  if (recoverableFrames.size() == 2) {
    auto rejected = decodeFrame(std::move(recoverableFrames[0]), kLimits);
    auto accepted = decodeFrame(std::move(recoverableFrames[1]), kLimits);
    CHECK(test, rejected.issue &&
                    rejected.issue->failureClass == FailureClass::RequestError);
    CHECK(test, accepted);
  }

  auto hugePromptCount = valid;
  storeU32(hugePromptCount, kFrameHeaderBytes + request_offset::promptCount,
           std::numeric_limits<uint32_t>::max());
  expect(hugePromptCount, IssueCode::LimitExceeded);

  // A wrong count desynchronizes the tail and fails the request.
  auto shortScores = peer::serialize(exampleScoreRequest());
  storeU32(shortScores, kFrameHeaderBytes + request_offset::scoreCount, 2);
  expect(shortScores, IssueCode::InvalidPayloadLength);

  auto zeroDeadline = valid;
  storeU64(zeroDeadline, kFrameHeaderBytes + request_offset::absoluteDeadline,
           0);
  expect(zeroDeadline, IssueCode::InvalidDeadline);

  auto nanSampling = valid;
  storeU32(nanSampling, kFrameHeaderBytes + request_offset::temperature,
           0x7fc00000U);
  expect(nanSampling, IssueCode::InvalidSampling);

  auto undefinedConstraint = valid;
  undefinedConstraint[kFrameHeaderBytes + request_offset::constraint] = 2;
  expect(undefinedConstraint, IssueCode::InvalidEnumValue);
}

// Every rejection of a request's prompt or image spans is that request's own
// error.
void testPromptAndImageSpanRejections() {
  constexpr std::string_view test = "prompt and image span rejections";
  RequestFrame empty = exampleRequest();
  empty.promptTokens.clear();
  expectRequestIssue(test, empty, IssueCode::LimitExceeded);
  ProtocolLimits fourTokens;
  fourTokens.maxPromptTokens = 4;
  expectRequestIssue(test, exampleRequest(), IssueCode::LimitExceeded,
                     fourTokens);
  RequestFrame wholePrompt = exampleRequest();
  wholePrompt.generationPromptTokens = wholePrompt.promptTokens.size();
  expectRequestIssue(test, wholePrompt, IssueCode::InvalidCount);

  const RequestFrame image = exampleImageRequest();
  auto withSpan = [&](auto change) {
    RequestFrame result = image;
    change(result.imageSpans[0]);
    result.imagePixels.resize(result.imageSpans[0].pixelBytes());
    return result;
  };
  expectRequestIssue(test,
                     withSpan([](ImageSpan &span) { span.gridHeight = 3; }),
                     IssueCode::InvalidCount);
  expectRequestIssue(test, withSpan([](ImageSpan &span) {
                       span.gridHeight = span.gridWidth = span.tokens = 0;
                     }),
                     IssueCode::InvalidCount);
  // A grid at the patch limit, and the smallest even-sided one past it, in
  // prompts that hold their tokens.
  auto withGrid = [&](uint32_t gridWidth) {
    RequestFrame result = withSpan([&](ImageSpan &span) {
      span.gridHeight = 2;
      span.gridWidth = gridWidth;
      span.tokens = gridWidth / 2;
    });
    result.promptTokens.resize(1 + result.imageSpans[0].tokens, 7);
    return result;
  };
  const RequestFrame largest = withGrid(splash::ops::kMaximumImagePatches / 2);
  CHECK(test, roundTrip(largest) == largest);
  expectRequestIssue(test, withGrid(splash::ops::kMaximumImagePatches / 2 + 2),
                     IssueCode::InvalidCount);
  // A grid whose patch count would wrap to zero in 32 bits, as its pixel
  // bytes do in 64, with no tokens or pixels.
  const RequestFrame wrapped = withSpan([](ImageSpan &span) {
    span.gridHeight = span.gridWidth = 1U << 28;
    span.tokens = 0;
  });
  CHECK(test, wrapped.imagePixels.empty());
  expectRequestIssue(test, wrapped, IssueCode::InvalidCount);
  expectRequestIssue(test, withSpan([](ImageSpan &span) { span.tokens = 2; }),
                     IssueCode::InvalidCount);
  expectRequestIssue(test, withSpan([](ImageSpan &span) { span.offset = 3; }),
                     IssueCode::InvalidCount);
  RequestFrame overlapping = image;
  overlapping.imageSpans.push_back(image.imageSpans[0]);
  overlapping.imagePixels.resize(2 * image.imagePixels.size());
  expectRequestIssue(test, overlapping, IssueCode::InvalidCount);
  RequestFrame shortPixels = image;
  shortPixels.imagePixels.pop_back();
  expectRequestIssue(test, shortPixels, IssueCode::InvalidPayloadLength);
  RequestFrame twoImages = overlapping;
  twoImages.imageSpans[0].offset = 0;
  CHECK(test, roundTrip(twoImages) == twoImages);
  ProtocolLimits oneImage;
  oneImage.maxImageSpans = 1;
  expectRequestIssue(test, twoImages, IssueCode::LimitExceeded, oneImage);
}

// The sampling parameters are one block after the image span count:
// temperature, top_p, top_k, the presence, frequency and repetition
// penalties, min_p and the seed. A request takes any top_k and is refused a
// subnormal temperature, a penalty or a min_p outside its range, and as a
// score request anything but the defaults.
void testSamplingBlock() {
  constexpr std::string_view test = "sampling block";
  RequestFrame request = exampleRequest();
  CHECK(test, roundTrip(request) == request);

  auto expectInvalid = [&](const RequestFrame &invalid) {
    expectRequestIssue(test, invalid, IssueCode::InvalidSampling);
  };
  constexpr float nan = std::numeric_limits<float>::quiet_NaN();
  constexpr float inf = std::numeric_limits<float>::infinity();
  constexpr float normalMinimum = std::numeric_limits<float>::min();
  // The kernels divide by a sampling temperature, and Metal flushes a
  // subnormal one to zero.
  for (const float temperature :
       {std::numeric_limits<float>::denorm_min(), 1e-40f,
        std::nextafter(normalMinimum, 0.0f), -1.0f, nan, inf}) {
    RequestFrame invalid = request;
    invalid.sampling.temperature = temperature;
    expectInvalid(invalid);
  }
  for (const float temperature : {normalMinimum, 0.0f, -0.0f}) {
    RequestFrame valid = request;
    valid.sampling.temperature = temperature;
    CHECK(test, roundTrip(valid) == valid);
  }
  for (const float penalty : {-2.01f, 2.01f, nan, -inf}) {
    RequestFrame presence = request;
    presence.sampling.presencePenalty = penalty;
    expectInvalid(presence);
    RequestFrame frequency = request;
    frequency.sampling.frequencyPenalty = penalty;
    expectInvalid(frequency);
  }
  for (const float repetition : {0.0f, -0.0f, -1.0f, nan, inf}) {
    RequestFrame invalid = request;
    invalid.sampling.repetitionPenalty = repetition;
    expectInvalid(invalid);
  }
  for (const float minP : {-0.01f, 1.01f, nan, inf, -inf}) {
    RequestFrame invalid = request;
    invalid.sampling.minP = minP;
    expectInvalid(invalid);
  }
  // The limits themselves, the smallest and largest repetition, min_p's 0
  // and 1, and any top_k, 0 keeping every token.
  for (const SamplingParameters limit :
       {SamplingParameters{0.8f, 0.95f, 32, -2.0f, 2.0f,
                           std::numeric_limits<float>::denorm_min(), 0.0f},
        SamplingParameters{0.8f, 0.95f, 32, 2.0f, -2.0f,
                           std::numeric_limits<float>::max(), 1.0f},
        SamplingParameters{0.8f, 0.95f, 0}, SamplingParameters{0.8f, 1.0f, 1000},
        SamplingParameters{1.0f, 0.9f, 0xffffffffU}}) {
    RequestFrame valid = request;
    valid.sampling = limit;
    CHECK(test, roundTrip(valid) == valid);
  }

  // Greedy requests carry penalties and min_p too; score requests carry
  // none.
  RequestFrame greedy = exampleScoreRequest();
  greedy.scoreTokens.clear();
  greedy.logicalMaxOutputTokens = 16;
  greedy.sampling.presencePenalty = 1.5f;
  greedy.sampling.minP = 0.1f;
  CHECK(test, roundTrip(greedy) == greedy);
  for (size_t field = 0; field < 4; ++field) {
    RequestFrame score = exampleScoreRequest();
    float *values[] = {&score.sampling.presencePenalty,
                       &score.sampling.frequencyPenalty,
                       &score.sampling.repetitionPenalty,
                       &score.sampling.minP};
    *values[field] += 0.5f;
    expectInvalid(score);
  }
}

// The flags word follows the generation prompt count. Any request can ask
// for its prompt's progress, unconstrained generation can ignore
// end-of-sequence; an undefined bit, or that flag on a constrained or score
// request, is the request's own error.
void testRequestFlags() {
  constexpr std::string_view test = "request flags";
  RequestFrame request = exampleIgnoreEosRequest();
  CHECK(test, roundTrip(request) == request);
  RequestFrame progress = request;
  progress.flags |= kReturnProgressFlag;
  CHECK(test, roundTrip(progress) == progress);
  for (const uint32_t flags : {1U << 2, 1U << 31, 0xffffffffU}) {
    RequestFrame undefined = request;
    undefined.flags = flags;
    expectRequestIssue(test, undefined, IssueCode::InvalidEnumValue);
  }
  RequestFrame constrained = exampleRequest();
  constrained.flags = RequestIgnoreEndOfSequence;
  expectRequestIssue(test, constrained, IssueCode::InvalidConstraint);
  RequestFrame score = exampleScoreRequest();
  score.flags = RequestIgnoreEndOfSequence;
  expectRequestIssue(test, score, IssueCode::InvalidConstraint);
}

std::string jsonOfExactSize(size_t bytes) {
  if (bytes < 8)
    throw std::invalid_argument("JSON size is too small");
  return "{\"x\":\"" + std::string(bytes - 8, 'a') + "\"}";
}

// The status JSON is opaque and only its size is bounded; a client frame over
// the frame limit is refused at its header.
void testBoundedArbitraryStatusJsonAndFrames() {
  constexpr std::string_view test = "bounded status JSON and frames";
  ProtocolLimits limits;
  limits.maxFramePayloadBytes = 128 * 1024;
  limits.maxStatusJsonBytes = limits.maxFramePayloadBytes - 8;
  limits.maxErrorStringBytes = 1024;
  StatusJsonEvent maximum{
      99, jsonOfExactSize(static_cast<size_t>(limits.maxStatusJsonBytes))};
  auto encoded = serializeEvent(maximum, limits);
  CHECK(test, encoded);
  if (encoded) {
    std::vector<EngineEvent> decoded = peer::decodeEvents(*encoded.value);
    CHECK(test, decoded.size() == 1 && decoded.front() == EngineEvent{maximum});
  }

  StatusJsonEvent tooLarge = maximum;
  tooLarge.json.push_back(' ');
  expectEventIssue(test, tooLarge, IssueCode::LimitExceeded, limits);
  expectEventIssue(test, StatusJsonEvent{1, ""}, IssueCode::LimitExceeded,
                   limits);

  RequestFrame oversized = exampleRequest();
  oversized.promptTokens.assign(limits.maxFramePayloadBytes / 4, 7);
  ProtocolIssue issue = parserIssue(peer::serialize(oversized), limits);
  CHECK(test, issue.failureClass == FailureClass::ProtocolFatal);
  CHECK(test, issue.code == IssueCode::FrameTooLarge);
}

void testFailureTaxonomy() {
  constexpr std::string_view test = "failure taxonomy";
  for (ErrorEvent expected : {
           ErrorEvent{FailureClass::RequestError, 5, true, "busy",
                      "retry this request"},
           ErrorEvent{FailureClass::EngineUnhealthy, 0, false, "metal_error",
                      "replace engine"},
           ErrorEvent{FailureClass::ProtocolFatal, 0, false, "framing_error",
                      "close stream"},
       }) {
    CHECK(test, roundTripEvent(expected) == expected);
  }

  CHECK(test, !connectionMustClose(FailureClass::RequestError));
  CHECK(test, connectionMustClose(FailureClass::EngineUnhealthy));
  CHECK(test, connectionMustClose(FailureClass::ProtocolFatal));

  RequestFrame invalid = exampleRequest();
  invalid.sampling.topP = 0.0f;
  expectRequestIssue(test, invalid, IssueCode::InvalidSampling);

  expectEventIssue(test, ReadyEvent{0, 4096, false}, IssueCode::InvalidCount);
  expectEventIssue(test, StartEvent{8, ExecutionLimits::maximumBatchWidth, 0},
                   IssueCode::InvalidCount);
  expectEventIssue(test,
                   TokensEvent{8, std::numeric_limits<uint32_t>::max(), {1}},
                   IssueCode::IntegerOverflow);
  expectEventIssue(test,
                   ErrorEvent{FailureClass::RequestError, 0, false, "bad", ""},
                   IssueCode::InvalidErrorClassification);
}

// A client frame's failure is its request's error only when the frame names
// the request; with request id 0 it closes the stream, however else the
// frame is malformed.
void testUnnamedRequestFailuresAreFatal() {
  constexpr std::string_view test = "unnamed request failures";
  auto expect = [&](const ClientMessage &message, IssueCode code,
                    FailureClass failureClass,
                    const ProtocolLimits &limits = kLimits) {
    auto decoded = decodeClient(peer::serialize(message), limits);
    CHECK(test, !decoded && decoded.issue);
    if (decoded.issue) {
      CHECK(test, decoded.issue->failureClass == failureClass);
      CHECK(test, decoded.issue->code == code);
    }
  };
  RequestFrame unnamed = exampleRequest();
  unnamed.requestId = 0;
  expect(unnamed, IssueCode::InvalidRequestId, FailureClass::ProtocolFatal);
  ProtocolLimits fourTokens;
  fourTokens.maxPromptTokens = 4;
  expect(unnamed, IssueCode::LimitExceeded, FailureClass::ProtocolFatal,
         fourTokens);
  expect(CancelFrame{0}, IssueCode::InvalidRequestId,
         FailureClass::ProtocolFatal);
  expect(MaskResponseFrame{0, 7, {1}}, IssueCode::InvalidRequestId,
         FailureClass::ProtocolFatal);
  expect(MaskResponseFrame{91, 0, {1}}, IssueCode::InvalidRequestId,
         FailureClass::RequestError);
}

void testInvalidLimitsAndOuterTruncation() {
  constexpr std::string_view test = "invalid limits and outer truncation";
  ProtocolLimits invalid;
  invalid.maxFramePayloadBytes = std::numeric_limits<uint64_t>::max();
  const std::optional<ProtocolIssue> refused = validateLimits(invalid);
  CHECK(test, refused);
  if (refused) {
    CHECK(test, refused->failureClass == FailureClass::ProtocolFatal);
    CHECK(test, refused->code == IssueCode::LimitExceeded);
  }

  auto claimsOneMore = peer::serialize(exampleRequest());
  storeU64(claimsOneMore, 8, loadU64(claimsOneMore, 8) + 1);
  ProtocolIssue issue = parserIssue(claimsOneMore, kLimits);
  CHECK(test, issue.code == IssueCode::TruncatedFrame);
  CHECK(test, issue.failureClass == FailureClass::ProtocolFatal);
}

void testFuzzLikeInputsAndMutations() {
  constexpr std::string_view test = "fuzz-like malformed inputs";
  std::mt19937_64 random(0x5eed1234ULL);

  for (uint32_t iteration = 0; iteration < 3000; ++iteration) {
    size_t length = static_cast<size_t>(random() % 257);
    std::vector<uint8_t> bytes(length);
    for (uint8_t &byte : bytes)
      byte = static_cast<uint8_t>(random());

    FrameParser parser(kLimits);
    size_t offset = 0;
    uint32_t steps = 0;
    bool failed = false;
    while (offset < bytes.size() && !failed) {
      size_t chunk = std::min<size_t>(1 + random() % 31, bytes.size() - offset);
      ParseStep step = parser.consume(
          std::span<const uint8_t>(bytes.data() + offset, chunk));
      CHECK(test, step.consumedBytes <= chunk);
      failed = step.issue.has_value();
      if (!step.consumedBytes && !failed) {
        CHECK(test, false);
        break;
      }
      offset += step.consumedBytes;
      if (step.frame) {
        static_cast<void>(decodeFrame(std::move(*step.frame), kLimits));
      }
      if (++steps > 1024) {
        CHECK(test, false);
        break;
      }
    }
    if (!failed)
      static_cast<void>(parser.finish());
  }

  const std::vector<uint8_t> valid = peer::serialize(exampleRequest());
  for (uint32_t iteration = 0; iteration < 2000; ++iteration) {
    std::vector<uint8_t> mutated = valid;
    size_t mutations = 1 + random() % 4;
    for (size_t count = 0; count < mutations; ++count) {
      mutated[random() % mutated.size()] = static_cast<uint8_t>(random());
    }
    FrameParser parser(kLimits);
    size_t offset = 0;
    bool failed = false;
    while (offset < mutated.size() && !failed) {
      ParseStep step =
          parser.consume(std::span<const uint8_t>(mutated).subspan(offset));
      failed = step.issue.has_value();
      if (!step.consumedBytes && !failed) {
        CHECK(test, false);
        break;
      }
      offset += step.consumedBytes;
      if (step.frame) {
        static_cast<void>(decodeFrame(std::move(*step.frame), kLimits));
      }
    }
    if (!failed)
      static_cast<void>(parser.finish());
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " protocol_golden.txt\n";
    return 2;
  }
  try {
    testGoldenVectors(argv[1]);
    testRequestRoundTrip();
    testScoreRequestAndDoneLogits();
    testClientAndEventStreams();
    testOneByteIncrementalParsing();
    testLargeIncrementalFrameHasNoGeometricCapacitySlack();
    testImagePixelsMoveOutOfThePayload();
    testHeaderFailures();
    testTruncationAtEveryBoundary();
    testMalformedPayloadClassification();
    testPromptAndImageSpanRejections();
    testSamplingBlock();
    testRequestFlags();
    testBoundedArbitraryStatusJsonAndFrames();
    testFailureTaxonomy();
    testUnnamedRequestFailuresAreFatal();
    testInvalidLimitsAndOuterTruncation();
    testFuzzLikeInputsAndMutations();
  } catch (const std::exception &error) {
    ++failures;
    std::cerr << "UNCAUGHT EXCEPTION: " << error.what() << '\n';
  }

  if (failures) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "native protocol tests passed\n";
  return 0;
}

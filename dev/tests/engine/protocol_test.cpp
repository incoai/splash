#include "engine/Protocol.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace splash::protocol;

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

uint16_t loadU16(const std::vector<uint8_t> &bytes, size_t offset) {
  return static_cast<uint16_t>(bytes.at(offset)) |
         (static_cast<uint16_t>(bytes.at(offset + 1)) << 8);
}

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
                            const ProtocolLimits &limits = {}) {
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

template <typename T>
T roundTrip(const T &message, const ProtocolLimits &limits = {}) {
  auto bytes = serializeMessage(Message{message}, limits);
  if (!bytes)
    throw std::runtime_error(bytes.issue->describe());
  std::vector<Frame> frames = parseAll(*bytes.value, limits);
  if (frames.size() != 1) {
    throw std::runtime_error("round trip did not produce one frame");
  }
  auto decoded = decodeFrame(frames.front(), limits);
  if (!decoded)
    throw std::runtime_error(decoded.issue->describe());
  if (!std::holds_alternative<T>(*decoded.value)) {
    throw std::runtime_error("round trip produced the wrong message type");
  }
  return std::get<T>(std::move(*decoded.value));
}

Frame decodeSingleFrame(const std::vector<uint8_t> &wire);

RequestFrame exampleRequest() {
  RequestFrame request;
  request.requestId = 0x0123456789abcdefULL;
  request.priority = RequestPriority::Foreground;
  request.absoluteDeadlineUnixMicros = 1'800'000'000'000'000ULL;
  request.remainingDeadlineMicros = 45'000'000;
  request.logicalMaxOutputTokens = 32'768;
  request.promptTokens = {0, 1, 42, 0x80000000U, 0xffffffffU};
  request.sampling = {0.8f, 0.95f, 32};
  request.seed = 0xfedcba9876543210ULL;
  request.cohort = Cohort::Constrained;
  request.constraint = ConstraintMode::TokenMask;
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
  RequestFrame request;
  request.requestId = 0x1111111111111111ULL;
  request.priority = RequestPriority::Normal;
  request.absoluteDeadlineUnixMicros = 1'800'000'000'000'000ULL;
  request.remainingDeadlineMicros = 45'000'000;
  request.logicalMaxOutputTokens = 0;
  request.promptTokens = {1, 2, 3, 4};
  request.sampling = {0.0f, 1.0f, 0};
  request.seed = 7;
  request.cohort = Cohort::Greedy;
  request.constraint = ConstraintMode::None;
  request.scoreTokens = {32, 65, 97};
  return request;
}

void testRequestWireAndRoundTrip() {
  constexpr std::string_view test = "request wire and round trip";
  RequestFrame request = exampleRequest();
  auto serialized = serializeMessage(Message{request});
  CHECK(test, serialized);
  if (!serialized)
    return;
  const auto &wire = *serialized.value;

  CHECK(test, wire.size() ==
                  kFrameHeaderBytes + 64 + request.promptTokens.size() * 4);
  CHECK(test, std::string(wire.begin(), wire.begin() + 4) == "SPLH");
  CHECK(test, loadU16(wire, 4) == kProtocolVersion);
  CHECK(test, loadU16(wire, 6) == kFrameHeaderBytes);
  CHECK(test, loadU16(wire, 8) == static_cast<uint16_t>(FrameType::Request));
  CHECK(test, loadU16(wire, 10) == 0);
  CHECK(test, loadU64(wire, 12) == 64 + request.promptTokens.size() * 4);
  CHECK(test, loadU32(wire, 20) == 0);
  CHECK(test, loadU64(wire, kFrameHeaderBytes) == request.requestId);
  CHECK(test,
        loadU32(wire, kFrameHeaderBytes + 31) == request.promptTokens.size());
  CHECK(test, loadU32(wire, kFrameHeaderBytes + 35) == 0);
  CHECK(test, loadU32(wire, kFrameHeaderBytes + 60) == 0);
  CHECK(test, loadU32(wire, kFrameHeaderBytes + 64) == 0);
  CHECK(test, loadU32(wire, kFrameHeaderBytes + 64 + 16) == 0xffffffffU);
  RequestFrame decoded = roundTrip(request);
  CHECK(test, decoded == request);

  RequestFrame withImage = exampleImageRequest();
  auto imageWire = serializeMessage(Message{withImage});
  CHECK(test, imageWire);
  if (!imageWire)
    return;
  const size_t spanOffset =
      kFrameHeaderBytes + 64 + withImage.promptTokens.size() * 4;
  CHECK(test, imageWire.value->size() ==
                  spanOffset + 32 + withImage.imagePixels.size());
  CHECK(test, loadU32(*imageWire.value, kFrameHeaderBytes + 35) == 1);
  CHECK(test, loadU32(*imageWire.value, spanOffset) == 1);
  CHECK(test, loadU32(*imageWire.value, spanOffset + 4) == 1);
  CHECK(test, loadU32(*imageWire.value, spanOffset + 8) == 2);
  CHECK(test, loadU32(*imageWire.value, spanOffset + 12) == 2);
  CHECK(test, loadU64(*imageWire.value, spanOffset + 16) ==
                  withImage.imageSpans[0].digestLo);
  CHECK(test, roundTrip(withImage) == withImage);
}

void testScoreRequestAndDoneLogits() {
  constexpr std::string_view test = "score request and done logits";
  RequestFrame request = exampleScoreRequest();
  auto serialized = serializeMessage(Message{request});
  CHECK(test, serialized);
  if (!serialized)
    return;
  const auto &wire = *serialized.value;
  const size_t scoreOffset =
      kFrameHeaderBytes + 64 + request.promptTokens.size() * 4;
  CHECK(test, wire.size() == scoreOffset + request.scoreTokens.size() * 4);
  CHECK(test, loadU32(wire, kFrameHeaderBytes + 27) == 0);
  CHECK(test, loadU32(wire, kFrameHeaderBytes + 60) ==
                  request.scoreTokens.size());
  CHECK(test, roundTrip(request) == request);

  RequestFrame emptyScores = request;
  emptyScores.scoreTokens.clear();
  emptyScores.logicalMaxOutputTokens = 16;
  CHECK(test, roundTrip(emptyScores) == emptyScores);

  auto expectRequestIssue = [&](RequestFrame invalid, IssueCode code) {
    auto encoded = encodeMessage(Message{invalid});
    CHECK(test, !encoded);
    if (encoded.issue) {
      CHECK(test, encoded.issue->failureClass == FailureClass::RequestError);
      CHECK(test, encoded.issue->code == code);
      CHECK(test, encoded.issue->requestId == invalid.requestId);
    }
  };

  RequestFrame tooFew = request;
  tooFew.scoreTokens = {32};
  expectRequestIssue(tooFew, IssueCode::InvalidCount);

  RequestFrame tooMany = request;
  tooMany.scoreTokens.resize(kMaximumScoreOptions + 1);
  for (uint32_t index = 0; index < tooMany.scoreTokens.size(); ++index)
    tooMany.scoreTokens[index] = index;
  expectRequestIssue(tooMany, IssueCode::InvalidCount);

  auto oversizedWire = wire;
  oversizedWire.resize(scoreOffset + tooMany.scoreTokens.size() * 4);
  storeU32(oversizedWire, kFrameHeaderBytes + 60, tooMany.scoreTokens.size());
  storeU64(oversizedWire, 12, oversizedWire.size() - kFrameHeaderBytes);
  for (size_t index = 0; index < tooMany.scoreTokens.size(); ++index)
    storeU32(oversizedWire, scoreOffset + index * 4, tooMany.scoreTokens[index]);
  auto oversizedDecoded = decodeFrame(decodeSingleFrame(oversizedWire));
  CHECK(test, !oversizedDecoded);
  if (oversizedDecoded.issue) {
    CHECK(test, oversizedDecoded.issue->failureClass == FailureClass::RequestError);
    CHECK(test, oversizedDecoded.issue->code == IssueCode::InvalidCount);
    CHECK(test, oversizedDecoded.issue->requestId == request.requestId);
  }

  RequestFrame duplicates = request;
  duplicates.scoreTokens = {32, 65, 32};
  expectRequestIssue(duplicates, IssueCode::InvalidCount);

  RequestFrame withOutput = request;
  withOutput.logicalMaxOutputTokens = 8;
  expectRequestIssue(withOutput, IssueCode::InvalidCount);

  RequestFrame withImage = request;
  withImage.imageSpans = {
      {0, 1, 2, 2, 0x1111222233334444ULL, 0x5555666677778888ULL}};
  withImage.imagePixels.resize(withImage.imageSpans[0].pixelBytes());
  expectRequestIssue(withImage, IssueCode::InvalidCount);

  RequestFrame constrained = request;
  constrained.constraint = ConstraintMode::TokenMask;
  constrained.cohort = Cohort::Constrained;
  expectRequestIssue(constrained, IssueCode::InvalidCohortConstraint);

  RequestFrame sampling = request;
  sampling.sampling = {0.8f, 0.95f, 32};
  sampling.cohort = Cohort::Sampling;
  expectRequestIssue(sampling, IssueCode::InvalidSampling);

  DoneEvent scored{91, FinishReason::Stop, 4096, 0, 1000, 0, 3500, {1.5f, -2.0f, 0.25f}};
  CHECK(test, roundTrip(scored) == scored);

  auto encodedDone = serializeMessage(Message{scored});
  CHECK(test, encodedDone);
  if (encodedDone) {
    CHECK(test, encodedDone.value->size() == kFrameHeaderBytes + 45 + 12);
    CHECK(test, loadU32(*encodedDone.value, kFrameHeaderBytes + 41) == 3);
  }

  DoneEvent generation{91, FinishReason::Length, 10, 4, 1, 2, 3, {}};
  CHECK(test, roundTrip(generation) == generation);

  DoneEvent cancelled{91, FinishReason::Cancelled, 10, 0, 1, 0, 3, {}};
  CHECK(test, roundTrip(cancelled) == cancelled);

  auto expectDoneIssue = [&](DoneEvent invalid, IssueCode code) {
    auto encoded = encodeMessage(Message{invalid});
    CHECK(test, !encoded);
    if (encoded.issue) {
      CHECK(test, encoded.issue->failureClass == FailureClass::EngineUnhealthy);
      CHECK(test, encoded.issue->code == code);
    }
  };

  DoneEvent oneLogit = scored;
  oneLogit.optionLogits = {1.0f};
  expectDoneIssue(oneLogit, IssueCode::InvalidCount);

  DoneEvent nanLogit = scored;
  nanLogit.optionLogits = {1.0f, std::numeric_limits<float>::quiet_NaN()};
  expectDoneIssue(nanLogit, IssueCode::InvalidCount);

  DoneEvent infLogit = scored;
  infLogit.optionLogits = {1.0f, std::numeric_limits<float>::infinity()};
  expectDoneIssue(infLogit, IssueCode::InvalidCount);

  DoneEvent withCompletion = scored;
  withCompletion.completionTokens = 1;
  expectDoneIssue(withCompletion, IssueCode::InvalidCount);

  DoneEvent withDecode = scored;
  withDecode.decodeMicros = 5;
  expectDoneIssue(withDecode, IssueCode::InvalidCount);

  DoneEvent cancelledScored = scored;
  cancelledScored.reason = FinishReason::Cancelled;
  expectDoneIssue(cancelledScored, IssueCode::InvalidCount);

  DoneEvent lengthScored = scored;
  lengthScored.reason = FinishReason::Length;
  expectDoneIssue(lengthScored, IssueCode::InvalidCount);

  if (encodedDone) {
    auto truncated = *encodedDone.value;
    truncated.resize(truncated.size() - 4);
    storeU64(truncated, 12, truncated.size() - kFrameHeaderBytes);
    auto result = decodeFrame(decodeSingleFrame(truncated));
    CHECK(test, !result);
    if (result.issue) {
      CHECK(test, result.issue->failureClass == FailureClass::ProtocolFatal);
      CHECK(test, result.issue->code == IssueCode::InvalidPayloadLength);
    }

    auto lengthWire = *encodedDone.value;
    lengthWire[kFrameHeaderBytes + 8] =
        static_cast<uint8_t>(FinishReason::Length);
    auto lengthResult = decodeFrame(decodeSingleFrame(lengthWire));
    CHECK(test, !lengthResult);
    if (lengthResult.issue) {
      CHECK(test,
            lengthResult.issue->failureClass == FailureClass::ProtocolFatal);
      CHECK(test, lengthResult.issue->code == IssueCode::InvalidCount);
    }
  }
}


std::vector<Message> everyOtherMessage() {
  return {
      Message{exampleImageRequest()},
      CancelFrame{91},
      MaskResponseFrame{91, 7, {0xffffffffU, 0, 0xa5a5a5a5U}},
      StatusRequestFrame{808},
      ReadyEvent{1001, 4, 524'288,
                 FeatureCancellation | FeatureTokenMasks | FeatureStatusJson |
                     FeatureMultiplexing},
      StartEvent{91, CacheDisposition::PrefixHit, 2, 4096, 131'072},
      PromptProgressEvent{91, 2048, 123456},
      TokensEvent{91, 17, {10, 11, 12}},
      MaskRequestEvent{91, 6, 4, {}},
      MaskRequestEvent{91, 7, 4, {101, 102, 103}},
      DoneEvent{91, FinishReason::Stop, 4096, 512, 1000, 2000, 3500},
      ErrorEvent{FailureClass::RequestError, 91, true, "deadline_exceeded",
                 "request deadline expired"},
      ErrorEvent{FailureClass::EngineUnhealthy, 0, false, "gpu_fault",
                 "Metal command buffer failed"},
      ErrorEvent{FailureClass::ProtocolFatal, 0, false, "bad_frame",
                 "stream framing cannot be trusted"},
      CapacityExhaustedEvent{92, 40, 12, 50'000},
      StatusJsonEvent{808, kStatusSchemaVersion,
                      "{\n  \"schema_version\": 4, \"ready\": true\n}"},
  };
}

void testEveryMessageAndMultiplexedStream() {
  constexpr std::string_view test = "all messages and multiplexing";
  std::vector<Message> expected = everyOtherMessage();
  std::vector<uint8_t> stream;
  for (const Message &message : expected) {
    auto wire = serializeMessage(message);
    CHECK(test, wire);
    if (!wire)
      continue;
    stream.insert(stream.end(), wire.value->begin(), wire.value->end());
  }

  std::vector<Frame> frames = parseAll(stream);
  CHECK(test, frames.size() == expected.size());
  if (frames.size() != expected.size())
    return;
  for (size_t index = 0; index < frames.size(); ++index) {
    auto decoded = decodeFrame(frames[index]);
    CHECK(test, decoded);
    if (decoded)
      CHECK(test, *decoded.value == expected[index]);
  }
}

void testOneByteIncrementalParsing() {
  constexpr std::string_view test = "one-byte incremental parser";
  RequestFrame request = exampleRequest();
  auto wire = serializeMessage(Message{request});
  CHECK(test, wire);
  if (!wire)
    return;

  FrameParser parser;
  std::optional<Frame> frame;
  for (uint8_t byte : *wire.value) {
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
  auto decoded = decodeFrame(*frame);
  CHECK(test, decoded);
  if (decoded)
    CHECK(test, std::get<RequestFrame>(*decoded.value) == request);
}

ProtocolIssue parserIssue(const std::vector<uint8_t> &wire,
                          const ProtocolLimits &limits = {}) {
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
  StatusJsonEvent expected{1, kStatusSchemaVersion,
                           "\"" + std::string(1024 * 1024 + 7, 'a') + "\""};
  auto encoded = serializeMessage(Message{expected});
  CHECK(test, encoded);
  if (!encoded)
    return;
  FrameParser parser;
  size_t offset = 0;
  std::optional<Frame> frame;
  while (offset < encoded.value->size()) {
    const size_t count =
        std::min<size_t>(64 * 1024, encoded.value->size() - offset);
    auto step = parser.consume(
        std::span<const uint8_t>(*encoded.value).subspan(offset, count));
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
  auto decoded = decodeFrame(*frame);
  CHECK(test, decoded);
  if (decoded)
    CHECK(test, std::get<StatusJsonEvent>(*decoded.value) == expected);
  auto next = serializeMessage(Message{StatusRequestFrame{2}});
  auto step = parser.consume(*next.value);
  CHECK(test, !step.issue && step.frame);
  if (step.frame)
    CHECK(test, step.frame->payload.size() == 8);
}

void testHeaderFailures() {
  constexpr std::string_view test = "fatal header validation";
  auto serialized = serializeMessage(Message{exampleRequest()});
  CHECK(test, serialized);
  if (!serialized)
    return;
  const std::vector<uint8_t> valid = *serialized.value;

  auto expect = [&](std::vector<uint8_t> wire, IssueCode code) {
    ProtocolIssue issue = parserIssue(wire);
    CHECK(test, issue.failureClass == FailureClass::ProtocolFatal);
    CHECK(test, issue.code == code);
  };

  auto badMagic = valid;
  badMagic[0] = 'X';
  expect(std::move(badMagic), IssueCode::BadMagic);

  auto version = valid;
  storeU16(version, 4, 1);
  expect(std::move(version), IssueCode::UnsupportedVersion);

  auto headerSize = valid;
  storeU16(headerSize, 6, 23);
  expect(std::move(headerSize), IssueCode::InvalidHeaderSize);

  auto unknownType = valid;
  storeU16(unknownType, 8, 0x7777);
  expect(std::move(unknownType), IssueCode::UnknownFrameType);

  auto flags = valid;
  storeU16(flags, 10, 1);
  expect(std::move(flags), IssueCode::NonZeroHeaderFlags);

  auto reserved = valid;
  storeU32(reserved, 20, 1);
  expect(std::move(reserved), IssueCode::NonZeroReservedField);

  auto enormous = valid;
  storeU64(enormous, 12, std::numeric_limits<uint64_t>::max());
  expect(std::move(enormous), IssueCode::FrameTooLarge);

  auto tooShort = valid;
  storeU64(tooShort, 12, 54);
  expect(std::move(tooShort), IssueCode::InvalidPayloadLength);

  std::vector<uint8_t> invalidMagic(24, 'r');
  expect(std::move(invalidMagic), IssueCode::BadMagic);

  FrameParser sticky;
  auto corrupted = valid;
  corrupted[0] = 0;
  ParseStep first = sticky.consume(corrupted);
  CHECK(test, first.issue);
  ParseStep second = sticky.consume(valid);
  CHECK(test, second.issue);
  if (second.issue) {
    CHECK(test, second.issue->code == IssueCode::ParserAlreadyFailed);
    CHECK(test, second.consumedBytes == 0);
  }
}

void testTruncationAtEveryBoundary() {
  constexpr std::string_view test = "truncation boundaries";
  auto serialized = serializeMessage(Message{exampleRequest()});
  CHECK(test, serialized);
  if (!serialized)
    return;
  const std::vector<uint8_t> &wire = *serialized.value;

  for (size_t cut = 1; cut < wire.size(); ++cut) {
    FrameParser parser;
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

  FrameParser empty;
  CHECK(test, !empty.finish());
}

Frame decodeSingleFrame(const std::vector<uint8_t> &wire) {
  std::vector<Frame> frames = parseAll(wire);
  if (frames.size() != 1) {
    throw std::runtime_error("expected one frame");
  }
  return std::move(frames.front());
}

void testMalformedPayloadClassification() {
  constexpr std::string_view test = "malformed payload classification";
  auto serialized = serializeMessage(Message{exampleRequest()});
  CHECK(test, serialized);
  if (!serialized)
    return;

  auto invalidPriority = *serialized.value;
  invalidPriority[kFrameHeaderBytes + 8] = 0xff;
  auto priorityResult = decodeFrame(decodeSingleFrame(invalidPriority));
  CHECK(test, !priorityResult);
  if (priorityResult.issue) {
    CHECK(test,
          priorityResult.issue->failureClass == FailureClass::RequestError);
    CHECK(test, priorityResult.issue->code == IssueCode::InvalidEnumValue);
  }

  std::vector<uint8_t> recoverableStream = invalidPriority;
  recoverableStream.insert(recoverableStream.end(), serialized.value->begin(),
                           serialized.value->end());
  std::vector<Frame> recoverableFrames = parseAll(recoverableStream);
  CHECK(test, recoverableFrames.size() == 2);
  if (recoverableFrames.size() == 2) {
    auto rejected = decodeFrame(recoverableFrames[0]);
    auto accepted = decodeFrame(recoverableFrames[1]);
    CHECK(test, rejected.issue &&
                    rejected.issue->failureClass == FailureClass::RequestError);
    CHECK(test, accepted);
  }

  auto hugePromptCount = *serialized.value;
  storeU32(hugePromptCount, kFrameHeaderBytes + 31,
           std::numeric_limits<uint32_t>::max());
  auto countResult = decodeFrame(decodeSingleFrame(hugePromptCount));
  CHECK(test, !countResult);
  if (countResult.issue) {
    CHECK(test, countResult.issue->failureClass == FailureClass::RequestError);
    CHECK(test, countResult.issue->code == IssueCode::LimitExceeded);
  }

  auto zeroDeadline = *serialized.value;
  storeU64(zeroDeadline, kFrameHeaderBytes + 11, 0);
  auto deadlineResult = decodeFrame(decodeSingleFrame(zeroDeadline));
  CHECK(test, !deadlineResult);
  if (deadlineResult.issue) {
    CHECK(test,
          deadlineResult.issue->failureClass == FailureClass::RequestError);
    CHECK(test, deadlineResult.issue->code == IssueCode::InvalidDeadline);
  }

  auto nanSampling = *serialized.value;
  storeU32(nanSampling, kFrameHeaderBytes + 39, 0x7fc00000U);
  auto samplingResult = decodeFrame(decodeSingleFrame(nanSampling));
  CHECK(test, !samplingResult);
  if (samplingResult.issue) {
    CHECK(test,
          samplingResult.issue->failureClass == FailureClass::RequestError);
    CHECK(test, samplingResult.issue->code == IssueCode::InvalidSampling);
  }

  auto mismatchedConstraint = *serialized.value;
  mismatchedConstraint[kFrameHeaderBytes + 10] =
      static_cast<uint8_t>(ConstraintMode::None);
  auto cohortResult = decodeFrame(decodeSingleFrame(mismatchedConstraint));
  CHECK(test, !cohortResult);
  if (cohortResult.issue) {
    CHECK(test, cohortResult.issue->failureClass == FailureClass::RequestError);
    CHECK(test, cohortResult.issue->code == IssueCode::InvalidCohortConstraint);
  }

  auto tokenWire = serializeMessage(Message{TokensEvent{7, 0, {1, 2}}});
  CHECK(test, tokenWire);
  if (tokenWire) {
    storeU32(*tokenWire.value, kFrameHeaderBytes + 12, 3);
    auto result = decodeFrame(decodeSingleFrame(*tokenWire.value));
    CHECK(test, !result);
    if (result.issue) {
      CHECK(test, result.issue->failureClass == FailureClass::ProtocolFatal);
      CHECK(test, result.issue->code == IssueCode::InvalidPayloadLength);
    }
  }

  auto errorWire = serializeMessage(Message{
      ErrorEvent{FailureClass::RequestError, 7, false, "bad", "message"}});
  CHECK(test, errorWire);
  if (errorWire) {
    storeU32(*errorWire.value, kFrameHeaderBytes + 10,
             std::numeric_limits<uint32_t>::max());
    auto result = decodeFrame(decodeSingleFrame(*errorWire.value));
    CHECK(test, !result);
    if (result.issue) {
      CHECK(test, result.issue->failureClass == FailureClass::ProtocolFatal);
      CHECK(test, result.issue->code == IssueCode::InvalidPayloadLength);
    }
  }

  auto statusWire = serializeMessage(Message{
      StatusJsonEvent{0, kStatusSchemaVersion, "{\"schema_version\":4}"}});
  CHECK(test, statusWire);
  if (statusWire) {
    storeU32(*statusWire.value, kFrameHeaderBytes + 8, 2);
    auto result = decodeFrame(decodeSingleFrame(*statusWire.value));
    CHECK(test, !result);
    if (result.issue) {
      CHECK(test, result.issue->failureClass == FailureClass::ProtocolFatal);
      CHECK(test, result.issue->code == IssueCode::InvalidStatusSchema);
    }
  }
}

std::string jsonOfExactSize(size_t bytes) {
  if (bytes < 8)
    throw std::invalid_argument("JSON size is too small");
  return "{\"x\":\"" + std::string(bytes - 8, 'a') + "\"}";
}

void testBoundedArbitraryStatusJson() {
  constexpr std::string_view test = "bounded arbitrary status JSON";
  ProtocolLimits limits;
  limits.maxFramePayloadBytes = 128 * 1024;
  limits.maxStatusJsonBytes = limits.maxFramePayloadBytes - 12;
  limits.maxErrorStringBytes = 1024;
  limits.maxPromptTokens = 1024;
  limits.maxTokenBatch = 128;
  limits.maxSimulationTokens = 8;
  limits.maxMaskWords = 4096;

  StatusJsonEvent maximum{
      99, kStatusSchemaVersion,
      jsonOfExactSize(static_cast<size_t>(limits.maxStatusJsonBytes))};
  StatusJsonEvent decoded = roundTrip(maximum, limits);
  CHECK(test, decoded == maximum);

  StatusJsonEvent tooLarge = maximum;
  tooLarge.json.push_back(' ');
  auto rejected = serializeMessage(Message{tooLarge}, limits);
  CHECK(test, !rejected);
  if (rejected.issue) {
    CHECK(test, rejected.issue->failureClass == FailureClass::EngineUnhealthy);
    CHECK(test, rejected.issue->code == IssueCode::LimitExceeded);
  }

  auto normalWire = serializeMessage(
      Message{StatusJsonEvent{1, kStatusSchemaVersion, jsonOfExactSize(4096)}});
  CHECK(test, normalWire);
  if (normalWire) {
    ProtocolLimits tiny = limits;
    tiny.maxStatusJsonBytes = 100;
    ProtocolIssue issue = parserIssue(*normalWire.value, tiny);
    CHECK(test, issue.failureClass == FailureClass::ProtocolFatal);
    CHECK(test, issue.code == IssueCode::FrameTooLarge);
  }
}

void testFailureTaxonomyAndCapacityEvent() {
  constexpr std::string_view test = "failure taxonomy";
  for (ErrorEvent expected : {
           ErrorEvent{FailureClass::RequestError, 5, true, "busy",
                      "retry this request"},
           ErrorEvent{FailureClass::EngineUnhealthy, 0, false, "metal_error",
                      "replace engine"},
           ErrorEvent{FailureClass::ProtocolFatal, 0, false, "framing_error",
                      "close stream"},
       }) {
    ErrorEvent decoded = roundTrip(expected);
    CHECK(test, decoded == expected);
  }

  CHECK(test, !connectionMustClose(FailureClass::RequestError));
  CHECK(test, connectionMustClose(FailureClass::EngineUnhealthy));
  CHECK(test, connectionMustClose(FailureClass::ProtocolFatal));

  RequestFrame invalid = exampleRequest();
  invalid.sampling.topP = 0.0f;
  auto requestIssue = serializeMessage(Message{invalid});
  CHECK(test, !requestIssue);
  if (requestIssue.issue) {
    CHECK(test, requestIssue.issue->failureClass == FailureClass::RequestError);
  }

  auto unhealthy = serializeMessage(Message{ReadyEvent{0, 4, 4096, 0}});
  CHECK(test, !unhealthy);
  if (unhealthy.issue) {
    CHECK(test, unhealthy.issue->failureClass == FailureClass::EngineUnhealthy);
  }

  CapacityExhaustedEvent capacity{8, 24, 3, 100'000};
  CHECK(test, roundTrip(capacity) == capacity);

  TokensEvent overflow{8, std::numeric_limits<uint32_t>::max(), {1}};
  auto overflowResult = serializeMessage(Message{overflow});
  CHECK(test, !overflowResult);
  if (overflowResult.issue) {
    CHECK(test,
          overflowResult.issue->failureClass == FailureClass::EngineUnhealthy);
    CHECK(test, overflowResult.issue->code == IssueCode::IntegerOverflow);
  }
}

void testOverflowLimitsAndOuterTruncation() {
  constexpr std::string_view test = "overflow limits and outer truncation";
  ProtocolLimits invalid;
  invalid.maxFramePayloadBytes = std::numeric_limits<uint64_t>::max();
  FrameParser parser(invalid);
  CHECK(test, parser.failed());
  ParseStep step = parser.consume({});
  CHECK(test, step.issue);
  if (step.issue) {
    CHECK(test, step.issue->code == IssueCode::ParserAlreadyFailed);
  }

  auto invalidEncode = serializeMessage(Message{exampleRequest()}, invalid);
  CHECK(test, !invalidEncode);
  if (invalidEncode.issue) {
    CHECK(test,
          invalidEncode.issue->failureClass == FailureClass::ProtocolFatal);
    CHECK(test, invalidEncode.issue->code == IssueCode::LimitExceeded);
  }

  ProtocolLimits tight;
  tight.maxFramePayloadBytes = 1024;
  tight.maxStatusJsonBytes = 100;
  tight.maxErrorStringBytes = 700;
  tight.maxPromptTokens = 1000;
  tight.maxLogicalOutputTokens = 1000;
  tight.maxTokenBatch = 16;
  tight.maxSimulationTokens = 4;
  tight.maxMaskWords = 64;

  ErrorEvent oversizedEvent{FailureClass::EngineUnhealthy, 0, false,
                            std::string(700, 'c'), std::string(700, 'm')};
  auto oversizedEventResult = encodeMessage(Message{oversizedEvent}, tight);
  CHECK(test, !oversizedEventResult);
  if (oversizedEventResult.issue) {
    CHECK(test, oversizedEventResult.issue->failureClass ==
                    FailureClass::EngineUnhealthy);
    CHECK(test, oversizedEventResult.issue->code == IssueCode::FrameTooLarge);
  }

  RequestFrame oversizedRequest = exampleRequest();
  oversizedRequest.logicalMaxOutputTokens = 100;
  oversizedRequest.promptTokens.assign(300, 7);
  auto oversizedRequestResult = encodeMessage(Message{oversizedRequest}, tight);
  CHECK(test, !oversizedRequestResult);
  if (oversizedRequestResult.issue) {
    CHECK(test, oversizedRequestResult.issue->failureClass ==
                    FailureClass::RequestError);
    CHECK(test, oversizedRequestResult.issue->code == IssueCode::FrameTooLarge);
    CHECK(test, oversizedRequestResult.issue->requestId ==
                    oversizedRequest.requestId);
  }

  auto serialized = serializeMessage(Message{exampleRequest()});
  CHECK(test, serialized);
  if (serialized) {
    auto claimsOneMore = *serialized.value;
    storeU64(claimsOneMore, 12, loadU64(claimsOneMore, 12) + 1);
    ProtocolIssue issue = parserIssue(claimsOneMore);
    CHECK(test, issue.code == IssueCode::TruncatedFrame);
    CHECK(test, issue.failureClass == FailureClass::ProtocolFatal);
  }
}

void testFuzzLikeInputsAndMutations() {
  constexpr std::string_view test = "fuzz-like malformed inputs";
  std::mt19937_64 random(0x5eed1234ULL);

  for (uint32_t iteration = 0; iteration < 3000; ++iteration) {
    size_t length = static_cast<size_t>(random() % 257);
    std::vector<uint8_t> bytes(length);
    for (uint8_t &byte : bytes)
      byte = static_cast<uint8_t>(random());

    FrameParser parser;
    size_t offset = 0;
    uint32_t steps = 0;
    while (offset < bytes.size() && !parser.failed()) {
      size_t chunk = std::min<size_t>(1 + random() % 31, bytes.size() - offset);
      ParseStep step = parser.consume(
          std::span<const uint8_t>(bytes.data() + offset, chunk));
      CHECK(test, step.consumedBytes <= chunk);
      if (!step.consumedBytes && !step.issue) {
        CHECK(test, false);
        break;
      }
      offset += step.consumedBytes;
      if (step.frame) {
        static_cast<void>(decodeFrame(*step.frame));
      }
      if (++steps > 1024) {
        CHECK(test, false);
        break;
      }
    }
    static_cast<void>(parser.finish());
  }

  auto valid = serializeMessage(Message{exampleRequest()});
  CHECK(test, valid);
  if (!valid)
    return;
  for (uint32_t iteration = 0; iteration < 2000; ++iteration) {
    std::vector<uint8_t> mutated = *valid.value;
    size_t mutations = 1 + random() % 4;
    for (size_t count = 0; count < mutations; ++count) {
      mutated[random() % mutated.size()] = static_cast<uint8_t>(random());
    }
    FrameParser parser;
    size_t offset = 0;
    while (offset < mutated.size() && !parser.failed()) {
      ParseStep step =
          parser.consume(std::span<const uint8_t>(mutated).subspan(offset));
      if (!step.consumedBytes && !step.issue) {
        CHECK(test, false);
        break;
      }
      offset += step.consumedBytes;
      if (step.frame) {
        static_cast<void>(decodeFrame(*step.frame));
      }
    }
    static_cast<void>(parser.finish());
  }
}

} // namespace

int main() {
  try {
    testRequestWireAndRoundTrip();
    testScoreRequestAndDoneLogits();
    testEveryMessageAndMultiplexedStream();
    testOneByteIncrementalParsing();
    testLargeIncrementalFrameHasNoGeometricCapacitySlack();
    testHeaderFailures();
    testTruncationAtEveryBoundary();
    testMalformedPayloadClassification();
    testBoundedArbitraryStatusJson();
    testFailureTaxonomyAndCapacityEvent();
    testOverflowLimitsAndOuterTruncation();
    testFuzzLikeInputsAndMutations();
  } catch (const std::exception &error) {
    ++failures;
    std::cerr << "UNCAUGHT EXCEPTION: " << error.what() << '\n';
  }

  if (failures) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "native current protocol tests passed\n";
  return 0;
}

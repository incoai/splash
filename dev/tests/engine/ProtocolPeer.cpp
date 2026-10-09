#include "ProtocolPeer.hpp"

#include <algorithm>
#include <bit>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>

namespace splash::protocol::peer {
namespace {

constexpr std::string_view kMagic = "SPLH";

class Writer {
public:
  void u8(uint8_t value) { bytes_.push_back(value); }
  void u16(uint16_t value) { little(value, 2); }
  void u32(uint32_t value) { little(value, 4); }
  void u64(uint64_t value) { little(value, 8); }
  void f32(float value) { u32(std::bit_cast<uint32_t>(value)); }
  void words(const std::vector<uint32_t> &values) {
    for (uint32_t value : values)
      u32(value);
  }
  void raw(std::span<const uint8_t> values) {
    bytes_.insert(bytes_.end(), values.begin(), values.end());
  }
  [[nodiscard]] std::vector<uint8_t> take() { return std::move(bytes_); }

private:
  void little(uint64_t value, uint32_t count) {
    for (uint32_t index = 0; index < count; ++index)
      u8(static_cast<uint8_t>(value >> (8 * index)));
  }

  std::vector<uint8_t> bytes_;
};

class Reader {
public:
  explicit Reader(std::span<const uint8_t> bytes) : bytes_(bytes) {}

  [[nodiscard]] std::span<const uint8_t> bytes(size_t count) {
    if (count > bytes_.size() - offset_)
      throw std::runtime_error("native frame is truncated");
    const auto result = bytes_.subspan(offset_, count);
    offset_ += count;
    return result;
  }
  [[nodiscard]] uint8_t u8() { return static_cast<uint8_t>(little(1)); }
  [[nodiscard]] uint16_t u16() { return static_cast<uint16_t>(little(2)); }
  [[nodiscard]] uint32_t u32() { return static_cast<uint32_t>(little(4)); }
  [[nodiscard]] uint64_t u64() { return little(8); }
  [[nodiscard]] float f32() { return std::bit_cast<float>(u32()); }
  [[nodiscard]] std::vector<uint32_t> words(uint32_t count) {
    std::vector<uint32_t> values(count);
    for (uint32_t &value : values)
      value = u32();
    return values;
  }
  [[nodiscard]] std::string text(size_t count) {
    const auto span = bytes(count);
    return {span.begin(), span.end()};
  }
  [[nodiscard]] std::string rest() { return text(bytes_.size() - offset_); }
  [[nodiscard]] bool done() const { return offset_ == bytes_.size(); }

private:
  uint64_t little(uint32_t count) {
    uint64_t value = 0;
    const auto span = bytes(count);
    for (uint32_t index = 0; index < count; ++index)
      value |= uint64_t{span[index]} << (8 * index);
    return value;
  }

  std::span<const uint8_t> bytes_;
  size_t offset_ = 0;
};

FrameType write(Writer &out, const RequestFrame &request) {
  out.u64(request.requestId);
  out.u8(static_cast<uint8_t>(request.priority));
  out.u8(static_cast<uint8_t>(request.constraint));
  out.u64(request.absoluteDeadlineUnixMicros);
  out.u64(request.remainingDeadlineMicros);
  out.u32(request.logicalMaxOutputTokens);
  out.u32(static_cast<uint32_t>(request.promptTokens.size()));
  out.u32(static_cast<uint32_t>(request.imageSpans.size()));
  out.f32(request.sampling.temperature);
  out.f32(request.sampling.topP);
  out.u32(request.sampling.topK);
  out.f32(request.sampling.presencePenalty);
  out.f32(request.sampling.frequencyPenalty);
  out.f32(request.sampling.repetitionPenalty);
  out.f32(request.sampling.minP);
  out.u64(request.sampling.seed);
  out.u32(static_cast<uint32_t>(request.scoreTokens.size()));
  out.u32(request.generationPromptTokens);
  out.u32(request.flags);
  out.words(request.promptTokens);
  for (const ImageSpan &span : request.imageSpans) {
    out.u32(span.offset);
    out.u32(span.tokens);
    out.u32(span.gridHeight);
    out.u32(span.gridWidth);
    out.u64(span.digestLo);
    out.u64(span.digestHi);
  }
  out.raw(request.imagePixels);
  out.words(request.scoreTokens);
  return FrameType::Request;
}

FrameType write(Writer &out, const CancelFrame &cancel) {
  out.u64(cancel.requestId);
  return FrameType::Cancel;
}

FrameType write(Writer &out, const MaskResponseFrame &response) {
  out.u64(response.requestId);
  out.u64(response.maskRequestId);
  out.u32(static_cast<uint32_t>(response.maskWords.size()));
  out.words(response.maskWords);
  return FrameType::MaskResponse;
}

FrameType write(Writer &out, const StatusRequestFrame &request) {
  out.u64(request.correlationId);
  return FrameType::StatusRequest;
}

struct Header {
  FrameType type;
  uint64_t payloadBytes;
};

// Reads a frame header, checking what identifies the native wire.
Header readHeader(std::span<const uint8_t> bytes) {
  Reader in(bytes);
  const auto magic = in.bytes(kMagic.size());
  if (!std::equal(magic.begin(), magic.end(), kMagic.begin()) ||
      in.u16() != kProtocolVersion) {
    throw std::runtime_error("native frame header is invalid");
  }
  const auto type = static_cast<FrameType>(in.u16());
  return {type, in.u64()};
}

EngineEvent decodeEvent(FrameType type, Reader &in) {
  switch (type) {
  case FrameType::Ready: {
    ReadyEvent event;
    event.maxConcurrentRequests = in.u32();
    event.maxContextTokens = in.u32();
    event.vision = in.u8() != 0;
    return event;
  }
  case FrameType::Start: {
    StartEvent event;
    event.requestId = in.u64();
    event.lane = in.u32();
    event.matchedPromptTokens = in.u32();
    return event;
  }
  case FrameType::PromptProgress: {
    PromptProgressEvent event;
    event.requestId = in.u64();
    event.processedTokens = in.u32();
    event.elapsedMicros = in.u64();
    return event;
  }
  case FrameType::Tokens: {
    TokensEvent event;
    event.requestId = in.u64();
    event.sequenceOffset = in.u32();
    event.tokens = in.words(in.u32());
    return event;
  }
  case FrameType::MaskRequest: {
    MaskRequestEvent event;
    event.requestId = in.u64();
    event.maskRequestId = in.u64();
    event.wordsPerMask = in.u32();
    event.simulationTokens = in.words(in.u32());
    return event;
  }
  case FrameType::Done: {
    DoneEvent event;
    event.requestId = in.u64();
    event.reason = static_cast<engine::EngineFinishReason>(in.u8());
    event.promptTokens = in.u32();
    event.completionTokens = in.u32();
    event.prefillMicros = in.u64();
    event.decodeMicros = in.u64();
    event.wallMicros = in.u64();
    event.optionLogits.resize(in.u32());
    for (float &logit : event.optionLogits)
      logit = in.f32();
    return event;
  }
  case FrameType::Error: {
    ErrorEvent event;
    event.failureClass = static_cast<FailureClass>(in.u8());
    event.retryable = in.u8() != 0;
    event.requestId = in.u64();
    const uint32_t codeBytes = in.u32();
    const uint32_t messageBytes = in.u32();
    event.code = in.text(codeBytes);
    event.message = in.text(messageBytes);
    return event;
  }
  case FrameType::StatusJson: {
    StatusJsonEvent event;
    event.correlationId = in.u64();
    event.json = in.rest();
    return event;
  }
  default:
    throw std::runtime_error("the engine sent a frame that is not an event");
  }
}

} // namespace

std::vector<uint8_t> serialize(const ClientMessage &message) {
  Writer payload;
  const FrameType type = std::visit(
      [&](const auto &value) { return write(payload, value); }, message);
  const std::vector<uint8_t> bytes = payload.take();
  Writer frame;
  frame.raw(std::span(reinterpret_cast<const uint8_t *>(kMagic.data()),
                      kMagic.size()));
  frame.u16(kProtocolVersion);
  frame.u16(static_cast<uint16_t>(type));
  frame.u64(bytes.size());
  frame.raw(bytes);
  return frame.take();
}

std::vector<EngineEvent> decodeEvents(std::span<const uint8_t> bytes) {
  std::vector<EngineEvent> events;
  Reader stream(bytes);
  while (!stream.done()) {
    const Header header = readHeader(stream.bytes(kFrameHeaderBytes));
    Reader payload(stream.bytes(header.payloadBytes));
    events.push_back(decodeEvent(header.type, payload));
    if (!payload.done())
      throw std::runtime_error("native event payload has trailing bytes");
  }
  return events;
}

std::vector<EngineEvent> EventReader::feed(std::span<const uint8_t> bytes) {
  pending_.insert(pending_.end(), bytes.begin(), bytes.end());
  const std::span<const uint8_t> pending(pending_);
  size_t whole = 0;
  while (pending.size() - whole >= kFrameHeaderBytes) {
    const uint64_t payloadBytes =
        readHeader(pending.subspan(whole, kFrameHeaderBytes)).payloadBytes;
    if (payloadBytes > pending.size() - whole - kFrameHeaderBytes)
      break;
    whole += kFrameHeaderBytes + payloadBytes;
  }
  std::vector<EngineEvent> events = decodeEvents(pending.first(whole));
  pending_.erase(pending_.begin(),
                 pending_.begin() + static_cast<std::ptrdiff_t>(whole));
  return events;
}

} // namespace splash::protocol::peer

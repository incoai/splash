#include "omni/OmniProtocol.hpp"

#include <cstring>

namespace splash::omni::protocol {
namespace {

constexpr char kMagic[4] = {'S', 'P', 'L', 'H'};

class Reader final {
public:
  explicit Reader(std::span<const uint8_t> bytes) : bytes_(bytes) {}
  template <class T> T take(const char *what) {
    T value;
    if (offset_ + sizeof(T) > bytes_.size()) throw ProtocolError(std::string("frame is short at ") + what);
    std::memcpy(&value, bytes_.data() + offset_, sizeof(T));
    offset_ += sizeof(T);
    return value;
  }
  template <class T> std::vector<T> takeArray(uint64_t count, const char *what) {
    if (count > bytes_.size() / sizeof(T) || offset_ + count * sizeof(T) > bytes_.size())
      throw ProtocolError(std::string("frame is short at ") + what);
    std::vector<T> values(count);
    std::memcpy(values.data(), bytes_.data() + offset_, count * sizeof(T));
    offset_ += count * sizeof(T);
    return values;
  }
  void finish(const char *what) const {
    if (offset_ != bytes_.size()) throw ProtocolError(std::string(what) + " has trailing bytes");
  }

private:
  std::span<const uint8_t> bytes_;
  size_t offset_ = 0;
};

class Writer final {
public:
  explicit Writer(FrameType type, size_t payloadBytes) : type_(type) {
    bytes_.resize(kHeaderBytes + payloadBytes);
    offset_ = kHeaderBytes;
  }
  template <class T> void put(T value) {
    std::memcpy(bytes_.data() + offset_, &value, sizeof(T));
    offset_ += sizeof(T);
  }
  void put(std::span<const uint8_t> raw) {
    std::memcpy(bytes_.data() + offset_, raw.data(), raw.size());
    offset_ += raw.size();
  }
  template <class T> void put(std::span<const T> values) {
    std::memcpy(bytes_.data() + offset_, values.data(), values.size() * sizeof(T));
    offset_ += values.size() * sizeof(T);
  }
  std::vector<uint8_t> finish() {
    if (offset_ != bytes_.size()) throw std::logic_error("frame payload size mismatch");
    encodeHeader(type_, bytes_.size() - kHeaderBytes, std::span<uint8_t, kHeaderBytes>(bytes_.data(), kHeaderBytes));
    return std::move(bytes_);
  }

private:
  FrameType type_;
  std::vector<uint8_t> bytes_;
  size_t offset_ = 0;
};

bool knownType(uint16_t type) {
  switch (static_cast<FrameType>(type)) {
  case FrameType::Cancel:
  case FrameType::StatusRequest:
  case FrameType::SpeechRequest:
  case FrameType::TranscribeRequest:
  case FrameType::Ready:
  case FrameType::Start:
  case FrameType::Tokens:
  case FrameType::Done:
  case FrameType::Error:
  case FrameType::StatusJson:
  case FrameType::Audio:
  case FrameType::Codes:
    return true;
  }
  return false;
}

void requireFinite(float value, const char *what) {
  if (value != value || value > 1e30F || value < -1e30F) throw ProtocolError(std::string(what) + " is not finite");
}

} // namespace

FrameHeader decodeHeader(std::span<const uint8_t, kHeaderBytes> bytes) {
  if (std::memcmp(bytes.data(), kMagic, 4) != 0) throw ProtocolError("bad frame magic");
  uint16_t version = 0, type = 0;
  uint64_t length = 0;
  std::memcpy(&version, bytes.data() + 4, 2);
  std::memcpy(&type, bytes.data() + 6, 2);
  std::memcpy(&length, bytes.data() + 8, 8);
  if (version != kVersion)
    throw ProtocolError("protocol version " + std::to_string(version) + ", expected " + std::to_string(kVersion));
  if (!knownType(type)) throw ProtocolError("unknown frame type " + std::to_string(type));
  if (length > kMaximumPayloadBytes) throw ProtocolError("frame payload exceeds 256 MiB");
  return {static_cast<FrameType>(type), length};
}

void encodeHeader(FrameType type, uint64_t length, std::span<uint8_t, kHeaderBytes> bytes) {
  std::memcpy(bytes.data(), kMagic, 4);
  const uint16_t version = kVersion, code = static_cast<uint16_t>(type);
  std::memcpy(bytes.data() + 4, &version, 2);
  std::memcpy(bytes.data() + 6, &code, 2);
  std::memcpy(bytes.data() + 8, &length, 8);
}

SpeechRequestFrame decodeSpeechRequest(std::span<const uint8_t> payload) {
  Reader reader(payload);
  SpeechRequestFrame f;
  f.requestId = reader.take<uint64_t>("request id");
  f.flags = reader.take<uint32_t>("flags");
  f.languageId = reader.take<uint32_t>("language id");
  f.temperature = reader.take<float>("temperature");
  f.topK = reader.take<uint32_t>("top_k");
  f.topP = reader.take<float>("top_p");
  f.repetitionPenalty = reader.take<float>("repetition penalty");
  f.predictorTemperature = reader.take<float>("predictor temperature");
  f.predictorTopK = reader.take<uint32_t>("predictor top_k");
  f.predictorTopP = reader.take<float>("predictor top_p");
  f.seed = reader.take<uint64_t>("seed");
  f.maxFrames = reader.take<uint32_t>("max frames");
  f.chunkFrames = reader.take<uint32_t>("chunk frames");
  const uint32_t textCount = reader.take<uint32_t>("text count");
  const uint32_t referenceTextCount = reader.take<uint32_t>("reference text count");
  const uint32_t referenceFrames = reader.take<uint32_t>("reference frame count");
  const uint32_t speakerPresent = reader.take<uint32_t>("speaker flag");
  const uint32_t instructionCount = reader.take<uint32_t>("instruction count");
  f.presetSpeaker = reader.take<uint32_t>("preset speaker");
  if (f.flags & ~kSpeechFlagCodes) throw ProtocolError("speech request flags hold reserved bits");
  if (!textCount || textCount > kMaximumTextTokens) throw ProtocolError("text must hold 1..4096 tokens");
  if (instructionCount > kMaximumInstructionTokens) throw ProtocolError("the instruction exceeds 512 tokens");
  if (referenceTextCount > kMaximumTextTokens || referenceFrames > 8192) throw ProtocolError("reference is too long");
  for (float value : {f.temperature, f.topP, f.repetitionPenalty, f.predictorTemperature, f.predictorTopP})
    requireFinite(value, "sampling parameter");
  if (f.temperature < 0.0F || f.predictorTemperature < 0.0F) throw ProtocolError("temperature must not be negative");
  if (f.topP <= 0.0F || f.topP > 1.0F || f.predictorTopP <= 0.0F || f.predictorTopP > 1.0F)
    throw ProtocolError("top_p must be in (0, 1]");
  if (f.repetitionPenalty <= 0.0F) throw ProtocolError("repetition penalty must be positive");
  if (f.topK > 256 || f.predictorTopK > 256) throw ProtocolError("top_k must be at most 256");
  f.text = reader.takeArray<uint32_t>(textCount, "text tokens");
  f.instruction = reader.takeArray<uint32_t>(instructionCount, "instruction tokens");
  f.referenceText = reader.takeArray<uint32_t>(referenceTextCount, "reference text tokens");
  f.referenceCodes = reader.takeArray<uint32_t>(uint64_t{referenceFrames} * 16, "reference codes");
  if (speakerPresent) f.speaker = reader.takeArray<float>(2048, "speaker embedding");
  reader.finish("speech request");
  return f;
}

TranscribeRequestFrame decodeTranscribeRequest(std::span<const uint8_t> payload) {
  Reader reader(payload);
  TranscribeRequestFrame f;
  f.requestId = reader.take<uint64_t>("request id");
  f.flags = reader.take<uint32_t>("flags");
  f.maxTokens = reader.take<uint32_t>("max tokens");
  const uint32_t prefixCount = reader.take<uint32_t>("prefix count");
  const uint32_t suffixCount = reader.take<uint32_t>("suffix count");
  const uint32_t sampleCount = reader.take<uint32_t>("sample count");
  if (!prefixCount || prefixCount > kMaximumTextTokens || !suffixCount || suffixCount > kMaximumTextTokens)
    throw ProtocolError("prefix and suffix must hold 1..4096 tokens");
  if (sampleCount > kMaximumSamples) throw ProtocolError("audio exceeds 20 minutes");
  f.prefix = reader.takeArray<uint32_t>(prefixCount, "prefix tokens");
  f.suffix = reader.takeArray<uint32_t>(suffixCount, "suffix tokens");
  f.samples = reader.takeArray<float>(sampleCount, "samples");
  for (float sample : f.samples) requireFinite(sample, "sample");
  reader.finish("transcribe request");
  return f;
}

uint64_t decodeCancel(std::span<const uint8_t> payload) {
  Reader reader(payload);
  const uint64_t id = reader.take<uint64_t>("request id");
  reader.finish("cancel");
  return id;
}

uint64_t decodeStatusRequest(std::span<const uint8_t> payload) {
  Reader reader(payload);
  const uint64_t id = reader.take<uint64_t>("correlation id");
  reader.finish("status request");
  return id;
}

std::vector<uint8_t> readyFrame(uint32_t maxConcurrent, uint32_t maxContext, ModelKindCode kind) {
  Writer writer(FrameType::Ready, 9);
  writer.put<uint32_t>(maxConcurrent);
  writer.put<uint32_t>(maxContext);
  writer.put<uint8_t>(static_cast<uint8_t>(kind));
  return writer.finish();
}

std::vector<uint8_t> startFrame(uint64_t requestId, uint32_t lane) {
  Writer writer(FrameType::Start, 16);
  writer.put<uint64_t>(requestId);
  writer.put<uint32_t>(lane);
  writer.put<uint32_t>(0);
  return writer.finish();
}

std::vector<uint8_t> audioFrame(uint64_t requestId, uint32_t sampleOffset, std::span<const int16_t> samples) {
  Writer writer(FrameType::Audio, 16 + samples.size() * 2);
  writer.put<uint64_t>(requestId);
  writer.put<uint32_t>(sampleOffset);
  writer.put<uint32_t>(static_cast<uint32_t>(samples.size()));
  writer.put<int16_t>(samples);
  return writer.finish();
}

std::vector<uint8_t> codesFrame(uint64_t requestId, std::span<const uint32_t> codes) {
  if (codes.size() % 16) throw std::logic_error("codes are not whole frames");
  Writer writer(FrameType::Codes, 12 + codes.size() * 4);
  writer.put<uint64_t>(requestId);
  writer.put<uint32_t>(static_cast<uint32_t>(codes.size() / 16));
  writer.put<uint32_t>(codes);
  return writer.finish();
}

std::vector<uint8_t> tokensFrame(uint64_t requestId, uint32_t sequenceOffset, std::span<const uint32_t> tokens) {
  Writer writer(FrameType::Tokens, 16 + tokens.size() * 4);
  writer.put<uint64_t>(requestId);
  writer.put<uint32_t>(sequenceOffset);
  writer.put<uint32_t>(static_cast<uint32_t>(tokens.size()));
  writer.put<uint32_t>(tokens);
  return writer.finish();
}

std::vector<uint8_t> doneFrame(uint64_t requestId, DoneReason reason, const DoneTimings &timings) {
  Writer writer(FrameType::Done, 8 + 1 + 4 + 4 + 8 + 8 + 8 + 4);
  writer.put<uint64_t>(requestId);
  writer.put<uint8_t>(static_cast<uint8_t>(reason));
  writer.put<uint32_t>(timings.promptTokens);
  writer.put<uint32_t>(timings.completion);
  writer.put<uint64_t>(timings.prefillMicros);
  writer.put<uint64_t>(timings.decodeMicros);
  writer.put<uint64_t>(timings.wallMicros);
  writer.put<uint32_t>(0);
  return writer.finish();
}

std::vector<uint8_t> errorFrame(FailureClass failure, bool retryable, uint64_t requestId, std::string_view code,
                                std::string_view message) {
  Writer writer(FrameType::Error, 1 + 1 + 8 + 4 + 4 + code.size() + message.size());
  writer.put<uint8_t>(static_cast<uint8_t>(failure));
  writer.put<uint8_t>(retryable ? 1 : 0);
  writer.put<uint64_t>(requestId);
  writer.put<uint32_t>(static_cast<uint32_t>(code.size()));
  writer.put<uint32_t>(static_cast<uint32_t>(message.size()));
  writer.put(std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(code.data()), code.size()));
  writer.put(std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(message.data()), message.size()));
  return writer.finish();
}

std::vector<uint8_t> statusJsonFrame(uint64_t correlationId, std::string_view json) {
  Writer writer(FrameType::StatusJson, 8 + json.size());
  writer.put<uint64_t>(correlationId);
  writer.put(std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(json.data()), json.size()));
  return writer.finish();
}

} // namespace splash::omni::protocol

#pragma once

// The Omni wire protocol, version 11 (docs/omni-design.md, "Protocol"):
// Splash's 16-byte frame header, the client frames SpeechRequest,
// TranscribeRequest, Cancel and StatusRequest, and the engine frames Ready,
// Start, Audio, Codes, Tokens, Done, Error and StatusJson. All integers are
// little-endian. Parsing throws ProtocolError naming what was wrong.

#include "omni/OmniEvents.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace splash::omni::protocol {

inline constexpr uint16_t kVersion = 11;
inline constexpr size_t kHeaderBytes = 16;
inline constexpr uint64_t kMaximumPayloadBytes = uint64_t{256} << 20;
inline constexpr uint32_t kMaximumTextTokens = 4096;
// A speech request's instruction turn (VoiceDesign, CustomVoice).
inline constexpr uint32_t kMaximumInstructionTokens = 512;
inline constexpr uint32_t kMaximumSamples = 20 * 60 * 16000;
// SpeechRequest flags: a Codes event with the request's frames before its
// Done. The other bits are reserved and must be 0.
inline constexpr uint32_t kSpeechFlagCodes = 1;

enum class FrameType : uint16_t {
  Cancel = 0x0002,
  StatusRequest = 0x0004,
  SpeechRequest = 0x0011,
  TranscribeRequest = 0x0012,
  Ready = 0x0100,
  Start = 0x0101,
  Tokens = 0x0102,
  Done = 0x0104,
  Error = 0x0105,
  StatusJson = 0x0107,
  Audio = 0x0110,
  Codes = 0x0111,
};

class ProtocolError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

struct FrameHeader final {
  FrameType type = FrameType::Cancel;
  uint64_t length = 0;
};
// Decodes a header; throws ProtocolError on a bad magic, version or length.
[[nodiscard]] FrameHeader decodeHeader(std::span<const uint8_t, kHeaderBytes> bytes);
void encodeHeader(FrameType type, uint64_t length, std::span<uint8_t, kHeaderBytes> bytes);

struct SpeechRequestFrame final {
  uint64_t requestId = 0;
  uint32_t flags = 0;
  uint32_t languageId = 0;
  float temperature = 0.9F;
  uint32_t topK = 50;
  float topP = 1.0F;
  float repetitionPenalty = 1.05F;
  float predictorTemperature = 0.9F;
  uint32_t predictorTopK = 50;
  float predictorTopP = 1.0F;
  uint64_t seed = 0;
  uint32_t maxFrames = 0;
  uint32_t chunkFrames = 0;
  uint32_t presetSpeaker = 0; // a CustomVoice speaker's codec id, 0 = none
  std::vector<uint32_t> text;
  std::vector<uint32_t> instruction; // the instruction's user turn, tokenized whole
  std::vector<uint32_t> referenceText;
  std::vector<uint32_t> referenceCodes;
  std::vector<float> speaker;
};
[[nodiscard]] SpeechRequestFrame decodeSpeechRequest(std::span<const uint8_t> payload);

struct TranscribeRequestFrame final {
  uint64_t requestId = 0;
  uint32_t flags = 0;
  uint32_t maxTokens = 0;
  std::vector<uint32_t> prefix;
  std::vector<uint32_t> suffix;
  std::vector<float> samples;
};
[[nodiscard]] TranscribeRequestFrame decodeTranscribeRequest(std::span<const uint8_t> payload);
[[nodiscard]] uint64_t decodeCancel(std::span<const uint8_t> payload);
[[nodiscard]] uint64_t decodeStatusRequest(std::span<const uint8_t> payload);

enum class ModelKindCode : uint8_t { Tts = 1, Asr = 2 };
enum class FailureClass : uint8_t { RequestError = 1, EngineUnhealthy = 2, ProtocolFatal = 3 };

// Whole frames, header included.
[[nodiscard]] std::vector<uint8_t> readyFrame(uint32_t maxConcurrent, uint32_t maxContext, ModelKindCode kind);
[[nodiscard]] std::vector<uint8_t> startFrame(uint64_t requestId, uint32_t lane);
[[nodiscard]] std::vector<uint8_t> audioFrame(uint64_t requestId, uint32_t sampleOffset, std::span<const int16_t> samples);
// A speech request's generated frames, 16 codes each.
[[nodiscard]] std::vector<uint8_t> codesFrame(uint64_t requestId, std::span<const uint32_t> codes);
[[nodiscard]] std::vector<uint8_t> tokensFrame(uint64_t requestId, uint32_t sequenceOffset, std::span<const uint32_t> tokens);
[[nodiscard]] std::vector<uint8_t> doneFrame(uint64_t requestId, DoneReason reason, const DoneTimings &timings);
[[nodiscard]] std::vector<uint8_t> errorFrame(FailureClass failure, bool retryable, uint64_t requestId, std::string_view code,
                                              std::string_view message);
[[nodiscard]] std::vector<uint8_t> statusJsonFrame(uint64_t correlationId, std::string_view json);

} // namespace splash::omni::protocol

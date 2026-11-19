#pragma once

// What the Omni engines report about a request, which the transport turns
// into frames (docs/omni-design.md, "Protocol").

#include <cstdint>
#include <span>
#include <string>

namespace splash::omni {

enum class DoneReason : uint8_t { Stop = 0, Length = 1, Cancelled = 2 };

struct DoneTimings final {
  uint32_t promptTokens = 0;
  uint32_t completion = 0; // TTS: frames; ASR: tokens
  uint64_t prefillMicros = 0;
  uint64_t decodeMicros = 0;
  uint64_t wallMicros = 0;
};

class EventSink {
public:
  virtual ~EventSink() = default;
  virtual void onStart(uint64_t requestId, uint32_t lane) = 0;
  // TTS: a chunk of 24 kHz int16 samples, in order.
  virtual void onAudio(uint64_t requestId, uint32_t sampleOffset, std::span<const int16_t> samples) = 0;
  // TTS: the request's generated frames (16 codes each), before its Done,
  // when the request asked for them.
  virtual void onCodes(uint64_t requestId, std::span<const uint32_t> codes) {
    (void)requestId;
    (void)codes;
  }
  // ASR: generated tokens, in order.
  virtual void onTokens(uint64_t requestId, uint32_t sequenceOffset, std::span<const uint32_t> tokens) = 0;
  virtual void onDone(uint64_t requestId, DoneReason reason, const DoneTimings &timings) = 0;
  // A request that failed; no Done follows.
  virtual void onError(uint64_t requestId, const std::string &code, const std::string &message) = 0;
};

} // namespace splash::omni

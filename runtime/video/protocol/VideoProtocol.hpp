#pragma once

#include "engine/Protocol.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

// The video engine's stream on its stdin and stdout. Its frames have the
// language model engine's header (engine/Protocol.hpp) with their own magic
// and version, so a frame sent to the other engine fails at its header with
// BadMagic, either way. Cancel, Error and StartFailed carry the engine
// protocol's payloads, through its codec. The video engine decodes client
// frames and encodes events; server/video_protocol.py does the reverse.
// dev/tests/engine/video_protocol_golden.txt pins the bytes both sides agree
// on.
namespace splash::video::wire {

// A frame is its header, the magic "SPLV", the u16 video protocol version,
// the u16 VideoFrameType and the u64 payload length, then the payload.
inline constexpr uint16_t kVideoProtocolVersion = 1;
inline constexpr std::array<uint8_t, 4> kVideoMagic{'S', 'P', 'L', 'V'};
static_assert(protocol::kFrameHeaderBytes == 16);
// A video request carries its Ref2VA references' normalized frames: three
// 15 s reference videos on a 768p canvas are 3.4 GB of RGB.
inline constexpr uint64_t kAbsoluteMaxVideoRequestPayloadBytes =
    4ULL * 1024 * 1024 * 1024;

// Every integer is little-endian on the wire. A type the engine protocol
// also has keeps its id and its payload.
enum class VideoFrameType : uint16_t {
  Cancel = 0x0002,
  VideoRequest = 0x0007,

  Error = 0x0105,
  StartFailed = 0x0109,
  VideoProgress = 0x010B,
  VideoOutput = 0x010C,
  VideoReady = 0x010D,
};
static_assert(static_cast<uint16_t>(VideoFrameType::Cancel) ==
              static_cast<uint16_t>(protocol::FrameType::Cancel));
static_assert(static_cast<uint16_t>(VideoFrameType::Error) ==
              static_cast<uint16_t>(protocol::FrameType::Error));
static_assert(static_cast<uint16_t>(VideoFrameType::StartFailed) ==
              static_cast<uint16_t>(protocol::FrameType::StartFailed));

struct VideoLimits {
  // The engine protocol's, valid as protocol::validateLimits checks them:
  // a VideoRequest's prompt tokens, a VideoOutput's frame, and Cancel's,
  // Error's and StartFailed's own.
  protocol::ProtocolLimits shared;
  // A VideoRequest's payload bound, at most
  // kAbsoluteMaxVideoRequestPayloadBytes.
  uint64_t maxVideoRequestPayloadBytes = kAbsoluteMaxVideoRequestPayloadBytes;
};

// The payloads, in wire order; counts precede what they count, and the
// variable parts come last:
//   VideoRequest   u64 requestId, u8 DiT name bytes, u8 outputs
//                  (VideoOutputFlag), u32 frames, u32 width, u32 height,
//                  u32 steps, u64 seed, u8 keyframes (VideoKeyframeFlag), u32
//                  prompt token count, u32 reference count, the DiT name
//                  (validDitName), the prompt tokens, per
//                  reference its header (u8 kind, u32 frames, u32 width,
//                  u32 height, u64 soundtrack samples), then the first
//                  frame's and the last frame's pixels, each height x width x
//                  3 bytes, then per reference its pixels (frames x height x
//                  width x 3 bytes) and its soundtrack (2 x samples f32)
//   Cancel         the engine protocol's
//   VideoReady     u32 maxConcurrentRequests, u32 maxPromptTokens
//   VideoProgress  u64 requestId, u8 phase, u32 step, u32 steps,
//                  u64 elapsedMicros, u64 peakFootprintBytes
//   VideoOutput    u64 requestId, u8 output, u32 frames, u64 wallMicros,
//                  the mp4
//   Error          the engine protocol's
//   StartFailed    the engine protocol's

// One clip of a video model, of a prompt of the text encoder's token ids.
// Answered by VideoProgress events, then the clip's VideoOutput, or an Error.
// It names the DiT it runs on: a directory name under the weights, 1 to 64
// of a-z, 0-9, '.', '_' and '-', without ".." (server/video_protocol.py
// valid_dit_name).
inline constexpr size_t kMaxDitNameBytes = 64;
[[nodiscard]] bool validDitName(std::string_view name);
enum VideoOutputFlag : uint8_t {
  VideoOutputPreview = 1U << 0,
  VideoOutputAudio = 1U << 1,
};
// The keyframes a video request carries.
enum VideoKeyframeFlag : uint8_t {
  VideoKeyframeFirst = 1U << 0,
  VideoKeyframeLast = 1U << 1,
};

// A Ref2VA reference of a video request, as the server normalized it
// (by Pika H3 Flex's rules): an image at its tier size, a
// video's frames at 24 fps, truncated to the clip's frame count, on its tier
// canvas, with its soundtrack at 32 kHz stereo truncated to the clip, or an
// audio reference, a soundtrack alone (no frames, width or height).
enum class VideoReferenceKind : uint8_t { Image = 0, Video = 1, Audio = 2 };

struct VideoReference {
  VideoReferenceKind kind = VideoReferenceKind::Image;
  // An image's frames are 1, an audio reference's 0.
  uint32_t frames = 1;
  uint32_t width = 0;
  uint32_t height = 0;
  // uint8 RGB [frames][height][width][3].
  std::vector<uint8_t> pixels;
  // fp32 [2][samples] at 32 kHz: a video's soundtrack (empty: none), an
  // audio reference's samples.
  std::vector<float> soundtrack;

  bool operator==(const VideoReference &) const = default;
};

// The references a video request carries at most (Pika H3 Flex's 12).
inline constexpr uint32_t kMaxVideoReferences = 12;

struct VideoRequestFrame {
  uint64_t requestId = 0;
  // The DiT's name (validDitName): its weights' dit-<name>/.
  std::string dit;
  // VideoOutputFlag bits: also the preview; the clip with its audio.
  uint8_t outputs = VideoOutputAudio;
  uint32_t frames = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  // Denoising steps: a VDN-H3 DiT's; a FastH3 DiT's zero or its rungs'.
  uint32_t steps = 0;
  uint64_t seed = 0;
  std::vector<uint32_t> promptTokens;
  // Keyframes (first-frame and first-and-last-frame conditioning), each
  // empty or uint8 RGB [height][width][3], already on the clip's canvas
  // (server/images.py keyframe): the first frame anchors the clip's first
  // latent frame, the last frame its last.
  std::vector<uint8_t> firstFrame;
  std::vector<uint8_t> lastFrame;
  // Ref2VA references in presentation order; none with keyframes.
  std::vector<VideoReference> references;

  bool operator==(const VideoRequestFrame &) const = default;
};

// Sent once the engine can take requests, in place of the language model
// engine's Ready; a start that fails sends StartFailed instead.
struct VideoReadyEvent {
  uint32_t maxConcurrentRequests = 0;
  // A presentation's tokens at most, its vision blocks' included.
  uint32_t maxPromptTokens = 0;

  bool operator==(const VideoReadyEvent &) const = default;
};

// The phases of a video request, in their order.
enum class VideoPhase : uint8_t {
  Encode = 1,
  Prepare,
  Denoise,
  Preview,
  Decode,
  Audio,
  Mux,
};

// A phase has done `step` of its `steps` (one but the denoising's), the
// phase's time so far and the process's peak footprint; step 0 starts it
// and `steps` completes it.
struct VideoProgressEvent {
  uint64_t requestId = 0;
  VideoPhase phase = VideoPhase::Encode;
  uint32_t step = 0;
  uint32_t steps = 1;
  uint64_t elapsedMicros = 0;
  uint64_t peakFootprintBytes = 0;

  bool operator==(const VideoProgressEvent &) const = default;
};

// An mp4 of the request: the preview, once its phase completes, or the clip,
// which ends the request.
enum class VideoOutputKind : uint8_t { Video = 0, Preview = 1 };

struct VideoOutputEvent {
  uint64_t requestId = 0;
  VideoOutputKind output = VideoOutputKind::Video;
  uint32_t frames = 0;
  // From the request's arrival.
  uint64_t wallMicros = 0;
  std::vector<uint8_t> mp4;

  bool operator==(const VideoOutputEvent &) const = default;
};

using VideoClientMessage = std::variant<VideoRequestFrame, protocol::CancelFrame>;
using VideoEvent =
    std::variant<VideoReadyEvent, VideoProgressEvent, VideoOutputEvent,
                 protocol::ErrorEvent, protocol::StartFailedEvent>;

struct VideoFrame {
  VideoFrameType type = VideoFrameType::VideoRequest;
  std::vector<uint8_t> payload;

  bool operator==(const VideoFrame &) const = default;
};

// A client frame's issues are the engine protocol's: a request's error
// rejects that request, any other closes the stream.
[[nodiscard]] protocol::ProtocolResult<VideoClientMessage>
decodeVideoFrame(VideoFrame &&frame, const VideoLimits &limits);
// The whole frame, header included. An event that breaks its rules is the
// engine's defect: the issue is EngineUnhealthy.
[[nodiscard]] protocol::ProtocolResult<std::vector<uint8_t>>
serializeVideoEvent(const VideoEvent &event, const VideoLimits &limits);

struct VideoParseStep {
  size_t consumedBytes = 0;
  std::optional<VideoFrame> frame;
  std::optional<protocol::ProtocolIssue> issue;
};

// Incremental one-frame-at-a-time parser of client frames, as
// protocol::FrameParser is: consume() stops at each complete frame, an event
// type is an unknown frame type, a frame of the engine protocol has bad
// magic, and finish() at EOF turns a partial frame into a protocol-fatal
// truncation.
class VideoFrameParser {
public:
  explicit VideoFrameParser(VideoLimits limits);

  [[nodiscard]] VideoParseStep consume(std::span<const uint8_t> bytes);
  [[nodiscard]] std::optional<protocol::ProtocolIssue> finish();

private:
  [[nodiscard]] std::optional<protocol::ProtocolIssue> parseHeader();
  [[nodiscard]] VideoParseStep fail(size_t consumed, protocol::ProtocolIssue issue);
  void resetCurrentFrame();

  VideoLimits limits_;
  std::array<uint8_t, protocol::kFrameHeaderBytes> header_{};
  size_t headerBytes_ = 0;
  bool readingPayload_ = false;
  VideoFrameType currentType_ = VideoFrameType::VideoRequest;
  uint64_t expectedPayloadBytes_ = 0;
  std::vector<uint8_t> payload_;
  std::optional<protocol::ProtocolIssue> terminalIssue_;
};

} // namespace splash::video::wire

#pragma once

// The packed DiT sequence of one request as a list of row classes.
// Every row belongs to one class; a class fixes the row's input stream, modality tag, adaLN slot, timestep, update rule
// and attention role (kRowClasses), and its (t, h, w) position rule (RowLayout::build). Packed order:
//   text (text and vision runs) | per reference: [soundtrack, video] / [image] / [audio] / [keyframe] | target audio |
//   target video
// Text-to-video is the text and target classes alone, FL2VA adds keyframe rows (one target frame each), Ref2VA the
// reference classes (Pika H3 Flex, matched bit for bit against a reference run's saved layout by
// dev/tools/video_ref2va_layout).
// Positions are float64 as the pipelines compute them; a shared rotary clock starts at the text length, every
// reference advances it (image + 1, audio + its latents, video + max(soundtrack latents, its temporal span)), keyframes
// do not, and the target audio and video start at the clock the references leave.
#include "video/ops/Geometry.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace splash::video::ops {

enum class RowClass : uint8_t {
  Text,           // prompt and presentation labels
  Vision,         // vision-tower rows inside the text segment (Qwen3-VL image / video blocks)
  Keyframe,       // FL2VA first / last frame: one target frame of rows
  RefSoundtrack,  // a reference video's audio latents
  RefVideo,       // a reference video's latents
  RefImage,       // a reference image's latents (one latent frame)
  RefAudio,       // a reference audio clip's latents
  TargetAudio,    // generated audio
  TargetVideo,    // generated video
};
inline constexpr int32_t kRowClassCount = 9;

enum class RowStream : uint8_t { Text, Video, Audio };  // the input rows: refined text, 96-channel video, 32-channel audio
// Timestep of a row: the target video's t, the conditioning max(t, 0.999), clean 1.0 (never noised), the audio's t
enum class RowTime : uint8_t { Video, Condition, Clean, Audio };
// Initial state and update: fixed (clean or text), augmented condition (0.999 clean + (1 - 0.999) eps, fixed), noised
// target (initial noise, updated by the sampler)
enum class RowUpdate : uint8_t { Fixed, Augmented, Noised };
// Attention role: dense prefix (its queries attend to every row; video queries keep all its keys) or the video
// region (VSA: video queries keep the prefix keys and their top-k video tiles)
enum class RowAttention : uint8_t { Prefix, VideoRegion };

inline constexpr float kConditionTime = 0.999f;  // KEYFRAME_NOISE_AUG: clean-time t of augmented condition rows
inline constexpr uint8_t kSlotCondAudio = 4;     // adaLN slot of clean audio rows: audio modulation at t = 1 (no table yet)

struct RowClassInfo {
  const char *name;
  RowStream stream;
  uint8_t tag;   // modality tag of the dumps' token_tags: 0 video, 1 text, 2 audio
  uint8_t slot;  // adaLN table slot (metal/abi/video/Layout.h tags): 0 video, 1 text, 2 audio, 3 conditioning video, 4
  RowTime time;
  RowUpdate update;
  RowAttention attention;
};

inline constexpr RowClassInfo kRowClasses[kRowClassCount] = {
    {"text", RowStream::Text, 1, 1, RowTime::Video, RowUpdate::Fixed, RowAttention::Prefix},
    {"vision", RowStream::Text, 0, 0, RowTime::Video, RowUpdate::Fixed, RowAttention::Prefix},
    {"keyframe", RowStream::Video, 0, 3, RowTime::Condition, RowUpdate::Augmented, RowAttention::Prefix},
    {"ref_soundtrack", RowStream::Audio, 2, kSlotCondAudio, RowTime::Clean, RowUpdate::Fixed, RowAttention::Prefix},
    {"ref_video", RowStream::Video, 0, 3, RowTime::Condition, RowUpdate::Augmented, RowAttention::Prefix},
    {"ref_image", RowStream::Video, 0, 3, RowTime::Condition, RowUpdate::Augmented, RowAttention::Prefix},
    {"ref_audio", RowStream::Audio, 2, kSlotCondAudio, RowTime::Clean, RowUpdate::Fixed, RowAttention::Prefix},
    {"target_audio", RowStream::Audio, 2, 2, RowTime::Audio, RowUpdate::Noised, RowAttention::Prefix},
    {"target_video", RowStream::Video, 0, 0, RowTime::Video, RowUpdate::Noised, RowAttention::VideoRegion},
};
[[nodiscard]] inline const RowClassInfo &rowClass(RowClass c) { return kRowClasses[static_cast<int>(c)]; }

// One conditioning input in presentation order. Video and image latents are VAE latents [F][h][w] (h, w even: 2 x 2
// patches); a keyframe is one frame of the target's latents at its first (anchor 0) or last (1) latent frame.
struct ReferenceShape {
  enum class Kind : uint8_t { Video, Image, Audio, Keyframe } kind = Kind::Image;
  int32_t frames = 1, latentHeight = 0, latentWidth = 0;
  int32_t audioLatents = 0;  // Video: its soundtrack's latents (0: silent); Audio: its latents
  int32_t anchor = 0;        // Keyframe
};

struct RowRange {
  int32_t start = 0, rows = 0;
  [[nodiscard]] int32_t end() const { return start + rows; }
};

struct RowSegment {
  RowClass cls = RowClass::Text;
  int32_t start = 0, rows = 0;
  int32_t ref = -1;                                     // index into the references, -1 for text and target rows
  int32_t frames = 0, latentHeight = 0, latentWidth = 0;  // video-stream segments: their latents
  [[nodiscard]] int32_t end() const { return start + rows; }
};

struct RowLayout {
  std::vector<RowSegment> segments;  // packed order, class-uniform
  std::vector<double> pos;           // [rows][3] (t, h, w)
  std::vector<uint8_t> tags;         // [rows] RowClassInfo::tag
  std::vector<uint8_t> slots;        // [rows] RowClassInfo::slot
  // the video-stream rows (conditions first, then the target video), the audio-stream rows (likewise), the text rows:
  // the order of the DiT's video_in / audio_in rows
  std::vector<int64_t> videoIndices, audioIndices, textIndices;
  int32_t textRows = 0, condVideoRows = 0, condAudioRows = 0, targetAudioRows = 0, targetVideoRows = 0;
  double targetClock = 0;  // rotary clock at the target audio and video
  // The DiT's row order (SequenceLayout): text, the video-stream conditions, the audio-stream conditions, the target
  // audio, the target video, each in packed order; engineOrder[e] = packed row (the identity for text-to-video and
  // FL2VA). Every DiT operation but the VSA tile map is per row or over all rows, so the order changes no result.
  std::vector<int32_t> engineOrder;
  // The dense prefix in engine order (SequenceLayout::segments): the text, each run of one conditioning class, the
  // audio conditions, the target audio
  std::vector<std::pair<int32_t, int32_t>> engineSegments;

  [[nodiscard]] int32_t rows() const { return static_cast<int32_t>(tags.size()); }
  // textTags [textRows]: 1 text, 0 vision rows (null: all text). Throws std::invalid_argument for a shape the rules
  // do not cover.
  static RowLayout build(const VideoGeometry &target, int32_t textRows, const uint8_t *textTags,
                         const std::vector<ReferenceShape> &references);
  // The forward's distinct timesteps (sorted, as numpy.unique) and each row's index into them, from the target video's
  // and audio's clean-time t.
  void timesteps(float videoT, float audioT, std::vector<float> &table, std::vector<int64_t> &index) const;
  // Attention metadata: the merged row ranges of a role, and the rows the sampler never updates.
  [[nodiscard]] std::vector<RowRange> ranges(RowAttention role) const;
  [[nodiscard]] std::vector<RowRange> fixedRanges() const;
  // The DiT's SequenceLayout of these rows (engine order) for the target geometry.
  [[nodiscard]] SequenceLayout sequence(const VideoGeometry &target, int32_t tailPad) const;
  // A per-row array of `width` elements from packed into engine order.
  template <class T>
  void toEngineOrder(const T *packed, T *engine, size_t width) const {
    for (size_t e = 0; e < engineOrder.size(); ++e)
      for (size_t c = 0; c < width; ++c) engine[e * width + c] = packed[size_t(engineOrder[e]) * width + c];
  }
};

} // namespace splash::video::ops

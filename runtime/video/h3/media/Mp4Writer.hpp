#pragma once

// MP4 muxer on AVFoundation (container and the system H.264 / AAC encoders; no compute of the engine runs here):
// uint8 RGB frames -> H.264 High (VideoToolbox, BT.709 tags) at 24 fps, fp32 stereo waveform -> AAC-LC 192 kb/s at
// 32 kHz, one AVAssetWriter. Every frame is written (no -shortest truncation); both tracks start at 0.
#include "video/h3/engine/Status.hpp"

#include <cstdint>
#include <string>

namespace h3::media {

struct Mp4Stats {
  double wallS = 0;
  int32_t frames = 0;
  int64_t audioSamples = 0;  // per channel
  double videoSeconds = 0, audioSeconds = 0;
  int64_t bytes = 0;
  int32_t videoBitrate = 0;  // bits/s requested from the encoder
};

// rgb: [frames][height][width][3]; wav (nullable): [2][samples] fp32 in [-1, 1] at 32 kHz. bitrate 0: videoBitrate().
Status writeMp4(const std::string &path, const uint8_t *rgb, int32_t frames, int32_t width, int32_t height,
                const float *wav, int64_t samples, Mp4Stats &stats, int32_t bitrate = 0);

// Bit rate of the video track for a frame size: the libx264 CRF 18 quality of the former ffmpeg mux, measured on the
// golden clips (19.8 Mb/s at 1344 x 768, see Mp4Writer.mm).
int32_t videoBitrate(int32_t width, int32_t height);

} // namespace h3::media

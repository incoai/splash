#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace splash::video {

// A video the decoder cannot read, or a frame it cannot return: the server
// answers the request with a 400.
class VideoError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// The video-frames command (server/videos.py runs it in a disposable,
// supervised process): `probe FILE` writes the first video track's displayed
// size and the presentation times of its displayed frames and keyframes;
// `decode FILE WIDTH HEIGHT` reads the frames to return from stdin and writes
// each, resized and turned upright, as RGB24 to stdout. Returns the exit
// status.
int runVideoFrames(int argc, char **argv);

// Pillow's two-pass bicubic resize (Image.resize with Resampling.BICUBIC and
// no reducing_gap), which server/images.py applies to images: the same
// support, rounding and 22-bit fixed-point weights, so the result is
// byte-identical. source holds height rows of width pixels of channels bytes,
// stride bytes apart; the result is outHeight packed rows of outWidth pixels
// of the same channels. Pillow resizes along x, then y; verticalFirst
// reverses the passes, which is Pillow's order for the frame turned a
// quarter.
std::vector<uint8_t> resizeBicubic(const uint8_t *source, int width, int height,
                                   size_t stride, int channels, int outWidth,
                                   int outHeight, bool verticalFirst);

} // namespace splash::video

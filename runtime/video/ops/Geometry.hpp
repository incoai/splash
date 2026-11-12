#pragma once

// MiniMax-H3 request geometry: latent shapes of a clip, the packed DiT sequence and the video VAE decode plan
// (VideoVAE.decode chunking and _split_tiles tiling of the MLX port): the shapes the operators take. Every shape
// derives from the canvas's token grid (W / 32, H / 32: one token is 2 x 2 latent pixels of 16 x 16 px); nothing is
// tabulated per canvas.
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace splash::video::ops {

// Canvas tiers, by the short edge: 768p and 480p. A canvas of a tier is any width and height, both multiples of 32,
// whose short edge is the tier's and whose long edge is at most twice it (16:9 1344 x 768, 9:16 768 x 1344; 832 x 480
// and 864 x 480, 480 x 832). The tier of a canvas, or 0 for one outside both; why it is refused, or empty.
inline constexpr int32_t kCanvasTiers[] = {768, 480};
inline constexpr int32_t kCanvasStep = 32;  // pixels per token side
[[nodiscard]] int32_t canvasTier(int32_t width, int32_t height);
[[nodiscard]] std::string canvasRefusal(int32_t width, int32_t height);

// Clip of 17 n + 5 frames at 24 fps (3 s = 73, 5 s = 124, 10 s = 243, 15 s = 362) -> 5 n + 2 latent frames.
struct VideoGeometry {
  int32_t frames = 0, width = 0, height = 0;
  int32_t latentFrames = 0, latentHeight = 0, latentWidth = 0;  // F, H / 16, W / 16
  int32_t gridH = 0, gridW = 0, tokensPerFrame = 0;             // 2 x 2 patches: H / 32 x W / 32 (768p 24 x 42 = 1008)
  int32_t audioRows = 0;                                        // 2 x round(frames / 24 x 40)

  [[nodiscard]] int32_t videoRows() const { return latentFrames * tokensPerFrame; }
  static VideoGeometry of(int32_t frames, int32_t width, int32_t height);
};

// Packed DiT rows [text L | video conditioning C | audio conditioning Ca | audio Na | video F x T], no pad rows (C and
// Ca are 0 for text-to-video; C holds FL2VA's keyframes or Ref2VA's reference video and image rows, Ca Ref2VA's
// reference audio rows, video/ops/RowLayout.hpp's engine order). Conditioning rows are global rows of the window softmax
// (dense in both directions, as OpenVDN's layout, whose target video is the trailing contiguous run of video rows) and
// are not part of the linear branch. The linear branch covers the inner frames 1 .. F - 2 (anchor frames 0 and F - 1
// excluded).
struct SequenceLayout {
  int32_t text = 0, audio = 0, frames = 0, tokensPerFrame = 0, gridH = 0, gridW = 0, cond = 0;
  int32_t condAudio = 0;  // audio conditioning rows, in front of the target audio (clean, never updated)
  int32_t tailPad = 0;    // rows of zeros kept after the last row (VSA: a query tile reads 64 rows from its start)
  // The dense prefix as segments [start, start + rows) in row order, each cut into 64-row chunks of its own by the
  // VSA tile map; empty: [text], [conditioning], [audio conditioning + audio].
  std::vector<std::pair<int32_t, int32_t>> segments;

  [[nodiscard]] int32_t videoStart() const { return text + cond + condAudio + audio; }
  [[nodiscard]] int32_t audioStart() const { return text + cond; }  // the audio rows, conditioning first
  [[nodiscard]] int32_t audioRows() const { return condAudio + audio; }
  // adaLN table slots per block: video, text, audio (+ conditioning video at max(t, 0.999), + audio at t = 1)
  [[nodiscard]] int32_t adaModalities() const { return condAudio > 0 ? 5 : (cond > 0 ? 4 : 3); }
  // final-layer adaLN slots: video, text, audio, and with audio conditioning rows (which it also runs) 3 and 4
  [[nodiscard]] int32_t finalSlots() const { return condAudio > 0 ? 5 : 3; }
  [[nodiscard]] std::vector<std::pair<int32_t, int32_t>> prefixSegments() const {
    if (!segments.empty()) return segments;
    return {{0, text}, {text, cond}, {audioStart(), audioRows()}};
  }
  [[nodiscard]] int32_t rows() const { return videoStart() + frames * tokensPerFrame; }
  [[nodiscard]] int32_t paddedRows() const { return (rows() + tailPad + 127) / 128 * 128; }
  [[nodiscard]] int32_t innerFrames() const { return frames - 2; }
  [[nodiscard]] int32_t innerRow0() const { return videoStart() + tokensPerFrame; }
  [[nodiscard]] int32_t innerRows() const { return innerFrames() * tokensPerFrame; }
  // Raw q / k rows kept for the branch: text rows, then the inner video rows.
  [[nodiscard]] int32_t rawRows() const { return text + innerRows(); }
};

// VideoVAE.decode: temporal clips of ct latent frames (+ tov overlap tokens), 256 px tiles with >= 64 px overlaps on
// either axis of any canvas.
struct VaePlan {
  int32_t ct = 0, tov = 0, fpp = 0, fov = 0;  // tokens per clip, overlap tokens, frame pre-padding, frame overlap
  int32_t padTokens = 0, chunks = 0, padFrames = 0;
  std::vector<int32_t> ys, ovy, xs, ovx;  // tile starts and overlaps (pixels)
  [[nodiscard]] int32_t clipFrames() const { return 4 * ct - fpp; }
  [[nodiscard]] int32_t decodedFrames() const { return chunks * clipFrames() + fov; }
  [[nodiscard]] int32_t outputFrames() const { return decodedFrames() - padFrames; }
};
[[nodiscard]] VaePlan planVaeDecode(const VideoGeometry &geometry);

// TAEH3 output frames for L latent frames: 4 L - 3 ceil(4 L / 20).
[[nodiscard]] int32_t previewFrames(int32_t latentFrames);


} // namespace splash::video::ops

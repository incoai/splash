#pragma once

// MiniMax-H3 request geometry: latent shapes of a clip, the packed DiT sequence, the video VAE decode plan
// (VideoVAE.decode chunking and _split_tiles tiling of the MLX port) and the latent unpatchify.
#include <cstdint>
#include <vector>

namespace h3::model {

// Clip of 17 n + 5 frames at 24 fps (3 s = 73, 10 s = 243, 15 s = 362) -> 5 n + 2 latent frames.
struct VideoGeometry {
  int32_t frames = 0, width = 0, height = 0;
  int32_t latentFrames = 0, latentHeight = 0, latentWidth = 0;  // F, H / 16, W / 16
  int32_t gridH = 0, gridW = 0, tokensPerFrame = 0;             // 2 x 2 patches: 24 x 42 = 1008
  int32_t audioRows = 0;                                        // 2 x round(frames / 24 x 40)

  [[nodiscard]] int32_t videoRows() const { return latentFrames * tokensPerFrame; }
  static VideoGeometry of(int32_t frames, int32_t width, int32_t height);
};

// Packed DiT rows [text L | keyframe conditioning C | audio Na | video F x T], no pad rows (C = 0 for text-to-video).
// Conditioning rows are global rows of the window softmax (dense in both directions, as OpenVDN's layout, whose
// target video is the trailing contiguous run of video rows) and are not part of the linear branch. The linear branch
// covers the inner frames 1 .. F - 2 (anchor frames 0 and F - 1 excluded).
struct SequenceLayout {
  int32_t text = 0, audio = 0, frames = 0, tokensPerFrame = 0, gridH = 0, gridW = 0, cond = 0;

  [[nodiscard]] int32_t videoStart() const { return text + cond + audio; }
  [[nodiscard]] int32_t audioStart() const { return text + cond; }
  // adaLN table slots per block: video, text, audio (+ conditioning video)
  [[nodiscard]] int32_t adaModalities() const { return cond > 0 ? 4 : 3; }
  [[nodiscard]] int32_t rows() const { return videoStart() + frames * tokensPerFrame; }
  [[nodiscard]] int32_t paddedRows() const { return (rows() + 127) / 128 * 128; }
  [[nodiscard]] int32_t innerFrames() const { return frames - 2; }
  [[nodiscard]] int32_t innerRow0() const { return videoStart() + tokensPerFrame; }
  [[nodiscard]] int32_t innerRows() const { return innerFrames() * tokensPerFrame; }
  // Raw q / k rows kept for the branch: text rows, then the inner video rows.
  [[nodiscard]] int32_t rawRows() const { return text + innerRows(); }
};

// VideoVAE.decode: temporal clips of ct latent frames (+ tov overlap tokens), 256 px tiles with >= 64 px overlaps.
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

// Packed video rows [F gh gw][24 c][2 py][2 px] -> channels-last [F][2 gh][2 gw][24] (x std + mean when given).
void unpatchifyVideo(const float *rows, const VideoGeometry &geometry, const float *std, const float *mean,
                     float *out);

} // namespace h3::model

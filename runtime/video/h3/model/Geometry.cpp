#include "video/h3/model/Geometry.hpp"

#include <algorithm>
#include <cmath>

namespace h3::model {
namespace {

constexpr int32_t kFramesPerChunk = 17, kLatentsPerChunk = 5;
constexpr double kFps = 24.0, kAudioLatentsPerSecond = 40.0;
constexpr int32_t kAudioChannels = 2;
// H3 video VAE (FL2VA/video_vae/config.json): vae_clip_length, vae_token_drop, temporal / spatial compression,
// decoder tile 256 px with >= 64 px overlaps.
constexpr int32_t kClipLength = 17, kTokenDrop = 3, kTemporalRatio = 4, kSpatialRatio = 16;
constexpr int32_t kTilePixels = 256, kTileMinOverlap = 64;

int32_t pyMod(int32_t a, int32_t b) { return ((a % b) + b) % b; }

// VideoVAE._split_tiles: smallest tile count covering length with overlaps >= min_overlap; slack distributed
// round-robin over the overlaps in whole ratio steps.
Status splitTiles(int32_t length, std::vector<int32_t> &starts, std::vector<int32_t> &overlaps) {
  if (kTilePixels >= length) return error(Code::Argument, "frame side %d smaller than one 256 px tile", length);
  int32_t n = (length + kTilePixels - 1) / kTilePixels;
  while (kTilePixels * n - kTileMinOverlap * (n - 1) - length < 0) ++n;
  overlaps.assign(n - 1, kTileMinOverlap);
  const int32_t remaining = kTilePixels * n - kTileMinOverlap * (n - 1) - length;
  for (int32_t i = 0; i < remaining / kSpatialRatio; ++i) overlaps[i % (n - 1)] += kSpatialRatio;
  starts.assign(1, 0);
  for (int32_t i = 0; i < n - 1; ++i) starts.push_back(starts.back() + kTilePixels - overlaps[i]);
  return {};
}

} // namespace

Status VideoGeometry::of(int32_t frames, int32_t width, int32_t height, VideoGeometry &out) {
  if (frames < kFramesPerChunk + kLatentsPerChunk || frames % kFramesPerChunk != kLatentsPerChunk)
    return error(Code::Argument, "frames must be 17 n + 5 (got %d)", frames);
  if (width <= 0 || height <= 0 || width % 32 || height % 32)
    return error(Code::Argument, "width and height must be multiples of 32 (got %d x %d)", width, height);
  VideoGeometry g;
  g.frames = frames;
  g.width = width;
  g.height = height;
  g.latentFrames = (frames - kLatentsPerChunk) / kFramesPerChunk * kLatentsPerChunk + 2;
  g.latentHeight = height / kSpatialRatio;
  g.latentWidth = width / kSpatialRatio;
  g.gridH = g.latentHeight / 2;
  g.gridW = g.latentWidth / 2;
  g.tokensPerFrame = g.gridH * g.gridW;
  // round() of the port is Python's (half to even): nearbyint in the default rounding mode
  const double audioLatents = std::nearbyint(static_cast<double>(frames) / kFps * kAudioLatentsPerSecond);
  g.audioRows = kAudioChannels * static_cast<int32_t>(audioLatents);
  out = g;
  return {};
}

Status planVaeDecode(const VideoGeometry &geometry, VaePlan &out) {
  VaePlan p;
  p.ct = (kClipLength + kTemporalRatio - 1) / kTemporalRatio;
  p.tov = pyMod(-kTokenDrop, p.ct);
  p.fpp = pyMod(-kClipLength, kTemporalRatio);
  p.fov = std::max(p.tov * kTemporalRatio - p.fpp, 0);
  const int32_t tokens = geometry.latentFrames + kTokenDrop;
  p.padTokens = pyMod(-tokens, p.ct);
  p.chunks = (tokens + p.padTokens) / p.ct - (kTokenDrop > 0 ? 1 : 0);
  if (p.chunks < 1) return error(Code::Argument, "too few latent frames for the VAE");
  H3_TRY(splitTiles(geometry.latentHeight * kSpatialRatio, p.ys, p.ovy));
  H3_TRY(splitTiles(geometry.latentWidth * kSpatialRatio, p.xs, p.ovx));
  for (int32_t o : p.ovy)
    if (o % kSpatialRatio) return error(Code::Argument, "tile overlap not latent-aligned");
  for (int32_t o : p.ovx)
    if (o % kSpatialRatio) return error(Code::Argument, "tile overlap not latent-aligned");
  const int32_t intraTail = kClipLength % kTemporalRatio;
  for (int32_t k = 0; k < p.padTokens; ++k)
    p.padFrames += (intraTail && (geometry.latentFrames + k) % p.ct == 0) ? intraTail : kTemporalRatio;
  out = std::move(p);
  return {};
}

int32_t previewFrames(int32_t latentFrames) { return 4 * latentFrames - 3 * ((4 * latentFrames + 19) / 20); }

void unpatchifyVideo(const float *rows, const VideoGeometry &g, const float *std, const float *mean, float *out) {
  const int32_t H = g.latentHeight, W = g.latentWidth, C = 24;
  for (int32_t t = 0; t < g.latentFrames; ++t)
    for (int32_t y = 0; y < g.gridH; ++y)
      for (int32_t x = 0; x < g.gridW; ++x) {
        const float *row = rows + ((static_cast<size_t>(t) * g.gridH + y) * g.gridW + x) * (C * 4);
        for (int32_t py = 0; py < 2; ++py)
          for (int32_t px = 0; px < 2; ++px) {
            float *dst = out + ((static_cast<size_t>(t) * H + 2 * y + py) * W + 2 * x + px) * C;
            for (int32_t c = 0; c < C; ++c) {
              float v = row[c * 4 + py * 2 + px];
              if (std) {
                v = v * std[c];
                v = v + mean[c];
              }
              dst[c] = v;
            }
          }
      }
}

} // namespace h3::model

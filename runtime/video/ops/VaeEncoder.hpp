#pragma once

// Video VAE encoder (kernels/video/vae/encoder.metal; weights <weights>/vae/vae_enc_f16.bin, written by the installer):
// the MLX port's VideoVAE._encode_clip for single frames (the FL2VA keyframe path of pipeline._encode_keyframes):
// 256 px tiles with >= 64 px overlaps (_split_tiles), each through Encoder3d + quant_conv, the posterior moments of
// the tiles blended into one latent frame (_stitch_tiles, fp32 on the host). A single frame makes every causal 3D
// convolution its last temporal tap; the kernels take the frame count and causal taps generally.
// Per tile: conv_in -> 6 levels x (2 ResnetBlock3d (GroupNorm + SiLU -> conv, GroupNorm + SiLU -> conv, + shortcut)
// + Downsample3d) -> GroupNorm + SiLU -> conv_out -> quant_conv, a batch of tiles per command. fp16 weights and conv
// inputs, fp32 accumulation, fp32 residual stream and GroupNorm statistics.
// Videos (Ref2VA reference videos, encodeVideoRgb): AutoencoderKLMiniMaxH3._encode's independent 17-frame clips, each
// tiled as a keyframe over all its frames (causal temporal taps with zero frames before the clip start, temporal
// strides 1, 2, 2: 17 -> 5 latent frames); the last clip's padding and token_drop are folded into encoding its 5 real
// frames alone (the encoder is causal and its GroupNorm per frame, so the kept latent frames are the same).
#include "video/metal/Encoder.hpp"

#include <map>
#include <string>
#include <vector>

namespace splash::video::ops {

// The encoder's weights (the video model layer loads them): fp16 conv weights and fp32 biases, by module name.
struct VaeEncoderConv {
  BufferView w, b;
  int32_t cin = 0, cinPad = 0, cout = 0, coutPad = 0, kd = 0, k = 0;
};
struct VaeEncoderNorm {
  BufferView w, b;
  int32_t c = 0;
};
struct VaeEncoderWeights {
  metal::MetalBuffer blob;  // every tensor; also what a kernel without a norm binds in its place
  std::map<std::string, VaeEncoderConv> convs;
  std::map<std::string, VaeEncoderNorm> norms;
  BufferView quantW, quantB;  // quant_conv
};

struct VaeEncodeStats {
  double wallS = 0, gpuS = 0;
  int32_t tiles = 0, batches = 0;
  int64_t nonfinite = 0;
};

class VaeEncoder {
public:
  static constexpr int32_t kLevels = 6;  // Encoder3d levels: 2 ResnetBlock3d + Downsample3d each

  VaeEncoder(metal::MetalBackend &backend, VaeEncoderWeights weights);

  // pixels: ImageNet-normalised RGB fp32 [n][H][W][3] (H, W multiples of 16) -> posterior moments fp32
  // [n][H / 16][W / 16][48] (mean 24 | logvar 24), channels last.
  void encodeFrames(const float *pixels, int32_t n, int32_t H, int32_t W, std::vector<float> &moments,
                    VaeEncodeStats &stats);

  // uint8 RGB [n][H][W][3] -> the same, after pipeline._encode_keyframes' pixel normalisation
  // ((p / 255 - PIXEL_MEAN) / PIXEL_STD, fp32, ImageNet constants).
  void encodeRgb(const uint8_t *rgb, int32_t n, int32_t H, int32_t W, std::vector<float> &moments,
                 VaeEncodeStats &stats);

  // uint8 RGB [n][H][W][3], n = 17 k + 5 (>= 22): the reference implementation's video encode (_encode: 17-frame
  // clips, the last padded by its final frame, token_drop 3) -> posterior moments fp32 [F][H / 16][W / 16][48]
  // (mean 24 | logvar 24), F = 5 k + 2. Pixels normalised as encodeRgb's.
  void encodeVideoRgb(const uint8_t *rgb, int32_t n, int32_t H, int32_t W, std::vector<float> &moments,
                      VaeEncodeStats &stats);

  static constexpr int32_t kClip = 17;      // clip_length
  static constexpr int32_t kClipLatents = 5;
  static constexpr int32_t kTailFrames = 5;  // the last clip's real frames (17 n + 5 frames)

  // VideoVAE._split_tiles: tile starts and overlaps (pixels) over `length`.
  static void splitTiles(int32_t length, int32_t tile, int32_t minOverlap, std::vector<int32_t> &starts,
                         std::vector<int32_t> &lengths, std::vector<int32_t> &overlaps);

private:
  struct Act {  // fp32 dense [B][H][W][C]
    metal::MetalBuffer buf;
    int32_t H = 0, W = 0, C = 0;
  };
  struct Pad {  // fp16 padded [B][FS][C]
    metal::MetalBuffer buf;
    int32_t H = 0, W = 0, Wp = 0, FS = 0, C = 0;
  };

  void reserve(int32_t frames);  // activation capacity: `frames` 256 px tile frames
  void pad(Encoder &e, const Act &src, Pad &dst, int32_t mode, const VaeEncoderNorm *norm, int32_t cdst, int32_t frames);
  // batch items of `din` frames each (item-major); temporal stride 2 leaves (din - 1) / 2 + 1 frames per item
  void conv(Encoder &e, const Pad &src, const VaeEncoderConv &w, int32_t stride, int32_t tstride, Act &dst, const Act *residual,
            int32_t batch, int32_t din);
  // `batch` tiles of `din` frames each -> moments_ [batch][return value][lh][lw][48]
  int32_t encodeBatch(Encoder &e, int32_t batch, int32_t din, int32_t tileH, int32_t tileW);

  metal::MetalBackend &b_;
  VaeEncoderWeights w_;
  int32_t capH_ = 0, capW_ = 0;  // activation capacity (tile size)
  int32_t capFrames_ = 0;        // tile frames per command
  Act x_[2], h_, s_, out_, moments_;
  Act pixels_;
  Pad p_, p2_;
  metal::MetalBuffer stats_;
};

} // namespace splash::video::ops

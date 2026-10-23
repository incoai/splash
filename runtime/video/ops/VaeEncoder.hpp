#pragma once

// Video VAE encoder (kernels/vae/encoder.metal; weights <weights>/vae/vae_enc_f16.bin, convert_vae.py --encoder):
// the MLX port's VideoVAE._encode_clip for single frames (the FL2VA keyframe path of pipeline._encode_keyframes):
// 256 px tiles with >= 64 px overlaps (_split_tiles), each through Encoder3d + quant_conv, the posterior moments of
// the tiles blended into one latent frame (_stitch_tiles, fp32 on the host). A single frame makes every causal 3D
// convolution its last temporal tap; the kernels take the frame count and causal taps generally.
// Per tile: conv_in -> 6 levels x (2 ResnetBlock3d (GroupNorm + SiLU -> conv, GroupNorm + SiLU -> conv, + shortcut)
// + Downsample3d) -> GroupNorm + SiLU -> conv_out -> quant_conv, a batch of tiles per command. fp16 weights and conv
// inputs, fp32 accumulation, fp32 residual stream and GroupNorm statistics.
#include "video/metal/Encoder.hpp"

#include <map>
#include <string>
#include <vector>

namespace splash::video::ops {

// The encoder's weights (model/VaeWeights.hpp loads them): fp16 conv weights and fp32 biases, by module name.
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

  void pad(Encoder &e, const Act &src, Pad &dst, int32_t mode, const VaeEncoderNorm *norm, int32_t cdst, int32_t frames);
  void conv(Encoder &e, const Pad &src, const VaeEncoderConv &w, int32_t stride, int32_t tstride, Act &dst, const Act *residual,
            int32_t frames);
  void encodeBatch(Encoder &e, int32_t frames, int32_t tileH, int32_t tileW);

  metal::MetalBackend &b_;
  VaeEncoderWeights w_;
  int32_t capH_ = 0, capW_ = 0;  // activation capacity (tile size)
  Act x_[2], h_, s_, out_, moments_;
  Act pixels_;
  Pad p_, p2_;
  metal::MetalBuffer stats_;
};

} // namespace splash::video::ops

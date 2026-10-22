#pragma once

// TAEH3 preview decoder (kernels/preview): per temporal chunk of 5 latent frames (MemBlock state carried across chunks
// in the memory slots): tanh(x / 3) * 3 -> conv 24 -> 256 + relu -> 3 x MemBlock(256) -> TGrow (t x1) -> up 2x +
// conv 256 -> 128 -> 3 x MemBlock(128) -> TGrow (t x2) -> up 2x + conv 128 -> 64 -> 3 x MemBlock(64) -> TGrow (t x2)
// -> up 2x + conv 64 -> 64 + relu -> conv 64 -> 12 + pixel shuffle 2x + clamp -> uint8 frames (H3 frame drop: 3 of
// every 20 decoded frames), optionally yuv420p for the encoder. One command per chunk; the host copies chunk k - 1's
// frames while chunk k runs. fp16 weights and activations, fp32 accumulation and epilogues.

#include "video/metal/Encoder.hpp"

#include <array>
#include <string>
#include <vector>

namespace splash::video::ops {

// The decoder's weights (model/PreviewWeights.hpp loads them): fp16 conv weights and fp32 biases.
//   conv3x3 W[src][tap 9][Cout][Cin] (MemBlock conv.0: src 0 = x, src 1 = past), tgrow 1x1 W[Cout][Cin],
//   up (nearest 2x + 3x3) W[phase 4][tap 2x2][Cout][Cin] pre-combined.
struct PreviewMemBlock {
  BufferView conv0, bias0, conv2, bias2, conv4, bias4;
};

struct PreviewDecoderWeights {
  BufferView inputConv, inputBias;           // decoder.1: 24 (32) -> 256
  std::array<PreviewMemBlock, 9> memBlocks;  // decoder.3-5, 9-11, 15-17
  std::array<BufferView, 3> tgrow;           // decoder.7, 13, 19
  std::array<BufferView, 3> upsample;        // decoder.8, 14, 20
  BufferView outputConv, outputBias;         // decoder.22: 64 -> 12 (16)
};

struct PreviewStats {
  double wallS = 0, gpuS = 0, hostCopyS = 0;
  int64_t nonfinite = 0;
  int32_t chunks = 0, frames = 0;
  double deviceMb = 0;
};

class PreviewDecoder {
public:
  PreviewDecoder(metal::MetalBackend &backend, PreviewDecoderWeights weights, int32_t latentHeight,
                 int32_t latentWidth);

  // latents: normalized fp32 [L][LH][LW][24], L = 5 k + 2. frames: previewFrames(L) x [16 LH][16 LW][3] uint8;
  // yuv (nullable): yuv420p frames [F][1.5 x 16 LH x 16 LW].
  void decode(const float *latents, int32_t latentFrames, uint8_t *frames, uint8_t *yuv, PreviewStats &stats);

private:
  struct Activation {  // fp16 padded frames with one leading memory slot
    metal::MetalBuffer buffer;
    int32_t H = 0, W = 0, Wp = 0, FS = 0, C = 0;
    [[nodiscard]] size_t base() const { return static_cast<size_t>(FS) * C * 2; }  // byte offset of frame 0
  };
  struct Conv {
    const char *kind;  // preview_conv_relu | _memory | _residual, preview_tgrow, preview_upsample, preview_conv_out
    std::string tile;
    const Activation *src;
    int32_t frames;
    BufferView weight, bias;
    const Activation *dst, *residual;
    int32_t cout, split, relu;
    bool upsample;
    int slot;  // output staging slot (preview_conv_out)
  };

  [[nodiscard]] Activation activation(int32_t H, int32_t W, int32_t C, int32_t frames);
  void conv(Encoder &e, const Conv &c);
  void encodeChunk(Encoder &e, int slot, int32_t frames, int32_t kept, bool yuv);

  metal::MetalBackend &b_;
  PreviewDecoderWeights w_;
  int32_t LH_ = 0, LW_ = 0, chunk_ = 0;
  std::string tile256_, tile128_, tile64_, tileOut_;  // conv tile per channel level (DeviceProfile)
  Activation input_, upsampled_;
  std::array<std::array<Activation, 4>, 3> blocks_{};  // MemBlock inputs / outputs per level
  std::array<Activation, 3> y1_{}, y2_{}, grown_{};
  std::array<metal::MetalBuffer, 2> latents_{}, frames_{}, yuv_{}, frameMap_{};
  metal::MetalBuffer bad_, null_;
};

} // namespace splash::video::ops

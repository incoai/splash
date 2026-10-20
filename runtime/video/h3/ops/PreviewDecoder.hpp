#pragma once

// TAEH3 preview decoder (kernels/preview): per temporal chunk of 5 latent frames (MemBlock state carried across chunks
// in the memory slots): tanh(x / 3) * 3 -> conv 24 -> 256 + relu -> 3 x MemBlock(256) -> TGrow (t x1) -> up 2x +
// conv 256 -> 128 -> 3 x MemBlock(128) -> TGrow (t x2) -> up 2x + conv 128 -> 64 -> 3 x MemBlock(64) -> TGrow (t x2)
// -> up 2x + conv 64 -> 64 + relu -> conv 64 -> 12 + pixel shuffle 2x + clamp -> uint8 frames (H3 frame drop: 3 of
// every 20 decoded frames), optionally yuv420p for the encoder. One command buffer per chunk; the host copies chunk
// k - 1's frames while chunk k runs. fp16 weights and activations, fp32 accumulation and epilogues.
#import <Metal/Metal.h>

#include "video/h3/engine/Types.hpp"
#include "video/h3/metal/MetalBackend.hpp"
#include "video/h3/model/PreviewWeights.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace h3::ops {

class PreviewDecoder {
public:
  static Status create(metal::MetalBackend &backend, std::unique_ptr<model::PreviewWeights> weights,
                       int32_t latentHeight, int32_t latentWidth, std::unique_ptr<PreviewDecoder> &out);

  // latents: normalized fp32 [L][LH][LW][24], L = 5 k + 2. frames: previewFrames(L) x [16 LH][16 LW][3] uint8;
  // yuv (nullable): yuv420p frames [F][1.5 x 16 LH x 16 LW].
  Status decode(const float *latents, int32_t latentFrames, uint8_t *frames, uint8_t *yuv, PreviewStats &stats);

private:
  struct Activation {  // fp16 padded frames with one leading memory slot
    id<MTLBuffer> buffer = nil;
    int32_t H = 0, W = 0, Wp = 0, FS = 0, C = 0;
    [[nodiscard]] size_t base() const { return static_cast<size_t>(FS) * C * 2; }  // byte offset of frame 0
  };
  struct Conv {
    const char *kind;  // preview_conv_relu | _memory | _residual, preview_tgrow, preview_upsample, preview_conv_out
    std::string tile;
    const Activation *src;
    int32_t frames;
    metal::BufferView weight, bias;
    const Activation *dst, *residual;
    int32_t cout, split, relu;
    bool upsample;
    int slot;  // output staging slot (preview_conv_out)
  };

  explicit PreviewDecoder(metal::MetalBackend &backend) : b_(backend) {}
  Status activation(int32_t H, int32_t W, int32_t C, int32_t frames, Activation &out);
  void conv(id<MTLComputeCommandEncoder> e, const Conv &c);
  void encodeChunk(id<MTLComputeCommandEncoder> e, int slot, int32_t frames, int32_t kept, bool yuv);

  metal::MetalBackend &b_;
  std::unique_ptr<model::PreviewWeights> w_;
  int32_t LH_ = 0, LW_ = 0, chunk_ = 0;
  std::string tile256_, tile128_, tile64_, tileOut_;  // conv tile per channel level (DeviceProfile)
  Activation input_, upsampled_;
  std::array<std::array<Activation, 4>, 3> blocks_{};  // MemBlock inputs / outputs per level
  std::array<Activation, 3> y1_{}, y2_{}, grown_{};
  std::array<id<MTLBuffer>, 2> latents_{}, frames_{}, yuv_{}, frameMap_{};
  id<MTLBuffer> bad_ = nil, null_ = nil;
};

} // namespace h3::ops

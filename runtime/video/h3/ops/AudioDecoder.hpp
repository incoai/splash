#pragma once

// Audio VAE decoder (kernels/audio): packed DiT audio rows [2 n][32] -> x std + mean -> dec_in_proj (1x1, 32 -> 2048)
// -> BigVGAN: conv_pre (k7, 2048 -> 1024), 7 x [transposed conv upsampler (rates 5 5 2 2 2 2 2) -> mean of 3 AMP
// blocks (kernels 3 / 7 / 11; per dilation 1 / 3 / 5: x += conv2(act(conv1(act(x)))), act = anti-aliased SnakeBeta)]
// -> act -> conv_post (k7, 8 -> 1) -> clip -> fp32 waveform [2][800 n] at 32 kHz. Stereo is two batch items.
// Conv inputs are fp16 hi / lo pairs and weights fp16 hi / lo pairs (three fp16 products per tap, fp32 accumulate);
// residual streams fp32. One command buffer per decode.
#import <Metal/Metal.h>

#include "video/h3/engine/Types.hpp"
#include "video/h3/metal/MetalBackend.hpp"
#include "video/h3/model/AudioWeights.hpp"

#include <array>
#include <memory>
#include <string>

namespace h3::ops {

class AudioDecoder {
public:
  static Status create(metal::MetalBackend &backend, std::unique_ptr<model::AudioWeights> weights,
                       std::unique_ptr<AudioDecoder> &out);

  // rows: [2 n][32] fp32 (channel-major, as the DiT emits them); wav: [2][800 n] fp32.
  Status decode(const float *rows, int32_t n, float *wav, AudioStats &stats);
  [[nodiscard]] static int64_t samples(int32_t n) { return int64_t(n) * 800; }

private:
  struct Pair {  // fp16 hi / lo padded activations [2][P + T + P][C] with slack rows before and after
    id<MTLBuffer> hi = nil, lo = nil;
  };
  struct Conv;

  explicit AudioDecoder(metal::MetalBackend &backend) : b_(backend) {}
  Status reserve(int32_t n);
  void conv(id<MTLComputeCommandEncoder> e, const Conv &c);
  void act(id<MTLComputeCommandEncoder> e, id<MTLBuffer> x, int32_t T, int32_t C, metal::BufferView snake);
  void zeroPads(id<MTLComputeCommandEncoder> e, const Pair &pair, int32_t T, int32_t C);
  void encode(id<MTLComputeCommandEncoder> e, int32_t n);

  metal::MetalBackend &b_;
  std::unique_ptr<model::AudioWeights> w_;
  int32_t capacity_ = 0;  // latents the buffers were sized for
  Pair a_, b2_;
  std::array<id<MTLBuffer>, 3> blocks_{};  // AMP-block residual streams
  id<MTLBuffer> x0_ = nil, r_ = nil, rows_ = nil, wav_ = nil, null_ = nil;
};

} // namespace h3::ops

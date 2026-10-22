#pragma once

// Audio VAE decoder (kernels/audio): packed DiT audio rows [2 n][32] -> x std + mean -> dec_in_proj (1x1, 32 -> 2048)
// -> BigVGAN: conv_pre (k7, 2048 -> 1024), 7 x [transposed conv upsampler (rates 5 5 2 2 2 2 2) -> mean of 3 AMP
// blocks (kernels 3 / 7 / 11; per dilation 1 / 3 / 5: x += conv2(act(conv1(act(x)))), act = anti-aliased SnakeBeta)]
// -> act -> conv_post (k7, 8 -> 1) -> clip -> fp32 waveform [2][800 n] at 32 kHz. Stereo is two batch items.
// Conv inputs are fp16 hi / lo pairs and weights fp16 hi / lo pairs (three fp16 products per tap, fp32 accumulate);
// residual streams fp32. One command per decode.

#include "video/metal/Encoder.hpp"

#include <array>
#include <string>

namespace splash::video::ops {

// The decoder's weights (model/AudioWeights.hpp loads them): fp16 hi / lo conv weights W[tap][Cout][Cin], fp32 biases,
// SnakeBeta parameters [2][C] (exp(alpha), 1 / (exp(beta) + 1e-9)), the 12-tap anti-aliasing filter and the latent
// de-normalisation.
struct AudioConv {
  BufferView w, wl, b;  // b empty for conv_post
  int32_t taps = 0, cout = 0, cin = 0;
};

// One upsampling stage: the transposed conv, then 3 AMP blocks of 3 dilations.
struct AudioStageWeights {
  AudioConv up;
  std::array<std::array<AudioConv, 3>, 3> convs1, convs2;  // [block][dilation]
  std::array<std::array<BufferView, 6>, 3> acts;          // [block][activation]: convs1[d] <- 2d, convs2[d] <- 2d+1
};

struct AudioDecoderWeights {
  AudioConv decIn, convPre, convPost;
  std::array<AudioStageWeights, 7> stages;
  BufferView postAct, filter, mean, std;
};

struct AudioStats {
  double wallS = 0, gpuS = 0;
  int64_t samples = 0, nonfinite = 0;  // per channel; non-finite values in the waveform
  double deviceMb = 0;
};

class AudioDecoder {
public:
  // BigVGAN: 7 upsampling stages (rates 5 5 2 2 2 2 2), 3 AMP blocks per stage (kernels 3 / 7 / 11, dilations
  // 1 / 3 / 5), channels 1024 / 2^(stage + 1).
  static constexpr int kStages = 7;
  static constexpr int kRates[kStages] = {5, 5, 2, 2, 2, 2, 2};
  static constexpr int kResKernels[3] = {3, 7, 11};
  static constexpr int kDilations[3] = {1, 3, 5};
  static constexpr int kDecoderDim = 1024, kLatentDim = 2048, kSampleRate = 32000;

  AudioDecoder(metal::MetalBackend &backend, AudioDecoderWeights weights);

  // rows: [2 n][32] fp32 (channel-major, as the DiT emits them); wav: [2][800 n] fp32.
  void decode(const float *rows, int32_t n, float *wav, AudioStats &stats);
  [[nodiscard]] static int64_t samples(int32_t n) { return int64_t(n) * 800; }

private:
  struct Pair {  // fp16 hi / lo padded activations [2][P + T + P][C] with slack rows before and after
    metal::MetalBuffer hi, lo;
  };
  struct Conv;

  void reserve(int32_t n);
  void conv(Encoder &e, const Conv &c);
  void act(Encoder &e, const metal::MetalBuffer &x, int32_t T, int32_t C, BufferView snake);
  void zeroPads(Encoder &e, const Pair &pair, int32_t T, int32_t C);
  void encode(Encoder &e, int32_t n);

  metal::MetalBackend &b_;
  AudioDecoderWeights w_;
  int32_t capacity_ = 0;  // latents the buffers were sized for
  Pair a_, b2_;
  std::array<metal::MetalBuffer, 3> blocks_{};  // AMP-block residual streams
  metal::MetalBuffer x0_, r_, rows_, wav_, null_;
};

} // namespace splash::video::ops

#pragma once

// Audio VAE encoder (kernels/video/audio): a stereo soundtrack fp32 [2][S] at 32 kHz -> the normalised posterior mean
// [T][64], T = ceil(S / 800), per latent channel 0's 32 | channel 1's 32 (the ReferenceInput::audio layout
// that requests use). Stereo is two batch items of the mono VAE; the waveform is right-padded with zeros to 800 T.
//   GPU: DAC encoder: conv_in (k7, 1 -> 64) -> 5 blocks (strides 2 4 4 5 5; channels C = 64 .. 1024): 3 residual
//   units x += conv1(snake(conv7_d(snake(x)))) (dilations 1 / 3 / 9) -> snake -> strided conv (k 2s, pad ceil(s / 2),
//   C -> 2 C) -> snake -> conv (k3, 2048 -> 2048). Conv inputs and weights fp16 hi / lo pairs (three fp16 products
//   per tap, fp32 accumulate, as the decoder); residual streams fp32. One command per encode.
//   Host, fp32: pre_block (LayerNorm, causal attention with 8 heads mean-pooled and adaptive-avg-pooled to 32,
//   + proj(norm3(x)), + GeGLU MLP), mean_proj, (mean - latents_mean) / latents_std.
#include "video/ops/AudioDecoder.hpp"

#include <array>
#include <vector>

namespace splash::video::ops {

// The encoder's weights (loadAudioEncoder of the video model layer): GPU convs as the decoder's AudioConv (the strided
// convs with taps = 2 row pairs and cin = s Cin), Snake1d [2][C] (alpha, 1 / (alpha + 1e-9)); the host part as fp32
// views into the same blob (linear weights transposed [in][out]).
struct AudioEncoderWeights {
  struct Unit {
    BufferView act1, act2;
    AudioConv conv7, conv1;
  };
  struct Block {
    std::array<Unit, 3> units;
    BufferView act;
    AudioConv down;
  };
  struct Linear {
    const float *wt = nullptr, *b = nullptr;  // [in][out], [out]
  };
  AudioConv convIn, convOut;
  std::array<Block, 5> blocks;
  BufferView finalAct;
  // pre_block / mean_proj / normalisation (host, fp32)
  const float *norm1W = nullptr, *norm1B = nullptr, *norm2W = nullptr, *norm2B = nullptr, *norm3W = nullptr,
              *norm3B = nullptr, *mlpNormW = nullptr, *mlpNormB = nullptr, *qkvT = nullptr, *qBias = nullptr,
              *vBias = nullptr, *mean = nullptr, *std = nullptr;
  Linear attnProj, proj, w0, w1, w2, meanProj;
};

struct AudioEncodeStats {
  double wallS = 0, gpuS = 0;
  int64_t nonfinite = 0;
};

class AudioEncoder {
public:
  static constexpr int kBlocks = 5;
  static constexpr int kRates[kBlocks] = {2, 4, 4, 5, 5};
  static constexpr int kDilations[3] = {1, 3, 9};
  static constexpr int kEncoderDim = 64, kLatentDim = 2048, kLatent = 32, kHeads = 8, kHop = 800;

  AudioEncoder(metal::MetalBackend &backend, AudioEncoderWeights weights);

  // wav: fp32 [2][samples] at 32 kHz (channel-major, as the soundtrack after Pika's truncation to the clip);
  // latents: normalized posterior mean [T][64], T = ceil(samples / 800), per latent channel 0's 32 | channel 1's 32.
  void encode(const float *wav, int64_t samples, std::vector<float> &latents, AudioEncodeStats &stats);

private:
  struct Pair {  // fp16 hi / lo padded [2][P + T + P][C], slack rows before, slack + tail rows after
    metal::MetalBuffer hi, lo;
  };

  void reserve(int32_t T0, int64_t samples);
  void conv(Encoder &e, const AudioConv &w, int32_t T, int32_t dil, int32_t pad, bool residual);
  void down(Encoder &e, const AudioConv &w, int32_t T, int32_t stride);
  void snake(Encoder &e, const metal::MetalBuffer &x, int32_t T, int32_t C, BufferView act);
  void record(Encoder &e, int32_t T0, int64_t samples);
  void preBlock(const float *x, int32_t T, float *out) const;  // one channel: [T][2048] -> normalised [T][32]

  metal::MetalBackend &b_;
  AudioEncoderWeights w_;
  int32_t capacity_ = 0;     // padded samples per channel the buffers were sized for
  int64_t wavCapacity_ = 0;  // samples per channel of wav_
  Pair a_;
  metal::MetalBuffer x_, r_, wav_, null_;
};

} // namespace splash::video::ops

#pragma once

// Converted audio VAE decoder (tools/convert_audio.py from FL2VA/audio_vae): fp16 hi / lo conv weights W[tap][Cout][Cin],
// fp32 biases, SnakeBeta parameters [2][C] (exp(alpha), 1 / (exp(beta) + 1e-9)), the 12-tap anti-aliasing filter and
// the latent de-normalisation, in one blob audio_dec_f16.bin + manifest.json (byte offsets, 256-byte aligned).
#import <Metal/Metal.h>

#include "video/h3/engine/Status.hpp"
#include "video/h3/metal/MetalBackend.hpp"

#include <array>
#include <memory>
#include <string>

namespace h3::model {

struct AudioConv {
  metal::BufferView w, wl, b;  // b empty for conv_post
  int32_t taps = 0, cout = 0, cin = 0;
};

// BigVGAN decoder: 7 upsampling stages (rates 5 5 2 2 2 2 2), 3 AMP blocks per stage (kernels 3 / 7 / 11, dilations
// 1 / 3 / 5), channels 1024 / 2^(stage + 1).
struct AudioStageWeights {
  AudioConv up;
  std::array<std::array<AudioConv, 3>, 3> convs1, convs2;  // [block][dilation]
  std::array<std::array<metal::BufferView, 6>, 3> acts;     // [block][activation]: convs1[d] <- 2d, convs2[d] <- 2d+1
};

class AudioWeights {
public:
  static constexpr int kStages = 7;
  static constexpr int kRates[kStages] = {5, 5, 2, 2, 2, 2, 2};
  static constexpr int kResKernels[3] = {3, 7, 11};
  static constexpr int kDilations[3] = {1, 3, 5};
  static constexpr int kDecoderDim = 1024, kLatentDim = 2048, kSampleRate = 32000;

  static Status load(metal::MetalBackend &backend, const std::string &dir, std::unique_ptr<AudioWeights> &out);

  AudioConv decIn, convPre, convPost;
  std::array<AudioStageWeights, kStages> stages;
  metal::BufferView postAct, filter, mean, std;
  [[nodiscard]] size_t bytes() const { return blob_ ? blob_.length : 0; }

private:
  id<MTLBuffer> blob_ = nil;
};

} // namespace h3::model

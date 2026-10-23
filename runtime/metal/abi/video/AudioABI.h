#pragma once

// Audio VAE decoder (BigVGAN, kernels/audio): host / kernel structs and shapes.
//
// Conv inputs are fp16 hi / lo pairs (x = hi + lo, ~22-bit) in a padded channels-last layout per stereo channel:
// [2][P + T + P][C], pad rows zero. Weights are fp16 hi / lo pairs W[tap][Cout][Cin]; a matmul2d conv accumulates
// hi*hi + lo*hi + hi*lo in fp32 (fp16-split arithmetic; fp16 x fp16 alone is ~42 dB against fp32, see
// tools/convert_audio.py). Residual streams and conv outputs that feed an activation are fp32 [2][T][C].
#include "metal/abi/video/Layout.h"

#define H3_AUDIO_PAD 32          // pad rows on each side of a channel's sequence (max conv reach is 25)
#define H3_AUDIO_SLACK 64        // rows of slack before and after every padded buffer (GEMM edge reads)
#define H3_AUDIO_FILTER 12       // Kaiser-sinc taps of the 2x anti-aliasing up / down samplers
#define H3_AUDIO_LATENT 32       // latent channels
#define H3_AUDIO_HOP 800         // samples per latent
#define H3_AUDIO_ACT_ROWS 64     // output rows per activation threadgroup

// audio_conv_* / audio_small_*: out[t] = bias + sum_j X[t + j dil - pad] W[j] (conv), or the transposed conv
// out[s i + tap - upad] += X[i] W[tap] by output phase (grid z = phase).
struct AudioConvParams {
  int32_t Cin, Cout;
  int32_t T, Mrows;      // input rows per channel; GEMM domain rows over the padded input layout (tile multiple)
  int32_t taps, dil, pad;
  int32_t stride, upad;  // transposed conv
  int32_t Tout;          // output rows per channel (T, or stride * T)
  int32_t pad0, pad1;
};

// audio_act_*: anti-aliased SnakeBeta (2x Kaiser-sinc upsample, x + b sin(a x)^2, 2x Kaiser-sinc downsample),
// fp32 [2][T][C] -> fp16 hi / lo padded. audio_input, audio_sum3_*: the stage glue.
struct AudioActParams {
  int32_t C, T, n, pad0;
};

#pragma once

// Audio VAE decoder (BigVGAN, kernels/video/audio): host / kernel structs and shapes.
//
// Conv inputs are fp16 hi / lo pairs (x = hi + lo, ~22-bit) in a padded channels-last layout per stereo channel:
// [2][P + T + P][C], pad rows zero. Weights are fp16 hi / lo pairs W[tap][Cout][Cin]; a matmul2d conv accumulates
// hi*hi + lo*hi + hi*lo in fp32 (fp16-split arithmetic; fp16 x fp16 alone is ~42 dB against fp32, which is why
// the installer stores hi / lo pairs). Residual streams and conv outputs that feed an activation are fp32 [2][T][C].
#include "metal/abi/video/Layout.h"

#define H3_AUDIO_PAD 32          // pad rows on each side of a channel's sequence (max conv reach is 25)
#define H3_AUDIO_SLACK 64        // rows of slack before and after every padded buffer (GEMM edge reads)
#define H3_AUDIO_FILTER 12       // Kaiser-sinc taps of the 2x anti-aliasing up / down samplers
#define H3_AUDIO_LATENT 32       // latent channels
#define H3_AUDIO_ACT_ROWS 64     // output rows per activation threadgroup
#define H3_AUDIO_ENC_TAIL 384    // encoder: rows after a padded buffer read by the strided convs edge tiles (64 x 5 + 64)

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

// Audio VAE encoder (DAC, kernels/video/audio/enc.metal). audio_enc_in: conv_in (k7, 1 -> 64) of the zero-extended waveform
// [2][n] -> fp32 [2][T][64] (AudioActParams: C 64, T padded samples, n samples). audio_snake: Snake1d
// x + sin(a x)^2 / (a + 1e-9) of fp32 [2][T][C] -> hi / lo padded (AudioActParams). audio_down_*: the strided
// conv (k 2s, stride s, pad p) as a GEMM over K = s Cin rows of the padded input (AudioConvParams: T input rows,
// Mrows output rows (tile multiple), taps 2 (row pairs), stride s, pad p, Tout = T / s; grid z = stereo channel),
// weights W[q][Cout][s Cin] (W[q][co][r Cin + ci] = conv weight [co][ci][q s + r]).

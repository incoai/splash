#pragma once

// Parameter layouts of the ASR audio tower's front-end kernels
// (kernels/omni/audio_encoder.metal) and their host dispatch
// (ops/OmniAudioEncoder.hpp). The tower's shapes are fixed: 128 mels in
// 100-frame chunks, three 3x3 stride-2 convolutions of 480 channels, 13
// tokens of 1024 per chunk, 2048-dim outputs.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#define OMNI_AUDIO_MEL_BINS 128u
#define OMNI_AUDIO_CHUNK_FRAMES 100u
#define OMNI_AUDIO_CHUNK_TOKENS 13u
#define OMNI_AUDIO_CONV_CHANNELS 480u
// The conv2/conv3 GEMMs run on ops::DenseLinear, whose K and N are whole
// 64s: the 480 x 3 x 3 taps are padded to 4352 columns, the 480 output
// channels to 512 rows (zero weights, zero bias).
#define OMNI_AUDIO_CONV_TAPS (OMNI_AUDIO_CONV_CHANNELS * 9u)
#define OMNI_AUDIO_CONV_PADDED_TAPS 4352u
#define OMNI_AUDIO_CONV_PADDED_CHANNELS 512u
#define OMNI_AUDIO_CONV1_HEIGHT 64u
#define OMNI_AUDIO_CONV2_HEIGHT 32u
#define OMNI_AUDIO_CONV3_HEIGHT 16u
#define OMNI_AUDIO_HIDDEN 1024u
// conv_out's input: the 480 channels x 16 frequency rows of a conv3 frame,
// channel-major (c * 16 + f).
#define OMNI_AUDIO_CONV_OUT_INPUT (OMNI_AUDIO_CONV_CHANNELS * OMNI_AUDIO_CONV3_HEIGHT)
#define OMNI_AUDIO_FFN 4096u
#define OMNI_AUDIO_OUTPUT 2048u
#define OMNI_AUDIO_WINDOW_TOKENS 104u

// conv2d1 over a group of chunks: 1 -> 480 channels, 3x3, stride 2, pad 1,
// GELU, direct. Reads the mel [frames][128] (frame-major) at chunk c's
// frames [c * 100, c * 100 + chunk_frames), zero past mel_frames; writes
// bf16 [chunk][64][output_width][480] (frequency rows, then time, channels
// innermost). chunk_frames is 100, or the frame count of a lone chunk.
struct OmniAudioConv1Params {
  uint32_t first_chunk;
  uint32_t chunks;
  uint32_t mel_frames;
  uint32_t chunk_frames;
  uint32_t output_width;
  uint32_t padding[3];
};
static_assert(sizeof(OmniAudioConv1Params) == 32, "conv1 parameters are 32 bytes on both sides");

// im2col of a 3x3 stride-2 pad-1 convolution over [chunk][input_height]
// [input_width][input_stride] bf16 pixels of 480 channels: row (chunk, ho,
// wo) of the output holds its taps at column (kh * 3 + kw) * 480 + ci, the
// 32 padding columns zero.
struct OmniAudioIm2colParams {
  uint32_t chunks;
  uint32_t input_height;
  uint32_t input_width;
  uint32_t input_stride;
  uint32_t output_height;
  uint32_t output_width;
  uint32_t rows;
  uint32_t padding;
};
static_assert(sizeof(OmniAudioIm2colParams) == 32, "im2col parameters are 32 bytes on both sides");

// The conv3 output [chunk][16][width][512] of a group of chunks to the
// conv_out rows [token][7680] (c * 16 + f): chunk c's frame t is token
// c * 13 + t when t is below its valid frames (13, or last_chunk_tokens for
// the last chunk).
struct OmniAudioFlattenParams {
  uint32_t first_chunk;
  uint32_t chunks;
  uint32_t width;
  uint32_t last_chunk;
  uint32_t last_chunk_tokens;
  uint32_t padding[3];
};
static_assert(sizeof(OmniAudioFlattenParams) == 32, "flatten parameters are 32 bytes on both sides");

// rows[r][d] += table[(r % period)][d] over fp32 rows of `width`: the
// sinusoidal positions restarting every chunk.
struct OmniAudioPositionParams {
  uint32_t rows;
  uint32_t width;
  uint32_t period;
  uint32_t padding;
};
static_assert(sizeof(OmniAudioPositionParams) == 16, "position parameters are 16 bytes on both sides");

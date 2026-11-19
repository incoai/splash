// The ASR audio tower's front end (ops::omni::AudioEncoder): conv2d1 as a
// direct 9-tap kernel, im2col for conv2d2 and conv2d3 (whose GEMMs run on
// omni_gemm_bf16, kernels/omni/gemm_bf16.metal), the channel-major flatten
// into conv_out's rows, and the per-chunk sinusoidal positions. Activations
// between the convolutions are bf16 pixels with channels innermost (NHWC:
// chunk, frequency, time, channel), the layout the MLX reference convolves
// and the im2col reads contiguously.

#include "metal/abi/OmniAudioEncoder.h"
#include <metal_stdlib>

using namespace metal;

// Abramowitz & Stegun 7.1.26, as the dense GEMM epilogue computes GELU.
inline float omni_audio_erf(float x) {
  const float sign = x < 0.0f ? -1.0f : 1.0f;
  x = fabs(x);
  const float t = 1.0f / (1.0f + 0.3275911f * x);
  const float polynomial =
      t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
  return sign * (1.0f - polynomial * precise::exp(-x * x));
}
inline float omni_audio_gelu(float x) { return 0.5f * x * (1.0f + omni_audio_erf(x * 0.7071067811865476f)); }

// One thread per eight output channels of a pixel (chunk, ho, wo, co..co+7),
// channels innermost: the nine mel taps are loaded once for the eight, whose
// 72 weights are contiguous, and the eight outputs stored as one 16-byte
// word. Each channel sums its taps in the same order as alone.
kernel void omni_audio_conv1(device const float *mel [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
                             device const float *bias [[buffer(2)]], device bfloat *output [[buffer(3)]],
                             constant OmniAudioConv1Params &p [[buffer(4)]],
                             uint index [[thread_position_in_grid]]) {
  constexpr uint vectors = OMNI_AUDIO_CONV_CHANNELS / 8u;
  constexpr uint height = OMNI_AUDIO_CONV1_HEIGHT;
  const uint co = (index % vectors) * 8u;
  const uint pixel = index / vectors;
  const uint wo = pixel % p.output_width;
  const uint ho = pixel / p.output_width % height;
  const uint g = pixel / p.output_width / height;
  if (g >= p.chunks) return;
  const uint chunk = p.first_chunk + g;
  float sum[8];
#pragma unroll
  for (uint c = 0; c < 8; ++c) sum[c] = bias[co + c];
  for (uint kh = 0; kh < 3; ++kh) {
    const int hi = int(ho) * 2 - 1 + int(kh);
    if (hi < 0 || hi >= int(OMNI_AUDIO_MEL_BINS)) continue;
    for (uint kw = 0; kw < 3; ++kw) {
      const int wi = int(wo) * 2 - 1 + int(kw);
      if (wi < 0 || wi >= int(p.chunk_frames)) continue;
      const uint frame = chunk * OMNI_AUDIO_CHUNK_FRAMES + uint(wi);
      if (frame >= p.mel_frames) continue;
      const float value = mel[frame * OMNI_AUDIO_MEL_BINS + uint(hi)];
#pragma unroll
      for (uint c = 0; c < 8; ++c) sum[c] += value * float(weight[(co + c) * 9u + kh * 3u + kw]);
    }
  }
  vec<bfloat, 8> out;
#pragma unroll
  for (uint c = 0; c < 8; ++c) out[c] = bfloat(omni_audio_gelu(sum[c]));
  *reinterpret_cast<device vec<bfloat, 8> *>(output + ulong(pixel) * OMNI_AUDIO_CONV_CHANNELS + co) = out;
}

// One thread per eight output elements (row, k..k+7), eight consecutive
// input channels of one tap as one 16-byte load and store (480 channels,
// the input strides and the 4352 columns are whole multiples of eight).
kernel void omni_audio_im2col(device const bfloat *input [[buffer(0)]], device bfloat *output [[buffer(1)]],
                              constant OmniAudioIm2colParams &p [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
  constexpr uint channels = OMNI_AUDIO_CONV_CHANNELS;
  constexpr uint vectors = OMNI_AUDIO_CONV_PADDED_TAPS / 8u;
  const uint row = index / vectors, k = (index % vectors) * 8u;
  if (row >= p.rows) return;
  uint4 value = uint4(0u);
  if (k < OMNI_AUDIO_CONV_TAPS) {
    const uint pixels = p.output_height * p.output_width;
    const uint g = row / pixels;
    const uint rest = row % pixels;
    const uint ho = rest / p.output_width, wo = rest % p.output_width;
    const uint tap = k / channels, ci = k % channels;
    const int hi = int(ho) * 2 - 1 + int(tap / 3u);
    const int wi = int(wo) * 2 - 1 + int(tap % 3u);
    if (hi >= 0 && hi < int(p.input_height) && wi >= 0 && wi < int(p.input_width))
      value = *reinterpret_cast<device const uint4 *>(
          input + ((g * p.input_height + uint(hi)) * p.input_width + uint(wi)) * p.input_stride + ci);
  }
  *reinterpret_cast<device uint4 *>(output + ulong(row) * OMNI_AUDIO_CONV_PADDED_TAPS + k) = value;
}

// One thread per (chunk, frame, c * 16 + f) of the conv3 output.
kernel void omni_audio_flatten(device const bfloat *conv [[buffer(0)]], device bfloat *output [[buffer(1)]],
                               constant OmniAudioFlattenParams &p [[buffer(2)]],
                               uint index [[thread_position_in_grid]]) {
  constexpr uint width = OMNI_AUDIO_CONV_OUT_INPUT;
  constexpr uint height = OMNI_AUDIO_CONV3_HEIGHT;
  const uint column = index % width;
  const uint rest = index / width;
  const uint t = rest % p.width, g = rest / p.width;
  if (g >= p.chunks) return;
  const uint chunk = p.first_chunk + g;
  const uint valid = chunk == p.last_chunk ? p.last_chunk_tokens : OMNI_AUDIO_CHUNK_TOKENS;
  if (t >= valid) return;
  const uint c = column / height, f = column % height;
  output[ulong(chunk * OMNI_AUDIO_CHUNK_TOKENS + t) * width + column] =
      conv[((g * height + f) * p.width + t) * OMNI_AUDIO_CONV_PADDED_CHANNELS + c];
}

kernel void omni_audio_positions(device float *rows [[buffer(0)]], device const float *table [[buffer(1)]],
                                 constant OmniAudioPositionParams &p [[buffer(2)]],
                                 uint index [[thread_position_in_grid]]) {
  if (index >= p.rows * p.width) return;
  const uint row = index / p.width, d = index % p.width;
  rows[index] += table[(row % p.period) * p.width + d];
}

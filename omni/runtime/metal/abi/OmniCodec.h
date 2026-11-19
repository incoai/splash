#pragma once

// Parameter layouts shared by the speech tokenizer decoder kernels
// (runtime/metal/kernels/omni/codec.metal) and their host dispatch code
// (ops/OmniCodec.hpp). All activations are fp32 rows of channels ([time]
// [channel], channel contiguous); every struct is plain data with
// fixed-width scalars and a static_assert pins its size on both sides.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The implicit-GEMM convolution: output[t][n] = epilogue(sum over taps k and
// input channels c of input[t - pad + k * dilation][c] * weight[k][c][n]).
// Rows before 0 read the stream's history (history_rows rows of in_channels
// before row 0, the last input rows of the previous chunk) and zero before
// that; rows at or past input_rows read zero. A plain
// linear is taps 1, pad 0; a stride-s transposed convolution of kernel k s
// is k taps, pad k - 1 with out_channels = s * channels (the output row t
// holds the s phases of input row t, which is the [s * t + r][channel]
// layout).
// Tiles: 64 output rows x 64 output channels per threadgroup of 128
// threads (4 simdgroups of 32 x 32), 32 input channels per step;
// in_channels must be a multiple of 32 and out_channels of 4.
// Flags: bit 0 adds bias[n]; bit 1 adds residual[t][n] last; bit 2 applies
// the exact GELU before the residual; bit 3 multiplies (sum + bias) by
// scale[n] before the residual.
#define OMNI_CODEC_CONV_FLAG_BIAS 1u
#define OMNI_CODEC_CONV_FLAG_RESIDUAL 2u
#define OMNI_CODEC_CONV_FLAG_GELU 4u
#define OMNI_CODEC_CONV_FLAG_SCALE 8u
#define OMNI_CODEC_CONV_TILE_ROWS 64u
#define OMNI_CODEC_CONV_TILE_COLUMNS 64u
#define OMNI_CODEC_CONV_TILE_DEPTH 32u
#define OMNI_CODEC_CONV_THREADS 128u
struct OmniCodecConvParams {
  uint32_t rows;         // output rows
  uint32_t input_rows;   // rows the input holds
  uint32_t in_channels;  // input channels (multiple of 32)
  uint32_t out_channels; // output channels (multiple of 4)
  uint32_t taps;
  uint32_t dilation;
  int32_t pad;           // left padding in input rows
  uint32_t input_stride; // elements between input rows
  uint32_t output_stride;
  uint32_t flags;
  uint32_t history_rows; // 0: no history
  uint32_t padding;
};
static_assert(sizeof(OmniCodecConvParams) == 48, "codec conv parameters are 48 bytes on both sides");

// Depthwise causal convolution (the ConvNeXt dwconv): output[t][c] =
// bias[c] + sum_k weight[k][c] * input[t - (taps - 1) + k][c], one thread
// per (row, channel); rows before 0 read the history as the GEMM does.
struct OmniCodecDepthwiseParams {
  uint32_t rows;
  uint32_t channels;
  uint32_t taps;
  uint32_t history_rows;
};
static_assert(sizeof(OmniCodecDepthwiseParams) == 16, "codec depthwise parameters are 16 bytes on both sides");

// Elementwise over rows of `channels`: 0 SnakeBeta with per-channel
// snake_alpha = exp(alpha), snake_beta = 1 / (exp(beta) + 1e-9);
// 1 SiLU-gated product: output[t][c] = silu(input[t][c]) * input[t][channels
// + c] over rows of 2 * channels (the transformer MLP's gate and up halves).
struct OmniCodecElementwiseParams {
  uint32_t rows;
  uint32_t channels;
  uint32_t kind;
  uint32_t padding;
};
static_assert(sizeof(OmniCodecElementwiseParams) == 16, "codec elementwise parameters are 16 bytes on both sides");

// RVQ lookup: for frame f, output[f][0 .. half_width) = codebook[0][codes[f][0]]
// and output[f][half_width .. 2 half_width) = sum over q in 1 .. quantizers of
// codebook[q][codes[f][q]]; codebooks are fp32 [quantizers][codebook_size]
// [half_width]. A code at or past codebook_size reads the last entry.
struct OmniCodecLookupParams {
  uint32_t frames;
  uint32_t quantizers;
  uint32_t codebook_size;
  uint32_t half_width;
};
static_assert(sizeof(OmniCodecLookupParams) == 16, "codec lookup parameters are 16 bytes on both sides");

// The transformer's rotate-half RoPE over the query and key heads of fp32
// rows [rows][3 * heads * 64] (queries, keys, values), position = position
// + row.
struct OmniCodecRopeParams {
  uint32_t rows;
  uint32_t heads;
  float theta;
  uint32_t position;
};
static_assert(sizeof(OmniCodecRopeParams) == 16, "codec RoPE parameters are 16 bytes on both sides");

// The keys and values of a layer live in rings of `capacity` rows of
// heads * 64 fp32: position p at row p % capacity. omni_codec_kv_append
// copies the rotated keys and values of rows [position, position + rows)
// into them; a stream's ring holds the window - 1 positions before its
// chunk, so capacity >= window - 1 + rows.
struct OmniCodecAppendParams {
  uint32_t rows;
  uint32_t heads;
  uint32_t position;
  uint32_t capacity;
};
static_assert(sizeof(OmniCodecAppendParams) == 16, "codec append parameters are 16 bytes on both sides");

// Causal sliding-window attention: the query of row t at position p =
// position + t attends key positions q <= p with q > p - window (1 ..
// OMNI_CODEC_MAXIMUM_WINDOW). One simdgroup per (row, head), heads of 64;
// output fp32 [rows][heads * 64].
#define OMNI_CODEC_HEAD_DIM 64u
#define OMNI_CODEC_ATTENTION_GROUPS 4u
#define OMNI_CODEC_MAXIMUM_WINDOW 256u
struct OmniCodecAttentionParams {
  uint32_t rows;
  uint32_t heads;
  uint32_t window;
  float scale;
  uint32_t position;
  uint32_t capacity;
  uint32_t padding[2];
};
static_assert(sizeof(OmniCodecAttentionParams) == 32, "codec attention parameters are 32 bytes on both sides");

// A stream's history of a convolution's input: the new history (the last
// history_rows rows of the old history followed by the input_rows rows of
// the input) is written from the old one and the input, one thread per
// element.
struct OmniCodecHistoryParams {
  uint32_t input_rows;
  uint32_t history_rows;
  uint32_t width;
  uint32_t padding;
};
static_assert(sizeof(OmniCodecHistoryParams) == 16, "codec history parameters are 16 bytes on both sides");

// The output convolution to one channel: samples[t] = clamp(bias + sum_k
// sum_c weight[k][c] * input[t - (taps - 1) + k][c], -1, 1), one thread per
// sample; rows before 0 read the history.
struct OmniCodecOutputParams {
  uint32_t rows;
  uint32_t channels;
  uint32_t taps;
  uint32_t history_rows;
};
static_assert(sizeof(OmniCodecOutputParams) == 16, "codec output parameters are 16 bytes on both sides");

// Rows to and from the Neural Engine's layout (ops::omni::CodecAneStack):
// a surface of `channels` rows of fp16, `stride` elements apart, whose
// column t holds row t of the fp32 rows [rows][channels].
// omni_codec_to_surface writes columns [0, columns): column t < rows holds
// rows[t], the others zero; one thread per (channel, column).
// omni_codec_from_surface reads columns [0, rows) into the fp32 rows; one
// thread per element.
struct OmniCodecSurfaceParams {
  uint32_t rows;
  uint32_t columns;
  uint32_t channels;
  uint32_t stride;
};
static_assert(sizeof(OmniCodecSurfaceParams) == 16, "codec surface parameters are 16 bytes on both sides");

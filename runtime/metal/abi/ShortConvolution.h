#pragma once

// Parameters of the short convolution kernels (kernels/shared/short_convolution.metal,
// ops/ShortConvolution.cpp): a causal depthwise convolution of
// SHORT_CONVOLUTION_TAPS taps over each channel of a projection's output
// rows, in fp32. The taps are static, F32 [channels][taps], or static plus
// the row's own dynamic taps, bf16 [rows][heads][taps], which every channel
// of a head shares, each in the order a GGUF stores it, which the loader
// keeps: a channel's static taps in a conv1d weight's order, its last column
// weighing the row itself and column taps - 1 - k the input k rows back
// (llama.cpp's ggml_ssm_conv); a head's dynamic taps by that lag, k the input
// k rows back (the projection's output, index head * taps + k). The sum
// plus the row's input, over `divisor`, is rounded to bf16 (the source
// returns its input's type) before an output adds it to a residual
// (SHORT_CONVOLUTION_OUTPUT_*).
//
// A sequence's state is the SHORT_CONVOLUTION_TAPS - 1 input rows before its
// first, oldest first, in bf16 as the projection writes them: rows of a
// lane's state matrix of `state_stride` columns, of which a convolution owns
// `channels` from `state_column` (ops::ShortConvState), so that every
// convolution of a target binds the same per-lane buffers and one commit
// writes all of them.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "metal/abi/ExecutionGeometry.h"

// The taps of every short convolution of a model in the tree (its
// configurations' convolution kernel sizes).
#define SHORT_CONVOLUTION_TAPS 4u
#define SHORT_CONVOLUTION_STATE_ROWS (SHORT_CONVOLUTION_TAPS - 1u)

// The most states captured inside a command's sequences
// (ops::ShortConvCapture) that one dispatch writes; a command of more runs
// dispatches of the rest alone (captures_only).
#define SHORT_CONVOLUTION_CAPTURES 4u

// The outputs (ops::ShortConvOutput): bf16(y) as bf16 rows; bf16 of an fp32
// residual plus bf16(y), a layer's output (short_convolution); fp32 rows of a
// bf16 residual, or none, plus bf16(y) (short_convolution_sum).
#define SHORT_CONVOLUTION_OUTPUT_PLAIN 0u
#define SHORT_CONVOLUTION_OUTPUT_SUM 1u
#define SHORT_CONVOLUTION_OUTPUT_ROUNDED_SUM 2u

// The convolution of a command's `sequences` sequences of its `rows` rows
// (sequence s's rows its entry of the command's layout, SplashLaneRows,
// metal/abi/VerifyTree.h), as ngram_embedding_conv runs its own: with
// write_state the state after each sequence's rows goes to its state_out, and
// capture q (q < captures) the state after the first capture_rows[q] rows of
// sequence capture_sequences[q] to its own buffer; captures_only writes those
// alone. With `tree` the sequences are verify blocks of at most 32 rows whose
// rows' taps follow their paths, and each row's input goes to the tape (rows
// of state_stride columns, the convolution's from state_column), which the
// commit reads. heads: 0 for static taps, else the dynamic taps' heads, whose
// rows are dynamic_stride values apart (a projection's rows, whole tiles).
// residual: whether a sum output adds a residual (none adds zero). The input
// rows are input_stride values apart, the convolution's channels from column
// input_column of each (a view of a wider projection's rows). A pair
// (short_convolution_pair: a verify step's two convolutions of one shape in
// one dispatch, plain) runs a second convolution of its own buffers, its
// input's channels from column pair_input_column, its state's and the tape's
// from pair_state_column. The commit takes `channels` as the state matrix's
// columns, every one of which it writes; with captures (one) it is a decode
// capture's fold instead, of the first capture_rows[0] rows of each lane's
// accepted path alone.
struct ShortConvolutionParams {
  uint32_t rows;
  uint32_t channels;
  uint32_t heads;
  uint32_t output;
  uint32_t residual;
  uint32_t write_state;
  uint32_t tree;
  uint32_t sequences;
  uint32_t state_stride;
  uint32_t state_column;
  float divisor;
  uint32_t captures;
  uint32_t captures_only;
  uint32_t dynamic_stride;
  uint32_t capture_sequences[SHORT_CONVOLUTION_CAPTURES];
  uint32_t capture_rows[SHORT_CONVOLUTION_CAPTURES];
  uint32_t input_stride;
  uint32_t input_column;
  uint32_t pair_input_column;
  uint32_t pair_state_column;
};

static_assert(sizeof(ShortConvolutionParams) == 104,
              "short convolution parameters are 104 bytes on both sides");

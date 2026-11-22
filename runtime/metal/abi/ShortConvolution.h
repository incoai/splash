#pragma once

// Parameters of the short convolution kernels (kernels/shared/short_convolution.metal,
// ops/ShortConvolution.cpp), Inkling 2's causal depthwise convolution of
// SHORT_CONVOLUTION_TAPS taps over `channels` channels, added to its input
// and scaled: out[c] = (x_p[c] + sum_j w[j][c] x_{p-3+j}[c] + dyn) * scale,
// tap 3 the current row's input. The dynamic taps (head_channels != 0) add
// sum_lag lambda[head(c) * 4 + lag] x_{p-lag}[c], the row's own per-head
// weights of the current input (lag 0) and the three before it, head(c) =
// c / head_channels. Each sequence's state, its history, is its inputs of
// the SHORT_CONVOLUTION_HISTORY rows before its first, oldest first: zeros
// before a sequence's start. A kernel reads a past input as the history
// keeps it (the state's element type), so a row's output is the same
// whether its past inputs come from its own command or from a history,
// whatever the command boundaries.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#define SHORT_CONVOLUTION_TAPS 4u
#define SHORT_CONVOLUTION_HISTORY (SHORT_CONVOLUTION_TAPS - 1u)
// Threads of every short convolution kernel's threadgroup: one channel each.
#define SHORT_CONVOLUTION_THREADS 256u

// The convolution (short_convolution_<input>_<state>) of a command's
// `sequences` sequences of its `rows` rows, sequence s's rows its entry of
// the command's layout (SplashLaneRows, metal/abi/VerifyTree.h): a prompt
// chunk's sequences' as its host writes them, a verify step's lanes' as its
// proposer does. Rows are input_stride, output_stride and lambda_stride
// elements apart, a history's rows history_stride. With write_history the
// kernel writes each sequence's history after its rows; with `tree` the
// sequences are verify blocks (SplashTreeRow a row) whose rows' past inputs
// follow their paths, and the kernel writes each row's input, as a history
// keeps it, to the tape row of its packed row (tape_stride elements apart),
// from which the commit (ShortConvolutionCommitParams) writes the history
// after a block's accepted path. With `accumulate` the output adds to the
// destination's value.
struct ShortConvolutionParams {
  uint32_t rows;
  uint32_t channels;
  uint32_t sequences;
  uint32_t input_stride;
  uint32_t output_stride;
  uint32_t lambda_stride;
  uint32_t history_stride;
  uint32_t tape_stride;
  uint32_t head_channels;
  uint32_t write_history;
  uint32_t tree;
  uint32_t accumulate;
  float scale;
};

static_assert(sizeof(ShortConvolutionParams) == 52, "short convolution parameters are 52 bytes on both sides");

// The history after a sequence's first `rows` rows of a prefill
// (short_convolution_history_<input>_<state>): a state captured inside its
// command, from its history before them and its inputs.
struct ShortConvolutionHistoryParams {
  uint32_t rows;
  uint32_t channels;
  uint32_t input_stride;
  uint32_t history_stride;
};

static_assert(sizeof(ShortConvolutionHistoryParams) == 16,
              "short convolution history parameters are 16 bytes on both sides");

// The commit after a verify (short_convolution_commit_<state>): each of
// `lanes` lanes' history after its accepted path (a row mask the GPU wrote)
// of its block's rows in the tape, for each of `units` units of `width`
// channels (a target's convolutions of one layer side by side): unit u's
// history at u * unit_stride elements of a lane's state, its rows
// history_stride apart, its tape at u * tape_unit_stride elements of the
// tape, the step's packed rows tape_stride apart.
struct ShortConvolutionCommitParams {
  uint32_t rows;
  uint32_t lanes;
  uint32_t width;
  uint32_t units;
  uint32_t history_stride;
  uint32_t unit_stride;
  uint32_t tape_stride;
  uint32_t tape_unit_stride;
};

static_assert(sizeof(ShortConvolutionCommitParams) == 32,
              "short convolution commit parameters are 32 bytes on both sides");

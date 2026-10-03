#pragma once

// Parameter layouts of the GPU side of the ANE FFN split (ops/AneFfn.cpp).
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct AneFfnRotateParams {
  uint32_t hidden;
};

// Rotated rows [rows][hidden] as the channel-major int8 inputs
// [channel, channel + 32 * grid.y) of the ANE, stride bytes per channel.
struct AneFfnPackParams {
  uint32_t hidden;
  uint32_t channel;
  uint32_t stride;
};

// Rotated int8 of a projection: its output rows [row, row + N) and inputs
// [input, input + width), where groups is its input count / 64 (affine Q4) or
// / 32 (a GGUF image tensor of GGUF_FMT_* format), into rows of stride bytes
// and their scales scale_stride halves apart.
struct AneFfnWeightParams {
  uint32_t groups;
  uint32_t row;
  uint32_t input;
  uint32_t width;
  uint32_t stride;
  uint32_t scale_stride;
  uint32_t format;
};

// output[t][c] += partial[c][t] for rows t < rows; stride halves per channel.
struct AneFfnJoinParams {
  uint32_t hidden;
  uint32_t stride;
  uint32_t rows;
};

static_assert(sizeof(AneFfnRotateParams) == 4, "ANE FFN rotate parameters are 4 bytes");
static_assert(sizeof(AneFfnPackParams) == 12, "ANE FFN pack parameters are 12 bytes");
static_assert(sizeof(AneFfnWeightParams) == 28, "ANE FFN weight parameters are 28 bytes");
static_assert(sizeof(AneFfnJoinParams) == 12, "ANE FFN join parameters are 12 bytes");

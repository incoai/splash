#pragma once

// Parameter layouts of the text encoder's vision tower and its vision rows
// (kernels/video/vision_tower.metal and its host operator). The tower's
// GEMM, LayerNorm, grid, QKV and attention parameters are the shared vision
// tower's (metal/abi/Vision.h): the H3 conditioner's tower, Qwen3-VL-32B's,
// has Qwen3.5's geometry, 27 blocks of 1152 with 16 heads of 72.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The rows of a prompt's vision blocks: features [rows][width] fp32 replace
// (mode 0) or are added to (mode 1, deepstack) the bf16 hidden rows the row
// index [rows] names, rounded once to bf16.
struct TextEncoderVisionRowsParams {
  uint32_t rows;
  uint32_t width;
  uint32_t mode;
};

static_assert(sizeof(TextEncoderVisionRowsParams) == 12,
              "text encoder vision row parameters are 12 bytes on both sides");

// Interleaved M-RoPE of Qwen3-VL's text model: of the 64 rotary frequencies,
// f % 3 == 1 below 3 x 20 rotates by the row's height position, f % 3 == 2
// below 3 x 20 by its width position, every other by its temporal position
// (mrope_section [24, 20, 20], interleaved).
#define SPLASH_TEXT_ENCODER_MROPE_HEIGHT 20u
#define SPLASH_TEXT_ENCODER_MROPE_WIDTH 20u

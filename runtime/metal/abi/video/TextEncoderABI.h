#pragma once

// Parameter layouts of the text encoder's attention kernels
// (kernels/video/text_encoder.metal, video/TextEncoder.mm).
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// The kernels run one simdgroup per head row, four dimensions per lane.
#define SPLASH_TEXT_ENCODER_HEAD_DIMENSION 128u

// The heads of a prompt's rows: Q/K RMS norms and rotation of the fused
// projection's rows, and causal attention over them.
struct TextEncoderHeadParams {
  uint32_t rows;
  uint32_t query_heads;
  uint32_t kv_heads;
  // The norms' epsilon (rotation), or the scores' scale (attention).
  float value;
};

static_assert(sizeof(TextEncoderHeadParams) == 16,
              "text encoder head parameters are 16 bytes on both sides");

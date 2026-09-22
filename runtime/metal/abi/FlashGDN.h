#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// Projected rows are lane-major and tightly packed. State strides are bytes,
// permitting independently padded convolution and recurrent allocations.
struct FlashGDNParams {
  uint32_t rows;
  uint32_t lanes;
  uint32_t key_heads;
  uint32_t value_heads;
  uint32_t key_dimension;
  uint32_t value_dimension;
  uint32_t convolution_taps;
  float norm_epsilon;
  uint64_t convolution_lane_stride_bytes;
  uint64_t recurrent_lane_stride_bytes;
};

enum FlashGDNDiagnostic : uint32_t {
  FlashGDNInvalidParameters = 2u,
  FlashGDNNonFinite = 4u,
};

static_assert(sizeof(FlashGDNParams) == 48,
              "Flash GDN parameters must match the shader ABI");

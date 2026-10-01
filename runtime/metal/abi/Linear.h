#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct Q4Params {
  uint32_t output_size;
  uint32_t input_size;
  uint32_t persistent_groups;
};

static_assert(sizeof(Q4Params) == 12,
              "Q4 decode projection parameters are 12 bytes on both sides");

// Separate from ops::LinearMatrix so host-only fields cannot change the ABI.
// plane_input_size: the inputs each weight row holds, of which the projection
// reads the first input_size (0 = input_size).
struct Q4PrefillParams {
  uint32_t output_size;
  uint32_t input_size;
  uint32_t plane_input_size;
};

static_assert(sizeof(Q4PrefillParams) == 12,
              "Q4 prefill projection parameters are 12 bytes on both sides");

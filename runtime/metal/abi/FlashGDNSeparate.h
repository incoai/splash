#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct FlashGDNSeparateParams {
  uint32_t lanes;
  float norm_epsilon;
};

static_assert(sizeof(FlashGDNSeparateParams) == 8,
              "Flash GDN separate-state parameters must match Metal");

#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

enum : uint32_t { kFlashMoEDirectAPaddingRows = 63 };

struct FlashMoEDirectAPrepareParams {
  uint32_t routes;
  uint32_t width;
  uint32_t padding_rows;
  uint32_t reserved;
};

static_assert(sizeof(FlashMoEDirectAPrepareParams) == 16,
              "Flash direct A prepare parameters are 16 bytes");

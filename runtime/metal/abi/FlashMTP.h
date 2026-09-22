#pragma once

#if defined(__METAL_VERSION__)
struct FlashMTPFuseParams {
  uint rows;
  uint width;
  uint streams;
};
#else
#include <cstdint>
struct FlashMTPFuseParams {
  uint32_t rows;
  uint32_t width;
  uint32_t streams;
};
static_assert(sizeof(FlashMTPFuseParams) == 12);
#endif

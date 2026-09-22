#pragma once

#if defined(__METAL_VERSION__)
struct FlashForwardActivationParams {
  uint rows;
  uint width;
  uint streams;
};
struct FlashForwardCopyParams { ulong words; };
#else
#include <cstdint>
struct FlashForwardActivationParams {
  uint32_t rows;
  uint32_t width;
  uint32_t streams;
};
static_assert(sizeof(FlashForwardActivationParams) == 12);
struct FlashForwardCopyParams { uint64_t words; };
static_assert(sizeof(FlashForwardCopyParams) == 8);
#endif

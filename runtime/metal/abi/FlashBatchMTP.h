#pragma once

#if defined(__METAL_VERSION__)
struct FlashBatchMTPGatherParams {
  uint lanes;
  uint width;
  uint rows;
};
#else
#include <cstdint>
struct FlashBatchMTPGatherParams {
  uint32_t lanes;
  uint32_t width;
  uint32_t rows;
};
static_assert(sizeof(FlashBatchMTPGatherParams) == 12);
#endif

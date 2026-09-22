#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

struct FlashPLESSDParams {
  uint32_t rows;
  uint32_t heads;
  uint32_t head_width;
  uint32_t row_stride_bytes;
  uint64_t table_rows;
};
static_assert(sizeof(FlashPLESSDParams) == 24,
              "Flash PLE SSD parameters are 24 bytes");

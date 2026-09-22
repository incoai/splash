#pragma once

#include "FlashGDN.h"

struct FlashGDNCaptureParams {
  FlashGDNParams gdn;
  uint64_t capture_row_stride_bytes;
  uint64_t capture_lane_stride_bytes;
  uint32_t capture_rows;
  uint32_t reserved;
};

static_assert(sizeof(FlashGDNCaptureParams) == 72,
              "Flash GDN capture parameters must match the shader ABI");

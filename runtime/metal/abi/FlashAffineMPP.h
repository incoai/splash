#pragma once

#include "FlashAffine.h"

struct FlashAffineMPPParams {
  FlashAffineParams affine;
  uint32_t mode;
  uint32_t tile_rows;
  uint32_t tile_outputs;
  uint32_t reserved;
};

static_assert(sizeof(FlashAffineMPPParams) == 80,
              "Flash affine MPP parameters must match host/shader layouts");

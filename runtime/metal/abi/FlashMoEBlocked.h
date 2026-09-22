#pragma once

#include "FlashMoEFused.h"

struct FlashMoEBlockedGateParams {
  FlashMoEFusedParams affine;
  uint32_t route_capacity;
  uint32_t job_capacity;
  uint32_t tile_rows;
  uint32_t reserved;
};

struct FlashMoEBlockedDownParams {
  FlashMoEDownFusedParams affine;
  uint32_t route_capacity;
  uint32_t job_capacity;
  uint32_t tile_rows;
  uint32_t reserved;
};

static_assert(sizeof(FlashMoEBlockedGateParams) == 112,
              "Flash blocked gate/up parameters are 112 bytes");
static_assert(sizeof(FlashMoEBlockedDownParams) == 80,
              "Flash blocked expert-down parameters are 80 bytes");

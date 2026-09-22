#pragma once

#include "FlashAffine.h"

struct FlashDenseCacheParams {
  uint32_t rows;
  uint32_t input_size;
  uint32_t output_size;
  uint32_t output_begin;
  uint32_t output_count;
  uint32_t tile_rows;
  uint32_t tile_outputs;
  uint32_t reserved;
};

static_assert(sizeof(FlashDenseCacheParams) == 32,
              "Flash dense cache parameters must match host/shader layouts");

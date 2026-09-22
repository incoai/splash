#pragma once

#include "FlashAffine.h"

struct FlashFloatDenseSmallRowsParams {
  uint32_t rows;
  uint32_t padded_rows;
  uint32_t input_size;
  uint32_t output_size;
  uint32_t output_begin;
  uint32_t output_count;
  uint32_t tile_rows;
  uint32_t tile_outputs;
};

static_assert(sizeof(FlashFloatDenseSmallRowsParams) == 32,
              "Flash float small-row parameters must match host/shader layouts");

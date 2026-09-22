#pragma once
#include "FlashQSAFast.h"

// Existing statistics/numerator scratch and 64-byte parameters are retained.
// Under this guard all complete blocks are selected in chronological order;
// selected slots and the incomplete causal tail equal direct token IDs.
static inline bool flash_qsa_row_tiles_geometry(uint32_t begin, uint32_t rows,
                                               uint32_t partitions) {
  return begin >= 512 && begin <= 2048 && rows >= 32 && rows <= 128 &&
         rows <= 2048 - begin && partitions == 4;
}

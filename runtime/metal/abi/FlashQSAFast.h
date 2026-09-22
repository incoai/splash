#pragma once

#include "FlashQSA.h"

struct FlashQSAFastParams {
  FlashQSAParams common;
  uint32_t norm_dtype_mask;       // bit0/1/2: raw Q/K/index-Q norm is F32
  uint32_t norm_convention_bits; // two bits per Q/K/index-Q norm, 0=1+w,1=w
  uint32_t partitions;
  uint32_t maximum_partitions;   // physical scratch stride
};

static_assert(sizeof(FlashQSAFastParams) == 64,
              "Flash QSA fast parameters are 64 bytes on host and Metal");

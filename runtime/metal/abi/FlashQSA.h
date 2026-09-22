#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// Fixed inspected Flash-Next text geometry; all extents are logical and
// unpadded. Byte/storage strides are not inherited from the 27B package.
struct FlashQSAParams {
  uint32_t rows;
  uint32_t begin;
  uint32_t capacity;
  uint32_t block_capacity;
  uint32_t first_block;
  uint32_t new_blocks;
  uint32_t norm_convention;
  uint32_t mode;
  float epsilon;
  float theta;
  uint32_t positions_supplied;
  uint32_t reserved;
};

static_assert(sizeof(FlashQSAParams) == 48,
              "Flash QSA parameters are 48 bytes on host and Metal");

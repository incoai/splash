#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// `rows` rows of `width` bf16 values, from row source_row of the source's
// rows of source_stride values, at source_column, to the destination's
// alike (ops::RowCopy).
struct RowCopyParams {
  uint32_t rows;
  uint32_t width;
  uint32_t source_row;
  uint32_t source_stride;
  uint32_t source_column;
  uint32_t destination_row;
  uint32_t destination_stride;
  uint32_t destination_column;
};

static_assert(sizeof(RowCopyParams) == 32,
              "Row copy parameters are 32 bytes on both sides");

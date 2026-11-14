#pragma once

// Parameter layouts shared by host dispatch code and Metal kernels.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// `rows` rows of `width` bf16 values, from row source_row of the source's
// rows of source_stride values, at source_column, to the destination's alike
// (ops::RowCopy). With `destination_lane_rows` set, the destination's rows
// are grouped `destination_lane_rows` to a lane `destination_lane_stride`
// apart: flat row r lands at destination_row + r / lane_rows * lane_stride +
// r % lane_rows, which scatters a pass's rows into the lanes' whole-block
// rows (a 16-row verify's second sub-pass's captures).
struct RowCopyParams {
  uint32_t rows;
  uint32_t width;
  uint32_t source_row;
  uint32_t source_stride;
  uint32_t source_column;
  uint32_t destination_row;
  uint32_t destination_stride;
  uint32_t destination_column;
  uint32_t destination_lane_rows;
  uint32_t destination_lane_stride;
};

static_assert(sizeof(RowCopyParams) == 40,
              "Row copy parameters are 40 bytes on both sides");

#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif
// Qwen4 Q8/G64 vocabulary head. Original unsigned UINT8 source codes and
// BF16 scale/bias remain exact; group-factored dot and affine epilog are F32.
struct FlashInt8HeadParams {
  uint32_t rows, padded_rows, input_size, output_size;
  uint32_t bits, group_size, tile_rows, tile_outputs;
  uint64_t weight_row_stride_bytes, parameter_row_stride_bytes;
};
static_assert(sizeof(FlashInt8HeadParams) == 48,
              "Private INT8 code ABI size changed");

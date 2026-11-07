#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <cstdint>
#endif
struct VerifyBodyPackParams {
  uint32_t lanes, output_size, input_size, residual, columns;
};
static_assert(sizeof(VerifyBodyPackParams) == 20);

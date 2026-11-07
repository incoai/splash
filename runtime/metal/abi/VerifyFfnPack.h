#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <cstdint>
#endif
struct VerifyPackParams {
  uint32_t lanes, hidden, intermediate, up_splits, down_splits;
};
struct VerifyPackControl {
  uint32_t total, compact, valid;
  uint32_t map[32], inverse[64], grids[7][3];
};
static_assert(sizeof(VerifyPackParams) == 20);
static_assert(sizeof(VerifyPackControl) == 480);

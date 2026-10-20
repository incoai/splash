#pragma once

// Deterministic synthetic values (h3 bench --synthetic, dev/tests/synthetic_check.sh): every tensor is a function of
// (tensor seed, element index) through integer hashing and one fused multiply-add, identical on host and GPU and
// independent of compiler flags.
//   z = seed + (i + 1) * 0x9E3779B97F4A7C15; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9;
//   z = (z ^ (z >> 27)) * 0x94D049BB133111EB; h = (z ^ (z >> 31)) >> 32                          (splitmix64)
//   int8: (h mod 255) - 127;  float: fma(scale, r, offset), r = (int32(h) >> 8) * 2^-23 in [-1, 1)
//   (fp16 / bf16: the float rounded to nearest even)
#include "metal/abi/video/Layout.h"

enum : int32_t { H3_SYN_I8 = 0, H3_SYN_F32 = 1, H3_SYN_F16 = 2, H3_SYN_BF16 = 3 };

struct SyntheticFill {
  uint32_t seed_lo, seed_hi;
  uint32_t count;
  int32_t type;
  float scale, offset;
};

#ifdef __METAL_VERSION__
inline uint32_t h3_syn_hash(ulong seed, ulong i) {
  ulong z = seed + (i + 1) * 0x9E3779B97F4A7C15ul;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
  return uint32_t((z ^ (z >> 31)) >> 32);
}
#else
inline uint32_t h3_syn_hash(uint64_t seed, uint64_t i) {
  uint64_t z = seed + (i + 1) * 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return static_cast<uint32_t>((z ^ (z >> 31)) >> 32);
}
#endif

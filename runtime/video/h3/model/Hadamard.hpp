#pragma once

// Sign vector of the Hadamard(128) rotation applied to every int8 linear (same definition in tools/convert_dit.py and
// tools/convert_vae.py had_signs): s[k] = -1 if splitmix64((0x48335138 << 32) + k) >> 63 else +1.
#include <cstdint>
#include <vector>

namespace h3::model {

inline constexpr uint64_t kHadamardSeed = 0x48335138ull;  // "H3Q8"

inline uint64_t splitmix64(uint64_t x) {
  uint64_t z = x + 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

inline std::vector<int8_t> hadamardSigns(int count) {
  std::vector<int8_t> s(count);
  for (int k = 0; k < count; ++k) s[k] = (splitmix64((kHadamardSeed << 32) + static_cast<uint64_t>(k)) >> 63) ? -1 : 1;
  return s;
}

} // namespace h3::model

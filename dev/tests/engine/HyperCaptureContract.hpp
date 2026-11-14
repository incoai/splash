#pragma once
// Test-only candidate contract. No production dispatch or feature flag is changed.
#include "metal/abi/HyperConnection.h"
#include <array>
#include <cstdint>
#include <stdexcept>
namespace splash::test {
struct CaptureRegion { uintptr_t identity; uint64_t offset, bytes, capacity; };
inline uint64_t captureProduct(uint64_t a, uint64_t b) {
  if (a && b > UINT64_MAX / a) throw std::invalid_argument("capture extent overflow");
  return a * b;
}
inline void validateHyperCapture(const HyperStreamMeanParams &p, const std::array<CaptureRegion, 4> &regions) {
  if (!p.rows || !p.hidden || !p.streams || p.streams > HYPER_CONNECTION_MAXIMUM_STREAMS ||
      uint64_t{p.column} + p.hidden > p.stride) throw std::invalid_argument("invalid capture geometry");
  const std::array<uint64_t, 4> need{
    captureProduct(captureProduct(p.rows, p.hidden), uint64_t{p.streams} * 4),
    captureProduct(captureProduct(p.rows, p.hidden), 2),
    captureProduct(captureProduct(p.rows, p.streams), 4),
    captureProduct(captureProduct(uint64_t{p.row} + p.rows, p.stride), 2)};
  for (size_t i = 0; i < regions.size(); ++i) {
    const auto &r = regions[i]; const uint64_t alignment = i == 0 || i == 2 ? 4 : 2;
    if (!r.identity || r.offset % alignment || r.offset > r.capacity || r.bytes > r.capacity - r.offset || r.bytes < need[i])
      throw std::invalid_argument("invalid capture buffer extent/alignment");
    for (size_t j = 0; j < i; ++j) {
      const auto &other = regions[j];
      if ((i == 3 || j == 0) && r.identity == other.identity &&
          r.offset < other.offset + other.bytes && other.offset < r.offset + r.bytes)
        throw std::invalid_argument("capture writable ranges overlap");
    }
  }
}
} // namespace splash::test

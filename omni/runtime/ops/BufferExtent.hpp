#pragma once

#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::ops {

// The bytes from the start of a buffer to one past the last element of
// `rows` rows of `width` elements of `elementBytes` bytes, `stride` elements
// apart: the extent a kernel that touches those rows reaches.
[[nodiscard]] constexpr uint64_t rowBytes(uint64_t rows, uint64_t stride, uint64_t width,
                                          uint64_t elementBytes) noexcept {
  return rows ? ((rows - 1) * stride + width) * elementBytes : 0;
}

// Throws unless `buffer` holds `bytes`, the extent its dispatch's kernel reads
// or writes. A buffer the dispatch does not touch needs no bytes and may be
// absent.
inline void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes, std::string_view what) {
  if (buffer.sizeBytes() < bytes)
    throw std::invalid_argument(std::string(what) + " buffer holds " + std::to_string(buffer.sizeBytes()) +
                                " bytes, needs " + std::to_string(bytes));
}

} // namespace splash::ops

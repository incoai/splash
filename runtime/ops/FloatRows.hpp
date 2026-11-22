#pragma once

// Rows of fp32 or bf16 values in a buffer, as the per-row kernels of
// Inkling 2's operators take them: a view from the first row's first value,
// and the elements from one row's start to the next's, so that a kernel reads
// one segment of wider rows (a projection's packed row) in place.

#include "metal/MetalBackend.hpp"

#include <cstdint>

namespace splash::ops {

enum class FloatElement : uint8_t { Float32, BFloat16 };

[[nodiscard]] constexpr uint64_t elementBytes(FloatElement element) noexcept {
  return element == FloatElement::Float32 ? sizeof(float) : sizeof(uint16_t);
}
// The element's token in a kernel instance's name.
[[nodiscard]] constexpr const char *elementToken(FloatElement element) noexcept {
  return element == FloatElement::Float32 ? "f32" : "bf16";
}

struct StridedRows final {
  metal::MetalBuffer buffer;
  uint32_t stride = 0;
};

} // namespace splash::ops

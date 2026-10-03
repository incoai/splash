#pragma once

// Editing this file re-prepares every GGUF and vision model.

#include <bit>
#include <cstdint>
#include <optional>

namespace splash::model {

// The bf16 bits of value when it is exactly a bf16: preparation never rounds
// a weight.
[[nodiscard]] inline std::optional<uint16_t> exactBfloat16(float value) {
  const auto bits = std::bit_cast<uint32_t>(value);
  if (bits & 0xFFFFu) return std::nullopt;
  return static_cast<uint16_t>(bits >> 16);
}

} // namespace splash::model

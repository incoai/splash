#pragma once

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

// The bf16 nearest value (ties to even), for a finite value: the one
// rounding preparation makes, of a Q8_0 weight's scale times its quant,
// which a GPU product of bf16 operands would round as it read it anyway
// (model/VisionPreparation.cpp).
[[nodiscard]] inline uint16_t roundBfloat16(float value) {
  const auto bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x7FFFu + (bits >> 16 & 1u)) >> 16);
}

// The float a bf16's bits hold.
[[nodiscard]] inline float widenBfloat16(uint16_t bits) { return std::bit_cast<float>(uint32_t{bits} << 16); }

} // namespace splash::model

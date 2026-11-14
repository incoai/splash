#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>

// bf16 arithmetic shared by the kernel tests' CPU references.
namespace splash::test {

inline float bf16ToFloat(uint16_t value) noexcept {
  return std::bit_cast<float>(uint32_t{value} << 16);
}
// Round to nearest even, as the kernels' bfloat conversion does.
inline uint16_t floatToBf16(float value) noexcept {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fff + ((bits >> 16) & 1);
  return uint16_t(bits >> 16);
}

// Spacing of the bf16 binade holding value: 2^(e - 7) for a normal value in
// [2^e, 2^(e+1)), the seven fraction bits. Zero and denormals take the
// smallest normal spacing so a reference that rounds to zero still admits
// the noise of the other summation order.
inline float ulpBf16(float value) noexcept {
  const float magnitude = std::fabs(value);
  if (!(magnitude >= std::ldexp(1.0f, -126)) || !std::isfinite(magnitude))
    return std::ldexp(1.0f, -133);
  int exponent = 0;
  (void)std::frexp(magnitude, &exponent); // magnitude = m * 2^exponent, m in [0.5, 1)
  return std::ldexp(1.0f, exponent - 8);
}

// The int8-activation quantizers' rule (runtime/metal/kernels/common/
// int8_activations.h): a block whose largest magnitude is m codes an input x
// as clamp(rint(x * (127 / m)), -127, 127), ties to even, which stands for
// code * scale, scale = m / 127, both quotients rounded to nearest; a block
// whose m is 2^-100 or less codes as zeros with scale 0.
struct Int8Block {
  float scale = 0, inverse = 0;
};
inline Int8Block int8Block(float largest) noexcept {
  if (!(largest > 0x1p-100f)) return {};
  return {largest / 127.0f, 127.0f / largest};
}
inline int int8Code(float x, const Int8Block &block) noexcept {
  return int(std::clamp(std::rint(x * block.inverse), -127.0f, 127.0f));
}

} // namespace splash::test

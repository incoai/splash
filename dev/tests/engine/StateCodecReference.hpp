#pragma once

#include "metal/abi/StateCodec.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace splash::test {

inline uint16_t codecBf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof bits);
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

inline float codecWiden(uint16_t value) {
  const uint32_t bits = uint32_t{value} << 16;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof result);
  return result;
}

// What ops::StateCodec stores of a layer of `rows` value rows of `columns`
// values: every value's code, then every block's scale
// (metal/abi/StateCodec.h). A block's scale is its largest magnitude over 127
// rounded to nearest even bf16; a value's code the value over the scale
// rounded to nearest even (nearbyint's default), within [-127, 127], or zero
// in a block of zeros. A block that holds NaN or an infinity has a NaN scale
// (common/state_codec.h) and, here, zero codes.
inline std::vector<uint8_t> stateCodesReference(const float *values, uint32_t rows, uint32_t columns) {
  constexpr uint32_t block = SPLASH_STATE_CODEC_BLOCK;
  std::vector<uint8_t> image(state_codec_layer_bytes(rows, columns));
  const uint64_t blocks = uint64_t{rows} * columns / block;
  for (uint64_t b = 0; b < blocks; ++b) {
    const float *blockValues = values + b * block;
    float amax = 0.0F;
    for (uint32_t i = 0; i < block; ++i)
      amax = std::isfinite(blockValues[i]) ? std::max(amax, std::fabs(blockValues[i])) : INFINITY;
    const uint16_t scale = std::isfinite(amax) ? codecBf16(amax * (1.0F / 127.0F)) : uint16_t{0x7FC0};
    const float wide = codecWiden(scale);
    for (uint32_t i = 0; i < block; ++i) {
      const auto code = static_cast<int8_t>(
          wide > 0.0F && std::isfinite(wide) ? std::clamp(std::nearbyint(blockValues[i] / wide), -127.0F, 127.0F)
                                             : 0.0F);
      std::memcpy(image.data() + b * block + i, &code, 1);
    }
    std::memcpy(image.data() + state_codec_scales_offset(rows, columns) + b * 2, &scale, 2);
  }
  return image;
}

} // namespace splash::test

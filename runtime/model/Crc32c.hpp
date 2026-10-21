#pragma once

#include <arm_acle.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace splash::model {

// CRC-32C (Castagnoli), which a persistent slot file keeps for each slot's
// payload and record. A running value continues across calls:
// crc32c(b, crc32c(a)) is the CRC of a followed by b.
[[nodiscard]] inline uint32_t crc32c(std::span<const std::byte> bytes,
                                     uint32_t crc = 0) noexcept {
  uint32_t state = ~crc;
  const std::byte *data = bytes.data();
  size_t size = bytes.size();
  for (; size >= sizeof(uint64_t); data += sizeof(uint64_t), size -= sizeof(uint64_t)) {
    uint64_t word;
    std::memcpy(&word, data, sizeof(word));
    state = __crc32cd(state, word);
  }
  for (; size; ++data, --size)
    state = __crc32cb(state, static_cast<uint8_t>(*data));
  return ~state;
}

} // namespace splash::model

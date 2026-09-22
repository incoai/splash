#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace splash::flash {

// Exact greedy selection for finite BF16 logits. Floating-point ties choose
// the first token, including a tie between positive and negative zero.
inline uint32_t flashGreedyToken(std::span<const uint16_t> values) {
  if (values.empty()) throw std::invalid_argument("empty greedy vocabulary row");

#if defined(__aarch64__) && defined(__ARM_NEON)
  const auto sign = vdupq_n_u16(0x8000);
  const auto magnitude = vdupq_n_u16(0x7fff);
  const auto exponent = vdupq_n_u16(0x7f80);
  const auto zeroRank = sign;
  auto maximum = vdupq_n_u16(0);
  auto nonfinite = vdupq_n_u16(0);
  size_t begin = 0;
  for (; values.size() - begin >= 8; begin += 8) {
    const auto bits = vld1q_u16(values.data() + begin);
    nonfinite = vorrq_u16(nonfinite,
        vceqq_u16(vandq_u16(bits, exponent), exponent));
    const auto negative = vceqq_u16(vandq_u16(bits, sign), sign);
    auto rank = vbslq_u16(negative, vmvnq_u16(bits), veorq_u16(bits, sign));
    rank = vbslq_u16(vceqq_u16(vandq_u16(bits, magnitude), vdupq_n_u16(0)),
        zeroRank, rank);
    maximum = vmaxq_u16(maximum, rank);
  }
  uint16_t bestRank = vmaxvq_u16(maximum);
  bool invalid = vmaxvq_u16(nonfinite) != 0;
  for (size_t token = begin; token < values.size(); ++token) {
    const uint16_t bits = values[token];
    invalid |= (bits & 0x7f80) == 0x7f80;
    const uint16_t rank = !(bits & 0x7fff) ? uint16_t{0x8000}
        : (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                         : static_cast<uint16_t>(bits ^ 0x8000);
    if (rank > bestRank) bestRank = rank;
  }
  if (invalid) throw std::runtime_error("non-finite Flash vocabulary logit");

  // Having reduced the row, search the original bit pattern in SIMD blocks.
  // The first match preserves the reference's lower-token-ID tie policy.
  const bool bestIsZero = bestRank == 0x8000;
  const uint16_t bestBits = bestIsZero ? uint16_t{0}
      : (bestRank & 0x8000) ? static_cast<uint16_t>(bestRank ^ 0x8000)
                           : static_cast<uint16_t>(~bestRank);
  const auto wanted = vdupq_n_u16(bestBits);
  begin = 0;
  for (; values.size() - begin >= 8; begin += 8) {
    auto bits = vld1q_u16(values.data() + begin);
    if (bestIsZero) bits = vandq_u16(bits, magnitude);
    if (vmaxvq_u16(vceqq_u16(bits, wanted))) {
      for (size_t token = begin; token < begin + 8; ++token)
        if ((bestIsZero ? values[token] & 0x7fff : values[token]) == bestBits)
          return static_cast<uint32_t>(token);
    }
  }
  for (size_t token = begin; token < values.size(); ++token)
    if ((bestIsZero ? values[token] & 0x7fff : values[token]) == bestBits)
      return static_cast<uint32_t>(token);
  throw std::logic_error("Flash greedy maximum was not found");
#else
  uint32_t best = 0;
  float maximum = -INFINITY;
  for (uint32_t token = 0; token < values.size(); ++token) {
    const float value = std::bit_cast<float>(uint32_t{values[token]} << 16);
    if (!std::isfinite(value)) throw std::runtime_error("non-finite Flash vocabulary logit");
    if (value > maximum) { maximum = value; best = token; }
  }
  return best;
#endif
}

} // namespace splash::flash

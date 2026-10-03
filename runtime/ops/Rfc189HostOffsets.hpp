#pragma once

// RFC 189 scaffolding: host-side position offsets.
// See docs/rfc/189-host-offsets.md and
// https://github.com/incoai/splash/issues/189.

#include <cstdint>

namespace splash::ops::rfc189 {

inline constexpr bool kEnabledDefault = false;
inline constexpr const char* kEnvVar = "SPLASH_HOST_POSITION_OFFSETS";

// Host-side position delivery: instead of one 1-elem device add per layer,
// the host broadcasts base+L once into a shared offset table the kernels
// read. Pure arithmetic — RoPE math itself is untouched.
//
// offsets[lane] = committedTokens[lane] + verifyLen, clamped to the
// context window. Returns false when the window would overflow.
struct LaneOffsets final {
  std::uint32_t lanes = 0;
  std::uint32_t verifyLen = 0;
};

[[nodiscard]] constexpr std::uint32_t offsetFor(std::uint32_t committed,
                                               std::uint32_t verifyLen) noexcept {
  return committed + verifyLen;
}

[[nodiscard]] constexpr bool offsetsFit(std::uint32_t committed,
                                        std::uint32_t verifyLen,
                                        std::uint32_t contextWindow) noexcept {
  return verifyLen > 0 && committed + verifyLen <= contextWindow;
}

static_assert(offsetFor(512, 8) == 520);
static_assert(offsetFor(0, 16) == 16);
static_assert(offsetsFit(512, 8, 262144) && !offsetsFit(262140, 8, 262144));
static_assert(!offsetsFit(100, 0, 4096));

}  // namespace splash::ops::rfc189

#pragma once

// RFC 187 scaffolding: deeper verify rows (8 -> 16 candidate).
// Not wired into ExecutionGeometry / verifyPlan yet.
// See docs/rfc/187-verify-16.md and
// https://github.com/incoai/splash/issues/187.

#include <cstdint>

namespace splash::ops::rfc187 {

inline constexpr bool kEnabledDefault = false;
inline constexpr const char* kEnvVar = "SPLASH_VERIFY_ROWS";

// Shipped default stays 8; 16 is the measured candidate.
inline constexpr std::uint32_t kVerifyRowsDefault = 8;
inline constexpr std::uint32_t kVerifyRowsCandidate = 16;

// Decode rows for a lane count at a verify width: lanes x verifyRows.
// Width 8 gives today's M8/M16/M24/M32 shapes; width 16 doubles them.
[[nodiscard]] constexpr std::uint32_t rowsForLanes(std::uint32_t lanes,
                                                   std::uint32_t verifyRows) noexcept {
  return lanes * verifyRows;
}

// Admissible decode shapes keep M8 granularity within the 4-lane window.
[[nodiscard]] constexpr bool shapeSupported(std::uint32_t rows) noexcept {
  return rows >= 8 && rows <= 64 && rows % 8 == 0;
}

// Warm-up roster (bonsai warmVerifyBlock analog): the M16 kernels the
// candidate width dispatches on its first verify. Pre-building them cuts
// the first-verify hitch; they already ship in linear_q4.metal, so no new
// .metal is needed for this RFC — the follow-up only pre-builds + retains.
inline constexpr const char* kWarmVerifyPipelines[] = {
    "decode_linear_q4_n128_m16",
    "decode_linear_q4_n256_m16",
    "decode_linear_q4_n128_residual_m16",
    "decode_linear_q4_n256_gate_up_m16",
};
inline constexpr std::uint32_t kWarmVerifyPipelineCount = 4;

static_assert(rowsForLanes(1, 8) == 8 && rowsForLanes(4, 8) == 32);
static_assert(rowsForLanes(1, 16) == 16 && rowsForLanes(4, 16) == 64);
static_assert(shapeSupported(16) && shapeSupported(64));
static_assert(!shapeSupported(12) && !shapeSupported(72));
static_assert(kWarmVerifyPipelineCount == 4);

}  // namespace splash::ops::rfc187

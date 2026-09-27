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

}  // namespace splash::ops::rfc187

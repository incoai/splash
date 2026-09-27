#pragma once

// RFC 189 scaffolding: host-side position offsets.
// See docs/rfc/189-host-offsets.md and
// https://github.com/incoai/splash/issues/189.

namespace splash::ops::rfc189 {

inline constexpr bool kEnabledDefault = false;
inline constexpr const char* kEnvVar = "SPLASH_HOST_POSITION_OFFSETS";

}  // namespace splash::ops::rfc189

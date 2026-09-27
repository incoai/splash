#pragma once

// RFC 193 scaffolding: P-core work-interval hint for verify dispatch.
// See docs/rfc/193-work-interval.md and
// https://github.com/incoai/splash/issues/193.

namespace splash::engine::rfc193 {

inline constexpr bool kEnabledDefault = false;
inline constexpr const char* kEnvVar = "SPLASH_ENGINE_WORK_INTERVAL";

}  // namespace splash::engine::rfc193

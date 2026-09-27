#pragma once

// RFC 190 scaffolding: deferred GDN recurrent replay.
// See docs/rfc/190-deferred-gdn.md and
// https://github.com/incoai/splash/issues/190.

namespace splash::ops::rfc190 {

inline constexpr bool kEnabledDefault = false;
inline constexpr const char* kEnvVar = "SPLASH_DEFERRED_GDN_REPLAY";

}  // namespace splash::ops::rfc190

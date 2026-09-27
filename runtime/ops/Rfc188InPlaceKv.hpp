#pragma once

// RFC 188 scaffolding: in-place KV append + squeezed verify update.
// See docs/rfc/188-inplace-kv.md and
// https://github.com/incoai/splash/issues/188.

namespace splash::ops::rfc188 {

inline constexpr bool kEnabledDefault = false;
inline constexpr const char* kEnvVar = "SPLASH_INPLACE_KV_APPEND";

// Squeeze batch to 1 before slice-assign on head-transposed verify slices.
inline constexpr bool kSqueezeBatchForVerify = true;

}  // namespace splash::ops::rfc188

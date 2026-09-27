#pragma once

// RFC 192 scaffolding: fused prompt-causal attention.
// See docs/rfc/192-fused-attn.md and
// https://github.com/incoai/splash/issues/192.

namespace splash::ops::rfc192 {

inline constexpr bool kEnabledDefault = false;
inline constexpr const char* kEnvVar = "SPLASH_FUSED_PROMPT_ATTN";

}  // namespace splash::ops::rfc192

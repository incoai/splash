#pragma once

// RFC 191 scaffolding: few-row QMM + wide-QMV + prompt split-K candidates.
// See docs/rfc/191-fewrow-gemm.md and
// https://github.com/incoai/splash/issues/191.

#include <cstdint>

namespace splash::ops::rfc191 {

inline constexpr bool kEnabledDefault = false;
inline constexpr const char* kEnvVar = "SPLASH_FEWROW_GEMM";

// Candidate geometry (matches bonsai qmm_m16_block shape class).
inline constexpr std::uint32_t kBlockM = 32;
inline constexpr std::uint32_t kBlockN = 64;
inline constexpr std::uint32_t kMaxKSplits = 8;

}  // namespace splash::ops::rfc191

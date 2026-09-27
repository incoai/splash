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

// N64 narrow-column decode kernels added in linear_q4.metal. Compiled into
// the metallib but undispatched until the tuner wires them (default off).
// Names must match the Q4_DECODE_AFFINE instances (plus _f32 variants).
inline constexpr const char* kN64Plain = "decode_linear_q4_n64";
inline constexpr const char* kN64M16 = "decode_linear_q4_n64_m16";
inline constexpr const char* kN64M32 = "decode_linear_q4_n64_m32";

}  // namespace splash::ops::rfc191

#pragma once

#include <metal_stdlib>

using namespace metal;

// Shared by dense and bucketed-expert MPP routes. Affine coefficients are
// retained in their original BF16 storage and explicitly promoted to F32 for
// reconstruction. BF16 matrix operands round only after that reconstruction;
// BF16(q*scale)+bias is a distinct arithmetic convention; its source-kernel
// compatibility needs separate actual-GPU qualification.
inline void flash_mpp_error(device uint *diagnostics, uint bits) {
  atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(diagnostics),
                           bits, memory_order_relaxed);
}

inline bool flash_mpp_finite(float v) {
  return (as_type<uint>(v) & 0x7f800000u) != 0x7f800000u;
}

inline bool flash_mpp_finite(bfloat v) {
  return (as_type<ushort>(v) & 0x7f80u) != 0x7f80u;
}

// Bounds are the caller's responsibility: k is a valid source-row channel.
// Source rows use the original unsigned little-endian 4/5/6/8-bit bitstream.
inline uint flash_mpp_code(device const uchar *row, uint k, uint bits) {
  const uint bit = k * bits, byte = bit / 8, shift = bit % 8;
  uint word = uint(row[byte]);
  if (shift + bits > 8) word |= uint(row[byte + 1]) << 8;
  return (word >> shift) & ((1u << bits) - 1u);
}

inline float flash_mpp_dequantize_f32(uint code, bfloat scale, bfloat bias) {
  return float(code) * float(scale) + float(bias);
}

inline bfloat flash_mpp_dequantize_bf16(uint code, bfloat scale, bfloat bias) {
  return bfloat(flash_mpp_dequantize_f32(code, scale, bias));
}

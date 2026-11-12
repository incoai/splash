#pragma once

// Device-tensor views for MPP matmul2d and small load helpers shared by the kernel families.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

namespace h3 {
using namespace metal;
using namespace mpp::tensor_ops;

template <class T>
using dmat = tensor<device T, dextents<int, 2>, tensor_inline>;

// Row-major [rows][k] matrix (k contiguous) as a matmul2d operand.
template <class T>
inline dmat<T> mk(device T *p, int k, int rows) {
  return dmat<T>(p, dextents<int, 2>(k, rows), array<int, 2>{1, k});
}

inline float4 ld4(device const bfloat *p) { return float4(*(device const vec<bfloat, 4> *)p); }
inline float4 ld4(device const float *p) { return *(device const float4 *)p; }
inline float4 ld4(device const half *p) { return float4(*(device const half4 *)p); }
inline float4 ld4s(device const char *p) { return float4(*(device const char4 *)p); }

// Tile-order swizzle: GM consecutive M-blocks are walked for each N tile before moving on, so concurrently resident
// threadgroups share weight tiles in cache.
inline uint2 swz(uint2 tg, uint2 ng, int GM) {
  if (GM <= 1) return tg;
  const uint L = tg.y * ng.x + tg.x;
  const uint per = uint(GM) * ng.x;
  const uint g = L / per, r = L % per;
  const uint gs = min(uint(GM), ng.y - g * uint(GM));
  return uint2(r / gs, g * uint(GM) + r % gs);
}
}  // namespace h3

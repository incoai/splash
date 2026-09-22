#if __METAL_VERSION__ >= 400
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashDenseCache.h"

#pragma METAL fp math_mode(safe)
#include "metal/kernels/common/flash_affine_mpp_common.h"

using namespace metal;
using namespace mpp::tensor_ops;
#include "metal/kernels/common/flash_dense_traversal.h"

// One-time source-layout conversion. Affine coefficients stay signed, F32
// reconstruction precedes the single BF16 cast, and the original buffers are
// read only. Complete N*K coefficient coverage is independent of inference M.
kernel void flash_dense_cache_expand_bf16(
    device const uchar *weights [[buffer(0)]],
    device const uchar *scaleBytes [[buffer(1)]],
    device const uchar *biasBytes [[buffer(2)]],
    device bfloat *output [[buffer(3)]], device uint *diagnostics [[buffer(4)]],
    constant FlashAffineParams &p [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (p.rows != 1 || p.selections != 1 || p.experts != 1 || p.flags ||
      !p.input_size || p.input_size > 32768 || !p.output_size ||
      (p.bits != 4 && p.bits != 5 && p.bits != 6 && p.bits != 8) ||
      (p.group_size != 32 && p.group_size != 64 && p.group_size != 128) ||
      p.input_size % p.group_size ||
      p.weight_row_stride_bytes < (ulong(p.input_size) * p.bits + 7) / 8 ||
      p.parameter_row_stride_bytes < ulong(p.input_size / p.group_size) * 2 ||
      p.parameter_row_stride_bytes % 2 || group.y || group.z ||
      threads.x != 256 || threads.y != 1 || threads.z != 1) {
    if (tid == 0) flash_mpp_error(diagnostics, 2u);
    return;
  }
  const ulong index = ulong(group.x) * 256 + tid;
  if (index >= ulong(p.output_size) * p.input_size) return;
  const uint n = uint(index / p.input_size), k = uint(index % p.input_size);
  const uint code = flash_mpp_code(weights + ulong(n) * p.weight_row_stride_bytes,
                                  k, p.bits);
  const ulong offset = ulong(n) * p.parameter_row_stride_bytes +
                       ulong(k / p.group_size) * 2;
  const bfloat scale = *reinterpret_cast<device const bfloat *>(scaleBytes + offset);
  const bfloat bias = *reinterpret_cast<device const bfloat *>(biasBytes + offset);
  const float value = flash_mpp_dequantize_f32(code, scale, bias);
  const bfloat converted = bfloat(value);
  if (!flash_mpp_finite(value) || !flash_mpp_finite(converted))
    flash_mpp_error(diagnostics, 4u);
  output[index] = converted;
}

template <ushort M, ushort N, bool Traversal = false>
inline void flash_dense_cache_tile(
    device bfloat *input, device bfloat *weights, device bfloat *output,
    device uint *diagnostics, constant FlashDenseCacheParams &p,
    uint3 group, uint3 threads, uint tid) {
  if (!p.rows || p.rows > 8192 || p.rows % M || !p.input_size ||
      p.input_size > 32768 || p.input_size % 32 || !p.output_size ||
      !p.output_count || p.output_count % N || p.output_begin > p.output_size ||
      p.output_count > p.output_size - p.output_begin ||
      p.tile_rows != M || p.tile_outputs != N ||
      (Traversal ? p.reserved > 4 : p.reserved != 0) || group.z ||
      (Traversal ? !flash_dense_traversal_group_valid(group.xy, p.rows / M,
                           p.output_count / N, p.reserved)
                 : group.x >= p.output_count / N || group.y >= p.rows / M) ||
      threads.x != 128 || threads.y != 1 || threads.z != 1) {
    if (tid == 0) flash_mpp_error(diagnostics, 2u);
    return;
  }
  const uint2 tile = Traversal ? flash_dense_traversal_tile(group.xy, p.reserved) : group.xy;
  if (Traversal && (tile.x >= p.output_count / N || tile.y >= p.rows / M)) return;
  const uint row = tile.y * M, column = p.output_begin + tile.x * N;
  const int k = int(p.input_size);
  auto a = tensor(input + ulong(row) * p.input_size,
                  dextents<int, 2>{k, M}, array<int, 2>{1, k});
  auto b = tensor(weights + ulong(column) * p.input_size,
                  dextents<int, 2>{k, N}, array<int, 2>{1, k});
  constexpr auto descriptor = matmul2d_descriptor(
      M, N, static_cast<int>(dynamic_extent), false, true, false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<4>> operation;
  auto dot = operation.template get_destination_cooperative_tensor<
      decltype(a), decltype(b), float>();
  operation.run(a, b, dot);
#pragma unroll
  for (ushort i = 0; i < dot.get_capacity(); ++i) {
    if (!dot.is_valid_element(i)) continue;
    const auto index = dot.get_multidimensional_index(i);
    const bfloat value = bfloat(dot[i]);
    if (!flash_mpp_finite(dot[i]) || !flash_mpp_finite(value))
      flash_mpp_error(diagnostics, 4u);
    output[ulong(row + index[1]) * p.output_size + column + index[0]] = value;
  }
}

#define FLASH_DENSE_CACHE_ENTRY(Name, M, N, Traversal)                       \
  kernel void Name(device bfloat *input [[buffer(0)]],                      \
      device bfloat *weights [[buffer(1)]], device bfloat *output [[buffer(2)]], \
      device uint *diagnostics [[buffer(3)]],                               \
      constant FlashDenseCacheParams &params [[buffer(4)]],                 \
      uint3 group [[threadgroup_position_in_grid]],                         \
      uint3 threads [[threads_per_threadgroup]],                           \
      uint tid [[thread_index_in_threadgroup]]) {                          \
    flash_dense_cache_tile<M, N, Traversal>(input, weights, output, diagnostics, \
                                 params, group, threads, tid);             \
  }

FLASH_DENSE_CACHE_ENTRY(flash_dense_cache_m8_n64, 8, 64, false)
FLASH_DENSE_CACHE_ENTRY(flash_dense_cache_m16_n64, 16, 64, false)
FLASH_DENSE_CACHE_ENTRY(flash_dense_cache_m16_n128, 16, 128, false)
FLASH_DENSE_CACHE_ENTRY(flash_dense_cache_m32_n64, 32, 64, false)
FLASH_DENSE_CACHE_ENTRY(flash_dense_cache_m32_n128, 32, 128, false)
FLASH_DENSE_CACHE_ENTRY(flash_dense_cache_m8_n64_traversal, 8, 64, true)
FLASH_DENSE_CACHE_ENTRY(flash_dense_cache_m16_n64_traversal, 16, 64, true)
FLASH_DENSE_CACHE_ENTRY(flash_dense_cache_m16_n128_traversal, 16, 128, true)
FLASH_DENSE_CACHE_ENTRY(flash_dense_cache_m32_n64_traversal, 32, 64, true)
FLASH_DENSE_CACHE_ENTRY(flash_dense_cache_m32_n128_traversal, 32, 128, true)

#undef FLASH_DENSE_CACHE_ENTRY

inline bfloat flash_cached_hc_precise_sigmoid(bfloat source) {
  const bfloat e = bfloat(precise::exp(abs(float(source))));
  const bfloat d = bfloat(1.0f + float(e));
  float reciprocal = 1.0f / float(d);
  const uint word = as_type<uint>(reciprocal);
  // The qualified standalone sigmoid flushes its F32 reciprocal intermediate;
  // products and final BF16 values retain subnormals.
  if (!(word & 0x7f800000u))
    reciprocal = as_type<float>(word & 0x80000000u);
  const bfloat tail = bfloat(reciprocal);
  return source < bfloat(0.0f) ? tail : bfloat(1.0f - float(tail));
}

template <ushort M, ushort N>
inline void flash_dense_cache_hc_up_mix_tile(
    device bfloat *activatedDown, device bfloat *weights,
    device const bfloat *normalized, device bfloat *mixed,
    device uint *diagnostics, constant FlashDenseCacheParams &p,
    uint3 group, uint3 threads, uint tid) {
  if (p.rows < 32 || p.rows > 8192 || p.rows % M || p.input_size != 320 ||
      p.output_size != 2560 || p.output_begin || p.output_count != 2560 ||
      p.tile_rows != M || p.tile_outputs != N || p.reserved || group.z ||
      group.x >= 2560 / N || group.y >= p.rows / M ||
      threads.x != 128 || threads.y != 1 || threads.z != 1) {
    if (tid == 0) flash_mpp_error(diagnostics, 2u);
    return;
  }
  const uint rowOrigin = group.y * M, columnOrigin = group.x * N;
  auto a = tensor(activatedDown + ulong(rowOrigin) * 320,
      dextents<int, 2>{320, M}, array<int, 2>{1, 320});
  auto firstB = tensor(weights + ulong(columnOrigin) * 320,
      dextents<int, 2>{320, N}, array<int, 2>{1, 320});
  constexpr auto descriptor = matmul2d_descriptor(
      M, N, static_cast<int>(dynamic_extent), false, true, false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<4>> operation;
  auto streamTotal = operation.template get_destination_cooperative_tensor<
      decltype(a), decltype(firstB), float>();
#pragma unroll
  for (ushort i = 0; i < streamTotal.get_capacity(); ++i)
    if (streamTotal.is_valid_element(i)) streamTotal[i] = 0.0f;
  // Each destination remains F32 typed, so intrinsic coordinates/validity
  // match between streamTotal and all four dots. Storage boundaries below are
  // explicit BF16 values rather than indexing a BF16 cooperative fragment.
  for (uint stream = 0; stream < 4; ++stream) {
    auto b = tensor(weights + ulong(stream * 2560 + columnOrigin) * 320,
        dextents<int, 2>{320, N}, array<int, 2>{1, 320});
    auto dot = operation.template get_destination_cooperative_tensor<
        decltype(a), decltype(b), float>();
    operation.run(a, b, dot);
#pragma unroll
    for (ushort i = 0; i < streamTotal.get_capacity(); ++i) {
      if (!streamTotal.is_valid_element(i)) continue;
      const auto index = streamTotal.get_multidimensional_index(i);
      const uint row = rowOrigin + index[1], column = columnOrigin + index[0];
      const bfloat raw = bfloat(dot[i]);
      if (!flash_mpp_finite(dot[i]) || !flash_mpp_finite(raw))
        flash_mpp_error(diagnostics, 4u);
      const bfloat gate = flash_cached_hc_precise_sigmoid(raw);
      const bfloat product = bfloat(float(gate) * float(
          normalized[(ulong(row) * 4 + stream) * 2560 + column]));
      streamTotal[i] = float(bfloat(float(product) + streamTotal[i]));
    }
  }
#pragma unroll
  for (ushort i = 0; i < streamTotal.get_capacity(); ++i) {
    if (!streamTotal.is_valid_element(i)) continue;
    const auto index = streamTotal.get_multidimensional_index(i);
    const bfloat value = bfloat(streamTotal[i] / 4.0f);
    if (!flash_mpp_finite(value)) flash_mpp_error(diagnostics, 4u);
    mixed[ulong(rowOrigin + index[1]) * 2560 + columnOrigin + index[0]] = value;
  }
}

#define FLASH_CACHED_HC_UP_MIX(Name, M, N)                                   \
  kernel void Name(device bfloat *activatedDown [[buffer(0)]],             \
      device bfloat *weights [[buffer(1)]],                                \
      device const bfloat *normalized [[buffer(2)]],                       \
      device bfloat *mixed [[buffer(3)]], device uint *diagnostics [[buffer(4)]], \
      constant FlashDenseCacheParams &params [[buffer(5)]],                 \
      uint3 group [[threadgroup_position_in_grid]],                         \
      uint3 threads [[threads_per_threadgroup]],                           \
      uint tid [[thread_index_in_threadgroup]]) {                          \
    flash_dense_cache_hc_up_mix_tile<M, N>(activatedDown, weights,          \
        normalized, mixed, diagnostics, params, group, threads, tid);      \
  }

FLASH_CACHED_HC_UP_MIX(flash_dense_cache_hc_up_mix_m8_n64, 8, 64)
FLASH_CACHED_HC_UP_MIX(flash_dense_cache_hc_up_mix_m16_n64, 16, 64)
FLASH_CACHED_HC_UP_MIX(flash_dense_cache_hc_up_mix_m16_n128, 16, 128)
FLASH_CACHED_HC_UP_MIX(flash_dense_cache_hc_up_mix_m32_n64, 32, 64)
FLASH_CACHED_HC_UP_MIX(flash_dense_cache_hc_up_mix_m32_n128, 32, 128)

#undef FLASH_CACHED_HC_UP_MIX
#endif

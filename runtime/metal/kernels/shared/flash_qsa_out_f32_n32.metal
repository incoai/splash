// Qualified QSA output original-F32 coefficient route; no storage conversion.
#if __METAL_VERSION__ >= 400
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashFloatDenseCache.h"

#pragma METAL fp math_mode(safe)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#include "metal/kernels/common/flash_affine_mpp_common.h"

using namespace metal;
using namespace mpp::tensor_ops;

inline bool flash_qsa_out_f32_n32_geometry(constant FlashFloatDenseSmallRowsParams &p) {
  return p.rows && p.rows <= 16 && (p.tile_rows == 8 || p.tile_rows == 16) &&
      (p.tile_outputs == 32 || p.tile_outputs == 64 || p.tile_outputs == 128) &&
      p.padded_rows == (p.rows + p.tile_rows - 1) / p.tile_rows * p.tile_rows &&
      p.padded_rows <= 16 && p.input_size == 6144 && p.output_size == 2560 && p.rows >= 4;
}

template <ushort M, ushort N, ushort SIMD>
inline void flash_qsa_out_f32_n32_tile(
    device bfloat *padded, device float *weights, device bfloat *output,
    device uint *diagnostics, constant FlashFloatDenseSmallRowsParams &p,
    uint3 group, uint3 threads, uint tid) {
  if (!flash_qsa_out_f32_n32_geometry(p) || p.tile_rows != M || p.tile_outputs != N ||
      !p.output_count || p.output_count % N || p.output_begin > p.output_size ||
      p.output_count > p.output_size - p.output_begin || group.z ||
      group.x >= p.output_count / N || group.y >= p.padded_rows / M ||
      threads.x != SIMD * 32 || threads.y != 1 || threads.z != 1) {
    if (tid == 0) flash_mpp_error(diagnostics, 2u);
    return;
  }
  const uint row = group.y * M, column = p.output_begin + group.x * N;
  const int k = int(p.input_size);
  auto a = tensor(padded + ulong(row) * p.input_size,
      dextents<int, 2>{k, M}, array<int, 2>{1, k});
  auto b = tensor(weights + ulong(column) * p.input_size,
      dextents<int, 2>{k, N}, array<int, 2>{1, k});
  constexpr auto descriptor = matmul2d_descriptor(
      M, N, static_cast<int>(dynamic_extent), false, true, false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<SIMD>> operation;
  auto dot = operation.template get_destination_cooperative_tensor<
      decltype(a), decltype(b), float>();
  operation.run(a, b, dot);
#pragma unroll
  for (ushort i = 0; i < dot.get_capacity(); ++i) {
    if (!dot.is_valid_element(i)) continue;
    const auto index = dot.get_multidimensional_index(i);
    const uint outputRow = row + index[1];
    if (outputRow >= p.rows) continue;
    const bfloat value = bfloat(dot[i]);
    if (!flash_mpp_finite(dot[i]) || !flash_mpp_finite(value))
      flash_mpp_error(diagnostics, 4u);
    output[ulong(outputRow) * p.output_size + column + index[0]] = value;
  }
}

#define FLASH_DENSE_SMALL_ROWS(Name, M, N, SIMD)                                   \
  kernel void Name(device bfloat *padded [[buffer(0)]],                    \
      device float *weights [[buffer(1)]], device bfloat *output [[buffer(2)]], \
      device uint *diagnostics [[buffer(3)]],                               \
      constant FlashFloatDenseSmallRowsParams &params [[buffer(4)]],             \
      uint3 group [[threadgroup_position_in_grid]],                         \
      uint3 threads [[threads_per_threadgroup]],                           \
      uint tid [[thread_index_in_threadgroup]]) {                          \
    flash_qsa_out_f32_n32_tile<M, N, SIMD>(padded, weights, output, diagnostics, \
                                      params, group, threads, tid);        \
  }

FLASH_DENSE_SMALL_ROWS(flash_qsa_out_f32_n32_m8_n32_s4, 8, 32, 4)
#undef FLASH_DENSE_SMALL_ROWS
#endif

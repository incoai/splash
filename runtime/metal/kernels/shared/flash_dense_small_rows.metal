#if __METAL_VERSION__ >= 400
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashDenseSmallRows.h"

#pragma METAL fp math_mode(safe)
#include "metal/kernels/common/flash_affine_mpp_common.h"

using namespace metal;
using namespace mpp::tensor_ops;

inline bool flash_dense_small_rows_geometry(constant FlashDenseSmallRowsParams &p) {
  return p.rows && p.rows <= 16 && (p.tile_rows == 8 || p.tile_rows == 16) &&
      (p.tile_outputs == 64 || p.tile_outputs == 128) &&
      p.padded_rows == (p.rows + p.tile_rows - 1) / p.tile_rows * p.tile_rows &&
      p.padded_rows <= 16 && p.input_size && p.input_size <= 32768 &&
      p.input_size % 32 == 0 && p.output_size && p.output_size % 64 == 0;
}

// Exact BF16 bit copy. Padding is positive zero, including when source data
// contains NaNs; nonfinite source words set diagnostics without modifying them.
// Only padded_rows*actualK words are touched in the reusable Shared arena.
kernel void flash_dense_small_rows_pad(
    device const ushort *input [[buffer(0)]], device ushort *padded [[buffer(1)]],
    device uint *diagnostics [[buffer(2)]],
    constant FlashDenseSmallRowsParams &p [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  if (!flash_dense_small_rows_geometry(p) || group.y || group.z ||
      threads.x != 256 || threads.y != 1 || threads.z != 1) {
    if (tid == 0) flash_mpp_error(diagnostics, 2u);
    return;
  }
  const ulong index = ulong(group.x) * 256 + tid;
  if (index >= ulong(p.padded_rows) * p.input_size) return;
  ushort word = 0;
  if (index < ulong(p.rows) * p.input_size) {
    word = input[index];
    if ((word & 0x7f80u) == 0x7f80u) flash_mpp_error(diagnostics, 4u);
  }
  padded[index] = word;
}

template <ushort M, ushort N>
inline void flash_dense_small_rows_tile(
    device bfloat *padded, device bfloat *weights, device bfloat *output,
    device uint *diagnostics, constant FlashDenseSmallRowsParams &p,
    uint3 group, uint3 threads, uint tid) {
  if (!flash_dense_small_rows_geometry(p) || p.tile_rows != M || p.tile_outputs != N ||
      !p.output_count || p.output_count % N || p.output_begin > p.output_size ||
      p.output_count > p.output_size - p.output_begin || group.z ||
      group.x >= p.output_count / N || group.y >= p.padded_rows / M ||
      threads.x != 128 || threads.y != 1 || threads.z != 1) {
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
  matmul2d<descriptor, execution_simdgroups<4>> operation;
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

#define FLASH_DENSE_SMALL_ROWS(Name, M, N)                                   \
  kernel void Name(device bfloat *padded [[buffer(0)]],                    \
      device bfloat *weights [[buffer(1)]], device bfloat *output [[buffer(2)]], \
      device uint *diagnostics [[buffer(3)]],                               \
      constant FlashDenseSmallRowsParams &params [[buffer(4)]],             \
      uint3 group [[threadgroup_position_in_grid]],                         \
      uint3 threads [[threads_per_threadgroup]],                           \
      uint tid [[thread_index_in_threadgroup]]) {                          \
    flash_dense_small_rows_tile<M, N>(padded, weights, output, diagnostics, \
                                      params, group, threads, tid);        \
  }

FLASH_DENSE_SMALL_ROWS(flash_dense_small_rows_m8_n64, 8, 64)
FLASH_DENSE_SMALL_ROWS(flash_dense_small_rows_m8_n128, 8, 128)
FLASH_DENSE_SMALL_ROWS(flash_dense_small_rows_m16_n64, 16, 64)
FLASH_DENSE_SMALL_ROWS(flash_dense_small_rows_m16_n128, 16, 128)

#undef FLASH_DENSE_SMALL_ROWS
#endif

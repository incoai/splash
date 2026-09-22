#if __METAL_VERSION__ >= 400
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashAffineMPP.h"

#pragma METAL fp math_mode(safe)
#include "metal/kernels/common/flash_affine_mpp_common.h"

using namespace metal;
using namespace mpp::tensor_ops;

// Dense prefill on original MLX unsigned little-endian row-major weights.
// Group-affine mode feeds exact integer codes (0..255) to BF16 matrix units,
// applying F32 signed scale/bias only to the F32 group dot. Mode 1 explicitly
// rounds reconstructed coefficients to BF16. Tails never read source padding.
// No activation quantization, full weight expansion, or source conversion.

template <ushort M, ushort N, ushort G>
inline void flash_affine_mpp_tile(
    device const bfloat *input, device const uchar *weights,
    device const uchar *scaleBytes, device const uchar *biasBytes,
    device bfloat *output, device uint *diagnostics,
    constant FlashAffineMPPParams &params, uint3 group, uint3 threads,
    uint tid, uint lane, uint simdgroup,
    threadgroup bfloat *stagedA, threadgroup bfloat *stagedB,
    threadgroup float *rowSums) {
  const constant FlashAffineParams &p = params.affine;
  if (!p.rows || p.rows > 8192 || !p.input_size || p.input_size > 32768 ||
      !p.output_size || p.experts != 1 || p.selections != 1 || p.flags ||
      p.group_size != G || p.input_size % G ||
      (p.bits != 4 && p.bits != 5 && p.bits != 6 && p.bits != 8) ||
      p.weight_row_stride_bytes < (ulong(p.input_size) * p.bits + 7) / 8 ||
      p.parameter_row_stride_bytes < ulong(p.input_size / G) * 2 ||
      p.parameter_row_stride_bytes % 2 || params.mode > 1 || params.reserved ||
      params.tile_rows != M || params.tile_outputs != N ||
      threads.x != 128 || threads.y != 1 || threads.z != 1 || group.z ||
      group.y >= (p.rows - 1) / M + 1 ||
      group.x >= (p.output_size - 1) / N + 1) {
    if (tid == 0) flash_mpp_error(diagnostics, 2u);
    return;
  }
  // Half-group chunks keep G128/N128 tiles within 32 KiB. Coefficients remain
  // addressed by the full quantization group, while each half contributes its
  // own affine dot and input sum to the F32 accumulator.
  constexpr ushort BK = G == 128 && N == 128 ? 64 : G;
  const uint rowOrigin = group.y * M, outputOrigin = group.x * N;
  auto a = tensor(stagedA, dextents<int, 2>{BK, M}, array<int, 2>{1, BK});
  auto b = tensor(stagedB, dextents<int, 2>{BK, N}, array<int, 2>{1, BK});
  auto a0 = a.template slice<BK, M>(0, 0);
  auto b0 = b.template slice<BK, N>(0, 0);
  constexpr auto descriptor = matmul2d_descriptor(
      M, N, BK, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<4>> operation;
  auto accumulated = operation.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i)
    if (accumulated.is_valid_element(i)) accumulated[i] = 0.0f;

  for (uint chunk = 0; chunk < p.input_size / BK; ++chunk) {
    const uint inputOrigin = chunk * BK, qgroup = inputOrigin / G;
    for (uint i = tid; i < uint(M) * BK; i += 128) {
      const uint row = i / BK, k = i % BK;
      bfloat value = bfloat(0.0f);
      if (rowOrigin + row < p.rows) {
        value = input[ulong(rowOrigin + row) * p.input_size + inputOrigin + k];
        if (!flash_mpp_finite(value)) {
          flash_mpp_error(diagnostics, 4u);
          value = bfloat(0.0f);
        }
      }
      stagedA[i] = value;
    }
    for (uint i = tid; i < uint(N) * BK; i += 128) {
      const uint column = i / BK, k = i % BK, n = outputOrigin + column;
      bfloat value = bfloat(0.0f);
      if (n < p.output_size) {
        const uint code = flash_mpp_code(
            weights + ulong(n) * p.weight_row_stride_bytes,
            inputOrigin + k, p.bits);
        if (params.mode == 0) {
          // All integer codes fit the exact BF16 integer range [0,256].
          value = bfloat(float(code));
        } else {
          const ulong offset = ulong(n) * p.parameter_row_stride_bytes +
                               ulong(qgroup) * 2;
          const bfloat scale = *reinterpret_cast<device const bfloat *>(
              scaleBytes + offset);
          const bfloat bias = *reinterpret_cast<device const bfloat *>(
              biasBytes + offset);
          const float reconstructed = flash_mpp_dequantize_f32(code, scale, bias);
          value = bfloat(reconstructed);
          if (!flash_mpp_finite(reconstructed) || !flash_mpp_finite(value)) {
            flash_mpp_error(diagnostics, 4u);
            value = bfloat(0.0f);
          }
        }
      }
      stagedB[i] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Four SIMD groups compute all input-row sums from the same staged BF16
    // data used by MPP. Each lane owns BK/32 values; no reduction in BF16.
    if (params.mode == 0) {
      for (uint row = simdgroup; row < M; row += 4) {
        float sum = 0.0f;
#pragma unroll
        for (ushort k = 0; k < BK; k += 32)
          sum += float(stagedA[row * BK + k + lane]);
        sum = simd_sum(sum);
        if (lane == 0) rowSums[row] = sum;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto partial = operation.template get_destination_cooperative_tensor<
        decltype(a0), decltype(b0), float>();
    operation.run(a0, b0, partial);
#pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
      if (!accumulated.is_valid_element(i)) continue;
      const auto index = accumulated.get_multidimensional_index(i);
      const uint n = outputOrigin + index[0];
      if (params.mode == 0 && n < p.output_size) {
        const ulong offset = ulong(n) * p.parameter_row_stride_bytes +
                             ulong(qgroup) * 2;
        const float scale = float(*reinterpret_cast<device const bfloat *>(
            scaleBytes + offset));
        const float bias = float(*reinterpret_cast<device const bfloat *>(
            biasBytes + offset));
        if (!flash_mpp_finite(scale) || !flash_mpp_finite(bias))
          flash_mpp_error(diagnostics, 4u);
        accumulated[i] += partial[i] * scale + rowSums[index[1]] * bias;
      } else {
        accumulated[i] += partial[i];
      }
    }
    // MPP must consume both stages before the next quantization group writes.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
    if (!accumulated.is_valid_element(i)) continue;
    const auto index = accumulated.get_multidimensional_index(i);
    const uint n = outputOrigin + index[0], row = rowOrigin + index[1];
    if (n >= p.output_size || row >= p.rows) continue;
    const bfloat value = bfloat(accumulated[i]);
    if (!flash_mpp_finite(accumulated[i]) || !flash_mpp_finite(value))
      flash_mpp_error(diagnostics, 4u);
    output[ulong(row) * p.output_size + n] = value;
  }
}

#define FLASH_AFFINE_MPP(Name, M, N, G)                                      \
  kernel void Name(                                                        \
      device const bfloat *input [[buffer(0)]],                             \
      device const uchar *weights [[buffer(1)]],                            \
      device const uchar *scales [[buffer(2)]],                             \
      device const uchar *biases [[buffer(3)]],                             \
      device bfloat *output [[buffer(4)]], device uint *diagnostics [[buffer(5)]], \
      constant FlashAffineMPPParams &params [[buffer(6)]],                 \
      uint3 group [[threadgroup_position_in_grid]],                         \
      uint3 threads [[threads_per_threadgroup]],                           \
      uint tid [[thread_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]],                             \
      uint simdgroup [[simdgroup_index_in_threadgroup]]) {                  \
    constexpr ushort BK = G == 128 && N == 128 ? 64 : G;                  \
    alignas(16) threadgroup bfloat a[M * BK], b[N * BK];                    \
    threadgroup float sums[M];                                            \
    flash_affine_mpp_tile<M, N, G>(input, weights, scales, biases, output,   \
        diagnostics, params, group, threads, tid, lane, simdgroup, a, b, sums); \
  }

FLASH_AFFINE_MPP(flash_affine_mpp_m8_n64_g32, 8, 64, 32)
FLASH_AFFINE_MPP(flash_affine_mpp_m16_n64_g32, 16, 64, 32)
FLASH_AFFINE_MPP(flash_affine_mpp_m16_n128_g32, 16, 128, 32)
FLASH_AFFINE_MPP(flash_affine_mpp_m32_n64_g32, 32, 64, 32)
FLASH_AFFINE_MPP(flash_affine_mpp_m32_n128_g32, 32, 128, 32)
FLASH_AFFINE_MPP(flash_affine_mpp_m8_n64_g64, 8, 64, 64)
FLASH_AFFINE_MPP(flash_affine_mpp_m16_n64_g64, 16, 64, 64)
FLASH_AFFINE_MPP(flash_affine_mpp_m16_n128_g64, 16, 128, 64)
FLASH_AFFINE_MPP(flash_affine_mpp_m32_n64_g64, 32, 64, 64)
FLASH_AFFINE_MPP(flash_affine_mpp_m32_n128_g64, 32, 128, 64)
FLASH_AFFINE_MPP(flash_affine_mpp_m8_n64_g128, 8, 64, 128)
FLASH_AFFINE_MPP(flash_affine_mpp_m16_n64_g128, 16, 64, 128)
FLASH_AFFINE_MPP(flash_affine_mpp_m16_n128_g128, 16, 128, 128)
FLASH_AFFINE_MPP(flash_affine_mpp_m32_n64_g128, 32, 64, 128)
FLASH_AFFINE_MPP(flash_affine_mpp_m32_n128_g128, 32, 128, 128)

#undef FLASH_AFFINE_MPP
#endif

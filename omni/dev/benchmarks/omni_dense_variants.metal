// Variants of the audio tower's MPP GEMM tile (kernels/omni/gemm_bf16.metal
// omni_gemm_tile) for omni-dense-bench --tower, linked beside the
// production kernels into a bench library: the tile's rows and columns, the
// simdgroups that run it, the K of one matmul call (0: the whole K in one
// call), how many calls run between threadgroup barriers (0: none), the
// weights K-contiguous as the checkpoint stores them ([N][K]) or a
// pretransposed copy ([K][N]), and bf16 or fp16 operands (fp32 sums
// always; bf16 or fp32 rows out). The epilogue is the production one.

#include "metal/abi/Omni.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

inline float variant_erf(float x) {
  const float sign = x < 0.0f ? -1.0f : 1.0f;
  x = fabs(x);
  const float t = 1.0f / (1.0f + 0.3275911f * x);
  const float polynomial =
      t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
  return sign * (1.0f - polynomial * precise::exp(-x * x));
}

inline float variant_activate(float x, uint activation) {
  if (activation == 1u) return 0.5f * x * (1.0f + variant_erf(x * 0.7071067811865476f));
  if (activation == 2u) return x / (1.0f + precise::exp(-x));
  return x;
}

template <typename Output>
inline void variant_store(float value, uint row, uint column, device const float *bias, device const Output *residual,
                          device Output *output, constant OmniGemmParams &p) {
  if (p.flags & OMNI_GEMM_FLAG_BIAS) value += bias[column];
  value = variant_activate(value, p.activation);
  const ulong index = ulong(row) * p.output_stride + column;
  if (p.flags & OMNI_GEMM_FLAG_RESIDUAL) value += float(residual[index]);
  output[index] = Output(value);
}

template <ushort TileM, ushort TileN, ushort BK, ushort SG, ushort Barrier, bool Pretransposed, typename T,
          typename Output>
inline void variant_tile(device T *input, device T *weights, device const float *bias, device const Output *residual,
                         device Output *output, constant OmniGemmParams &p, uint2 group) {
  const uint row_origin = group.x * TileM;
  const uint output_origin = group.y * TileN;
  const int K = int(p.input_size), N = int(p.output_size);
  auto a = tensor(input + ulong(row_origin) * p.input_stride, dextents<int, 2>{K, TileM},
                  array<int, 2>{1, int(p.input_stride)});
  // [N][K] read transposed, or [K][N] as it is.
  auto b = Pretransposed ? tensor(weights + output_origin, dextents<int, 2>{TileN, K}, array<int, 2>{1, N})
                         : tensor(weights + ulong(output_origin) * p.input_size, dextents<int, 2>{K, TileN},
                                  array<int, 2>{1, K});
  constexpr int CallK = BK ? int(BK) : int(dynamic_extent);
  constexpr auto descriptor = matmul2d_descriptor(TileM, TileN, CallK, false, !Pretransposed, false,
                                                  matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<SG>> operation;
  auto accumulated = operation.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i) accumulated[i] = 0.0f;
  if constexpr (BK == 0) {
    operation.run(a, b, accumulated);
  } else {
    for (uint chunk = 0; chunk < p.input_size / BK; ++chunk) {
      auto a_slice = a.template slice<BK, TileM>(chunk * BK, 0);
      if constexpr (Pretransposed) {
        auto b_slice = b.template slice<TileN, BK>(0, chunk * BK);
        operation.run(a_slice, b_slice, accumulated);
      } else {
        auto b_slice = b.template slice<BK, TileN>(chunk * BK, 0);
        operation.run(a_slice, b_slice, accumulated);
      }
      if (Barrier && (chunk + 1) % Barrier == 0) threadgroup_barrier(mem_flags::mem_none);
    }
  }
#pragma unroll
  for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
    if (!accumulated.is_valid_element(i)) continue;
    auto index = accumulated.get_multidimensional_index(i);
    const uint column = output_origin + index[0];
    const uint row = row_origin + index[1];
    if (row >= p.rows) continue;
    variant_store<Output>(accumulated[i], row, column, bias, residual, output, p);
  }
}

#define VARIANT_KERNEL(Name, M, N, BK, SG, B, PT, T, Output)                                                     \
  kernel void Name(device T *input [[buffer(0)]], device T *weights [[buffer(1)]],                              \
                   device const float *bias [[buffer(2)]], device const Output *residual [[buffer(3)]],         \
                   device Output *output [[buffer(4)]], constant OmniGemmParams &params [[buffer(5)]],          \
                   uint2 group [[threadgroup_position_in_grid]]) {                                             \
    variant_tile<M, N, BK, SG, B, PT, T, Output>(input, weights, bias, residual, output, params, group);       \
  }
// omni_tv_m{M}n{N}k{BK}s{SG}b{barrier}{t: [N][K] | p: [K][N]}{bf | hf}, and _f32 for fp32 rows.
#define VARIANT(M, N, BK, SG, B, PT, L, T, TN)                                                                   \
  VARIANT_KERNEL(omni_tv_m##M##n##N##k##BK##s##SG##b##B##L##TN, M, N, BK, SG, B, PT, T, bfloat)                       \
  VARIANT_KERNEL(omni_tv_m##M##n##N##k##BK##s##SG##b##B##L##TN##_f32, M, N, BK, SG, B, PT, T, float)
#define VARIANT_BK(M, N, SG)                                                                                      \
  VARIANT(M, N, 32, SG, 1, false, t, bfloat, bf)                                                                 \
  VARIANT(M, N, 64, SG, 1, false, t, bfloat, bf)                                                                 \
  VARIANT(M, N, 128, SG, 1, false, t, bfloat, bf)                                                                \
  VARIANT(M, N, 256, SG, 1, false, t, bfloat, bf)                                                                \
  VARIANT(M, N, 512, SG, 1, false, t, bfloat, bf)                                                                \
  VARIANT(M, N, 0, SG, 0, false, t, bfloat, bf)                                                                  \
  VARIANT(M, N, 64, SG, 0, false, t, bfloat, bf)                                                                 \
  VARIANT(M, N, 64, SG, 4, false, t, bfloat, bf)                                                                 \
  VARIANT(M, N, 128, SG, 0, false, t, bfloat, bf)                                                                \
  VARIANT(M, N, 256, SG, 0, false, t, bfloat, bf)                                                                \
  VARIANT(M, N, 64, SG, 1, true, p, bfloat, bf)                                                                  \
  VARIANT(M, N, 256, SG, 0, true, p, bfloat, bf)                                                                 \
  VARIANT(M, N, 64, SG, 1, false, t, half, hf)                                                                   \
  VARIANT(M, N, 256, SG, 0, false, t, half, hf)                                                                  \
  VARIANT(M, N, 0, SG, 0, false, t, half, hf)
VARIANT_BK(32, 64, 4)
VARIANT_BK(32, 64, 8)
VARIANT_BK(64, 64, 4)
VARIANT_BK(64, 64, 8)
VARIANT_BK(64, 128, 4)
VARIANT_BK(64, 128, 8)
VARIANT_BK(32, 128, 4)
VARIANT_BK(128, 64, 8)
VARIANT_BK(128, 128, 8)
VARIANT_BK(32, 32, 4)

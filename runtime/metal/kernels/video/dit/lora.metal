// Runtime LoRA down-projections of an installed adapter (abi/video/LinearABI.h): T = scale (A x), fp16 [Mpad][ldt], one
// 64 x 128 tile of T per threadgroup (MPP matmul2d, fp32 accumulation).
//   dit_lora_down      x the int8 rows of a DiT GEMM's input (Hadamard-rotated, one fp32 scale per row) x A bf16
//                      (rotated as the int8 weights): int8 x bfloat -> float, times the row's scale
//   dit_lora_down_f16  x fp16 rows (the QKFIX blocks' unrotated q / k / v GEMM input) x A bf16: half x bfloat -> float
// Rows past M are not written (the up-projection's rows past M are discarded).
#include "metal/abi/video/LinearABI.h"
#include "metal/kernels/video/common/tensor.h"

using namespace h3;

template <int TM, int TN, bool SCALED, class Op, class TA, class TB>
inline void lora_down_tile(thread Op &op, thread TA &a, thread TB &b, int m0, int n0, device const float *sa,
                           device half *T, constant LoraDownParams &p) {
  auto c = op.template get_destination_cooperative_tensor<TA, TB, float>();
#pragma unroll
  for (ushort i = 0; i < c.get_capacity(); ++i)
    if (c.is_valid_element(i)) c[i] = 0.0f;
  op.run(a, b, c);
#pragma unroll
  for (ushort i = 0; i < c.get_capacity(); ++i) {
    if (!c.is_valid_element(i)) continue;
    auto idx = c.get_multidimensional_index(i);
    const int row = m0 + idx[1], col = n0 + idx[0];
    if (row >= p.M) continue;
    float v = float(c[i]) * p.scale;
    if (SCALED) v *= sa[row];
    T[ulong(row) * p.ldt + col] = half(clamp(v, -65504.0f, 65504.0f));
  }
}

template <int TM, int TN, int NSG, class TX, bool SCALED>
kernel void dit_lora_down(device TX *X [[buffer(0)]], device bfloat *A [[buffer(1)]],
                          device const float *sa [[buffer(2)]], device half *T [[buffer(3)]],
                          constant LoraDownParams &p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]]) {
  auto tX = mk<TX>(X, p.K, p.M);
  auto tA = mk<bfloat>(A, p.K, p.R);
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  const int m0 = tg.y * TM, n0 = tg.x * TN;
  auto b = tA.template slice<dynamic_extent, TN>(0, n0);
  if (m0 + TM <= p.M) {
    auto a = tX.template slice<dynamic_extent, TM>(0, m0);
    lora_down_tile<TM, TN, SCALED>(op, a, b, m0, n0, sa, T, p);
  } else {
    auto a = tX.slice(0, m0);
    lora_down_tile<TM, TN, SCALED>(op, a, b, m0, n0, sa, T, p);
  }
}
template [[host_name("dit_lora_down")]] [[kernel]] void
dit_lora_down<H3_LINEAR_TM, H3_LORA_RANK, H3_LINEAR_NSG, int8_t, true>(device int8_t *, device bfloat *,
                                                                       device const float *, device half *,
                                                                       constant LoraDownParams &, uint2);
template [[host_name("dit_lora_down_f16")]] [[kernel]] void
dit_lora_down<H3_LINEAR_TM, H3_LORA_RANK, H3_LINEAR_NSG, half, false>(device half *, device bfloat *,
                                                                      device const float *, device half *,
                                                                      constant LoraDownParams &, uint2);

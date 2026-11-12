#pragma once

// The up-projection of a runtime LoRA inside a DiT GEMM's epilogue (abi/video/LinearABI.h): the TM x TN tile of
// T[:, col0 .. col0 + rank) B[brow .. brow + TN)^T as an MPP matmul2d (half x bfloat -> float) destination of the same
// shape and execution as the GEMM's own, whose element i is element i of the GEMM's.
#include "metal/abi/video/LinearABI.h"
#include "metal/kernels/video/common/tensor.h"

namespace h3 {

template <int TM, int TN, int NSG>
inline auto lora_up(device const half *T, device const bfloat *B, LoraUpParams lp, int m0, int col0, int brow) {
  auto tT = tensor<device half, dextents<int, 2>, tensor_inline>(const_cast<device half *>(T) + col0,
                                                                  dextents<int, 2>(H3_LORA_RANK, lp.mpad),
                                                                  array<int, 2>{1, lp.ldt});
  auto tB = mk<bfloat>(const_cast<device bfloat *>(B) + ulong(brow) * H3_LORA_RANK, H3_LORA_RANK, TN);
  constexpr auto desc = matmul2d_descriptor(TM, TN, H3_LORA_RANK, false, true, false);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  auto a = tT.template slice<H3_LORA_RANK, TM>(0, m0);
  auto b = tB.template slice<H3_LORA_RANK, TN>(0, 0);
  auto c = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
#pragma unroll
  for (ushort i = 0; i < c.get_capacity(); ++i)
    if (c.is_valid_element(i)) c[i] = 0.0f;
  op.run(a, b, c);
  return c;
}

// What a GEMM needs to compute its up-projection tile itself after the K loop (the _lora_post kernels).
struct LoraArgs {
  device const half *LT;
  device const bfloat *LB;
  LoraUpParams lp;
  int col0, brow;
};

}  // namespace h3

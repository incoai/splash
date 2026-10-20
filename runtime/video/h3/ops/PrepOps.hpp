#pragma once

// Dispatches of the native prep kernels (kernels/prep, MLX-identical arithmetic). Each call encodes one MLX primitive
// with MLX's routing for its shape (e.g. a bf16 linear with M <= 15 rows takes MLX's gemv_wide, otherwise the NAX
// GEMM), so the results match the former Python prep bit for bit.
#import <Metal/Metal.h>

#include "video/h3/metal/MetalBackend.hpp"

#include <string>
#include <vector>

namespace h3::ops {

class PrepOps {
public:
  using View = metal::BufferView;
  static std::vector<std::string> kernels();
  explicit PrepOps(metal::MetalBackend &backend) : b_(backend) {}

  // y[M][N] = x[M][K] W[N][K]^T (+ bias[N]) as nn.Linear / mx.addmm. bf16: gemv_wide for 2 <= M <= 15, else the
  // NAX GEMM; fp32 (TF32 NAX GEMM, MLX's default) always the GEMM.
  void linear(id<MTLComputeCommandEncoder> e, bool f32, View x, View w, View bias, View y, int M, int N, int K);
  // mx.fast.rms_norm over rows of `axis` (bf16), input / output row strides in elements
  void rms(id<MTLComputeCommandEncoder> e, View x, View w, View y, int rows, int axis, int inLd, int outLd, float eps);
  // mx.fast.scaled_dot_product_attention, bf16, head dim 128, no mask; strides in elements
  struct AttnStrides {
    int qH, qL, kH, kL, vH, vL, oH, oL;
  };
  void attention(id<MTLComputeCommandEncoder> e, View q, View k, View v, View o, int L, int heads, float scale,
                 const AttnStrides &s);
  void siluF32(id<MTLComputeCommandEncoder> e, View x, View y, int n);
  void swiglu(id<MTLComputeCommandEncoder> e, View fused, View y, int rows, int ffn);  // silu(gate) * value, bf16
  void add(id<MTLComputeCommandEncoder> e, View a, View b, View y, int n);              // bf16
  void castBf16(id<MTLComputeCommandEncoder> e, View x, View y, int n);
  void ropeInvFreq(id<MTLComputeCommandEncoder> e, View inv);
  void ropeTrig(id<MTLComputeCommandEncoder> e, View pos, View inv, View cos, View sin, int rows);
  void tembSinusoid(id<MTLComputeCommandEncoder> e, View t, View y, int T);
  void normal(id<MTLComputeCommandEncoder> e, View bits, View y, int n);

private:
  metal::MetalBackend &b_;
};

} // namespace h3::ops

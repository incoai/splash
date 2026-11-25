#pragma once
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/softcapped_swiglu.h"

// What both GGUF GEMM families (kernels/shared/gguf_linear.metal and
// kernels/decode/linear_gguf_sgmatrix.metal) do to a finished fp32 sum.
enum GgufEpilogue : ushort { EpNone, EpResidual, EpUpWithGate, EpUpWithGateSoftcap4_25 };

// The output of sum v at element `at`: v, v plus the residual aux[at], or the bf16-rounded up value v gated by
// aux[at] through silu or a compiled routing's softcapped SwiGLU, as the destination's type Out: bf16, or fp32
// (the plain epilogue's logits).
template <GgufEpilogue Ep, class Out = bfloat> inline Out gguf_epilogue(float v, device const bfloat *aux, ulong at) {
  if constexpr (Ep == EpResidual) v += float(aux[at]);
  if constexpr (Ep == EpUpWithGate) v = float(bfloat(v)) * splash_silu(float(aux[at]));
  if constexpr (Ep == EpUpWithGateSoftcap4_25)
    v = SoftcappedSwiglu<4, 25>::value(float(bfloat(v)), float(aux[at]));
  return Out(v);
}

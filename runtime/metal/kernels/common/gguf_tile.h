#pragma once
#include "metal/kernels/common/activation.h"

// What both GGUF GEMM families (kernels/shared/gguf_linear.metal and
// kernels/decode/linear_gguf_sgmatrix.metal) do to a finished fp32 sum.
enum GgufEpilogue : ushort { EpNone, EpResidual, EpUpWithGate };

// The output of sum v at element `at`: v, v plus the residual aux[at], or the bf16-rounded up value v times silu of
// the gate aux[at], both clamped by a SwiGLU limit > 0 (splash_swiglu), as the destination's type Out: bf16, or fp32
// (the plain epilogue's logits).
template <GgufEpilogue Ep, class Out = bfloat>
inline Out gguf_epilogue(float v, device const bfloat *aux, ulong at, float swiglu_limit = 0.0f) {
  if constexpr (Ep == EpResidual) v += float(aux[at]);
  if constexpr (Ep == EpUpWithGate) v = splash_swiglu(float(bfloat(v)), float(aux[at]), swiglu_limit);
  return Out(v);
}

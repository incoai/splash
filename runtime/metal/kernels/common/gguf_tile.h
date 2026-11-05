#pragma once
#include "metal/kernels/common/activation.h"

// What both GGUF GEMM families (kernels/shared/gguf_linear.metal and
// kernels/decode/linear_gguf_sgmatrix.metal) do to a finished fp32 sum.
enum GgufEpilogue : ushort { EpNone, EpResidual, EpUpWithGate, EpClampedUpWithGate };

// The output of sum v at element `at`: v, v plus the residual aux[at], the
// bf16-rounded up value v times silu of the gate aux[at], or the same gated
// product with both activations clamped at the clamp aux's value before the
// silu and the product (a family whose SwiGLU clamps: the gate scratch holds
// its unclamped sums, whose clamp commutes with their bf16 rounding, as the
// threshold is exactly representable), as the destination's type Out: bf16,
// or fp32 (the plain epilogue's logits).
template <GgufEpilogue Ep, class Out = bfloat>
inline Out gguf_epilogue(float v, device const bfloat *aux, ulong at, float clamped = 0.0f) {
  if constexpr (Ep == EpResidual) v += float(aux[at]);
  if constexpr (Ep == EpUpWithGate) v = float(bfloat(v)) * splash_silu(float(aux[at]));
  if constexpr (Ep == EpClampedUpWithGate) {
    // An infinity clamps nothing, which an unclamped dispatch's value is.
    v = splash_silu(fmin(fmax(float(aux[at]), -clamped), clamped)) *
        fmin(fmax(float(bfloat(v)), -clamped), clamped);
  }
  return Out(v);
}

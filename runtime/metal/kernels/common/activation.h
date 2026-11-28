#pragma once
#include <metal_stdlib>
using namespace metal;

// SiLU and the logistic sigmoid of every kernel, e^-x taken as
// fast::exp2(-x * log2(e)). A gated product multiplies splash_silu(gate)
// into its other factor, so silu(g) * up evaluates g / (1 + e^-g) * up.
inline float splash_silu(float x) {
  return x / (1.0f + fast::exp2(-1.44269504089f * x));
}
inline float2 splash_silu(float2 x) {
  return x / (1.0f + fast::exp2(-1.44269504089f * x));
}
inline float splash_sigmoid(float x) {
  return 1.0f / (1.0f + fast::exp2(-1.44269504089f * x));
}

// GELU, x Phi(x), with the normal CDF Phi taken through tanh
// (gelu_pytorch_tanh) or through erf. erf is Abramowitz & Stegun 7.1.26,
// whose error of 1.5e-7 lies far below bf16 rounding.
inline float splash_gelu_tanh(float x) {
  return 0.5f * x *
      (1.0f + precise::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}
inline float splash_erf(float x) {
  float sign = x < 0.0f ? -1.0f : 1.0f;
  x = fabs(x);
  float t = 1.0f / (1.0f + 0.3275911f * x);
  float polynomial = t * (0.254829592f + t * (-0.284496736f +
      t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
  return sign * (1.0f - polynomial * precise::exp(-x * x));
}
inline float splash_gelu_erf(float x) {
  return 0.5f * x * (1.0f + splash_erf(x * 0.7071067811865476f));
}

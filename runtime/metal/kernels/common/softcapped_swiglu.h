#pragma once

#include "metal/kernels/common/activation.h"

#include <metal_stdlib>
using namespace metal;

// A SwiGLU whose sigmoid sees the raw gate g while softcaps bound its linear
// gate factor and its up value u:
// GateCap tanh(g / GateCap) * sigmoid(g) * UpCap tanh(u / UpCap), in fp32,
// the factors in that order. The caps are a compiled MoE routing's
// (ops::kCompiledMoeRoutings); tanh is the precise one.
template <uint GateCap, uint UpCap> struct SoftcappedSwiglu {
  static float value(float up, float gate) {
    const float gate_cap = float(GateCap), up_cap = float(UpCap);
    return gate_cap * precise::tanh(gate / gate_cap) * splash_sigmoid(gate) *
           (up_cap * precise::tanh(up / up_cap));
  }
};

#pragma once
// What a dense projection (ops::DenseEpilogue) does with a finished fp32 sum of output `column`: adds its bias, then
// stores it (bf16), stores its GELU (bf16), or adds it to the fp32 row it stores into, as it is or times its
// output's scale (a LayerScale's). The bf16 dense tiles (kernels/shared/dense_linear.metal) and the GGUF prefill
// tiles' dense instances (gguf_linear.metal, gguf_prefill_int8.metal) store through it, so a quantized tower's
// projections end as a bf16 one's do.
#include "metal/kernels/common/activation.h"
#include <metal_stdlib>
using namespace metal;

enum DenseEpilogue { DenseNone, DenseGeluTanh, DenseGeluErf, DenseResidual, DenseScaledResidual };

template <DenseEpilogue Epilogue, typename Output>
inline void dense_epilogue_store(float sum, device const bfloat *bias, device const bfloat *scale, device Output *out,
                                 uint column) {
  float value = sum + float(bias[column]);
  if constexpr (Epilogue == DenseGeluTanh) value = splash_gelu_tanh(value);
  else if constexpr (Epilogue == DenseGeluErf) value = splash_gelu_erf(value);
  if constexpr (Epilogue == DenseScaledResidual) value = value * float(scale[column]) + float(*out);
  else if constexpr (Epilogue == DenseResidual) value += float(*out);
  *out = Output(value);
}

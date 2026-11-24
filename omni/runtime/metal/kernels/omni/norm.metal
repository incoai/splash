// Row norms with a run-time epsilon and elementwise activations
// (ops::OmniNorm): RMSNorm and LayerNorm over rows of `width` values, 256
// threads per row, fp32 statistics in source order. Instances name their
// input, output and weight types: omni_rms_norm_f32 (fp32 rows and weights),
// omni_layer_norm_f32_bf16 (fp32 rows, bf16 weights and bias, bf16 output),
// omni_layer_norm_f32 (fp32 everything).

#include "metal/abi/Omni.h"
#include <metal_stdlib>

using namespace metal;

inline float omni_block_sum(float value, threadgroup float *reductions, uint tid, uint lane, uint sg) {
#pragma clang fp reassociate(off)
  value = simd_sum(value);
  if (lane == 0) reductions[sg] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint i = 0; i < 8; ++i) total += reductions[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  (void)tid;
  return total;
}

template <typename In, typename Out, typename W>
inline void omni_rms_norm_row(device const In *input, device const W *weight, device Out *output,
                              constant OmniNormParams &p, uint row, uint tid, uint lane, uint sg,
                              threadgroup float *reductions) {
#pragma clang fp reassociate(off)
  device const In *x = input + ulong(row) * p.width;
  float sum = 0.0f;
  for (uint column = tid; column < p.width; column += 256) {
    const float value = float(x[column]);
    sum += value * value;
  }
  const float total = omni_block_sum(sum, reductions, tid, lane, sg);
  const float inverse = rsqrt(total / float(p.width) + p.epsilon);
  for (uint column = tid; column < p.width; column += 256)
    output[ulong(row) * p.width + column] = Out(float(x[column]) * inverse * float(weight[column]));
}

template <typename In, typename Out, typename W>
inline void omni_layer_norm_row(device const In *input, device const W *weight, device const W *bias,
                                device Out *output, constant OmniNormParams &p, uint row, uint tid, uint lane,
                                uint sg, threadgroup float *reductions) {
#pragma clang fp reassociate(off)
  device const In *x = input + ulong(row) * p.width;
  float sum = 0.0f;
  for (uint column = tid; column < p.width; column += 256) sum += float(x[column]);
  const float mean = omni_block_sum(sum, reductions, tid, lane, sg) / float(p.width);
  float squares = 0.0f;
  for (uint column = tid; column < p.width; column += 256) {
    const float centered = float(x[column]) - mean;
    squares += centered * centered;
  }
  const float variance = omni_block_sum(squares, reductions, tid, lane, sg) / float(p.width);
  const float inverse = rsqrt(variance + p.epsilon);
  for (uint column = tid; column < p.width; column += 256)
    output[ulong(row) * p.width + column] =
        Out((float(x[column]) - mean) * inverse * float(weight[column]) + float(bias[column]));
}

#define OMNI_RMS_NORM(Name, In, Out, W)                                                                          \
  kernel void Name(device const In *input [[buffer(0)]], device const W *weight [[buffer(1)]],                  \
                   device Out *output [[buffer(2)]], constant OmniNormParams &p [[buffer(3)]],                  \
                   uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],         \
                   uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {       \
    threadgroup float reductions[8];                                                                           \
    if (row >= p.rows) return;                                                                                 \
    omni_rms_norm_row<In, Out, W>(input, weight, output, p, row, tid, lane, sg, reductions);                   \
  }
OMNI_RMS_NORM(omni_rms_norm_f32, float, float, float)
OMNI_RMS_NORM(omni_rms_norm_bf16, bfloat, bfloat, bfloat)
#undef OMNI_RMS_NORM

#define OMNI_LAYER_NORM(Name, In, Out, W)                                                                        \
  kernel void Name(device const In *input [[buffer(0)]], device const W *weight [[buffer(1)]],                  \
                   device const W *bias [[buffer(2)]], device Out *output [[buffer(3)]],                        \
                   constant OmniNormParams &p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],         \
                   uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],           \
                   uint sg [[simdgroup_index_in_threadgroup]]) {                                                \
    threadgroup float reductions[8];                                                                           \
    if (row >= p.rows) return;                                                                                 \
    omni_layer_norm_row<In, Out, W>(input, weight, bias, output, p, row, tid, lane, sg, reductions);           \
  }
OMNI_LAYER_NORM(omni_layer_norm_f32_bf16, float, bfloat, bfloat)
OMNI_LAYER_NORM(omni_layer_norm_f32, float, float, float)
#undef OMNI_LAYER_NORM

inline float omni_norm_erf(float x) {
  const float sign = x < 0.0f ? -1.0f : 1.0f;
  x = fabs(x);
  const float t = 1.0f / (1.0f + 0.3275911f * x);
  const float polynomial =
      t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
  return sign * (1.0f - polynomial * precise::exp(-x * x));
}

template <typename T>
inline void omni_activate_elements(device const T *input, device T *output, constant OmniElementwiseParams &p,
                                   uint index) {
  if (index >= p.count) return;
  float value = float(input[index]);
  if (p.activation == 1u) value = 0.5f * value * (1.0f + omni_norm_erf(value * 0.7071067811865476f));
  else if (p.activation == 2u) value = value / (1.0f + precise::exp(-value));
  output[index] = T(value);
}

#define OMNI_ACTIVATE(Name, T)                                                                                  \
  kernel void Name(device const T *input [[buffer(0)]], device T *output [[buffer(1)]],                        \
                   constant OmniElementwiseParams &p [[buffer(2)]], uint index [[thread_position_in_grid]]) {   \
    omni_activate_elements<T>(input, output, p, index);                                                        \
  }
OMNI_ACTIVATE(omni_activate_f32, float)
OMNI_ACTIVATE(omni_activate_bf16, bfloat)
#undef OMNI_ACTIVATE

// out[i] = bf16(a[i]) for fp32 rows, and the reverse: conversions between
// the fp32 residual streams of the encoders and the bf16 projection inputs.
kernel void omni_convert_f32_bf16(device const float *input [[buffer(0)]], device bfloat *output [[buffer(1)]],
                                  constant OmniElementwiseParams &p [[buffer(2)]],
                                  uint index [[thread_position_in_grid]]) {
  if (index < p.count) output[index] = bfloat(input[index]);
}
kernel void omni_convert_bf16_f32(device const bfloat *input [[buffer(0)]], device float *output [[buffer(1)]],
                                  constant OmniElementwiseParams &p [[buffer(2)]],
                                  uint index [[thread_position_in_grid]]) {
  if (index < p.count) output[index] = float(input[index]);
}

// out[r][n] = act(in[r][n] + bias[n]) over bf16 rows of `width` values and
// an fp32 bias: the bias of a Q4 projection, which the Q4 tiles do not add.
kernel void omni_bias_activate(device const bfloat *input [[buffer(0)]], device const float *bias [[buffer(1)]],
                               device bfloat *output [[buffer(2)]], constant OmniBiasParams &p [[buffer(3)]],
                               uint index [[thread_position_in_grid]]) {
  const uint elements = p.rows * p.width;
  if (index >= elements) return;
  float value = float(input[index]) + bias[index % p.width];
  if (p.activation == 1u) value = 0.5f * value * (1.0f + omni_norm_erf(value * 0.7071067811865476f));
  else if (p.activation == 2u) value = value / (1.0f + precise::exp(-value));
  output[index] = bfloat(value);
}
